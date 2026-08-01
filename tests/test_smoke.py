import numpy as np
import torch

from unoc.calibration import (
    OperatorCalibrator,
    _higher_quantile,
    calibrate_trajectory_model,
    polyhedral_tightening_margin,
    trajectory_coverage_metrics,
)
from unoc.data import TrajectoryArrays
from unoc.forced_control import (
    ForcedBurgersCase,
    ForcedBurgersConfig,
    ForcedBurgersSolver,
    OracleCEMConfig,
    oracle_cem_action,
)
from unoc.models import PerturbationScaleWorldModel, build_model
from unoc.mpc import _sequence_cost
from unoc.pde import BurgersConfig, BurgersSolver
from unoc.value_gap import (
    finite_horizon_value_bound,
    fixed_policy_infinite_bound,
    simulate_sharpness_witness,
)


def test_conformal_quantile_uses_infinity_when_sample_is_too_small() -> None:
    assert np.isinf(_higher_quantile(np.arange(5.0), 0.90))
    assert _higher_quantile(np.arange(20.0), 0.90) == 18.0


def test_solver_preserves_boundary_conditions() -> None:
    solver = BurgersSolver(BurgersConfig(grid_size=32))
    rng = np.random.default_rng(2)
    state = solver.random_state(rng, -0.1, 0.08)
    next_state = solver.step(state, 0.4, 0.02, -0.1, 0.08)
    assert np.isfinite(next_state).all()
    assert next_state[0] == -0.1
    assert next_state[-1] == 0.08


def test_forced_solver_preserves_boundaries_and_accepts_vector_actions() -> None:
    solver = ForcedBurgersSolver(
        ForcedBurgersConfig(grid_size=32, control_dt=0.01, solver_dt=0.0005)
    )
    case = ForcedBurgersCase(
        viscosity=0.012,
        left_boundary=-0.02,
        right_boundary=0.03,
        actuator_gain=1.1,
        forcing_amplitude=0.6,
        forcing_frequency=1.0,
        forcing_phase=0.2,
        initial_amplitude=0.05,
        initial_seed=4,
    )
    state = solver.initial_state(case)
    next_state = solver.step(state, np.array([-0.3, 0.2]), 0.0, case)
    assert next_state.shape == state.shape
    assert np.isfinite(next_state).all()
    assert next_state[0] == case.left_boundary
    assert next_state[-1] == case.right_boundary
    assert np.allclose(solver.external_force(0.1, case)[[0, -1]], 0.0)


def test_oracle_cem_action_is_reproducible_and_feasible() -> None:
    solver = ForcedBurgersSolver(
        ForcedBurgersConfig(grid_size=24, control_dt=0.01, solver_dt=0.0005)
    )
    case = ForcedBurgersCase(
        viscosity=0.012,
        left_boundary=0.0,
        right_boundary=0.0,
        actuator_gain=1.0,
        forcing_amplitude=0.6,
        forcing_frequency=1.0,
        forcing_phase=0.0,
        initial_amplitude=0.05,
        initial_seed=7,
    )
    config = OracleCEMConfig(horizon=2, candidates=8, elites=2, iterations=2)
    state = solver.initial_state(case)
    first = oracle_cem_action(solver, state, 0.0, case, config, seed=11)
    second = oracle_cem_action(solver, state, 0.0, case, config, seed=11)
    assert np.allclose(first, second)
    assert first.shape == (solver.action_dimension,)
    assert np.max(np.abs(first)) <= solver.config.action_limit


def test_all_world_models_return_mean_and_positive_scale() -> None:
    batch, grid = 3, 32
    state = torch.randn(batch, grid) * 0.1
    action = torch.zeros(batch)
    viscosity = torch.full((batch,), 0.02)
    boundary = torch.zeros(batch, 2)
    for kind in ("fno", "tno", "dscdno", "moe"):
        model = build_model(kind, width=12, modes=6, layers=2)
        mean, scale = model(state, action, viscosity, boundary)
        assert mean.shape == state.shape
        assert scale.shape == state.shape
        assert torch.all(scale > 0)


def test_radius_propagation_is_monotone() -> None:
    calibrator = OperatorCalibrator(
        multiplier=2.0,
        coverage=0.9,
        lipschitz=1.1,
        norm_kind="decision",
    )
    previous = torch.tensor([0.2])
    one_step = torch.tensor([0.1])
    propagated = calibrator.propagate(previous, one_step)
    assert float(propagated) > float(previous)


def test_adjoint_radius_is_exact_ellipsoid_support() -> None:
    calibrator = OperatorCalibrator(
        multiplier=1.7,
        coverage=0.9,
        lipschitz=1.0,
        norm_kind="ellipsoid",
    )
    scale = torch.tensor([[0.2, 0.4, 0.1, 0.3]])
    sensitivity = torch.tensor([[1.0, -2.0, 0.5, 3.0]])
    radius = calibrator.radius(scale, sensitivity)
    weighted_gradient = scale * sensitivity
    grid = scale.shape[1]
    direction = calibrator.multiplier * np.sqrt(grid) * weighted_gradient
    direction = direction / torch.linalg.vector_norm(weighted_gradient, dim=1)[:, None]
    perturbation = scale * direction
    objective = torch.mean(sensitivity * perturbation, dim=1)
    constraint = torch.sqrt(torch.mean((perturbation / scale).square(), dim=1))
    assert torch.allclose(objective, radius, atol=1e-6)
    assert torch.allclose(
        constraint,
        torch.tensor([calibrator.multiplier]),
        atol=1e-6,
    )


def test_max_score_is_simultaneous_coordinate_score() -> None:
    calibrator = OperatorCalibrator(2.0, 0.9, 1.0, "max")
    error = torch.tensor([[0.2, -0.8, 0.3]])
    scale = torch.tensor([[0.1, 0.4, 0.1]])
    score = calibrator.standardized_score(error, scale)
    assert torch.allclose(score, torch.tensor([3.0]))
    assert bool(score <= calibrator.multiplier) is False


def test_box_support_and_polyhedral_tightening_are_exact() -> None:
    calibrator = OperatorCalibrator(1.5, 0.9, 1.0, "max")
    scale = torch.tensor([[0.2, 0.4, 0.1]])
    sensitivity = torch.tensor([[1.0, -2.0, 0.5]])
    support = calibrator.radius(scale, sensitivity)
    expected_normalized_support = 1.5 * torch.mean(sensitivity.abs() * scale, dim=1)
    assert torch.allclose(support, expected_normalized_support)

    matrix = torch.tensor([[1.0, -2.0, 0.5], [-1.0, 0.0, 3.0]])
    margin = polyhedral_tightening_margin(calibrator, scale, matrix)
    expected_margin = 1.5 * torch.einsum("mn,bn->bm", matrix.abs(), scale)
    assert torch.allclose(margin, expected_margin)


def test_perturbation_wrapper_uses_smoothed_disagreement_and_floor() -> None:
    class AffineModel(torch.nn.Module):
        def __init__(self, offset: torch.Tensor):
            super().__init__()
            self.register_buffer("offset", offset)

        def forward(self, state, action, viscosity, boundary):
            mean = state + self.offset[None, :]
            return mean, torch.ones_like(mean)

    base = AffineModel(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]))
    perturbed = AffineModel(torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0]))
    wrapper = PerturbationScaleWorldModel(base, perturbed, smoothing_window=3, scale_floor=0.2)
    state = torch.zeros(1, 5)
    mean, scale = wrapper(
        state,
        torch.zeros(1),
        torch.ones(1),
        torch.zeros(1, 2),
    )
    assert torch.allclose(mean, state)
    assert torch.all(scale >= 0.2)
    assert float(scale[0, 2]) > float(scale[0, 0])


def test_rollout_recursion_matches_geometric_bound() -> None:
    lipschitz, epsilon, steps = 0.8, 0.03, 7
    error = 0.0
    for _ in range(steps):
        error = lipschitz * error + epsilon
    closed_form = epsilon * (1.0 - lipschitz**steps) / (1.0 - lipschitz)
    assert abs(error - closed_form) < 1e-12


def test_value_gap_witness_attains_infinite_horizon_bound() -> None:
    epsilon, gamma, lipschitz, reward_lipschitz = 0.01, 0.95, 1.0, 1.3
    exact = fixed_policy_infinite_bound(
        epsilon, gamma, reward_lipschitz, lipschitz
    )
    simulated = simulate_sharpness_witness(
        epsilon,
        gamma,
        lipschitz,
        reward_lipschitz,
        horizon=5000,
    )
    assert abs(simulated - exact) / exact < 1e-10


def test_finite_value_bound_converges_to_squared_discount_rate() -> None:
    epsilon, gamma, reward_lipschitz = 0.02, 0.9, 0.7
    finite = finite_horizon_value_bound(
        epsilon,
        gamma,
        reward_lipschitz,
        dynamics_lipschitz=1.0,
        horizon=1000,
    )
    exact = gamma * reward_lipschitz * epsilon / (1.0 - gamma) ** 2
    assert abs(finite - exact) / exact < 1e-10


def test_trajectory_calibration_covers_exact_behavior_rollouts() -> None:
    class LinearWorldModel(torch.nn.Module):
        def forward(self, state, action, viscosity, boundary):
            return 0.9 * state + 0.1 * action[:, None], torch.full_like(state, 0.05)

    samples, horizon, grid = 6, 3, 4
    actions = np.linspace(-0.5, 0.5, samples * horizon, dtype=np.float32).reshape(
        samples, horizon
    )
    states = np.zeros((samples, horizon + 1, grid), dtype=np.float32)
    states[:, 0] = np.linspace(-0.2, 0.2, samples, dtype=np.float32)[:, None]
    for step in range(horizon):
        states[:, step + 1] = 0.9 * states[:, step] + 0.1 * actions[:, step, None]
    trajectories = TrajectoryArrays(
        states,
        actions,
        np.full(samples, 0.02, dtype=np.float32),
        np.zeros((samples, 2), dtype=np.float32),
    )
    calibrator, info = calibrate_trajectory_model(
        LinearWorldModel(), trajectories, torch.device("cpu"), coverage=0.8
    )
    metrics = trajectory_coverage_metrics(
        LinearWorldModel(), calibrator, trajectories, torch.device("cpu")
    )
    assert info["horizon"] == horizon
    assert metrics["coverage"] == 1.0


def test_autograd_conversion_matches_normalized_cost_gradient() -> None:
    state = torch.tensor([[0.2, -0.4, 0.1, 0.3]], requires_grad=True)
    weight = torch.tensor([[0.3, 0.5, 0.8, 1.0]])
    cost = torch.mean(weight * state.square())
    euclidean_gradient = torch.autograd.grad(cost, state)[0]
    normalized_inner_product_gradient = state.shape[1] * euclidean_gradient
    assert torch.allclose(
        normalized_inner_product_gradient,
        2.0 * weight * state,
        atol=1e-7,
    )


def test_adversarial_rollout_cost_dominates_nominal_cost() -> None:
    class LinearWorldModel(torch.nn.Module):
        def forward(self, state, action, viscosity, boundary):
            mean = 0.9 * state + 0.1 * action[:, None]
            scale = torch.full_like(mean, 0.08)
            return mean, scale

    model = LinearWorldModel()
    calibrator = OperatorCalibrator(
        multiplier=1.2,
        coverage=0.9,
        lipschitz=1.0,
        norm_kind="ellipsoid",
    )
    state = torch.tensor([0.2, -0.1, 0.3, -0.2])
    sequences = torch.tensor([[0.5, -0.2], [-0.4, 0.1]])
    nominal = _sequence_cost(
        model,
        calibrator,
        state,
        sequences,
        0.02,
        (0.0, 0.0),
        False,
        0.002,
    )
    adversarial = _sequence_cost(
        model,
        calibrator,
        state,
        sequences,
        0.02,
        (0.0, 0.0),
        True,
        0.002,
        adversarial=True,
        adversary_iterations=4,
    )
    assert torch.all(adversarial >= nominal)
