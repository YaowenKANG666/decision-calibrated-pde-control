import numpy as np
import torch

from unoc.calibration import OperatorCalibrator
from unoc.models import build_model
from unoc.mpc import _sequence_cost
from unoc.pde import BurgersConfig, BurgersSolver


def test_solver_preserves_boundary_conditions() -> None:
    solver = BurgersSolver(BurgersConfig(grid_size=32))
    rng = np.random.default_rng(2)
    state = solver.random_state(rng, -0.1, 0.08)
    next_state = solver.step(state, 0.4, 0.02, -0.1, 0.08)
    assert np.isfinite(next_state).all()
    assert next_state[0] == -0.1
    assert next_state[-1] == 0.08


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


def test_rollout_recursion_matches_geometric_bound() -> None:
    lipschitz, epsilon, steps = 0.8, 0.03, 7
    error = 0.0
    for _ in range(steps):
        error = lipschitz * error + epsilon
    closed_form = epsilon * (1.0 - lipschitz**steps) / (1.0 - lipschitz)
    assert abs(error - closed_form) < 1e-12


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
