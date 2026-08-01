"""Task validation for controlled Burgers dynamics under persistent forcing.

The module deliberately precedes neural-operator training.  It asks whether a
closed-loop controller with access to the numerical plant can improve over
zero control on an independently sampled population of forced PDE instances.
If this oracle-model comparison fails, learned-model and uncertainty studies
on the task are not scientifically informative.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ForcedBurgersConfig:
    """Numerical and control constants fixed before drawing test cases."""

    grid_size: int = 48
    control_dt: float = 0.025
    solver_dt: float = 0.0005
    actuator_centers: tuple[float, ...] = (0.28, 0.72)
    actuator_width: float = 0.10
    action_limit: float = 2.0
    control_weight: float = 0.002
    terminal_weight: float = 4.0
    failure_threshold: float = 0.55


@dataclass(frozen=True)
class ForcedBurgersCase:
    """One draw from the joint deployment distribution."""

    viscosity: float
    left_boundary: float
    right_boundary: float
    actuator_gain: float
    forcing_amplitude: float
    forcing_frequency: float
    forcing_phase: float
    initial_amplitude: float
    initial_seed: int


@dataclass(frozen=True)
class OracleCEMConfig:
    """Sampling budget for PDE-oracle model-predictive control."""

    horizon: int = 6
    candidates: int = 64
    elites: int = 8
    iterations: int = 3
    initial_std: float = 0.9


class ForcedBurgersSolver:
    r"""Finite-difference solver for a persistently forced controlled plant.

    The equation is

    .. math::
       u_t + u u_x = \nu u_{xx} + f_{\rm ext}(x,t)
       + g_{\rm act}\sum_k a_{t,k} b_k(x).

    The reference is zero in the released task. Persistent forcing, rather
    than the initial condition, prevents zero control from solving the task by
    passive viscous decay.
    """

    def __init__(self, config: ForcedBurgersConfig | None = None):
        self.config = config or ForcedBurgersConfig()
        self.x = np.linspace(0.0, 1.0, self.config.grid_size)
        profiles = []
        for center in self.config.actuator_centers:
            profile = np.exp(-0.5 * ((self.x - center) / self.config.actuator_width) ** 2)
            profile[0] = profile[-1] = 0.0
            profiles.append(profile / np.max(profile))
        self.actuators = np.stack(profiles, axis=0)

    @property
    def action_dimension(self) -> int:
        return int(self.actuators.shape[0])

    def reference(self, time: float) -> np.ndarray:
        """Return the prescribed state reference at a physical time."""

        del time
        return np.zeros_like(self.x)

    def external_force(self, time: float, case: ForcedBurgersCase) -> np.ndarray:
        """Persistent smooth forcing with a time-varying secondary mode."""

        temporal = np.sin(2.0 * np.pi * case.forcing_frequency * time + case.forcing_phase)
        force = case.forcing_amplitude * (
            np.sin(np.pi * self.x) + 0.35 * temporal * np.sin(2.0 * np.pi * self.x)
        )
        force[0] = force[-1] = 0.0
        return force

    def initial_state(self, case: ForcedBurgersCase) -> np.ndarray:
        """Draw a reproducible smooth initial field satisfying the boundaries."""

        rng = np.random.default_rng(case.initial_seed)
        state = case.left_boundary * (1.0 - self.x) + case.right_boundary * self.x
        for mode in range(1, 5):
            coefficient = case.initial_amplitude * rng.normal() / mode
            state += coefficient * np.sin(mode * np.pi * self.x)
        state[0], state[-1] = case.left_boundary, case.right_boundary
        return state.astype(np.float64)

    def step_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        time: float,
        case: ForcedBurgersCase,
    ) -> np.ndarray:
        """Advance a batch through one control interval with identical physics."""

        states = np.asarray(states, dtype=np.float64)
        if states.ndim == 1:
            states = states[None, :]
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim == 1:
            actions = actions[None, :]
        if states.shape[0] != actions.shape[0]:
            if states.shape[0] == 1:
                states = np.broadcast_to(states, (actions.shape[0], states.shape[1])).copy()
            else:
                raise ValueError("state and action batch sizes do not match")
        if actions.shape[1] != self.action_dimension:
            raise ValueError("wrong action dimension")

        cfg = self.config
        actions = np.clip(actions, -cfg.action_limit, cfg.action_limit)
        substeps = int(np.ceil(cfg.control_dt / cfg.solver_dt))
        dt = cfg.control_dt / substeps
        dx = 1.0 / (cfg.grid_size - 1)
        if dt * case.viscosity / (dx * dx) > 0.49:
            raise ValueError("diffusion CFL condition violated")

        controlled_force = case.actuator_gain * np.einsum("bm,mn->bn", actions, self.actuators)
        u = states.copy()
        for substep in range(substeps):
            backward = (u[:, 1:-1] - u[:, :-2]) / dx
            forward = (u[:, 2:] - u[:, 1:-1]) / dx
            upwind = np.where(u[:, 1:-1] >= 0.0, backward, forward)
            laplacian = (u[:, 2:] - 2.0 * u[:, 1:-1] + u[:, :-2]) / (dx * dx)
            forcing = self.external_force(time + (substep + 0.5) * dt, case)
            u[:, 1:-1] += dt * (
                -u[:, 1:-1] * upwind
                + case.viscosity * laplacian
                + forcing[None, 1:-1]
                + controlled_force[:, 1:-1]
            )
            u[:, 0] = case.left_boundary
            u[:, -1] = case.right_boundary
        return u

    def step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        time: float,
        case: ForcedBurgersCase,
    ) -> np.ndarray:
        return self.step_batch(state[None, :], action[None, :], time, case)[0]

    def state_cost_batch(self, states: np.ndarray, time: float) -> np.ndarray:
        reference = self.reference(time)
        return np.mean(np.square(states - reference[None, :]), axis=1)

    def stage_cost(
        self,
        state: np.ndarray,
        action: np.ndarray,
        time: float,
    ) -> float:
        tracking = float(self.state_cost_batch(state[None, :], time)[0])
        effort = self.config.control_weight * float(np.sum(np.square(action)))
        return tracking + effort


def sample_test_cases(size: int, seed: int) -> list[ForcedBurgersCase]:
    """Draw independent cases from a fixed joint deployment distribution."""

    if size < 1:
        raise ValueError("size must be positive")
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(size):
        cases.append(
            ForcedBurgersCase(
                viscosity=float(rng.uniform(0.008, 0.018)),
                left_boundary=float(rng.uniform(-0.03, 0.03)),
                right_boundary=float(rng.uniform(-0.03, 0.03)),
                actuator_gain=float(rng.uniform(0.65, 1.35)),
                forcing_amplitude=float(rng.uniform(0.40, 0.80)),
                forcing_frequency=float(rng.uniform(0.50, 1.50)),
                forcing_phase=float(rng.uniform(0.0, 2.0 * np.pi)),
                initial_amplitude=float(rng.uniform(0.03, 0.12)),
                initial_seed=int(rng.integers(0, 2**31 - 1)),
            )
        )
    return cases


def oracle_cem_action(
    solver: ForcedBurgersSolver,
    state: np.ndarray,
    time: float,
    case: ForcedBurgersCase,
    config: OracleCEMConfig,
    seed: int,
) -> np.ndarray:
    """Plan with direct access to the numerical PDE transition function.

    "Oracle" refers only to model access. The finite-budget CEM solve is an
    approximate optimizer and does not certify global optimality.
    """

    rng = np.random.default_rng(seed)
    action_dim = solver.action_dimension
    mean = np.zeros((config.horizon, action_dim), dtype=np.float64)
    std = np.full_like(mean, config.initial_std)
    for _ in range(config.iterations):
        noise = rng.normal(size=(config.candidates, config.horizon, action_dim))
        sequences = np.clip(
            mean[None, :, :] + std[None, :, :] * noise,
            -solver.config.action_limit,
            solver.config.action_limit,
        )
        predicted = np.broadcast_to(state, (config.candidates, state.size)).copy()
        cost = np.zeros(config.candidates, dtype=np.float64)
        for lookahead in range(config.horizon):
            lookahead_time = time + lookahead * solver.config.control_dt
            cost += solver.state_cost_batch(
                predicted,
                lookahead_time,
            )
            cost += solver.config.control_weight * np.sum(
                np.square(sequences[:, lookahead, :]), axis=1
            )
            predicted = solver.step_batch(
                predicted,
                sequences[:, lookahead, :],
                lookahead_time,
                case,
            )
        cost += solver.config.terminal_weight * solver.state_cost_batch(
            predicted,
            time + config.horizon * solver.config.control_dt,
        )
        elite_index = np.argpartition(cost, config.elites - 1)[: config.elites]
        elite = sequences[elite_index]
        mean = elite.mean(axis=0)
        std = np.maximum(elite.std(axis=0), 0.05)
    return mean[0]


def _bootstrap_interval(
    values: np.ndarray,
    statistic,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[float, float]:
    size = values.shape[0]
    samples = rng.integers(0, size, size=(replicates, size))
    estimates = np.asarray([statistic(values[index]) for index in samples])
    lower, upper = np.quantile(estimates, (0.025, 0.975))
    return float(lower), float(upper)


def _paired_bootstrap(
    uncontrolled: np.ndarray,
    oracle: np.ndarray,
    seed: int,
    replicates: int = 5_000,
) -> dict[str, list[float] | float]:
    rng = np.random.default_rng(seed)
    difference = oracle - uncontrolled
    mean_ci = _bootstrap_interval(difference, np.mean, rng, replicates)

    size = uncontrolled.size
    samples = rng.integers(0, size, size=(replicates, size))
    p90_difference = np.asarray(
        [
            np.quantile(oracle[index], 0.90) - np.quantile(uncontrolled[index], 0.90)
            for index in samples
        ]
    )
    return {
        "mean_difference": float(np.mean(difference)),
        "mean_difference_ci95": [float(mean_ci[0]), float(mean_ci[1])],
        "p90_difference": float(np.quantile(oracle, 0.90) - np.quantile(uncontrolled, 0.90)),
        "p90_difference_ci95": [
            float(np.quantile(p90_difference, 0.025)),
            float(np.quantile(p90_difference, 0.975)),
        ],
        "fraction_oracle_better": float(np.mean(oracle < uncontrolled)),
    }


def _controller_summary(
    costs: np.ndarray,
    efforts: np.ndarray,
    failures: np.ndarray,
    seed: int,
    replicates: int,
) -> dict[str, float | list[float]]:
    rng = np.random.default_rng(seed)
    return {
        "mean_cost": float(np.mean(costs)),
        "mean_cost_ci95": list(_bootstrap_interval(costs, np.mean, rng, replicates)),
        "median_cost": float(np.median(costs)),
        "median_cost_ci95": list(_bootstrap_interval(costs, np.median, rng, replicates)),
        "p90_cost": float(np.quantile(costs, 0.90)),
        "p90_cost_ci95": list(
            _bootstrap_interval(costs, lambda x: np.quantile(x, 0.90), rng, replicates)
        ),
        "failure_rate": float(np.mean(failures)),
        "mean_control_effort": float(np.mean(efforts)),
    }


def _figure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def _save_figure(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(figure)


def _make_figures(
    output_dir: Path,
    rows: list[dict[str, float]],
    representative: dict[str, list[float] | list[list[float]]],
    paired_summary: dict[str, list[float] | float],
) -> None:
    _figure_style()
    figure_dir = output_dir / "figures"
    uncontrolled = np.asarray([row["uncontrolled_cost"] for row in rows])
    oracle = np.asarray([row["oracle_cost"] for row in rows])

    figure, axis = plt.subplots(figsize=(3.50, 3.05), constrained_layout=True)
    axis.scatter(uncontrolled, oracle, s=15, alpha=0.72, color="#32688E", edgecolor="none")
    limit = 1.04 * max(float(np.max(uncontrolled)), float(np.max(oracle)))
    axis.plot([0.0, limit], [0.0, limit], color="#777777", lw=0.9, ls="--")
    axis.set_xlim(0.0, limit)
    axis.set_ylim(0.0, limit)
    axis.set_xlabel("Uncontrolled cumulative cost")
    axis.set_ylabel("PDE-oracle MPC cumulative cost")
    axis.set_title("Paired task-validity comparison")
    axis.grid(alpha=0.16)
    _save_figure(figure, figure_dir / "oracle_01_paired_cost")

    difference = oracle - uncontrolled
    mean_difference = float(paired_summary["mean_difference"])
    mean_ci = np.asarray(paired_summary["mean_difference_ci95"], dtype=float)
    figure, axis = plt.subplots(figsize=(3.50, 2.65), constrained_layout=True)
    jitter = np.linspace(-0.08, 0.08, difference.size)
    axis.scatter(jitter, difference, s=13, alpha=0.65, color="#32688E", edgecolor="none")
    axis.errorbar(
        0.30,
        mean_difference,
        yerr=np.asarray(
            [[mean_difference - mean_ci[0]], [mean_ci[1] - mean_difference]]
        ),
        fmt="o",
        color="#C4513B",
        capsize=3,
        lw=1.2,
    )
    axis.text(
        0.33,
        mean_difference,
        "Mean\n95% paired-bootstrap CI",
        fontsize=6.2,
        va="center",
    )
    axis.axhline(0.0, color="#777777", lw=0.9, ls="--")
    axis.set_xlim(-0.18, 0.48)
    axis.set_xticks([])
    axis.set_ylabel("Paired cost difference (oracle minus uncontrolled)")
    axis.set_title("PDE-oracle MPC lowers cost on matched cases")
    axis.grid(alpha=0.16, axis="y")
    _save_figure(figure, figure_dir / "oracle_02_paired_difference")

    times = np.asarray(representative["time"])
    figure, axis = plt.subplots(figsize=(3.50, 2.65), constrained_layout=True)
    axis.plot(times, representative["uncontrolled_rmse"], color="#888888", lw=1.4,
              label="Uncontrolled")
    axis.plot(times, representative["oracle_rmse"], color="#32688E", lw=1.4,
              label="PDE-oracle MPC")
    axis.set_xlabel("Physical time")
    axis.set_ylabel("State RMSE from reference")
    axis.set_title("Representative forced trajectory")
    axis.legend()
    axis.grid(alpha=0.16)
    _save_figure(figure, figure_dir / "oracle_03_tracking_trajectory")

    actions = np.asarray(representative["oracle_actions"])
    action_times = times[:-1]
    figure, axis = plt.subplots(figsize=(3.50, 2.65), constrained_layout=True)
    for index in range(actions.shape[1]):
        axis.step(
            action_times,
            actions[:, index],
            where="post",
            lw=1.3,
            label=f"Actuator {index + 1}",
        )
    axis.axhline(0.0, color="#777777", lw=0.8)
    axis.set_xlabel("Physical time")
    axis.set_ylabel("Applied action")
    axis.set_title("PDE-oracle MPC control input")
    axis.legend()
    axis.grid(alpha=0.16)
    _save_figure(figure, figure_dir / "oracle_04_control_actions")


def run_task_validation(
    output_dir: Path,
    cases: int = 100,
    rollout_horizon: int = 20,
    seed: int = 27,
    bootstrap_replicates: int = 5_000,
    cem_config: OracleCEMConfig | None = None,
) -> dict[str, object]:
    """Compare zero control and PDE-oracle MPC on matched independent cases."""

    solver = ForcedBurgersSolver()
    cem = cem_config or OracleCEMConfig()
    sampled_cases = sample_test_cases(cases, seed)
    rows: list[dict[str, float]] = []
    representative: dict[str, list[float] | list[list[float]]] | None = None

    for case_index, case in enumerate(sampled_cases):
        initial = solver.initial_state(case)
        controller_records: dict[str, dict[str, object]] = {}
        for controller in ("uncontrolled", "oracle"):
            state = initial.copy()
            cumulative = 0.0
            effort = 0.0
            failed = False
            rmse = [float(np.sqrt(solver.state_cost_batch(state[None, :], 0.0)[0]))]
            actions: list[list[float]] = []
            for step in range(rollout_horizon):
                time = step * solver.config.control_dt
                if controller == "uncontrolled":
                    action = np.zeros(solver.action_dimension, dtype=np.float64)
                else:
                    action = oracle_cem_action(
                        solver,
                        state,
                        time,
                        case,
                        cem,
                        seed=1_000_003 * case_index + step,
                    )
                cumulative += solver.stage_cost(
                    state,
                    action,
                    time,
                )
                effort += float(np.sum(np.square(action)))
                state = solver.step(state, action, time, case)
                current_rmse = float(
                    np.sqrt(
                        solver.state_cost_batch(
                            state[None, :],
                            time + solver.config.control_dt,
                        )[0]
                    )
                )
                failed = failed or current_rmse > solver.config.failure_threshold
                rmse.append(current_rmse)
                actions.append(action.tolist())
            cumulative += solver.config.terminal_weight * float(
                solver.state_cost_batch(
                    state[None, :],
                    rollout_horizon * solver.config.control_dt,
                )[0]
            )
            controller_records[controller] = {
                "cost": cumulative,
                "effort": effort / rollout_horizon,
                "failed": failed,
                "rmse": rmse,
                "actions": actions,
            }
        rows.append(
            {
                "case": float(case_index),
                **{key: float(value) for key, value in asdict(case).items()},
                "uncontrolled_cost": float(controller_records["uncontrolled"]["cost"]),
                "oracle_cost": float(controller_records["oracle"]["cost"]),
                "uncontrolled_effort": float(controller_records["uncontrolled"]["effort"]),
                "oracle_effort": float(controller_records["oracle"]["effort"]),
                "uncontrolled_failure": float(controller_records["uncontrolled"]["failed"]),
                "oracle_failure": float(controller_records["oracle"]["failed"]),
            }
        )
        if case_index == 0:
            representative = {
                "time": (np.arange(rollout_horizon + 1) * solver.config.control_dt).tolist(),
                "uncontrolled_rmse": controller_records["uncontrolled"]["rmse"],
                "oracle_rmse": controller_records["oracle"]["rmse"],
                "oracle_actions": controller_records["oracle"]["actions"],
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "case_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    uncontrolled = np.asarray([row["uncontrolled_cost"] for row in rows])
    oracle = np.asarray([row["oracle_cost"] for row in rows])
    uncontrolled_effort = np.asarray([row["uncontrolled_effort"] for row in rows])
    oracle_effort = np.asarray([row["oracle_effort"] for row in rows])
    uncontrolled_failure = np.asarray([row["uncontrolled_failure"] for row in rows])
    oracle_failure = np.asarray([row["oracle_failure"] for row in rows])
    summary: dict[str, object] = {
        "claim": "persistent-forcing task requires active control",
        "oracle_definition": "MPC with direct numerical-PDE model access and finite-budget CEM",
        "test_cases": cases,
        "rollout_horizon": rollout_horizon,
        "physical_horizon": rollout_horizon * solver.config.control_dt,
        "test_seed": seed,
        "bootstrap_replicates": bootstrap_replicates,
        "plant_config": asdict(solver.config),
        "cem_config": asdict(cem),
        "test_distribution": {
            "viscosity": [0.008, 0.018],
            "boundary_each": [-0.03, 0.03],
            "actuator_gain": [0.65, 1.35],
            "forcing_amplitude": [0.40, 0.80],
            "forcing_frequency": [0.50, 1.50],
            "forcing_phase": [0.0, float(2.0 * np.pi)],
            "initial_amplitude": [0.03, 0.12],
        },
        "uncontrolled": _controller_summary(
            uncontrolled,
            uncontrolled_effort,
            uncontrolled_failure,
            seed + 101,
            bootstrap_replicates,
        ),
        "pde_oracle_mpc": _controller_summary(
            oracle,
            oracle_effort,
            oracle_failure,
            seed + 102,
            bootstrap_replicates,
        ),
        "paired": _paired_bootstrap(
            uncontrolled,
            oracle,
            seed + 103,
            bootstrap_replicates,
        ),
        "limitations": [
            "Oracle denotes access to the numerical transition model, not a globally optimal policy.",
            "Finite-budget CEM can underestimate the benefit attainable with exact optimization.",
            "This task-validation experiment precedes learned-FNO comparisons.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    assert representative is not None
    (output_dir / "representative_trajectory.json").write_text(
        json.dumps(representative, indent=2),
        encoding="utf-8",
    )
    _make_figures(output_dir, rows, representative, summary["paired"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/forced_oracle_validation"))
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--rollout-horizon", type=int, default=20)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--bootstrap-replicates", type=int, default=5_000)
    parser.add_argument("--cem-horizon", type=int, default=6)
    parser.add_argument("--cem-candidates", type=int, default=64)
    parser.add_argument("--cem-elites", type=int, default=8)
    parser.add_argument("--cem-iterations", type=int, default=3)
    args = parser.parse_args()
    summary = run_task_validation(
        output_dir=args.output_dir,
        cases=args.cases,
        rollout_horizon=args.rollout_horizon,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
        cem_config=OracleCEMConfig(
            horizon=args.cem_horizon,
            candidates=args.cem_candidates,
            elites=args.cem_elites,
            iterations=args.cem_iterations,
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
