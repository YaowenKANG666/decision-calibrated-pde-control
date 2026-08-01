"""Reward/value-gap experiments for dynamics-model error propagation.

The module contains three deliberately separate pieces of evidence:

1. an analytic sharpness witness attaining the fixed-policy
   ``epsilon / (1-gamma)^2`` rate;
2. a controlled Burgers misspecification with exactly known one-step error;
3. an optional learned-FNO audit on the finite set of visited rollout states.

Only the first item is a theorem-level construction.  Lipschitz and error
constants in the Burgers/FNO audits are trajectory-local empirical maxima.
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
import torch

from .models import load_perturbation_world_model
from .pde import BurgersSolver


def normalized_l2(value: np.ndarray) -> float:
    """Discrete function-space L2 norm used by the theory and experiments."""

    return float(np.sqrt(np.mean(np.square(np.asarray(value, dtype=float)))))


def discounted_value(rewards: np.ndarray, gamma: float) -> float:
    """Return the finite-horizon discounted value of a reward sequence."""

    rewards = np.asarray(rewards, dtype=float)
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must lie strictly between zero and one")
    return float(np.dot(np.power(gamma, np.arange(rewards.size)), rewards))


def fixed_policy_infinite_bound(
    epsilon: float,
    gamma: float,
    reward_lipschitz: float,
    dynamics_lipschitz: float,
) -> float:
    r"""Infinite-horizon fixed-policy value bound.

    This requires ``gamma * dynamics_lipschitz < 1`` and equals

    ``gamma L_r epsilon / ((1-gamma)(1-gamma L_G))``.
    """

    if epsilon < 0.0 or reward_lipschitz < 0.0 or dynamics_lipschitz < 0.0:
        raise ValueError("epsilon and Lipschitz constants must be nonnegative")
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must lie strictly between zero and one")
    if gamma * dynamics_lipschitz >= 1.0:
        raise ValueError("the infinite-horizon formula requires gamma * L_G < 1")
    return float(
        gamma
        * reward_lipschitz
        * epsilon
        / ((1.0 - gamma) * (1.0 - gamma * dynamics_lipschitz))
    )


def finite_horizon_value_bound(
    epsilon: float,
    gamma: float,
    reward_lipschitz: float,
    dynamics_lipschitz: float,
    horizon: int,
) -> float:
    r"""Compute ``L_r sum gamma^t e_t`` with ``e_{t+1}=L_G e_t+epsilon``."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    error_envelope = 0.0
    value_bound = 0.0
    for step in range(horizon):
        value_bound += (gamma**step) * reward_lipschitz * error_envelope
        error_envelope = dynamics_lipschitz * error_envelope + epsilon
    return float(value_bound)


def simulate_sharpness_witness(
    epsilon: float,
    gamma: float,
    dynamics_lipschitz: float,
    reward_lipschitz: float,
    horizon: int,
) -> float:
    r"""Simulate the scalar construction attaining the fixed-policy bound.

    True and learned dynamics are ``G(x)=L_G x`` and
    ``G_hat(x)=L_G x+epsilon`` from ``x_0=0``.  On the resulting nonnegative
    trajectory, ``r(x)=-L_r |x|`` is ``L_r``-Lipschitz and makes the bound an
    equality as the horizon tends to infinity.
    """

    true_state = 0.0
    model_state = 0.0
    true_rewards = np.empty(horizon, dtype=float)
    model_rewards = np.empty(horizon, dtype=float)
    for step in range(horizon):
        true_rewards[step] = -reward_lipschitz * abs(true_state)
        model_rewards[step] = -reward_lipschitz * abs(model_state)
        true_state = dynamics_lipschitz * true_state
        model_state = dynamics_lipschitz * model_state + epsilon
    return abs(
        discounted_value(true_rewards, gamma)
        - discounted_value(model_rewards, gamma)
    )


@dataclass(frozen=True)
class ValueGapExperimentConfig:
    seed: int = 27
    analytic_horizon: int = 5000
    burgers_horizon: int = 40
    burgers_cases: int = 10
    reward_lipschitz: float = 1.0
    analytic_epsilon: tuple[float, ...] = (
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
        3e-2,
    )
    burgers_epsilon: tuple[float, ...] = (
        5e-4,
        1e-3,
        2e-3,
        4e-3,
        8e-3,
        1.6e-2,
    )
    gammas: tuple[float, ...] = (0.80, 0.90, 0.95, 0.97, 0.98, 0.99)


def _write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analytic_scaling_rows(config: ValueGapExperimentConfig) -> list[dict[str, float]]:
    """Generate exact and simulated sharpness-witness data."""

    rows: list[dict[str, float]] = []
    for dynamics_lipschitz in (0.80, 1.00):
        for epsilon in config.analytic_epsilon:
            for gamma in config.gammas:
                exact = fixed_policy_infinite_bound(
                    epsilon,
                    gamma,
                    config.reward_lipschitz,
                    dynamics_lipschitz,
                )
                simulated = simulate_sharpness_witness(
                    epsilon,
                    gamma,
                    dynamics_lipschitz,
                    config.reward_lipschitz,
                    config.analytic_horizon,
                )
                rows.append(
                    {
                        "epsilon": epsilon,
                        "gamma": gamma,
                        "dynamics_lipschitz": dynamics_lipschitz,
                        "reward_lipschitz": config.reward_lipschitz,
                        "simulated_value_gap": simulated,
                        "exact_fixed_policy_bound": exact,
                        "optimal_policy_bound": 2.0 * exact,
                        "simulation_to_exact_ratio": simulated / exact,
                        "normalized_gamma_gap": exact / (gamma * epsilon),
                    }
                )
    return rows


def _bias_direction(solver: BurgersSolver) -> np.ndarray:
    direction = np.sin(np.pi * solver.x) + 0.35 * np.sin(3.0 * np.pi * solver.x)
    direction[0] = direction[-1] = 0.0
    return direction / normalized_l2(direction)


def _stage_reward(solver: BurgersSolver, state: np.ndarray, action: float) -> float:
    return -solver.stage_cost(state, action)


def _sample_burgers_case(
    solver: BurgersSolver,
    rng: np.random.Generator,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    viscosity = float(rng.uniform(0.007, 0.012))
    left = float(rng.uniform(-0.12, 0.12))
    right = float(rng.uniform(-0.12, 0.12))
    actuator_gain = float(rng.uniform(0.55, 1.45))
    state = solver.random_state(rng, left, right, (0.25, 0.65))
    raw_actions = rng.normal(0.0, 0.65, horizon)
    # A mild temporal filter avoids an artificial white-noise control signal.
    actions = np.empty(horizon, dtype=float)
    actions[0] = raw_actions[0]
    for step in range(1, horizon):
        actions[step] = 0.72 * actions[step - 1] + 0.28 * raw_actions[step]
    actions = np.clip(actions, -solver.config.action_limit, solver.config.action_limit)
    return state, actions, viscosity, left, right, actuator_gain


def controlled_burgers_rows(
    config: ValueGapExperimentConfig,
    solver: BurgersSolver,
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
    """Audit value propagation for an exactly epsilon-misspecified PDE model."""

    rng = np.random.default_rng(config.seed)
    direction = _bias_direction(solver)
    cases = [
        _sample_burgers_case(solver, rng, config.burgers_horizon)
        for _ in range(config.burgers_cases)
    ]
    value_rows: list[dict[str, float | int]] = []
    rollout_rows: list[dict[str, float | int]] = []
    for epsilon in config.burgers_epsilon:
        for case_index, case in enumerate(cases):
            state0, actions, viscosity, left, right, gain = case
            true_state = state0.copy()
            model_state = state0.copy()
            true_rewards = np.empty(config.burgers_horizon, dtype=float)
            model_rewards = np.empty_like(true_rewards)
            errors = np.empty(config.burgers_horizon + 1, dtype=float)
            errors[0] = 0.0
            local_ratios: list[float] = []
            max_abs_state = float(np.max(np.abs(state0)))
            for step, action in enumerate(actions):
                true_rewards[step] = _stage_reward(solver, true_state, float(action))
                model_rewards[step] = _stage_reward(solver, model_state, float(action))
                true_nominal_next = solver.step(
                    true_state, float(action), viscosity, left, right, gain
                )
                model_nominal_next = solver.step(
                    model_state, float(action), viscosity, left, right, gain
                )
                denominator = normalized_l2(true_state - model_state)
                if denominator > 1e-10:
                    local_ratios.append(
                        normalized_l2(true_nominal_next - model_nominal_next)
                        / denominator
                    )
                model_next = model_nominal_next + epsilon * direction
                model_next[0], model_next[-1] = left, right
                true_state, model_state = true_nominal_next, model_next
                errors[step + 1] = normalized_l2(true_state - model_state)
                max_abs_state = max(
                    max_abs_state,
                    float(np.max(np.abs(true_state))),
                    float(np.max(np.abs(model_state))),
                )

            empirical_lipschitz = max(local_ratios, default=0.0)
            # For r(x,a)=-mean(w*x^2)-c*a^2, w<=1 and actions are shared:
            # |r(x,a)-r(y,a)| <= 2 B_inf ||x-y||_{2,n}.
            reward_lipschitz = 2.0 * max_abs_state
            envelope = 0.0
            for step, error in enumerate(errors):
                rollout_rows.append(
                    {
                        "epsilon": epsilon,
                        "case": case_index,
                        "step": step,
                        "state_error": error,
                        "error_envelope": envelope,
                        "empirical_dynamics_lipschitz": empirical_lipschitz,
                    }
                )
                envelope = empirical_lipschitz * envelope + epsilon
            for gamma in config.gammas:
                true_value = discounted_value(true_rewards, gamma)
                model_value = discounted_value(model_rewards, gamma)
                bound = finite_horizon_value_bound(
                    epsilon,
                    gamma,
                    reward_lipschitz,
                    empirical_lipschitz,
                    config.burgers_horizon,
                )
                value_rows.append(
                    {
                        "epsilon": epsilon,
                        "gamma": gamma,
                        "case": case_index,
                        "horizon": config.burgers_horizon,
                        "true_value": true_value,
                        "model_value": model_value,
                        "absolute_value_gap": abs(true_value - model_value),
                        "finite_horizon_bound": bound,
                        "gap_to_bound_ratio": abs(true_value - model_value)
                        / max(bound, 1e-15),
                        "reward_lipschitz": reward_lipschitz,
                        "empirical_dynamics_lipschitz": empirical_lipschitz,
                        "max_rollout_state_error": float(np.max(errors)),
                    }
                )
    return value_rows, rollout_rows


@torch.no_grad()
def learned_fno_value_rows(
    config: ValueGapExperimentConfig,
    solver: BurgersSolver,
    checkpoint: Path,
) -> tuple[list[dict[str, float | int]], dict[str, str | float]]:
    """Audit a trained quick FNO on visited combined-shift rollout states.

    The returned epsilon and Lipschitz constants are maxima only over the
    finite visited set.  Consequently the computed envelope is a diagnostic,
    not a certified global theorem bound.
    """

    if not checkpoint.exists():
        return [], {"status": "skipped", "reason": "checkpoint_not_found"}
    device = torch.device("cpu")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("model_kind") != "fno":
        return [], {"status": "skipped", "reason": "checkpoint_is_not_fno"}
    model, payload = load_perturbation_world_model(checkpoint, device)

    rng = np.random.default_rng(config.seed + 91_003)
    rows: list[dict[str, float | int]] = []
    for case_index in range(config.burgers_cases):
        state0, actions, viscosity, left, right, gain = _sample_burgers_case(
            solver, rng, config.burgers_horizon
        )
        true_state = state0.copy()
        model_state = state0.copy()
        true_rewards = np.empty(config.burgers_horizon, dtype=float)
        model_rewards = np.empty_like(true_rewards)
        one_step_errors: list[float] = []
        local_ratios: list[float] = []
        max_abs_state = float(np.max(np.abs(state0)))
        for step, action in enumerate(actions):
            true_rewards[step] = _stage_reward(solver, true_state, float(action))
            model_rewards[step] = _stage_reward(solver, model_state, float(action))

            true_next = solver.step(true_state, float(action), viscosity, left, right, gain)
            model_true_next = solver.step(
                model_state, float(action), viscosity, left, right, gain
            )
            denominator = normalized_l2(true_state - model_state)
            if denominator > 1e-10:
                local_ratios.append(
                    normalized_l2(true_next - model_true_next) / denominator
                )

            state_tensor = torch.as_tensor(model_state[None, :], dtype=torch.float32)
            action_tensor = torch.tensor([action], dtype=torch.float32)
            viscosity_tensor = torch.tensor([viscosity], dtype=torch.float32)
            boundary_tensor = torch.tensor([[left, right]], dtype=torch.float32)
            predicted_next, _ = model(
                state_tensor,
                action_tensor,
                viscosity_tensor,
                boundary_tensor,
            )
            predicted_next_np = predicted_next[0].cpu().numpy().astype(float)
            # Error is evaluated where the learned rollout actually queries
            # the world model, matching the recursion's visited state set.
            one_step_errors.append(normalized_l2(model_true_next - predicted_next_np))
            true_state, model_state = true_next, predicted_next_np
            max_abs_state = max(
                max_abs_state,
                float(np.max(np.abs(true_state))),
                float(np.max(np.abs(model_state))),
            )

        epsilon = max(one_step_errors)
        empirical_lipschitz = max(local_ratios, default=0.0)
        reward_lipschitz = 2.0 * max_abs_state
        for gamma in config.gammas:
            true_value = discounted_value(true_rewards, gamma)
            model_value = discounted_value(model_rewards, gamma)
            bound = finite_horizon_value_bound(
                epsilon,
                gamma,
                reward_lipschitz,
                empirical_lipschitz,
                config.burgers_horizon,
            )
            rows.append(
                {
                    "gamma": gamma,
                    "case": case_index,
                    "horizon": config.burgers_horizon,
                    "empirical_epsilon": epsilon,
                    "true_value": true_value,
                    "model_value": model_value,
                    "absolute_value_gap": abs(true_value - model_value),
                    "finite_horizon_local_envelope": bound,
                    "gap_to_envelope_ratio": abs(true_value - model_value)
                    / max(bound, 1e-15),
                    "reward_lipschitz": reward_lipschitz,
                    "empirical_dynamics_lipschitz": empirical_lipschitz,
                }
            )
    return rows, {
        "status": "completed",
        "checkpoint": checkpoint.name,
        "guarantee": "finite_visited_set_empirical_audit_only",
    }


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.polyfit(np.log(np.asarray(x)), np.log(np.asarray(y)), 1)[0])


def _aggregate_burgers(
    rows: list[dict[str, float | int]], gamma: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    epsilon_values = sorted({float(row["epsilon"]) for row in rows})
    means, maxima, bounds = [], [], []
    for epsilon in epsilon_values:
        selected = [
            row
            for row in rows
            if float(row["epsilon"]) == epsilon and float(row["gamma"]) == gamma
        ]
        gaps = np.asarray([float(row["absolute_value_gap"]) for row in selected])
        local_bounds = np.asarray([float(row["finite_horizon_bound"]) for row in selected])
        means.append(float(np.mean(gaps)))
        maxima.append(float(np.max(gaps)))
        bounds.append(float(np.max(local_bounds)))
    return (
        np.asarray(epsilon_values),
        np.asarray(means),
        np.asarray(maxima),
        np.asarray(bounds),
    )


def _value_figure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def _save_value_figure(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def make_individual_value_figures(
    analytic_rows: list[dict[str, float]],
    burgers_rows: list[dict[str, float | int]],
    rollout_rows: list[dict[str, float | int]],
    figure_dir: Path,
) -> list[str]:
    """Export four standalone plots, one conclusion per file."""

    _value_figure_style()
    navy, blue, orange, red, grey = "#3A5A78", "#77A6C5", "#D8843F", "#B9544D", "#777777"
    gamma_fixed = 0.95
    epsilon_fixed = 0.01
    stems: list[str] = []

    selected = [
        row for row in analytic_rows
        if row["gamma"] == gamma_fixed and row["dynamics_lipschitz"] == 1.0
    ]
    epsilon = np.asarray([row["epsilon"] for row in selected])
    gap = np.asarray([row["exact_fixed_policy_bound"] for row in selected])
    figure, axis = plt.subplots(figsize=(3.45, 2.75), constrained_layout=True)
    axis.loglog(epsilon, gap, "o-", color=navy, lw=1.6, ms=4.2)
    axis.text(0.05, 0.93, f"log–log slope = {_linear_slope(epsilon, gap):.3f}", transform=axis.transAxes, va="top", color=navy)
    axis.set_xlabel(r"One-step error $\epsilon$")
    axis.set_ylabel(r"$|V_G-V_{\widehat G}|$")
    axis.set_title(r"Linear dependence on $\epsilon$")
    axis.grid(alpha=0.17, which="both")
    _save_value_figure(figure, figure_dir / "value_01_epsilon_scaling")
    stems.append("value_01_epsilon_scaling")

    selected = [
        row for row in analytic_rows
        if row["epsilon"] == epsilon_fixed and row["dynamics_lipschitz"] == 1.0
    ]
    gamma = np.asarray([row["gamma"] for row in selected])
    horizon = 1.0 / (1.0 - gamma)
    normalized_gap = np.asarray([row["normalized_gamma_gap"] for row in selected])
    figure, axis = plt.subplots(figsize=(3.45, 2.75), constrained_layout=True)
    axis.loglog(horizon, normalized_gap, "o-", color=orange, lw=1.7, ms=4.2, label=r"$|\Delta V|/(\gamma\epsilon)$")
    axis.loglog(horizon, horizon**2, "--", color=grey, lw=1.0, label=r"$(1-\gamma)^{-2}$")
    axis.text(0.05, 0.93, f"log–log slope = {_linear_slope(horizon, normalized_gap):.3f}", transform=axis.transAxes, va="top", color=orange)
    axis.set_xlabel(r"Effective horizon $(1-\gamma)^{-1}$")
    axis.set_ylabel("Normalized value gap")
    axis.set_title(r"Sharp $(1-\gamma)^{-2}$ scaling")
    axis.legend(fontsize=6.6)
    axis.grid(alpha=0.17, which="both")
    _save_value_figure(figure, figure_dir / "value_02_discount_scaling")
    stems.append("value_02_discount_scaling")

    eps, mean_gap, max_gap, max_bound = _aggregate_burgers(burgers_rows, gamma_fixed)
    figure, axis = plt.subplots(figsize=(3.45, 2.75), constrained_layout=True)
    axis.loglog(eps, mean_gap, "o-", color=blue, lw=1.5, ms=4, label="Mean observed gap")
    axis.loglog(eps, max_gap, "s-", color=navy, lw=1.3, ms=3.6, label="Maximum observed gap")
    axis.loglog(eps, max_bound, "--", color=red, lw=1.4, label="Finite-horizon envelope")
    axis.set_xlabel(r"Injected PDE error $\epsilon$")
    axis.set_ylabel("Absolute discounted-value gap")
    axis.set_title("Controlled Burgers misspecification")
    axis.legend(fontsize=6.5)
    axis.grid(alpha=0.17, which="both")
    _save_value_figure(figure, figure_dir / "value_03_burgers_gap_bound")
    stems.append("value_03_burgers_gap_bound")

    largest = float(np.max(eps))
    selected = [row for row in rollout_rows if float(row["epsilon"]) == largest and int(row["case"]) == 0]
    steps = np.asarray([row["step"] for row in selected])
    state_error = np.asarray([row["state_error"] for row in selected])
    envelope = np.asarray([row["error_envelope"] for row in selected])
    figure, axis = plt.subplots(figsize=(3.45, 2.75), constrained_layout=True)
    axis.plot(steps, state_error, color=navy, lw=1.7, label="Observed rollout error")
    axis.plot(steps, envelope, "--", color=red, lw=1.4, label="Recursive envelope")
    axis.fill_between(steps, state_error, envelope, color=red, alpha=0.10, lw=0)
    axis.set_xlabel("Rollout step")
    axis.set_ylabel(r"State error $\|x_t-\widehat x_t\|_{2,n}$")
    axis.set_title("Multi-step propagation audit")
    axis.legend(fontsize=6.5)
    axis.grid(alpha=0.17)
    _save_value_figure(figure, figure_dir / "value_04_rollout_error")
    stems.append("value_04_rollout_error")
    return stems


def make_individual_learned_figures(
    learned_rows: list[dict[str, float | int]],
    figure_dir: Path,
) -> list[str]:
    if not learned_rows:
        return []
    _value_figure_style()
    navy, blue, orange, grey = "#3A5A78", "#77A6C5", "#D8843F", "#777777"
    gammas = sorted({float(row["gamma"]) for row in learned_rows})
    mean_gap, max_gap, max_envelope = [], [], []
    for gamma in gammas:
        selected = [row for row in learned_rows if float(row["gamma"]) == gamma]
        gaps = np.asarray([float(row["absolute_value_gap"]) for row in selected])
        envelopes = np.asarray([float(row["finite_horizon_local_envelope"]) for row in selected])
        mean_gap.append(float(np.mean(gaps)))
        max_gap.append(float(np.max(gaps)))
        max_envelope.append(float(np.max(envelopes)))
    figure, axis = plt.subplots(figsize=(3.45, 2.75), constrained_layout=True)
    axis.plot(gammas, mean_gap, "o-", color=blue, lw=1.6, ms=4, label="Mean gap")
    axis.plot(gammas, max_gap, "s-", color=navy, lw=1.4, ms=3.7, label="Maximum gap")
    axis.plot(gammas, max_envelope, "--", color=orange, lw=1.4, label="Maximum local envelope")
    axis.set_xlabel(r"Discount factor $\gamma$")
    axis.set_ylabel("Absolute discounted-value gap")
    axis.set_yscale("log")
    axis.set_title("Learned FNO value error under joint shift")
    axis.legend(fontsize=6.5)
    axis.grid(alpha=0.17, which="both")
    _save_value_figure(figure, figure_dir / "value_05_fno_gap_discount")

    envelopes = np.asarray([float(row["finite_horizon_local_envelope"]) for row in learned_rows])
    gaps = np.asarray([float(row["absolute_value_gap"]) for row in learned_rows])
    gamma_points = np.asarray([float(row["gamma"]) for row in learned_rows])
    figure, axis = plt.subplots(figsize=(3.45, 2.9), constrained_layout=True)
    scatter = axis.scatter(envelopes, gaps, c=gamma_points, cmap="cividis", s=22, alpha=0.82, edgecolors="white", linewidths=0.35)
    positive = np.concatenate((envelopes[envelopes > 0], gaps[gaps > 0]))
    lower, upper = float(np.min(positive)) * 0.75, float(np.max(positive)) * 1.35
    axis.plot([lower, upper], [lower, upper], "--", color=grey, lw=1.0, label="Equality")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_xlabel("Finite-visited-set local envelope")
    axis.set_ylabel("Observed value gap")
    axis.set_title("Observed gap versus audit envelope")
    colorbar = figure.colorbar(scatter, ax=axis, pad=0.02, fraction=0.05)
    colorbar.set_label(r"Discount factor $\gamma$")
    axis.grid(alpha=0.17, which="both")
    _save_value_figure(figure, figure_dir / "value_06_fno_gap_envelope")
    return ["value_05_fno_gap_discount", "value_06_fno_gap_envelope"]


def make_publication_figure(
    analytic_rows: list[dict[str, float]],
    burgers_rows: list[dict[str, float | int]],
    rollout_rows: list[dict[str, float | int]],
    figure_stem: Path,
) -> None:
    """Create the quantitative-grid figure and publication exports."""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
        }
    )
    navy = "#3A5A78"
    blue = "#77A6C5"
    orange = "#D8843F"
    red = "#B9544D"
    grey = "#777777"

    figure = plt.figure(figsize=(7.25, 5.15), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.08, 1.0))
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])
    axis_c = figure.add_subplot(grid[1, 0])
    axis_d = figure.add_subplot(grid[1, 1])

    gamma_fixed = 0.95
    analytic_fixed = [
        row
        for row in analytic_rows
        if row["gamma"] == gamma_fixed and row["dynamics_lipschitz"] == 1.0
    ]
    epsilon = np.asarray([row["epsilon"] for row in analytic_fixed])
    gap = np.asarray([row["exact_fixed_policy_bound"] for row in analytic_fixed])
    axis_a.loglog(epsilon, gap, "o-", color=navy, lw=1.6, ms=4.2, label="Exact value gap")
    slope_epsilon = _linear_slope(epsilon, gap)
    axis_a.text(
        0.05,
        0.93,
        rf"log–log slope = {slope_epsilon:.3f}",
        transform=axis_a.transAxes,
        ha="left",
        va="top",
        color=navy,
    )
    axis_a.set_xlabel(r"One-step error $\epsilon$")
    axis_a.set_ylabel(r"$|V_G-V_{\widehat G}|$")
    axis_a.set_title(r"Linear dependence on $\epsilon$")

    epsilon_fixed = 0.01
    gamma_rows = [
        row
        for row in analytic_rows
        if row["epsilon"] == epsilon_fixed and row["dynamics_lipschitz"] == 1.0
    ]
    gamma = np.asarray([row["gamma"] for row in gamma_rows])
    inverse_discount = 1.0 / (1.0 - gamma)
    normalized_gap = np.asarray([row["normalized_gamma_gap"] for row in gamma_rows])
    reference = inverse_discount**2
    axis_b.loglog(
        inverse_discount,
        normalized_gap,
        "o-",
        color=orange,
        lw=1.8,
        ms=4.5,
        label=r"$|\Delta V|/(\gamma\epsilon)$",
    )
    axis_b.loglog(
        inverse_discount,
        reference,
        "--",
        color=grey,
        lw=1.1,
        label=r"$(1-\gamma)^{-2}$",
    )
    slope_gamma = _linear_slope(inverse_discount, normalized_gap)
    axis_b.text(
        0.05,
        0.93,
        rf"collapse slope = {slope_gamma:.3f}",
        transform=axis_b.transAxes,
        ha="left",
        va="top",
        color=orange,
    )
    axis_b.set_xlabel(r"Effective horizon $(1-\gamma)^{-1}$")
    axis_b.set_ylabel(r"Normalized value gap")
    axis_b.set_title(r"Sharp $(1-\gamma)^{-2}$ scaling")
    axis_b.legend(loc="lower right")

    burgers_epsilon, mean_gap, max_gap, max_bound = _aggregate_burgers(
        burgers_rows, gamma_fixed
    )
    axis_c.loglog(
        burgers_epsilon,
        mean_gap,
        "o-",
        color=blue,
        lw=1.5,
        ms=4,
        label="Mean observed gap",
    )
    axis_c.loglog(
        burgers_epsilon,
        max_gap,
        "s-",
        color=navy,
        lw=1.3,
        ms=3.6,
        label="Maximum observed gap",
    )
    axis_c.loglog(
        burgers_epsilon,
        max_bound,
        "--",
        color=red,
        lw=1.4,
        label="Finite-horizon envelope",
    )
    axis_c.set_xlabel(r"Injected PDE error $\epsilon$")
    axis_c.set_ylabel("Absolute discounted-value gap")
    axis_c.set_title("Controlled Burgers misspecification")
    axis_c.legend(loc="upper left", fontsize=6.6)

    largest_epsilon = float(np.max(burgers_epsilon))
    chosen = [
        row
        for row in rollout_rows
        if float(row["epsilon"]) == largest_epsilon and int(row["case"]) == 0
    ]
    steps = np.asarray([row["step"] for row in chosen])
    state_error = np.asarray([row["state_error"] for row in chosen])
    envelope = np.asarray([row["error_envelope"] for row in chosen])
    axis_d.plot(steps, state_error, color=navy, lw=1.7, label="Observed rollout error")
    axis_d.plot(steps, envelope, "--", color=red, lw=1.4, label="Recursive envelope")
    axis_d.fill_between(steps, state_error, envelope, color=red, alpha=0.10, lw=0)
    axis_d.set_xlabel("Rollout step")
    axis_d.set_ylabel(r"State error $\|x_t-\widehat x_t\|_{2,n}$")
    axis_d.set_title("Multi-step propagation audit")
    axis_d.legend(loc="upper left", fontsize=6.6)

    for label, axis in zip("abcd", (axis_a, axis_b, axis_c, axis_d)):
        axis.text(
            -0.16,
            1.08,
            label,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )
        axis.grid(alpha=0.17, which="both", linewidth=0.55)

    figure_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(figure_stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(figure_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def make_learned_fno_figure(
    learned_rows: list[dict[str, float | int]],
    figure_stem: Path,
) -> None:
    """Visualize the learned FNO's finite-visited-set value audit."""

    if not learned_rows:
        return
    navy = "#3A5A78"
    blue = "#77A6C5"
    orange = "#D8843F"
    grey = "#777777"
    gammas = sorted({float(row["gamma"]) for row in learned_rows})
    mean_gap, max_gap, max_envelope = [], [], []
    for gamma in gammas:
        selected = [row for row in learned_rows if float(row["gamma"]) == gamma]
        gaps = np.asarray([float(row["absolute_value_gap"]) for row in selected])
        envelopes = np.asarray(
            [float(row["finite_horizon_local_envelope"]) for row in selected]
        )
        mean_gap.append(float(np.mean(gaps)))
        max_gap.append(float(np.max(gaps)))
        max_envelope.append(float(np.max(envelopes)))

    figure, (axis_a, axis_b) = plt.subplots(
        1, 2, figsize=(7.25, 3.05), constrained_layout=True
    )
    axis_a.plot(gammas, mean_gap, "o-", color=blue, lw=1.6, ms=4, label="Mean gap")
    axis_a.plot(gammas, max_gap, "s-", color=navy, lw=1.4, ms=3.7, label="Maximum gap")
    axis_a.plot(
        gammas,
        max_envelope,
        "--",
        color=orange,
        lw=1.4,
        label="Maximum local envelope",
    )
    axis_a.set_xlabel(r"Discount factor $\gamma$")
    axis_a.set_ylabel("Absolute discounted-value gap")
    axis_a.set_yscale("log")
    axis_a.set_title("Learned FNO value error under joint shift")
    axis_a.legend(loc="upper left", fontsize=6.7)

    envelopes = np.asarray(
        [float(row["finite_horizon_local_envelope"]) for row in learned_rows]
    )
    gaps = np.asarray([float(row["absolute_value_gap"]) for row in learned_rows])
    gamma_points = np.asarray([float(row["gamma"]) for row in learned_rows])
    scatter = axis_b.scatter(
        envelopes,
        gaps,
        c=gamma_points,
        cmap="cividis",
        s=22,
        alpha=0.82,
        edgecolors="white",
        linewidths=0.35,
    )
    lower = min(float(np.min(envelopes)), float(np.min(gaps))) * 0.75
    upper = max(float(np.max(envelopes)), float(np.max(gaps))) * 1.35
    axis_b.plot([lower, upper], [lower, upper], "--", color=grey, lw=1.0, label="Equality")
    axis_b.set_xscale("log")
    axis_b.set_yscale("log")
    axis_b.set_xlim(lower, upper)
    axis_b.set_ylim(lower, upper)
    axis_b.set_xlabel("Finite-visited-set local envelope")
    axis_b.set_ylabel("Observed value gap")
    axis_b.set_title("Observed gaps remain below the audit envelope")
    colorbar = figure.colorbar(scatter, ax=axis_b, pad=0.02, fraction=0.05)
    colorbar.set_label(r"Discount factor $\gamma$")

    for label, axis in zip("ab", (axis_a, axis_b)):
        axis.text(
            -0.15,
            1.07,
            label,
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )
        axis.grid(alpha=0.17, which="both", linewidth=0.55)
    figure_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(figure_stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(figure_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def run_experiment(
    output_root: Path,
    checkpoint: Path | None = None,
    config: ValueGapExperimentConfig | None = None,
) -> dict[str, object]:
    """Run all value-gap experiments and persist data, summaries, and figures."""

    cfg = config or ValueGapExperimentConfig()
    data_dir = output_root / "data"
    figure_dir = output_root / "figures"
    result_dir = output_root / "results"
    for directory in (data_dir, figure_dir, result_dir):
        directory.mkdir(parents=True, exist_ok=True)

    analytic_rows = analytic_scaling_rows(cfg)
    solver = BurgersSolver()
    burgers_rows, rollout_rows = controlled_burgers_rows(cfg, solver)
    learned_rows: list[dict[str, float | int]] = []
    learned_status: dict[str, str | float] = {
        "status": "skipped",
        "reason": "no_checkpoint_requested",
    }
    if checkpoint is not None:
        learned_rows, learned_status = learned_fno_value_rows(cfg, solver, checkpoint)

    _write_rows(data_dir / "analytic_sharpness.csv", analytic_rows)
    _write_rows(data_dir / "burgers_value_gap.csv", burgers_rows)
    _write_rows(data_dir / "burgers_rollout_error.csv", rollout_rows)
    if learned_rows:
        _write_rows(data_dir / "learned_fno_value_gap.csv", learned_rows)
    individual_value_figures = make_individual_value_figures(
        analytic_rows,
        burgers_rows,
        rollout_rows,
        figure_dir,
    )
    individual_learned_figures = make_individual_learned_figures(
        learned_rows,
        figure_dir,
    )

    analytic_epsilon_slice = [
        row
        for row in analytic_rows
        if row["gamma"] == 0.95 and row["dynamics_lipschitz"] == 1.0
    ]
    analytic_gamma_slice = [
        row
        for row in analytic_rows
        if row["epsilon"] == 0.01 and row["dynamics_lipschitz"] == 1.0
    ]
    epsilon_slope = _linear_slope(
        np.asarray([row["epsilon"] for row in analytic_epsilon_slice]),
        np.asarray([row["exact_fixed_policy_bound"] for row in analytic_epsilon_slice]),
    )
    gamma_slope = _linear_slope(
        1.0 / (1.0 - np.asarray([row["gamma"] for row in analytic_gamma_slice])),
        np.asarray([row["normalized_gamma_gap"] for row in analytic_gamma_slice]),
    )
    maximum_burgers_ratio = max(float(row["gap_to_bound_ratio"]) for row in burgers_rows)
    learned_summary: dict[str, object] = dict(learned_status)
    if learned_rows:
        learned_ratios = np.asarray(
            [float(row["gap_to_envelope_ratio"]) for row in learned_rows]
        )
        learned_epsilon = np.asarray(
            [float(row["empirical_epsilon"]) for row in learned_rows]
        )
        learned_gaps = np.asarray(
            [float(row["absolute_value_gap"]) for row in learned_rows]
        )
        learned_summary.update(
            {
                "maximum_gap_to_local_envelope_ratio": float(np.max(learned_ratios)),
                "mean_gap_to_local_envelope_ratio": float(np.mean(learned_ratios)),
                "empirical_epsilon_range": [
                    float(np.min(learned_epsilon)),
                    float(np.max(learned_epsilon)),
                ],
                "observed_value_gap_range": [
                    float(np.min(learned_gaps)),
                    float(np.max(learned_gaps)),
                ],
            }
        )
    summary: dict[str, object] = {
        "config": asdict(cfg),
        "analytic": {
            "epsilon_log_log_slope": epsilon_slope,
            "discount_horizon_log_log_slope": gamma_slope,
            "discount_slope_response": "absolute_value_gap / (gamma * epsilon)",
            "max_simulation_relative_error": max(
                abs(float(row["simulation_to_exact_ratio"]) - 1.0)
                for row in analytic_rows
            ),
            "interpretation": (
                "The scalar Lipschitz witness attains the fixed-policy bound; "
                "the generic optimal-policy regret theorem is twice this value."
            ),
        },
        "controlled_burgers": {
            "maximum_gap_to_finite_horizon_bound_ratio": maximum_burgers_ratio,
            "cases": cfg.burgers_cases,
            "horizon": cfg.burgers_horizon,
            "guarantee": (
                "Exact injected one-step error with a trajectory-local secant "
                "Lipschitz envelope; not a global PDE certificate."
            ),
        },
        "learned_fno": learned_summary,
        "artifacts": {
            "analytic_data": "data/analytic_sharpness.csv",
            "burgers_value_data": "data/burgers_value_gap.csv",
            "burgers_rollout_data": "data/burgers_rollout_error.csv",
            "learned_fno_data": (
                "data/learned_fno_value_gap.csv" if learned_rows else None
            ),
            "individual_value_figures": [f"figures/{name}" for name in individual_value_figures],
            "individual_learned_fno_figures": [
                f"figures/{name}" for name in individual_learned_figures
            ],
        },
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/reward_value_gap"),
    )
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    summary = run_experiment(args.output_root, args.checkpoint)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
