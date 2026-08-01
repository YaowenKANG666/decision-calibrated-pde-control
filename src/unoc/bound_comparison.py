"""Independent calibration/test comparison of four discounted-value bounds.

The experiment compares:

``global``
    one trajectory-wide epsilon and Lipschitz recursion;
``local``
    state-dependent uncertainty radii with the same global expansion factor;
``adjoint``
    first-order support of the anisotropic ambiguity set;
``adjoint_curvature``
    adjoint support plus an audit-calibrated quadratic remainder proxy.

Every method is finally conformalized at the value level on a calibration
split. This produces a fair coverage--tightness comparison on an independent
test split. The curvature coefficient and empirical dynamics constants are not
claimed to be uniform analytic certificates.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from .calibration import _higher_quantile
from .models import load_perturbation_world_model
from .pde import BurgersSolver

METHODS = ("global", "local", "adjoint", "adjoint_curvature")
METHOD_LABELS = {
    "global": "Global max",
    "local": "Local recursion",
    "adjoint": "Adjoint support",
    "adjoint_curvature": "Adjoint + curvature",
}
COLORS = {
    "global": "#8A8A8A",
    "local": "#77A6C5",
    "adjoint": "#3A5A78",
    "adjoint_curvature": "#D8843F",
}


def norm2(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value))))


def stage_reward_tensor(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    grid = state.shape[-1]
    x = torch.linspace(0.0, 1.0, grid, dtype=state.dtype, device=state.device)
    weight = 0.15 + 0.85 * torch.exp(-0.5 * ((x - 0.72) / 0.14).square())
    return -torch.mean(weight[None, :] * state.square(), dim=1) - 0.002 * action.square()


@dataclass
class RolloutRecord:
    observed_gap: float
    max_one_step_error: float
    max_l2_score: float
    max_ellipsoid_score: float
    max_secant_lipschitz: float
    reward_lipschitz: float
    scale_l2: list[float]
    adjoint_unit_support: float
    curvature_unit_feature: float


def sample_case(solver, rng, horizon):
    viscosity = float(rng.uniform(0.007, 0.012))
    left = float(rng.uniform(-0.12, 0.12))
    right = float(rng.uniform(-0.12, 0.12))
    gain = float(rng.uniform(0.55, 1.45))
    state = solver.random_state(rng, left, right, (0.25, 0.65))
    white = rng.normal(0.0, 0.65, horizon)
    actions = np.empty(horizon)
    actions[0] = white[0]
    for step in range(1, horizon):
        actions[step] = 0.72 * actions[step - 1] + 0.28 * white[step]
    return state, np.clip(actions, -2.0, 2.0), viscosity, left, right, gain


def model_step(model, state, action, viscosity, left, right, device):
    state_t = torch.as_tensor(state[None, :], dtype=torch.float32, device=device)
    action_t = torch.tensor([action], dtype=torch.float32, device=device)
    viscosity_t = torch.tensor([viscosity], dtype=torch.float32, device=device)
    boundary_t = torch.tensor([[left, right]], dtype=torch.float32, device=device)
    with torch.no_grad():
        mean, scale = model(state_t, action_t, viscosity_t, boundary_t)
    return mean[0].cpu().numpy(), scale[0].cpu().numpy()


def adjoint_unit_support(model, state0, actions, viscosity, left, right, gamma, device):
    state = torch.as_tensor(state0[None, :], dtype=torch.float32, device=device)
    actions_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
    viscosity_t = torch.tensor([viscosity], dtype=torch.float32, device=device)
    boundary_t = torch.tensor([[left, right]], dtype=torch.float32, device=device)
    perturbations, scales = [], []
    value = torch.zeros(1, dtype=state.dtype, device=device)
    with torch.enable_grad():
        for step, action in enumerate(actions_t):
            mean, scale = model(
                state,
                action.reshape(1),
                viscosity_t,
                boundary_t,
            )
            delta = torch.zeros_like(mean, requires_grad=True)
            perturbations.append(delta)
            scales.append(scale)
            state = mean + delta
            value = value + (gamma**step) * stage_reward_tensor(
                state, action.reshape(1)
            )
        gradients = torch.autograd.grad(value.sum(), perturbations)
    support = 0.0
    curvature_feature = 0.0
    grid = state.shape[1]
    for step, (gradient, scale) in enumerate(zip(gradients, scales)):
        normalized_gradient = grid * gradient
        support += float(
            torch.sqrt(torch.mean((normalized_gradient * scale).square())).detach().cpu()
        )
        scale_norm = float(torch.sqrt(torch.mean(scale.square())).detach().cpu())
        curvature_feature += 0.5 * (gamma**step) * scale_norm**2
    return support, curvature_feature


def evaluate_case(model, solver, case, horizon, gamma, device):
    state0, actions, viscosity, left, right, gain = case
    true_state = state0.copy()
    learned_state = state0.copy()
    true_rewards, model_rewards = [], []
    one_step_errors, l2_scores, ellipsoid_scores, scales = [], [], [], []
    secants = []
    max_abs_state = float(np.max(np.abs(state0)))
    for action in actions:
        true_next = solver.step(true_state, float(action), viscosity, left, right, gain)
        true_from_model_state = solver.step(
            learned_state, float(action), viscosity, left, right, gain
        )
        mean, scale = model_step(
            model, learned_state, float(action), viscosity, left, right, device
        )
        residual = true_from_model_state - mean
        scale_floor = np.maximum(scale, 1e-6)
        scale_l2 = norm2(scale_floor)
        one_step_errors.append(norm2(residual))
        l2_scores.append(norm2(residual) / max(scale_l2, 1e-8))
        ellipsoid_scores.append(norm2(residual / scale_floor))
        scales.append(scale_l2)
        denominator = norm2(true_state - learned_state)
        if denominator > 1e-9:
            secants.append(norm2(true_next - true_from_model_state) / denominator)
        true_state, learned_state = true_next, mean
        true_rewards.append(-solver.stage_cost(true_state, float(action)))
        model_rewards.append(-solver.stage_cost(learned_state, float(action)))
        max_abs_state = max(
            max_abs_state,
            float(np.max(np.abs(true_state))),
            float(np.max(np.abs(learned_state))),
        )
    discounts = gamma ** np.arange(horizon)
    gap = abs(float(np.dot(discounts, true_rewards) - np.dot(discounts, model_rewards)))
    adjoint, curvature = adjoint_unit_support(
        model, state0, actions, viscosity, left, right, gamma, device
    )
    return RolloutRecord(
        observed_gap=gap,
        max_one_step_error=max(one_step_errors),
        max_l2_score=max(l2_scores),
        max_ellipsoid_score=max(ellipsoid_scores),
        max_secant_lipschitz=max(secants, default=1.0),
        reward_lipschitz=2.0 * max_abs_state,
        scale_l2=scales,
        adjoint_unit_support=adjoint,
        curvature_unit_feature=curvature,
    )


def recursion_bound(radii, lipschitz, reward_lipschitz, gamma):
    envelope = 0.0
    bound = 0.0
    for step, radius in enumerate(radii):
        envelope = lipschitz * envelope + radius
        bound += (gamma**step) * reward_lipschitz * envelope
    return float(bound)


def raw_bounds(records, constants):
    raw = []
    for record in records:
        global_bound = recursion_bound(
            [constants["epsilon_global"]] * len(record.scale_l2),
            constants["lipschitz_global"],
            constants["reward_lipschitz_global"],
            constants["gamma"],
        )
        local_bound = recursion_bound(
            [constants["q_l2"] * scale for scale in record.scale_l2],
            constants["lipschitz_global"],
            constants["reward_lipschitz_global"],
            constants["gamma"],
        )
        adjoint = constants["q_ellipsoid"] * record.adjoint_unit_support
        curvature_feature = constants["q_l2"] ** 2 * record.curvature_unit_feature
        raw.append(
            {
                "global": max(global_bound, 1e-12),
                "local": max(local_bound, 1e-12),
                "adjoint": max(adjoint, 1e-12),
                "curvature_feature": max(curvature_feature, 1e-12),
            }
        )
    curvature_scores = [
        max(record.observed_gap - item["adjoint"], 0.0) / item["curvature_feature"]
        for record, item in zip(records, raw)
    ]
    return raw, curvature_scores


def conformal_scale(records, raw, method, coverage):
    scores = np.asarray(
        [
            record.observed_gap / max(item[method], 1e-12)
            for record, item in zip(records, raw)
        ]
    )
    return _higher_quantile(scores, coverage)


def summarize(test_records, test_raw, multipliers, target):
    rows = []
    for method in METHODS:
        bounds = np.asarray(
            [multipliers[method] * item[method] for item in test_raw], dtype=float
        )
        gaps = np.asarray([record.observed_gap for record in test_records])
        utilization = gaps / np.maximum(bounds, 1e-12)
        rows.append(
            {
                "method": METHOD_LABELS[method],
                "method_key": method,
                "target_coverage": target,
                "coverage": float(np.mean(gaps <= bounds)),
                "mean_bound": float(np.mean(bounds)),
                "median_utilization": float(np.median(utilization)),
                "p90_utilization": float(np.quantile(utilization, 0.90)),
                "max_utilization": float(np.max(utilization)),
                "mean_value_gap": float(np.mean(gaps)),
                "calibration_multiplier": float(multipliers[method]),
            }
        )
    return rows


def style():
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


def save(fig, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def bar_figure(rows, key, ylabel, title, stem, line=None):
    fig, ax = plt.subplots(figsize=(3.45, 2.75), constrained_layout=True)
    methods = [row["method_key"] for row in rows]
    values = [row[key] for row in rows]
    ax.bar(
        np.arange(len(rows)),
        values,
        color=[COLORS[method] for method in methods],
        width=0.72,
    )
    ax.set_xticks(np.arange(len(rows)), [METHOD_LABELS[m] for m in methods], rotation=22, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if line is not None:
        ax.axhline(line, color="#555555", ls="--", lw=1.0)
    ax.grid(alpha=0.17, axis="y")
    save(fig, stem)


def make_figures(summary, curves, test_records, test_raw, multipliers, figure_dir):
    style()
    bar_figure(
        summary,
        "coverage",
        "Independent-test coverage",
        "Matched-target value-bound coverage",
        figure_dir / "bound_01_coverage",
        line=summary[0]["target_coverage"],
    )
    bar_figure(
        summary,
        "mean_bound",
        "Mean discounted-value bound",
        "Bound conservatism at matched calibration",
        figure_dir / "bound_02_mean_bound",
    )
    bar_figure(
        summary,
        "median_utilization",
        "Median gap / bound",
        "Median bound utilization",
        figure_dir / "bound_03_median_utilization",
        line=1.0,
    )
    bar_figure(
        summary,
        "p90_utilization",
        "p90 gap / bound",
        "Upper-tail bound utilization",
        figure_dir / "bound_04_p90_utilization",
        line=1.0,
    )
    bar_figure(
        summary,
        "max_utilization",
        "Maximum gap / bound",
        "Worst observed bound utilization",
        figure_dir / "bound_05_max_utilization",
        line=1.0,
    )

    fig, ax = plt.subplots(figsize=(3.45, 2.75), constrained_layout=True)
    for method in METHODS:
        selected = [row for row in curves if row["method_key"] == method]
        ax.plot(
            [row["mean_bound"] for row in selected],
            [row["coverage"] for row in selected],
            "o-",
            color=COLORS[method],
            lw=1.4,
            ms=3.8,
            label=METHOD_LABELS[method],
        )
    ax.set_xlabel("Mean discounted-value bound")
    ax.set_ylabel("Independent-test coverage")
    ax.set_title("Coverage–conservatism frontier")
    ax.legend(fontsize=6.3)
    ax.grid(alpha=0.17)
    save(fig, figure_dir / "bound_06_coverage_mean_frontier")

    gaps = np.asarray([record.observed_gap for record in test_records])
    distributions = []
    for method in METHODS:
        bounds = np.asarray([multipliers[method] * item[method] for item in test_raw])
        distributions.append(gaps / np.maximum(bounds, 1e-12))
    fig, ax = plt.subplots(figsize=(3.45, 2.75), constrained_layout=True)
    boxes = ax.boxplot(distributions, patch_artist=True, showfliers=False)
    for patch, method in zip(boxes["boxes"], METHODS):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.8)
    ax.axhline(1.0, color="#555555", ls="--", lw=1.0)
    ax.set_xticks(np.arange(1, len(METHODS) + 1), [METHOD_LABELS[m] for m in METHODS], rotation=22, ha="right")
    ax.set_ylabel("Observed value gap / bound")
    ax.set_title("Independent-test utilization distribution")
    ax.grid(alpha=0.17, axis="y")
    save(fig, figure_dir / "bound_07_utilization_distribution")


def run(checkpoint, output_root, seed=27, calibration_cases=30, test_cases=80, horizon=15, gamma=0.95):
    if calibration_cases < 20:
        raise ValueError(
            "At least 20 calibration trajectories are required for a finite 95% "
            "split-conformal auxiliary quantile."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_perturbation_world_model(checkpoint, device)
    if payload.get("model_kind") != "fno":
        raise ValueError("The current bound comparison is validated for FNO checkpoints.")
    solver = BurgersSolver()
    rng = np.random.default_rng(seed)
    cases = [sample_case(solver, rng, horizon) for _ in range(calibration_cases + test_cases)]
    records = [
        evaluate_case(model, solver, case, horizon, gamma, device) for case in cases
    ]
    calibration_records = records[:calibration_cases]
    test_records = records[calibration_cases:]

    auxiliary_coverage = 0.95
    constants = {
        "epsilon_global": _higher_quantile(
            np.asarray([r.max_one_step_error for r in calibration_records]), auxiliary_coverage
        ),
        "q_l2": _higher_quantile(
            np.asarray([r.max_l2_score for r in calibration_records]), auxiliary_coverage
        ),
        "q_ellipsoid": _higher_quantile(
            np.asarray([r.max_ellipsoid_score for r in calibration_records]), auxiliary_coverage
        ),
        "lipschitz_global": max(
            1.0,
            _higher_quantile(
                np.asarray([r.max_secant_lipschitz for r in calibration_records]),
                auxiliary_coverage,
            ),
        ),
        "reward_lipschitz_global": max(r.reward_lipschitz for r in calibration_records),
        "gamma": gamma,
    }
    calibration_raw, curvature_scores = raw_bounds(calibration_records, constants)
    curvature_coefficient = _higher_quantile(np.asarray(curvature_scores), 0.90)
    for item in calibration_raw:
        item["adjoint_curvature"] = item["adjoint"] + curvature_coefficient * item["curvature_feature"]
    test_raw, _ = raw_bounds(test_records, constants)
    for item in test_raw:
        item["adjoint_curvature"] = item["adjoint"] + curvature_coefficient * item["curvature_feature"]

    coverage_targets = (0.80, 0.85, 0.90, 0.925, 0.95)
    curves = []
    table = None
    target_multipliers = None
    for target in coverage_targets:
        multipliers = {
            method: conformal_scale(
                calibration_records, calibration_raw, method, target
            )
            for method in METHODS
        }
        rows = summarize(test_records, test_raw, multipliers, target)
        curves.extend(rows)
        if target == 0.90:
            table = rows
            target_multipliers = multipliers
    assert table is not None and target_multipliers is not None

    data_dir = output_root / "data"
    result_dir = output_root / "results"
    figure_dir = output_root / "figures"
    for directory in (data_dir, result_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    with (result_dir / "bound_table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    with (data_dir / "coverage_bound_curve.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    test_rows = []
    for index, (record, item) in enumerate(zip(test_records, test_raw)):
        row = {"case": index, "observed_value_gap": record.observed_gap}
        for method in METHODS:
            bound = target_multipliers[method] * item[method]
            row[f"{method}_bound"] = bound
            row[f"{method}_utilization"] = record.observed_gap / max(bound, 1e-12)
        test_rows.append(row)
    with (data_dir / "test_case_bounds.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(test_rows[0]))
        writer.writeheader()
        writer.writerows(test_rows)
    make_figures(table, curves, test_records, test_raw, target_multipliers, figure_dir)
    result = {
        "device": str(device),
        "calibration_cases": calibration_cases,
        "test_cases": test_cases,
        "horizon": horizon,
        "gamma": gamma,
        "auxiliary_constants": constants,
        "curvature_coefficient": curvature_coefficient,
        "target_coverage": 0.90,
        "table": table,
        "interpretation": (
            "All final value-bound multipliers were selected on calibration trajectories; "
            "the reported table uses independent test trajectories."
        ),
    }
    (result_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("experiments/bound_comparison"))
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--calibration-cases", type=int, default=30)
    parser.add_argument("--test-cases", type=int, default=80)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--gamma", type=float, default=0.95)
    args = parser.parse_args()
    run(
        args.checkpoint,
        args.output_root,
        args.seed,
        args.calibration_cases,
        args.test_cases,
        args.horizon,
        args.gamma,
    )


if __name__ == "__main__":
    main()
