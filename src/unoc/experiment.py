"""End-to-end training, calibration, shifted evaluation, and robust MPC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .calibration import (
    calibrate_model,
    calibrate_trajectory_model,
    coverage_metrics,
    estimate_perturbation_floor,
    trajectory_coverage_metrics,
)
from .data import (
    BOUNDARY_SHIFT,
    COMBINED_SHIFT,
    PARAMETER_SHIFT,
    TRAIN_REGIME,
    TransitionDataset,
    generate_trajectories,
    generate_transitions,
)
from .models import PerturbationScaleWorldModel, build_model, count_parameters
from .mpc import CEMConfig, cem_action
from .pde import BurgersSolver
from .train import TrainConfig, train_model


def _loader(arrays, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(TransitionDataset(arrays), batch_size=batch_size, shuffle=shuffle)


def _figure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
        }
    )


def _save_figure(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _make_individual_figures(
    output_dir: Path,
    model_kind: str,
    history: list[dict[str, float]],
    evaluation: dict[str, dict[str, dict[str, float]]],
    control: dict[str, list[float]],
    control_actions: dict[str, list[list[float]]],
    control_cases: list[dict[str, float]],
) -> None:
    """Write one scientific conclusion per file; never create panel grids."""

    _figure_style()
    figure_dir = output_dir / "figures"
    colors = ("#999999", "#4C78A8", "#9ECAE1", "#6BAED6", "#F28E2B", "#59A14F", "#D95F02")
    labels = list(control)
    display_labels = [label.replace("_", " ") for label in labels]

    figure, axis = plt.subplots(figsize=(3.45, 2.75), constrained_layout=True)
    axis.semilogy(
        [item["epoch"] for item in history],
        [item["train_loss"] for item in history],
        color="#3A5A78",
        label="Train",
    )
    axis.semilogy(
        [item["epoch"] for item in history],
        [item["validation_loss"] for item in history],
        color="#D8843F",
        label="Validation",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Training objective")
    axis.set_title(f"{model_kind.upper()} world-model optimization")
    axis.legend()
    axis.grid(alpha=0.18)
    _save_figure(figure, figure_dir / f"{model_kind}_01_training_curve")

    for index, (metric, ylabel, title, stem) in enumerate(
        (
            ("mean", "Mean shifted closed-loop cost", "Expected control cost under joint shift", "02_control_mean_cost"),
            ("p90", "p90 shifted closed-loop cost", "Upper-tail control cost under joint shift", "03_control_p90_cost"),
            ("action", "Mean absolute action", "Control effort under joint shift", "04_mean_absolute_action"),
        )
    ):
        del index
        if metric == "mean":
            values = [float(np.mean(control[label])) for label in labels]
        elif metric == "p90":
            values = [float(np.quantile(control[label], 0.90)) for label in labels]
        else:
            values = [float(np.mean(np.abs(control_actions[label]))) for label in labels]
        figure, axis = plt.subplots(figsize=(4.35, 2.9), constrained_layout=True)
        axis.bar(np.arange(len(labels)), values, color=colors)
        axis.set_xticks(np.arange(len(labels)), display_labels, rotation=24, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.18, axis="y")
        _save_figure(figure, figure_dir / f"{model_kind}_{stem}")

    gains = np.asarray([case["actuator_gain"] for case in control_cases])
    if gains.size > 1:
        order = np.argsort(gains)
        figure, axis = plt.subplots(figsize=(3.45, 2.75), constrained_layout=True)
        for label, color in zip(labels, colors):
            axis.plot(
                gains[order],
                np.asarray(control[label])[order],
                "o-",
                color=color,
                lw=1.1,
                ms=3.0,
                label=label.replace("_", " "),
            )
        axis.set_xlabel("True actuator gain")
        axis.set_ylabel("Closed-loop cost")
        axis.set_title("Robustness to actuator-gain shift")
        axis.legend(fontsize=5.4, ncol=2)
        axis.grid(alpha=0.18)
        _save_figure(figure, figure_dir / f"{model_kind}_05_cost_vs_actuator_gain")

    regimes = list(evaluation)
    methods = ("id_l2", "audit_l2", "audit_ellipsoid", "audit_simultaneous_box")
    method_colors = ("#9ECAE1", "#6BAED6", "#F28E2B", "#59A14F")
    for metric, ylabel, title, stem in (
        ("coverage", "Function-level coverage", "Coverage under deployment shift", "06_coverage_by_shift"),
        ("mean_radius", "Mean ambiguity radius", "Ambiguity-set size under deployment shift", "07_radius_by_shift"),
        ("radius_error_correlation", "Radius–error correlation", "Does uncertainty localize model error?", "08_error_scale_association"),
    ):
        figure, axis = plt.subplots(figsize=(4.25, 2.85), constrained_layout=True)
        x = np.arange(len(regimes))
        for method, color in zip(methods, method_colors):
            axis.plot(
                x,
                [evaluation[regime][method][metric] for regime in regimes],
                "o-",
                color=color,
                lw=1.3,
                ms=3.4,
                label=method.replace("_", " "),
            )
        if metric == "coverage":
            axis.axhline(0.90, color="#666666", ls="--", lw=0.9, label="90% target")
        axis.set_xticks(x, [regime.replace("_", " ") for regime in regimes], rotation=20, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.legend(fontsize=5.7, ncol=2)
        axis.grid(alpha=0.18)
        _save_figure(figure, figure_dir / f"{model_kind}_{stem}")


def _control_study(
    model,
    id_l2_calibrator,
    audit_l2_calibrator,
    decision_calibrator,
    simultaneous_calibrator,
    solver: BurgersSolver,
    device: torch.device,
    cases: int,
    horizon: int,
    seed: int,
    config: CEMConfig,
) -> tuple[dict[str, list[float]], dict[str, list[list[float]]], list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    costs = {
        "uncontrolled": [],
        "nominal_mpc": [],
        "id_l2_robust_mpc": [],
        "audit_l2_robust_mpc": [],
        "adjoint_robust_mpc": [],
        "box_adjoint_robust_mpc": [],
        "adversarial_robust_mpc": [],
    }
    action_records = {method: [] for method in costs}
    case_records: list[dict[str, float]] = []
    gain_levels = np.linspace(
        COMBINED_SHIFT.actuator_gain[0],
        COMBINED_SHIFT.actuator_gain[1],
        cases,
    )
    viscosity = float(rng.uniform(*PARAMETER_SHIFT.viscosity))
    left, right = rng.uniform(*BOUNDARY_SHIFT.boundary, 2)
    state0 = solver.random_state(rng, float(left), float(right), (0.35, 0.70))
    for case in range(cases):
        actuator_gain = float(gain_levels[case])
        case_records.append(
            {
                "viscosity": viscosity,
                "left_boundary": float(left),
                "right_boundary": float(right),
                "actuator_gain": actuator_gain,
            }
        )
        for method, method_costs in costs.items():
            state = state0.copy()
            cumulative = 0.0
            method_actions: list[float] = []
            for step in range(horizon):
                if method == "uncontrolled":
                    action = 0.0
                else:
                    calibrator = {
                        "nominal_mpc": decision_calibrator,
                        "id_l2_robust_mpc": id_l2_calibrator,
                        "audit_l2_robust_mpc": audit_l2_calibrator,
                        "adjoint_robust_mpc": decision_calibrator,
                        "box_adjoint_robust_mpc": simultaneous_calibrator,
                        "adversarial_robust_mpc": decision_calibrator,
                    }[method]
                    action = cem_action(
                        model,
                        calibrator,
                        state,
                        viscosity,
                        (float(left), float(right)),
                        config,
                        device,
                        robust=method != "nominal_mpc",
                        seed=10_000 * case + step,
                        adversarial=method == "adversarial_robust_mpc",
                    )
                method_actions.append(float(action))
                cumulative += solver.stage_cost(state, action)
                state = solver.step(
                    state,
                    action,
                    viscosity,
                    float(left),
                    float(right),
                    actuator_gain,
                )
            method_costs.append(cumulative)
            action_records[method].append(method_actions)
    return costs, action_records, case_records


def run(
    model_kind: str,
    output_dir: Path,
    seed: int,
    quick: bool,
    uncertainty: str = "perturbation",
    control_cases: int | None = None,
    control_horizon: int | None = None,
    device_name: str = "auto",
) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    solver = BurgersSolver()
    sizes = {
        "train": 700 if quick else 4_000,
        "validation": 160 if quick else 600,
        "calibration": 180 if quick else 800,
        "deployment_audit": 90 if quick else 300,
        "test": 180 if quick else 800,
    }
    regimes = {
        "train": TRAIN_REGIME,
        "validation": TRAIN_REGIME,
        "calibration": TRAIN_REGIME,
        "deployment_audit": COMBINED_SHIFT,
        "id_test": TRAIN_REGIME,
        "parameter_shift": PARAMETER_SHIFT,
        "boundary_shift": BOUNDARY_SHIFT,
        "combined_shift": COMBINED_SHIFT,
    }
    arrays = {
        name: generate_transitions(
            solver,
            sizes["test"] if name not in sizes else sizes[name],
            regime,
            seed + 101 * index,
        )
        for index, (name, regime) in enumerate(regimes.items())
    }
    batch_size = 64
    architecture = {
        "width": 20 if quick else 32,
        "modes": 10 if quick else 14,
        "layers": 3 if quick else 4,
    }
    base_model = build_model(
        model_kind,
        **architecture,
    ).to(device)
    train_config = TrainConfig(
        epochs=20 if quick else 60,
        uncertainty_weight=0.0 if uncertainty == "perturbation" else 0.50,
    )
    history = train_model(
        base_model,
        _loader(arrays["train"], batch_size, True),
        _loader(arrays["validation"], batch_size),
        train_config,
        device,
    )
    perturbation_history: list[dict[str, float]] | None = None
    uncertainty_details: dict[str, float | str] = {"method": uncertainty}
    if uncertainty == "perturbation":
        perturbed_model = build_model(
            model_kind,
            **architecture,
        ).to(device)
        perturbed_training = arrays["train"].with_perturbed_targets(
            noise_multiplier=0.05,
            seed=seed + 71_003,
        )
        perturbation_history = train_model(
            perturbed_model,
            _loader(perturbed_training, batch_size, True),
            _loader(arrays["validation"], batch_size),
            train_config,
            device,
        )
        model = PerturbationScaleWorldModel(
            base_model,
            perturbed_model,
            smoothing_window=5,
        ).to(device)
        floor = estimate_perturbation_floor(
            model,
            _loader(arrays["train"], batch_size),
            device,
        )
        uncertainty_details.update(
            {
                "label_noise_multiplier": 0.05,
                "smoothing_window": 5.0,
                "scale_floor": floor,
            }
        )
    elif uncertainty == "head":
        model = base_model
    else:
        raise ValueError(f"Unknown uncertainty method: {uncertainty}")
    id_l2_calibrator, id_l2_info = calibrate_model(
        model,
        _loader(arrays["calibration"], batch_size),
        device,
        norm_kind="l2",
    )
    audit_l2_calibrator, audit_l2_info = calibrate_model(
        model,
        _loader(arrays["deployment_audit"], batch_size),
        device,
        norm_kind="l2",
    )
    decision_calibrator, decision_info = calibrate_model(
        model,
        _loader(arrays["deployment_audit"], batch_size),
        device,
        norm_kind="ellipsoid",
    )
    simultaneous_calibrator, simultaneous_info = calibrate_model(
        model,
        _loader(arrays["deployment_audit"], batch_size),
        device,
        norm_kind="max",
    )
    trajectory_horizon = 5 if quick else 10
    trajectory_audit = generate_trajectories(
        solver,
        size=60 if quick else 200,
        horizon=trajectory_horizon,
        regime=COMBINED_SHIFT,
        seed=seed + 82_001,
    )
    trajectory_test = generate_trajectories(
        solver,
        size=120 if quick else 400,
        horizon=trajectory_horizon,
        regime=COMBINED_SHIFT,
        seed=seed + 82_002,
    )
    trajectory_calibrator, trajectory_info = calibrate_trajectory_model(
        model,
        trajectory_audit,
        device,
    )
    trajectory_evaluation = trajectory_coverage_metrics(
        model,
        trajectory_calibrator,
        trajectory_test,
        device,
    )
    evaluation = {
        name: {
            "id_l2": coverage_metrics(
                model,
                id_l2_calibrator,
                _loader(arrays[name], batch_size),
                device,
            ),
            "audit_l2": coverage_metrics(
                model,
                audit_l2_calibrator,
                _loader(arrays[name], batch_size),
                device,
            ),
            "audit_ellipsoid": coverage_metrics(
                model,
                decision_calibrator,
                _loader(arrays[name], batch_size),
                device,
            ),
            "audit_simultaneous_box": coverage_metrics(
                model,
                simultaneous_calibrator,
                _loader(arrays[name], batch_size),
                device,
            ),
        }
        for name in ("id_test", "parameter_shift", "boundary_shift", "combined_shift")
    }
    control, control_actions, control_cases_info = _control_study(
        model,
        id_l2_calibrator,
        audit_l2_calibrator,
        decision_calibrator,
        simultaneous_calibrator,
        solver,
        device,
        cases=control_cases if control_cases is not None else (3 if quick else 12),
        horizon=control_horizon if control_horizon is not None else (10 if quick else 25),
        seed=seed + 999,
        config=(
            CEMConfig(candidates=64, elites=8, iterations=3)
            if quick
            else CEMConfig()
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_kind": model_kind,
            "architecture": architecture,
            "quick": quick,
            "device_used": str(device),
            "state_dict": model.state_dict(),
            "id_l2_calibrator": id_l2_calibrator.__dict__,
            "audit_l2_calibrator": audit_l2_calibrator.__dict__,
            "decision_calibrator": decision_calibrator.__dict__,
            "simultaneous_calibrator": simultaneous_calibrator.__dict__,
            "trajectory_calibrator": trajectory_calibrator.__dict__,
            "uncertainty": uncertainty_details,
        },
        output_dir / f"{model_kind}_{uncertainty}_world_model.pt",
    )

    _make_individual_figures(
        output_dir,
        model_kind,
        history,
        evaluation,
        control,
        control_actions,
        control_cases_info,
    )

    result: dict[str, object] = {
        "model": model_kind,
        "parameters": count_parameters(model),
        "quick": quick,
        "device": str(device),
        "architecture": architecture,
        "uncertainty": uncertainty_details,
        "calibration": {
            "id_l2": id_l2_info,
            "audit_l2": audit_l2_info,
            "audit_ellipsoid": decision_info,
            "audit_simultaneous_box": simultaneous_info,
            "audit_trajectory_box": trajectory_info,
        },
        "evaluation": evaluation,
        "trajectory_evaluation": trajectory_evaluation,
        "control": {
            method: {
                "mean": float(np.mean(values_)),
                "std": float(np.std(values_)),
                "p90": float(np.quantile(values_, 0.90)),
                "raw": values_,
                "mean_absolute_action": float(
                    np.mean(np.abs(control_actions[method]))
                ),
                "actions": control_actions[method],
            }
            for method, values_ in control.items()
        },
        "control_cases": control_cases_info,
        "final_history": history[-1],
        "training_history": history,
        "perturbation_training_history": perturbation_history,
        "perturbation_final_history": (
            perturbation_history[-1] if perturbation_history is not None else None
        ),
    }
    (output_dir / f"{model_kind}_metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("fno", "tno", "dscdno", "moe"), default="fno")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--uncertainty", choices=("perturbation", "head"), default="perturbation")
    parser.add_argument("--control-cases", type=int)
    parser.add_argument("--control-horizon", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    result = run(
        args.model,
        args.output_dir,
        args.seed,
        args.quick,
        args.uncertainty,
        args.control_cases,
        args.control_horizon,
        args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
