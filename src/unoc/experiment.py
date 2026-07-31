"""End-to-end training, calibration, shifted evaluation, and robust MPC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from .calibration import calibrate_model, coverage_metrics
from .data import (
    BOUNDARY_SHIFT,
    COMBINED_SHIFT,
    PARAMETER_SHIFT,
    TRAIN_REGIME,
    TransitionDataset,
    generate_transitions,
)
from .models import build_model, count_parameters
from .mpc import CEMConfig, cem_action
from .pde import BurgersSolver
from .train import TrainConfig, train_model


def _loader(arrays, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(TransitionDataset(arrays), batch_size=batch_size, shuffle=shuffle)


def _control_study(
    model,
    id_l2_calibrator,
    audit_l2_calibrator,
    decision_calibrator,
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
        for method in costs:
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
            costs[method].append(cumulative)
            action_records[method].append(method_actions)
    return costs, action_records, case_records


def run(
    model_kind: str,
    output_dir: Path,
    seed: int,
    quick: bool,
    control_cases: int | None = None,
    control_horizon: int | None = None,
) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    device = torch.device("cpu")
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
    model = build_model(
        model_kind,
        width=20 if quick else 32,
        modes=10 if quick else 14,
        layers=3 if quick else 4,
    ).to(device)
    train_config = TrainConfig(epochs=20 if quick else 60)
    history = train_model(
        model,
        _loader(arrays["train"], batch_size, True),
        _loader(arrays["validation"], batch_size),
        train_config,
        device,
    )
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
        }
        for name in ("id_test", "parameter_shift", "boundary_shift", "combined_shift")
    }
    control, control_actions, control_cases_info = _control_study(
        model,
        id_l2_calibrator,
        audit_l2_calibrator,
        decision_calibrator,
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
            "state_dict": model.state_dict(),
            "id_l2_calibrator": id_l2_calibrator.__dict__,
            "audit_l2_calibrator": audit_l2_calibrator.__dict__,
            "decision_calibrator": decision_calibrator.__dict__,
        },
        output_dir / f"{model_kind}_world_model.pt",
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    axes[0].plot([item["epoch"] for item in history], [item["train_loss"] for item in history])
    axes[0].plot(
        [item["epoch"] for item in history],
        [item["validation_loss"] for item in history],
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Probabilistic loss")
    axes[0].set_title(f"{model_kind.upper()} world-model training")
    axes[0].legend(("train", "validation"), frameon=False)
    axes[0].grid(alpha=0.25)

    labels = list(control)
    values = [np.mean(control[label]) for label in labels]
    errors = [np.std(control[label]) for label in labels]
    axes[1].bar(
        labels,
        values,
        yerr=errors,
        color=("#999999", "#4C78A8", "#9ECAE1", "#6BAED6", "#F28E2B", "#D95F02"),
    )
    axes[1].set_ylabel("Shifted closed-loop cost (lower is better)")
    axes[1].set_title("Parameter + boundary shift")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].grid(alpha=0.2, axis="y")
    figure.tight_layout()
    figure.savefig(output_dir / f"{model_kind}_summary.png", dpi=180)
    plt.close(figure)

    result: dict[str, object] = {
        "model": model_kind,
        "parameters": count_parameters(model),
        "quick": quick,
        "calibration": {
            "id_l2": id_l2_info,
            "audit_l2": audit_l2_info,
            "audit_ellipsoid": decision_info,
        },
        "evaluation": evaluation,
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
    parser.add_argument("--control-cases", type=int)
    parser.add_argument("--control-horizon", type=int)
    args = parser.parse_args()
    result = run(
        args.model,
        args.output_dir,
        args.seed,
        args.quick,
        args.control_cases,
        args.control_horizon,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
