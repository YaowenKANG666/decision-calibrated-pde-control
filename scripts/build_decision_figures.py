"""Create standalone decision-effectiveness figures from a Burgers metrics file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "id_l2": "#A7B8C8",
    "audit_l2": "#77A6C5",
    "audit_ellipsoid": "#D8843F",
    "audit_simultaneous_box": "#6D9F71",
    "nominal_mpc": "#8A8A8A",
    "id_l2_robust_mpc": "#A7B8C8",
    "audit_l2_robust_mpc": "#77A6C5",
    "adjoint_robust_mpc": "#D8843F",
    "box_adjoint_robust_mpc": "#6D9F71",
    "adversarial_robust_mpc": "#B9544D",
}

DISPLAY = {
    "id_l2": "Source L2",
    "audit_l2": "Audit L2",
    "audit_ellipsoid": "Audit ellipsoid",
    "audit_simultaneous_box": "Audit box",
    "nominal_mpc": "Nominal",
    "id_l2_robust_mpc": "Source L2",
    "audit_l2_robust_mpc": "Audit L2",
    "adjoint_robust_mpc": "Adjoint",
    "box_adjoint_robust_mpc": "Box adjoint",
    "adversarial_robust_mpc": "Adversarial",
}


def style() -> None:
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


def save(figure: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(figure)


def coverage_support(payload: dict, output: Path) -> None:
    shifted = payload["evaluation"]["combined_shift"]
    methods = ["id_l2", "audit_l2", "audit_ellipsoid", "audit_simultaneous_box"]
    evaluation_cases = 180 if payload.get("quick", False) else 800
    figure, axis = plt.subplots(figsize=(3.45, 2.8), constrained_layout=True)
    for method in methods:
        metrics = shifted[method]
        x = float(metrics["mean_decision_support"])
        y = float(metrics["coverage"])
        axis.scatter(x, y, s=42, color=COLORS[method], edgecolor="white", linewidth=0.6, zorder=3)
        axis.annotate(DISPLAY[method], (x, y), xytext=(4, 3), textcoords="offset points", fontsize=6.5)
    axis.axhline(0.90, color="#666666", ls="--", lw=1.0, label="90% target")
    axis.set_xlabel("Mean support in stage-cost direction")
    axis.set_ylabel("Combined-shift function coverage")
    axis.set_title(
        f"Coverage is not the same as decision tightness (n={evaluation_cases})"
    )
    axis.grid(alpha=0.17)
    axis.legend(loc="lower right", fontsize=6.4)
    save(figure, output / "decision_01_coverage_support")


def control_tail(payload: dict, output: Path) -> None:
    control = payload["control"]
    methods = [
        "nominal_mpc",
        "id_l2_robust_mpc",
        "audit_l2_robust_mpc",
        "adjoint_robust_mpc",
        "box_adjoint_robust_mpc",
        "adversarial_robust_mpc",
    ]
    p90 = np.asarray([float(control[method]["p90"]) for method in methods])
    nominal = float(control["nominal_mpc"]["p90"])
    change = 100.0 * (p90 - nominal) / nominal
    cases = len(control["nominal_mpc"]["raw"])
    figure, axis = plt.subplots(figsize=(4.1, 2.85), constrained_layout=True)
    bars = axis.bar(
        np.arange(len(methods)),
        change,
        color=[COLORS[method] for method in methods],
        width=0.72,
    )
    axis.axhline(0.0, color="#555555", lw=1.0)
    for bar, value in zip(bars, change):
        if abs(value) < 1e-12:
            continue
        axis.annotate(
            f"{value:+.2f}%",
            (bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.2,
        )
    axis.set_xticks(
        np.arange(len(methods)),
        [DISPLAY[method] for method in methods],
        rotation=24,
        ha="right",
    )
    axis.set_ylabel("p90 cost change versus nominal (%)")
    axis.set_title(f"Upper-tail control change across {cases} matched gain cases")
    axis.grid(alpha=0.17, axis="y")
    save(figure, output / "decision_02_control_p90")


def trajectory_validation(payload: dict, output: Path) -> None:
    shifted = payload["evaluation"]["combined_shift"]
    trajectory = payload["trajectory_evaluation"]
    one_step_cases = 180 if payload.get("quick", False) else 800
    trajectory_cases = 120 if payload.get("quick", False) else 400
    labels = ["One-step ellipsoid", "One-step box", "Trajectory max-score"]
    values = [
        float(shifted["audit_ellipsoid"]["coverage"]),
        float(shifted["audit_simultaneous_box"]["coverage"]),
        float(trajectory["coverage"]),
    ]
    figure, axis = plt.subplots(figsize=(3.45, 2.75), constrained_layout=True)
    bars = axis.bar(
        np.arange(3),
        values,
        color=[COLORS["audit_ellipsoid"], COLORS["audit_simultaneous_box"], "#3A5A78"],
        width=0.68,
    )
    axis.axhline(0.90, color="#666666", ls="--", lw=1.0, label="90% target")
    axis.set_xticks(np.arange(3), labels, rotation=18, ha="right")
    axis.set_ylim(max(0.0, min(values) - 0.08), 1.01)
    axis.set_ylabel("Independent-test coverage")
    axis.set_title(
        "Separate trajectory calibration covers time and space\n"
        f"behavior-policy horizon = {int(float(trajectory['horizon']))}"
    )
    axis.grid(alpha=0.17, axis="y")
    axis.legend(fontsize=6.4)
    for bar, sample_size in zip(bars, (one_step_cases, one_step_cases, trajectory_cases)):
        axis.annotate(
            f"n={sample_size}",
            (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.2,
        )
    save(figure, output / "decision_03_trajectory_coverage")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    style()
    coverage_support(payload, args.output_dir)
    control_tail(payload, args.output_dir)
    trajectory_validation(payload, args.output_dir)


if __name__ == "__main__":
    main()
