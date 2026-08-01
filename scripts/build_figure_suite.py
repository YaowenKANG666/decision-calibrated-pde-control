"""Collect standalone experiment figures and write an honest manifest.

The script never invents a missing experiment.  A requested figure is either
copied with all available publication formats or marked ``pending``.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path


FORMATS = (".png", ".svg", ".pdf")


@dataclass(frozen=True)
class FigureSpec:
    stem: str
    conclusion: str
    source: str


SPECS = (
    FigureSpec("method_01_chain_schematic", "Prediction scale to calibrated set to robust decision", "figures"),
    FigureSpec("fno_01_training_curve", "FNO optimization is numerically stable", "results/colab_fno_seed27/figures"),
    FigureSpec("fno_02_control_mean_cost", "Mean control cost under joint shift", "results/colab_fno_seed27/figures"),
    FigureSpec("fno_03_control_p90_cost", "Upper-tail control cost under joint shift", "results/colab_fno_seed27/figures"),
    FigureSpec("fno_04_mean_absolute_action", "Robustness is not purchased by hidden control effort", "results/colab_fno_seed27/figures"),
    FigureSpec("fno_05_cost_vs_actuator_gain", "Controller sensitivity to actuator-gain shift", "results/colab_fno_seed27/figures"),
    FigureSpec("fno_06_coverage_by_shift", "Coverage under parameter and boundary shifts", "results/colab_fno_seed27/figures"),
    FigureSpec("fno_07_radius_by_shift", "Ambiguity radius expands under shift", "results/colab_fno_seed27/figures"),
    FigureSpec("fno_08_error_scale_association", "Uncertainty scale localizes prediction error", "results/colab_fno_seed27/figures"),
    FigureSpec("value_01_epsilon_scaling", "Value error is linear in one-step model error", "experiments/reward_value_gap/figures"),
    FigureSpec("value_02_discount_scaling", "The sharp witness has squared effective-horizon scaling", "experiments/reward_value_gap/figures"),
    FigureSpec("value_03_burgers_gap_bound", "Finite-horizon Burgers gaps are covered by the recursion", "experiments/reward_value_gap/figures"),
    FigureSpec("value_04_rollout_error", "Observed rollout error versus recursive envelope", "experiments/reward_value_gap/figures"),
    FigureSpec("value_05_fno_gap_discount", "Learned-FNO value error across discount factors", "experiments/reward_value_gap/figures"),
    FigureSpec("value_06_fno_gap_envelope", "Observed learned-FNO gap versus local envelope", "experiments/reward_value_gap/figures"),
    FigureSpec("bound_01_coverage", "Four bounds compared at independent-test coverage", "experiments/bound_comparison/figures"),
    FigureSpec("bound_02_mean_bound", "Mean conservatism at matched calibration", "experiments/bound_comparison/figures"),
    FigureSpec("bound_03_median_utilization", "Typical bound utilization", "experiments/bound_comparison/figures"),
    FigureSpec("bound_04_p90_utilization", "Upper-tail bound utilization", "experiments/bound_comparison/figures"),
    FigureSpec("bound_05_max_utilization", "Worst independent-test utilization", "experiments/bound_comparison/figures"),
    FigureSpec("bound_06_coverage_mean_frontier", "Coverage-conservatism frontier", "experiments/bound_comparison/figures"),
    FigureSpec("bound_07_utilization_distribution", "Distribution of gap-to-bound ratios", "experiments/bound_comparison/figures"),
    FigureSpec("ns2d_01_input", "Representative input vorticity field", "experiments/ns2d/figures"),
    FigureSpec("ns2d_02_target", "Representative target vorticity field", "experiments/ns2d/figures"),
    FigureSpec("ns2d_03_prediction", "Representative FNO vorticity prediction", "experiments/ns2d/figures"),
    FigureSpec("ns2d_04_absolute_error", "Spatial distribution of prediction error", "experiments/ns2d/figures"),
    FigureSpec("ns2d_05_uncertainty_halfwidth", "Calibrated function-valued uncertainty width", "experiments/ns2d/figures"),
    FigureSpec("ns2d_06_coverage_mask", "Pointwise realization of simultaneous coverage", "experiments/ns2d/figures"),
    FigureSpec("ns2d_07_training_curve", "Base two-dimensional FNO optimization", "experiments/ns2d/figures"),
    FigureSpec("ns2d_08_score_ecdf", "Max-type nonconformity score distribution", "experiments/ns2d/figures"),
    FigureSpec("ns2d_09_coverage_reliability", "Nominal versus test simultaneous coverage", "experiments/ns2d/figures"),
    FigureSpec("ns2d_10_coverage_width", "Coverage-width trade-off", "experiments/ns2d/figures"),
    FigureSpec("ns2d_11_error_scale", "Disagreement scale versus realized pointwise error", "experiments/ns2d/figures"),
    FigureSpec("ablation_01_label_noise", "Sensitivity to perturbation-label noise", "planned"),
    FigureSpec("ablation_02_smoothing_window", "Sensitivity to spatial smoothing", "planned"),
    FigureSpec("ablation_03_scale_floor", "Sensitivity to the minimum scale floor", "planned"),
    FigureSpec("ablation_04_calibration_size", "Coverage and width versus audit-set size", "planned"),
    FigureSpec("trajectory_01_horizon_coverage", "Simultaneous trajectory coverage versus horizon", "planned"),
    FigureSpec("architecture_01_backbone_coverage", "FNO/TNO/DSC-DNO coverage comparison", "planned"),
    FigureSpec("architecture_02_backbone_control", "FNO/TNO/DSC-DNO control-cost comparison", "planned"),
)


def build(project_root: Path, output_dir: Path) -> list[dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for spec in SPECS:
        source_dir = project_root / spec.source
        copied: list[str] = []
        if spec.source != "planned":
            for suffix in FORMATS:
                source = source_dir / f"{spec.stem}{suffix}"
                if source.exists():
                    destination = output_dir / source.name
                    shutil.copy2(source, destination)
                    copied.append(suffix.lstrip("."))
        rows.append(
            {
                "figure": spec.stem,
                "status": "available" if copied else "pending",
                "formats": ",".join(copied),
                "conclusion": spec.conclusion,
                "source": spec.source,
            }
        )

    with (output_dir / "figure_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Individual Figure Manifest",
        "",
        "Each available file carries one conclusion. Pending items require a separate experiment and are not fabricated.",
        "",
        "| Figure | Status | Conclusion |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| `{row['figure']}` | {row['status']} | {row['conclusion']} |" for row in rows
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("figures/all_individual"))
    args = parser.parse_args()
    rows = build(args.project_root.resolve(), args.output_dir.resolve())
    available = sum(row["status"] == "available" for row in rows)
    print(f"Collected {available}/{len(rows)} standalone figures in {args.output_dir}")


if __name__ == "__main__":
    main()
