"""Build a traceable Markdown/JSON summary from the final experiment files.

The script avoids copying numbers by hand into the manuscript.  Every reported
percentage is recomputed from the stored raw metrics, and unavailable
experiments are explicitly marked rather than silently replaced by pilot data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path | None):
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def relative_change(value: float, reference: float) -> float:
    return 100.0 * (value - reference) / reference


def summarize_burgers(payload: dict) -> dict:
    shifted = payload["evaluation"]["combined_shift"]
    control = payload["control"]
    nominal_mean = float(control["nominal_mpc"]["mean"])
    nominal_p90 = float(control["nominal_mpc"]["p90"])
    control_rows = []
    for method, values in control.items():
        mean = float(values["mean"])
        p90 = float(values["p90"])
        control_rows.append(
            {
                "method": method,
                "mean": mean,
                "p90": p90,
                "mean_change_vs_nominal_percent": relative_change(mean, nominal_mean),
                "p90_change_vs_nominal_percent": relative_change(p90, nominal_p90),
                "mean_absolute_action": float(values["mean_absolute_action"]),
            }
        )
    return {
        "training": {
            "parameters": int(payload["parameters"]),
            "final_train_loss": float(payload["final_history"]["train_loss"]),
            "final_validation_loss": float(payload["final_history"]["validation_loss"]),
        },
        "combined_shift_sets": {
            method: {
                "coverage": float(metrics["coverage"]),
                "mean_radius": float(metrics["mean_radius"]),
                "decision_linear_coverage": float(metrics["decision_linear_coverage"]),
                "mean_decision_support": float(metrics["mean_decision_support"]),
            }
            for method, metrics in shifted.items()
        },
        "trajectory": payload["trajectory_evaluation"],
        "control": control_rows,
    }


def markdown_table(rows: list[dict], columns: list[tuple[str, str, str]]) -> str:
    header = "| " + " | ".join(title for _, title, _ in columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    body = []
    for row in rows:
        body.append(
            "| "
            + " | ".join(
                format(row[key], spec) if not isinstance(row[key], str) else row[key]
                for key, _, spec in columns
            )
            + " |"
        )
    return "\n".join([header, separator, *body])


def render_markdown(summary: dict) -> str:
    lines = [
        "# Final result summary",
        "",
        "Generated directly from stored JSON files. Smaller closed-loop cost is better.",
        "",
    ]
    burgers = summary.get("burgers")
    if burgers is not None:
        lines.extend(["## Controlled Burgers", ""])
        set_rows = [
            {"method": method, **metrics}
            for method, metrics in burgers["combined_shift_sets"].items()
        ]
        lines.extend(
            [
                "### Function-space ambiguity sets under combined shift",
                "",
                markdown_table(
                    set_rows,
                    [
                        ("method", "Set", ""),
                        ("coverage", "Coverage", ".3f"),
                        ("mean_radius", "Mean radius", ".4f"),
                        ("decision_linear_coverage", "Decision coverage", ".3f"),
                        ("mean_decision_support", "Mean decision support", ".4f"),
                    ],
                ),
                "",
                "### Closed-loop control",
                "",
                markdown_table(
                    burgers["control"],
                    [
                        ("method", "Method", ""),
                        ("mean", "Mean cost", ".4f"),
                        ("p90", "p90 cost", ".4f"),
                        ("mean_change_vs_nominal_percent", "Mean change (%)", "+.2f"),
                        ("p90_change_vs_nominal_percent", "p90 change (%)", "+.2f"),
                    ],
                ),
                "",
                "Trajectory max-score coverage: "
                f"{float(burgers['trajectory']['coverage']):.3f} at horizon "
                f"{int(float(burgers['trajectory']['horizon']))}.",
                "",
            ]
        )
    bounds = summary.get("bounds")
    if bounds is not None:
        lines.extend(
            [
                "## Independent value-bound comparison",
                "",
                markdown_table(
                    bounds["table"],
                    [
                        ("method", "Method", ""),
                        ("coverage", "Coverage", ".3f"),
                        ("mean_bound", "Mean bound", ".5f"),
                        ("median_utilization", "Median utilization", ".3f"),
                        ("p90_utilization", "p90 utilization", ".3f"),
                        ("max_utilization", "Max utilization", ".3f"),
                    ],
                ),
                "",
            ]
        )
    ns2d = summary.get("ns2d")
    if ns2d is not None:
        lines.extend(
            [
                "## Official NS2D function-valued benchmark",
                "",
                f"- Simultaneous test coverage: {float(ns2d['test_simultaneous_coverage']):.3f}",
                f"- Mean L2 error: {float(ns2d['mean_l2_error']):.6f}",
                f"- Mean full band width: {float(ns2d['mean_full_band_width']):.6f}",
                f"- Proper train/audit/test sizes: {ns2d['n_train']}/{ns2d['n_audit']}/{ns2d['n_test']}",
                "- Scope: function-valued uncertainty only; the public pairs have no action channel.",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--burgers-metrics", type=Path)
    parser.add_argument("--bound-summary", type=Path)
    parser.add_argument("--ns2d-metrics", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/final_summary"))
    args = parser.parse_args()

    burgers_raw = read_json(args.burgers_metrics)
    summary = {
        "burgers": summarize_burgers(burgers_raw) if burgers_raw is not None else None,
        "bounds": read_json(args.bound_summary),
        "ns2d": read_json(args.ns2d_metrics),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.output_dir / "SUMMARY.md").write_text(
        render_markdown(summary), encoding="utf-8"
    )
    print(args.output_dir / "SUMMARY.md")


if __name__ == "__main__":
    main()
