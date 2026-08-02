"""Audit the public persistent-forcing task-validity result.

The audit recomputes every headline statistic that can be recovered from the
released case-level CSV. It does not certify provenance or global optimality of
the finite-budget CEM planner.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


class Audit:
    def __init__(self) -> None:
        self.rows: list[dict[str, str]] = []

    def check(self, name: str, condition: bool, detail: str) -> None:
        self.rows.append(
            {"name": name, "status": "PASS" if condition else "FAIL", "detail": detail}
        )

    def close(
        self,
        name: str,
        observed: float,
        expected: float,
        tolerance: float = 1e-10,
    ) -> None:
        difference = abs(observed - expected)
        self.check(
            name,
            difference <= tolerance,
            f"observed={observed:.12g}, stored={expected:.12g}, "
            f"abs_diff={difference:.3g}",
        )

    @property
    def passed(self) -> bool:
        return all(row["status"] == "PASS" for row in self.rows)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def audit_forced_task(root: Path, audit: Audit) -> None:
    result_root = root / "results" / "forced_oracle_validation"
    summary = load_json(result_root / "summary.json")
    rows = load_csv(result_root / "case_metrics.csv")

    expected = int(summary["test_cases"])
    audit.check(
        "Independent test-case count",
        len(rows) == expected,
        f"rows={len(rows)}, stored={expected}",
    )
    audit.check(
        "Unique case identifiers",
        len({row["case"] for row in rows}) == expected,
        f"unique={len({row['case'] for row in rows})}",
    )

    uncontrolled = np.asarray([float(row["uncontrolled_cost"]) for row in rows])
    oracle = np.asarray([float(row["oracle_cost"]) for row in rows])
    uncontrolled_effort = np.asarray(
        [float(row["uncontrolled_effort"]) for row in rows]
    )
    oracle_effort = np.asarray([float(row["oracle_effort"]) for row in rows])
    uncontrolled_failure = np.asarray(
        [float(row["uncontrolled_failure"]) for row in rows]
    )
    oracle_failure = np.asarray([float(row["oracle_failure"]) for row in rows])

    audit.check(
        "All released values are finite",
        bool(
            np.isfinite(uncontrolled).all()
            and np.isfinite(oracle).all()
            and np.isfinite(oracle_effort).all()
        ),
        "costs and efforts",
    )

    for key, costs, efforts, failures in (
        ("uncontrolled", uncontrolled, uncontrolled_effort, uncontrolled_failure),
        ("pde_oracle_mpc", oracle, oracle_effort, oracle_failure),
    ):
        stored = summary[key]
        audit.close(f"{key}: mean cost", float(np.mean(costs)), float(stored["mean_cost"]))
        audit.close(
            f"{key}: median cost", float(np.median(costs)), float(stored["median_cost"])
        )
        audit.close(
            f"{key}: p90 cost", float(np.quantile(costs, 0.90)), float(stored["p90_cost"])
        )
        audit.close(
            f"{key}: mean control effort",
            float(np.mean(efforts)),
            float(stored["mean_control_effort"]),
        )
        audit.close(
            f"{key}: failure rate",
            float(np.mean(failures)),
            float(stored["failure_rate"]),
        )

    difference = oracle - uncontrolled
    paired = summary["paired"]
    audit.close(
        "Paired mean difference",
        float(np.mean(difference)),
        float(paired["mean_difference"]),
    )
    audit.close(
        "Paired p90 difference",
        float(np.quantile(oracle, 0.90) - np.quantile(uncontrolled, 0.90)),
        float(paired["p90_difference"]),
    )
    audit.close(
        "Fraction oracle better",
        float(np.mean(oracle < uncontrolled)),
        float(paired["fraction_oracle_better"]),
    )

    budget = load_json(result_root / "cem_budget_audit.json")
    first_twenty = float(np.mean(oracle[: int(budget["comparison_cases"])]))
    audit.close(
        "Standard-budget first-20 mean",
        first_twenty,
        float(budget["standard_budget"]["mean_oracle_cost"]),
    )
    audit.check(
        "Stored high-budget audit improves the standard budget",
        float(budget["high_minus_standard_mean_cost"]) < 0.0
        and float(budget["fraction_high_budget_lower"]) == 1.0,
        "high-budget case-level outputs are not redistributed; "
        "this check validates the stored aggregate only",
    )


def render_markdown(audit: Audit) -> str:
    lines = [
        "# Public result audit",
        "",
        f"**Status: {'PASS' if audit.passed else 'FAIL'}**",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for row in audit.rows:
        lines.append(f"| {row['name']} | {row['status']} | {row['detail']} |")
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "The audit recomputes the headline costs from 100 released case-level rows.",
            "It checks the paired design and the stored task-validity summary.",
            "It does not certify experiment provenance, FNO performance, conformal",
            "coverage, or global optimality of the finite-budget CEM planner.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/data_integrity")
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    audit = Audit()
    audit_forced_task(root, audit)

    payload = {
        "status": "PASS" if audit.passed else "FAIL",
        "checks": audit.rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "data_audit.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (output / "DATA_AUDIT.md").write_text(
        render_markdown(audit), encoding="utf-8"
    )
    print(output / "DATA_AUDIT.md")
    if not audit.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
