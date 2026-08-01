"""Draw the paper's one-figure uncertainty-to-control method chain."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def draw(output_stem: Path, write_tiff: bool = False) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
        }
    )
    figure, axis = plt.subplots(figsize=(7.20, 2.64), constrained_layout=True)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    boxes = (
        (0.025, "Two world models", "clean labels\nperturbed labels", "#E8EEF3"),
        (0.225, "Disagreement scale", r"$\sigma=\mathrm{Smooth}(|\mu-\widetilde\mu|)$", "#DCEAF3"),
        (0.425, "Audit-calibrated set", "ellipsoid / max box\nmarginal one-step coverage", "#F6E8D8"),
        (0.625, "Decision query", "adjoint support\nor adversarial rollout", "#E5EFE5"),
        (0.825, "Independent evaluation", "coverage + tightness\nclosed-loop cost", "#EDE6F2"),
    )
    width, height, y = 0.15, 0.48, 0.33
    for x, title, body, color in boxes:
        box = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=0.85,
            edgecolor="#435260",
            facecolor=color,
        )
        axis.add_patch(box)
        axis.text(x + width / 2, y + 0.36, title, ha="center", va="center", fontweight="bold", fontsize=7.2)
        axis.text(x + width / 2, y + 0.18, body, ha="center", va="center", fontsize=6.6, linespacing=1.25)
    for x in (0.175, 0.375, 0.575, 0.775):
        axis.add_patch(
            FancyArrowPatch(
                (x + 0.006, y + height / 2),
                (x + 0.043, y + height / 2),
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=1.0,
                color="#526778",
            )
        )

    axis.text(
        0.5,
        0.92,
        "From predictive uncertainty to a decision-effective dynamics ambiguity set",
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
    )
    axis.text(0.50, 0.16, "Finite-sample statement: marginal one-transition coverage under exchangeability", ha="center", va="center", color="#9A5A22", fontsize=6.6)
    axis.text(0.50, 0.07, "Empirical statement: control benefit must be established independently; coverage alone is insufficient", ha="center", va="center", color="#4A6650", fontsize=6.6)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    if write_tiff:
        figure.savefig(output_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-stem", type=Path, default=Path("figures/method_01_chain_schematic"))
    parser.add_argument("--tiff", action="store_true")
    args = parser.parse_args()
    draw(args.output_stem, args.tiff)


if __name__ == "__main__":
    main()
