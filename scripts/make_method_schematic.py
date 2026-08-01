"""Draw the compact uncertainty-to-control schematic used in the paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

INK = "#243447"
MUTED = "#718096"
BLUE = "#315F8C"
BLUE_LIGHT = "#EAF1F8"
TEAL = "#2F7E79"
TEAL_LIGHT = "#E7F3F1"
GOLD = "#B8792A"
GOLD_LIGHT = "#FAF1E3"
VIOLET = "#71558B"
VIOLET_LIGHT = "#F0EBF5"


def arrow(axis, x0: float, x1: float, y: float = 0.50, color: str = MUTED) -> None:
    axis.add_patch(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.05,
            color=color,
            shrinkA=1,
            shrinkB=1,
        )
    )


def card(axis, center: float, face: str, edge: str, title: str, symbol: str):
    width, y0, height = 0.155, 0.21, 0.61
    x0 = center - width / 2
    patch = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.010,rounding_size=0.022",
        linewidth=0.9,
        edgecolor=edge,
        facecolor=face,
    )
    axis.add_patch(patch)
    axis.text(
        center,
        0.745,
        title,
        ha="center",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color=edge,
    )
    axis.text(center, 0.275, symbol, ha="center", va="center", fontsize=7.2, color=INK)
    return x0, width


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

    figure, axis = plt.subplots(figsize=(7.2, 2.15), constrained_layout=True)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    centers = [0.095, 0.292, 0.489, 0.686, 0.895]
    cards = [
        card(axis, centers[0], BLUE_LIGHT, BLUE, "Twin FNOs", r"$\widehat G,\,\widetilde G$"),
        card(axis, centers[1], BLUE_LIGHT, BLUE, "Disagreement", r"$\sigma(x,a)$"),
        card(axis, centers[2], GOLD_LIGHT, GOLD, "Split conformal", r"$q_{1-\alpha}$"),
        card(axis, centers[3], TEAL_LIGHT, TEAL, "Field set", r"$\mathcal{U}(x,a)$"),
        card(axis, centers[4], VIOLET_LIGHT, VIOLET, "Robust MPC", r"$a_t^*$"),
    ]

    for (_, width), (next_x, _) in zip(cards[:-1], cards[1:]):
        current_x = next_x - (centers[1] - centers[0]) + width
        arrow(axis, current_x + 0.012, next_x - 0.012)

    # Twin-operator icon: identical inputs, clean and perturbed output traces.
    x = np.linspace(centers[0] - 0.052, centers[0] + 0.052, 80)
    phase = (x - x.min()) / (x.max() - x.min())
    axis.plot(x, 0.575 + 0.025 * np.sin(2.3 * np.pi * phase), color=BLUE, lw=1.35)
    axis.plot(
        x,
        0.505 + 0.025 * np.sin(2.3 * np.pi * phase + 0.22),
        color=BLUE,
        lw=1.15,
        ls="--",
    )
    axis.add_patch(
        Rectangle(
            (centers[0] - 0.058, 0.475),
            0.116,
            0.13,
            fill=False,
            ec=BLUE,
            lw=0.65,
        )
    )

    # Disagreement icon: a localized, spatially varying scale.
    x = np.linspace(centers[1] - 0.055, centers[1] + 0.055, 120)
    z = (x - x.min()) / (x.max() - x.min())
    scale = 0.013 + 0.045 * np.exp(-((z - 0.64) / 0.22) ** 2)
    axis.fill_between(x, 0.535 - scale, 0.535 + scale, color=TEAL, alpha=0.24, lw=0)
    axis.plot(x, 0.535 + 0.012 * np.sin(3 * np.pi * z), color=TEAL, lw=1.2)

    # Conformal icon: empirical scores and the selected finite-sample order statistic.
    score_x = centers[2] + np.linspace(-0.052, 0.052, 8)
    score_y = np.array([0.485, 0.497, 0.512, 0.526, 0.545, 0.566, 0.585, 0.628])
    axis.scatter(score_x, score_y, s=8, facecolor="white", edgecolor=GOLD, linewidth=0.8, zorder=3)
    axis.plot([centers[2] - 0.061, centers[2] + 0.061], [0.587, 0.587], color=GOLD, lw=1.15)

    # Function-space geometry icon: anisotropic set and decision direction.
    axis.add_patch(Ellipse((centers[3], 0.54), 0.104, 0.073, angle=18, fc="white", ec=TEAL, lw=1.1))
    axis.add_patch(
        Ellipse(
            (centers[3], 0.54),
            0.052,
            0.036,
            angle=18,
            fill=False,
            ec=TEAL,
            lw=0.65,
            ls="--",
        )
    )
    axis.add_patch(
        FancyArrowPatch(
            (centers[3] - 0.006, 0.536),
            (centers[3] + 0.058, 0.596),
            arrowstyle="-|>",
            mutation_scale=7,
            color=INK,
            lw=0.9,
        )
    )

    # Closed-loop icon: controller and deployed PDE linked in feedback.
    left, right, y = centers[4] - 0.037, centers[4] + 0.037, 0.54
    axis.add_patch(Circle((left, y), 0.026, fc="white", ec=VIOLET, lw=1.0))
    axis.add_patch(Circle((right, y), 0.026, fc="white", ec=INK, lw=1.0))
    axis.text(left, y, "M", ha="center", va="center", fontsize=6.2, fontweight="bold", color=VIOLET)
    axis.text(right, y, "P", ha="center", va="center", fontsize=6.2, fontweight="bold", color=INK)
    axis.add_patch(
        FancyArrowPatch(
            (left + 0.025, y + 0.012),
            (right - 0.025, y + 0.012),
            arrowstyle="-|>",
            mutation_scale=6,
            color=VIOLET,
            lw=0.9,
        )
    )
    axis.add_patch(
        FancyArrowPatch(
            (right - 0.025, y - 0.015),
            (left + 0.025, y - 0.015),
            arrowstyle="-|>",
            mutation_scale=6,
            color=INK,
            lw=0.9,
        )
    )

    # Three data roles, shown once rather than repeated inside every stage.
    axis.text(0.194, 0.91, "proper training", ha="center", va="center", fontsize=6.4, color=BLUE)
    axis.plot([0.035, 0.353], [0.875, 0.875], color=BLUE, lw=1.2)
    axis.text(0.588, 0.91, "deployment audit", ha="center", va="center", fontsize=6.4, color=GOLD)
    axis.plot([0.429, 0.747], [0.875, 0.875], color=GOLD, lw=1.2)
    axis.text(0.895, 0.91, "closed loop", ha="center", va="center", fontsize=6.4, color=VIOLET)
    axis.plot([0.818, 0.972], [0.875, 0.875], color=VIOLET, lw=1.2)

    axis.text(
        0.5,
        0.085,
        "predictive sensitivity  →  calibrated dynamics geometry  →  objective-aware action",
        ha="center",
        va="center",
        fontsize=6.7,
        color=MUTED,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    if write_tiff:
        figure.savefig(output_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path("figures/method_01_chain_schematic"),
    )
    parser.add_argument("--tiff", action="store_true")
    args = parser.parse_args()
    draw(args.output_stem, args.tiff)


if __name__ == "__main__":
    main()
