"""Draw the paper's uncertainty-to-control schematic and claim ledger."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

INK = "#273444"
MUTED = "#667585"
BLUE = "#315F8C"
BLUE_LIGHT = "#E4EEF7"
TEAL = "#347C78"
TEAL_LIGHT = "#E1F0EE"
GOLD = "#A66A1F"
GOLD_LIGHT = "#F7EBD8"
VIOLET = "#70558B"
VIOLET_LIGHT = "#EEE8F3"
RED = "#A6463D"
RED_LIGHT = "#F6E5E2"


def rounded_box(axis, xy, width, height, face, title, body, edge=INK, title_color=INK):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.010,rounding_size=0.016",
        linewidth=0.9,
        edgecolor=edge,
        facecolor=face,
    )
    axis.add_patch(patch)
    x, y = xy
    axis.text(
        x + 0.018,
        y + height - 0.050,
        title,
        ha="left",
        va="center",
        fontsize=7.3,
        fontweight="bold",
        color=title_color,
    )
    axis.text(
        x + 0.018,
        y + height - 0.105,
        body,
        ha="left",
        va="top",
        fontsize=6.35,
        linespacing=1.28,
        color=INK,
    )
    return patch


def arrow(axis, start, end, color=MUTED, width=1.05, connectionstyle="arc3"):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=width,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=2,
            shrinkB=2,
        )
    )


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
    figure, axis = plt.subplots(figsize=(7.25, 4.25), constrained_layout=True)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(
        0.02,
        0.965,
        "Predictive scale  →  calibrated function-space set  →  decision-effective robust control",
        ha="left",
        va="center",
        fontsize=9.3,
        fontweight="bold",
        color=INK,
    )
    axis.text(
        0.02,
        0.925,
        "The FNO is a replaceable world model; the contribution is the uncertainty-to-decision interface.",
        ha="left",
        va="center",
        fontsize=6.55,
        color=MUTED,
    )

    # a: proper training, with two explicit operator branches.
    rounded_box(
        axis,
        (0.02, 0.54),
        0.205,
        0.33,
        BLUE_LIGHT,
        "a  Proper training split",
        r"$\mathcal{D}_{\mathrm{tr}}$ is used twice" + "\nclean targets  →  base FNO" + "\nperturbed targets  →  replica",
        edge=BLUE,
        title_color=BLUE,
    )
    axis.add_patch(Rectangle((0.044, 0.584), 0.066, 0.035, facecolor="white", edgecolor=BLUE, lw=0.7))
    axis.add_patch(Rectangle((0.135, 0.584), 0.066, 0.035, facecolor="white", edgecolor=BLUE, lw=0.7))
    axis.text(0.077, 0.601, r"$\widehat G$", ha="center", va="center", fontsize=6.7, color=BLUE)
    axis.text(0.168, 0.601, r"$\widetilde G$", ha="center", va="center", fontsize=6.7, color=BLUE)
    axis.text(0.122, 0.555, r"$\sigma=\mathrm{Smooth}(|\widehat G-\widetilde G|)\vee\tau_0$", ha="center", va="center", fontsize=5.8)

    # b: disjoint audit and split conformal quantile.
    rounded_box(
        axis,
        (0.265, 0.54),
        0.205,
        0.33,
        GOLD_LIGHT,
        "b  Deployment audit split",
        r"$\mathcal{D}_{\mathrm{cal}}\perp\mathcal{D}_{\mathrm{tr}}$" + "\nnormalized residual score" + "\nfinite-sample order statistic",
        edge=GOLD,
        title_color=GOLD,
    )
    axis.text(0.367, 0.612, r"$S_i=\| (y_i-\widehat G_i)/\sigma_i\|$", ha="center", va="center", fontsize=6.15)
    axis.text(0.367, 0.568, r"$q=S_{(\lceil(m+1)(1-\alpha)\rceil)}$", ha="center", va="center", fontsize=6.15, fontweight="bold", color=GOLD)

    # c: function-space geometry and support functions.
    rounded_box(
        axis,
        (0.510, 0.54),
        0.205,
        0.33,
        TEAL_LIGHT,
        "c  Dynamics ambiguity set",
        r"$\mathcal{U}(x,a)=\widehat G+\sigma\odot Z_q$" + "\nellipsoid or simultaneous box" + "\ngeometry controls support",
        edge=TEAL,
        title_color=TEAL,
    )
    axis.add_patch(Ellipse((0.565, 0.585), 0.075, 0.045, facecolor="white", edgecolor=TEAL, lw=0.9))
    axis.add_patch(Rectangle((0.625, 0.5625), 0.055, 0.045, facecolor="white", edgecolor=TEAL, lw=0.9))
    axis.text(0.622, 0.625, r"$h_{\mathcal{U}}(\lambda)$", ha="center", va="center", fontsize=6.1, color=TEAL)

    # d: planning block, intentionally dominant.
    rounded_box(
        axis,
        (0.755, 0.49),
        0.225,
        0.38,
        VIOLET_LIGHT,
        "d  Decision-effective MPC",
        "candidate actions  →  FNO rollout\ninner query: adjoint support or PGD\nouter search: receding-horizon CEM",
        edge=VIOLET,
        title_color=VIOLET,
    )
    axis.text(0.868, 0.585, r"$\min_{\mathbf{a}}\;\max_{\Delta_{0:H-1}\in\mathcal{U}^{H}} J_H$", ha="center", va="center", fontsize=7.0, fontweight="bold", color=VIOLET)
    axis.text(0.868, 0.535, r"fast: $q\|\lambda\odot\sigma\|$   |   nonlinear: PGD", ha="center", va="center", fontsize=5.9)

    arrow(axis, (0.225, 0.705), (0.265, 0.705), BLUE)
    arrow(axis, (0.470, 0.705), (0.510, 0.705), GOLD)
    arrow(axis, (0.715, 0.705), (0.755, 0.705), TEAL)

    # Closed-loop deployment connection.
    plant = FancyBboxPatch((0.755, 0.345), 0.105, 0.075, boxstyle="round,pad=0.008", facecolor="white", edgecolor=INK, lw=0.85)
    controller = FancyBboxPatch((0.875, 0.345), 0.105, 0.075, boxstyle="round,pad=0.008", facecolor="white", edgecolor=VIOLET, lw=0.85)
    axis.add_patch(plant)
    axis.add_patch(controller)
    axis.text(0.8075, 0.382, "deployed PDE", ha="center", va="center", fontsize=6.2, fontweight="bold")
    axis.text(0.9275, 0.382, "robust MPC", ha="center", va="center", fontsize=6.2, fontweight="bold", color=VIOLET)
    arrow(axis, (0.875, 0.382), (0.860, 0.382), VIOLET)
    arrow(axis, (0.807, 0.345), (0.928, 0.345), INK, connectionstyle="arc3,rad=-0.35")
    axis.text(0.864, 0.322, r"state $x_t$ / action $a_t$", ha="center", va="center", fontsize=5.8, color=MUTED)
    arrow(axis, (0.868, 0.49), (0.927, 0.420), VIOLET)

    # Guarantee ledger: three statements that must not be collapsed.
    axis.text(0.02, 0.405, "e  Claim and guarantee ledger", ha="left", va="center", fontsize=7.3, fontweight="bold", color=INK)
    ledger = [
        (0.02, BLUE_LIGHT, BLUE, "Finite-sample statistical", "Exchangeable audit + split CP\nimplies marginal field coverage"),
        (0.263, RED_LIGHT, RED, "Deterministic propagation", r"uniform $\epsilon$ + Lipschitz dynamics" + "\nimplies rollout/value bounds"),
        (0.506, TEAL_LIGHT, TEAL, "Independent decision evidence", "matched closed-loop tests\nimply mean/tail-cost comparison"),
    ]
    for x, face, edge, title, body in ledger:
        rounded_box(axis, (x, 0.075), 0.205, 0.265, face, title, body, edge=edge, title_color=edge)
    for x in (0.235, 0.478):
        axis.text(x, 0.205, "≠", ha="center", va="center", fontsize=12, fontweight="bold", color=RED)
    axis.text(0.755, 0.206, "No automatic implication", ha="left", va="center", fontsize=6.25, fontweight="bold", color=RED)
    axis.text(0.755, 0.159, "coverage is not a safety certificate\nand does not prove control benefit", ha="left", va="center", fontsize=6.15, linespacing=1.3, color=INK)

    # Small signal trace for a PDE-flavoured visual cue without using raster art.
    xs = np.linspace(0.767, 0.965, 120)
    ys = 0.105 + 0.018 * np.sin(10 * np.pi * (xs - xs.min()) / (xs.max() - xs.min())) * np.exp(-1.6 * (xs - xs.min()))
    axis.plot(xs, ys, color=VIOLET, lw=1.0)

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
