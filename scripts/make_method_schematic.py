"""Draw the closed-loop uncertainty-to-control schematic used in the paper.

Figure contract
---------------
Core conclusion: calibrated predictive uncertainty becomes operational only
when the learned set is repeatedly queried inside a receding-horizon feedback
loop with the physical PDE plant.
Archetype: schematic-led method figure.
Backend: Python/matplotlib only.
Export: editable SVG/PDF plus a 400-dpi PNG preview (and optional TIFF).
Reviewer risk: one-step conformal calibration must not be depicted as a
closed-loop safety guarantee; offline transfers are therefore dashed and the
online feedback loop is solid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#243447"
MUTED = "#6E7F91"
LINE = "#8796A7"
BLUE = "#315F8C"
BLUE_LIGHT = "#EAF1F8"
TEAL = "#2F7E79"
TEAL_LIGHT = "#E7F3F1"
GOLD = "#B8792A"
GOLD_LIGHT = "#FAF1E3"
VIOLET = "#71558B"
VIOLET_LIGHT = "#F0EBF5"
PLANT = "#3E5268"
PLANT_LIGHT = "#EDF1F4"


def rounded_box(
    axis,
    xy: tuple[float, float],
    width: float,
    height: float,
    face: str,
    edge: str,
    title: str,
    symbol: str,
    equation: str,
    *,
    title_size: float = 7.1,
) -> None:
    x0, y0 = xy
    axis.add_patch(
        FancyBboxPatch(
            (x0, y0),
            width,
            height,
            boxstyle="round,pad=0.009,rounding_size=0.018",
            linewidth=0.95,
            edgecolor=edge,
            facecolor=face,
        )
    )
    axis.text(
        x0 + width / 2,
        y0 + 0.72 * height,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=edge,
    )
    axis.text(
        x0 + width / 2,
        y0 + 0.43 * height,
        symbol,
        ha="center",
        va="center",
        fontsize=7.2,
        color=INK,
    )
    axis.text(
        x0 + width / 2,
        y0 + 0.16 * height,
        equation,
        ha="center",
        va="center",
        fontsize=5.9,
        color=MUTED,
    )


def straight_arrow(
    axis,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = LINE,
    dashed: bool = False,
    width: float = 1.05,
    label: str | None = None,
    label_xy: tuple[float, float] | None = None,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8.5,
            linewidth=width,
            linestyle=(0, (3, 2)) if dashed else "solid",
            color=color,
            shrinkA=1,
            shrinkB=1,
        )
    )
    if label and label_xy:
        axis.text(
            *label_xy,
            label,
            ha="center",
            va="center",
            fontsize=5.9,
            color=color,
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

    figure, axis = plt.subplots(figsize=(7.2, 3.15), constrained_layout=True)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    # Phase labels make the guarantee boundary explicit.
    axis.text(
        0.018,
        0.953,
        "OFFLINE  learn + audit-calibrate",
        fontsize=6.4,
        fontweight="bold",
        color=BLUE,
    )
    axis.plot([0.018, 0.982], [0.925, 0.925], color="#D8E0E8", lw=0.8)
    axis.text(
        0.018,
        0.522,
        "ONLINE  receding-horizon feedback",
        fontsize=6.4,
        fontweight="bold",
        color=VIOLET,
    )

    # Offline row: proper training and audit calibration have distinct data roles.
    rounded_box(axis, (0.025, 0.625), 0.155, 0.235, BLUE_LIGHT, BLUE,
                "Data split", r"$D_{\rm tr},D_{\rm val},D_{\rm cal}$", "Sec. 2.3")
    rounded_box(
        axis,
        (0.225, 0.625),
        0.175,
        0.235,
        BLUE_LIGHT,
        BLUE,
        "Twin FNOs",
        r"$\widehat G_\theta,\widetilde G_{\widetilde\theta}$",
        "Eqs. (14)--(16)",
    )
    rounded_box(axis, (0.445, 0.625), 0.160, 0.235, TEAL_LIGHT, TEAL,
                "Spatial scale", r"$\sigma_\theta(u,a)$", "Eqs. (17)--(19)")
    rounded_box(axis, (0.650, 0.625), 0.165, 0.235, GOLD_LIGHT, GOLD,
                "Split conformal", r"$q_g$ at $1-\alpha$", "Eqs. (12), (20)--(22)")
    rounded_box(axis, (0.860, 0.625), 0.115, 0.235, PLANT_LIGHT, PLANT,
                "Freeze", r"$\widehat G,\sigma,q_g$", "deploy", title_size=6.8)

    straight_arrow(axis, (0.182, 0.742), (0.222, 0.742), color=BLUE)
    straight_arrow(axis, (0.402, 0.742), (0.442, 0.742), color=TEAL)
    straight_arrow(axis, (0.607, 0.742), (0.647, 0.742), color=GOLD)
    straight_arrow(axis, (0.817, 0.742), (0.857, 0.742), color=PLANT)
    axis.text(0.300, 0.590, "proper-training labels", ha="center", fontsize=5.6, color=BLUE)
    axis.text(0.728, 0.590, "held-out audit labels", ha="center", fontsize=5.6, color=GOLD)

    # Online row.  The model/set and controller interact while the selected
    # action is applied only to the physical PDE plant.
    state_x, state_y = 0.043, 0.205
    axis.text(
        state_x,
        state_y + 0.095,
        r"observed field $u_t$",
        ha="center",
        fontsize=6.2,
        color=INK,
    )
    wave_x = np.linspace(0.012, 0.080, 100)
    phase = (wave_x - wave_x.min()) / (wave_x.max() - wave_x.min())
    axis.plot(wave_x, state_y + 0.020 + 0.025 * np.sin(2.2 * np.pi * phase), color=INK, lw=1.2)

    rounded_box(axis, (0.125, 0.125), 0.220, 0.285, TEAL_LIGHT, TEAL,
                "Calibrated dynamics set", r"$\mathcal{U}_g(u_t,a)$", "Eqs. (23)--(26)")
    rounded_box(axis, (0.445, 0.125), 0.180, 0.285, VIOLET_LIGHT, VIOLET,
                "Robust MPC", r"$a_t^*=\arg\min_a\max_\Delta J$", "Eq. (27)")
    rounded_box(axis, (0.740, 0.125), 0.205, 0.285, PLANT_LIGHT, PLANT,
                "Physical PDE plant", r"$u_{t+1}=G_B^{\Delta T}(u_t,a_t^*;\xi)$", "Eqs. (1), (3)")

    straight_arrow(axis, (0.080, 0.267), (0.122, 0.267), color=INK)
    straight_arrow(
        axis, (0.347, 0.310), (0.442, 0.310), color=TEAL,
        label="worst-case next fields", label_xy=(0.394, 0.340),
    )
    straight_arrow(
        axis, (0.442, 0.222), (0.347, 0.222), color=VIOLET,
        label=r"queries $(u,a)$", label_xy=(0.394, 0.191),
    )
    straight_arrow(
        axis, (0.627, 0.267), (0.737, 0.267), color=VIOLET, width=1.3,
        label=r"apply $a_t^*$", label_xy=(0.682, 0.303),
    )

    # Dashed arrows transfer frozen learned/calibrated objects to deployment.
    straight_arrow(axis, (0.905, 0.622), (0.330, 0.414), color=MUTED, dashed=True, width=0.9)
    axis.text(
        0.635,
        0.491,
        "frozen model, scale and quantile",
        ha="center",
        fontsize=5.6,
        color=MUTED,
    )

    # A long return arrow makes the physical feedback loop unambiguous.
    axis.add_patch(
        FancyArrowPatch(
            (0.842, 0.121),
            (0.047, 0.188),
            connectionstyle="arc3,rad=-0.10",
            arrowstyle="-|>",
            mutation_scale=9.5,
            linewidth=1.35,
            color=INK,
        )
    )
    axis.text(
        0.492,
        0.035,
        r"measure $u_{t+1}$, shift horizon, and repeat",
        ha="center",
        fontsize=6.1,
        color=INK,
    )

    axis.text(
        0.982,
        0.505,
        "dashed: offline transfer   |   solid: online feedback",
        ha="right",
        va="center",
        fontsize=5.5,
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
