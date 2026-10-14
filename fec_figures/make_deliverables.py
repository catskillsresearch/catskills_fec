"""q95 sits on a near-edge track; betaN is p over magnetic pressure, then scaled."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse
from scipy.interpolate import RegularGridInterpolator

from make_tokamak_symbols import (
    ACCENT,
    BG,
    BLUE,
    GRAY,
    ORANGE,
    PURPLE,
    TEAL,
    faint_lcfs,
    load_frame,
    save,
    style_poloidal,
)

OUT = Path(__file__).resolve().parent
Q_EDGE = 5.0  # cartoon edge twist, not a fit to 203702


def fig_q95() -> Path:
    d = load_frame()
    psi, R, Z = d["psi"], d["R"], d["Z"]
    interp = RegularGridInterpolator((Z, R), psi, bounds_error=False)
    psi_ax = float(interp([[d["z_ax"], d["r_ax"]]])[0])
    pts = np.column_stack([d["lc_z"], d["lc_r"]])
    psi_b = float(np.nanmedian(interp(pts)))
    depth = psi_b - psi_ax
    psi_n = np.where(d["inside"], (psi - psi_ax) / depth, np.nan)

    fig, (ax, ux) = plt.subplots(1, 2, figsize=(10.4, 4.6), dpi=140)
    style_poloidal(ax, d)
    ax.contour(d["RR"], d["ZZ"], psi_n, levels=[0.3, 0.55, 0.75], colors=GRAY, linewidths=0.9)
    ax.contour(d["RR"], d["ZZ"], psi_n, levels=[0.95], colors=PURPLE, linewidths=2.6)
    faint_lcfs(ax, d)
    ax.plot(d["r_ax"], d["z_ax"], "o", color=ACCENT, ms=7, zorder=6)
    ax.text(d["r_ax"] - 0.50, d["z_ax"] + 0.12, "axis", color=ACCENT, fontsize=9)
    ax.text(2.32, -0.08, r"$q_{95}$", color=PURPLE, fontsize=11)
    ax.text(1.05, -1.22, "LCFS", color=ORANGE, fontsize=9)
    ax.set_title("same nested tracks — pick the 95% one", fontsize=11)

    phi = np.linspace(0.0, Q_EDGE, 500)
    theta = phi / Q_EDGE
    ux.set_facecolor(BG)
    ux.plot(phi, theta, color=PURPLE, lw=2.4)
    ux.plot([Q_EDGE], [1.0], "o", color=PURPLE, ms=7)
    ux.set_xlim(0, Q_EDGE + 0.15)
    ux.set_ylim(0, 1.18)
    ux.set_xlabel("laps the long way")
    ux.set_ylabel("laps the short way")
    ux.set_title(rf"unroll that purple track  —  $q_{{95}}={Q_EDGE:.0f}$", fontsize=11, color=PURPLE)
    ux.axhline(1.0, color=GRAY, lw=0.8, ls="--")
    ux.axvline(Q_EDGE, color=GRAY, lw=0.8, ls="--")
    ux.text(Q_EDGE - 1.55, 1.05, "once around the D", color=GRAY, fontsize=8)
    ux.text(0.15, 0.08, "five long-way laps", color=GRAY, fontsize=8)
    ux.grid(True, color="#d8d8d4", lw=0.6)
    return save(fig, "tokamak_q95.png")


def fig_betan() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.8), dpi=140)

    ax = axes[0]
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.45, 1.45)
    ax.set_axis_off()
    ax.set_title("balloon shove  $p$", fontsize=11, color=ORANGE)
    for s, alpha in [(1.0, 0.18), (0.7, 0.28), (0.4, 0.42)]:
        ax.add_patch(Ellipse((0, 0), 2.0 * s, 2.5 * s, fc=ORANGE, ec="none", alpha=alpha))
    ax.add_patch(Ellipse((0, 0), 2.0, 2.5, fill=False, ec=ORANGE, lw=1.8))
    for ang in (90, 210, 330):
        rad = np.deg2rad(ang)
        ax.annotate(
            "",
            xy=(1.05 * np.cos(rad), 1.25 * np.sin(rad)),
            xytext=(0.35 * np.cos(rad), 0.45 * np.sin(rad)),
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4),
        )
    ax.text(0, 0, "hot\ncore", ha="center", va="center", fontsize=9, color="#333")

    ax = axes[1]
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.45, 1.45)
    ax.set_axis_off()
    ax.set_title(r"field shove  $B^2/(2\mu_0)$", fontsize=11, color=BLUE)
    ax.add_patch(Ellipse((0, 0), 2.0, 2.5, fill=False, ec=BLUE, lw=1.8))
    for ang in (90, 210, 330):
        rad = np.deg2rad(ang)
        ax.annotate(
            "",
            xy=(0.35 * np.cos(rad), 0.45 * np.sin(rad)),
            xytext=(1.05 * np.cos(rad), 1.25 * np.sin(rad)),
            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4),
        )
    ax.text(0, 0, "holds\nstill", ha="center", va="center", fontsize=9, color="#333")

    ax = axes[2]
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_title(r"ratio, then a size ruler", fontsize=11, color=TEAL)
    ax.text(0.50, 0.78, r"$\beta = p / [B^2 / (2\mu_0)]$", ha="center", fontsize=12, color="#333")
    ax.text(0.50, 0.52, r"$\beta_N = 100\,\beta\, a B_{t0} / I_p$", ha="center", fontsize=12, color=TEAL)
    ax.add_patch(Ellipse((0.28, 0.18), 0.22, 0.22, fill=False, ec=TEAL, lw=1.6))
    ax.add_patch(Ellipse((0.72, 0.18), 0.38, 0.28, fill=False, ec=TEAL, lw=1.6))
    ax.text(0.28, 0.18, "skinny", ha="center", va="center", fontsize=8, color=TEAL)
    ax.text(0.72, 0.18, "fat", ha="center", va="center", fontsize=8, color=TEAL)
    ax.text(0.50, 0.02, r"same $\beta_N$ can mean different raw $\beta$", ha="center", fontsize=8, color=GRAY)
    return save(fig, "tokamak_betan.png")


def fig_model_scalars() -> Path:
    fig, ax = plt.subplots(figsize=(9.6, 2.8), dpi=140)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    ax.set_axis_off()
    ax.set_facecolor(BG)

    def box(x, y, w, h, text, color, *, dashed=False):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.08,rounding_size=0.12",
                fc="white",
                ec=color,
                lw=1.6,
                ls="--" if dashed else "-",
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, color=color)

    box(0.3, 1.05, 2.2, 1.15, r"track map" "\n" r"$\psi(R,Z)$", BLUE)
    box(3.6, 2.05, 2.1, 0.95, r"$F$" "\n" r"(not submitted)", PURPLE, dashed=True)
    box(3.6, 0.25, 2.1, 0.95, r"$p$" "\n" r"(not submitted)", ORANGE, dashed=True)
    box(7.4, 2.05, 2.2, 0.95, r"edge twist" "\n" r"$q_{95}$", PURPLE)
    box(7.4, 0.25, 2.2, 0.95, r"normalised $p$" "\n" r"$\beta_N$", ORANGE)
    for y in (2.52, 0.72):
        ax.add_patch(
            FancyArrowPatch((2.55, 1.6), (3.55, y), arrowstyle="-|>", mutation_scale=12, color=GRAY, lw=1.1)
        )
    ax.annotate("", xy=(7.35, 2.52), xytext=(5.75, 2.52), arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.4))
    ax.annotate("", xy=(7.35, 0.72), xytext=(5.75, 0.72), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4))
    ax.text(6.55, 2.72, "you guess", color=PURPLE, fontsize=8, ha="center")
    ax.text(6.55, 0.92, "you guess", color=ORANGE, fontsize=8, ha="center")
    return save(fig, "tokamak_model_scalars.png")


def main() -> None:
    for fn in (fig_model_scalars, fig_q95, fig_betan):
        print("wrote", fn())


if __name__ == "__main__":
    main()
