"""Motivate BR, BZ, Bφ: across vs along vs out of the page."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

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


def _components(d):
    psi, R, Z = d["psi"], d["R"], d["Z"]
    dpsidR = np.gradient(psi, R, axis=1)
    dpsidZ = np.gradient(psi, Z, axis=0)
    BR = -dpsidZ / np.maximum(d["RR"], 1e-6)
    BZ = dpsidR / np.maximum(d["RR"], 1e-6)
    return dpsidR, dpsidZ, BR, BZ


def _grid(d, step=5):
    mask = d["inside"]
    sl = (slice(2, None, step), slice(2, None, step))
    return sl, mask[sl]


def fig_across_along() -> None:
    d = load_frame()
    dpsidR, dpsidZ, BR, BZ = _components(d)
    sl, m = _grid(d)
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.4), dpi=140)

    ax = axes[0]
    style_poloidal(ax, d, xlim=(0.7, 2.45), ylim=(-1.55, 1.55))
    ax.contour(d["RR"], d["ZZ"], d["psi"], levels=12, colors=GRAY, linewidths=0.5, zorder=1)
    faint_lcfs(ax, d)
    ax.quiver(
        d["RR"][sl][m], d["ZZ"][sl][m], dpsidR[sl][m], dpsidZ[sl][m],
        color=PURPLE, angles="xy", scale_units="xy", scale=2.4, width=0.004, zorder=4,
    )
    ax.plot(d["r_ax"], d["z_ax"], "o", color=ACCENT, ms=5, zorder=5)
    ax.set_title(r"$\nabla\psi$  points across the tracks", fontsize=11, color=PURPLE)

    ax = axes[1]
    style_poloidal(ax, d, xlim=(0.7, 2.45), ylim=(-1.55, 1.55))
    ax.contour(d["RR"], d["ZZ"], d["psi"], levels=12, colors=GRAY, linewidths=0.5, zorder=1)
    faint_lcfs(ax, d)
    ax.quiver(
        d["RR"][sl][m], d["ZZ"][sl][m], BR[sl][m], np.zeros_like(BR[sl][m]),
        color=ACCENT, angles="xy", scale_units="xy", scale=1.6, width=0.0035, zorder=4,
    )
    ax.quiver(
        d["RR"][sl][m], d["ZZ"][sl][m], np.zeros_like(BZ[sl][m]), BZ[sl][m],
        color=BLUE, angles="xy", scale_units="xy", scale=1.6, width=0.0035, zorder=4,
    )
    ax.plot(d["r_ax"], d["z_ax"], "o", color=ACCENT, ms=5, zorder=5)
    ax.set_title(r"$B_R$ (red) and $B_Z$ (blue) run along", fontsize=11)

    ax = axes[2]
    style_poloidal(ax, d, xlim=(0.7, 2.45), ylim=(-1.55, 1.55))
    ax.contour(d["RR"], d["ZZ"], d["psi"], levels=12, colors=GRAY, linewidths=0.5, zorder=1)
    faint_lcfs(ax, d)
    ax.plot(d["r_ax"], d["z_ax"], "o", color=ACCENT, ms=5, zorder=5)
    rs = np.linspace(1.15, 2.15, 5)
    from matplotlib.path import Path as MPath
    poly = MPath(np.column_stack([d["lc_r"], d["lc_z"]]))
    for r in rs:
        if not poly.contains_point((r, 0.0)):
            continue
        ax.add_patch(Circle((r, 0.0), 0.055, fill=False, ec=TEAL, lw=1.6, zorder=6))
        ax.plot(r, 0.0, "o", color=TEAL, ms=3.5, zorder=6)
    ax.set_title(r"$B_\varphi$  out of the page (long way)", fontsize=11, color=TEAL)
    print("wrote", save(fig, "tokamak_b_across_along.png"))


def fig_turn() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6), dpi=140)

    def frame(ax, title, color):
        ax.set_facecolor(BG)
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title, fontsize=11, color=color)
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(0.82 * np.cos(th), 0.55 * np.sin(th), color=GRAY, lw=1.8)
        ax.plot(0.52 * np.cos(th), 0.34 * np.sin(th), color=GRAY, lw=1.1)

    ax = axes[0]
    frame(ax, r"$\nabla\psi$  across  (hops layers)", PURPLE)
    ax.annotate("", xy=(0.95, 0.08), xytext=(0.22, 0.08),
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=2.2, mutation_scale=14))
    ax.text(0.55, 0.22, r"$\nabla\psi$", color=PURPLE, fontsize=11, ha="center")
    ax.text(0.0, -0.95, "steepest change of " r"$\psi$", color=GRAY, fontsize=8, ha="center")

    ax = axes[1]
    frame(ax, r"turn $90^\circ$ with $\mathbf{e}_\varphi$", "#333")
    ax.annotate("", xy=(0.95, 0.08), xytext=(0.22, 0.08),
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.4, mutation_scale=12))
    ax.annotate("", xy=(0.22, 0.78), xytext=(0.22, 0.08),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.2, mutation_scale=14))
    ax.annotate(
        "", xy=(0.12, 0.55), xytext=(0.72, 0.12),
        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.3,
                        connectionstyle="arc3,rad=-0.45"),
    )
    ax.text(0.58, -0.08, r"$\nabla\psi$", color=PURPLE, fontsize=10)
    ax.text(0.32, 0.82, r"$B_Z$", color=BLUE, fontsize=11)
    ax.text(0.55, 0.42, r"$90^\circ$", color=ORANGE, fontsize=9)
    ax.add_patch(Circle((0.22, 0.08), 0.06, fill=False, ec=TEAL, lw=1.5))
    ax.plot(0.22, 0.08, "o", color=TEAL, ms=3)
    ax.text(0.34, -0.22, r"$\mathbf{e}_\varphi$", color=TEAL, fontsize=9)

    ax = axes[2]
    frame(ax, r"pieces: $B_R$ sideways, $B_Z$ up", "#333")
    ax.annotate("", xy=(0.95, 0.08), xytext=(0.22, 0.08),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2.2, mutation_scale=14))
    ax.annotate("", xy=(0.22, 0.78), xytext=(0.22, 0.08),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.2, mutation_scale=14))
    ax.text(0.70, -0.12, r"$B_R$", color=ACCENT, fontsize=12)
    ax.text(0.00, 0.82, r"$B_Z$", color=BLUE, fontsize=12)
    ax.text(0.0, -0.95, r"along the track  —  no hop", color=GRAY, fontsize=8, ha="center")
    print("wrote", save(fig, "tokamak_b_turn.png"))


def main() -> None:
    fig_across_along()
    fig_turn()


if __name__ == "__main__":
    main()
