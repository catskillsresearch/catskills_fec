"""Cartoon (R,Z) slice: hole, axes, nested psi tracks. Rebuilds tokamak_rz_slice.png."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from scipy.interpolate import RegularGridInterpolator

from make_tokamak_symbols import (
    ACCENT,
    BG,
    BLUE,
    GRAY,
    ORANGE,
    PURPLE,
    faint_lcfs,
    load_frame,
    save,
    style_poloidal,
)

OUT_NAME = "tokamak_rz_slice.png"


def main() -> None:
    d = load_frame()
    psi, R, Z = d["psi"], d["R"], d["Z"]
    interp = RegularGridInterpolator((Z, R), psi, bounds_error=False)
    psi_ax = float(interp([[d["z_ax"], d["r_ax"]]])[0])
    pts = np.column_stack([d["lc_z"], d["lc_r"]])
    psi_b = float(np.nanmedian(interp(pts)))
    psi_n = np.where(d["inside"], (psi - psi_ax) / (psi_b - psi_ax), np.nan)

    fig, ax = plt.subplots(figsize=(5.6, 5.4), dpi=140)
    style_poloidal(ax, d, xlim=(-0.15, 2.75), ylim=(-1.75, 1.75))
    ax.contour(d["RR"], d["ZZ"], psi_n, levels=[0.25, 0.5, 0.75, 0.92], colors=PURPLE, linewidths=1.0)
    faint_lcfs(ax, d)
    ax.plot(d["r_ax"], d["z_ax"], "o", color=ACCENT, ms=8, zorder=6)
    ax.axvline(0.0, color="#333", lw=1.0, ls="--", zorder=1)
    ax.add_patch(
        FancyArrowPatch((0.08, -0.08), (1.15, -0.08), arrowstyle="-|>", color=BLUE, lw=1.4, mutation_scale=12)
    )
    ax.add_patch(
        FancyArrowPatch((0.08, -0.08), (0.08, 0.85), arrowstyle="-|>", color=BLUE, lw=1.4, mutation_scale=12)
    )
    ax.text(1.18, -0.22, r"$R$", color=BLUE, fontsize=12)
    ax.text(0.14, 0.88, r"$Z$", color=BLUE, fontsize=12)
    ax.text(-0.08, 0.12, "central\nhole", color=GRAY, fontsize=8, ha="right", va="bottom")
    ax.text(d["r_ax"] + 0.10, d["z_ax"] + 0.14, "axis", color=ACCENT, fontsize=9)
    ax.text(2.12, 0.55, "nested\ntracks of " r"$\psi$", color=PURPLE, fontsize=9)
    ax.text(2.18, -1.38, "(R, Z) slice", color=ORANGE, fontsize=9)
    ax.set_title("one cut through the doughnut", fontsize=11)
    print("wrote", save(fig, OUT_NAME))


if __name__ == "__main__":
    main()
