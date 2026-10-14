"""ψ(R,Z) is a field; F is constant on its level sets. No 1-D ψ axis."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.interpolate import RegularGridInterpolator

from make_tokamak_symbols import (
    ACCENT,
    GRAY,
    ORANGE,
    TEAL,
    faint_lcfs,
    load_frame,
    save,
    style_poloidal,
)

LEVELS = (0.25, 0.55, 0.85)
F_EDGE = 1.0
F_BUMP = 0.18


def _psi_ends(d):
    interp = RegularGridInterpolator((d["Z"], d["R"]), d["psi"], bounds_error=False)
    psi_ax = float(interp([[d["z_ax"], d["r_ax"]]])[0])
    pts = np.column_stack([d["lc_z"], d["lc_r"]])
    psi_b = float(np.nanmedian(interp(pts)))
    return psi_ax, psi_b


def _F_of_psi(psi, psi_ax, psi_b):
    t = np.clip((psi - psi_ax) / (psi_b - psi_ax), 0.0, 1.0)
    return F_EDGE + F_BUMP * (1.0 - t) ** 1.3


def main() -> None:
    d = load_frame()
    psi_ax, psi_b = _psi_ends(d)
    t = np.where(d["inside"], (d["psi"] - psi_ax) / (psi_b - psi_ax), np.nan)
    psi_show = np.where(d["inside"], d["psi"], np.nan)
    Fmap = np.where(d["inside"], _F_of_psi(d["psi"], psi_ax, psi_b), np.nan)

    fig, (ax, fx) = plt.subplots(1, 2, figsize=(10.4, 4.6), dpi=140)

    style_poloidal(ax, d, xlim=(0.7, 2.45), ylim=(-1.55, 1.55))
    cf = ax.contourf(d["RR"], d["ZZ"], psi_show, levels=12, cmap="Purples", zorder=1)
    ax.contour(d["RR"], d["ZZ"], psi_show, levels=12, colors=GRAY, linewidths=0.4, zorder=2)
    faint_lcfs(ax, d)
    ax.plot(d["r_ax"], d["z_ax"], "o", color=ACCENT, ms=6, zorder=6)
    ax.text(d["r_ax"] - 0.42, d["z_ax"] + 0.10, "axis", color=ACCENT, fontsize=9)
    mid = LEVELS[1]
    ax.contour(d["RR"], d["ZZ"], t, levels=[mid], colors=TEAL, linewidths=2.2, zorder=4)
    row = np.nanargmin(np.abs(d["Z"]))
    xs, ys = d["R"], t[row]
    good = np.isfinite(ys)
    crossings = np.where(np.diff(np.sign(ys[good] - mid)))[0]
    r_hits = xs[good][crossings]
    if len(r_hits) >= 2:
        rin, rout = float(r_hits[0]), float(r_hits[-1])
        ax.plot([rin, rout], [0, 0], "o", color=TEAL, ms=6, zorder=7)
        ax.annotate(
            rf"same $\psi$, two $R$" "\n" rf"${rin:.2f}\,\mathrm{{m}}$, ${rout:.2f}\,\mathrm{{m}}$",
            xy=(rout, 0.0),
            xytext=(2.00, 0.58),
            fontsize=8,
            color=TEAL,
            arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=1.0),
        )
    ax.text(1.05, -1.22, "LCFS", color=ORANGE, fontsize=9)
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$\psi(R,Z)$  ($\mathrm{V\cdot s/rad}$)")
    ax.set_title(r"$\psi$ is a field on the slice", fontsize=11)

    style_poloidal(fx, d, xlim=(0.7, 2.45), ylim=(-1.55, 1.55))
    ff = fx.contourf(d["RR"], d["ZZ"], Fmap, levels=12, cmap="GnBu", zorder=1)
    for lev, lw in zip(LEVELS, (1.6, 2.2, 1.6)):
        fx.contour(d["RR"], d["ZZ"], t, levels=[lev], colors=TEAL, linewidths=lw, zorder=4)
    faint_lcfs(fx, d)
    fx.plot(d["r_ax"], d["z_ax"], "o", color=ACCENT, ms=6, zorder=6)
    fx.text(1.05, -1.22, "LCFS", color=ORANGE, fontsize=9)
    cb2 = fig.colorbar(ff, ax=fx, fraction=0.046, pad=0.04)
    cb2.set_label(r"$F$  ($\mathrm{T\cdot m}$), schematic")
    fx.set_title(r"$F$ is constant on each $\psi$ loop", fontsize=11)
    fx.legend(
        handles=[Line2D([0], [0], color=TEAL, lw=2.2, label=r"same $F$ on a whole loop")],
        loc="upper right",
        fontsize=8,
        frameon=True,
    )
    print("wrote", save(fig, "tokamak_F_of_psi.png"))


if __name__ == "__main__":
    main()
