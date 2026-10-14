"""Pedagogical (R,Z) figures for fec.md — one highlight per symbol table."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from matplotlib.patches import Circle
from matplotlib.path import Path as MPath
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter, minimum_filter

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
MU0 = 4e-7 * np.pi
BG = "#f7f7f4"
ACCENT = "#c0392b"
BLUE = "#1a5f7a"
ORANGE = "#e07a2f"
TEAL = "#148f77"
PURPLE = "#6c3483"
GRAY = "#9aa3ad"


def load_frame() -> dict:
    table = pq.read_table(ROOT / "parquet_data" / "d3d_shot_203702.parquet")
    row = {c: table.column(c)[0].as_py() for c in table.column_names}
    t = int(np.asarray(row["efit_psirz"]).shape[0] // 2)
    psi = np.asarray(row["efit_psirz"], float)[t]
    R = np.asarray(row["efit_grid_R"], float)
    Z = np.asarray(row["efit_grid_Z"], float)
    RR, ZZ = np.meshgrid(R, Z)
    n = int(np.asarray(row["efit_lcfs_n"])[t])
    lc_r = np.asarray(row["efit_lcfs_r"], float)[t, :n]
    lc_z = np.asarray(row["efit_lcfs_z"], float)[t, :n]
    inside = MPath(np.column_stack([lc_r, lc_z])).contains_points(
        np.column_stack([RR.ravel(), ZZ.ravel()])
    ).reshape(RR.shape)
    return {
        "psi": psi,
        "R": R,
        "Z": Z,
        "RR": RR,
        "ZZ": ZZ,
        "lc_r": lc_r,
        "lc_z": lc_z,
        "inside": inside,
        "r_ax": float(np.asarray(row["efit_r_axis"])[t]),
        "z_ax": float(np.asarray(row["efit_z_axis"])[t]),
    }


def style_poloidal(ax, d, *, xlim=(0.55, 2.75), ylim=(-1.75, 1.75)):
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"major radius  $R$  (m)")
    ax.set_ylabel(r"vertical height  $Z$  (m)")
    ax.axhline(0.0, color="#d0d0cc", lw=0.6, zorder=0)
    ax.axvline(0.0, color="#d0d0cc", lw=0.6, zorder=0)


def faint_lcfs(ax, d):
    ax.plot(d["lc_r"], d["lc_z"], color=ORANGE, lw=1.6, zorder=3)


def save(fig, name: str) -> Path:
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def find_nulls(d) -> list[tuple[str, float, float, float]]:
    psi, R, Z = d["psi"], d["R"], d["Z"]
    dR = np.gradient(psi, R, axis=1)
    dZg = np.gradient(psi, Z, axis=0)
    g2 = dR**2 + dZg**2
    ismin = (g2 == minimum_filter(g2, size=5)) & (g2 < np.quantile(g2, 0.08))
    out = []
    dR0, dZ0 = R[1] - R[0], Z[1] - Z[0]
    for i, j in zip(*np.where(ismin)):
        if i < 3 or i >= len(Z) - 3 or j < 3 or j >= len(R) - 3:
            continue
        hrr = (psi[i, j + 1] - 2 * psi[i, j] + psi[i, j - 1]) / dR0**2
        hzz = (psi[i + 1, j] - 2 * psi[i, j] + psi[i - 1, j]) / dZ0**2
        hrz = (
            psi[i + 1, j + 1]
            - psi[i + 1, j - 1]
            - psi[i - 1, j + 1]
            + psi[i - 1, j - 1]
        ) / (4 * dR0 * dZ0)
        det = hrr * hzz - hrz**2
        kind = "O" if det > 0 else "X" if det < 0 else "?"
        out.append((kind, float(R[j]), float(Z[i]), float(np.sqrt(g2[i, j]))))
    out.sort(key=lambda x: x[3])
    return out


def lcfs_xpoint(d) -> tuple[float, float]:
    """X-point nearest the LCFS (the diverted saddle)."""
    xs = [(r, z, g) for k, r, z, g in find_nulls(d) if k == "X"]
    best = min(
        xs,
        key=lambda p: np.min(np.hypot(d["lc_r"] - p[0], d["lc_z"] - p[1])),
    )
    return best[0], best[1]


def delta_star(d, sigma=0.9):
    ps = gaussian_filter(d["psi"], sigma)
    dR = np.gradient(ps, d["R"], axis=1)
    dZ = np.gradient(ps, d["Z"], axis=0)
    term_R = d["RR"] * np.gradient(dR / np.maximum(d["RR"], 1e-6), d["R"], axis=1)
    term_Z = np.gradient(dZ, d["Z"], axis=0)
    return term_R + term_Z


def fig_coords(d) -> Path:
    """R, Z, φ — cutaway torus plus the poloidal cut."""
    fig = plt.figure(figsize=(11.2, 5.6), dpi=140)

    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    R0, a = 1.67, 0.58
    u = np.linspace(0.22 * np.pi, 2 * np.pi - 0.08 * np.pi, 90)
    v = np.linspace(0, 2 * np.pi, 56)
    U, V = np.meshgrid(u, v)
    X = (R0 + a * np.cos(V)) * np.cos(U)
    Y = (R0 + a * np.cos(V)) * np.sin(U)
    Zt = a * np.sin(V)
    ax3.plot_surface(
        X, Y, Zt, rstride=2, cstride=2, color="#c5d4e0", alpha=0.72, linewidth=0, shade=True
    )
    th = np.linspace(0, 2 * np.pi, 80)
    face = np.column_stack([(R0 + a * np.cos(th)), np.zeros_like(th), a * np.sin(th)])
    ax3.add_collection3d(
        Poly3DCollection([face], facecolor="#f4e4c8", edgecolor="#333", linewidths=0.9, alpha=0.97)
    )
    ax3.plot([0, 0], [0, 0], [-1.25, 1.25], color="#333", lw=1.3, ls="--")
    ax3.plot([0, R0], [0, 0], [0, 0], color=ACCENT, lw=2.6)
    ax3.scatter([R0], [0], [0], color=ACCENT, s=18)
    ax3.text(R0 * 0.42, -0.35, 0.18, r"$R$", color=ACCENT, fontsize=14)
    ax3.plot([R0, R0], [0, 0], [0, 0.95], color=BLUE, lw=2.6)
    ax3.text(R0 + 0.22, 0.12, 0.62, r"$Z$", color=BLUE, fontsize=14)
    ph = np.linspace(0.55, 2.15, 50)
    rp = R0 + 0.18
    ax3.plot(rp * np.cos(ph), rp * np.sin(ph), np.full_like(ph, 0.08), color=TEAL, lw=2.6)
    ax3.quiver(
        rp * np.cos(ph[-1]),
        rp * np.sin(ph[-1]),
        0.08,
        -0.45 * np.sin(ph[-1]),
        0.45 * np.cos(ph[-1]),
        0,
        color=TEAL,
        arrow_length_ratio=0.45,
        lw=2.0,
    )
    ax3.text(-0.15, 1.85, 0.35, r"$\varphi$", color=TEAL, fontsize=14)
    ax3.set_xlim(-2.3, 2.5)
    ax3.set_ylim(-2.3, 2.5)
    ax3.set_zlim(-1.2, 1.2)
    ax3.set_axis_off()
    ax3.set_box_aspect((1.7, 1.7, 1.0))
    ax3.view_init(elev=20, azim=-58)
    ax3.set_title(r"doughnut: $R$, $Z$, $\varphi$", fontsize=11, pad=2)

    ax = fig.add_subplot(1, 2, 2)
    style_poloidal(ax, d)
    ax.plot(d["lc_r"], d["lc_z"], color="#c5d4e0", lw=6, zorder=1)
    faint_lcfs(ax, d)
    ax.plot([0, 0], [-1.7, 1.7], color="#333", lw=1.1, ls="--", zorder=2)
    ax.annotate(
        "",
        xy=(d["r_ax"], 0.0),
        xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.8),
        zorder=7,
    )
    ax.text(d["r_ax"] * 0.42, 0.14, r"$R$", color=ACCENT, fontsize=13, fontweight="bold")
    ax.annotate(
        "",
        xy=(d["r_ax"], 0.95),
        xytext=(d["r_ax"], 0.0),
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.8),
        zorder=7,
    )
    ax.text(d["r_ax"] + 0.08, 0.52, r"$Z$", color=BLUE, fontsize=13, fontweight="bold")
    ax.plot(d["r_ax"], 0.0, "o", color=TEAL, ms=8, zorder=8)
    ax.text(0.62, -1.45, r"$\varphi$: around the doughnut, out of this page", color=TEAL, fontsize=9)
    ax.set_title(r"poloidal plane $(R,Z)$  —  $\varphi$ suppressed", fontsize=11)
    fig.text(0.5, 0.01, "DIII-D shot 203702, mid-flattop", ha="center", fontsize=9, color="#555")
    return save(fig, "tokamak_coords.png")


def fig_psi(d) -> Path:
    fig, ax = plt.subplots(figsize=(7.6, 7.0), dpi=140)
    style_poloidal(ax, d, xlim=(0.55, 2.55), ylim=(-1.75, 1.75))
    levels = np.linspace(d["psi"].min(), np.quantile(d["psi"], 0.92), 16)
    cf = ax.contourf(d["RR"], d["ZZ"], d["psi"], levels=levels, cmap="Blues_r", zorder=1)
    ax.contour(d["RR"], d["ZZ"], d["psi"], levels=levels, colors="#2c3e50", linewidths=0.65, zorder=2)
    faint_lcfs(ax, d)
    ax.plot(d["r_ax"], d["z_ax"], "o", color=ACCENT, ms=7, zorder=6)
    ax.text(
        0.62,
        0.55,
        r"magnetic axis" "\n" r"(extremum of $\psi$)",
        color=ACCENT,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9),
        zorder=8,
    )
    ax.annotate(
        "",
        xy=(d["r_ax"], d["z_ax"]),
        xytext=(0.95, 0.48),
        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.9),
        zorder=8,
    )
    # a mid closed contour, left side
    ax.annotate(
        r"$\psi=\mathrm{const}$" "\n" r"(a flux surface)",
        xy=(1.22, 0.42),
        xytext=(0.62, 1.35),
        fontsize=9,
        color=BLUE,
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.9),
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9),
        zorder=8,
    )
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$\psi(R,Z)$  (V·s/rad)")
    ax.set_title(r"poloidal flux  $\psi(R,Z)$")
    return save(fig, "tokamak_psi.png")


def fig_nulls(d) -> Path:
    fig, ax = plt.subplots(figsize=(7.6, 7.0), dpi=140)
    style_poloidal(ax, d, xlim=(0.55, 2.55), ylim=(-1.75, 1.75))
    psi, R, Z = d["psi"], d["R"], d["Z"]
    dpsidR = np.gradient(psi, R, axis=1)
    dpsidZ = np.gradient(psi, Z, axis=0)
    ax.contour(d["RR"], d["ZZ"], psi, levels=18, colors=GRAY, linewidths=0.5, zorder=1)
    faint_lcfs(ax, d)
    step = 4
    sl = (slice(1, None, step), slice(1, None, step))
    ax.quiver(
        d["RR"][sl],
        d["ZZ"][sl],
        dpsidR[sl],
        dpsidZ[sl],
        color=BLUE,
        angles="xy",
        scale_units="xy",
        scale=2.8,
        width=0.003,
        zorder=4,
    )
    ax.plot(d["r_ax"], d["z_ax"], "o", color=ACCENT, ms=8, zorder=6)
    ax.text(
        d["r_ax"] - 0.48,
        d["z_ax"] + 0.10,
        r"O-point  $\nabla\psi=0$",
        color=ACCENT,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.92),
        zorder=8,
    )
    r_x, z_x = lcfs_xpoint(d)
    XGREEN = "#00c853"
    ax.plot(r_x, z_x, "X", color=XGREEN, ms=12, mew=2.0, zorder=6)
    ax.text(
        r_x + 0.08,
        z_x - 0.22,
        r"X-point  $\nabla\psi=0$",
        color=XGREEN,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.92),
        zorder=8,
    )
    ax.text(
        0.62,
        1.55,
        r"arrows: $\nabla\psi$  in the $(R,Z)$ plane",
        fontsize=9,
        color=BLUE,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.88),
    )
    ax.set_title(r"magnetic nulls of $\nabla\psi$")
    return save(fig, "tokamak_nulls.png")


def fig_force(d) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 7.0), dpi=140)
    style_poloidal(ax, d, xlim=(0.55, 2.55), ylim=(-1.75, 1.75))
    ax.contour(d["RR"], d["ZZ"], d["psi"], levels=14, colors=GRAY, linewidths=0.45, zorder=1)
    faint_lcfs(ax, d)
    ax.plot(d["r_ax"], d["z_ax"], "o", color="#888", ms=5, zorder=4)
    r0, z0 = 2.02, 0.12
    ax.annotate(
        "",
        xy=(r0 - 0.32, z0),
        xytext=(r0, z0),
        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2.4),
        zorder=7,
    )
    ax.text(r0 - 0.36, z0 + 0.14, r"$\nabla p$", color=ACCENT, fontsize=12, ha="right")
    ax.annotate(
        "",
        xy=(r0 + 0.02, z0 + 0.38),
        xytext=(r0 + 0.02, z0 - 0.26),
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=2.4),
        zorder=7,
    )
    ax.text(r0 + 0.10, z0 + 0.34, r"$\mathbf{B}$", color=BLUE, fontsize=12)
    ax.add_patch(Circle((r0, z0), 0.075, fill=False, ec=PURPLE, lw=2.1, zorder=8))
    ax.plot(r0, z0, "o", color=PURPLE, ms=5.5, zorder=8)
    ax.text(r0 + 0.12, z0 - 0.08, r"$\mathbf{j}$  (toroidal)", color=PURPLE, fontsize=10)
    ax.text(
        0.62,
        -1.45,
        r"$\mathbf{j}\times\mathbf{B}=\nabla p$" "\n" r"magnetic force cancels the pressure gradient",
        fontsize=10,
        color="#222",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.92),
        zorder=8,
    )
    ax.set_title(r"ideal MHD force balance")
    return save(fig, "tokamak_force.png")


def fig_B(d) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 7.0), dpi=140)
    style_poloidal(ax, d, xlim=(0.55, 2.55), ylim=(-1.75, 1.75))
    psi, R, Z = d["psi"], d["R"], d["Z"]
    dpsidR = np.gradient(psi, R, axis=1)
    dpsidZ = np.gradient(psi, Z, axis=0)
    BR = -dpsidZ / np.maximum(d["RR"], 1e-6)
    BZ = dpsidR / np.maximum(d["RR"], 1e-6)
    ax.contour(d["RR"], d["ZZ"], psi, levels=14, colors=GRAY, linewidths=0.45, zorder=1)
    faint_lcfs(ax, d)
    mask = d["inside"]
    step = 5
    sl = (slice(2, None, step), slice(2, None, step))
    m = mask[sl]
    ax.quiver(
        d["RR"][sl][m],
        d["ZZ"][sl][m],
        BR[sl][m],
        np.zeros_like(BR[sl][m]),
        color=ACCENT,
        angles="xy",
        scale_units="xy",
        scale=1.6,
        width=0.0035,
        zorder=4,
    )
    ax.quiver(
        d["RR"][sl][m],
        d["ZZ"][sl][m],
        np.zeros_like(BZ[sl][m]),
        BZ[sl][m],
        color=BLUE,
        angles="xy",
        scale_units="xy",
        scale=1.6,
        width=0.0035,
        zorder=4,
    )
    # Bφ / eφ markers along midplane inside plasma
    rs = np.linspace(1.15, 2.15, 5)
    for r in rs:
        if not MPath(np.column_stack([d["lc_r"], d["lc_z"]])).contains_point((r, 0.0)):
            continue
        ax.add_patch(Circle((r, 0.0), 0.055, fill=False, ec=TEAL, lw=1.6, zorder=6))
        ax.plot(r, 0.0, "o", color=TEAL, ms=3.5, zorder=6)
    ax.plot(d["r_ax"], d["z_ax"], "o", color="#888", ms=5, zorder=5)
    ax.text(
        0.62,
        1.48,
        r"$p(\psi),\,F(\psi)$  constant on each contour",
        color=ORANGE,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9),
        zorder=8,
    )
    ax.text(
        0.62,
        -1.12,
        r"$B_R$: sideways (along $R$)",
        color=ACCENT,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9),
        zorder=8,
    )
    ax.text(
        0.62,
        -1.32,
        r"$B_Z$: up and down (along $Z$)",
        color=BLUE,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9),
        zorder=8,
    )
    ax.text(
        0.62,
        -1.52,
        r"$B_\varphi$ and $\mathbf{e}_\varphi$: out of the page",
        color=TEAL,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9),
        zorder=8,
    )
    ax.set_title(r"magnetic field from $\psi$ and $F(\psi)$")
    return save(fig, "tokamak_B.png")


def fig_gs(d) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.6), dpi=140)
    ax = axes[0]
    style_poloidal(ax, d, xlim=(0.55, 2.75), ylim=(-1.75, 1.75))
    dstar = delta_star(d)
    show = np.ma.array(dstar, mask=~d["inside"])
    vmax = np.nanpercentile(np.abs(show.compressed()), 96)
    cf = ax.contourf(
        d["RR"],
        d["ZZ"],
        show,
        levels=18,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        zorder=1,
    )
    ax.contour(d["RR"], d["ZZ"], d["psi"], levels=12, colors="#333", linewidths=0.35, alpha=0.45, zorder=2)
    faint_lcfs(ax, d)
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$\Delta^*\psi$")
    ax.set_title(r"elliptic operator  $\Delta^*\psi$")

    ax2 = axes[1]
    ax2.set_facecolor(BG)
    psi_n = np.linspace(0, 1, 200)
    # typical qualitative shapes (not a fit): p' negative, FF' can change sign
    pprime = -2.2 * (1 - psi_n) ** 1.4
    ffprime = 0.55 * (1 - psi_n) ** 0.8 - 0.15
    ax2.plot(psi_n, pprime, color=ACCENT, lw=2.0, label=r"$p'(\psi)$")
    ax2.plot(psi_n, ffprime, color=BLUE, lw=2.0, label=r"$FF'(\psi)$")
    ax2.axhline(0.0, color="#bbb", lw=0.7)
    ax2.set_xlabel(r"normalized flux  $\psi_N$")
    ax2.set_ylabel("flux-function slope  (schematic)")
    ax2.set_title(r"the two profiles on the right-hand side")
    ax2.legend(framealpha=0.92, loc="lower right")
    ax2.text(
        0.05,
        0.92,
        r"$\Delta^*\psi=-\mu_0 R^2\,p'(\psi)-FF'(\psi)$" "\n"
        r"$\mu_0=4\pi\times 10^{-7}\,\mathrm{N\,A^{-2}}$" "\n"
        r"curves: typical shapes, not this shot",
        transform=ax2.transAxes,
        fontsize=9.5,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ddd"),
    )
    return save(fig, "tokamak_gs.png")


def fig_jphi(d) -> Path:
    fig, ax = plt.subplots(figsize=(7.6, 7.0), dpi=140)
    style_poloidal(ax, d, xlim=(0.55, 2.55), ylim=(-1.75, 1.75))
    dstar = delta_star(d)
    jphi = -dstar / (MU0 * np.maximum(d["RR"], 1e-6))
    # DIII-D ψ is a minimum: Ampère gives j_φ < 0. Plot −j_φ so the core reads as current.
    show = np.ma.array(-jphi, mask=~d["inside"])
    lo, hi = np.nanpercentile(show.compressed(), [4, 96])
    cf = ax.contourf(
        d["RR"],
        d["ZZ"],
        show,
        levels=18,
        cmap="magma",
        vmin=max(lo, 0.0),
        vmax=hi,
        zorder=1,
    )
    ax.contour(d["RR"], d["ZZ"], d["psi"], levels=12, colors="white", linewidths=0.35, alpha=0.35, zorder=2)
    faint_lcfs(ax, d)
    ax.plot(d["r_ax"], d["z_ax"], "o", color="white", ms=5, zorder=5)
    ax.text(
        0.62,
        1.45,
        r"$j_\varphi$  around the doughnut" "\n" r"(this shot: into the page)",
        color=ACCENT,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9),
        zorder=8,
    )
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$-j_\varphi$  (A·m$^{-2}$), from $\Delta^*\psi=-\mu_0 R\,j_\varphi$")
    ax.set_title(r"toroidal current density  $j_\varphi$")
    return save(fig, "tokamak_jphi.png")


def main() -> None:
    d = load_frame()
    paths = [
        fig_coords(d),
        fig_psi(d),
        fig_nulls(d),
        fig_force(d),
        fig_B(d),
        fig_gs(d),
        fig_jphi(d),
    ]
    for p in paths:
        print("wrote", p)


if __name__ == "__main__":
    main()
