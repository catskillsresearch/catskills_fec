"""Generic CS / PF / TF placement — not DIII-D or MAST."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OUT = Path(__file__).resolve().parent / "tokamak_coils.png"
BG = "#f7f7f4"
PF = "#2E6B8A"       # central solenoid + poloidal-field coils
TF = "#6c3483"       # toroidal-field coils
ORANGE = "#e07a2f"

# Cartoon geometry (metres, not a real machine).
R0, A, KAPPA, DELTA = 1.68, 0.62, 1.55, 0.48
SOL_R0, SOL_R1 = 0.20, 0.40
SOL_Z = 1.28
TF_PAD = 0.22


def plasma_rz(n: int = 240) -> tuple[np.ndarray, np.ndarray]:
    th = np.linspace(0, 2 * np.pi, n)
    r = R0 + A * np.cos(th + np.arcsin(DELTA) * np.sin(th))
    z = KAPPA * A * np.sin(th) - 0.10 * A * (1 - np.cos(th))
    return r, z


def tf_rz(n: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """One D-shaped TF coil, just outside the plasma."""
    r, z = plasma_rz(n)
    return R0 + (1.0 + TF_PAD / A) * (r - R0), (1.0 + TF_PAD / (KAPPA * A)) * z


def torus_xyz(u, v):
    x = (R0 + A * np.cos(v)) * np.cos(u)
    y = (R0 + A * np.cos(v)) * np.sin(u)
    z = A * np.sin(v)
    return x, y, z


def add_ring(ax, radius: float, z: float, color: str, lw: float = 2.4) -> None:
    ph = np.linspace(0, 2 * np.pi, 120)
    ax.plot(radius * np.cos(ph), radius * np.sin(ph), np.full_like(ph, z), color=color, lw=lw)


def add_tf_coil(ax, phi: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, z = tf_rz()
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    ax.plot(x, y, z, color=TF, lw=2.4)
    return x, y, z


def fig_3d(ax) -> None:
    u = np.linspace(0.18 * np.pi, 2 * np.pi - 0.06 * np.pi, 80)
    v = np.linspace(0, 2 * np.pi, 48)
    U, V = np.meshgrid(u, v)
    X, Y, Zt = torus_xyz(U, V)
    ax.plot_surface(X, Y, Zt, rstride=2, cstride=2, color="#c5d4e0", alpha=0.32, linewidth=0, shade=True)

    pr, pz = plasma_rz(80)
    face = np.column_stack([pr, np.zeros_like(pr), pz])
    ax.add_collection3d(
        Poly3DCollection([face], facecolor="#f4e4c8", edgecolor=ORANGE, linewidths=1.1, alpha=0.95)
    )

    ph = np.linspace(0, 2 * np.pi, 40)
    zs = np.linspace(-SOL_Z, SOL_Z, 2)
    PH, ZZ = np.meshgrid(ph, zs)
    rs = 0.5 * (SOL_R0 + SOL_R1)
    ax.plot_surface(
        rs * np.cos(PH), rs * np.sin(PH), ZZ,
        rstride=1, cstride=1, color=PF, alpha=0.92, linewidth=0, shade=True,
    )
    for z in (-SOL_Z, SOL_Z):
        add_ring(ax, rs, z, PF, lw=1.2)

    for rad, z in (
        (0.85, 0.95),
        (0.85, -0.95),
        (1.70, 1.22),
        (1.70, -1.22),
        (2.45, 0.45),
        (2.45, -0.45),
    ):
        add_ring(ax, rad, z, PF, lw=2.6)

    # TF coils wrap the short way. None sit in the cut plane (φ = 0).
    for phi in (0.55 * np.pi, 0.95 * np.pi, 1.45 * np.pi, 1.75 * np.pi):
        add_tf_coil(ax, phi)

    # 3D world-text clips through the mesh; pin labels in the panel next to the coils.
    ax.text2D(0.50, 0.90, "central solenoid", transform=ax.transAxes, color=PF, fontsize=9, ha="center")
    ax.text2D(0.98, 0.72, "poloidal-field\ncoils", transform=ax.transAxes, color=PF, fontsize=9, ha="right", va="center")
    ax.text2D(0.52, 0.04, "toroidal-field\ncoils", transform=ax.transAxes, color=TF, fontsize=9, ha="center", va="bottom")
    ax.plot([0, 0], [0, 0], [-1.35, 1.35], color="#333", lw=1.1, ls="--")
    ax.set_xlim(-2.4, 2.6)
    ax.set_ylim(-2.4, 2.6)
    ax.set_zlim(-1.35, 1.35)
    ax.set_axis_off()
    ax.set_box_aspect((1.7, 1.7, 1.05))
    ax.view_init(elev=18, azim=-56)
    ax.set_title("coils on the doughnut", fontsize=11, pad=2)


def box(ax, r0, z0, w, h, color, **kw) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (r0, z0), w, h,
            boxstyle="square,pad=0",
            facecolor=color, edgecolor="#222", linewidth=0.5,
            zorder=4, **kw,
        )
    )


def fig_poloidal(ax) -> None:
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.set_xlim(-0.20, 3.45)
    ax.set_ylim(-1.95, 1.95)
    ax.set_xlabel(r"major radius  $R$  (m)")
    ax.set_ylabel(r"vertical height  $Z$  (m)")
    ax.axhline(0.0, color="#d0d0cc", lw=0.6, zorder=0)
    ax.plot([0, 0], [-1.75, 1.75], color="#333", lw=1.1, ls="--", zorder=1)
    ax.text(0.06, 1.62, "axis", color="#333", fontsize=8)

    r, z = plasma_rz()
    ax.fill(r, z, color="#f4e4c8", zorder=2)
    ax.plot(r, z, color=ORANGE, lw=1.8, zorder=3)
    ax.plot(R0, 0.0, "o", color="#148f77", ms=6, zorder=5)

    box(ax, SOL_R0, -SOL_Z, SOL_R1 - SOL_R0, 2 * SOL_Z, PF)
    ax.annotate(
        "central solenoid",
        xy=(0.32, 0.15),
        xytext=(-0.05, -1.55),
        color=PF, fontsize=9, ha="left",
        arrowprops=dict(arrowstyle="->", color=PF, lw=1.2),
        zorder=8,
    )

    for rc, zc in ((0.72, 0.95), (0.72, 0.48), (0.72, -0.48), (0.72, -0.95)):
        box(ax, rc - 0.07, zc - 0.09, 0.14, 0.18, PF)
    for rc, zc in (
        (1.35, 1.38),
        (1.95, 1.38),
        (1.35, -1.38),
        (1.95, -1.38),
        (2.58, 0.48),
        (2.58, -0.48),
    ):
        box(ax, rc - 0.10, zc - 0.10, 0.20, 0.20, PF)

    ax.annotate(
        "poloidal-field coils",
        xy=(1.35, 1.38),
        xytext=(1.85, 1.68),
        color=PF, fontsize=9,
        arrowprops=dict(arrowstyle="->", color=PF, lw=1.2),
        zorder=8,
    )

    # One TF coil in this cut: inboard and outboard limbs (not a ring about the axis).
    box(ax, 0.88, -0.16, 0.14, 0.32, TF)
    box(ax, 2.28, -0.16, 0.16, 0.32, TF)
    ax.annotate(
        "toroidal-field coils",
        xy=(2.36, -0.16),
        xytext=(2.00, -1.72),
        color=TF, fontsize=9,
        arrowprops=dict(arrowstyle="->", color=TF, lw=1.2),
        zorder=8,
    )
    ax.text(1.48, 0.06, "plasma", color=ORANGE, fontsize=9, zorder=8)
    ax.set_title(r"poloidal cut  —  typical placement, not a real machine", fontsize=11)


def main() -> None:
    fig = plt.figure(figsize=(11.4, 5.7), dpi=140)
    fig_3d(fig.add_subplot(1, 2, 1, projection="3d"))
    fig_poloidal(fig.add_subplot(1, 2, 2))
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
