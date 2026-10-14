"""Low-Ip vs high-Ip helical field-line pitch — cartoon, not a real machine."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

OUT = Path(__file__).resolve().parent / "tokamak_helix.png"
BG = "#f7f7f4"
LINE = "#6c3483"
TORUS = "#c5d4e0"

R0, A = 1.70, 0.48
N_LAPS = 3.0
Q_LOW = 5.0   # low Ip — gentle twist
Q_HIGH = 1.8  # high Ip — tight twist


def field_line(q: float, n: int = 900) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phi = np.linspace(0.0, 2 * np.pi * N_LAPS, n)
    theta = phi / q
    x = (R0 + A * np.cos(theta)) * np.cos(phi)
    y = (R0 + A * np.cos(theta)) * np.sin(phi)
    z = A * np.sin(theta)
    return x, y, z, phi, theta


def faint_torus(ax) -> None:
    u = np.linspace(0.15 * np.pi, 2 * np.pi - 0.05 * np.pi, 70)
    v = np.linspace(0, 2 * np.pi, 40)
    U, V = np.meshgrid(u, v)
    X = (R0 + A * np.cos(V)) * np.cos(U)
    Y = (R0 + A * np.cos(V)) * np.sin(U)
    Z = A * np.sin(V)
    ax.plot_surface(X, Y, Z, rstride=2, cstride=2, color=TORUS, alpha=0.28, linewidth=0, shade=True)


def style_3d(ax) -> None:
    faint_torus(ax)
    ax.plot([0, 0], [0, 0], [-0.85, 0.85], color="#333", lw=0.8, ls="--")
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-2.3, 2.3)
    ax.set_zlim(-0.9, 0.9)
    ax.set_axis_off()
    ax.set_box_aspect((1.7, 1.7, 0.75))
    ax.view_init(elev=20, azim=-58)


def draw_helix(ax, q: float) -> None:
    x, y, z, _, _ = field_line(q)
    # Colour the line along its length so you can follow one lap.
    pts = np.column_stack([x, y, z]).reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    t = np.linspace(0, 1, len(segs))
    colors = plt.cm.plasma(0.15 + 0.7 * t)
    ax.add_collection3d(Line3DCollection(segs, colors=colors, linewidths=2.2))
    ax.scatter([x[0]], [y[0]], [z[0]], color=LINE, s=22, zorder=5)
    ax.scatter([x[-1]], [y[-1]], [z[-1]], color=LINE, s=22, zorder=5)


def draw_unwrap(ax, q: float, title: str) -> None:
    _, _, _, phi, theta = field_line(q)
    ax.set_facecolor(BG)
    ax.plot(phi / (2 * np.pi), theta / (2 * np.pi), color=LINE, lw=2.2)
    ax.set_xlim(0, N_LAPS)
    ax.set_ylim(0, N_LAPS / Q_HIGH * 1.05)
    ax.set_xlabel("laps the long way")
    ax.set_ylabel("laps the short way")
    ax.set_title(title, fontsize=11, color=LINE)
    ax.grid(True, color="#d8d8d4", lw=0.6)


def main() -> None:
    fig = plt.figure(figsize=(10.6, 7.2), dpi=140)
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    style_3d(ax1)
    draw_helix(ax1, Q_LOW)
    ax1.set_title(r"low $I_p$  —  gentle twist", fontsize=12, color=LINE)

    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    style_3d(ax2)
    draw_helix(ax2, Q_HIGH)
    ax2.set_title(r"high $I_p$  —  tight twist", fontsize=12, color=LINE)

    ax3 = fig.add_subplot(2, 2, 3)
    draw_unwrap(ax3, Q_LOW, r"same line, unrolled  —  low $I_p$")

    ax4 = fig.add_subplot(2, 2, 4)
    draw_unwrap(ax4, Q_HIGH, r"same line, unrolled  —  high $I_p$")

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
