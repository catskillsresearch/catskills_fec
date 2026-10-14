"""One-way tickets vs Brownian jitter — pedagogical plasma-traffic figure."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

OUT = Path(__file__).resolve().parent / "tokamak_plasma_traffic.png"
BG = "#f7f7f4"
PLUS = "#2E6B8A"
MINUS = "#c0392b"
TORUS = "#c5d4e0"
GRAY = "#6d767e"

R0, A, Q = 1.70, 0.48, 3.2


def helix(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = phi / Q
    x = (R0 + A * np.cos(theta)) * np.cos(phi)
    y = (R0 + A * np.cos(theta)) * np.sin(phi)
    z = A * np.sin(theta)
    return x, y, z


def paint_helix(ax, phi: np.ndarray, color: str, lw: float = 2.2) -> None:
    x, y, z = helix(phi)
    pts = np.column_stack([x, y, z]).reshape(-1, 1, 3)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    ax.add_collection3d(Line3DCollection(segs, colors=color, linewidths=lw))


def arrow_on(ax, phi0: float, sign: int, color: str) -> None:
    ph = np.array([phi0, phi0 + sign * 0.22])
    x, y, z = helix(ph)
    ax.plot(x, y, z, color=color, lw=3.0)
    ax.scatter([x[1]], [y[1]], [z[1]], color=color, s=36, zorder=6)


def fig_brownian(ax) -> None:
    rng = np.random.default_rng(4)
    n = 400
    step = rng.normal(0, 0.045, size=(n, 2))
    xy = np.cumsum(step, axis=0)
    xy -= xy.mean(axis=0)
    ax.set_facecolor(BG)
    ax.plot(xy[:, 0], xy[:, 1], color=GRAY, lw=1.1)
    ax.plot(0, 0, "o", color=GRAY, ms=5)
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.axis("off")
    ax.set_title("not this: Brownian jitter\nin one particle", fontsize=11)


def fig_oneway(ax) -> None:
    th = np.linspace(0.15, 2 * np.pi - 0.35, 200)
    ax.set_facecolor(BG)
    ax.plot(np.cos(th), np.sin(th), color=PLUS, lw=2.4)
    ax.annotate(
        "",
        xy=(np.cos(th[-1]), np.sin(th[-1])),
        xytext=(np.cos(th[-8]), np.sin(th[-8])),
        arrowprops=dict(arrowstyle="->", color=PLUS, lw=2.0),
    )
    ax.plot(0, 0, "o", color="#148f77", ms=6)
    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.axis("off")
    ax.set_title("this: one smooth one-way ride\n(this particle keeps its sign)", fontsize=11)


def fig_torus(ax) -> None:
    u = np.linspace(0.18 * np.pi, 2 * np.pi - 0.06 * np.pi, 70)
    v = np.linspace(0, 2 * np.pi, 40)
    U, V = np.meshgrid(u, v)
    X = (R0 + A * np.cos(V)) * np.cos(U)
    Y = (R0 + A * np.cos(V)) * np.sin(U)
    Z = A * np.sin(V)
    ax.plot_surface(X, Y, Z, rstride=2, cstride=2, color=TORUS, alpha=0.28, linewidth=0, shade=True)
    paint_helix(ax, np.linspace(0.0, 2.4 * np.pi, 220), "#5a5a5a")
    arrow_on(ax, 0.55, +1, PLUS)
    arrow_on(ax, 1.85, -1, MINUS)
    ax.text2D(0.04, 0.90, "one way", transform=ax.transAxes, color=PLUS, fontsize=9)
    ax.text2D(0.04, 0.80, "the other way", transform=ax.transAxes, color=MINUS, fontsize=9)
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-2.3, 2.3)
    ax.set_zlim(-0.9, 0.9)
    ax.set_axis_off()
    ax.set_box_aspect((1.7, 1.7, 0.75))
    ax.view_init(elev=20, azim=-58)
    ax.set_title("two particles, same track,\nopposite tickets", fontsize=11)


def fig_tickets(ax) -> None:
    v = np.linspace(-3.2, 3.2, 400)
    f = np.exp(-0.5 * v**2)
    ax.set_facecolor(BG)
    ax.fill_between(v[v >= 0], 0, f[v >= 0], color=PLUS, alpha=0.55)
    ax.fill_between(v[v <= 0], 0, f[v <= 0], color=MINUS, alpha=0.55)
    ax.plot(v, f, color="#333", lw=1.2)
    ax.axvline(0.0, color="#333", lw=0.7, ls="--")
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([])
    ax.set_xlabel(r"speed along the track  (sign $=$ which way)")
    ax.set_title("the random draw: who got which ticket\n(balanced → soup as a whole is not spinning)", fontsize=11)


def main() -> None:
    fig = plt.figure(figsize=(11.0, 8.0), dpi=140)
    fig_brownian(fig.add_subplot(2, 2, 1))
    fig_oneway(fig.add_subplot(2, 2, 2))
    fig_torus(fig.add_subplot(2, 2, 3, projection="3d"))
    fig_tickets(fig.add_subplot(2, 2, 4))
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
