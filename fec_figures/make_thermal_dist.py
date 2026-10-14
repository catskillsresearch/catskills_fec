"""1 keV speed distributions along the track: electrons vs deuterium ions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "tokamak_thermal_dist.png"
BG = "#f7f7f4"
PLUS = "#2E6B8A"
MINUS = "#c0392b"
MARK = "#6c3483"

E = 1.602176634e-19
ME = 9.1093837e-31
MD = 2.014 * 1.66053906660e-27
T = 1000 * E  # 1 keV in joules


def maxwell_1d(v: np.ndarray, mass: float) -> np.ndarray:
    return np.sqrt(mass / (2 * np.pi * T)) * np.exp(-mass * v**2 / (2 * T))


def panel(ax, mass: float, xmax: float, title: str, v_unit: str) -> None:
    v = np.linspace(-xmax, xmax, 800)
    f = maxwell_1d(v, mass)
    v_t = np.sqrt(2 * T / mass)
    v_abs = np.sqrt(2 * T / (np.pi * mass))
    ax.set_facecolor(BG)
    ax.fill_between(v[v >= 0], 0, f[v >= 0], color=PLUS, alpha=0.50)
    ax.fill_between(v[v <= 0], 0, f[v <= 0], color=MINUS, alpha=0.50)
    ax.plot(v, f, color="#333", lw=1.3)
    ax.axvline(0.0, color="#333", lw=0.7, ls="--")
    ymax = float(f.max())
    ax.plot([v_t, v_t], [0, ymax * 0.72], color=MARK, lw=1.4)
    ax.plot([-v_t, -v_t], [0, ymax * 0.72], color=MARK, lw=1.4)
    ax.plot([v_abs, v_abs], [0, ymax * 0.42], color="#148f77", lw=1.3, ls=":")
    ax.plot([-v_abs, -v_abs], [0, ymax * 0.42], color="#148f77", lw=1.3, ls=":")
    ax.text(0.02, 0.92, r"$\langle v\rangle=0$  (no net wind)", transform=ax.transAxes, fontsize=9)
    ax.text(v_t, ymax * 0.76, r"$v=\sqrt{2T/m}$", color=MARK, fontsize=8, ha="center")
    ax.text(v_abs, ymax * 0.46, r"$\langle|v|\rangle$", color="#148f77", fontsize=8, ha="left")
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(0, ymax * 1.18)
    ax.set_yticks([])
    ax.set_xlabel(f"speed along the track  ({v_unit})")
    ax.set_title(title, fontsize=12)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=140)
    panel(
        axes[0], ME, 5.5e7,
        r"electrons at $1\,\mathrm{keV}$  —  wide distribution",
        r"$10^7\,\mathrm{m/s}$",
    )
    axes[0].xaxis.set_major_formatter(lambda x, _pos: f"{x/1e7:.0f}")
    panel(
        axes[1], MD, 9.0e5,
        r"$\mathrm{D}^{+}$ ions at $1\,\mathrm{keV}$  —  same $T$, narrow distribution",
        r"$10^5\,\mathrm{m/s}$",
    )
    axes[1].xaxis.set_major_formatter(lambda x, _pos: f"{x/1e5:.0f}")
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
