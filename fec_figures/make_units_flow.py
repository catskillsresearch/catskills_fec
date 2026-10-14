"""Draw fec_figures/units_flow.png for fec.md.

The graph (green = irreducible) is also written as Mermaid in units_flow.mmd.
Do not paste that fence into the markdown page: one preview draws it and the
other dumps the source, so the page would show both.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "units_flow.png"
BG = "#f7f7f4"
GREEN = "#d4edda"
GREEN_EDGE = "#3d8b40"
YELLOW = "#fff4cc"
YELLOW_EDGE = "#c4a035"
ARROW = "#4a4a4a"

# id -> (x, y, headline, unit line, [writings], irreducible)
NODES = {
    "m":  (0.13, 0.94, "length", "metre  m", [], True),
    "kg": (0.38, 0.94, "mass", "kilogram  kg", ["gram  g", "1 kg = 1000 g"], True),
    "s":  (0.63, 0.94, "time", "second  s", [], True),
    "e":  (0.87, 0.94, "charge", "electron  e", [], True),
    "N":  (0.38, 0.79, "how hard the shove", "newton  N", ["N = kg·m/s²"], False),
    "J":  (0.26, 0.63, "how much work", "joule  J", ["J = N·m", "J = kg·m²/s²"], False),
    "C":  (0.87, 0.63, "a pile of charge", "coulomb  C", ["C ≈ 6.2×10¹⁸ e"], False),
    "W":  (0.26, 0.49, "work per second", "watt  W", ["W = J/s"], False),
    "A":  (0.63, 0.49, "charge per second", "ampere  A", ["A = C/s"], False),
    "V":  (0.50, 0.31, "push per coulomb", "volt  V",
          ["V = J/C", "V = N·m/C", "V = W/A", "V = kg·m²/(A·s³)"], False),
    "T":  (0.26, 0.10, "how strong the track is", "tesla  T",
          ["T = (V·s)/m²", "T = N/(A·m)", "T = kg/(A·s²)"], False),
    "Wb": (0.74, 0.10, "field through an area", "weber  Wb",
          ["Wb = V·s", "Wb = T·m²"], False),
}

# Transitive reduction: if A→B→C is drawn, do not also draw A→C.
EDGES = [
    ("kg", "N"), ("m", "N"), ("s", "N"),
    ("N", "J"),
    ("e", "C"),
    ("C", "A"), ("s", "A"),
    ("J", "W"),
    ("W", "V"), ("A", "V"),
    ("V", "T"),
    ("T", "Wb"),
]


def lines_of(spec: tuple) -> list[str]:
    _, _, headline, unit, writings, _ = spec
    return [headline, unit, *writings]


def box_size(spec: tuple) -> tuple[float, float]:
    n = len(lines_of(spec))
    w = 0.20 if n <= 3 else 0.24
    h = 0.024 * n + 0.010
    return w, h


def edge_point(x: float, y: float, w: float, h: float, x2: float, y2: float, gap: float) -> tuple[float, float]:
    dx, dy = x2 - x, y2 - y
    if dx == 0 and dy == 0:
        return x, y
    tx = (w / 2) / abs(dx) if dx else float("inf")
    ty = (h / 2) / abs(dy) if dy else float("inf")
    t = min(tx, ty)
    ux, uy = dx * t, dy * t
    length = (ux ** 2 + uy ** 2) ** 0.5
    if length:
        ux += gap * ux / length
        uy += gap * uy / length
    return x + ux, y + uy


def draw_box(ax, spec: tuple) -> None:
    x, y, _, _, _, raw = spec
    w, h = box_size(spec)
    fill = GREEN if raw else YELLOW
    edge = GREEN_EDGE if raw else YELLOW_EDGE
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.003,rounding_size=0.008",
            facecolor=fill, edgecolor=edge, linewidth=1.0, mutation_aspect=0.6,
        )
    )
    lines = lines_of(spec)
    step = 0.022
    y0 = y + (len(lines) - 1) * step / 2
    for i, line in enumerate(lines):
        ax.text(
            x, y0 - i * step, line,
            ha="center", va="center",
            fontsize=9 if i == 0 else 7.5,
            fontweight="bold" if i == 0 else "normal",
            color="#222222",
        )


def main() -> None:
    fig, ax = plt.subplots(figsize=(9.6, 10.4), facecolor=BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for spec in NODES.values():
        draw_box(ax, spec)
    for a, b in EDGES:
        sa, sb = NODES[a], NODES[b]
        xa, ya = sa[0], sa[1]
        xb, yb = sb[0], sb[1]
        wa, ha = box_size(sa)
        wb, hb = box_size(sb)
        if (a, b) == ("s", "A"):
            # Straight drop onto the top of ampere; coulomb sits to the right.
            x0, y0 = xa, ya - ha / 2 - 0.004
            x1, y1 = xb, yb + hb / 2 + 0.008
            style = "arc3,rad=0"
        else:
            x0, y0 = edge_point(xa, ya, wa, ha, xb, yb, gap=0.006)
            x1, y1 = edge_point(xb, yb, wb, hb, xa, ya, gap=0.010)
            style = "arc3,rad=0"
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0), (x1, y1),
                arrowstyle="-|>",
                mutation_scale=14,
                color=ARROW,
                lw=1.05,
                shrinkA=0,
                shrinkB=0,
                connectionstyle=style,
                transform=ax.transData,
            )
        )
    fig.savefig(OUT, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
