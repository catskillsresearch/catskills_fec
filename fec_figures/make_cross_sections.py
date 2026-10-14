"""Annotated (R,Z) cross-sections for fec.md from official parquet geometry."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent


def load_row(path: Path) -> dict:
    table = pq.read_table(path)
    return {c: table.column(c)[0].as_py() for c in table.column_names}


def mid_frame(row: dict) -> int:
    psi = np.asarray(row["efit_psirz"])
    return int(psi.shape[0] // 2)


def lcfs_xy(row: dict, t: int) -> tuple[np.ndarray, np.ndarray] | None:
    n = int(np.asarray(row["efit_lcfs_n"])[t])
    if n < 8:
        return None
    r = np.asarray(row["efit_lcfs_r"])[t, :n]
    z = np.asarray(row["efit_lcfs_z"])[t, :n]
    return r, z


def draw_coils(ax, row: dict, *, mast: bool) -> None:
    names = np.asarray(row["coil_input_column"]).astype(str)
    R = np.asarray(row["coil_R"], float)
    Z = np.asarray(row["coil_Z"], float)
    W = np.asarray(row["coil_width"], float)
    H = np.asarray(row["coil_height"], float)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, col in enumerate(names):
        groups[col].append(i)
    palette = {
        "magnetics_ECOILA": "#8B4513",
        "magnetics_sol_current": "#8B4513",
    }
    for col, idxs in groups.items():
        short = col.replace("magnetics_", "").replace("_current", "")
        color = palette.get(col, "#2E6B8A")
        if mast:
            color = "#8B4513" if "sol" in col else "#2E6B8A"
            for i in idxs:
                ax.add_patch(
                    Rectangle(
                        (R[i] - W[i] / 2, Z[i] - H[i] / 2),
                        max(W[i], 0.012),
                        max(H[i], 0.012),
                        facecolor=color,
                        edgecolor="none",
                        alpha=0.85,
                        zorder=3,
                    )
                )
        else:
            i = idxs[0]
            ax.add_patch(
                Rectangle(
                    (R[i] - W[i] / 2, Z[i] - H[i] / 2),
                    W[i],
                    H[i],
                    facecolor=color,
                    edgecolor="k",
                    linewidth=0.4,
                    alpha=0.9,
                    zorder=3,
                )
            )
            ax.text(
                R[i],
                Z[i],
                short.replace("ECOILA", "E"),
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if short == "ECOILA" else "#102030",
                zorder=4,
            )


def annotate(ax, xy, text, xytext, color="#222", ha=None):
    if ha is None:
        ha = "left" if xytext[0] > xy[0] else "right"
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        textcoords="data",
        fontsize=8,
        color=color,
        ha=ha,
        va="center",
        arrowprops=dict(arrowstyle="->", color=color, lw=0.85),
        zorder=6,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.82),
    )


def plot_d3d(row: dict, out: Path) -> None:
    t = mid_frame(row)
    psi = np.asarray(row["efit_psirz"])[t]
    Rg = np.asarray(row["efit_grid_R"], float)
    Zg = np.asarray(row["efit_grid_Z"], float)
    RR, ZZ = np.meshgrid(Rg, Zg)
    ch_name = np.asarray(row["thomson_chord_name"]).astype(str)
    ch_r = np.asarray(row["thomson_chord_R"], float)
    ch_z = np.asarray(row["thomson_chord_Z"], float)

    fig, ax = plt.subplots(figsize=(9.2, 8.2), dpi=140)
    ax.set_facecolor("#f7f7f4")
    cs = ax.contour(RR, ZZ, psi, levels=18, colors="#9aa3ad", linewidths=0.45, zorder=1)
    ax.clabel(cs, fmt="", inline=False)
    lc = lcfs_xy(row, t)
    if lc:
        ax.plot(*lc, color="#e07a2f", lw=2.0, zorder=2, label="LCFS (diverted)")
        ax.fill(*lc, color="#e07a2f", alpha=0.08, zorder=1)
    draw_coils(ax, row, mast=False)

    families = {
        "TS_core": ("#c0392b", "o"),
        "TS_tangential": ("#6c3483", "s"),
        "TS_divertor": ("#148f77", "D"),
    }
    for prefix, (color, marker) in families.items():
        m = np.array([n.startswith(prefix) for n in ch_name])
        if m.any():
            ax.scatter(
                ch_r[m],
                ch_z[m],
                s=18,
                c=color,
                marker=marker,
                zorder=5,
                label=prefix.replace("TS_", "Thomson "),
            )

    annotate(ax, (0.723, 0.00), "ECOILA solenoid", (0.18, 0.28), "#8B4513")
    annotate(ax, (0.861, 1.00), "inner PF F1–F4 A/B", (0.12, 1.35))
    annotate(ax, (1.69, 1.59), "upper PF F5A F8A F9A", (2.85, 1.88))
    annotate(ax, (2.61, 0.44), "outer PF F6A F7A", (3.02, 0.85))
    annotate(ax, (2.61, -0.44), "outer PF F6B F7B", (3.02, -0.85))
    annotate(ax, (1.69, -1.58), "lower PF F5B F8B F9B", (2.85, -1.92))
    annotate(ax, (1.70, 0.00), "plasma current Ip", (3.02, 0.22), "#e07a2f")
    core = ch_name.astype(str)
    mcore = np.array([n.startswith("TS_core") for n in core])
    if mcore.any():
        annotate(
            ax,
            (float(np.median(ch_r[mcore])), 0.55),
            "Thomson core Te, ne",
            (3.02, 1.15),
            "#c0392b",
        )
    mtan = np.array([n.startswith("TS_tangential") for n in core])
    if mtan.any():
        annotate(
            ax,
            (float(np.median(ch_r[mtan])), float(np.median(ch_z[mtan]))),
            "Thomson tangential Te, ne",
            (3.02, -0.18),
            "#6c3483",
        )
    mdiv = np.array([n.startswith("TS_divertor") for n in core])
    if mdiv.any():
        annotate(
            ax,
            (float(np.min(ch_r[mdiv])), float(np.min(ch_z[mdiv]))),
            "Thomson divertor Te, ne",
            (0.18, -1.85),
            "#148f77",
        )
    ax.text(
        0.12,
        -2.08,
        "bcoil: toroidal field (out of plane; no coil rectangle)\n"
        "grid: 65×65   R ∈ [0.84, 2.54] m    Z ∈ [−1.60, 1.60] m",
        fontsize=7.5,
        color="#333",
    )
    ax.set_aspect("equal")
    ax.set_xlim(0.05, 3.25)
    ax.set_ylim(-2.15, 2.15)
    ax.set_xlabel("major radius  R  (m)")
    ax.set_ylabel("vertical height  Z  (m)")
    ax.set_title("DIII-D poloidal cross-section  —  shot 203702, mid-flattop")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_mast(row: dict, out: Path) -> None:
    t = mid_frame(row)
    psi = np.asarray(row["efit_psirz"])[t]
    Rg = np.asarray(row["efit_grid_R"], float)
    Zg = np.asarray(row["efit_grid_Z"], float)
    RR, ZZ = np.meshgrid(Rg, Zg)
    ch_name = np.asarray(row["thomson_chord_name"]).astype(str)
    ch_r = np.asarray(row["thomson_chord_R"], float)
    ch_z = np.asarray(row["thomson_chord_Z"], float)

    fig, ax = plt.subplots(figsize=(8.6, 8.6), dpi=140)
    ax.set_facecolor("#f7f7f4")
    ax.contour(RR, ZZ, psi, levels=18, colors="#9aa3ad", linewidths=0.45, zorder=1)
    lc = lcfs_xy(row, t)
    if lc:
        ax.plot(*lc, color="#e07a2f", lw=2.0, zorder=2, label="LCFS (diverted)")
        ax.fill(*lc, color="#e07a2f", alpha=0.08, zorder=1)
    draw_coils(ax, row, mast=True)

    mcore = np.array(["core" in n.lower() and "edge" not in n.lower() for n in ch_name])
    medge = np.array(["edge" in n.lower() for n in ch_name])
    if mcore.any():
        ax.scatter(ch_r[mcore], ch_z[mcore], s=14, c="#c0392b", marker="o", zorder=5, label="Thomson core")
    if medge.any():
        ax.scatter(ch_r[medge], ch_z[medge], s=22, c="#6c3483", marker="s", zorder=5, label="Thomson edge")

    annotate(ax, (0.14, 0.00), "solenoid (P1)", (-0.55, 0.40), "#8B4513")
    annotate(ax, (0.50, 1.75), "P2U", (2.15, 2.12))
    annotate(ax, (0.50, -1.75), "P2L", (2.15, -2.12))
    annotate(ax, (1.10, 1.10), "P3U", (2.15, 1.38))
    annotate(ax, (1.50, 1.10), "P4U", (2.15, 1.10))
    annotate(ax, (1.44, 0.90), "P6U", (2.15, 0.82))
    annotate(ax, (1.65, 0.50), "P5U", (2.15, 0.50))
    annotate(ax, (1.10, -1.10), "P3L", (2.15, -1.38))
    annotate(ax, (1.50, -1.10), "P4L", (2.15, -1.10))
    annotate(ax, (1.44, -0.90), "P6L", (2.15, -0.82))
    annotate(ax, (1.65, -0.50), "P5L", (2.15, -0.50))
    annotate(ax, (0.90, 0.00), "plasma current Ip", (2.15, 0.18), "#e07a2f")
    if mcore.any():
        annotate(
            ax,
            (float(np.median(ch_r[mcore])), 0.0),
            "Thomson core Te, ne  (Z = 0)",
            (2.15, -0.12),
            "#c0392b",
        )
    if medge.any():
        annotate(
            ax,
            (float(np.max(ch_r[medge])), 0.0),
            "Thomson edge Te, ne",
            (2.15, -0.38),
            "#6c3483",
        )
    ax.text(
        -0.55,
        -2.45,
        "tf_current: toroidal field (out of plane)\n"
        "efps_current: error-field correction (no rectangle)\n"
        "grid: 65×65   R ∈ [0.06, 2.00] m    Z ∈ [−2.00, 2.00] m",
        fontsize=7.5,
        color="#333",
    )
    ax.set_aspect("equal")
    ax.set_xlim(-0.70, 2.45)
    ax.set_ylim(-2.55, 2.45)
    ax.set_xlabel("major radius  R  (m)")
    ax.set_ylabel("vertical height  Z  (m)")
    ax.set_title("MAST poloidal cross-section  —  shot 28348, mid-flattop")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    d3d = load_row(ROOT / "parquet_data" / "d3d_shot_203702.parquet")
    mast = load_row(ROOT / "parquet_data" / "mast_shot_28348.parquet")
    plot_d3d(d3d, OUT / "diii_d_poloidal.png")
    plot_mast(mast, OUT / "mast_poloidal.png")
    print("wrote", OUT / "diii_d_poloidal.png")
    print("wrote", OUT / "mast_poloidal.png")


if __name__ == "__main__":
    main()
