"""Curl as extras through a stamp, then the map on the (R,Z) slice."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from scipy.interpolate import RegularGridInterpolator

from make_tokamak_symbols import (
    ACCENT,
    BG,
    BLUE,
    GRAY,
    ORANGE,
    TEAL,
    delta_star,
    faint_lcfs,
    load_frame,
    save,
    style_poloidal,
)

YELLOW = "#e6b800"


def _psi_n(d):
    interp = RegularGridInterpolator((d["Z"], d["R"]), d["psi"], bounds_error=False)
    psi_ax = float(interp([[d["z_ax"], d["r_ax"]]])[0])
    pts = np.column_stack([d["lc_z"], d["lc_r"]])
    psi_b = float(np.nanmedian(interp(pts)))
    return np.where(d["inside"], (d["psi"] - psi_ax) / (psi_b - psi_ax), np.nan)


def _curl_phi(d):
    # (∇×B)_φ = −Δ*ψ / R   (same combination as ∂BR/∂Z − ∂BZ/∂R)
    return -delta_star(d) / np.maximum(d["RR"], 1e-6)


def _br_bz(d):
    dpsidR = np.gradient(d["psi"], d["R"], axis=1)
    dpsidZ = np.gradient(d["psi"], d["Z"], axis=0)
    BR = -dpsidZ / np.maximum(d["RR"], 1e-6)
    BZ = dpsidR / np.maximum(d["RR"], 1e-6)
    return BR, BZ


def _paddle(ax, x, y, *, spinning, scale=1.0):
    r = 0.22 * scale
    ax.add_patch(Circle((x, y), r, fill=False, ec=YELLOW, lw=1.8, zorder=6))
    ax.add_patch(Circle((x, y), 0.035 * scale, fc=YELLOW, ec="#333", lw=0.6, zorder=7))
    for ang in (20, 110, 200, 290):
        th = np.deg2rad(ang)
        ax.plot(
            [x + 0.04 * scale * np.cos(th), x + r * 0.92 * np.cos(th)],
            [y + 0.04 * scale * np.sin(th), y + r * 0.92 * np.sin(th)],
            color=YELLOW, lw=2.0, zorder=6,
        )
    if spinning:
        ax.annotate(
            "",
            xy=(x + r * 0.55, y + r * 1.15),
            xytext=(x - r * 0.15, y + r * 1.05),
            arrowprops=dict(arrowstyle="-|>", color=YELLOW, lw=1.4, mutation_scale=10),
        )
    return r


def fig_paddle() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), dpi=140)

    ax = axes[0]
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.2, 1.25)
    ax.set_axis_off()
    ax.set_title("tight swirl  —  paddlewheel spins", fontsize=11, color=ACCENT)
    th = np.linspace(0, 2 * np.pi, 13)[:-1]
    for t in th:
        ax.annotate(
            "",
            xy=(0.72 * np.cos(t + 0.35), 0.72 * np.sin(t + 0.35)),
            xytext=(0.72 * np.cos(t), 0.72 * np.sin(t)),
            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.5, mutation_scale=11),
        )
    _paddle(ax, 0.0, 0.0, spinning=True, scale=1.15)
    ax.add_patch(Circle((0.0, 0.0), 0.08, fill=False, ec=TEAL, lw=1.8, zorder=8))
    ax.plot(0.0, 0.0, "o", color=TEAL, ms=5, zorder=8)
    ax.text(0.0, -1.08, r"axle out of the page  $(\nabla\times\mathbf{B})_\varphi$", ha="center", color=TEAL, fontsize=9)
    ax.text(0.0, 1.08, r"$\mathbf{B}$ circles the point", ha="center", color=BLUE, fontsize=9)

    ax = axes[1]
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.2, 1.25)
    ax.set_axis_off()
    ax.set_title("no swirl  —  paddlewheel still", fontsize=11, color=GRAY)
    for y in np.linspace(-0.75, 0.75, 6):
        ax.annotate(
            "",
            xy=(0.85, y),
            xytext=(-0.85, y),
            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4, mutation_scale=10),
        )
    _paddle(ax, 0.0, 0.0, spinning=False, scale=1.15)
    ax.text(0.0, -1.08, r"curl is zero  —  arrows never turn", ha="center", color=GRAY, fontsize=9)
    ax.text(0.0, 1.08, r"$\mathbf{B}$ is straight, like spaghetti", ha="center", color=BLUE, fontsize=9)
    print("wrote", save(fig, "tokamak_curl_paddle.png"))


def fig_levels() -> None:
    d = load_frame()
    BR, BZ = _br_bz(d)
    curl = _curl_phi(d)
    show = np.ma.array(np.abs(curl), mask=~d["inside"])
    psi_n = _psi_n(d)
    # three levels on the outboard midplane
    targets = (0.15, 0.50, 0.88)
    points = []
    row = int(np.argmin(np.abs(d["Z"] - d["z_ax"])))
    rs = d["R"]
    tn = psi_n[row]
    for lev in targets:
        good = np.isfinite(tn) & (rs > d["r_ax"])
        j = int(np.nanargmin(np.abs(tn[good] - lev)))
        r = float(rs[good][j])
        points.append((r, float(d["z_ax"])))

    fig, (ax, hx) = plt.subplots(1, 2, figsize=(11.0, 4.8), dpi=140)

    style_poloidal(ax, d, xlim=(0.7, 2.45), ylim=(-1.55, 1.55))
    ax.contour(d["RR"], d["ZZ"], d["psi"], levels=12, colors=GRAY, linewidths=0.45, zorder=1)
    faint_lcfs(ax, d)
    sl = (slice(2, None, 5), slice(2, None, 5))
    m = d["inside"][sl]
    ax.quiver(
        d["RR"][sl][m], d["ZZ"][sl][m], BR[sl][m], BZ[sl][m],
        color=BLUE, angles="xy", scale_units="xy", scale=1.35, width=0.004, zorder=3,
    )
    ax.plot(d["r_ax"], d["z_ax"], "o", color=ACCENT, ms=6, zorder=6)
    ax.text(d["r_ax"] - 0.38, d["z_ax"] + 0.10, "axis", color=ACCENT, fontsize=8)
    for r, z in points:
        ax.add_patch(Circle((r, z), 0.11, fill=False, ec=YELLOW, lw=1.8, zorder=7))
        ax.add_patch(Circle((r, z), 0.035, fill=False, ec=TEAL, lw=1.4, zorder=8))
        ax.plot(r, z, "o", color=TEAL, ms=3.5, zorder=8)
    ax.set_title(r"what can turn: the $\mathbf{B}$ arrows (not particles)", fontsize=11)
    ax.text(
        0.78, -1.42,
        r"blue = $\mathbf{B}$    yellow = empty measuring loop    teal $\odot$ = axle",
        fontsize=8, color=GRAY,
    )

    style_poloidal(hx, d, xlim=(0.7, 2.45), ylim=(-1.55, 1.55))
    vmax = float(np.nanpercentile(show.compressed(), 96))
    cf = hx.contourf(
        d["RR"], d["ZZ"], show, levels=14, cmap="YlOrRd", vmin=0, vmax=vmax, zorder=1,
    )
    faint_lcfs(hx, d)
    hx.plot(d["r_ax"], d["z_ax"], "o", color=ACCENT, ms=6, zorder=6)
    for r, z in points:
        hx.plot(r, z, "o", color=TEAL, ms=7, zorder=7)
        hx.plot(r, z, "o", color="white", ms=2.5, zorder=8)
    cb = fig.colorbar(cf, ax=hx, fraction=0.046, pad=0.04)
    cb.set_label(r"how much the $\mathbf{B}$ arrows turn  $|(\nabla\times\mathbf{B})_\varphi|$")
    hx.set_title(r"a number at each place — still not particles", fontsize=11)
    print("wrote", save(fig, "tokamak_curl_levels.png"))


def fig_not_tracks() -> None:
    """Kill 'red = packed tracks': |∇ψ| is pale on axis; |curl| is red there."""
    d = load_frame()
    dpsidR = np.gradient(d["psi"], d["R"], axis=1)
    dpsidZ = np.gradient(d["psi"], d["Z"], axis=0)
    gpsi = np.hypot(dpsidR, dpsidZ)
    pack = np.ma.array(gpsi, mask=~d["inside"])
    curl = np.ma.array(np.abs(_curl_phi(d)), mask=~d["inside"])

    fig, (ax, hx, lx) = plt.subplots(1, 3, figsize=(13.2, 4.6), dpi=140)
    style_poloidal(ax, d, xlim=(0.7, 2.45), ylim=(-1.55, 1.55))
    vmax_p = float(np.nanpercentile(pack.compressed(), 96))
    cf1 = ax.contourf(
        d["RR"], d["ZZ"], pack, levels=14, cmap="YlOrRd", vmin=0, vmax=vmax_p, zorder=1,
    )
    ax.contour(d["RR"], d["ZZ"], d["psi"], levels=12, colors="#333", linewidths=0.35, alpha=0.45, zorder=2)
    faint_lcfs(ax, d)
    ax.plot(d["r_ax"], d["z_ax"], "o", color=ACCENT, ms=6, zorder=6)
    ax.text(d["r_ax"] - 0.42, d["z_ax"] + 0.12, "axis: pale", color=ACCENT, fontsize=8)
    cb1 = fig.colorbar(cf1, ax=ax, fraction=0.046, pad=0.04)
    cb1.set_label(r"track packing  $|\nabla\psi|$")
    ax.set_title(r"how close the loops sit  —  pale on axis", fontsize=11)

    style_poloidal(hx, d, xlim=(0.7, 2.45), ylim=(-1.55, 1.55))
    vmax_c = float(np.nanpercentile(curl.compressed(), 96))
    cf2 = hx.contourf(
        d["RR"], d["ZZ"], curl, levels=14, cmap="YlOrRd", vmin=0, vmax=vmax_c, zorder=1,
    )
    faint_lcfs(hx, d)
    hx.plot(d["r_ax"], d["z_ax"], "o", color=ACCENT, ms=6, zorder=6)
    hx.text(d["r_ax"] - 0.42, d["z_ax"] + 0.12, "axis: red", color=ACCENT, fontsize=8)
    cb2 = fig.colorbar(cf2, ax=hx, fraction=0.046, pad=0.04)
    cb2.set_label(r"long-way current (curl)  $|(\nabla\times\mathbf{B})_\varphi|$")
    hx.set_title(r"where current punches through  —  red on axis", fontsize=11)

    row = int(np.argmin(np.abs(d["Z"] - d["z_ax"])))
    inside_row = d["inside"][row]
    R = d["R"]
    g = np.where(inside_row, gpsi[row], np.nan)
    c = np.where(inside_row, np.abs(_curl_phi(d)[row]), np.nan)
    lx.set_facecolor(BG)
    lx.plot(R, g / np.nanmax(g), color=ORANGE, lw=2.2, label=r"$|\nabla\psi|$  (packing)")
    lx.plot(R, c / np.nanmax(c), color=TEAL, lw=2.2, label=r"$|(\nabla\times\mathbf{B})_\varphi|$  (current)")
    lx.axvline(d["r_ax"], color=ACCENT, lw=1.0, ls="--")
    lx.text(d["r_ax"] + 0.04, 0.92, "axis", color=ACCENT, fontsize=8)
    lx.set_xlim(0.9, 2.35)
    lx.set_ylim(-0.05, 1.15)
    lx.set_xlabel(r"major radius  $R$  (m)")
    lx.set_ylabel("each curve scaled to its own max")
    lx.set_title("same midplane: packing dies, current peaks", fontsize=11)
    lx.grid(True, color="#d8d8d4", lw=0.6)
    lx.legend(fontsize=7, loc="upper right")
    print("wrote", save(fig, "tokamak_curl_not_tracks.png"))


def fig_net_flow() -> None:
    """Two-way tickets through a stamp: extras one way are current."""
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.0), dpi=140)

    def row(ax, y, n, marker, color, label=None):
        xs = np.linspace(-0.42, 0.42, max(n, 2)) if n > 1 else [0.0]
        for x in xs[:n]:
            ax.plot(x, y, marker=marker, color=color, ms=16)
        if label is not None:
            ax.text(0.72, y, label, va="center", fontsize=9, color=color)

    def stamp(ax, title, n_to, n_away, note):
        ax.set_facecolor(BG)
        ax.set_xlim(-1.05, 1.35)
        ax.set_ylim(-1.25, 1.15)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title, fontsize=12)
        ax.add_patch(Circle((0, 0), 0.78, fill=False, ec=ORANGE, lw=1.8))
        extras = n_to - n_away
        row(ax, 0.32, n_to, r"$\odot$", TEAL if extras else BLUE, f"{n_to} toward")
        row(ax, -0.08, n_away, r"$\otimes$", GRAY, f"{n_away} away")
        ax.text(
            0,
            -0.62,
            f"extras = {n_to} − {n_away} = {extras}",
            ha="center",
            fontsize=10,
            color=TEAL if extras else GRAY,
            fontweight="bold",
        )
        ax.text(0, -1.08, note, ha="center", fontsize=8.5, color=GRAY)

    stamp(
        axes[0],
        "core stamp — high curl",
        n_to=6,
        n_away=4,
        note="more bits, and a lean one way",
    )
    stamp(
        axes[1],
        "edge stamp — low curl",
        n_to=3,
        n_away=3,
        note="fewer bits, and no lean — no current",
    )
    fig.text(
        0.5,
        0.02,
        r"$\odot$ through the stamp toward you    $\otimes$ away    extras (toward − away) = current",
        ha="center",
        fontsize=9,
        color=GRAY,
    )
    print("wrote", save(fig, "tokamak_curl_net_flow.png"))


def fig_stamp_walk() -> None:
    """Walk the edge of the stamp: B on the edge is extras through the middle."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), dpi=140)

    def edge(ax, p0, p1, color):
        ax.add_patch(
            FancyArrowPatch(
                p0,
                p1,
                arrowstyle="-|>",
                mutation_scale=14,
                lw=2.0,
                color=color,
                shrinkA=2,
                shrinkB=2,
            )
        )

    def panel(ax, title, circulating, note):
        ax.set_facecolor(BG)
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(title, fontsize=12)
        ax.add_patch(
            Rectangle((-0.55, -0.42), 1.1, 0.84, fill=False, ec=ORANGE, lw=1.8)
        )
        if circulating:
            edge(ax, (-0.35, -0.42), (0.35, -0.42), BLUE)   # bottom +R
            edge(ax, (0.55, -0.22), (0.55, 0.22), BLUE)     # right +Z
            edge(ax, (0.35, 0.42), (-0.35, 0.42), BLUE)     # top −R
            edge(ax, (-0.55, 0.22), (-0.55, -0.22), BLUE)   # left −Z
            ax.plot(0, 0, marker=r"$\odot$", color=TEAL, ms=22)
            ax.text(0, -0.18, "extras", ha="center", color=TEAL, fontsize=10)
        else:
            edge(ax, (-0.35, 0.22), (0.35, 0.22), BLUE)
            edge(ax, (-0.35, -0.22), (0.35, -0.22), BLUE)
            edge(ax, (-0.35, 0.0), (0.35, 0.0), BLUE)
            ax.plot(0, 0, marker=r"$\cdot$", color=GRAY, ms=14)
            ax.text(0, -0.18, "no extras", ha="center", color=GRAY, fontsize=10)
        ax.text(0, -1.05, note, ha="center", fontsize=8.5, color=GRAY)

    panel(
        axes[0],
        "walk the stamp — edge",
        True,
        "arrows help you one way around  —  extras through the middle",
    )
    panel(
        axes[1],
        "walk the stamp — cancel",
        False,
        "same arrows on every edge  —  toward equals away",
    )
    print("wrote", save(fig, "tokamak_curl_stamp_walk.png"))


def main() -> None:
    fig_levels()
    fig_not_tracks()
    fig_net_flow()
    fig_stamp_walk()


if __name__ == "__main__":
    main()
