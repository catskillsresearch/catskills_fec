"""The two GS profiles p'(ψ), FF'(ψ) — schematic curves, not a fit."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from scipy.interpolate import RegularGridInterpolator

from make_tokamak_symbols import (
    ACCENT,
    BG,
    BLUE,
    GRAY,
    ORANGE,
    PURPLE,
    TEAL,
    faint_lcfs,
    load_frame,
    save,
    style_poloidal,
)

# Track coordinate t: 0 at the magnetic axis, 1 at the last closed track.
# DIII-D ψ is a minimum on axis, so t rises outward with ψ. Schematic only.
T = np.linspace(0.0, 1.0, 240)
P = 1.15 * (1.0 - T) ** 1.55
F = 0.92 + 0.10 * (1.0 - T) ** 1.2
DP = np.gradient(P, T)
DFF = np.gradient(F * F, T)


def _axis_track(ax) -> None:
    ax.set_facecolor(BG)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["magnetic\naxis", "last closed\ntrack"])
    ax.axhline(0.0, color="#ddd", lw=0.7)
    ax.grid(True, axis="y", color="#e6e6e2", lw=0.6)


def fig_profiles() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.4), dpi=140)

    ax = axes[0, 0]
    _axis_track(ax)
    ax.plot(T, P, color=ORANGE, lw=2.2)
    ax.set_ylim(-0.05, 1.35)
    ax.set_ylabel("pressure  $p$")
    ax.set_title("soup push on each track", fontsize=11)
    ax.annotate(
        "",
        xy=(0.42, float(np.interp(0.42, T, P))),
        xytext=(0.62, float(np.interp(0.22, T, P))),
        arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.4),
    )
    ax.text(0.48, 0.95, r"$p'$ is this slope", color=ACCENT, fontsize=9)

    ax = axes[0, 1]
    _axis_track(ax)
    ax.plot(T, F, color=TEAL, lw=2.2)
    ax.set_ylim(0.86, 1.08)
    ax.set_ylabel(r"long-way $F=RB_\varphi$")
    ax.set_title("one $F$ per track", fontsize=11)
    ax.annotate(
        "",
        xy=(0.55, float(np.interp(0.55, T, F))),
        xytext=(0.78, float(np.interp(0.28, T, F))),
        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4),
    )
    ax.text(0.18, 0.895, r"$FF'$ uses this slope", color=BLUE, fontsize=9)

    ax = axes[1, 0]
    _axis_track(ax)
    ax.plot(T, DP, color=ACCENT, lw=2.2)
    ax.set_ylabel(r"$p'(\psi)=dp/d\psi$")
    ax.set_title("missing profile 1 — pressure hop", fontsize=11)
    ax.text(0.04, 0.12, "usually negative:\npressure drops outward", color=GRAY, fontsize=8)

    ax = axes[1, 1]
    _axis_track(ax)
    ax.plot(T, DFF, color=BLUE, lw=2.2)
    ax.set_ylabel(r"$FF'(\psi)=F\,dF/d\psi$")
    ax.set_title("missing profile 2 — long-way hop", fontsize=11)
    ax.text(0.04, 0.02, "small, can change sign", color=GRAY, fontsize=8)

    fig.suptitle(
        r"schematic curves, not a fit  —  one value on every track",
        fontsize=10,
        color=GRAY,
        y=0.02,
    )
    print("wrote", save(fig, "tokamak_gs_profiles.png"))


def _box(ax, xy, wh, text, *, ec, fc="#fff"):
    x, y = xy
    w, h = wh
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=fc,
            edgecolor=ec,
            lw=1.6,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5, color="#222")


def _arrow(ax, p0, p1, color):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=12,
            lw=1.3,
            color=color,
            shrinkA=1,
            shrinkB=1,
        )
    )


def fig_fingerprints() -> None:
    """q95 names F on one track. F only."""
    fig, ax = plt.subplots(figsize=(5.4, 4.6), dpi=140)
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.35, 1.25)
    ax.set_axis_off()
    ax.set_title(r"$q_{95}$ names $F$ on one track", fontsize=11)
    ts = np.linspace(0, 2 * np.pi, 240)
    for a, b, c, lw in (
        (0.28, 0.34, GRAY, 0.9),
        (0.48, 0.58, GRAY, 0.9),
        (0.68, 0.82, GRAY, 0.9),
        (0.88, 1.05, PURPLE, 2.4),
        (1.02, 1.18, ORANGE, 1.6),
    ):
        ax.plot(a * np.cos(ts), b * np.sin(ts) - 0.04 * a, color=c, lw=lw)
    ax.plot(0, 0, "o", color=ACCENT, ms=6)
    ax.text(0.08, 0.06, "axis", color=ACCENT, fontsize=8)
    ax.text(0.55, -0.08, r"$q_{95}$", color=PURPLE, fontsize=11)
    ax.text(-0.15, -1.22, "LCFS", color=ORANGE, fontsize=8)
    ax.text(
        0,
        1.12,
        r"$F(t;q_{95})=2\pi q_{95}/W+f(t)$",
        ha="center",
        fontsize=9,
        color=GRAY,
    )
    print("wrote", save(fig, "tokamak_gs_fingerprints.png"))


def fig_need() -> None:
    fig, ax = plt.subplots(figsize=(10.6, 4.2), dpi=140)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 4.2)
    ax.set_axis_off()
    ax.set_title("what the scorer actually wants", fontsize=12)

    _box(ax, (0.25, 1.45), (2.5, 1.3), r"track map  $\psi(R,Z)$", ec=BLUE, fc="#eaf4f7")
    _arrow(ax, (2.85, 2.55), (3.55, 3.15), PURPLE)
    _arrow(ax, (2.85, 1.65), (3.55, 1.05), ORANGE)

    _box(
        ax,
        (3.55, 2.55),
        (2.7, 1.2),
        "whole curve $F(\\psi)$\nnot submitted",
        ec=PURPLE,
        fc="#f4eef8",
    )
    _box(
        ax,
        (3.55, 0.45),
        (2.7, 1.2),
        "whole curve $p(\\psi)$\nnot submitted",
        ec=ORANGE,
        fc="#fff4ec",
    )
    _arrow(ax, (6.35, 3.15), (7.15, 3.15), PURPLE)
    _arrow(ax, (6.35, 1.05), (7.15, 1.05), ORANGE)

    _box(
        ax,
        (7.15, 2.55),
        (3.15, 1.2),
        r"one number  $q_{95}$" "\nsubmit this",
        ec=PURPLE,
        fc="#f4eef8",
    )
    _box(
        ax,
        (7.15, 0.45),
        (3.15, 1.2),
        r"one number  $\beta_N$" "\nsubmit this",
        ec=ORANGE,
        fc="#fff4ec",
    )

    ax.text(
        5.3,
        0.12,
        r"$p'$ and $FF'$ stay off the scorecard  —  matching $\psi$ already sat on some pair",
        ha="center",
        fontsize=9,
        color=GRAY,
    )
    print("wrote", save(fig, "tokamak_gs_need.png"))


def fig_family_F() -> None:
    """F(t; q95) family. F only."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=140)
    t = np.linspace(0.0, 1.0, 240)
    t95 = 0.95
    f95 = 1.00
    _axis_track(ax)
    members = (
        (0.0, 1.0, TEAL, r"$f=0$"),
        (0.14, 1.3, BLUE, "core bump"),
        (0.22, 0.7, "#5dade2", "bigger bump"),
        (-0.10, 1.1, "#148f77", "core dip"),
    )
    for amp, pow_, c, lab in members:
        y = f95 + amp * ((1.0 - t) ** pow_ - (1.0 - t95) ** pow_)
        ax.plot(t, y, color=c, lw=2.0, label=lab)
    ax.plot([t95], [f95], "o", color=PURPLE, ms=9, zorder=5)
    ax.annotate(
        r"$F_{95}=2\pi q_{95}/W$",
        xy=(t95, f95),
        xytext=(0.28, 1.16),
        color=PURPLE,
        fontsize=9,
        arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.1),
    )
    ax.set_ylim(0.86, 1.24)
    ax.set_ylabel(r"$F(t;q_{95})$  ($\mathrm{T\cdot m}$)")
    ax.set_title(r"$F(t;q_{95})=2\pi q_{95}/W+f(t)$,  $f(t_{95})=0$", fontsize=11)
    ax.legend(framealpha=0.92, fontsize=8, loc="upper right")
    print("wrote", save(fig, "tokamak_gs_family_F.png"))


def fig_family_p() -> None:
    """p(t; betaN) family. p only."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=140)
    t = np.linspace(0.0, 1.0, 240)
    _axis_track(ax)
    mean = 0.42
    raw = (
        ((1 - t) ** 2.2, ACCENT, "peaked $s$"),
        ((1 - t) ** 1.2, ORANGE, "medium $s$"),
        ((1 - t) ** 0.55, "#e67e22", "broader $s$"),
        ((1 - 0.10 * t), TEAL, "almost flat $s$"),
    )
    for y0, c, lab in raw:
        y = y0 * (mean / np.mean(y0))
        ax.plot(t, y, color=c, lw=2.0, label=lab)
    ax.axhline(mean, color=GRAY, lw=1.0, ls="--")
    ax.text(0.04, mean + 0.08, r"$\langle p\rangle(\beta_N)$", color=GRAY, fontsize=9)
    ax.set_ylim(-0.02, 1.45)
    ax.set_ylabel(r"$p(t;\beta_N)$  (Pa, schematic)")
    ax.set_title(r"$p(t;\beta_N)=\langle p\rangle(\beta_N)\,s(t)/\langle s\rangle$", fontsize=11)
    ax.legend(framealpha=0.92, fontsize=8, loc="upper right")
    print("wrote", save(fig, "tokamak_gs_family_p.png"))


def fig_fit_loop() -> None:
    fig, ax = plt.subplots(figsize=(10.6, 4.4), dpi=140)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 4.4)
    ax.set_axis_off()
    ax.set_title(r"fitted $F$, $p$, boundary  $\to$  $\psi$", fontsize=12)
    boxes = (
        (0.20, 2.35, 2.35, 1.45, r"fitted $F(\tau)$" "\n" r"fitted $p(\tau)$", BLUE, "#eaf4f7"),
        (2.90, 2.35, 2.35, 1.45, "boundary\ncoils or named curve", PURPLE, "#f4eef8"),
        (5.60, 2.35, 2.35, 1.45, "Grad–Shafranov\n" r"$\Delta^*\psi=\cdots$", ORANGE, "#fff4ec"),
        (8.30, 2.35, 2.05, 1.45, r"new map $\psi$", TEAL, "#e8f6f3"),
    )
    for x, y, w, h, text, ec, fc in boxes:
        _box(ax, (x, y), (w, h), text, ec=ec, fc=fc)
    for x0, x1 in ((2.55, 2.90), (5.25, 5.60), (7.95, 8.30)):
        _arrow(ax, (x0, 3.07), (x1, 3.07), GRAY)
    _arrow(ax, (9.32, 2.35), (9.32, 0.85), GRAY)
    _box(
        ax,
        (5.60, 0.25),
        (4.75, 1.15),
        r"recompute $W$, $\Delta\psi$, $\langle s\rangle_V$",
        ec=ACCENT,
        fc="#fdedec",
    )
    _arrow(ax, (5.60, 0.82), (1.38, 0.82), GRAY)
    _arrow(ax, (1.38, 0.82), (1.38, 2.35), GRAY)
    print("wrote", save(fig, "tokamak_gs_fit_loop.png"))


def fig_fit_family() -> None:
    """Two shapes that share the two numbers can still make different maps."""
    d = load_frame()
    interp = RegularGridInterpolator((d["Z"], d["R"]), d["psi"], bounds_error=False)
    psi_ax = float(interp([[d["z_ax"], d["r_ax"]]])[0])
    pts = np.column_stack([d["lc_z"], d["lc_r"]])
    psi_b = float(np.nanmedian(interp(pts)))
    t = np.where(d["inside"], (d["psi"] - psi_ax) / (psi_b - psi_ax), np.nan)
    peaked = np.where(d["inside"], (1.0 - np.clip(t, 0, 1)) ** 3.0, np.nan)
    broad = np.where(d["inside"], (1.0 - np.clip(t, 0, 1)) ** 0.55, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), dpi=140)
    for ax, field, title in (
        (axes[0], peaked, "peaked $p$  —  extras sit on axis"),
        (axes[1], broad, "broad $p$  —  extras spread out"),
    ):
        style_poloidal(ax, d, xlim=(0.7, 2.45), ylim=(-1.55, 1.55))
        show = np.ma.array(field, mask=~d["inside"])
        ax.contourf(d["RR"], d["ZZ"], show, levels=14, cmap="magma", zorder=1)
        ax.contour(d["RR"], d["ZZ"], t, levels=8, colors="#fff", linewidths=0.35, alpha=0.35, zorder=2)
        faint_lcfs(ax, d)
        ax.plot(d["r_ax"], d["z_ax"], "o", color="white", ms=5, zorder=6)
        ax.set_title(title, fontsize=11)
    fig.suptitle(
        r"same $\beta_N$ family, same last closed track  —  different $\psi$ seating",
        fontsize=10,
        color=GRAY,
        y=0.02,
    )
    print("wrote", save(fig, "tokamak_gs_fit_family.png"))


def main() -> None:
    fig_profiles()
    fig_need()
    fig_fingerprints()
    fig_family_F()
    fig_family_p()
    fig_fit_loop()
    fig_fit_family()


if __name__ == "__main__":
    main()
