"""d3d_0.0.2_osf_0.0.12 — DIII-D calibration of solver osf_0.0.12.

Symbols are one-shot-forward.md. Numbers are diii-d-calibration.md.
Same shot as 0.0.1. Plasma current and the solenoid scale follow b.
The q95 gates are a least-squares assignment of the existing readout on this shot:
R²(q95) +0.902, 10.15 at 180 ms (EFIT 9.91), 4.25 at 1680 ms (EFIT 4.22).
betaN reads the tangential midplane Thomson block. On this shot R²(betaN) is +0.791.
"""
from __future__ import annotations

from osf.osf_0_0_12 import Calibration, solve_one_frame as _solve_one_frame, solve_shot as _solve_shot, write_shot

MODEL_ID = "d3d_0.0.2_osf_0.0.12"
SOLVER_ID = "osf_0.0.12"

CAL = Calibration(
    calibration_id=MODEL_ID,
    machine="DIII-D",
    r_star=0.34,
    i_star_a=-46261.19891905632,
    r00=1.68,
    rw=0.0,
    rb=0.0,
    z00=-0.163,
    zb=-0.303,
    ar0=0.58,
    aw=0.0,
    a0=0.42,
    ab=0.0,
    ad=0.0,
    gamma_bar=-1.0,
    k0=0.0,
    kw=0.0,
    kb=0.0,
    kd=0.0,
    i0_a=1.804347826e6,
    i_b_a=-1.304347826e6,
    lam_v0=1.0,
    lam_vb=0.0,
    lam_l=0.0,
    lam_delta=1.0,
    lam_o=0.0,
    lam_s0=-52.463768,
    lam_sb=172.463768,
    q_hi=4.207,
    q_lo=1.802,
    alpha_q=1.646,
    eta_q=4.892,
    zeta_q=-5.316,
    p_q=2.524,
    n_beta=7,
    eta_beta=0.0,
    beta_lo=0.01,
    beta_hi=4.0,
    beta_fb=0.53,
    s_beta=1.0,
    rt_lo=1.00,
    rt_hi=2.30,
    n_tf=144.0,
    sigma_ax=-1.0,
    r_axis_min=1.40,
    r_axis_max=2.15,
    z_axis_max=0.55,
    kappa_max=2.8,
    z_span_max=2.6,
    r_out_min=1.70,
    r_in_max=1.35,
    edge_cells=2.0,
    dz_x=2.8,
    x_r_min=1.05,
    x_r_max=2.20,
    x_z_abs=0.55,
    x_dist=0.35,
    sol="magnetics_ECOILA",
    v_u="magnetics_F6A",
    v_l="magnetics_F6B",
    delta=("magnetics_F5A", "magnetics_F5B"),
    outboard=("magnetics_F7A", "magnetics_F7B"),
    tf="magnetics_bcoil",
    ip="magnetics_plasma_current",
    ip_time="magnetics_plasma_current_times",
    loader="named",
    walk_on_measured=False,
    pressure_on_measured=False,
    beta_guess="window",
    readout_pressure="tangential",
    readout_ip="target",
    axis_on_stored=True,
    axis_gate_z=False,
    fallback_closed=False,
    factor_fallback=True,
    smooth_inplace=True,
    post_clip_beta=True,
    fb_from_bt=False,
)


def solve_one_frame(row: dict, t_index: int, *, maps=None, green=None) -> dict:
    return _solve_one_frame(row, t_index, CAL, maps=maps, green=green)


def solve_shot(row: dict, *, green=None, maps=None) -> dict:
    return _solve_shot(row, CAL, green=green, maps=maps)
