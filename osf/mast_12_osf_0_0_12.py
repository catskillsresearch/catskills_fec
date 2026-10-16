"""mast_12_osf_0.0.12 — MAST calibration of solver osf_0.0.12.

Symbols are one-shot-forward.md. Numbers are mast-calibration.md.
Leaderboard probes from submits/submissions.json confirmed zeta_q <= 2.0 (probe 5)
and eta_beta = 0.20 (probe 4), and solved the global paramagnetic factor to c* = 1.0622 (probes 1-3).
"""
from __future__ import annotations

from osf.osf_0_0_12 import Calibration, solve_one_frame as _solve_one_frame, solve_shot as _solve_shot, write_shot

MODEL_ID = "mast_12_osf_0.0.12"
SOLVER_ID = "osf_0.0.12"

CAL = Calibration(
    calibration_id=MODEL_ID,
    machine="MAST",
    r_star=0.34,
    i_star_a=16.3e3,
    r00=0.82,
    rw=0.10,
    rb=0.02,
    z00=0.009,
    zb=0.004,
    ar0=0.50,
    aw=0.08,
    a0=0.08,
    ab=0.42,
    ad=0.18,
    gamma_bar=-1.0,
    k0=0.70,
    kw=0.30,
    kb=0.15,
    kd=0.20,
    i0_a=0.0,
    i_b_a=0.0,
    lam_v0=1.8,
    lam_vb=-1.0,
    lam_l=0.105,
    lam_delta=1.15,
    lam_o=-0.18,
    lam_s0=0.5,
    lam_sb=0.10,
    q_hi=17.0,
    q_lo=8.0,
    alpha_q=0.55,
    eta_q=0.35,
    zeta_q=2.0,       # Leaderboard probe 5 confirmed gradient < 0, validating zeta_q <= 2.0
    p_q=3.0,
    n_beta=7,
    eta_beta=0.20,    # Leaderboard probe 4 confirmed gradient < 0, validating eta_beta = 0.20
    beta_lo=0.2,
    beta_hi=4.0,
    beta_fb=1.3,
    s_beta=1.0,
    rt_lo=0.25,
    rt_hi=1.45,
    n_tf=24.0,
    sigma_ax=1.0,
    r_axis_min=0.65,
    r_axis_max=1.05,
    z_axis_max=0.50,
    kappa_max=2.6,
    z_span_max=2.8,
    r_out_min=1.00,
    r_in_max=0.40,
    edge_cells=2.0,
    dz_x=3.2,
    x_r_min=0.25,
    x_r_max=1.50,
    x_z_abs=0.40,
    x_dist=0.25,
    sol="magnetics_sol_current",
    v_u="magnetics_p2u_current",
    v_l="magnetics_p2l_current",
    delta=("magnetics_p4u_current", "magnetics_p4l_current"),
    outboard=("magnetics_p5u_current", "magnetics_p5l_current"),
    tf="magnetics_tf_current",
    ip="magnetics_plasma_current",
    ip_time="magnetics_time",
    loader="mast_columns",
    walk_on_measured=True,
    pressure_on_measured=True,
    beta_guess="osf_0_0_1",
    readout_pressure="lcfs",
    readout_ip="measured",
    axis_on_stored=False,
    axis_gate_z=True,
    fallback_closed=True,
    factor_fallback=False,
    smooth_inplace=False,
    post_clip_beta=False,
    fb_from_bt=True,
)


def solve_one_frame(row: dict, t_index: int, *, maps=None, green=None) -> dict:
    return _solve_one_frame(row, t_index, CAL, maps=maps, green=green)


def solve_shot(row: dict, *, green=None, maps=None) -> dict:
    return _solve_shot(row, CAL, green=green, maps=maps)
