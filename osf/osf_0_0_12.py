"""osf_0.0.12 — one-shot forward Grad–Shafranov solver.

Symbols are one-shot-forward.md. A calibration assigns every symbol on this
release. MAST is mast_12_osf_0.0.12. DIII-D is d3d_0.0.1_osf_0.0.12.
The numbers below are the MAST assignment, so a call that has not bound a
calibration still solves that calibration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.feq.data import interpolate

from osf.osf_0_0_1 import (
    E_CHARGE,
    MAX_ITERS,
    MU0,
    PSI_ABS_TOL,
    SCALAR_RTOL,
    TAU_95,
    FrameResult,
    _bilinear,
    _beta_n_raw,
    _closed_around,
    knobs_for_frame,
    p_prime_of_tau,
    s_of_tau,
    thomson_pressure,
    w_on_track,
)

MODEL_ID = "osf_0.0.12"

R0_M = 0.82
A_TRIAL_M = 0.50
A_Z_M = 0.58  # A_TRIAL + A_Z_BASE
KAPPA_MAX = 2.6
Z_SPAN_MAX_M = 2.8
R_AXIS_MIN_M = 0.65
R_AXIS_MAX_M = 1.05
Z_AXIS_MAX_M = 0.50
R_LCFS_MAX_MIN_M = 1.00
R_LCFS_INBOARD_MAX_M = 0.40
EDGE_CELLS = 2.0
IP_SCALE = 0.70
GAMMA_BAR = -1.0  # li ~0.40 high at −2. Still FF' > 0.
P2_SCALE = 1.8  # diverted end. ×2 squeezed volume; ×1.8 is the pinch/volume fit
BLEED = 0.0
P4_SCALE = 1.15  # seats both pinches. ×0.85 left δ low. Volume pays.
F4_COL = ""
F4_SCALE = 1.0
SOL_SCALE = 0.5  # diverted end (centre-column shield)
P2_IP_REF = 0.34  # |P2|/|Ip| at the seated 90 ms frame
SOL_BLEED_REF_A = 16.3e3  # amperes. First-frame OH at limited end; bleed=1 there, 0 at I_sol ≤ 0.
A_Z_BASE_M = 0.08  # diverted kappa / li; support was a circle
A_Z_EXTRA_M = 0.42  # limited end (~70 ms): taller limited oval (was 0.35)
A_Z_DIVERT_M = 0.18  # diverted: +6 cm on the 0.0.11 ellipse. κ was 15 cm short.
P2L_DIVERT = 0.105  # extra on P2L only. ×1.15 left δ_top low once a_Z grew.
P5_DIVERT = -0.18  # diverted ×0.82. ×0.85 was 0.0.11; ×0.80 reopened δ_bot on the bundle.
R0_BLEED_M = 0.02  # limited end (~70 ms): R0 inboard while limited (was fat outboard)
Z0_DIVERT_M = 0.009  # diverted end. Fixed 8 mm was anti-correlated with EFIT (r=−0.68).
Z0_BLEED_M = 0.004  # b=1 → 5 mm, near the early-frame EFIT axis.
Z0_M = 0.009
IP_BLEED = 0.15  # limited end (~70 ms): 0.85 Ip at b=1 (sol well on; do not add full Ip)
IP_DIVERT = 0.20  # diverted ×0.90. ×0.85 left volume low after P5 ×0.82.
SOL_BLEED = 0.10  # limited end (~70 ms): sol ×0.60 at b=1 (was 0.50 → full sol)
Q95_BLEED_SCALE = 0.55  # map q × this on the OH well
Q95_WALK_SCALE = 0.35  # Ampère W is short just after divert; ×(1+0.35(1-w)(1-b))
Q95_WELL_MAP = 17.0  # q_map still the solenoid well (W long)
Q95_DIVERT_MAP = 8.0  # Ampère has shortened W; walk law from here down
Q95_HOLE_SCALE = 2.0  # +2 b (1-b)^3 while OH is still on and W has collapsed
Q95_HOLE_POWER = 3.0  # peaks near b=1/4 (divert approach). Off at b=0
BETAN_TIME_MEDIAN = 7  # efit frames (~35 ms). One laser peak is not equilibrium ⟨p⟩.
BETAN_WALK_SCALE = 0.20  # midplane ⟨p⟩ vs volume; ×(1-0.20 w) as P2 walks off
USE_VACUUM_XD = False  # existence miss: j on an X-D does not seat extract_lcfs

SUPPORT_INSIDE: np.ndarray | None = None
SUPPORT_TAU: np.ndarray | None = None
SUPPORT_INFO: dict = {}

# Symbol coefficients. Defaults are the MAST assignment. bind() overwrites them.
CALIBRATION_ID = "mast_12_osf_0.0.12"
MACHINE_NAME = "MAST"
R_STAR = P2_IP_REF
I_STAR_A = SOL_BLEED_REF_A
R00 = 0.82
RW = 0.10
AR0 = 0.50
AW = 0.08
K0 = 0.70
KW = 0.30
LAM_V0 = 1.8
LAM_VB = -1.0
LAM_S0 = 0.5
LAM_L = P2L_DIVERT
LAM_O = P5_DIVERT
I0_A = 0.0
I_B_A = 0.0
N_TF_TURNS = 24.0
SIGMA_AX = 1.0
DZ_X = 3.2
X_R_MIN = 0.25
X_R_MAX = 1.50
X_Z_ABS = 0.40
X_DIST = 0.25
BETA_LO = 0.2
BETA_HI = 4.0
BETA_FB = 1.3
S_BETA = 1.0
RT_LO = 0.25
RT_HI = 1.45
SOL_COL = "magnetics_sol_current"
V_U = "magnetics_p2u_current"
V_L = "magnetics_p2l_current"
DELTA_COLS = ("magnetics_p4u_current", "magnetics_p4l_current")
OUTBOARD_COLS = ("magnetics_p5u_current", "magnetics_p5l_current")
TF_COL = "magnetics_tf_current"
IP_COL = "magnetics_plasma_current"
IP_TIME_KEY = "magnetics_time"
LOADER = "mast_columns"
WALK_ON_MEASURED = True
PRESSURE_ON_MEASURED = True
BETA_GUESS = "osf_0_0_1"
READOUT_PRESSURE = "lcfs"
READOUT_IP = "measured"
AXIS_ON_STORED = False
AXIS_GATE_Z = True
FALLBACK_CLOSED = True
FACTOR_FALLBACK = False
SMOOTH_INPLACE = False
POST_CLIP_BETA = False
FB_FROM_BT = True


@dataclass(frozen=True)
class Calibration:
    """One assignment of the osf_0.0.12 symbols. The solver reads these after bind."""

    calibration_id: str
    machine: str
    r_star: float
    i_star_a: float
    r00: float
    rw: float
    rb: float
    z00: float
    zb: float
    ar0: float
    aw: float
    a0: float
    ab: float
    ad: float
    gamma_bar: float
    k0: float
    kw: float
    kb: float
    kd: float
    i0_a: float
    i_b_a: float
    lam_v0: float
    lam_vb: float
    lam_l: float
    lam_delta: float
    lam_o: float
    lam_s0: float
    lam_sb: float
    q_hi: float
    q_lo: float
    alpha_q: float
    eta_q: float
    zeta_q: float
    p_q: float
    n_beta: int
    eta_beta: float
    beta_lo: float
    beta_hi: float
    beta_fb: float
    s_beta: float
    rt_lo: float
    rt_hi: float
    n_tf: float
    sigma_ax: float
    r_axis_min: float
    r_axis_max: float
    z_axis_max: float
    kappa_max: float
    z_span_max: float
    r_out_min: float
    r_in_max: float
    edge_cells: float
    dz_x: float
    x_r_min: float
    x_r_max: float
    x_z_abs: float
    x_dist: float
    sol: str
    v_u: str
    v_l: str
    delta: tuple[str, str]
    outboard: tuple[str, str]
    tf: str
    ip: str
    ip_time: str
    loader: str
    walk_on_measured: bool
    pressure_on_measured: bool
    beta_guess: str
    readout_pressure: str
    readout_ip: str
    axis_on_stored: bool
    axis_gate_z: bool
    fallback_closed: bool
    factor_fallback: bool
    smooth_inplace: bool
    post_clip_beta: bool
    fb_from_bt: bool
    psi00: float = 0.0
    psi0b: float = 0.0
    lam_f4: float = 1.0
    f4: str = ""


def bind(cal: Calibration) -> None:
    """Copy one calibration onto the symbols the solve reads."""
    global CALIBRATION_ID, MACHINE_NAME, R_STAR, I_STAR_A, R00, RW, AR0, AW, K0, KW
    global LAM_V0, LAM_VB, LAM_S0, LAM_L, LAM_O, I0_A, I_B_A, N_TF_TURNS, SIGMA_AX, DZ_X
    global X_R_MIN, X_R_MAX, X_Z_ABS, X_DIST, BETA_LO, BETA_HI, BETA_FB, S_BETA, RT_LO, RT_HI
    global SOL_COL, V_U, V_L, DELTA_COLS, OUTBOARD_COLS, TF_COL, IP_COL, IP_TIME_KEY
    global LOADER, WALK_ON_MEASURED, PRESSURE_ON_MEASURED, BETA_GUESS, READOUT_PRESSURE, READOUT_IP
    global AXIS_ON_STORED, AXIS_GATE_Z, FALLBACK_CLOSED, FACTOR_FALLBACK, SMOOTH_INPLACE, POST_CLIP_BETA, FB_FROM_BT
    global R0_BLEED_M, Z0_DIVERT_M, Z0_BLEED_M, A_Z_BASE_M, A_Z_EXTRA_M, A_Z_DIVERT_M
    global IP_BLEED, IP_DIVERT, SOL_BLEED, P2L_DIVERT, P5_DIVERT, P4_SCALE, GAMMA_BAR
    global Q95_BLEED_SCALE, Q95_WALK_SCALE, Q95_WELL_MAP, Q95_DIVERT_MAP, Q95_HOLE_SCALE, Q95_HOLE_POWER
    global BETAN_TIME_MEDIAN, BETAN_WALK_SCALE
    global R_AXIS_MIN_M, R_AXIS_MAX_M, Z_AXIS_MAX_M, KAPPA_MAX, Z_SPAN_MAX_M
    global R_LCFS_MAX_MIN_M, R_LCFS_INBOARD_MAX_M, EDGE_CELLS
    global PSI00, PSI0B, F4_COL, F4_SCALE
    CALIBRATION_ID = cal.calibration_id
    MACHINE_NAME = cal.machine
    R_STAR = cal.r_star
    I_STAR_A = cal.i_star_a
    R00 = cal.r00
    RW = cal.rw
    R0_BLEED_M = cal.rb
    Z0_DIVERT_M = cal.z00
    Z0_BLEED_M = cal.zb
    AR0 = cal.ar0
    AW = cal.aw
    A_Z_BASE_M = cal.a0
    A_Z_EXTRA_M = cal.ab
    A_Z_DIVERT_M = cal.ad
    GAMMA_BAR = cal.gamma_bar
    K0 = cal.k0
    KW = cal.kw
    IP_BLEED = cal.kb
    IP_DIVERT = cal.kd
    I0_A = cal.i0_a
    I_B_A = cal.i_b_a
    LAM_V0 = cal.lam_v0
    LAM_VB = cal.lam_vb
    LAM_L = cal.lam_l
    P2L_DIVERT = cal.lam_l
    P4_SCALE = cal.lam_delta
    LAM_O = cal.lam_o
    P5_DIVERT = cal.lam_o
    LAM_S0 = cal.lam_s0
    SOL_BLEED = cal.lam_sb
    Q95_WELL_MAP = cal.q_hi
    Q95_DIVERT_MAP = cal.q_lo
    Q95_BLEED_SCALE = cal.alpha_q
    Q95_WALK_SCALE = cal.eta_q
    Q95_HOLE_SCALE = cal.zeta_q
    Q95_HOLE_POWER = cal.p_q
    BETAN_TIME_MEDIAN = cal.n_beta
    BETAN_WALK_SCALE = cal.eta_beta
    BETA_LO = cal.beta_lo
    BETA_HI = cal.beta_hi
    BETA_FB = cal.beta_fb
    S_BETA = cal.s_beta
    RT_LO = cal.rt_lo
    RT_HI = cal.rt_hi
    N_TF_TURNS = cal.n_tf
    SIGMA_AX = cal.sigma_ax
    R_AXIS_MIN_M = cal.r_axis_min
    R_AXIS_MAX_M = cal.r_axis_max
    Z_AXIS_MAX_M = cal.z_axis_max
    KAPPA_MAX = cal.kappa_max
    Z_SPAN_MAX_M = cal.z_span_max
    R_LCFS_MAX_MIN_M = cal.r_out_min
    R_LCFS_INBOARD_MAX_M = cal.r_in_max
    EDGE_CELLS = cal.edge_cells
    DZ_X = cal.dz_x
    X_R_MIN = cal.x_r_min
    X_R_MAX = cal.x_r_max
    X_Z_ABS = cal.x_z_abs
    X_DIST = cal.x_dist
    SOL_COL = cal.sol
    V_U = cal.v_u
    V_L = cal.v_l
    DELTA_COLS = cal.delta
    OUTBOARD_COLS = cal.outboard
    TF_COL = cal.tf
    IP_COL = cal.ip
    IP_TIME_KEY = cal.ip_time
    LOADER = cal.loader
    WALK_ON_MEASURED = cal.walk_on_measured
    PRESSURE_ON_MEASURED = cal.pressure_on_measured
    BETA_GUESS = cal.beta_guess
    READOUT_PRESSURE = cal.readout_pressure
    READOUT_IP = cal.readout_ip
    AXIS_ON_STORED = cal.axis_on_stored
    AXIS_GATE_Z = cal.axis_gate_z
    FALLBACK_CLOSED = cal.fallback_closed
    FACTOR_FALLBACK = cal.factor_fallback
    SMOOTH_INPLACE = cal.smooth_inplace
    POST_CLIP_BETA = cal.post_clip_beta
    FB_FROM_BT = cal.fb_from_bt
    PSI00 = cal.psi00
    PSI0B = cal.psi0b
    F4_COL = cal.f4
    F4_SCALE = cal.lam_f4


def magnetics_knobs(currents_a: dict[str, float], ip_a: float) -> dict[str, float]:
    """Ellipse and coil scales from the walk and bleed clocks."""
    global R0_M, A_TRIAL_M, A_Z_M, IP_SCALE, SOL_SCALE, P2_SCALE, BLEED, Z0_M
    iv = 0.5 * (
        abs(float(currents_a.get(V_U, 0.0)))
        + abs(float(currents_a.get(V_L, 0.0)))
    )
    sol = float(currents_a.get(SOL_COL, 0.0))
    bleed = min(max(sol / I_STAR_A, 0.0), 1.0) if I_STAR_A != 0.0 else 0.0
    BLEED = bleed
    if WALK_ON_MEASURED:
        denom = max(abs(float(ip_a)), 1.0)
    else:
        # k_w is applied after w exists. This ratio uses the target at w = 0.
        pre = K0 + IP_BLEED * bleed + IP_DIVERT * (1.0 - bleed)
        denom = max(abs(pre * float(ip_a) + I0_A + I_B_A * bleed), 1.0)
    ratio = iv / denom
    walk = (1.0 - min(ratio / R_STAR, 1.0)) ** 2
    A_TRIAL_M = AR0 + AW * walk
    R0_M = R00 + RW * walk - R0_BLEED_M * bleed
    Z0_M = Z0_DIVERT_M - Z0_BLEED_M * bleed
    A_Z_M = A_TRIAL_M + A_Z_BASE_M + A_Z_EXTRA_M * bleed + A_Z_DIVERT_M * (1.0 - bleed)
    IP_SCALE = K0 + KW * walk + IP_BLEED * bleed + IP_DIVERT * (1.0 - bleed)
    # λ_Vb = −1 is λ_V0 − b, the same subtraction as the MAST line 1.8 − b.
    if LAM_VB == -1.0:
        P2_SCALE = LAM_V0 - bleed
    else:
        P2_SCALE = LAM_V0 + LAM_VB * bleed
    SOL_SCALE = LAM_S0 + SOL_BLEED * bleed
    return {
        "walk": walk,
        "p2_ip": ratio,
        "r0": R0_M,
        "z0": Z0_M,
        "a": A_TRIAL_M,
        "a_z": A_Z_M,
        "ip_scale": IP_SCALE,
        "bleed": bleed,
        "p2_scale": P2_SCALE,
        "p2l_scale": P2_SCALE * (1.0 + P2L_DIVERT * (1.0 - bleed)),
        "p5_scale": 1.0 + P5_DIVERT * (1.0 - bleed),
        "sol_scale": SOL_SCALE,
    }


def scale_vacuum_currents(currents_a: dict[str, float]) -> dict[str, float]:
    """Scale the five coil roles. Unlisted coils stay at the measured current."""
    out = dict(currents_a)
    if SOL_COL in out:
        out[SOL_COL] *= SOL_SCALE
    for key in (V_U, V_L):
        if key in out:
            out[key] *= P2_SCALE
    # Pair scale first, then extra on the lower member only. Limited b=1: ×1.
    if V_L in out:
        out[V_L] *= 1.0 + LAM_L * (1.0 - BLEED)
    for key in DELTA_COLS:
        if key in out:
            out[key] *= P4_SCALE
    if LAM_O != 0.0:
        g = 1.0 + LAM_O * (1.0 - BLEED)
        for key in OUTBOARD_COLS:
            if key in out:
                out[key] *= g
    if F4_COL and F4_COL in out:
        out[F4_COL] *= F4_SCALE
    return out


def trial_disk(r: np.ndarray, z: np.ndarray) -> np.ndarray:
    """j_φ support: ellipse, or vacuum-X Miller D after prepare_support."""
    if SUPPORT_INSIDE is not None:
        return SUPPORT_INSIDE
    zz, rr = np.meshgrid(z, r, indexing="ij")
    a_r = max(A_TRIAL_M, 1.0e-6)
    a_z = max(A_Z_M, 1.0e-6)
    return ((rr - R0_M) / a_r) ** 2 + ((zz - Z0_M) / a_z) ** 2 <= 1.0


def _points_in_poly(rq: np.ndarray, zq: np.ndarray, poly: np.ndarray) -> np.ndarray:
    r1 = poly[:, 0]
    z1 = poly[:, 1]
    r2 = np.roll(r1, -1)
    z2 = np.roll(z1, -1)
    inside = np.zeros(rq.shape, dtype=bool)
    for e in range(len(r1)):
        za, zb = z1[e], z2[e]
        if za == zb:
            continue
        cross = (za > zq) != (zb > zq)
        if not cross.any():
            continue
        xint = r1[e] + (zq - za) * (r2[e] - r1[e]) / (zb - za)
        inside ^= cross & (rq < xint)
    return inside


def miller_poly(
    r0: float,
    z0: float,
    a: float,
    kappa_u: float,
    kappa_d: float,
    d_top: float,
    d_bot: float,
    *,
    n: int = 256,
) -> np.ndarray:
    """Closed (R,Z) Miller D. θ=0 outboard, θ=π/2 top. Tips at δ, height at κ."""
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    s = np.sin(th)
    kap = np.where(s >= 0.0, kappa_u, kappa_d)
    den = np.where(s >= 0.0, d_top, d_bot)
    rr = r0 + a * np.cos(th + den * s)
    zz = z0 + a * kap * s
    return np.column_stack([rr, zz])


def divertor_x_pair(
    r: np.ndarray, z: np.ndarray, psi: np.ndarray
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Sharpest saddle in the divertor box on this map, each half-plane."""
    zz, rr = np.meshgrid(z, r, indexing="ij")
    dpsz, dpsr = np.gradient(psi, z, r)
    g = np.hypot(dpsr, dpsz)
    d2r = np.gradient(dpsr, r, axis=1)
    d2z = np.gradient(dpsz, z, axis=0)
    d2rz = np.gradient(dpsr, z, axis=0)
    hess = d2r * d2z - d2rz**2
    out = []
    for zlo, zhi in ((0.70, 1.50), (-1.50, -0.70)):
        box = (rr > 0.40) & (rr < 1.00) & (zz > zlo) & (zz < zhi) & (hess < 0)
        if not box.any():
            return None
        work = np.where(box, g, np.inf)
        iz, ir = np.unravel_index(int(np.argmin(work)), work.shape)
        if not np.isfinite(work[iz, ir]):
            return None
        out.append((float(r[ir]), float(z[iz])))
    return out[0], out[1]


def prepare_support(r: np.ndarray, z: np.ndarray, psi_vac: np.ndarray) -> dict:
    """Ellipse at b=1. Blend to a D whose tips are the vacuum X as b→0."""
    global SUPPORT_INSIDE, SUPPORT_TAU, SUPPORT_INFO
    SUPPORT_INSIDE = None
    SUPPORT_TAU = None
    SUPPORT_INFO = {"alpha": 0.0, "x_top": None, "x_bot": None}
    alpha = (1.0 - float(BLEED)) if USE_VACUUM_XD else 0.0
    SUPPORT_INFO["alpha"] = alpha
    if alpha < 1.0e-6:
        return SUPPORT_INFO
    pair = divertor_x_pair(r, z, psi_vac)
    if pair is None:
        return SUPPORT_INFO
    (rt, zt), (rb, zb) = pair
    SUPPORT_INFO["x_top"] = (rt, zt)
    SUPPORT_INFO["x_bot"] = (rb, zb)
    a = max(A_TRIAL_M, 1.0e-6)
    kap_ell = max(A_Z_M, 1.0e-6) / a
    d_top = alpha * float(np.clip((R0_M - rt) / a, -1.0, 1.2))
    d_bot = alpha * float(np.clip((R0_M - rb) / a, -1.0, 1.2))
    kap_u = (1.0 - alpha) * kap_ell + alpha * max((zt - Z0_M) / a, 0.4)
    kap_d = (1.0 - alpha) * kap_ell + alpha * max((Z0_M - zb) / a, 0.4)
    poly = miller_poly(R0_M, Z0_M, a, kap_u, kap_d, d_top, d_bot)
    zz, rr = np.meshgrid(z, r, indexing="ij")
    inside = _points_in_poly(rr, zz, poly)
    if int(inside.sum()) < 16:
        return SUPPORT_INFO
    kap_z = np.where(zz >= Z0_M, kap_u, kap_d) * a
    rho2 = ((rr - R0_M) / a) ** 2 + ((zz - Z0_M) / np.maximum(kap_z, 1.0e-6)) ** 2
    SUPPORT_INSIDE = inside
    SUPPORT_TAU = np.where(inside, np.clip(rho2, 0.0, 1.0), 2.0)
    SUPPORT_INFO.update(
        {
            "delta_top": d_top,
            "delta_bot": d_bot,
            "kappa_u": kap_u,
            "kappa_d": kap_d,
            "n_inside": int(inside.sum()),
        }
    )
    return SUPPORT_INFO


def _edge_pad(r: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    d_r = abs(float(r[1] - r[0])) if r.size > 1 else 0.03
    d_z = abs(float(z[1] - z[0])) if z.size > 1 else 0.06
    return EDGE_CELLS * d_r, EDGE_CELLS * d_z


def x_points(
    r: np.ndarray, z: np.ndarray, psi: np.ndarray, axis_r: float, axis_z: float, *, n: int = 4
) -> list[tuple[float, float, float]]:
    """Saddles of ψ away from the axis: (R, Z, ψ), smallest |∇ψ| first."""
    dpsi_z, dpsi_r = np.gradient(psi, z, r)
    g = np.hypot(dpsi_r, dpsi_z)
    zz, rr = np.meshgrid(z, r, indexing="ij")
    d2r = np.gradient(dpsi_r, r, axis=1)
    d2z = np.gradient(dpsi_z, z, axis=0)
    d2rz = np.gradient(dpsi_r, z, axis=0)
    hess = d2r * d2z - d2rz**2
    mask = (
        (rr > X_R_MIN)
        & (rr < X_R_MAX)
        & (np.abs(zz) > X_Z_ABS)
        & (np.hypot(rr - axis_r, zz - axis_z) > X_DIST)
        & (hess < 0)
    )
    work = np.where(mask, g, np.inf)
    out: list[tuple[float, float, float]] = []
    for _ in range(n):
        k = int(np.argmin(work))
        iz, ir = np.unravel_index(k, work.shape)
        if not np.isfinite(work[iz, ir]):
            break
        out.append((float(r[ir]), float(z[iz]), float(psi[iz, ir])))
        work[max(0, iz - 2) : iz + 3, max(0, ir - 2) : ir + 3] = np.inf
    return out


def last_closed(
    r: np.ndarray, z: np.ndarray, psi: np.ndarray, axis_r: float, axis_z: float
) -> tuple[np.ndarray | None, float, tuple[float, float] | None]:
    """Contour through an X-point if it closes around the axis; else compact last-closed."""
    xs = x_points(r, z, psi, axis_r, axis_z)
    closed: list[tuple[float, np.ndarray, tuple[float, float]]] = []
    for xr, xz, xp in xs:
        cand = _closed_around(r, z, psi, xp, axis_r, axis_z)
        if cand is None:
            continue
        zspan = float(cand[:, 1].max() - cand[:, 1].min())
        if zspan > DZ_X:
            continue
        closed.append((xp, cand, (xr, xz)))
    if closed:
        closed.sort(key=lambda t: t[0])  # lowest ψ = outermost on MAST
        xp, poly, xz = closed[0]
        return poly, float(xp), xz
    poly, psi_b = compact_last_closed(r, z, psi, axis_r, axis_z)
    return poly, psi_b, None


def compact_last_closed(
    r: np.ndarray, z: np.ndarray, psi: np.ndarray, axis_r: float, axis_z: float
) -> tuple[np.ndarray | None, float]:
    """Lowest closed contour around (axis_r, axis_z) that is compact, or None."""
    if not (np.isfinite(axis_r) and np.isfinite(axis_z)):
        return None, float("nan")
    iz = int(np.argmin(np.abs(z - axis_z)))
    ir = int(np.argmin(np.abs(r - axis_r)))
    psi_ax = float(psi[iz, ir])
    floor = float(np.nanmin(psi[np.isfinite(psi)]))
    lo, hi = floor, psi_ax
    poly = None
    pad_r, pad_z = _edge_pad(r, z)
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        cand = _closed_around(r, z, psi, mid, axis_r, axis_z)
        if cand is None or not _compact(r, z, cand, axis_r, pad_r, pad_z):
            lo = mid
        else:
            hi = mid
            poly = cand
    psi_b = float(hi) if poly is not None else float("nan")
    return poly, psi_b


def _compact(
    r: np.ndarray, z: np.ndarray, poly: np.ndarray, axis_r: float, pad_r: float, pad_z: float
) -> bool:
    rmin, rmax = float(poly[:, 0].min()), float(poly[:, 0].max())
    zmin, zmax = float(poly[:, 1].min()), float(poly[:, 1].max())
    width = max(rmax - rmin, 1.0e-9)
    kappa = (zmax - zmin) / width
    if kappa > KAPPA_MAX or (zmax - zmin) > Z_SPAN_MAX_M:
        return False
    if not axis_ok(axis_r, 0.0) or rmax < R_LCFS_MAX_MIN_M or rmin > R_LCFS_INBOARD_MAX_M:
        return False
    if zmin <= z[0] + pad_z or zmax >= z[-1] - pad_z:
        return False
    if rmin <= r[0] + pad_r or rmax >= r[-1] - pad_r:
        return False
    return True


def plasma_axis(r: np.ndarray, z: np.ndarray, psi: np.ndarray) -> tuple[float, float, float]:
    """Maximum of ψ near R0. Skip the solenoid and an axis that has walked outboard."""
    zz, rr = np.meshgrid(z, r, indexing="ij")
    box = (rr >= R_AXIS_MIN_M) & (rr <= R_AXIS_MAX_M) & (np.abs(zz) <= Z_AXIS_MAX_M)
    work = np.where(box, psi, -np.inf)
    if not np.isfinite(work).any() or np.isneginf(work).all():
        return R0_M, Z0_M, float(np.nanmax(psi))
    iz, ir = np.unravel_index(int(np.argmax(work)), work.shape)
    return float(r[ir]), float(z[iz]), float(psi[iz, ir])


def axis_ok(axis_r: float, axis_z: float) -> bool:
    return R_AXIS_MIN_M <= axis_r <= R_AXIS_MAX_M and abs(axis_z) <= Z_AXIS_MAX_M


_EXTRACT_MASKS: dict = {}


def _extract_lcfs(psi: np.ndarray, r: np.ndarray, z: np.ndarray, machine: str = "MAST", axis_sign: float = 1.0):
    """Scorer last-closed. machine selects the envelope and the axis sign."""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    scoring = root / "fusion_scoring"
    if str(scoring) not in sys.path:
        sys.path.insert(0, str(scoring))
    from lcfs import extract_lcfs

    if machine not in _EXTRACT_MASKS:
        fname = "mast_envelope.npz" if machine == "MAST" else "d3d_envelope.npz"
        envelope = np.load(scoring / "masks" / fname)
        _EXTRACT_MASKS[machine] = np.asarray(envelope["mask_coarse"], dtype=bool)
    mask = _EXTRACT_MASKS[machine]
    return extract_lcfs(psi, r, z, machine, mask, mask, axis_sign=float(axis_sign))


def thomson_pressure_lcfs(row: dict, t: float, r_in: float, r_out: float) -> float:
    """⟨p⟩ = mean of 2 n_e T_e e on aligned midplane chords with R in [r_in, r_out].

    Core R is 130 channels; Te/ne are 131. Use min(Te, ne, R) so each sample
    has an R. The all-chord median includes scrape-off and sits low of EFIT.
    """
    pressures: list[float] = []
    fallback: list[float] = []
    blocks = (
        ("thomson_core", "thomson_core_R"),
        ("thomson_edge", "thomson_edge_spatial"),
    )
    for prefix, r_key in blocks:
        t_key = f"{prefix}_times"
        if t_key not in row or f"{prefix}_Te" not in row:
            continue
        times = np.asarray(row[t_key], dtype=np.float64).reshape(-1)
        te = np.asarray(row[f"{prefix}_Te"], dtype=np.float64)
        ne = np.asarray(row[f"{prefix}_ne"], dtype=np.float64)
        if times.size == 0 or te.size == 0:
            continue
        te_t = interpolate(times, te, np.array([t]))[0]
        ne_t = interpolate(times, ne, np.array([t]))[0]
        rch = np.asarray(row.get(r_key, []), dtype=np.float64).reshape(-1)
        n = min(te_t.size, ne_t.size, rch.size) if rch.size else min(te_t.size, ne_t.size)
        for j in range(n):
            if not (np.isfinite(te_t[j]) and np.isfinite(ne_t[j]) and te_t[j] > 0 and ne_t[j] > 0):
                continue
            p = 2.0 * float(ne_t[j]) * float(te_t[j]) * E_CHARGE
            fallback.append(p)
            if rch.size >= n:
                rq = float(rch[j])
                if np.isfinite(rq) and r_in <= rq <= r_out:
                    pressures.append(p)
            else:
                pressures.append(p)
    if pressures:
        return float(np.mean(pressures))
    if fallback:
        return float(np.mean(fallback))
    return float("nan")


def smooth_betan(beta: np.ndarray, *, inplace: bool = False) -> np.ndarray:
    """Median over BETAN_TIME_MEDIAN efit frames. Kills a single laser peak.

    inplace=False reads the original samples. inplace=True reads the median
    already written, which is the DIII-D window.
    """
    y = np.asarray(beta, dtype=np.float64).ravel()
    out = y.copy()
    h = BETAN_TIME_MEDIAN // 2
    src = out if inplace else y
    for i in range(y.size):
        sl = src[max(0, i - h) : i + h + 1]
        sl = sl[np.isfinite(sl)]
        if sl.size:
            out[i] = float(np.median(sl))
    return out


def thomson_window_median(row: dict, t: float) -> float:
    """Median 2 n_e T_e e on core chords with R in [RT_LO, RT_HI]."""
    pressures: list[float] = []
    if "thomson_core_times" not in row or "thomson_core_Te" not in row:
        return float("nan")
    times = np.asarray(row["thomson_core_times"], dtype=np.float64).reshape(-1)
    te = np.asarray(row["thomson_core_Te"], dtype=np.float64)
    ne = np.asarray(row["thomson_core_ne"], dtype=np.float64)
    if times.size == 0 or te.size == 0:
        return float("nan")
    te_t = interpolate(times, te, np.array([t]))[0]
    ne_t = interpolate(times, ne, np.array([t]))[0]
    rch = np.asarray(row.get("thomson_core_R", []), dtype=np.float64).reshape(-1)
    n = min(te_t.size, ne_t.size)
    for j in range(n):
        if not (np.isfinite(te_t[j]) and np.isfinite(ne_t[j]) and te_t[j] > 0 and ne_t[j] > 0):
            continue
        if rch.size == n and not (RT_LO <= float(rch[j]) <= RT_HI):
            continue
        pressures.append(2.0 * float(ne_t[j]) * float(te_t[j]) * E_CHARGE)
    if not pressures:
        return float("nan")
    return float(np.median(pressures))


def thomson_tangential_mean(row: dict, t: float) -> float:
    """Mean 2 n_e T_e e on the tangential Thomson block.

    On DIII-D that block is the midplane laser. Its spatial column is Z, so the
    radius filter used for the vertical core does not apply. Channel order matches
    the TS_tangential chords.
    """
    if "thomson_edge_times" not in row or "thomson_edge_Te" not in row:
        return float("nan")
    times = np.asarray(row["thomson_edge_times"], dtype=np.float64).reshape(-1)
    te = np.asarray(row["thomson_edge_Te"], dtype=np.float64)
    ne = np.asarray(row["thomson_edge_ne"], dtype=np.float64)
    if times.size == 0 or te.size == 0:
        return float("nan")
    te_t = interpolate(times, te, np.array([t]))[0]
    ne_t = interpolate(times, ne, np.array([t]))[0]
    n = min(te_t.size, ne_t.size)
    pressures: list[float] = []
    for j in range(n):
        if not (np.isfinite(te_t[j]) and np.isfinite(ne_t[j]) and te_t[j] > 0 and ne_t[j] > 0):
            continue
        pressures.append(2.0 * float(ne_t[j]) * float(te_t[j]) * E_CHARGE)
    if not pressures:
        return float("nan")
    return float(np.mean(pressures))


def read_q95_betan(
    r: np.ndarray,
    z: np.ndarray,
    psi: np.ndarray,
    f_b: float,
    b_t: float,
    ip_ma: float,
    row: dict,
    t: float,
    *,
    bleed: float = 0.0,
    walk: float = 0.0,
    axis_r: float | None = None,
    axis_z: float | None = None,
    machine: str = "MAST",
    axis_sign: float = 1.0,
    pressure: str = "lcfs",
    fallback_closed: bool = True,
    factor_fallback: bool = False,
) -> tuple[float, float]:
    """q95 = F_b W / 2π on the 95% extract_lcfs track, then the well / walk readout.

    ×0.55 while q_map is still the OH well (W long). Once Ampère shortens W
    (q_map ≤ 8), ×(1+0.35(1-w)(1-b)). +2 b (1-b)^3 at the divert approach
    (EFIT q95 peaks while W has already collapsed). Off at b=0 so late walk
    is not moved.
    """
    ip = max(abs(float(ip_ma)), 1.0e-3)
    if axis_r is None or axis_z is None:
        axis_r, axis_z, psi_ax = plasma_axis(r, z, psi)
    else:
        iz = int(np.argmin(np.abs(z - axis_z)))
        ir = int(np.argmin(np.abs(r - axis_r)))
        psi_ax = float(psi[iz, ir])
    poly = _extract_lcfs(psi, r, z, machine, axis_sign)
    psi_b = float("nan")
    if poly is not None:
        psi_b = float(np.mean(_bilinear(r, z, psi, poly[:, 0], poly[:, 1])))
    if (poly is None or not np.isfinite(psi_ax) or not np.isfinite(psi_b)) and fallback_closed:
        poly, psi_b, _x = last_closed(r, z, psi, axis_r, axis_z)
    if poly is not None:
        r_in = float(poly[:, 0].min())
        r_out = float(poly[:, 0].max())
        a = max(0.5 * (r_out - r_in), 0.05)
    else:
        r_in, r_out, a = 0.25, 1.45, max(A_TRIAL_M, 0.05)
    if pressure == "lcfs":
        p_mean = thomson_pressure_lcfs(row, t, r_in, r_out)
        if not np.isfinite(p_mean):
            p_mean = thomson_pressure(row, t)
    elif pressure == "tangential":
        p_mean = thomson_tangential_mean(row, t)
        if not np.isfinite(p_mean):
            p_mean = thomson_window_median(row, t)
    else:
        p_mean = thomson_window_median(row, t)
    beta = _beta_n_raw(p_mean, b_t, ip_ma, a)
    if not np.isfinite(beta):
        beta = BETA_FB
    else:
        beta = float(np.clip(beta, BETA_LO, BETA_HI))
    if poly is None or not np.isfinite(psi_ax) or not np.isfinite(psi_b):
        q_star = 5.0 * (A_TRIAL_M**2) * abs(b_t) / (R0_M * ip)
        if not factor_fallback:
            return float(q_star), float(beta)
        q_map = q_star
        well = np.clip(
            (q_map - Q95_DIVERT_MAP) / (Q95_WELL_MAP - Q95_DIVERT_MAP),
            0.0,
            1.0,
        )
        diverted = 1.0 + Q95_WALK_SCALE * (1.0 - float(walk)) * (1.0 - float(bleed))
        hole = Q95_HOLE_SCALE * float(bleed) * (1.0 - float(bleed)) ** Q95_HOLE_POWER
        q95 = (well * Q95_BLEED_SCALE + (1.0 - well) * diverted + hole) * q_map
        return float(q95), float(beta)
    level_95 = psi_ax - TAU_95 * (psi_ax - psi_b)
    poly_95 = _closed_around(r, z, psi, level_95, axis_r, axis_z)
    track = poly_95 if poly_95 is not None else poly
    w = w_on_track(r, z, psi, track)
    q_map = abs(float(f_b)) * w / (2.0 * np.pi)
    if not np.isfinite(q_map) or q_map <= 0:
        q_map = 5.0 * (a**2) * abs(b_t) / (R0_M * ip)
    # Well (W long): ×0.55. Ampère (W short): walk. Gate on q_map, not on b.
    well = np.clip(
        (q_map - Q95_DIVERT_MAP) / (Q95_WELL_MAP - Q95_DIVERT_MAP),
        0.0,
        1.0,
    )
    diverted = 1.0 + Q95_WALK_SCALE * (1.0 - float(walk)) * (1.0 - float(bleed))
    hole = Q95_HOLE_SCALE * float(bleed) * (1.0 - float(bleed)) ** Q95_HOLE_POWER
    q95 = (well * Q95_BLEED_SCALE + (1.0 - well) * diverted + hole) * q_map
    return float(q95), float(beta)


def geometric_tau(r: np.ndarray, z: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """τ = ρ² on the support. Not from ψ."""
    if SUPPORT_TAU is not None:
        return SUPPORT_TAU
    zz, rr = np.meshgrid(z, r, indexing="ij")
    a_r = max(A_TRIAL_M, 1.0e-6)
    a_z = max(A_Z_M, 1.0e-6)
    rho2 = ((rr - R0_M) / a_r) ** 2 + ((zz - Z0_M) / a_z) ** 2
    return np.where(inside, np.clip(rho2, 0.0, 1.0), 2.0)


def scale_j_to_ip(j_phi: np.ndarray, r: np.ndarray, z: np.ndarray, ip_a: float) -> np.ndarray:
    d_r = abs(float(np.mean(np.diff(r))))
    d_z = abs(float(np.mean(np.diff(z))))
    ip_hat = float(np.sum(j_phi) * d_r * d_z)
    if abs(ip_hat) < 1.0 or not np.isfinite(ip_hat) or abs(ip_a) < 1.0:
        return j_phi
    return j_phi * (ip_a / ip_hat)


def solve_frame(
    *,
    r: np.ndarray,
    z: np.ndarray,
    psi_vac: np.ndarray,
    green: np.ndarray,
    b_t: float,
    ip_ma: float,
    q95: float,
    beta_n: float,
) -> FrameResult:
    psi = np.array(psi_vac, dtype=np.float64, copy=True)
    w = 2.0 * np.pi * A_TRIAL_M**2 / (R0_M * 0.05)
    delta_psi = 0.05
    s_vol = 0.5
    axis_r, axis_z = R0_M, Z0_M
    n_iter = 0
    prev_w, prev_delta, prev_s = w, delta_psi, s_vol
    ip_a = ip_ma * 1.0e6
    for n_iter in range(1, MAX_ITERS + 1):
        axis_r, axis_z, psi_ax = plasma_axis(r, z, psi)
        poly, psi_b, _xpt = last_closed(r, z, psi, axis_r, axis_z)
        # Hold the trial disk and a geometric τ so current cannot chase an outboard max.
        inside = trial_disk(r, z)
        a = A_TRIAL_M
        tau = geometric_tau(r, z, inside)
        seated = axis_ok(axis_r, axis_z) if AXIS_GATE_Z else (R_AXIS_MIN_M <= axis_r <= R_AXIS_MAX_M)
        if poly is not None and seated:
            delta_psi = max(abs(psi_ax - psi_b), 1.0e-4)
            level_95 = psi_ax - TAU_95 * (psi_ax - psi_b)
            poly_95 = _closed_around(r, z, psi, level_95, axis_r, axis_z)
            w = w_on_track(r, z, psi, poly_95 if poly_95 is not None else poly)
        else:
            psi_ax = float(np.nanmax(psi[inside])) if inside.any() else float(np.nanmax(psi))
            psi_b = float(np.nanmean(psi[~inside])) if (~inside).any() else psi_ax - 0.05
            delta_psi = max(abs(psi_ax - psi_b), 1.0e-4)
            w = 2.0 * np.pi * a**2 / (R0_M * delta_psi)
        a = max(a, 0.05)
        w = max(w, 1.0e-6)
        zz, rr = np.meshgrid(z, r, indexing="ij")
        s = s_of_tau(np.clip(tau, 0.0, 1.0))
        d_r = abs(float(np.mean(np.diff(r))))
        d_z = abs(float(np.mean(np.diff(z))))
        vol_w = np.where(inside, 2.0 * np.pi * rr * d_r * d_z, 0.0)
        s_vol = float(np.sum(s * vol_w) / max(float(np.sum(vol_w)), 1.0e-30))
        s_vol = max(s_vol, 1.0e-12)
        if PRESSURE_ON_MEASURED and I0_A == 0.0:
            p_mean = (beta_n / 100.0) * (abs(b_t) / (2.0 * MU0)) * (abs(ip_ma) / a)
        else:
            target_a = IP_SCALE * ip_a + I0_A + I_B_A * BLEED
            ip_for_p = abs(ip_ma) if PRESSURE_ON_MEASURED else abs(target_a) / 1.0e6
            p_mean = (beta_n / 100.0) * (abs(b_t) / (2.0 * MU0)) * (ip_for_p / a)
        tau_c = np.clip(tau, 0.0, 1.0)
        p_prime = p_mean * p_prime_of_tau(tau_c) / (delta_psi * s_vol)
        # MAST: FF' = −γ̄ G(τ), G=(1−τ) at c=(0,0). See fec.md slope sign.
        g_tau = (1.0 - tau_c)
        ff_prime = np.where(inside, -GAMMA_BAR * g_tau, 0.0)
        j_phi = rr * p_prime + ff_prime / (MU0 * np.maximum(rr, 1.0e-6))
        j_phi = np.where(inside, j_phi, 0.0)
        if PRESSURE_ON_MEASURED and I0_A == 0.0:
            if ip_ma < 0:
                j_phi = -j_phi
            j_phi = scale_j_to_ip(j_phi, r, z, IP_SCALE * ip_a)
        else:
            target_a = IP_SCALE * ip_a + I0_A + I_B_A * BLEED
            if target_a < 0.0:
                j_phi = -j_phi
            j_phi = scale_j_to_ip(j_phi, r, z, target_a)
        psi_pl = (green @ j_phi.ravel()).reshape(psi.shape)
        psi_new = psi_vac + psi_pl
        dpsi = float(np.nanmax(np.abs(psi_new - psi)))
        rel_w = abs(w - prev_w) / (abs(prev_w) + 1.0e-12)
        rel_d = abs(delta_psi - prev_delta) / (abs(prev_delta) + 1.0e-12)
        rel_s = abs(s_vol - prev_s) / (abs(prev_s) + 1.0e-12)
        psi = psi_new
        prev_w, prev_delta, prev_s = w, delta_psi, s_vol
        # After one ellipse pass the divertor X is on total ψ. Freeze that D.
        if n_iter == 1 and USE_VACUUM_XD and (1.0 - float(BLEED)) > 1.0e-6:
            prepare_support(r, z, psi)
        if dpsi < PSI_ABS_TOL and rel_w < SCALAR_RTOL and rel_d < SCALAR_RTOL and rel_s < SCALAR_RTOL:
            break
    return FrameResult(
        psi=psi,
        q95=float(q95),
        beta_n=float(beta_n),
        n_iter=n_iter,
        axis_r=float(axis_r),
        axis_z=float(axis_z),
        w=float(w),
        delta_psi=float(delta_psi),
    )


def _default_cal() -> Calibration:
    from osf.mast_12_osf_0_0_12 import CAL

    return CAL


def _named_currents(row: dict, t: float) -> tuple[dict[str, float], float, float]:
    """Coil currents in A for a calibration that names its own columns."""
    mt_cols = [c for c in row if str(c).startswith("magnetics_F") or c == SOL_COL]
    currents = {c: _interp_at(row, c, t) * 1.0e3 for c in mt_cols}
    tf_a = _interp_at(row, TF_COL, t) * 1.0e3
    ip_a = _interp_at(row, IP_COL, t, IP_TIME_KEY) * 1.0e3
    return currents, tf_a, ip_a


def _interp_at(row: dict, column: str, t: float, time_key: str = "magnetics_time") -> float:
    times = np.asarray(row[time_key], dtype=np.float64).reshape(-1)
    return float(interpolate(times, row[column], np.array([t]))[0])


def solve_one_frame(row: dict, t_index: int, cal: Calibration | None = None, *, maps=None, green=None) -> dict:
    from osf.osf_0_0_1 import (
        _as1,
        coil_unit_maps,
        frame_currents_a,
        plasma_green,
        vacuum_psi,
    )

    if cal is None:
        cal = _default_cal()
    bind(cal)
    r = np.asarray(row["efit_grid_R"], dtype=np.float64).reshape(-1)
    z = np.asarray(row["efit_grid_Z"], dtype=np.float64).reshape(-1)
    t_dst = _as1(row["efit_times"])
    if maps is None:
        maps = coil_unit_maps(row)
    if green is None:
        green = plasma_green(r, z)
    if LOADER == "mast_columns":
        currents_a, tf_a, ip_a = frame_currents_a(row, t_index, t_dst)
    else:
        currents_a, tf_a, ip_a = _named_currents(row, float(t_dst[t_index]))
    knobs = magnetics_knobs(currents_a, ip_a)
    currents_a = scale_vacuum_currents(currents_a)
    psi_vac = vacuum_psi(maps, currents_a)
    global SUPPORT_INSIDE, SUPPORT_TAU, SUPPORT_INFO
    SUPPORT_INSIDE = None
    SUPPORT_TAU = None
    SUPPORT_INFO = {"alpha": 0.0, "x_top": None, "x_bot": None}
    f_b = MU0 * N_TF_TURNS * abs(tf_a) / (2.0 * np.pi)
    b_t = f_b / R0_M
    ip_ma = ip_a / 1.0e6
    if BETA_GUESS == "osf_0_0_1":
        q_i, b_i = knobs_for_frame(row, float(t_dst[t_index]), b_t, ip_ma)
    else:
        p_med = thomson_window_median(row, float(t_dst[t_index]))
        target_for_beta = IP_SCALE * ip_a + I0_A
        b_i = _beta_n_raw(p_med, b_t, abs(target_for_beta) / 1.0e6, A_TRIAL_M)
        if not np.isfinite(b_i):
            b_i = BETA_FB
        else:
            b_i = float(np.clip(b_i, BETA_LO, BETA_HI))
        q_i = 0.0
    if SIGMA_AX == 1.0:
        phi_vac = psi_vac
    else:
        phi_vac = SIGMA_AX * psi_vac
    out = solve_frame(
        r=r, z=z, psi_vac=phi_vac, green=green, b_t=b_t, ip_ma=ip_ma, q95=q_i, beta_n=b_i
    )
    psi = out.psi if SIGMA_AX == 1.0 else SIGMA_AX * out.psi
    psi = psi + (PSI00 + PSI0B * BLEED)
    poly, psi_b, xpt = last_closed(r, z, out.psi, out.axis_r, out.axis_z)
    if I0_A == 0.0:
        ip_target_a = IP_SCALE * ip_a
    else:
        ip_target_a = IP_SCALE * ip_a + I0_A + I_B_A * BLEED
    return {
        "model_id": CALIBRATION_ID,
        "t": float(t_dst[t_index]),
        "t_index": int(t_index),
        "psi": psi,
        "psi_vac": psi_vac,
        "q95": out.q95,
        "betaN": out.beta_n,
        "axis_r": out.axis_r,
        "axis_z": out.axis_z,
        "n_iter": out.n_iter,
        "w": out.w,
        "delta_psi": out.delta_psi,
        "poly": poly,
        "psi_b": psi_b,
        "x_point": xpt,
        "r": r,
        "z": z,
        "ip_ma": ip_ma,
        "b_t": b_t,
        "f_b": float(f_b),
        "gamma_bar": float(GAMMA_BAR),
        "ip_scale": float(IP_SCALE),
        "p2_scale": float(P2_SCALE),
        "p2l_scale": float(knobs.get("p2l_scale", P2_SCALE)),
        "p4_scale": float(P4_SCALE),
        "p5_scale": float(knobs.get("p5_scale", 1.0)),
        "sol_scale": float(SOL_SCALE),
        "r0": float(R0_M),
        "z0": float(Z0_M),
        "ip_target_a": float(ip_target_a),
        "a_trial": float(A_TRIAL_M),
        "a_z": float(A_Z_M),
        "walk": float(knobs["walk"]),
        "bleed": float(knobs["bleed"]),
        "support": dict(SUPPORT_INFO),
    }


def _stored_axis(r: np.ndarray, z: np.ndarray, psi: np.ndarray, r0: float, z0: float) -> tuple[float, float]:
    """Extremum of the stored map in the axis box. A minimum when σ_ax = −1."""
    axis_r, axis_z = r0, z0
    zz, rr = np.meshgrid(z, r, indexing="ij")
    box = (rr >= R_AXIS_MIN_M) & (rr <= R_AXIS_MAX_M) & (np.abs(zz) <= Z_AXIS_MAX_M)
    if SIGMA_AX < 0.0:
        work = np.where(box, psi, np.inf)
        if np.isfinite(work).any() and not np.isposinf(work).all():
            iz, ir = np.unravel_index(int(np.argmin(work)), work.shape)
            axis_r, axis_z = float(r[ir]), float(z[iz])
    else:
        work = np.where(box, psi, -np.inf)
        if np.isfinite(work).any() and not np.isneginf(work).all():
            iz, ir = np.unravel_index(int(np.argmax(work)), work.shape)
            axis_r, axis_z = float(r[ir]), float(z[iz])
    return axis_r, axis_z


def solve_shot(row: dict, cal: Calibration | None = None, *, green=None, maps=None, cal_at=None) -> dict:
    from osf.osf_0_0_1 import (
        _as1,
        coil_unit_maps,
        plasma_green,
    )

    if cal is None and cal_at is None:
        cal = _default_cal()
    if cal is not None:
        bind(cal)
    r = np.asarray(row["efit_grid_R"], dtype=np.float64).reshape(-1)
    z = np.asarray(row["efit_grid_Z"], dtype=np.float64).reshape(-1)
    t_dst = _as1(row["efit_times"])
    if maps is None:
        maps = coil_unit_maps(row)
    if green is None:
        green = plasma_green(r, z)
    n = int(t_dst.size)
    psirz = np.zeros((n, z.size, r.size), dtype=np.float64)
    q95 = np.zeros(n, dtype=np.float64)
    beta_n = np.zeros(n, dtype=np.float64)
    axis_r = np.zeros(n, dtype=np.float64)
    axis_z = np.zeros(n, dtype=np.float64)
    n_iter = np.zeros(n, dtype=np.int32)
    walk = np.zeros(n, dtype=np.float64)
    for t_index in range(n):
        frame_cal = cal_at(float(t_dst[t_index])) if cal_at is not None else cal
        pack = solve_one_frame(row, t_index, frame_cal, maps=maps, green=green)
        psirz[t_index] = pack["psi"]
        if AXIS_ON_STORED:
            ar, az = _stored_axis(r, z, pack["psi"], float(pack["r0"]), float(pack["z0"]))
        else:
            ar, az = pack["axis_r"], pack["axis_z"]
        ip_read = pack["ip_ma"] if READOUT_IP == "measured" else abs(pack["ip_target_a"]) / 1.0e6
        if FB_FROM_BT:
            f_b = pack["b_t"] * pack.get("r0", R0_M)
        else:
            f_b = pack["f_b"]
        q_r, b_r = read_q95_betan(
            r,
            z,
            pack["psi"],
            f_b,
            pack["b_t"],
            ip_read,
            row,
            float(t_dst[t_index]),
            bleed=float(pack.get("bleed", 0.0)),
            walk=float(pack.get("walk", 0.0)),
            axis_r=ar,
            axis_z=az,
            machine=MACHINE_NAME,
            axis_sign=SIGMA_AX,
            pressure=READOUT_PRESSURE,
            fallback_closed=FALLBACK_CLOSED,
            factor_fallback=FACTOR_FALLBACK,
        )
        q95[t_index] = q_r
        beta_n[t_index] = b_r
        walk[t_index] = float(pack.get("walk", 0.0))
        axis_r[t_index] = ar
        axis_z[t_index] = az
        n_iter[t_index] = pack["n_iter"]
    beta_n = smooth_betan(beta_n, inplace=SMOOTH_INPLACE) * (1.0 - BETAN_WALK_SCALE * walk)
    if S_BETA != 1.0:
        beta_n = beta_n * S_BETA
    if POST_CLIP_BETA:
        beta_n = np.clip(beta_n, BETA_LO, BETA_HI)
    return {
        "model_id": CALIBRATION_ID,
        "psirz": psirz,
        "q95": q95,
        "betaN": beta_n,
        "axis_r": axis_r,
        "axis_z": axis_z,
        "n_iter": n_iter,
        "efit_times": t_dst,
        "efit_grid_R": r,
        "efit_grid_Z": z,
    }


def write_shot(path, pack: dict) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{k: v for k, v in pack.items() if k != "model_id"},
        model_id=np.asarray(pack["model_id"]),
    )
