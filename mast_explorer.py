#!/usr/bin/env python3
"""MAST Equilibrium Explorer — Interactive Calibration & Physics Visualization App.

Runs an interactive web application allowing users to browse the 1,206 unlabelled
MAST test shots, adjust bounded free parameters within literature-enclosing ranges,
and observe real-time Grad–Shafranov flux maps and derived challenge measures.

Usage:
    .venv/bin/python mast_explorer.py [--port 8775] [--host 127.0.0.1]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCORING = ROOT / "fusion_scoring"
if str(SCORING) not in sys.path:
    sys.path.insert(0, str(SCORING))

from src.feq.data import data_root, load_shot, shot_paths
from osf.osf_0_0_1 import coil_unit_maps, plasma_green
from osf.osf_0_0_12 import Calibration, solve_one_frame, read_q95_betan
from osf.mast_12_osf_0_0_12 import CAL
from fusion_scoring.lcfs import extract_lcfs
from fusion_scoring.derive import derive_frame

# Pre-cached Global Operators (initialized on server startup)
CACHED_MAPS = None
CACHED_GREEN = None
CACHED_MASK = None
ALL_SHOT_PATHS = []
SHOT_METADATA_INDEX = {}
LOADED_SHOT_CACHE = {}

# -----------------------------------------------------------------------------
# Parameter Specifications Catalog (Table 1 & Table 2)
# -----------------------------------------------------------------------------

LOCKED_PARAMETERS = [
    {
        "symbol": "μ₀",
        "name": "Permeability of Free Space",
        "value": "4π × 10⁻⁷ H/m",
        "role": "Biot–Savart operator and Ampère law scale",
        "citation": "Fundamental electromagnetic physical constant",
    },
    {
        "symbol": "e",
        "name": "Elementary Charge",
        "value": "1.6022 × 10⁻¹⁹ C",
        "role": "Converts Thomson Te (eV) to kinetic pressure p = 2 ne e Te",
        "citation": "Fundamental physical constant",
    },
    {
        "symbol": "R Grid",
        "name": "Major Radius Computational Domain",
        "value": "[0.06, 2.00] m (65 nodes)",
        "role": "Covers entire interior chamber and center column",
        "citation": "Competition standard grid definition (ΔR = 30.3 mm)",
    },
    {
        "symbol": "Z Grid",
        "name": "Vertical Computational Domain",
        "value": "[-2.00, +2.00] m (65 nodes)",
        "role": "Up-down symmetric domain covering both divertors",
        "citation": "Competition standard grid definition (ΔZ = 62.5 mm)",
    },
    {
        "symbol": "NTF",
        "name": "Toroidal Field Coil Turns",
        "value": "24 turns (12 limbs × 2 turns)",
        "role": "Relates power supply current to toroidal field Fb = μ₀ NTF |ITF| / (2π)",
        "citation": "Sykes et al., Nucl. Fusion 41, 1423 (2001)",
    },
    {
        "symbol": "σax",
        "name": "Magnetic Axis Polarity Sign",
        "value": "+1 (Maximum)",
        "role": "Defines axis as flux maximum (ψaxis > ψboundary)",
        "citation": "Appel et al., Comput. Phys. Commun. 175, 473 (2006) (EFIT++)",
    },
    {
        "symbol": "C",
        "name": "Poloidal Conductor Tiles",
        "value": "812 discrete tiles (3×3 sub-filaments)",
        "role": "Geometric representation of P1–P6 coil bundles",
        "citation": "Sykes et al. (2001); Lloyd et al. (2011)",
    },
    {
        "symbol": "Rcasing",
        "name": "Center Column Armor Radius",
        "value": "0.20 m",
        "role": "Inboard physical mechanical clearance boundary",
        "citation": "Sykes et al., Nucl. Fusion 41, 1423 (2001)",
    },
    {
        "symbol": "Rlimiter",
        "name": "Outboard Limiter Radius",
        "value": "1.45 m",
        "role": "Outboard physical mechanical clearance boundary",
        "citation": "Sykes et al., Nucl. Fusion 41, 1423 (2001)",
    },
    {
        "symbol": "κmax",
        "name": "Elongation Rejection Ceiling",
        "value": "2.60",
        "role": "Rejection gate on pathological VDE contours",
        "citation": "Peng & Strickler, Nucl. Fusion 26, 769 (1986)",
    },
    {
        "symbol": "ΔZmax",
        "name": "Divertor Target Clearance",
        "value": "2.80 m",
        "role": "Mechanical clearance between upper and lower divertors",
        "citation": "Sykes et al., Nucl. Fusion 41, 1423 (2001)",
    },
    {
        "symbol": "γ̄",
        "name": "Profile Internal Current Scale",
        "value": "-1.00",
        "role": "Enforces FF' > 0 so paramagnetic current aligns with Ip",
        "citation": "Freidberg, Ideal MHD (1987)",
    },
]

SEARCHABLE_PARAMS = [
    # Group A: Geometry & Trajectory
    {
        "key": "r00",
        "group": "geometry",
        "group_name": "A. Geometry & Trajectory",
        "symbol": "R₀₀",
        "name": "Baseline Major Radius",
        "min": 0.70,
        "max": 0.95,
        "default": 0.82,
        "step": 0.005,
        "unit": "m",
        "rationale": "Chamber design centerline is 0.85 m; operational axis spans [0.70, 0.95] m (Sykes 2001).",
    },
    {
        "key": "rw",
        "group": "geometry",
        "group_name": "A. Geometry & Trajectory",
        "symbol": "Rw",
        "name": "Shafranov Walk Shift",
        "min": 0.00,
        "max": 0.20,
        "default": 0.10,
        "step": 0.005,
        "unit": "m",
        "rationale": "Low-aspect-ratio equilibrium Shafranov shift ΔR ~ (a²/2R₀)(βp + ℓi/2) (Shafranov 1966).",
    },
    {
        "key": "rb",
        "group": "geometry",
        "group_name": "A. Geometry & Trajectory",
        "symbol": "Rb",
        "name": "Ohmic Inboard Shift",
        "min": 0.00,
        "max": 0.08,
        "default": 0.02,
        "step": 0.002,
        "unit": "m",
        "rationale": "Inboard limiter clearance displacement during early breakdown (Sykes 2001).",
    },
    {
        "key": "z00",
        "group": "geometry",
        "group_name": "A. Geometry & Trajectory",
        "symbol": "Z₀₀",
        "name": "Baseline Vertical Center",
        "min": -0.030,
        "max": 0.030,
        "default": 0.009,
        "step": 0.001,
        "unit": "m",
        "rationale": "Up-down symmetric coil geometry; fast control restricts |ΔZ| ≤ 30 mm to avoid VDEs.",
    },
    {
        "key": "zb",
        "group": "geometry",
        "group_name": "A. Geometry & Trajectory",
        "symbol": "Zb",
        "name": "Ohmic Vertical Shift",
        "min": -0.020,
        "max": 0.020,
        "default": 0.004,
        "step": 0.001,
        "unit": "m",
        "rationale": "Solenoid eddy field vertical axis alignment during startup breakdown.",
    },
    {
        "key": "ar0",
        "group": "geometry",
        "group_name": "A. Geometry & Trajectory",
        "symbol": "aR0",
        "name": "Baseline Minor Radius",
        "min": 0.40,
        "max": 0.65,
        "default": 0.50,
        "step": 0.005,
        "unit": "m",
        "rationale": "MAST aspect ratio A = R/a in [1.3, 1.7] bounded by vessel casing (Akers 2002).",
    },
    {
        "key": "aw",
        "group": "geometry",
        "group_name": "A. Geometry & Trajectory",
        "symbol": "aw",
        "name": "Minor Radius Growth",
        "min": 0.00,
        "max": 0.15,
        "default": 0.08,
        "step": 0.005,
        "unit": "m",
        "rationale": "Expansion from compact limiter filament (a ≈ 0.40 m) to diverted chamber (a ≈ 0.58 m).",
    },
    {
        "key": "a0",
        "group": "geometry",
        "group_name": "A. Geometry & Trajectory",
        "symbol": "a₀",
        "name": "Base Elongation Offset",
        "min": 0.00,
        "max": 0.25,
        "default": 0.08,
        "step": 0.005,
        "unit": "m",
        "rationale": "Natural ST elongation κnat ~ 1 + ε yields vertical excess aZ - aR (Peng & Strickler 1986).",
    },
    {
        "key": "ab",
        "group": "geometry",
        "group_name": "A. Geometry & Trajectory",
        "symbol": "ab",
        "name": "Limiter Elongation Boost",
        "min": 0.10,
        "max": 0.60,
        "default": 0.42,
        "step": 0.01,
        "unit": "m",
        "rationale": "Quadrupole field elongation near central solenoid return flux (Menard 2001).",
    },
    {
        "key": "ad",
        "group": "geometry",
        "group_name": "A. Geometry & Trajectory",
        "symbol": "ad",
        "name": "Diverted Elongation Mod.",
        "min": 0.00,
        "max": 0.40,
        "default": 0.18,
        "step": 0.01,
        "unit": "m",
        "rationale": "Operational double-null flattop elongation limits κ in [1.5, 2.0] (Lloyd 2011).",
    },

    # Group B: Ampère Current
    {
        "key": "k0",
        "group": "ampere",
        "group_name": "B. Ampère Enclosed Current",
        "symbol": "k₀",
        "name": "Current Enclosed Floor",
        "min": 0.40,
        "max": 0.90,
        "default": 0.70,
        "step": 0.02,
        "unit": "× Ip",
        "rationale": "Scrape-off layer (SOL) edge current fraction outside core trial ellipse (Appel 2006).",
    },
    {
        "key": "kw",
        "group": "ampere",
        "group_name": "B. Ampère Enclosed Current",
        "symbol": "kw",
        "name": "Walk Restore Fraction",
        "min": 0.10,
        "max": 0.50,
        "default": 0.30,
        "step": 0.02,
        "unit": "× Ip",
        "rationale": "Current profile broadening recovery as plasma detaches into flattop (Akers 2002).",
    },
    {
        "key": "kb",
        "group": "ampere",
        "group_name": "B. Ampère Enclosed Current",
        "symbol": "kb",
        "name": "Limiter Enclosed Fraction",
        "min": 0.00,
        "max": 0.35,
        "default": 0.15,
        "step": 0.01,
        "unit": "× Ip",
        "rationale": "Enforces (k₀ + kb) in [0.75, 0.95] Ip while solenoid ohmic well is engaged.",
    },
    {
        "key": "kd",
        "group": "ampere",
        "group_name": "B. Ampère Enclosed Current",
        "symbol": "kd",
        "name": "Diverted Enclosed Fraction",
        "min": 0.05,
        "max": 0.40,
        "default": 0.20,
        "step": 0.01,
        "unit": "× Ip",
        "rationale": "Enforces (k₀ + kd) in [0.80, 0.95] Ip after limiter detachment.",
    },

    # Group C: Vacuum Scales
    {
        "key": "lam_v0",
        "group": "vacuum",
        "group_name": "C. Poloidal Vacuum Scales",
        "symbol": "λV0",
        "name": "Vertical Field Scale (P2)",
        "min": 1.00,
        "max": 2.40,
        "default": 1.80,
        "step": 0.05,
        "unit": "scale",
        "rationale": "Shafranov vertical holding field Bz and vessel screening compensation (Shafranov 1971).",
    },
    {
        "key": "lam_vb",
        "group": "vacuum",
        "group_name": "C. Poloidal Vacuum Scales",
        "symbol": "λVb",
        "name": "Ohmic Vertical Blend",
        "min": -1.50,
        "max": -0.40,
        "default": -1.00,
        "step": 0.05,
        "unit": "scale",
        "rationale": "Current ramp holding field ratio: λV(1) = 0.80 vs λV(0) = 1.80 (Shafranov 1966).",
    },
    {
        "key": "lam_l",
        "group": "vacuum",
        "group_name": "C. Poloidal Vacuum Scales",
        "symbol": "λL",
        "name": "Lower Asymmetry (P2L)",
        "min": 0.00,
        "max": 0.30,
        "default": 0.105,
        "step": 0.005,
        "unit": "scale",
        "rationale": "Lower strike-point divertor plate biasing for exhaust management (Lloyd 2011).",
    },
    {
        "key": "lam_delta",
        "group": "vacuum",
        "group_name": "C. Poloidal Vacuum Scales",
        "symbol": "λδ",
        "name": "Triangularity Scale (P4)",
        "min": 0.90,
        "max": 1.40,
        "default": 1.15,
        "step": 0.02,
        "unit": "scale",
        "rationale": "Vessel wall eddy screening attenuation compensation on P4 fields (Cunningham 2003).",
    },
    {
        "key": "lam_o",
        "group": "vacuum",
        "group_name": "C. Poloidal Vacuum Scales",
        "symbol": "λo",
        "name": "Outboard Scale (P5)",
        "min": -0.35,
        "max": 0.10,
        "default": -0.18,
        "step": 0.02,
        "unit": "scale",
        "rationale": "Passive vessel screening of large outboard shaping currents (Appel 2006).",
    },
    {
        "key": "lam_s0",
        "group": "vacuum",
        "group_name": "C. Poloidal Vacuum Scales",
        "symbol": "λS0",
        "name": "Diverted Solenoid Scale (P1)",
        "min": 0.25,
        "max": 0.80,
        "default": 0.50,
        "step": 0.05,
        "unit": "scale",
        "rationale": "Center-column casing screening of central solenoid core flux well (Sykes 2001).",
    },
    {
        "key": "lam_sb",
        "group": "vacuum",
        "group_name": "C. Poloidal Vacuum Scales",
        "symbol": "λSb",
        "name": "Ohmic Solenoid Blend",
        "min": -0.10,
        "max": 0.35,
        "default": 0.10,
        "step": 0.02,
        "unit": "scale",
        "rationale": "Dynamic coupling during active central solenoid flux swing.",
    },

    # Group D: Clocks & Thresholds
    {
        "key": "r_star",
        "group": "clocks",
        "group_name": "D. Clocks & Thresholds",
        "symbol": "r★",
        "name": "Walk Ratio Threshold",
        "min": 0.15,
        "max": 0.50,
        "default": 0.34,
        "step": 0.01,
        "unit": "|P2|/|Ip|",
        "rationale": "Ratio threshold defining active divertor X-point formation detachment (Appel 2006).",
    },
    {
        "key": "i_star_a",
        "group": "clocks",
        "group_name": "D. Clocks & Thresholds",
        "symbol": "I★",
        "name": "Solenoid Anchor Current",
        "min": 10000.0,
        "max": 25000.0,
        "default": 16300.0,
        "step": 500.0,
        "unit": "A",
        "rationale": "MAST central solenoid power supply pre-charge operating rating (Sykes 2001).",
    },

    # Group E: Profiles & Scales
    {
        "key": "q95_boost",
        "group": "profiles",
        "group_name": "E. Profile Readouts & Boost",
        "symbol": "q₉₅-boost",
        "name": "Paramagnetic Pitch Factor",
        "min": 1.00,
        "max": 1.30,
        "default": 1.0622,
        "step": 0.005,
        "unit": "scale",
        "rationale": "ST internal toroidal field enhancement F/F₀ ~ √(1 + 2ε²(1-βp)) (Peng 1986, Sykes 1997).",
    },
    {
        "key": "s_beta",
        "group": "profiles",
        "group_name": "E. Profile Readouts & Boost",
        "symbol": "sβ",
        "name": "Beta-N Calibration Scale",
        "min": 0.80,
        "max": 1.30,
        "default": 1.0785,
        "step": 0.01,
        "unit": "scale",
        "rationale": "Fast-ion beam pressure fraction scaling on kinetic Thomson chords (Akers 2002).",
    },
    {
        "key": "alpha_q",
        "group": "profiles",
        "group_name": "E. Profile Readouts & Boost",
        "symbol": "αq",
        "name": "Startup Well Scale",
        "min": 0.35,
        "max": 0.85,
        "default": 0.55,
        "step": 0.02,
        "unit": "scale",
        "rationale": "Ohmic solenoid core curvature correction during early startup breakdown.",
    },
    {
        "key": "eta_q",
        "group": "profiles",
        "group_name": "E. Profile Readouts & Boost",
        "symbol": "ηq",
        "name": "Diverted Walk Factor",
        "min": 0.10,
        "max": 0.60,
        "default": 0.35,
        "step": 0.02,
        "unit": "scale",
        "rationale": "Magnetic pitch expansion rate following center column detachment.",
    },
    {
        "key": "zeta_q",
        "group": "profiles",
        "group_name": "E. Profile Readouts & Boost",
        "symbol": "ζq",
        "name": "Divertor Hole Amplitude",
        "min": 0.50,
        "max": 3.50,
        "default": 2.00,
        "step": 0.05,
        "unit": "amp",
        "rationale": "Transient safety factor elevation bump during divertor formation.",
    },
    {
        "key": "p_q",
        "group": "profiles",
        "group_name": "E. Profile Readouts & Boost",
        "symbol": "pq",
        "name": "Divertor Hole Exponent",
        "min": 1.50,
        "max": 4.50,
        "default": 3.00,
        "step": 0.10,
        "unit": "exp",
        "rationale": "Calculus of transition peaking: b* = 1/(1+p) matches detachment timing.",
    },
    {
        "key": "eta_beta",
        "group": "profiles",
        "group_name": "E. Profile Readouts & Boost",
        "symbol": "ηβ",
        "name": "Thomson Walk Taper",
        "min": 0.05,
        "max": 0.40,
        "default": 0.20,
        "step": 0.02,
        "unit": "scale",
        "rationale": "Fixed midplane laser chord geometric offset as magnetic axis shifts outward.",
    },
    {
        "key": "beta_fb",
        "group": "profiles",
        "group_name": "E. Profile Readouts & Boost",
        "symbol": "βfb",
        "name": "Empty Thomson Fallback",
        "min": 0.80,
        "max": 2.00,
        "default": 1.30,
        "step": 0.05,
        "unit": "betaN",
        "rationale": "Operational database median beta across NBI-heated MAST discharges (Akers 2002).",
    },
    {
        "key": "q_hi",
        "group": "profiles",
        "group_name": "E. Profile Readouts & Boost",
        "symbol": "qhi",
        "name": "Safety Factor Upper Gate",
        "min": 12.0,
        "max": 24.0,
        "default": 17.0,
        "step": 0.5,
        "unit": "q",
        "rationale": "Low-Bt spherical tokamak startup safety factor excursion (Peng 1986).",
    },
    {
        "key": "q_lo",
        "group": "profiles",
        "group_name": "E. Profile Readouts & Boost",
        "symbol": "qlo",
        "name": "Safety Factor Lower Gate",
        "min": 5.0,
        "max": 12.0,
        "default": 8.0,
        "step": 0.2,
        "unit": "q",
        "rationale": "Divertor detachment boundary where edge magnetic shear diverges (Appel 2006).",
    },
]

# -----------------------------------------------------------------------------
# Operator & Data Pre-Caching
# -----------------------------------------------------------------------------

def init_environment():
    global CACHED_MAPS, CACHED_GREEN, CACHED_MASK, ALL_SHOT_PATHS, SHOT_METADATA_INDEX
    print("[INIT] Scanning MAST datasets...", flush=True)

    # Gather unlabelled test shots
    test_paths = shot_paths("mast_public_test")
    
    # Gather demo shots if available
    demo_files = [
        ROOT / "parquet_data" / "mast_shot_28348.parquet",
        ROOT / "parquet_data" / "mast_shot_28350.parquet",
        ROOT / "parquet_data" / "mast_shot_28351.parquet",
    ]
    demo_paths = [p for p in demo_files if p.is_file()]

    ALL_SHOT_PATHS = demo_paths + test_paths
    print(f"[INIT] Indexed {len(test_paths)} unlabelled test shots + {len(demo_paths)} demo shots.", flush=True)

    # Load sample shot to compute Green's operators once
    sample_path = ALL_SHOT_PATHS[0]
    sample_row = load_shot(sample_path)
    r = np.asarray(sample_row["efit_grid_R"], dtype=float)
    z = np.asarray(sample_row["efit_grid_Z"], dtype=float)

    print("[INIT] Pre-computing static Green's response operators (plasma & coil)...", flush=True)
    t0 = time.time()
    CACHED_MAPS = coil_unit_maps(sample_row)
    CACHED_GREEN = plasma_green(r, z)
    envelope = np.load(SCORING / "masks" / "mast_envelope.npz")
    CACHED_MASK = envelope["mask_coarse"]
    dt = (time.time() - t0) * 1000
    print(f"[INIT] Operators cached in {dt:.1f} ms. Single-frame solve time will be ~60 ms.", flush=True)

    # Build quick metadata index
    for p in ALL_SHOT_PATHS:
        SHOT_METADATA_INDEX[p.stem] = {
            "stem": p.stem,
            "path": str(p),
            "is_demo": "28348" in p.stem or "28350" in p.stem or "28351" in p.stem,
        }


def get_shot_row(stem: str) -> dict:
    if stem not in SHOT_METADATA_INDEX:
        raise KeyError(f"Unknown shot {stem}")
    if stem in LOADED_SHOT_CACHE:
        return LOADED_SHOT_CACHE[stem]
    path = Path(SHOT_METADATA_INDEX[stem]["path"])
    row = load_shot(path)
    if len(LOADED_SHOT_CACHE) > 16:
        LOADED_SHOT_CACHE.pop(next(iter(LOADED_SHOT_CACHE)))
    LOADED_SHOT_CACHE[stem] = row
    return row

# -----------------------------------------------------------------------------
# Equilibrium Computation Engine
# -----------------------------------------------------------------------------

def solve_with_overrides(row: dict, t_index: int, overrides: dict) -> dict:
    global CACHED_MAPS, CACHED_GREEN, CACHED_MASK
    t0 = time.time()

    # Filter overrides for dataclass fields
    cal_fields = {f.name for f in dataclasses.fields(CAL)}
    cal_args = {k: float(v) for k, v in overrides.items() if k in cal_fields}

    # Special handling for q95_boost
    q95_boost = float(overrides.get("q95_boost", 1.0622))

    # Construct custom calibration
    custom_cal = dataclasses.replace(CAL, **cal_args)

    r = np.asarray(row["efit_grid_R"], dtype=float)
    z = np.asarray(row["efit_grid_Z"], dtype=float)

    # Solve frame with cached Green operators
    pack = solve_one_frame(row, t_index, custom_cal, maps=CACHED_MAPS, green=CACHED_GREEN)

    # Read flux-surface q95 and betaN
    f_b = pack["b_t"] * pack.get("r0", custom_cal.r00)
    q_r, b_r = read_q95_betan(
        r,
        z,
        pack["psi"],
        f_b,
        pack["b_t"],
        pack["ip_ma"],
        row,
        float(row["efit_times"][t_index]),
        bleed=float(pack.get("bleed", 0.0)),
        walk=float(pack.get("walk", 0.0)),
        axis_r=pack["axis_r"],
        axis_z=pack["axis_z"],
        machine="MAST",
        axis_sign=1.0,
        pressure="lcfs",
    )

    # Apply paramagnetic boost and beta scale
    final_q95 = float(q_r * q95_boost)
    final_beta = float(b_r * custom_cal.s_beta)

    # Extract LCFS boundary contour and consistency shape functionals
    contour = extract_lcfs(pack["psi"], r, z, "MAST", CACHED_MASK, CACHED_MASK, axis_sign=1.0)
    scalars = derive_frame(pack["psi"], r, z, "MAST", CACHED_MASK, CACHED_MASK, contour=contour, axis_sign=1.0)

    solve_time_ms = (time.time() - t0) * 1000

    # Clean contour for JSON
    contour_coords = []
    if contour is not None and hasattr(contour, "__len__") and len(contour) > 0:
        contour_coords = [[round(float(pt[0]), 4), round(float(pt[1]), 4)] for pt in contour]

    return {
        "solve_ms": round(solve_time_ms, 1),
        "t": float(row["efit_times"][t_index]),
        "t_index": int(t_index),
        "n_iter": int(pack.get("n_iter", 0)),
        "q95": round(final_q95, 4),
        "betaN": round(final_beta, 4),
        "delta_psi": round(float(pack.get("delta_psi", 0.0)), 5),
        "axis_r": round(float(pack.get("axis_r", 0.0)), 4),
        "axis_z": round(float(pack.get("axis_z", 0.0)), 4),
        "ip_ma": round(float(pack.get("ip_ma", 0.0)), 4),
        "b_t": round(float(pack.get("b_t", 0.0)), 4),
        "walk": round(float(pack.get("walk", 0.0)), 4),
        "bleed": round(float(pack.get("bleed", 0.0)), 4),
        "r": r.tolist(),
        "z": z.tolist(),
        "psi": pack["psi"].tolist(),
        "contour": contour_coords,
        "scalars": {
            "R_axis": round(float(scalars.get("R_axis", float("nan"))), 4),
            "Z_axis": round(float(scalars.get("Z_axis", float("nan"))), 4),
            "kappa": round(float(scalars.get("kappa", float("nan"))), 4),
            "tri_top": round(float(scalars.get("tri_top", float("nan"))), 4),
            "tri_bot": round(float(scalars.get("tri_bot", float("nan"))), 4),
            "volume": round(float(scalars.get("volume", float("nan"))), 4),
            "li": round(float(scalars.get("li", float("nan"))), 4),
        },
    }

# -----------------------------------------------------------------------------
# HTTP Request Handler & REST API
# -----------------------------------------------------------------------------

class ExplorerHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Concise logging
        print(f"[{self.address_string()}] {fmt % args}")

    def send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, allow_nan=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, html: str, status: int = 200):
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/":
            return self.send_html(render_app_html())

        if path == "/api/shots":
            shots_list = []
            for p in ALL_SHOT_PATHS:
                stem = p.stem
                is_demo = "28348" in stem or "28350" in stem or "28351" in stem
                shots_list.append({
                    "stem": stem,
                    "is_demo": is_demo,
                    "label": f"{'[DEMO LABELED] ' if is_demo else ''}{stem}",
                })
            return self.send_json({"shots": shots_list, "total": len(shots_list)})

        if path == "/api/specs":
            return self.send_json({
                "locked": LOCKED_PARAMETERS,
                "searchable": SEARCHABLE_PARAMS,
            })

        if path.startswith("/api/shot/"):
            stem = path.split("/")[3]
            try:
                row = get_shot_row(stem)
                times = [float(t) for t in row["efit_times"]]
                ip = [float(val) for val in row["magnetics_plasma_current"]]
                return self.send_json({
                    "stem": stem,
                    "is_demo": "28348" in stem or "28350" in stem or "28351" in stem,
                    "n_frames": len(times),
                    "times": times,
                    "t_min": round(min(times), 1) if times else 0,
                    "t_max": round(max(times), 1) if times else 0,
                    "ip_max_ka": round(max([abs(x) for x in ip]) / 1000.0, 1) if ip else 0,
                })
            except Exception as e:
                return self.send_json({"error": str(e)}, 400)

        self.send_error(404, "Endpoint not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/api/solve":
            try:
                content_len = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(content_len).decode("utf-8"))
                stem = body.get("shot")
                t_index = int(body.get("t_index", 0))
                overrides = body.get("params", {})

                row = get_shot_row(stem)
                res = solve_with_overrides(row, t_index, overrides)
                return self.send_json(res)
            except Exception as e:
                import traceback
                traceback.print_exc()
                return self.send_json({"error": str(e)}, 500)

        self.send_error(404, "Endpoint not found")

# -----------------------------------------------------------------------------
# Embedded Single-Page Application (HTML/CSS/JS)
# -----------------------------------------------------------------------------

def render_app_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>MAST Equilibrium Explorer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {
  --bg: #0b0f19;
  --surface: #131b2e;
  --surface-border: #1e2942;
  --surface-hover: #1c2740;
  --primary: #38bdf8;
  --primary-hover: #0284c7;
  --accent: #f59e0b;
  --success: #10b981;
  --danger: #ef4444;
  --text: #f1f5f9;
  --text-muted: #94a3b8;
  --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font);
  font-size: 14px;
  line-height: 1.5;
  overflow-x: hidden;
}

header {
  background: var(--surface);
  border-bottom: 1px solid var(--surface-border);
  padding: 12px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 16px;
}

.brand {
  display: flex;
  align-items: center;
  gap: 12px;
}
.brand-badge {
  background: rgba(56, 189, 248, 0.15);
  color: var(--primary);
  border: 1px solid rgba(56, 189, 248, 0.3);
  padding: 2px 8px;
  border-radius: 4px;
  font-weight: 700;
  font-size: 11px;
  letter-spacing: 0.5px;
}
h1 { font-size: 18px; font-weight: 700; color: #fff; }
.subtitle { font-size: 12px; color: var(--text-muted); }

.top-actions {
  display: flex;
  align-items: center;
  gap: 10px;
}
.btn {
  background: var(--surface);
  color: var(--text);
  border: 1px solid var(--surface-border);
  padding: 6px 14px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 12px;
  font-weight: 600;
  transition: all 0.15s ease;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.btn:hover { background: var(--surface-hover); border-color: var(--primary); color: #fff; }
.btn-primary { background: var(--primary); color: #000; border: none; font-weight: 700; }
.btn-primary:hover { background: var(--primary-hover); color: #fff; }

.shot-bar {
  background: #0f1626;
  border-bottom: 1px solid var(--surface-border);
  padding: 10px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 16px;
}
.shot-select-wrap {
  display: flex;
  align-items: center;
  gap: 10px;
}
select, input[type="text"] {
  background: var(--surface);
  color: #fff;
  border: 1px solid var(--surface-border);
  padding: 6px 12px;
  border-radius: 6px;
  font-size: 13px;
  outline: none;
}
select:focus, input[type="text"]:focus { border-color: var(--primary); }

.time-scrub-wrap {
  display: flex;
  align-items: center;
  gap: 12px;
  flex: 1;
  max-width: 650px;
}
.time-scrub {
  flex: 1;
  accent-color: var(--primary);
  cursor: pointer;
}
.clock-display {
  font-family: var(--mono);
  font-size: 13px;
  color: var(--accent);
  background: rgba(245, 158, 11, 0.1);
  padding: 3px 8px;
  border-radius: 4px;
  white-space: nowrap;
}

.main-layout {
  display: grid;
  grid-template-columns: 58% 42%;
  height: calc(100vh - 110px);
}
@media (max-width: 1200px) {
  .main-layout { grid-template-columns: 1fr; height: auto; }
}

.left-panel {
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  border-right: 1px solid var(--surface-border);
  overflow-y: auto;
}
.right-panel {
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  overflow-y: auto;
  background: #0e1424;
}

.card {
  background: var(--surface);
  border: 1px solid var(--surface-border);
  border-radius: 8px;
  padding: 16px;
}
.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
  padding-bottom: 8px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}
.card-title {
  font-size: 13px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--primary);
}

#flux-plot {
  width: 100%;
  height: 520px;
  border-radius: 6px;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 10px;
}
.metric-tile {
  background: rgba(11, 15, 25, 0.6);
  border: 1px solid var(--surface-border);
  border-radius: 6px;
  padding: 10px 12px;
}
.metric-label { font-size: 11px; color: var(--text-muted); margin-bottom: 4px; }
.metric-val { font-family: var(--mono); font-size: 16px; font-weight: 700; color: #fff; }
.metric-delta { font-size: 11px; font-family: var(--mono); margin-left: 6px; }
.delta-pos { color: var(--success); }
.delta-neg { color: var(--danger); }
.delta-zero { color: var(--text-muted); }

.consistency-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8px;
  margin-top: 10px;
}
.cons-tile {
  background: #162035;
  border: 1px solid var(--surface-border);
  border-radius: 6px;
  padding: 8px 10px;
}
.cons-name { font-size: 11px; color: #94a3b8; }
.cons-num { font-family: var(--mono); font-size: 13px; font-weight: 700; color: #38bdf8; }

.param-tabs {
  display: flex;
  gap: 4px;
  background: var(--bg);
  padding: 4px;
  border-radius: 6px;
  margin-bottom: 12px;
}
.param-tab {
  flex: 1;
  text-align: center;
  padding: 6px 8px;
  font-size: 11px;
  font-weight: 600;
  border-radius: 4px;
  cursor: pointer;
  color: var(--text-muted);
  transition: all 0.15s ease;
}
.param-tab.active { background: var(--primary); color: #000; font-weight: 700; }

.slider-group {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.slider-row {
  background: rgba(11, 15, 25, 0.4);
  border: 1px solid rgba(255, 255, 255, 0.05);
  border-radius: 6px;
  padding: 8px 12px;
}
.slider-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 6px;
}
.slider-title {
  font-size: 12px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 6px;
}
.slider-symbol {
  font-family: var(--mono);
  color: var(--primary);
  background: rgba(56, 189, 248, 0.1);
  padding: 1px 6px;
  border-radius: 4px;
}
.slider-range-badge {
  font-size: 10px;
  font-family: var(--mono);
  color: var(--text-muted);
}
.slider-val-box {
  font-family: var(--mono);
  font-size: 13px;
  font-weight: 700;
  color: var(--accent);
}
.slider-input-line {
  display: flex;
  align-items: center;
  gap: 10px;
}
.param-slider {
  flex: 1;
  accent-color: var(--primary);
  cursor: pointer;
}
.slider-rationale {
  font-size: 10px;
  color: #64748b;
  margin-top: 4px;
}

/* Modal */
.modal-overlay {
  display: none;
  position: fixed;
  top: 0; left: 0; width: 100vw; height: 100vh;
  background: rgba(0, 0, 0, 0.75);
  backdrop-filter: blur(4px);
  z-index: 9999;
  justify-content: center;
  align-items: center;
}
.modal-card {
  background: var(--surface);
  border: 1px solid var(--surface-border);
  border-radius: 12px;
  max-width: 900px;
  width: 90%;
  max-height: 85vh;
  display: flex;
  flex-direction: column;
}
.modal-head {
  padding: 16px 20px;
  border-bottom: 1px solid var(--surface-border);
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.modal-body {
  padding: 20px;
  overflow-y: auto;
}
.table-custom {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}
.table-custom th, .table-custom td {
  padding: 8px 12px;
  border-bottom: 1px solid var(--surface-border);
  text-align: left;
}
.table-custom th { color: var(--primary); font-weight: 700; }
</style>
</head>
<body>

<header>
  <div class="brand">
    <span class="brand-badge">TRACK 2 ZERO-SHOT</span>
    <div>
      <h1>MAST Equilibrium Explorer</h1>
      <div class="subtitle">One-Shot Forward Physics Model (<code>osf_0.0.12</code>) Calibration Playground</div>
    </div>
  </div>
  <div class="top-actions">
    <button class="btn" id="btn-show-locked">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>
      Locked Physics Specs (Table 1)
    </button>
    <select id="preset-selector">
      <option value="baseline">Preset: Current Baseline (0.9.1 / mast_12)</option>
      <option value="optimal">Preset: Optimal Vertex (cq=1.0622, sβ=1.0785)</option>
      <option value="unboosted">Preset: Unboosted Paramagnetic (cq=1.00, sβ=1.00)</option>
      <option value="ohmic">Preset: Ohmic Limit Boundary (b=1, w=0)</option>
      <option value="flattop">Preset: Flattop Double-Null (b=0, w=1)</option>
    </select>
    <button class="btn" id="btn-reset-all">Reset All Sliders</button>
  </div>
</header>

<div class="shot-bar">
  <div class="shot-select-wrap">
    <label style="font-weight:600; color:var(--text-muted); font-size:12px;">SHOT:</label>
    <select id="shot-dropdown" style="min-width: 280px;"></select>
    <span id="shot-info-pill" style="font-size:12px; color:#38bdf8; font-family:var(--mono);"></span>
  </div>

  <div class="time-scrub-wrap">
    <button class="btn" id="btn-prev-frame" title="Previous frame">◀</button>
    <button class="btn" id="btn-play">Play</button>
    <button class="btn" id="btn-next-frame" title="Next frame">▶</button>
    <input type="range" class="time-scrub" id="frame-scrubber" min="0" max="0" value="0"/>
    <div class="clock-display" id="clock-readout">0.0 ms (Frame 0)</div>
  </div>
</div>

<div class="main-layout">
  <!-- LEFT: Flux Map & Solver Diag -->
  <div class="left-panel">
    <div class="card" style="padding:12px;">
      <div class="card-header">
        <span class="card-title">Poloidal Flux Map ψ(R, Z) & Extracted LCFS</span>
        <span style="font-size:11px; font-family:var(--mono); color:var(--text-muted);" id="solve-latency">Latency: — ms</span>
      </div>
      <div id="flux-plot"></div>
    </div>

    <div class="card">
      <div class="card-header">
        <span class="card-title">Picard Solver Convergence & State</span>
      </div>
      <div class="metrics-grid">
        <div class="metric-tile">
          <div class="metric-label">Iterations Nit</div>
          <div class="metric-val" id="diag-iter">—</div>
        </div>
        <div class="metric-tile">
          <div class="metric-label">Walk Clock w</div>
          <div class="metric-val" id="diag-walk">—</div>
        </div>
        <div class="metric-tile">
          <div class="metric-label">Bleed Clock b</div>
          <div class="metric-val" id="diag-bleed">—</div>
        </div>
        <div class="metric-tile">
          <div class="metric-label">Ip / Bt Axis</div>
          <div class="metric-val" id="diag-currents" style="font-size:13px;">—</div>
        </div>
      </div>
    </div>
  </div>

  <!-- RIGHT: Derived Measures & Sliders -->
  <div class="right-panel">
    <!-- Challenge Measures -->
    <div class="card">
      <div class="card-header">
        <span class="card-title">Derived Challenge Measures</span>
        <span style="font-size:11px; color:#10b981;">Live vs Baseline Delta</span>
      </div>
      <div class="metrics-grid">
        <div class="metric-tile">
          <div class="metric-label">Safety Factor q₉₅</div>
          <div class="metric-val" id="val-q95">—</div>
        </div>
        <div class="metric-tile">
          <div class="metric-label">Normalized Beta βN</div>
          <div class="metric-val" id="val-betaN">—</div>
        </div>
        <div class="metric-tile">
          <div class="metric-label">Flux Depth Δψ</div>
          <div class="metric-val" id="val-deltapsi">—</div>
        </div>
        <div class="metric-tile">
          <div class="metric-label">Axis (R, Z)</div>
          <div class="metric-val" id="val-axis" style="font-size:13px;">—</div>
        </div>
      </div>

      <div style="font-size:11px; font-weight:700; color:var(--text-muted); margin-top:14px; text-transform:uppercase;">
        7 Consistency Shape Functionals (Competition Scoring)
      </div>
      <div class="consistency-grid">
        <div class="cons-tile">
          <div class="cons-name">R_axis (m)</div>
          <div class="cons-num" id="cons-raxis">—</div>
        </div>
        <div class="cons-tile">
          <div class="cons-name">Z_axis (m)</div>
          <div class="cons-num" id="cons-zaxis">—</div>
        </div>
        <div class="cons-tile">
          <div class="cons-name">Elongation κ</div>
          <div class="cons-num" id="cons-kappa">—</div>
        </div>
        <div class="cons-tile">
          <div class="cons-name">Volume (m³)</div>
          <div class="cons-num" id="cons-vol">—</div>
        </div>
        <div class="cons-tile">
          <div class="cons-name">Upper Tri δtop</div>
          <div class="cons-num" id="cons-tritop">—</div>
        </div>
        <div class="cons-tile">
          <div class="cons-name">Lower Tri δbot</div>
          <div class="cons-num" id="cons-tribot">—</div>
        </div>
        <div class="cons-tile">
          <div class="cons-name">Inductance ℓi</div>
          <div class="cons-num" id="cons-li">—</div>
        </div>
        <div class="cons-tile">
          <div class="cons-name">Convergence</div>
          <div class="cons-num" id="cons-status" style="color:#10b981;">OK</div>
        </div>
      </div>
    </div>

    <!-- Sliders Control Deck -->
    <div class="card" style="flex:1;">
      <div class="card-header">
        <span class="card-title">Searchable Parameters (Literature Enclosed)</span>
      </div>

      <div class="param-tabs">
        <div class="param-tab active" data-tab="geometry">Geometry (10)</div>
        <div class="param-tab" data-tab="ampere">Ampère (4)</div>
        <div class="param-tab" data-tab="vacuum">Vacuum (7)</div>
        <div class="param-tab" data-tab="clocks">Clocks (2)</div>
        <div class="param-tab" data-tab="profiles">Profiles (10)</div>
      </div>

      <div class="slider-group" id="sliders-container"></div>
    </div>
  </div>
</div>

<!-- Modal: Locked Parameters (Table 1) -->
<div class="modal-overlay" id="locked-modal">
  <div class="modal-card">
    <div class="modal-head">
      <h2 style="font-size:16px;">Locked / Non-Searchable Parameters (Table 1 Reference)</h2>
      <button class="btn small" id="btn-close-modal">✕ Close</button>
    </div>
    <div class="modal-body">
      <p style="font-size:12px; color:var(--text-muted); margin-bottom:12px;">
        These parameters are firmly established by fundamental physical constants, machine engineering drawings, or published facility specifications. They remain constant across all hypothetical slider explorations.
      </p>
      <table class="table-custom">
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Name</th>
            <th>Locked Value</th>
            <th>Physical Role</th>
            <th>Literature Source</th>
          </tr>
        </thead>
        <tbody id="locked-table-body"></tbody>
      </table>
    </div>
  </div>
</div>

<script>
let currentShot = null;
let shotData = null;
let currentFrame = 0;
let isPlaying = false;
let playInterval = null;
let paramSpecs = [];
let lockedSpecs = [];
let currentParams = {};
let baselineParams = {};
let baselineResults = {};
let solveDebounceTimer = null;
let activeTab = "geometry";

async function initApp() {
  // Load specs
  const specRes = await fetch("/api/specs").then(r => r.json());
  paramSpecs = specRes.searchable;
  lockedSpecs = specRes.locked;

  // Render locked table
  const tbody = document.getElementById("locked-table-body");
  lockedSpecs.forEach(p => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td style="font-family:var(--mono); font-weight:700; color:var(--primary);">${p.symbol}</td>
      <td style="font-weight:600;">${p.name}</td>
      <td style="font-family:var(--mono); color:var(--accent);">${p.value}</td>
      <td>${p.role}</td>
      <td style="color:var(--text-muted); font-size:11px;">${p.citation}</td>
    `;
    tbody.appendChild(tr);
  });

  // Populate defaults
  paramSpecs.forEach(p => {
    currentParams[p.key] = p.default;
    baselineParams[p.key] = p.default;
  });

  // Render sliders
  renderSliders();

  // Load shots list
  const shotsRes = await fetch("/api/shots").then(r => r.json());
  const select = document.getElementById("shot-dropdown");
  shotsRes.shots.forEach(s => {
    const opt = document.createElement("option");
    opt.value = s.stem;
    opt.textContent = s.label;
    select.appendChild(opt);
  });

  select.addEventListener("change", (e) => loadShot(e.target.value));

  // Default to first shot
  if (shotsRes.shots.length > 0) {
    loadShot(shotsRes.shots[0].stem);
  }

  // Set up event listeners
  setupEvents();
}

function setupEvents() {
  document.getElementById("btn-show-locked").onclick = () => {
    document.getElementById("locked-modal").style.display = "flex";
  };
  document.getElementById("btn-close-modal").onclick = () => {
    document.getElementById("locked-modal").style.display = "none";
  };
  document.getElementById("locked-modal").onclick = (e) => {
    if (e.target.id === "locked-modal") document.getElementById("locked-modal").style.display = "none";
  };

  document.querySelectorAll(".param-tab").forEach(tab => {
    tab.onclick = () => {
      document.querySelectorAll(".param-tab").forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      activeTab = tab.dataset.tab;
      renderSliders();
    };
  });

  document.getElementById("btn-reset-all").onclick = () => {
    paramSpecs.forEach(p => currentParams[p.key] = p.default);
    renderSliders();
    triggerSolve();
  };

  document.getElementById("preset-selector").onchange = (e) => {
    const mode = e.target.value;
    if (mode === "baseline") {
      paramSpecs.forEach(p => currentParams[p.key] = p.default);
      currentParams["q95_boost"] = 1.20;
      currentParams["s_beta"] = 1.00;
    } else if (mode === "optimal") {
      paramSpecs.forEach(p => currentParams[p.key] = p.default);
      currentParams["q95_boost"] = 1.0622;
      currentParams["s_beta"] = 1.0785;
    } else if (mode === "unboosted") {
      paramSpecs.forEach(p => currentParams[p.key] = p.default);
      currentParams["q95_boost"] = 1.00;
      currentParams["s_beta"] = 1.00;
    } else if (mode === "ohmic") {
      currentParams["k0"] = 0.70;
      currentParams["kb"] = 0.25;
      currentParams["ab"] = 0.50;
      currentParams["rb"] = 0.04;
    } else if (mode === "flattop") {
      currentParams["k0"] = 0.70;
      currentParams["kw"] = 0.35;
      currentParams["kd"] = 0.25;
      currentParams["rw"] = 0.12;
      currentParams["ad"] = 0.22;
    }
    renderSliders();
    triggerSolve();
  };

  // Time scrubber
  const scrubber = document.getElementById("frame-scrubber");
  scrubber.oninput = (e) => {
    currentFrame = parseInt(e.target.value);
    updateClock();
    triggerSolve();
  };

  document.getElementById("btn-prev-frame").onclick = () => {
    if (currentFrame > 0) {
      currentFrame--;
      scrubber.value = currentFrame;
      updateClock();
      triggerSolve();
    }
  };

  document.getElementById("btn-next-frame").onclick = () => {
    if (shotData && currentFrame < shotData.n_frames - 1) {
      currentFrame++;
      scrubber.value = currentFrame;
      updateClock();
      triggerSolve();
    }
  };

  document.getElementById("btn-play").onclick = () => {
    if (isPlaying) {
      clearInterval(playInterval);
      isPlaying = false;
      document.getElementById("btn-play").textContent = "Play";
    } else {
      isPlaying = true;
      document.getElementById("btn-play").textContent = "Pause";
      playInterval = setInterval(() => {
        if (!shotData) return;
        currentFrame = (currentFrame + 1) % shotData.n_frames;
        scrubber.value = currentFrame;
        updateClock();
        triggerSolve();
      }, 150);
    }
  };
}

async function loadShot(stem) {
  currentShot = stem;
  shotData = await fetch(`/api/shot/${stem}`).then(r => r.json());
  currentFrame = 0;

  const scrubber = document.getElementById("frame-scrubber");
  scrubber.max = shotData.n_frames - 1;
  scrubber.value = 0;

  document.getElementById("shot-info-pill").textContent = 
    `${shotData.n_frames} frames • ${shotData.t_min}–${shotData.t_max} ms • Ip ≈ ${shotData.ip_max_ka} kA`;

  updateClock();
  triggerSolve();
}

function updateClock() {
  if (!shotData) return;
  const t = shotData.times[currentFrame];
  document.getElementById("clock-readout").textContent = `${t.toFixed(1)} ms (Frame ${currentFrame})`;
}

function renderSliders() {
  const container = document.getElementById("sliders-container");
  container.innerHTML = "";

  const filtered = paramSpecs.filter(p => p.group === activeTab);
  filtered.forEach(p => {
    const val = currentParams[p.key];
    const row = document.createElement("div");
    row.className = "slider-row";
    row.innerHTML = `
      <div class="slider-meta">
        <div class="slider-title">
          <span class="slider-symbol">${p.symbol}</span>
          <span>${p.name}</span>
        </div>
        <div style="display:flex; align-items:center; gap:8px;">
          <span class="slider-range-badge">[${p.min}, ${p.max}] ${p.unit}</span>
          <span class="slider-val-box" id="val-${p.key}">${Number(val).toFixed(p.step < 0.01 ? 3 : 2)}</span>
        </div>
      </div>
      <div class="slider-input-line">
        <input type="range" class="param-slider" 
          min="${p.min}" max="${p.max}" step="${p.step}" value="${val}"
          data-key="${p.key}" id="slider-${p.key}"/>
      </div>
      <div class="slider-rationale">${p.rationale}</div>
    `;

    const slider = row.querySelector(`#slider-${p.key}`);
    slider.oninput = (e) => {
      const v = parseFloat(e.target.value);
      currentParams[p.key] = v;
      row.querySelector(`#val-${p.key}`).textContent = v.toFixed(p.step < 0.01 ? 3 : 2);
      triggerSolve();
    };

    container.appendChild(row);
  });
}

function triggerSolve() {
  if (solveDebounceTimer) clearTimeout(solveDebounceTimer);
  solveDebounceTimer = setTimeout(performSolve, 50);
}

async function performSolve() {
  if (!currentShot) return;

  const payload = {
    shot: currentShot,
    t_index: currentFrame,
    params: currentParams,
  };

  try {
    const res = await fetch("/api/solve", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    }).then(r => r.json());

    if (res.error) {
      console.error(res.error);
      return;
    }

    renderResults(res);
  } catch (err) {
    console.error(err);
  }
}

function renderResults(res) {
  // Latency & Diagnostics
  document.getElementById("solve-latency").textContent = `Latency: ${res.solve_ms} ms (Nit: ${res.n_iter})`;
  document.getElementById("diag-iter").textContent = `${res.n_iter} passes`;
  document.getElementById("diag-walk").textContent = res.walk.toFixed(3);
  document.getElementById("diag-bleed").textContent = res.bleed.toFixed(3);
  document.getElementById("diag-currents").textContent = `${(res.ip_ma * 1000).toFixed(0)} kA / ${res.b_t.toFixed(2)} T`;

  // Global Profile Scalars
  document.getElementById("val-q95").textContent = res.q95.toFixed(3);
  document.getElementById("val-betaN").textContent = res.betaN.toFixed(3);
  document.getElementById("val-deltapsi").textContent = `${(res.delta_psi * 1000).toFixed(1)} mWb`;
  document.getElementById("val-axis").textContent = `(${res.axis_r.toFixed(3)}, ${res.axis_z.toFixed(3)})`;

  // Consistency Functionals
  document.getElementById("cons-raxis").textContent = res.scalars.R_axis.toFixed(3);
  document.getElementById("cons-zaxis").textContent = res.scalars.Z_axis.toFixed(3);
  document.getElementById("cons-kappa").textContent = res.scalars.kappa.toFixed(3);
  document.getElementById("cons-vol").textContent = res.scalars.volume.toFixed(2);
  document.getElementById("cons-tritop").textContent = res.scalars.tri_top.toFixed(3);
  document.getElementById("cons-tribot").textContent = res.scalars.tri_bot.toFixed(3);
  document.getElementById("cons-li").textContent = res.scalars.li.toFixed(3);

  // Plot 2D Poloidal Flux Map with Overlays
  renderPlotly(res);
}

function renderPlotly(res) {
  const r = res.r;
  const z = res.z;
  const psi = res.psi;

  const data = [
    // 2D Contour
    {
      z: psi,
      x: r,
      y: z,
      type: "contour",
      colorscale: "Plasma",
      ncontours: 28,
      line: { width: 0.8, color: "rgba(255,255,255,0.25)" },
      contours: { coloring: "heatmap" },
      colorbar: {
        title: "ψ (Wb)",
        titleside: "right",
        tickfont: { color: "#94a3b8", family: "monospace", size: 10 },
        titlefont: { color: "#f1f5f9", size: 11 },
        thickness: 12,
        len: 0.95,
      },
    },
  ];

  // LCFS Contour Polygon Overlay
  if (res.contour && res.contour.length > 0) {
    const cx = res.contour.map(pt => pt[0]);
    const cy = res.contour.map(pt => pt[1]);
    cx.push(cx[0]);
    cy.push(cy[0]);
    data.push({
      x: cx,
      y: cy,
      mode: "lines",
      line: { color: "#38bdf8", width: 2.8 },
      name: "LCFS (Last-Closed)",
      hoverinfo: "skip",
    });
  }

  // O-point Magnetic Axis
  data.push({
    x: [res.axis_r],
    y: [res.axis_z],
    mode: "markers",
    marker: { color: "#ef4444", size: 10, symbol: "cross-thin", line: { width: 3, color: "#ef4444" } },
    name: "Magnetic Axis (O-point)",
  });

  const layout = {
    margin: { t: 10, r: 10, b: 40, l: 50 },
    paper_bgcolor: "#131b2e",
    plot_bgcolor: "#0b0f19",
    showlegend: false,
    xaxis: {
      title: "Major Radius R (m)",
      range: [0.06, 2.00],
      color: "#94a3b8",
      gridcolor: "rgba(255,255,255,0.06)",
      zerolinecolor: "rgba(255,255,255,0.1)",
    },
    yaxis: {
      title: "Vertical Elevation Z (m)",
      range: [-1.80, 1.80],
      scaleanchor: "x",
      scaleratio: 1,
      color: "#94a3b8",
      gridcolor: "rgba(255,255,255,0.06)",
      zerolinecolor: "rgba(255,255,255,0.1)",
    },
    shapes: [
      // Center column armor wall (R = 0.20 m)
      {
        type: "line",
        x0: 0.20, x1: 0.20, y0: -1.8, y1: 1.8,
        line: { color: "rgba(239, 68, 68, 0.7)", width: 2, dash: "dot" },
      },
      // Outboard limiter wall (R = 1.45 m)
      {
        type: "line",
        x0: 1.45, x1: 1.45, y0: -1.8, y1: 1.8,
        line: { color: "rgba(245, 158, 11, 0.7)", width: 2, dash: "dot" },
      },
      // Thomson midplane laser chord (Z = 0.0 m, R in [0.25, 1.45])
      {
        type: "line",
        x0: 0.25, x1: 1.45, y0: 0.0, y1: 0.0,
        line: { color: "rgba(16, 185, 129, 0.8)", width: 1.5, dash: "dash" },
      },
    ],
    annotations: [
      {
        x: 0.20, y: 1.65,
        text: "Center Column (0.20m)",
        showarrow: false,
        font: { size: 9, color: "#ef4444", family: "monospace" },
        xanchor: "left",
      },
      {
        x: 1.45, y: 1.65,
        text: "Limiter (1.45m)",
        showarrow: false,
        font: { size: 9, color: "#f59e0b", family: "monospace" },
        xanchor: "right",
      },
    ],
  };

  const config = { responsive: true, displayModeBar: false };
  Plotly.react("flux-plot", data, layout, config);
}

window.addEventListener("DOMContentLoaded", initApp);
</script>
</body>
</html>
"""

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MAST Equilibrium Explorer")
    parser.add_argument("--port", type=int, default=8775, help="Server port (default: 8775)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    args = parser.parse_args()

    init_environment()

    server_address = (args.host, args.port)
    httpd = ThreadingHTTPServer(server_address, ExplorerHandler)
    print(f"\n==========================================================================")
    print(f" MAST EQUILIBRIUM EXPLORER READY")
    print(f" Access app in browser: http://{args.host}:{args.port}")
    print(f" 1,206 Unlabelled MAST Test Shots + 3 Labeled Demo Shots")
    print(f" 32 Bounded Free Parameters with Instant (~60ms) Grad–Shafranov Solves")
    print(f"==========================================================================\n", flush=True)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    main()
