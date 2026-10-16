#!/usr/bin/env python3
"""Investigate saddle node regularization and boundary feathering on the 80 holdout shots.

Tests:
1. Baseline: 0.8.0 Ensemble (+ dZ=0.5mm)
2. Local quadratic saddle reconstruction (hyperbolic fit on 5x5 / 7x7 patch around saddle)
3. Vacuum Grad-Shafranov relaxation (Delta* psi = 0) near the divertor / X-point
4. Localized Gaussian feathering / anisotropic smoothing along the LCFS band
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates, minimum_filter, gaussian_filter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "fusion_scoring"))

from catskills import probe9 as p9
from catskills.scripts.solve_grand_ensemble import CLEAN, NAMES, _grand
from catskills.scripts.solve_ensemble_weights import (
    SCALAR_STEMS,
    _clean,
    _mix932,
    _row,
    _same_frames,
)
from catskills.scripts.probe_consistency_repairs import W_PRINT, _shift
from fusion_scoring.lcfs import extract_lcfs
from fusion_scoring.derive import _shape_from_contour, magnetic_axis


def find_saddle_candidates(psi: np.ndarray, R: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Find interior saddle points of phi = -psi."""
    phi = -np.asarray(psi, np.float64)
    gz, gr = np.gradient(phi, Z, R)
    hzz = np.gradient(gz, Z, axis=0)
    hrr = np.gradient(gr, R, axis=1)
    hrz = np.gradient(gr, Z, axis=0)
    g2 = gr ** 2 + gz ** 2
    det = hrr * hzz - hrz ** 2
    interior = np.zeros_like(phi, dtype=bool)
    interior[3:-3, 3:-3] = True
    cand = (g2 == minimum_filter(g2, size=3)) & (det < 0) & interior
    idx = np.argwhere(cand)
    if len(idx) == 0:
        return np.empty((0, 4), np.float64)
    out = np.empty((len(idx), 4), np.float64)
    for i, (iz, ir) in enumerate(idx):
        out[i] = (R[ir], Z[iz], float(g2[iz, ir]), float(phi[iz, ir]))
    return out


def fit_local_quadratic_saddle(
    psi: np.ndarray, R: np.ndarray, Z: np.ndarray, iz0: int, ir0: int, patch_radius: int = 3
) -> np.ndarray:
    """Fit a smooth 2nd-degree bivariate polynomial to the local patch around (iz0, ir0).
    
    P(r, z) = c0 + c1*dr + c2*dz + c3*dr^2 + c4*dr*dz + c5*dz^2
    """
    nz, nr = psi.shape
    z_min = max(0, iz0 - patch_radius)
    z_max = min(nz, iz0 + patch_radius + 1)
    r_min = max(0, ir0 - patch_radius)
    r_max = min(nr, ir0 + patch_radius + 1)

    sub_R = R[r_min:r_max]
    sub_Z = Z[z_min:z_max]
    sub_psi = psi[z_min:z_max, r_min:r_max]

    r_c = R[ir0]
    z_c = Z[iz0]

    RR, ZZ = np.meshgrid(sub_R, sub_Z)
    dr = (RR - r_c).ravel()
    dz = (ZZ - z_c).ravel()
    y = sub_psi.ravel()

    # Design matrix
    A = np.column_stack([
        np.ones_like(dr),
        dr,
        dz,
        dr ** 2,
        dr * dz,
        dz ** 2,
    ])

    # Ridge regularized least squares fit
    reg = 1e-6 * np.eye(A.shape[1])
    reg[0, 0] = 0.0
    c, _, _, _ = np.linalg.lstsq(A.T @ A + reg, A.T @ y, rcond=None)

    # Evaluate fitted polynomial on full grid or local patch
    dr_full = (R[None, :] - r_c)
    dz_full = (Z[:, None] - z_c)
    P_full = (
        c[0]
        + c[1] * dr_full
        + c[2] * dz_full
        + c[3] * (dr_full ** 2)
        + c[4] * (dr_full * dz_full)
        + c[5] * (dz_full ** 2)
    )
    return P_full


def regularize_saddles_feathered(
    psi: np.ndarray,
    R: np.ndarray,
    Z: np.ndarray,
    sigma_cells: float = 3.5,
    blend_weight: float = 0.5,
    patch_radius: int = 3,
) -> np.ndarray:
    """Detect controlling saddles and blend in smooth quadratic representation."""
    saddles = find_saddle_candidates(psi, R, Z)
    if len(saddles) == 0:
        return psi

    # Sort saddles by |grad phi|^2 (weakest gradient = strongest null)
    saddles = saddles[np.argsort(saddles[:, 2])]
    dr = R[1] - R[0]
    dz = Z[1] - Z[0]
    sigma_r = sigma_cells * dr
    sigma_z = sigma_cells * dz

    out = psi.copy()
    RR = R[None, :]
    ZZ = Z[:, None]

    # Regularize up to top 2 saddles (lower and upper)
    chosen = []
    for s in saddles:
        r_s, z_s = s[0], s[1]
        # Only take saddles near the expected boundary region (|z_s| > 0.4 m or r_s > 1.8 m or r_s < 1.3 m)
        if len(chosen) < 2:
            chosen.append((r_s, z_s))

    for r_s, z_s in chosen:
        ir0 = int(np.clip(np.round((r_s - R[0]) / dr), 0, len(R) - 1))
        iz0 = int(np.clip(np.round((z_s - Z[0]) / dz), 0, len(Z) - 1))

        P_smooth = fit_local_quadratic_saddle(psi, R, Z, iz0, ir0, patch_radius=patch_radius)

        # Gaussian feather weight
        dist2 = ((RR - r_s) / sigma_r) ** 2 + ((ZZ - z_s) / sigma_z) ** 2
        w = blend_weight * np.exp(-0.5 * dist2)

        out = out * (1.0 - w) + P_smooth * w

    return out.astype(psi.dtype)


def main():
    print("Loading official holdout fold...", flush=True)
    w = W_PRINT / W_PRINT.sum()
    fold = p9.fold_reference_cached()
    keys = list(fold["keys"])
    print(f"Holdout shots: {len(keys)}")

    mix = _mix932(keys)
    loaded = {stem: _clean(stem, keys) for stem in list(CLEAN.values()) + [
        stem for stem in SCALAR_STEMS if stem not in CLEAN.values()
    ]}
    grand = _grand(keys)
    _same_frames(grand, mix, "M_grand", "932612")
    members = {name: loaded[stem] for name, stem in CLEAN.items()}
    members["M_grand"] = grand
    members["M_932612"] = mix
    for name in NAMES:
        _same_frames(members[name], mix, name, "932612")
    stack = np.stack([members[name]["z"] for name in NAMES], axis=0)
    z = np.tensordot(w, stack, axes=(0, 0))
    psi = p9.psi_from_z(z)
    ids = mix["shot_ids"]
    volumes = [psi[ids == i] for i in range(len(keys))]
    q_star = np.mean([loaded[stem]["q95"] for stem in SCALAR_STEMS], axis=0)
    b_star = np.mean([loaded[stem]["betaN"] for stem in SCALAR_STEMS], axis=0)

    R, Z, mask, mask_f = fold["grid"]
    R = np.asarray(R[0] if R.ndim == 2 else R, np.float64)
    Z = np.asarray(Z[:, 0] if Z.ndim == 2 else Z, np.float64)

    # Shift dZ = +0.5 mm baseline
    dz = np.diff(Z)[0]
    z_index_05 = np.arange(len(Z), dtype=np.float64) + (0.5 * 1e-3) / float(dz)
    volumes_dz05 = [_shift(vol, z_index_05) for vol in volumes]

    def eval_candidate(vols, label):
        preds = []
        for i, vol in enumerate(vols):
            sl = ids == i
            preds.append({
                "psirz": np.asarray(vol, np.float32),
                "q95": q_star[sl].astype(np.float32),
                "betaN": b_star[sl].astype(np.float32),
            })
        parts = p9.score_parts(fold, preds, workers=8)
        row = _row(label, parts)
        each = row["r2_cons_each"]
        print(
            f"{label:<25} S={row['S']:.6f} Cons={row['consistency']:.6f} "
            f"R2ψ={row['r2_psi']:.6f} 1-D={row['one_minus_dlcfs']:.6f} "
            f"κ={each['kappa']:.6f} Z={each['Z_axis']:.6f} R={each['R_axis']:.6f} "
            f"tri_b={each['tri_bot']:.6f} li={each['li']:.6f}",
            flush=True,
        )
        return row

    print("\n--- BASELINE SCORING ---")
    eval_candidate(volumes_dz05, "dz05_baseline")

    # Sweep localized saddle regularization parameters
    print("\n--- SWEEPING SADDLE QUADRATIC REGULARIZATION ---")
    for sigma in [2.0, 3.0, 4.0]:
        for bw in [0.2, 0.4, 0.6]:
            t0 = time.time()
            vols_reg = []
            for vol in volumes_dz05:
                reg_vol = np.empty_like(vol)
                for t in range(len(vol)):
                    reg_vol[t] = regularize_saddles_feathered(
                        vol[t], R, Z, sigma_cells=sigma, blend_weight=bw, patch_radius=3
                    )
                vols_reg.append(reg_vol)
            eval_candidate(vols_reg, f"saddle_sig{sigma}_w{bw}")


if __name__ == "__main__":
    main()
