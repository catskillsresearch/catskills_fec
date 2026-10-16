#!/usr/bin/env python3
"""Diagnose exact jump frames on the official 80-shot holdout."""
import sys
import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "fusion_scoring")

from catskills import probe9 as p9
from catskills.scripts.solve_grand_ensemble import CLEAN, NAMES, _grand
from catskills.scripts.solve_ensemble_weights import SCALAR_STEMS, _clean, _mix932
from catskills.scripts.probe_consistency_repairs import W_PRINT, _shift
from fusion_scoring.lcfs import extract_lcfs, _best_contour_at
from fusion_scoring.o_point import find_o_point

fold = p9.fold_reference_cached()
keys = list(fold["keys"])
mix = _mix932(keys)
loaded = {stem: _clean(stem, keys) for stem in list(CLEAN.values()) + [
    stem for stem in SCALAR_STEMS if stem not in CLEAN.values()
]}
grand = _grand(keys)
members = {name: loaded[stem] for name, stem in CLEAN.items()}
members["M_grand"] = grand
members["M_932612"] = mix
w = W_PRINT / W_PRINT.sum()
stack = np.stack([members[name]["z"] for name in NAMES], axis=0)
z = np.tensordot(w, stack, axes=(0, 0))
psi = p9.psi_from_z(z)
ids = mix["shot_ids"]
volumes = [psi[ids == i] for i in range(len(keys))]

R, Z, mask, mask_f = fold["grid"]
R = np.asarray(R[0] if R.ndim == 2 else R, np.float64)
Z = np.asarray(Z[:, 0] if Z.ndim == 2 else Z, np.float64)
dz = np.diff(Z)[0]
z_index = np.arange(len(Z), dtype=np.float64) + (0.5 * 1e-3) / float(dz)
shifted_vols = [_shift(vol, z_index) for vol in volumes]

jump_count = 0
total_frames = 0
jump_frames = []

for shot_idx, (s_vol, ref) in enumerate(zip(shifted_vols[:15], fold["refs"][:15])):
    gt_cons = ref[1]
    gt_cmask = ref[2]
    # gt_cons has columns: R_axis, Z_axis, kappa, tri_top, tri_bot, volume, li
    for t in range(len(s_vol)):
        total_frames += 1
        if not gt_cmask[t, 2]: # kappa mask
            continue
        gt_kappa = gt_cons[t, 2]
        pred_c = extract_lcfs(s_vol[t], R, Z, "DIII-D", mask, mask_f)
        if pred_c is None:
            continue
        z_min = float(pred_c[:, 1].min())
        z_max = float(pred_c[:, 1].max())
        r_min = float(pred_c[:, 0].min())
        r_max = float(pred_c[:, 0].max())
        w_p = r_max - r_min
        pred_kappa = (z_max - z_min) / w_p
        
        err_kappa = pred_kappa - gt_kappa
        if abs(err_kappa) > 0.15:
            jump_count += 1
            if len(jump_frames) < 10:
                jump_frames.append({
                    "shot": shot_idx,
                    "t": t,
                    "gt_kappa": gt_kappa,
                    "pred_kappa": pred_kappa,
                    "err": err_kappa,
                    "z_min_pred": z_min,
                    "z_max_pred": z_max,
                })

print(f"Total frames: {total_frames}, Jumps (|dkappa| > 0.15): {jump_count} ({jump_count/total_frames*100:.2f}%)")
print("First few jump frames:")
for jf in jump_frames:
    print(f"  Shot {jf['shot']} t={jf['t']}: GT_kappa={jf['gt_kappa']:.3f}, Pred_kappa={jf['pred_kappa']:.3f}, diff={jf['err']:+.3f}, Zmin_pred={jf['z_min_pred']:.3f}")
