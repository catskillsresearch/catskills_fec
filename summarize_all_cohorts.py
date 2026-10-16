import json
from pathlib import Path

results_dir = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/results/d3d_100shot_pretest")
c1 = json.load(open(results_dir / "receipt_100_shot_pretest.json"))
c2 = json.load(open(results_dir / "receipt_100_shot_pretest_seed101.json"))
c3 = json.load(open(results_dir / "receipt_200_shot_osf_scalars_seed2024.json"))
c4 = json.load(open(results_dir / "receipt_200_shot_midnight_5pairs_seed2025.json"))

nf1, nf2, nf3, nf4 = c1["n_frames"], c2["n_frames"], c3["n_frames"], c4["n_frames"]
total_frames = nf1 + nf2 + nf3 + nf4

s4b_c1 = c1["scores"]["Solution 4b: Combined (dZ = +0.5 mm + 100% Scalars)"]["S"]
s4b_c2 = c2["scores"]["Solution 4b: Combined (dZ = +0.5 mm + 100% Scalars)"]["S"]
s4b_c3 = c3["scores"]["Sol 4b: Champion Deep Ensemble (dZ=+0.5mm + Dedicated)"]["S"]
s4b_c4 = c4["scores"]["Package #4: Champion Sol 4b (dZ = +0.5 mm + DedicatedScalarNet)"]["S"]

w_s4b = (s4b_c1*nf1 + s4b_c2*nf2 + s4b_c3*nf3 + s4b_c4*nf4) / total_frames
print(f"Sol 4b Champion across 600 unique shots ({total_frames:,} frames):")
print(f"  Cohort 1 (100 shots, {nf1:,} frames): {s4b_c1:.5f}")
print(f"  Cohort 2 (100 shots, {nf2:,} frames): {s4b_c2:.5f}")
print(f"  Cohort 3 (200 shots, {nf3:,} frames): {s4b_c3:.5f}")
print(f"  Cohort 4 (200 shots, {nf4:,} frames): {s4b_c4:.5f}")
print(f"  Weighted Grand Mean : {w_s4b:.5f}")

s5b_c3 = c3["scores"]["Sol 5b: OSF Modal + DedicatedScalarNet (Raw)"]["S"]
s5b_c4 = c4["scores"]["Package #5: Upgraded OSF Modal Sol 5b (Modal + DedicatedScalarNet)"]["S"]
n_5b = nf3 + nf4
w_s5b = (s5b_c3*nf3 + s5b_c4*nf4) / n_5b
print(f"\nSol 5b (OSF Modal + DedicatedScalarNet) across 400 unique shots ({n_5b:,} frames):")
print(f"  Cohort 3 (200 shots, {nf3:,} frames): {s5b_c3:.5f}")
print(f"  Cohort 4 (200 shots, {nf4:,} frames): {s5b_c4:.5f}")
print(f"  Weighted Grand Mean : {w_s5b:.5f}")

c3_scal = c3["scalar_calibration"]
c4_scal = c4["scalar_calibration"]
print("\nDedicatedScalarNet Raw R2 on Fresh Holdouts:")
print(f"  Cohort 3 (200 shots, {nf3:,} frames): q95 = {c3_scal['q95_raw_r2']:.6f}, betaN = {c3_scal['betaN_raw_r2']:.6f}")
print(f"  Cohort 4 (200 shots, {nf4:,} frames): q95 = {c4_scal['q95_raw_r2']:.6f}, betaN = {c4_scal['betaN_raw_r2']:.6f}")
