"""generate_4shots_cache.py — Generate and cache predictions, scores, and diagnostics for top 4 DIII-D shots."""
import time
from pathlib import Path
from src.feq.data import load_shot, shot_paths
from osf.d3d_0_0_15_osf_modal import solve_shot, MODEL_ID
from osf.osf_0_0_12 import write_shot
from fec_site.score import load_or_compute
from fec_site.diag import load_or_compute_diag

ROOT = Path(__file__).resolve().parent
out_dir = ROOT / "results" / MODEL_ID
out_dir.mkdir(parents=True, exist_ok=True)

paths = shot_paths("diii_d_train")
stems = sorted([p.stem for p in paths])[:4]

for stem in stems:
    p = [x for x in paths if x.stem == stem][0]
    dest = out_dir / f"{stem}.npz"
    print(f"\nProcessing {stem} ({p})...")
    row = load_shot(p)
    t0 = time.time()
    pred = solve_shot(row, k_modes=16, reg=1e-2)
    t1 = time.time()
    print(f"  Solved in {t1 - t0:.2f} s")
    
    print(f"  Writing {dest}...")
    write_shot(dest, pred)
    
    print(f"  Computing score cache...")
    score_pack = load_or_compute(out_dir, stem, row, pred)
    
    print(f"  Computing diagnostic cache...")
    diag_pack = load_or_compute_diag(out_dir, stem, row, pred, score_pack)
    
    s = score_pack["shot"]
    print(f"  Done! S = {s['S']:.5f}, Cons = {s['Consistency']:+.5f}, R2(psi) = {s['r2_psi']:.6f}, 1 - D = {s['one_minus_D']:.5f}")

print("\nAll 4 shots cached successfully!")
