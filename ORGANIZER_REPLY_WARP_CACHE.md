# Organizer reply — warp_coeff_cache / LS_O1_W1

**To:** FusionEC-26 organizers  
**Re:** `lambda-w > 0 but warp_basis cache missing` at commit `907d045`

---

Dear organizers,

You are right: at `907d045`, `psi_tailfit.py` names `warp_coeff_cache` and reads `results/brown/warp_basis/<key>.npz`, but the builder was not in that tree. Sorry — that was an omission in the rebuild packaging.

**Fix:** we are pushing `catskills/warp_coeff.py` and `catskills/scripts/warp_coeff_cache.py` onto `main`, and updating `scripts/phase3_rebuild_all_d3d.sh` to:

1. `mkdir -p $CATSKILLS_FUSION_LAB/results/brown`
2. run `maroon_m0_cache` (PCA / `coeffs_v1`, as in `REPRODUCE.md`)
3. run `warp_coeff_cache` **before** the `LS_O1_W1` train

Until that commit lands, you can drop the two modules from this reply’s sibling PR / pull `main` after the push.

### Build the cache (settings we used)

Prerequisite: coefficient cache for the train keys (`maroon_m0_cache` / `coeffs_v1`).

```bash
# Same split psi_tailfit uses: sample_cache_keys(7000+80, seed=42)
.venv/bin/python -m catskills.scripts.warp_coeff_cache --keys train --workers 12
```

| Setting | Value |
|---|---|
| `--keys` | `train` → keys `[80:]` of `sample_cache_keys(7080, 42, include_thomson=True)` (6,961 shots) |
| PCA | \(k=60\), Thomson on |
| Workers | `12` (docstring default for the house build; any `≥1` is fine) |
| Output | `$CATSKILLS_FUSION_LAB/results/brown/warp_basis/<key>.npz` |
| Payload | `B` shape `(60, 5)`, `names = (iso, stretch_z, shift_r, shift_z, gain)` |

Then the W1 command unchanged:

```bash
.venv/bin/python -m catskills.scripts.psi_tailfit --parent O1 --power 0.0 --seed 42 \
  --lambda-w 1 --tag LS_O1_W1 --epochs 220 --patience 30
```

`LS_O1_101` does **not** need the warp cache (`--lambda-w` defaults to 0).

### Lab provenance (for forensics only)

The builder first landed in the lab tree as `catskills/scripts/warp_coeff_cache.py` + `catskills/warp_coeff.py` around commit `b686e27c…` (2026-09-07). It was never copied into `catskills_fec` for `907d045`. You should not need that lab SHA once `main` has the port.

### Your other notes

Agreed on both:

- **`maroon_m0_cache`** — required for coeffs; now step 0 of `phase3_rebuild_all_d3d.sh`.
- **`results/brown`** — `psi_tailfit` / warp cache write there; the rebuild script now `mkdir`s it, and `psi_tailfit` creates the directory itself before writing.

Thank you again for the careful rebuild.

Best regards,  
Catskills Research Company (`catskills`)
