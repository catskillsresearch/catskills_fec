# Phase 3 (Reproduce) — organizer how-to

**Self-contained package.** No Fusion lab drive, no `/media/catskills/...` paths.
Shipped checkpoints live in-repo at `weights/phase3/*.pt` (real files in the
SHA256 archive). Outputs write under `benchmarks/CRC_FEQ_phase3/` inside the
extracted tree unless you pass `--out-dir`.

This package rebuilds the Phase-2 Final entries:

- Codabench **1003664** (HF `178dc841…`) — forensic narrow ZGRID
- **EM-catskills-3** (HF `9db5e666…`) — standing path

Tolerances (`final.tt` / rule 6): retrain composite S within **0.01**; shipped weights within **1e-4**.

## Architecture

| Machine | Pipeline |
|---|---|
| DIII-D | Charcoal / O1G ensemble (`generate_final_predictions`) |
| MAST | M9 ψ + OSF q95/βN |

## ZGRID

```bash
# Discover from the train fold you supply for Phase 3:
.venv/bin/python -m catskills.zgrid_discover --train-dir /path/to/diii_d_train --out zgrid.json --apply-env
```

Rule: `half = round(0.75 * max|efit_grid_Z|, 2)`, `n_bins=32`.  
Train and private infer must both load this freeze (`CRC_FEQ_ZGRID_*`). Never re-discover on `private_test`.

## Weights-only (match submitted S ≤ 1e-4)

```bash
./reproduce.sh --weights-only
# optional smoke:
./reproduce.sh --weights-only --dry-run
```

Uses `weights/phase3/*.pt` + `weights/phase3/zgrid.json` → writes `diii_d_private_test.npz` and `mast_private_test.npz`.

Forensic 1003664:

```bash
./reproduce.sh --weights-only --legacy-zgrid-1003664
```

## Train from scratch (match S ≤ 0.01)

```bash
./reproduce.sh --train --train-dir /path/to/organizer_diii_d_train
```

Steps: discover ZGRID → charcoal O1G train (`phase3_train_charcoal` / `o1_geom_train`) → private infer with freeze. MAST OSF needs no train; M9 weights ship in `weights/phase3/` (retrain optional; see COMPUTE.md).

### Short convergence probe (makefile-style)

```bash
./reproduce.sh --short
# resumes newest benchmarks/CRC_FEQ_phase3/reproduce_* if present:
#   skips charcoal if O1G_control.pt already in repro_weights/
#   skips each M9 member that already exists (--skip-existing)
#   builds mast caches only if missing, then continues M9 → diff → infer

./reproduce.sh --short --out-dir benchmarks/CRC_FEQ_phase3/reproduce_20261030_124935
./reproduce.sh --short --fresh      # new stamp
./reproduce.sh --short --force      # redo charcoal/M9/infer even if present
```

DIII-D charcoal needs the lab feature + coefficient caches (`features_v1`, `coeffs_v1`).
If you see `coefficient-cache keys missing`, restore or rebuild:

```bash
.venv/bin/python -m catskills.scripts.phase3_cache_preflight --fix   # lab archive → live
# or from scratch:
.venv/bin/python -m catskills.warm_feature_cache
.venv/bin/python -m catskills.scripts.maroon_m0_cache
```

MAST from-scratch builds caches then retrains M9 (no shipped-weight fallback):

```bash
.venv/bin/python -m catskills.scripts.phase3_prepare_mast_caches   # warm features + qc/win
```

`./reproduce.sh --short` / `--train` calls that automatically before M9. If warm/train fails, the run **stops**.


`--short` / `--probe` / `--quick`:

| Machine | What `--short` does |
|---|---|
| **DIII-D** | Truncated charcoal (2 epochs) — watch early loss |
| **MAST** | **Full** M9 trio (50 ep × 3, ~7 min) + frozen OSF — already ~10 min end-to-end; no truncated MAST path |

OSF constants stay frozen (no leaderboard re-probe).

## Where things live (organizer / judge machine)

| What | Path |
|---|---|
| Shipped Phase-2 weights (diff baseline) | `weights/phase3/*.pt` + `zgrid.json` |
| Checksums | `weights/phase3/SHA256SUMS` |
| Reproduce outputs | `benchmarks/CRC_FEQ_phase3/reproduce_<stamp>/` (default) |
| Retrained weights | `<out>/repro_weights/` |
| Smell-check report | `<out>/weight_diff.html` (+ `.md`, `.json`, `_figures/`) |
| Predictions | `<out>/diii_d_private_test.npz`, `mast_private_test.npz` |

Diff always compares **`weights/phase3`** (shipped, never overwritten) vs
**`<out>/repro_weights`**. Override shipped location only with
`CRC_FEQ_SHIPPED_WEIGHTS_DIR` if you unpack weights elsewhere.

Hand-over artifact: the SHA256 tarball already materializes real `.pt` files
(not symlinks). If you clone a git tree that still has lab symlinks, run
`weights/phase3/stage_weights.sh --copy` from a local mirror, or use the tarball.

## Weight diff (after `--train` / `--short`)

Retrain writes new checkpoints under `<out>/repro_weights/` and **does not**
overwrite `weights/phase3/` (shipped). Step 1c prints a DIII-D / MAST summary:

```text
[MAST] CRC_FEQ_mast_M9_win50_i101: DIFF  max_abs=…  mean_abs=…  rel_l2=…
DIII-D summary: …
MAST summary: …
```

Visual smell-check report:

- `<out>/weight_diff.html` — tables + charts (open in a browser)
- `<out>/weight_diff.md` — same tables + linked PNGs
- `<out>/weight_diff.json` — machine-readable
- `<out>/weight_diff_figures/` — PNG bar charts

```bash
.venv/bin/python -m catskills.scripts.phase3_weight_diff \
  --shipped-dir weights/phase3 --repro-dir <out>/repro_weights \
  --out-json <out>/weight_diff.json
```

Exact stem matches for M9; short charcoal may use `--cross` against `O1G_grand`.

## Validation

```bash
.venv/bin/python validate_submission.py <out>/diii_d_private_test.npz --config diii_d_private_test
.venv/bin/python validate_submission.py <out>/mast_private_test.npz --config mast_private_test
```

## Post-hoc package rule

Phase-3 docs and `reproduce.sh` may be completed after Phase-2 submit. Scored blobs on HF are not rewritten; this package must regenerate equivalent private predictions from Final inputs + shipped weights / retrain.
