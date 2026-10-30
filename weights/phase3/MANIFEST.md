# Phase 3 shipped weights (Phase-2 Final)

Target submission: **EM-catskills-3** / HF `9db5e666…`  
Forensic sibling: Codabench **1003664** / `178dc841…` (narrow ZGRID; use `zgrid_legacy_1003664.json`).

## Checksums

Verify with `sha256sum -c SHA256SUMS` in this directory.

## DIII-D (charcoal / O1G)

| Stem | Role |
|---|---|
| `CRC_FEQ_olive_geom_O1G_grand` | Flux member (grand clock) |
| `CRC_FEQ_olive_geom_O1G_clean_ip_seed{45,44,43}_best` | Flux members (Ip-shift) |
| `CRC_FEQ_olive_geom_O1G_clean_ip_scratch_best` | Flux member (Ip-shift) |
| `CRC_FEQ_olive_geom_O1G_early_b2048`, `O1G_early`, `O1G_place` | Legacy flux members |
| `CRC_FEQ_brown_TF_LS_O1_{101,W1}` | Legacy flux members |
| `CRC_FEQ_olive_geom_O1G_clean_ip_scratch_{best,final}`, `seed43_{best,final}` | Scalar heads |

## MAST (M9)

| Stem | Role |
|---|---|
| `CRC_FEQ_mast_M9_win50_i{101,202,303}` | M9 flux ensemble |

MAST q95 / βN come from OSF (`osf/mast_12_osf_0_0_12.py`), not from these weights.

## ZGRID

- `zgrid.json` — freeze for 9db5e666 (±1.2, 32 bins; 0.75 × EFIT Z half-width).
- `zgrid_legacy_1003664.json` — forensic (−0.6…+0.8).

## Staging

If `.pt` files are symlinks to the lab archive, materialize before shipping an organizer tarball:

```bash
./weights/phase3/stage_weights.sh --copy
```
