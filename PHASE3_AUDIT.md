# Phase 3 (Reproduce) compliance audit vs `final.tt`

**Date:** 2026-10-30  
**Team:** catskills  
**Organizer request:** `final.tt` — packages for Codabench **1003664** and **EM-catskills-3** (HF `9db5e666…`) by 31 Oct 2026 23:59 CDT.

## Phase naming

| Phase | Name | Role |
|---|---|---|
| 1 | Development | Public fold |
| 2 | Final | Blind private fold (scored submissions) |
| 3 | Reproduce | Organizer rebuild (this package) |

## What Phase 2 actually shipped

| Machine | Model | Entry |
|---|---|---|
| DIII-D | Charcoal / O1G 6-member ensemble | `generate_final_predictions.py` |
| MAST | M9 flux + OSF scalars (`osf_0_0_12` + \(c^*/s^*\)) | same driver |

## Gap vs pre-fix package

| Requirement (`final.tt`) | Before | After (this work) |
|---|---|---|
| Model source | Present, but docs pointed at OSF-DIII-D | Dual pipeline: charcoal DIII-D + OSF/M9 MAST |
| Pinned environment | `environment.yml` / requirements | Same + freeze note in COMPUTE.md |
| Trained weights | Lab disk only | `weights/phase3/` + SHA256 |
| Train from scratch | `reproduce.sh` → `train_fleet_full` (wrong DIII-D) | charcoal O1G train scripts |
| Infer regenerates submitted `.npz` | public_test OSF export | private_test charcoal+M9/OSF |
| Provenance per machine | OSF DIII-D narrative | Rewritten dual-pipeline PROVENANCE.md |
| Compute ≤24 GPU-h | Declared for OSF path | COMPUTE.md for charcoal + M9 + OSF CPU |
| ZGRID | Hardcoded (±1.2 after hotfix) | Discover from Phase-3 train split → `zgrid.json` |

## Submission mapping

| ID | HF revision | Local attempt |
|---|---|---|
| 1003664 | `178dc841…` | `CRC_FEQ_final/1.0.0_final_champion_20261019_115731` (narrow ZGRID bug) |
| EM-catskills-3 | `9db5e666…` | `CRC_FEQ_final/1.0.1_final_zgrid_fix_20261028_175604` |

Standing Phase-3 path targets **9db5e666**. Forensic `--legacy-zgrid-1003664` optional.

## Implementation status (2026-10-30)

- `catskills/zgrid_discover.py` — EFIT Z × 0.75 rule; freeze I/O; env apply.
- `reproduce.sh` — Phase 3 dual pipeline; `--weights-only` / `--train` / `--dry-run` / `--legacy-zgrid-1003664`.
- `weights/phase3/` — shipped `.pt` + `SHA256SUMS` + `zgrid.json`.
- Docs: `PROVENANCE.md`, `COMPUTE.md`, `REPRODUCE.md`, `fec.md` Phase 3 section.
