# Compute declaration — Phase 3 (Reproduce)

Ceiling: **≤ 24 GPU-hours** on a single 80 GB A100-class GPU for both machines together (challenge rule 6).

## Summary

| Work | Device | Wall-clock (approx.) | GPU-h |
|---|---|---|---|
| Charcoal O1G DIII-D ensemble (full ladder + clean-Ip seeds) | 1× A100 80 GB | ~12–18 h | ~12–18 |
| M9 MAST flux trio (3×50 ep; Phase-2 wall-clock) | 1× GPU | ~7 min (~2 min/member) | ≪1 |
| M9 Phase-3 budget pad (cold cache) | 1× GPU | up to ~1–3 h | ≤3 |
| OSF MAST scalars | CPU only | ~2–5 min | **0** |
| Private-test inference (both machines, shipped weights) | 1× GPU ≥16 GB | ~45–90 min | ~1 |
| ZGRID discovery over train parquets | CPU | ~1–3 min | **0** |

**Weights-only rebuild** (organizer check that shipped weights match submitted S within **1e-4**): inference only — well under the GPU-h budget.

**From-scratch retrain** (tolerance: composite S within **0.01**): charcoal + optional M9 as above; OSF remains CPU-only. Combined estimate stays under 24 GPU-h on A100-class hardware.

## Hardware notes

- Peak VRAM for charcoal O1G members: typically ≤ 16 GB at batch 2048 (placement_early); grand/signature path may use more — stay within one 80 GB device.
- CPU data generation (parquet feature caches, OSF Green function): uncapped under the rules; declare wall-clock if material.

## Environment

Pinned via `environment.yml` and `requirements.txt` / `requirements-pytorch.txt` at the package commit.
