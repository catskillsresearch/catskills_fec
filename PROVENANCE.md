# Data Provenance Statement — Phase 3 (Reproduce)

**Challenge:** Fusion Equilibrium Challenge (NeurIPS 2026)  
**Team:** Catskills Research Company (`catskills`)  
**Package phase:** Phase 3 organizer rebuild  
**Scored artifacts verified:** Codabench **1003664** (`178dc841…`) and **EM-catskills-3** (`9db5e666…`)

---

## Dual pipelines (Phase 2 Final)

| Machine | Model | Weights / code |
|---|---|---|
| **DIII-D** | Charcoal / O1G 6-member neural ensemble | `weights/phase3/CRC_FEQ_olive_geom_*.pt`, `CRC_FEQ_brown_TF_LS_O1_*.pt` |
| **MAST** | M9 neural flux + OSF analytic q95/βN | `weights/phase3/CRC_FEQ_mast_M9_*.pt` + `osf/mast_12_osf_0_0_12.py` |

Inference driver: `python -m catskills.scripts.generate_final_predictions`  
Entrypoint: `./reproduce.sh --weights-only` or `./reproduce.sh --train`

---

## Thomson ZGRID (DIII-D)

ZGRID is **not** a permanent hardcode for Phase 3.

1. Discover once on the **organizer-supplied Reproduce-Phase train split** (`efit_grid_Z` → half-width × 0.75, `n_bins=32`).
2. Freeze to `zgrid.json` / `CRC_FEQ_ZGRID_{LO,HI,N}`.
3. Train and infer with that freeze. **Do not** re-discover from `private_test`.

Shipped `weights/phase3/zgrid.json` matches the Dev/public `*_thz` law (±1.2) used for **9db5e666**.  
`zgrid_legacy_1003664.json` (−0.6…+0.8) is forensic only for submission **1003664**.

---

## Award / Rule 3 compliance

### DIII-D (Award #1 path)

| Component | Source |
|---|---|
| Inputs | Official challenge `diii_d_train` magnetics + Thomson |
| Targets | Official `diii_d_train` EFIT (`efit_psirz`, `efit_q95`, `efit_beta_n`) |
| External pre-trained models | None beyond checkpoints trained on that fold |
| ZGRID | Discovered from train EFIT Z mesh; frozen for infer |

### MAST (Award #2 / transfer path)

| Data | Used? |
|---|---|
| FAIR-MAST EFIT++ / archive reconstructions | **No** |
| Demo shots 28348/28350/28351 as fit targets | **No** (inspect-only historically; not in training loop) |
| `mast_envelope.npz` as prior/mask | **No** |
| Official MAST sensor inputs | **Yes** |
| Public machine geometry / literature constants | **Yes** |
| OSF electrodynamic closures + frozen `c*`, `s*` | **Yes** (code hyperparameters) |
| M9 flux nets | Trained without forbidden MAST equilibrium labels (challenge-legal regime documented in training logs) |

---

## Submission mapping

| ID | HF revision | Standing Phase-3 path |
|---|---|---|
| EM-catskills-3 | `9db5e666…` | Train-discovered / shipped ±1.2 freeze + charcoal+M9/OSF |
| 1003664 | `178dc841…` | Same models; optional `--legacy-zgrid-1003664` |
