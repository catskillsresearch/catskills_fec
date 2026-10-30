# Organizer reply draft — Phase 3 (Reproduce)

**To:** FusionEC-26 organizers  
**Re:** Reproduction packages for Codabench 1003664 and EM-catskills-3  
**Deadline:** 31 October 2026 23:59 CDT  

---

Dear organizers,

Please find our Phase 3 (Reproduce) package for the two Final submissions you listed.

## Identifiers

| Codabench / EM id | HF revision | Package path |
|---|---|---|
| **1003664** | `178dc841…` | Same dual pipelines; optional forensic ZGRID (see below) |
| **EM-catskills-3** | `9db5e666d9a01ef9a79920dc293c04b9663eae7c` | Standing rebuild target |

## Hand-over artifact

Versioned archive (SHA-256):

```
sha256:dc29bb15607bb208e339750b51e0955b2b97de1c6290b3f642915d019d07f854
```

File (lab):  
`/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/benchmarks/CRC_FEQ_phase3/catskills_phase3_reproduce_20261030.tar.gz`  
(private hand-over as needed; companion `PHASE3_ORGANIZER_REPLY.md` / `REPLAY_RECEIPT.md` in the same directory).

Contents include model source, `environment.yml` / requirements, trained weights under `weights/phase3/` with `SHA256SUMS`, training entrypoint (`./reproduce.sh --train` / `catskills.scripts.phase3_train_charcoal`), inference entrypoint (`./reproduce.sh --weights-only` → `generate_final_predictions`), `PROVENANCE.md`, and `COMPUTE.md`.

The archive is **self-contained** (real `.pt` files under `weights/phase3/`). No lab NAS / Fusion drive is required. Default outputs land in `benchmarks/CRC_FEQ_phase3/reproduce_<stamp>/` inside the extracted tree (`weight_diff.html`, `repro_weights/`, prediction `.npz`). See `REPRODUCE.md`.

## Dual pipelines

- **DIII-D:** Charcoal / O1G 6-member neural ensemble.  
- **MAST:** M9 flux ensemble + OSF analytic q95/βN (`osf_0_0_12`).  

Thomson **ZGRID** is discovered once from the Reproduce-Phase train split you supply (`efit_grid_Z` × 0.75 → freeze `zgrid.json`), then used for train and private infer. We do not re-estimate from `private_test`. The shipped freeze matches the Dev/public feature law (±1.2) used for **9db5e666**.

Submission **1003664** was produced under a mistaken narrow ZGRID (−0.6…+0.8). For forensics only: `./reproduce.sh --weights-only --legacy-zgrid-1003664`.

## How to rebuild

```bash
# Weights replay (S within 1e-4 of submitted):
./reproduce.sh --weights-only

# From-scratch train (S within 0.01), using your Phase-3 train drop:
./reproduce.sh --train --train-dir /path/to/diii_d_train
```

Details: `REPRODUCE.md`. Compute budget: `COMPUTE.md` (≤24 GPU-h A100-class; OSF is CPU-only).

We are happy to arrange a private hand-over if preferred before results are announced.

Best regards,  
Catskills Research Company (`catskills`)
