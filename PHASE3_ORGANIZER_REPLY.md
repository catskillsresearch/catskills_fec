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

Public GitHub Release (pinned source + trained weights):

- Release: https://github.com/catskillsresearch/catskills_fec/releases/tag/phase3-reproduce-03becd2  
- Commit: `03becd27c531906281cf8310a49f8b33a4d8a282`  
- Archive: `phase3-reproduce-03becd2.tar.gz`  
- SHA-256: `500f76d4e82acfc476f0aece384ec99c46f3ac5a6563021ca8e02ea41f2dbf7d`

Long-form methods document (assembled from the `fec.md` wiki: OSF + MAST/DIII-D calibration + DIII-D ML fit + Phase 2/3):  
`docs/FEC.pdf` in the release / repository (`python3 scripts/build_fec_pdf.py`).

Contents include model source, `environment.yml` / requirements, trained weights under `weights/phase3/` with `SHA256SUMS`, training entrypoint (`./reproduce.sh --train`), inference entrypoint (`./reproduce.sh --weights-only` → `generate_final_predictions`), `PROVENANCE.md`, and `COMPUTE.md`.

The archive is **self-contained** (real `.pt` files under `weights/phase3/`). No lab NAS / Fusion drive is required. Default outputs land in `benchmarks/CRC_FEQ_phase3/reproduce_<stamp>/` inside the extracted tree (`weight_diff.html`, `repro_weights/`, prediction `.npz`). Short how-to: `REPRODUCE.md`.

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

Details: `docs/FEC.pdf` (long) and `REPRODUCE.md` (short). Compute budget: `COMPUTE.md` (≤24 GPU-h A100-class; OSF is CPU-only).

We are happy to arrange a private hand-over if preferred before results are announced.

Best regards,  
Catskills Research Company (`catskills`)
