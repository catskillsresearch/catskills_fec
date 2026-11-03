# Organizer reply — from-scratch rebuild gaps

**To:** FusionEC-26 organizers  
**Re:** Rebuild steps / missing pieces (no decision requested on earlier questions)  

---

Dear organizers,

Thank you for the careful check. You are right: `./reproduce.sh --train` as shipped in `phase3-reproduce-03becd2` only rebuilds a **subset** of the DIII-D charcoal ladder (control / place / early@2048) and then falls back to shipped weights for the rest of the ensemble. Below are the exact rebuild recipes, the ensemble/dZ policy, the lab commits, and the MAST training data.

We are packaging these into a new tagged commit on `catskills_fec` (`scripts/phase3_rebuild_all_d3d.sh`, the missing `o1g_*` / `build_grand_features` / `psi_tailfit` entrypoints, MAST shot/QC lists under `weights/phase3/mast_data/`, and a git bundle of the lab commits). Until that tag lands, this email is the authority.

---

## 1. DIII-D — training every shipped member

**Environment (all jobs):**

```bash
export CRC_FEQ_MODEL_ID=CRC_FEQ_olive_geom   # except brown TF tags → CRC_FEQ_brown_TF via psi_tailfit parent
export CRC_FEQ_THOMSON_LAYOUT=zgrid
# Apply the Phase-3 ZGRID freeze (±1.2 / 32 bins for 9db5e666), e.g.:
#   python -m catskills.zgrid_discover --out weights/phase3/zgrid.json
#   export CRC_FEQ_ZGRID_LO=… CRC_FEQ_ZGRID_HI=… CRC_FEQ_ZGRID_N=32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**Single script (preferred):** `scripts/phase3_rebuild_all_d3d.sh`  
One-liner sequence (no shipped `.pt` required except the Olive O1 parent for brown tailfit — see §1.4):

### 1.1 Feature caches

| Cache | Width | How built |
|---|---|---|
| `{key}_thz.npz` | 149 | Charcoal / warm feature cache on `diii_d_train` with `CRC_FEQ_THOMSON_LAYOUT=zgrid` |
| `{key}_thz_ipshift.npz` | 149 | `python -m catskills.scripts.o1g_clean_train train` (rebuilds Ip-corrected clock; also trains scratch) |
| `{key}_grand.npz` | **169** = 149 + 20 path signatures | `python -m catskills.scripts.build_grand_features` (appends level-2 actuator signatures to ipshift) |

### 1.2 Clean-Ip dualctx members (seed 42 / 43 / 44 / 45)

Recipe (88 cosine epochs, batch 2048, `placement_early`, λ_geom=0.020, n_pca=60, trunk 1024×8, dropout 0.05). Data seed stays **42**; only `--init-seed` changes. Charcoal writes `_best.pt` and `_final.pt`.

```bash
python -m catskills.scripts.o1g_clean_train train          # → O1G_clean_ip_scratch_{best,final}
python -m catskills.scripts.o1g_clean_train_seed43 train   # → O1G_clean_ip_seed43_{best,final}
python -m catskills.scripts.o1g_clean_train_seed44 train
python -m catskills.scripts.o1g_clean_train_seed45 train
```

Checkpoint metadata (shipped): n_train=6961, n_train_frames=1 540 767, thomson_layout=zgrid.

### 1.3 O1G_grand (169-column)

```bash
python -m catskills.scripts.build_grand_features   # if not already done
python -m catskills.scripts.o1g_grand_train
# env inside script: CRC_FEQ_GRAND=1 CRC_FEQ_IP_SHIFT=1 CRC_FEQ_LIMITED_MIXUP=1
# arch=grand, init-seed=46, batch=2048, geom-warmup=14, trunk 1536×8 / ctx 768×3, n_pca=60, 88 ep
```

### 1.4 Legacy geometry ladder + early@256

```bash
python -m catskills.scripts.o1_geom_train control
python -m catskills.scripts.o1_geom_train place
python -m catskills.scripts.o1_geom_train early --batch 256    # → O1G_early
python -m catskills.scripts.o1_geom_train early --batch 2048   # → O1G_early_b2048
```

(`reproduce.sh --train` already ran control / place / early@2048 only.)

### 1.5 Brown `TF_LS_O1_101` and `TF_LS_O1_W1`

These are **psi_tailfit** fine-tunes of parent Olive O1 (`CRC_FEQ_olive_O1_dualctx`), not O1G:

```bash
# Requires CRC_FEQ_olive_O1_dualctx.pt (shipped under weights/phase3/ in the follow-up tag;
# Olive dualctx stack at lab commit 65cfe77). Place it in artifacts/ or CRC_FEQ_WEIGHTS_DIR.
python -m catskills.scripts.psi_tailfit --parent O1 --power 0.0 --seed 101 \
  --tag LS_O1_101 --epochs 220 --patience 30
# → CRC_FEQ_brown_TF_LS_O1_101.pt

python -m catskills.scripts.psi_tailfit --parent O1 --power 0.0 --seed 42 \
  --lambda-w 1 --tag LS_O1_W1 --epochs 220 --patience 30
# → CRC_FEQ_brown_TF_LS_O1_W1.pt
```

Receipts: `results/brown/P30_TAILFIT_LS_O1_{101,W1}.json`.

---

## 2. DIII-D — ensemble weights and dZ = +0.5 mm

**Keep as frozen constants in `generate_final_predictions.py`. Do not refit for a rebuild that targets the submitted private score.**

| Knob | Value | Source |
|---|---|---|
| Flux simplex weights `W` | `[0.3668, 0.2680, 0.0585, 0.2161, 0.0560, 0.0345]` (renormalized) | Nelder–Mead polish on holdout after SLSQP (`solve_ensemble_weights` / grand-master blend); **locked for Final** |
| Legacy slot | `W[5]` split equally across the five `LEGACY_MEMBERS` | Same file |
| Scalar heads | Equal mean of four clean-Ip heads (scratch best/final, seed43 best/final) | Same file |
| Vertical micro-shift | `DZ_MM = 0.5` → `psi_new(R,Z) = psi_old(R, Z+0.5 mm)` via `map_coordinates` | Same file |

Refitting `W` on organizer holdout would change the submission identity. Rebuild = retrain members → run **the same** `generate_final_predictions.py` constants.

---

## 3. Code versions / lab commits

Those SHAs are **not** on `catskills_fec` history; they are from the lab working tree `fusion-equilibrium-challenge` recorded inside each `.pt` as `git_commit`:

| Commit | Used by |
|---|---|
| `65cfe77…` | brown `TF_LS_O1_{101,W1}` (Olive era) |
| `8cfd0b3…` | `O1G_early`, `O1G_place` |
| `b686e27…` | `O1G_early_b2048` |
| `43144a1…` | `O1G_grand` |
| `213e9d2…` | clean_ip scratch + seed43 |
| `ae8ace4…` | clean_ip seed44 |
| `fb79bee…` | clean_ip seed45 |

**`03becd27` does not bit-reproduce those historical trains**; it is the Phase-3 packaging commit (reproduce entrypoints + shipped weights). Training *recipes* above match the checkpoint `training_params` and the lab scripts at those commits.

We are shipping `weights/phase3/lab_training_commits.bundle` so you can:

```bash
git clone <empty> lab_tree && cd lab_tree
git bundle unbundle /path/to/lab_training_commits.bundle
# or: git fetch lab_training_commits.bundle 'refs/heads/lab/*:refs/heads/lab/*'
```

---

## 4. MAST — M9 training data

### 4.1 Shot list / QC → 3 622 train shots

Shipped under `weights/phase3/mast_data/`:

- `mast_pool3.txt` — 3 738 FAIR-MAST shot ids present in our `fair_mast_train` (space-separated; replaces `/tmp/mast_pool3.txt`)
- `mast_qc_keep.txt` / `mast_qc_v1_index.json` — QC keep list (**3 622** keys) + thresholds

**How 3 622 arises** (`catskills.scripts.mast_qc_index`):

1. Load all `mast_features_v1` keys except the three starter-kit demo holdouts.
2. Shuffle with seed **42**; seal first 80; remainder = train pool (`n_train` = 3 655 in our index).
3. Fit maxabs fence on train: `median + 10·MAD` (`maxabs_hi ≈ 2.981`).
4. **keep** = train shots that pass (`n_keep` = **3 622**).  
5. M9: `python -m catskills.run_crc_feq_pinn_mast --qc --shot-balanced --mask-edge --win --seed 42 --init-seed {101,202,303} --epochs 50 --tag M9_win50_i{101,202,303}`  
   With `--qc`, train keys = that keep list → checkpoint `train_shots: 3622`.

### 4.2 Building `fair_mast_train`

```bash
# Pool file (now in the package):
export POOL=weights/phase3/mast_data/mast_pool3.txt
# Ingest from public FAIR-MAST S3 (CC BY-SA 4.0); challenge schema + efit_psirz for offline QC only
python -m catskills.scripts.mast_ingest --pool "$POOL" --start 0 --shots 4000 \
  --out "$LAB/datasets/fair_mast_train"
# Warm features → QC → win index → M9
python -m catskills.scripts.phase3_prepare_mast_caches
python -m catskills.scripts.phase3_train_mast_m9 --weights-dir weights/phase3
```

**Filtering:** ingest keeps archive shots that successfully map into the challenge parquet layout (magnetics + Thomson + EFIT grid). QC then drops amplitude outliers as above. **M9 never uses EFIT ψ / LCFS / scalars as training targets** (Rule 3); features are sensors only; QC uses feature-space maxabs, not equilibrium labels.

---

## 5. What we will tag next

1. `scripts/phase3_rebuild_all_d3d.sh` + ported `o1g_*` / `build_grand_features` / `psi_tailfit`.  
2. `weights/phase3/mast_data/{mast_pool3.txt,mast_qc_*}`.  
3. `lab_training_commits.bundle`.  
4. Wire `reproduce.sh --train` to call the full DIII-D rebuild (or document `--train-full`).

Weights-only replay (`./reproduce.sh --weights-only`) remains the path that matches submitted S within 1e-4; from-scratch is the recipes above.

Best regards,  
Catskills Research Company (`catskills`)
