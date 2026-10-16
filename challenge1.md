# Catskills Research Company — Challenge 1 (Intra-machine DIII-D)

**Team:** Catskills Research Company  
**Goal:** Maximize composite $S$ on DIII-D — win Award #1  
**Metric:** v3.1.0 (vendored `fusion_scoring/` / `local_score.py`)  
**Companion:** [`challenge2.md`](challenge2.md) (zero-shot MAST — deferred focus)

---

## 1. Objective

Reconstruct $\psi(R,Z)$ plus $q_{95}$ and $\beta_N$ from coil currents and Thomson on **DIII-D only**.

$$
S = 0.55\cdot\max(0,R^2_\psi)
  + 0.15\cdot\max(0,R^2_{\{q_{95},\beta_N\}})
  + 0.10\cdot(1-\min(1,D_{\mathrm{LCFS}}))
  + 0.20\cdot\mathrm{Consistency}
$$

Submit `psirz`, `q95`, `betaN`. LCFS / axis / shape / $\ell_i$ are derived from $\psi$ by the scorer.

One Codabench upload can carry both machines; Challenge 1 ranking only needs a strong DIII-D file. We still emit MAST placeholders in Route B zips so the same package remains dual-eligible later (see `challenge2.md`).

---

## 2. Sample solutions — what they are

**zeros / ridge_pca / mlp_pca** are **challenge baselines** (not Catskills-original models). We re-implement and package them for Codabench; label them `baseline:*`. **`CRC_FEQ_black`** is the first Catskills luminance revision (see `catskills/MODEL_DESCRIPTION_CRC_FEQ_black.md`, color chart in `CRC_FEQ_COLORS.md`).

| Method | Mechanism | Differentiator |
|--------|-----------|----------------|
| **zeros** | Constant-zero $\psi$, $q_{95}$, $\beta_N$ | Submission plumbing / floor ($S=0$) |
| **ridge_pca** | Coil features → `StandardScaler` → PCA(40) → `RidgeCV`; Ridge scalar heads | Challenge linear baseline; B ∝ coil current prior |
| **mlp_pca** | Same PCA target space; `MLPRegressor(256,128)` flux head | Challenge nonlinear baseline in compressed flux space |
| **CRC_FEQ_black** | Same as ridge_pca **plus** soft Grad–Shafranov residual scale nudge at inference | **Catskills** luminance line (black); *not* full residual backprop yet |

All trained models in this smoke table used **5 DIII-D train shots** (shuffled stream) and magnetics-only features (no Thomson). Packaged Route B zips live in dated attempt folders under `benchmarks/<method>/` (beside `REPORT.md`).

---

## 3. Challenge 1 baseline scores

### 3.0 Robust local benchmark (preferred)

Use labelled Hugging Face `diii_d_train` with a **random shot-level** train/holdout split and the **official** composite (same functionals as Codabench). Holdout is still ⊂ train corpus — not the public/private test folds — but it is the right pre-submit comparison.

```bash
export HF_TOKEN="$(tr -d '\n\r' < ../huggingface.token)"
poetry run python -m catskills.local_benchmark \
  --pool 80 --eval-shots 15 --seed 42 \
  --methods zeros ridge_pca mlp_pca
```

Outputs JSON under `results/challenge1_local_benchmark/` (includes `train_s` / `infer_s` / `score_s`).  
Scale `--pool` toward hundreds as RAM allows (~20 MB/shot). Full 7k-shot fits need a local download + more disk/RAM, not a change in metric.

### 3.1 Smoke table (tiny — plumbing only)

**Protocol:** 5-shot train + 3 contiguous eval shots (`skip=20`). Harness anchors: perfect → $S=1$, zeros → $S=0$.  
Treat as plumbing proof, not a method ranking.

**What the columns mean (plain English)**

| Column | Weight in $S$ | Dummy-friendly meaning |
|--------|--------------:|------------------------|
| **Method** | — | Which sample solution produced the predictions. |
| **$S$** | 100% | The **official leaderboard number** for Challenge 1. Higher is better; 1.0 is perfect, 0 is “no credit.” It is a weighted blend of the four terms below (negative $R^2$ values are clipped to 0 before mixing). |
| **$R^2_\psi$** | 55% | “Did we get the **flux-map image** right?” Compares every pixel of your 65×65 $\psi$ to EFIT’s. **1.0** = identical; **0** = no better than predicting the average map; **negative** = worse than that average (common when the model’s amplitude/scale is wrong). |
| **$R^2_{\{q95,\beta_N\}}$** | 15% | “Did we get the **two submitted numbers** right?” Edge safety factor $q_{95}$ and normalized beta $\beta_N$ — the only scalars you hand in as values (a flux map alone cannot contain them). Same $R^2$ intuition: 1 good, ≤0 useless after clipping. |
| **$1-D_{\mathrm{LCFS}}$** | 10% | “Is the **plasma outline** in the right place?” The scorer traces the last closed flux surface (the edge of the plasma blob) from your $\psi$ and from truth, measures how far apart those outlines are ($D_{\mathrm{LCFS}}$), then reports $1-D$ so **higher is better**. 1.0 = outlines match; 0 = as bad as the metric allows (or no closed contour found). |
| **Consistency** | 20% | “If we **read geometry off your $\psi$** the same way EFIT does, do we get the right answers?” Axis position, elongation, triangularity, volume, internal inductance $\ell_i$ are *not* submitted — they are derived from your map. This term averages how well those derived numbers match the same derivation on the true map. High Consistency means your $\psi$ isn’t just blurry-correct in pixels; it implies the right plasma shape. |

| Method | $S$ | $R^2_\psi$ | $R^2_{\{q95,\beta_N\}}$ | $1-D_{\mathrm{LCFS}}$ | Consistency |
|--------|----:|----------:|------------------------:|----------------------:|------------:|
| perfect (harness) | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 |
| zeros | **0.000** | −0.019 | −7.24 | 0.000 | 0.000 |
| ridge_pca | **0.070** | −5.48 | −21.7 | 0.696 | 0.000 |
| **mlp_pca** | **0.350** | **0.482** | −21.7 | **0.847** | 0.000 |
| CRC_FEQ_black | **0.070** | −5.39 | −21.7 | 0.697 | 0.000 |

Raw logs / JSON: `results/challenge1_baseline_scores/`.

Reproduce:

```bash
export HF_TOKEN="$(tr -d '\n\r' < ../huggingface.token)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
poetry run python -m catskills.score_challenge1_baselines \
  --n-shots 3 --skip 20 --methods zeros ridge_pca mlp_pca pinn_gs
```

---

## 4. Compare, contrast, and opinion

### 4.1 zeros vs everyone

Zeros fail LCFS extraction (100%) and score $S=0$. Useful only as a Codabench format / Route B smoke. Any nontrivial flux that yields a closed contour immediately picks up the $0.10\cdot(1-D_{\mathrm{LCFS}})$ term.

### 4.2 ridge_pca vs mlp_pca (main contrast)

With **only five training shots**, MLP dominates Ridge on the terms that matter for $S$:

- **Flux fidelity:** MLP reaches $R^2_\psi\approx 0.48$ (contributes $\sim 0.26$ to $S$). Ridge is deeply negative ($R^2_\psi\approx -5.5$ → clipped to 0 in the composite). Linear maps under-capacity / wrong amplitude when the PCA basis is estimated from almost no data.
- **Boundary:** Both extract LCFS on every frame; MLP’s Hausdorff term is better ($1-D\approx 0.85$ vs $0.70$). Nonlinear coupling among coils helps shape, not just pixel MSE.
- **Scalars / Consistency:** Both are **zero contribution**. Scalar heads are garbage ($R^2_{\{q95,\beta_N\}}\ll 0$), and every ψ-derived Consistency $R^2$ is negative → clipped. Contours exist, but axis / κ / volume / $\ell_i$ implied by $\hat\psi$ do not match EFIT.

**Opinion:** For Challenge 1 smoke under extreme data poverty, **mlp_pca is the only sample that looks like a real model** ($S\approx 0.35$ from flux + LCFS alone). Ridge’s “physics is linear” story needs hundreds of shots before it should beat a small MLP; that is still the preferred compute-light horse once we scale training data.

### 4.3 `CRC_FEQ_black` vs ridge_pca

`CRC_FEQ_black` (first Catskills luminance revision) is Ridge plus an inference-time amplitude search against a soft $\Delta^*$ residual. Smoke scores were **statistically identical** to ridge_pca ($S\approx 0.070$). The GS nudge did not fix amplitude collapse or Consistency. See `catskills/MODEL_DESCRIPTION_CRC_FEQ_black.md`.

**Opinion:** Soft post-hoc GS scaling is not a winning ingredient. Challenge 1 needs either (a) enough data that Ridge/PCA already gets $R^2_\psi$ positive, or (b) a **training-time** hybrid loss (true residual backprop / neural $p',FF'$) that moves Consistency — the 0.20 weight we currently leave on the table. Keep the physics module; do not expect **black** to move the leaderboard.

### 4.4 Shared failure modes (all learned models)

1. **Scalar heads broken / undertrained** — $q_{95}$, $\beta_N$ dominate a 0.15 weight and currently contribute nothing.  
2. **Consistency = 0** — even MLP’s decent $R^2_\psi$ does not yet imply EFIT’s geometric scalars. Optimizing pixel/PCA MSE ≠ optimizing the official 0.20 term.  
3. **5-shot training** — numbers are a lower bound on method quality, not a ceiling. Next: ≥500–2000 shots, shot-level val, early-stop on `local_score` $S$ not MSE.

### 4.5 What we should scale first for a Challenge 1 win

1. Retrain **mlp_pca** and **ridge_pca** on large DIII-D (Thomson on).  
2. Add scalar heads that work (or multi-task aux).  
3. Score-aware training / selection via `local_score.py`.  
4. Only then invest in training-time PINN / consistency surrogates.  
5. Full public-test Route B zip (`--max-shots 0`) when Codabench is live.

### 4.6 Model-planning note — PINN terms for every input

When we move to a real training-time hybrid / PINN loss, **do not only hang a bare $\Delta^*\hat\psi$ residual on the map**. Tie each challenge **input** to a physics term in the objective so the network is pressured to use that channel, not ignore it:

| Input | Physics / PINN-style term to put in the loss |
|-------|-----------------------------------------------|
| **Coil currents** (F-coils, ECOILA, bcoil, …) | Free-boundary / vacuum coupling: exterior / boundary $\psi$ should be consistent with the poloidal field those coils produce (Green-function / mutual-inductance / Biot–Savart style mismatch — “coil → vacuum $B$ → boundary $\psi$”). |
| **Plasma current $I_p$** | Global Ampère constraint: $\int J_\varphi\,dA \approx I_p$ (with $J_\varphi$ from the GS right-hand side / $p',\ FF'$), not only a soft pixel match on $\psi$. |
| **Thomson $n_e$, $T_e$** | Kinetic pressure source: $p\sim n_e T_e$ (or a learned EOS) must agree with the $p'(\psi)$ that enters Grad–Shafranov; equivalently a force-balance / flux-surface pressure consistency term so Thomson is not a free spectator feature. |
| **All of the above → $\hat\psi$** | Interior Grad–Shafranov residual $\Delta^*\hat\psi + \mu_0 R^2 p'(\hat\psi) + FF'(\hat\psi)\approx 0$ (training-time, not post-hoc amplitude nudge). |

Optional auxiliaries (still physics-linked, not Codabench terms): $q_{95}$ from $\hat\psi$ geometry vs submitted head; $\beta_N$ vs volume-averaged $p/B^2$ from the same kinetic + field picture. Official $S$ still does not score the PDE residual — these terms are **training regularizers** so $\hat\psi$ stays equilibrium-like and input-faithful.

---

## 5. Packaging — Route B (always)

Upload **`direct_submission.zip`** to Codabench. We and the organizers always use this
self-contained zip (both `.npz` at archive root, no HF pointer). See
[`catskills/SUBMISSION.md`](catskills/SUBMISSION.md). Do not use the HF pointer path
unless explicitly requested.

```bash
poetry run python -m catskills.package_submissions \
  --methods mlp_pca ridge_pca zeros pinn_gs \
  --max-shots 0 --n-train-shots 500 --validate
```

---

## 6. Separability note

Challenge 1 research can proceed **without** Challenge 2 features. Shared: dataset access, scorer, submission contract. See `challenge2.md` for why we keep one repo and defer MAST transfer work.

---

## 7. Challenge vocabulary (dummy-friendly)

| Term | Spelled out | What it is |
|------|-------------|------------|
| **Tokamak** | — | Donut-shaped fusion device; plasma held by magnetic fields. |
| **DIII-D** | “D-three-D” | General Atomics tokamak in San Diego; Challenge 1 training machine. |
| **MAST** | Mega Amp Spherical Tokamak | UK spherical tokamak; Challenge 2 zero-shot target (different shape/coils). |
| **Shot** | — | One plasma discharge / pulse. |
| **EFIT** | Equilibrium Fitting | Classic reconstruction code; its $\psi$ maps are our **ground truth** labels. |
| **$\psi$ / psirz / flux map** | Poloidal flux on $(R,Z)$ | 65×65 “topographic map” of magnetic surfaces — the main thing you predict. |
| **$R$, $Z$** | Major radius, vertical height | Physical coordinates in meters, not pixel indices. |
| **LCFS** | Last Closed Flux Surface | Outer closed contour = plasma boundary / “edge of the blob.” |
| **X-point** | — | Magnetic null that lets the plasma divert heat to the divertor; kidney-bean topology. |
| **dsep** | divertor / X-point gap | Signed gap (meters): **>0 diverted**, **<0 limited** (touching a wall). |
| **Thomson / TS** | Thomson scattering | Laser diagnostic for electron **temperature $T_e$** and **density $n_e$** (profiles over channels × time — **not** scalars). |
| **$T_e$, $n_e$** | Electron temperature, density | Kinetic **inputs** (plus coils); **not** the scored target. |
| **Coil currents** | Magnetics | Currents in shaping / ohmic / toroidal coils (e.g. **ECOILA**, **F1A**…**F9B**, **bcoil**, **Ip**). |
| **ECOILA** | Ohmic solenoid coil (DIII-D naming) | Central solenoid current — drives plasma current; a 1D time series. |
| **Ip** | Plasma current | Total current in the plasma (kA). |
| **$q_{95}$** | Edge safety factor | How field lines wind near the edge; **you submit this scalar**. |
| **$\beta_N$** | Normalized beta | Plasma pressure vs magnetic pressure (normalized); **you submit this scalar**. |
| **$\ell_i$** | Internal inductance | How peaked the current profile is; **derived from $\psi$ by the scorer**, not submitted. |
| **PCA** | Principal Component Analysis | Compress 65×65 maps into ~40 coefficients before Ridge/MLP. |
| **PINN** | Physics-Informed Neural Network | Model trained with a PDE residual in the loss (here: Grad–Shafranov). |
| **GS** | Grad–Shafranov | Equilibrium PDE that $\psi$ should obey; same physics on every machine. |

**Challenge story:** Magnets and Thomson tell you how the machine was driven and how hot/dense the electrons were. We must learn $\psi$ **without** magnetic wall diagnostics — only coils + Thomson + Ip — then submit $\psi$, $q_{95}$, $\beta_N$. Smell-check predictions with dated local benchmark reports (§8), not an external GUI.

---

## 8. Full local dataset (CRC)

Download all Hub configs once onto the data drive (~7041 train + public tests):

```bash
source crc_fusion_lab.env
poetry run python -m catskills.download_dataset
# status: poetry run python -m catskills.download_dataset --verify-only
```

Mirror path: `$CATSKILLS_FUSION_LAB/datasets/Sophelio__fusion-equilibrium-challenge`  
Catskills loaders prefer that path automatically when present (else Hub streaming).

---

## 9. Local attempts (report + submission, one folder)

Each dated attempt is a **self-contained package** under CRC:

`$CATSKILLS_FUSION_LAB/benchmarks/<method>/<model_version>_<UTC>/`

That folder holds **both** the smell-check report and the Codabench upload:

| File | Role |
|------|------|
| `REPORT.md` + `figures/` | Dated attempt log / GT–Pred figures |
| `scores.json` / `meta.json` | Holdout $S$ + run identity |
| `direct_submission.zip` | **Upload this to Codabench** (Route B) |
| `diii_d_public_test.npz` / `mast_public_test.npz` | Prediction files inside the zip |

**Model version** string: `{catskills.__version__}+{method}+{git_shortsha}` (e.g. `0.1.0+mlp_pca+c1226a3`), stamped on the trained bundle and report `meta.json`.

Each report includes:

- Official holdout $S$ breakdown (`scores.json`)
- Figure pack: **Inputs** first (coils, Thomson, dsep, EFIT $\ell_i$/axis context — one panel each, with definitions); then **Outputs** with **GT | Pred | Diff** only for $\psi$ and submitted $q_{95}$/$\beta_N$
- Heuristic qualitative bullets (edge vs core error, amplitude scale, $q_{95}$/$\beta_N$ RMSE)

```bash
# produced automatically by run_baseline_mlp_pca (unless --skip-benchmark-report)
poetry run python -m catskills.run_baseline_mlp_pca \
  --max-train-shots 25 --eval-shots 3 --max-package-shots 2 --benchmark-shots 2

# or from an existing bundle:
poetry run python -m catskills.benchmark_report \
  --bundle /media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/artifacts/baseline_mlp_pca.pkl \
  --max-train-shots 25 --eval-shots 3 --max-figure-shots 2
```

Dated competition attempt docs live on the data drive (`benchmarks/*/REPORT.md`). For the GPU residual trainer (`CRC_FEQ_navy`), use local **TensorBoard** under `results/tensorboard/CRC_FEQ_navy/` for train/val curves — that does **not** replace the attempt report.

Starter-kit leftovers (`fusion_data_provider.py`, demo `parquet_data/`) may remain in the repo for schema reference; Catskills visualization/smell-check goes through §8 only.
