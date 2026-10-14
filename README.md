# Catskills Physics-Informed Equilibrium Reconstruction

**One-Shot Forward (OSF) Modal Pipeline for the Fusion Equilibrium Challenge (NeurIPS)**

This repository contains the standalone codebase for the **Catskills Physics-Informed Equilibrium Reconstruction Pipeline**. The model reconstructs 2D magnetic flux surfaces $\psi(R, Z)$ and global profile scalars ($q_{95}, \beta_N$) directly from diagnostic magnetic sensor time-series and Thomson scattering measurements without iterative non-linear Grad-Shafranov PDE loop overhead.

---

## Architecture & Physics Formulation

```
                       Diagnostic Magnetics & Thomson Features (64-dim)
                                            │
                                            ▼
                     ┌──────────────────────────────────────────────┐
                     │ Multi-Head Residual MLP (MultiHeadResMLP)    │
                     │   Stem: Linear(64->256) -> LayerNorm -> SiLU  │
                     │   ResBlock 1 & 2: 256->256 with SiLU Skips   │
                     └──────────────┬────────────────────────┬──────┘
                                    │                        │
                                    ▼                        ▼
                     ┌────────────────────────┐    ┌────────────────────────┐
                     │ Modal Head (256->24)   │    │ Profile Head (256->2)  │
                     │   Outputs c_k(t)       │    │   Outputs q95, beta_N  │
                     └──────────────┬─────────┘    └────────────────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────────────────────┐
                     │ Biot-Savart Reconstruction & External Field  │
                     │   ψ = c(t) @ (Φ @ P^T) + f_dyn41 @ W_ext     │
                     └──────────────────────────────────────────────┘
```

### 1. 2D Modal Current Expansion & Biot–Savart Operator
The plasma toroidal current distribution is expanded into a $K=24$ spatial modal basis:
$$j_\varphi(R, Z, t) = \sum_{k=1}^{24} c_k(t) \phi_k(R, Z)$$

The plasma poloidal flux $\psi_{\mathrm{plasma}}$ is evaluated via the discrete Biot–Savart Green's response operator $P \in \mathbb{R}^{4225 \times 3969}$:
$$\psi_{\mathrm{plasma}}(R, Z, t) = \sum_{k=1}^{24} c_k(t) \cdot (\phi_k P^T)$$

Every reconstructed plasma flux surface rigorously satisfies the elliptic PDE $\Delta^* \psi_{\mathrm{plasma}} = -\mu_0 R j_\varphi$, guaranteeing $C^\infty$ smoothness and eliminating high-frequency numerical artifacts.

### 2. Calibrated 41-Feature Dynamic External Response ($W_{\mathrm{ext\_dyn41}}$)
Static coil response matrices omit inductive coil ramping rates ($\dot{I}$) and plasma current ($I_p$), causing passive vacuum vessel eddy current distortions, a $>40\,\mathrm{mWb}$ core flux depression, and severe elongation ($\kappa$) / volume contour dilation.

The pipeline utilizes a calibrated 41-feature dynamic Ridge regression matrix:
- **19 shaping coil currents**: `ECOILA` + `F1A`–`F9B`
- **19 coil derivatives ($\dot{I}_{\mathrm{coil}}$)**: Eddy current proxy for vessel time constants
- **1 toroidal field current**: `magnetics_bcoil`
- **1 plasma current estimate**: $I_p$
- **1 constant bias intercept**

This reduces external field error by $>63\%$ ($R^2 = 0.9276$ on holdout), flipping $R^2(\kappa)$ from $-1.501 \to \mathbf{+0.3533}$ and recovering $R^2(\text{volume})$ by $+2.110$.

---

## Benchmark Progression (Official 80-Shot Holdout Benchmark)

Evaluated across all 21,500 frames of the official 80-shot holdout using the official Marching Squares composite scorer:

| Metric | Pre-Step 4 (`d3d_0.0.17_fleet_full`) | Step 4 Calibrated ($W_{\mathrm{ext\_dyn41}}$) | Delta / Status |
| :--- | :---: | :---: | :---: |
| **Composite Score $S$** | $0.69594$ | **$0.71842$** | **$+0.02248$** |
| **Overall Consistency** | $+0.43502$ | **$+0.52184$** | **$+0.08682$** |
| **Elongation $\kappa$** | $-1.5011$ | **$+0.3533$** | **$+1.8544$ (negative $\to$ positive)** |
| **Plasma Volume** | $-3.9973$ | **$-1.8869$** | **$+2.1104$ improvement** |
| **Magnetic Axis $R_{\mathrm{axis}}$** | $+0.2354$ | **$+0.3120$** | **$+0.0766$** |
| **Magnetic Axis $Z_{\mathrm{axis}}$** | $+0.9690$ | **$+0.9702$** | Near-perfect vertical tracking |
| **Lower Triangularity $\delta_{\mathrm{bot}}$** | $+0.7039$ | **$+0.7042$** | Robustly $> 0.70$ |
| **Upper Triangularity $\delta_{\mathrm{top}}$** | $+0.5398$ | **$+0.5391$** | Preserved $> 0.53$ |
| **Internal Inductance $\ell_i$** | $+0.5971$ | **$+0.5980$** | Robustly $> 0.59$ |
| **Profile $R^2_{\{q,\beta\}}$** | $0.99504$ | **$0.99504$** | Unchanged high fidelity |
| **LCFS Fail Fraction** | $0.00\%$ | **$0.00\%$** | 100% closed equilibria |

---

## Repository Structure

```
├── README.md                      # Architecture and operational documentation
├── LICENSE                        # Catskills Research Company proprietary license
├── pyproject.toml                 # Package and build specifications
├── requirements.txt               # Core runtime dependencies
├── requirements-pytorch.txt       # CUDA PyTorch dependency spec
│
├── train_fleet_full.py            # High-occupancy full fleet OSF model training
├── fit_dynamic_wext.py            # Dynamic 41-feature external matrix calibration
├── calibrate_probe_100.py         # 100-shot calibration probe & feature extraction
├── local_score.py                 # Official competition metric benchmark scorer
├── validate_submission.py         # Submission format and shape validator
├── export_calibrated_osf_d3d.py   # Public test submission exporter
├── push_predictions.py            # Codabench Route A (Hugging Face pointer) publisher
│
├── src/feq/                       # Data loading and physics abstractions
│   ├── data.py                    # Parquet loader and EFIT target definitions
│   ├── labdisk.py                 # Lab disk path management
│   ├── physics.py                 # Fundamental constants
│   ├── timebase.py                # Magnetics / plasma current clock alignment
│   └── machine.py                 # Machine geometry constants and sensor layouts
│
├── fusion_scoring/                # Official competition evaluation suite
│   ├── common.py                  # Scoring weights and constants
│   ├── contour.py                 # Symmetric Hausdorff distance calculations
│   ├── derive.py                  # Axis, LCFS, volume, elongation, triangularity
│   ├── lcfs.py                    # Level-set marching & boundary extraction
│   ├── metrics.py                 # Official composite score functionals
│   └── masks/                     # Vacuum vessel limiter/divertor boundary masks
│
├── catskills/                     # Core physics and dataset utilities
│   ├── gs_solver.py               # Biot-Savart operator & PDE solvers
│   ├── vacuum_green.py            # Green's function elliptic filament integrals
│   ├── soft_derive.py             # Differentiable boundary/scalar functionals
│   ├── features.py                # Centered magnetics windowing + Thomson interpolation
│   ├── feature_cache.py           # On-disk feature cache manager on lab drive
│   ├── dataset_local.py           # Local parquet dataset mirror resolver
│   ├── lab_paths.py               # Lab drive directory resolution
│   └── local_benchmark.py         # Holdout scoring harness
│
└── fec_site/                      # Interactive web viewer and live inference
    ├── app.py                     # Flask web viewer server
    ├── d3d_infer.py               # Real-time OSF model inference pipeline
    ├── diag.py                    # Diagnostic extraction and time-traces
    └── score.py                   # Single-shot metric evaluator
```

---

## Lab Drive Architecture

All generated artifacts, model checkpoints, tensor caches, and parquet mirrors reside on the external laboratory drive:
- **Lab Root**: `/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/`
- **Model Checkpoints**: `results/d3d_0.0.17_fleet_full/`
  - `weights_best.pt`: PyTorch weights for `MultiHeadResMLP`
  - `weights.npz`: Normalization scalers and modal basis
  - `W_ext_dyn41.npz`: Calibrated 41-feature dynamic external matrix
- **Feature Caches**: `cache/features_v1/`
- **Precomputed Green's Operators**: `cache/gs_response_v1/P_d3.npy`
- **Dataset Parquets**: `datasets/Sophelio__fusion-equilibrium-challenge/data/`

---

## Quickstart

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-pytorch.txt
```

### 2. Launch the Interactive Web Viewer
Run the web viewer to inspect equilibria, scalar traces, and consistency breakdowns across shots:
```bash
python -m fec_site
```
Open `http://127.0.0.1:8770/d3d/labelled` in your browser.

### 3. Evaluate on the Official Holdout Benchmark
```bash
python calibrate_probe_100.py
```

### 4. Train the Full Fleet OSF Model
```bash
python train_fleet_full.py
```

### 5. Calibrate the Dynamic External Field Matrix
```bash
python fit_dynamic_wext.py
```

### 6. Package and Validate Public Test Submissions
```bash
python export_calibrated_osf_d3d.py
```
This evaluates all 874 public test shots, packages them alongside the pinned MAST 0.9.1 paramagnetic upload, validates both files with `validate_submission.py`, and stages the submission to `/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-cleanroom/outputs/diii_d_public_test.npz`.

### 7. Push Route A Pointer to Codabench
```bash
./catskills/scripts/push_hf_submission.sh
```
Upload the resulting `<attempt-dir>/submission_pointer.zip` directly to Codabench.
