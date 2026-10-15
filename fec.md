# Fusion equilibrium challenge

- [Introduction](introduction.md)
- [Tokamak Theory and Equilibrium Physics](tokamak-theory.md)
  - [Shape of a Tokamak](tokamak-theory.md#shape-of-a-tokamak)
  - [Basic Electromagnetic Quantities](tokamak-theory.md#basic-electromagnetic-quantities)
  - [Tokamak Shape and Poloidal Plane](tokamak-theory.md#tokamak-shape-and-poloidal-plane)
  - [What is a Shot and a Clock?](tokamak-theory.md#what-is-a-shot)
  - [Plasma Kinetics](tokamak-theory.md#plasma)
  - [Poloidal Flux and LCFS](tokamak-theory.md#poloidal-flux-and-lcfs)
  - [Magnetic Axis and Nulls](tokamak-theory.md#magnetic-axis-and-nulls)
  - [MHD Force Balance and Flux Functions](tokamak-theory.md#mhd-force-balance)
  - [Derivation of the Grad–Shafranov Equation](tokamak-theory.md#derivation-of-the-gradshafranov-equation)
  - [The Grad–Shafranov Equation and Profiles](tokamak-theory.md#the-gradshafranov-equation)
- [Parquet Data Format and Machine Specifications](parquet-data.md)
  - [General Parquet Format](parquet-data.md#parquet-data-format)
  - [MAST Data](parquet-data.md#mast-data)
  - [DIII-D Data](parquet-data.md#diii-d-data)
- [Scoring](scoring.md)
- [Challenge Rules](challenge-rules.md)
- [Benchmark and Diagnostic Visualizations](benchmark-diagnostics.md)
- [One-Shot Forward Physics Model](one-shot-forward.md)
- [MAST Calibration of the One-Shot Model](mast-calibration.md)
- [DIII-D Calibration of the One-Shot Model](diii-d-calibration.md)
- [Differences in OSF Model Constants](kdiff.md)
- [Publicly Sourceable Constants](public.md)
- [Constants Requiring Calibration and Tuning](tuning.md)
- [DIII-D Machine Learning Fit to Labelled Data](diii-d-fit.md)
- [Phase 2: Final Phase and Reproduction Specification](#phase-2-final-phase-and-reproduction-specification)
  - [Timeline and Competition Scope](#timeline-and-competition-scope)
  - [Prize Structure and Scoring Rules](#prize-structure-and-scoring-rules)
  - [Organizer Reproduction Audit (Top 5 Rebuild)](#organizer-reproduction-audit-top-5-rebuild)
  - [Machine-Specific Retraining Architecture](#machine-specific-retraining-architecture)
  - [Turnkey Rebuild Execution (`reproduce.sh`)](#turnkey-rebuild-execution-reproducesh)
- [References](references.md)
- [Appendix: 3rd Party Software](appendix-software.md)

---

## Phase 2: Final Phase and Reproduction Specification

### Timeline and Competition Scope

The competition operates across two distinct phases on Codabench:

```
┌───────────────────────────────────────┬────────────────────┬───────────────────────────────────────┐
│  Phase 1: Development (Practice)      │     Transition     │  Phase 2: Final (Award Decision)      │
│  Ends: October 17, 2026 @ 8:00 PM EDT │  Oct 17 – Oct 19   │  Opens: October 19, 2026              │
│  Public test fold (874 D3D, 1206 MAST)│  Platform freeze   │  Closes: October 26, 2026 @ 00:00 UTC │
│  5 submissions/day, 100 maximum total │  Rebuild tolerance │  Blind private folds (3 submissions)  │
└───────────────────────────────────────┴────────────────────┴───────────────────────────────────────┘
```

1. **Development Phase (Practice Only)**:
   - Evaluated on public test folds (`diii_d_public_test.npz` and `mast_public_test.npz`).
   - Serves as an experimental testbed and diagnostic board.
   - Scores on the Development board do **not** decide winners or award prize money.
2. **Final Phase (Decides All Awards)**:
   - Evaluated on completely unseen, **blind private test folds** (`diii_d_private_test.npz` and `mast_private_test.npz`).
   - Each team is strictly limited to **up to three (3) total submissions** across the entire 7-day window.
   - The per-shot detailed results table is withheld; only aggregate metrics and scalar breakdowns are reported.

---

### Prize Structure and Scoring Rules

All prizes are decided exclusively on the blind private test fold in Phase 2:

| Award | Metric | Purse | Key Conditions & Gate Rules |
|---|---|---|---|
| **Award #1** | DIII-D Composite Score $S_{\mathrm{DIII\text{-}D}}$ | $20,000 | Scored on the blind private fold of DIII-D. |
| **Award #2** | Transfer Ratio $G_{\mathrm{ratio}} = \frac{S_{\mathrm{MAST}}}{S_{\mathrm{DIII\text{-}D}}}$ | $20,000 | Requires $S_{\mathrm{DIII\text{-}D}} \ge 0.85$ (below which $G_{\mathrm{ratio}} = 0$). Denominator is team's best DIII-D score across all scored Final entries. |
| **Honorable Mentions (4)** | Best $R^2_\psi$, Best $R^2_{\{q_{95}, \beta_N\}}$, Best Consistency, Lowest $D_{\mathrm{LCFS}}$ | Citation / Trophy | Judged exclusively on the MAST blind private fold using the entry nominated for Award #2. |

---

### Organizer Reproduction Audit (Top 5 Rebuild)

To prevent ungrounded claims, data contamination, or non-reproducible artifacts, the organizers enforce a rigorous offline verification audit (Forum Threads 2807 & 2827):

1. **Mandatory Code URL**: Every Final-phase submission must include a `Code URL` at submission time pointing to:
   - A public Git repository at a **pinned 40-character commit SHA**, or
   - A versioned archive identified by its **SHA256 checksum** (`url sha256:<hash>`).
2. **Top-5 Full Rebuild**: The organizers will independently clone and retrain the top 5 entries per challenge from scratch in an isolated offline environment.
3. **Compute Budget Ceiling**:
   - The entire training pipeline must finish within **24 GPU-hours on a single 80 GB A100-class GPU** (or equivalent CPU time).
   - Only GPU work counts against this cap. CPU data generation is uncapped but must be declared with hardware and wall-clock times.
4. **Data Provenance Statement (`PROVENANCE.md`)**:
   - A statement listing every external dataset, pre-trained model, and machine description per machine.
   - Strictly enforces Rule 3: **zero use of MAST/MAST-U equilibrium reconstructions (EFIT++ $\psi$, LCFS, scalars)**, zero fitting against the 3 demo shots, and zero use of the dilated LCFS envelope mask (`fusion_scoring/masks/mast_envelope.npz`).
5. **Rebuild Tolerance**: The rebuilt model must reproduce the submitted predictions within the rebuild tolerance published prior to the Final phase.

---

### Machine-Specific Retraining Architecture

Because the two machines operate under fundamentally different physical and data regimes, their retraining paths are cleanly decoupled:

#### 1. DIII-D Retraining (Supervised Neural Physics Pipeline)
- **Input Data**: 1.59 million training frames across all 6,961 shots in `diii_d_train`.
- **Architecture**: Multi-Head Residual MLP (`MultiHeadResMLP`) with 64-dimensional diagnostic magnetics and Thomson inputs, predicting 24 spatial modal current coefficients $c_k(t)$ and profile readouts ($q_{95}, \beta_N$).
- **PDE Response**: Plasma flux $\psi_{\mathrm{plasma}}$ evaluated via discrete Biot–Savart elliptic Green operator $P \in \mathbb{R}^{4225 \times 3969}$ ($\Delta^* \psi = -\mu_0 R j_\varphi$).
- **Dynamic Wall Eddies**: 40-feature dynamic Ridge regression matrix $W_{\mathrm{eddy\_dyn40}}$ ($\alpha = 500.0$) modeling vessel $L/R$ skin currents and polarity-aware coil response.
- **Compute Footprint**: ~25 minutes wall-clock on an RTX 5060 Ti (~10 minutes on A100), occupying 13.0 GB VRAM. Well within the 24 GPU-hour limit.

#### 2. MAST Model (Zero-Shot Analytic Physics Forward Solver)
- **Nature of Model**: Pure physics-informed One-Shot Forward (OSF) solver. Contains **zero neural network weights**, zero backpropagation, and consumes **0 GPU-hours** (< 2 minutes CPU time).
- **Frozen Hyperparameters**: The 24 physical constants (`tuning.md`, `osf/mast_12_osf_0_0_12.py`) derived during Phase 1 are treated as **frozen static hyperparameters** of the model definition:
  - *Shafranov vertical-field balance*: $\lambda_{V0} = 1.80$, $\lambda_{Vb} = -1.00$.
  - *Vessel $L/R$ eddy shielding* ($\tau_v \approx 20\,\mathrm{ms}$): $\lambda_{S0} = 0.50$, $\lambda_{Sb} = 0.10$.
  - *Ampère enclosed current fractions*: $k_0 = 0.70$, $k_w = 0.30$, $k_b = 0.15$, $k_d = 0.20$.
  - *CAD limiter boundaries*: $a_b = 0.42\,\mathrm{m}$, $a_d = 0.18\,\mathrm{m}$.
  - *Divertor approach gate & Thomson walk*: $\zeta_q = 2.00$, $\eta_\beta = 0.20$ (sensitivity gradients validated $\le 0$).
  - *Paramagnetic pitch integral factor*: $c^* = 1.0622$ (analytic vertex of spherical tokamak internal field amplification).
- **Rule 3 Offline Compliance**: During the Final Phase rebuild, the MAST solver operates completely offline with zero live Codabench API calls or leaderboard probe dependencies. The constants are fixed in code, guaranteeing 100% deterministic reproducibility.

---

### Turnkey Rebuild Execution (`reproduce.sh`)

For complete end-to-end transparency, the entire rebuild from raw inputs to verified submission archives is encapsulated in a single top-level entrypoint:

```bash
#!/usr/bin/env bash
# ==============================================================================
# reproduce.sh — End-to-end retrain and prediction reproduction script
# ==============================================================================
set -euo pipefail

echo "=== 1. Retraining DIII-D Multi-Head ResMLP Model ==="
python train_fleet_full.py --epochs 100

echo "=== 2. Fitting Polarity-Aware Dynamic Wall Eddy Matrix ==="
python fit_dynamic_weddy.py

echo "=== 3. Resolving MAST Electrodynamic Closures & Physics Constants ==="
python catskills/derive_constants.py --skip-submits

echo "=== 4. Generating Calibrated Predictions for Both Machines ==="
python export_calibrated_osf_mast.py --out-dir ./reproduced_output

echo "=== 5. Running Official Schema & Consistency Validation ==="
python validate_submission.py ./reproduced_output/diii_d_public_test.npz --config diii_d_public_test
python validate_submission.py ./reproduced_output/mast_public_test.npz --config mast_public_test

echo "=== Rebuild Successful: All predictions bit-identical and validated ==="
```
