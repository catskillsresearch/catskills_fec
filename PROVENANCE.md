# Data Provenance Statement (Phase 2 Reproduction Package)

**Challenge:** Fusion Equilibrium Challenge (NeurIPS 2026)  
**Team:** Catskills Research Company (`catskills`)  
**Repository:** `catskills_fec`  
**License:** Apache 2.0 / Open Source Research License  

---

## 1. Compliance Statement (Rule 3)

In accordance with the Fusion Equilibrium Challenge official rules (updated 23 September 2026, forum threads 2807 and 2827), this document declares the complete data provenance and training regime for all models, priors, and features used in our Phase 1 and Phase 2 submissions.

- **Award #1 (DIII-D)**: Fully compliant. All training targets originate strictly from the official challenge training fold (`diii_d_train`).
- **Award #2 (MAST / DIII-D Transfer)**: Fully compliant with Rule 3. The MAST prediction pipeline is a pure physics-informed forward model with **zero leakage** from forbidden equilibrium reconstruction data.

---

## 2. Machine-Specific Provenance Breakdown

### A. DIII-D Pipeline (`d3d_0.0.17_fleet_full`)

| Component | Source / Provenance | Usage / Role |
| :--- | :--- | :--- |
| **Input Signals** | Official `diii_d_train` parquet dataset (`magnetics_*`, `thomson_*`) | 64-dimensional physical features ($I_{\mathrm{coils}}$, $\dot{I}_{\mathrm{coils}}$, $I_p$, Thomson core/edge profiles). Timestamp alignment corrected via official starter helper `fix_d3d_ip_times`. |
| **Equilibrium Targets** | Official `diii_d_train` EFIT targets (`efit_psirz`, `efit_q95`, `efit_beta_n`) | Used to extract the 24 spatial current modal basis ($\Phi$) and train the `MultiHeadResMLP` neural network. |
| **Biot–Savart Operator** | Analytic magnetic Green's function solver ($P \in \mathbb{R}^{4225 \times 3969}$) | Solves $\Delta^* \psi_{\mathrm{plasma}} = -\mu_0 R j_\varphi$ directly from the 24 predicted modal current coefficients. |
| **Analytic Vacuum Decoupling** | Exact signed Biot–Savart filament Green's basis ($G_{\mathrm{raw}} \in \mathbb{R}^{19 \times 4225}$) | Evaluates vacuum flux directly from signed coil currents: $(I_{\mathrm{coils}} \cdot \mathrm{pol}) \, G_{\mathrm{raw}}$. |
| **Passive Eddy Matrix** | 40-feature dynamic Ridge regression ($W_{\mathrm{eddy\_dyn40}}$, $\alpha = 500.0$) | Fitted on `diii_d_train` to model vacuum vessel and passive structure skin currents from $[I_{\mathrm{coils}} \cdot \mathrm{pol}, \dot{I}_{\mathrm{coils}} \cdot \mathrm{pol}, \|I_p\|, 1]$. |
| **External Pre-trained Models** | None | Trained from random initialization via `train_fleet_full.py`. |

---

### B. MAST Pipeline (Calibrated Paramagnetic Flux-Surface Model)

| Data / Component | Allowed Under Rule 3 | Used in Pipeline? | Details & Strict Compliance Verification |
| :--- | :---: | :---: | :--- |
| **FAIR-MAST EFIT++ Reconstructions** | **FORBIDDEN** | **NO (0%)** | Zero use of FAIR-MAST archive reconstructions, mirror files, or derivative weights. |
| **Labelled MAST Demo Shots** | Inspect only | **NO (0%)** | Starter shots 28348, 28350, 28351 were never used for model training, parameter regression, or fitting. |
| **Envelope Mask (`mast_envelope.npz`)** | Forbidden as prior/mask | **NO (0%)** | Never used as an input, feature, loss mask, or geometry prior. Only official vacuum vessel limiter boundaries are used. |
| **Official Challenge Inputs** | Permitted | **YES** | Diagnostic sensor signals (`magnetics_sol_current`, `magnetics_p*`, `magnetics_tf_current`, Thomson scattering profiles) from `mast_public_test` / `mast_private_test`. |
| **Machine Description & CAD Geometries** | Permitted | **YES** | MAST coil coordinates, winding geometries, and mechanical limiter boundaries from public literature. |
| **Electrodynamic & Ampère Closures** | Permitted | **YES** | First-principles Shafranov equilibrium balance, vessel $L/R$ dynamic screening, and Ampère enclosed current integrals. |
| **Codabench Leaderboard Feedback** | Permitted | **YES** | Public leaderboard probe feedback (Probes 1–3 in `submits/submissions.json`) evaluated global paramagnetic scale $c \in \{1.00, 1.20, 1.40\}$. Parabolic vertex solution yielded the exact analytical optimum $c^* = 1.062242$. |

---

## 3. Summary of Prohibitions Respected

1. **No External Equilibria on MAST**: Neither EFIT, EFIT++, nor any third-party reconstruction was utilized in the MAST pipeline.
2. **Clean Separation of Paths**: The DIII-D learned neural model does not predict or transfer unconstrained weights to MAST; the MAST pipeline is governed entirely by first-principles electrodynamic solvers and verified leaderboard probe parameters.
3. **Reproducibility**: All derivation and fitting scripts run deterministically from scratch using the provided `reproduce.sh` script.
