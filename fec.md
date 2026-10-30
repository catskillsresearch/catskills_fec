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
- [Phase 3: Reproduce](#phase-3-reproduce)
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

Phase-2 Final (and Phase-3 rebuild) uses **two machine-specific models**:

#### 1. DIII-D — Charcoal / O1G neural ensemble
- Supervised charcoal PINN / O1G geometry-aware members trained on official `diii_d_train`.
- Six-member flux ensemble + clean-Ip scalar heads; inference via `catskills.scripts.generate_final_predictions`.
- **Thomson ZGRID**: discovered once from the train split’s `efit_grid_Z` (half-width × 0.75, 32 bins), frozen in `zgrid.json`, then used for train + private infer. Do not re-estimate from `private_test`.
- Compute: see `COMPUTE.md` (charcoal ladder within the 24 GPU-h A100 ceiling).

#### 2. MAST — M9 flux + OSF analytic scalars
- ψ from the M9 win50 ensemble (`CRC_FEQ_mast_M9_win50_i{101,202,303}`).
- q95 / βN from OSF `osf_0_0_12` with frozen calibrations $c^*=1.062242$, $s^*=1.078508$ (CPU-only; 0 GPU-h).
- Rule 3: zero FAIR-MAST EFIT++ / envelope-mask priors (see `PROVENANCE.md`).

---

### Turnkey Rebuild Execution (`reproduce.sh`)

```bash
./reproduce.sh --weights-only          # shipped .pt + zgrid → private_test npz
./reproduce.sh --train                 # discover ZGRID → charcoal train → infer
./reproduce.sh --dry-run               # smoke (2 shots)
./reproduce.sh --legacy-zgrid-1003664  # forensic ZGRID for Codabench 1003664
```

Full organizer instructions: [`REPRODUCE.md`](REPRODUCE.md). Provenance: [`PROVENANCE.md`](PROVENANCE.md). Compute: [`COMPUTE.md`](COMPUTE.md).

---

## Phase 3: Reproduce

| Phase | Name | Role |
|---|---|---|
| 1 | Development | Public fold / Codabench Dev |
| 2 | Final | Blind private fold (award scores) |
| 3 | **Reproduce** | Organizer rebuild / internal review (`final.tt`) |

**Deadline:** 31 October 2026 23:59 CDT.  
**Packages required for:** Codabench **1003664** and **EM-catskills-3** (`9db5e666…`).

**Tolerances:** retrain composite $S$ within **0.01** of submitted; shipped weights within **1e-4**.

**Post-hoc package rule:** Phase-3 docs and entrypoints may be finalized after Phase-2 submit; HF prediction blobs are not rewritten — regenerate from Final inputs + weights / retrain.

Standing rebuild path matches **9db5e666** (train-discovered ZGRID freeze equivalent to Dev ±1.2). Submission **1003664** used a mistaken narrow ZGRID (−0.6…+0.8); use `--legacy-zgrid-1003664` only for forensics.
