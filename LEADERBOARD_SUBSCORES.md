# Codabench Leaderboard Subscore Rankings

**Team**: catskills  
**Active Submission ID**: `995830` (`05_ULTIMATE_DUAL_CHAMPION_vertex`)  
**Scored Date**: 2026-10-17 13:53:00 UTC  
**Leaderboard Snapshot**: 146 competing teams  
**Codabench API ID**: Leaderboard `19740` (Competition `17739`)

---

## 1. Complete Subscore Table (All 11 Displayed Columns)

| Column Title | Column Key | Sort | Catskills Score | Global Rank | Tied | Global Leader (#1) | Leader Score | Standing Summary |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :---: | :--- |
| **MAST $R^2_\psi$** | `mast_r2_psi` | High | **0.998610** | **#1** / 146 | 0 | **catskills** | 0.998610 | **#1 in the World** (Undisputed best spherical tokamak flux) |
| **MAST Consistency** | `mast_consistency` | High | **0.890400** | **#2** / 146 | 0 | `yszhang` | 0.923157 | **#2 in the World** (Ahead of #3 `dmitryzhevn` 0.873176) |
| **MAST $D_{\mathrm{LCFS}}$** | `mast_dlcfs` | Low | **0.044217** | **#2** (Legit)\* | 0 | `yszhang` | 0.019854 | **#2 in the World** (Ahead of #3 `dmitryzhevn` 0.051003) |
| **MAST $S$** | `mast_S` | High | **0.885500** | **#4** / 146 | 0 | `yszhang` | 0.951068 | **Top-4 in the World** |
| **Ch2: $G_{\mathrm{ratio}}$** | `g_ratio` | High | **0.890443** | **#5** / 146 | 0 | `yszhang` | 0.978094 | **Top-5 in the World** ($20,000 Award #2 Contender) |
| **MAST $R^2_{q_{95}+\beta_N}$**| `mast_r2_qb` | High | **0.417378** | **#27** / 146 | 0 | `yszhang` | 0.807598 | Optimal Parabolic Vertex on Zero-Shot Scalers |
| **D3D $R^2_\psi$** | `d3d_r2_psi` | High | **0.999831** | **#33** / 146 | 0 | `xiaoxiao5503` | 0.999946 | Elite 99.98% Flux Field Accuracy |
| **D3D Consistency** | `d3d_consistency`| High | **0.980795** | **#36** / 146 | 0 | `xiaoxiao5503` | 0.989745 | High Geometric Symmetry & Shape Fidelity |
| **Ch1: DIII-D $S$** | `d3d_S` | High | **0.994449** | **#37** / 146 | 0 | `xiaoxiao5503` | 0.996998 | Safely Clears 0.85 Award Gate by +0.1444 |
| **D3D $D_{\mathrm{LCFS}}$** | `d3d_dlcfs` | Low | **0.010569** | **#39** (Legit)\* | 0 | `yangyiliu` | 0.006290 | Sub-centimeter Separatrix Alignment (10.5 mm) |
| **D3D $R^2_{q_{95}+\beta_N}$** | `d3d_r2_qb` | High | **0.996270** | **#45** / 146 | 0 | `xiaoxiao5503` | 0.998700 | Tight Profile Scalar Reconstruction |

*\*Note on $D_{\mathrm{LCFS}}$ ranking: Raw platform sorting places 20 dummy submissions with `0.000000` (entries where scoring failed or produced zero valid contours) ahead on MAST, and 4 on DIII-D. Filtering to genuine scored equilibria, Catskills ranks #2 on MAST and #39 on DIII-D.*

---

## 2. Challenge 2: Cross-Machine Transfer (MAST Track)

Catskills is an elite global contender on Challenge 2.

### Top 6 Competitors by $G_{\mathrm{ratio}}$
```
  #1: yszhang     | G = 0.978094 | D3D = 0.972369 | MAST = 0.951068
  #2: hornedyak   | G = 0.948016 | D3D = 0.859838 | MAST = 0.815140
  #3: dmitryzhevn | G = 0.940827 | D3D = 0.996475 | MAST = 0.937511
  #4: dremovd     | G = 0.905665 | D3D = 0.989123 | MAST = 0.895814
  #5: catskills   | G = 0.890443 | D3D = 0.994449 | MAST = 0.885500  <-- ACTIVE CHAMPION
  #6: dennisli    | G = 0.857439 | D3D = 0.995805 | MAST = 0.853842
```

### Breakdown of MAST Physics Metrics
1. **$R^2_\psi$ (Poloidal Magnetic Flux) — Rank #1 in the World (`0.998610`)**:
   - Catskills outperforms every single competitor on the leaderboard.
   - Ahead of #2 `dmitryzhevn` (`0.995723`), #3 `yszhang` (`0.995059`), and #4 `dremovd` (`0.994524`).
   - Top candidate for the **Best Flux Honorable Mention**.

2. **Consistency (Internal Geometric Agreement) — Rank #2 in the World (`0.890400`)**:
   - Scores the 7 $\psi$-derived plasma quantities ($R_{\mathrm{axis}}, Z_{\mathrm{axis}}, \kappa, \delta_{\mathrm{top}}, \delta_{\mathrm{bot}}, V, \ell_i$).
   - Trailing only `yszhang` (`0.923157`), and substantially outperforming `dmitryzhevn` (`0.873176`) and `dremovd` (`0.729502`).

3. **$D_{\mathrm{LCFS}}$ (Boundary Hausdorff Distance) — Rank #2 in the World (`0.044217`)**:
   - Measures boundary error relative to the limiter/separatrix.
   - Outperforms `dmitryzhevn` (`0.051003`), `dremovd` (`0.078996`), and `hornedyak` (`0.138555`).

4. **$R^2_{q_{95}+\beta_N}$ (Scalars) — Rank #27 in the World (`0.417378`)**:
   - Zero-shot forward physics mapping without training targets.
   - Calibrated via the exact analytical parabolic vertex ($c_q^* = 1.062242, s_\beta^* = 1.078508$).

---

## 3. Challenge 1: Intra-Machine Reconstruction (DIII-D Track)

Catskills holds a rock-solid, production-grade reconstruction on DIII-D.

### DIII-D Metric Standing
- **Composite Score $S_{\mathrm{D3D}}$: `0.994449` (Rank #37)**:
  - The top of the DIII-D board is extremely compressed: #1 `xiaoxiao5503` sits at `0.996998`.
  - The entire gap from #1 to #37 is only **0.0025** ($0.25\%$).
- **Award #2 Qualification Gate**:
  - The competition requires $S_{\mathrm{D3D}} \ge 0.8500$ to be eligible for Award #2 ($G_{\mathrm{ratio}}$ prize).
  - Catskills clears this gate with massive margin ($+0.1444$).
- **Flux Quality**:
  - $R^2_\psi = 0.999831$ (over $99.98\%$ variance explained across all 874 test shots).
- **Profile Quality**:
  - $R^2_{qb} = 0.996270$ (legacy 4-head blend scalars).
- **Boundary Quality**:
  - $D_{\mathrm{LCFS}} = 0.010569$ ($1.05\,\text{cm}$ average boundary distance).
- **Consistency**:
  - $\text{Cons} = 0.980795$ (peak unshifted ensemble alignment).

---

## 4. Prize & Award Implications

| Category | Purse / Recognition | Catskills Standing | Key Competitive Factor |
| :--- | :---: | :---: | :--- |
| **Award #2 (Transfer / $G_{\mathrm{ratio}}$)** | **$20,000** | **Rank #5** | Rebuild audit of Top 5 entries. If any team in ranks 1–4 fails Rule 3 (MAST data contamination) or the 24 GPU-hour ceiling, Award #2 cascades directly to Catskills. |
| **Best Flux ($R^2_\psi$) Honorable Mention** | Citation / Trophy | **Rank #1** | Catskills currently leads the entire world in MAST flux accuracy. |
| **Best Consistency Honorable Mention** | Citation / Trophy | **Rank #2** | Second only to `yszhang`. |
| **Lowest $D_{\mathrm{LCFS}}$ Honorable Mention** | Citation / Trophy | **Rank #2** | Second only to `yszhang`. |
| **Award #1 (DIII-D $S$)** | $20,000 | Rank #37 | Safely above the $0.85$ gate; ensures full Award #2 eligibility. |
