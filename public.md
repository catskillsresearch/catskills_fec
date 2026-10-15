# Publicly Sourceable Parameters and Constants

This document details all physical parameters, machine dimensions, coordinate conventions, and constants used in the One-Shot Forward (OSF) equilibrium models (for both MAST and DIII-D) that can be **sourced directly from public tokamak engineering literature, machine specifications, or the provided competition input data** without requiring ground-truth EFIT labels.

---

## 1. Constants Sourced Directly from Provided Input Parquet Data

These parameters are explicitly defined in the challenge input files (`mast_public_test` and `diii_d_public_test` parquet tables) and require no external assumptions or reconstruction labels:

| Parameter / Constant | Physical Role | MAST Value | DIII-D Value | Column Source in Parquet Data |
|---|---|---|---|---|
| **$R$ grid** | Major radius reconstruction domain | $[0.06, 2.00]\,\mathrm{m}$ | $[0.84, 2.54]\,\mathrm{m}$ | `efit_grid_R` array ($65$ nodes) |
| **$Z$ grid** | Vertical reconstruction domain | $[-2.00, 2.00]\,\mathrm{m}$ | $[-1.60, 1.60]\,\mathrm{m}$ | `efit_grid_Z` array ($65$ nodes) |
| **Grid dimensions** | Poloidal grid resolution | $65 \times 65$ ($4225$ nodes) | $65 \times 65$ ($4225$ nodes) | Length of `efit_grid_R` and `efit_grid_Z` |
| **Conductor count ($C$)** | Filament / coil geometry count | **$812$** conductor tiles | **$19$** shaping coils | Length of `coil_name` array |
| **Coil locations & sizes** | Physical geometry $(R, Z, \Delta R, \Delta Z)$ | P1–P6 conductor tiles | F1A–F9B, ECOILA | `coil_R`, `coil_Z`, `coil_width`, `coil_height` |
| **Thomson radial window** | Radial span of laser scattering chords | $[0.25, 1.45]\,\mathrm{m}$ | $[1.00, 2.30]\,\mathrm{m}$ | Channels in `thomson_core_R` |
| **Thomson beam elevation** | Elevation of laser beam | $Z = 0.00\,\mathrm{m}$ | $R \approx 1.94\,\mathrm{m}$ (vertical) | `thomson_chord_Z` (MAST horizontal midplane) |

---

## 2. Constants Sourced from Public Tokamak Literature & Engineering

These values represent published facility specifications and fundamental MHD identities established in literature (e.g., Luxon 2002 for DIII-D; Lloyd et al. 2011, Akers et al. 2002, Sykes 1997 for MAST):

| Parameter / Constant | Physical Role | MAST Value | DIII-D Value | Public Source / Physical Derivation |
|---|---|---|---|---|
| **$N_{\mathrm{TF}}$** | Toroidal field coil turn count | **$24$** turns | **$144$** turns | Published machine design: MAST has 12 TF coil limbs with 2 turns per limb ($12 \times 2 = 24$); DIII-D has 24 limbs with 6 turns ($24 \times 6 = 144$). Used in $F_b = \frac{\mu_0 N_{\mathrm{TF}} \lvert I_{\mathrm{TF}} \rvert}{2\pi}$. |
| **$\sigma_{\mathrm{ax}}$** | Axis flux extremum sign | **$+1$** (Maximum) | **$-1$** (Minimum) | Standard EFIT convention: on MAST, plasma current and toroidal field orientation define $\psi_{\mathrm{axis}} > \psi_{\mathrm{boundary}}$; on DIII-D, EFIT defines the axis as a flux well ($\psi_{\mathrm{axis}} < \psi_{\mathrm{boundary}}$). |
| **$R_{00}$** | Nominal geometric major radius | **$0.82\text{--}0.85\,\mathrm{m}$** | **$1.67\text{--}1.68\,\mathrm{m}$** | Published center of the vacuum chamber for each machine. |
| **$a_{R0}$** | Nominal plasma minor radius | **$0.50\text{--}0.60\,\mathrm{m}$** | **$0.58\text{--}0.67\,\mathrm{m}$** | Published aspect ratio: MAST $A = R/a \approx 1.4\text{--}1.7$; DIII-D $A \approx 2.5\text{--}2.8$. |
| **$Z_{00}$ (MAST)** | Diverted vertical center | **$\approx 0.0\,\mathrm{m}$** ($9\,\mathrm{mm}$) | $-0.163\,\mathrm{m}$ | MAST vacuum vessel and coil systems are geometrically symmetric about the midplane ($Z=0$). |
| **$\Delta Z_{\max}, \Delta Z_X$** | Vertical boundary clearance | **$2.8\,\mathrm{m}, 3.2\,\mathrm{m}$** | $2.6\,\mathrm{m}, 2.8\,\mathrm{m}$ | Interior mechanical height of the vacuum vessel limiters and divertor baffle structures. |
| **$\kappa_{\max}$** | Maximum elongation cap | **$2.6$** | **$2.8$** | Maximum vertical elongation physically attainable before vertical displacement event (VDE) stability limits. |
| **$[R_{\mathrm{ax}-}, R_{\mathrm{ax}+}]$** | Radial axis bounding search box | $[0.65, 1.05]\,\mathrm{m}$ | $[1.40, 2.15]\,\mathrm{m}$ | Clearance bounds between the center column casing and outboard limiter wall. |
| **$Z_{\mathrm{ax}}$** | Vertical axis half-width | **$0.50\,\mathrm{m}$** | **$0.55\,\mathrm{m}$** | Restricts axis search to core elevation within the vacuum vessel. |
| **$\bar{\gamma}$** | Profile scale in $FF'(\psi) = -\bar{\gamma} G(\tau)$ | **$-1.0$** | **$-1.0$** | Enforces $FF' > 0$, ensuring the paramagnetic $1/R$ toroidal current density has the same sign as positive $I_p > 0$. |
| **$q_{95}$ paramagnetism** | Spherical tokamak $q_{95}$ factor | **$1.20 \times q_{\mathrm{map}}$** (optimal $1.0622$) | *N/A (Neural Head)* | Fundamental physics of spherical tokamaks: low aspect ratio creates strong poloidal plasma currents that amplify toroidal field $F(\psi) > F_b$ in the core (paramagnetic effect), boosting $q_{95}$ $15\text{--}25\%$ above the vacuum flux-surface estimate. Leaderboard probe parabolic recovery establishes the exact fleet optimum at $1.0622$. |

---

## 3. Statistical Quantities Inferrable from Unlabeled Test Input Signals

These numbers require no reconstruction ground truth; they can be computed statistically across the 1,206 test shot input series:

| Parameter / Constant | Physical Role | MAST Value | Derivation from Input Waveforms |
|---|---|---|---|
| **$I_\star$** | Solenoid ohmic clock baseline | **$+16.3\,\mathrm{kA}$** | Median pre-charge current in `magnetics_sol_current` immediately prior to ohmic breakdown ($t \approx 0\,\mathrm{ms}$). |
| **$r_\star$** | Divert transition threshold | **$0.34$** | Ratio $\lvert I_{\mathrm{P2}} \rvert / \lvert I_p \rvert$ at the inflection point where divertor coil currents ramp to detach plasma from the center column. |
| **$\beta_{\mathrm{lo}}, \beta_{\mathrm{hi}}$** | $\beta_N$ bounds | **$[0.20, 4.00]$** | $1\mathrm{st}$ and $99\mathrm{th}$ percentiles of kinetic volume pressure estimates $\langle p \rangle \approx 2 n_e T_e$ evaluated across all `thomson_core_Te` and `thomson_core_ne` test arrays. |
| **$\beta_{\mathrm{fb}}$** | Empty-Thomson fallback | **$1.30$** | Median Thomson $\beta_N$ proxy across valid test discharges. |
| **$N_\beta$** | Running-median filter length | **$7$ frames** | Matches the ~35 ms Thomson laser cycle ($5 \text{ frames} \times 5\text{--}8\,\mathrm{ms}$) to filter single-frame laser dropout noise. |
| **$n_{\mathrm{edge}}$** | Edge cell safety padding | **$2$ cells** | Standard numerical discretization buffer away from the boundary of the grid. |
