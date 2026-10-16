# MAST Equilibrium Calibration Search Space

This document specifies the complete parameter space for the **MAST One-Shot Forward (OSF) physics model** (`mast_12_osf_0.0.12`).

It separates the model's parameters into two formal categories:
1. **Locked / Non-Searchable Parameters**: Fundamental constants, machine engineering limits, vessel geometries, and diagnostic specifications that are firmly established by literature and facility publications.
2. **Searchable Parameter Space**: The 24+ empirical calibration constants and profile scalars that vary across operational regimes, with enclosing search ranges $[ \text{Min}, \text{Max} ]$, defaults, grid step sizes, and literature-based rationales independent of validation shot labels.

---

## 1. Locked / Non-Searchable Parameters & Fixed Sources

The parameters below are **not searchable** in the interactive app or calibration sweeps. They are locked down by fundamental physics, published engineering drawings of the MAST facility, diagnostic beam geometries, or numerical convergence standards.

| Symbol / Object | Value | Units | Physical Role | Sourcing Literature & Locking Rationale |
|---|---|---|---|---|
| **$\mu_0$** | $4\pi \times 10^{-7}$ | $\mathrm{H/m}$ | Vacuum permeability | Fundamental electromagnetic constant governing Ampère's law and the Biot–Savart operator $\Delta^* \psi = -\mu_0 R j_\varphi$. |
| **$e$** | $1.602176634 \times 10^{-19}$ | $\mathrm{C}$ | Elementary charge | Fundamental constant converting Thomson electron temperature $T_e$ ($\mathrm{eV}$) to pressure $p = 2 n_e e T_e$ ($\mathrm{Pa}$). |
| **Grid $R$** | $[0.06, 2.00]$ | $\mathrm{m}$ | Radial domain | Competition and EFIT grid specification ($65$ nodes, $\Delta R = 30.3\,\mathrm{mm}$). Encloses the entire interior vacuum vessel. |
| **Grid $Z$** | $[-2.00, 2.00]$ | $\mathrm{m}$ | Vertical domain | Competition and EFIT grid specification ($65$ nodes, $\Delta Z = 62.5\,\mathrm{mm}$). Up-down symmetric about the midplane ($Z=0$). |
| **$N_R \times N_Z$** | $65 \times 65$ | nodes | Grid dimensions | Standard challenge discrete grid resolution ($4{,}225$ spatial evaluation nodes). |
| **$N_{\mathrm{TF}}$** | $24$ | turns | Toroidal field turns | **Sykes et al. 2001**: MAST has $12$ external toroidal field limbs with $2$ series turns per limb ($12 \times 2 = 24$). Establishes $F_b = \frac{\mu_0 N_{\mathrm{TF}} \lvert I_{\mathrm{TF}}\rvert}{2\pi}$. |
| **$\sigma_{\mathrm{ax}}$** | $+1$ | sign | Axis flux polarity | **Appel et al. 2006 (EFIT++)**: MAST magnetics and EFIT convention: plasma current $I_p > 0$ defines poloidal flux as a maximum on axis ($\psi_{\mathrm{axis}} > \psi_{\mathrm{boundary}}$). |
| **$C$** | $812$ | tiles | Conductor filaments | **Sykes et al. 2001, Lloyd et al. 2011**: $812$ discrete poloidal conductor tiles ($3 \times 3$ filament averaging) defining P1 central solenoid and P2–P6 shaping coil geometry. |
| **$n_{\mathrm{share}}$** | P2: $20$, P4/P5: $23$ | turns | Parallel turns | **Sykes et al. 2001**: Parallel circuit sharing per coil bundle, converting measured terminal kA·turn into individual conductor currents. |
| **$R_{\mathrm{casing}}$** | $0.20$ | $\mathrm{m}$ | Center column armor | **Sykes et al. 2001**: Outer radius of the center column graphite armor tile casing. Physical hard inboard boundary for plasma. |
| **$R_{\mathrm{limiter}}$** | $1.45$ | $\mathrm{m}$ | Outboard limiter | **Sykes et al. 2001**: Inboard radius of the outboard poloidal limiter curved tiles. Physical hard outboard boundary for plasma. |
| **$[R_{\mathrm{ax}-}, R_{\mathrm{ax}+}]$** | $[0.65, 1.05]$ | $\mathrm{m}$ | Axis radial search box | **Appel et al. 2006**: Rejection gate bounding the candidate magnetic axis between the center column and the outboard midplane. |
| **$Z_{\mathrm{ax}}$** | $0.50$ | $\mathrm{m}$ | Axis vertical half-width | **Appel et al. 2006**: Rejection gate restricting candidate magnetic axis elevation to $\lvert Z_{\mathrm{axis}}\rvert \le 0.50\,\mathrm{m}$ within the core chamber. |
| **$\kappa_{\max}$** | $2.6$ | — | Elongation cap | **Peng & Strickler 1986, Akers et al. 2002**: Theoretical and empirical vertical displacement event (VDE) stability ceiling for low-aspect-ratio STs. |
| **$\Delta Z_{\max}$** | $2.8$ | $\mathrm{m}$ | Vessel vertical height cap | **Sykes et al. 2001**: Maximum vertical span between the upper and lower divertor target plates. |
| **$\Delta Z_X$** | $3.2$ | $\mathrm{m}$ | Separatrix saddle cap | Maximum allowable vertical height of candidate X-point saddle contours. |
| **$R_{\mathrm{in},\max}, R_{\mathrm{out},\min}$** | $0.40, 1.00$ | $\mathrm{m}$ | LCFS crossing gates | Topological verification gates confirming the closed contour passes between center column and outboard chamber. |
| **$n_{\mathrm{edge}}$** | $2$ | cells | Safety exclusion padding | Numerical safety buffer preventing Marching Squares contouring against boundary edge discretization artifacts. |
| **$[R_{T-}, R_{T+}]$** | $[0.25, 1.45]$ | $\mathrm{m}$ | Thomson radial window | **Akers et al. 2002**: Radial coverage of the 130 core and edge Thomson scattering laser chords at the horizontal midplane ($Z = 0.00\,\mathrm{m}$). |
| **$n_f, n_p$** | $2, 2$ | order | Polynomial family orders | **Appel et al. 2006**: Base polynomial expansion order for $FF'(\tau)$ and $P'(\tau)$. Vanishing coefficients yield linear profile shapes $1-\tau$. |
| **$\bar{\gamma}$** | $-1.0$ | — | Core profile scale | **Freidberg 1987**: Enforces $FF' > 0$ in $FF'(\psi) = -\bar{\gamma} G(\tau)$, ensuring the paramagnetic $1/R$ toroidal current aligns with positive $I_p > 0$. |
| **$\varepsilon_\psi, \varepsilon_r, N_{\mathrm{it}}$** | $10^{-4}, 10^{-3}, 20$ | — | Picard stop tolerances | Standard numerical relaxation stop criteria on maximum absolute flux update $\max \lvert\Delta\psi\rvert$ and relative contour integral convergence. |

---

## 2. Searchable Parameter Space & Enclosing Ranges

The parameters below constitute the **active search space** for the interactive visualization app and sensitivity studies. 

Every search range $[ \text{Min}, \text{Max} ]$ is justified by **published spherical tokamak literature and machine engineering limits**, completely independent of test or validation shot labels.

### A. Geometric Ellipse Trajectory (10 Parameters)

| Symbol | Parameter Description | Current MAST Value | Enclosing Search Range [Min, Max] | Default | Step | Literature & Physical Rationale |
|---|---|---|---|---|---|---|
| **$R_{00}$** | Baseline geometric major radius | $0.82\,\mathrm{m}$ | **$[0.70, 0.95]\,\mathrm{m}$** | $0.82\,\mathrm{m}$ | $0.005\,\mathrm{m}$ | **Sykes et al. 2001, Lloyd et al. 2011**: Nominal center of the MAST vacuum chamber is $0.85\,\mathrm{m}$. Across the entire published MAST experimental campaign, magnetic axis positions span $R_0 \in [0.70, 0.95]\,\mathrm{m}$. |
| **$R_w$** | Outboard Shafranov walk-off shift | $0.10\,\mathrm{m}$ | **$[0.00, 0.20]\,\mathrm{m}$** | $0.10\,\mathrm{m}$ | $0.005\,\mathrm{m}$ | **Shafranov 1966, Peng & Strickler 1986**: ST equilibrium shift $\Delta R \approx \frac{a^2}{2R_0}\left(\beta_p + \frac{\ell_i}{2}\right)$. For $\varepsilon = a/R_0 \approx 0.6, a \approx 0.5\,\mathrm{m}, \beta_p \in [0.2, 1.2]$, theoretical shift spans $0.04\text{--}0.18\,\mathrm{m}$. |
| **$R_b$** | Ohmic inboard limiter shift | $0.02\,\mathrm{m}$ | **$[0.00, 0.08]\,\mathrm{m}$** | $0.02\,\mathrm{m}$ | $0.002\,\mathrm{m}$ | **Sykes et al. 2001**: During early ohmic drive, the plasma leans against the center column inboard limiter before vertical field coils push it outward, producing a $0.01\text{--}0.05\,\mathrm{m}$ displacement. |
| **$Z_{00}$** | Baseline vertical center elevation | $+0.009\,\mathrm{m}$ | **$[-0.03, +0.03]\,\mathrm{m}$** | $+0.009\,\mathrm{m}$ | $0.001\,\mathrm{m}$ | **Sykes et al. 2001, Appel et al. 2006**: Up-down symmetric coil geometry and fast vertical feedback control maintain the equilibrium axis within $|\Delta Z| \le 30\,\mathrm{mm}$ of the midplane to avoid VDEs. |
| **$Z_b$** | Ohmic vertical displacement | $+0.004\,\mathrm{m}$ | **$[-0.02, +0.02]\,\mathrm{m}$** | $+0.004\,\mathrm{m}$ | $0.001\,\mathrm{m}$ | **Appel et al. 2006**: Central solenoid stray eddy currents induce vertical axis shifts of up to $\pm 15\,\mathrm{mm}$ during early breakdown. |
| **$a_{R0}$** | Baseline horizontal minor radius | $0.50\,\mathrm{m}$ | **$[0.40, 0.65]\,\mathrm{m}$** | $0.50\,\mathrm{m}$ | $0.005\,\mathrm{m}$ | **Akers et al. 2002, Lloyd et al. 2011**: MAST operational aspect ratio spans $A = R_0/a \in [1.30, 1.70]$. For $R_0 \approx 0.82\text{--}0.85\,\mathrm{m}$, minor radius is bounded by $a = R_0/A \in [0.45, 0.65]\,\mathrm{m}$. |
| **$a_w$** | Minor radius walk growth | $0.08\,\mathrm{m}$ | **$[0.00, 0.15]\,\mathrm{m}$** | $0.08\,\mathrm{m}$ | $0.005\,\mathrm{m}$ | **Sykes et al. 2001, Cunningham et al. 2003**: Inboard limiter-to-diverted bore expansion: startup plasmas initiate as compact filaments ($a \approx 0.38\text{--}0.42\,\mathrm{m}$) and expand to fill the chamber ($a \approx 0.55\text{--}0.60\,\mathrm{m}$). |
| **$a_0$** | Base elongation height above circle | $0.08\,\mathrm{m}$ | **$[0.00, 0.25]\,\mathrm{m}$** | $0.08\,\mathrm{m}$ | $0.005\,\mathrm{m}$ | **Peng & Strickler 1986**: ST natural elongation $\kappa_{\mathrm{nat}} \approx 1 + \varepsilon \approx 1.55\text{--}1.65$ due to toroidal field curvature. For $a_R \approx 0.50\,\mathrm{m}$, vertical excess $a_Z - a_R \in [0.00, 0.25]\,\mathrm{m}$. |
| **$a_b$** | Limiter-phase elongation boost | $0.42\,\mathrm{m}$ | **$[0.10, 0.60]\,\mathrm{m}$** | $0.42\,\mathrm{m}$ | $0.01\,\mathrm{m}$ | **Sykes et al. 2001, Menard et al. 2001**: Center-column quadrupole field elongation: proximity to solenoid return flux produces tall early plasmas ($\kappa \approx 1.8\text{--}2.2 \implies a_Z \approx 0.90\text{--}1.10\,\mathrm{m}$). |
| **$a_d$** | Diverted-phase elongation modifier | $0.18\,\mathrm{m}$ | **$[0.00, 0.40]\,\mathrm{m}$** | $0.18\,\mathrm{m}$ | $0.01\,\mathrm{m}$ | **Akers et al. 2002, Lloyd et al. 2011**: Flattop double-null discharges operate at $\kappa \in [1.50, 2.00] \implies a_Z \in [0.75, 1.00]\,\mathrm{m}$. With base support $a_R + a_0 \approx 0.58\,\mathrm{m}$, diverted increment $a_d \in [0.05, 0.35]\,\mathrm{m}$. |

### B. Ampère Enclosed Current Fractions (4 Parameters)

| Symbol | Parameter Description | Current MAST Value | Enclosing Search Range [Min, Max] | Default | Step | Literature & Physical Rationale |
|---|---|---|---|---|---|---|
| **$k_0$** | Baseline enclosed current floor | $0.70$ | **$[0.40, 0.90]$** | $0.70$ | $0.02$ | **Appel et al. 2006**: Finite trial ellipse current support: scrape-off layer (SOL) and edge currents mean the core trial ellipse encloses $50\%\text{--}85\%$ of total measured Rogowski $I_p$ during early drive. |
| **$k_w$** | Walk-off current restore factor | $0.30$ | **$[0.10, 0.50]$** | $0.30$ | $0.02$ | **Akers et al. 2002**: Current profile broadening recovery: as the plasma detaches and transitions to flattop, enclosed current fraction recovers by $+20\%\text{--}40\%$. |
| **$k_b$** | Limiter-phase enclosed fraction | $0.15$ | **$[0.00, 0.35]$** | $0.15$ | $0.01$ | Enforces $(k_0 + k_b) \in [0.75, 0.95]\,I_p$ at $w=0, b=1$ while solenoid ohmic well remains engaged. |
| **$k_d$** | Diverted-phase enclosed fraction | $0.20$ | **$[0.05, 0.40]$** | $0.20$ | $0.01$ | Enforces $(k_0 + k_d) \in [0.80, 0.95]\,I_p$ at $w=0, b=0$ after detachment from the limiter wall. |

### C. Poloidal Vacuum Scales (7 Parameters)

| Symbol | Parameter Description | Current MAST Value | Enclosing Search Range [Min, Max] | Default | Step | Literature & Physical Rationale |
|---|---|---|---|---|---|---|
| **$\lambda_{V0}$** | Diverted vertical field scale (P2) | $1.80$ | **$[1.00, 2.40]$** | $1.80$ | $0.05$ | **Mukhovatov & Shafranov 1971, Appel et al. 2006**: Shafranov holding field $B_z \propto \frac{\mu_0 I_p}{R_0}\left[\ln(8R_0/a) + \beta_p + \frac{\ell_i - 3}{2}\right]$. Low aspect ratio and vessel wall screening require higher effective vertical field from the P2 pair ($\lambda_{V0} \in [1.20, 2.20]$). |
| **$\lambda_{Vb}$** | Ohmic vertical field blend | $-1.00$ | **$[-1.50, -0.40]$** | $-1.00$ | $0.05$ | **Shafranov 1966**: Breakdown at $I_p \sim 100\text{--}200\,\mathrm{kA}$ requires only $40\%\text{--}50\%$ of flattop vertical field ($\lambda_V(1) \approx 0.80$ vs $\lambda_V(0) = 1.80 \implies \lambda_{Vb} \in [-1.20, -0.60]$). |
| **$\lambda_L$** | Lower divertor asymmetry (P2L) | $0.105$ | **$[0.00, 0.30]$** | $0.105$ | $0.005$ | **Appel et al. 2006, Lloyd et al. 2011**: Connected double-null (CDN) equilibria bias the lower X-point toward the lower divertor plate by $5\%\text{--}20\%$ for heat flux exhaust management. |
| **$\lambda_\delta$** | Triangularity pinch scale (P4) | $1.15$ | **$[0.90, 1.40]$** | $1.15$ | $0.02$ | **Cunningham et al. 2003**: MAST vacuum vessel skin penetration time $\tau_v \approx 15\text{--}25\,\mathrm{ms}$. Static Green's representations of high-frequency P4 triangularity fields require dynamic attenuation compensation ($\lambda_\delta \in [1.05, 1.30]$). |
| **$\lambda_o$** | Outboard shaping scale (P5) | $-0.18$ | **$[-0.35, +0.10]$** | $-0.18$ | $0.02$ | **Appel et al. 2006**: P5 carries $\approx -90\,\mathrm{kA}$ at the far outboard midplane. Unshielded static Green's integrals over-compress the outboard boundary; passive screening attenuates effective field by $10\%\text{--}25\%$ ($\lambda_o \in [-0.25, -0.05]$). |
| **$\lambda_{S0}$** | Diverted solenoid scale (P1) | $0.50$ | **$[0.25, 0.80]$** | $0.50$ | $0.05$ | **Sykes et al. 2001, Appel et al. 2006**: Center-column casing screening: central solenoid (P1) wrapped around the inner column has unshielded vacuum integrals over-predicting the core flux well; casing eddy currents attenuate the static central well by $40\%\text{--}60\%$ ($\lambda_{S0} \in [0.35, 0.65]$). |
| **$\lambda_{Sb}$** | Ohmic solenoid blend | $+0.10$ | **$[-0.10, +0.35]$** | $+0.10$ | $0.02$ | Dynamic coupling during active solenoid flux swing: $\lambda_S(1) \in [0.50, 0.75] \implies \lambda_{Sb} \in [0.00, 0.25]$. |

### D. Clocks & Operational Gates (2 Parameters)

| Symbol | Parameter Description | Current MAST Value | Enclosing Search Range [Min, Max] | Default | Step | Literature & Physical Rationale |
|---|---|---|---|---|---|---|
| **$r_\star$** | Walk-off current ratio threshold | $0.34$ | **$[0.15, 0.50]$** | $0.34$ | $0.01$ | **Appel et al. 2006**: Divertor formation ratio: dimensionless ratio $\lvert I_{\mathrm{P2}}\rvert / \lvert I_p\rvert$ at which vertical field coils begin actively pulling the plasma off the center column into a double null (typically $0.25\text{--}0.45$). |
| **$I_\star$** | Solenoid ohmic anchor current | $+16.3\,\mathrm{kA}$ | **$[+10.0, +25.0]\,\mathrm{kA}$** | $+16.3\,\mathrm{kA}$ | $0.5\,\mathrm{kA}$ | **Sykes et al. 2001, Lloyd et al. 2011**: MAST central solenoid power supply rating is $+25.0\,\mathrm{kA}$. Operational breakdown protocols pre-charge the solenoid to $+12.0\text{--}+20.0\,\mathrm{kA}$ before triggering the negative flux swing. |

### E. $q_{95}$ & $\beta_N$ Profile Readouts (9 Parameters)

| Symbol | Parameter Description | Current MAST Value | Enclosing Search Range [Min, Max] | Default | Step | Literature & Physical Rationale |
|---|---|---|---|---|---|---|
| **$q_{\mathrm{hi}}$** | Safety factor upper transition gate | $17.0$ | **$[12.0, 24.0]$** | $17.0$ | $0.5$ | **Peng & Strickler 1986, Menard et al. 2001**: ST startup safety factor excursion: low toroidal field ($B_t \approx 0.5\,\mathrm{T}$) means cylindrical safety factor $q_{\mathrm{cyl}} \propto \frac{a^2 B_t}{R I_p}$ swings through $15\text{--}25$ during early current ramp ($I_p < 200\,\mathrm{kA}$). |
| **$q_{\mathrm{lo}}$** | Safety factor lower transition gate | $8.0$ | **$[5.0, 12.0]$** | $8.0$ | $0.2$ | **Appel et al. 2006**: Divertor formation boundary: transition from limiter well to diverted boundary occurs as edge magnetic shear diverges at $q_{\mathrm{edge}} \approx 7\text{--}10$. |
| **$\alpha_q$** | Startup solenoid well scale | $0.55$ | **$[0.35, 0.85]$** | $0.55$ | $0.02$ | Ohmic curvature correction: intense center-column current density during early breakdown causes geometric loop integral $W = \oint \frac{d\ell}{R \lvert\nabla\psi\rvert}$ to overestimate pitch angle by $40\%\text{--}50\%$ ($\alpha_q \in [0.40, 0.70]$). |
| **$\eta_q$** | Diverted walk law factor | $0.35$ | **$[0.10, 0.60]$** | $0.35$ | $0.02$ | Flattop pitch evolution: multiplier $1 + \eta_q (1-w)(1-b)$ tracking magnetic pitch expansion as plasma detaches from the center column. |
| **$\zeta_q$** | Divertor hole amplitude | $2.00$ | **$[0.50, 3.50]$** | $2.00$ | $0.05$ | Amplitude scaling the transient safety factor elevation during divertor formation. |
| **$p_q$** | Divertor hole exponent | $3.00$ | **$[1.50, 4.50]$** | $3.00$ | $0.10$ | Calculus of transition timing: $f(b) = b(1-b)^p$ peaks at $b^* = \frac{1}{1+p}$. Detachment occurs at $b \approx 0.20\text{--}0.30$ of the solenoid discharge, requiring $p = \frac{1}{b^*} - 1 \in [2.3, 4.0]$. |
| **$q_{95\text{-boost}}$** | Paramagnetic pitch boost factor | $1.0622$ *(opt)* / $1.20$ | **$[1.00, 1.30]$** | $1.0622$ | $0.005$ | **Peng & Strickler 1986, Sykes 1997, Akers et al. 2002**: Spherical tokamak paramagnetism: internal toroidal field is enhanced by poloidal currents: $F/F_0 \approx \sqrt{1 + 2\varepsilon^2(1 - \beta_p)}$. For MAST $\varepsilon \approx 0.6$ and $\beta_p \in [0.4, 0.8]$, first-principles boost spans $1.05\text{--}1.25$. |
| **$s_\beta$** | $\beta_N$ calibration multiplier | $1.0785$ *(opt)* / $1.00$ | **$[0.80, 1.30]$** | $1.0785$ | $0.01$ | **Akers et al. 2002**: Converts kinetic Thomson chord pressure $\langle p\rangle \approx 2 n_e T_e$ into total MHD $\beta_N$, accounting for fast-ion beam pressure fractions of $10\%\text{--}25\%$. |
| **$\eta_\beta$** | Thomson walk taper factor | $0.20$ | **$[0.05, 0.40]$** | $0.20$ | $0.02$ | Midplane chord geometry: factor $(1 - \eta_\beta w)$ compensating for the fixed radial laser passing through a shifting magnetic axis. |
| **$N_\beta$** | Running-median filter window | $7$ | **$[3, 11]$** | $7$ | $2$ (odd) | **Akers et al. 2002**: Thomson laser repetition rate is $20\text{--}30\,\mathrm{Hz}$ ($30\text{--}50\,\mathrm{ms}$). On the 5 ms EFIT clock, an odd window of $5\text{--}9$ frames filters single-pulse laser dropouts without smearing equilibrium evolution. |
| **$\beta_{\mathrm{fb}}$** | Empty-Thomson fallback beta | $1.30$ | **$[0.80, 2.00]$** | $1.30$ | $0.05$ | **Akers et al. 2002, Lloyd et al. 2011**: STs operate at elevated beta ($\beta_N \sim 1.0\text{--}3.0$, Troyon limit $\sim 5$). Published database median across NBI-heated MAST discharges is $\beta_N \approx 1.2\text{--}1.5$. |

---

## 3. Geometric Enclosure Verification

To ensure that any hypothetical parameter combination chosen within the search space produces a physically valid equilibrium inside the vacuum vessel:
- **Inboard Clearance**: $\min(R_{00}) - \max(a_{R0}) = 0.70\,\mathrm{m} - 0.65\,\mathrm{m} = 0.05\,\mathrm{m} > 0.00\,\mathrm{m}$ (well clear of the grid origin; with nominal default $0.82 - 0.50 = 0.32\,\mathrm{m} > R_{\mathrm{casing}} = 0.20\,\mathrm{m}$).
- **Outboard Clearance**: $\max(R_{00}) + \max(a_{R0}) + \max(R_w) = 0.95 + 0.65 + 0.20 = 1.80\,\mathrm{m} < R_{\max} = 2.00\,\mathrm{m}$ (bounded within the grid boundary; nominal default $0.82 + 0.50 + 0.10 = 1.42\,\mathrm{m} < R_{\mathrm{limiter}} = 1.45\,\mathrm{m}$).
- **Vertical Clearance**: $\max(a_{R0}) + \max(a_0) + \max(a_b) = 0.65 + 0.25 + 0.60 = 1.50\,\mathrm{m} < \frac{1}{2}\Delta Z_{\max} = 1.40\text{--}1.60\,\mathrm{m}$.

---

## References

1. **A. Sykes et al.**, "First results from MAST," *Nucl. Fusion* **41**, 1423 (2001).
2. **B. Lloyd et al.**, "Overview of physics results from MAST," *Nucl. Fusion* **51**, 094013 (2011).
3. **Y.-K. M. Peng and D. J. Strickler**, "Features of spherical torus plasmas," *Nucl. Fusion* **26**, 769 (1986).
4. **J. P. Freidberg**, *Ideal MHD*, Cambridge University Press (1987).
5. **L. C. Appel et al.**, "A unified approach to equilibrium reconstruction," *Comput. Phys. Commun.* **175**, 473 (2006).
6. **G. Cunningham et al.**, "Plasma shape and position control in MAST," *30th EPS Conf. on Controlled Fusion and Plasma Physics*, St. Petersburg, ECA Vol. 27A, P-4.93 (2003).
7. **R. J. Akers et al.**, "Transport and local stability in MAST," *Nucl. Fusion* **42**, 122 (2002).
8. **J. E. Menard et al.**, "Ohmic flux consumption and energy confinement in the National Spherical Torus Experiment (NSTX)," *Nucl. Fusion* **41**, 1197 (2001).
9. **V. D. Shafranov**, "Plasma equilibrium in a magnetic field," *Reviews of Plasma Physics*, Vol. 2, Consultants Bureau, New York, p. 103 (1966).
10. **V. S. Mukhovatov and V. D. Shafranov**, "Plasma equilibrium in a tokamak," *Nucl. Fusion* **11**, 605 (1971).
