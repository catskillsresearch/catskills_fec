# Differences in OSF Model Constants: MAST vs. DIII-D

This document tabulates all physical parameters, coordinate conventions, and calibration constants that **differ** between the MAST One-Shot Forward (OSF) physics model (`mast_12_osf_0.0.12` with paramagnetic $q_{95}$) and the DIII-D physics-informed OSF modal model (`d3d_0.0.17_fleet_full`).

---

## Side-by-Side Comparison of Differing Constants

| Symbol / Constant | Physical Role | MAST OSF (`mast_12_osf_0.0.12`) | DIII-D OSF (`d3d_0.0.17_fleet_full`) | Physical Reason for Difference |
|---|---|---|---|---|
| **$\sigma_{\mathrm{ax}}$** | Axis flux extremum sign | **$+1$** (Maximum) | **$-1$** (Minimum) | EFIT convention: DIII-D axis is a flux well ($\psi_{\mathrm{axis}} < \psi_{\mathrm{bnd}}$); MAST is a peak |
| **$N_{\mathrm{TF}}$** | Toroidal field turn count | **$24$** turns | **$144$** turns | Multiplies supply current to compute vacuum toroidal flux $F_b$ |
| **$R$ grid** | Major radius domain | $[0.06, 2.00]\,\mathrm{m}$ | $[0.84, 2.54]\,\mathrm{m}$ | Spherical tokamak tight central hole (MAST) vs conventional aspect ratio (DIII-D) |
| **$Z$ grid** | Vertical domain | $[-2.00, 2.00]\,\mathrm{m}$ | $[-1.60, 1.60]\,\mathrm{m}$ | MAST vacuum vessel is taller vertically |
| **Conductors ($C$)** | Filament / coil count | **$812$** conductor tiles | **$19$** shaping coils | MAST uses FAIR filament tiles; DIII-D uses 18 F-coils + ECOILA |
| **$I_\star$** | Solenoid ohmic clock baseline | **$+16.3\,\mathrm{kA}$** | **$-46.26\,\mathrm{kA}$** | Clock baseline for $b = \mathrm{clip}(I_{\mathrm{sol}}/I_\star, 0, 1)$; opposite solenoid polarity |
| **$R_{00}$** | Diverted major radius center | **$0.82\,\mathrm{m}$** | **$1.68\,\mathrm{m}$** | Geometric plasma center of the respective machines |
| **$R_w$** | Walk-off major radius shift | **$+0.10\,\mathrm{m}$** | **$0.00\,\mathrm{m}$** | MAST plasma shifts outward as vertical coils walk off; DIII-D remains centered |
| **$R_b$** | Ohmic-inboard shift | **$+0.02\,\mathrm{m}$** | **$0.00\,\mathrm{m}$** | MAST shifts inboard during early ohmic solenoid phase |
| **$Z_{00}$** | Diverted vertical center | **$+0.009\,\mathrm{m}$** | **$-0.163\,\mathrm{m}$** | MAST is nearly midplane-symmetric; DIII-D equilibrium is lowered toward divertor |
| **$Z_b$** | Ohmic vertical shift | **$+0.004\,\mathrm{m}$** | **$-0.303\,\mathrm{m}$** | DIII-D early ohmic plasma is located significantly higher ($Z \approx +0.14\,\mathrm{m}$) |
| **$a_{R0}$** | Minor radius baseline | **$0.50\,\mathrm{m}$** | **$0.58\,\mathrm{m}$** | Difference in machine minor radius ($a$) |
| **$a_w$** | Minor radius walk growth | **$+0.08\,\mathrm{m}$** | **$0.00\,\mathrm{m}$** | MAST plasma broadens horizontally during walk-off |
| **$a_0$** | Support height above $a_R$ | **$0.08\,\mathrm{m}$** | **$0.42\,\mathrm{m}$** | Baseline vertical extension ($a_Z = a_R + a_0$) |
| **$a_b$** | Limited height modifier | **$+0.42\,\mathrm{m}$** | **$0.00\,\mathrm{m}$** | MAST limited phase is significantly taller ($a_Z = 1.00\,\mathrm{m}$) |
| **$a_d$** | Diverted height modifier | **$+0.18\,\mathrm{m}$** | **$0.00\,\mathrm{m}$** | MAST diverted phase height modifier ($a_Z = 0.76\,\mathrm{m}$) |
| **Current basis** | Plasma current representation | Rigid parametric ellipse | **24 spatial modes ($P_{\mathrm{sub}}$)** | DIII-D uses 24-mode SVD Biot–Savart operator to capture $\delta_{\mathrm{top}}, \delta_{\mathrm{bot}}, \kappa, \ell_i$ |
| **$k_0$** | Floor Ampère multiplier | **$0.70$** | **$0.00$** | MAST baseline fraction of $I_p$ |
| **$k_w$** | Walk-clock Ampère factor | **$+0.30$** | **$0.00$** | Restores current as MAST walks off ($w \to 1$) |
| **$k_b$** | Solenoid Ampère factor | **$+0.15$** | **$0.00$** | Limited end on MAST ($w=0, b=1$) is $0.85\,I_p$ |
| **$k_d$** | Diverted Ampère factor | **$+0.20$** | **$0.00$** | Diverted end on MAST ($w=0, b=0$) is $0.90\,I_p$ |
| **$I_0$** | Flat added current offset | **$0.00\,\mathrm{MA}$** | **$1.40\,\mathrm{MA}$** | DIII-D Rogowski compensation in base OSF |
| **$\lambda_{V0}$** | Vertical-field pair scale | **$1.80$** (P2 pair) | **$1.00$** (F6 pair) | Diverted vertical field scale |
| **$\lambda_{Vb}$** | Vertical-field ohmic scale | **$-1.00$** | **$0.00$** | Ohmic end scale on MAST is $\lambda_V(1) = 0.80$ |
| **$\lambda_L$** | Lower-member multiplier | **$+0.105$** (P2L only) | **$0.00$** | Pulls lower boundary tip toward divertor null on MAST |
| **$\lambda_\delta$** | Triangularity pair scale | **$1.15$** (P4 pair) | **$1.00$** (F5 pair) | Shapes upper and lower pinch coils on MAST |
| **$\lambda_o$** | Outboard pair scale | **$-0.18$** (P5 pair) | **$0.00$** (F7 pair) | Diverted scale $= 0.82$ on MAST |
| **$\lambda_{S0}$** | Solenoid scale | **$0.50$** | **$1.00$** | Halves center-column well on MAST |
| **$\lambda_{Sb}$** | Solenoid ohmic scale | **$+0.10$** | **$0.00$** | Ohmic end scale on MAST is $\lambda_S(1) = 0.60$ |
| **$W_{\mathrm{ext}}$ model** | External field coupling | Static Green's filaments | **$G_{\mathrm{raw}} + W_{\mathrm{eddy\_dyn40}}$** | Polarity-aware analytic Green basis $G_{\mathrm{raw}}$ plus 40-feature dynamic vessel eddy matrix $W_{\mathrm{eddy}}$ (fallback to $W_{\mathrm{ext\_dyn41}}$) |
| **$\alpha_q$** | Well factor ($h=1$) | **$0.55$** | **$1.00$** | Ohmic-well $q_{95}$ factor on MAST; unused on DIII-D ($h=0$) |
| **$\eta_q$** | Diverted walk law | **$0.35$** | **$0.00$** | Late-walk $q_{95}$ multiplier on MAST |
| **$\zeta_q$** | Hole amplitude | **$2.00$** | **$0.00$** | Added $q_{95}$ multiplier during divert approach on MAST |
| **$q_{95}$ calibration** | Safety factor determination | **$1.20 \times q_{\mathrm{map}}$** (optimal $1.0622$) | **Neural profile head** | MAST uses paramagnetic flux-surface integral (optimal $c^*=1.0622$ recovered via Codabench parabolic probe); DIII-D uses trained MLP head |
| **$\eta_\beta$** | Walk $\beta_N$ taper | **$0.20$** | **$0.00$** | Tapers hot midplane Thomson chords on MAST |
| **$\beta_{\mathrm{lo}}$** | Lower $\beta_N$ clamp floor | **$0.20$** | **$0.01$** | DIII-D allows lower startup values |
| **$\beta_{\mathrm{fb}}$** | Empty-Thomson fallback | **$1.30$** | **$0.53$** | Typical discharge median fallback |
| **$[R_{T-}, R_{T+}]$** | Thomson radial window | $[0.25, 1.45]\,\mathrm{m}$ | $[1.00, 2.30]\,\mathrm{m}$ | Intersects horizontal laser (MAST) vs vertical laser (DIII-D) |
| **$[R_{\mathrm{ax}-}, R_{\mathrm{ax}+}]$** | Axis search window | $[0.65, 1.05]\,\mathrm{m}$ | $[1.40, 2.15]\,\mathrm{m}$ | Core radial search bounding box |
| **$Z_{\mathrm{ax}}$** | Axis vertical half-width | **$0.50\,\mathrm{m}$** | **$0.55\,\mathrm{m}$** | Core vertical bounding box |
| **$\kappa_{\max}$** | Elongation filter cap | **$2.6$** | **$2.8$** | Maximum allowable elongation for compact contour |
| **$\Delta Z_{\max}$** | Vertical span filter cap | **$2.8\,\mathrm{m}$** | **$2.6\,\mathrm{m}$** | Bounding contour height filter |
| **$R_{\mathrm{out,min}}$** | Minimum outboard radius | **$1.00\,\mathrm{m}$** | **$1.70\,\mathrm{m}$** | Contour must reach outboard region |
| **$R_{\mathrm{in,max}}$** | Maximum inboard radius | **$0.40\,\mathrm{m}$** | **$1.35\,\mathrm{m}$** | Contour must reach inboard region |
| **$\Delta Z_X$** | Max X-point contour height | **$3.2\,\mathrm{m}$** | **$2.8\,\mathrm{m}$** | Upper limit on diverted separatrix contour |
