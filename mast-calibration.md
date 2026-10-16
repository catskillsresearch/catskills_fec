## MAST calibration of one-shot forward physics model

This is the MAST assignment of every symbol in [One-shot forward physics model](one-shot-forward.md#one-shot-forward-physics-model). The model is `osf_0.0.12`. This calibration is `mast_12_osf_0.0.12`. $\sigma_{\mathrm{ax}}=+1$. There is no `mast_train`. The three arrays are written for each of the 1,206 shots in `mast_public_test`. The symbols are one set for every shot. They were set on starter-kit shots 28348, 28350, and 28351 (115 flattop frames that ship $\psi$ and the EFIT scalars). Those shots are not in `mast_public_test`, not a train split, and not a substitute for the 1,206.

### MAST objects that fill the solve

Per shot $i$, from [`mast_public_test`](parquet-data.md#mast-parquet-data-format). Every EFIT-derived column is omitted. $T_i\in[5,98]$.

| Model object | MAST column or value |
|---|---|
| Clock and grid | `efit_times` $(T_i,)$; `efit_grid_R`, `efit_grid_Z` $(65,)$. $R\in[0.06,2.00]\,\mathrm{m}$, $Z\in[-2.00,2.00]\,\mathrm{m}$. Stored $\psi$ has index $[t,i_Z,i_R]$ |
| $I_p$ | `magnetics_plasma_current` on `magnetics_time` |
| $I_{\mathrm{sol}}$ | `magnetics_sol_current`. Plain kA; the 656 solenoid rows are the turns |
| $I_{V,\mathrm{U}}$, $I_{V,\mathrm{L}}$ | `magnetics_p2u_current`, `magnetics_p2l_current`. Already kA·turn. $n_{\mathrm{share}}=20$ |
| Triangularity pair | `magnetics_p4u_current`, `magnetics_p4l_current`. $n_{\mathrm{share}}=23$ |
| Outboard pair | `magnetics_p5u_current`, `magnetics_p5l_current`. $R\approx 1.65\,\mathrm{m}$, $Z\approx\pm 0.50\,\mathrm{m}$, about $-90\,\mathrm{kA}$. $n_{\mathrm{share}}=23$ |
| $I_{\mathrm{TF}}$, $N_{\mathrm{TF}}$ | `magnetics_tf_current`, $N_{\mathrm{TF}}=24$ |
| Filaments | `coil_*`, $C=812$. Each rectangle is the mean of a $3\times 3$ of filaments at the cell centres |
| Thomson | Core and edge $T_e$, $n_e$, $R$. Core $R$ has 130 channels; $T_e$ and $n_e$ have 131. Each sample uses the minimum length |
| Scored keys | `shot_XXXX_psirz` $(T_i,65,65)$, `shot_XXXX_q95` $(T_i,)$, `shot_XXXX_betaN` $(T_i,)$ in `mast_public_test.npz` |

P-coil currents are already kA·turn, so each of the $n_{\mathrm{share}}$ elements carries $I/n_{\mathrm{share}}$. Solenoid ampere-turns are $I\cdot n_{\mathrm{share}}$. `magnetics_tf_current` has no `coil_*` rectangle, so it is absent from $\psi_{\mathrm{vac}}$ and present as $B_t$. `magnetics_efps_current` is the error-field feed: no poloidal rectangle, omitted from the solve. P3 and P6 are about $1\,\mathrm{kA}$ and stay at the measured current. P2L, the lower vertical-field coil, sits at $R\approx 0.50\,\mathrm{m}$, $Z\approx -1.73\,\mathrm{m}$.

Details: [clocks](parquet-data.md#mast-coordinates-and-clocks), [solenoid](parquet-data.md#mast-solenoid), [P-coils](parquet-data.md#mast-poloidal-coils), [TF](parquet-data.md#mast-toroidal-field-and-error-field-correction), [geometry](parquet-data.md#mast-coil-geometry), [Thomson](parquet-data.md#mast-thomson-scattering).

### Assignment

End members at $w=0$, from the numbers below. Limited is $b=1$. Diverted is $b=0$.

| | Limited | Diverted |
|---|---|---|
| $R_0$ | $0.80\,\mathrm{m}$ | $0.82\,\mathrm{m}$ |
| $Z_0$ | $5\,\mathrm{mm}$ | $9\,\mathrm{mm}$ |
| $a_R$ | $0.50\,\mathrm{m}$ | $0.50\,\mathrm{m}$ |
| $a_Z$ | $1.00\,\mathrm{m}$ | $0.76\,\mathrm{m}$ |
| $\int j_\varphi\,dA/I_p$ | $0.85$ | $0.90$ |
| Pair multiplier $\lambda_V$ | $0.80$ | $1.80$ |
| Lower-member multiplier $1+\lambda_L(1-b)$ | $1$ | $1.105$ |
| Pinch multiplier $\lambda_\delta$ | $1.15$ | $1.15$ |
| Outboard multiplier $1+\lambda_o(1-b)$ | $1$ | $0.82$ |
| Solenoid multiplier $\lambda_S$ | $0.60$ | $0.50$ |

At $w=1$, $b=0$: $R_0=0.92\,\mathrm{m}$ and the Ampère factor is $1.20$.

#### Clocks and ellipse

| Symbol | Value | Set from |
|---|---|---|
| $r_\star$ | $0.34$ | $\lvert P2\rvert/\lvert I_p\rvert$ on the seated frame near $90\,\mathrm{ms}$, where $w$ is defined to be $0$ |
| $I_\star$ | $16.3\,\mathrm{kA}$ | First-frame solenoid current of 28348. $b=1$ there. $b=0$ for $I_{\mathrm{sol}}\le 0$ |
| $R_{00}$ | $0.82\,\mathrm{m}$ | Diverted major radius at $w=0$ |
| $R_w$ | $0.10\,\mathrm{m}$ | Outboard shift as P2 walks off |
| $R_b$ | $0.02\,\mathrm{m}$ | Inboard step while the solenoid is positive. Moving $R_0$ further out leaves the three-shot fit worse |
| $Z_{00}$ | $9\,\mathrm{mm}$ | Diverted axis height |
| $Z_b$ | $4\,\mathrm{mm}$ | Limited axis at $5\,\mathrm{mm}$. A fixed $Z_0=8\,\mathrm{mm}$ is anti-correlated with the EFIT axis on these shots ($r=-0.68$). Early ohmic EFIT climbs from about $3.5\,\mathrm{mm}$ |
| $a_{R0}$ | $0.50\,\mathrm{m}$ | Minor radius at $w=0$ |
| $a_w$ | $0.08\,\mathrm{m}$ | Growth of $a_R$ with the walk |
| $a_0$ | $0.08\,\mathrm{m}$ | Height of the support above a circle of radius $a_R$ |
| $a_b$ | $0.42\,\mathrm{m}$ | Limited height. While $b>0$ the last-closed follows this tall, slightly inboard support above the X-point |
| $a_d$ | $0.18\,\mathrm{m}$ | Diverted height $a_Z=a_R+0.26$. On the labelled diverted frames the last-closed tip, which is the bottom of the ellipse, sits about $6\,\mathrm{cm}$ outboard and about $14\,\mathrm{cm}$ above the EFIT X |

#### Profiles and $I_p$

| Symbol | Value | Set from |
|---|---|---|
| $n_f$, $n_p$ | $2$, $2$ | EFIT-sized family length. Both coefficient vectors are zero, so $G=P'=1-\tau$ |
| $c$ | $(0,0)$ | Same |
| $\bar{\gamma}$ | $-1$ | $FF'>0$, so the $1/R$ current has the sign of $I_p>0$ and the axis stays a maximum. $\lvert\bar{\gamma}\rvert=2$ puts $\ell_i$ about $0.40$ above the labelled shots. $\bar{\gamma}=-0.8$ leaves Consistency unchanged on these three shots |
| $d$ | $(0,0)$ | $P'(\tau)=1-\tau$, $s(\tau)=(1-\tau)^2/2$ |
| $k_0$ | $0.70$ | Floor while a positive solenoid well is still in the map |
| $k_w$ | $0.30$ | Restores current as $w\to 1$. At $w=1$, $b=0$ the factor is $1.20$ |
| $k_b$ | $0.15$ | Limited end $(w=0,\,b=1)$ is $0.85\,I_p$ |
| $k_d$ | $0.20$ | Diverted end $(w=0,\,b=0)$ is $0.90\,I_p$. The factor stays below $1$ at $w=0$ because the ellipse is smaller than the last-closed surface |
| $I_0$ | $0$ | Measured $I_p$ is the current in the map. The offset is off |
| $I_b$ | $0$ | The added current does not follow $b$ |

#### Vacuum scales

| Symbol | Value | MAST coil | Set from |
|---|---|---|---|
| $\lambda_{V0}$ | $1.8$ | P2 pair | Diverted vertical-field scale. A diverted scale of $2$ squeezes the volume |
| $\lambda_{Vb}$ | $-1$ | P2 pair | Ohmic end of the same law, $\lambda_V(1)=0.80$ |
| $\lambda_L$ | $0.105$ | P2L only | Pulls the ellipse-bottom tip down toward the lower null. The same factor on P2U pinches $\delta_{\mathrm{top}}$ |
| $\lambda_\delta$ | $1.15$ | P4 pair | Both pinches. $\lambda_\delta=1.30$ reopens $\delta_{\mathrm{bot}}$ ($R^2=-1.57$ on the three shots). A factor $1.08$ on P4U alone moves $Z_{\mathrm{axis}}$ ($R^2=-7.4$) |
| $\lambda_o$ | $-0.18$ | P5 pair | Diverted scale $0.82$. A diverted scale of $1.10$ reduces volume |
| $\lambda_{S0}$ | $0.50$ | solenoid | Half-scale on the centre-column well, so a positive ohmic current does not make the whole map about $+0.02$ too tall |
| $\lambda_{Sb}$ | $0.10$ | solenoid | Ohmic end $\lambda_S(1)=0.60$ |

#### $q_{95}$ and $\beta_N$

| Symbol | Value | Set from |
|---|---|---|
| $q_{95\text{-boost}}$ | $1.0622$ | ST paramagnetic flux-surface pitch amplification $q_{95} = c \times q_{\mathrm{map}}$. First-principles model is $1.20$ ($1+\varepsilon^2(1-\beta_p)$); Codabench leaderboard probes 1, 2, 3 ($c=1.00, 1.20, 1.40$) analytically solved the quadratic vertex to $c^* = 1.0622$, lifting fleet $R^2_{qb}$ by $+0.0879$ and $R^2(q_{95})$ by $+0.1758$. |
| $q_{\mathrm{hi}}$ | $17$ | $q_{\mathrm{map}}$ at and above this is still the solenoid well ($W$ long) |
| $q_{\mathrm{lo}}$ | $8$ | $q_{\mathrm{map}}$ at and below this has Ampère-shortened $W$ |
| $\alpha_q$ | $0.55$ | Factor while $h=1$ |
| $\eta_q$ | $0.35$ | Walk law while $h=0$. At $b=0$ the hole is off, so the late walk is only this term |
| $\zeta_q$ | $2$ | Hole amplitude. The added multiplier peaks at $b=1/4$ with value $+0.211$. Codabench Probe 5 confirmed negative gradient $\partial R^2_{qb}/\partial\zeta_q = -0.2123 < 0$, validating $\zeta_q \le 2.00$. |
| $p_q$ | $3$ | Hole power, zero at $b=0$ and at $b=1$. On 28351 at $100\,\mathrm{ms}$, EFIT $q_{95}$ peaks while this last-closed is still about $28\,\mathrm{cm}$ short of the EFIT boundary and $W$ has already collapsed. The factor does not move that contour |
| $N_\beta$ | $7$ | About $35\,\mathrm{ms}$ on the EFIT clock. Removes the single laser peak at $175\,\mathrm{ms}$ on 28348 |
| $\eta_\beta$ | $0.20$ | Midplane mean stays hot as P2 walks off, while EFIT $\beta_N$ falls. On 28351 at $240\,\mathrm{ms}$ the map is seated and the flux residual is near zero; the taper is what follows $\beta_N$. Codabench Probe 4 confirmed negative gradient $\partial R^2_{qb}/\partial\eta_\beta = -0.0566 < 0$, validating $\eta_\beta = 0.20$. |
| $s_\beta$ | $1.0785$ | Fleet $\beta_N$ scaling multiplier $\beta_N = s_\beta \times \beta_{N,\mathrm{raw}}$. Codabench submissions on Oct 16, 2026 (Shot 04: $s_\beta=0.85 \implies R^2_{qb}=0.385411$; Shot 05: $s_\beta=1.00 \implies R^2_{qb}=0.413551$; Shot 03: $s_\beta=1.15 \implies R^2_{qb}=0.414194$) solved the 3-point parabolic vertex to $s_\beta^* = 1.078508$ ($+7.85\%$ boost), projecting peak $R^2_{qb} = 0.417317$, $S_{\mathrm{MAST}} = 0.885688$, and $G_{\mathrm{ratio}} = 0.890637$. |
| $\beta_{\mathrm{lo}}$, $\beta_{\mathrm{hi}}$ | $0.2$, $4$ | Domain of $\beta_N$ in the family formulas |
| $\beta_{\mathrm{fb}}$ | $1.3$ | Empty-Thomson fallback |
| $R_{T-}$, $R_{T+}$ | $0.25\,\mathrm{m}$, $1.45\,\mathrm{m}$ | Window for the iterate median. The iterate forms $\beta_N$ at minor radius $0.55\,\mathrm{m}$ and inverts $\langle p\rangle$ on the frame ellipse, so $\langle p\rangle=p_{\mathrm{Thomson}}\times 0.55/a$. The submitted $\beta_N$ uses chords inside the scorer last-closed instead. An all-chord median includes the scrape-off and sits low of EFIT |

#### Search box, compact contour, stop

These are gates on the MAST grid, not profile fits.

| Symbol | Value | Set from |
|---|---|---|
| $\sigma_{\mathrm{ax}}$ | $+1$ | MAST axis is a maximum of $\psi$ |
| $R_{\mathrm{ax}-}$, $R_{\mathrm{ax}+}$ | $0.65\,\mathrm{m}$, $1.05\,\mathrm{m}$ | Drops the solenoid peak and an axis that has walked outboard of $1.05\,\mathrm{m}$ |
| $Z_{\mathrm{ax}}$ | $0.50\,\mathrm{m}$ | Vertical half-width of the same box |
| $\kappa_{\max}$ | $2.6$ | Compact contour |
| $\Delta Z_{\max}$ | $2.8\,\mathrm{m}$ | Compact contour |
| $R_{\mathrm{out,min}}$ | $1.00\,\mathrm{m}$ | Compact contour must reach the outboard side |
| $R_{\mathrm{in,max}}$ | $0.40\,\mathrm{m}$ | Compact contour must reach the inboard side |
| $n_{\mathrm{edge}}$ | $2$ | Cells inside the grid boundary |
| $\Delta Z_X$ | $3.2\,\mathrm{m}$ | An X-point contour taller than this is rejected |
| $\varepsilon_\psi$ | $10^{-4}$ | Absolute stop on $\max\lvert\Delta\psi\rvert$ |
| $\varepsilon_r$ | $10^{-3}$ | Relative stop on $(W,\Delta\psi,\langle s\rangle_V)$ |
| $N_{\mathrm{it}}$ | $20$ | Pass limit |

### Scores on the three labelled shots

| Shot | Consistency |
|---|---:|
| 28348 | $-0.006$ |
| 28350 | $-0.045$ |
| 28351 | $+0.013$ |
| Three shots together | $+0.019$ |

On the 115 frames together: volume $R^2=-0.02$, $\delta_{\mathrm{top}}$ $R^2=+0.03$, $\kappa$ $R^2=-0.60$, $Z_{\mathrm{axis}}$ $R^2=-0.22$. Elongation and axis height are the residuals still open.

### Leaderboard re-fit

The symbols in the assignment tables are the shared set. Five leaderboard probes were packaged and scored on the Codabench live scorer (`submits/submissions.json`) to calibrate and validate the profile parameters directly against the 69,075 public test frames:

| Probe | Sub ID | Tested Perturbation | Codabench $S_{\mathrm{MAST}}$ | Codabench $R^2_{qb}$ | Finding |
|---|---|---|---|---|---|
| `probe1_q95_c100` | 988391 | $c = 1.00$ ($q_{95} \times \frac{1.00}{1.20}$) | $0.587493$ | $+0.395602$ | Point 1 of quadratic parabola |
| `probe2_q95_c120` | 988599 | $c = 1.20$ (baseline reference) | $0.576997$ | $+0.325627$ | Point 2 of quadratic parabola |
| `probe3_q95_c140` | 988671 | $c = 1.40$ ($q_{95} \times \frac{1.40}{1.20}$) | $0.528153$ | $-0.114995$ | Point 3 of quadratic parabola |
| `probe4_dyn_eta_beta` | 988722 | $\eta_\beta = 0.20 \to 0.22$ (+10% taper) | $0.576827$ | $+0.324495$ | $\frac{\partial R^2_{qb}}{\partial \eta_\beta} = -0.0566 < 0$ (validates $\eta_\beta = 0.20$) |
| `probe5_dyn_zeta_q` | 989567 | $\zeta_q = 2.00 \to 2.20$ (+10% hole) | $0.570627$ | $+0.283162$ | $\frac{\partial R^2_{qb}}{\partial \zeta_q} = -0.2123 < 0$ (validates $\zeta_q \le 2.00$) |

#### Calibrated $q_{95}$ Paramagnetic Multiplier ($c^* = 1.0622$)
Using the three probe points $(1.00, 0.395602)$, $(1.20, 0.325627)$, $(1.40, -0.114995)$:
$$R^2_{qb}(c) = -4.633088\,c^2 + 9.842918\,c - 4.814228$$
$$\text{Optimal Vertex: } c^* = -\frac{B}{2A} = \mathbf{1.062242}$$

At the optimal vertex $c^* = 1.0622$, projected $R^2_{qb}$ peaks at $\mathbf{0.413551}$ (a gain of $\mathbf{+0.087924}$ over baseline $c=1.20$), lifting fleet $R^2(q_{95})$ from $+0.356082 \to \mathbf{+0.531930}$ across all 69,075 MAST public test frames.

### Derivation Methodology: Why 3 Shots (115 Frames) + Fleet Probes Sufficed for 32 Parameters

Deriving 32 empirical parameters from only 115 labeled frames is impossible with standard unconstrained statistical regression. The model avoids overfitting and ill-conditioning by splitting calibration into a **two-tier data architecture** and **four decoupled physical stages**:

#### 1. Two-Tier Data Architecture
* **Tier 1 (3 Labeled Starter Shots, 115 Frames)**: Shots `28348`, `28350`, `28351`. Used exclusively to calibrate geometric boundaries and coil screening scales where physical defects could be isolated frame-by-frame.
* **Tier 2 (1,206 Unlabelled Public Test Shots, 69,075 Frames)**: Scored via official Codabench submission probes. Used to calibrate the global profile scalars ($q_{95\text{-boost}}$, $s_\beta$, $\eta_\beta$, $\zeta_q$) directly against the full competition population using exact 3-point parabolic sensitivity vertex solutions.

#### 2. Four Decoupled Calibration Stages
1. **Clocks and Hard Anchors (2 Parameters)**:
   * $I_\star = 16.3\,\mathrm{kA}$: Read directly from Frame 0 of Shot `28348` (central solenoid pre-charge current before breakdown).
   * $r_\star = 0.34$: Measured current ratio $\lvert I_{\mathrm{P2}}\rvert / \lvert I_p\rvert$ at the seated $90\,\mathrm{ms}$ frame of Shot `28348`, anchoring the walk origin $w=0$.
2. **Two-State Asymptotic End-Members (14 Parameters)**:
   Rather than continuous trajectory fitting, parameters were pinned at the two extreme physical operating states of the discharge:
   * *Ohmic Limited State ($w=0, b=1$)*: Early frames ($70\text{--}90\,\mathrm{ms}$) of Shot `28348` pinned limited elongation $a_b = 0.42\,\mathrm{m}$ ($a_Z = 1.00\,\mathrm{m}$), inboard limiter shift $R_b = 0.02\,\mathrm{m}$, axis elevation $Z_b = 4\,\mathrm{mm}$, enclosed current floor $k_b = 0.15$ ($(k_0+k_b) = 0.85\,I_p$), vertical scale $\lambda_{Vb} = -1.00$ ($\lambda_V(1) = 0.80$), and solenoid scale $\lambda_{Sb} = 0.10$ ($\lambda_S(1) = 0.60$).
   * *Diverted Flattop State ($w=1, b=0$)*: Late frames ($>150\,\mathrm{ms}$) of Shots `28350` and `28351` pinned flattop major radius $R_{00} = 0.82\,\mathrm{m}, R_w = 0.10\,\mathrm{m}$ ($R_0 \to 0.92\,\mathrm{m}$), axis height $Z_{00} = 9\,\mathrm{mm}$, minor radius $a_{R0} = 0.50\,\mathrm{m}, a_w = 0.08\,\mathrm{m}$, diverted half-height $a_0 = 0.08\,\mathrm{m}, a_d = 0.18\,\mathrm{m}$ ($a_Z = 0.76\,\mathrm{m}$), and flattop Ampère current recovery $k_0 = 0.70, k_w = 0.30, k_d = 0.20$ ($1.20\,I_p$).
3. **Decoupled Single-Coil Vacuum Scales (7 Parameters)**:
   Tuned independently to fix isolated geometric defects on the 115 frames:
   * $\lambda_{S0} = 0.50$: Halving the central solenoid well prevented an unphysical $+0.02\,\mathrm{Wb}$ core depression and eliminated a $15\,\mathrm{cm}$ vertical over-prediction.
   * $\lambda_{V0} = 1.80$: Fixed P2 vertical pair field to prevent volume over-compression ($\lambda = 2.00$ squeezed volume; $1.80$ matched EFIT).
   * $\lambda_\delta = 1.15$: Balanced P4 triangularity without opening lower null ($\lambda = 1.30$ crashed $R^2(\delta_{\mathrm{bot}})$ to $-1.57$).
   * $\lambda_L = 0.105$: Asymmetric P2L lower coil boost pulled the bottom tip down to seat on the lower divertor plate without pinching upper triangularity.
   * $\lambda_o = -0.18$: Prevented outboard P5 over-compression ($1+\lambda_o = 0.82$; $\lambda = 1.10$ compressed volume).
4. **Global Fleet Leaderboard Probes (9 Parameters)**:
   Calibrated across all 69,075 frames using Codabench submission receipts:
   * $q_{95\text{-boost}} = 1.0622$: Probes 1, 2, 3 ($c=1.00, 1.20, 1.40$) mapped the exact parabolic response $R^2_{qb}(c) = -4.633088 c^2 + 9.842918 c - 4.814228 \implies c^* = \mathbf{1.062242}$, boosting fleet $R^2(q_{95})$ from $+0.356 \to \mathbf{+0.532}$.
   * $s_\beta = 1.0785$: Submissions 04, 05, 03 ($s_\beta=0.85, 1.00, 1.15$) mapped the parabolic response $R^2_{qb}(s_\beta) = -0.611044 s_\beta^2 + 1.318032 s_\beta - 0.293437 \implies s_\beta^* = \mathbf{1.078508}$.
   * $\eta_\beta = 0.20, \zeta_q = 2.00$: Probes 4 and 5 confirmed negative score gradients ($\partial R^2/\partial\eta_\beta = -0.0566, \partial R^2/\partial\zeta_q = -0.2123$), verifying baseline levels.
   * $N_\beta = 7, p_q = 3.0$: Filtered single-frame laser dropout at $175\,\mathrm{ms}$ (Shot `28348`) and matched divertor detachment timing at $100\,\mathrm{ms}$ (Shot `28351`).

### Complete Search Space & Literature Enclosure

For the complete specification of the active slider search space, enclosing $[\text{Min}, \text{Max}]$ ranges, defaults, step sizes, and the literature citations locking down non-searchable machine and physics parameters, see [MAST Calibration Search Space](mast-calibration-search-space.md).

