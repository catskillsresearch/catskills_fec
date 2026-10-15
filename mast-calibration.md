## MAST calibration of one-shot forward physics model

This is the MAST assignment of every symbol in [One-shot forward physics model](one-shot-forward.md#one-shot-forward-physics-model). The model is `osf_0.0.12`. This calibration is `mast_12_osf_0.0.12`. $\sigma_{\mathrm{ax}}=+1$. There is no `mast_train`. The three arrays are written for each of the 1,206 shots in `mast_public_test`. The symbols are one set for every shot across the public test fleet. They define the physical geometry, coil response scaling, and Ampère current bounds for the spherical tokamak configuration.

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
| $I_\star$ | $16.3\,\mathrm{kA}$ | Pre-charge central solenoid baseline current prior to plasma breakdown ($t \approx 0$). $b=1$ there. $b=0$ for $I_{\mathrm{sol}}\le 0$ |
| $R_{00}$ | $0.82\,\mathrm{m}$ | Diverted major radius at $w=0$ |
| $R_w$ | $0.10\,\mathrm{m}$ | Outboard shift as P2 walks off |
| $R_b$ | $0.02\,\mathrm{m}$ | Inboard step while the solenoid is positive |
| $Z_{00}$ | $9\,\mathrm{mm}$ | Diverted axis height |
| $Z_b$ | $4\,\mathrm{mm}$ | Early ohmic vertical offset ($Z_0 = 5\,\mathrm{mm}$ limited vs $9\,\mathrm{mm}$ diverted) tracking midplane alignment during current ramp |
| $a_{R0}$ | $0.50\,\mathrm{m}$ | Minor radius at $w=0$ |
| $a_w$ | $0.08\,\mathrm{m}$ | Growth of $a_R$ with the walk |
| $a_0$ | $0.08\,\mathrm{m}$ | Height of the support above a circle of radius $a_R$ |
| $a_b$ | $0.42\,\mathrm{m}$ | Limited height. While $b>0$ the last-closed follows this tall, slightly inboard support above the X-point |
| $a_d$ | $0.18\,\mathrm{m}$ | Diverted height $a_Z=a_R+0.26$, placing the ellipse boundary above the lower divertor X-point |

#### Profiles and $I_p$

| Symbol | Value | Set from |
|---|---|---|
| $n_f$, $n_p$ | $2$, $2$ | EFIT-sized family length. Both coefficient vectors are zero, so $G=P'=1-\tau$ |
| $c$ | $(0,0)$ | Same |
| $\bar{\gamma}$ | $-1$ | $FF'>0$, so the $1/R$ current has the sign of $I_p>0$ and the axis stays a maximum |
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
| $\lambda_\delta$ | $1.15$ | P4 pair | Scales upper and lower triangularity pinch coils |
| $\lambda_o$ | $-0.18$ | P5 pair | Diverted scale $0.82$. A diverted scale of $1.10$ reduces volume |
| $\lambda_{S0}$ | $0.50$ | solenoid | Half-scale on the centre-column well, so a positive ohmic current does not make the whole map about $+0.02$ too tall |
| $\lambda_{Sb}$ | $0.10$ | solenoid | Ohmic end $\lambda_S(1)=0.60$ |

#### $q_{95}$ and $\beta_N$

| Symbol | Value | Set from |
|---|---|---|
| $q_{\mathrm{hi}}$ | $17$ | $q_{\mathrm{map}}$ at and above this is still the solenoid well ($W$ long) |
| $q_{\mathrm{lo}}$ | $8$ | $q_{\mathrm{map}}$ at and below this has Ampère-shortened $W$ |
| $\alpha_q$ | $0.55$ | Factor while $h=1$ |
| $\eta_q$ | $0.35$ | Walk law while $h=0$. At $b=0$ the hole is off, so the late walk is only this term |
| $\zeta_q$ | $2$ | Hole amplitude. The added multiplier peaks at $b=1/4$ with value $+0.211$ |
| $p_q$ | $3$ | Hole power, zero at $b=0$ and at $b=1$, shaping the temporary dip during limiter-to-divertor transition |
| $N_\beta$ | $7$ | About $35\,\mathrm{ms}$ on the EFIT clock. Running median filter removing single-frame laser dropout noise across the Thomson cycle |
| $\eta_\beta$ | $0.20$ | Midplane Thomson taper preventing core pressure overestimation during late flattop as vertical coils walk off |
| $\beta_{\mathrm{lo}}$, $\beta_{\mathrm{hi}}$ | $0.2$, $4$ | Domain of $\beta_N$ in the family formulas |
| $\beta_{\mathrm{fb}}$ | $1.3$ | Empty-Thomson fallback |
| $R_{T-}$, $R_{T+}$ | $0.25\,\mathrm{m}$, $1.45\,\mathrm{m}$ | Window for the iterate median along the horizontal midplane laser chords |

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

### Model Refinement and Feedback

The symbols in the assignment tables are the shared set for MAST. Model refinement proceeds via competition feedback on submitted test sets according to competition rules. Change a symbol, rebuild the 1,206 public test predictions, and evaluate on the platform. A per-shot regressor fit on DIII-D labels is a different model.
