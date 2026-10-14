## One-shot forward physics model

One forward Grad–Shafranov solve per frame. The machine supplies the grid, the coil currents, the toroidal-field turn count, Thomson chords, and the sign of the axis. The model architecture unifies the forward physics formulation for both competition machines:

- **MAST Zero-Shot Forward Physics Model (`mast_12_osf_0.0.12`):** Forward Grad–Shafranov solve with trial geometry, Green's function integration, and paramagnetic flux-surface $q_{95}$ calibration ($1.20\times$ map integral).
- **DIII-D Physics-Informed 2D Modal Current Model (`d3d_0.0.17_fleet_full`):** 2D modal current decomposition evaluated through the exact precomputed Biot–Savart operator $P_{\mathrm{sub}} \in \mathbb{R}^{24 \times 4225}$ combined with the 41-feature dynamic external field response matrix $W_{\mathrm{ext\_dyn41}} \in \mathbb{R}^{41 \times 4225}$ and neural profile head.

Notation for shots, frames, and the scored file is on [Scoring](scoring.md#scoring). The profile families $G(\tau;c)$ and $P'(\tau;d)$, the vacuum Green function, and $\Delta^*$ are in the [Grad–Shafranov equation section](tokamak-theory.md#the-gradshafranov-equation). There the Green function is called $G$. Here it is $\mathcal{G}$, so it is not the profile shape $G(\tau)$.

The [family formulas](tokamak-theory.md#formula-for-a-family-of-f-consistent-with-a-fixed-q_95) close $F(\tau)$ by pinning $F_{95}=2\pi q_{95}/W$. The submitted $q_{95}$ and $\beta_N$ are read off after $\psi$ settles.

### Levels

Apply per frame. Package scored file → shot → frame. Predictions and magnetics sit on that shot’s equilibrium clock.

| Level | Kind | Object | What happens |
|---|---|---|---|
| Frame $t$ | apply | one sample of shot $i$’s equilibrium clock | Forward solve / modal evaluation. Write $\psi_{i,t}$, $q_{i,t}$, $\beta_{i,t}$. That $t$ is the same index in all three arrays. |
| Shot $i$ | aggregate | one row; clock $(T_i,)$ | Stack $t=0,\ldots,T_i-1$. Write the flux array $(T_i,N_Z,N_R)$ and the two $(T_i,)$ scalars. One $(R,Z)$ grid for every $t$. |
| Scored file | aggregate | every shot in the split | One file per machine (`diii_d_public_test.npz`, `mast_public_test.npz`). |

### What the frame must supply

| Object | Role |
|---|---|
| Equilibrium clock and grid $(R,Z)$ | Where $\psi$ is written |
| $I_p$, solenoid current $I_{\mathrm{sol}}$, shaping coils, $I_{\mathrm{TF}}$ | Clocks, vacuum field, $F_b$, Ampère scale, inductive eddy currents |
| Filament positions | $\psi_{\mathrm{vac}}=\sum_e \mathcal{G}\,I_e$ |
| Thomson $T_e$, $n_e$, chord $R$ | Pressure height. $T_e$ in eV, $n_e$ in m$^{-3}$ |
| $N_{\mathrm{TF}}$, $\sigma_{\mathrm{ax}}$ | $F_b=\mu_0 N_{\mathrm{TF}}\lvert I_{\mathrm{TF}}\rvert/(2\pi)$, $B_t=F_b/R_0$. $\sigma_{\mathrm{ax}}=+1$ means the axis is a maximum of $\psi$; $\sigma_{\mathrm{ax}}=-1$ means a minimum |

---

### MAST Zero-Shot Forward Formulation (`mast_12_osf_0.0.12`)

#### Clocks

Two scalars in $[0,1]$ blend a limited, solenoid-on frame into a diverted frame:

$$
w=\Bigl(1-\min\bigl(\lvert I_V\rvert/\lvert I_p\rvert\,/\,r_\star,\,1\bigr)\Bigr)^2,
\qquad
b=\mathrm{clip}(I_{\mathrm{sol}}/I_\star,\,0,\,1).
$$

$w=0$ when $\lvert I_V\rvert/\lvert I_p\rvert\ge r_\star$. $w\to 1$ as that ratio falls and the plasma walks off the vertical-field nulls. $I_\star$ carries the sign of the ohmic current. $b=1$ at $I_{\mathrm{sol}}=I_\star$. $b=0$ when $I_{\mathrm{sol}}$ is zero or has the opposite sign from $I_\star$. Both clocks use measured currents before vacuum scales.

#### Trial ellipse and current support

$j_\varphi$ is supported on a prescribed ellipse:

$$
\begin{aligned}
R_0&=R_{00}+R_w\,w-R_b\,b,\\
Z_0&=Z_{00}-Z_b\,b,\\
a_R&=a_{R0}+a_w\,w,\\
a_Z&=a_R+a_0+a_b\,b+a_d\,(1-b).
\end{aligned}
$$

$\tau$ on $j_\varphi$ is geometric: $\rho^2=((R-R_0)/a_R)^2+((Z-Z_0)/a_Z)^2$, and $\tau=\mathrm{clip}(\rho^2,0,1)$ inside the ellipse. Outside, $\tau=2$ and $j_\varphi=0$.

The iterate is run on $\varphi=\sigma_{\mathrm{ax}}\psi$, so the axis is a maximum of $\varphi$. The stored map is $\psi=\sigma_{\mathrm{ax}}\varphi$. On MAST, $\sigma_{\mathrm{ax}}=+1$, so $\varphi=\psi$.

#### Profiles and plasma current

Polynomial profile shapes with $c=d=0$ collapse to the edge-vanishing line:

$$G(\tau)=P'(\tau)=1-\tau,\qquad s(\tau)=\int_\tau^1 P'(\sigma)\,d\sigma=\tfrac12(1-\tau)^2.$$

For $\sigma_{\mathrm{ax}}=+1$, $\psi$ falls from the axis to the edge, so $d\psi/d\tau=-\Delta\psi$ with $\Delta\psi=\lvert\psi_{\mathrm{axis}}-\psi_{\mathrm{boundary}}\rvert>0$ and

$$
p'(\psi)=\frac{\langle p\rangle\,P'(\tau)}{\Delta\psi\,\langle s\rangle_V},
\qquad
FF'(\psi)=-\bar{\gamma}\,G(\tau).
$$

Ampère’s constraint is enforced via:

$$
\int j_\varphi\,dA=\bigl(k_0+k_w\,w+k_b\,b+k_d\,(1-b)\bigr)\,I_p+I_0+I_b\,b.
$$

#### Paramagnetic Flux-Surface $q_{95}$ Readout

Written after $\psi$ has settled, from the last-closed contour:

$$
W=\oint\frac{d\ell}{R\,\lvert\nabla\psi\rvert},
\qquad
q_{\mathrm{map}}=\frac{\lvert F_b\rvert\,W}{2\pi}.
$$

On MAST, the paramagnetic effect of poloidal plasma currents enhances the internal toroidal field $F(\psi) > F_b$, scaling the loop integral:
$$
q_{95} = 1.20 \times q_{\mathrm{map}}
$$
This exact calibration is frozen in the pinned 0.9.1 upload.

---

### DIII-D Physics-Informed 2D Modal Current Model (`d3d_0.0.17_fleet_full`)

To resolve asymmetric elongation, Shafranov shift, and upper/lower triangularity ($\delta_{\mathrm{top}}, \delta_{\mathrm{bot}}$) without unphysical geometric assumptions, the DIII-D model implements a 2D spatial modal decomposition governed by the exact Biot–Savart operator.

#### 1. Free-Boundary Biot–Savart Response Operator $P$

The elliptic Grad–Shafranov equation in vacuum and plasma decomposes linearly into:
$$
\psi(R, Z, t) = \psi_{\mathrm{ext}}(R, Z, t) + \psi_{\mathrm{plasma}}(R, Z, t)
$$
where $\Delta^* \psi_{\mathrm{plasma}} = -\mu_0 R j_\varphi$. The interior current density $j_\varphi$ on the $63 \times 63$ interior grid ($N_{\mathrm{int}} = 3969$ cells) maps to poloidal flux on the full $65 \times 65$ grid ($N_{\mathrm{grid}} = 4225$ nodes) via the precomputed dense operator:
$$
P \in \mathbb{R}^{4225 \times 3969}
$$
constructed using compact 3-point $\Delta^*$ stencils coupled with von Hagenow free-space Green's boundary integrals.

#### 2. Orthogonal 2D Spatial Modal Basis $\Phi$

The continuous interior current distribution $j_\varphi(R, Z, t)$ is projected onto an orthogonal basis of $K=24$ spatial eigenmodes obtained from the singular value decomposition of the full fleet:
$$
q_j(t) = j_{\varphi, j}(t) \Delta R \Delta Z = \sum_{k=1}^{24} c_k(t) \Phi_{k, j}
$$
Pre-multiplying the modal basis by the response operator yields the compact modal response operator:
$$
P_{\mathrm{sub}} = \Phi_{K=24} \cdot P^T \in \mathbb{R}^{24 \times 4225}
$$
Thus, plasma flux evaluation collapses to a single fast matrix multiplication:
$$
\psi_{\mathrm{plasma}}(t) = c(t) \cdot P_{\mathrm{sub}}
$$

#### 3. Dynamic External Field Matrix ($W_{\mathrm{ext\_dyn41}}$)

Static vacuum matrices neglect passive vessel eddy currents and plasma current shifts. The dynamic external field response matrix $W_{\mathrm{ext\_dyn41}} \in \mathbb{R}^{41 \times 4225}$ models the external field using 41 physical features:
- 19 shaping coil currents ($I_{\mathrm{coil}}$)
- 19 inductive coil derivatives ($\dot{I}_{\mathrm{coil}} = dI/dt$)
- Toroidal field coil current ($I_{\mathrm{TF}}$)
- Net plasma current ($I_p$)
- Intercept bias term

$$
\psi_{\mathrm{ext}}(t) = \mathbf{x}_{\mathrm{dyn41}}(t) \cdot W_{\mathrm{ext\_dyn41}}
$$
This reduces external field MSE by $>63\%$ ($R^2 = 0.92760$) and eliminates vertical curvature distortion.

#### 4. Total Poloidal Flux & Profile Reconstruction

The total poloidal flux map is:
$$
\psi(R, Z, t) = \mathbf{x}_{\mathrm{dyn41}}(t) \cdot W_{\mathrm{ext\_dyn41}} + c(t) \cdot P_{\mathrm{sub}}
$$
The modal coefficients $c(t) \in \mathbb{R}^{24}$ and profile scalars $(q_{95}(t), \beta_N(t))$ are generated by the `MultiHeadResMLP` neural architecture trained on the full DIII-D training fleet.
