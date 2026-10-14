## DIII-D Machine Learning Fit to Labelled Data

This document details the architecture, mathematical formulation, training protocol, and evaluation of the physics-informed One-Shot Forward (OSF) modal model (`d3d_0.0.17_fleet_full`) trained on the full DIII-D labelled fleet.

---

### Physics-Informed 2D Modal Current Equilibrium Model

Rather than predicting raw pixel arrays directly with black-box neural networks (which suffer from numerical non-smoothness and unphysical field gradients), the model unifies machine learning parameter estimation with exact magnetohydrodynamic (MHD) elliptic operators.

```
                    Allowed Tokamak Diagnostic Signals u(t)
    (19 Shaping Coils F1A-F9B & ECOILA, Ip, Bt, Thomson proxy, Coil derivatives dI/dt)
                                      │
                                      ▼
       ┌─────────────────────────────────────────────────────────────┐
       │     64-Dimensional Aligned Physical Feature Engine          │
       │  Currents, dimensionless ratios I_coil/|Ip|, coil slopes    │
       └──────────────────────────────┬──────────────────────────────┘
                                      │
                                      ▼
       ┌─────────────────────────────────────────────────────────────┐
       │      Multi-Head Residual MLP (MultiHeadResMLP)              │
       │   Stem: Linear(64->256) -> LayerNorm -> SiLU               │
       │   Blocks: 2x Pre-activation ResBlocks (256->256)            │
       └──────────────┬──────────────────────────────┬───────────────┘
                      │                              │
                      ▼                              ▼
       ┌─────────────────────────────┐┌──────────────────────────────┐
       │     Modal Current Head      ││     Scalar Profile Head      │
       │  Linear(256->128) -> SiLU   ││   Linear(256->64) -> SiLU    │
       │  -> Linear(128->24)         ││   -> Linear(64->2)           │
       │  Outputs: c_k(t) (k=1..24)  ││   Outputs: q95(t), betaN(t)  │
       └──────────────┬──────────────┘└──────────────────────────────┘
                      │
                      ▼ Modal Current Projection
       ┌─────────────────────────────────────────────────────────────┐
       │        Discrete Biot–Savart Response Operator               │
       │  P_sub = Φ_24 · P^T  ∈ R^(24 x 4225)                        │
       │  ψ_plasma(R, Z, t) = c(t) · P_sub                           │
       └──────────────────────────────┬──────────────────────────────┘
                                      │
                                      ▼ Dynamic External Field Summation
       ┌─────────────────────────────────────────────────────────────┐
       │  ψ(R, Z, t) = x_dyn41(t) · W_ext_dyn41 + ψ_plasma(R, Z, t)   │
       │  Outputs: Full-resolution 65x65 poloidal flux map           │
       └─────────────────────────────────────────────────────────────┘
```

---

### Core Mathematical Foundations

#### 1. Free-Boundary Biot–Savart Response Operator $P$

In ideal MHD equilibrium, poloidal flux satisfies the Grad–Shafranov equation:
$$
\Delta^* \psi = -\mu_0 R j_\varphi
$$
where $\Delta^* \equiv R \frac{\partial}{\partial R} \left( \frac{1}{R} \frac{\partial}{\partial R} \right) + \frac{\partial^2}{\partial Z^2}$.

The total flux decomposes linearly into vacuum coil fields and internal plasma response:
$$
\psi(R, Z) = \psi_{\mathrm{ext}}(R, Z) + \psi_{\mathrm{plasma}}(R, Z)
$$
The plasma contribution is governed by the discrete elliptic boundary-value problem:
$$
\psi_{\mathrm{plasma}} = P \cdot q
$$
where $q_j = j_{\varphi, j} \cdot \Delta R \Delta Z$ is the toroidal current in interior cell $j$ ($N_{\mathrm{int}} = 3969$), and $P \in \mathbb{R}^{4225 \times 3969}$ is the precomputed dense response operator:
$$
P = \begin{bmatrix} -L^{-1} \left( \mu_0 \operatorname{diag}(R)/(\Delta R \Delta Z) + C B \right) \\ B \end{bmatrix}
$$
Here $B$ is the free-space Green's boundary block and $L$ is the compact 3-point elliptic stencil. Every generated field automatically satisfies $\Delta^* \psi = -\mu_0 R j_\varphi$ by construction.

#### 2. Orthogonal 2D Spatial Modal Basis

The interior current distribution is decomposed into $K=24$ orthogonal spatial modes:
$$
j_\varphi(R, Z, t) = \sum_{k=1}^{24} c_k(t) \Phi_k(R, Z)
$$
- **Mode 1:** Ohmic monopole current channel ($95.17\%$ variance).
- **Mode 2:** Shafranov shift and elongation $\kappa$ ($99.59\%$ cumulative variance).
- **Modes 3–6:** Asymmetric upper and lower triangularity $\delta_{\mathrm{top}}, \delta_{\mathrm{bot}}$ ($99.96\%$ cumulative variance).
- **Modes 7–24:** Higher-order internal squareness and current peaking $\ell_i$.

By contracting the basis directly with the response operator:
$$
P_{\mathrm{sub}} = \Phi_{K=24} \cdot P^T \in \mathbb{R}^{24 \times 4225}
$$
computing the plasma flux field requires only a single matrix multiplication:
$$
\psi_{\mathrm{plasma}}(t) = c(t) \cdot P_{\mathrm{sub}}
$$

#### 3. Dynamic External Field Matrix ($W_{\mathrm{ext\_dyn41}}$)

Static vacuum matrices neglect passive vessel eddy currents induced during fast coil current ramps and plasma image currents. The 41-feature dynamic external field response matrix $W_{\mathrm{ext\_dyn41}}$ incorporates:
- 19 shaping coil currents ($I_{\mathrm{coil}}$ from `ECOILA` + `F1A`–`F9B`)
- 19 coil time derivatives ($\dot{I}_{\mathrm{coil}} = dI_{\mathrm{coil}}/dt$)
- Toroidal field coil current ($I_{\mathrm{TF}}$ from `bcoil`)
- Net plasma current ($I_p$)
- Intercept bias term

$$
\psi_{\mathrm{ext}}(t) = \mathbf{x}_{\mathrm{dyn41}}(t) \cdot W_{\mathrm{ext\_dyn41}}
$$
This reduces external field MSE by $>63\%$ ($R^2 = 0.92760$) and resolves the vertical curvature distortion that previously inflated the computed plasma volume.

---

### Fleet-Scale Training Protocol (`train_fleet_full.py`)

1. **Dataset**: All 6,961 training shots from `diii_d_train` (~1.59 million frames).
   - 6,861 training shots (~1.57 million frames).
   - 100 validation shots (22,918 frames).
2. **Hardware Optimization**: High-occupancy batch size ($B = 610,000$) maintaining $13.05\,\mathrm{GiB}$ VRAM occupancy on the local 16 GB RTX 5060 Ti.
3. **Training Dynamics**:
   - Optimizer: AdamW ($\mathrm{lr} = 1 \times 10^{-3}$, weight decay $1 \times 10^{-4}$).
   - Loss Function: Multi-task normalized MSE:
     $$
     \mathcal{L} = \frac{1}{K} \sum_{k=1}^{24} \frac{(c_k - \hat{c}_k)^2}{\sigma_{c_k}^2} + \lambda_q \frac{(q_{95} - \hat{q}_{95})^2}{\sigma_q^2} + \lambda_\beta \frac{(\beta_N - \hat{\beta}_N)^2}{\sigma_\beta^2}
     $$
   - Learning Rate Scheduler: `ReduceLROnPlateau(factor=0.5, patience=20, min_lr=1e-6)`.
   - Early Stopping: Triggered at Epoch 859 with best validation loss $0.111069$. Total training time: 10.29 minutes.

---

### Official 80-Shot Holdout Benchmark Scorecard

Evaluated on the official 80-shot holdout benchmark (21,500 frames, never seen during training or validation) using the official competition Marching Squares composite scorer:

| Metric | Static 19-Coil Baseline | **Dynamic 41-Feature (`W_ext_dyn41`)** | Impact / Delta |
|---|---|---|---|
| **Composite Score $S$** | $0.69594$ | **$0.71842$** | **$+0.02248$** |
| **Consistency** | $+0.43502$ | **$+0.52184$** | **$+0.08682$** |
| **Profile Fidelity $R^2_{\{q,\beta\}}$** | $0.99504$ | **$0.99504$** | Unaffected ($>0.99$) |
| **Boundary Contour $1 - D_{\mathrm{LCFS}}$** | $0.93421$ | **$0.93418$** | Maintained ($D_{\mathrm{LCFS}} \le 7.6\,\mathrm{cm}$) |
| **Poloidal Flux $R^2_\psi$** | $0.66593$ | **$0.66595$** | Preserved |
| **LCFS Fail Fraction** | $0.00\%$ | **$0.00\%$** | 0 failures across 21,500 frames |
| **Consistency Fail Fraction** | $0.00\%$ | **$0.00\%$** | 0 failures across 21,500 frames |

#### Consistency Factor Recovery

| Consistency Scalar | Static Baseline | Dynamic Matrix ($W_{\mathrm{ext\_dyn41}}$) | Recovery Delta | Status |
|---|---|---|---|---|
| $\kappa$ (Elongation) | $-1.5011$ | **$+0.3533$** | **$+1.8544$** | **Flipped to robustly positive** |
| $\text{volume}$ (Plasma Volume) | $-3.9973$ | **$-1.8869$** | **$+2.1104$** | **$+52.8\%$ MSE reduction** |
| $Z_{\mathrm{axis}}$ (Vertical Axis) | $+0.9690$ | **$+0.9691$** | $+0.0001$ | Near-perfect vertical alignment |
| $\delta_{\mathrm{bot}}$ (Lower Triangularity) | $+0.7039$ | **$+0.7039$** | $0.0000$ | Robustly $> 0.70$ |
| $\ell_i$ (Internal Inductance) | $+0.5971$ | **$+0.5971$** | $0.0000$ | Robustly $> 0.59$ |
| $\delta_{\mathrm{top}}$ (Upper Triangularity) | $+0.5398$ | **$+0.5398$** | $0.0000$ | Robustly $> 0.50$ |
| $R_{\mathrm{axis}}$ (Radial Axis) | $+0.2354$ | **$+0.2354$** | $0.0000$ | Monotonic climb |
