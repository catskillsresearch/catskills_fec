## Benchmark and Diagnostic Visualizations and Quality Measurements

This guide defines the diagnostic visualizations, quality metrics, and error patterns used to benchmark and validate tokamak equilibrium reconstructions.

All diagnostics described here are implemented interactively in the web viewer (`fec_site/`) and programmatically in `local_score.py` and `catskills/local_benchmark.py`.

---

## Per-Frame Diagnostics

One $(R,Z)$ map at time $t$. Evaluates 2D flux fidelity, contour geometry, and current distribution:

| Overlay | Quality Measurement | Role |
|---|---|---|
| $\psi_{\mathrm{GT}}$, $\hat\psi$, residual, histogram | $R^2_\psi$, pixel MSE | 2D poloidal flux reconstruction accuracy |
| Residual + $j_\varphi$ | $R^2_\psi$, Ampère ratio | Current channel alignment and magnetic force balance |
| Last-Closed Flux Surface (LCFS) | $D_{\mathrm{LCFS}}$ (Hausdorff distance) | Plasma boundary shape fidelity |
| X-point saddle zoom | Saddle coordinate delta $\Delta Z_X$ | Divertor null location and leg angle |
| Equal-$T_e$ Thomson pairs | Surface alignment fraction | Isothermal flux surface consistency |
| Modal current residual | $j_\varphi - \sum c_k \Phi_k$ | Basis truncation and higher-order moments |

### 1. Flux Residual Anatomy

Draw $\psi_{\mathrm{GT}}$, $\hat\psi - \psi_{\mathrm{GT}}$, and $j_\varphi$:
- **Ring pattern** in the residual indicates an overall scale or plasma current amplitude error.
- **Dipole pattern** (positive lobe adjacent to negative lobe) indicates a spatial centroid shift (Shafranov shift or vertical displacement).
- **Quadrupole pattern** indicates an elongation or triangularity distortion.
- An accurate physical model produces a near-zero, white residual map with error spikes concentrated only at the boundary grid cells.

### 2. LCFS Contour Overlay

Draw the predicted and ground-truth last-closed flux surfaces on the $(R,Z)$ grid:
- Uses the official Marching Squares contour extractor (`extract_lcfs`).
- Evaluates symmetric Hausdorff distance $D_{\mathrm{LCFS}}$ normalized by major radius $R_{\mathrm{geo}}$:
  $$
  1 - D_{\mathrm{LCFS}} = 1 - \frac{\max\left(\max_{p \in C_{\mathrm{pred}}} \min_{q \in C_{\mathrm{GT}}} \|p-q\|, \max_{q \in C_{\mathrm{GT}}} \min_{p \in C_{\mathrm{pred}}} \|p-q\|\right)}{R_{\mathrm{geo}}}
  $$
- Catches unphysical under-elongation ($\kappa$), missing triangularity ($\delta$), or divertor leg detachment.

### 3. Pinch / Saddle Zoom

In diverted configurations, zoom into the region surrounding the magnetic X-point:
- Identifies the hyperbolic saddle null ($\nabla\psi = 0$) where poloidal field vanishes.
- Verifies that the reconstructed separatrix bifurcates cleanly into private flux and divertor legs without grid jump artifacts.

### 4. Equal-$T_e$ Thomson Isothermal Diagnostic

In ideal MHD equilibrium, electron temperature is a flux function ($T_e = T_e(\psi)$):
- Chords measuring equal electron temperature inboard and outboard must lie on the same poloidal flux contour $\psi(R,Z) = \text{const}$.
- **Pair validity** measures the fraction of equal-$T_e$ chord pairs whose mapped flux agrees within $2\%$ of the total flux span $|\psi_{\mathrm{axis}} - \psi_{\mathrm{boundary}}|$.

---

## Per-Shot Diagnostics

Evaluates the temporal evolution of one discharge across its entire equilibrium clock ($t = 0, \ldots, T-1$):

### 1. Input Traces
- **Coil Currents**: Tracking shaping coil currents ($I_{\mathrm{coil}}$), solenoid ($I_{\mathrm{sol}}$), and toroidal field ($I_{\mathrm{TF}}$).
- **Inductive Slopes**: Time derivatives $\dot{I}_{\mathrm{coil}} = dI/dt$ driving vacuum vessel eddy currents.
- **Plasma Current**: Total Rogowski current $I_p(t)$ aligned with the magnetic reconstruction clock.

### 2. Output Scalar Time-Traces
- **Edge Safety Factor $q_{95}(t)$**: Safety factor at $95\%$ normalized poloidal flux, reflecting field line twist.
- **Normalized Beta $\beta_N(t)$**: Ratio of kinetic pressure to magnetic pressure normalized by Troyon scaling:
  $$
  \beta_N = 100 \cdot \beta_t \cdot \frac{a B_t}{|I_p|\ (\mathrm{MA})}
  $$
- **Ampère Balance**: Verification that the surface integral of toroidal current density recovers total plasma current:
  $$
  \frac{\iint j_\varphi\,dR\,dZ}{I_p} \approx 1.0
  $$

---

## Scoring Measures & Evaluation Contract

The competition evaluates submissions using four weighted composite terms:

$$
S = 0.55 \cdot R^2_\psi + 0.15 \cdot R^2_{\{q95, \beta_N\}} + 0.10 \cdot (1 - D_{\mathrm{LCFS}}) + 0.20 \cdot \mathrm{Consistency}
$$

### 1. Poloidal Flux ($R^2_\psi$, weight $0.55$)
Vectorized coefficient of determination over all pixels across all frames in the split:
$$
R^2_\psi = 1 - \frac{\sum_{i,t,z,r} (\psi_{\mathrm{pred}} - \psi_{\mathrm{GT}})^2}{\sum_{i,t,z,r} (\psi_{\mathrm{GT}} - \bar\psi_{\mathrm{GT}})^2}
$$

### 2. Profile Scalars ($R^2_{\{q, \beta\}}$, weight $0.15$)
Harmonic blend of safety factor and normalized beta fidelity:
$$
R^2_{\{q, \beta\}} = \frac{1}{2} \left[ \max(0, R^2(q_{95})) + \max(0, R^2(\beta_N)) \right]
$$

### 3. Boundary Fidelity ($1 - D_{\mathrm{LCFS}}$, weight $0.10$)
Mean normalized Hausdorff distance between extracted and ground-truth separatrices.

### 4. Consistency ($\mathrm{Consistency}$, weight $0.20$)
Mean of $R^2$ across seven geometric and physical equilibrium functionals derived from the flux map:
1. **$R_{\mathrm{axis}}$**: Radial position of the magnetic axis (O-point).
2. **$Z_{\mathrm{axis}}$**: Vertical position of the magnetic axis.
3. **$\kappa$**: Plasma elongation $(Z_{\max} - Z_{\min}) / (R_{\max} - R_{\min})$.
4. **$\delta_{\mathrm{top}}$**: Upper triangularity $(R_{\mathrm{geo}} - R_{Z_{\max}}) / a$.
5. **$\delta_{\mathrm{bot}}$**: Lower triangularity $(R_{\mathrm{geo}} - R_{Z_{\min}}) / a$.
6. **$\text{volume}$**: Enclosed plasma volume inside the LCFS: $\oint 2\pi R\,dA$.
7. **$\ell_i$**: Normalized internal inductance: $\frac{\langle B_p^2 \rangle_V}{(\mu_0 I_p / L)^2}$.

$$
\mathrm{Consistency} = \frac{1}{7} \sum_{k=1}^7 \max\left(0, R^2(f_k)\right)
$$

---

## Interactive Diagnostic Browser (`fec_site`)

All metrics, 2D contour overlays, difference maps, and time traces can be inspected interactively in the web viewer:

```bash
# Launch the web viewer
.venv/bin/python -m fec_site

# Navigate to
http://127.0.0.1:8770/d3d/labelled
```

The browser displays:
- Side-by-side ground truth vs. predicted 2D flux maps and separatrices.
- Error difference maps ($\hat\psi - \psi$) with zero-centered RdBu colormaps.
- Dynamic time-series scrubbers across all frames of a discharge.
- Complete metric scorecards showing $S$, $R^2_\psi$, $R^2_{\{q, \beta\}}$, $1 - D_{\mathrm{LCFS}}$, and all seven Consistency scalars.
