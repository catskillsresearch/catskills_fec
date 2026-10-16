# FEC interface

Inference is one shot. Align inputs onto that shot’s `efit_times`. Emit the three outputs. Do this for every DIII-D public-test shot and every MAST public-test shot. Current intervals below are corpus envelopes, not clips; missing values are `NaN`.

## Parameters

Size parameters. The input (and raw-output) arrays are functions of these. They are not sensors. \(T\) and \(T_{\mathrm{mag}}\) measured on the public-test parquets (874 DIII-D, 1206 MAST).

| Symbol | Dim | Range | Definition |
|---|---|---|---|
| \(T\) | \(1\) | DIII-D: 275 values in \([3,444]\); MAST: 87 values in \([5,98]\) | \(\lvert\texttt{efit\_times}\rvert\) |
| \(T_{\mathrm{mag}}\) | \(1\) | DIII-D: \(\{49152,50176,480256\}\) (143 / 145 / 586 shots); MAST: \(\{15482,30000\}\) (114 / 1092 shots) | \(\lvert\texttt{magnetics\_time}\rvert\) |
| \(T_{I_p}\) | \(1\) | per-shot | \(\lvert\texttt{magnetics\_plasma\_current}\rvert\) (own ADC on DIII-D) |
| \(T_{\mathrm{th}}\) | \(1\) | per-shot | Thomson time-base length |
| \(n\) | \(1\) | \(\{65\}\) | flux-grid side |
| \(C\) | \(1\) | \(\{19\}\) DIII-D; \(\{812\}\) MAST | coil-geometry rows |
| \(N\) | \(1\) | DIII-D \([59,138]\) | Thomson chords |
| \(N_{\mathrm{ch}}\) | \(1\) | per-shot | Thomson spatial channels |

## Inputs — D3 only

Not present on a MAST row. A model that requires these cannot run on MAST.

| Field | Dim | Range | \(\Psi_{\mathrm{vac}}\) |
|---|---|---|---|
| `magnetics_F{1–9}{A,B}` | \((T_{\mathrm{mag}},)\) | \(\approx[-580,580]\) kA·turn | used |
| `magnetics_ECOILA` | \((T_{\mathrm{mag}},)\) | kA | used |
| `magnetics_bcoil` | \((T_{\mathrm{mag}},)\) | kA | unused; no `coil_*` row |
| `magnetics_plasma_current_times` | \((T_{I_p},)\) | ms; D3 \(I_p\) clock | unused; \(I_p\) is not in the sum |

## Inputs — MAST only

Not present on a D3 row.

| Field | Dim | Range | \(\Psi_{\mathrm{vac}}\) |
|---|---|---|---|
| `magnetics_p{2–6}{l,u}_current` | \((T_{\mathrm{mag}},)\) | \(\approx[-600,600]\) kA·turn | used |
| `magnetics_sol_current` | \((T_{\mathrm{mag}},)\) | kA | used |
| `magnetics_tf_current` | \((T_{\mathrm{mag}},)\) | kA | unused; no `coil_*` row |
| `magnetics_efps_current` | \((T_{\mathrm{mag}},)\) | kA | unused; no `coil_*` row |

## Derived common input \(\Psi_{\mathrm{vac}}\)

Do not call this \(C\). \(C\) is already the coil-geometry row count (\(\{19\}\) D3, \(\{812\}\) MAST), and later \(C=\mathrm{LCFS}(\psi)\).

One object, both machines, one `efit_times` index \(t\):

\[
\Psi_{\mathrm{vac}}(t)\in\mathbb{R}^{n\times n}=\mathbb{R}^{65\times 65}
\]

It is the vacuum flux from that slice’s coil currents. Not a concat of named currents (that would be \(\mathbb{R}^{19}\) on D3 and \(\mathbb{R}^{11}\) on MAST). Implemented by Sky P0 / `rectangle_green`.

**Algorithm** (same steps on D3 and MAST):

1. Read common `efit_grid_R`, `efit_grid_Z` \(\rightarrow\) build the \(65\times 65\) mesh \((R,Z)\).
2. Set \(\Psi_{\mathrm{vac}}\leftarrow 0\in\mathbb{R}^{65\times 65}\).
3. For each geometry row \(k=1,\ldots,C\):
   - \(\texttt{col}\leftarrow\texttt{coil\_input\_column}[k]\)
   - \(I_{\mathrm{coil}}(t)\leftarrow\mathrm{interp}(\texttt{magnetics\_time},\;\texttt{row}[\texttt{col}],\;t)\)
   - \(I^{\mathrm{el}}\leftarrow I_{\mathrm{coil}}(t)/n_{\mathrm{share},k}\)
   - \(\Psi_{\mathrm{vac}}\leftarrow \Psi_{\mathrm{vac}}+I^{\mathrm{el}}\,G(R,Z;R_k,Z_k,w_k,h_k)\)
4. Return \(\Psi_{\mathrm{vac}}\).

\(G\) is the rectangle Green function. Rows with no `coil_*` (`bcoil`, `tf_current`, `efps_current`) are skipped.

### From D3-only currents

\(C=19\). \(\texttt{col}\) is `magnetics_F1A` … `magnetics_F9B` or `magnetics_ECOILA`. \(n_{\mathrm{share}}=1\). `bcoil` is not used.

### From MAST-only currents

\(C=812\). \(\texttt{col}\) is `magnetics_p2l_current` … `magnetics_p6u_current` or `magnetics_sol_current`. \(n_{\mathrm{share}}\) is the turn count that share that column. `tf_current` and `efps_current` are not used.

\(\Psi_{\mathrm{vac}}(t)\) is vacuum \(\psi\), not the submitted \(\psi\), not \(q_{95}\), not \(\beta_N\). \(I_p\) is common and is not mixed into this sum.

The consumer is one operator on both machines:

\[
M\bigl(\Psi_{\mathrm{vac}}(t),\,I_p(t),\,\ldots\bigr)
=\bigl(\psi(t),\,q_{95}(t),\,\beta_N(t)\bigr)
\]

Same \(M\), same three outputs, same zip keys. The machine disappears inside the sum that built \(\Psi_{\mathrm{vac}}\). \(M\) does not branch on F-coil names or on \(C\in\{19,812\}\). The grid extents still differ (D3 \(R\in[0.84,2.54]\), MAST \(R\in[0.06,2.0]\)); those come in as the common `efit_grid_*` that defined the mesh, not as a second current layout.

## Inputs — Common (both machines)

Same field names on a D3 shot and a MAST shot. Lengths and physical ranges still differ (see Parameters). Join currents to geometry with `coil_input_column`; do not assume \(C=19\) or F-coil names. `magnetics_dsep` and every `efit_*` label are withheld on both public tests.

| Field | Dim | Notes |
|---|---|---|
| `efit_times` | \((T,)\) | ms; prediction clock. D3 \(T\in[3,444]\); MAST \(T\in[5,98]\) |
| `efit_grid_R` | \((n,)\) | meters. D3 \([0.84,2.54]\); MAST \([0.06,2.0]\) |
| `efit_grid_Z` | \((n,)\) | meters. D3 \([-1.60,1.60]\); MAST \([-2.0,2.0]\) |
| `magnetics_time` | \((T_{\mathrm{mag}},)\) | ms. D3 \(T_{\mathrm{mag}}\in\{49152,50176,480256\}\); MAST \(\{15482,30000\}\) |
| `magnetics_plasma_current` \(I_p\) | \((T_{I_p},)\) | kA on both machines. The one apples-to-apples current. D3: interpolate with `magnetics_plasma_current_times`. MAST: interpolate with `magnetics_time` |
| `coil_R`, `coil_Z` | \((C,)\) | meters. \(C=19\) on D3; \(C=812\) on MAST |
| `coil_width`, `coil_height` | \((C,)\) | meters |
| `coil_angle1`, `coil_angle2` | \((C,)\) | degrees; \(0\) = rectangle |
| `coil_name`, `coil_input_column` | \((C,)\) | join key to that machine’s current columns |
| `derived_psi` | \((T,n,n)\) | derived common input \(\Psi_{\mathrm{vac}}\). One \(n\times n\) vacuum-flux frame per `efit_times` index. Not in the parquet; built above from that machine’s currents and `coil_*` |
| `thomson_core_*`, `thomson_edge_*` \(T_e\) | \((T_{\mathrm{th}},N_{\mathrm{ch}})\) | eV. D3 core is a vertical chord; MAST core is midplane horizontal |
| `thomson_core_*`, `thomson_edge_*` \(n_e\) | \((T_{\mathrm{th}},N_{\mathrm{ch}})\) | m\(^{-3}\) |
| `thomson_chord_R`, `thomson_chord_Z` | \((N,)\) | meters. D3 \(N\in[59,138]\); MAST \(Z=0\) |

## Raw outputs — D3

File: `diii_d_public_test.npz`. You do not write \(S\), \(G\), or any \(^{\mathrm{gt}}\). DIII-D train has labels; this public-test file does not.

| You write | Symbol | Zip key | Dim | Range |
|---|---|---|---|---|
| Poloidal flux | \(\psi(R,Z)\) | `shot_XXXX_psirz` | \((T,n,n)\) | \(\mathbb{R}\) |
| Safety factor | \(q_{95}\) | `shot_XXXX_q95` | \((T,)\) | \(\mathbb{R}\) |
| Normalized beta | \(\beta_N\) | `shot_XXXX_betaN` | \((T,)\) | \(\mathbb{R}\) |

## Raw outputs — MAST

File: `mast_public_test.npz`. Same three arrays, MAST inputs. You do not write \(S_{\mathrm{MAST}}\) or any \(^{\mathrm{gt}}\). There is no `mast_train`.

| You write | Symbol | Zip key | Dim | Range |
|---|---|---|---|---|
| Poloidal flux | \(\psi(R,Z)\) | `shot_XXXX_psirz` | \((T,n,n)\) | \(\mathbb{R}\) |
| Safety factor | \(q_{95}\) | `shot_XXXX_q95` | \((T,)\) | \(\mathbb{R}\) |
| Normalized beta | \(\beta_N\) | `shot_XXXX_betaN` | \((T,)\) | \(\mathbb{R}\) |

## Per-frame slice

The prediction clock is `efit_times`. Index \(t\in\{0,\ldots,T-1\}\) is one frame. \(T\) is not a constant. Magnetics and Thomson keep their own clocks; they are not sliced by row index. They are interpolated to \(\tau=\texttt{efit\_times}[t]\) (ms). Geometry does not depend on \(t\).

### Common inputs at frame \(t\)

| Field | Shot dim | Frame slice | How |
|---|---|---|---|
| `efit_times` | \((T,)\) | scalar \(\tau\) | \(\texttt{efit\_times}[t]\) |
| `efit_grid_R`, `efit_grid_Z` | \((n,)\) | \((n,)\) | copy; same mesh every frame |
| `magnetics_time` | \((T_{\mathrm{mag}},)\) | — | source clock only; do not index by \(t\) |
| `magnetics_plasma_current` \(I_p\) | \((T_{I_p},)\) | scalar | D3: \(\mathrm{interp}(\texttt{magnetics\_plasma\_current\_times},I_p,\tau)\). MAST: \(\mathrm{interp}(\texttt{magnetics\_time},I_p,\tau)\) |
| `coil_*` | \((C,)\) | \((C,)\) | copy; geometry is shot-constant |
| `derived_psi` \(\Psi_{\mathrm{vac}}\) | \((T,n,n)\) | \((n,n)\) | \(\texttt{derived\_psi}[t]\). Already on the EFIT clock |
| `thomson_*` \(T_e,n_e\) | \((T_{\mathrm{th}},N_{\mathrm{ch}})\) | \((N_{\mathrm{ch}},)\) | \(\mathrm{interp}(\texttt{thomson\_*\_times},\,\cdot\,,\tau)\) |
| `thomson_chord_R`, `thomson_chord_Z` | \((N,)\) | \((N,)\) | copy |

After the slice, \(M\) sees one \(\Psi_{\mathrm{vac}}(t)\in\mathbb{R}^{65\times 65}\), one \(I_p(t)\), the grid, the chords, and Thomson at \(\tau\). It does not see \(T_{\mathrm{mag}}\) or named F/P columns.

### Raw outputs at frame \(t\)

| You write | Shot dim | Frame slice |
|---|---|---|
| \(\psi(R,Z)\) | \((T,n,n)\) | \(\psi[t]\in\mathbb{R}^{n\times n}\) |
| \(q_{95}\) | \((T,)\) | \(q_{95}[t]\in\mathbb{R}\) |
| \(\beta_N\) | \((T,)\) | \(\beta_N[t]\in\mathbb{R}\) |

Write the shot arrays by stacking those \(T\) slices in `efit_times` order. On `diii_d_train` the labels are the same slice of `efit_psirz`, `efit_q95`, `efit_beta_n`. Public test has no labels.

### D3 train pool on CRC_FUSION_LAB

Code: [`catskills/psivac.py`](catskills/psivac.py) slices one row; [`catskills/scripts/psivac_pool.py`](catskills/scripts/psivac_pool.py) writes the flat pool. `features_v1` is the old 129-d magnetics/Thomson cache. It is not \(\Psi_{\mathrm{vac}}\). Its `n_frames` column is only a count of \(\lvert\texttt{efit\_times}\rvert\).

`diii_d_train` is 7041 shots under `$LAB/datasets/Sophelio__fusion-equilibrium-challenge/data/diii_d_train/`. Expected size of the pool, from that count:

| | |
|---|---|
| Shots | \(7041\) |
| Frames | \(1\,559\,340\) |
| \(T\) | 401 values in \([1,445]\); mean \(221.5\); median \(241\); 5–95% \([75,324]\) |
| `derived_psi` float32 | \(1\,559\,340\times 65\times 65\times 4\,\mathrm{B}\approx 24.5\,\mathrm{GiB}\) |
| Same plus `efit_psirz` | \(\approx 49\,\mathrm{GiB}\) |
| Path | `$LAB/cache/psivac_pool_v1/` (`MANIFEST.json`, `derived_psi.f32`, `ip.f32`, `psi_gt.f32`, `q95.f32`, `betaN.f32`) |
| Drive | CRC_FUSION_LAB \(916\,\mathrm{GiB}\), \(\approx 616\,\mathrm{GiB}\) free (2026-08-26) |

Build:

```
poetry run python -m catskills.scripts.psivac_pool
poetry run python -m catskills.scripts.psivac_pool --split mast_public_test
```

Every D3 train frame has both sides: `derived_psi`, \(I_p\), and labels `efit_psirz`, `efit_q95`, `efit_beta_n`. The builder refuses a shot that is missing any of those three.

### MAST public-test pool (inputs only)

There is no `mast_train`. MAST public-test frames are the same input slice (`derived_psi`, \(I_p\), grid) and have **no** output to pair. Path: `$LAB/cache/psivac_pool_v1_mast/`. About 1206 shots, \(T\) typically in \([12,91]\).

## Frame I/O

This is what \(M\) consumes and emits at one `efit_times` index. Not a shot. Not a named current vector.

**Input frame** (both machines):

| Tensor | Dim | |
|---|---|---|
| \(\Psi_{\mathrm{vac}}\) `derived_psi` | \((65,65)\) | vacuum flux from that machine’s coils |
| \(I_p\) | scalar | kA, interpolated to this frame |
| `efit_grid_R`, `efit_grid_Z` | \((65,)\) | meters; shot-constant, needed to read the map |

Thomson at \(\tau\) is optional extra. Named F/P columns are not an input. \(C\in\{19,812\}\) is not an input.

**Output frame** (what \(M\) writes; D3 train also has this as \(R^{\mathrm{gt}}\)):

| Tensor | Dim | |
|---|---|---|
| \(\psi(R,Z)\) | \((65,65)\) | submitted poloidal flux |
| \(q_{95}\) | scalar | |
| \(\beta_N\) | scalar | |

D3 train: every frame is a pair (input, output). MAST public test: input only. Public D3 test: input only.

## Representation test

Official \(S\) is a later filter. First question: can a net **overfit** a small random D3 sample of these pairs? If train \(R^2_\psi\) cannot approach 1, the architecture cannot represent the map. How to spend a 1000-trial walk of the zoo on one fixed sample: [Optimal configuration space traversal](FEC_model_zoo.md#optimal-configuration-space-traversal) in [`FEC_model_zoo.md`](FEC_model_zoo.md).

Zoo in [`catskills/frame_models.py`](catskills/frame_models.py), run by [`catskills/scripts/frame_overfit.py`](catskills/scripts/frame_overfit.py): `linear`, `mlp`, `residual_mlp` (\(\psi=a\Psi_{\mathrm{vac}}+\mathrm{MLP}\)), `conv`, `unet`, `film`. Vary width, `mse` / `huber` / `mse_scalars`, Adam / AdamW.

```
poetry run python -m catskills.scripts.frame_overfit --shots 8
```

## Aggregate Scores

These scores are over a set of shots (the public-test set, or later the private-test set), each run through \(M\), then pooled. They are not a score on one shot. Inference is still one shot; \(S\), \(D_{\mathrm{LCFS}}\), Consistency, and the board columns are the aggregate after every shot in that set has a written triple.

### \(S_0\) and \(S\)

\[
R^2(a,b)=1-\frac{\lVert a-b\rVert_2^2}{\lVert b-\bar b\rVert_2^2}
\]

\[
\begin{aligned}
S_0(\psi,q_{95},\beta_N,\psi^{\mathrm{gt}},q_{95}^{\mathrm{gt}},\beta_N^{\mathrm{gt}})
&=
0.55\max\bigl(0,R^2(\psi,\psi^{\mathrm{gt}})\bigr)
+0.15\max\bigl(0,\tfrac12\bigl(R^2(q_{95},q_{95}^{\mathrm{gt}})+R^2(\beta_N,\beta_N^{\mathrm{gt}})\bigr)\bigr)\\
&\quad+0.10\bigl(1-\min(1,D_{\mathrm{LCFS}}(\psi,\psi^{\mathrm{gt}}))\bigr)
+0.20\,\mathrm{Consistency}(\psi,\psi^{\mathrm{gt}})
\end{aligned}
\]

\[
S(\psi,q_{95},\beta_N,\psi^{\mathrm{gt}},q_{95}^{\mathrm{gt}},\beta_N^{\mathrm{gt}})
=
\begin{cases}
S_0(\psi,q_{95},\beta_N,\psi^{\mathrm{gt}},q_{95}^{\mathrm{gt}},\beta_N^{\mathrm{gt}})
& R^2(\psi,\psi^{\mathrm{gt}})\ge R^2(-\psi,\psi^{\mathrm{gt}})\\[6pt]
S_0(-\psi,q_{95},\beta_N,\psi^{\mathrm{gt}},q_{95}^{\mathrm{gt}},\beta_N^{\mathrm{gt}})
& R^2(-\psi,\psi^{\mathrm{gt}})>R^2(\psi,\psi^{\mathrm{gt}})
\end{cases}
\]

### \(D_{\mathrm{LCFS}}\) and \(\mathrm{Consistency}\)

Both take two flux maps. Neither uses \(q_{95}\) or \(\beta_N\). You do not submit an LCFS or the seven scalars; the scorer reads them off \(\psi\) and \(\psi^{\mathrm{gt}}\).

Per frame, \(\mathrm{LCFS}(\psi)\) is the **last closed flux surface**. Work in \(\phi=s\psi\) with \(s\in\{+1,-1\}\) chosen so the magnetic axis is a **maximum** of \(\phi\).

The magnetic axis is a **point** \((R_a,Z_a)\) in the poloidal plane (the O-point), not a segment and not a pair of endpoints. In 3-D it is a toroidal loop; the scorer only sees the \((R,Z)\) plane. It is the strongest **interior local** maximum of \(\phi\): a grid cell \((i_Z,i_R)\) that is \(\ge\) all eight neighbours, not on the envelope rim (mask eroded by 2 cells), then refined to sub-pixel by a 1-D parabola in \(R\) and in \(Z\). If no interior local max exists, fall back to \(\arg\max\phi\) inside the mask.

A contour at level \(\lambda\) is a connected component of \(\{(R,Z):\phi(R,Z)=\lambda\}\) (marching squares). It is **closed** if the first and last samples coincide. It **encloses** the axis if the even-odd ray-cast of the polygon contains the axis cell \((i_Z,i_R)\): a horizontal ray from that cell crosses the boundary an odd number of times. It must also lie inside the machine envelope.

Let \(\mathcal{V}(\lambda)\) be the set of closed, in-envelope contours at level \(\lambda\) that enclose the axis. A level is valid when \(\mathcal{V}(\lambda)\ne\emptyset\). “Lowest” means the smallest such \(\lambda\) (further down, the surface opens onto the wall or splits at an X-point):

\[
\lambda^\star
=\inf\{\lambda:\mathcal{V}(\lambda)\ne\emptyset\}
\]

The search brackets \(\lambda\in[\min\phi|_{\partial\text{grid}},\,\phi(i_Z,i_R))\) and bisects (22 steps). Then

\[
\mathrm{LCFS}(\psi)=\text{the unique contour in }\mathcal{V}(\lambda^\star)\text{, resampled to 512 arc-length points}
\]

or missing if even the tight loop around the axis is not closed and enclosing.

For two point sets \(A,B\subset\mathbb{R}^2\) (the two LCFS polygons, metres):

\[
h(A,B)=\max_{a\in A}\min_{b\in B}\lVert a-b\rVert_2,\qquad
d_H(A,B)=\max\bigl(h(A,B),\,h(B,A)\bigr)
\]

\(R_{\mathrm{geo}}\) is that shot’s mean major radius from the GT contour. No contour on a frame that has a GT contour \(\rightarrow D=1\). Machine value is the mean of per-shot means.

\[
D_{\mathrm{LCFS}}(\psi,\psi^{\mathrm{gt}})
=\operatorname{mean}_{i,t}\min\bigl(1,\,d_H(\mathrm{LCFS}(\psi_{i,t}),\mathrm{LCFS}(\psi^{\mathrm{gt}}_{i,t}))/R_{\mathrm{geo},i}\bigr)
\]

Let \(C=\mathrm{LCFS}(\psi)\) with points \((R_k,Z_k)\), \(R_{\max}=\max R_k\), \(R_{\min}=\min R_k\), \(Z_{\max}=\max Z_k\), \(Z_{\min}=\min Z_k\), \(a=(R_{\max}-R_{\min})/2\), \(R_{\mathrm{geo}}=(R_{\max}+R_{\min})/2\). \(B_p=\lvert\nabla\psi\rvert/R\). \(\Omega\) is the interior of \(C\). \(\ell_p\) is the poloidal perimeter of \(C\). \(\oint_C B_p\,\mathrm{d}\ell\) is Ampere’s \(\mu_0 I_p\).

\[
\begin{aligned}
f_{R_{\mathrm{axis}}}(\psi)&=R\text{ of the O-point of }\pm\psi\\
f_{Z_{\mathrm{axis}}}(\psi)&=Z\text{ of the O-point of }\pm\psi\\
f_{\kappa}(\psi)&=(Z_{\max}-Z_{\min})/(R_{\max}-R_{\min})\\
f_{\delta_{\mathrm{top}}}(\psi)&=(R_{\mathrm{geo}}-R|_{Z_{\max}})/a\\
f_{\delta_{\mathrm{bot}}}(\psi)&=(R_{\mathrm{geo}}-R|_{Z_{\min}})/a\\
f_{V}(\psi)&=2\pi\bigl\lvert\tfrac16\sum_k(R_k+R_{k+1})(R_k Z_{k+1}-R_{k+1}Z_k)\bigr\rvert\\
f_{\ell_i}(\psi)&=\langle B_p^2\rangle_{\Omega}\,/\,\bigl(\tfrac{1}{\ell_p}\oint_C B_p\,\mathrm{d}\ell\bigr)^2
\end{aligned}
\]

\(\mathrm{Consistency}\) applies the same \(f_j\) to both maps, then pooled \(R^2\), then a clipped mean:

\[
\mathrm{Consistency}(\psi,\psi^{\mathrm{gt}})
=\operatorname{mean}_j\max\bigl(0,\,R^2(f_j(\psi),f_j(\psi^{\mathrm{gt}}))\bigr)
\]

\(j\) runs over \(\{R_{\mathrm{axis}},Z_{\mathrm{axis}},\kappa,\delta_{\mathrm{top}},\delta_{\mathrm{bot}},V,\ell_i\}\). Failed or implausible extractions are replaced by the mean over that scored set.

### Derived outputs — D3

The scorer applies \(S\) to the D3 file and hidden DIII-D ground truth. You do not write these.

| Board column | Symbol | Dim | Range | Value |
|---|---|---|---|---|
| `d3d_S` | \(S\) | \(1\) | \([0,1]\) | \(S(\psi,q_{95},\beta_N,\psi^{\mathrm{gt}},q_{95}^{\mathrm{gt}},\beta_N^{\mathrm{gt}})\) on `diii_d_public_test.npz` |
| `d3d_r2_psi` | \(R^2_\psi\) | \(1\) | \((-\infty,1]\) | \(R^2(\psi,\psi^{\mathrm{gt}})\) as used inside that \(S\) |
| `d3d_r2_qb` | \(R^2_{\{q_{95},\beta_N\}}\) | \(1\) | \((-\infty,1]\) | \(\tfrac12\bigl(R^2(q_{95},q_{95}^{\mathrm{gt}})+R^2(\beta_N,\beta_N^{\mathrm{gt}})\bigr)\) |
| `d3d_dlcfs` | \(D_{\mathrm{LCFS}}\) | \(1\) | \([0,1]\) | \(D_{\mathrm{LCFS}}(\psi,\psi^{\mathrm{gt}})\) |
| `d3d_consistency` | \(\mathrm{Consistency}\) | \(1\) | \([0,1]\) | \(\mathrm{Consistency}(\psi,\psi^{\mathrm{gt}})\) |
| `g_ratio` | \(G\) | \(1\) | \([0,+\infty)\) | `mast_S` \(/\) `d3d_S` if `d3d_S` \(>0\), else \(0\) |

### Derived outputs — MAST

The scorer applies \(S\) to the MAST file and hidden MAST ground truth. You do not write these.

| Board column | Symbol | Dim | Range | Value |
|---|---|---|---|---|
| `mast_S` | \(S_{\mathrm{MAST}}\) | \(1\) | \([0,1]\) | \(S(\psi,q_{95},\beta_N,\psi^{\mathrm{gt}},q_{95}^{\mathrm{gt}},\beta_N^{\mathrm{gt}})\) on `mast_public_test.npz` |
| `mast_r2_psi` | \(R^2_\psi\) | \(1\) | \((-\infty,1]\) | \(R^2(\psi,\psi^{\mathrm{gt}})\) as used inside that \(S\) |
| `mast_r2_qb` | \(R^2_{\{q_{95},\beta_N\}}\) | \(1\) | \((-\infty,1]\) | \(\tfrac12\bigl(R^2(q_{95},q_{95}^{\mathrm{gt}})+R^2(\beta_N,\beta_N^{\mathrm{gt}})\bigr)\) |
| `mast_dlcfs` | \(D_{\mathrm{LCFS}}\) | \(1\) | \([0,1]\) | \(D_{\mathrm{LCFS}}(\psi,\psi^{\mathrm{gt}})\) |
| `mast_consistency` | \(\mathrm{Consistency}\) | \(1\) | \([0,1]\) | \(\mathrm{Consistency}(\psi,\psi^{\mathrm{gt}})\) |
| `g_ratio` | \(G\) | \(1\) | \([0,+\infty)\) | `mast_S` \(/\) `d3d_S` if `d3d_S` \(>0\), else \(0\) |

### Current leaderboard

Public board [19740](https://www.codabench.org/api/leaderboards/19740/), 2026-08-26. Ranked by `d3d_S`. You do not compute these; Codabench does.

**D3**

| Board column | dmitryzhevn | pshishkin | mwaller | dgaytanv | mnn31 |
|---|---:|---:|---:|---:|---:|
| `d3d_S` | 0.994376 | 0.994081 | 0.993620 | 0.993203 | 0.992076 |
| `d3d_r2_psi` | 0.999789 | 0.999832 | 0.999742 | 0.999864 | 0.999765 |
| `d3d_r2_qb` | 0.996797 | 0.996539 | 0.996104 | 0.995456 | 0.991979 |
| `d3d_dlcfs` | 0.010309 | 0.011227 | 0.011168 | 0.012566 | 0.013401 |
| `d3d_consistency` | 0.980018 | 0.979077 | 0.977314 | 0.976079 | 0.973744 |
| `g_ratio` | 0.747531 | 0.571475 | 0.614716 | 0.000000 | 0.564182 |

**MAST**

| Board column | dmitryzhevn | pshishkin | mwaller | dgaytanv | mnn31 |
|---|---:|---:|---:|---:|---:|
| `mast_S` | 0.743327 | 0.568092 | 0.610794 | 0.000000 | 0.559711 |
| `mast_r2_psi` | 0.940590 | 0.875286 | 0.936987 | 0.000000 | 0.848303 |
| `mast_r2_qb` | 0.527987 | 0.184925 | 0.086423 | 0.000000 | 0.144451 |
| `mast_dlcfs` | 0.163970 | 0.429622 | 0.308633 | 0.000000 | 0.285230 |
| `mast_consistency` | 0.316007 | 0.009540 | 0.066758 | 0.000000 | 0.000000 |
| `g_ratio` | 0.747531 | 0.571475 | 0.614716 | 0.000000 | 0.564182 |
