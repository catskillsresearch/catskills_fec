# FEC model zoo

Fixed mesh. No later refinement. One frame in, one frame out. I/O is [`FEC_interface.md`](FEC_interface.md). This file is only the flatten family: linear, MLP, residual. A model selection is one tuple

\[
(\mathrm{NN},\;\mathrm{sizing},\;\mathrm{trainer},\;\mathrm{loss},\;\mathrm{optimizer},\;\mathrm{stop})
\]

`stop` is required. `epochs` is a cap, not a promise to run that long.

## Frame tensors

\(n=65\). \(P=n^2=4225\).

| | Symbol | Dim | Pool |
|---|---|---|---|
| Input map | \(\Psi_{\mathrm{vac}}\) | \((n,n)\) | D3 `derived_psi`; MAST same shape, no label |
| Input current | \(I_p\) | scalar | kA |
| Output map | \(\psi\) | \((n,n)\) | D3 `efit_psirz` only |
| Output scalars | \(q_{95},\;\beta_N\) | scalar, scalar | D3 only |

Flattened write (this zoo):

\[
x=\bigl[\mathrm{vec}(\Psi_{\mathrm{vac}}),\;I_p\bigr]\in\mathbb{R}^{P+1}=\mathbb{R}^{4226}
\]

\[
\hat y=\bigl[\mathrm{vec}(\hat\psi),\;\hat q_{95},\;\hat\beta_N\bigr]\in\mathbb{R}^{P+2}=\mathbb{R}^{4227}
\]

**Normalize by channel, not one global 4227-mean.** \(\Psi_{\mathrm{vac}}\), \(I_p\), \(\psi\), \(q_{95}\), \(\beta_N\) are different quantities. Fit five affines on the **train shots only**. Apply the same five at val and at MAST infer. The net and the generic losses see the normalized concat. Invert \(\psi,q_{95},\beta_N\) before you write an npz.

D3 pool: \(N_{\mathrm{pool}}=1\,559\,340\) labeled frames, 7041 shots, `$LAB/cache/psivac_pool_v1/`. MAST: 69 075 input-only frames. Do not put MAST in any loss.

---

## Neural networks

All three consume \(x\in\mathbb{R}^{4226}\) and emit \(\hat y\in\mathbb{R}^{4227}\). Activation \(a\) is a sizing choice (`GELU`, `ReLU`, `SiLU`). Bias on every linear layer unless noted.

### Linear

\[
\hat y = W x + b,\qquad W\in\mathbb{R}^{4227\times 4226},\; b\in\mathbb{R}^{4227}
\]

No hidden width. Capacity is the matrix.

| Hyperparameter | Symbol | Grid |
|---|---|---|
| Bias | `bias` | `{true, false}` |
| Input dropout | `p_in` | `{0, 0.05, 0.1}` |

Parameters \(\approx 4227\times 4226 \approx 1.79\times 10^7\) (with bias).

### MLP

Hidden width \(w\), depth \(d\) (number of hidden layers). Last layer is linear onto \(4227\).

\[
h_0=x,\quad
h_{\ell+1}=a(W_\ell h_\ell+b_\ell)\ \ (\ell=0,\ldots,d-1),\quad
\hat y=W_d h_d+b_d
\]

\(W_0\in\mathbb{R}^{w\times 4226}\), \(W_\ell\in\mathbb{R}^{w\times w}\) for hidden-to-hidden, \(W_d\in\mathbb{R}^{4227\times w}\).

| Hyperparameter | Symbol | Grid |
|---|---|---|
| Width | \(w\) | `{64, 128, 256, 512, 1024}` |
| Depth | \(d\) | `{1, 2, 3, 4, 6}` |
| Activation | \(a\) | `{GELU, ReLU, SiLU}` |
| Hidden dropout | `p` | `{0, 0.1}` |
| Layer-norm | `ln` | `{off, on}` (after each hidden) |

Parameter count (bias on, no extra heads):

\[
(4226\,w + w) + (d-1)(w^2+w) + (4227\,w + 4227)
\approx w\,(8453 + (d-1)w)
\]

| \(w\) | \(d=2\) | \(d=4\) |
|---|---:|---:|
| 128 | \(1.1\times 10^6\) | \(1.1\times 10^6\) |
| 256 | \(2.2\times 10^6\) | \(2.4\times 10^6\) |
| 512 | \(4.6\times 10^6\) | \(5.1\times 10^6\) |
| 1024 | \(9.7\times 10^6\) | \(1.2\times 10^7\) |

The \(P{\to}w\) and \(w{\to}P\) maps dominate until \(w\) is large.

Optional head split (`heads=split`): last hidden \(h_d\) feeds \(W_\psi\in\mathbb{R}^{P\times w}\) and \(W_s\in\mathbb{R}^{2\times w}\) instead of one \(4227\) layer. Same \(w,d\); two readout matrices.

### Residual

Same MLP body as above. The map head is a correction to vacuum:

\[
\hat\psi = \alpha\,\Psi_{\mathrm{vac}} + \mathrm{reshape}(r),\qquad
(r,\;\hat q_{95},\;\hat\beta_N)=\mathrm{MLP}(x)
\]

\(\alpha\in\mathbb{R}\) is a learnable scalar (one number). Vacuum and \(\psi\) stay in normalized units so \(\alpha\) can start at \(0\) or \(1\).

| Hyperparameter | Symbol | Grid |
|---|---|---|
| Width, depth, \(a\), dropout, LN | as MLP | same grids |
| Scale init | \(\alpha_0\) | `{0, 1}` |
| Scale frozen | `alpha_freeze` | `{false, true}` |

One extra parameter over the matching MLP.

---

## Losses

The net always emits one vector of length \(P+2=4227\):

\[
\hat y=\bigl[\mathrm{vec}(\hat\psi),\;\hat q_{95},\;\hat\beta_N\bigr],\qquad
y=\bigl[\mathrm{vec}(\psi^{\mathrm{gt}}),\;q_{95}^{\mathrm{gt}},\;\beta_N^{\mathrm{gt}}\bigr]
\in\mathbb{R}^{4227}
\]

Residual concatenates \(\mathrm{vec}(\hat\psi)\) with the two scalars before the loss. Official \(S\) is not a train loss. Pick **one** named loss per selection.

### Torch (standard)

One call on the full pair. `ŷ.shape == y.shape == (B, 4227)`. Reduction is `mean` over the \(4227\) entries, then the batch.

`F.mse_loss` has no per-slot norm argument. You do not pass a list of scales into the loss. You **fit five channel stats once**, apply them so \(\hat y\) and \(y\) are already dimensionless, then call the stock function.

**Specify it on the trainer, not on the loss ID:**

```
norm=zscore  channels=vac,ip,psi,q95,betaN
```

or `norm=rms` with the same `channels` list. Required whenever the loss is `mse` / `l1` / `smoothl1` / `huber`.

| Channel | Lives in | Slice | How the two stats are computed (train shots only) |
|---|---|---|---|
| `vac` | \(x\) | \(x_{1:P}\) | mean/std (or RMS) of all `derived_psi` pixels |
| `ip` | \(x\) | \(x_{P+1}\) | mean/std (or RMS) of \(I_p\) |
| `psi` | \(y\) | \(y_{1:P}\) | mean/std (or RMS) of all `efit_psirz` pixels |
| `q95` | \(y\) | \(y_{P+1}\) | mean/std (or RMS) of \(q_{95}\) |
| `betaN` | \(y\) | \(y_{P+2}\) | mean/std (or RMS) of \(\beta_N\) |

One mean and one scale per channel. Not one mean over \(4227\). Not a per-pixel map of 4225 means.

**Program it as a small affine, not as a custom `nn.Module` loss:**

1. After the shot hold-out, scan train frames and write `norm.json` (or `.npz`): `{vac: {mean, std}, ip: {…}, psi: {…}, q95: {…}, betaN: {…}}`. `std = max(std, 1e-8)`.
2. In the loader, build raw \(x,y\), then

```python
x_n[:, :P]  = (x[:, :P]  - st["vac"]["mean"])  / st["vac"]["std"]
x_n[:, P]   = (x[:, P]   - st["ip"]["mean"])   / st["ip"]["std"]
y_n[:, :P]  = (y[:, :P]  - st["psi"]["mean"])  / st["psi"]["std"]
y_n[:, P]   = (y[:, P]   - st["q95"]["mean"])  / st["q95"]["std"]
y_n[:, P+1] = (y[:, P+1] - st["betaN"]["mean"]) / st["betaN"]["std"]
```

3. The net is \(M(x_n)=\hat y_n\). The loss is exactly `F.mse_loss(yhat_n, y_n)` (or `l1` / `smoothl1` / `huber`). No extra weights.
4. At infer, apply the **same** `vac`/`ip` stats to MAST \(x\). Invert only the three output channels before the npz:

```python
psi  = yhat_n[:, :P]  * st["psi"]["std"]  + st["psi"]["mean"]
q95  = yhat_n[:, P]   * st["q95"]["std"]  + st["q95"]["mean"]
beta = yhat_n[:, P+1] * st["betaN"]["std"] + st["betaN"]["mean"]
```

`rms` is the same code with `mean=0` and `std=sqrt(mean(channel**2))`.

Ship `norm.json` next to the checkpoint. A run that says `loss=mse` without those five stats is incomplete.

| ID | Call on already-normalized \((\hat y_n, y_n)\) |
|---|---|
| `mse` | `F.mse_loss` |
| `l1` | `F.l1_loss` |
| `smoothl1` | `F.smooth_l1_loss` |
| `huber` | `F.huber_loss` (`delta=1`) |

### Bespoke

These may slice \(\hat y\) and \(y\) into the three pieces the board already scores separately:

\[
\hat\psi=\mathrm{reshape}(\hat y_{1:P}),\qquad
\hat q_{95}=\hat y_{P+1},\qquad
\hat\beta_N=\hat y_{P+2}
\]

That helps. \(S_0\) is three \(R^2\) terms, not one \(R^2\) on a 4227-pile. A single `inv_r2(\hat y,y)` is almost \(R^2_\psi\); the two scalars vanish in the sum of squares.

\[
L_{R^2}(a,b)=1-R^2(a,b)=\frac{\lVert a-b\rVert_2^2}{\lVert b-\bar b\rVert_2^2+\varepsilon}
\]

**`s_proxy`** — Torch MSE on each piece, weights from the 0.55 / 0.15 part of \(S_0\) (0.15 is the mean of the two scalar \(R^2\)):

\[
L = 0.55\,\mathrm{MSE}(\hat\psi,\psi^{\mathrm{gt}})
+ 0.075\,\mathrm{MSE}(\hat q_{95},q_{95}^{\mathrm{gt}})
+ 0.075\,\mathrm{MSE}(\hat\beta_N,\beta_N^{\mathrm{gt}})
\]

**`inv_r2`** — same split, scale-free:

\[
L = 0.55\,L_{R^2}(\hat\psi,\psi^{\mathrm{gt}})
+ 0.075\,L_{R^2}(\hat q_{95},q_{95}^{\mathrm{gt}})
+ 0.075\,L_{R^2}(\hat\beta_N,\beta_N^{\mathrm{gt}})
\]

**`signed_mse`** — `s_proxy` on \(\hat y\) and on the vector with the first \(P\) entries flipped; keep the better of the two. Scalars stay put. The scorer’s sign case. Use after a model already fits the map; do not start here.

Do not backprop \(D_{\mathrm{LCFS}}\) or Consistency in this zoo. Read them on a held-out shot set after train, as a filter, the way \(S\) does.

---

## Trainer

Assume the D3 pool of \(1\,559\,340\) frames. Shuffle **frames** for SGD. Hold out **shots**, not random pixels from the same shot.

| Knob | Grid | Meaning |
|---|---|---|
| `val_shots` | `{80, 200}` | whole shots left out of the shuffle; never in the loss |
| `train_shots` | rest of 7041 | \(\approx 1.54\times 10^6\) frames when `val_shots=80` |
| `batch` | `{64, 128, 256, 512}` | frames per step |
| `epochs` | `{1, 3, 8, 20}` | **cap** on passes over the shuffled train frames. The stopper may exit earlier |
| `amp` | `{off, fp16}` | CUDA only |
| `norm` | `{zscore, rms}` | **required** for generic losses. Five channel affines on train shots: \(\Psi_{\mathrm{vac}}\), \(I_p\), \(\psi\), \(q_{95}\), \(\beta_N\). Not one mean over \(4227\). |

Shuffle: each epoch, permute the train-frame index and walk it in `batch` chunks. Last incomplete batch drops or pads; drop is fine.

Steps per epoch at `batch=256`: \(1\,540\,000/256\approx 6000\). Eight epochs is \(\sim 5\times 10^4\) steps.

Do not shuffle MAST into the loader. Infer MAST only if the **after** stopper says keep.

---

## Stopping

The grid is \(10^8\) rows. You do not train them to the epoch cap. One named policy `stop=r2_psi` (not a search axis). Three gates, same metric: \(R^2_\psi\) on inverted \(\psi\) (physical units), plus the two scalar \(R^2\) only as a note.

### Before (do not start the 1.5M run)

Overfit probe on a **fixed** 4-shot sample (seed 0), `steps=400`, full batch, no val. Same NN, sizing, loss, optimizer as the candidate. No channel-norm fit on the full train set yet: fit the five affines on those 4 shots only.

| Kill if | Meaning |
|---|---|
| train \(R^2_\psi < 0.90\) | this write cannot represent the map |
| loss is `signed_mse` and no sibling (`mse` or `s_proxy`) has passed this gate | sign-case is a polish, not a first try |

A kill here discards that **NN + sizing** for every trainer/optimizer combo. That is the cheap cut.

### During (leave the epoch cap)

After each epoch, score the **val shots** (the `val_shots` hold-out). Keep the checkpoint with the best val \(R^2_\psi\).

| Knob | Default | Meaning |
|---|---|---|
| `patience` | `2` | epochs with no val \(R^2_\psi\) rise of at least `delta` |
| `delta` | `1e-4` | minimum rise to count as improvement |
| `min_epochs` | `1` | do not stop before this |
| `train_floor` | `0.50` | if train \(R^2_\psi < 0.50\) after `min_epochs`, abort (cannot fit) |

Exit when `patience` is exhausted or `train_floor` fails. Reload the best checkpoint. `epochs` is only the maximum.

### After (do not spend more)

On the best checkpoint:

| Kill if | Meaning |
|---|---|
| val \(R^2_\psi <\) `keep` (`0.98`) | not a contest map; do not infer MAST; do not add epochs |
| val \(R^2_\psi\) below the running best by more than `delta` | do not spawn neighbors (wider, deeper, other `lr`) |

If it **keeps**, you may infer MAST and write the public D3 npz. Official \(S\) on the val shots is a later filter, not this stopper.

`stop=r2_psi` with the defaults above is **one** policy. It does not multiply \(\lvert\mathcal{C}\rvert\).

---

## Optimizers

One Torch optimizer per selection. Learning rate and weight decay are part of the optimizer slot (they travel with the name).

| ID | Class | `lr` grid | Extra |
|---|---|---|---|
| `sgd` | `torch.optim.SGD` | `{1e-2, 1e-1}` | `momentum=0` |
| `sgd_m` | `torch.optim.SGD` | `{1e-2, 3e-2, 1e-1}` | `momentum=0.9` |
| `adam` | `torch.optim.Adam` | `{1e-4, 3e-4, 1e-3}` | \(\beta=(0.9,0.999)\) |
| `adamw` | `torch.optim.AdamW` | `{1e-4, 3e-4, 1e-3}` | `weight_decay` `{0, 1e-4, 1e-2}` |
| `rmsprop` | `torch.optim.RMSprop` | `{1e-4, 1e-3}` | `alpha=0.99` |

Optional schedule, attached to the trainer, not a fifth axis unless you name it: `cosine` over the planned steps, or `none`. Default `none` until AdamW is stable.

---

## Configuration

A configuration is one legal tuple \((\mathrm{NN},\;\mathrm{sizing},\;\mathrm{trainer},\;\mathrm{loss},\;\mathrm{optimizer},\;\mathrm{stop})\). Sizing is **not** a single cartesian product: linear does not have \(w,d\); residual adds \(\alpha\). Count sizing **given** the NN, then multiply by the shared axes.

`channels=vac,ip,psi,q95,betaN` is fixed (not a choice). `train_shots` is determined by `val_shots`. The val-shot **seed** is fixed when comparing runs. `stop=r2_psi` (probe bar \(0.90\), `patience=2`, `keep=0.98`) is fixed (count \(1\)). It does not multiply the space. `epochs` remains a cap in the trainer product.

### NN + sizing

| NN | Choices | Count |
|---|---|---:|
| `linear` | `bias` 2 × `p_in` 3 | \(6\) |
| `mlp` | \(w\) 5 × \(d\) 5 × \(a\) 3 × `p` 2 × `ln` 2 × `heads` 2 | \(600\) |
| `residual` | MLP’s \(600\) × \(\alpha_0\) 2 × `alpha_freeze` 2 | \(2\,400\) |
| **NN + sizing** | union | \(\mathbf{3\,006}\) |

### Loss

| Item | Grid | Count |
|---|---|---:|
| `loss` | `mse`, `l1`, `smoothl1`, `huber`, `s_proxy`, `inv_r2`, `signed_mse` | \(7\) |

### Trainer

| Item | Grid | Count |
|---|---|---:|
| `val_shots` | `{80, 200}` | \(2\) |
| `batch` | `{64, 128, 256, 512}` | \(4\) |
| `epochs` | `{1, 3, 8, 20}` | \(4\) |
| `amp` | `{off, fp16}` | \(2\) |
| `norm` | `{zscore, rms}` | \(2\) |
| `schedule` | `{none, cosine}` | \(2\) |
| **Trainer** | product | \(\mathbf{256}\) |

### Optimizer

`lr` and `weight_decay` travel with the optimizer name. Different names have different `lr` lists, so this is a **sum**, not `5 × |lr|`.

| ID | `lr` | Extra | Count |
|---|---|---|---:|
| `sgd` | 2 | — | \(2\) |
| `sgd_m` | 3 | momentum fixed | \(3\) |
| `adam` | 3 | — | \(3\) |
| `adamw` | 3 | `weight_decay` 3 | \(9\) |
| `rmsprop` | 2 | — | \(2\) |
| **Optimizer** | | | \(\mathbf{19}\) |

### Size of the space

\[
\begin{aligned}
\lvert\mathcal{C}\rvert
&= \lvert\mathrm{NN{+}sizing}\rvert
\times \lvert\mathrm{loss}\rvert
\times \lvert\mathrm{trainer}\rvert
\times \lvert\mathrm{optimizer}\rvert \\
&= 3006 \times 7 \times 256 \times 19 \\
&= 102\,348\,288
\end{aligned}
\]

About \(1.02\times 10^8\) configurations. That is the written grid, not a recommended search. Do not enumerate it. Walk it with [Optimal configuration space traversal](#optimal-configuration-space-traversal): **broad on NN, then refine** (every body once under a common-sense freeze, then 774 trials on the other slots), then promote a handful. The **before** stopper throws away NN+sizing that cannot overfit four shots. The **during** stopper leaves the epoch cap. The **after** stopper refuses MAST infer and neighbor search.

If you freeze the usual defaults (`heads=joint`, `schedule=none`, `norm=zscore`), the space drops to

\[
(6+300+1200)\times 7\times 128\times 19 = 25\,638\,144
\]

(\(1.51\times 10^3\) nets × \(7\) losses × \(128\) trainers × \(19\) optimizers).

---

## Optimal configuration space traversal

\(\lvert\mathcal{C}\rvert\approx 10^8\). One GPU. The question is not “train them all.” It is: **broad on NN, then refine.** First pass: every neural-network body once, other slots frozen at one common-sense tuple. Later passes: only those other slots, and only on bodies that already hit `bar`. Each trial overfits the **same** tiny labeled sample until `stop=r2_psi` fires, records the row, and moves on. Promote a handful to the 1.5M trainer.

This section is that walk. It does not multiply \(\lvert\mathcal{C}\rvert\). It does not change `stop`.

### What a trial is

1. Draw the next tuple from the current pass (Phase A exhausts NN bodies; Phase B refines the rest).
2. Train on the **same** probe frames \(\mathcal{S}\) until the **before** stopper exits (or the step cap).
3. Write one row: the tuple string, train \(R^2_\psi\), the two scalar \(R^2\), steps to \(0.90\), steps to best, wall time, kill/keep.
4. Move to the next tuple. Do not resample \(\mathcal{S}\). Do not infer MAST. Do not write a public npz.

Phase A is not a sample of nets. It is every body, once. Phase B is the budgeted refine (\(B_{\mathrm{refine}}=774\) if the total stay is 1000). Rank the whole log when promoting; do not crown Phase A alone.

### Fix the probe first

Draw \(\mathcal{S}\) once, seed 0, and freeze it for the whole tournament. A new sample per tuple makes the 1000 incomparable.

| Probe | Size | Use |
|---|---|---|
| \(\mathcal{S}_{100}\) | 100 frames from **100 distinct shots** | the 1000-trial ranking |
| \(\mathcal{S}_{4\mathrm{s}}\) | the existing 4-shot set (\(\sim 900\) frames, seed 0) | promotion only; this is the **before** gate already in Stopping |

**Use 100 frames, not 10, for the tournament.** Ten frames is \(\sim 4\times 10^4\) output numbers. Every net that takes a step will hit \(R^2_\psi\approx 1\). The board then ranks optimizer luck, not the map. One hundred frames, one shot each, is still a memorize-able set (linear has \(1.8\times 10^7\) parameters), but step-to-threshold and scalar \(R^2\) start to separate tuples. Do not put two frames from the same shot in \(\mathcal{S}_{100}\): that leaks the shot into “diversity.”

Fit the five channel affines on \(\mathcal{S}\) only (`norm=zscore` unless the tuple says `rms`). Full-batch: \(B_{\mathrm{probe}}=\lvert\mathcal{S}\rvert\). No val split on \(\mathcal{S}\). `epochs` in the tuple is ignored here; the step cap is `400`.

### Stop on the probe

Same metric as `stop=r2_psi`, train-only (there is no val shot in \(\mathcal{S}_{100}\)):

| Knob | Value | Meaning |
|---|---|---|
| `steps` | `400` | cap |
| `delta` | `1e-4` | rise in train \(R^2_\psi\) that resets patience |
| `patience` | `40` steps | no rise → exit and keep the best step |
| `bar` | `0.90` | never reached → **kill** the tuple (and, if it is the first loss for that NN+sizing, kill that body for later waves) |

`signed_mse` is still blocked until a sibling `mse` or `s_proxy` on the same NN+sizing has passed `bar`.

### Rank a finished trial

On ten frames, final \(R^2_\psi\) is useless (everything is \(1\)). On 100 frames it is weak. Rank in this order:

1. Did it hit `bar`? Kill goes to the bottom (recorded, not retried).
2. **Steps to \(R^2_\psi\ge 0.90\)** (smaller is better). This is the useful number.
3. Train \(R^2_\psi\) at the kept step.
4. Mean of the two scalar \(R^2\) at that step.

Official \(S\) is not a column. \(D_{\mathrm{LCFS}}\) is not a column.

### Broad, then refine

Do not sample \(\mathcal{C}\) uniformly. Uniform is a residual search (residual is \(2400/3006\) of NN+sizing) and it mixes untried nets with untried losses. First see **every net**. Then refine the other slots on the nets that work.

A **body** is the architecture, not the regularizer:

| NN | What is enumerated in Phase A | Count |
|---|---|---:|
| `linear` | the map \(Wx+b\) (`bias=true`, `p_in=0`) | \(1\) |
| `mlp` | every \((w,d,a)\) | \(5\times 5\times 3=75\) |
| `residual` | every \((w,d,a,\alpha_0)\) | \(75\times 2=150\) |
| **Bodies** | | \(\mathbf{226}\) |

That is “all the neural networks” for the first pass. Dropout, layer-norm, split heads, `alpha_freeze`, and linear `p_in` / `bias=false` are **not** a different net. They wait for refine. Running the raw \(3006\) NN+sizing rows first would spend most of the day on residual×dropout×LN before you knew whether \(w=128\) or \(w=1024\) can represent \(\psi\).

### Phase A — broad on NN (226 trials, required)

Every body once. **No sampling. No Sobol. No skip.** Common-sense freeze on everything that is not the body:

```
loss=mse
adamw  lr=3e-4  wd=1e-4
batch=256  amp=fp16  norm=zscore  schedule=none  val_shots=80  epochs=8
heads=joint  p=0  p_in=0  ln=off  bias=true  alpha_freeze=false
stop=r2_psi
```

`val_shots` and `epochs` do not affect the probe. They are written so a promoted run is a complete tuple.

Walk order (so a crash still leaves a usable prefix): `linear`, then `mlp` by increasing \(w\), then \(d\), then \(a\) (`GELU`, `ReLU`, `SiLU`), then `residual` in the same \((w,d,a)\) order with \(\alpha_0=1\) before \(\alpha_0=0\).

A body that never hits `bar` is **dead**. Phase B does not revive it with a new loss or optimizer. A kill here is the cheap cut the **before** stopper was for.

Phase A answers one question: which of the 226 nets can represent the map under a sane trainer? It does not answer which loss or `lr` is best. Do not start Phase B until all 226 rows are in the log.

### Phase B — refine the other options (budget 774)

Only bodies that hit `bar`. Thaw **one factor family at a time**, best bodies first. If a factor level never beats the Phase A default, drop it from later waves.

| Wave | Budget | What varies | Who |
|---|---:|---|---|
| B1 loss | 200 | `mse`, `l1`, `smoothl1`, `huber`, `s_proxy`, `inv_r2` | top bodies from Phase A |
| B2 opt | 200 | the 19 optimizer rows | top (body, loss) |
| B3 trainer | 200 | `batch`, `amp`, `norm`, `schedule` | top (body, loss, opt) |
| B4 leftover sizing | 174 | `p`, `ln`, `heads`, `alpha_freeze`, linear `p_in` / `bias`; then one-knob neighbors (next \(w\), \(d\), `lr`) | top 20 of the log |

`signed_mse` only in B1, and only if that body already passed `bar` under `mse` or `s_proxy`.

Inside B1–B3, draw a **Sobol** sequence on the thawed axes among the allowed bodies, not iid. Phase A already spent the “try every net” budget. Phase B does not re-enumerate 226 nets.

### Common-sense freeze (the Phase A tuple)

This is the “other configuration options” selection. Change it only if Phase A itself cannot run (OOM, `lr` explodes). If you change it, restart Phase A. A half-finished A under two defaults is two incomparable lists.

```
mse | adamw lr=3e-4 wd=1e-4 | batch=256 amp=fp16 norm=zscore schedule=none
```

### Promote, do not crown

Best-of-1000 on \(\mathcal{S}_{100}\) is a cheap ranking, not a champion. Memorization on 100 frames is not a D3 map.

| Stage | Who | What |
|---|---|---|
| Phase A | all 226 bodies | \(\mathcal{S}_{100}\), common-sense freeze, 400 steps |
| Phase B | 774 refine trials | only bodies that hit `bar` |
| Before | top 16 of the full log that hit `bar` | existing 4-shot gate \(\mathcal{S}_{4\mathrm{s}}\). Fail \(R^2_\psi<0.90\) → discard that body |
| During | survivors, at most 8 | 1.5M trainer, shot hold-out, patience 2, epoch cap |
| After | keepers with val \(R^2_\psi\ge 0.98\) | MAST infer, public D3 npz. Official \(S\) on val shots is a later filter |

One GPU: finish Phase A before Phase B. Finish the probe log before any 1.5M run. The 226+774 overfits are the day’s work. The eight promoted trains are the night.

### Suggested first cut

```
S = 100 frames / 100 shots / seed 0
Phase A = 226 bodies, exhaustive, common-sense freeze
Phase B = 774, loss → opt → trainer → leftover sizing
stop = r2_psi (bar 0.90, 400 steps, patience 40)
promote = 16 → S_4s → ≤8 full trains
```

Do not start a second tournament on a new \(\mathcal{S}\) until the first eight full trains have a val \(R^2_\psi\). Changing the probe is how you fool yourself into a new “best of 1000.”

---

## Selection

A row in the zoo is one combination. Example:

```
mlp | w=256 d=3 GELU p=0 ln=off heads=joint | val_shots=80 batch=256 epochs=8 amp=fp16 norm=zscore | mse | adamw lr=3e-4 wd=1e-4 | stop=r2_psi
```

Write that string (or a JSON of the five fields) on every run. Compare on the **same** `val_shots` seed. Rank on val \(R^2_\psi\) first; break ties on the scalar \(R^2\) mean. Official \(S\) on those shots is a later filter, not the train loss.

Conv / U-Net / FiLM / FNO are not in this file. They keep the \(65\times 65\) layout and get their own sizing if this flatten family saturates.
