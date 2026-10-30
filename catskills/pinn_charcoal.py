"""``CRC_FEQ_charcoal`` — teal's PCA/MLP flux map, trained in the scorer's own norm.

Charcoal is deliberately *not* a new architecture. Probe 0a settled that: ground-truth psi pushed
through teal's trained k=60 basis rescores at Consistency 0.9971, so the basis can already express
the geometry to three decimals and teal's 0.708 is a fitting failure, not a representation failure.
The planned FieldRefiner and full-field decoder (CHARCOAL_PLAN.md 4.1/4.2) are therefore cancelled,
and the entire budget goes into *what we optimize* rather than *what we optimize over*.

Three changes against teal, each traceable to a measurement:

1. **Geometry loss in the scorer's norm** (plan 6). ``soft_derive`` gives the seven consistency
   scalars differentiably at R2 >= 0.98 against ``fusion_scoring.derive.derive_frame``. Each residual
   is divided by that scalar's corpus variance, which is exactly the denominator the scorer's pooled
   R2 uses -- so the term is proportional to the score deficit it causes, and summing over j with
   equal weights is a differentiable surrogate for -Consistency. Teal's bounding-box proxies were
   biased estimators of the same quantities, which is the likeliest reason its ladder found
   ``lambda_z`` actively harmful and drove R2_kappa negative on two units.

   The target is ``soft_derive(psi_gt)``, *not* the scorer's value. Both sides then pass through the
   same surrogate and its residual bias cancels to first order, which matters most for tri_top (the
   one scalar still carrying a -2.1% bias after P1 tuning).

2. **Temporal features** (plan 4.3). Probe 0c showed test-time smoothing over w=5 is worth +0.0014 S
   on frozen predictions, and probe 0b showed every scalar is shrinkage- rather than noise-dominated,
   i.e. information-limited. So instead of only smoothing the output we give the network the
   temporal information as input: each frame sees its own features, a centred moving average, and a
   central difference. The difference channel is the one plan 4.3 argues for on physical grounds --
   vertical position control is a feedback loop acting through Idot on the shaping coils, and Z_axis
   (R2 0.670) is the scalar most plausibly limited by not seeing it.

3. **Model selection on Consistency, not psi MSE** (plan 6). Teal early-stopped on the term with
   0.0028 of headroom left while the term with 0.0585 went unwatched.

Everything else -- features, PCA, CoeffMLP, optimizer, L-BFGS polish -- is teal's, so a charcoal run
with ``lambda_geom = lambda_h1 = lambda_disp = 0`` and ``temporal_window = 0`` is teal, and any
difference measured at equal scale is attributable to the three changes above.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

# MODEL_ID and the tag resolvers live in lab_paths (they are a path policy, and the ladders need them
# without importing torch); re-exported here because every caller in the project imports them from
# this module and the name is part of its public surface.
from catskills.lab_paths import (  # noqa: F401
    LINEAGE_ID,
    MODEL_ID,
    artifact_name,
    find_checkpoint,
    find_score,
    results_dir,
)
from catskills.model_version import git_full_sha, make_model_version
from catskills.coeff_cache import (
    build_coeff_matrix_from_keys,
    load_basis as load_coeff_basis,
    trim_keys_to_frame_budget as coeff_trim_keys,
)
from catskills.feature_cache import (
    build_feature_matrix_cached,
    build_feature_matrix_from_keys,
)
from catskills.pinn_teal import (
    CoeffMLP,
    _activation,
    _default_grids_torch,
    _parse_hidden,
    _trim_keys_to_frame_budget,
    make_optimizer,
    require_cuda,
)
from catskills.soft_derive import SCALAR_NAMES, SoftDerive, SoftDeriveConfig
from experiments import EFIT_SCALAR_LABELS

# Scalars the scorer derives from psi. soft_derive returns them under these names.
N_CONS = len(SCALAR_NAMES)


# --------------------------------------------------------------------------- temporal features


def augment_temporal(X: np.ndarray, shot_ids: np.ndarray, window: int,
                     mode: str = "avgdiff", seq_window: int = 0) -> np.ndarray:
    """``(T, F) -> (T, 3F)``: raw frame, centred moving average, central difference.

    Applied per shot, so no window ever straddles a shot boundary, and applied to the *full*
    time-ordered matrix before any frame subsampling -- the feature cache concatenates shots
    contiguously in EFIT-time order, and teal's ``max_frames`` subsample shuffles that ordering away.

    The moving average supplies the noise reduction probe 0c measured at inference time, and the
    central difference supplies Idot. Edges are handled by clamping, which biases the first and last
    frame of a shot slightly; those are ramp-up/ramp-down frames where the equilibrium is least
    well posed anyway.
    """
    X = np.asarray(X, dtype=np.float32)
    if str(mode) == "smoothdiff+stack":
        # D2: C7b's 447-d [raw|avg|diff] plus a raw-frame stack for the GRU branch.
        # ``window`` is the dualctx smoothdiff width (must stay 9 to warm-start C7b).
        # ``seq_window`` is the GRU stack; 0 means the same as ``window``.
        sw = int(seq_window) if int(seq_window) > 1 else int(window)
        a = augment_temporal(X, shot_ids, window, "smoothdiff")
        b = augment_temporal(X, shot_ids, sw, "stack")
        return np.concatenate([a, b], axis=1)
    if window <= 1:
        return X
    half = int(window) // 2
    if mode == "stack":
        # D1's input: the raw frames themselves, [x_{t-half} .. x_{t+half}] concatenated per frame
        # and shot-clamped at the edges — NOT [raw | avg | diff]. The flat layout keeps the entire
        # training/inference/packaging pipeline unchanged; CoeffSeq reshapes to (B, W, F) inside
        # forward. Columns are ordered oldest-first so a recurrent encoder consumes them in time
        # order. Standardization stays per-column, which is per (offset, feature) pair here.
        W = 2 * half + 1
        F = X.shape[1]
        out = np.empty((X.shape[0], F * W), dtype=np.float32)
        for s in np.unique(shot_ids):
            sl = np.flatnonzero(shot_ids == s)
            if sl.size == 0:
                continue
            blk = X[sl]
            idx = np.arange(blk.shape[0])
            for k, d in enumerate(range(-half, half + 1)):
                out[sl, k * F : (k + 1) * F] = blk[np.clip(idx + d, 0, blk.shape[0] - 1)]
        return out
    out = np.empty((X.shape[0], X.shape[1] * 3), dtype=np.float32)
    out[:, : X.shape[1]] = X
    for s in np.unique(shot_ids):
        sl = np.flatnonzero(shot_ids == s)
        if sl.size == 0:
            continue
        blk = X[sl]
        T = blk.shape[0]
        idx = np.arange(T)
        # Centred moving average with clamped edges.
        acc = np.zeros_like(blk)
        for d in range(-half, half + 1):
            acc += blk[np.clip(idx + d, 0, T - 1)]
        acc /= float(2 * half + 1)
        # Central difference, clamped at the ends.
        #
        # ``mode="avgdiff"`` differences the RAW signal one frame apart, which is what R2a and R4 ran
        # and why both lost to a no-context baseline (0.95938 and 0.96979 against R2b's 0.97698). It
        # is the noisiest derivative available: the average above smooths over ``window`` frames while
        # this channel keeps full-bandwidth sensor noise, and standardizing it then divides by a very
        # small std, so noise arrives at the trunk as large z-scores. R4's clip could not repair a
        # channel that is mostly noise by construction.
        #
        # ``mode="smoothdiff"`` differences ``acc`` instead, spaced by ``half`` rather than 1. Same
        # Idot information, ~sqrt(2*half+1) less noise in it, and a std large enough that the channel
        # standardizes to a sane range. The raw and average channels are untouched, so this is a
        # single-variable change against R2a.
        if mode == "smoothdiff":
            h = max(1, half)
            dif = (acc[np.clip(idx + h, 0, T - 1)] - acc[np.clip(idx - h, 0, T - 1)]) / (2.0 * h)
        else:
            dif = 0.5 * (blk[np.clip(idx + 1, 0, T - 1)] - blk[np.clip(idx - 1, 0, T - 1)])
        out[sl, X.shape[1] : 2 * X.shape[1]] = acc
        out[sl, 2 * X.shape[1] :] = dif
    return out


@dataclass
class CharcoalBundle:
    checkpoint: Path
    n_features: int
    feature_mean: np.ndarray
    feature_std: np.ndarray
    include_thomson: bool
    n_pca: int
    device_name: str
    # Raw (pre-augmentation) width the training corpus was padded to. Inference must pad to this
    # *before* augmenting, or the [raw | avg | diff] blocks shift; see predict_from_raw_features.
    thomson_layout: str = ""
    n_raw_features: int = 0
    temporal_window: int = 5
    temporal_mode: str = "avgdiff"
    seq_window: int = 0
    raw_feature_mean: np.ndarray | None = None
    raw_feature_std: np.ndarray | None = None
    training_params: dict = field(default_factory=dict)
    model_version: str = ""
    pca_mean: np.ndarray | None = None
    pca_components: np.ndarray | None = None
    hidden: tuple[int, ...] = (256, 128)
    activation: str = "tanh"
    # "mlp" (CoeffMLP, everything before R1) or "resnet" (CoeffResNet).
    arch: str = "mlp"
    # Sigma clamp on standardized inputs; 0 = off. Inference must apply the same one.
    feature_clip: float = 0.0
    # teal-compat: charcoal never uses a residual map (4.1 cancelled).
    use_residual: bool = False


# --------------------------------------------------------------------------- input hygiene


def _clip_features(Xs: np.ndarray, clip: float, *, label: str = "") -> np.ndarray:
    """Clamp standardized features to +-``clip`` sigma. ``clip <= 0`` disables it.

    R1's psi_diag found the flux residual concentrated in a handful of holdout shots -- the worst 5%
    carry 35.9% of it -- and those shots are the ones whose inputs reach |z| = 53 on several columns.
    A 50-sigma input is a sensor artefact, not a plasma state, and an unbounded first layer turns it
    into an unbounded activation. Clipping is deliberately the crudest possible fix: it changes only
    the tail, leaves every in-range frame identical, and so keeps this a single-variable arm against
    R1. It must be applied at inference too, which is why the threshold rides in the checkpoint.
    """
    if clip is None or clip <= 0:
        return Xs
    hot = np.abs(Xs) > clip
    n_hot = int(hot.sum())
    if n_hot and label:
        cols = np.flatnonzero(hot.any(axis=0))
        worst = int(np.argmax(np.abs(Xs).max(axis=0)))
        print(
            f"[{MODEL_ID}] feature clip {label}: |z|>{clip:g} on {n_hot:,} cells "
            f"({100.0 * n_hot / Xs.size:.4f}% of {Xs.size:,}) across {cols.size} of "
            f"{Xs.shape[1]} columns; worst col {worst} at |z|={np.abs(Xs[:, worst]).max():.1f}",
            flush=True,
        )
    return np.clip(Xs, -clip, clip, out=Xs)


# --------------------------------------------------------------------------- trunks


def _act(name: str) -> torch.nn.Module:
    """``pinn_teal._activation`` plus ReLU, which the teal line never offered."""
    return torch.nn.ReLU() if str(name).lower() == "relu" else _activation(name)


class _ResBlock(torch.nn.Module):
    """Linear - act - dropout - Linear, added back to its own input."""

    def __init__(self, width: int, dropout: float, activation: str):
        super().__init__()
        self.fc1 = torch.nn.Linear(width, width)
        self.act = _act(activation)
        self.drop = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.fc2 = torch.nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.drop(self.act(self.fc1(x))))


class CoeffResNet(torch.nn.Module):
    """R1's trunk: a residual tabular network behind the same heads ``CoeffMLP`` exposes.

    Every charcoal and maroon model to date is a 2-layer 256/128 MLP, and R1 asks one question about
    that: is the flux ceiling at R2_psi ~0.9957 a property of the data or of this trunk? Rank 1 runs
    a ResNet over the same PCA formulation and reaches 0.9994, so the cheapest discriminating test is
    to change the trunk and nothing else -- same features, same frozen basis, same holdout, geometry
    off so the comparison is against M1c rather than against P1's geometry trade.

    The forward signature and the head names match ``CoeffMLP`` exactly so that the loss, the
    validation path, L-BFGS and the scorer all work on it unchanged; only ``load_charcoal`` has to
    know which class to rebuild, which is what the checkpoint's ``arch`` key records.
    """

    def __init__(
        self,
        n_in: int,
        n_pca: int,
        width: int = 512,
        blocks: int = 4,
        dropout: float = 0.05,
        activation: str = "relu",
    ):
        super().__init__()
        self.width, self.blocks_n = int(width), int(blocks)
        self.dropout_p, self.activation_name = float(dropout), activation
        self.stem = torch.nn.Linear(n_in, self.width)
        self.act = _act(activation)
        self.blocks = torch.nn.Sequential(
            *[_ResBlock(self.width, float(dropout), activation) for _ in range(self.blocks_n)]
        )
        self.coeff_head = torch.nn.Linear(self.width, n_pca)
        self.q95_head = torch.nn.Linear(self.width, 1)
        self.beta_head = torch.nn.Linear(self.width, 1)
        self.r_axis_head = torch.nn.Linear(self.width, 1)
        self.z_axis_head = torch.nn.Linear(self.width, 1)
        self.li_head = torch.nn.Linear(self.width, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.blocks(self.act(self.stem(x)))
        return (
            self.coeff_head(h),
            self.q95_head(h).squeeze(-1),
            self.beta_head(h).squeeze(-1),
            self.r_axis_head(h).squeeze(-1),
            self.z_axis_head(h).squeeze(-1),
            self.li_head(h).squeeze(-1),
        )


class CoeffDualContext(torch.nn.Module):
    """Purple's trunk: two encoders — present frame, temporal context — fused by a **zero-init gate**.

    Temporal context has now lost three times as a *concatenated* input (R2a −0.0157, R4 −0.0072,
    R10a −0.0036), and R10a is the control that matters: it fixed the noisy-derivative defect and lost
    anyway, so the surviving explanation is not the channel's noise but the single shared stem. One
    trunk fed ``[raw | moving-average | difference]`` as one flat vector has to serve a clean signal
    and a marginal one through the same first projection, and it pays for the marginal one in capacity
    it would otherwise spend on the flux map.

    Here the present frame gets its own stem and blocks, the context blocks get theirs, and the two
    meet as *embeddings* rather than as raw columns::

        h = blocks(act(stem(x[:, :F])))                    # exactly CoeffResNet on the raw frame
        c = ctx_blocks(act(ctx_stem(x[:, F:])))             # its own path, its own scale
        h = h + gate * ctx_proj(c)                          # gate starts at ZERO

    ``gate`` is a learned scalar initialised to 0, so **at initialisation this network is functionally
    identical to R7** and the context branch contributes nothing. The optimizer has to *opt into*
    context wherever it lowers the loss, instead of being force-fed it. If temporal information is
    worthless the gate stays near zero and we recover R7 rather than handing back 0.004–0.016 of S,
    which is precisely what the three concatenation arms did. That makes the downside of this
    experiment bounded by construction rather than by hope, and it gives a hard correctness test:
    with the gate at zero the output must be *exactly* invariant to the context columns.

    The forward signature and head names match ``CoeffMLP``/``CoeffResNet`` so the loss, validation,
    L-BFGS, the scorer, the packager and the ensembler all work on it unchanged; only
    ``load_charcoal`` needs the ``arch`` key to rebuild the right class.
    """

    def __init__(
        self,
        n_in: int,
        n_pca: int,
        n_split: int,
        width: int = 1024,
        blocks: int = 8,
        ctx_width: int = 512,
        ctx_blocks: int = 3,
        dropout: float = 0.05,
        activation: str = "relu",
    ):
        super().__init__()
        if not 0 < n_split < n_in:
            raise ValueError(f"n_split {n_split} must lie inside (0, n_in={n_in})")
        self.n_split = int(n_split)
        self.width, self.blocks_n = int(width), int(blocks)
        self.ctx_width, self.ctx_blocks_n = int(ctx_width), int(ctx_blocks)
        self.dropout_p, self.activation_name = float(dropout), activation

        self.stem = torch.nn.Linear(self.n_split, self.width)
        self.act = _act(activation)
        self.blocks = torch.nn.Sequential(
            *[_ResBlock(self.width, float(dropout), activation) for _ in range(self.blocks_n)]
        )
        self.ctx_stem = torch.nn.Linear(n_in - self.n_split, self.ctx_width)
        self.ctx_blocks = torch.nn.Sequential(
            *[_ResBlock(self.ctx_width, float(dropout), activation)
              for _ in range(self.ctx_blocks_n)]
        )
        self.ctx_proj = torch.nn.Linear(self.ctx_width, self.width)
        # LayerScale-style: one scalar, zero at init, so the branch is off until it earns its way on.
        self.gate = torch.nn.Parameter(torch.zeros(1))

        self.coeff_head = torch.nn.Linear(self.width, n_pca)
        self.q95_head = torch.nn.Linear(self.width, 1)
        self.beta_head = torch.nn.Linear(self.width, 1)
        self.r_axis_head = torch.nn.Linear(self.width, 1)
        self.z_axis_head = torch.nn.Linear(self.width, 1)
        self.li_head = torch.nn.Linear(self.width, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.embed(x)
        return self.heads(h)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.act(self.stem(x[:, : self.n_split])))
        c = self.ctx_blocks(self.act(self.ctx_stem(x[:, self.n_split:])))
        return h + self.gate * self.ctx_proj(c)

    def heads(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.coeff_head(h),
            self.q95_head(h).squeeze(-1),
            self.beta_head(h).squeeze(-1),
            self.r_axis_head(h).squeeze(-1),
            self.z_axis_head(h).squeeze(-1),
            self.li_head(h).squeeze(-1),
        )


class CoeffGrand(CoeffDualContext):
    """Dual-context trunk with a 3-layer head for q95 and betaN.

    GradNorm calls ``embed`` and then ``heads`` with no batch, so the raw
    neutral-beam slice has to be stashed in ``embed``. The last ``n_nbi``
    columns of the present frame are that slice. ``n_nbi=0`` when the parquet
    has no beam-power column: the head is then an MLP on the trunk embedding
    alone, and no zero-power channel is invented.
    """

    def __init__(
        self,
        n_in: int,
        n_pca: int,
        n_split: int,
        width: int = 1536,
        blocks: int = 8,
        ctx_width: int = 768,
        ctx_blocks: int = 3,
        dropout: float = 0.05,
        activation: str = "relu",
        n_nbi: int = 0,
    ):
        super().__init__(
            n_in, n_pca, n_split, width=width, blocks=blocks,
            ctx_width=ctx_width, ctx_blocks=ctx_blocks,
            dropout=dropout, activation=activation,
        )
        self.n_nbi = int(n_nbi)
        if self.n_nbi < 0 or self.n_nbi >= self.n_split:
            raise ValueError(f"n_nbi={self.n_nbi} does not fit in n_split={self.n_split}")
        hid = max(256, self.width // 3)
        self.qb_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.width + self.n_nbi, hid),
            _act(activation),
            torch.nn.Linear(hid, hid),
            _act(activation),
            torch.nn.Linear(hid, 2),
        )
        for p in (*self.q95_head.parameters(), *self.beta_head.parameters()):
            p.requires_grad_(False)
        self._nbi: torch.Tensor | None = None

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_nbi:
            self._nbi = x[:, self.n_split - self.n_nbi : self.n_split]
        else:
            self._nbi = None
        return super().embed(x)

    def heads(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        nbi = self._nbi
        if self.n_nbi and (nbi is None or nbi.shape[0] != h.shape[0]):
            nbi = h.new_zeros((h.shape[0], self.n_nbi))
        feat = torch.cat([h, nbi], dim=1) if self.n_nbi else h
        qb = self.qb_mlp(feat)
        return (
            self.coeff_head(h),
            qb[:, 0],
            qb[:, 1],
            self.r_axis_head(h).squeeze(-1),
            self.z_axis_head(h).squeeze(-1),
            self.li_head(h).squeeze(-1),
        )


class CoeffDualSeq(CoeffDualContext):
    """D2: C7b's dualctx plus a gated GRU on a raw-frame stack.

    Module names of the parent match a C7b checkpoint, so ``strict=False`` warm-start loads the
    flux encoder and leaves ``gru`` / ``seq_proj`` / ``gate_seq`` at init. ``gate_seq`` starts at
    zero, so the network is C7b's present-frame+ctx-ResNet path until the optimizer opens it.
    """

    def __init__(
        self,
        n_in: int,
        n_pca: int,
        n_split: int,
        width: int = 1024,
        blocks: int = 8,
        ctx_width: int = 512,
        ctx_blocks: int = 3,
        dropout: float = 0.05,
        activation: str = "relu",
        seq_hidden: int = 256,
        seq_layers: int = 2,
    ):
        dual_w = 3 * int(n_split)
        if n_in <= dual_w or (n_in - dual_w) % int(n_split) != 0:
            raise ValueError(
                f"dualseq needs n_in ({n_in}) = 3*n_split + W*n_split "
                f"(n_split={n_split}); got remainder {n_in - dual_w}"
            )
        super().__init__(
            dual_w, n_pca, n_split, width=width, blocks=blocks,
            ctx_width=ctx_width, ctx_blocks=ctx_blocks, dropout=dropout, activation=activation,
        )
        self.seq_window = int((n_in - dual_w) // n_split)
        self.seq_hidden_n, self.seq_layers_n = int(seq_hidden), int(seq_layers)
        self.gru = torch.nn.GRU(
            int(n_split), self.seq_hidden_n, num_layers=self.seq_layers_n, batch_first=True,
            dropout=self.dropout_p if self.seq_layers_n > 1 else 0.0,
        )
        self.seq_proj = torch.nn.Linear(self.seq_hidden_n, self.width)
        self.gate_seq = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dual_w = 3 * self.n_split
        h = self.embed(x[:, :dual_w])
        seq = x[:, dual_w:].view(x.shape[0], self.seq_window, self.n_split)
        out, _ = self.gru(seq)
        h = h + self.gate_seq * self.seq_proj(out[:, -1])
        return self.heads(h)


class CoeffDualShot(CoeffDualContext):
    """C7b dualctx plus a gated bidirectional GRU over the whole shot.

    Parent is inited at n_in=3*n_split so a C7b ``state_dict`` loads. The GRU reads the raw
    149-d frames of the discharge and writes a *per-frame* residual into the dualctx embedding.
    ``gate_seq`` starts at 0, so step 0 is C7b. This is D2's topology on Forest's axis.
    """

    def __init__(
        self,
        n_in: int,
        n_pca: int,
        n_split: int,
        width: int = 1024,
        blocks: int = 8,
        ctx_width: int = 512,
        ctx_blocks: int = 3,
        dropout: float = 0.05,
        activation: str = "relu",
        seq_hidden: int = 256,
        seq_layers: int = 1,
        bidirectional: bool = True,
    ):
        dual_w = 3 * int(n_split)
        if int(n_in) != dual_w:
            raise ValueError(
                f"dualshot dualctx block is 3*n_split={dual_w}; got n_in={n_in}"
            )
        super().__init__(
            dual_w, n_pca, n_split, width=width, blocks=blocks,
            ctx_width=ctx_width, ctx_blocks=ctx_blocks, dropout=dropout, activation=activation,
        )
        self.seq_hidden_n, self.seq_layers_n = int(seq_hidden), int(seq_layers)
        self.bidirectional = bool(bidirectional)
        self.gru = torch.nn.GRU(
            int(n_split), self.seq_hidden_n, num_layers=self.seq_layers_n, batch_first=True,
            dropout=self.dropout_p if self.seq_layers_n > 1 else 0.0,
            bidirectional=self.bidirectional,
        )
        h_out = self.seq_hidden_n * (2 if self.bidirectional else 1)
        self.seq_proj = torch.nn.Linear(h_out, self.width)
        self.gate_seq = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x_dual: torch.Tensor,
        x_raw: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x_raw is None:
            return self.heads(self.embed(x_dual))
        squeeze = x_dual.dim() == 2
        if squeeze:
            x_dual = x_dual.unsqueeze(0)
            x_raw = x_raw.unsqueeze(0)
        B, T, _ = x_dual.shape
        h = self.embed(x_dual.reshape(B * T, -1)).reshape(B, T, self.width)
        if lengths is None:
            lengths = torch.full((B,), T, dtype=torch.long)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x_raw, lengths.to(dtype=torch.long, device="cpu"),
            batch_first=True, enforce_sorted=False,
        )
        packed_out, _ = self.gru(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=T,
        )
        h = h + self.gate_seq * self.seq_proj(out)
        z, q, b, ra, za, li = self.heads(h.reshape(B * T, self.width))
        z = z.view(B, T, -1)
        q, b, ra, za, li = (t.view(B, T) for t in (q, b, ra, za, li))
        if squeeze:
            return z[0], q[0], b[0], ra[0], za[0], li[0]
        return z, q, b, ra, za, li

    def embed_fuse(
        self,
        x_dual: torch.Tensor,
        x_raw: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fused embedding immediately before the linear heads (Olive GradNorm layer)."""
        if x_raw is None:
            return self.embed(x_dual)
        squeeze = x_dual.dim() == 2
        if squeeze:
            x_dual = x_dual.unsqueeze(0)
            x_raw = x_raw.unsqueeze(0)
        B, T, _ = x_dual.shape
        h = self.embed(x_dual.reshape(B * T, -1)).reshape(B, T, self.width)
        if lengths is None:
            lengths = torch.full((B,), T, dtype=torch.long)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x_raw, lengths.to(dtype=torch.long, device="cpu"),
            batch_first=True, enforce_sorted=False,
        )
        packed_out, _ = self.gru(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=T,
        )
        h = h + self.gate_seq * self.seq_proj(out)
        h = h.reshape(B * T, self.width)
        return h[0] if squeeze else h


class CoeffSeq(torch.nn.Module):
    """D1's trunk: a small GRU over the raw frame window, MLP-style heads on the last state.

    The 2026-08-15 19:58 board falsified the premise of the architecture lock: the new #1
    (pshishkin, 0.9932) declares ``seq + mlp`` with the compute-light badge, so a *recurrent*
    delivery of temporal context trains in under 2 h and beats every dualctx mix we own. Context
    has now been delivered three ways: flat concat lost three times (R2a/R4/R10a), a gated second
    encoder won (C1) and saturated (C7/C8/C9), and a sequence encoder is the one delivery never
    tried. This class is that test, kept deliberately small (~0.7M parameters at 2x256) so the arm
    is itself compute-light.

    Input stays FLAT: ``augment_temporal(mode="stack")`` lays the window out as
    ``[x_{t-h} .. x_{t+h}]`` per frame, so the loss, validation, packager, ensembler and scorer
    all work unchanged; ``forward`` reshapes to ``(B, W, F)`` and runs the GRU oldest-first. The
    heads read the LAST hidden state, which by then has consumed the whole window. The forward
    signature and head names match ``CoeffMLP``/``CoeffResNet``/``CoeffDualContext`` exactly;
    ``load_charcoal`` rebuilds via the checkpoint's ``arch`` key as with every other trunk.
    """

    def __init__(
        self,
        n_in: int,
        n_pca: int,
        n_split: int,
        hidden: int = 256,
        layers: int = 2,
        dropout: float = 0.05,
    ):
        super().__init__()
        if n_split <= 0 or n_in % n_split != 0:
            raise ValueError(
                f"seq trunk needs n_in ({n_in}) divisible by n_split ({n_split}); "
                "n_split is the RAW feature width and n_in/n_split the window length"
            )
        self.n_split = int(n_split)
        self.window = int(n_in // n_split)
        self.hidden_n, self.layers_n = int(hidden), int(layers)
        self.dropout_p = float(dropout)
        self.gru = torch.nn.GRU(
            self.n_split, self.hidden_n, num_layers=self.layers_n, batch_first=True,
            dropout=self.dropout_p if self.layers_n > 1 else 0.0,
        )
        self.coeff_head = torch.nn.Linear(self.hidden_n, n_pca)
        self.q95_head = torch.nn.Linear(self.hidden_n, 1)
        self.beta_head = torch.nn.Linear(self.hidden_n, 1)
        self.r_axis_head = torch.nn.Linear(self.hidden_n, 1)
        self.z_axis_head = torch.nn.Linear(self.hidden_n, 1)
        self.li_head = torch.nn.Linear(self.hidden_n, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = x.view(x.shape[0], self.window, self.n_split)
        out, _ = self.gru(seq)
        h = out[:, -1]
        return (
            self.coeff_head(h),
            self.q95_head(h).squeeze(-1),
            self.beta_head(h).squeeze(-1),
            self.r_axis_head(h).squeeze(-1),
            self.z_axis_head(h).squeeze(-1),
            self.li_head(h).squeeze(-1),
        )


def build_trunk(
    arch: str,
    n_in: int,
    n_pca: int,
    *,
    hidden: tuple[int, ...] = (256, 128),
    activation: str = "tanh",
    trunk_width: int = 512,
    trunk_blocks: int = 4,
    trunk_dropout: float = 0.05,
    ctx_width: int = 512,
    ctx_blocks: int = 3,
    n_split: int = 0,
    seq_hidden: int = 256,
    seq_layers: int = 2,
    shot_bidirectional: bool = True,
    shot_mlp_blocks: int = 2,
    n_nbi: int = 0,
) -> torch.nn.Module:
    """One place that maps a saved ``arch`` string to a model, so training and inference cannot
    disagree about what a checkpoint contains."""
    if str(arch).lower() in ("", "mlp"):
        return CoeffMLP(n_in, n_pca, hidden=hidden, activation=activation)
    if str(arch).lower() == "resnet":
        return CoeffResNet(
            n_in, n_pca, width=trunk_width, blocks=trunk_blocks,
            dropout=trunk_dropout, activation=activation,
        )
    if str(arch).lower() == "dualctx":
        if not n_split:
            raise ValueError(
                "arch 'dualctx' needs n_split (the RAW feature width); it splits the augmented "
                "vector into [present frame | context], so it requires --temporal-window > 0"
            )
        return CoeffDualContext(
            n_in, n_pca, n_split, width=trunk_width, blocks=trunk_blocks,
            ctx_width=ctx_width, ctx_blocks=ctx_blocks,
            dropout=trunk_dropout, activation=activation,
        )
    if str(arch).lower() == "grand":
        if not n_split:
            raise ValueError("arch 'grand' needs n_split (the RAW feature width)")
        return CoeffGrand(
            n_in, n_pca, n_split, width=trunk_width, blocks=trunk_blocks,
            ctx_width=ctx_width, ctx_blocks=ctx_blocks,
            dropout=trunk_dropout, activation=activation, n_nbi=n_nbi,
        )
    if str(arch).lower() == "seq":
        if not n_split:
            raise ValueError(
                "arch 'seq' needs n_split (the RAW feature width); it reshapes the stacked "
                "window into (B, W, F), so it requires --temporal-window > 1 with "
                "--temporal-mode stack"
            )
        return CoeffSeq(
            n_in, n_pca, n_split, hidden=seq_hidden, layers=seq_layers, dropout=trunk_dropout,
        )
    if str(arch).lower() == "dualseq":
        if not n_split:
            raise ValueError(
                "arch 'dualseq' needs n_split (the RAW feature width); it splits "
                "[raw|avg|diff | stack] so it requires --temporal-mode smoothdiff+stack"
            )
        return CoeffDualSeq(
            n_in, n_pca, n_split, width=trunk_width, blocks=trunk_blocks,
            ctx_width=ctx_width, ctx_blocks=ctx_blocks,
            dropout=trunk_dropout, activation=activation,
            seq_hidden=seq_hidden, seq_layers=seq_layers,
        )
    if str(arch).lower() == "shotseq":
        from catskills.pinn_forest import CoeffShotSeq
        return CoeffShotSeq(
            n_in, n_pca, hidden=seq_hidden, layers=seq_layers,
            dropout=trunk_dropout, bidirectional=shot_bidirectional,
            mlp_blocks=shot_mlp_blocks, activation=activation,
        )
    if str(arch).lower() == "dualshot":
        if not n_split:
            raise ValueError(
                "arch 'dualshot' needs n_split (the RAW feature width); the dualctx "
                "block is 3*n_split and the GRU reads the raw shot"
            )
        return CoeffDualShot(
            n_in, n_pca, n_split, width=trunk_width, blocks=trunk_blocks,
            ctx_width=ctx_width, ctx_blocks=ctx_blocks,
            dropout=trunk_dropout, activation=activation,
            seq_hidden=seq_hidden, seq_layers=seq_layers,
            bidirectional=shot_bidirectional,
        )
    raise ValueError(
        f"unknown arch {arch!r} (expected 'mlp', 'resnet', 'dualctx', 'grand', "
        f"'seq', 'dualseq', 'shotseq' or 'dualshot')"
    )


# --------------------------------------------------------------------------- geometry targets


def _geom_gt(
    sd: SoftDerive, y_t: torch.Tensor, chunk: int = 512, progress: bool = True
) -> torch.Tensor:
    """``soft_derive`` on every ground-truth frame -> ``[n, 7]`` targets.

    Precomputed once: the targets never change, so paying for them per batch would double the cost
    of the geometry loss for nothing.
    """
    outs = []
    rng = range(0, int(y_t.shape[0]), chunk)
    if progress:
        rng = tqdm(rng, desc="geometry targets", unit="chunk", leave=False)
    with torch.no_grad():
        for start in rng:
            g = sd(y_t[start : start + chunk])
            outs.append(torch.stack([g[k] for k in SCALAR_NAMES], dim=1))
    return torch.cat(outs, dim=0)


def _plausible_mask(g: torch.Tensor) -> torch.Tensor:
    """Finite and inside ``derive.PLAUSIBLE`` -- the scorer nan's the rest, so we must not fit it."""
    from catskills.soft_derive import PLAUSIBLE

    ok = torch.isfinite(g)
    for j, name in enumerate(SCALAR_NAMES):
        if name in PLAUSIBLE:
            lo, hi = PLAUSIBLE[name]
            ok[:, j] &= (g[:, j] >= lo) & (g[:, j] <= hi)
    return ok


# Per-scalar Consistency from the PUBLIC charcoal scorecard (submission 879268, 2026-08-05, 874
# DIII-D shots). Public rather than local because the 80-shot holdout understates Consistency by
# 0.016-0.021 unevenly across scalars -- see CHARCOAL_PLAN.md section 15.1.
PUBLIC_CONS_R2 = {
    "R_axis": 0.885426, "Z_axis": 0.810280, "kappa": 0.833633, "tri_top": 0.952071,
    "tri_bot": 0.943116, "volume": 0.839940, "li": 0.792975,
}


def derive_geom_weights(target: float = 0.95, floor: float = 0.25) -> np.ndarray:
    """Per-scalar geometry weights from the remaining public deficit, normalised to mean 1.

    Uniform weighting spends 1/7 of the geometry gradient on tri_top, which is already at 0.952 --
    at rank #2's own level, with nothing left to win -- while li sits at 0.793. Reweighting is
    therefore not a bet on deficit-proportionality being optimal; it is declining to spend effort
    where there is no deficit. ``floor`` is what stops that from becoming a regression: a scalar at
    the target keeps a quarter of its share rather than being dropped, because a term with zero
    weight is free to drift back down.
    """
    d = np.array([max(0.0, target - PUBLIC_CONS_R2[n]) for n in SCALAR_NAMES], dtype=np.float64)
    hi = d.max()
    w = np.full_like(d, 1.0) if hi <= 0 else floor + (1.0 - floor) * (d / hi)
    return (w / w.mean()).astype(np.float32)


# The three knobs Forest's oracles proved move S (F15 s_Z / kappa, F32 s_iso / volume,
# F57 dZ / Z_axis) and its probes proved F9 never learned (F16, F21, F58). Brown puts
# the geometry gradient on those three from epoch 1. Floor keeps R_axis / tri / li
# from drifting off the graph; zero weight is free to regress.
STACK_GEOM_FOCUS = {"kappa": 1.5, "Z_axis": 1.0, "volume": 1.0}
STACK_GEOM_FLOOR = 0.15


def derive_stack_geom_weights(
    floor: float = STACK_GEOM_FLOOR,
    focus: dict[str, float] | None = None,
) -> np.ndarray:
    """Brown's geometry weights: kappa, Z_axis, volume. Normalised to mean 1.

    Pass Olive's focus (includes ``li``) to promote internal inductance. The
    default focus is B8's and must not change.
    """
    w = np.full(len(SCALAR_NAMES), float(floor), dtype=np.float64)
    for name, val in (focus or STACK_GEOM_FOCUS).items():
        w[SCALAR_NAMES.index(name)] = float(val)
    return (w / w.mean()).astype(np.float32)


def _geom_loss(
    g_hat: torch.Tensor,
    g_ref: torch.Tensor,
    ok: torch.Tensor,
    sigma: torch.Tensor,
    huber: float,
    w: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean over valid (frame, scalar) of the variance-normalized residual, Huberised.

    Quadratic within ``huber`` sigmas, so for a model that is already close the term still equals
    the Consistency deficit; linear beyond, so a handful of frames whose soft geometry has gone
    haywire cannot dominate the batch gradient. That matters most for R_axis, whose sigma is 0.034 m
    -- a third of a grid cell -- so a single badly-placed axis is worth hundreds of sigma squared.

    ``w`` reweights the seven scalars (mean 1, so lambda_geom keeps its meaning). It divides by the
    weighted count, not the raw one, so the term stays on the same scale as the uniform version and
    the two are directly comparable across runs.
    """
    r = (g_hat - g_ref) / sigma
    if huber and huber > 0:
        a = r.abs()
        d = torch.where(a <= huber, r * r, huber * (2.0 * a - huber))
    else:
        d = r * r
    okf = ok.to(d.dtype)
    if w is not None:
        okf = okf * w
    return (torch.where(ok, d, torch.zeros_like(d)) * okf).sum() / okf.sum().clamp_min(1.0)


def _cons_proxy(
    g_hat: torch.Tensor, g_ref: torch.Tensor, ok: torch.Tensor
) -> tuple[float, list[float]]:
    """Pooled per-scalar R2 then ``mean(max(0, .))`` -- the scorer's Consistency, on our surrogate.

    Mirrors ``fusion_scoring.metrics``: SS_tot comes from the reference over the whole fold, and a
    scalar with no valid frames is dropped from the mean rather than scored 0.
    """
    per: list[float] = []
    for j in range(N_CONS):
        m = ok[:, j]
        if int(m.sum()) < 2:
            per.append(float("nan"))
            continue
        ref = g_ref[m, j]
        res = ((g_hat[m, j] - ref) ** 2).sum()
        tot = ((ref - ref.mean()) ** 2).sum()
        per.append(float(1.0 - res / tot) if float(tot) > 0 else float("nan"))
    good = [p for p in per if np.isfinite(p)]
    return (float(np.mean([max(0.0, p) for p in good])) if good else 0.0), per


# --------------------------------------------------------------------------- coefficient targets


class CoeffTargets:
    """Ground-truth psi held as PCA coefficients, reconstructed on demand (maroon M0).

    Charcoal parked the targets on the GPU as 65x65 grids, which is 26.4 GB for the indexed corpus
    against a 12 GB card -- that memory wall, not a modelling choice, is where ``--max-frames 200000``
    came from and why every shipped charcoal run trained on 12.8% of the data. As 60 coefficients the
    same targets are 374 MB, and probe 0a measured the basis reconstructing ground truth at R2_psi
    1.000000 (M0 re-gates it on shots the basis fit never saw), so nothing the scorer can read is
    lost.

    Reconstruction is the matmul the training step already performs for its own prediction, so the
    per-batch cost is one extra ``(B, 60) @ (60, 4225)`` -- negligible beside the loss itself. Reads
    the same way a tensor does, so ``_gather`` and ``_geom_gt`` are unchanged.
    """

    def __init__(self, z: torch.Tensor, comp: torch.Tensor, mean: torch.Tensor) -> None:
        self._z, self._comp, self._mean = z, comp, mean
        self.shape = (int(z.shape[0]), 65, 65)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx) -> torch.Tensor:
        return (self._z[idx] @ self._comp + self._mean).view(-1, 65, 65)

    def var(self, chunk: int = 8192) -> torch.Tensor:
        """Variance over every pixel of every frame, matching ``y_t.var()`` on the grid tensor.

        Accumulated in float64 over chunks: the grid form never exists at full size, and a float32
        sum over 6.6e9 pixels would lose the low digits that ``lambda_geom`` is derived from.
        """
        n = self.shape[0]
        total = torch.zeros((), dtype=torch.float64, device=self._z.device)
        total_sq = torch.zeros((), dtype=torch.float64, device=self._z.device)
        count = 0
        with torch.no_grad():
            for start in range(0, n, chunk):
                y = self[start : start + chunk].double()
                total += y.sum()
                total_sq += (y * y).sum()
                count += y.numel()
        m = total / max(count, 1)
        return (total_sq / max(count, 1) - m * m).float()


# --------------------------------------------------------------------------- training


def train_charcoal(
    shots: list[dict] | None = None,
    *,
    cache_keys: list[str] | None = None,
    # maroon M1: read targets from the coefficient cache instead of the grid cache. This is the flag
    # that lets a run exceed ~250k frames at all; see CoeffTargets.
    coeff_targets: bool = False,
    include_thomson: bool = True,
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 1e-3,
    max_frames: int = 32768,
    n_pca: int = 60,
    temporal_window: int = 5,
    # "avgdiff" is what R2a/R4 ran and is kept as the default so those runs still reproduce exactly.
    # "smoothdiff" differences the smoothed signal instead; see augment_temporal.
    temporal_mode: str = "avgdiff",
    lambda_qb: float = 1.0,
    lambda_geom: float = -1.0,
    geom_emphasis: float = 30.0,
    lambda_h1: float = 0.1,
    lambda_disp: float = 0.1,
    # C5: weight of a per-mode-rescaled coefficient term. Both flux terms in the loss are
    # variance-weighted -- measured over 88,105 frames, mode 1's coefficient std is 10.494 and mode
    # 60's is 0.00074, so mode 1 carries 2e8 times the variance, and because the basis is orthonormal
    # L_data weights modes exactly as L_coeff does. Above roughly mode 20 the gradient is numerically
    # nil. That was ruled harmless in maroon 11.17, but that argument was made in psi-MSE units and
    # C3 then showed psi MSE is anti-aligned with Consistency. 0 = off, which reproduces every run to
    # date bit for bit.
    lambda_coeff_white: float = 0.0,
    # C5: derive geometry targets on a slice of the VALIDATION frames even when the geometry loss is
    # off, purely so the checkpoint selector can see Consistency. Without it ``val_cons`` is NaN, the
    # S-proxy branch below is unreachable, and every run from R1 through C4 selected on ``-val_mse``
    # alone (their logs all print ``best selection=-0.0000``). C3 then showed psi MSE and Consistency
    # move in opposite directions late in training, which makes psi-only selection an active mistake
    # rather than a harmless simplification. 0 disables it and restores the pre-C5 selector exactly.
    #
    # Default 0, i.e. opt-in, deliberately: flipping the default would silently change which epoch
    # every queued run selects, so the C4 members trained after this edit would not be comparable to
    # C4a trained before it. C5b measures what the selector is worth; the default moves when there is
    # a number behind it.
    val_geom_frames: int = 0,
    # Tempering. Dividing by the raw std (exponent 1.0) is full whitening, which 11.17 argues against:
    # 14 of 60 components are predicted worse than the basis mean, so equal weighting pours capacity
    # into modes that are noise to us. The clamp floors the divisor at mode ``coeff_white_clamp``'s
    # std, so modes past it share one weight instead of escalating without bound.
    coeff_white_power: float = 1.0,
    coeff_white_clamp: int = 15,
    # C8: reweight L_data in (R,Z), not in PCA space. "off" is every run to date. Weights are
    # mean-normalized to 1 so L_data stays in the same units and lambda_coeff_white=1.35 stays
    # calibrated. "grad" uses |∇ψ_gt|^power (LCFS / X-point light up). "band" is a donut on
    # normalized ψ (1 on edge_band_lo <= psi_n <= edge_band_hi, 0 elsewhere). edge_mix is the
    # convex weight on that map vs uniform 1: 0 = C7b, 1 = pure edge map.
    edge_weight: str = "off",
    edge_mix: float = 0.6,
    edge_power: float = 1.0,
    edge_band_lo: float = 0.70,
    edge_band_hi: float = 1.05,
    geom_frac: float = 1.0,
    geom_warmup: int = 10,
    geom_sd_chunk: int = 256,
    probe_epochs: int = 0,
    geom_huber: float = 3.0,
    # P6b: "uniform" is the pre-P6 behaviour, "deficit" weights by the public per-scalar gap.
    geom_weights: str = "uniform",
    geom_weight_target: float = 0.95,
    geom_weight_floor: float = 0.25,
    geom_kappa_mult: float = 1.0,
    geom_raxis_mult: float = 1.0,
    geom_zaxis_mult: float = 1.0,
    geom_weight_vec: tuple[float, ...] | None = None,
    lr_eta_min: float = 0.0,
    inherit_feature_stats: bool = False,
    # P6a: the soft symmetric-Hausdorff D_LCFS term. 0 = off, which reproduces P4 exactly.
    lcfs_emphasis: float = 0.0,
    lambda_lcfs: float = -1.0,
    # The scorer clamps D_LCFS at 1.0; training clamps higher so that a frame the scorer has already
    # written off still has somewhere to go.
    lcfs_cap: float = 2.0,
    # P6c: decay the geometry emphasis over the last `geom_anneal_tail` fraction of epochs, down to
    # `geom_emphasis_final` times the original. 1.0 = no anneal, which is P4's behaviour.
    geom_emphasis_final: float = 1.0,
    geom_anneal_tail: float = 0.25,
    # P6d: divide each scalar residual by its own corpus std, making L_qb identically (1 - R2_qb) and
    # lambda_qb a score-unit quantity like lambda_geom. Off by default = P4's raw-MSE behaviour.
    qb_normalize: bool = False,
    qb_emphasis: float = 1.0,
    tau_hi: float = 0.05,
    tau_lo: float = 0.005,
    tau_li_hi: float = 0.08,
    tau_li_lo: float = 0.02,
    anneal_frac: float = 0.5,
    band_lo: float = 0.85,
    band_hi: float = 1.05,
    val_fraction: float = 0.1,
    early_stop_patience: int = 0,
    kill_r2psi: float = 0.0,
    kill_r2psi_after: int = 8,
    init_checkpoint: Path | str | None = None,
    optimizer: str = "adamw",
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    hidden: str | tuple[int, ...] = (256, 128),
    activation: str = "tanh",
    # R2: clamp standardized inputs to +-this many sigma. 0 = off, which is every run before R2b.
    feature_clip: float = 0.0,
    # R1: "mlp" is every model shipped to date; "resnet" swaps the trunk for CoeffResNet and makes
    # the three trunk_* arguments live. "dualctx" (purple) additionally splits the augmented vector
    # into [present frame | context] and gives the context its own encoder behind a zero-init gate,
    # which needs temporal_window > 0. Nothing else in the run changes.
    arch: str = "mlp",
    trunk_width: int = 512,
    trunk_blocks: int = 4,
    trunk_dropout: float = 0.05,
    ctx_width: int = 512,
    ctx_blocks: int = 3,
    # D1: the "seq" trunk (CoeffSeq) — GRU size over the stacked raw-frame window.
    seq_hidden: int = 256,
    seq_layers: int = 2,
    # D2: GRU stack width independent of dualctx smoothdiff window; 0 = same as temporal_window.
    seq_window: int = 0,
    freeze_dualctx: bool = False,
    shot_bidirectional: bool = True,
    shot_mlp_blocks: int = 2,
    token_budget: int = 1024,
    # R6: the two stabilizers R5 lacked. Both default to off so earlier runs reproduce exactly.
    grad_clip: float = 0.0,
    lr_warmup: int = 0,
    # Continue a .partial.pt written by this same config. Only partials carrying a "resume" block
    # qualify; anything older is scoreable but not continuable, and is refused rather than warm-started.
    resume: Path | None = None,
    lbfgs_steps: int = 0,
    lbfgs_max_frames: int = 8192,
    seed: int = 42,
    init_seed: int | None = None,
    output: Path | None = None,
    log_every: int = 10,
    ckpt_every: int = 25,
    loss_report: bool = False,
    geom_ramp: int = 0,
    gradnorm: bool = False,
    gradnorm_alpha: float = 0.5,
    gradnorm_w_from: int = 10,
    pcgrad: bool = False,
    stack_li: bool = False,
    lp_floor: float = 0.0,
    kill_cons: float = 0.0,
    kill_cons_after: int = 8,
    psi_collapse_r2psi: float = 0.0,
    psi_collapse_streak: int = 3,
    adaptive_warmup: bool = False,
    warmup_cap: int = 25,
    warmup_r2psi: float = 0.999,
    warmup_cons: float = 0.96,
    nondim: bool = False,
    gray: bool = False,
    sky: bool = False,
    sky_tokens: bool = False,
    yellow: bool = False,
) -> CharcoalBundle:
    device = require_cuda()
    # ``seed`` fixes the data: shot sample, frame subsample, PCA, validation shots. ``init_seed``
    # fixes only weight init and batch order. Separating them is what makes a replication study
    # possible -- varying ``seed`` would also move the holdout, so the scores would not compare.
    torch.manual_seed(seed if init_seed is None else init_seed)
    np.random.seed(seed)
    hidden_t = _parse_hidden(hidden)

    # Y is the grid targets when they fit in memory and None when they do not; the coefficient path
    # carries Z instead and materialises grids only per batch.
    Y = Z_pre = None
    if coeff_targets:
        if cache_keys is None:
            raise ValueError("coeff_targets needs cache_keys (the coefficient cache is keyed)")
        if max_frames > 0:
            cache_keys = coeff_trim_keys(
                cache_keys, max_frames, include_thomson=include_thomson, seed=seed
            )
        X, Z_pre, S, frame_shot_idx, cache_stats = build_coeff_matrix_from_keys(
            cache_keys, k=n_pca, include_thomson=include_thomson
        )
        print(
            f"[{MODEL_ID}] coefficient targets: {len(X):,} frames x {Z_pre.shape[1]} coeffs "
            f"= {Z_pre.nbytes / 1e6:.0f} MB (grids would be "
            f"{len(X) * 65 * 65 * 4 / 1e9:.1f} GB)",
            flush=True,
        )
    elif cache_keys is not None:
        if max_frames > 0:
            cache_keys = _trim_keys_to_frame_budget(
                cache_keys, max_frames, include_thomson=include_thomson, seed=seed
            )
        X, Y, S, frame_shot_idx, cache_stats = build_feature_matrix_from_keys(
            cache_keys, include_thomson=include_thomson
        )
    else:
        X, Y, S, frame_shot_idx, cache_stats = build_feature_matrix_cached(
            shots or [], include_thomson=include_thomson
        )

    if nondim:
        from catskills.red_nondim import nondim_matrix
        X = nondim_matrix(X, source="DIII-D")
        print(f"[{MODEL_ID}] Red non-dim X  magnetics I/Ip, Bt/B0, Te keV, ne a²/|Ip|",
              flush=True)
    if gray:
        from catskills.gray_roles import N_ROLES, roles_from_d3d_matrix
        X = roles_from_d3d_matrix(X, source="DIII-D")
        print(f"[{MODEL_ID}] Gray role X  {N_ROLES} roles + Thomson "
              f"(Ip, sol, TF, E_shape)",
              flush=True)
    if sky:
        from catskills.sky_geom import VAC_N, ensure_d3d_green, vac_from_d3d_matrix
        pack = ensure_d3d_green()
        X = vac_from_d3d_matrix(X, pack)
        print(
            f"[{MODEL_ID}] Sky vac X  ψ_vac {VAC_N}×{VAC_N} + Ip + TF + Thomson "
            f"(not 21 names, not 4 PCA)",
            flush=True,
        )
    if sky_tokens:
        from catskills.sky_geom import ensure_d3d_green, tokens_from_d3d_matrix
        pack = ensure_d3d_green()
        X = tokens_from_d3d_matrix(X, pack)
        print(
            f"[{MODEL_ID}] Sky token X  (r,z,I/|Ip|)×{len(pack['coil_idx'])} + Ip + TF + Thomson",
            flush=True,
        )
    if yellow:
        from catskills.yellow_geom import N_AUG, TS_N, ensure_ts_green, yellow_from_d3d_matrix
        pack = ensure_ts_green()
        X = yellow_from_d3d_matrix(X, pack)
        print(
            f"[{MODEL_ID}] Yellow X  S1 tokens + vac-aug thz "
            f"({TS_N} Z × 7 = {N_AUG} at R={pack['R_th']:.3f} m)",
            flush=True,
        )

    # Temporal augmentation must precede the frame subsample: subsampling shuffles time order away.
    n_raw_feat = int(X.shape[1])
    X_raw = np.ascontiguousarray(X) if str(arch).lower() == "dualshot" else None
    X = augment_temporal(
        X, frame_shot_idx, temporal_window, temporal_mode, seq_window=seq_window,
    )
    print(
        f"[{MODEL_ID}] temporal window={temporal_window} mode={temporal_mode}  "
        f"seq_window={int(seq_window) or int(temporal_window)}  "
        f"features {n_raw_feat} -> {X.shape[1]}",
        flush=True,
    )

    n_all = len(X)
    if max_frames > 0 and n_all > max_frames:
        rng = np.random.default_rng(seed)
        pick = rng.choice(n_all, size=int(max_frames), replace=False)
        rng.shuffle(pick)
        X, S = X[pick], S[pick]
        if X_raw is not None:
            X_raw = X_raw[pick]
        if Y is not None:
            Y = Y[pick]
        if Z_pre is not None:
            Z_pre = Z_pre[pick]
        frame_shot_idx = frame_shot_idx[pick]
    n_total = len(X)

    init_ck = None
    if init_checkpoint is not None:
        init_path = Path(init_checkpoint)
        if not init_path.is_file():
            raise FileNotFoundError(f"init_checkpoint not found: {init_path}")
        init_ck = torch.load(init_path, map_location="cpu", weights_only=False)
        print(f"[{MODEL_ID}] warm-start from {init_path}", flush=True)

    mean = X.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = X.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    if (
        inherit_feature_stats
        and init_ck is not None
        and init_ck.get("feature_mean") is not None
        and init_ck.get("feature_std") is not None
    ):
        im = np.asarray(init_ck["feature_mean"], dtype=np.float32).reshape(-1)
        istd = np.asarray(init_ck["feature_std"], dtype=np.float32).reshape(-1)
        if im.shape == mean.shape and istd.shape == std.shape:
            mean = im
            std = istd
            print(
                f"[{MODEL_ID}] inherited feature mean/std from the warm-start checkpoint",
                flush=True,
            )
        else:
            raise RuntimeError(
                f"warm-start feature stats are {im.shape}, this run is {mean.shape}"
            )
    # dualseq warm-start: the first 3F columns ARE C7b's input. Inherit that checkpoint's
    # mean/std so gate_seq=0 is the C7b map, not a re-standardized cousin.
    if (
        init_ck is not None
        and str(arch).lower() in ("dualseq", "dualshot")
        and init_ck.get("feature_mean") is not None
    ):
        dual_w = 3 * n_raw_feat
        im = np.asarray(init_ck["feature_mean"], dtype=np.float32).reshape(-1)
        istd = np.asarray(init_ck["feature_std"], dtype=np.float32).reshape(-1)
        if im.size == dual_w and mean.size >= dual_w:
            mean[:dual_w] = im
            std[:dual_w] = istd
            print(
                f"[{MODEL_ID}] inherited init mean/std on dualctx block "
                f"({dual_w} of {mean.size} cols)",
                flush=True,
            )
    Xs = ((X - mean) / std).astype(np.float32)
    del X
    Xs = _clip_features(Xs, feature_clip, label="train")

    if coeff_targets:
        # The basis is frozen in the cache and shared by every run, which is what makes the M1 arms
        # comparable: a per-arm re-fit would confound "more data" with "different target axes".
        b_mean, b_comp, b_meta = load_coeff_basis(n_pca, include_thomson)
        n_comp = int(b_comp.shape[0])
        pca = PCA(n_components=n_comp)
        pca.mean_ = b_mean.astype(np.float64)
        pca.components_ = b_comp.astype(np.float64)
        pca.explained_variance_ratio_ = np.asarray(
            b_meta.get("explained_variance_ratio_sum", np.nan), dtype=np.float64
        ).repeat(n_comp) / n_comp
        Zc = np.ascontiguousarray(Z_pre[:, :n_comp], dtype=np.float32)
        del Z_pre
        Z_pre = None
        print(f"[{MODEL_ID}] frozen corpus basis k={n_comp} from the coefficient cache", flush=True)
    else:
        Y_flat = Y.reshape(n_total, -1).astype(np.float32)
        n_comp = int(min(n_pca, max(1, n_total - 1), Y_flat.shape[1]))
        pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=seed, iterated_power=3)
        if (
            init_ck is not None
            and init_ck.get("pca_components") is not None
            and int(init_ck.get("n_pca", 0)) == n_comp
        ):
            pca.mean_ = np.asarray(init_ck["pca_mean"], dtype=np.float64)
            pca.components_ = np.asarray(init_ck["pca_components"], dtype=np.float64)
            pca.explained_variance_ratio_ = np.ones(n_comp) / n_comp
            Zc = (
                (Y_flat - pca.mean_.astype(np.float32)) @ pca.components_.T.astype(np.float32)
            ).astype(np.float32)
            print(f"[{MODEL_ID}] reused PCA from init_checkpoint (n={n_comp})", flush=True)
        else:
            Zc = pca.fit_transform(Y_flat).astype(np.float32)
        del Y_flat
    print(
        f"[{MODEL_ID}] PCA n={n_comp}  explained="
        f"{float(np.asarray(pca.explained_variance_ratio_).sum()):.3f}  frames={n_total}  "
        f"hidden={hidden_t}  act={activation}  opt={optimizer}",
        flush=True,
    )

    i_q = EFIT_SCALAR_LABELS.index("efit_q95")
    i_b = EFIT_SCALAR_LABELS.index("efit_beta_n")
    q95 = np.nan_to_num(S[:, i_q], nan=0.0).astype(np.float32)
    beta = np.nan_to_num(S[:, i_b], nan=0.0).astype(np.float32)

    n_features = int(Xs.shape[1])
    if str(arch).lower() == "shotseq":
        from catskills.pinn_forest import train_shotseq_from_frames
        return train_shotseq_from_frames(
            Xs=Xs, Zc=Zc, q95=q95, beta=beta, frame_shot_idx=frame_shot_idx,
            mean=mean, std=std, pca=pca, n_raw_feat=n_raw_feat, n_comp=n_comp,
            device=device, out=output or Path(f"catskills_artifacts/{MODEL_ID}.pt"),
            hidden=int(seq_hidden), layers=int(seq_layers),
            bidirectional=bool(shot_bidirectional), mlp_blocks=int(shot_mlp_blocks),
            token_budget=int(token_budget) if int(token_budget) > 0 else 1024,
            dropout=float(trunk_dropout), activation=str(activation),
            epochs=int(epochs), lr=float(lr), weight_decay=float(weight_decay),
            optimizer=str(optimizer), lr_warmup=int(lr_warmup),
            grad_clip=float(grad_clip), feature_clip=float(feature_clip),
            seed=int(seed), init_seed=init_seed, val_fraction=float(val_fraction),
            early_stop_patience=int(early_stop_patience),
            log_every=int(log_every), ckpt_every=int(ckpt_every),
            lambda_qb=float(lambda_qb), qb_normalize=bool(qb_normalize),
            qb_emphasis=float(qb_emphasis),
            lambda_coeff_white=float(lambda_coeff_white),
            coeff_white_power=float(coeff_white_power),
            coeff_white_clamp=int(coeff_white_clamp),
            include_thomson=bool(include_thomson),
            cache_keys=cache_keys, cache_stats=cache_stats,
        )
    if str(arch).lower() == "dualshot":
        from catskills.pinn_forest import train_dualshot_from_frames
        if X_raw is None:
            raise RuntimeError("dualshot lost the raw feature matrix")
        raw_mean = X_raw.mean(axis=0, dtype=np.float64).astype(np.float32)
        raw_std = X_raw.std(axis=0, dtype=np.float64).astype(np.float32)
        raw_std = np.where(raw_std < 1e-6, 1.0, raw_std)
        Xs_raw = _clip_features(
            ((X_raw - raw_mean) / raw_std).astype(np.float32), feature_clip, label="raw",
        )
        return train_dualshot_from_frames(
            Xs_dual=Xs, Xs_raw=Xs_raw, Zc=Zc, q95=q95, beta=beta,
            frame_shot_idx=frame_shot_idx, mean=mean, std=std,
            raw_mean=raw_mean, raw_std=raw_std, pca=pca,
            n_raw_feat=n_raw_feat, n_comp=n_comp, device=device,
            out=output or Path(f"catskills_artifacts/{MODEL_ID}.pt"),
            hidden=int(seq_hidden), layers=int(seq_layers),
            bidirectional=bool(shot_bidirectional),
            token_budget=int(token_budget) if int(token_budget) > 0 else 1024,
            dropout=float(trunk_dropout), activation=str(activation),
            trunk_width=int(trunk_width), trunk_blocks=int(trunk_blocks),
            ctx_width=int(ctx_width), ctx_blocks=int(ctx_blocks),
            epochs=int(epochs), lr=float(lr), weight_decay=float(weight_decay),
            optimizer=str(optimizer), lr_warmup=int(lr_warmup),
            grad_clip=float(grad_clip), feature_clip=float(feature_clip),
            seed=int(seed), init_seed=init_seed, val_fraction=float(val_fraction),
            log_every=int(log_every), ckpt_every=int(ckpt_every),
            lambda_qb=float(lambda_qb), qb_normalize=bool(qb_normalize),
            lambda_coeff_white=float(lambda_coeff_white),
            coeff_white_power=float(coeff_white_power),
            coeff_white_clamp=int(coeff_white_clamp),
            include_thomson=bool(include_thomson),
            cache_keys=cache_keys, cache_stats=cache_stats,
            init_ck=init_ck, freeze_dualctx=bool(freeze_dualctx),
            temporal_window=int(temporal_window), temporal_mode=str(temporal_mode),
            lambda_geom=float(lambda_geom), geom_emphasis=float(geom_emphasis),
            geom_weights=str(geom_weights),
            geom_weight_target=float(geom_weight_target),
            geom_weight_floor=float(geom_weight_floor),
            geom_kappa_mult=float(geom_kappa_mult),
            geom_raxis_mult=float(geom_raxis_mult),
            geom_zaxis_mult=float(geom_zaxis_mult),
            geom_emphasis_final=float(geom_emphasis_final),
            geom_anneal_tail=float(geom_anneal_tail),
            geom_huber=float(geom_huber), geom_frac=float(geom_frac),
            geom_warmup=int(geom_warmup),
            geom_sd_chunk=int(geom_sd_chunk),
            edge_weight=str(edge_weight),
            edge_mix=float(edge_mix),
            edge_power=float(edge_power),
            edge_band_lo=float(edge_band_lo),
            edge_band_hi=float(edge_band_hi),
            probe_epochs=int(probe_epochs),
            tau_hi=float(tau_hi), tau_lo=float(tau_lo),
            tau_li_hi=float(tau_li_hi), tau_li_lo=float(tau_li_lo),
            anneal_frac=float(anneal_frac),
            resume=resume,
            geom_ramp=int(geom_ramp),
            gradnorm=bool(gradnorm),
            gradnorm_alpha=float(gradnorm_alpha),
            gradnorm_w_from=int(gradnorm_w_from),
            pcgrad=bool(pcgrad),
            stack_li=bool(stack_li),
            lp_floor=float(lp_floor),
            kill_r2psi=float(kill_r2psi),
            kill_r2psi_after=int(kill_r2psi_after),
            kill_cons=float(kill_cons),
            kill_cons_after=int(kill_cons_after),
            psi_collapse_r2psi=float(psi_collapse_r2psi),
            psi_collapse_streak=int(psi_collapse_streak),
        )
    # Wide stacks (dualseq W=9 is 11 GB of features) do not fit on a 12 GB card next to
    # zc + the 19M trunk. Host residency + per-batch H2D is the same math; only the device changes.
    feat_bytes = int(Xs.nbytes)
    free_b, _total_b = torch.cuda.mem_get_info()
    headroom = 3 * 1024 ** 3
    if feat_bytes + headroom > free_b:
        xs_t = torch.from_numpy(np.ascontiguousarray(Xs))
        try:
            xs_t = xs_t.pin_memory()
        except RuntimeError:
            pass
        print(
            f"[{MODEL_ID}] features on host ({feat_bytes / 1e9:.1f} GB); "
            f"GPU free {free_b / 1e9:.1f} GB — batches move per step",
            flush=True,
        )
    else:
        xs_t = torch.from_numpy(Xs).to(device)
    zc_t = torch.from_numpy(Zc).to(device)
    # The basis has to reach the device before the targets do on the coefficient path, so both live
    # here rather than beside the model.
    pca_mean = torch.from_numpy(pca.mean_.astype(np.float32)).to(device)
    pca_comp = torch.from_numpy(pca.components_.astype(np.float32)).to(device)
    if coeff_targets:
        y_t = CoeffTargets(zc_t, pca_comp, pca_mean)
    else:
        y_t = torch.from_numpy(Y.astype(np.float32)).to(device)

    # Per-mode divisor for the tempered coefficient term, from THIS run's targets rather than a
    # constant, so it stays correct at any k or frame budget. Built even when the term is off: it
    # costs one reduction and lets the run log what the weighting would have been.
    coeff_sd = zc_t.std(dim=0).clamp_min(1e-12)
    _clamp_i = min(max(int(coeff_white_clamp), 1), int(coeff_sd.numel())) - 1
    coeff_scale = coeff_sd.clamp_min(coeff_sd[_clamp_i]) ** float(coeff_white_power)
    if lambda_coeff_white > 0:
        # Relative weight each mode's squared error receives, normalized so mode 1 reads 1.0. This is
        # the number the experiment is actually about, so it goes in the log rather than staying implicit.
        w = (coeff_scale[0] / coeff_scale) ** 2
        print(
            f"[{MODEL_ID}] tempered coeff term: lambda={lambda_coeff_white:g} "
            f"power={coeff_white_power:g} clamp=mode{_clamp_i + 1}  "
            f"sd[1]={float(coeff_sd[0]):.4g} sd[{_clamp_i + 1}]={float(coeff_sd[_clamp_i]):.4g} "
            f"sd[{coeff_sd.numel()}]={float(coeff_sd[-1]):.4g}  "
            f"relative mode weight: 1 -> {float(w[0]):.1f}, "
            f"{_clamp_i + 1} -> {float(w[_clamp_i]):.1f}, "
            f"{coeff_sd.numel()} -> {float(w[-1]):.1f}",
            flush=True,
        )
    if edge_weight != "off" and edge_mix > 0:
        print(
            f"[{MODEL_ID}] edge weight: {edge_weight} mix={edge_mix:g} "
            f"power={edge_power:g} band=[{edge_band_lo:g},{edge_band_hi:g}]  "
            f"L_data = mean(w * err^2), w = (1-mix) + mix*(s/mean(s)), then mean-normalized "
            f"so lambda_coeff_white stays calibrated",
            flush=True,
        )
    q_t = torch.from_numpy(q95).to(device)
    b_t = torch.from_numpy(beta).to(device)
    del Xs, Zc, Y, S, q95, beta

    root = Path(__file__).resolve().parents[1]
    env = np.load(root / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    sd = SoftDerive(
        env["grid_R"], env["grid_Z"], env["mask_coarse"].astype(bool),
        cfg=SoftDeriveConfig(lp_floor=float(lp_floor)),
    ).to(device)

    # Put the geometry weight in score units. The smoke run measured L_geom = 0.320 against a
    # Consistency of 0.6575, confirming that the variance-normalized sum *is* the Consistency
    # deficit (1 - Cons); and L_data / Var(psi) is likewise (1 - R2_psi). So the S-proportional
    # loss is  W_PSI*(1-R2) + W_CONS*(1-Cons)  ->  L_data + (W_CONS/W_PSI)*Var(psi)*L_geom, and
    # geom_emphasis is how many times harder we push geometry than its raw score weight. We push
    # >1 on purpose: R2_psi has 0.0028 of headroom left and Consistency has 0.0585, so the marginal
    # return per unit of loss is wildly asymmetric. Deriving it from Var(psi) rather than fixing a
    # raw lambda is what makes one setting valid at both probe and scale.
    var_psi = float(y_t.var())  # CoeffTargets.var matches the grid tensor's, accumulated in chunks
    W_PSI, W_CONS, W_LCFS, W_QB = 0.55, 0.20, 0.10, 0.15

    # The scorer scores q95 and betaN as two SEPARATE pooled R2 and then means them, so they are
    # equally weighted there. Raw-MSE training is not: std(q95)/std(betaN) = 2.18 on the corpus, so
    # q95's squared error carries 4.75x the weight of betaN's. Normalising makes L_qb identically the
    # R2_qb deficit, which is what lets lambda_qb be derived in score units and carry between probe
    # and scale -- the same argument that made lambda_geom transferable.
    if qb_normalize:
        sig_q = float(q_t.std().clamp_min(1e-6))
        sig_b = float(b_t.std().clamp_min(1e-6))
        if lambda_qb < 0:
            lambda_qb = float(qb_emphasis * (W_QB / W_PSI) * var_psi)
        print(
            f"[{MODEL_ID}] qb normalized: sigma_q95={sig_q:.4g} sigma_betaN={sig_b:.4g}"
            f"  qb_emphasis={qb_emphasis:g}  ->  lambda_qb={lambda_qb:.4g}",
            flush=True,
        )
    else:
        sig_q = sig_b = 1.0
        if lambda_qb < 0:
            raise ValueError("--lambda-qb < 0 needs --qb-normalize to derive it")
    if lambda_geom < 0:
        lambda_geom = float(geom_emphasis * (W_CONS / W_PSI) * var_psi)
        print(
            f"[{MODEL_ID}] Var(psi)={var_psi:.4g}  geom_emphasis={geom_emphasis:g}"
            f"  ->  lambda_geom={lambda_geom:.4g}",
            flush=True,
        )
    # Same construction for D_LCFS, against its own score weight. The soft term is already in metric
    # units (it *is* D_LCFS), so lcfs_emphasis reads on the same "times its score weight" scale as
    # geom_emphasis -- but note the two terms differ in power: L_geom is a squared residual and this
    # is linear in the metric, so the two emphases are not interchangeable and each needs its own
    # calibration. The per-epoch loss report prints both magnitudes for exactly that reason.
    if lambda_lcfs < 0:
        lambda_lcfs = float(lcfs_emphasis * (W_LCFS / W_PSI) * var_psi)
        if lcfs_emphasis > 0:
            print(
                f"[{MODEL_ID}] lcfs_emphasis={lcfs_emphasis:g}  ->  "
                f"lambda_lcfs={lambda_lcfs:.4g}  (cap={lcfs_cap:g})",
                flush=True,
            )

    use_geom = lambda_geom > 0
    use_lcfs = lambda_lcfs > 0
    if kill_r2psi > 0:
        print(
            f"[{MODEL_ID}] kill_r2psi={float(kill_r2psi):g} after epoch "
            f"{int(kill_r2psi_after)}",
            flush=True,
        )

    geom_w = None
    if geom_weights == "deficit":
        geom_w = torch.tensor(
            derive_geom_weights(geom_weight_target, geom_weight_floor), device=device
        )
        if geom_kappa_mult != 1.0:
            ik = SCALAR_NAMES.index("kappa")
            geom_w = geom_w.clone()
            geom_w[ik] = geom_w[ik] * float(geom_kappa_mult)
            geom_w = geom_w / geom_w.mean()
        if geom_raxis_mult != 1.0:
            ir = SCALAR_NAMES.index("R_axis")
            geom_w = geom_w.clone()
            geom_w[ir] = geom_w[ir] * float(geom_raxis_mult)
            geom_w = geom_w / geom_w.mean()
        if geom_zaxis_mult != 1.0:
            iz = SCALAR_NAMES.index("Z_axis")
            geom_w = geom_w.clone()
            geom_w[iz] = geom_w[iz] * float(geom_zaxis_mult)
            geom_w = geom_w / geom_w.mean()
        print(
            f"[{MODEL_ID}] geometry weights (target {geom_weight_target:g}, "
            f"floor {geom_weight_floor:g}, kappa×{geom_kappa_mult:g}, "
            f"R_axis×{geom_raxis_mult:g}, Z_axis×{geom_zaxis_mult:g}): "
            + "  ".join(f"{n}={float(geom_w[j]):.3f}" for j, n in enumerate(SCALAR_NAMES)),
            flush=True,
        )
    elif geom_weights == "stack":
        from catskills.olive_loss import OLIVE_STACK_FOCUS
        focus = OLIVE_STACK_FOCUS if stack_li else None
        geom_w = torch.tensor(derive_stack_geom_weights(focus=focus), device=device)
    elif geom_weights == "placement":
        from catskills.olive_loss import PLACEMENT_STACK_FOCUS
        geom_w = torch.tensor(
            derive_stack_geom_weights(focus=PLACEMENT_STACK_FOCUS), device=device
        )
        label = "placement κ/R_axis/Z_axis"
        print(
            f"[{MODEL_ID}] geometry weights ({label}): "
            + "  ".join(f"{n}={float(geom_w[j]):.3f}" for j, n in enumerate(SCALAR_NAMES)),
            flush=True,
        )
    elif geom_weights == "placement_early":
        from catskills.olive_loss import PLACEMENT_EARLY_FLOOR, PLACEMENT_EARLY_FOCUS
        geom_w = torch.tensor(
            derive_stack_geom_weights(
                floor=PLACEMENT_EARLY_FLOOR, focus=PLACEMENT_EARLY_FOCUS
            ),
            device=device,
        )
        label = "placement_early κ/R/Z + ℓ_i floor"
        print(
            f"[{MODEL_ID}] geometry weights ({label}): "
            + "  ".join(f"{n}={float(geom_w[j]):.3f}" for j, n in enumerate(SCALAR_NAMES)),
            flush=True,
        )
    elif geom_weights == "explicit":
        if geom_weight_vec is None or len(geom_weight_vec) != len(SCALAR_NAMES):
            raise ValueError(
                f"explicit geom weights need {len(SCALAR_NAMES)} values "
                f"in {SCALAR_NAMES} order, got {geom_weight_vec!r}"
            )
        geom_w = torch.tensor(geom_weight_vec, dtype=torch.float32, device=device)
        print(
            f"[{MODEL_ID}] geometry weights (explicit, not renormalized): "
            + "  ".join(f"{n}={float(geom_w[j]):.3f}" for j, n in enumerate(SCALAR_NAMES)),
            flush=True,
        )
    elif geom_weights != "uniform":
        raise ValueError(
            f"geom_weights must be 'uniform', 'deficit', 'stack', 'placement', "
            f"'placement_early', or 'explicit', got {geom_weights!r}"
        )
    if use_geom:
        t_g = time.perf_counter()
        g_gt = _geom_gt(sd, y_t)
        ok_gt = _plausible_mask(g_gt)
        # sigma_j is the scorer's SS_tot denominator: corpus std of the *reference* scalar.
        sigma = torch.stack(
            [
                g_gt[ok_gt[:, j], j].std() if int(ok_gt[:, j].sum()) > 1 else torch.ones((), device=device)
                for j in range(N_CONS)
            ]
        ).clamp_min(1e-6)
        print(
            f"[{MODEL_ID}] geometry targets in {time.perf_counter() - t_g:.1f}s  "
            + "  ".join(
                f"{n}:sigma={float(sigma[j]):.3g},ok={float(ok_gt[:, j].float().mean()):.3f}"
                for j, n in enumerate(SCALAR_NAMES)
            ),
            flush=True,
        )
    else:
        g_gt = ok_gt = sigma = None

    n_nbi = int(os.environ.get("CRC_FEQ_NBI_COLS", "0") or 0)
    model = build_trunk(
        arch, n_features, n_comp, hidden=hidden_t, activation=activation,
        trunk_width=trunk_width, trunk_blocks=trunk_blocks, trunk_dropout=trunk_dropout,
        ctx_width=ctx_width, ctx_blocks=ctx_blocks, n_split=n_raw_feat,
        seq_hidden=seq_hidden, seq_layers=seq_layers, n_nbi=n_nbi,
    ).to(device)
    print(
        f"[{MODEL_ID}] trunk={arch} params={sum(p.numel() for p in model.parameters()):,}"
        + (f" width={trunk_width} blocks={trunk_blocks} dropout={trunk_dropout}"
           if arch in ("resnet", "dualctx", "grand") else "")
        + (f" hidden={list(hidden_t)}" if arch in ("", "mlp") else "")
        + (f" ctx={ctx_width}x{ctx_blocks} split={n_raw_feat} gate=0(zero-init)"
           if arch in ("dualctx", "grand") else "")
        + (f" qb_mlp=3 n_nbi={n_nbi}" if arch == "grand" else "")
        + (f" gru={seq_hidden}x{seq_layers} window={n_features // max(n_raw_feat, 1)} "
           f"split={n_raw_feat}" if arch == "seq" else "")
        + (f" dualseq gru={seq_hidden}x{seq_layers} seq_window="
           f"{int(seq_window) or int(temporal_window)} split={n_raw_feat} gate_seq=0"
           if arch == "dualseq" else ""),
        flush=True,
    )
    if init_ck is not None and init_ck.get("model_state") is not None:
        src = init_ck["model_state"]
        cur = model.state_dict()
        filtered = {}
        skipped_shape = []
        for k, v in src.items():
            if k not in cur:
                continue
            if tuple(v.shape) != tuple(cur[k].shape):
                skipped_shape.append(k)
                continue
            filtered[k] = v
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if skipped_shape:
            print(
                f"[{MODEL_ID}] init skipped shape-mismatch keys {skipped_shape[:8]}",
                flush=True,
            )
        if missing:
            print(f"[{MODEL_ID}] init missing keys {list(missing)[:6]}…", flush=True)
        if unexpected:
            print(f"[{MODEL_ID}] init unexpected keys {list(unexpected)[:6]}…", flush=True)
        if str(arch).lower() == "dualseq" and hasattr(model, "gate_seq"):
            model.gate_seq.data.zero_()
            print(f"[{MODEL_ID}] dualseq gate_seq reset to 0 after warm-start", flush=True)
    if freeze_dualctx:
        keep = {"gru", "seq_proj", "gate_seq"}
        for name, p in model.named_parameters():
            if name.split(".", 1)[0] not in keep:
                p.requires_grad = False
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        print(f"[{MODEL_ID}] freeze-dualctx: training {n_tr:,} / {n_all:,} params", flush=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters after freeze-dualctx")
    opt = make_optimizer(
        trainable, name=optimizer, lr=lr, weight_decay=weight_decay, momentum=momentum
    )
    w_g = torch.nn.Parameter(torch.ones((), device=device))
    opt_w = torch.optim.Adam([w_g], lr=1e-3)
    _gn_l0: tuple[float, float] | None = None
    val_cons_warmup: float | None = None
    if gradnorm:
        print(
            f"[{MODEL_ID}] GradNorm α={float(gradnorm_alpha):g} on h_fuse; "
            f"w_g Adam from epoch {int(gradnorm_w_from)}; ramp={int(geom_ramp)}",
            flush=True,
        )
    if pcgrad:
        print(f"[{MODEL_ID}] PCGrad on (L_data, L_stack)", flush=True)
    if float(lp_floor) > 0:
        print(f"[{MODEL_ID}] SoftDerive ℓ_p floor={float(lp_floor):g}", flush=True)
    if float(psi_collapse_r2psi) > 0:
        print(
            f"[{MODEL_ID}] post-gate ψ collapse abort: val_R2ψ < "
            f"{float(psi_collapse_r2psi):g} for {int(psi_collapse_streak)} consecutive epochs",
            flush=True,
        )
    adapt_end: list[int | None] = [None]
    psi_collapse_run: list[int] = [0]
    if adaptive_warmup:
        print(
            f"[{MODEL_ID}] adaptive warmup: data-only until "
            f"val_R2ψ>={float(warmup_r2psi):g} and val_Cons>={float(warmup_cons):g}; "
            f"cap={int(warmup_cap)}; then ramp={int(geom_ramp)}",
            flush=True,
        )
    # R5 (section 11.8) diverged at width 1024 x 8 blocks: loss 0.089 at epoch 5 to 7.9e6 at epoch 20.
    # A 17M-parameter residual trunk starts at full lr with random skip paths, and the first epochs are
    # where it blows up, so the schedule ramps in linearly before the cosine takes over. warmup=0
    # reproduces the plain cosine every earlier run used.
    if lr_warmup > 0:
        w = min(int(lr_warmup), max(epochs - 1, 1))

        def _lr_at(ep: int) -> float:
            if ep < w:
                return (ep + 1) / float(w + 1)
            frac = (ep - w) / max(epochs - w, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(frac, 1.0)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_at)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(epochs, 1), eta_min=float(lr_eta_min)
        )

    axis_sign = -1.0

    def _geom_scale(ep: int) -> float:
        """P6c. Multiplier on both geometry weights, 1.0 until the tail then decaying geometrically.

        The premise is that boundary *structure* is learned early while pixel accuracy is a late
        refinement, so charcoal's 0.0042 R2_psi regression against teal (section 15.2) may be the
        price of holding full geometry emphasis all the way into the L-BFGS polish. Decaying it over
        the tail lets the data term finish. P4 selected epoch 301 of 420, so the tail exists.
        """
        from catskills.olive_loss import geom_ramp_mult
        warm = (
            int(adapt_end[0]) if adaptive_warmup and adapt_end[0] is not None
            else (10**9 if adaptive_warmup else int(geom_warmup))
        )
        ramp = geom_ramp_mult(ep, warm, int(geom_ramp))
        if geom_emphasis_final >= 1.0 or geom_anneal_tail <= 0 or epochs <= 1:
            return ramp
        start = (1.0 - geom_anneal_tail) * epochs
        if ep <= start:
            return ramp
        f = min(1.0, (ep - start) / max(1.0, geom_anneal_tail * epochs))
        return ramp * float(geom_emphasis_final ** f)

    def _taus(ep: int) -> tuple[float, float]:
        """Anneal the mask widths from wide (gradients everywhere) to the P1-tuned values."""
        if epochs <= 1 or anneal_frac <= 0:
            return tau_lo, tau_li_lo
        f = min(1.0, (ep - 1) / max(1.0, anneal_frac * epochs))
        return (
            float(tau_hi * (tau_lo / tau_hi) ** f),
            float(tau_li_hi * (tau_li_lo / tau_li_hi) ** f),
        )

    def _grad_rz(p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gz = torch.zeros_like(p)
        gz[:, 1:-1, :] = (p[:, 2:, :] - p[:, :-2, :]) * 0.5
        gr = torch.zeros_like(p)
        gr[:, :, 1:-1] = (p[:, :, 2:] - p[:, :, :-2]) * 0.5
        return gz, gr

    def _lcfs_loss(psi_hat: torch.Tensor, y_ref: torch.Tensor) -> torch.Tensor:
        """Mean soft D_LCFS over the frames where both boundaries could be found.

        Safe here only because ``psi_hat`` is a PCA reconstruction: section 15.9 measured that the
        same gradient applied to unconstrained pixels drives this term down 2x further while moving
        the scorer's D_LCFS by 0.4%, i.e. the replica has exploitable slack and the k=60 basis is
        what closes it off.
        """
        d, v = sd.soft_dlcfs(psi_hat, y_ref, cap=lcfs_cap)
        vf = v.to(d.dtype)
        return (d * vf).sum() / vf.sum().clamp_min(1.0)

    def _batch_loss(xb, zb, yb, qb, bb, gb, okb, tau, tau_li, want_geom: bool,
                    gscale: float = 1.0, precomputed=None, idx=None):
        if precomputed is None:
            z_hat, q_hat, b_hat, *_ = model(xb)
        else:
            z_hat, q_hat, b_hat, *_ = precomputed
        psi = (z_hat @ pca_comp + pca_mean).view(-1, 65, 65)

        err2 = (psi - yb) ** 2
        if edge_weight != "off" and edge_mix > 0:
            with torch.no_grad():
                if edge_weight == "grad":
                    gz_w, gr_w = _grad_rz(yb)
                    s = (gz_w.square() + gr_w.square()).sqrt().clamp_min(0.0)
                    s = s.pow(float(edge_power))
                    s = s / s.mean().clamp_min(1e-12)
                elif edge_weight == "band":
                    psi_min_w = yb.amin(dim=(1, 2), keepdim=True)
                    psi_max_w = yb.amax(dim=(1, 2), keepdim=True)
                    psi_n_w = (yb - psi_min_w) / (psi_max_w - psi_min_w).clamp_min(1e-9)
                    s = ((psi_n_w >= float(edge_band_lo))
                         & (psi_n_w <= float(edge_band_hi))).to(psi.dtype)
                    s = s / s.mean().clamp_min(1e-12)
                else:
                    raise ValueError(f"edge_weight={edge_weight!r}; expected off|grad|band")
                w_edge = (1.0 - float(edge_mix)) + float(edge_mix) * s
                w_edge = w_edge / w_edge.mean().clamp_min(1e-12)
            L_data = (err2 * w_edge).mean()
        else:
            L_data = err2.mean()
        L_coeff = ((z_hat - zb) ** 2).mean()
        if lambda_coeff_white > 0:
            L_coeff_w = (((z_hat - zb) / coeff_scale) ** 2).mean()
        else:
            L_coeff_w = torch.zeros((), device=device)
        if qb_normalize:
            L_qb = 0.5 * (((q_hat - qb) / sig_q) ** 2).mean() + \
                   0.5 * (((b_hat - bb) / sig_b) ** 2).mean()
        else:
            L_qb = ((q_hat - qb) ** 2).mean() + ((b_hat - bb) ** 2).mean()

        gz_h, gr_h = _grad_rz(psi)
        gz_g, gr_g = _grad_rz(yb)
        gmag2 = (gz_g ** 2 + gr_g ** 2).detach()
        # Dimensionless: fraction of GT gradient energy we get wrong, so lambda_h1 is a priority
        # rather than a unit conversion.
        L_h1 = ((gz_h - gz_g) ** 2 + (gr_h - gr_g) ** 2).mean() / gmag2.mean().clamp_min(1e-12)

        # Normal displacement: L2 reweighted by 1/|grad psi| inside the near-separatrix band. This
        # is the linearization of "move the contour to the right place" and puts weight exactly
        # where kappa and the triangularities are decided.
        with torch.no_grad():
            psi_min = yb.amin(dim=(1, 2), keepdim=True)
            psi_max = yb.amax(dim=(1, 2), keepdim=True)
            psi_n = (yb - psi_min) / (psi_max - psi_min).clamp_min(1e-9)
            if axis_sign < 0:
                pass  # axis at the minimum: psi_n = 0 at axis, 1 at edge
            band = ((psi_n >= band_lo) & (psi_n <= band_hi)).to(psi.dtype)
            wnorm = band / (gmag2.sqrt() + 1e-3)
            wsum = wnorm.sum().clamp_min(1.0)
        L_disp = ((psi - yb) ** 2 * wnorm).sum() / wsum

        L_geom = torch.zeros((), device=device)
        L_lcfs = torch.zeros((), device=device)
        if want_geom and gb is not None:
            g = sd(psi, tau=tau, tau_li=tau_li)
            g_hat = torch.stack([g[k] for k in SCALAR_NAMES], dim=1)
            L_geom = _geom_loss(g_hat, gb, okb, sigma, geom_huber, geom_w)
            if use_lcfs:
                L_lcfs = _lcfs_loss(psi, yb)

        loss = (
            L_data
            + 0.25 * L_coeff
            + lambda_coeff_white * L_coeff_w
            + lambda_qb * L_qb
            + lambda_h1 * L_h1
            + lambda_disp * L_disp
            + gscale * lambda_geom * L_geom
            + gscale * lambda_lcfs * L_lcfs
        )
        parts = {
            "L_data": L_data,
            "L_coeff": L_coeff,
            "L_coeff_w": L_coeff_w,
            "L_qb": L_qb,
            "L_h1": L_h1,
            "L_disp": L_disp,
            "L_geom": L_geom,
            "L_lcfs": L_lcfs,
        }
        if (
            want_geom
            and idx is not None
            and os.environ.get("CRC_FEQ_DIFF_CONS", "").strip().lower() in ("1", "true", "yes", "on")
        ):
            from catskills.scripts.o1g_diff_consistency_train import diff_cons_penalty

            extra, extra_parts = diff_cons_penalty(
                psi, yb, gb, okb, xb, idx, float(mean[20]), float(std[20]),
            )
            loss = loss + extra
            parts.update(extra_parts)
        return loss, L_data, parts

    # Crash insurance. P5 seed 101 segfaulted inside libc10 at epoch ~370 of 420 after 5h08m and left
    # nothing behind, because the only torch.save is at the very end. The fault was not ours to fix
    # (no OOM, 27 GB free, and the identical command succeeded on the next seed), so the answer is to
    # stop making a crash expensive: every ``ckpt_every`` epochs the best-so-far state is written in the
    # *same* format as the final checkpoint, so load_charcoal, the report script and the packager all
    # read it with no new code. Marked partial=True so a partial can never be mistaken for a finished
    # run. Written to a temp path and renamed, so a crash mid-write cannot corrupt the previous one.
    #
    # ``partial_path`` is bound below, after ``out`` exists; _save_partial reads it at call time.

    def _static_ckpt() -> dict:
        return {
            "residual_state": None,
            "n_features": n_features,
            "n_raw_features": n_raw_feat,
            "feature_mean": mean,
            "feature_std": std,
            "include_thomson": include_thomson,
            "thomson_layout": __import__("os").environ.get("CRC_FEQ_THOMSON_LAYOUT", "").strip().lower(),
            "n_pca": int(n_comp),
            "use_residual": False,
            "temporal_window": int(temporal_window),
            "temporal_mode": str(temporal_mode),
            "hidden": list(hidden_t),
            "activation": activation,
            "arch": arch,
            "arch_config": {"width": int(trunk_width), "blocks": int(trunk_blocks),
                            "dropout": float(trunk_dropout),
                            "ctx_width": int(ctx_width), "ctx_blocks": int(ctx_blocks),
                            "seq_hidden": int(seq_hidden), "seq_layers": int(seq_layers),
                            "seq_window": int(seq_window) or int(temporal_window),
                            "n_split": int(n_raw_feat), "n_nbi": int(n_nbi)},
            "feature_clip": float(feature_clip),
            "pca_mean": pca.mean_.astype(np.float32),
            "pca_components": pca.components_.astype(np.float32),
            "method": MODEL_ID,
            "model_version": make_model_version(MODEL_ID),
            "git_commit": git_full_sha(),
        }

    def _clone_state() -> dict:
        return {k2: v.detach().cpu().clone() for k2, v in model.state_dict().items()}

    def _save_partial(
        ep: int,
        state: dict | None,
        sel_ep: int | None,
        sel: float,
        live_state: dict | None = None,
    ) -> None:
        if state is None:
            return
        try:
            tmp = partial_path.with_suffix(".tmp")
            torch.save(
                {
                    **_static_ckpt(),
                    "model_state": state,
                    "history": list(history),
                    # P1 lost 13 h because "checkpointed" was not "restartable". ``model_state`` above
                    # is best-so-far in the finished-model format, which is what makes a partial
                    # scoreable -- and it is the wrong thing to resume from. Resuming needs the LIVE
                    # weights, AdamW's moment buffers, where the cosine had got to, and the batch-order
                    # RNG; without those, "resume" silently means "warm-start a different run with the
                    # LR back at peak and the moments at zero". These four keys are the difference.
                    "resume": {
                        "epoch": int(ep),
                        "live_state": live_state if live_state is not None else _clone_state(),
                        "opt_state": opt.state_dict(),
                        "sched_state": sched.state_dict(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": (torch.cuda.get_rng_state(device)
                                     if device.type == "cuda" else None),
                        "best_selection": float(sel),
                        "best_epoch": int(sel_ep or ep),
                        "best_state": state,
                        "stall": int(stall),
                        # Anything that would make a resumed run incomparable to the one it continues.
                        "guard": {"seed": int(seed), "epochs": int(epochs),
                                  "batch_size": int(batch_size), "n_pca": int(n_comp),
                                  "arch": arch, "trunk_width": int(trunk_width),
                                  "trunk_blocks": int(trunk_blocks),
                                  "ctx_width": int(ctx_width),
                                  "ctx_blocks": int(ctx_blocks),
                                  "seq_hidden": int(seq_hidden),
                                  "seq_layers": int(seq_layers),
                                  "n_features": int(n_features)},
                    },
                    "training_params": {
                        "partial": True,
                        "epochs": int(epochs),
                        "epochs_ran": int(ep),
                        "best_epoch": int(sel_ep or ep),
                        "best_selection": float(sel),
                        "n_pca": int(n_comp),
                        "seed": int(seed),
                        "init_seed": int(seed if init_seed is None else init_seed),
                        "elapsed_s": time.perf_counter() - t0,
                    },
                },
                tmp,
            )
            tmp.replace(partial_path)
            print(f"[{MODEL_ID}] partial checkpoint @ ep {ep} -> {partial_path.name}", flush=True)
        except Exception as exc:  # never let insurance take the run down
            print(f"[{MODEL_ID}] partial checkpoint failed: {type(exc).__name__}: {exc}", flush=True)

    t0 = time.perf_counter()
    history: list[dict] = []
    log_every = max(1, int(log_every))
    use_tqdm = sys.stderr.isatty() and epochs <= 60
    out = output or Path(f"catskills_artifacts/{MODEL_ID}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    partial_path = out.with_name(out.stem + ".partial.pt")

    # Held-out shots, not held-out frames. psi(t) is smooth on the EFIT timebase, so a frame-wise
    # split leaves both temporal neighbours of every val frame in the training set — val Consistency
    # then reads ~0.94 against ~0.50 on unseen shots, and selection on the S proxy chases the leak.
    val_fraction = float(max(0.0, min(0.5, val_fraction)))
    val_idx = None
    if val_fraction > 0:
        uniq = np.unique(frame_shot_idx)
        if uniq.size >= 2:
            order = np.random.default_rng(seed).permutation(uniq)
            target = n_total * val_fraction
            counts = {int(s): int((frame_shot_idx == s).sum()) for s in uniq}
            held: list[int] = []
            taken = 0
            for s in order:
                if taken >= target or n_total - taken - counts[int(s)] < batch_size:
                    break
                held.append(int(s))
                taken += counts[int(s)]
            if held and n_total - taken >= batch_size:
                mask = np.isin(frame_shot_idx, held)
                val_idx = torch.from_numpy(np.flatnonzero(mask)).to(device)
                train_idx_pool = torch.from_numpy(np.flatnonzero(~mask)).to(device)
                print(
                    f"[{MODEL_ID}] val split: {len(held)}/{uniq.size} shots, "
                    f"{int(val_idx.numel())}/{n_total} frames held out",
                    flush=True,
                )
    if val_idx is None:
        train_idx_pool = torch.arange(n_total, device=device)
    n_train_frames = int(train_idx_pool.numel())

    if os.environ.get("CRC_FEQ_DIFF_CONS", "").strip().lower() in ("1", "true", "yes", "on"):
        from catskills.scripts.o1g_diff_consistency_train import prepare_diff_cons

        try:
            prepare_diff_cons(cache_keys, frame_shot_idx, n_total)
        except Exception as exc:
            print(f"[{MODEL_ID}] diff cons prepare failed: {exc}; terms stay off", flush=True)

    # Intra-regime mixup. Off unless CRC_FEQ_LIMITED_MIXUP=1, so every other
    # charcoal run is unchanged. Partners are training frames with kappa < 1.45
    # from a different shot. Default geometry targets are soft_derive of the
    # mixed psi. CRC_FEQ_STRATIFIED_BATCH instead draws a 90/10 batch and
    # linearly mixes the geometry targets of the two limited frames.
    limited_pool = None
    shot_of = None
    limited_flags = None
    if os.environ.get("CRC_FEQ_LIMITED_MIXUP", "").strip().lower() in (
        "1", "true", "yes", "on",
    ) and cache_keys is not None:
        from fusion_scoring.common import CONS_SCALARS
        from catskills.score_ref_cache import load_reference

        kap_i = CONS_SCALARS.index("kappa")
        flags = np.zeros(n_total, dtype=bool)
        cuts = np.flatnonzero(np.diff(frame_shot_idx)) + 1
        bounds = np.r_[0, cuts, n_total]
        n_miss = 0
        n_shots_lim = 0
        if len(bounds) == len(cache_keys) + 1:
            for s, key in enumerate(cache_keys):
                ref = load_reference(key)
                if ref is None:
                    n_miss += 1
                    continue
                cons = np.asarray(ref[1])
                a, b = int(bounds[s]), int(bounds[s + 1])
                n = min(b - a, len(cons))
                kap = cons[:n, kap_i]
                hit = np.isfinite(kap) & (kap < 1.45)
                flags[a : a + n] = hit
                n_shots_lim += int(hit.any())
                if (s + 1) % 1000 == 0:
                    print(f"[{MODEL_ID}] limited index {s + 1}/{len(cache_keys)}", flush=True)
        train_np = train_idx_pool.detach().cpu().numpy()
        lim_train = train_np[flags[train_np]]
        limited_pool = torch.from_numpy(np.ascontiguousarray(lim_train, dtype=np.int64)).to(device)
        shot_of = torch.from_numpy(np.ascontiguousarray(frame_shot_idx, dtype=np.int64)).to(device)
        limited_flags = torch.from_numpy(flags).to(device)
        mix_p_print = os.environ.get("CRC_FEQ_LIMITED_MIXUP_P", "0.5")
        geom_mix = (
            "linear y_geom"
            if os.environ.get("CRC_FEQ_STRATIFIED_BATCH", "").strip().lower()
            in ("1", "true", "yes", "on")
            else "soft_derive(mixed psi)"
        )
        print(
            f"[{MODEL_ID}] limited mixup: {int(flags.sum())} frames, "
            f"{n_shots_lim} shots with kappa<1.45, train pool {int(limited_pool.numel())}, "
            f"missing refs {n_miss}, p={mix_p_print} Beta(2,2), geom={geom_mix}",
            flush=True,
        )
        if int(limited_pool.numel()) < 2:
            limited_pool = None
            print(f"[{MODEL_ID}] limited mixup disabled: pool shorter than 2", flush=True)

    # Geometry targets for the SELECTOR when the geometry loss is off. Scattered into a full-length
    # NaN-filled tensor so `_gather` keeps indexing by global frame index and nothing downstream
    # changes shape; only the sampled validation rows are finite, and `_plausible_mask` rejects the
    # rest, so a training batch can never pick up a target from here. It cannot anyway -- `want_geom`
    # is False for the whole run when `use_geom` is False -- but relying on two independent guards is
    # cheaper than relying on one.
    #
    # Sampled rather than exhaustive: soft_derive over every held-out frame would run each epoch, and
    # a 7-scalar R2 over 20k frames spanning all held-out shots is already tight enough to rank
    # epochs. The sample is drawn once, with a fixed generator, so the selector compares like with
    # like across epochs.
    if (not use_geom) and val_geom_frames > 0 and val_idx is not None and val_idx.numel() > 0:
        t_g = time.perf_counter()
        n_val = int(val_idx.numel())
        if n_val > val_geom_frames:
            g_cpu = torch.Generator(device="cpu").manual_seed(seed)
            pick = torch.randperm(n_val, generator=g_cpu)[:val_geom_frames].to(device)
            sel_idx = val_idx[pick]
        else:
            sel_idx = val_idx
        g_gt = torch.full((n_total, N_CONS), float("nan"), device=device)
        with torch.no_grad():
            for start in range(0, int(sel_idx.numel()), 512):
                sl = sel_idx[start : start + 512]
                g = sd(y_t[sl])
                g_gt[sl] = torch.stack([g[k] for k in SCALAR_NAMES], dim=1)
        ok_gt = _plausible_mask(g_gt)
        # Restrict the selector's val set to the sampled frames, so `_validate` does not pay for
        # soft_derive on frames whose targets are NaN and would be masked out regardless.
        val_idx = sel_idx.sort().values
        print(
            f"[{MODEL_ID}] selector geometry: {int(sel_idx.numel()):,} of {n_val:,} val frames in "
            f"{time.perf_counter() - t_g:.1f}s -> selection sees Consistency, not psi MSE alone",
            flush=True,
        )

    best_sel = -float("inf")
    best_state: dict | None = None
    best_epoch = 0
    stall = 0
    epochs_ran = 0
    start_epoch = 1

    if resume is not None:
        rp = Path(resume)
        if not rp.is_file():
            raise FileNotFoundError(f"--resume {rp} does not exist")
        rck = torch.load(rp, map_location=device, weights_only=False)
        rs = rck.get("resume")
        if not rs:
            raise ValueError(
                f"{rp.name} has no resume state: it was written before resume support existed, so it "
                f"can be SCORED but not continued. Start the run again."
            )
        want = {"seed": int(seed), "epochs": int(epochs), "batch_size": int(batch_size),
                "n_pca": int(n_comp), "arch": arch, "trunk_width": int(trunk_width),
                "trunk_blocks": int(trunk_blocks), "ctx_width": int(ctx_width),
                "ctx_blocks": int(ctx_blocks), "seq_hidden": int(seq_hidden),
                "seq_layers": int(seq_layers),
                "seq_window": int(seq_window) or int(temporal_window),
                "n_features": int(n_features)}
        bad = {k2: (v, want[k2]) for k2, v in (rs.get("guard") or {}).items()
               if k2 in want and v != want[k2]}
        if bad:
            raise ValueError(
                f"refusing to resume {rp.name}: " + ", ".join(
                    f"{k2} was {a} now {b}" for k2, (a, b) in bad.items())
                + ". A resume that changes the split or the schedule is a new run wearing an old name."
            )
        model.load_state_dict(rs["live_state"])
        opt.load_state_dict(rs["opt_state"])
        sched.load_state_dict(rs["sched_state"])
        torch.set_rng_state(rs["torch_rng"].cpu() if hasattr(rs["torch_rng"], "cpu")
                            else rs["torch_rng"])
        if rs.get("cuda_rng") is not None and device.type == "cuda":
            torch.cuda.set_rng_state(rs["cuda_rng"].cpu(), device)
        history = list(rck.get("history") or [])
        # Patience 0 used to leave best_selection at -inf while still restoring best_state.
        # That snapshot is the last partial's weights, not a scored winner. Ignore it.
        try:
            resumed_sel = float(rs.get("best_selection", -float("inf")))
        except (TypeError, ValueError):
            resumed_sel = -float("inf")
        if math.isfinite(resumed_sel) and rs.get("best_state"):
            best_sel = resumed_sel
            best_epoch = int(rs.get("best_epoch", 0))
            best_state = rs.get("best_state")
        else:
            best_sel = -float("inf")
            best_epoch = 0
            best_state = None
            print(
                f"[{MODEL_ID}] resume best_selection was not finite; "
                "selector starts unset and will not restore a warmup snapshot",
                flush=True,
            )
        stall = int(rs.get("stall", 0))
        start_epoch = int(rs["epoch"]) + 1
        epochs_ran = int(rs["epoch"])
        adopted = ""
        if rp.name.endswith(".partial.pt"):
            best_file = rp.with_name(rp.name.replace(".partial.pt", "_best.pt"))
            if best_file.is_file():
                bck = torch.load(best_file, map_location="cpu", weights_only=False)
                btp = bck.get("training_params") or {}
                try:
                    file_sel = float(btp.get("best_selection", float("nan")))
                except (TypeError, ValueError):
                    file_sel = float("nan")
                file_state = bck.get("model_state")
                if (
                    math.isfinite(file_sel)
                    and isinstance(file_state, dict)
                    and file_state
                    and (not math.isfinite(best_sel) or file_sel > best_sel)
                ):
                    best_sel = file_sel
                    best_epoch = int(btp.get("best_epoch") or btp.get("epochs_ran") or best_epoch)
                    best_state = {
                        k2: v.detach().cpu().clone() if torch.is_tensor(v) else v
                        for k2, v in file_state.items()
                    }
                    adopted = f"  adopted {best_file.name}"
        print(
            f"[{MODEL_ID}] RESUMED {rp.name} at epoch {rs['epoch']} -> continuing from "
            f"{start_epoch}/{epochs}  lr={opt.param_groups[0]['lr']:.3e}  "
            f"best selection {best_sel:.6f} @ ep {best_epoch}  stall {stall}{adopted}",
            flush=True,
        )
        if start_epoch > epochs:
            print(f"[{MODEL_ID}] nothing to do: {rs['epoch']} of {epochs} epochs already ran",
                  flush=True)

    def _gather(idx):
        gb = g_gt[idx] if g_gt is not None else None
        okb = ok_gt[idx] if ok_gt is not None else None
        if xs_t.device.type == "cpu":
            xb = xs_t[idx.detach().to("cpu")].to(device, non_blocking=True)
        else:
            xb = xs_t[idx]
        return xb, zc_t[idx], y_t[idx], q_t[idx], b_t[idx], gb, okb

    @torch.no_grad()
    def _validate(tau: float, tau_li: float) -> tuple[float, float, list[float], float]:
        """Returns (val psi MSE, val Consistency proxy, per-scalar R2, val soft D_LCFS)."""
        if val_idx is None or val_idx.numel() == 0:
            return float("nan"), float("nan"), [], float("nan")
        model.eval()
        tot, n = 0.0, 0
        gh, gr_, okv = [], [], []
        dl_sum, dl_n = 0.0, 0.0
        for start in range(0, int(val_idx.numel()), batch_size):
            sl = val_idx[start : start + batch_size]
            xb, zb, yb, qb, bb, gb, okb = _gather(sl)
            z_hat = model(xb)[0]
            psi = (z_hat @ pca_comp + pca_mean).view(-1, 65, 65)
            tot += float(((psi - yb) ** 2).mean()) * int(sl.numel())
            n += int(sl.numel())
            if gb is not None:
                g = sd(psi, tau=tau, tau_li=tau_li)
                gh.append(torch.stack([g[k] for k in SCALAR_NAMES], dim=1))
                gr_.append(gb)
                okv.append(okb)
            if use_lcfs:
                # cap 1.0 here, not lcfs_cap: this is a *readout* meant to be comparable with the
                # scorer's own clamp, not the training penalty.
                d, v = sd.soft_dlcfs(psi, yb, cap=1.0)
                vf = v.to(d.dtype)
                dl_sum += float((d * vf).sum())
                dl_n += float(vf.sum())
        model.train()
        dl = dl_sum / dl_n if dl_n > 0 else float("nan")
        if not gh:
            return tot / max(n, 1), float("nan"), [], dl
        cons, per = _cons_proxy(torch.cat(gh), torch.cat(gr_), torch.cat(okv))
        return tot / max(n, 1), cons, per, dl

    def _mix_limited(idx, xb, zb, yb, qb, bb, gb, okb, tau, tau_li):
        if limited_pool is None or shot_of is None or limited_flags is None:
            return xb, zb, yb, qb, bb, gb, okb
        mix_p = float(os.environ.get("CRC_FEQ_LIMITED_MIXUP_P", "0.5"))
        take = torch.nonzero(
            limited_flags[idx] & (torch.rand(idx.shape[0], device=device) < mix_p),
            as_tuple=False,
        ).flatten()
        if take.numel() == 0:
            return xb, zb, yb, qb, bb, gb, okb
        n = int(take.numel())
        partner = limited_pool[torch.randint(0, int(limited_pool.numel()), (n,), device=device)]
        src_shot = shot_of[idx[take]]
        for _ in range(8):
            bad = shot_of[partner] == src_shot
            if not bool(bad.any()):
                break
            redraw = torch.randint(0, int(limited_pool.numel()), (int(bad.sum()),), device=device)
            partner = partner.clone()
            partner[bad] = limited_pool[redraw]
        keep = shot_of[partner] != src_shot
        if not bool(keep.any()):
            return xb, zb, yb, qb, bb, gb, okb
        take, partner = take[keep], partner[keep]
        linear_geom = os.environ.get("CRC_FEQ_STRATIFIED_BATCH", "").strip().lower() in (
            "1", "true", "yes", "on",
        )
        xb_b, zb_b, yb_b, qb_b, bb_b, gb_b, ok_b = _gather(partner)
        lam = torch.distributions.Beta(
            torch.tensor(2.0, device=device), torch.tensor(2.0, device=device),
        ).sample((int(take.numel()), 1))
        one = 1.0 - lam
        xb, zb, qb, bb = xb.clone(), zb.clone(), qb.clone(), bb.clone()
        yb = yb.clone()
        xb[take] = lam * xb[take] + one * xb_b
        zb[take] = lam * zb[take] + one * zb_b
        yb[take] = lam.view(-1, 1, 1) * yb[take] + one.view(-1, 1, 1) * yb_b
        qb[take] = lam.view(-1) * qb[take] + one.view(-1) * qb_b
        bb[take] = lam.view(-1) * bb[take] + one.view(-1) * bb_b
        if gb is not None and okb is not None:
            gb, okb = gb.clone(), okb.clone()
            if linear_geom and gb_b is not None and ok_b is not None:
                mixed = lam * gb[take] + one * gb_b
                finite = torch.isfinite(mixed) & ok_b
                gb[take] = torch.where(finite, mixed, gb[take])
                okb[take] = okb[take] & finite
            else:
                with torch.no_grad():
                    g = sd(yb[take], tau=tau, tau_li=tau_li)
                    gb[take] = torch.stack([g[name] for name in SCALAR_NAMES], dim=1)
                    okb[take] = _plausible_mask(gb[take])
        return xb, zb, yb, qb, bb, gb, okb

    for ep in range(start_epoch, epochs + 1):
        epochs_ran = ep
        model.train()
        tau, tau_li = _taus(ep)
        gscale = _geom_scale(ep)
        warm_end = (
            int(adapt_end[0]) if adaptive_warmup and adapt_end[0] is not None
            else (10**9 if adaptive_warmup else int(geom_warmup))
        )
        want_geom = use_geom and ep > warm_end
        w_from = (
            int(adapt_end[0]) + 2 if adaptive_warmup and adapt_end[0] is not None
            else int(gradnorm_w_from)
        )
        loss_sum, n_batches = 0.0, 0
        stratified = os.environ.get("CRC_FEQ_STRATIFIED_BATCH", "").strip().lower() in (
            "1", "true", "yes", "on",
        )
        if stratified and limited_pool is not None and int(limited_pool.numel()) > 1:
            from catskills.scripts.o1g_stratified_finetune import balanced_batch_perm

            perm = balanced_batch_perm(
                train_idx_pool, limited_pool, int(batch_size), device,
            )
        else:
            perm = train_idx_pool[torch.randperm(n_train_frames, device=device)]
        batch_iter = range(0, int(perm.numel()), batch_size)
        if use_tqdm:
            batch_iter = tqdm(batch_iter, desc=f"epoch {ep}/{epochs}", unit="batch", leave=False)
        parts_acc: dict[str, float] = {}
        w_g_ep = g_psi_ep = g_g_ep = None
        for start in batch_iter:
            idx = perm[start : start + batch_size]
            xb, zb, yb, qb, bb, gb, okb = _gather(idx)
            xb, zb, yb, qb, bb, gb, okb = _mix_limited(
                idx, xb, zb, yb, qb, bb, gb, okb, tau, tau_li,
            )
            # Geometry on a subsample of the batch keeps the scale run affordable; the plan budgets
            # for this explicitly. At probe scale geom_frac=1.0 and this is a no-op.
            use_gn = bool(gradnorm) and want_geom and hasattr(model, "embed")
            use_pc = bool(pcgrad) and want_geom
            h_fuse = None
            pred = None
            if use_gn:
                h_fuse = model.embed(xb)
                pred = model.heads(h_fuse)
            if want_geom and geom_frac < 1.0 and not use_gn:
                k = max(1, int(round(idx.numel() * geom_frac)))
                sub = torch.randperm(int(idx.numel()), device=device)[:k]
                loss, _, parts = _batch_loss(
                    xb, zb, yb, qb, bb, gb, okb, tau, tau_li, False, gscale, idx=idx,
                )
                psi_sub = ((model(xb[sub])[0]) @ pca_comp + pca_mean).view(-1, 65, 65)
                g = sd(psi_sub, tau=tau, tau_li=tau_li)
                g_hat = torch.stack([g[k2] for k2 in SCALAR_NAMES], dim=1)
                L_geom = _geom_loss(g_hat, gb[sub], okb[sub], sigma, geom_huber, geom_w)
                loss = loss + gscale * lambda_geom * L_geom
                parts["L_geom"] = L_geom
                if use_lcfs:
                    L_lcfs = _lcfs_loss(psi_sub, yb[sub])
                    loss = loss + gscale * lambda_lcfs * L_lcfs
                    parts["L_lcfs"] = L_lcfs
            else:
                loss, _, parts = _batch_loss(
                    xb, zb, yb, qb, bb, gb, okb, tau, tau_li, want_geom, gscale,
                    precomputed=pred, idx=idx,
                )
            if use_gn and h_fuse is not None:
                from catskills.olive_loss import update_gradnorm
                l_data = loss - gscale * lambda_geom * parts.get(
                    "L_geom", loss.new_zeros(())
                )
                l_stack = parts.get("L_geom", loss.new_zeros(()))
                if _gn_l0 is None and float(l_stack.detach()) > 0:
                    _gn_l0 = (float(l_data.detach()), float(l_stack.detach()))
                g_psi = torch.autograd.grad(
                    l_data, h_fuse, retain_graph=True, allow_unused=True,
                )[0]
                g_st = torch.autograd.grad(
                    w_g * gscale * lambda_geom * l_stack, h_fuse,
                    retain_graph=True, allow_unused=True,
                )[0]
                if (
                    g_psi is not None and g_st is not None
                    and ep >= int(w_from) and _gn_l0 is not None
                ):
                    w_g_ep, g_psi_ep, g_g_ep = update_gradnorm(
                        w_g, opt_w, g_psi, g_st, l_data, l_stack,
                        _gn_l0[0], _gn_l0[1], alpha=float(gradnorm_alpha),
                    )
                loss = l_data + w_g * gscale * lambda_geom * l_stack
            opt.zero_grad(set_to_none=True)
            if use_pc and "L_geom" in parts:
                from catskills.olive_loss import pcgrad_combine
                l_data = loss - gscale * lambda_geom * parts["L_geom"]
                l_stack = gscale * lambda_geom * parts["L_geom"]
                params = [p for p in model.parameters() if p.requires_grad]
                l_data.backward(retain_graph=True)
                g_data = [p.grad.detach().clone() if p.grad is not None else None for p in params]
                opt.zero_grad(set_to_none=True)
                l_stack.backward()
                g_st = [p.grad.detach().clone() if p.grad is not None else None for p in params]
                for p, g in zip(params, pcgrad_combine(g_data, g_st)):
                    p.grad = g
            else:
                loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            loss_sum += float(loss.detach())
            n_batches += 1
            if loss_report:
                for k2, v in parts.items():
                    parts_acc[k2] = parts_acc.get(k2, 0.0) + float(v.detach())
        sched.step()
        mean_loss = loss_sum / max(n_batches, 1)
        val_mse, val_cons, per, val_dlcfs = _validate(tau, tau_li)
        val_r2psi = (
            1.0 - val_mse / max(var_psi, 1e-12) if np.isfinite(val_mse) else float("nan")
        )
        if adaptive_warmup and adapt_end[0] is None:
            from catskills.olive_loss import warmup_healthy
            if warmup_healthy(
                val_r2psi, val_cons,
                r2_floor=float(warmup_r2psi), cons_floor=float(warmup_cons),
            ):
                adapt_end[0] = ep
                val_cons_warmup = float(val_cons)
                print(
                    f"[{MODEL_ID}] adaptive warmup end @ epoch {ep}: "
                    f"val_R2ψ={val_r2psi:.4f} val_Cons={val_cons:.4f}; "
                    f"λ_geom ramps next {int(geom_ramp)} epochs; "
                    f"w_g Adam from epoch {ep + 2}",
                    flush=True,
                )
            elif ep >= int(warmup_cap):
                print(
                    f"[{MODEL_ID}] KILLED at epoch {ep}: adaptive warmup "
                    f"val_R2ψ={val_r2psi:.4f} val_Cons={val_cons:.4f} "
                    f"never reached {float(warmup_r2psi):g} / {float(warmup_cons):g} "
                    f"by cap {int(warmup_cap)}",
                    flush=True,
                )
                break
        if (
            kill_r2psi > 0
            and ep >= int(kill_r2psi_after)
            and np.isfinite(val_r2psi)
            and val_r2psi < float(kill_r2psi)
        ):
            print(
                f"[{MODEL_ID}] KILLED at epoch {ep}: val_R2ψ={val_r2psi:.4f} "
                f"< {float(kill_r2psi):g} (floor after ep {int(kill_r2psi_after)})",
                flush=True,
            )
            break
        if ep == int(geom_warmup) and np.isfinite(val_cons):
            val_cons_warmup = float(val_cons)
        if (
            kill_cons > 0
            and ep >= int(kill_cons_after)
            and np.isfinite(val_cons)
            and val_cons < float(kill_cons)
        ):
            print(
                f"[{MODEL_ID}] KILLED at epoch {ep}: val_Cons={val_cons:.4f} "
                f"< {float(kill_cons):g}",
                flush=True,
            )
            break
        if float(psi_collapse_r2psi) > 0 and np.isfinite(val_r2psi):
            gate_ep = (
                int(adapt_end[0]) if adaptive_warmup and adapt_end[0] is not None
                else (int(geom_warmup) if not adaptive_warmup else None)
            )
            if gate_ep is not None and ep > gate_ep:
                if val_r2psi < float(psi_collapse_r2psi):
                    psi_collapse_run[0] += 1
                else:
                    psi_collapse_run[0] = 0
                if psi_collapse_run[0] >= int(psi_collapse_streak):
                    print(
                        f"[{MODEL_ID}] KILLED at epoch {ep}: post-gate val_R2ψ={val_r2psi:.4f} "
                        f"< {float(psi_collapse_r2psi):g} for "
                        f"{int(psi_collapse_streak)} consecutive epochs",
                        flush=True,
                    )
                    break
        if (
            gradnorm
            and ep >= 20
            and w_g_ep is not None
            and val_cons_warmup is not None
            and np.isfinite(val_cons)
            and (w_g_ep < 1e-3 or w_g_ep > 5.0)
            and (val_cons - val_cons_warmup) < 0.005
        ):
            print(
                f"[{MODEL_ID}] COLLAPSE at epoch {ep}: w_g={w_g_ep:.4g} "
                f"val_Cons={val_cons:.4f} (warmup {val_cons_warmup:.4f}). "
                f"Stop O1; run O1.1.",
                flush=True,
            )
            collapse_path = (results_dir() / "olive" / "O1_collapse.json")
            collapse_path.parent.mkdir(parents=True, exist_ok=True)
            collapse_path.write_text(
                json.dumps({
                    "epoch": ep, "w_g": w_g_ep, "val_cons": val_cons,
                    "val_cons_warmup": val_cons_warmup,
                }, indent=2) + "\n",
                encoding="utf-8",
            )
            break
        history.append(
            {
                "epoch": ep,
                "loss": mean_loss,
                "val_psi_mse": val_mse,
                "val_cons_proxy": val_cons,
                "val_r2_cons": {
                    n: float(per[i]) for i, n in enumerate(SCALAR_NAMES) if i < len(per)
                },
                "val_dlcfs_proxy": val_dlcfs,
                "tau": tau,
                "tau_li": tau_li,
                "phase": "adamish",
                "w_g": w_g_ep,
                "G_psi": g_psi_ep,
                "G_g": g_g_ep,
            }
        )
        if loss_report and n_batches:
            print(
                f"[{MODEL_ID}] ep{ep} terms  "
                + "  ".join(f"{k2}={v / n_batches:.4e}" for k2, v in parts_acc.items()),
                flush=True,
            )
        if ep == 1 or ep == epochs or ep % log_every == 0:
            extra = f"  val_ψ={val_mse:.4e}" if np.isfinite(val_mse) else ""
            if np.isfinite(val_cons):
                extra += (f"  val_Cons={val_cons:.4f}"
                          f"  val_R2ψ={1.0 - val_mse / max(var_psi, 1e-12):.4f}")
            if per:
                extra += "  " + " ".join(
                    f"{n}={float(per[i]):.4f}"
                    for i, n in enumerate(SCALAR_NAMES) if i < len(per)
                )
            if np.isfinite(val_dlcfs):
                extra += f"  val_D_LCFS={val_dlcfs:.4f}"
            if gscale < 1.0:
                extra += f"  gscale={gscale:.3f}"
            if w_g_ep is not None:
                extra += f"  w_g={w_g_ep:.3f}"
            # purple: the context gate is the experiment's own instrument. It starts at exactly 0, so
            # its trajectory answers "does the network WANT temporal context?" directly, and a gate
            # that decays toward 0 is the model telling us the branch is worthless -- which is a far
            # cleaner readout than inferring it from a final score, as R2a/R4/R10a had to.
            gate_p = getattr(model, "gate", None)
            if gate_p is not None:
                extra += f"  gate={float(gate_p.detach().mean()):+.4f}"
            gate_s = getattr(model, "gate_seq", None)
            if gate_s is not None:
                extra += f"  gate_seq={float(gate_s.detach().mean()):+.4f}"
            # Wall clock and a projected finish. Without these a 420-epoch run is unreadable: the log
            # cannot answer "is this on pace or has it slowed down", which is the only question worth
            # asking of an unattended run, and reconstructing it from file mtimes does not work.
            rate = (time.perf_counter() - t0) / max(ep, 1)
            eta = time.strftime("%H:%M", time.localtime(time.time() + rate * (epochs - ep)))
            extra += f"  {rate:.1f}s/ep  eta {eta}"
            print(
                f"[{MODEL_ID}] {time.strftime('%H:%M:%S')} epoch {ep}/{epochs}  "
                f"loss={mean_loss:.4e}{extra}  τ={tau:.4f}",
                flush=True,
            )

        # Select on the S proxy, not on Consistency alone. Plan 6 is right that teal early-stopped
        # on the wrong term, but selecting on Consistency *only* is worse: the geometry surrogate is
        # validated on ground-truth frames, so once a run drifts off the data manifold the proxy can
        # keep climbing while the real score collapses -- which is exactly how the first P3 attempt
        # failed (surrogate Cons rising, scorer Cons 0.000, R2_psi 0.18). Weighting psi fidelity back
        # in at its true score weight makes that trade visible to the selector.
        if np.isfinite(val_mse) and np.isfinite(val_cons):
            sel = W_PSI * (1.0 - val_mse / max(var_psi, 1e-12)) + W_CONS * val_cons
            # With P6a active the selector has to see D_LCFS too, or it will happily pick the epoch
            # that bought Consistency by pushing the boundary out -- which is precisely the trade
            # that cost 0.0034 S between teal_ensemble and P4 (section 15.2).
            if np.isfinite(val_dlcfs):
                sel += W_LCFS * (1.0 - min(1.0, val_dlcfs))
        elif np.isfinite(val_mse):
            sel = -val_mse
        else:
            sel = None
        # The improvement threshold has to match the selector's SCALE. The S-proxy branch above lives
        # near 1.0, where a fixed 1e-6 is a sensible "did anything really change". The geometry-off
        # branch is -val_mse, which for a 17M trunk is ~3.5e-5 -- so 1e-6 demands a 3% relative gain
        # per step and silently discards everything smaller. R7 hit exactly that: val_psi fell
        # monotonically 3.5734e-5 -> 3.4979e-5 over epochs 85-100, a 7.6e-7 total gain that never
        # cleared the bar, so the saved model is epoch 85 and the last 15 epochs were thrown away
        # while `stall` climbed toward early-stop on a run that was still improving.
        sel_tol = 1e-6 if (sel is not None and abs(sel) > 1e-3) else 1e-3 * abs(sel or 0.0)
        # Selection is independent of early-stop. Patience 0 still has to remember the best
        # epoch; it only means "do not stop". A resume that left best_selection at -inf
        # must be allowed to record the first finite score.
        sel_finite = sel is not None and math.isfinite(float(sel))
        improved = sel_finite and (
            not math.isfinite(best_sel) or float(sel) > best_sel + sel_tol
        )
        if improved:
            best_sel, best_epoch, stall = float(sel), ep, 0
            best_state = _clone_state()
            best_path = out.with_name(out.stem + "_best.pt")
            torch.save(
                {
                    **_static_ckpt(),
                    "model_state": best_state,
                    "training_params": {
                        "snapshot": "best",
                        "epochs_ran": int(ep),
                        "best_epoch": int(ep),
                        "best_selection": float(best_sel),
                    },
                },
                best_path,
            )
            print(
                f"[CHECKPOINT] New best model saved @ epoch {ep} (score: {best_sel:.6f})",
                flush=True,
            )
        elif early_stop_patience > 0 and sel_finite:
            stall += 1
            if stall >= early_stop_patience:
                print(
                    f"[{MODEL_ID}] early-stop at epoch {ep}: selection stalled {stall} "
                    f"(best {best_sel:.4f} @ ep {best_epoch})",
                    flush=True,
                )
                break
        snap_root = __import__("os").environ.get("CRC_FEQ_EPOCH_SNAPSHOT_DIR", "").strip()
        if snap_root:
            snap_path = Path(snap_root) / f"epoch_{ep:02d}.pt"
            snap_path.parent.mkdir(parents=True, exist_ok=True)
            live = {k2: v.detach().cpu().clone() for k2, v in model.state_dict().items()}
            torch.save(
                {
                    **_static_ckpt(),
                    "model_state": live,
                    "training_params": {
                        "partial": True,
                        "snapshot": "live",
                        "epochs_ran": int(ep),
                    },
                },
                snap_path,
            )
            print(f"[{MODEL_ID}] live snapshot epoch {ep} -> {snap_path}", flush=True)
        if ckpt_every > 0 and ep % ckpt_every == 0:
            # A non-finite selector must not freeze an earlier warmup snapshot into model_state.
            live_now = _clone_state()
            scored = best_state is not None and math.isfinite(best_sel)
            _save_partial(
                ep,
                best_state if scored else live_now,
                best_epoch if scored else ep,
                best_sel if scored else float("nan"),
                live_state=live_now,
            )
        if int(probe_epochs) > 0 and (ep - start_epoch + 1) >= int(probe_epochs):
            live_now = _clone_state()
            scored = best_state is not None and math.isfinite(best_sel)
            _save_partial(
                ep,
                best_state if scored else live_now,
                best_epoch if scored else ep,
                best_sel if scored else float("nan"),
                live_state=live_now,
            )
            print(
                f"[{MODEL_ID}] probe done after {probe_epochs} epoch(s); "
                f"partial kept at {partial_path.name}",
                flush=True,
            )
            return CharcoalBundle(
                checkpoint=partial_path,
                n_features=n_features,
                feature_mean=mean,
                feature_std=std,
                include_thomson=include_thomson,
                n_pca=int(n_comp),
                device_name=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                n_raw_features=n_raw_feat,
                temporal_window=int(temporal_window),
                temporal_mode=str(temporal_mode),
                arch=arch,
                feature_clip=float(feature_clip),
                pca_mean=pca.mean_.astype(np.float32),
                pca_components=pca.components_.astype(np.float32),
                hidden=hidden_t,
                activation=activation,
            )

    live_final = _clone_state()
    finite_best = best_state is not None and math.isfinite(best_sel)
    _save_partial(
        epochs_ran,
        best_state if finite_best else live_final,
        best_epoch if finite_best else epochs_ran,
        best_sel if finite_best else float("nan"),
        live_state=live_final,
    )
    if finite_best:
        model.load_state_dict(best_state)
        print(f"[{MODEL_ID}] restored best selection={best_sel:.4f} @ epoch {best_epoch}", flush=True)
    else:
        print(
            f"[{MODEL_ID}] best_selection was not finite; keeping final epoch {epochs_ran} weights",
            flush=True,
        )
        best_state = live_final
        best_epoch = int(epochs_ran)

    lbfgs_ran = 0
    if lbfgs_steps > 0 and str(arch).lower() in ("seq", "dualseq"):
        # cuDNN GRU backward requires train mode; train mode turns inter-layer dropout on and
        # makes strong_wolfe non-deterministic. Skip the polish rather than ship a noisy one.
        print(
            f"[{MODEL_ID}] L-BFGS skipped for arch={arch}: cuDNN GRU cannot backward in eval, "
            f"and train-mode dropout breaks strong_wolfe",
            flush=True,
        )
    elif lbfgs_steps > 0:
        # strong_wolfe needs a deterministic objective, and a dropout trunk left in train mode does
        # not give it one. A no-op for CoeffMLP, which has no dropout and no batch norm.
        model.eval()
        pool = train_idx_pool
        n_pool = int(pool.numel())
        idx_lb = (
            pool[torch.randperm(n_pool, device=device)[:lbfgs_max_frames]]
            if lbfgs_max_frames > 0 and n_pool > lbfgs_max_frames
            else pool
        )
        lb = torch.optim.LBFGS(
            list(model.parameters()),
            max_iter=int(lbfgs_steps),
            history_size=10,
            line_search_fn="strong_wolfe",
        )
        tau, tau_li = _taus(epochs)

        def _closure():
            lb.zero_grad(set_to_none=True)
            total = torch.zeros((), device=device)
            for start in range(0, int(idx_lb.numel()), batch_size):
                sl = idx_lb[start : start + batch_size]
                l, _, _ = _batch_loss(*_gather(sl), tau, tau_li, use_geom, _geom_scale(epochs))
                l.backward()
                total = total + l.detach()
            return total

        lb.step(_closure)
        lbfgs_ran = int(lbfgs_steps)
        val_mse, val_cons, _, val_dlcfs = _validate(tau, tau_li)
        print(
            f"[{MODEL_ID}] L-BFGS {lbfgs_ran} steps on {int(idx_lb.numel())} frames  "
            f"val_ψ={val_mse:.4e}  val_Cons={val_cons:.4f}"
            + (f"  val_D_LCFS={val_dlcfs:.4f}" if np.isfinite(val_dlcfs) else ""),
            flush=True,
        )
        history.append({"phase": "lbfgs", "val_psi_mse": val_mse, "val_cons_proxy": val_cons,
                        "val_dlcfs_proxy": val_dlcfs})

    train_s = time.perf_counter() - t0
    tp = {
        "n_train": len(cache_keys) if cache_keys is not None else len(shots or []),
        "n_train_frames": n_total,
        "max_frames": int(max_frames),
        "coeff_targets": bool(coeff_targets),
        "epochs": int(epochs),
        "epochs_ran": int(epochs_ran),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "optimizer": str(optimizer),
        "weight_decay": float(weight_decay),
        "hidden_layer_sizes": list(hidden_t),
        "activation": str(activation),
        "arch": str(arch),
        "trunk_width": int(trunk_width),
        "trunk_blocks": int(trunk_blocks),
        "trunk_dropout": float(trunk_dropout),
        "ctx_width": int(ctx_width),
        "ctx_blocks": int(ctx_blocks),
        "seq_hidden": int(seq_hidden),
        "seq_layers": int(seq_layers),
        "seq_window": int(seq_window) or int(temporal_window),
        "freeze_dualctx": bool(freeze_dualctx),
        "gate_seq": (
            float(model.gate_seq.detach().mean())
            if getattr(model, "gate_seq", None) is not None else None
        ),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "n_pca": int(n_comp),
        "temporal_window": int(temporal_window),
        "lambda_qb": float(lambda_qb),
        "lambda_geom": float(lambda_geom),
        "geom_emphasis": float(geom_emphasis),
        "geom_weights": str(geom_weights),
        "geom_weight_target": float(geom_weight_target),
        "geom_weight_floor": float(geom_weight_floor),
        "geom_weight_vec": (
            {n: float(geom_w[j]) for j, n in enumerate(SCALAR_NAMES)} if geom_w is not None else None
        ),
        "lambda_lcfs": float(lambda_lcfs),
        "lcfs_emphasis": float(lcfs_emphasis),
        "lcfs_cap": float(lcfs_cap),
        "geom_emphasis_final": float(geom_emphasis_final),
        "geom_anneal_tail": float(geom_anneal_tail),
        "qb_normalize": bool(qb_normalize),
        "qb_emphasis": float(qb_emphasis),
        "lambda_qb": float(lambda_qb),
        "var_psi": var_psi,
        "lambda_h1": float(lambda_h1),
        "lambda_disp": float(lambda_disp),
        "lambda_coeff_white": float(lambda_coeff_white),
        "coeff_white_power": float(coeff_white_power),
        "coeff_white_clamp": int(coeff_white_clamp),
        "edge_weight": str(edge_weight),
        "edge_mix": float(edge_mix),
        "edge_power": float(edge_power),
        "edge_band_lo": float(edge_band_lo),
        "edge_band_hi": float(edge_band_hi),
        "val_geom_frames": int(val_geom_frames),
        "geom_frac": float(geom_frac),
        "grad_clip": float(grad_clip),
        "lr_warmup": int(lr_warmup),
        "geom_warmup": int(geom_warmup),
        "geom_huber": float(geom_huber),
        "tau_hi": float(tau_hi),
        "tau_lo": float(tau_lo),
        "band": [float(band_lo), float(band_hi)],
        "val_fraction": float(val_fraction),
        "early_stop_patience": int(early_stop_patience),
        "kill_r2psi": float(kill_r2psi),
        "kill_r2psi_after": int(kill_r2psi_after),
        "psi_collapse_r2psi": float(psi_collapse_r2psi),
        "psi_collapse_streak": int(psi_collapse_streak),
        "best_epoch": int(best_epoch or epochs_ran),
        "best_selection": float(best_sel) if best_state is not None else None,
        "lbfgs_steps_ran": lbfgs_ran,
        "seed": int(seed),
        "init_seed": int(seed if init_seed is None else init_seed),
        "device": torch.cuda.get_device_name(0),
        "train_s": float(train_s),
        "feature_cache_hits": int(cache_stats.get("hits", 0)),
        "git_commit": git_full_sha(),
        "thomson_layout": __import__("os").environ.get("CRC_FEQ_THOMSON_LAYOUT", "").strip().lower(),
        "vac_in": __import__("os").environ.get("CRC_FEQ_VAC_IN", "").strip().lower()
        not in ("", "0", "false", "off"),
        "sky": bool(sky),
        "sky_tokens": bool(sky_tokens),
        "yellow": bool(yellow),
        "sky_vac_n": 16 if sky else 0,
    }
    ckpt = {
        "model_state": model.state_dict(),
        "residual_state": None,
        "n_features": n_features,
        "n_raw_features": n_raw_feat,
        "feature_mean": mean,
        "feature_std": std,
        "include_thomson": include_thomson,
        "thomson_layout": __import__("os").environ.get("CRC_FEQ_THOMSON_LAYOUT", "").strip().lower(),
        "n_pca": int(n_comp),
        "use_residual": False,
        "temporal_window": int(temporal_window),
        # Must be here and not only in _static_ckpt: load_charcoal reads THIS dict, so without it
        # inference silently rebuilds features with the default avgdiff while the weights were
        # trained on smoothdiff. Absent on every checkpoint written before R10, read as "avgdiff".
        "temporal_mode": str(temporal_mode),
        "hidden": list(hidden_t),
        "activation": activation,
        # Absent on every checkpoint written before R1, which load_charcoal reads as "mlp".
        "arch": arch,
        "arch_config": {"width": int(trunk_width), "blocks": int(trunk_blocks),
                        "dropout": float(trunk_dropout),
                        "ctx_width": int(ctx_width), "ctx_blocks": int(ctx_blocks),
                        "seq_hidden": int(seq_hidden), "seq_layers": int(seq_layers),
                        "seq_window": int(seq_window) or int(temporal_window),
                        "n_split": int(n_raw_feat), "n_nbi": int(n_nbi)},
        "feature_clip": float(feature_clip),
        "pca_mean": pca.mean_.astype(np.float32),
        "pca_components": pca.components_.astype(np.float32),
        "training_params": tp,
        "history": history,
        "method": MODEL_ID,
        "model_version": make_model_version(MODEL_ID),
        "git_commit": tp["git_commit"],
    }
    torch.save(ckpt, out)
    final_path = out.with_name(out.stem + "_final.pt")
    best_path = out.with_name(out.stem + "_best.pt")
    torch.save({**ckpt, "model_state": live_final, "training_params": {**tp, "snapshot": "final_live"}}, final_path)
    torch.save({**ckpt, "model_state": best_state, "training_params": {**tp, "snapshot": "best"}}, best_path)
    print(
        f"[{MODEL_ID}] wrote {out.name}; kept {partial_path.name}; "
        f"wrote {best_path.name} and {final_path.name}  train_s={train_s:.1f}s",
        flush=True,
    )

    return CharcoalBundle(
        checkpoint=out,
        n_features=n_features,
        feature_mean=mean,
        feature_std=std,
        include_thomson=include_thomson,
        n_pca=int(n_comp),
        device_name=tp["device"],
        n_raw_features=n_raw_feat,
        temporal_window=int(temporal_window),
        temporal_mode=str(temporal_mode),
        seq_window=int(seq_window) or int(temporal_window),
        training_params=tp,
        model_version=ckpt["model_version"],
        pca_mean=ckpt["pca_mean"],
        pca_components=ckpt["pca_components"],
        hidden=hidden_t,
        activation=activation,
        arch=arch,
        feature_clip=float(feature_clip),
    )


# --------------------------------------------------------------------------- inference


def load_charcoal(path: Path, device: Optional[torch.device] = None):
    device = device or require_cuda()
    ck = torch.load(Path(path), map_location=device, weights_only=False)
    hidden = tuple(int(h) for h in ck.get("hidden", (256, 128)))
    ac = dict(ck.get("arch_config") or {})
    model = build_trunk(
        str(ck.get("arch", "mlp")), int(ck["n_features"]), int(ck["n_pca"]),
        hidden=hidden, activation=ck.get("activation", "tanh"),
        trunk_width=int(ac.get("width", 512)), trunk_blocks=int(ac.get("blocks", 4)),
        trunk_dropout=float(ac.get("dropout", 0.05)),
        ctx_width=int(ac.get("ctx_width", 512)), ctx_blocks=int(ac.get("ctx_blocks", 3)),
        n_split=int(ac.get("n_split", 0)),
        seq_hidden=int(ac.get("seq_hidden", ac.get("hidden", 256))),
        seq_layers=int(ac.get("seq_layers", ac.get("layers", 2))),
        shot_bidirectional=bool(ac.get("bidirectional", True)),
        shot_mlp_blocks=int(ac.get("mlp_blocks", 2)),
        n_nbi=int(ac.get("n_nbi", 0)),
    ).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    bundle = CharcoalBundle(
        checkpoint=Path(path),
        n_features=int(ck["n_features"]),
        feature_mean=np.asarray(ck["feature_mean"], dtype=np.float32),
        feature_std=np.asarray(ck["feature_std"], dtype=np.float32),
        include_thomson=bool(ck.get("include_thomson", True)),
        thomson_layout=str(ck.get("thomson_layout") or (ck.get("training_params") or {}).get("thomson_layout") or ""),
        n_pca=int(ck["n_pca"]),
        device_name=str(ck.get("training_params", {}).get("device", "")),
        n_raw_features=int(ck.get("n_raw_features", ck["n_features"])),
        temporal_window=int(ck.get("temporal_window", 5)),
        temporal_mode=str(ck.get("temporal_mode", "avgdiff")),
        seq_window=int((ck.get("arch_config") or {}).get("seq_window")
                       or ck.get("temporal_window", 5)),
        raw_feature_mean=(
            np.asarray(ck["raw_feature_mean"], dtype=np.float32)
            if ck.get("raw_feature_mean") is not None else None
        ),
        raw_feature_std=(
            np.asarray(ck["raw_feature_std"], dtype=np.float32)
            if ck.get("raw_feature_std") is not None else None
        ),
        training_params=dict(ck.get("training_params") or {}),
        model_version=str(ck.get("model_version", "")),
        pca_mean=np.asarray(ck["pca_mean"], dtype=np.float32),
        pca_components=np.asarray(ck["pca_components"], dtype=np.float32),
        hidden=hidden,
        activation=ck.get("activation", "tanh"),
        arch=str(ck.get("arch", "mlp")),
        feature_clip=float(ck.get("feature_clip", 0.0) or 0.0),
    )
    env = __import__("os").environ
    for key in (
        "CRC_FEQ_VAC_IN", "CRC_FEQ_SKY", "CRC_FEQ_SKY_TOKENS",
        "CRC_FEQ_YELLOW", "CRC_FEQ_WHITE", "CRC_FEQ_GRAY", "CRC_FEQ_CORAL", "CRC_FEQ_NONDIM",
    ):
        env.pop(key, None)
    if bundle.training_params.get("vac_in"):
        env["CRC_FEQ_VAC_IN"] = "1"
    if bundle.training_params.get("sky"):
        env["CRC_FEQ_SKY"] = "1"
    if bundle.training_params.get("sky_tokens"):
        env["CRC_FEQ_SKY_TOKENS"] = "1"
    if bundle.training_params.get("yellow"):
        env["CRC_FEQ_YELLOW"] = "1"
    if bundle.training_params.get("gray"):
        env["CRC_FEQ_GRAY"] = "1"
    if bundle.training_params.get("coral"):
        env["CRC_FEQ_CORAL"] = "1"
    if bundle.training_params.get("nondim"):
        env["CRC_FEQ_NONDIM"] = "1"
    return model, bundle, ck


@torch.inference_mode()
def predict_from_raw_features(
    bundle: CharcoalBundle,
    X_list: list[np.ndarray],
    *,
    model: CoeffMLP | None = None,
    device: Optional[torch.device] = None,
    max_frames: int = 16384,
    smooth_window: int = 0,
) -> list[dict]:
    """Per-shot raw features -> psi/q95/betaN, applying the same temporal augmentation as training.

    ``smooth_window`` optionally applies probe 0c's output smoothing along t as well; it is off by
    default because the temporal input channels already carry most of that gain, and stacking both
    is measured at P5 rather than assumed.
    """
    if not X_list:
        return []
    device = device or require_cuda()
    if model is None:
        model, bundle, _ = load_charcoal(bundle.checkpoint, device=device)

    normed = []
    for X in X_list:
        # Pad to the training *raw* width BEFORE augmenting. Training pads every shot to the corpus
        # max (133) while most shots are natively 129, and the augmented layout is three contiguous
        # blocks [raw | moving-average | difference]. Padding after augmentation would shift the
        # second and third blocks by the width deficit and silently misalign every column.
        Xr = np.asarray(X, dtype=np.float32)
        if bundle.training_params.get("vac_in"):
            from catskills.vacuum_green import append_vac_features
            Xr = append_vac_features(Xr)
        if bundle.training_params.get("sky"):
            from catskills.red_nondim import N_MAG
            from catskills.sky_geom import vac_from_d3d_matrix
            if Xr.shape[1] >= N_MAG and Xr.shape[1] != bundle.n_raw_features:
                Xr = vac_from_d3d_matrix(Xr)
        if bundle.training_params.get("sky_tokens"):
            from catskills.red_nondim import N_MAG
            from catskills.sky_geom import tokens_from_d3d_matrix
            if Xr.shape[1] >= N_MAG and Xr.shape[1] != bundle.n_raw_features:
                Xr = tokens_from_d3d_matrix(Xr)
        if bundle.training_params.get("yellow"):
            from catskills.red_nondim import N_MAG
            from catskills.yellow_geom import yellow_from_d3d_matrix
            if Xr.shape[1] >= N_MAG and Xr.shape[1] != bundle.n_raw_features:
                Xr = yellow_from_d3d_matrix(Xr)
        if Xr.shape[1] < bundle.n_raw_features:
            Xr = np.hstack(
                [Xr, np.zeros((len(Xr), bundle.n_raw_features - Xr.shape[1]), dtype=np.float32)]
            )
        elif Xr.shape[1] > bundle.n_raw_features:
            Xr = Xr[:, : bundle.n_raw_features]
        Xa = augment_temporal(
            Xr, np.zeros(len(Xr), dtype=np.int32), bundle.temporal_window,
            bundle.temporal_mode, seq_window=int(bundle.seq_window or 0),
        )
        if Xa.shape[1] != bundle.n_features:
            raise RuntimeError(
                f"augmented width {Xa.shape[1]} != trained {bundle.n_features} "
                f"(raw {bundle.n_raw_features}, window {bundle.temporal_window})"
            )
        normed.append(_clip_features(
            ((Xa - bundle.feature_mean) / bundle.feature_std).astype(np.float32),
            bundle.feature_clip,
        ))

    lengths = [int(x.shape[0]) for x in normed]
    Xs = np.concatenate(normed, axis=0)
    pca_mean = torch.from_numpy(bundle.pca_mean).to(device)
    pca_comp = torch.from_numpy(bundle.pca_components).to(device)

    psi_parts, q_parts, b_parts = [], [], []
    if str(bundle.arch).lower() == "dualshot":
        if bundle.raw_feature_mean is None or bundle.raw_feature_std is None:
            raise RuntimeError("dualshot checkpoint is missing raw_feature_mean/std")
        for Xa, Xr0 in zip(normed, X_list):
            Xr = np.asarray(Xr0, dtype=np.float32)
            if Xr.shape[1] < bundle.n_raw_features:
                Xr = np.hstack(
                    [Xr, np.zeros((len(Xr), bundle.n_raw_features - Xr.shape[1]),
                                  dtype=np.float32)]
                )
            elif Xr.shape[1] > bundle.n_raw_features:
                Xr = Xr[:, : bundle.n_raw_features]
            Xrn = _clip_features(
                ((Xr - bundle.raw_feature_mean) / bundle.raw_feature_std).astype(np.float32),
                bundle.feature_clip,
            )
            xd = torch.from_numpy(np.ascontiguousarray(Xa)).to(device)
            xr = torch.from_numpy(np.ascontiguousarray(Xrn)).to(device)
            z_hat, q_hat, b_hat = model(xd, xr)[:3]
            psi = (z_hat @ pca_comp + pca_mean).view(-1, 65, 65)
            psi_parts.append(psi.cpu().numpy().astype(np.float32))
            q_parts.append(q_hat.cpu().numpy().astype(np.float32))
            b_parts.append(b_hat.cpu().numpy().astype(np.float32))
    elif str(bundle.arch).lower() == "shotseq":
        # A shot-GRU cannot see concatenated shots or a mid-shot chunk. One call per discharge.
        for Xa in normed:
            xb = torch.from_numpy(np.ascontiguousarray(Xa)).to(device, non_blocking=True)
            z_hat, q_hat, b_hat = model(xb)[:3]
            psi = (z_hat @ pca_comp + pca_mean).view(-1, 65, 65)
            psi_parts.append(psi.cpu().numpy().astype(np.float32))
            q_parts.append(q_hat.cpu().numpy().astype(np.float32))
            b_parts.append(b_hat.cpu().numpy().astype(np.float32))
    else:
        for start in range(0, Xs.shape[0], max(1, int(max_frames))):
            xb = torch.from_numpy(Xs[start : start + max_frames]).to(device, non_blocking=True)
            z_hat, q_hat, b_hat = model(xb)[:3]
            psi = (z_hat @ pca_comp + pca_mean).view(-1, 65, 65)
            psi_parts.append(psi.cpu().numpy().astype(np.float32))
            q_parts.append(q_hat.cpu().numpy().astype(np.float32))
            b_parts.append(b_hat.cpu().numpy().astype(np.float32))

    psi_np = np.concatenate(psi_parts, axis=0)
    q_np = np.concatenate(q_parts, axis=0)
    b_np = np.concatenate(b_parts, axis=0)
    out, off = [], 0
    for T in lengths:
        p = psi_np[off : off + T]
        q = q_np[off : off + T]
        b = b_np[off : off + T]
        if smooth_window and smooth_window > 1 and T > 1:
            p, q, b = (_smooth_t(a, smooth_window) for a in (p, q, b))
        out.append({"psirz": p, "q95": q, "betaN": b})
        off += T
    return out


def _smooth_t(a: np.ndarray, w: int) -> np.ndarray:
    """Centred moving average along axis 0 with clamped edges (probe 0c's w=5)."""
    half, T = int(w) // 2, a.shape[0]
    idx = np.arange(T)
    acc = np.zeros_like(a, dtype=np.float32)
    for d in range(-half, half + 1):
        acc += a[np.clip(idx + d, 0, T - 1)]
    return acc / float(2 * half + 1)


@torch.inference_mode()
def predict_row(
    bundle: CharcoalBundle,
    row: dict,
    source: str,
    *,
    model: CoeffMLP | None = None,
    device: Optional[torch.device] = None,
    smooth_window: int = 0,
) -> dict:
    from catskills.features import features_from_shot, shot_from_row_inputs_only

    del source
    X = features_from_shot(
        shot_from_row_inputs_only(row),
        include_thomson=bundle.include_thomson,
        thomson_layout=bundle.thomson_layout,
    ).astype(np.float32)
    return predict_from_raw_features(
        bundle, [X], model=model, device=device, smooth_window=smooth_window
    )[0]
