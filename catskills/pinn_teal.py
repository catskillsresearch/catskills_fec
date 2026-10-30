"""CRC_FEQ_teal — PCA-manifold flux head (+ optional residual), GPU.

V3a: torch MLP ``c → z`` then ``ψ̂ = PCA⁻¹(z)`` (mean-anchored by construction).
V3b: ``ψ̂ = PCA⁻¹(z) + δ_θ`` with ``λ_δ · mean(δ²)``.

Public ID is opaque (no method keyword). See ``CRC_FEQ_COLORS.md``.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from catskills.feature_cache import (
    build_feature_matrix_cached,
    build_feature_matrix_from_keys,
    load_cache_index,
)
from catskills.features import features_from_shot, shot_from_row_inputs_only
from catskills.model_version import git_full_sha, make_model_version
from experiments import EFIT_SCALAR_LABELS

MODEL_ID = "CRC_FEQ_teal"

R_LO, R_HI = 0.8, 2.5
Z_LO, Z_HI = -1.5, 1.5


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{MODEL_ID} requires a CUDA GPU (torch.cuda.is_available() is False)."
        )
    return torch.device("cuda")


def _activation(name: str) -> nn.Module:
    key = (name or "tanh").lower()
    if key == "tanh":
        return nn.Tanh()
    if key == "gelu":
        return nn.GELU()
    if key in ("silu", "swish"):
        return nn.SiLU()
    raise ValueError(f"Unknown activation {name!r}; choose tanh|gelu|silu")


def make_optimizer(
    params,
    *,
    name: str,
    lr: float,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    key = (name or "adam").lower().strip()
    if key == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if key == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay or 1e-4)
    if key == "sgd":
        return torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=momentum > 0
        )
    if key == "radam":
        return torch.optim.RAdam(params, lr=lr, weight_decay=weight_decay)
    if key == "nadam":
        return torch.optim.NAdam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer {name!r}; choose adam|adamw|sgd|radam|nadam")


class CoeffMLP(nn.Module):
    """Normalized features → PCA coefficients (+ q95/βN + optional axis/li)."""

    def __init__(
        self,
        n_in: int,
        n_pca: int,
        hidden: tuple[int, ...] = (256, 128),
        activation: str = "tanh",
    ):
        super().__init__()
        self.hidden = tuple(int(h) for h in hidden)
        self.activation_name = activation
        layers: list[nn.Module] = []
        d = n_in
        for h in self.hidden:
            layers.extend([nn.Linear(d, h), _activation(activation)])
            d = h
        self.backbone = nn.Sequential(*layers)
        self.coeff_head = nn.Linear(d, n_pca)
        self.q95_head = nn.Linear(d, 1)
        self.beta_head = nn.Linear(d, 1)
        self.r_axis_head = nn.Linear(d, 1)
        self.z_axis_head = nn.Linear(d, 1)
        self.li_head = nn.Linear(d, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return (
            self.coeff_head(h),
            self.q95_head(h).squeeze(-1),
            self.beta_head(h).squeeze(-1),
            self.r_axis_head(h).squeeze(-1),
            self.z_axis_head(h).squeeze(-1),
            self.li_head(h).squeeze(-1),
        )


class ResidualMap(nn.Module):
    """Low-capacity δ(R,Z|c) map residual on the PCA reconstruction."""

    def __init__(self, n_cond: int, hidden: int = 64, layers: int = 3, activation: str = "tanh"):
        super().__init__()
        d_in = 2 + n_cond
        mods: list[nn.Module] = []
        d = d_in
        for _ in range(layers):
            mods.extend([nn.Linear(d, hidden), _activation(activation)])
            d = hidden
        mods.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*mods)
        self.scale = nn.Parameter(torch.tensor(0.05))

    def forward(
        self,
        c: torch.Tensor,
        R: torch.Tensor,
        Z: torch.Tensor,
        *,
        chunk: int = 32,
    ) -> torch.Tensor:
        """c (B,d), R (W,), Z (H,) → δ (B,H,W). Chunked to avoid LBFGS OOMs."""
        B = c.shape[0]
        if B > chunk:
            parts = [
                self.forward(c[i : i + chunk], R, Z, chunk=chunk)
                for i in range(0, B, chunk)
            ]
            return torch.cat(parts, dim=0)
        H, W = Z.numel(), R.numel()
        ZZ, RR = torch.meshgrid(Z, R, indexing="ij")
        Rn = 2.0 * (RR - R_LO) / (R_HI - R_LO) - 1.0
        Zn = 2.0 * (ZZ - Z_LO) / (Z_HI - Z_LO) - 1.0
        rz = torch.stack([Rn, Zn], dim=-1).reshape(1, H * W, 2).expand(B, -1, -1)
        c_exp = c.unsqueeze(1).expand(B, H * W, -1)
        inp = torch.cat([rz, c_exp], dim=-1).reshape(B * H * W, -1)
        return (self.scale * self.net(inp).squeeze(-1)).reshape(B, H, W)


@dataclass
class TealBundle:
    checkpoint: Path
    n_features: int
    feature_mean: np.ndarray
    feature_std: np.ndarray
    include_thomson: bool
    n_pca: int
    use_residual: bool
    device_name: str
    training_params: dict = field(default_factory=dict)
    model_version: str = ""
    pca_mean: np.ndarray | None = None
    pca_components: np.ndarray | None = None
    hidden: tuple[int, ...] = (256, 128)
    activation: str = "tanh"


def _default_grids_torch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    R = torch.linspace(R_LO, R_HI, 65, device=device)
    Z = torch.linspace(Z_LO, Z_HI, 65, device=device)
    return R, Z


def _parse_hidden(hidden: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(hidden, (tuple, list)):
        return tuple(int(h) for h in hidden)
    parts = [p.strip() for p in str(hidden).replace("x", ",").split(",") if p.strip()]
    if not parts:
        return (256, 128)
    return tuple(int(p) for p in parts)


def _trim_keys_to_frame_budget(
    cache_keys: list[str],
    max_frames: int,
    *,
    include_thomson: bool,
    seed: int,
) -> list[str]:
    """Pick a random shot subset whose cached frame count ≤ max_frames.

    Avoids loading millions of ψ maps into RAM then discarding them.
    """
    if max_frames <= 0:
        return cache_keys
    idx = {e["key"]: int(e["n_frames"]) for e in load_cache_index(include_thomson)}
    rng = np.random.default_rng(seed)
    order = list(cache_keys)
    rng.shuffle(order)
    chosen: list[str] = []
    frames = 0
    for key in order:
        n = idx.get(key, 0)
        if n <= 0:
            continue
        if chosen and frames + n > max_frames:
            continue
        chosen.append(key)
        frames += n
        if frames >= max_frames:
            break
    if not chosen:
        return cache_keys[:1]
    print(
        f"[{MODEL_ID}] frame budget: kept {len(chosen)}/{len(cache_keys)} shots "
        f"≈{frames} frames (cap={max_frames})",
        flush=True,
    )
    return chosen


def train_teal(
    shots: list[dict] | None = None,
    *,
    cache_keys: list[str] | None = None,
    include_thomson: bool = True,
    epochs: int = 12,
    batch_size: int = 16,
    lr: float = 1e-3,
    max_frames: int = 2048,
    n_pca: int = 20,
    use_residual: bool = False,
    lambda_delta: float = 0.1,
    lambda_qb: float = 1.0,
    lambda_axis: float = 0.0,
    lambda_z_axis: float = 0.0,
    lambda_li: float = 0.0,
    lambda_lcfs: float = 0.0,
    lambda_shape: float = 0.0,
    lambda_volume: float = 0.0,
    lambda_cons_gt: float = 0.0,
    lcfs_lo: float = 0.92,
    lcfs_hi: float = 0.995,
    hard_shot_keys: list[str] | None = None,
    hard_frac: float = 0.0,
    val_fraction: float = 0.0,
    early_stop_patience: int = 0,
    init_checkpoint: Path | str | None = None,
    optimizer: str = "adam",
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    hidden: str | tuple[int, ...] = (256, 128),
    activation: str = "tanh",
    lbfgs_steps: int = 0,
    lbfgs_max_frames: int = 32768,
    seed: int = 42,
    output: Path | None = None,
    log_every: int = 10,
) -> TealBundle:
    device = require_cuda()
    torch.manual_seed(seed)
    np.random.seed(seed)
    hidden_t = _parse_hidden(hidden)

    if cache_keys is not None:
        if max_frames > 0:
            cache_keys = _trim_keys_to_frame_budget(
                cache_keys,
                max_frames,
                include_thomson=include_thomson,
                seed=seed,
            )
        X, Y, S, frame_shot_idx, cache_stats = build_feature_matrix_from_keys(
            cache_keys, include_thomson=include_thomson
        )
        key_list = list(cache_keys)
    else:
        X, Y, S, frame_shot_idx, cache_stats = build_feature_matrix_cached(
            shots or [], include_thomson=include_thomson
        )
        key_list = []

    hard_set = {str(k).strip() for k in (hard_shot_keys or []) if str(k).strip()}
    hard_shot_i = {
        i for i, k in enumerate(key_list) if k in hard_set
    }
    hard_frame = np.isin(frame_shot_idx, np.fromiter(hard_shot_i, dtype=np.int32)) if hard_shot_i else np.zeros(len(X), dtype=bool)

    n_all = len(X)
    if max_frames > 0 and n_all > max_frames:
        rng = np.random.default_rng(seed)
        # Prefer keeping hard frames when curriculum is on.
        n_keep = int(max_frames)
        hard_ix = np.flatnonzero(hard_frame)
        easy_ix = np.flatnonzero(~hard_frame)
        n_hard_keep = min(len(hard_ix), max(0, int(round(n_keep * max(hard_frac, 0.0)))))
        n_easy_keep = n_keep - n_hard_keep
        pick_h = rng.choice(hard_ix, size=n_hard_keep, replace=False) if n_hard_keep else np.array([], dtype=np.int64)
        pick_e = rng.choice(easy_ix, size=min(n_easy_keep, len(easy_ix)), replace=False) if n_easy_keep and len(easy_ix) else np.array([], dtype=np.int64)
        pick = np.concatenate([pick_h, pick_e])
        if len(pick) < n_keep:
            rest = np.setdiff1d(np.arange(n_all), pick, assume_unique=False)
            need = n_keep - len(pick)
            pick = np.concatenate([pick, rng.choice(rest, size=need, replace=False)])
        rng.shuffle(pick)
        X, Y, S = X[pick], Y[pick], S[pick]
        hard_frame = hard_frame[pick]
        frame_shot_idx = frame_shot_idx[pick]
    n_total = len(X)

    init_ck = None
    if init_checkpoint is not None:
        init_path = Path(init_checkpoint)
        if not init_path.is_file():
            raise FileNotFoundError(f"init_checkpoint not found: {init_path}")
        init_ck = torch.load(init_path, map_location="cpu", weights_only=False)
        print(f"[{MODEL_ID}] warm-start from {init_path}", flush=True)

    # float64 stats — float32 Σx² overflows on large/wide feature matrices.
    if init_ck is not None and "feature_mean" in init_ck and "feature_std" in init_ck:
        mean = np.asarray(init_ck["feature_mean"], dtype=np.float32)
        std = np.asarray(init_ck["feature_std"], dtype=np.float32)
        if mean.shape[0] != X.shape[1]:
            mean = X.mean(axis=0, dtype=np.float64).astype(np.float32)
            std = X.std(axis=0, dtype=np.float64).astype(np.float32)
            std = np.where(std < 1e-6, 1.0, std)
    else:
        mean = X.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = X.std(axis=0, dtype=np.float64).astype(np.float32)
        std = np.where(std < 1e-6, 1.0, std)
    Xs = ((X - mean) / std).astype(np.float32)
    Y_flat = Y.reshape(n_total, -1).astype(np.float32)

    n_comp = int(min(n_pca, max(1, n_total - 1), Y_flat.shape[1]))
    pca = PCA(
        n_components=n_comp,
        svd_solver="randomized",
        random_state=seed,
        iterated_power=3,
    )
    if (
        init_ck is not None
        and init_ck.get("pca_components") is not None
        and int(init_ck.get("n_pca", 0)) == n_comp
        and np.asarray(init_ck["pca_components"]).shape[0] == n_comp
    ):
        pca.mean_ = np.asarray(init_ck["pca_mean"], dtype=np.float64)
        pca.components_ = np.asarray(init_ck["pca_components"], dtype=np.float64)
        pca.explained_variance_ratio_ = np.ones(n_comp) / n_comp
        Zc = ((Y_flat - pca.mean_.astype(np.float32)) @ pca.components_.T.astype(np.float32)).astype(np.float32)
        print(f"[{MODEL_ID}] reused PCA from init_checkpoint (n={n_comp})", flush=True)
    else:
        Zc = pca.fit_transform(Y_flat).astype(np.float32)
    del Y_flat
    print(
        f"[{MODEL_ID}] PCA n={n_comp}  explained={float(np.asarray(pca.explained_variance_ratio_).sum()):.3f}  "
        f"frames={n_total}  hard_frames={int(hard_frame.sum())}  hard_frac={hard_frac}  "
        f"residual={'ON' if use_residual else 'OFF'}  "
        f"hidden={hidden_t}  act={activation}  opt={optimizer}",
        flush=True,
    )

    i_q = EFIT_SCALAR_LABELS.index("efit_q95")
    i_b = EFIT_SCALAR_LABELS.index("efit_beta_n")
    i_r = EFIT_SCALAR_LABELS.index("efit_r_axis")
    i_z = EFIT_SCALAR_LABELS.index("efit_z_axis")
    i_li = EFIT_SCALAR_LABELS.index("efit_li")
    q95 = np.nan_to_num(S[:, i_q], nan=0.0).astype(np.float32)
    beta = np.nan_to_num(S[:, i_b], nan=0.0).astype(np.float32)
    r_axis = S[:, i_r].astype(np.float32)
    z_axis = S[:, i_z].astype(np.float32)
    li = S[:, i_li].astype(np.float32)

    cons_arr = None
    if lambda_cons_gt > 0:
        from catskills.cons_gt_cache import build_cons_scalars_for_maps

        print(f"[{MODEL_ID}] building cons GT for {n_total} frames…", flush=True)
        cons_arr, cons_stats = build_cons_scalars_for_maps(Y, progress=True)
        print(
            f"[{MODEL_ID}] cons GT hits={cons_stats['hits']} misses={cons_stats['misses']}",
            flush=True,
        )

    # Keep the whole frame set on GPU — H2D every batch was pinning 1 CPU core
    # and starving the tiny MLP (nvtop looked "idle" at ~few hundred MiB).
    n_features = int(Xs.shape[1])
    xs_t = torch.from_numpy(Xs).to(device)
    zc_t = torch.from_numpy(Zc).to(device)
    y_t = torch.from_numpy(Y.astype(np.float32)).to(device)
    q_t = torch.from_numpy(q95).to(device)
    b_t = torch.from_numpy(beta).to(device)
    r_t = torch.from_numpy(r_axis).to(device)
    zax_t = torch.from_numpy(z_axis).to(device)
    li_t = torch.from_numpy(li).to(device)
    kappa_t = vol_t = tri_t_t = tri_b_t = None
    if cons_arr is not None:
        kappa_t = torch.from_numpy(cons_arr["kappa"]).to(device)
        vol_t = torch.from_numpy(cons_arr["volume"]).to(device)
        tri_t_t = torch.from_numpy(cons_arr["tri_top"]).to(device)
        tri_b_t = torch.from_numpy(cons_arr["tri_bot"]).to(device)
    del Xs, Zc, Y, S, q95, beta, r_axis, z_axis, li, cons_arr
    n_total = int(xs_t.shape[0])
    bytes_gb = (
        xs_t.nbytes + zc_t.nbytes + y_t.nbytes + q_t.nbytes + b_t.nbytes
    ) / 1e9
    print(
        f"[{MODEL_ID}] GPU-resident tensors: frames={n_total}  "
        f"batch={batch_size}  data≈{bytes_gb:.2f} GB  "
        f"λ_qb={lambda_qb} λ_axis={lambda_axis} λ_z={lambda_z_axis} λ_li={lambda_li} "
        f"λ_lcfs={lambda_lcfs} λ_shape={lambda_shape} λ_vol={lambda_volume} "
        f"λ_cons_gt={lambda_cons_gt}",
        flush=True,
    )

    model = CoeffMLP(n_features, n_comp, hidden=hidden_t, activation=activation).to(device)
    residual: ResidualMap | None = None
    params = list(model.parameters())
    if use_residual:
        residual = ResidualMap(n_features, activation=activation).to(device)
        params += list(residual.parameters())
    if init_ck is not None and init_ck.get("model_state") is not None:
        missing, unexpected = model.load_state_dict(init_ck["model_state"], strict=False)
        if missing:
            print(f"[{MODEL_ID}] init missing keys {list(missing)[:4]}…", flush=True)
        if unexpected:
            print(f"[{MODEL_ID}] init unexpected keys {list(unexpected)[:4]}…", flush=True)
        if (
            use_residual
            and residual is not None
            and init_ck.get("residual_state") is not None
        ):
            residual.load_state_dict(init_ck["residual_state"], strict=False)
    opt = make_optimizer(
        params, name=optimizer, lr=lr, weight_decay=weight_decay, momentum=momentum
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    hard_mask_t = torch.from_numpy(hard_frame.astype(np.bool_)).to(device)

    pca_mean = torch.from_numpy(pca.mean_.astype(np.float32)).to(device)
    pca_comp = torch.from_numpy(pca.components_.astype(np.float32)).to(device)
    R_grid, Z_grid = _default_grids_torch(device)
    axis_sign = -1.0  # DIII-D EFIT: axis at ψ minimum

    def _finite_mse(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        valid = torch.isfinite(gt)
        if not bool(valid.any()):
            return torch.zeros((), device=device)
        return ((pred[valid] - gt[valid]) ** 2).mean()

    def _batch_loss(xb, zb, yb, qb, bb, r_gt, z_gt, li_gt, k_gt, v_gt, tt_gt, tb_gt):
        from catskills.pinn_navy import (
            normalize_flux_torch,
            soft_lcfs_extents,
            soft_plasma_moments,
            soft_triangularity,
        )

        z_hat, q_hat, b_hat, r_hat, z_hat_ax, li_hat = model(xb)
        psi_flat = z_hat @ pca_comp + pca_mean
        psi = psi_flat.view(-1, 65, 65)
        L_delta = torch.zeros((), device=device)
        if residual is not None:
            delta = residual(xb, R_grid, Z_grid)
            psi = psi + delta
            L_delta = (delta ** 2).mean()
        L_data = ((psi - yb) ** 2).mean()
        L_coeff = ((z_hat - zb) ** 2).mean()
        L_qb = ((q_hat - qb) ** 2).mean() + ((b_hat - bb) ** 2).mean()
        L_axis = torch.zeros((), device=device)
        if lambda_axis > 0:
            L_axis = _finite_mse(r_hat, r_gt) + _finite_mse(z_hat_ax, z_gt)
        L_z = torch.zeros((), device=device)
        if lambda_z_axis > 0:
            L_z = _finite_mse(z_hat_ax, z_gt)
        L_li = torch.zeros((), device=device)
        if lambda_li > 0:
            L_li = _finite_mse(li_hat, li_gt)
        L_lcfs = torch.zeros((), device=device)
        if lambda_lcfs > 0:
            psi_n_gt = normalize_flux_torch(yb.detach(), axis_sign=axis_sign)
            band = ((psi_n_gt >= lcfs_lo) & (psi_n_gt <= lcfs_hi)).to(dtype=psi.dtype)
            denom = band.sum().clamp_min(1.0)
            L_lcfs = (((psi - yb) ** 2) * band).sum() / denom
            psi_n_hat = normalize_flux_torch(psi, axis_sign=axis_sign)
            L_lcfs = L_lcfs + (((psi_n_hat - psi_n_gt) ** 2) * band).sum() / denom
        L_shape = torch.zeros((), device=device)
        if lambda_shape > 0:
            area_h, kappa_h = soft_plasma_moments(psi, R_grid, Z_grid, axis_sign=axis_sign)
            area_g, kappa_g = soft_plasma_moments(
                yb.detach(), R_grid, Z_grid, axis_sign=axis_sign
            )
            area_scale = area_g.detach().abs().mean().clamp_min(1e-6)
            L_shape = (
                ((area_h - area_g) / area_scale) ** 2 + (kappa_h - kappa_g) ** 2
            ).mean()
            rmin_h, rmax_h, zmin_h, zmax_h = soft_lcfs_extents(
                psi, R_grid, Z_grid, axis_sign=axis_sign
            )
            rmin_g, rmax_g, zmin_g, zmax_g = soft_lcfs_extents(
                yb.detach(), R_grid, Z_grid, axis_sign=axis_sign
            )
            width_h = (rmax_h - rmin_h).clamp_min(1e-3)
            width_g = (rmax_g - rmin_g).clamp_min(1e-3)
            area_box_h = width_h * (zmax_h - zmin_h).clamp_min(1e-3)
            area_box_g = width_g * (zmax_g - zmin_g).clamp_min(1e-3)
            box_scale = area_box_g.detach().abs().mean().clamp_min(1e-6)
            L_shape = L_shape + ((area_box_h - area_box_g) / box_scale).pow(2).mean()
        L_vol = torch.zeros((), device=device)
        if lambda_volume > 0:
            area_h, _ = soft_plasma_moments(psi, R_grid, Z_grid, axis_sign=axis_sign)
            area_g, _ = soft_plasma_moments(
                yb.detach(), R_grid, Z_grid, axis_sign=axis_sign
            )
            area_scale = area_g.detach().abs().mean().clamp_min(1e-6)
            rmin_h, rmax_h, zmin_h, zmax_h = soft_lcfs_extents(
                psi, R_grid, Z_grid, axis_sign=axis_sign
            )
            rmin_g, rmax_g, zmin_g, zmax_g = soft_lcfs_extents(
                yb.detach(), R_grid, Z_grid, axis_sign=axis_sign
            )
            area_box_h = (rmax_h - rmin_h).clamp_min(1e-3) * (zmax_h - zmin_h).clamp_min(1e-3)
            area_box_g = (rmax_g - rmin_g).clamp_min(1e-3) * (zmax_g - zmin_g).clamp_min(1e-3)
            box_scale = area_box_g.detach().abs().mean().clamp_min(1e-6)
            L_vol = (
                ((area_h - area_g) / area_scale) ** 2
                + ((area_box_h - area_box_g) / box_scale) ** 2
            ).mean()
        L_cons = torch.zeros((), device=device)
        if lambda_cons_gt > 0 and k_gt is not None:
            _, kappa_h = soft_plasma_moments(psi, R_grid, Z_grid, axis_sign=axis_sign)
            rmin_h, rmax_h, zmin_h, zmax_h = soft_lcfs_extents(
                psi, R_grid, Z_grid, axis_sign=axis_sign
            )
            width_h = (rmax_h - rmin_h).clamp_min(1e-3)
            kappa_box = (zmax_h - zmin_h) / width_h
            area_box = width_h * (zmax_h - zmin_h).clamp_min(1e-3)
            L_cons = _finite_mse(kappa_h, k_gt) + _finite_mse(kappa_box, k_gt)
            if v_gt is not None:
                v_scale = torch.nanmean(torch.abs(v_gt)).clamp_min(1e-3)
                L_cons = L_cons + _finite_mse(area_box / v_scale, v_gt / v_scale)
            if tt_gt is not None and tb_gt is not None:
                tri_t_h, tri_b_h = soft_triangularity(
                    psi, R_grid, Z_grid, axis_sign=axis_sign
                )
                L_cons = L_cons + _finite_mse(tri_t_h, tt_gt) + _finite_mse(tri_b_h, tb_gt)
        loss = (
            L_data
            + 0.25 * L_coeff
            + lambda_qb * L_qb
            + lambda_delta * L_delta
            + lambda_axis * L_axis
            + lambda_z_axis * L_z
            + lambda_li * L_li
            + lambda_lcfs * L_lcfs
            + lambda_shape * L_shape
            + lambda_volume * L_vol
            + lambda_cons_gt * L_cons
        )
        return loss, L_data

    def _gather(idx):
        k = v = tt = tb = None
        if kappa_t is not None:
            k = kappa_t[idx]
            v = vol_t[idx]
            tt = tri_t_t[idx]
            tb = tri_b_t[idx]
        return (
            xs_t[idx],
            zc_t[idx],
            y_t[idx],
            q_t[idx],
            b_t[idx],
            r_t[idx],
            zax_t[idx],
            li_t[idx],
            k,
            v,
            tt,
            tb,
        )

    t0 = time.perf_counter()
    history: list[dict] = []
    log_every = max(1, int(log_every))
    # Quiet long runs: tqdm batch spam can blow agent/terminal capture caps (~1MB).
    use_tqdm = sys.stderr.isatty() and epochs <= 60
    out = output or Path(f"catskills_artifacts/{MODEL_ID}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Optional frame holdout for early-stop on val ψ MSE (probes).
    val_fraction = float(max(0.0, min(0.5, val_fraction)))
    n_val = int(round(n_total * val_fraction)) if val_fraction > 0 else 0
    if n_val > 0 and n_total - n_val < batch_size:
        n_val = 0
    if n_val > 0:
        perm0 = torch.randperm(n_total, device=device)
        val_idx = perm0[:n_val]
        train_idx_pool = perm0[n_val:]
    else:
        val_idx = None
        train_idx_pool = torch.arange(n_total, device=device)
    n_train_frames = int(train_idx_pool.numel())
    n_batches_ep = max(1, (n_train_frames + batch_size - 1) // batch_size)
    hard_train = hard_mask_t[train_idx_pool]
    hard_pool = train_idx_pool[hard_train]
    easy_pool = train_idx_pool[~hard_train]
    use_hard = hard_frac > 0 and hard_pool.numel() > 0

    best_val = float("inf")
    best_state: dict | None = None
    best_res_state: dict | None = None
    best_epoch = 0
    stall = 0
    epochs_ran = 0

    def _sample_batch_idx() -> torch.Tensor:
        if not use_hard:
            return train_idx_pool[torch.randperm(n_train_frames, device=device)]
        # Rebuild a perm that oversamples hard frames to ~hard_frac of each epoch.
        n_hard_ep = min(
            int(hard_pool.numel()),
            max(1, int(round(n_train_frames * float(hard_frac)))),
        )
        n_easy_ep = max(0, n_train_frames - n_hard_ep)
        h = hard_pool[torch.randint(0, hard_pool.numel(), (n_hard_ep,), device=device)]
        if n_easy_ep > 0 and easy_pool.numel() > 0:
            e = easy_pool[torch.randint(0, easy_pool.numel(), (n_easy_ep,), device=device)]
            mix = torch.cat([h, e])
        else:
            mix = h
        return mix[torch.randperm(mix.numel(), device=device)]

    @torch.no_grad()
    def _val_psi_mse() -> float:
        if val_idx is None or val_idx.numel() == 0:
            return float("nan")
        model.eval()
        if residual is not None:
            residual.eval()
        total = 0.0
        n = 0
        for start in range(0, int(val_idx.numel()), batch_size):
            sl = val_idx[start : start + batch_size]
            loss_c, l_data = _batch_loss(*_gather(sl))
            del loss_c
            total += float(l_data.item()) * int(sl.numel())
            n += int(sl.numel())
        model.train()
        if residual is not None:
            residual.train()
        return total / max(n, 1)

    for ep in range(1, epochs + 1):
        epochs_ran = ep
        model.train()
        if residual is not None:
            residual.train()
        loss_sum = 0.0
        n_batches = 0
        perm = _sample_batch_idx()
        batch_iter = range(0, int(perm.numel()), batch_size)
        if use_tqdm:
            batch_iter = tqdm(
                batch_iter,
                desc=f"epoch {ep}/{epochs}",
                unit="batch",
                leave=False,
                total=n_batches_ep,
                mininterval=0.5,
            )
        last_loss = 0.0
        for start in batch_iter:
            idx = perm[start : start + batch_size]
            batch = _gather(idx)
            loss, _ = _batch_loss(*batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach())
            loss_sum += last_loss
            n_batches += 1
            if use_tqdm and (n_batches == 1 or n_batches % 50 == 0):
                batch_iter.set_postfix(loss=f"{last_loss:.3e}")
        sched.step()
        mean_loss = loss_sum / max(n_batches, 1)
        val_mse = _val_psi_mse() if n_val > 0 else float("nan")
        history.append(
            {
                "epoch": ep,
                "loss": mean_loss,
                "val_psi_mse": val_mse,
                "phase": "adamish",
            }
        )
        if ep == 1 or ep == epochs or ep % log_every == 0:
            extra = f"  val_ψ={val_mse:.4e}" if np.isfinite(val_mse) else ""
            print(
                f"[{MODEL_ID}] epoch {ep}/{epochs}  loss={mean_loss:.4e}{extra}",
                flush=True,
            )

        if early_stop_patience > 0 and np.isfinite(val_mse):
            if val_mse < best_val - 1e-8:
                best_val = val_mse
                best_epoch = ep
                stall = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if residual is not None:
                    best_res_state = {
                        k: v.detach().cpu().clone() for k, v in residual.state_dict().items()
                    }
            else:
                stall += 1
                if stall >= early_stop_patience:
                    print(
                        f"[{MODEL_ID}] early-stop at epoch {ep}: "
                        f"val_ψ stalled {stall} (best {best_val:.4e} @ ep {best_epoch})",
                        flush=True,
                    )
                    break

        if epochs >= 100 and ep % 50 == 0:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "residual_state": None if residual is None else residual.state_dict(),
                    "n_features": n_features,
                    "feature_mean": mean,
                    "feature_std": std,
                    "include_thomson": include_thomson,
                    "n_pca": int(n_comp),
                    "use_residual": bool(use_residual),
                    "hidden": list(hidden_t),
                    "activation": activation,
                    "pca_mean": pca.mean_.astype(np.float32),
                    "pca_components": pca.components_.astype(np.float32),
                    "training_params": {
                        "epochs_ran": int(ep),
                        "n_train_frames": n_train_frames,
                        "partial": True,
                    },
                    "history": history,
                    "method": MODEL_ID,
                    "model_version": make_model_version(MODEL_ID),
                    "git_commit": git_full_sha(),
                },
                out.with_name(out.stem + f"_ep{ep}.pt"),
            )
            print(f"[{MODEL_ID}] snapshot {out.stem}_ep{ep}.pt", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
        if residual is not None and best_res_state is not None:
            residual.load_state_dict(best_res_state)
        print(
            f"[{MODEL_ID}] restored best val_ψ={best_val:.4e} @ epoch {best_epoch}",
            flush=True,
        )

    def _build_ckpt(train_s: float, lbfgs_ran: int = 0) -> dict:
        n_shots_caller = len(cache_keys) if cache_keys is not None else len(shots or [])
        tp = {
            "n_train": n_shots_caller,
            "from_feature_cache": cache_keys is not None,
            "n_train_frames": n_total,
            "max_frames": int(max_frames),
            "epochs": int(epochs),
            "epochs_ran": int(epochs_ran),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "optimizer": str(optimizer),
            "weight_decay": float(weight_decay),
            "momentum": float(momentum),
            "hidden_layer_sizes": list(hidden_t),
            "activation": str(activation),
            "lbfgs_steps": int(lbfgs_steps),
            "lbfgs_steps_ran": int(lbfgs_ran),
            "lbfgs_max_frames": int(lbfgs_max_frames),
            "n_pca": int(n_comp),
            "use_residual": bool(use_residual),
            "lambda_delta": float(lambda_delta),
            "lambda_qb": float(lambda_qb),
            "lambda_axis": float(lambda_axis),
            "lambda_z_axis": float(lambda_z_axis),
            "lambda_li": float(lambda_li),
            "lambda_lcfs": float(lambda_lcfs),
            "lambda_shape": float(lambda_shape),
            "lambda_volume": float(lambda_volume),
            "lambda_cons_gt": float(lambda_cons_gt),
            "hard_frac": float(hard_frac),
            "n_hard_keys": int(len(hard_set)),
            "val_fraction": float(val_fraction),
            "early_stop_patience": int(early_stop_patience),
            "best_epoch": int(best_epoch or epochs_ran),
            "best_val_psi_mse": float(best_val) if best_state is not None else None,
            "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
            "seed": int(seed),
            "device": torch.cuda.get_device_name(0),
            "train_s": float(train_s),
            "feature_cache_hits": int(cache_stats.get("hits", 0)),
            "feature_cache_misses": int(cache_stats.get("misses", 0)),
            "git_commit": git_full_sha(),
        }
        return {
            "model_state": model.state_dict(),
            "residual_state": None if residual is None else residual.state_dict(),
            "n_features": n_features,
            "feature_mean": mean,
            "feature_std": std,
            "include_thomson": include_thomson,
            "n_pca": int(n_comp),
            "use_residual": bool(use_residual),
            "hidden": list(hidden_t),
            "activation": activation,
            "pca_mean": pca.mean_.astype(np.float32),
            "pca_components": pca.components_.astype(np.float32),
            "training_params": tp,
            "history": history,
            "method": MODEL_ID,
            "model_version": make_model_version(MODEL_ID),
            "git_commit": tp["git_commit"],
        }

    # Persist AdamW weights before L-BFGS — soft shape/LCFS polish can OOM.
    pre_lbfgs = time.perf_counter() - t0
    torch.save(_build_ckpt(pre_lbfgs, lbfgs_ran=0), out)
    print(f"[{MODEL_ID}] wrote pre-LBFGS {out}  train_s={pre_lbfgs:.1f}s", flush=True)

    lbfgs_ran = 0
    if lbfgs_steps > 0:
        # PINN-classic stage-2 (arxiv:2402.01868). Soft topology losses OOM if the
        # polish set is evaluated as one giant batch — chunk like AdamW.
        pool = train_idx_pool
        n_all_t = int(pool.numel())
        if lbfgs_max_frames > 0 and n_all_t > lbfgs_max_frames:
            n_lb = int(lbfgs_max_frames)
            idx_lb = pool[torch.randperm(n_all_t, device=device)[:n_lb]]
        else:
            n_lb = n_all_t
            idx_lb = pool
        # Soft ψ ops need ~tens of MB per frame of intermediates; keep chunks small.
        lb_chunk = max(1, min(int(batch_size), 128))
        print(
            f"[{MODEL_ID}] L-BFGS polish steps={lbfgs_steps}  "
            f"frames={n_lb}/{n_all_t}  chunk={lb_chunk}",
            flush=True,
        )
        model.train()
        if residual is not None:
            residual.train()
        opt_l = torch.optim.LBFGS(
            params,
            lr=1.0,
            max_iter=20,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        try:
            for step in range(1, lbfgs_steps + 1):
                def closure():
                    opt_l.zero_grad(set_to_none=True)
                    weighted = 0.0
                    for start in range(0, n_lb, lb_chunk):
                        sl = idx_lb[start : start + lb_chunk]
                        loss_c, _ = _batch_loss(*_gather(sl))
                        w = float(sl.numel()) / float(n_lb)
                        (loss_c * w).backward()
                        weighted += float(loss_c.detach()) * w
                    # Grads already accumulated; return value is for LBFGS line search.
                    return torch.tensor(weighted, device=device)

                loss = opt_l.step(closure)
                loss_f = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
                lbfgs_ran = step
                history.append({"epoch": epochs + step, "loss": loss_f, "phase": "lbfgs"})
                if step == 1 or step == lbfgs_steps or step % max(1, lbfgs_steps // 5) == 0:
                    print(
                        f"[{MODEL_ID}] LBFGS {step}/{lbfgs_steps}  loss={loss_f:.4e}",
                        flush=True,
                    )
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            print(
                f"[{MODEL_ID}] L-BFGS OOM ({e}); keeping pre-LBFGS checkpoint",
                flush=True,
            )

    train_s = time.perf_counter() - t0
    ckpt = _build_ckpt(train_s, lbfgs_ran=lbfgs_ran)
    torch.save(ckpt, out)
    print(f"[{MODEL_ID}] wrote {out}  train_s={train_s:.1f}s", flush=True)
    return TealBundle(
        checkpoint=out,
        n_features=n_features,
        feature_mean=mean,
        feature_std=std,
        include_thomson=include_thomson,
        n_pca=int(n_comp),
        use_residual=bool(use_residual),
        device_name=torch.cuda.get_device_name(0),
        training_params=ckpt["training_params"],
        model_version=ckpt["model_version"],
        pca_mean=pca.mean_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        hidden=hidden_t,
        activation=activation,
    )


def load_teal(path: Path, device: Optional[torch.device] = None):
    device = device or require_cuda()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    tp = dict(ckpt.get("training_params") or {})
    hidden = tuple(ckpt.get("hidden") or tp.get("hidden_layer_sizes") or (256, 128))
    activation = str(ckpt.get("activation") or tp.get("activation") or "tanh")
    model = CoeffMLP(
        int(ckpt["n_features"]), int(ckpt["n_pca"]), hidden=hidden, activation=activation
    ).to(device)
    # Older teal ckpts lack axis/li heads — allow partial load.
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing:
        print(f"[{MODEL_ID}] load: missing keys {list(missing)[:6]}…", flush=True)
    if unexpected:
        print(f"[{MODEL_ID}] load: unexpected keys {list(unexpected)[:6]}…", flush=True)
    model.eval()
    residual = None
    if ckpt.get("use_residual") and ckpt.get("residual_state") is not None:
        residual = ResidualMap(int(ckpt["n_features"]), activation=activation).to(device)
        residual.load_state_dict(ckpt["residual_state"])
        residual.eval()
    bundle = TealBundle(
        checkpoint=Path(path),
        n_features=int(ckpt["n_features"]),
        feature_mean=np.asarray(ckpt["feature_mean"], dtype=np.float32),
        feature_std=np.asarray(ckpt["feature_std"], dtype=np.float32),
        include_thomson=bool(ckpt["include_thomson"]),
        n_pca=int(ckpt["n_pca"]),
        use_residual=bool(ckpt.get("use_residual")),
        device_name=str(tp.get("device", "cuda")),
        training_params=tp,
        model_version=str(ckpt.get("model_version") or ""),
        pca_mean=np.asarray(ckpt["pca_mean"], dtype=np.float32),
        pca_components=np.asarray(ckpt["pca_components"], dtype=np.float32),
        hidden=hidden,
        activation=activation,
    )
    return model, residual, bundle, ckpt


def _pca_tensors(bundle: TealBundle, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Device-cached PCA mean/components (avoid H2D every shot)."""
    mean_t = getattr(bundle, "_pca_mean_t", None)
    comp_t = getattr(bundle, "_pca_comp_t", None)
    if (
        mean_t is None
        or comp_t is None
        or mean_t.device != device
        or comp_t.device != device
    ):
        if bundle.pca_mean is None or bundle.pca_components is None:
            ckpt = torch.load(bundle.checkpoint, map_location="cpu", weights_only=False)
            mean_np = np.asarray(ckpt["pca_mean"], dtype=np.float32)
            comp_np = np.asarray(ckpt["pca_components"], dtype=np.float32)
            bundle.pca_mean = mean_np
            bundle.pca_components = comp_np
        else:
            mean_np = bundle.pca_mean
            comp_np = bundle.pca_components
        mean_t = torch.from_numpy(np.asarray(mean_np, dtype=np.float32)).to(device)
        comp_t = torch.from_numpy(np.asarray(comp_np, dtype=np.float32)).to(device)
        bundle._pca_mean_t = mean_t  # type: ignore[attr-defined]
        bundle._pca_comp_t = comp_t  # type: ignore[attr-defined]
    return mean_t, comp_t


def _align_raw_features(bundle: TealBundle, X: np.ndarray) -> np.ndarray:
    """Pad/truncate raw (T, F) to bundle.n_features, then apply .pt mean/std."""
    X = np.asarray(X, dtype=np.float32)
    if X.shape[1] < bundle.n_features:
        X = np.hstack(
            [X, np.zeros((len(X), bundle.n_features - X.shape[1]), dtype=np.float32)]
        )
    elif X.shape[1] > bundle.n_features:
        X = X[:, : bundle.n_features]
    return ((X - bundle.feature_mean) / bundle.feature_std).astype(np.float32)


@torch.inference_mode()
def predict_from_raw_features(
    bundle: TealBundle,
    X_list: list[np.ndarray],
    *,
    model: CoeffMLP | None = None,
    residual: ResidualMap | None = None,
    device: Optional[torch.device] = None,
    max_frames: int = 16384,
) -> list[dict]:
    """GPU predict from **cached raw** features; applies .pt mean/std here.

    Chunks by ``max_frames`` so large batches stay within VRAM while keeping the
    MLP busy (far better util than one-shot packaging).
    """
    if not X_list:
        return []
    device = device or require_cuda()
    if model is None:
        model, residual, bundle, _ = load_teal(bundle.checkpoint, device=device)

    normed = [_align_raw_features(bundle, X) for X in X_list]
    lengths = [int(x.shape[0]) for x in normed]
    Xs = np.concatenate(normed, axis=0)
    pca_mean, pca_comp = _pca_tensors(bundle, device)

    psi_parts: list[np.ndarray] = []
    q_parts: list[np.ndarray] = []
    b_parts: list[np.ndarray] = []
    n_all = Xs.shape[0]
    for start in range(0, n_all, max(1, int(max_frames))):
        sl = slice(start, start + max_frames)
        xb = torch.from_numpy(Xs[sl]).to(device, non_blocking=True)
        z_hat, q_hat, b_hat = model(xb)[:3]
        psi = (z_hat @ pca_comp + pca_mean).view(-1, 65, 65)
        if residual is not None:
            R, Z = _default_grids_torch(device)
            psi = psi + residual(xb, R, Z)
        psi_parts.append(psi.detach().cpu().numpy().astype(np.float32))
        q_parts.append(q_hat.detach().cpu().numpy().astype(np.float32))
        b_parts.append(b_hat.detach().cpu().numpy().astype(np.float32))

    psi_np = np.concatenate(psi_parts, axis=0)
    q_np = np.concatenate(q_parts, axis=0)
    b_np = np.concatenate(b_parts, axis=0)
    out: list[dict] = []
    off = 0
    for T in lengths:
        out.append(
            {
                "psirz": psi_np[off : off + T],
                "q95": q_np[off : off + T],
                "betaN": b_np[off : off + T],
            }
        )
        off += T
    return out


@torch.inference_mode()
def predict_rows(
    bundle: TealBundle,
    rows: list[dict],
    sources: list[str] | None = None,
    *,
    model: CoeffMLP | None = None,
    residual: ResidualMap | None = None,
    device: Optional[torch.device] = None,
    feature_workers: int = 8,
) -> list[dict]:
    """Batched GPU predict from public-test rows (parquet → features → model).

    Prefer cached raw features + ``predict_from_raw_features`` for packaging.
    """
    del sources
    if not rows:
        return []
    if len(rows) == 1 or feature_workers <= 1:
        raw_list = [
            features_from_shot(
                shot_from_row_inputs_only(r),
                include_thomson=bundle.include_thomson,
            ).astype(np.float32)
            for r in rows
        ]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=feature_workers) as pool:
            raw_list = list(
                pool.map(
                    lambda r: features_from_shot(
                        shot_from_row_inputs_only(r),
                        include_thomson=bundle.include_thomson,
                    ).astype(np.float32),
                    rows,
                )
            )
    return predict_from_raw_features(
        bundle, raw_list, model=model, residual=residual, device=device
    )


@torch.inference_mode()
def predict_row(
    bundle: TealBundle,
    row: dict,
    source: str,
    *,
    model: CoeffMLP | None = None,
    residual: ResidualMap | None = None,
    device: Optional[torch.device] = None,
) -> dict:
    return predict_rows(
        bundle,
        [row],
        [source],
        model=model,
        residual=residual,
        device=device,
        feature_workers=1,
    )[0]
