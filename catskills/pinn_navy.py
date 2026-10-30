"""CRC_FEQ_navy — training-time Grad–Shafranov residual model (no PCA), GPU.

See ``MODEL_DESCRIPTION_CRC_FEQ_navy.md``. Jang / Raissi style: MLP maps
(R, Z | diagnostics) → ψ̂; GS residual enters the training loss via autodiff.
Public model ID is opaque (no method keyword); see ``CRC_FEQ_COLORS.md``.
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from catskills.feature_cache import (
    build_feature_matrix_cached,
    build_feature_matrix_from_keys,
)
from catskills.features import features_from_shot, shot_from_row_inputs_only
from catskills.lab_paths import results_dir
from catskills.model_version import git_full_sha, make_model_version
from catskills.physics import MU0, default_grid_from_row
from experiments import EFIT_SCALAR_LABELS

MODEL_ID = "CRC_FEQ_navy"

# DIII-D-ish physical box used to normalize (R, Z) for the network.
R_LO, R_HI = 0.8, 2.5
Z_LO, Z_HI = -1.5, 1.5

# Magnetics layout in ``features``: 20 coils then plasma_current (index 20).
N_COIL = 20
IP_INDEX = 20


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{MODEL_ID} requires a CUDA GPU (torch.cuda.is_available() is False)."
        )
    return torch.device("cuda")


class TanhMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, n_layers: int):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NavyPINN(nn.Module):
    """Parametric PINN: ψ̂(R,Z|c), p'/FF'(ψ_N|c), vacuum ψ(R,Z|coils), q95/βN(c)."""

    def __init__(
        self,
        n_cond: int,
        flux_hidden: int = 128,
        flux_layers: int = 6,
        prof_hidden: int = 64,
        prof_layers: int = 3,
        scalar_hidden: int = 64,
        scalar_layers: int = 3,
        vac_hidden: int = 64,
        vac_layers: int = 3,
        n_coil: int = N_COIL,
        n_fourier: int = 0,
    ):
        super().__init__()
        self.n_cond = n_cond
        self.n_coil = min(n_coil, n_cond)
        self.n_fourier = int(n_fourier)
        # Raw (Rn,Zn) + optional sin/cos Fourier features (off by default; hurt S vs champion).
        rz_dim = 2 + 4 * self.n_fourier  # 2 + 2 axes * 2 (sin/cos) * n_fourier
        self.flux_net = TanhMLP(rz_dim + n_cond, 1, flux_hidden, flux_layers)
        self.p_prime_net = TanhMLP(1 + n_cond, 1, prof_hidden, prof_layers)
        self.ff_prime_net = TanhMLP(1 + n_cond, 1, prof_hidden, prof_layers)
        # Coil → vacuum / free-boundary surrogate (Green-function stand-in).
        self.vacuum_net = TanhMLP(rz_dim + self.n_coil, 1, vac_hidden, vac_layers)
        self.q95_net = TanhMLP(n_cond, 1, scalar_hidden, scalar_layers)
        self.betaN_net = TanhMLP(n_cond, 1, scalar_hidden, scalar_layers)
        # Direct (R,Z) axis head — Consistency campaign teacher for soft-axis / EFIT labels.
        self.axis_net = TanhMLP(n_cond, 2, scalar_hidden, scalar_layers)
        # Internal inductance head (Consistency campaign C3) vs efit_li.
        self.li_net = TanhMLP(n_cond, 1, scalar_hidden, scalar_layers)
        # Elongation head vs derive_frame κ (Consistency — κ is the dead term).
        self.kappa_net = TanhMLP(n_cond, 1, scalar_hidden, scalar_layers)
        # Mean-anchored residual: ψ̂ = ψ̄_train + scale * Δθ (off until set_psi_mean).
        self.mean_anchor = False
        self.delta_scale = nn.Parameter(torch.tensor(0.1))
        self.register_buffer("psi_mean", torch.zeros(65, 65), persistent=True)

    def set_psi_mean(self, psi_mean: np.ndarray | torch.Tensor, *, enabled: bool = True) -> None:
        """Enable mean-anchored residual field ψ̂ = ψ̄ + scale·Δ."""
        t = torch.as_tensor(psi_mean, dtype=torch.float32).detach().clone()
        if t.ndim != 2:
            raise ValueError(f"psi_mean must be 2D, got {tuple(t.shape)}")
        if tuple(self.psi_mean.shape) != tuple(t.shape):
            self.register_buffer("psi_mean", t, persistent=True)
        else:
            self.psi_mean.copy_(t.to(device=self.psi_mean.device))
        self.mean_anchor = bool(enabled)

    def sample_psi_mean(self, R: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """Bilinear sample of ψ̄ at physical (R,Z) points → (N,)."""
        # grid_sample: x↔R, y↔Z in [-1, 1], align_corners with linspace box.
        Rn = 2.0 * (R - R_LO) / (R_HI - R_LO) - 1.0
        Zn = 2.0 * (Z - Z_LO) / (Z_HI - Z_LO) - 1.0
        grid = torch.stack([Rn, Zn], dim=-1).view(1, 1, -1, 2)
        mean = self.psi_mean.to(device=R.device, dtype=R.dtype).view(1, 1, *self.psi_mean.shape[-2:])
        return F.grid_sample(mean, grid, mode="bilinear", align_corners=True, padding_mode="border").view(-1)

    @staticmethod
    def normalize_rz(R: torch.Tensor, Z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        Rn = 2.0 * (R - R_LO) / (R_HI - R_LO) - 1.0
        Zn = 2.0 * (Z - Z_LO) / (Z_HI - Z_LO) - 1.0
        return Rn, Zn

    def fourier_rz(self, Rn: torch.Tensor, Zn: torch.Tensor) -> torch.Tensor:
        """(N,) Rn,Zn in [-1,1] → (N, 2+4*n_fourier) positional features."""
        feats = [Rn.unsqueeze(-1), Zn.unsqueeze(-1)]
        for k in range(self.n_fourier):
            freq = (2.0 ** k) * math.pi
            feats.extend(
                [
                    torch.sin(freq * Rn).unsqueeze(-1),
                    torch.cos(freq * Rn).unsqueeze(-1),
                    torch.sin(freq * Zn).unsqueeze(-1),
                    torch.cos(freq * Zn).unsqueeze(-1),
                ]
            )
        return torch.cat(feats, dim=-1)

    def psi_points(self, R: torch.Tensor, Z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """ψ at scattered points. R,Z (N,), c (N, d) or (d,) broadcast."""
        if c.ndim == 1:
            c = c.unsqueeze(0).expand(R.shape[0], -1)
        Rn, Zn = self.normalize_rz(R, Z)
        inp = torch.cat([self.fourier_rz(Rn, Zn), c], dim=-1)
        delta = self.flux_net(inp).squeeze(-1)
        if self.mean_anchor:
            # ψ̄ is a fixed field. Detach (R,Z) so GS autograd (2nd order) does not
            # need grid_sample backward (unimplemented for higher-order grads).
            mean = self.sample_psi_mean(R.detach(), Z.detach())
            return mean + self.delta_scale * delta
        return delta

    def predict_map(self, c: torch.Tensor, R: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """c (B,d) or (d,), R (W,), Z (H,) → ψ (B,H,W) or (H,W)."""
        single = c.ndim == 1
        if single:
            c = c.unsqueeze(0)
        B = c.shape[0]
        H, W = Z.numel(), R.numel()
        # On the native EFIT grid, add ψ̄ exactly (no resample) for data/Consistency.
        if (
            self.mean_anchor
            and H == int(self.psi_mean.shape[0])
            and W == int(self.psi_mean.shape[1])
        ):
            ZZ, RR = torch.meshgrid(Z, R, indexing="ij")
            RR = RR.reshape(-1)
            ZZ = ZZ.reshape(-1)
            c_exp = c.unsqueeze(1).expand(B, H * W, -1).reshape(B * H * W, -1)
            RR_b = RR.unsqueeze(0).expand(B, -1).reshape(-1)
            ZZ_b = ZZ.unsqueeze(0).expand(B, -1).reshape(-1)
            # Temporarily disable anchor so psi_points returns Δ only.
            was = self.mean_anchor
            self.mean_anchor = False
            try:
                delta = self.psi_points(RR_b, ZZ_b, c_exp).reshape(B, H, W)
            finally:
                self.mean_anchor = was
            mean = self.psi_mean.to(device=c.device, dtype=delta.dtype).unsqueeze(0)
            psi = mean + self.delta_scale * delta
            return psi.squeeze(0) if single else psi
        ZZ, RR = torch.meshgrid(Z, R, indexing="ij")
        RR = RR.reshape(-1)
        ZZ = ZZ.reshape(-1)
        c_exp = c.unsqueeze(1).expand(B, H * W, -1).reshape(B * H * W, -1)
        RR_b = RR.unsqueeze(0).expand(B, -1).reshape(-1)
        ZZ_b = ZZ.unsqueeze(0).expand(B, -1).reshape(-1)
        psi = self.psi_points(RR_b, ZZ_b, c_exp).reshape(B, H, W)
        return psi.squeeze(0) if single else psi

    def profiles(self, psi_n: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if c.ndim == 1:
            c = c.unsqueeze(0).expand(psi_n.shape[0], -1)
        elif c.shape[0] == 1 and psi_n.shape[0] > 1:
            c = c.expand(psi_n.shape[0], -1)
        inp = torch.cat([psi_n.unsqueeze(-1), c], dim=-1)
        return self.p_prime_net(inp).squeeze(-1), self.ff_prime_net(inp).squeeze(-1)

    def scalars(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q95_net(c).squeeze(-1), self.betaN_net(c).squeeze(-1)

    def axis_rz(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Diagnostic-conditioned magnetic-axis (R,Z) in meters (sigmoid-boxed)."""
        raw = self.axis_net(c)
        R = R_LO + (R_HI - R_LO) * torch.sigmoid(raw[..., 0])
        Z = Z_LO + (Z_HI - Z_LO) * torch.sigmoid(raw[..., 1])
        return R, Z

    def li_hat(self, c: torch.Tensor) -> torch.Tensor:
        """Diagnostic-conditioned ℓ_i proxy (positive, typical DIII-D ~0.5–1.5)."""
        return 0.2 + 2.0 * torch.sigmoid(self.li_net(c).squeeze(-1))

    def kappa_hat(self, c: torch.Tensor) -> torch.Tensor:
        """Diagnostic-conditioned elongation (typical DIII-D ~1.2–2.2)."""
        return 1.0 + 1.5 * torch.sigmoid(self.kappa_net(c).squeeze(-1))

    def coil_slice(self, c: torch.Tensor) -> torch.Tensor:
        """Coil currents only (no Ip / Thomson) — free-boundary drivers."""
        return c[..., : self.n_coil]

    def psi_vacuum(self, R: torch.Tensor, Z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Vacuum / exterior flux surrogate conditioned on coils alone."""
        coils = self.coil_slice(c)
        if coils.ndim == 1:
            coils = coils.unsqueeze(0).expand(R.shape[0], -1)
        Rn, Zn = self.normalize_rz(R, Z)
        inp = torch.cat([self.fourier_rz(Rn, Zn), coils], dim=-1)
        return self.vacuum_net(inp).squeeze(-1)


def normalize_flux_torch(psi: torch.Tensor, axis_sign: float = -1.0) -> torch.Tensor:
    """ψ_N in [0,1]; DIII-D axis at ψ min (axis_sign=-1)."""
    flat = psi.reshape(psi.shape[0], -1) if psi.ndim == 3 else psi.reshape(1, -1)
    if axis_sign < 0:
        axis = flat.min(dim=-1).values
        edge = flat.max(dim=-1).values
    else:
        axis = flat.max(dim=-1).values
        edge = flat.min(dim=-1).values
    denom = (edge - axis).abs().clamp_min(1e-8)
    while axis.ndim < psi.ndim:
        axis = axis.unsqueeze(-1)
        denom = denom.unsqueeze(-1)
    psi_n = (psi - axis) / denom
    if axis_sign > 0:
        psi_n = 1.0 - psi_n
    return psi_n.clamp(0.0, 1.0)


def soft_axis_rz(
    psi: torch.Tensor,
    R: torch.Tensor,
    Z: torch.Tensor,
    *,
    axis_sign: float = -1.0,
    temp: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable magnetic axis (R,Z) via spatial softmax.

    DIII-D: axis at ψ minimum → softmax(-ψ/T). Returns (B,) each.
    """
    if psi.ndim == 2:
        psi = psi.unsqueeze(0)
    B, H, W = psi.shape
    logits = ((-psi if axis_sign < 0 else psi) / max(temp, 1e-4)).reshape(B, -1)
    w = torch.softmax(logits, dim=-1)
    ZZ, RR = torch.meshgrid(Z, R, indexing="ij")
    RR = RR.reshape(1, -1)
    ZZ = ZZ.reshape(1, -1)
    return (w * RR).sum(dim=-1), (w * ZZ).sum(dim=-1)


def soft_plasma_moments(
    psi: torch.Tensor,
    R: torch.Tensor,
    Z: torch.Tensor,
    *,
    axis_sign: float = -1.0,
    edge_tau: float = 0.95,
    temp: float = 0.08,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft area + elongation proxies from ψ_N membership (Consistency surrogates)."""
    if psi.ndim == 2:
        psi = psi.unsqueeze(0)
    psi_n = normalize_flux_torch(psi, axis_sign=axis_sign)
    m = torch.sigmoid((edge_tau - psi_n) / max(temp, 1e-4))
    ZZ, RR = torch.meshgrid(Z, R, indexing="ij")
    dR = (R[-1] - R[0]) / max(R.numel() - 1, 1)
    dZ = (Z[-1] - Z[0]) / max(Z.numel() - 1, 1)
    area = (m * dR * dZ).sum(dim=(-1, -2))
    mass = m.sum(dim=(-1, -2)).clamp_min(1e-6)
    R_c = (m * RR).sum(dim=(-1, -2)) / mass
    Z_c = (m * ZZ).sum(dim=(-1, -2)) / mass
    var_R = (m * (RR - R_c[:, None, None]) ** 2).sum(dim=(-1, -2)) / mass
    var_Z = (m * (ZZ - Z_c[:, None, None]) ** 2).sum(dim=(-1, -2)) / mass
    kappa = torch.sqrt(var_Z.clamp_min(1e-8) / var_R.clamp_min(1e-8))
    return area, kappa


def soft_lcfs_extents(
    psi: torch.Tensor,
    R: torch.Tensor,
    Z: torch.Tensor,
    *,
    axis_sign: float = -1.0,
    edge_tau: float = 0.97,
    temp: float = 0.05,
    alpha: float = 40.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable (Rmin,Rmax,Zmin,Zmax) of the soft plasma domain."""
    rmin, rmax, zmin, zmax, _, _ = soft_lcfs_geom(
        psi, R, Z, axis_sign=axis_sign, edge_tau=edge_tau, temp=temp, alpha=alpha
    )
    return rmin, rmax, zmin, zmax


def soft_lcfs_geom(
    psi: torch.Tensor,
    R: torch.Tensor,
    Z: torch.Tensor,
    *,
    axis_sign: float = -1.0,
    edge_tau: float = 0.97,
    temp: float = 0.05,
    alpha: float = 40.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Soft LCFS bbox + R at Zmax/Zmin tips (for triangularity)."""
    if psi.ndim == 2:
        psi = psi.unsqueeze(0)
    B = psi.shape[0]
    psi_n = normalize_flux_torch(psi, axis_sign=axis_sign)
    m = torch.sigmoid((edge_tau - psi_n) / max(temp, 1e-4)).reshape(B, -1)
    ZZ, RR = torch.meshgrid(Z, R, indexing="ij")
    RR = RR.reshape(1, -1).expand(B, -1)
    ZZ = ZZ.reshape(1, -1).expand(B, -1)
    log_m = torch.log(m.clamp_min(1e-6))
    w_rmax = torch.softmax(alpha * RR + log_m, dim=-1)
    w_rmin = torch.softmax(-alpha * RR + log_m, dim=-1)
    w_zmax = torch.softmax(alpha * ZZ + log_m, dim=-1)
    w_zmin = torch.softmax(-alpha * ZZ + log_m, dim=-1)
    rmin = (w_rmin * RR).sum(dim=-1)
    rmax = (w_rmax * RR).sum(dim=-1)
    zmin = (w_zmin * ZZ).sum(dim=-1)
    zmax = (w_zmax * ZZ).sum(dim=-1)
    r_atzmax = (w_zmax * RR).sum(dim=-1)
    r_atzmin = (w_zmin * RR).sum(dim=-1)
    return rmin, rmax, zmin, zmax, r_atzmax, r_atzmin


def soft_triangularity(
    psi: torch.Tensor,
    R: torch.Tensor,
    Z: torch.Tensor,
    *,
    axis_sign: float = -1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft (tri_top, tri_bot) ≈ (Rgeo - R|Zmax|Zmin) / a."""
    rmin, rmax, _, _, r_atzmax, r_atzmin = soft_lcfs_geom(
        psi, R, Z, axis_sign=axis_sign
    )
    rgeo = 0.5 * (rmin + rmax)
    a = (0.5 * (rmax - rmin)).clamp_min(1e-3)
    return (rgeo - r_atzmax) / a, (rgeo - r_atzmin) / a


def soft_li_proxy(
    psi: torch.Tensor,
    R: torch.Tensor,
    Z: torch.Tensor,
    *,
    axis_sign: float = -1.0,
) -> torch.Tensor:
    """Peakedness proxy correlated with ℓ_i: core mass / plasma mass."""
    if psi.ndim == 2:
        psi = psi.unsqueeze(0)
    psi_n = normalize_flux_torch(psi, axis_sign=axis_sign)
    m_all = torch.sigmoid((0.95 - psi_n) / 0.08)
    m_core = torch.sigmoid((0.45 - psi_n) / 0.08)
    return m_core.sum(dim=(-1, -2)) / m_all.sum(dim=(-1, -2)).clamp_min(1e-6)


def _mse_over_var(
    pred: torch.Tensor,
    gt: torch.Tensor,
    *,
    valid: Optional[torch.Tensor] = None,
    use_r2: bool = False,
) -> torch.Tensor:
    """MSE, or MSE/Var(gt) when ``use_r2`` (batch R²-style, Consistency-aligned)."""
    if valid is not None:
        if not bool(valid.any()):
            return torch.zeros((), device=pred.device, dtype=pred.dtype)
        pred = pred[valid]
        gt = gt[valid]
    mse = ((pred - gt) ** 2).mean()
    if not use_r2 or pred.numel() < 2:
        return mse
    var = gt.var(unbiased=False).clamp_min(1e-6)
    return mse / var


def topology_surrogate_losses(
    psi_hat: torch.Tensor,
    psi_gt: torch.Tensor,
    R: torch.Tensor,
    Z: torch.Tensor,
    *,
    axis_sign: float = -1.0,
    lcfs_lo: float = 0.92,
    lcfs_hi: float = 0.995,
    r_axis_gt: Optional[torch.Tensor] = None,
    z_axis_gt: Optional[torch.Tensor] = None,
    kappa_gt: Optional[torch.Tensor] = None,
    volume_gt: Optional[torch.Tensor] = None,
    tri_top_gt: Optional[torch.Tensor] = None,
    tri_bot_gt: Optional[torch.Tensor] = None,
    soft_axis_temp: float = 0.05,
    cons_r2_loss: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score-aligned surrogates: axis, LCFS band, soft shape, mild Consistency.

    Returns (L_axis, L_lcfs, L_shape, L_cons).
    Prefer EFIT / derive_frame axis labels for L_axis when provided.
    L_cons: soft κ + tri vs derive_frame GT; volume soft-to-soft (λ_cons keep small).
    """
    R_h, Z_h = soft_axis_rz(
        psi_hat, R, Z, axis_sign=axis_sign, temp=soft_axis_temp
    )
    if r_axis_gt is not None and z_axis_gt is not None:
        valid = torch.isfinite(r_axis_gt) & torch.isfinite(z_axis_gt)
        if bool(valid.any()):
            if cons_r2_loss:
                L_axis = _mse_over_var(
                    R_h, r_axis_gt, valid=valid, use_r2=True
                ) + _mse_over_var(Z_h, z_axis_gt, valid=valid, use_r2=True)
            else:
                L_axis = (
                    ((R_h - r_axis_gt) / max(R_HI - R_LO, 1e-3)) ** 2
                    + ((Z_h - z_axis_gt) / max(Z_HI - Z_LO, 1e-3)) ** 2
                )[valid].mean()
        else:
            R_g, Z_g = soft_axis_rz(
                psi_gt.detach(), R, Z, axis_sign=axis_sign, temp=soft_axis_temp
            )
            L_axis = (
                ((R_h - R_g) / max(R_HI - R_LO, 1e-3)) ** 2
                + ((Z_h - Z_g) / max(Z_HI - Z_LO, 1e-3)) ** 2
            ).mean()
    else:
        R_g, Z_g = soft_axis_rz(
            psi_gt.detach(), R, Z, axis_sign=axis_sign, temp=soft_axis_temp
        )
        L_axis = (
            ((R_h - R_g) / max(R_HI - R_LO, 1e-3)) ** 2
            + ((Z_h - Z_g) / max(Z_HI - Z_LO, 1e-3)) ** 2
        ).mean()

    psi_n_gt = normalize_flux_torch(psi_gt.detach(), axis_sign=axis_sign)
    band = ((psi_n_gt >= lcfs_lo) & (psi_n_gt <= lcfs_hi)).to(dtype=psi_hat.dtype)
    denom = band.sum().clamp_min(1.0)
    L_lcfs = (((psi_hat - psi_gt) ** 2) * band).sum() / denom
    psi_n_hat = normalize_flux_torch(psi_hat, axis_sign=axis_sign)
    L_lcfs = L_lcfs + (((psi_n_hat - psi_n_gt) ** 2) * band).sum() / denom

    area_h, kappa_h = soft_plasma_moments(psi_hat, R, Z, axis_sign=axis_sign)
    area_g, kappa_g = soft_plasma_moments(psi_gt.detach(), R, Z, axis_sign=axis_sign)
    area_scale = area_g.detach().abs().mean().clamp_min(1e-6)
    L_shape = (
        ((area_h - area_g) / area_scale) ** 2 + (kappa_h - kappa_g) ** 2
    ).mean()

    rmin_h, rmax_h, zmin_h, zmax_h = soft_lcfs_extents(psi_hat, R, Z, axis_sign=axis_sign)
    rmin_g, rmax_g, zmin_g, zmax_g = soft_lcfs_extents(
        psi_gt.detach(), R, Z, axis_sign=axis_sign
    )
    width_h = (rmax_h - rmin_h).clamp_min(1e-3)
    width_g = (rmax_g - rmin_g).clamp_min(1e-3)
    kappa_box_h = (zmax_h - zmin_h) / width_h
    area_box_h = width_h * (zmax_h - zmin_h).clamp_min(1e-3)
    area_box_g = width_g * (zmax_g - zmin_g).clamp_min(1e-3)
    area_box_scale = area_box_g.detach().abs().mean().clamp_min(1e-6)
    L_vol = ((area_box_h - area_box_g) / area_box_scale) ** 2
    L_kappa = torch.zeros((), device=psi_hat.device)
    if kappa_gt is not None:
        valid_k = torch.isfinite(kappa_gt)
        if bool(valid_k.any()):
            # Dual soft κ: LCFS bbox elongation + moment-based κ vs derive_frame.
            _, kappa_mom_h = soft_plasma_moments(psi_hat, R, Z, axis_sign=axis_sign)
            L_kappa = _mse_over_var(
                kappa_box_h, kappa_gt, valid=valid_k, use_r2=cons_r2_loss
            ) + _mse_over_var(
                kappa_mom_h, kappa_gt, valid=valid_k, use_r2=cons_r2_loss
            )

    L_tri = torch.zeros((), device=psi_hat.device)
    if tri_top_gt is not None and tri_bot_gt is not None:
        tri_t_h, tri_b_h = soft_triangularity(psi_hat, R, Z, axis_sign=axis_sign)
        valid_t = torch.isfinite(tri_top_gt) & torch.isfinite(tri_bot_gt)
        if bool(valid_t.any()):
            L_tri = _mse_over_var(
                tri_t_h, tri_top_gt, valid=valid_t, use_r2=cons_r2_loss
            ) + _mse_over_var(
                tri_b_h, tri_bot_gt, valid=valid_t, use_r2=cons_r2_loss
            )

    if kappa_gt is not None or (tri_top_gt is not None and tri_bot_gt is not None):
        L_cons = L_kappa + L_vol.mean() + L_tri
    else:
        L_cons = _soft_cons_bbox(psi_hat, psi_gt, R, Z, axis_sign=axis_sign)
    return L_axis, L_lcfs, L_shape, L_cons


def _soft_cons_bbox(
    psi_hat: torch.Tensor,
    psi_gt: torch.Tensor,
    R: torch.Tensor,
    Z: torch.Tensor,
    *,
    axis_sign: float = -1.0,
) -> torch.Tensor:
    """Fallback Consistency surrogate: soft LCFS bbox hat vs gt."""
    rmin_h, rmax_h, zmin_h, zmax_h = soft_lcfs_extents(psi_hat, R, Z, axis_sign=axis_sign)
    rmin_g, rmax_g, zmin_g, zmax_g = soft_lcfs_extents(
        psi_gt.detach(), R, Z, axis_sign=axis_sign
    )
    width_h = (rmax_h - rmin_h).clamp_min(1e-3)
    width_g = (rmax_g - rmin_g).clamp_min(1e-3)
    kappa_box_h = (zmax_h - zmin_h) / width_h
    kappa_box_g = (zmax_g - zmin_g) / width_g
    area_box_h = width_h * (zmax_h - zmin_h).clamp_min(1e-3)
    area_box_g = width_g * (zmax_g - zmin_g).clamp_min(1e-3)
    area_box_scale = area_box_g.detach().abs().mean().clamp_min(1e-6)
    return (
        (kappa_box_h - kappa_box_g) ** 2
        + ((area_box_h - area_box_g) / area_box_scale) ** 2
    ).mean()


def gs_residual_ad(
    model: NavyPINN,
    R: torch.Tensor,
    Z: torch.Tensor,
    c: torch.Tensor,
    *,
    n_per_sample: Optional[int] = None,
    axis_sign: float = -1.0,
    create_graph: bool = True,
) -> torch.Tensor:
    """Pointwise GS residual via automatic differentiation (Jang/Raissi style).

    ``R``, ``Z`` are (N,); ``c`` is (N, d) or (d,). If ``n_per_sample`` is set,
    ψ is normalized per sample block of that length (batched collocation).
    """
    R = R.requires_grad_(True)
    Z = Z.requires_grad_(True)
    if c.ndim == 1:
        c = c.unsqueeze(0).expand(R.shape[0], -1)
    psi = model.psi_points(R, Z, c)
    ones = torch.ones_like(psi)
    # First-order grads must keep the graph so second derivatives exist (even for val).
    dpsi_dR, dpsi_dZ = torch.autograd.grad(
        psi, (R, Z), grad_outputs=ones, create_graph=True, retain_graph=True
    )
    d2psi_dR2 = torch.autograd.grad(
        dpsi_dR, R, grad_outputs=ones, create_graph=create_graph, retain_graph=True
    )[0]
    d2psi_dZ2 = torch.autograd.grad(
        dpsi_dZ, Z, grad_outputs=ones, create_graph=create_graph, retain_graph=True
    )[0]
    R_safe = R.clamp_min(1e-3)
    delta_star = d2psi_dR2 - (1.0 / R_safe) * dpsi_dR + d2psi_dZ2

    if n_per_sample is None or n_per_sample <= 0 or psi.numel() % n_per_sample != 0:
        psi_min = psi.min()
        psi_max = psi.max()
        if axis_sign < 0:
            psi_n = (psi - psi_min) / (psi_max - psi_min).clamp_min(1e-8)
        else:
            psi_n = (psi_max - psi) / (psi_max - psi_min).clamp_min(1e-8)
    else:
        B = psi.numel() // n_per_sample
        psi_b = psi.view(B, n_per_sample)
        if axis_sign < 0:
            mn = psi_b.min(dim=1, keepdim=True).values
            mx = psi_b.max(dim=1, keepdim=True).values
            psi_n = ((psi_b - mn) / (mx - mn).clamp_min(1e-8)).reshape(-1)
        else:
            mn = psi_b.min(dim=1, keepdim=True).values
            mx = psi_b.max(dim=1, keepdim=True).values
            psi_n = ((mx - psi_b) / (mx - mn).clamp_min(1e-8)).reshape(-1)
    psi_n = psi_n.clamp(0.0, 1.0)
    p_prime, ff_prime = model.profiles(psi_n, c)
    rhs = -MU0 * (R_safe ** 2) * p_prime - ff_prime
    return delta_star - rhs


def delta_star_ad(
    model: NavyPINN,
    R: torch.Tensor,
    Z: torch.Tensor,
    c: torch.Tensor,
    *,
    create_graph: bool = True,
) -> torch.Tensor:
    """Δ*ψ via AD (vacuum exterior uses Δ*ψ ≈ 0)."""
    R = R.requires_grad_(True)
    Z = Z.requires_grad_(True)
    if c.ndim == 1:
        c = c.unsqueeze(0).expand(R.shape[0], -1)
    psi = model.psi_points(R, Z, c)
    ones = torch.ones_like(psi)
    dpsi_dR, dpsi_dZ = torch.autograd.grad(
        psi, (R, Z), grad_outputs=ones, create_graph=True, retain_graph=True
    )
    d2psi_dR2 = torch.autograd.grad(
        dpsi_dR, R, grad_outputs=ones, create_graph=create_graph, retain_graph=True
    )[0]
    d2psi_dZ2 = torch.autograd.grad(
        dpsi_dZ, Z, grad_outputs=ones, create_graph=create_graph, retain_graph=True
    )[0]
    R_safe = R.clamp_min(1e-3)
    return d2psi_dR2 - (1.0 / R_safe) * dpsi_dR + d2psi_dZ2


def sample_boundary_points(
    batch_size: int, n_pts: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (R,Z) on the computational-box boundary (free-boundary proxy)."""
    # Four edges, roughly equal share.
    n_each = max(1, n_pts // 4)
    extras = n_pts - 4 * n_each
    chunks_r, chunks_z = [], []
    # left / right R walls
    for R_wall in (R_LO, R_HI):
        z = torch.empty(batch_size, n_each, device=device).uniform_(Z_LO, Z_HI)
        r = torch.full((batch_size, n_each), R_wall, device=device)
        chunks_r.append(r)
        chunks_z.append(z)
    # bottom / top Z walls
    for Z_wall in (Z_LO, Z_HI):
        r = torch.empty(batch_size, n_each, device=device).uniform_(R_LO, R_HI)
        z = torch.full((batch_size, n_each), Z_wall, device=device)
        chunks_r.append(r)
        chunks_z.append(z)
    R = torch.cat(chunks_r, dim=1)
    Z = torch.cat(chunks_z, dim=1)
    if extras > 0:
        R = torch.cat(
            [R, torch.empty(batch_size, extras, device=device).uniform_(R_LO, R_HI)], dim=1
        )
        Z = torch.cat(
            [Z, torch.empty(batch_size, extras, device=device).uniform_(Z_LO, Z_HI)], dim=1
        )
    return R, Z


def sample_exterior_points(
    batch_size: int, n_pts: int, device: torch.device, edge_frac: float = 0.15
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample points in a near-wall exterior band (vacuum region proxy)."""
    R = torch.empty(batch_size, n_pts, device=device).uniform_(R_LO, R_HI)
    Z = torch.empty(batch_size, n_pts, device=device).uniform_(Z_LO, Z_HI)
    # Push a random subset onto the outer band so Δ* vacuum is meaningful.
    band = (torch.rand(batch_size, n_pts, device=device) < 0.7)
    side = torch.randint(0, 4, (batch_size, n_pts), device=device)
    r_w = edge_frac * (R_HI - R_LO)
    z_w = edge_frac * (Z_HI - Z_LO)
    R = torch.where(band & (side == 0), R_LO + torch.rand_like(R) * r_w, R)
    R = torch.where(band & (side == 1), R_HI - torch.rand_like(R) * r_w, R)
    Z = torch.where(band & (side == 2), Z_LO + torch.rand_like(Z) * z_w, Z)
    Z = torch.where(band & (side == 3), Z_HI - torch.rand_like(Z) * z_w, Z)
    return R, Z


LOSS_TERM_KEYS = (
    "data",
    "gs",
    "coil",
    "ip",
    "th",
    "qb",
    "axis",
    "lcfs",
    "shape",
    "cons",
    "li",
)


class LossEmaNorm:
    """Divide each loss term by an EMA of its magnitude so λ weights stay comparable.

    Keeps collapsed easy terms (qb/data) from starving topology surrogates once their
    raw scale is tiny. Scales are detached — only the normalized residual backprops.
    """

    def __init__(self, beta: float = 0.99, eps: float = 1e-6):
        self.beta = float(beta)
        self.eps = float(eps)
        self.ema: dict[str, float] = {}

    def weighted_sum(
        self,
        terms: dict[str, torch.Tensor],
        lambdas: dict[str, float],
        *,
        update: bool,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss: Optional[torch.Tensor] = None
        scales: dict[str, float] = {}
        for k, L in terms.items():
            v = float(L.detach().item())
            if update:
                prev = self.ema.get(k)
                if prev is None:
                    self.ema[k] = max(v, self.eps)
                else:
                    self.ema[k] = self.beta * prev + (1.0 - self.beta) * v
            scale = max(self.ema.get(k, max(v, self.eps)), self.eps)
            scales[k] = scale
            Ln = L / scale
            w = float(lambdas.get(k, 0.0))
            contrib = w * Ln
            loss = contrib if loss is None else loss + contrib
        assert loss is not None
        return loss, scales


def _batch_losses(
    model: NavyPINN,
    xb: torch.Tensor,
    yb: torch.Tensor,
    qsb: torch.Tensor,
    bsb: torch.Tensor,
    ipb: torch.Tensor,
    R_grid: torch.Tensor,
    Z_grid: torch.Tensor,
    *,
    n_collocation: int,
    n_boundary: int,
    include_thomson: bool,
    lambda_data: float,
    lambda_gs: float,
    lambda_coil: float,
    lambda_ip: float,
    lambda_th: float,
    lambda_qb: float,
    lambda_axis: float = 0.5,
    lambda_lcfs: float = 1.0,
    lambda_shape: float = 0.2,
    lambda_cons: float = 0.1,
    lambda_li: float = 0.0,
    r_axis_gt: Optional[torch.Tensor] = None,
    z_axis_gt: Optional[torch.Tensor] = None,
    kappa_gt: Optional[torch.Tensor] = None,
    volume_gt: Optional[torch.Tensor] = None,
    tri_top_gt: Optional[torch.Tensor] = None,
    tri_bot_gt: Optional[torch.Tensor] = None,
    li_gt: Optional[torch.Tensor] = None,
    soft_axis_temp: float = 0.05,
    cons_r2_loss: bool = False,
    ema_norm: Optional[LossEmaNorm] = None,
    ema_update: bool = True,
    create_graph: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Forward hybrid loss for one batch (GPU)."""
    B = xb.shape[0]
    psi_hat = model.predict_map(xb, R_grid, Z_grid)
    L_data = torch.mean((psi_hat - yb) ** 2)
    L_axis, L_lcfs, L_shape, L_cons = topology_surrogate_losses(
        psi_hat,
        yb,
        R_grid,
        Z_grid,
        axis_sign=-1.0,
        r_axis_gt=r_axis_gt,
        z_axis_gt=z_axis_gt,
        kappa_gt=kappa_gt,
        volume_gt=volume_gt,
        tri_top_gt=tri_top_gt,
        tri_bot_gt=tri_bot_gt,
        soft_axis_temp=soft_axis_temp,
        cons_r2_loss=cons_r2_loss,
    )
    # Axis head teacher: head ↔ labels; soft-axis ↔ head (score-time path is soft/derive).
    if r_axis_gt is not None and z_axis_gt is not None:
        R_head, Z_head = model.axis_rz(xb)
        R_soft, Z_soft = soft_axis_rz(
            psi_hat, R_grid, Z_grid, axis_sign=-1.0, temp=soft_axis_temp
        )
        valid = torch.isfinite(r_axis_gt) & torch.isfinite(z_axis_gt)
        if bool(valid.any()):
            L_head = (
                ((R_head - r_axis_gt) / max(R_HI - R_LO, 1e-3)) ** 2
                + ((Z_head - z_axis_gt) / max(Z_HI - Z_LO, 1e-3)) ** 2
            )
            L_soft = (
                ((R_soft - R_head.detach()) / max(R_HI - R_LO, 1e-3)) ** 2
                + ((Z_soft - Z_head.detach()) / max(Z_HI - Z_LO, 1e-3)) ** 2
            )
            L_axis = L_head[valid].mean() + L_soft[valid].mean()

    L_li = torch.zeros((), device=xb.device)
    if li_gt is not None and lambda_li > 0:
        li_h = model.li_hat(xb)
        li_soft = soft_li_proxy(psi_hat, R_grid, Z_grid, axis_sign=-1.0)
        # Map peaking proxy ~[0,1] into typical ℓ_i range via head teacher.
        valid_li = torch.isfinite(li_gt)
        if bool(valid_li.any()):
            L_li_head = (li_h - li_gt) ** 2
            L_li_soft = (li_soft - (li_h.detach() / 2.0).clamp(0.05, 0.95)) ** 2
            L_li = L_li_head[valid_li].mean() + L_li_soft[valid_li].mean()

    # κ head teacher: head ↔ derive κ; soft bbox/moment κ ↔ head (score path is derive).
    if kappa_gt is not None:
        kap_h = model.kappa_hat(xb)
        rmin_h, rmax_h, zmin_h, zmax_h = soft_lcfs_extents(
            psi_hat, R_grid, Z_grid, axis_sign=-1.0
        )
        kappa_box = (zmax_h - zmin_h) / (rmax_h - rmin_h).clamp_min(1e-3)
        valid_k = torch.isfinite(kappa_gt)
        if bool(valid_k.any()):
            L_kap_head = (kap_h - kappa_gt) ** 2
            L_kap_soft = (kappa_box - kap_h.detach()) ** 2
            L_cons = L_cons + L_kap_head[valid_k].mean() + L_kap_soft[valid_k].mean()

    R_col = torch.empty(B, n_collocation, device=xb.device).uniform_(R_LO, R_HI)
    Z_col = torch.empty(B, n_collocation, device=xb.device).uniform_(Z_LO, Z_HI)
    c_col = xb.unsqueeze(1).expand(B, n_collocation, -1).reshape(-1, xb.shape[-1])
    resid = gs_residual_ad(
        model,
        R_col.reshape(-1),
        Z_col.reshape(-1),
        c_col,
        n_per_sample=n_collocation,
        axis_sign=-1.0,
        create_graph=create_graph,
    )
    L_gs = torch.mean(resid ** 2)

    # --- Coil → vacuum / free-boundary (challenge1 §4.6) ---
    # (1) Exterior vacuum: Δ*ψ ≈ 0 near the box wall (no plasma current source).
    # (2) Boundary match: ψ̂(R,Z|c) ≈ ψ_vac(R,Z|coils) — coils drive the vacuum field.
    n_ext = max(16, n_boundary)
    R_ext, Z_ext = sample_exterior_points(B, n_ext, xb.device)
    c_ext = xb.unsqueeze(1).expand(B, n_ext, -1).reshape(-1, xb.shape[-1])
    delta_ext = delta_star_ad(
        model, R_ext.reshape(-1), Z_ext.reshape(-1), c_ext, create_graph=create_graph
    )
    L_vac = torch.mean(delta_ext ** 2)

    R_bd, Z_bd = sample_boundary_points(B, n_boundary, xb.device)
    c_bd = xb.unsqueeze(1).expand(B, n_boundary, -1).reshape(-1, xb.shape[-1])
    psi_bd = model.psi_points(R_bd.reshape(-1), Z_bd.reshape(-1), c_bd)
    psi_vac = model.psi_vacuum(R_bd.reshape(-1), Z_bd.reshape(-1), c_bd)
    L_coil_bc = torch.mean((psi_bd - psi_vac) ** 2)
    L_coil = L_vac + L_coil_bc

    psi_n_map = normalize_flux_torch(psi_hat, axis_sign=-1.0)
    psi_flat = psi_n_map.reshape(B, -1)
    idx = torch.randint(0, psi_flat.shape[1], (min(256, psi_flat.shape[1]),), device=xb.device)
    psi_s = psi_flat[:, idx].reshape(-1)
    c_s = xb.unsqueeze(1).expand(B, idx.numel(), -1).reshape(-1, xb.shape[-1])
    p_p, ff_p = model.profiles(psi_s, c_s)
    j_proxy = (p_p.abs() + ff_p.abs()).reshape(B, -1).mean(dim=-1)
    ip_n = ipb.abs() / (ipb.abs().mean().clamp_min(1.0))
    j_n = j_proxy / (j_proxy.mean().detach().clamp_min(1e-6))
    L_ip = torch.mean((j_n - ip_n) ** 2)

    if include_thomson and xb.shape[1] > 21:
        th = xb[:, 21:]
        pe_proxy = th.abs().mean(dim=-1)
        pe_n = pe_proxy / (pe_proxy.mean().detach().clamp_min(1e-6))
        pp_n = p_p.abs().reshape(B, -1).mean(dim=-1)
        pp_n = pp_n / (pp_n.mean().detach().clamp_min(1e-6))
        L_th = torch.mean((pp_n - pe_n) ** 2)
    else:
        L_th = torch.zeros((), device=xb.device)

    q_hat, b_hat = model.scalars(xb)
    L_qb = torch.mean((q_hat - qsb) ** 2) + torch.mean((b_hat - bsb) ** 2)

    terms = {
        "data": L_data,
        "gs": L_gs,
        "coil": L_coil,
        "ip": L_ip,
        "th": L_th,
        "qb": L_qb,
        "axis": L_axis,
        "lcfs": L_lcfs,
        "shape": L_shape,
        "cons": L_cons,
        "li": L_li,
    }
    lambdas = {
        "data": lambda_data,
        "gs": lambda_gs,
        "coil": lambda_coil,
        "ip": lambda_ip,
        "th": lambda_th,
        "qb": lambda_qb,
        "axis": lambda_axis,
        "lcfs": lambda_lcfs,
        "shape": lambda_shape,
        "cons": lambda_cons,
        "li": lambda_li,
    }
    if ema_norm is not None:
        loss, scales = ema_norm.weighted_sum(terms, lambdas, update=ema_update)
    else:
        loss = (
            lambda_data * L_data
            + lambda_gs * L_gs
            + lambda_coil * L_coil
            + lambda_ip * L_ip
            + lambda_th * L_th
            + lambda_qb * L_qb
            + lambda_axis * L_axis
            + lambda_lcfs * L_lcfs
            + lambda_shape * L_shape
            + lambda_cons * L_cons
            + lambda_li * L_li
        )
        scales = {k: 1.0 for k in terms}

    parts = {
        "loss": loss.detach(),
        "data": L_data.detach(),
        "gs": L_gs.detach(),
        "coil": L_coil.detach(),
        "vac": L_vac.detach(),
        "coil_bc": L_coil_bc.detach(),
        "ip": L_ip.detach(),
        "th": L_th.detach(),
        "qb": L_qb.detach(),
        "axis": L_axis.detach(),
        "lcfs": L_lcfs.detach(),
        "shape": L_shape.detach(),
        "cons": L_cons.detach(),
        "li": L_li.detach(),
        **{f"ema_{k}": torch.tensor(scales[k], device=loss.device) for k in scales},
    }
    return loss, parts


@dataclass
class NavyBundle:
    checkpoint: Path
    n_features: int
    feature_mean: np.ndarray
    feature_std: np.ndarray
    include_thomson: bool
    device_name: str
    training_params: dict
    method: str = MODEL_ID
    model_version: str = field(default="")


class FrameDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        q95: np.ndarray,
        betaN: np.ndarray,
        ip: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        r_axis: Optional[np.ndarray] = None,
        z_axis: Optional[np.ndarray] = None,
        kappa: Optional[np.ndarray] = None,
        volume: Optional[np.ndarray] = None,
        tri_top: Optional[np.ndarray] = None,
        tri_bot: Optional[np.ndarray] = None,
        li: Optional[np.ndarray] = None,
    ):
        self.X = ((X - mean) / std).astype(np.float32)
        self.Y = Y.astype(np.float32)
        self.q95 = q95.astype(np.float32)
        self.betaN = betaN.astype(np.float32)
        self.ip = ip.astype(np.float32)
        n = len(self.X)

        def _or_nan(a: Optional[np.ndarray]) -> np.ndarray:
            if a is None:
                return np.full(n, np.nan, dtype=np.float32)
            return np.asarray(a, dtype=np.float32)

        self.r_axis = _or_nan(r_axis)
        self.z_axis = _or_nan(z_axis)
        self.kappa = _or_nan(kappa)
        self.volume = _or_nan(volume)
        self.tri_top = _or_nan(tri_top)
        self.tri_bot = _or_nan(tri_bot)
        self.li = _or_nan(li)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        return (
            self.X[i],
            self.Y[i],
            self.q95[i],
            self.betaN[i],
            self.ip[i],
            self.r_axis[i],
            self.z_axis[i],
            self.kappa[i],
            self.volume[i],
            self.tri_top[i],
            self.tri_bot[i],
            self.li[i],
        )


def _default_grids_torch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    R = torch.linspace(R_LO, R_HI, 65, device=device, dtype=torch.float32)
    Z = torch.linspace(Z_LO, Z_HI, 65, device=device, dtype=torch.float32)
    return R, Z


def _default_tb_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return results_dir() / "tensorboard" / MODEL_ID / ts


def _eval_loader(
    model: NavyPINN,
    loader: DataLoader,
    device: torch.device,
    R_grid: torch.Tensor,
    Z_grid: torch.Tensor,
    loss_kwargs: dict,
) -> dict[str, float]:
    """Val metrics; GS residual still needs AD w.r.t. (R,Z) but not create_graph."""
    model.eval()
    totals = {
        k: 0.0
        for k in (
            "loss", "data", "gs", "coil", "ip", "th", "qb",
            "axis", "lcfs", "shape", "cons", "li",
        )
    }
    n = 0
    kw = {**loss_kwargs, "create_graph": False, "ema_update": False}
    for xb, yb, qsb, bsb, ipb, r_ax, z_ax, kap, vol, tri_t, tri_b, li_v in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        qsb = qsb.to(device, non_blocking=True)
        bsb = bsb.to(device, non_blocking=True)
        ipb = ipb.to(device, non_blocking=True)
        _, parts = _batch_losses(
            model,
            xb,
            yb,
            qsb,
            bsb,
            ipb,
            R_grid,
            Z_grid,
            r_axis_gt=r_ax.to(device, non_blocking=True),
            z_axis_gt=z_ax.to(device, non_blocking=True),
            kappa_gt=kap.to(device, non_blocking=True),
            volume_gt=vol.to(device, non_blocking=True),
            tri_top_gt=tri_t.to(device, non_blocking=True),
            tri_bot_gt=tri_b.to(device, non_blocking=True),
            li_gt=li_v.to(device, non_blocking=True),
            **kw,
        )
        for k in totals:
            totals[k] += float(parts[k])
        n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in totals.items()}


def make_optimizer(
    model: nn.Module,
    *,
    name: str,
    lr: float,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """Build optimizer by name (adam / adamw / sgd / radam / nadam)."""
    key = (name or "adam").lower().strip()
    params = model.parameters()
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
    raise ValueError(
        f"Unknown optimizer {name!r}; choose adam|adamw|sgd|radam|nadam"
    )


def train_navy(
    shots: Optional[list[dict]] = None,
    *,
    cache_keys: Optional[list[str]] = None,
    include_thomson: bool = True,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    optimizer: str = "adam",
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    n_collocation: int = 256,
    n_boundary: int = 256,
    max_frames: int = 0,
    val_fraction: float = 0.15,
    lambda_data: float = 1.0,
    lambda_gs: float = 0.05,
    lambda_coil: float = 0.05,
    lambda_ip: float = 0.01,
    lambda_th: float = 0.01,
    lambda_qb: float = 1.0,
    lambda_axis: float = 0.5,
    lambda_lcfs: float = 1.0,
    lambda_shape: float = 0.2,
    lambda_cons: float = 0.1,
    lambda_li: float = 0.0,
    n_fourier: int = 0,
    use_cons_gt: bool = True,
    use_cons_tri: bool = False,
    axis_gt_source: str = "efit",
    soft_axis_temp: float = 0.05,
    mean_anchor: bool = False,
    cons_r2_loss: bool = False,
    init_checkpoint: Optional[Path] = None,
    normalize_loss: bool = False,
    loss_ema_beta: float = 0.99,
    early_stop_patience: int = 10,
    early_stop_metric: str = "val_data",
    seed: int = 42,
    output: Path = Path("catskills_artifacts/CRC_FEQ_navy.pt"),
    log_dir: Optional[Path] = None,
) -> NavyBundle:
    device = require_cuda()
    torch.manual_seed(seed)
    np.random.seed(seed)

    if cache_keys is not None:
        print(
            f"[{MODEL_ID}] loading feature cache keys "
            f"(no parquet decode; {len(cache_keys)} shots)…",
            flush=True,
        )
        X, Y, S, _, cache_stats = build_feature_matrix_from_keys(
            cache_keys, include_thomson=include_thomson, progress=True
        )
    elif shots is not None:
        print(
            f"[{MODEL_ID}] building feature matrix on CPU "
            f"(magnetics+Thomson→EFIT times; per-shot disk cache)…",
            flush=True,
        )
        X, Y, S, _, cache_stats = build_feature_matrix_cached(
            shots, include_thomson=include_thomson, progress=True
        )
    else:
        raise ValueError("train_navy requires shots= or cache_keys=")
    print(
        f"[{MODEL_ID}] feature cache: hits={cache_stats['hits']}  "
        f"misses={cache_stats['misses']}  dir={cache_stats['cache_dir']}",
        flush=True,
    )
    i_q = EFIT_SCALAR_LABELS.index("efit_q95")
    i_b = EFIT_SCALAR_LABELS.index("efit_beta_n")
    i_r = EFIT_SCALAR_LABELS.index("efit_r_axis")
    i_z = EFIT_SCALAR_LABELS.index("efit_z_axis")
    i_li = EFIT_SCALAR_LABELS.index("efit_li")
    q95 = np.nan_to_num(S[:, i_q], nan=0.0)
    betaN = np.nan_to_num(S[:, i_b], nan=0.0)
    # EFIT axis / li labels — unclipped Consistency supervision.
    r_axis = S[:, i_r].astype(np.float32)
    z_axis = S[:, i_z].astype(np.float32)
    li_gt = S[:, i_li].astype(np.float32)
    # plasma_current is last magnetics channel (index 20) before Thomson pad.
    ip = X[:, 20].astype(np.float32)

    init_ck: Optional[dict] = None
    if init_checkpoint is not None:
        init_path = Path(init_checkpoint)
        print(f"[{MODEL_ID}] finetune init: {init_path}", flush=True)
        init_ck = torch.load(init_path, map_location="cpu", weights_only=False)
        mean = np.asarray(init_ck["feature_mean"], dtype=np.float32)
        std = np.asarray(init_ck["feature_std"], dtype=np.float32)
        n_fourier = int((init_ck.get("training_params") or {}).get("n_fourier", n_fourier))
    else:
        mean = np.mean(X, axis=0, dtype=np.float64)
        std = np.std(X, axis=0, dtype=np.float64)
        std = np.where(~np.isfinite(std) | (std < 1e-6), 1.0, std)
        mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    n_all = len(X)
    if max_frames > 0 and n_all > max_frames:
        idx = np.random.RandomState(seed).choice(n_all, size=max_frames, replace=False)
        idx.sort()
        X, Y = X[idx], Y[idx]
        q95, betaN, ip = q95[idx], betaN[idx], ip[idx]
        r_axis, z_axis, li_gt = r_axis[idx], z_axis[idx], li_gt[idx]
        print(
            f"[{MODEL_ID}] subsampled frames {n_all} → {len(X)} (max_frames={max_frames})",
            flush=True,
        )

    kappa_gt = np.full(len(Y), np.nan, dtype=np.float32)
    volume_gt = np.full(len(Y), np.nan, dtype=np.float32)
    tri_top_gt = np.full(len(Y), np.nan, dtype=np.float32)
    tri_bot_gt = np.full(len(Y), np.nan, dtype=np.float32)
    der_r_axis = np.full(len(Y), np.nan, dtype=np.float32)
    der_z_axis = np.full(len(Y), np.nan, dtype=np.float32)
    if use_cons_gt:
        from catskills.cons_gt_cache import build_cons_scalars_for_maps

        print(
            f"[{MODEL_ID}] Consistency GT via derive_frame (disk-cached)…",
            flush=True,
        )
        cons_arr, cons_stats = build_cons_scalars_for_maps(Y, progress=True)
        kappa_gt = cons_arr["kappa"]
        volume_gt = cons_arr["volume"]
        tri_top_gt = cons_arr["tri_top"]
        tri_bot_gt = cons_arr["tri_bot"]
        der_r_axis = cons_arr["R_axis"]
        der_z_axis = cons_arr["Z_axis"]
        print(
            f"[{MODEL_ID}] cons GT cache: hits={cons_stats['hits']}  "
            f"misses={cons_stats['misses']}  "
            f"finite_κ={int(np.isfinite(kappa_gt).sum())}/{len(kappa_gt)}",
            flush=True,
        )
    if not use_cons_tri:
        tri_top_gt = np.full(len(Y), np.nan, dtype=np.float32)
        tri_bot_gt = np.full(len(Y), np.nan, dtype=np.float32)

    axis_src = (axis_gt_source or "efit").lower().strip()
    if axis_src == "derive":
        r_axis_sup = der_r_axis
        z_axis_sup = der_z_axis
        print(f"[{MODEL_ID}] axis GT source: derive_frame (scorer-aligned)", flush=True)
    else:
        r_axis_sup = r_axis
        z_axis_sup = z_axis
        print(f"[{MODEL_ID}] axis GT source: EFIT labels", flush=True)

    ds = FrameDataset(
        X, Y, q95, betaN, ip, mean, std,
        r_axis=r_axis_sup,
        z_axis=z_axis_sup,
        kappa=kappa_gt,
        volume=volume_gt,
        tri_top=tri_top_gt,
        tri_bot=tri_bot_gt,
        li=li_gt,
    )
    n_total = len(ds)
    n_val = max(1, int(round(n_total * val_fraction))) if val_fraction > 0 else 0
    n_val = min(n_val, n_total - 1) if n_total > 1 else 0
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(seed)
    if n_val > 0:
        train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)
    else:
        train_ds, val_ds = ds, None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
        pin_memory=True, num_workers=0,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, drop_last=False,
            pin_memory=True, num_workers=0,
        )

    model = NavyPINN(n_cond=X.shape[1], n_fourier=n_fourier).to(device)
    if init_ck is not None:
        missing, unexpected = model.load_state_dict(init_ck["model_state"], strict=False)
        print(
            f"[{MODEL_ID}] loaded init weights  missing={list(missing)[:6]}"
            f"{'…' if len(missing) > 6 else ''}  unexpected={len(unexpected)}",
            flush=True,
        )
    if mean_anchor:
        # Prefer train-fold mean (manifold anchor); overwrite any init buffer.
        model.set_psi_mean(Y.mean(axis=0), enabled=True)
        print(
            f"[{MODEL_ID}] mean-anchor ON  ψ̄ range "
            f"[{float(model.psi_mean.min()):.3f}, {float(model.psi_mean.max()):.3f}]",
            flush=True,
        )
    opt = make_optimizer(
        model,
        name=optimizer,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    R_grid, Z_grid = _default_grids_torch(device)

    tb_dir = Path(log_dir) if log_dir is not None else _default_tb_dir()
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    n_shots_caller = (
        len(cache_keys) if cache_keys is not None else len(shots or [])
    )
    n_params = sum(p.numel() for p in model.parameters())
    print("=" * 64, flush=True)
    print(f"  {MODEL_ID} training", flush=True)
    print(f"  CUDA device:     {torch.cuda.get_device_name(0)}", flush=True)
    print(f"  params:          {n_params:,}", flush=True)
    print(f"  shots (caller):  {n_shots_caller}", flush=True)
    print(f"  frames total:    {n_total}  (from {n_all} available before cap)", flush=True)
    print(f"  train frames:    {n_train}  ({100 * n_train / max(n_total, 1):.1f}%)", flush=True)
    print(f"  val frames:      {n_val}  ({100 * n_val / max(n_total, 1):.1f}%)", flush=True)
    print(f"  epochs/batch:    {epochs} / {batch_size}", flush=True)
    print(
        f"  optimizer:       {optimizer}  lr={lr}  wd={weight_decay}"
        + (f"  momentum={momentum}" if str(optimizer).lower() == "sgd" else ""),
        flush=True,
    )
    print(f"  collocation:     {n_collocation} interior + {n_boundary} boundary (GPU AD)", flush=True)
    print(f"  λ_coil (vac+bc): {lambda_coil}", flush=True)
    print(
        f"  λ_topo:          axis={lambda_axis}  lcfs={lambda_lcfs}  "
        f"shape={lambda_shape}  cons={lambda_cons}  li={lambda_li}",
        flush=True,
    )
    print(
        f"  n_fourier:       {n_fourier}  cons_gt={'ON' if use_cons_gt else 'OFF'}  "
        f"cons_tri={'ON' if use_cons_tri else 'OFF'}  axis_gt={axis_src}  "
        f"soft_axis_T={soft_axis_temp}  mean_anchor={'ON' if mean_anchor else 'OFF'}  "
        f"cons_r2={'ON' if cons_r2_loss else 'OFF'}",
        flush=True,
    )
    if init_checkpoint is not None:
        print(f"  init_checkpoint: {init_checkpoint}", flush=True)
    print(
        f"  λ_qb / data:     qb={lambda_qb}  data={lambda_data}  "
        f"EMA-norm={'ON β='+str(loss_ema_beta) if normalize_loss else 'OFF'}",
        flush=True,
    )
    print(
        f"  early-stop:      metric={early_stop_metric}  patience={early_stop_patience}"
        if early_stop_patience > 0
        else "  early-stop:      OFF",
        flush=True,
    )
    print(f"  TensorBoard:     {tb_dir}", flush=True)
    print(f"  Launch TB with:  poetry run tensorboard --logdir {tb_dir.parent}", flush=True)
    print("  NOTE: GPU is idle during shot load/feature build; watch util once batch bars move.", flush=True)
    print("=" * 64, flush=True)

    ema_norm = LossEmaNorm(beta=loss_ema_beta) if normalize_loss else None
    loss_kwargs = dict(
        n_collocation=n_collocation,
        n_boundary=n_boundary,
        include_thomson=include_thomson,
        lambda_data=lambda_data,
        lambda_gs=lambda_gs,
        lambda_coil=lambda_coil,
        lambda_ip=lambda_ip,
        lambda_th=lambda_th,
        lambda_qb=lambda_qb,
        lambda_axis=lambda_axis,
        lambda_lcfs=lambda_lcfs,
        lambda_shape=lambda_shape,
        lambda_cons=lambda_cons,
        lambda_li=lambda_li,
        soft_axis_temp=soft_axis_temp,
        cons_r2_loss=cons_r2_loss,
        ema_norm=ema_norm,
    )

    writer.add_text(
        "run/config",
        (
            f"device={torch.cuda.get_device_name(0)}  shots={n_shots_caller}  "
            f"train_frames={n_train}  val_frames={n_val}  "
            f"epochs={epochs}  batch={batch_size}  lr={lr}  "
            f"N_col={n_collocation}  N_bdry={n_boundary}  "
            f"lambda_coil={lambda_coil}  lambda_qb={lambda_qb}  "
            f"lambda_axis={lambda_axis}  lambda_lcfs={lambda_lcfs}  "
            f"lambda_shape={lambda_shape}  lambda_cons={lambda_cons}  "
            f"n_fourier={n_fourier}  cons_gt={use_cons_gt}  "
            f"loss_ema_norm={normalize_loss}  beta={loss_ema_beta}  "
            f"thomson={include_thomson}"
        ),
        0,
    )

    model.train()
    t0 = time.perf_counter()
    history: list[dict] = []
    global_step = 0
    best_metric = float("inf")
    best_state: Optional[dict] = None
    best_epoch = 0
    stall = 0
    epochs_ran = 0
    epoch_bar = tqdm(range(1, epochs + 1), desc="epochs", unit="ep", leave=True)

    for epoch in epoch_bar:
        epochs_ran = epoch
        totals = {
            k: 0.0
            for k in (
                "loss", "data", "gs", "coil", "ip", "th", "qb",
                "axis", "lcfs", "shape", "cons", "li",
            )
        }
        n_batches = 0
        batch_bar = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{epochs} train",
            unit="batch",
            leave=False,
        )
        for xb, yb, qsb, bsb, ipb, r_ax, z_ax, kap, vol, tri_t, tri_b, li_v in batch_bar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            qsb = qsb.to(device, non_blocking=True)
            bsb = bsb.to(device, non_blocking=True)
            ipb = ipb.to(device, non_blocking=True)

            loss, parts = _batch_losses(
                model,
                xb,
                yb,
                qsb,
                bsb,
                ipb,
                R_grid,
                Z_grid,
                r_axis_gt=r_ax.to(device, non_blocking=True),
                z_axis_gt=z_ax.to(device, non_blocking=True),
                kappa_gt=kap.to(device, non_blocking=True),
                volume_gt=vol.to(device, non_blocking=True),
                tri_top_gt=tri_t.to(device, non_blocking=True),
                tri_bot_gt=tri_b.to(device, non_blocking=True),
                li_gt=li_v.to(device, non_blocking=True),
                ema_update=True,
                **loss_kwargs,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            for k in totals:
                totals[k] += float(parts[k])
            n_batches += 1
            global_step += 1

            mem_gb = torch.cuda.memory_allocated() / 1e9
            if global_step % 5 == 0:
                writer.add_scalar("batch/loss", float(parts["loss"]), global_step)
                writer.add_scalar("batch/data", float(parts["data"]), global_step)
                writer.add_scalar("batch/gs", float(parts["gs"]), global_step)
                writer.add_scalar("batch/coil", float(parts["coil"]), global_step)
                writer.add_scalar("batch/qb", float(parts["qb"]), global_step)
                writer.add_scalar("batch/axis", float(parts["axis"]), global_step)
                writer.add_scalar("batch/lcfs", float(parts["lcfs"]), global_step)
                writer.add_scalar("batch/shape", float(parts["shape"]), global_step)
                writer.add_scalar("batch/cons", float(parts["cons"]), global_step)
                for k in LOSS_TERM_KEYS:
                    ek = f"ema_{k}"
                    if ek in parts:
                        writer.add_scalar(f"ema_scale/{k}", float(parts[ek]), global_step)
                writer.add_scalar("system/gpu_mem_alloc_GB", mem_gb, global_step)
                writer.add_scalar(
                    "system/gpu_mem_reserved_GB",
                    torch.cuda.memory_reserved() / 1e9,
                    global_step,
                )

            batch_bar.set_postfix(
                loss=f"{float(parts['loss']):.3e}",
                axis=f"{float(parts['axis']):.3e}",
                cons=f"{float(parts['cons']):.3e}",
                gpuGB=f"{mem_gb:.2f}",
                refresh=False,
            )

        sched.step()
        train_avg = {k: v / max(n_batches, 1) for k, v in totals.items()}

        val_avg = {}
        if val_loader is not None:
            val_avg = _eval_loader(
                model, val_loader, device, R_grid, Z_grid, loss_kwargs
            )

        lr_now = float(opt.param_groups[0]["lr"])
        for k, v in train_avg.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_avg.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        writer.add_scalar("train/lr", lr_now, epoch)
        writer.add_scalar("system/gpu_mem_alloc_GB_epoch", torch.cuda.memory_allocated() / 1e9, epoch)
        writer.flush()

        row = {"epoch": epoch, "lr": lr_now, **{f"train_{k}": v for k, v in train_avg.items()}}
        row.update({f"val_{k}": v for k, v in val_avg.items()})
        history.append(row)

        postfix = {
            "tr_loss": f"{train_avg['loss']:.3e}",
            "tr_data": f"{train_avg['data']:.3e}",
        }
        if val_avg:
            postfix["va_loss"] = f"{val_avg['loss']:.3e}"
            postfix["va_cons"] = f"{val_avg.get('cons', float('nan')):.3e}"
            postfix["va_axis"] = f"{val_avg.get('axis', float('nan')):.3e}"

        # Early-stop on a fixed (non-EMA) metric — default val_data (protect ψ).
        # Composite: val_cons_proxy = val_axis + 0.25*val_cons + 0.5*val_data
        metric_key = early_stop_metric
        if metric_key == "val_cons_proxy" and val_avg:
            metric_val = (
                float(val_avg.get("axis", float("nan")))
                + 0.25 * float(val_avg.get("cons", float("nan")))
                + 0.5 * float(val_avg.get("data", float("nan")))
            )
        elif metric_key.startswith("val_") and val_avg:
            metric_val = float(val_avg.get(metric_key[4:], float("nan")))
        elif metric_key.startswith("train_"):
            metric_val = float(train_avg.get(metric_key[6:], float("nan")))
        else:
            metric_val = float(train_avg.get(metric_key, float("nan")))
        if early_stop_patience > 0 and np.isfinite(metric_val):
            if metric_val < best_metric - 1e-6:
                best_metric = metric_val
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                stall = 0
            else:
                stall += 1
            postfix["best"] = f"{best_metric:.3e}@ep{best_epoch}"
            if stall >= early_stop_patience:
                epoch_bar.set_postfix(postfix, refresh=True)
                print(
                    f"[{MODEL_ID}] early-stop at epoch {epoch}: "
                    f"{metric_key} stalled {stall} ep (best {best_metric:.4e} @ ep {best_epoch})",
                    flush=True,
                )
                break

        epoch_bar.set_postfix(postfix, refresh=True)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"[{MODEL_ID}] restored best checkpoint "
            f"({early_stop_metric}={best_metric:.4e} @ epoch {best_epoch})",
            flush=True,
        )

    train_s = time.perf_counter() - t0
    writer.add_scalar("run/train_wall_s", train_s, 0)
    writer.add_scalar("run/best_epoch", float(best_epoch or epochs_ran), 0)
    writer.close()

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tp = {
        "n_train": int(n_shots_caller),
        "from_feature_cache": bool(cache_keys is not None),
        "n_train_frames": int(n_train),
        "n_val_frames": int(n_val),
        "val_fraction": float(val_fraction),
        "n_train_frames_available": int(n_all),
        "max_frames": max_frames,
        "epochs": epochs,
        "epochs_ran": int(epochs_ran),
        "best_epoch": int(best_epoch or epochs_ran),
        "early_stop_patience": int(early_stop_patience),
        "early_stop_metric": early_stop_metric,
        "best_early_stop_metric": float(best_metric) if best_state is not None else None,
        "batch_size": batch_size,
        "optimizer": str(optimizer),
        "weight_decay": float(weight_decay),
        "momentum": float(momentum),
        "lr": lr,
        "n_collocation": n_collocation,
        "n_boundary": n_boundary,
        "lambda_data": lambda_data,
        "lambda_gs": lambda_gs,
        "lambda_coil": lambda_coil,
        "lambda_ip": lambda_ip,
        "lambda_th": lambda_th,
        "lambda_qb": lambda_qb,
        "lambda_axis": lambda_axis,
        "lambda_lcfs": lambda_lcfs,
        "lambda_shape": lambda_shape,
        "lambda_cons": lambda_cons,
        "lambda_li": lambda_li,
        "normalize_loss": bool(normalize_loss),
        "loss_ema_beta": float(loss_ema_beta),
        "flux_hidden": 128,
        "flux_layers": 6,
        "n_fourier": int(n_fourier),
        "use_cons_gt": bool(use_cons_gt),
        "use_cons_tri": bool(use_cons_tri),
        "axis_gt_source": axis_src,
        "soft_axis_temp": float(soft_axis_temp),
        "mean_anchor": bool(mean_anchor),
        "cons_r2_loss": bool(cons_r2_loss),
        "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
        "activation": "tanh",
        "include_thomson": include_thomson,
        "device": torch.cuda.get_device_name(0),
        "train_s": train_s,
        "seed": seed,
        "tensorboard_logdir": str(tb_dir),
        "feature_cache_hits": int(cache_stats["hits"]),
        "feature_cache_misses": int(cache_stats["misses"]),
        "feature_cache_dir": str(cache_stats["cache_dir"]),
        "git_commit": git_full_sha(),
    }
    ckpt = {
        "model_state": model.state_dict(),
        "n_features": int(X.shape[1]),
        "feature_mean": mean,
        "feature_std": std,
        "include_thomson": include_thomson,
        "training_params": tp,
        "history": history,
        "method": MODEL_ID,
        "model_version": make_model_version(MODEL_ID),
        "git_commit": tp["git_commit"],
    }
    torch.save(ckpt, output)
    print(
        f"[{MODEL_ID}] wrote {output}  train_s={train_s:.1f}s  "
        f"version={ckpt['model_version']}\n"
        f"[{MODEL_ID}] TensorBoard logs: {tb_dir}\n"
        f"[{MODEL_ID}] poetry run tensorboard --logdir {tb_dir.parent}",
        flush=True,
    )

    return NavyBundle(
        checkpoint=output,
        n_features=int(X.shape[1]),
        feature_mean=mean,
        feature_std=std,
        include_thomson=include_thomson,
        device_name=torch.cuda.get_device_name(0),
        training_params=tp,
        model_version=ckpt["model_version"],
    )


def load_navy(path: Path, device: Optional[torch.device] = None) -> tuple[NavyPINN, NavyBundle, dict]:
    device = device or require_cuda()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    tp = dict(ckpt.get("training_params") or {})
    n_fourier = int(tp.get("n_fourier", 0))
    model = NavyPINN(n_cond=int(ckpt["n_features"]), n_fourier=n_fourier).to(device)
    # axis_net / mean-anchor buffers — older ckpts load with strict=False.
    model.load_state_dict(ckpt["model_state"], strict=False)
    if bool(tp.get("mean_anchor")) and "psi_mean" in ckpt["model_state"]:
        model.mean_anchor = True
    model.eval()
    bundle = NavyBundle(
        checkpoint=Path(path),
        n_features=int(ckpt["n_features"]),
        feature_mean=np.asarray(ckpt["feature_mean"], dtype=np.float32),
        feature_std=np.asarray(ckpt["feature_std"], dtype=np.float32),
        include_thomson=bool(ckpt["include_thomson"]),
        device_name=str(ckpt.get("training_params", {}).get("device", "cuda")),
        training_params=dict(ckpt.get("training_params") or {}),
        model_version=str(ckpt.get("model_version") or make_model_version(MODEL_ID)),
    )
    return model, bundle, ckpt


@torch.no_grad()
def predict_row(bundle: NavyBundle, row: dict, source: str, model: Optional[NavyPINN] = None) -> dict:
    device = require_cuda()
    if model is None:
        model, bundle, _ = load_navy(bundle.checkpoint, device=device)
    shot = shot_from_row_inputs_only(row)
    X = features_from_shot(shot, include_thomson=bundle.include_thomson)
    if X.shape[1] < bundle.n_features:
        X = np.hstack([X, np.zeros((len(X), bundle.n_features - X.shape[1]), dtype=np.float32)])
    elif X.shape[1] > bundle.n_features:
        X = X[:, : bundle.n_features]
    Xs = (X - bundle.feature_mean) / bundle.feature_std
    c = torch.from_numpy(Xs.astype(np.float32)).to(device)
    R_np, Z_np = default_grid_from_row(row, source)
    R = torch.from_numpy(R_np.astype(np.float32)).to(device)
    Z = torch.from_numpy(Z_np.astype(np.float32)).to(device)
    psi = model.predict_map(c, R, Z).detach().cpu().numpy().astype(np.float32)
    q_hat, b_hat = model.scalars(c)
    return {
        "psirz": psi,
        "q95": q_hat.detach().cpu().numpy().astype(np.float32),
        "betaN": b_hat.detach().cpu().numpy().astype(np.float32),
    }


def main():
    ap = argparse.ArgumentParser(description=f"Train {MODEL_ID}")
    ap.add_argument("--n-shots", type=int, default=65)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("catskills_artifacts/CRC_FEQ_navy.pt"))
    args = ap.parse_args()
    from experiments import ExperimentConfig, load_shots

    cfg = ExperimentConfig(n_shots=args.n_shots, include_thomson=True)
    shots = load_shots(cfg)
    train_navy(shots, epochs=args.epochs, batch_size=args.batch_size, output=args.out)


if __name__ == "__main__":
    main()
