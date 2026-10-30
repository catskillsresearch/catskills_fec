"""Label-free physics losses for MLP3 MAST co-training.

Grad-Shafranov on MAST uses the machine's own grid from :mod:`catskills.machine_env`.
Vacuum-boundary agreement compares denormalized psi to the known Psi_vac away from the
plasma. Both need no EFIT labels.

A warning about :func:`gs_vacuum_loss`, retained for the record
--------------------------------------------------------------
``gs_vacuum_loss`` drives ``Delta* psi -> 0`` *inside* the vessel mask. That is the
vacuum equation, and it only holds where there is no current. Inside the vessel is
precisely where the plasma is, and there ``Delta* psi = -mu0 * R * j_phi`` is not
zero. Paired with :func:`vacuum_boundary_loss` pulling psi towards Psi_vac on the
rim, the global minimum of that objective is exactly ``psi = Psi_vac`` -- a field
with no plasma in it at all.

MLP3 was trained with this term on 25% of every batch for 12 hours and learned it
correctly: on MAST it emitted the vacuum field (R2(vac|pred) = 0.88, corr 0.958, no
closed LCFS on any frame). The MAST collapse was a correctly optimized wrong
objective, not a failure of transfer.

:func:`ampere_ip_loss` is the replacement. Ampere's law fixes the *total* current
the map must carry, which the vacuum solution cannot satisfy, so the degenerate
optimum is removed rather than merely outweighed.
"""
from __future__ import annotations

import torch

from catskills.machine_env import MachineEnv, load_env

MU0 = 4.0e-7 * torch.pi


def delta_star(psi: torch.Tensor, R: torch.Tensor, dR: float, dZ: float) -> torch.Tensor:
    """Axisymmetric elliptic operator on (B, nZ, nR). Same stencil as brown_steel_gs."""
    dpsi_dR = torch.gradient(psi, spacing=float(dR), dim=-1)[0]
    dpsi_dZ = torch.gradient(psi, spacing=float(dZ), dim=-2)[0]
    d2R = torch.gradient(dpsi_dR, spacing=float(dR), dim=-1)[0]
    d2Z = torch.gradient(dpsi_dZ, spacing=float(dZ), dim=-2)[0]
    return d2R - dpsi_dR / R.clamp(min=1e-3) + d2Z


def gs_vacuum_loss(
    psi: torch.Tensor,
    env: MachineEnv,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scale-free ||Delta* psi||^2 inside the vessel mask (vacuum limit, no RHS)."""
    if psi.dim() == 4:
        psi = psi[:, 0]
    R = torch.as_tensor(env.R_grid, device=psi.device, dtype=psi.dtype)
    m = mask if mask is not None else torch.as_tensor(env.mask_coarse, device=psi.device)
    m = m.to(psi.dtype)
    ds = delta_star(psi, R, env.dR, env.dZ)
    resid = ds * m
    num = resid.square().mean()
    den = delta_star(psi.detach(), R, env.dR, env.dZ).square().mul(m).mean().clamp(min=1e-12)
    return num / den


def enclosed_ip_ka(
    psi: torch.Tensor,
    env: MachineEnv,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Plasma current (kA) implied by psi (Wb) via Ampere's law, one value per frame.

    ``I_p = -(1/mu0) * integral of (Delta* psi / R) dR dZ`` over ``mask``. The mask
    must be eroded a few cells off the vessel edge: ``torch.gradient`` falls back to
    one-sided stencils at the boundary and those cells otherwise dominate the sum
    (measured: the un-eroded vessel integral lands at 0.08x the true current with
    20x scatter, the 3-cell-eroded one at 1.017x with 1% scatter).
    """
    if psi.dim() == 4:
        psi = psi[:, 0]
    R = torch.as_tensor(env.R_grid, device=psi.device, dtype=psi.dtype)
    ds = delta_star(psi, R, env.dR, env.dZ)
    dens = -(ds / R.clamp(min=1e-3)) * (env.dR * env.dZ / MU0)
    m = mask.to(psi.dtype)
    return (dens * m).flatten(start_dim=-2).sum(dim=-1) / 1.0e3


def ampere_ip_loss(
    psi: torch.Tensor,
    ip_ka: torch.Tensor,
    env: MachineEnv,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Relative squared error between the map's enclosed current and measured I_p.

    Label-free: I_p is an input column on both machines, and after
    :func:`catskills.psivac.ip_align_shift` it is trustworthy on both. Scored as a
    ratio so the term is dimensionless and needs no per-machine weight.

    This is the term that forbids the vacuum solution. Psi_vac encloses no current,
    so it sits at a relative error of 1.0 per frame instead of at the free minimum
    ``gs_vacuum_loss`` used to hand it.
    """
    pred = enclosed_ip_ka(psi, env, mask)
    tgt = ip_ka.to(pred.dtype).reshape(-1)
    if tgt.numel() != pred.numel():
        raise ValueError(f"ip has {tgt.numel()} entries, psi batch has {pred.numel()}")
    scale = tgt.abs().clamp(min=50.0)
    return ((pred.abs() - tgt.abs()) / scale).square().mean()


def vacuum_boundary_loss(
    psi: torch.Tensor,
    vac_phys: torch.Tensor,
    env: MachineEnv,
    *,
    rim_cells: int = 3,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """(psi - Psi_vac)^2 on the envelope rim -- exterior agreement without labels."""
    if psi.dim() == 4:
        psi = psi[:, 0]
    if vac_phys.dim() == 4:
        vac_phys = vac_phys[:, 0]
    m = mask if mask is not None else torch.as_tensor(env.mask_coarse, device=psi.device)
    rim = _mask_rim(m, rim_cells).to(psi.dtype)
    if rim.sum() < 1:
        return psi.new_zeros(())
    diff = (psi - vac_phys) * rim
    return diff.square().sum() / rim.sum().clamp(min=1.0)


def mask_erode(mask: torch.Tensor, cells: int) -> torch.Tensor:
    """Shrink a mask by ``cells``. Erosion is dilation of the complement.

    Uses the 4-neighbourhood cross, which is what ``scipy.ndimage.binary_erosion``
    applies by default. The pool-side rebuild in
    ``catskills.scripts.fix_pool_ip`` goes through scipy, so a 3x3 square here
    would silently integrate over a different region than the values in ip.f32
    were derived from.
    """
    f = torch.nn.functional
    inv = (~mask.bool()).to(torch.float32).unsqueeze(0).unsqueeze(0)
    for _ in range(max(0, int(cells))):
        vert = f.max_pool2d(f.pad(inv, (0, 0, 1, 1), mode="constant", value=1.0), (3, 1), stride=1)
        horiz = f.max_pool2d(f.pad(inv, (1, 1, 0, 0), mode="constant", value=1.0), (1, 3), stride=1)
        inv = torch.maximum(vert, horiz)
    return inv.squeeze() < 0.5


def _mask_rim(mask: torch.Tensor, cells: int) -> torch.Tensor:
    """Cells just outside the coarse envelope (where psi should match vacuum)."""
    m = mask.bool()
    out = m.clone()
    for _ in range(max(1, int(cells))):
        pad = torch.nn.functional.pad(m.float().unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode="constant", value=0)
        dil = torch.nn.functional.max_pool2d(pad, 3, stride=1).squeeze() > 0
        out = out | dil
    return out & (~m)


AMPERE_ERODE_CELLS = 3


def env_tensors(env: MachineEnv, device: torch.device, dtype: torch.dtype = torch.float32) -> dict:
    """Cached mask/R tensors for a machine on device.

    ``mask_core`` is the vessel eroded by :data:`AMPERE_ERODE_CELLS`, which is what
    the Ampere integral must be taken over; see :func:`enclosed_ip_ka`.
    """
    mask = torch.as_tensor(env.mask_coarse, device=device, dtype=torch.bool)
    return {
        "env": env,
        "mask": mask.to(dtype),
        "mask_core": mask_erode(mask, AMPERE_ERODE_CELLS).to(dtype),
        "R": torch.as_tensor(env.R_grid, device=device, dtype=dtype),
    }


def mast_env(device: torch.device) -> MachineEnv:
    return load_env("mast")
