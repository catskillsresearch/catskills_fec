"""Olive trainer contracts: GradNorm, PCGrad, λ_geom ramp, promoted ℓ_i stack.

Defaults are off. Brown / Forest / Steel runs that do not pass the new flags
are bit-identical to before this file existed.
"""
from __future__ import annotations

from typing import Iterable

import torch

# Olive stack: B8's three knobs plus ℓ_i promoted off the floor. Mean-1 after
# derive_stack_geom_weights. Do not assign this to STACK_GEOM_FOCUS.
OLIVE_STACK_FOCUS = {"kappa": 1.5, "Z_axis": 1.0, "volume": 1.0, "li": 1.0}
# O1 geometry rebuild: κ / R_axis / Z_axis only. Other four scalars stay on the floor.
PLACEMENT_STACK_FOCUS = {"kappa": 1.5, "R_axis": 1.0, "Z_axis": 1.0}
# O1G-early: same placement knobs, ℓ_i protected, higher floor on the rest.
PLACEMENT_EARLY_FOCUS = {"kappa": 1.5, "R_axis": 1.0, "Z_axis": 1.0, "li": 0.75}
PLACEMENT_EARLY_FLOOR = 0.5
LP_FLOOR = 1e-3
GRADNORM_ALPHA = 0.5
GRADNORM_W_FROM = 10
GEOM_RAMP = 4


WARMUP_R2PSI = 0.999
WARMUP_CONS = 0.96
WARMUP_CAP = 25


def warmup_healthy(
    r2psi: float,
    cons: float,
    *,
    r2_floor: float = WARMUP_R2PSI,
    cons_floor: float = WARMUP_CONS,
) -> bool:
    """True when a data-only map is ready for the λ_geom ramp."""
    return (
        r2psi == r2psi and r2psi >= float(r2_floor)
        and cons == cons and cons >= float(cons_floor)
    )


def geom_ramp_mult(ep: int, warmup: int, ramp: int) -> float:
    """0 through ``warmup``, then linear over ``ramp`` epochs, then 1.

    warmup=8, ramp=4 → epochs 9..12 are 0.25, 0.50, 0.75, 1.0.
    """
    if int(ep) <= int(warmup):
        return 0.0
    r = int(ramp)
    if r <= 0:
        return 1.0
    return min(1.0, (int(ep) - int(warmup)) / float(r))


def update_gradnorm(
    w_g: torch.nn.Parameter,
    opt_w: torch.optim.Optimizer,
    g_psi: torch.Tensor,
    g_stack: torch.Tensor,
    l_data: torch.Tensor,
    l_stack: torch.Tensor,
    l0_data: float,
    l0_stack: float,
    *,
    alpha: float = GRADNORM_ALPHA,
) -> tuple[float, float, float]:
    """One GradNorm step on ``w_g``. ``w_ψ`` is fixed at 1; then ``1 + w_g`` is renormalised to 2.

    ``g_psi`` / ``g_stack`` are ∇_{h_fuse} of the two task losses (stack already
    includes the current ``w_g``). Returns (w_g, G_ψ, G_g).
    """
    g_psi_n = g_psi.detach().float().norm()
    g_g_n = g_stack.detach().float().norm()
    g_bar = 0.5 * (g_psi_n + g_g_n)
    # l0_* are Python floats from the first positive stack loss. r_g is a
    # detached scale: w_g still backprops through g_stack into g_g_n, and
    # target is detached before loss_w. A zero l_data used to make the
    # denominator 0 and kill the step.
    eps = 1e-8
    l0_s = max(float(l0_stack), eps)
    l0_d = max(float(l0_data), eps)
    num_ratio = l_stack.detach().float() / l0_s
    den_ratio = torch.clamp(l_data.detach().float() / l0_d, min=eps)
    r_g = float(torch.clamp(num_ratio / den_ratio, min=0.01, max=100.0))
    target = g_bar * (r_g ** float(alpha))
    loss_w = (g_g_n - target.detach()).abs()
    opt_w.zero_grad(set_to_none=True)
    if w_g.grad is not None:
        w_g.grad = None
    # w_g lives in the graph of g_stack only if create_graph=True was used.
    # Fall back to a finite-difference-free scale step when it is detached.
    if loss_w.requires_grad:
        loss_w.backward()
        opt_w.step()
    else:
        with torch.no_grad():
            if g_g_n > target:
                w_g.mul_(0.99)
            else:
                w_g.mul_(1.01)
    with torch.no_grad():
        w_g.clamp_(1e-3, 10.0)
        s = 1.0 + float(w_g)
        w_g.mul_(2.0 / max(s, 1e-6))
    return float(w_g.detach()), float(g_psi_n), float(g_g_n)


def pcgrad_combine(
    g_data: Iterable[torch.Tensor | None],
    g_stack: Iterable[torch.Tensor | None],
) -> list[torch.Tensor | None]:
    """Project conflicting task grads, then add. None slots stay None."""
    d = list(g_data)
    s = list(g_stack)
    if len(d) != len(s):
        raise ValueError("pcgrad_combine length mismatch")

    def _dot(a: list[torch.Tensor | None], b: list[torch.Tensor | None]) -> torch.Tensor:
        acc = None
        for x, y in zip(a, b):
            if x is None or y is None:
                continue
            v = (x * y).sum()
            acc = v if acc is None else acc + v
        if acc is None:
            return torch.zeros((), device="cpu")
        return acc

    def _proj(src: list[torch.Tensor | None], other: list[torch.Tensor | None]):
        dot = _dot(src, other)
        if float(dot) >= 0:
            return src
        n2 = _dot(other, other).clamp_min(1e-12)
        scale = dot / n2
        out: list[torch.Tensor | None] = []
        for x, y in zip(src, other):
            if x is None:
                out.append(None)
            elif y is None:
                out.append(x)
            else:
                out.append(x - scale * y)
        return out

    d2 = _proj(d, s)
    s2 = _proj(s, d)
    out: list[torch.Tensor | None] = []
    for a, b in zip(d2, s2):
        if a is None:
            out.append(b)
        elif b is None:
            out.append(a)
        else:
            out.append(a + b)
    return out
