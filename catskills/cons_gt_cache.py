"""Disk cache for Consistency GT scalars from ``derive_frame`` on EFIT ψ.

Used by navy Consistency campaign: unclipped soft match to scorer-aligned
κ / volume / tri (and axis) without re-extracting LCFS every train epoch.

``li`` is opt-in
----------------
Six of the seven scored scalars come off the LCFS polygon, which ``derive_frame``
extracts once per frame anyway. ``li`` is the seventh and the only field-based one:
it needs ``|grad psi|`` integrated over the enclosed volume, which is why
``derive_frame`` takes a ``with_li`` flag at all and why this cache was built without
it.

It is needed now, because ``li`` is the term most sensitive to the flux *gradient* and
so the one a geometry loss has most to say about. But it is requested per call rather
than added unconditionally: the existing entries were written without it, and making
it part of the required set would invalidate every one of them and force the navy and
teal callers -- which never look at ``li`` -- to pay for a full rebuild. So a cached
entry is a hit for a caller that does not want ``li``, and a miss only for one that
does; the rebuild then writes all seven, so the upgrade happens once per frame and
only for the frames a li-consumer actually touches.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np

from catskills.lab_paths import lab_root

CACHE_VERSION = "cons_gt_v1"
ROOT = Path(__file__).resolve().parents[1]


def cache_root() -> Path:
    root = lab_root() / "cache" / CACHE_VERSION
    root.mkdir(parents=True, exist_ok=True)
    return root


def _frame_key(psi: np.ndarray) -> str:
    a = np.asarray(psi, dtype=np.float32)
    h = hashlib.sha1()
    h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
    h.update(a[::8, ::8].tobytes())
    return h.hexdigest()[:20]


def _path_for(key: str) -> Path:
    return cache_root() / f"{key}.npz"


def load_frame(key: str) -> Optional[dict[str, float]]:
    path = _path_for(key)
    if not path.is_file():
        return None
    z = np.load(path)
    return {k: float(z[k]) for k in z.files}


def save_frame(key: str, vals: dict[str, float]) -> Path:
    path = _path_for(key)
    # np.savez appends ".npz" unless the path already ends with it — use *.tmp.npz.
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **{k: np.float32(v) for k, v in vals.items()})
    tmp.replace(path)
    return path


def _machine_grids():
    mask = np.load(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    R = np.asarray(mask["grid_R"], dtype=np.float64)
    Z = np.asarray(mask["grid_Z"], dtype=np.float64)
    mask_coarse = mask["mask_coarse"].astype(bool)
    mask_f = mask_coarse.astype(np.float64)
    return R, Z, mask_coarse, mask_f


SHAPE_KEYS = ("kappa", "volume", "tri_top", "tri_bot", "R_axis", "Z_axis")
# The seven scored Consistency scalars, in the scorer's own order.
ALL_KEYS = SHAPE_KEYS + ("li",)


def derive_frame_gt(psi: np.ndarray, *, with_li: bool = False) -> dict[str, float]:
    """Run official ``derive_frame`` for one (nZ, nR) map."""
    import sys

    scoring_dir = str(ROOT / "fusion_scoring")
    if scoring_dir not in sys.path:
        sys.path.insert(0, scoring_dir)
    from derive import derive_frame  # noqa: E402

    R, Z, mask_coarse, mask_f = _machine_grids()
    vals = derive_frame(
        np.asarray(psi, dtype=np.float64),
        R,
        Z,
        "DIII-D",
        mask_coarse,
        mask_f,
        contour="auto",
        with_li=bool(with_li),
    )
    keys = ALL_KEYS if with_li else SHAPE_KEYS
    return {k: float(vals.get(k, np.nan)) for k in keys}


def get_or_build_frame_gt(psi: np.ndarray, *,
                          with_li: bool = False) -> tuple[dict[str, float], bool]:
    key = _frame_key(psi)
    cached = load_frame(key)
    # Older cache entries may lack tri/axis, and entries written before li was
    # cached lack that too — either way an incomplete entry is a miss.
    need = ALL_KEYS if with_li else SHAPE_KEYS
    if cached is not None and all(k in cached for k in need):
        return cached, True
    # Recompute the union rather than only the missing field: derive_frame is
    # deterministic and shares one LCFS extraction across all seven, so upgrading an
    # entry in place costs the same as building it and leaves nothing half-written.
    vals = derive_frame_gt(psi, with_li=with_li)
    save_frame(key, vals)
    return vals, False


def build_cons_scalars_for_maps(
    Y: np.ndarray,
    *,
    progress: bool = True,
    with_li: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Per-frame Consistency GT arrays aligned with ``Y`` (N, nZ, nR).

    ``with_li`` adds the seventh scalar and, for entries written before it was
    cached, forces a rebuild of those frames. Callers that do not ask for it see the
    cache exactly as it was.
    """
    from tqdm.auto import tqdm

    N = len(Y)
    keys = ALL_KEYS if with_li else SHAPE_KEYS
    out = {k: np.full(N, np.nan, dtype=np.float32) for k in keys}
    hits = misses = 0
    it = range(N)
    if progress:
        desc = "cons GT (derive_frame%s)" % (" +li" if with_li else "")
        it = tqdm(it, desc=desc, unit="frm", leave=False)
    for i in it:
        vals, hit = get_or_build_frame_gt(Y[i], with_li=with_li)
        hits += int(hit)
        misses += int(not hit)
        for k in out:
            out[k][i] = vals[k]
    return out, {"hits": hits, "misses": misses, "n": N}


def build_kappa_volume_for_maps(
    Y: np.ndarray,
    *,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Back-compat: κ / volume only."""
    arrays, stats = build_cons_scalars_for_maps(Y, progress=progress)
    return arrays["kappa"], arrays["volume"], stats
