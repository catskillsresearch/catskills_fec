"""One place to decide how much of the machine a batch job may take.

The lab box has 16 cores and is also the interactive machine. A ``ProcessPoolExecutor``
sized at ``cpu_count() - 2`` reparsing the 7,041-shot DIII-D corpus pins every core
for tens of minutes and makes the desktop unusable, so the default here is half the
cores rather than nearly all of them. Override per invocation with ``CRC_FEQ_WORKERS``.

Two knobs, because they multiply. ``workers()`` sizes the process pool;
:func:`limit_threads` caps the BLAS/OpenMP threads *inside* each worker, which would
otherwise each spin up their own core-count pool and oversubscribe by 16x.
"""
from __future__ import annotations

import os

ENV_VAR = "CRC_FEQ_WORKERS"
FRACTION = 0.5


def n_cores() -> int:
    return max(1, os.cpu_count() or 4)


def workers(requested: int | None = None, *, cap: int | None = None) -> int:
    """Process-pool size: an explicit request, else ``CRC_FEQ_WORKERS``, else half."""
    if requested is not None and int(requested) > 0:
        n = int(requested)
    else:
        env = os.environ.get(ENV_VAR)
        n = int(env) if env and env.isdigit() and int(env) > 0 else \
            max(1, int(n_cores() * FRACTION))
    if cap is not None:
        n = min(n, max(1, int(cap)))
    return max(1, min(n, n_cores()))


def limit_threads(per_worker: int = 1) -> None:
    """Cap in-process BLAS/OpenMP threads. Call before importing numpy or torch."""
    v = str(max(1, int(per_worker)))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(name, v)
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(max(1, int(per_worker)))
