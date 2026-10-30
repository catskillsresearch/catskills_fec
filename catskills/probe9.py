#!/usr/bin/env python3
"""Probe9 shared library: house harness pattern plus the paired-S bootstrap instrument.

Why this module exists. Every gate we have ever shipped selects on psi-MSE or on a
per-scalar sigma. The score is ``S = 0.55 R2_psi + 0.15 R2_qb + 0.10 (1 - D_LCFS)
+ 0.20 Cons`` and our gap to the top of the board decomposes as Consistency 72.8%,
R2_qb 15.6%, D_LCFS 9.7%, R2_psi 1.9%. So the selector and the objective disagree on
98% of the gap, and ``cons_foldvar --compare`` bootstraps Consistency alone.

``paired_S_bootstrap`` closes that. The official scorer is additive over shots by
construction -- ``Accum.add(part)`` sums per-shot residual sums and ``ref_stats``
is a per-shot sum of GT moments (see fusion_scoring/metrics.py) -- so a shot
resample is just a re-sum of cached per-shot parts followed by
``finalize_machine``. That makes 2000 replicates on the full composite S cost two
scoring passes instead of 4000.

Two nuisance parameters are held at their full-fold values across replicates: the
fold-global ``psi_sign`` bit and the fold means used for non-finite substitution.
They are properties of the fold, not of the candidate, and freezing them is what
keeps A and B paired.

  from catskills.probe9 import paired_S_bootstrap, score_parts_for_mix
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "fusion_scoring"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from catskills.d3d_analog import write_json_atomic  # noqa: E402
from catskills.lab_paths import artifacts_dir, results_dir  # noqa: E402

N_HOLD = 80
HOLD_SEED = 42
GRID = 65
K_PCA = 60
INCUMBENT_MIX = "o1x5_42for7"
QB_SOURCE = "st_support_cache:B8+F9"
N_BOOT = 2000
#: how much of a child's tail we keep in memory for OOM classification
TAIL_CHARS = 200_000
BOOT_SEED = 42
#: p_better both folds must clear before a probe counts as promote-worthy. The delta-S
#: instrument is the research selector; the house gates still govern packaging.
P_BETTER = 0.95
OOM_MARKERS = (
    "out of memory",
    "cuda out of memory",
    "cudnn_status_alloc_failed",
    "cudn_status_alloc_failed",
    "can't allocate memory",
    "cuda error: out of memory",
)
RECEIPT_KEYS = (
    "probe",
    "status",
    "delta_S",
    "delta_S_sigma",
    "p_better",
    "gpu_hours",
    "verdict",
    "reasons",
    "detail",
)
STATUSES = ("ok", "failed", "skipped", "pending")


# --------------------------------------------------------------------------- paths


def work_dir() -> Path:
    d = results_dir() / "probe9"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_dir() -> Path:
    d = work_dir() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir() -> Path:
    d = work_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def receipt_path(probe: str) -> Path:
    return work_dir() / f"{probe}.json"


def parts_dir(probe: str) -> Path:
    """Per-arm intermediate receipts, so a killed probe resumes at the arm boundary."""
    d = work_dir() / f"{probe}_parts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def part_path(probe: str, name: str) -> Path:
    return parts_dir(probe) / f"{_slug(name)}.json"


def load_part(probe: str, name: str) -> dict | None:
    return load_json(part_path(probe, name))


def save_part(probe: str, name: str, body: dict) -> Path:
    return write_json_atomic(part_path(probe, name), body)


def coverage_path() -> Path:
    return results_dir() / "brown" / "P33_COVERAGE.json"


def support_cache_dir() -> Path:
    return results_dir() / "brown" / "st_support_cache"


def load_json(path: Path) -> dict | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def sha_of(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _slug(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name)


# ------------------------------------------------------------------- progress/receipts


def progress_path() -> Path:
    return work_dir() / "RUN_PROGRESS.json"


def write_progress(**fields) -> dict:
    base = load_json(progress_path()) or {"steps_done": []}
    base["updated"] = datetime.now(timezone.utc).isoformat()
    base.update(fields)
    write_json_atomic(progress_path(), base)
    return base


def mark_step(step_id: str) -> list[str]:
    prog = load_json(progress_path()) or {"steps_done": []}
    steps = list(prog.get("steps_done") or [])
    if step_id not in steps:
        steps.append(step_id)
    write_progress(steps_done=steps)
    return steps


def step_done(step_id: str) -> bool:
    prog = load_json(progress_path()) or {}
    return step_id in (prog.get("steps_done") or [])


def make_receipt(
    probe: str,
    *,
    status: str = "ok",
    delta_S: float | None = None,
    delta_S_sigma: float | None = None,
    p_better: float | None = None,
    gpu_hours: float = 0.0,
    verdict: str = "",
    reasons: Sequence[str] = (),
    detail: dict | None = None,
) -> dict:
    if status not in STATUSES:
        raise ValueError(f"bad status {status!r}; expected one of {STATUSES}")
    return {
        "probe": probe,
        "status": status,
        "delta_S": None if delta_S is None else float(delta_S),
        "delta_S_sigma": None if delta_S_sigma is None else float(delta_S_sigma),
        "p_better": None if p_better is None else float(p_better),
        "gpu_hours": float(gpu_hours),
        "verdict": verdict,
        "reasons": list(reasons),
        "detail": dict(detail or {}),
        "written": datetime.now(timezone.utc).isoformat(),
    }


def valid_receipt(body: dict | None) -> bool:
    if not isinstance(body, dict):
        return False
    if any(k not in body for k in RECEIPT_KEYS):
        return False
    return body.get("status") in STATUSES


def write_receipt(receipt: dict) -> Path:
    if not valid_receipt(receipt):
        raise ValueError(f"incomplete receipt: {sorted(receipt)}")
    dest = receipt_path(receipt["probe"])
    write_json_atomic(dest, receipt)
    mark_step(f"receipt_{receipt['probe']}")
    return dest


def read_receipt(probe: str) -> dict | None:
    body = load_json(receipt_path(probe))
    return body if valid_receipt(body) else None


def have_receipt(probe: str) -> bool:
    """A resumable probe skips itself when a terminal receipt already exists."""
    body = read_receipt(probe)
    return bool(body) and body["status"] in ("ok", "skipped")


class GpuClock:
    """Wall-clock hours attributable to a probe, so the report can rank by gain per hour."""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    @property
    def hours(self) -> float:
        return (time.perf_counter() - self.t0) / 3600.0


# ------------------------------------------------------------------- subprocess/OOM


def looks_like_oom(text: str) -> bool:
    low = (text or "").lower()
    return any(m in low for m in OOM_MARKERS)


def probe_env() -> dict:
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("FEQ_SCORE_WORKERS", "8")
    return env


def train_until_artifact(
    cmd: Sequence[str],
    artifact: Path,
    *,
    log_name: str,
    retries: int = 2,
) -> tuple[bool, str]:
    """Run a trainer until its artifact exists. The artifact is the success condition.

    Every trainer in this house exits nonzero when its own promotion gate fails, and for
    probe9 that is the expected outcome -- those gates are what the delta-S instrument is
    being measured against. Retrying on a nonzero exit therefore re-runs an hour of
    training that already succeeded, so only a crash that left no artifact is retried.
    """
    last = ""
    for _ in range(max(1, retries)):
        if artifact.is_file():
            return True, last
        _rc, last = run_isolated(cmd, log_name=log_name, retries=1)
        if artifact.is_file() or not looks_like_oom(last):
            break
    return artifact.is_file(), last


def run_isolated(
    cmd: Sequence[str],
    *,
    log_name: str,
    retries: int = 3,
    backoff_s: float = 30.0,
) -> tuple[int, str]:
    """Run a step in a fresh process, retrying OOM with a cooldown. Log is appended.

    Output streams straight into the log rather than being captured, so a training arm
    that runs for hours can be watched while it runs.
    """
    log_path = logs_dir() / log_name
    env = probe_env()
    last, rc = "", 1
    for attempt in range(1, max(1, retries) + 1):
        print(f"+ ({attempt}/{retries})", " ".join(str(c) for c in cmd), flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"\n--- {datetime.now(timezone.utc).isoformat()} attempt {attempt} "
                     f"start ---\n")
            fh.flush()
            start = fh.tell()
            proc = subprocess.run(
                [str(c) for c in cmd], cwd=str(ROOT), env=env,
                stdout=fh, stderr=subprocess.STDOUT, text=True,
            )
            fh.write(f"\n--- attempt {attempt} rc={proc.returncode} ---\n")
        with log_path.open("r", encoding="utf-8", errors="replace") as fh:
            fh.seek(start)
            last = fh.read()[-TAIL_CHARS:]
        rc = proc.returncode
        if rc == 0:
            return 0, last
        if attempt < retries:
            oom = looks_like_oom(last)
            write_progress(last_error=f"{'OOM' if oom else 'fail'} on {log_name}",
                           last_log=str(log_path))
            time.sleep(backoff_s * (attempt if oom else 1))
    write_progress(last_error=f"failed {log_name}", last_log=str(log_path))
    if last:
        print("\n".join(last.strip().splitlines()[-8:]), file=sys.stderr)
    return rc, last


# ------------------------------------------------------------------- holdout / folds


def holdout_keys() -> list[str]:
    """The legacy 80. Same draw as tailfit_mix_score and cons_foldvar, so paired is legal."""
    from catskills.feature_cache import sample_cache_keys

    return list(sample_cache_keys(7000 + N_HOLD, HOLD_SEED, include_thomson=True)[:N_HOLD])


def entries_for(keys: Sequence[str]) -> list[dict]:
    """``{row, shot, cache_key}`` per shot, read from the parquet mirror."""
    from catskills.warm_feature_cache import row_from_parquet, train_parquet_dir
    from experiments import load_shot_from_hf_row

    pq = train_parquet_dir()
    out = []
    for key in keys:
        row = row_from_parquet(pq / f"{key}.parquet")
        out.append({"row": row, "shot": load_shot_from_hf_row(row), "cache_key": key})
    return out


_ENTRIES: list[dict] | None = None


def holdout_entries() -> list[dict]:
    global _ENTRIES
    if _ENTRIES is None:
        _ENTRIES = entries_for(holdout_keys())
    return _ENTRIES


def worst40_indices(keys: Sequence[str] | None = None) -> np.ndarray:
    """S33.7 fold: the 40 least-covered shots, the fold that brackets the public score."""
    cov = load_json(coverage_path())
    if not cov or not cov.get("rank_worst_first"):
        raise FileNotFoundError(f"missing rank_worst_first in {coverage_path()}")
    keys = list(keys or holdout_keys())
    picked = [k for k in cov["rank_worst_first"][:40] if k in keys]
    return np.array(sorted(keys.index(k) for k in picked), dtype=int)


def fold_indices(name: str, keys: Sequence[str] | None = None) -> np.ndarray:
    keys = list(keys or holdout_keys())
    if name == "all80":
        return np.arange(len(keys), dtype=int)
    if name == "worst40":
        return worst40_indices(keys)
    raise ValueError(f"unknown fold {name!r}")


# ------------------------------------------------------------------- mix registry


def mix_registry() -> dict[str, list[str]]:
    """Mix name -> checkpoint stems, inherited from cons_foldvar so names never diverge."""
    from catskills.scripts.cons_foldvar import CANDIDATES

    return {k: list(v) for k, v in CANDIDATES.items() if k != "pair"}


def member_registry() -> dict[str, str]:
    from catskills.scripts.tailfit_eval import CANDS

    return dict(CANDS)


def mix_stems(mix: str) -> list[str]:
    reg = mix_registry()
    if mix not in reg:
        raise KeyError(f"unknown mix {mix!r}; known: {sorted(reg)}")
    return reg[mix]


def stems_present(stems: Iterable[str]) -> dict[str, bool]:
    return {s: (artifacts_dir() / f"{s}.pt").is_file() for s in stems}


# ------------------------------------------------------------------- member inference cache


def basis_path() -> Path:
    return cache_dir() / "pca_basis.npz"


def member_cache_path(stem: str, tag: str = "") -> Path:
    suffix = f"_{_slug(tag)}" if tag and tag != "all80" else ""
    return cache_dir() / f"member_{_slug(stem)}{suffix}.npz"


def member_z(
    stem: str,
    *,
    keys: Sequence[str] | None = None,
    label: str = "hold",
    tag: str = "",
    device=None,
    refresh: bool = False,
) -> dict:
    """Holdout PCA coefficients + q/betaN heads for one checkpoint, cached to disk.

    The cache is what makes a mix search affordable: every mix is a coefficient-space
    mean of members, so after one GPU pass per checkpoint no further inference runs.
    """
    dest = member_cache_path(stem, tag)
    if dest.is_file() and not refresh:
        rec = np.load(dest)
        return {
            "z": np.asarray(rec["z"], np.float64),
            "q95": np.asarray(rec["q95"], np.float64),
            "betaN": np.asarray(rec["betaN"], np.float64),
            "shot_ids": np.asarray(rec["shot_ids"], np.int64),
            "stem": stem,
        }

    import torch

    from catskills.d3d_analog import save_npz_atomic
    from catskills.pinn_charcoal import load_charcoal
    from catskills.scripts.psi_decorrelate import _activate
    from catskills.scripts.psi_tailfit import build

    path = artifacts_dir() / f"{stem}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"missing checkpoint {path}")
    dev = device or torch.device("cuda")
    torch.backends.cudnn.enabled = False
    keys = list(keys) if keys else holdout_keys()
    model, bundle, ck = load_charcoal(path, device=dev)
    _activate(bundle)
    n_raw = int(ck.get("n_raw_features") or len(bundle.feature_mean) // 3)
    Xh, Zh, ids = build(keys, bundle, label, n_raw)
    z, q, bn = _infer_full(model, torch.from_numpy(np.ascontiguousarray(Xh)), dev)
    del model
    torch.cuda.empty_cache()

    if not basis_path().is_file():
        save_npz_atomic(
            basis_path(),
            pca_mean=np.asarray(ck["pca_mean"], np.float64).reshape(-1),
            pca_components=np.asarray(ck["pca_components"], np.float64),
        )
    if not coeff_gt_path(tag).is_file():
        save_npz_atomic(
            coeff_gt_path(tag),
            z_gt=np.asarray(Zh, np.float64),
            shot_ids=np.asarray(ids, np.int64),
        )
    save_npz_atomic(
        dest,
        z=z.astype(np.float64),
        q95=q.astype(np.float64).reshape(-1),
        betaN=bn.astype(np.float64).reshape(-1),
        shot_ids=np.asarray(ids, np.int64),
    )
    return member_z(stem, keys=keys, label=label, tag=tag, device=dev)


def coeff_gt_path(tag: str = "") -> Path:
    suffix = f"_{_slug(tag)}" if tag and tag != "all80" else ""
    return cache_dir() / f"coeff_gt{suffix}.npz"


def coeff_gt(tag: str = "") -> dict:
    rec = np.load(coeff_gt_path(tag))
    return {
        "z_gt": np.asarray(rec["z_gt"], np.float64),
        "shot_ids": np.asarray(rec["shot_ids"], np.int64),
    }


def _infer_full(model, xs, device, bs: int = 4096):
    import torch

    with torch.no_grad():
        model.eval()
        zs, qs, bns = [], [], []
        for s in range(0, len(xs), bs):
            out = model(xs[s:s + bs].to(device, non_blocking=True))
            zs.append(out[0].detach().float().cpu())
            qs.append(out[1].detach().float().cpu())
            bns.append(out[2].detach().float().cpu())
        return (
            torch.cat(zs).numpy(),
            torch.cat(qs).numpy(),
            torch.cat(bns).numpy(),
        )


def pca_basis() -> tuple[np.ndarray, np.ndarray]:
    rec = np.load(basis_path())
    return (
        np.asarray(rec["pca_mean"], np.float64).reshape(-1),
        np.asarray(rec["pca_components"], np.float64),
    )


def mix_z(
    stems: Sequence[str],
    *,
    keys: Sequence[str] | None = None,
    label: str = "hold",
    tag: str = "",
    device=None,
) -> dict:
    """Coefficient-space mean over members, with the member q/betaN heads averaged too."""
    recs = [member_z(s, keys=keys, label=label, tag=tag, device=device) for s in stems]
    ids = recs[0]["shot_ids"]
    for r in recs[1:]:
        if r["shot_ids"].shape != ids.shape or not np.array_equal(r["shot_ids"], ids):
            raise ValueError(f"frame misalignment between {recs[0]['stem']} and {r['stem']}")
    return {
        "z": np.mean([r["z"] for r in recs], axis=0),
        "q95": np.mean([r["q95"] for r in recs], axis=0),
        "betaN": np.mean([r["betaN"] for r in recs], axis=0),
        "shot_ids": ids,
        "stems": list(stems),
    }


def psi_from_z(z: np.ndarray) -> np.ndarray:
    mean, comp = pca_basis()
    return (np.asarray(z, np.float64) @ comp + mean).reshape(-1, GRID, GRID).astype(np.float32)


def lock_qb(keys: Sequence[str], shot_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shipped B8+F9 q95/betaN from the support cache, concatenated in frame order."""
    q_all, bn_all = [], []
    for i, key in enumerate(keys):
        dest = support_cache_dir() / f"{key}.npz"
        if not dest.is_file():
            raise FileNotFoundError(f"missing support-cache shot {dest}")
        rec = np.load(dest)
        n = int((shot_ids == i).sum())
        q = np.asarray(rec["q95_mean"], np.float64).reshape(-1)
        bn = np.asarray(rec["betaN_mean"], np.float64).reshape(-1)
        if q.size != n or bn.size != n:
            raise ValueError(f"{key}: lock qB {q.size}/{bn.size} != mix frames {n}")
        q_all.append(q)
        bn_all.append(bn)
    return np.concatenate(q_all), np.concatenate(bn_all)


def preds_from_z(
    z: np.ndarray,
    shot_ids: np.ndarray,
    keys: Sequence[str],
    *,
    q95: np.ndarray | None = None,
    betaN: np.ndarray | None = None,
    qb_from_pair: bool = True,
) -> list[dict]:
    """Per-shot ``{psirz, q95, betaN}`` in holdout order, ready for the official scorer."""
    psi = psi_from_z(z)
    if qb_from_pair:
        q95, betaN = lock_qb(keys, shot_ids)
    if q95 is None or betaN is None:
        raise ValueError("need q95/betaN when qb_from_pair is False")
    q95 = np.asarray(q95, np.float64).reshape(-1)
    betaN = np.asarray(betaN, np.float64).reshape(-1)
    out = []
    for i in range(len(keys)):
        sl = shot_ids == i
        out.append({
            "psirz": psi[sl],
            "q95": q95[sl].astype(np.float32),
            "betaN": betaN[sl].astype(np.float32),
        })
    return out


# ------------------------------------------------------------------- the instrument


def _ref_moments(gt: dict, ref: tuple) -> dict:
    """One shot's contribution to ``ref_stats``. Mirrors local_benchmark.official_score."""
    from common import N_CONS, N_SCALARS, SCALARS

    g = np.asarray(gt["psi"], np.float64)
    fin = np.isfinite(g)
    scal_sum = np.zeros(N_SCALARS)
    scal_sumsq = np.zeros(N_SCALARS)
    scal_n = np.zeros(N_SCALARS)
    for j, name in enumerate(SCALARS):
        v = np.asarray(gt[name], np.float64)
        v = v[np.isfinite(v)]
        scal_sum[j] = v.sum()
        scal_sumsq[j] = (v ** 2).sum()
        scal_n[j] = v.size
    cons_sum = np.zeros(N_CONS)
    cons_sumsq = np.zeros(N_CONS)
    cons_n = np.zeros(N_CONS)
    cons_gt, cmask = ref[1], ref[2]
    for j in range(N_CONS):
        v = cons_gt[cmask[:, j], j]
        cons_sum[j] = v.sum()
        cons_sumsq[j] = (v ** 2).sum()
        cons_n[j] = v.size
    return {
        "psi_sum": float(g[fin].sum()),
        "psi_sumsq": float((g[fin] ** 2).sum()),
        "psi_n": int(fin.sum()),
        "scal_sum": scal_sum,
        "scal_sumsq": scal_sumsq,
        "scal_n": scal_n,
        "cons_sum": cons_sum,
        "cons_sumsq": cons_sumsq,
        "cons_n": cons_n,
    }


def sum_ref_moments(moments: Sequence[dict], idx: Sequence[int]) -> dict:
    """``ref_stats`` for an arbitrary (possibly multiset) shot selection."""
    keys_scalar = ("psi_sum", "psi_sumsq", "psi_n")
    keys_array = ("scal_sum", "scal_sumsq", "scal_n", "cons_sum", "cons_sumsq", "cons_n")
    out: dict = {k: 0.0 for k in keys_scalar}
    for k in keys_array:
        out[k] = np.zeros_like(moments[0][k], dtype=np.float64)
    for i in idx:
        m = moments[int(i)]
        for k in keys_scalar:
            out[k] += m[k]
        for k in keys_array:
            out[k] = out[k] + m[k]
    return out


def fold_cache_path(tag: str = "all80") -> Path:
    return cache_dir() / f"fold_{_slug(tag)}.pkl"


def fold_reference_cached(
    tag: str = "all80",
    *,
    keys: Sequence[str] | None = None,
    member_label: str = "hold",
    refresh: bool = False,
) -> dict:
    """The fold's GT state, cached whole.

    Parsing the holdout parquet shards costs minutes and every probe runs in its own
    process, so the fold is pickled once. The GT reference contours are already disk
    cached per shot; what this adds is skipping the parquet parse and the moment pass.

    ``keys`` and ``member_label`` are what let Tier C build a second, sealed fold without
    colliding with the legacy 80's feature and reference caches.
    """
    import pickle

    dest = fold_cache_path(tag)
    if dest.is_file() and not refresh:
        with dest.open("rb") as fh:
            return pickle.load(fh)
    fold = fold_reference(entries_for(keys) if keys else holdout_entries())
    fold["name"] = tag
    fold["member_label"] = member_label
    tmp = dest.with_suffix(".pkl.tmp")
    with tmp.open("wb") as fh:
        pickle.dump(fold, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(dest)
    return fold


def fold_lite(tag: str = "all80", *, refresh: bool = False) -> dict:
    """Fold state without the flux cubes: enough for scalar-only re-scoring.

    P5 and P8 change value scalars only, and ``rescore_scalars`` needs the GT scalars,
    the substitution means, and each shot's frame count -- not the 330 MB of psi.
    """
    import pickle

    dest = cache_dir() / f"foldlite_{_slug(tag)}.pkl"
    if dest.is_file() and not refresh:
        with dest.open("rb") as fh:
            return pickle.load(fh)
    fold = fold_reference_cached(tag, refresh=refresh)
    lite = {
        "keys": list(fold["keys"]),
        "means": fold["means"],
        "moments": fold["moments"],
        "T": [int(np.asarray(gt["psi"]).shape[0]) for gt in fold["gts"]],
        "gts": [{"q95": np.asarray(gt["q95"], np.float64),
                 "betaN": np.asarray(gt["betaN"], np.float64)} for gt in fold["gts"]],
        "lite": True,
    }
    tmp = dest.with_suffix(".pkl.tmp")
    with tmp.open("wb") as fh:
        pickle.dump(lite, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(dest)
    return lite


def fold_reference(entries: Sequence[dict]) -> dict:
    """GT-side fold state: refs, per-shot moments, and the substitution means.

    Shared by every candidate, which is exactly why a delta is paired: the only thing
    that varies between A and B is the prediction.
    """
    from catskills.local_benchmark import gt_for_score
    from catskills.score_ref_cache import entry_cache_key, get_or_build_reference
    from local_score import build_reference

    mask = np.load(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    R, Z = mask["grid_R"], mask["grid_Z"]
    mask_coarse = mask["mask_coarse"].astype(bool)
    mask_f = mask_coarse.astype(np.float64)

    gts = [gt_for_score(e) for e in entries]
    refs, moments = [], []
    for entry, gt in zip(entries, gts):
        ref, _hit = get_or_build_reference(
            gt["psi"], R, Z, mask_coarse, mask_f,
            cache_key=entry_cache_key(entry), build_fn=build_reference,
        )
        refs.append(ref)
        moments.append(_ref_moments(gt, ref))
    total = sum_ref_moments(moments, range(len(moments)))
    mean_psi = total["psi_sum"] / total["psi_n"] if total["psi_n"] else 0.0
    scal_n, cons_n = total["scal_n"], total["cons_n"]
    means = (
        mean_psi,
        np.where(scal_n > 0, total["scal_sum"] / np.where(scal_n > 0, scal_n, 1), 0.0),
        np.where(cons_n > 0, total["cons_sum"] / np.where(cons_n > 0, cons_n, 1), 0.0),
    )
    return {
        "gts": gts,
        "refs": refs,
        "moments": moments,
        "means": means,
        "total": total,
        "grid": (R, Z, mask_coarse, mask_f),
        "keys": [e["cache_key"] for e in entries],
    }


def score_parts(fold: dict, preds: Sequence[dict], *, workers: int | None = None) -> dict:
    """Per-shot scorer parts for one candidate, plus the fold-global flux sign bit."""
    from local_score import psi_residuals, score_shots
    from common import PSI_SIGNS

    R, Z, mask_coarse, mask_f = fold["grid"]
    mean_psi = fold["means"][0]
    totals = {s: 0.0 for s in PSI_SIGNS}
    for gt, p in zip(fold["gts"], preds):
        rr, _, _ = psi_residuals(gt["psi"], p["psirz"], mean_psi)
        for sg in PSI_SIGNS:
            totals[sg] += rr[sg]
    psi_sign = min(PSI_SIGNS, key=lambda sg: totals[sg])
    parts = list(score_shots(
        fold["gts"], fold["refs"], preds, R, Z, mask_coarse, mask_f, psi_sign,
        fold["means"], workers=workers, desc="probe9 score",
    ))
    return {"parts": parts, "psi_sign": psi_sign, "moments": fold["moments"]}


def _score_with_sign(fold: dict, preds: Sequence[dict], psi_sign: float, *,
                     workers: int | None) -> list[dict]:
    from local_score import score_shots

    R, Z, mask_coarse, mask_f = fold["grid"]
    return list(score_shots(
        fold["gts"], fold["refs"], preds, R, Z, mask_coarse, mask_f, psi_sign,
        fold["means"], workers=workers, desc="probe9 score",
    ))


def _pick_sign(parts: Sequence[dict]) -> tuple[float, dict]:
    from common import PSI_SIGNS

    totals = {sg: float(sum(p["ss_res_psi"][sg] for p in parts)) for sg in PSI_SIGNS}
    return min(totals, key=lambda sg: totals[sg]), totals


def scored_chunked(
    candidates: dict[str, Sequence[str]],
    keys: Sequence[str],
    *,
    fold_name: str,
    member_label: str = "hold",
    chunk: int = 40,
    qb_from_pair: bool = False,
    workers: int | None = None,
    device=None,
) -> dict[str, dict]:
    """Score several mixes on a large fold without ever holding the whole fold in memory.

    ``fold_reference`` keeps every shot's GT flux grids and reference contours resident,
    which is 468 MB for the legacy 80 and enough to take the box down at 800. The per-shot
    scorer parts are about a kilobyte each, so the fold is walked in chunks and only the
    parts and moments are kept.

    Two fold-global quantities have to survive the chunking, and both are handled exactly
    rather than approximated:

    ``psi_sign`` composes into the geometry extractors, so it cannot be chosen after the
    fact. The first chunk is scored under both signs to fix it, and because every part
    carries ``ss_res_psi`` for both signs regardless, the choice is re-verified against the
    whole fold at the end for free.

    ``means`` (the scorer's substitution values) are chunk-local here, which is only sound
    because they are read exclusively when a prediction is missing or non-finite. That
    condition is asserted rather than assumed.
    """
    from common import PSI_SIGNS

    names = list(candidates)
    packed = {
        name: mix_z(list(stems), keys=keys, label=member_label, tag=fold_name,
                    device=device)
        for name, stems in candidates.items()
    }
    parts: dict[str, list[dict]] = {n: [] for n in names}
    sign: dict[str, float | None] = {n: None for n in names}
    sign_probe: dict[str, dict] = {}
    moments: list[dict] = []

    for start in range(0, len(keys), chunk):
        stop = min(start + chunk, len(keys))
        chunk_keys = list(keys[start:stop])
        fold_c = fold_reference(entries_for(chunk_keys))
        moments.extend(fold_c["moments"])
        for name in names:
            pk = packed[name]
            sel = (pk["shot_ids"] >= start) & (pk["shot_ids"] < stop)
            preds = preds_from_z(
                pk["z"][sel], pk["shot_ids"][sel] - start, chunk_keys,
                q95=pk["q95"][sel], betaN=pk["betaN"][sel], qb_from_pair=qb_from_pair,
            )
            if sign[name] is None:
                probe = {sg: _score_with_sign(fold_c, preds, sg, workers=workers)
                         for sg in PSI_SIGNS}
                sign[name], totals = _pick_sign(probe[PSI_SIGNS[0]])
                sign_probe[name] = {"chunk_totals": totals, "from_shots": len(chunk_keys)}
                got = probe[sign[name]]
            else:
                got = _score_with_sign(fold_c, preds, sign[name], workers=workers)
            parts[name].extend(got)
        del fold_c
        print(f"[chunked] shots {stop}/{len(keys)}", flush=True)

    out: dict[str, dict] = {}
    for name in names:
        best, totals = _pick_sign(parts[name])
        if best != sign[name]:
            raise ValueError(
                f"{name}: flux sign chosen on the first chunk ({sign[name]:+g}) is not the "
                f"whole-fold choice ({best:+g}); totals {totals}"
            )
        substituted = sum(int(p["psi_nonfinite"]) for p in parts[name])
        absent = sum(1 for p in parts[name] if p["missing"])
        if substituted or absent:
            raise ValueError(
                f"{name}: {substituted} non-finite flux cells and {absent} missing shots, "
                "so the scorer substituted fold means -- those are chunk-local here and "
                "the result would not match a whole-fold score"
            )
        out[name] = {
            "parts": parts[name], "psi_sign": sign[name], "moments": moments,
            "mix": name, "stems": list(candidates[name]),
            "chunked": {"chunk": chunk, "n_shots": len(keys),
                        "sign_probe": sign_probe.get(name), "sign_totals": totals},
        }
    return out


def scored_chunked_cached(
    candidates: dict[str, Sequence[str]],
    keys: Sequence[str],
    *,
    fold_name: str,
    member_label: str = "hold",
    qb_from_pair: bool = False,
    chunk: int = 40,
    workers: int | None = None,
    device=None,
) -> dict[str, dict]:
    """``scored_chunked`` with the parts pickled per candidate, so a kill costs one pass."""
    import pickle

    have, want = {}, {}
    for name, stems in candidates.items():
        dest = scored_cache_path(name, qb_from_pair=qb_from_pair, fold=fold_name)
        if dest.is_file():
            with dest.open("rb") as fh:
                have[name] = pickle.load(fh)
        else:
            want[name] = stems
    if want:
        print(f"[chunked] scoring {len(want)} candidate(s) over {len(keys)} shots; "
              f"{len(have)} cached", flush=True)
        fresh = scored_chunked(
            want, keys, fold_name=fold_name, member_label=member_label, chunk=chunk,
            qb_from_pair=qb_from_pair, workers=workers, device=device,
        )
        for name, scored in fresh.items():
            dest = scored_cache_path(name, qb_from_pair=qb_from_pair, fold=fold_name)
            tmp = dest.with_suffix(".pkl.tmp")
            with tmp.open("wb") as fh:
                pickle.dump(scored, fh, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(dest)
        have.update(fresh)
    return have


def reduce_exact(acc, ref_stats: dict) -> dict:
    """Unrounded mirror of ``metrics.finalize_machine``'s composite arithmetic.

    ``finalize_machine`` rounds every term to 6 decimals, which quantizes the 1e-4
    effects this harness exists to resolve. This reproduces the same reduction without
    the rounding; ``tests/test_probe9.py`` pins it against the official row.
    """
    from common import N_CONS, N_SCALARS, W_CONS, W_LCFS, W_PSI, W_QB
    from metrics import ss_tot

    tot_psi = ss_tot(ref_stats["psi_sum"], ref_stats["psi_sumsq"], ref_stats["psi_n"])
    r2_psi = 1.0 - acc.ss_res_psi[acc.psi_sign] / tot_psi if tot_psi > 0 else 0.0

    r2_each = np.zeros(N_SCALARS)
    for j in range(N_SCALARS):
        tot_j = ss_tot(ref_stats["scal_sum"][j], ref_stats["scal_sumsq"][j],
                       ref_stats["scal_n"][j])
        r2_each[j] = 1.0 - acc.ss_res_scal[j] / tot_j if tot_j > 0 else 0.0
    r2_qb = float(np.mean(r2_each))

    r2_cons = np.full(N_CONS, np.nan)
    for j in range(N_CONS):
        tot_j = ss_tot(ref_stats["cons_sum"][j], ref_stats["cons_sumsq"][j],
                       ref_stats["cons_n"][j])
        if tot_j > 0:
            r2_cons[j] = 1.0 - acc.ss_res_cons[j] / tot_j
    ok = np.isfinite(r2_cons)
    consistency = float(np.mean(np.maximum(0.0, r2_cons[ok]))) if ok.any() else 0.0
    dlcfs = float(np.mean(acc.shot_dlcfs)) if acc.shot_dlcfs else 1.0
    s = (W_PSI * max(0.0, r2_psi)
         + W_QB * max(0.0, r2_qb)
         + W_LCFS * (1.0 - min(1.0, dlcfs))
         + W_CONS * consistency)
    return {
        "S": float(s),
        "r2_psi": float(r2_psi),
        "r2_qb": float(r2_qb),
        "dlcfs": float(dlcfs),
        "consistency": float(consistency),
        "r2_cons_each": [None if not np.isfinite(v) else float(v) for v in r2_cons],
        "n_shots": int(acc.n_shots_scored),
    }


def finalize_subset(scored: dict, idx: Sequence[int], *, official: bool = False) -> dict:
    """Unrounded official row for an arbitrary shot multiset, from cached parts only."""
    from metrics import Accum, finalize_machine

    acc = Accum()
    acc.psi_sign = scored["psi_sign"]
    for i in idx:
        acc.add(scored["parts"][int(i)])
    ref_stats = sum_ref_moments(scored["moments"], idx)
    out = reduce_exact(acc, ref_stats)
    if official:
        out["official"] = finalize_machine(acc, ref_stats)
    return out


def per_shot_S(scored: dict, idx: Sequence[int] | None = None) -> list[float]:
    """Leave-one-in S per shot. Only a diagnostic; pooled R2 is not a per-shot mean."""
    idx = list(range(len(scored["parts"]))) if idx is None else list(idx)
    return [float(finalize_subset(scored, [i])["S"]) for i in idx]


def bootstrap_delta(
    score_a: Callable[[np.ndarray], float],
    score_b: Callable[[np.ndarray], float],
    n_shots: int,
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    """Paired shot bootstrap of ``score_a - score_b``.

    One shared resample per replicate is what makes the comparison paired: the fold
    noise that dominates an 80-shot aggregate (about 0.00233 sigma on Consistency)
    cancels, leaving the candidate difference.
    """
    if n_shots <= 0:
        raise ValueError("n_shots must be positive")
    full = np.arange(n_shots)
    s_a_full, s_b_full = float(score_a(full)), float(score_b(full))
    rng = np.random.default_rng(seed)
    deltas = np.empty(int(n_boot), dtype=np.float64)
    a_vals = np.empty(int(n_boot), dtype=np.float64)
    b_vals = np.empty(int(n_boot), dtype=np.float64)
    for r in range(int(n_boot)):
        idx = rng.choice(n_shots, n_shots, replace=True)
        a_vals[r] = score_a(idx)
        b_vals[r] = score_b(idx)
        deltas[r] = a_vals[r] - b_vals[r]
    sd = float(deltas.std(ddof=1)) if deltas.size > 1 else 0.0
    mean = float(deltas.mean())
    return {
        "n_shots": int(n_shots),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "S_a": s_a_full,
        "S_b": s_b_full,
        "delta_full": s_a_full - s_b_full,
        "delta_mean": mean,
        "paired_sd": sd,
        "paired_sigma": mean / sd if sd > 0 else (0.0 if mean == 0 else float("inf")),
        "p_better": float(np.mean(deltas > 0.0)),
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "S_a_boot_mean": float(a_vals.mean()),
        "S_b_boot_mean": float(b_vals.mean()),
    }


def paired_S_bootstrap(
    scored_a: dict,
    scored_b: dict,
    *,
    folds: Sequence[str] = ("all80", "worst40"),
    keys: Sequence[str] | None = None,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    fold_index_map: dict[str, Sequence[int]] | None = None,
) -> dict:
    """Paired bootstrap on the full composite S, on each named fold.

    ``scored_*`` come from :func:`score_parts`. The resample is over shots inside the
    fold, so ``worst40`` bootstraps 40 of 40 rather than a sub-subsample of 80.
    """
    if scored_a["psi_sign"] != scored_b["psi_sign"]:
        sign_note = [f"psi_sign_mismatch:{scored_a['psi_sign']}vs{scored_b['psi_sign']}"]
    else:
        sign_note = []
    out: dict = {"folds": {}, "notes": sign_note}
    if fold_index_map:
        # An explicit map is authoritative: silently falling back to the legacy 80's
        # indices for a fold the caller did not supply is how folds get crossed.
        folds = tuple(f for f in folds if f in fold_index_map) or tuple(fold_index_map)
    for fold in folds:
        if fold_index_map and fold in fold_index_map:
            base = np.asarray(list(fold_index_map[fold]), dtype=int)
        else:
            base = fold_indices(fold, keys)
        full_a = finalize_subset(scored_a, base, official=True)
        full_b = finalize_subset(scored_b, base, official=True)

        def _mk(scored):
            def _score(local_idx: np.ndarray) -> float:
                return float(finalize_subset(scored, base[np.asarray(local_idx, dtype=int)])["S"])
            return _score

        boot = bootstrap_delta(
            _mk(scored_a), _mk(scored_b), len(base), n_boot=n_boot, seed=seed
        )
        boot["terms_a"] = {k: full_a[k] for k in
                           ("S", "r2_psi", "r2_qb", "dlcfs", "consistency")}
        boot["terms_b"] = {k: full_b[k] for k in
                           ("S", "r2_psi", "r2_qb", "dlcfs", "consistency")}
        boot["delta_terms"] = {
            k: float(full_a[k] - full_b[k])
            for k in ("S", "r2_psi", "r2_qb", "dlcfs", "consistency")
        }
        boot["official_a"] = full_a.get("official")
        boot["official_b"] = full_b.get("official")
        out["folds"][fold] = boot
    return out


def promote_verdict(
    comparison: dict,
    *,
    folds: Sequence[str] = ("all80", "worst40"),
    p_better: float = P_BETTER,
) -> tuple[str, list[str]]:
    """PROMOTE-WORTHY only if delta S is positive and clears p_better on every fold.

    Requiring both folds is the guard against fold overfitting: worst40 is the fold
    that brackets the public score, all80 is the one every prior receipt used.
    """
    reasons: list[str] = []
    for fold in folds:
        body = (comparison.get("folds") or {}).get(fold)
        if not body:
            reasons.append(f"missing_{fold}")
            continue
        if float(body["delta_mean"]) <= 0.0:
            reasons.append(f"delta_S_{fold}")
        if float(body["p_better"]) < p_better:
            reasons.append(f"p_better_{fold}")
    verdict = "PROMOTE-WORTHY" if not reasons else "NULL"
    return verdict, reasons


def headline(comparison: dict, fold: str = "all80") -> dict:
    """The three numbers every receipt carries, taken from the named fold."""
    body = (comparison.get("folds") or {}).get(fold) or {}
    return {
        "delta_S": body.get("delta_mean"),
        "delta_S_sigma": body.get("paired_sd"),
        "p_better": body.get("p_better"),
    }


def scored_for_mix(
    mix: str,
    fold: dict,
    *,
    stems: Sequence[str] | None = None,
    qb_from_pair: bool = True,
    workers: int | None = None,
    device=None,
) -> dict:
    """Score a mix on the holdout, returning cached per-shot parts.

    ``stems`` lets a Tier B arm score a mix that is not in the registry yet -- a fresh
    tailfit member swapped into the shipped five, for instance.
    """
    stems = list(stems) if stems else mix_stems(mix)
    packed = mix_z(
        stems, keys=fold["keys"], label=fold.get("member_label", "hold"),
        tag=fold.get("name", ""), device=device,
    )
    preds = preds_from_z(
        packed["z"], packed["shot_ids"], fold["keys"],
        q95=packed["q95"], betaN=packed["betaN"], qb_from_pair=qb_from_pair,
    )
    scored = score_parts(fold, preds, workers=workers)
    scored["mix"] = mix
    scored["stems"] = stems
    return scored


def rescore_scalars(
    scored: dict,
    fold: dict,
    *,
    q95: np.ndarray,
    betaN: np.ndarray,
    shot_ids: np.ndarray,
) -> dict:
    """Swap in new q95/betaN predictions without re-running the geometry pass.

    Only ``ss_res_scal`` depends on the value scalars, and its formula is three lines of
    ``local_score.score_shot``: non-finite predictions are replaced by the fold mean, so
    a NaN frame scores exactly as a constant predictor. Recomputing just that term makes
    an abstention grid search free instead of one scoring pass per rule.
    """
    from common import N_SCALARS, SCALARS

    mean_scal = fold["means"][1]
    q95 = np.asarray(q95, np.float64).reshape(-1)
    betaN = np.asarray(betaN, np.float64).reshape(-1)
    preds = {"q95": q95, "betaN": betaN}
    frames = fold.get("T")
    parts = []
    for i, (gt, part) in enumerate(zip(fold["gts"], scored["parts"])):
        sl = shot_ids == i
        T = int(frames[i]) if frames else int(np.asarray(gt["psi"]).shape[0])
        ss = np.zeros(N_SCALARS)
        for j, name in enumerate(SCALARS):
            g = np.asarray(gt[name], np.float64)
            defined = np.isfinite(g)
            if not defined.any():
                continue
            p = preds[name][sl]
            if p.shape != (T,):
                ss[j] = float(np.sum((g[defined] - mean_scal[j]) ** 2))
                continue
            pj = np.where(np.isfinite(p[defined]), p[defined], mean_scal[j])
            ss[j] = float(np.sum((g[defined] - pj) ** 2))
        new = dict(part)
        new["ss_res_scal"] = ss
        parts.append(new)
    return {**scored, "parts": parts}


def scored_cache_path(mix: str, *, qb_from_pair: bool = True, fold: str = "all80") -> Path:
    suffix = f"_{_slug(fold)}" if fold and fold != "all80" else ""
    return (cache_dir()
            / f"scored_{_slug(mix)}_{'lockqb' if qb_from_pair else 'headqb'}{suffix}.pkl")


def scored_cached(
    mix: str,
    fold: dict,
    *,
    stems: Sequence[str] | None = None,
    qb_from_pair: bool = True,
    refresh: bool = False,
    workers: int | None = None,
    device=None,
) -> dict:
    """Per-shot scorer parts for a mix, cached to disk.

    Parts are a few kilobytes of residual sums per shot, so caching them turns every
    later probe's re-adjudication into pure arithmetic instead of another scoring pass.
    """
    import pickle

    dest = scored_cache_path(mix, qb_from_pair=qb_from_pair,
                             fold=fold.get("name", "all80"))
    if dest.is_file() and not refresh:
        with dest.open("rb") as fh:
            scored = pickle.load(fh)
        if scored.get("moments") is None:
            scored["moments"] = fold["moments"]
        return scored
    scored = scored_for_mix(
        mix, fold, stems=stems, qb_from_pair=qb_from_pair, workers=workers, device=device
    )
    tmp = dest.with_suffix(".pkl.tmp")
    with tmp.open("wb") as fh:
        pickle.dump(scored, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(dest)
    return scored


def bootstrap_aggregate(
    scored: dict, idx: Sequence[int], *, n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> dict:
    """One-arm shot bootstrap: how much fold noise is in an aggregate on this fold.

    This is what makes Tier C decidable or not. Aggregate Consistency carries about
    0.00233 one-sigma on 80 shots, which is larger than every effect the harness is
    trying to resolve; the same number on 800 shots is the point of sealing them.
    """
    base = np.asarray(list(idx), dtype=int)
    rng = np.random.default_rng(seed)
    S = np.empty(int(n_boot))
    cons = np.empty(int(n_boot))
    for r in range(int(n_boot)):
        pick = base[rng.choice(len(base), len(base), replace=True)]
        row = finalize_subset(scored, pick)
        S[r] = row["S"]
        cons[r] = row["consistency"]
    full = finalize_subset(scored, base)
    return {
        "n_shots": int(len(base)),
        "n_boot": int(n_boot),
        "S": float(full["S"]),
        "consistency": float(full["consistency"]),
        "sigma_S": float(S.std(ddof=1)),
        "sigma_cons": float(cons.std(ddof=1)),
    }


def load_scored(mix: str, *, qb_from_pair: bool = True, fold: str = "all80") -> dict | None:
    import pickle

    dest = scored_cache_path(mix, qb_from_pair=qb_from_pair, fold=fold)
    if not dest.is_file():
        return None
    with dest.open("rb") as fh:
        return pickle.load(fh)


def scored_and_fold(
    mix: str = INCUMBENT_MIX, *, workers: int | None = None, lite: bool = False
) -> tuple[dict, dict]:
    """Fold state plus a mix's cached parts, taking the cheapest route available.

    ``lite=True`` skips the flux cubes, which is all a scalar-only probe needs.
    """
    scored = load_scored(mix)
    if scored is not None:
        return (fold_lite() if lite else fold_reference_cached()), scored
    fold = fold_reference_cached()
    return fold, scored_cached(mix, fold, workers=workers)


def score_parts_for_mix(mix: str, **kw) -> tuple[dict, dict]:
    """Convenience: build the fold once and score a mix against it."""
    fold = fold_reference_cached()
    return fold, scored_for_mix(mix, fold, **kw)


def instrument_self_test(
    scored: dict,
    *,
    n_boot: int = 64,
    tol: float = 1e-9,
    fold_index_map: dict[str, Sequence[int]] | None = None,
) -> dict:
    """A candidate against itself must give exactly zero delta on every fold.

    Cheap, but it is the check that catches the failure that would silently invalidate
    every number in the report: fold indices misaligned between the two arms.
    """
    comp = paired_S_bootstrap(scored, scored, n_boot=n_boot,
                              fold_index_map=fold_index_map)
    worst = 0.0
    for body in comp["folds"].values():
        worst = max(worst, abs(float(body["delta_mean"])), abs(float(body["paired_sd"])))
    return {
        "ok": worst <= tol,
        "worst_abs": worst,
        "tol": tol,
        "n_boot": n_boot,
        "folds": {k: {"delta_mean": v["delta_mean"], "paired_sd": v["paired_sd"]}
                  for k, v in comp["folds"].items()},
    }
