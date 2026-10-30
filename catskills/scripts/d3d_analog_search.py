#!/usr/bin/env python3
"""Donor-pool in-fit analog screen. Never uses the canonical 80 to pick a cell.

kNN groups (field, embed, metric) × fold are independent. ``--workers`` runs
them in a fork pool so each job has its own BLAS threads. Thread-pooling
the GEMMs would share one OpenBLAS pool and stay ~2-core.
``d3d_analog_run --workers`` forwards here, not cache-only.

  .venv/bin/python -m catskills.scripts.d3d_analog_search
  .venv/bin/python -m catskills.scripts.d3d_analog_search --max-donor 2048 --folds 3
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import signal
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

from catskills.cpu_budget import limit_threads, workers as cpu_workers

limit_threads(2)

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fusion_scoring"))

from catskills.d3d_analog import (  # noqa: E402
    ALPHA_GRID, GAUSS_TEMPS, K_GRID, N_HOLD, PHASE_LENGTHS, TAG, AnalogConfig,
    FoldPreprocessor, LeakageError, analog_root, apply_correction,
    assert_no_leakage, blocked_distances, canonical_split, coeff_mse,
    desc_cache_dir, descriptor_a5, donor_folds, interpolate_phase_residual,
    knn_from_distances, load_shot_payload, lock_rank, neighbor_mask,
    neighbor_weights, permute_payloads, random_neighbors, reverse_descriptor_phase,
    dtw_rerank, shot_cache_dir, write_json_atomic,
)
from catskills.feature_cache import sample_cache_keys  # noqa: E402


def _heartbeat(stage: str, **extra) -> None:
    """Last-known stage on disk. Survives a kill if we get SIGTERM."""
    rec = {"stage": stage, "unix": time.time()}
    for k, v in extra.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            rec[k] = v
        else:
            rec[k] = str(v)
    try:
        write_json_atomic(analog_root() / "search_progress.json", rec)
    except Exception:
        pass
    bits = " ".join(f"{k}={rec[k]}" for k in extra if k in rec)
    print(f"[{TAG}] {stage}" + (f" {bits}" if bits else ""), flush=True)


def _install_death_hooks() -> None:
    def hook(signum, _frame):
        _heartbeat("killed", signum=int(signum))
        raise SystemExit(128 + int(signum))
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, hook)
        except Exception:
            pass

REPR_KEYS = {
    "A0": ["A0"],
    "A4": ["A4"],
    "A1": [f"A1_L{L}" for L in PHASE_LENGTHS],
    "A2": [f"A2_L{L}" for L in PHASE_LENGTHS],
    "A3": [f"A3_L{L}" for L in PHASE_LENGTHS],
    "A5": [f"A3_L{L}" for L in PHASE_LENGTHS],
}
EMBEDS = ("none", "pca32", "pca64", "pca128", "whitened64")
METRICS = ("l2", "cosine")
WEIGHTS = ("uniform", "inverse", "gaussian")
PAYLOADS = ("constant", "p16", "p32")
SCREEN_REPR = ("A0", "A2", "A3", "A4")
SCREEN_LEN = (64,)
SCREEN_K = (4, 8, 16)
SCREEN_ALPHA = (0.5, 1.0)
PROMOTE_N = 20
CONS_TOP_N = 5


def _load_desc(key: str) -> dict[str, np.ndarray] | None:
    p = desc_cache_dir() / f"{key}.npz"
    if not p.is_file():
        return None
    with np.load(p) as d:
        return {n: np.asarray(d[n], np.float64) for n in d.files if n not in ("key",)}


def _pad_stack(rows: list[np.ndarray | None]) -> np.ndarray:
    from catskills.d3d_analog import pad_or_trim
    width = next(r.size for r in rows if r is not None)
    out = np.zeros((len(rows), width), np.float64)
    for i, r in enumerate(rows):
        if r is not None:
            out[i] = pad_or_trim(r, width)
    return out


def load_banks(keys: list[str], fields: list[str]) -> dict:
    """One pass over desc + shot caches. Prints every 1000 keys."""
    desc_rows = {f: [] for f in fields}
    pay = {"constant": [], "p16": [], "p32": []}
    mse0, e_mean, n_frames = [], [], []
    t0 = time.perf_counter()
    for n, k in enumerate(keys, 1):
        d = _load_desc(k)
        for f in fields:
            desc_rows[f].append(None if d is None or f not in d else d[f].ravel())
        pl = load_shot_payload(k)
        if pl is None or pl.z_gt is None:
            pay["constant"].append(np.zeros(60))
            pay["p16"].append(np.zeros(16 * 60))
            pay["p32"].append(np.zeros(32 * 60))
            mse0.append(0.0)
            e_mean.append(np.zeros(60))
            n_frames.append(1)
        else:
            zb = np.asarray(pl.z_base, np.float64)
            zg = np.asarray(pl.z_gt, np.float64)
            T = min(len(zb), len(zg))
            e = zb[:T] - zg[:T]
            pay["constant"].append(pl.residual_const.ravel())
            pay["p16"].append(pl.residual_p16.reshape(-1))
            pay["p32"].append(pl.residual_p32.reshape(-1))
            mse0.append(float((e ** 2).mean()))
            e_mean.append(e.mean(0))
            n_frames.append(T)
        if n % 1000 == 0 or n == len(keys):
            print(f"  load {n}/{len(keys)} {time.perf_counter() - t0:.0f}s", flush=True)
    out = {f: _pad_stack(desc_rows[f]) for f in fields}
    out["pay_constant"] = _pad_stack(pay["constant"])
    out["pay_p16"] = _pad_stack(pay["p16"])
    out["pay_p32"] = _pad_stack(pay["p32"])
    out["mse0"] = np.asarray(mse0, np.float64)
    out["e_mean"] = np.vstack(e_mean)
    out["n_frames"] = np.asarray(n_frames, np.int32)
    return out


def _stack_desc(keys: list[str], field: str) -> np.ndarray:
    return load_banks(keys, [field])[field]


def _payload_bank(keys: list[str], kind: str) -> np.ndarray:
    b = load_banks(keys, [])
    return b[f"pay_{kind}"]


def _z_pair(keys: list[str]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    zb, zg = [], []
    for k in keys:
        pl = load_shot_payload(k)
        if pl is None or pl.z_gt is None:
            zb.append(np.zeros((1, 60)))
            zg.append(np.zeros((1, 60)))
        else:
            zb.append(np.asarray(pl.z_base, np.float64))
            zg.append(np.asarray(pl.z_gt, np.float64))
    return zb, zg


def _cut_from_hats(mse0_q: np.ndarray, e_mean_q: np.ndarray, hats: np.ndarray,
                   n_q: np.ndarray, alpha: float) -> float:
    """Pooled coeff-MSE cut for shot-constant hats (n_query, 60)."""
    a = float(alpha)
    mse1 = mse0_q + 2.0 * a * (e_mean_q * hats).mean(1) + (a * a) * (hats ** 2).mean(1)
    w = np.maximum(n_q.astype(np.float64), 1.0)
    base = float((mse0_q * w).sum())
    cut = float((mse1 * w).sum())
    if base <= 0:
        return float("nan")
    return 1.0 - cut / base


def _mse_cut(zb, zg, hats, alpha) -> float:
    base = cut = 0.0
    n = 0
    for z0, z1, r in zip(zb, zg, hats):
        mse0 = coeff_mse(z0, z1)
        zc = apply_correction(z0, r, alpha=alpha)
        mse1 = coeff_mse(zc, z1)
        T = min(len(z0), len(z1))
        base += mse0 * T
        cut += mse1 * T
        n += T
    if n == 0 or base <= 0:
        return float("nan")
    return 1.0 - cut / base


def _payload_for_query(kind: str, raw: np.ndarray, n_frames: int) -> np.ndarray:
    if kind == "constant":
        return raw.reshape(-1)[:60]
    if kind == "p16":
        return interpolate_phase_residual(raw.reshape(16, 60), n_frames)
    if kind == "p32":
        return interpolate_phase_residual(raw.reshape(32, 60), n_frames)
    raise ValueError(kind)


def _embed_spec(name: str) -> tuple[int | None, bool]:
    if name == "none":
        return None, False
    if name.startswith("whitened"):
        return int(name.replace("whitened", "")), True
    if name.startswith("pca"):
        return int(name.replace("pca", "")), False
    raise ValueError(name)


# Fork workers inherit this. Do not pickle the banks per job.
_KNN_SHARE: dict = {}


def _legal_mask(q_keys: list[str], tr_keys: list[str], hold,
                n_q: int, n_tr: int) -> np.ndarray:
    """True = usable neighbor. Vectorized; no per-pair Python loop."""
    mask = np.ones((n_q, n_tr), dtype=bool)
    qk = np.asarray(q_keys, dtype=object)
    tk = np.asarray(tr_keys, dtype=object)
    mask &= qk[:, None] != tk[None, :]
    hset = set(hold)
    if hset:
        mask[:, np.fromiter((t in hset for t in tr_keys), dtype=bool, count=n_tr)] = False
    return mask


def _knn_ranks(X: np.ndarray, embed: str, metric: str,
               q_keys: list[str], tr_keys: list[str], hold,
               qp: np.ndarray, tp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fold-local embed + exact k=64 ranks. Pure; same result in a worker."""
    dim, whiten = _embed_spec(embed)
    prep = FoldPreprocessor(forbidden_keys=set(hold) | set(q_keys))
    prep.fit(tr_keys, X[tp], dim=dim, whiten=whiten)
    Q = prep.transform(X[qp])
    Tm = prep.transform(X[tp])
    dist = blocked_distances(Q, Tm, metric)
    mask = _legal_mask(q_keys, tr_keys, hold, dist.shape[0], dist.shape[1])
    return knn_from_distances(dist, 64, mask=mask)


def _knn_share_job(job: tuple) -> tuple:
    fi, field, embed, metric = job
    fold = _KNN_SHARE["folds"][fi]
    idx, dd = _knn_ranks(
        _KNN_SHARE["desc"][field], embed, metric,
        fold["q_keys"], fold["tr_keys"], _KNN_SHARE["hold"],
        fold["qp"], fold["tp"],
    )
    return (fi, field, embed, metric), (idx, dd)


def _run_knn_jobs(jobs: list[tuple], n_workers: int) -> list[tuple]:
    if n_workers <= 1 or len(jobs) <= 1:
        out = []
        for j in jobs:
            rec = _knn_share_job(j)
            fi, field, embed, metric = rec[0]
            print(f"  knn fold={fi} {field} {embed} {metric} done", flush=True)
            out.append(rec)
        return out
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        return _run_knn_jobs(jobs, 1)
    out = []
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        futs = {ex.submit(_knn_share_job, j): j for j in jobs}
        for fut in as_completed(futs):
            rec = fut.result()
            fi, field, embed, metric = rec[0]
            print(f"  knn fold={fi} {field} {embed} {metric} done", flush=True)
            out.append(rec)
    return out


def evaluate_cell(cfg: AnalogConfig, q_idx, tr_idx, desc_q, desc_tr,
                  pay_tr, zb_q, zg_q, q_keys, tr_keys, hold_keys,
                  control: str = "") -> dict:
    dim, whiten = _embed_spec(cfg.embed)
    prep = FoldPreprocessor(forbidden_keys=set(hold_keys) | set(q_keys))
    prep.fit([tr_keys[i] for i in range(len(tr_keys))], desc_tr, dim=dim, whiten=whiten)
    Q = prep.transform(desc_q)
    T = prep.transform(desc_tr)
    metric = "cosine" if cfg.metric == "cosine" else "l2"
    dist = blocked_distances(Q, T, metric=metric)
    masks = []
    excluded = []
    for i, qk in enumerate(q_keys):
        try:
            m = neighbor_mask(tr_keys, qk, forbidden=hold_keys,
                              index_vecs=T, query_vec=Q[i], dup_thresh=cfg.dup_thresh)
        except LeakageError:
            m = np.zeros(len(tr_keys), dtype=bool)
            excluded.append(qk)
        masks.append(m)
        assert_no_leakage([tr_keys[j] for j in np.flatnonzero(m)[:cfg.k]],
                          set(hold_keys) | {qk})
    mask = np.stack(masks)
    k_use = min(cfg.k, max(1, int(mask.sum(1).min()) if mask.any() else 1))
    if control == "random":
        idx = random_neighbors(len(tr_keys), len(q_keys), k_use, seed=0)
        # still drop illegal
        for i, qk in enumerate(q_keys):
            for j, jdx in enumerate(idx[i]):
                if tr_keys[int(jdx)] in set(hold_keys) | {qk}:
                    idx[i, j] = int(np.flatnonzero(mask[i])[0]) if mask[i].any() else 0
        dd = np.take_along_axis(dist, idx, axis=1)
    else:
        idx, dd = knn_from_distances(dist, k_use, mask=mask)
    if control == "k256":
        idx, dd = knn_from_distances(dist, min(256, len(tr_keys)), mask=mask)
        k_use = idx.shape[1]
    tau = cfg.gauss_temp
    if cfg.weight == "gaussian":
        med = np.nanmedian(dd[:, -1]) if dd.size else 1.0
        tau = cfg.gauss_temp * (med if np.isfinite(med) and med > 0 else 1.0)
    w = neighbor_weights(dd, cfg.weight, tau)
    pay = pay_tr
    if control == "permute":
        pay = permute_payloads(pay_tr, seed=1)
    hats = []
    if control == "global_mean":
        mu = pay_tr.mean(0)
        for z0 in zb_q:
            hats.append(_payload_for_query(cfg.payload, mu, len(z0)))
    elif control == "oracle":
        for z0, z1 in zip(zb_q, zg_q):
            hats.append((z1 - z0).mean(0))
    else:
        for i, z0 in enumerate(zb_q):
            neigh = pay[idx[i]]
            hat = (w[i][:, None] * neigh).sum(0)
            hats.append(_payload_for_query(cfg.payload, hat, len(z0)))
    cut = _mse_cut(zb_q, zg_q, hats, 0.0 if control == "alpha0" else cfg.alpha)
    return {
        "cfg": cfg,
        "cell": cfg.cell_id() + (f"__{control}" if control else ""),
        "mse_cut": float(cut),
        "n_query": len(q_keys),
        "n_index": len(tr_keys),
        "n_excluded_dup": len(excluded),
        "control": control,
    }


def cheap_grid() -> list[AnalogConfig]:
    out = []
    for rep in SCREEN_REPR:
        lengths = SCREEN_LEN if rep not in ("A0", "A4") else (32,)
        for L in lengths:
            for emb in ("none", "pca64"):
                for met in METRICS:
                    for k in SCREEN_K:
                        for wt in ("uniform", "inverse"):
                            for pay in ("constant", "p16"):
                                for a in SCREEN_ALPHA:
                                    out.append(AnalogConfig(
                                        representation=rep, phase_length=L, embed=emb,
                                        metric=met, k=k, weight=wt, payload=pay, alpha=a,
                                    ))
    return out


def full_grid_around(winners: list[AnalogConfig]) -> list[AnalogConfig]:
    """Re-score the cheap top-N on all folds. Sweep k/alpha only on the leader."""
    extra = list(winners)
    seen = {c.cell_id() for c in winners}
    if winners:
        base = winners[0]
        for k in (8, 16, 32):
            for a in (0.25, 0.5, 0.75, 1.0):
                cfg = AnalogConfig(
                    representation=base.representation,
                    phase_length=base.phase_length, embed=base.embed,
                    metric=base.metric, k=k, weight=base.weight,
                    payload=base.payload, alpha=a, gauss_temp=base.gauss_temp,
                )
                if cfg.cell_id() not in seen:
                    extra.append(cfg)
                    seen.add(cfg.cell_id())
    return extra


def parse_cell_id(cell: str) -> AnalogConfig:
    """Invert AnalogConfig.cell_id for --from-cheap."""
    tok = cell.split("_")
    rep = tok[0]
    length = 64
    rest = tok[1:]
    if rest and rest[0].startswith("L") and rest[0][1:].isdigit():
        length = int(rest[0][1:])
        rest = rest[1:]
    embed = rest[0] if rest else "none"
    metric = rest[1] if len(rest) > 1 else "l2"
    k, weight, payload, alpha = 8, "uniform", "constant", 1.0
    for p in rest[2:]:
        if p.startswith("k") and p[1:].isdigit():
            k = int(p[1:])
        elif p in ("uniform", "inverse", "gaussian"):
            weight = p
        elif p in PAYLOADS:
            payload = p
        elif p.startswith("a"):
            try:
                alpha = float(p[1:])
            except ValueError:
                pass
    return AnalogConfig(representation=rep, phase_length=length, embed=embed,
                        metric=metric, k=k, weight=weight, payload=payload, alpha=alpha)


def _field_for(cfg: AnalogConfig) -> str:
    if cfg.representation in ("A0", "A4"):
        return cfg.representation
    if cfg.representation == "A5":
        return f"A3_L{cfg.phase_length}"
    return f"{cfg.representation}_L{cfg.phase_length}"


def run_screen(donor: list[str], hold: list[str], folds_idx: list[np.ndarray],
               used_folds: int, train_cap: int, cells: list[AnalogConfig],
               controls: bool, n_workers: int = 1) -> list[dict]:
    # restrict to shots that have both desc and payload
    ready = [k for k in donor
             if (desc_cache_dir() / f"{k}.npz").is_file()
             and (shot_cache_dir() / f"{k}.npz").is_file()]
    ready_set = set(ready)
    print(f"[{TAG}] ready donor shots {len(ready)}/{len(donor)}", flush=True)
    results = []
    # cache descriptor fields we need
    fields = sorted({_field_for(c) for c in cells} | {"A0", "A3_L64"})
    print(f"[{TAG}] loading banks fields={fields}", flush=True)
    banks = load_banks(ready, fields)
    desc = {f: banks[f] for f in fields}
    if any(c.representation == "A5" for c in cells):
        desc["A5_L64"] = np.hstack([desc["A0"], desc["A3_L64"]])
    pay = {kind: banks[f"pay_{kind}"] for kind in PAYLOADS}
    mse0_all, e_mean_all, n_all = banks["mse0"], banks["e_mean"], banks["n_frames"]
    zb_all, zg_all = _z_pair(ready)

    groups: dict[tuple, list[AnalogConfig]] = {}
    for cfg in cells:
        field = _field_for(cfg)
        if cfg.representation == "A5":
            field = "A5_L64" if "A5_L64" in desc else field
        groups.setdefault((field, cfg.embed, cfg.metric), []).append(cfg)

    fold_meta: list[dict] = []
    jobs: list[tuple] = []
    for fi, fidx in enumerate(folds_idx[:used_folds]):
        q_keys = [donor[i] for i in fidx if donor[i] in ready_set]
        tr_keys = [k for k in ready if k not in set(q_keys)]
        if train_cap and len(tr_keys) > train_cap:
            rng = np.random.default_rng(0)
            pick = np.sort(rng.choice(len(tr_keys), train_cap, replace=False))
            tr_keys = [tr_keys[i] for i in pick]
        if len(q_keys) < 8 or len(tr_keys) < 16:
            print(f"[{TAG}] fold {fi} too small q={len(q_keys)} tr={len(tr_keys)}", flush=True)
            continue
        q_pos = [ready.index(k) for k in q_keys]
        tr_pos = [ready.index(k) for k in tr_keys]
        qp = np.asarray(q_pos)
        tp = np.asarray(tr_pos)
        print(f"[{TAG}] fold {fi}: query {len(q_keys)} index {len(tr_keys)} "
              f"cells {len(cells)}", flush=True)
        fold_meta.append({
            "fi": fi, "q_keys": q_keys, "tr_keys": tr_keys,
            "q_pos": q_pos, "tr_pos": tr_pos, "qp": qp, "tp": tp,
            "zb_q": [zb_all[i] for i in q_pos],
            "zg_q": [zg_all[i] for i in q_pos],
        })
        for field, embed, metric in groups:
            if field in desc:
                jobs.append((fi, field, embed, metric))

    _KNN_SHARE.clear()
    _KNN_SHARE["desc"] = desc
    _KNN_SHARE["hold"] = list(hold)
    _KNN_SHARE["folds"] = {
        m["fi"]: {k: m[k] for k in ("q_keys", "tr_keys", "qp", "tp")}
        for m in fold_meta
    }
    print(f"[{TAG}] knn {len(jobs)} jobs workers={n_workers}", flush=True)
    _heartbeat("knn", n_jobs=len(jobs), n_workers=n_workers, n_cells=len(cells))
    caches: dict[int, dict] = {}
    for (fi, field, embed, metric), ranks in _run_knn_jobs(jobs, n_workers):
        caches.setdefault(fi, {})[(field, embed, metric)] = ranks
    _heartbeat("knn_done", n_jobs=len(jobs), n_folds=len(fold_meta))

    n_score = len(fold_meta) * len(cells)
    scored = 0
    t_score = time.perf_counter()
    for m in fold_meta:
        fi, q_keys, tr_keys = m["fi"], m["q_keys"], m["tr_keys"]
        q_pos, tr_pos, qp, tp = m["q_pos"], m["tr_pos"], m["qp"], m["tp"]
        zb_q, zg_q = m["zb_q"], m["zg_q"]
        knn_cache = caches.get(fi, {})
        _heartbeat("score_fold", fold=fi, n_cells=len(cells), n_query=len(q_keys))
        bar = tqdm(cells, desc=f"score fold {fi}", unit="cell",
                   file=sys.stdout, mininterval=1.0, dynamic_ncols=True)
        for cfg in bar:
            field = _field_for(cfg)
            if cfg.representation == "A5":
                field = "A5_L64" if "A5_L64" in desc else field
            key = (field, cfg.embed, cfg.metric)
            if key not in knn_cache:
                continue
            idx64, dd64 = knn_cache[key]
            k_use = min(cfg.k, idx64.shape[1])
            idx, dd = idx64[:, :k_use], dd64[:, :k_use]
            w = neighbor_weights(dd, cfg.weight, cfg.gauss_temp)
            pay_tr = pay[cfg.payload][tp]
            hats = (w[:, :, None] * pay_tr[idx]).sum(1)
            if cfg.payload == "constant":
                cut = _cut_from_hats(mse0_all[qp], e_mean_all[qp], hats[:, :60],
                                     n_all[qp], cfg.alpha)
            else:
                hats_use = [_payload_for_query(cfg.payload, hats[i], len(z0))
                            for i, z0 in enumerate(zb_q)]
                cut = _mse_cut(zb_q, zg_q, hats_use, cfg.alpha)
            results.append({
                "cfg": cfg, "cell": cfg.cell_id(), "mse_cut": float(cut),
                "n_query": len(q_keys), "n_index": len(tr_keys),
                "n_excluded_dup": 0, "control": "", "fold": fi,
            })
            scored += 1
            if scored == 1 or scored % 25 == 0:
                write_json_atomic(analog_root() / "search_progress.json", {
                    "stage": "score", "fold": fi, "scored": scored,
                    "n_score": n_score, "cell": cfg.cell_id(),
                    "payload": cfg.payload, "unix": time.time(),
                    "seconds": round(time.perf_counter() - t_score, 1),
                })
        if controls:
            proto = AnalogConfig(representation="A2", phase_length=64, embed="pca64",
                                 metric="l2", k=8, weight="uniform", payload="constant",
                                 alpha=1.0)
            field = _field_for(proto)
            ctls = ("random", "permute", "global_mean", "k256", "oracle", "alpha0")
            for ctl in tqdm(ctls, desc=f"controls fold {fi}", unit="ctl",
                            file=sys.stdout, mininterval=0.5):
                _heartbeat("control", fold=fi, cell=ctl)
                rec = evaluate_cell(
                    proto, q_pos, tr_pos, desc[field][q_pos], desc[field][tr_pos],
                    pay[proto.payload][tr_pos], zb_q, zg_q, q_keys, tr_keys, hold,
                    control=ctl,
                )
                rec["fold"] = fi
                results.append(rec)
            a0 = AnalogConfig(representation="A0", phase_length=32, embed="none",
                              metric="l2", k=8, weight="uniform", payload="constant",
                              alpha=1.0)
            _heartbeat("control", fold=fi, cell="A0_aggregate")
            rec = evaluate_cell(
                a0, q_pos, tr_pos, desc["A0"][q_pos], desc["A0"][tr_pos],
                pay["constant"][tr_pos], zb_q, zg_q, q_keys, tr_keys, hold,
            )
            rec["fold"] = fi
            rec["ablation"] = "A0_aggregate"
            results.append(rec)
            rev = desc[field][q_pos].copy()
            from catskills.red_nondim import N_MAG
            rev = np.vstack([reverse_descriptor_phase(v, 64, N_MAG) for v in rev])
            _heartbeat("control", fold=fi, cell="phase_reverse")
            rec = evaluate_cell(
                proto, q_pos, tr_pos, rev, desc[field][tr_pos],
                pay[proto.payload][tr_pos], zb_q, zg_q, q_keys, tr_keys, hold,
                control="phase_reverse",
            )
            rec["fold"] = fi
            results.append(rec)
    _heartbeat("score_done", scored=scored, seconds=round(time.perf_counter() - t_score, 1))
    return results


def aggregate_cells(rows: list[dict]) -> list[dict]:
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["cell"], []).append(r)
    out = []
    for cell, rs in by.items():
        cuts = [r["mse_cut"] for r in rs if np.isfinite(r.get("mse_cut", np.nan))]
        cfg = rs[0]["cfg"]
        out.append({
            "cell": cell,
            "cfg": cfg,
            "cfg_dict": cfg.as_dict() if isinstance(cfg, AnalogConfig) else {},
            "control": rs[0].get("control", ""),
            "ablation": rs[0].get("ablation", ""),
            "mse_cut": float(np.median(cuts)) if cuts else float("nan"),
            "mean_cut": float(np.mean(cuts)) if cuts else float("nan"),
            "worst_fold_cut": float(np.min(cuts)) if cuts else float("nan"),
            "n_folds": len(cuts),
            "pos_folds": int(sum(c > 0 for c in cuts)),
            "folds": cuts,
            "delta_cons": 0.0,  # filled later for top cells
        })
    out.sort(key=lock_rank)
    return out


def cheap_cons_proxy(cfg: AnalogConfig, donor: list[str], hold: list[str],
                     folds_idx: list[np.ndarray]) -> dict:
    """Paired bootstrap on coefficient-space residual energy as a Cons stand-in.

    Official derive is reserved for the locked holdout. Inner ranking uses the
    same paired-shot bootstrap machinery on per-shot coeff MSE (lower is better),
    reported as a signed sigma so the lock rule stays well-defined.
    """
    ready = [k for k in donor
             if (desc_cache_dir() / f"{k}.npz").is_file()
             and (shot_cache_dir() / f"{k}.npz").is_file()]
    ready_set = set(ready)
    field = _field_for(cfg)
    desc = _stack_desc(ready, field if field != "A5_L64" else "A3_L64")
    if cfg.representation == "A5":
        desc = np.hstack([_stack_desc(ready, "A0"), desc])
    pay = _payload_bank(ready, cfg.payload)
    zb_all, zg_all = _z_pair(ready)
    deltas = []
    per_fold = []
    for fi, fidx in enumerate(folds_idx):
        q_keys = [donor[i] for i in fidx if donor[i] in ready_set]
        tr_keys = [k for k in ready if k not in set(q_keys)]
        if len(q_keys) < 8:
            continue
        q_pos = [ready.index(k) for k in q_keys]
        tr_pos = [ready.index(k) for k in tr_keys]
        rec = evaluate_cell(
            cfg, q_pos, tr_pos, desc[q_pos], desc[tr_pos], pay[tr_pos],
            [zb_all[i] for i in q_pos], [zg_all[i] for i in q_pos],
            q_keys, tr_keys, hold,
        )
        per_fold.append(rec["mse_cut"])
        # shot-level paired: base mse vs corrected mse
        dim, whiten = _embed_spec(cfg.embed)
        prep = FoldPreprocessor(forbidden_keys=set(hold) | set(q_keys))
        prep.fit(tr_keys, desc[tr_pos], dim=dim, whiten=whiten)
        Q = prep.transform(desc[q_pos])
        Tm = prep.transform(desc[tr_pos])
        dist = blocked_distances(Q, Tm, cfg.metric)
        masks = [neighbor_mask(tr_keys, qk, forbidden=hold, index_vecs=Tm,
                               query_vec=Q[i], dup_thresh=cfg.dup_thresh)
                 for i, qk in enumerate(q_keys)]
        idx, dd = knn_from_distances(dist, cfg.k, mask=np.stack(masks))
        w = neighbor_weights(dd, cfg.weight, cfg.gauss_temp)
        dshot = []
        for i, qi in enumerate(q_pos):
            hat = (w[i][:, None] * pay[tr_pos][idx[i]]).sum(0)
            r = _payload_for_query(cfg.payload, hat, len(zb_all[qi]))
            b = coeff_mse(zb_all[qi], zg_all[qi])
            c = coeff_mse(apply_correction(zb_all[qi], r, cfg.alpha), zg_all[qi])
            dshot.append(b - c)
        rng = np.random.default_rng(42)
        boot = []
        x = np.asarray(dshot, np.float64)
        for _ in range(400):
            boot.append(float(rng.choice(x, len(x), replace=True).mean()))
        boot = np.asarray(boot)
        sigma = float(boot.mean() / max(boot.std(), 1e-12))
        p_better = float((boot > 0).mean())
        deltas.append({"fold": fi, "sigma": sigma, "p_better": p_better,
                       "mse_cut": rec["mse_cut"]})
    if not deltas:
        return {"delta_cons": 0.0, "p_better": 0.0, "pos_folds": 0, "per_fold": []}
    return {
        "delta_cons": float(np.mean([d["sigma"] for d in deltas])),
        "p_better": float(np.mean([d["p_better"] for d in deltas])),
        "pos_folds": int(sum(d["sigma"] > 0 for d in deltas)),
        "per_fold": deltas,
        "worst_fold_cut": float(np.min(per_fold)) if per_fold else float("nan"),
    }


def _load_ip(keys: list[str], length: int = 64) -> np.ndarray:
    rows = []
    for k in keys:
        p = desc_cache_dir() / f"{k}.npz"
        if not p.is_file():
            rows.append(np.zeros(length, np.float64))
            continue
        with np.load(p) as d:
            name = f"ip_L{length}"
            v = np.asarray(d[name] if name in d.files else np.zeros(length), np.float64).ravel()
        rows.append(v if v.size == length else np.pad(v, (0, max(0, length - v.size)))[:length])
    return np.vstack(rows)


def dtw_ablation(cfg: AnalogConfig, donor: list[str], hold: list[str],
                 folds_idx: list[np.ndarray], max_query: int = 128) -> dict:
    """Rerank the locked cell's top-64 by Ip DTW; retest k=4/8/16. Donor-pool only."""
    ready = [k for k in donor
             if (desc_cache_dir() / f"{k}.npz").is_file()
             and (shot_cache_dir() / f"{k}.npz").is_file()]
    ready_set = set(ready)
    field = _field_for(cfg)
    print(f"[{TAG}] DTW rerank {cfg.cell_id()} max_query={max_query}", flush=True)
    desc = _stack_desc(ready, field)
    pay = _payload_bank(ready, cfg.payload)
    zb_all, zg_all = _z_pair(ready)
    ip = _load_ip(ready, cfg.phase_length if cfg.phase_length in PHASE_LENGTHS else 64)
    rows = []
    for fi, fidx in enumerate(folds_idx):
        q_keys = [donor[i] for i in fidx if donor[i] in ready_set]
        tr_keys = [k for k in ready if k not in set(q_keys)]
        if len(q_keys) < 8 or len(tr_keys) < 16:
            continue
        rng = np.random.default_rng(0)
        if len(q_keys) > max_query:
            pick = np.sort(rng.choice(len(q_keys), max_query, replace=False))
            q_keys = [q_keys[int(i)] for i in pick]
        q_pos = [ready.index(k) for k in q_keys]
        tr_pos = [ready.index(k) for k in tr_keys]
        dim, whiten = _embed_spec(cfg.embed)
        prep = FoldPreprocessor(forbidden_keys=set(hold) | set(q_keys))
        prep.fit(tr_keys, desc[tr_pos], dim=dim, whiten=whiten)
        Q = prep.transform(desc[q_pos])
        Tm = prep.transform(desc[tr_pos])
        dist = blocked_distances(Q, Tm, cfg.metric)
        mask = _legal_mask(q_keys, tr_keys, hold, dist.shape[0], dist.shape[1])
        idx64, dd64 = knn_from_distances(dist, 64, mask=mask)
        for k in (4, 8, 16):
            hats = []
            for i, qi in enumerate(q_pos):
                cand = idx64[i]
                series_c = [ip[tr_pos[int(j)]] for j in cand]
                new_idx = dtw_rerank(ip[qi], series_c, cand, k)
                dd = np.array([
                    float(dd64[i, int(np.flatnonzero(cand == j)[0])])
                    if np.any(cand == j) else 1.0
                    for j in new_idx
                ], np.float64)
                w = neighbor_weights(dd[None, :], cfg.weight, cfg.gauss_temp)[0]
                hat = (w[:, None] * pay[tr_pos][new_idx]).sum(0)
                hats.append(_payload_for_query(cfg.payload, hat, len(zb_all[qi])))
            cut = _mse_cut([zb_all[i] for i in q_pos], [zg_all[i] for i in q_pos],
                           hats, cfg.alpha)
            rows.append({"fold": fi, "k": k, "mse_cut": float(cut),
                         "n_query": len(q_keys), "cell": cfg.cell_id() + f"_dtw_k{k}"})
        print(f"  DTW fold {fi} done", flush=True)
    by_k: dict[int, list[float]] = {}
    for r in rows:
        by_k.setdefault(int(r["k"]), []).append(float(r["mse_cut"]))
    summary = [
        {"k": k, "mse_cut": float(np.median(v)), "folds": v}
        for k, v in sorted(by_k.items())
    ]
    return {"rows": rows, "by_k": summary,
            "best": max(summary, key=lambda r: r["mse_cut"]) if summary else {}}


def in_fit_gate(winner: dict, controls: list[dict]) -> tuple[bool, str]:
    if winner.get("n_folds", 0) < 5 and winner.get("control"):
        return False, "winner is a control"
    if winner.get("control"):
        return False, "best cell is a control"
    if winner.get("pos_folds", 0) < winner.get("n_folds", 0):
        return False, "mse cut not positive in every fold"
    if float(winner.get("mse_cut") or 0) < 0.03:
        return False, f"median mse cut {winner.get('mse_cut')} < 0.03"
    if int(winner.get("cons_pos_folds") or 0) < 4:
        return False, "paired Cons proxy not positive in >=4 folds"
    if float(winner.get("p_better") or 0) < 0.95:
        return False, f"P(better) {winner.get('p_better')} < 0.95"
    # controls should be null-ish vs winner
    for c in controls:
        if c.get("control") == "oracle":
            continue
        if c.get("control") and float(c.get("mse_cut") or 0) > float(winner["mse_cut"]) + 1e-6:
            return False, f"control {c['control']} beat winner"
    return True, "pass"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--screen-folds", type=int, default=3)
    ap.add_argument("--max-donor", type=int, default=0, help="0 = full donor pool")
    ap.add_argument("--screen-train", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--from-cheap", action="store_true",
                    help="reuse in_fit_screen_cheap.json and skip the 3-fold screen")
    args = ap.parse_args()
    n_workers = cpu_workers(args.workers)
    _install_death_hooks()
    _heartbeat("start", workers=n_workers)

    keys = sample_cache_keys(0, 42, include_thomson=True)
    hold, donor = canonical_split(keys)
    if args.max_donor:
        donor = donor[: args.max_donor]
    folds = donor_folds(donor, n_folds=args.folds, seed=0)
    t0 = time.perf_counter()

    cheap_path = analog_root() / "in_fit_screen_cheap.json"
    if args.from_cheap and cheap_path.is_file():
        cheap = json.loads(cheap_path.read_text())
        promote_cfgs = [parse_cell_id(r["cell"]) for r in cheap.get("top") or []]
        print(f"[{TAG}] from-cheap {len(promote_cfgs)} cells", flush=True)
    else:
        cells = cheap_grid()
        print(f"[{TAG}] screen {len(cells)} cells on {args.screen_folds} folds, "
              f"train cap {args.screen_train} workers={n_workers}", flush=True)
        screen_rows = run_screen(donor, hold, folds, args.screen_folds, args.screen_train,
                                 cells, controls=True, n_workers=n_workers)
        screen = aggregate_cells(screen_rows)
        legal = [r for r in screen if not r.get("control")]
        top = legal[:PROMOTE_N] if legal else screen[:PROMOTE_N]
        write_json_atomic(cheap_path, {
            "n_cells": len(cells), "n_rows": len(screen_rows),
            "top": [{"cell": r["cell"], "mse_cut": r["mse_cut"], "folds": r["folds"]}
                    for r in top],
            "seconds": round(time.perf_counter() - t0, 1),
        })
        promote_cfgs = [r["cfg"] for r in top if isinstance(r["cfg"], AnalogConfig)]
    full_cells = full_grid_around(promote_cfgs) if promote_cfgs else cells[:20]
    print(f"[{TAG}] promote {len(promote_cfgs)} -> {len(full_cells)} cells "
          f"on {args.folds} folds", flush=True)
    full_rows = run_screen(donor, hold, folds, args.folds, 0, full_cells,
                           controls=True, n_workers=n_workers)
    full = aggregate_cells(full_rows)
    legal = [r for r in full if not r.get("control")]
    controls = [r for r in full if r.get("control")]
    write_json_atomic(analog_root() / "controls.json", {
        "rows": [{k: v for k, v in r.items() if k != "cfg"} | {"cfg": r.get("cfg_dict")}
                 for r in controls],
    })
    write_json_atomic(analog_root() / "ablations.json", {
        "rows": [{k: v for k, v in r.items() if k != "cfg"} | {"cfg": r.get("cfg_dict")}
                 for r in full if r.get("ablation") or r["cell"].startswith("A0")],
        "dtw": {},
    })

    top5 = legal[:CONS_TOP_N]
    print(f"[{TAG}] Cons-proxy on top {len(top5)}", flush=True)
    for r in tqdm(top5, desc="cons-proxy", unit="cell", file=sys.stdout):
        _heartbeat("cons_proxy", cell=r["cell"])
        prox = cheap_cons_proxy(r["cfg"], donor, hold, folds)
        r.update({
            "delta_cons": prox["delta_cons"],
            "p_better": prox["p_better"],
            "cons_pos_folds": prox["pos_folds"],
            "cons_per_fold": prox["per_fold"],
        })
    top5.sort(key=lock_rank)
    winner = top5[0] if top5 else (legal[0] if legal else full[0])
    dtw = {}
    if isinstance(winner.get("cfg"), AnalogConfig):
        try:
            dtw = dtw_ablation(winner["cfg"], donor, hold, folds)
        except Exception as e:
            dtw = {"error": str(e)}
    ok, why = in_fit_gate(winner, controls)
    payload = {
        "winner": {k: v for k, v in winner.items() if k != "cfg"} | {
            "cfg": winner["cfg"].as_dict() if isinstance(winner.get("cfg"), AnalogConfig)
            else winner.get("cfg_dict", {}),
        },
        "advanced": ok,
        "gate": why,
        "top5": [{k: v for k, v in r.items() if k != "cfg"} | {
            "cfg": r["cfg"].as_dict() if isinstance(r.get("cfg"), AnalogConfig) else r.get("cfg_dict"),
        } for r in top5],
        "dtw": dtw,
        "n_legal": len(legal),
        "seconds": round(time.perf_counter() - t0, 1),
        "note": "in_fit_screen: o1x5 saw these donor shots; not OOF proof",
    }
    write_json_atomic(analog_root() / "in_fit_screen.json", payload)
    abl_path = analog_root() / "ablations.json"
    if abl_path.is_file():
        abl = json.loads(abl_path.read_text())
        abl["dtw"] = dtw
        write_json_atomic(abl_path, abl)
    write_json_atomic(analog_root() / "winner.json", {
        "cfg": payload["winner"]["cfg"],
        "advanced": ok,
        "gate": why,
        "mse_cut": winner.get("mse_cut"),
        "delta_cons": winner.get("delta_cons"),
        "frozen_from": "in_fit_screen",
        "locked": True,
    })
    print(f"[{TAG}] winner {payload['winner'].get('cell')} cut={winner.get('mse_cut')} "
          f"advanced={ok} ({why})", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        try:
            write_json_atomic(analog_root() / "search_fail.json", {
                "error": traceback.format_exc(),
                "unix": time.time(),
            })
        except Exception:
            pass
        raise
