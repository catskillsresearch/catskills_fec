#!/usr/bin/env python3
"""Track 2: train ψ on FAIR-MAST archive labels. DIII-D ship stays o1x5_42for7.

  .venv/bin/python -m catskills.scripts.mast_warm_cache
  CRC_FEQ_MODEL_ID=CRC_FEQ_mast .venv/bin/python -m catskills.run_crc_feq_pinn_mast

Gate: official 3-shot holdout R²ψ >= 0.9976. q95/βN stay NaN (§31.3).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CRC_FEQ_MODEL_ID", "CRC_FEQ_mast")

import numpy as np
import torch
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from catskills.coeff_cache import N_PIX, reconstruct  # noqa: E402
from catskills.lab_paths import artifacts_dir, results_dir  # noqa: E402
from catskills.mast_cache import (  # noqa: E402
    CACHE_VERSION, EDGE_START, HOLDOUT_KEYS, KNOWN_FEATURE_VERSIONS, N_FEATURES,
    build_one, load_index, load_shot, load_row, load_vac, shot_from_mast_row,
    vac_from_row,
)
from catskills.mast_eval import load_holdout, score_holdout  # noqa: E402
from catskills.mast_qc import keep_set, load_qc_index, quarantine_set  # noqa: E402
from catskills.mast_win import frame_mask, win_ip_lo  # noqa: E402
from catskills.pinn_charcoal import (  # noqa: E402
    _clip_features, augment_temporal, build_trunk,
)

GATE_R2 = 0.9976
K = 60


def resolve_init_seed(seed: int, init_seed: int | None) -> int:
    """Torch init only. None means follow the data `--seed`."""
    return int(seed if init_seed is None else init_seed)


def _split(train_shots: int, seed: int, qc: bool = False, cache_version: str = CACHE_VERSION):
    entries = load_index(cache_version)
    if not entries:
        raise SystemExit(f"empty {cache_version} index; run mast_warm_cache --version {cache_version}")
    keys = [e["key"] for e in entries if e["key"] not in HOLDOUT_KEYS]
    rng = random.Random(seed)
    rng.shuffle(keys)
    sealed = keys[:80]
    train = keys[80:]
    if qc:
        keep = keep_set()
        if not keep:
            raise SystemExit("empty mast_qc_v1 keep list; run mast_qc_index")
        train = [k for k in train if k in keep]
    if train_shots > 0:
        train = train[:train_shots]
    return train, sealed


def _fit_width(X: np.ndarray, n_raw: int) -> np.ndarray:
    if X.shape[1] == n_raw:
        return X
    if X.shape[1] < n_raw:
        return np.hstack([X, np.zeros((X.shape[0], n_raw - X.shape[1]), dtype=X.dtype)])
    return X[:, :n_raw]


def _load_matrix(keys: list[str], residual: bool = False, cache_version: str = CACHE_VERSION,
                 ip_lo: float | None = None):
    Xs, Ys, ids = [], [], []
    skipped = 0
    win_empty = 0
    for i, key in enumerate(keys):
        got = load_shot(key, cache_version)
        if got is None:
            continue
        X, Y = got
        n = min(len(X), len(Y))
        X, Y = X[:n], Y[:n]
        if ip_lo is not None:
            keep = frame_mask(X, ip_lo)
            if not keep.any():
                win_empty += 1
                continue
            X, Y = X[keep], Y[keep]
            n = len(X)
        if residual:
            V = load_vac(key)
            if V is None:
                skipped += 1
                continue
            n = min(n, len(V))
            Y = Y[:n] - V[:n]
            X = X[:n]
        Xs.append(X)
        Ys.append(np.asarray(Y, np.float32).reshape(n, N_PIX))
        ids.append(np.full(n, i, np.int32))
    if not Xs:
        raise RuntimeError("no MAST cache rows loaded" + (" (vac sidecar empty)" if residual else ""))
    if skipped:
        print(f"[mast] residual skip {skipped} shots with no mast_vac_v1", flush=True)
    if win_empty:
        print(f"[mast] win skip {win_empty} shots with no frames above ip_lo", flush=True)
    n_raw = max(x.shape[1] for x in Xs)
    Xs = [_fit_width(x, n_raw) for x in Xs]
    return np.concatenate(Xs), np.concatenate(Ys), np.concatenate(ids), n_raw


def _with_vac(psi: np.ndarray, key: str | None = None, row: dict | None = None) -> np.ndarray:
    if row is not None:
        V = vac_from_row(row)
    elif key is not None:
        V = load_vac(key)
    else:
        return psi
    if V is None:
        return psi
    n = min(len(psi), len(V))
    out = np.asarray(psi, np.float32).copy()
    out[:n] = out[:n] + np.asarray(V[:n], np.float32)
    return out


def _pooled_r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, np.float64).reshape(-1)
    yhat = np.asarray(yhat, np.float64).reshape(-1)
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot else float("nan")


@torch.no_grad()
def _infer_z(model, xs, device, bs=4096):
    model.eval()
    out = []
    for s in range(0, len(xs), bs):
        z = model(xs[s:s + bs].to(device, non_blocking=True))[0]
        out.append(z.detach().float().cpu())
    return torch.cat(out).numpy().astype(np.float32)


def _prepare(X, ids, n_raw, mean, std, window, mode, clip):
    X = augment_temporal(_fit_width(X, n_raw), ids, window, mode)
    return _clip_features(((X - mean) / std).astype(np.float32), clip, label="mast")


def predict_keys(model, ck, keys, device):
    mean, std = ck["feature_mean"], ck["feature_std"]
    n_raw = int(ck["n_raw_features"])
    pca_mean, pca_comp = ck["pca_mean"], ck["pca_components"]
    cache_version = ck.get("cache_version", CACHE_VERSION)
    psi = {}
    for key in keys:
        got = load_shot(key, cache_version)
        if got is None:
            continue
        X, Y = got
        ids = np.zeros(len(X), np.int32)
        Xp = _prepare(X, ids, n_raw, mean, std, ck["temporal_window"], ck["temporal_mode"],
                      ck["feature_clip"])
        z = _infer_z(model, torch.from_numpy(np.ascontiguousarray(Xp)), device)
        n = min(len(z), len(Y))
        rec = reconstruct(z[:n], pca_mean, pca_comp)
        psi[key] = _with_vac(rec, key=key) if ck.get("residual_vac") else rec
    return psi


def predict_holdout(model, ck, device, mask_edge: bool = False):
    preds = []
    for key in HOLDOUT_KEYS:
        row = load_row(key, prefer_demo=True)
        shot = shot_from_mast_row(row)
        X, Y = build_one(shot)
        if mask_edge:
            X = np.array(X, copy=True)
            X[:, EDGE_START:N_FEATURES] = 0
        ids = np.zeros(len(X), np.int32)
        Xp = _prepare(
            X, ids, int(ck["n_raw_features"]), ck["feature_mean"], ck["feature_std"],
            ck["temporal_window"], ck["temporal_mode"], ck["feature_clip"],
        )
        z = _infer_z(model, torch.from_numpy(np.ascontiguousarray(Xp)), device)
        psi = reconstruct(z, ck["pca_mean"], ck["pca_components"])
        if ck.get("residual_vac"):
            psi = _with_vac(psi, row=row)
        n = min(len(psi), len(Y))
        preds.append({
            "psirz": psi[:n],
            "q95": np.full(n, np.nan, np.float64),
            "betaN": np.full(n, np.nan, np.float64),
        })
    return preds


def _ingest_aligned_r2(model, ck, device):
    """Raw pixel R² on ingest-cache holdout keys (not official; different T/scale than demo)."""
    per = {}
    ys, ps = [], []
    for key in HOLDOUT_KEYS:
        got = load_shot(key, ck.get("cache_version", CACHE_VERSION))
        if got is None:
            continue
        X, Y = got
        ids = np.zeros(len(X), np.int32)
        Xp = _prepare(
            X, ids, int(ck["n_raw_features"]), ck["feature_mean"], ck["feature_std"],
            ck["temporal_window"], ck["temporal_mode"], ck["feature_clip"],
        )
        z = _infer_z(model, torch.from_numpy(np.ascontiguousarray(Xp)), device)
        psi = reconstruct(z, ck["pca_mean"], ck["pca_components"])
        if ck.get("residual_vac"):
            psi = _with_vac(psi, key=key)
        n = min(len(psi), len(Y))
        per[key] = _pooled_r2(Y[:n], psi[:n])
        ys.append(np.asarray(Y[:n], np.float64).reshape(-1))
        ps.append(np.asarray(psi[:n], np.float64).reshape(-1))
    pooled = _pooled_r2(np.concatenate(ys), np.concatenate(ps)) if ys else float("nan")
    return per, pooled


def _train(train_keys, args, device):
    t0 = time.perf_counter()
    ip_lo = win_ip_lo() if args.win else None
    if args.win and ip_lo is None:
        raise RuntimeError("empty mast_win_v1; run mast_win_index")
    X, Yflat, ids, n_raw = _load_matrix(
        train_keys, residual=bool(args.residual_vac), cache_version=args.cache_version,
        ip_lo=ip_lo,
    )
    rng = np.random.default_rng(args.seed)
    sample_n = min(200_000, len(Yflat))
    if args.shot_balanced:
        picks = []
        for sid in np.unique(ids):
            ix = np.flatnonzero(ids == sid)
            n = max(1, sample_n // int(len(np.unique(ids))))
            picks.append(rng.choice(ix, size=min(n, len(ix)), replace=False))
        pick = np.concatenate(picks)
        if len(pick) > sample_n:
            pick = rng.choice(pick, size=sample_n, replace=False)
    else:
        pick = rng.choice(len(Yflat), size=sample_n, replace=False)
    pca = PCA(n_components=K, svd_solver="randomized", random_state=args.seed, iterated_power=3)
    pca.fit(Yflat[np.sort(pick)])
    Z = ((Yflat - pca.mean_) @ pca.components_.T).astype(np.float32)
    print(f"[mast] PCA k={K} explained={float(pca.explained_variance_ratio_.sum()):.6f} "
          f"frames={len(X):,} residual_vac={bool(args.residual_vac)} "
          f"shot_balanced={bool(args.shot_balanced)}", flush=True)

    window, mode, clip = 9, "smoothdiff", 10.0
    shots = np.unique(ids)
    rng.shuffle(shots)
    n_val = max(1, int(0.1 * len(shots)))
    val_m = np.isin(ids, shots[:n_val])
    X_aug = augment_temporal(_fit_width(X, n_raw), ids, window, mode)
    X64 = np.asarray(X_aug[~val_m], dtype=np.float64)
    mean = np.mean(X64, axis=0)
    std = np.std(X64, axis=0)
    std = np.where(~np.isfinite(std) | (std < 1e-6), 1.0, std)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    mean, std = mean.astype(np.float32), std.astype(np.float32)
    Xs = _clip_features(((X_aug - mean) / std).astype(np.float32), clip, label="train")
    xs = torch.from_numpy(np.ascontiguousarray(Xs))
    zt = torch.from_numpy(Z)

    torch.manual_seed(int(args.init_seed))
    model = build_trunk(
        "dualctx", Xs.shape[1], K, n_split=n_raw,
        trunk_width=512, trunk_blocks=4, ctx_width=512, ctx_blocks=3,
        trunk_dropout=0.05, activation="relu",
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    tr_i = np.flatnonzero(~val_m)
    va_i = np.flatnonzero(val_m)
    tr_by_shot = [tr_i[ids[tr_i] == s] for s in np.unique(ids[tr_i])]
    tr_by_shot = [a for a in tr_by_shot if len(a)]
    n_tr_shots = len(tr_by_shot)
    n_steps = max(1, int(np.ceil(len(tr_i) / args.batch)))
    best = {"val": 1e30, "ep": -1, "state": None}
    val_history: list[float] = []
    for ep in range(args.epochs):
        model.train()
        if args.shot_balanced:
            for _ in range(n_steps):
                sid = rng.integers(0, n_tr_shots, size=args.batch)
                idx = np.array([int(rng.choice(tr_by_shot[int(s)])) for s in sid], dtype=np.int64)
                pred = model(xs[torch.from_numpy(idx)].to(device, non_blocking=True))[0]
                loss = ((pred - zt[torch.from_numpy(idx)].to(device)) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        else:
            perm = rng.permutation(tr_i)
            for s in range(0, len(perm), args.batch):
                idx = torch.from_numpy(perm[s:s + args.batch].astype(np.int64))
                pred = model(xs[idx].to(device, non_blocking=True))[0]
                loss = ((pred - zt[idx].to(device)) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        pz = _infer_z(model, xs[va_i], device)
        val = float(((pz - Z[va_i]) ** 2).mean())
        flag = " *" if val < best["val"] else ""
        val_history.append(val)
        if val < best["val"]:
            best = {"val": val, "ep": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        print(f"[mast] ep {ep:3d}  val mse {val:.6e}{flag}", flush=True)
    model.load_state_dict(best["state"])
    ck = {
        "model_state": best["state"],
        "feature_mean": mean, "feature_std": std,
        "n_raw_features": int(n_raw), "n_features": int(Xs.shape[1]),
        "n_pca": K,
        "pca_mean": pca.mean_.astype(np.float32),
        "pca_components": pca.components_.astype(np.float32),
        "arch": "dualctx", "temporal_window": window, "temporal_mode": mode,
        "feature_clip": clip, "best_epoch": best["ep"], "val_mse": best["val"],
        "val_history": val_history,
        "train_shots": len(train_keys), "seed": args.seed,
        "init_seed": int(args.init_seed),
        "cache_version": args.cache_version,
        "vac_cache_version": "mast_vac_v1" if args.residual_vac else None,
        "residual_vac": bool(args.residual_vac),
        "qc": bool(args.qc),
        "shot_balanced": bool(args.shot_balanced),
        "mask_edge": bool(args.mask_edge),
        "win": bool(args.win),
        "win_ip_lo": (win_ip_lo() if args.win else None),
        "holdout_excluded": list(HOLDOUT_KEYS),
        "epochs": int(args.epochs),
        "train_seconds": round(time.perf_counter() - t0, 1),
    }
    return model, ck, best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-shots", type=int, default=0, help="0 = all non-holdout after sealed 80")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init-seed", type=int, default=None,
                    help="torch init only; default equals --seed. Split/PCA/batches stay on --seed")
    ap.add_argument("--tag", default="M1_archive")
    ap.add_argument("--eval-only", action="store_true",
                    help="score an existing CRC_FEQ_mast_<tag>.pt; do not train")
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting an existing CRC_FEQ_mast_<tag>.pt")
    ap.add_argument("--residual-vac", action="store_true",
                    help="train PCA/trunk on ψ−Ψ_vac; add Ψ_vac back at infer")
    ap.add_argument("--qc", action="store_true",
                    help="train only on mast_qc_v1 keep list (sidecar; features_v1 unchanged)")
    ap.add_argument("--shot-balanced", action="store_true",
                    help="sample shots uniformly in PCA and each train batch")
    ap.add_argument("--cache-version", default=CACHE_VERSION, choices=list(KNOWN_FEATURE_VERSIONS),
                    help="mast feature cache; v2 overlays aye edge Thomson")
    ap.add_argument("--mask-edge", action="store_true",
                    help="zero official-demo edge Thomson (cols 276:308) at infer")
    ap.add_argument("--win", action="store_true",
                    help="drop train frames below mast_win_v1 |Ip| fence")
    args = ap.parse_args()
    args.init_seed = resolve_init_seed(args.seed, args.init_seed)

    device = torch.device("cuda")
    torch.backends.cudnn.enabled = False
    ckpt = artifacts_dir() / f"CRC_FEQ_mast_{args.tag}.pt"
    if (not args.eval_only) and ckpt.is_file() and (not args.force):
        raise SystemExit(f"refusing to overwrite {ckpt}; pass --force or a new --tag")
    train_keys, sealed = _split(
        args.train_shots, args.seed, qc=bool(args.qc) and not args.eval_only,
        cache_version=args.cache_version,
    )
    if args.eval_only:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = build_trunk(
            "dualctx", int(ck["n_features"]), int(ck["n_pca"]),
            n_split=int(ck["n_raw_features"]),
            trunk_width=512, trunk_blocks=4, ctx_width=512, ctx_blocks=3,
            trunk_dropout=0.05, activation="relu",
        ).to(device)
        model.load_state_dict(ck["model_state"])
        train_n = int(ck.get("train_shots", len(train_keys)))
        t0 = time.perf_counter()
        best = {"ep": ck.get("best_epoch", -1), "val": ck.get("val_mse")}
        print(f"### MAST {args.tag} eval-only {ckpt}", flush=True)
    else:
        print(f"### MAST {args.tag} train={len(train_keys)} sealed={len(sealed)} "
              f"holdout={len(HOLDOUT_KEYS)} batch={args.batch} "
              f"seed={args.seed} init_seed={args.init_seed} "
              f"residual_vac={args.residual_vac} qc={args.qc} "
              f"shot_balanced={args.shot_balanced} cache={args.cache_version} "
              f"mask_edge={args.mask_edge} win={args.win}", flush=True)
        t0 = time.perf_counter()
        model, ck, best = _train(train_keys, args, device)
        torch.save(ck, ckpt)
        train_n = len(train_keys)

    sealed_psi = predict_keys(model, ck, sealed, device)
    feat_ver = ck.get("cache_version", args.cache_version)
    ys, ps = [], []
    for key, yh in sealed_psi.items():
        Y = load_shot(key, feat_ver)[1]
        nfr = min(len(Y), len(yh))
        ys.append(np.asarray(Y[:nfr], np.float64).reshape(-1))
        ps.append(np.asarray(yh[:nfr], np.float64).reshape(-1))
    sealed_r2 = _pooled_r2(np.concatenate(ys), np.concatenate(ps)) if ys else float("nan")
    qset = quarantine_set()
    clean_keys = [k for k in sealed_psi if k not in qset]
    clean_r2 = float("nan")
    if clean_keys:
        cys, cps = [], []
        for key in clean_keys:
            Y = load_shot(key, feat_ver)[1]
            yh = sealed_psi[key]
            nfr = min(len(Y), len(yh))
            cys.append(np.asarray(Y[:nfr], np.float64).reshape(-1))
            cps.append(np.asarray(yh[:nfr], np.float64).reshape(-1))
        clean_r2 = _pooled_r2(np.concatenate(cys), np.concatenate(cps))

    shots = load_holdout()
    mask_edge = bool(args.mask_edge or ck.get("mask_edge"))
    preds = predict_holdout(model, ck, device, mask_edge=mask_edge)
    scored = score_holdout(shots, preds, workers=1)
    ingest_per, ingest_pooled = _ingest_aligned_r2(model, ck, device)
    r2 = float(scored["r2_psi"])
    verdict = "PASS" if r2 >= GATE_R2 else "MISS"
    out = {
        "tag": args.tag,
        "verdict": verdict,
        "gate_r2_psi": GATE_R2,
        "holdout_r2_psi": r2,
        "holdout_S": float(scored["S"]),
        "holdout_consistency": float(scored["consistency"]),
        "holdout_dlcfs": float(scored["dlcfs"]),
        "holdout_r2_psi_unflipped": float(scored.get("r2_psi_unflipped", float("nan"))),
        "sealed_r2_psi": sealed_r2,
        "sealed_clean_r2_psi": clean_r2,
        "sealed_clean_shots": len(clean_keys),
        "qc": bool(ck.get("qc", args.qc)),
        "shot_balanced": bool(ck.get("shot_balanced", args.shot_balanced)),
        "qc_thresholds": (load_qc_index().get("thresholds") if args.qc or ck.get("qc") else None),
        "ingest_aligned_r2_psi": ingest_pooled,
        "ingest_aligned_per_shot": {k: float(v) for k, v in ingest_per.items()},
        "train_shots": train_n,
        "sealed_shots": len(sealed),
        "best_epoch": best["ep"],
        "epochs": int(ck.get("epochs", args.epochs)),
        "residual_vac": bool(ck.get("residual_vac", args.residual_vac)),
        "cache_version": feat_ver,
        "mask_edge": mask_edge,
        "win": bool(ck.get("win", args.win)),
        "win_ip_lo": ck.get("win_ip_lo"),
        "seed": int(ck.get("seed", args.seed)),
        "init_seed": int(ck.get("init_seed", ck.get("seed", args.seed))),
        "val_history": ck.get("val_history"),
        "eval_only": bool(args.eval_only),
        "checkpoint": str(ckpt),
        "note": "FAIR-MAST archive training; overlaps hidden test fold; not a zero-shot claim. "
                "Official holdout uses parquet_data demo (prefer_demo=True). "
                "Ingest holdout keys are a different recording than the demo parquets. DIII-D o1x5_42for7 unchanged. "
                "q95/βN submitted as NaN.",
        "seconds": round(time.perf_counter() - t0, 1),
    }
    dest = results_dir() / "mast_archive" / f"P36_{args.tag}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[mast] {verdict}  holdout R2ψ {r2:.6f}  (gate {GATE_R2})  "
          f"sealed {sealed_r2:.6f}  clean {clean_r2:.6f}  "
          f"S={scored['S']:.4f}  wrote {dest}", flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
