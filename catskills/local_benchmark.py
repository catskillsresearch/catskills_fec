#!/usr/bin/env python3
"""Robust Challenge-1 local benchmark on labelled ``diii_d_train``.

Unlike the tiny smoke table (5 train / 3 contiguous eval shots), this:

1. Streams a shuffled pool from Hugging Face ``diii_d_train`` (full labelled set).
2. Splits **by shot** into train / holdout (random, seeded).
3. Trains baselines on the train fold only.
4. Scores the holdout with the **official** composite metric (same functionals as
   Codabench via ``local_score`` / ``fusion_scoring``).

This is still a proxy (holdout ⊂ train corpus, not the public/private test folds),
but it is the right tool for method comparison before a leaderboard submit.

Memory ~20 MB/shot — keep ``--pool`` in a comfortable range (e.g. 80–200).
Scoring cost is dominated by LCFS extraction on holdout frames.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fusion_scoring"))

from tqdm.auto import tqdm  # noqa: E402

from catskills.hf_auth import ensure_hf_token  # noqa: E402
from catskills.lab_paths import results_dir  # noqa: E402
from catskills.score_ref_cache import (  # noqa: E402
    cache_root as score_ref_cache_root,
    entry_cache_key,
    get_or_build_reference,
)
from catskills.features import load_shot_from_hf_row  # noqa: E402
from local_score import (  # noqa: E402
    MACHINE,
    build_reference,
    default_score_workers,
    psi_residuals,
    score_shots,
)
from common import CONS_SCALARS, N_CONS, N_SCALARS, PSI_SIGNS, SCALARS, SCORING_VERSION  # noqa: E402
from metrics import Accum, finalize_machine  # noqa: E402


def load_pool(n_pool: int, seed: int) -> list[dict]:
    """Shuffle-stream ``n_pool`` labelled DIII-D train shots (keep raw row + shot)."""
    from catskills.dataset_local import load_challenge_dataset

    print(f"Loading shuffled pool of {n_pool} shots from diii_d_train (seed={seed})...")
    buf = min(max(n_pool * 2, 16), 64)
    print(f"  shuffle buffer_size={buf}")
    ds = load_challenge_dataset("diii_d_train", "train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=buf)
    pool = []
    for i, row in enumerate(ds):
        if i >= n_pool:
            break
        shot = load_shot_from_hf_row(row)
        shot["shot_index"] = i
        pool.append({"row": row, "shot": shot})
        print(f"  pool[{i}] T={len(shot['efit_times'])}")
    if len(pool) < n_pool:
        raise SystemExit(f"Only loaded {len(pool)}/{n_pool} shots from the Hub")
    return pool


def split_pool(pool: list[dict], n_eval: int, seed: int) -> tuple[list[dict], list[dict]]:
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(pool))
    eval_idx = set(idx[:n_eval].tolist())
    train = [pool[i] for i in range(len(pool)) if i not in eval_idx]
    holdout = [pool[i] for i in range(len(pool)) if i in eval_idx]
    print(f"Split: train={len(train)} shots, holdout={len(holdout)} shots")
    return train, holdout


def gt_for_score(entry: dict) -> dict:
    shot = entry["shot"]
    return {
        "psi": np.asarray(shot["efit_psirz"], dtype=np.float32),
        "q95": np.asarray(shot["scalars"].get("efit_q95", np.full(len(shot["efit_times"]), np.nan)),
                          dtype=np.float64),
        "betaN": np.asarray(shot["scalars"].get("efit_beta_n", np.full(len(shot["efit_times"]), np.nan)),
                            dtype=np.float64),
    }


def official_score(
    holdout: list[dict],
    preds: list[dict],
    *,
    workers: int | None = None,
) -> dict:
    """In-memory official composite on holdout (no re-stream / no skip).

    Per-shot LCFS + derive_frame is CPU-bound and independent, so the scoring loop
    uses a process pool (default: all cores, override with ``workers`` or
    ``FEQ_SCORE_WORKERS``). Ground-truth reference build stays sequential — after
    the first run it is a disk-cache hit.
    """
    mask = np.load(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    R, Z = mask["grid_R"], mask["grid_Z"]
    mask_coarse = mask["mask_coarse"].astype(bool)
    mask_f = mask_coarse.astype(np.float64)

    gts = [gt_for_score(e) for e in holdout]
    refs, psi_sum, psi_sumsq, psi_n = [], 0.0, 0.0, 0.0
    scal_sum = np.zeros(N_SCALARS)
    scal_sumsq = np.zeros(N_SCALARS)
    scal_n = np.zeros(N_SCALARS)
    cons_sum = np.zeros(N_CONS)
    cons_sumsq = np.zeros(N_CONS)
    cons_n = np.zeros(N_CONS)

    print(
        "Building holdout reference targets (LCFS + Consistency; "
        f"disk cache {score_ref_cache_root()})...",
        flush=True,
    )
    hits = misses = 0
    for entry, gt in tqdm(
        list(zip(holdout, gts)),
        desc="holdout GT refs",
        unit="shot",
        leave=True,
    ):
        key = entry_cache_key(entry)
        ref, hit = get_or_build_reference(
            gt["psi"],
            R,
            Z,
            mask_coarse,
            mask_f,
            cache_key=key,
            build_fn=build_reference,
        )
        if hit:
            hits += 1
        else:
            misses += 1
        refs.append(ref)
        g = gt["psi"].astype(np.float64)
        fin = np.isfinite(g)
        psi_sum += float(g[fin].sum())
        psi_sumsq += float((g[fin] ** 2).sum())
        psi_n += int(fin.sum())
        for j, name in enumerate(SCALARS):
            v = np.asarray(gt[name], dtype=np.float64)
            v = v[np.isfinite(v)]
            scal_sum[j] += v.sum()
            scal_sumsq[j] += (v ** 2).sum()
            scal_n[j] += v.size
        cons_gt, cmask = ref[1], ref[2]
        for j in range(N_CONS):
            v = cons_gt[cmask[:, j], j]
            cons_sum[j] += v.sum()
            cons_sumsq[j] += (v ** 2).sum()
            cons_n[j] += v.size
    print(f"  GT ref cache: hits={hits}  misses={misses}", flush=True)

    ref_stats = {
        "psi_sum": psi_sum, "psi_sumsq": psi_sumsq, "psi_n": psi_n,
        "scal_sum": scal_sum, "scal_sumsq": scal_sumsq, "scal_n": scal_n,
        "cons_sum": cons_sum, "cons_sumsq": cons_sumsq, "cons_n": cons_n,
    }
    mean_psi = psi_sum / psi_n if psi_n else 0.0
    means = (
        mean_psi,
        np.where(scal_n > 0, scal_sum / np.where(scal_n > 0, scal_n, 1), 0.0),
        np.where(cons_n > 0, cons_sum / np.where(cons_n > 0, cons_n, 1), 0.0),
    )

    totals = {s: 0.0 for s in PSI_SIGNS}
    for gt, p in zip(gts, preds):
        rr, _, _ = psi_residuals(gt["psi"], p["psirz"], mean_psi)
        for sg in PSI_SIGNS:
            totals[sg] += rr[sg]
    psi_sign = min(PSI_SIGNS, key=lambda sg: totals[sg])

    if workers is None:
        workers = default_score_workers()
    workers = max(1, min(int(workers), max(1, len(gts))))
    print(f"Scoring holdout with official metric ({workers} workers)...", flush=True)
    acc = Accum()
    acc.psi_sign = psi_sign
    t0 = time.perf_counter()
    for part in score_shots(
        gts, refs, preds, R, Z, mask_coarse, mask_f, psi_sign, means,
        workers=workers, desc="score holdout",
    ):
        acc.add(part)
    score_s = time.perf_counter() - t0
    res = finalize_machine(acc, ref_stats)
    one_minus = 1.0 - min(1.0, res["dlcfs"])
    return {
        "S": float(res["S"]),
        "r2_psi": float(res["r2_psi"]),
        "r2_qb": float(res["r2_qb"]),
        "one_minus_dlcfs": float(one_minus),
        "consistency": float(res["consistency"]),
        "r2_cons_each": dict(res.get("r2_cons_each") or {}),
        "cons_fail_each": dict(res.get("cons_fail_each") or {}),
        "cons_fail_frac": float(res["cons_fail_frac"]),
        "psi_sign": int(res["psi_sign"]),
        "lcfs_fail_frac": float(res["lcfs_fail_frac"]),
        "score_s": score_s,
        "score_workers": workers,
        "metric": SCORING_VERSION,
        "machine": MACHINE,
    }


def predict_zeros(holdout: list[dict]) -> list[dict]:
    out = []
    for e in holdout:
        T = len(e["shot"]["efit_times"])
        out.append({
            "psirz": np.zeros((T, 65, 65), dtype=np.float32),
            "q95": np.zeros(T, dtype=np.float32),
            "betaN": np.zeros(T, dtype=np.float32),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=int, default=80,
                    help="shuffled shots drawn from diii_d_train (train+holdout)")
    ap.add_argument("--eval-shots", type=int, default=15,
                    help="holdout size (rest of pool = train)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-thomson", action="store_true")
    ap.add_argument(
        "--score-workers",
        type=int,
        default=None,
        help="process-pool size for official scoring (default: FEQ_SCORE_WORKERS or CPU count)",
    )
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["zeros", "ridge_pca", "mlp_pca"],
        choices=["zeros", "ridge_pca", "mlp_pca", "pinn_gs"],
    )
    args = ap.parse_args()
    ensure_hf_token()
    OUT = results_dir() / "challenge1_local_benchmark"
    OUT.mkdir(parents=True, exist_ok=True)

    if args.eval_shots >= args.pool:
        raise SystemExit("--eval-shots must be < --pool")

    t_load = time.perf_counter()
    pool = load_pool(args.pool, args.seed)
    train_entries, holdout_entries = split_pool(pool, args.eval_shots, args.seed + 1)
    load_s = time.perf_counter() - t_load

    train_shots = [e["shot"] for e in train_entries]
    rows = []
    for method in args.methods:
        print(f"\n======== {method} ========")
        t0 = time.perf_counter()
        if method == "zeros":
            preds = predict_zeros(holdout_entries)
            train_s = 0.0
        elif method in ("ridge_pca", "mlp_pca"):
            from catskills.sklearn_baseline import train_sklearn_baseline, predict_row
            bundle = train_sklearn_baseline(
                method=method,
                include_thomson=args.include_thomson,
                shots=train_shots,
                output=OUT / f"{method}_benchmark.pkl",
            )
            train_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            preds = [predict_row(bundle, e["row"], "DIII-D") for e in holdout_entries]
            infer_s = time.perf_counter() - t1
        else:  # pinn_gs — Ridge+GS on the same train fold
            from catskills import pinn_baseline as pb
            from catskills.pinn_baseline import PINNModelBundle, predict_row
            from experiments import EFIT_SCALAR_LABELS, build_feature_matrix
            from sklearn.decomposition import PCA
            from sklearn.linear_model import RidgeCV
            from sklearn.preprocessing import StandardScaler

            X, Y, S, _ = build_feature_matrix(train_shots, include_thomson=args.include_thomson)
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            pca = PCA(n_components=min(40, Xs.shape[0] - 1), random_state=42)
            Zc = pca.fit_transform(Y.reshape(len(Y), -1))
            flux = RidgeCV(alphas=np.logspace(-3, 3, 13))
            flux.fit(Xs, Zc)
            Rgrid = np.linspace(0.8, 2.5, 65)
            Zgrid = np.linspace(-1.5, 1.5, 65)
            recon = pca.inverse_transform(flux.predict(Xs[: min(64, len(Xs))])).reshape(-1, 65, 65)
            cps, cfs = [], []
            for i in range(len(recon)):
                cp, cf = pb._fit_profiles_for_shot(recon[i], Rgrid, Zgrid, axis_sign=-1.0)
                cps.append(cp)
                cfs.append(cf)
            i_q = EFIT_SCALAR_LABELS.index("efit_q95")
            i_b = EFIT_SCALAR_LABELS.index("efit_beta_n")
            q95, betaN = S[:, i_q], S[:, i_b]
            q_model = RidgeCV(alphas=np.logspace(-3, 3, 13))
            b_model = RidgeCV(alphas=np.logspace(-3, 3, 13))
            m_q, m_b = np.isfinite(q95), np.isfinite(betaN)
            q_model.fit(Xs[m_q] if m_q.sum() > 10 else Xs,
                        q95[m_q] if m_q.sum() > 10 else np.zeros(len(Xs)))
            b_model.fit(Xs[m_b] if m_b.sum() > 10 else Xs,
                        betaN[m_b] if m_b.sum() > 10 else np.zeros(len(Xs)))
            bundle = PINNModelBundle(
                feature_scaler=scaler, pca=pca, flux_model=flux,
                q95_model=q_model, betaN_model=b_model, n_features=Xs.shape[1],
                gs_lambda=1e-3, coeffs_p=np.mean(np.stack(cps), 0),
                coeffs_ff=np.mean(np.stack(cfs), 0), include_thomson=args.include_thomson,
            )
            train_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            preds = [predict_row(bundle, e["row"], "DIII-D") for e in holdout_entries]
            infer_s = time.perf_counter() - t1

        if method == "zeros":
            infer_s = 0.0

        scores = official_score(holdout_entries, preds, workers=args.score_workers)
        row = {
            "method": method,
            **scores,
            "train_s": train_s,
            "infer_s": infer_s if method != "zeros" else 0.0,
            "load_s": load_s,
            "pool": args.pool,
            "n_train": len(train_shots),
            "n_eval": len(holdout_entries),
            "seed": args.seed,
            "include_thomson": args.include_thomson,
        }
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("method", "S", "r2_psi", "r2_qb",
                                              "one_minus_dlcfs", "consistency",
                                              "train_s", "infer_s", "score_s")}, indent=2))

    out = OUT / f"benchmark_pool{args.pool}_eval{args.eval_shots}_seed{args.seed}.json"
    payload = {
        "protocol": "random_shot_holdout_official_S",
        "note": "Holdout is labelled diii_d_train, not Codabench public/private test.",
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
