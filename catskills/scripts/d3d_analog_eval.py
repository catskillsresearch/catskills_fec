#!/usr/bin/env python3
"""One frozen-candidate evaluation on the canonical 80. No hyperparameter search.

  .venv/bin/python -m catskills.scripts.d3d_analog_eval
  .venv/bin/python -m catskills.scripts.d3d_analog_eval --winner path/to/winner.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from catskills.cpu_budget import limit_threads

limit_threads(2)

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fusion_scoring"))

from catskills.d3d_analog import (  # noqa: E402
    HOUSE_S, N_PCA, TAG, AnalogConfig, FoldPreprocessor, analog_root,
    apply_correction, apply_qb_correction, assert_no_leakage, blocked_distances,
    canonical_split, coeff_mse, desc_cache_dir, interpolate_phase_residual,
    knn_from_distances, load_shot_payload, neighbor_mask, neighbor_weights,
    pooled_r2, reconstruct_psi, shot_cache_dir, write_json_atomic,
)
from catskills.feature_cache import sample_cache_keys  # noqa: E402
from catskills.lab_paths import artifacts_dir, results_dir  # noqa: E402
from catskills.operator_targets import CONS_NAMES, official_grid  # noqa: E402
from catskills.scripts.cons_foldvar import consistency, pooled_r2 as fold_r2  # noqa: E402

from local_score import build_reference  # noqa: E402


def _field(cfg: AnalogConfig) -> str:
    if cfg.representation in ("A0", "A4"):
        return cfg.representation
    return f"{cfg.representation}_L{cfg.phase_length}"


def _load_desc_stack(keys: list[str], field: str) -> np.ndarray:
    from catskills.d3d_analog import pad_or_trim
    rows, width = [], None
    for k in keys:
        p = desc_cache_dir() / f"{k}.npz"
        with np.load(p) as d:
            name = field if field in d.files else (
                "A3_L64" if field.startswith("A5") else field)
            v = np.asarray(d[name], np.float64).ravel()
        width = width or v.size
        rows.append(pad_or_trim(v, width))
    return np.vstack(rows)


def _payloads(keys: list[str], kind: str) -> np.ndarray:
    rows = []
    for k in keys:
        pl = load_shot_payload(k)
        if kind == "constant":
            rows.append(pl.residual_const.ravel())
        elif kind == "p16":
            rows.append(pl.residual_p16.reshape(-1))
        else:
            rows.append(pl.residual_p32.reshape(-1))
    return np.vstack(rows)


def _embed(name: str) -> tuple[int | None, bool]:
    if name == "none":
        return None, False
    if name.startswith("whitened"):
        return int(name.replace("whitened", "")), True
    if name.startswith("pca"):
        return int(name.replace("pca", "")), False
    return None, False


def apply_locked(cfg: AnalogConfig, query: list[str], index: list[str],
                 hold: list[str]) -> tuple[list[np.ndarray], list[dict]]:
    field = _field(cfg)
    Q = _load_desc_stack(query, field)
    T = _load_desc_stack(index, field)
    dim, whiten = _embed(*[cfg.embed])
    prep = FoldPreprocessor(forbidden_keys=set(hold) | set(query))
    prep.fit(index, T, dim=dim, whiten=whiten)
    Qe, Te = prep.transform(Q), prep.transform(T)
    dist = blocked_distances(Qe, Te, cfg.metric)
    masks = []
    audits = []
    for i, qk in enumerate(query):
        m = neighbor_mask(index, qk, forbidden=hold, index_vecs=Te,
                          query_vec=Qe[i], dup_thresh=cfg.dup_thresh)
        masks.append(m)
        assert_no_leakage([index[j] for j in np.flatnonzero(m)[: cfg.k]], set(hold) | {qk})
    idx, dd = knn_from_distances(dist, cfg.k, mask=np.stack(masks))
    w = neighbor_weights(dd, cfg.weight, cfg.gauss_temp)
    pay = _payloads(index, cfg.payload)
    zc = []
    for i, qk in enumerate(query):
        pl = load_shot_payload(qk)
        hat = (w[i][:, None] * pay[idx[i]]).sum(0)
        if cfg.payload == "constant":
            r = hat[:60]
        elif cfg.payload == "p16":
            r = interpolate_phase_residual(hat.reshape(16, 60), pl.n_frames)
        else:
            r = interpolate_phase_residual(hat.reshape(32, 60), pl.n_frames)
        zc.append(apply_correction(pl.z_base, r, cfg.alpha, cfg.clip))
        audits.append({
            "key": qk,
            "neighbors": [index[int(j)] for j in idx[i]],
            "distances": [float(x) for x in dd[i]],
            "from_donor_only": all(index[int(j)] not in hold for j in idx[i]),
            "corr_norm": float(np.linalg.norm(r if r.ndim == 1 else r.mean(0))),
        })
    return zc, audits


def scalars_from_z(z_list, keys, mean, comp, stride: int = 1):
    R, Z, mask = official_grid()
    grid = (R, Z, mask, mask.astype(np.float64))
    gt_c, pr_c, shot = [], [], []
    for i, (z, key) in enumerate(zip(z_list, keys)):
        pl = load_shot_payload(key)
        zg = np.asarray(pl.z_gt, np.float64)
        zp = np.asarray(z, np.float64)
        T = min(len(zg), len(zp))
        sel = np.arange(0, T, max(1, stride))
        psi_g = reconstruct_psi(zg[sel], mean, comp)
        psi_p = reconstruct_psi(zp[sel], mean, comp)
        cg = build_reference(psi_g, *grid)[1]
        cp = build_reference(psi_p, *grid)[1]
        gt_c.append(cg)
        pr_c.append(cp)
        shot.append(np.full(len(sel), i))
    return np.vstack(gt_c), np.vstack(pr_c), np.concatenate(shot)


def paired_boot(gt, pa, pb, shot, boot: int, seed: int) -> dict:
    shots = np.unique(shot)
    idx_of = {s: np.flatnonzero(shot == s) for s in shots}
    keep = np.flatnonzero(np.isin(shot, shots))
    fa, ea = consistency(gt, pa, keep)
    fb, eb = consistency(gt, pb, keep)
    rng = np.random.default_rng(seed)
    da = np.empty(boot)
    for k in range(boot):
        fi = np.concatenate([idx_of[s] for s in rng.choice(shots, len(shots), replace=True)])
        ca, _ = consistency(gt, pa, fi)
        cb, _ = consistency(gt, pb, fi)
        da[k] = ca - cb
    per = {}
    for j, n in enumerate(CONS_NAMES):
        # reuse last each? recompute from full
        per[n] = {"full": float(ea[j] - eb[j])}
    return {
        "full_a": fa, "full_b": fb, "delta_full": fa - fb,
        "delta_mean": float(da.mean()), "paired_sd": float(da.std()),
        "p_better": float((da > 0).mean()),
        "sigma": float(da.mean() / max(da.std(), 1e-12)),
        "per_scalar": {n: {"full": float(ea[j] - eb[j])} for j, n in enumerate(CONS_NAMES)},
    }


def official_proxy(keys, z_corr, z_base, mean, comp, apply_qb: bool, cfg: AnalogConfig):
    """Local official-like S using reconstructed ψ and cached mix qβ (or analog qβ)."""
    from catskills.local_benchmark import official_score

    holdout, preds_b, preds_c = [], [], []
    for key, zc, zb in zip(keys, z_corr, z_base):
        packed = None
        from catskills.d3d_analog import load_thz_features
        packed = load_thz_features(key)
        X, Y, S = packed
        pl = load_shot_payload(key)
        T = min(len(Y), len(zc), len(zb))
        psi_c = reconstruct_psi(zc[:T], mean, comp)
        psi_b = reconstruct_psi(zb[:T], mean, comp)
        q = np.asarray(pl.q95_base, np.float64)[:T]
        b = np.asarray(pl.beta_base, np.float64)[:T]
        if apply_qb:
            q = apply_qb_correction(q, pl.q95_res, cfg.alpha)
            b = apply_qb_correction(b, pl.beta_res, cfg.alpha)
        shot = {
            "efit_times": np.arange(T, dtype=np.float64),
            "efit_psirz": np.asarray(Y[:T], np.float32),
            "scalars": {
                "efit_q95": np.asarray(S[:T, 2], np.float64) if S.shape[1] > 2 else np.full(T, np.nan),
                "efit_beta_n": np.asarray(S[:T, 0], np.float64) if S.shape[1] > 0 else np.full(T, np.nan),
            },
            "source": "DIII-D",
        }
        holdout.append({"shot": shot, "cache_key": key})
        preds_b.append({"psirz": psi_b.astype(np.float32), "q95": q.astype(np.float32),
                        "betaN": b.astype(np.float32)})
        preds_c.append({"psirz": psi_c.astype(np.float32), "q95": q.astype(np.float32),
                        "betaN": b.astype(np.float32)})
    sb = official_score(holdout, preds_b, workers=4)
    sc = official_score(holdout, preds_c, workers=4)
    return sb, sc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--winner", type=Path, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--skip-official", action="store_true")
    ap.add_argument("--apply-qb", action="store_true")
    args = ap.parse_args()

    win_path = args.winner or (analog_root() / "winner.json")
    win = json.loads(Path(win_path).read_text())
    cfg = AnalogConfig(**{k: v for k, v in win["cfg"].items() if k in AnalogConfig.__dataclass_fields__})
    keys = sample_cache_keys(0, 42, include_thomson=True)
    hold, donor = canonical_split(keys)
    hold = [k for k in hold if (shot_cache_dir() / f"{k}.npz").is_file()]
    donor = [k for k in donor if (shot_cache_dir() / f"{k}.npz").is_file()
             and (desc_cache_dir() / f"{k}.npz").is_file()]
    print(f"[{TAG}] eval locked {cfg.cell_id()} on {len(hold)} hold / {len(donor)} donor",
          flush=True)
    t0 = time.perf_counter()
    zc, audits = apply_locked(cfg, hold, donor, hold)
    if not all(a["from_donor_only"] for a in audits):
        raise RuntimeError("neighbor audit failed: holdout leaked into index")

    import torch
    ck = torch.load(artifacts_dir() / "CRC_FEQ_brown_TF_L_B8.pt",
                    map_location="cpu", weights_only=False)
    mean = np.asarray(ck["pca_mean"], np.float64)
    comp = np.asarray(ck["pca_components"], np.float64)

    zb = [np.asarray(load_shot_payload(k).z_base, np.float64) for k in hold]
    zg = [np.asarray(load_shot_payload(k).z_gt, np.float64) for k in hold]
    mse0 = float(np.mean([coeff_mse(a, b) for a, b in zip(zb, zg)]))
    mse1 = float(np.mean([coeff_mse(a, b) for a, b in zip(zc, zg)]))
    cut = 1.0 - mse1 / max(mse0, 1e-30)
    # alpha0 identity
    z_id = [apply_correction(a, np.zeros(60), 0.0) for a in zb]
    identity = all(np.allclose(a, b) for a, b in zip(z_id, zb))

    per_shot = []
    gains = []
    for key, a, b, g, au in zip(hold, zb, zc, zg, audits):
        e0 = coeff_mse(a, g)
        e1 = coeff_mse(b, g)
        gains.append(e0 - e1)
        per_shot.append({"key": key, "mse0": e0, "mse1": e1, "gain": e0 - e1,
                         **{k: au[k] for k in ("neighbors", "distances", "corr_norm")}})
    gains = np.asarray(gains, np.float64)
    total = float(gains[gains > 0].sum()) if (gains > 0).any() else 0.0
    max_frac = float(gains.max() / total) if total > 0 else 0.0

    print(f"[{TAG}] derive Consistency stride={args.stride}", flush=True)
    gt, pred_c, shot = scalars_from_z(zc, hold, mean, comp, args.stride)
    gt_b, pred_b, shot_b = scalars_from_z(zb, hold, mean, comp, args.stride)
    paired = paired_boot(gt, pred_c, pred_b, shot, args.boot, 42)

    worst40 = None
    cov_path = results_dir() / "brown" / "P33_COVERAGE.json"
    if cov_path.is_file():
        cov = json.loads(cov_path.read_text())
        order = cov["rank_worst_first"]
        picked = [k for k in order if k in hold][:40]
        sub = np.array([hold.index(k) for k in picked])
        keep = np.flatnonzero(np.isin(shot, sub))
        if keep.size:
            worst40 = paired_boot(gt[keep], pred_c[keep], pred_b[keep], shot[keep],
                                  args.boot, 42)

    official = None
    if not args.skip_official:
        print(f"[{TAG}] official_score proxy", flush=True)
        try:
            sb, sc = official_proxy(hold, zc, zb, mean, comp, args.apply_qb, cfg)
            official = {"base": sb, "corr": sc, "delta_S": sc["S"] - sb["S"]}
        except Exception as e:
            official = {"error": str(e)}

    scalar_floor = min((v["full"] for v in paired["per_scalar"].values()), default=0.0)
    gate = {
        "identity": identity,
        "mse_cut": cut,
        "delta_s": None if not official or "delta_S" not in official else official["delta_S"],
        "S": None if not official or "corr" not in official else official["corr"]["S"],
        "house_s": HOUSE_S,
        "cons_sigma": paired["sigma"],
        "cons_sigma_worst40": None if worst40 is None else worst40["sigma"],
        "r2_psi_delta": None if not official or "corr" not in official
        else official["corr"]["r2_psi"] - official["base"]["r2_psi"],
        "scalar_floor": scalar_floor,
        "max_shot_gain_frac": max_frac,
        "neighbor_audit_ok": all(a["from_donor_only"] for a in audits),
    }
    ready = False
    reasons = []
    if official and "delta_S" in official:
        if official["delta_S"] < 0.0010:
            reasons.append("delta_S")
        if official["corr"]["S"] < HOUSE_S:
            reasons.append("house_S")
        if official["corr"]["r2_psi"] < official["base"]["r2_psi"] - 2e-5:
            reasons.append("r2_psi")
    else:
        reasons.append("no_official")
    if paired["sigma"] < 2.0:
        reasons.append("cons_sigma")
    if worst40 is not None and worst40["sigma"] < 1.5:
        reasons.append("worst40")
    if scalar_floor < -1.0:
        reasons.append("scalar_floor")
    if max_frac > 0.20:
        reasons.append("shot_concentration")
    ready = len(reasons) == 0

    out = {
        "cfg": cfg.as_dict(),
        "n_hold": len(hold),
        "mse0": mse0, "mse1": mse1, "mse_cut": cut,
        "paired": paired,
        "worst40": worst40,
        "official": official,
        "per_shot": per_shot,
        "audits": audits[:8],
        "gate": gate,
        "candidate_ready": ready,
        "fail_reasons": reasons,
        "seconds": round(time.perf_counter() - t0, 1),
        "frozen_from": win.get("frozen_from", "winner.json"),
    }
    write_json_atomic(analog_root() / "holdout_final.json", out)
    print(f"[{TAG}] holdout cut={cut:+.4f} Consσ={paired['sigma']:+.2f} "
          f"ready={ready} {reasons}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
