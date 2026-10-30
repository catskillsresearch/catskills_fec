"""Shared helpers for the P38 three-arm follow-up chain.

Receipts live under ``analog_root()/followups/``. Never writes
``decision.json``, ``winner.json``, or ``holdout_final.json``.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from catskills.d3d_analog import (
    AnalogConfig, FoldPreprocessor, analog_root, apply_correction,
    apply_qb_correction, blocked_distances, canonical_split, coeff_mse,
    desc_cache_dir, donor_folds, hash_keys, interpolate_phase_residual,
    knn_from_distances, load_shot_payload, neighbor_weights, phase_bin_residual,
    pooled_r2, qb_phase_residual, qb_residual, random_neighbors, save_npz_atomic,
    sha256_bytes, sha256_file, shot_cache_dir, shot_constant_residual,
    write_json_atomic,
)
from catskills.feature_cache import load_cache_index, sample_cache_keys
from catskills.model_version import git_full_sha

ARM_VERSION = "p38_followups_v1"
OOF_REQUIRED = (
    "z_base", "z_gt", "residual_const", "residual_p16", "residual_p32",
    "q95_base", "q95_gt", "q95_res", "q95_p16",
    "beta_base", "beta_gt", "beta_res", "beta_p16", "n_frames",
    "model_sha256",
)
TERMINAL_OK = ("PASS", "MISS", "BLOCKED")
LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0)
SVD_RANKS = (4, 8, 16)
RAW_ARM_VERSION = "p38_raw_p16_v1"
INNER_SPLIT_SEED = 1
N_INNER = 5
# Comparison-only: psi_shrink λ=0 nested-OOF median (cross-model residuals).
CROSS_MODEL_P16_MEDIAN = 0.047310321427806


class ContractError(RuntimeError):
    """Leakage / allowlist / hash failure. Do not retry."""


def followups_root() -> Path:
    d = analog_root() / "followups"
    d.mkdir(parents=True, exist_ok=True)
    return d


def oof_dir() -> Path:
    d = followups_root() / "oof"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_train_keys_json(path: Path, *, seed: int = 42,
                         include_thomson: bool = True) -> list[str]:
    """Exact ordered allowlist. Raises ContractError on any violation."""
    obj = json.loads(Path(path).read_text())
    keys = [str(k) for k in obj.get("keys") or []]
    if not keys:
        raise ContractError(f"{path}: empty keys")
    if len(keys) != len(set(keys)):
        raise ContractError(f"{path}: duplicate train keys")
    expected = obj.get("keys_sha256")
    got = hash_keys(keys)
    if expected and expected != got:
        raise ContractError(f"{path}: keys_sha256 mismatch")
    # Canonical-80 leak check is always the P38 split (seed 42), never the trainer seed.
    hold, _ = canonical_split(sample_cache_keys(0, 42, include_thomson=True))
    leaked = [k for k in keys if k in set(hold)]
    if leaked:
        raise ContractError(f"{path}: canonical-80 leak {leaked[:4]}")
    available = {e["key"] for e in load_cache_index(include_thomson)}
    missing = [k for k in keys if k not in available]
    if missing:
        raise ContractError(f"{path}: missing cache keys {missing[:4]}")
    return keys


def write_train_keys_json(path: Path, keys: Sequence[str]) -> Path:
    keys = [str(k) for k in keys]
    return write_json_atomic(path, {"keys": keys, "keys_sha256": hash_keys(keys),
                                    "n": len(keys)})


def input_fingerprint(**extra) -> dict:
    keys = sample_cache_keys(0, 42, include_thomson=True)
    hold, donor = canonical_split(keys)
    win_p = analog_root() / "winner.json"
    win = json.loads(win_p.read_text()) if win_p.is_file() else {}
    rec = {
        "arm_version": ARM_VERSION,
        "git": git_full_sha(),
        "hold_sha256": hash_keys(hold),
        "donor_sha256": hash_keys(donor),
        "n_hold": len(hold),
        "n_donor": len(donor),
        "winner_cfg": win.get("cfg"),
    }
    rec.update(extra)
    if rec.get("stage") in ("qb_oof", "psi_shrink", "svd_phase", "raw_p16_ensemble"):
        rec.update(oof_stamp())
    return rec


def raw_p16_root() -> Path:
    d = followups_root() / "raw_p16_ensemble"
    d.mkdir(parents=True, exist_ok=True)
    return d


def shadow_ckpt(fold: int) -> Path:
    from catskills.lab_paths import artifacts_dir
    return artifacts_dir() / f"CRC_FEQ_P38FU_shadow_f{int(fold)}.pt"


def payload_model_sha(path: Path) -> str:
    with np.load(path, allow_pickle=True) as d:
        if "model_sha256" not in d.files:
            raise ContractError(f"{path}: missing model_sha256")
        return str(d["model_sha256"])


def pca_sha_from_ckpt(path: Path) -> str:
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    mu = np.ascontiguousarray(ck["pca_mean"], np.float64)
    comp = np.ascontiguousarray(ck["pca_components"], np.float64)
    return sha256_bytes(mu.tobytes() + b"\0" + comp.tobytes())


def inner_partitions(keys: Sequence[str], n_inner: int = N_INNER,
                     seed: int = INNER_SPLIT_SEED) -> list[np.ndarray]:
    return donor_folds(list(keys), n_inner, seed)


def average_z(members: Sequence[Sequence[np.ndarray]]) -> list[np.ndarray]:
    """Mean coefficient series across members, truncated to the shortest T."""
    if not members:
        return []
    n = len(members[0])
    out = []
    for i in range(n):
        rows = [np.asarray(m[i], np.float64) for m in members]
        T = min(r.shape[0] for r in rows)
        C = min(r.shape[1] if r.ndim == 2 else 0 for r in rows)
        out.append(np.mean([r[:T, :C] if r.ndim == 2 else r[:T] for r in rows], 0))
    return out


def assert_frozen_winner(cfg: AnalogConfig) -> None:
    ok = (cfg.representation == "A0" and int(cfg.phase_length) == 32
          and cfg.embed == "none" and cfg.metric == "cosine" and int(cfg.k) == 16
          and cfg.weight == "inverse" and cfg.payload == "p16"
          and abs(float(cfg.alpha) - 0.5) <= 1e-12
          and abs(float(cfg.clip) - 0.5) <= 1e-12)
    if not ok:
        raise ContractError(
            f"winner is not frozen A0/L32/none/cosine/k16/inverse/p16/a0.5/clip0.5: {cfg.as_dict()}")


def assert_payloads_match_shadow(keys: Sequence[str], model_sha: str,
                                 root: Path | None = None) -> None:
    root = Path(root) if root is not None else oof_dir()
    for k in keys:
        dest = root / f"{k}.npz"
        if not oof_schema_ok(dest, model_sha):
            got = payload_model_sha(dest) if dest.is_file() else None
            raise ContractError(
                f"payload {k} model_sha mismatch got={got} want={model_sha}")


def assert_shared_pca(paths: Sequence[Path]) -> str:
    shas = [pca_sha_from_ckpt(Path(p)) for p in paths]
    if not shas or len(set(shas)) != 1:
        raise ContractError(f"shadow PCA mismatch {shas}")
    return shas[0]


def mse_cut_halves(zb, zg, zc) -> tuple[float, float]:
    def _cut(a, b, c):
        base = tot = 0.0
        for z0, z1, z2 in zip(a, b, c):
            T = min(len(z0), len(z1), len(z2))
            if T <= 0:
                continue
            base += coeff_mse(z0[:T], z1[:T]) * T
            tot += coeff_mse(z2[:T], z1[:T]) * T
        if base <= 0:
            return float("nan")
        return 1.0 - tot / base

    zb1, zg1, zc1, zb2, zg2, zc2 = [], [], [], [], [], []
    for z0, z1, z2 in zip(zb, zg, zc):
        mid = min(len(z0), len(z1), len(z2)) // 2
        zb1.append(z0[:mid]); zg1.append(z1[:mid]); zc1.append(z2[:mid])
        zb2.append(z0[mid:]); zg2.append(z1[mid:]); zc2.append(z2[mid:])
    return _cut(zb1, zg1, zc1), _cut(zb2, zg2, zc2)


def oof_stamp() -> dict:
    man = oof_dir() / "manifest.json"
    if not man.is_file():
        return {"oof_manifest": None, "oof_n_ready": None, "oof_unix": None}
    rec = json.loads(man.read_text())
    return {
        "oof_manifest": str(man),
        "oof_n_ready": rec.get("n_ready"),
        "oof_unix": rec.get("unix"),
        "oof_status": rec.get("status"),
    }


def locked_cfg() -> AnalogConfig:
    win = json.loads((analog_root() / "winner.json").read_text())
    return AnalogConfig(**{k: v for k, v in win["cfg"].items()
                           if k in AnalogConfig.__dataclass_fields__})


def receipt_skippable(path: Path, fingerprint: dict) -> bool:
    if not path.is_file():
        return False
    try:
        rec = json.loads(path.read_text())
    except Exception:
        return False
    if rec.get("status") not in TERMINAL_OK:
        return False
    fp = rec.get("fingerprint") or {}
    keys = ("arm_version", "hold_sha256", "donor_sha256", "winner_cfg",
            "oof_n_ready", "oof_unix", "pca_sha256", "model_shas",
            "raw_arm_version")
    return all(fp.get(k) == fingerprint.get(k) for k in keys if k in fingerprint)


def write_receipt(path: Path, status: str, fingerprint: dict, **extra) -> Path:
    rec = {
        "status": status,
        "fingerprint": fingerprint,
        "unix": time.time(),
        "arm_version": ARM_VERSION,
    }
    rec.update(extra)
    return write_json_atomic(path, rec)


def heartbeat(path: Path, stage: str, **extra) -> None:
    rec = {"stage": stage, "unix": time.time(), "arm_version": ARM_VERSION}
    rec.update(extra)
    write_json_atomic(path, rec)


def oof_schema_ok(path: Path, model_sha: str | None = None) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=True) as d:
            if any(n not in d.files for n in OOF_REQUIRED):
                return False
            if model_sha is not None:
                got = str(d["model_sha256"])
                if got != model_sha:
                    return False
    except Exception:
        return False
    return True


def save_oof_payload(dest: Path, *, key: str, z_base, z_gt, q_base, b_base,
                     q_gt, b_gt, model_sha: str) -> Path:
    z_base = np.asarray(z_base, np.float64)
    z_gt = np.asarray(z_gt, np.float64)
    T = min(len(z_base), len(z_gt))
    z_base, z_gt = z_base[:T], z_gt[:T]
    q_base = np.asarray(q_base, np.float64)[:T]
    b_base = np.asarray(b_base, np.float64)[:T]
    q_gt = np.asarray(q_gt, np.float64)[:T]
    b_gt = np.asarray(b_gt, np.float64)[:T]
    q_res, _ = qb_residual(q_base, q_gt)
    b_res, _ = qb_residual(b_base, b_gt)
    return save_npz_atomic(
        dest,
        key=np.asarray(key),
        n_frames=np.int32(T),
        z_base=z_base.astype(np.float32),
        z_gt=z_gt.astype(np.float32),
        residual_const=shot_constant_residual(z_base, z_gt).astype(np.float32),
        residual_p16=phase_bin_residual(z_base, z_gt, 16).astype(np.float32),
        residual_p32=phase_bin_residual(z_base, z_gt, 32).astype(np.float32),
        q95_base=q_base.astype(np.float32),
        beta_base=b_base.astype(np.float32),
        q95_gt=q_gt.astype(np.float32),
        beta_gt=b_gt.astype(np.float32),
        q95_res=np.float32(q_res),
        beta_res=np.float32(b_res),
        q95_p16=qb_phase_residual(q_base, q_gt, 16).astype(np.float32),
        beta_p16=qb_phase_residual(b_base, b_gt, 16).astype(np.float32),
        model_sha256=np.asarray(model_sha),
    )


def load_desc_stack(keys: Sequence[str], field: str) -> np.ndarray:
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
    return np.vstack(rows) if rows else np.zeros((0, 1), np.float64)


def field_for(cfg: AnalogConfig) -> str:
    if cfg.representation in ("A0", "A4"):
        return cfg.representation
    if cfg.representation == "A5":
        return f"A3_L{cfg.phase_length}"
    return f"{cfg.representation}_L{cfg.phase_length}"


def embed_spec(name: str) -> tuple[int | None, bool]:
    if name == "none":
        return None, False
    if name.startswith("whitened"):
        return int(name.replace("whitened", "")), True
    if name.startswith("pca"):
        return int(name.replace("pca", "")), False
    return None, False


def legal_mask(q_keys, tr_keys, hold, n_q, n_tr) -> np.ndarray:
    from catskills.scripts.d3d_analog_search import _legal_mask
    return _legal_mask(q_keys, tr_keys, hold, n_q, n_tr)


def assert_disjoint(query: Sequence[str], train: Sequence[str], label: str = "") -> None:
    overlap = set(query) & set(train)
    if overlap:
        raise ContractError(f"{label} train/query overlap {sorted(overlap)[:4]}")


def evaluate_cell_eff(cfg: AnalogConfig, q_keys, tr_keys, hold,
                      desc_q, desc_tr, pay_tr, zb_q, zg_q,
                      alphas=None, control: str = "") -> dict:
    """Like search.evaluate_cell, but ``alphas`` may be scalar or per-query."""
    from catskills.d3d_analog import permute_payloads
    assert_disjoint(q_keys, tr_keys, "evaluate_cell_eff")
    idx, dd, w = fold_neighbors(cfg, q_keys, tr_keys, hold, desc_q, desc_tr)
    pay = np.asarray(pay_tr, np.float64)
    if control == "permute":
        pay = permute_payloads(pay, seed=1)
    if control == "random":
        idx = random_neighbors(len(tr_keys), len(q_keys), cfg.k, seed=0)
    hats = []
    if control == "global_mean":
        mu = pay.mean(0)
        hats = [payload_for_query(cfg.payload, mu, len(z0)) for z0 in zb_q]
    else:
        for i, z0 in enumerate(zb_q):
            hat = (w[i][:, None] * pay[idx[i]]).sum(0)
            hats.append(payload_for_query(cfg.payload, hat, len(z0)))
    a = 0.0 if control == "alpha0" else (cfg.alpha if alphas is None else alphas)
    cut, gains = mse_cut_from_corr(zb_q, zg_q, hats, a)
    return {
        "cfg": cfg,
        "cell": cfg.cell_id() + (f"__{control}" if control else ""),
        "mse_cut": float(cut),
        "gains": gains,
        "n_query": len(q_keys),
        "n_index": len(tr_keys),
        "control": control,
    }


def fold_neighbors(cfg: AnalogConfig, q_keys, tr_keys, hold,
                   desc_q, desc_tr) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dim, whiten = embed_spec(cfg.embed)
    prep = FoldPreprocessor(forbidden_keys=set(hold) | set(q_keys))
    prep.fit(tr_keys, desc_tr, dim=dim, whiten=whiten)
    Qe, Te = prep.transform(desc_q), prep.transform(desc_tr)
    dist = blocked_distances(Qe, Te, cfg.metric)
    mask = legal_mask(q_keys, tr_keys, hold, dist.shape[0], dist.shape[1])
    idx, dd = knn_from_distances(dist, cfg.k, mask=mask)
    w = neighbor_weights(dd, cfg.weight, cfg.gauss_temp)
    return idx, dd, w


def qb_hat(w, idx, res) -> np.ndarray:
    neigh = np.asarray(res, np.float64)[idx]
    num = np.nansum(w * neigh, axis=1)
    den = np.nansum(w * np.isfinite(neigh), axis=1)
    return num / np.maximum(den, 1e-9)


def qb_fold_r2(q_pls, hat_q, hat_b, alpha, second_half, control="") -> dict:
    n = len(q_pls)
    hat_q = np.asarray(hat_q, np.float64).copy()
    hat_b = np.asarray(hat_b, np.float64).copy()
    if control == "permute":
        rng = np.random.default_rng(1)
        hat_q = hat_q[rng.permutation(n)]
        hat_b = hat_b[rng.permutation(n)]
    elif control == "global_mean":
        hat_q = np.full(n, np.nanmean(hat_q))
        hat_b = np.full(n, np.nanmean(hat_b))
    gq, pq, gb, pb = [], [], [], []
    gq1, pq1, gq2, pq2 = [], [], [], []
    for i, pl in enumerate(q_pls):
        if pl.q95_base is None or pl.q95_gt is None:
            continue
        qc = apply_qb_correction(pl.q95_base, hat_q[i], alpha,
                                 second_half_only=second_half)
        bc = apply_qb_correction(pl.beta_base, hat_b[i], alpha,
                                 second_half_only=second_half)
        gq.append(np.ravel(pl.q95_gt)); pq.append(np.ravel(qc))
        gb.append(np.ravel(pl.beta_gt)); pb.append(np.ravel(bc))
        T = len(pl.q95_gt)
        mid = T // 2
        gq1.append(np.ravel(pl.q95_gt)[:mid]); pq1.append(np.ravel(qc)[:mid])
        gq2.append(np.ravel(pl.q95_gt)[mid:]); pq2.append(np.ravel(qc)[mid:])
    if not gq:
        return {"gain": float("nan"), "dq": float("nan"), "db": float("nan"),
                "first": float("nan"), "second": float("nan")}
    r2q0 = pooled_r2(np.concatenate([np.ravel(p.q95_gt) for p in q_pls if p.q95_gt is not None]),
                     np.concatenate([np.ravel(p.q95_base) for p in q_pls if p.q95_base is not None]))
    r2b0 = pooled_r2(np.concatenate([np.ravel(p.beta_gt) for p in q_pls if p.beta_gt is not None]),
                     np.concatenate([np.ravel(p.beta_base) for p in q_pls if p.beta_base is not None]))
    r2q1 = pooled_r2(np.concatenate(gq), np.concatenate(pq))
    r2b1 = pooled_r2(np.concatenate(gb), np.concatenate(pb))
    r2q1_h = pooled_r2(np.concatenate(gq1), np.concatenate(pq1)) if gq1 else float("nan")
    r2q2_h = pooled_r2(np.concatenate(gq2), np.concatenate(pq2)) if gq2 else float("nan")
    r2q0_1 = pooled_r2(
        np.concatenate([np.ravel(p.q95_gt)[:len(p.q95_gt) // 2] for p in q_pls if p.q95_gt is not None]),
        np.concatenate([np.ravel(p.q95_base)[:len(p.q95_base) // 2] for p in q_pls if p.q95_base is not None]))
    r2q0_2 = pooled_r2(
        np.concatenate([np.ravel(p.q95_gt)[len(p.q95_gt) // 2:] for p in q_pls if p.q95_gt is not None]),
        np.concatenate([np.ravel(p.q95_base)[len(p.q95_base) // 2:] for p in q_pls if p.q95_base is not None]))
    return {
        "gain": 0.5 * ((r2q1 - r2q0) + (r2b1 - r2b0)),
        "dq": r2q1 - r2q0,
        "db": r2b1 - r2b0,
        "first": r2q1_h - r2q0_1,
        "second": r2q2_h - r2q0_2,
        "comb0": 0.5 * (r2q0 + r2b0),
        "comb1": 0.5 * (r2q1 + r2b1),
    }


def fold_sigma(gains: Sequence[float], seed: int = 42) -> float:
    x = np.asarray(gains, np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    boot = np.array([float(rng.choice(x, len(x), replace=True).mean()) for _ in range(400)])
    return float(boot.mean() / max(boot.std(), 1e-12))


def neighbor_uncertainty(neigh: np.ndarray, w: np.ndarray) -> float:
    """Weighted RMS dispersion of neighbor residuals / weighted residual magnitude."""
    neigh = np.asarray(neigh, np.float64)
    w = np.asarray(w, np.float64).ravel()
    if neigh.ndim == 1:
        neigh = neigh[:, None]
    w = np.where(np.isfinite(w), w, 0.0)
    sw = float(w.sum())
    if sw <= 0 or len(neigh) == 0:
        return 1.0
    wn = w / sw
    hat = (wn[:, None] * np.where(np.isfinite(neigh), neigh, 0.0)).sum(0)
    d2 = ((np.where(np.isfinite(neigh), neigh, hat) - hat) ** 2).sum(1)
    rms = float(np.sqrt(np.dot(wn, d2)))
    mag = float(np.sqrt(np.dot(hat, hat)))
    return float(rms / max(mag, 1e-9))


def shrink_alpha(alpha: float, lam: float, uncertainty: float) -> float:
    return float(alpha) / (1.0 + float(lam) * float(uncertainty))


def shot_concentration(gains: Sequence[float]) -> float:
    g = np.asarray(gains, np.float64)
    pos = g[g > 0]
    total = float(pos.sum()) if pos.size else 0.0
    return float(g.max() / total) if total > 0 else 0.0


def mse_cut_from_corr(zb, zg, hats, alphas) -> tuple[float, list[float]]:
    """Pooled cut plus per-shot gains. ``alphas`` is scalar or per-query."""
    if np.isscalar(alphas):
        alphas = [float(alphas)] * len(zb)
    base = cut = 0.0
    gains = []
    for z0, z1, r, a in zip(zb, zg, hats, alphas):
        mse0 = coeff_mse(z0, z1)
        zc = apply_correction(z0, r, alpha=float(a))
        mse1 = coeff_mse(zc, z1)
        T = min(len(z0), len(z1))
        base += mse0 * T
        cut += mse1 * T
        gains.append(mse0 - mse1)
    if base <= 0:
        return float("nan"), gains
    return 1.0 - cut / base, gains


def pick_nested(scores_by_choice: dict[float, list[float]],
                outer: int, prefer_small: bool = True) -> float:
    """Argmax of median score on folds != outer. Tie → smaller choice."""
    best_s, best_c = -1e300, None
    for choice, folds in scores_by_choice.items():
        others = [v for i, v in enumerate(folds) if i != outer and np.isfinite(v)]
        if not others:
            continue
        med = float(np.median(others))
        if med > best_s + 1e-15 or (
            abs(med - best_s) <= 1e-15 and prefer_small
            and (best_c is None or choice < best_c)
        ):
            best_s, best_c = med, choice
    if best_c is None:
        return min(scores_by_choice)
    return best_c


def modal_choice(choices: Sequence[float], prefer_small: bool = True) -> float:
    counts: dict[float, int] = {}
    for c in choices:
        counts[float(c)] = counts.get(float(c), 0) + 1
    best_n, best = -1, None
    for c, n in counts.items():
        if n > best_n or (n == best_n and prefer_small and (best is None or c < best)):
            best_n, best = n, c
    return float(best if best is not None else 0.0)


def fit_phase_svd(residuals: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """``residuals`` (n, 16*60). Returns mean (D,) and components (rank, D)."""
    X = np.asarray(residuals, np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ContractError(f"SVD needs a 2-D donor stack, got {X.shape}")
    X = np.where(np.isfinite(X), X, 0.0)
    mu = X.mean(0)
    Z = X - mu
    _, _, vt = np.linalg.svd(Z, full_matrices=False)
    r = max(1, min(int(rank), vt.shape[0], vt.shape[1]))
    return mu, vt[:r]


def svd_encode(residuals: np.ndarray, mu: np.ndarray, comp: np.ndarray) -> np.ndarray:
    X = np.asarray(residuals, np.float64)
    if X.ndim == 1:
        X = X[None]
    X = np.where(np.isfinite(X), X, 0.0)
    return (X - mu) @ comp.T


def svd_decode(coeff: np.ndarray, mu: np.ndarray, comp: np.ndarray) -> np.ndarray:
    C = np.asarray(coeff, np.float64)
    if C.ndim == 1:
        C = C[None]
    return C @ comp + mu


def explained_energy(residuals: np.ndarray, mu: np.ndarray, comp: np.ndarray) -> float:
    X = np.where(np.isfinite(np.asarray(residuals, np.float64)), residuals, 0.0)
    Z = X - mu
    rec = svd_decode(svd_encode(Z + mu, mu, comp), mu, comp) - mu
    num = float((rec ** 2).sum())
    den = float((Z ** 2).sum())
    return num / max(den, 1e-12)


def p16_stack(keys: Sequence[str], root: Path) -> np.ndarray:
    rows = []
    for k in keys:
        pl = load_shot_payload(k, root=root)
        if pl is None:
            rows.append(np.zeros(16 * 60, np.float64))
        else:
            rows.append(np.asarray(pl.residual_p16, np.float64).reshape(-1))
    return np.vstack(rows) if rows else np.zeros((0, 960), np.float64)


def payload_for_query(kind: str, raw: np.ndarray, n_frames: int) -> np.ndarray:
    if kind == "constant":
        return raw.reshape(-1)[:60]
    if kind == "p16":
        return interpolate_phase_residual(raw.reshape(16, 60), n_frames)
    if kind == "p32":
        return interpolate_phase_residual(raw.reshape(32, 60), n_frames)
    raise ValueError(kind)


def alpha0_identity(z_base: np.ndarray) -> bool:
    zc = apply_correction(z_base, np.zeros(z_base.shape[-1] if z_base.ndim == 2 else 60), 0.0)
    return bool(np.allclose(zc, z_base))


def holdout_eval_core(cfg: AnalogConfig, zc: list, audits: list,
                      dest: Path, fingerprint: dict, extra: dict | None = None) -> dict:
    """Official/Consistency gates; writes dest, never holdout_final.json."""
    from catskills.d3d_analog import HOUSE_S
    from catskills.lab_paths import artifacts_dir, results_dir
    from catskills.scripts.d3d_analog_eval import (
        official_proxy, paired_boot, scalars_from_z,
    )
    import torch

    keys = sample_cache_keys(0, 42, include_thomson=True)
    hold, _ = canonical_split(keys)
    hold = [k for k in hold if (shot_cache_dir() / f"{k}.npz").is_file()]
    ck = torch.load(artifacts_dir() / "CRC_FEQ_brown_TF_L_B8.pt",
                    map_location="cpu", weights_only=False)
    mean = np.asarray(ck["pca_mean"], np.float64)
    comp = np.asarray(ck["pca_components"], np.float64)
    zb = [np.asarray(load_shot_payload(k).z_base, np.float64) for k in hold]
    zg = [np.asarray(load_shot_payload(k).z_gt, np.float64) for k in hold]
    mse0 = float(np.mean([coeff_mse(a, b) for a, b in zip(zb, zg)]))
    mse1 = float(np.mean([coeff_mse(a, b) for a, b in zip(zc, zg)]))
    cut = 1.0 - mse1 / max(mse0, 1e-30)
    identity = all(alpha0_identity(a) for a in zb)
    gains = [coeff_mse(a, g) - coeff_mse(c, g) for a, c, g in zip(zb, zc, zg)]
    max_frac = shot_concentration(gains)
    gt, pred_c, shot = scalars_from_z(zc, hold, mean, comp, 1)
    _, pred_b, _ = scalars_from_z(zb, hold, mean, comp, 1)
    paired = paired_boot(gt, pred_c, pred_b, shot, 2000, 42)
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
                                  2000, 42)
    official = None
    try:
        sb, sc = official_proxy(hold, zc, zb, mean, comp, False, cfg)
        official = {"base": sb, "corr": sc, "delta_S": sc["S"] - sb["S"]}
    except Exception as e:
        official = {"error": str(e)}
    scalar_floor = min((v["full"] for v in paired["per_scalar"].values()), default=0.0)
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
    if not all(a.get("from_donor_only", True) for a in audits):
        reasons.append("neighbor_leak")
    out = {
        "cfg": cfg.as_dict(),
        "n_hold": len(hold),
        "mse0": mse0, "mse1": mse1, "mse_cut": cut,
        "paired": paired,
        "worst40": worst40,
        "official": official,
        "max_shot_gain_frac": max_frac,
        "identity": identity,
        "candidate_ready": len(reasons) == 0,
        "fail_reasons": reasons,
        "fingerprint": fingerprint,
    }
    if extra:
        out.update(extra)
    write_json_atomic(dest, out)
    return out


def catch_status(fn):
    """Run ``fn`` → (status, payload). ContractError → BLOCKED; else ERROR."""
    try:
        return fn()
    except ContractError as e:
        return "BLOCKED", {"error": str(e), "traceback": traceback.format_exc()}
    except Exception as e:
        return "ERROR", {"error": str(e), "traceback": traceback.format_exc()}
