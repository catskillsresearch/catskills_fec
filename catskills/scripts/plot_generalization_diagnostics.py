#!/usr/bin/env python3
"""Where the DIII-D generalisation gap sits, on shots that still have EFIT labels.

The public test has no ground truth here. The official 80 is the scored holdout
(S = 0.995643). The next set is the first 200 shots of the seed-42 validation
split the clean models actually held out of training (10% of the train pool,
batch 2048). q95 and betaN are the clean four-head mean. Shape scalars and
D_LCFS come from the six-member flux blend. Nothing is uploaded.

    .venv/bin/python -m catskills.scripts.plot_generalization_diagnostics
"""
from __future__ import annotations

import ast
import os
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "fusion_scoring"))
os.environ.setdefault("CRC_FEQ_MODEL_ID", "CRC_FEQ_olive_geom")
os.environ["CRC_FEQ_IP_SHIFT"] = "1"
os.environ["CRC_FEQ_THOMSON_LAYOUT"] = "zgrid"

from catskills import probe9 as p9  # noqa: E402
from catskills.feature_cache import features_path, sample_cache_keys  # noqa: E402
from catskills.lab_paths import artifacts_dir, results_dir  # noqa: E402
from catskills.score_ref_cache import load_reference  # noqa: E402
from fusion_scoring.common import AXIS_SIGN, CONS_SCALARS  # noqa: E402

# Nelder-Mead winner. M5 is the seed-42 checkpoint, so those two weights add.
W = np.array([
    0.04173094785130809,
    0.2644250942857631,
    0.10972847300443292,
    0.35214062350084935,
    0.04290408837936068,
    0.1890707729782859,
], np.float64)
FLUX = (
    ("CRC_FEQ_olive_geom_O1G_clean_ip_scratch_best", W[0] + W[4], True),
    ("CRC_FEQ_olive_geom_O1G_clean_ip_seed43_best", W[1], True),
    ("CRC_FEQ_olive_geom_O1G_clean_ip_seed44_best", W[2], True),
    ("CRC_FEQ_olive_geom_O1G_clean_ip_seed45_best", W[3], True),
)
LEGACY = (
    "CRC_FEQ_olive_geom_O1G_early_b2048",
    "CRC_FEQ_olive_geom_O1G_early",
    "CRC_FEQ_brown_TF_LS_O1_101",
    "CRC_FEQ_olive_geom_O1G_place",
    "CRC_FEQ_brown_TF_LS_O1_W1",
)
SCALAR = (
    "CRC_FEQ_olive_geom_O1G_clean_ip_scratch_best",
    "CRC_FEQ_olive_geom_O1G_clean_ip_scratch_final",
    "CRC_FEQ_olive_geom_O1G_clean_ip_seed43_best",
    "CRC_FEQ_olive_geom_O1G_clean_ip_seed43_final",
)
CONS = list(CONS_SCALARS)
# One consistency scalar is 0.20/7 of S. Drag is the points lost by sitting under 0.985.
W_CONS_EACH = 0.20 / 7.0
TARGET = 0.985
IP_COL = 20
# Thomson Z-grid: 21 magnetics, then core Te, core ne, edge Te, edge ne, 32 bins each.
TE0, NE0, ZN = 21, 53, 32
BATCH_TRAIN = 2048
N_VAL = 200
AXIS = float(AXIS_SIGN["DIII-D"])


def _npy_rows(path: Path, name: str = "X.npy") -> int:
    with zipfile.ZipFile(path) as zf, zf.open(name) as fh:
        fh.read(6)
        major = fh.read(2)[0]
        hlen = int.from_bytes(fh.read(2 if major == 1 else 4), "little")
        header = fh.read(hlen).decode("latin1")
    i, j = header.find("("), header.find(")")
    return int(ast.literal_eval(header[i:j + 1])[0])


def _val_keys(hold: list[str]) -> list[str]:
    """First 200 shots of the seed-42, 10% shot-level validation split."""
    pool = sample_cache_keys(7080, 42, include_thomson=True)
    if pool[: len(hold)] != list(hold):
        raise SystemExit("feature-cache shuffle does not start with the official holdout")
    train = pool[len(hold):]
    counts = np.empty(len(train), np.int64)
    for i, key in enumerate(train):
        counts[i] = _npy_rows(features_path(key, True))
        if (i + 1) % 1000 == 0:
            print(f"val-split frame counts {i + 1}/{len(train)}", flush=True)
    order = np.random.default_rng(42).permutation(len(train))
    n_total = int(counts.sum())
    target = n_total * 0.1
    held: list[str] = []
    taken = 0
    for s in order:
        c = int(counts[int(s)])
        if taken >= target or n_total - taken - c < BATCH_TRAIN:
            break
        held.append(train[int(s)])
        taken += c
    print(f"validation split {len(held)} shots, {taken} frames; using {min(N_VAL, len(held))}", flush=True)
    return held[:N_VAL]


def _save(fig, name: str) -> list[Path]:
    paths = []
    for folder in (results_dir() / "olive_geom", ROOT / "results" / "olive_geom"):
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        fig.savefig(path, dpi=140)
        paths.append(path)
    plt.close(fig)
    return paths


def _load_xy(keys: list[str]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    xs, ss = [], []
    for key in keys:
        with np.load(features_path(key, True)) as blob:
            xs.append(np.asarray(blob["X"], np.float32))
            ss.append(np.asarray(blob["S"], np.float32))
    return xs, ss


def _holdout_scalars(keys: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Same clock the solver scored: Ip-shifted features, tag ipshift.
    recs = [p9.member_z(stem, keys=keys, label="hold_ipshift", tag="ipshift") for stem in SCALAR]
    ids = recs[0]["shot_ids"]
    q = np.mean([r["q95"] for r in recs], axis=0)
    b = np.mean([r["betaN"] for r in recs], axis=0)
    return q, b, ids


def _holdout_psi(keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    acc = None
    ids = None
    for stem, weight, _shift in FLUX:
        rec = p9.member_z(stem, keys=keys, label="hold_ipshift", tag="ipshift")
        ids = rec["shot_ids"]
        acc = weight * rec["z"] if acc is None else acc + weight * rec["z"]
    legacy_w = float(W[5] / len(LEGACY))
    # 932612 was inferred with the Ip shift unset. Its cache is the untagged file.
    for stem in LEGACY:
        rec = p9.member_z(stem, keys=keys, label="hold", tag="")
        if not np.array_equal(rec["shot_ids"], ids):
            raise SystemExit(f"{stem} frame ids do not match the clean members")
        acc = acc + legacy_w * rec["z"]
    return p9.psi_from_z(acc), ids


def _infer_shots(stem: str, xs: list[np.ndarray], *, shifted: bool) -> list[dict]:
    if shifted:
        os.environ["CRC_FEQ_IP_SHIFT"] = "1"
    else:
        os.environ.pop("CRC_FEQ_IP_SHIFT", None)
    os.environ["CRC_FEQ_THOMSON_LAYOUT"] = "zgrid"
    from catskills.pinn_charcoal import load_charcoal, predict_from_raw_features
    from catskills.scripts.psi_decorrelate import _activate

    model, bundle, _ = load_charcoal(artifacts_dir() / f"{stem}.pt")
    _activate(bundle)
    out: list[dict] = []
    batch = 8
    for start in range(0, len(xs), batch):
        out.extend(predict_from_raw_features(bundle, xs[start:start + batch], model=model))
    del model
    import torch
    torch.cuda.empty_cache()
    os.environ["CRC_FEQ_IP_SHIFT"] = "1"
    return out


def _val_predictions(keys: list[str]) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    shifted, _ss = _load_xy(keys)
    raw = []
    for key in keys:
        path = features_path(key, True).with_name(f"{key}_thz.npz")
        with np.load(path) as blob:
            raw.append(np.asarray(blob["X"], np.float32))
    if any(a.shape[0] != b.shape[0] for a, b in zip(shifted, raw)):
        raise SystemExit("ipshift and raw frame counts differ on the validation shots")
    psi = [np.zeros((x.shape[0], 65, 65), np.float64) for x in shifted]
    q = [np.zeros(x.shape[0], np.float64) for x in shifted]
    b = [np.zeros(x.shape[0], np.float64) for x in shifted]
    for stem, weight, use_shift in FLUX:
        print(f"val flux {stem} w={weight:.6f}", flush=True)
        preds = _infer_shots(stem, shifted if use_shift else raw, shifted=use_shift)
        for i, pred in enumerate(preds):
            psi[i] += weight * np.asarray(pred["psirz"], np.float64)
    legacy_w = float(W[5] / len(LEGACY))
    for stem in LEGACY:
        print(f"val flux {stem} w={legacy_w:.6f}", flush=True)
        preds = _infer_shots(stem, raw, shifted=False)
        for i, pred in enumerate(preds):
            psi[i] += legacy_w * np.asarray(pred["psirz"], np.float64)
    for stem in SCALAR:
        print(f"val scalar {stem}", flush=True)
        preds = _infer_shots(stem, shifted, shifted=True)
        for i, pred in enumerate(preds):
            q[i] += np.asarray(pred["q95"], np.float64).reshape(-1)
            b[i] += np.asarray(pred["betaN"], np.float64).reshape(-1)
    q = np.concatenate(q) / len(SCALAR)
    b = np.concatenate(b) / len(SCALAR)
    return [p.astype(np.float32) for p in psi], q, b


def _pressure(x: np.ndarray) -> np.ndarray:
    te = x[:, TE0:TE0 + ZN]
    ne = x[:, NE0:NE0 + ZN]
    prod = te * ne
    ok = np.isfinite(prod) & (ne > 0) & (te > 0)
    prod = np.where(ok, prod, np.nan)
    with np.errstate(all="ignore"):
        return np.nanmean(prod, axis=1).astype(np.float64)


def _shot_frames(job: tuple[str, np.ndarray]) -> dict:
    """Per-frame boundary error and the seven derived scalars for one shot."""
    key, psi = job
    from contour import symmetric_hausdorff
    from fusion_scoring.derive import derive_frame
    from fusion_scoring.lcfs import extract_lcfs_with_sign

    ref = load_reference(key)
    if ref is None:
        return {"key": key, "missing": True}
    contours, cons, cmask, rgeo = ref
    mask = np.load(ROOT / "fusion_scoring" / "masks" / "d3d_envelope.npz")
    R, Z = np.asarray(mask["grid_R"], np.float64), np.asarray(mask["grid_Z"], np.float64)
    coarse = mask["mask_coarse"].astype(bool)
    mask_f = coarse.astype(np.float64)
    t = min(len(psi), len(contours), len(cons))
    pred = np.full((t, 7), np.nan, np.float64)
    dist = np.full(t, np.nan, np.float64)
    for k in range(t):
        ex, _sign = extract_lcfs_with_sign(
            np.asarray(psi[k], np.float64), R, Z, "DIII-D", coarse, mask_f, axis_sign=AXIS,
        )
        ref_c = contours[k]
        if ref_c is not None:
            if ex is None or not np.isfinite(rgeo) or rgeo <= 0:
                dist[k] = 1.0
            else:
                dist[k] = min(1.0, symmetric_hausdorff(ex, np.asarray(ref_c, np.float64)) / rgeo)
        if cmask[k].any():
            vals = derive_frame(
                np.asarray(psi[k], np.float64), R, Z, "DIII-D", coarse, mask_f,
                contour=ex, with_li=True, axis_sign=AXIS,
            )
            for j, name in enumerate(CONS):
                pred[k, j] = vals[name]
    return {
        "key": key,
        "missing": False,
        "gt": np.asarray(cons[:t], np.float64),
        "ok": np.asarray(cmask[:t], bool),
        "pred": pred,
        "D": dist,
        "rgeo": float(rgeo),
    }


def _r2(gt: np.ndarray, pred: np.ndarray, ok: np.ndarray) -> float:
    g = gt[ok]
    if g.size < 2:
        return float("nan")
    p = pred[ok]
    mean = float(g.mean())
    p = np.where(np.isfinite(p), p, mean)
    tot = float(np.sum((g - mean) ** 2))
    if tot <= 0:
        return float("nan")
    return 1.0 - float(np.sum((g - p) ** 2)) / tot


def _pooled(rows: list[dict]) -> dict[str, float]:
    gt = np.concatenate([r["gt"] for r in rows])
    pred = np.concatenate([r["pred"] for r in rows])
    ok = np.concatenate([r["ok"] for r in rows])
    return {name: _r2(gt[:, j], pred[:, j], ok[:, j]) for j, name in enumerate(CONS)}


def _drag(r2: float) -> float:
    if not np.isfinite(r2):
        return float("nan")
    return W_CONS_EACH * max(0.0, TARGET - r2)


def _wing(dq: np.ndarray, ip: np.ndarray, q: np.ndarray) -> dict:
    ok = np.isfinite(dq) & np.isfinite(ip) & np.isfinite(q)
    dq, ip, q = dq[ok], np.abs(ip[ok]), q[ok]
    sse = dq ** 2
    total = float(sse.sum())
    lo = float(np.quantile(ip, 0.2))
    wing = ip <= lo
    ramp = ip < 0.4
    high = q > 8.0

    def share(mask: np.ndarray) -> float:
        return float(sse[mask].sum() / total) if total > 0 and mask.any() else float("nan")

    def med(mask: np.ndarray) -> float:
        return float(np.median(np.abs(dq[mask]))) if mask.any() else float("nan")

    return {
        "frames": int(dq.size),
        "ip_p20_MA": lo,
        "low_ip_frame_frac": float(wing.mean()),
        "low_ip_sse_frac": share(wing),
        "low_ip_median_abs": med(wing),
        "core_median_abs": med(~wing),
        "ramp_lt_0p4MA_frame_frac": float(ramp.mean()) if dq.size else float("nan"),
        "ramp_sse_frac": share(ramp),
        "ramp_median_abs": med(ramp),
        "flattop_median_abs": med(~ramp),
        "q_gt8_frame_frac": float(high.mean()) if dq.size else float("nan"),
        "q_gt8_sse_frac": share(high),
        "q_gt8_median_abs": med(high),
        "q_le8_median_abs": med(~high),
    }


def _scatter(ax, x, y, color, label):
    ax.scatter(x, y, s=6, c=color, alpha=0.35, linewidths=0, rasterized=True, label=label)


def _fig_scalars(pack: dict, *, title: str | None = None) -> list[Path]:
    fig = plt.figure(figsize=(11.2, 12.2))
    grid = fig.add_gridspec(3, 2, hspace=0.32, wspace=0.28)
    ax_q = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_qi = fig.add_subplot(grid[1, 0])
    ax_qq = fig.add_subplot(grid[1, 1])
    ax_bi = fig.add_subplot(grid[2, 0])
    ax_bp = fig.add_subplot(grid[2, 1])
    colors = {"holdout": "#9a3412", "validation": "#1d4ed8"}
    for name, color in colors.items():
        d = pack[name]
        _scatter(ax_q, d["q_true"], d["q_pred"], color, name)
        _scatter(ax_b, d["b_true"], d["b_pred"], color, name)
        _scatter(ax_qi, np.abs(d["ip"]), d["q_pred"] - d["q_true"], color, name)
        _scatter(ax_qq, d["q_true"], d["q_pred"] - d["q_true"], color, name)
        _scatter(ax_bi, np.abs(d["ip"]), d["b_pred"] - d["b_true"], color, name)
        _scatter(ax_bp, d["pe"], d["b_pred"] - d["b_true"], color, name)
    for ax, x, y in ((ax_q, pack["holdout"]["q_true"], pack["holdout"]["q_pred"]),
                     (ax_b, pack["holdout"]["b_true"], pack["holdout"]["b_pred"])):
        both = np.concatenate([
            np.concatenate([pack[k]["q_true"] if ax is ax_q else pack[k]["b_true"] for k in colors]),
            np.concatenate([pack[k]["q_pred"] if ax is ax_q else pack[k]["b_pred"] for k in colors]),
        ])
        both = both[np.isfinite(both)]
        lo, hi = float(np.quantile(both, 0.001)), float(np.quantile(both, 0.999))
        pad = 0.04 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.25", lw=1.0)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
    ax_q.set_title("A  $q_{95}$")
    ax_b.set_title("B  $\\beta_N$")
    ax_q.set_xlabel("EFIT $q_{95}$")
    ax_b.set_xlabel("EFIT $\\beta_N$")
    ax_q.set_ylabel("prediction" if title else "clean head")
    ax_b.set_ylabel("prediction" if title else "clean head")
    ax_qi.axhline(0, color="0.3", lw=0.8)
    ax_qi.axvline(0.4, color="0.45", lw=0.8, ls="--")
    ax_qq.axhline(0, color="0.3", lw=0.8)
    ax_qq.axvline(8, color="0.45", lw=0.8, ls="--")
    ax_bi.axhline(0, color="0.3", lw=0.8)
    ax_bp.axhline(0, color="0.3", lw=0.8)
    ax_qi.set_title("C  $\\Delta q_{95}$ vs $|I_p|$")
    ax_qq.set_title("C  $\\Delta q_{95}$ vs true $q_{95}$")
    ax_bi.set_title("D  $\\Delta\\beta_N$ vs $|I_p|$")
    ax_bp.set_title("D  $\\Delta\\beta_N$ vs core Thomson $T_e n_e$")
    ax_qi.set_xlabel("$|I_p|$ (MA)")
    ax_qq.set_xlabel("EFIT $q_{95}$")
    ax_bi.set_xlabel("$|I_p|$ (MA)")
    ax_bp.set_xlabel("mean core $T_e \\cdot n_e$")
    for ax in (ax_qi, ax_qq, ax_bi, ax_bp):
        ax.set_ylabel("prediction − EFIT")
    ax_q.legend(frameon=False, markerscale=2)
    fig.suptitle(title or "Clean scalar head  ·  holdout 80 and validation 200", y=0.995)
    return _save(fig, "d3d_scalar_error_regimes.png")


def _fig_cons(tables: dict[str, dict[str, float]]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    x = np.arange(len(CONS))
    width = 0.38
    colors = {"holdout": "#9a3412", "validation": "#1d4ed8"}
    for i, (name, color) in enumerate(colors.items()):
        vals = [tables[name][k] for k in CONS]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, color=color, label=name)
        for bar, key, val in zip(bars, CONS, vals):
            drag = _drag(val)
            if drag > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, val + 0.0015,
                    f"−{drag:.5f}", ha="center", va="bottom", fontsize=7, color=color,
                )
    ax.axhline(TARGET, color="0.2", ls="--", lw=1.0, label="0.985")
    ax.set_xticks(x)
    ax.set_xticklabels(CONS)
    finite = [v for table in tables.values() for v in table.values() if np.isfinite(v)]
    floor = 0.94 if not finite else min(0.94, min(finite) - 0.01)
    ax.set_ylim(floor, 1.005)
    ax.set_ylabel("pooled $R^2$")
    ax.set_title("Consistency scalars  ·  labels are the S-point drag under 0.985")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    return _save(fig, "d3d_consistency_breakdown.png")


def _fig_heat(rows: list[dict], *, title: str | None = None) -> list[Path]:
    kappa = np.concatenate([r["gt"][:, 2] for r in rows])
    delta = 0.5 * (
        np.concatenate([r["gt"][:, 3] for r in rows])
        + np.concatenate([r["gt"][:, 4] for r in rows])
    )
    dist = np.concatenate([r["D"] for r in rows])
    ok = np.isfinite(kappa) & np.isfinite(delta) & np.isfinite(dist)
    kappa, delta, dist = kappa[ok], delta[ok], dist[ok]
    kx = np.linspace(1.2, 2.2, 21)
    dy = np.linspace(0.0, 0.9, 19)
    total = np.histogram2d(kappa, delta, bins=[kx, dy])[0]
    summed = np.histogram2d(kappa, delta, bins=[kx, dy], weights=dist)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = summed / total
    mean = np.where(total >= 8, mean, np.nan)
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    image = ax.pcolormesh(kx, dy, mean.T, cmap="magma_r", shading="flat")
    bar = fig.colorbar(image, ax=ax, pad=0.02)
    bar.set_label("mean $D_{LCFS}$")
    outside = int(np.sum((kappa < 1.2) | (kappa > 2.2) | (delta < 0) | (delta > 0.9)))
    neg = delta < 0
    high_k = kappa >= 1.9
    inside = (kappa >= 1.2) & (kappa <= 2.2) & (delta >= 0.0) & (delta <= 0.9)
    print(
        f"mean D  all {float(dist.mean()):.4f}  "
        f"in-window {float(dist[inside].mean()):.4f} n {int(inside.sum())}  "
        f"δ<0 {float(dist[neg].mean()) if neg.any() else float('nan'):.4f} n {int(neg.sum())}  "
        f"κ≥1.9 {float(dist[high_k].mean()) if high_k.any() else float('nan'):.4f} n {int(high_k.sum())}",
        flush=True,
    )
    ax.set_xlabel("EFIT elongation $\\kappa$")
    ax.set_ylabel(r"EFIT triangularity $\delta=(\delta_{top}+\delta_{bot})/2$")
    ax.set_title(
        (title or "Boundary error by shape")
        + f"  ·  {kappa.size - outside} frames in range"
        + f"  ·  {int(neg.sum())} with $\\delta<0$"
    )
    fig.tight_layout()
    paths = _save(fig, "d3d_shape_error_heatmap.png")
    finite = mean[np.isfinite(mean)]
    if finite.size:
        ij = np.unravel_index(np.nanargmax(mean), mean.shape)
        print(
            f"worst bin κ {kx[ij[0]]:.2f}–{kx[ij[0]+1]:.2f}  "
            f"δ {dy[ij[1]]:.2f}–{dy[ij[1]+1]:.2f}  "
            f"mean D {mean[ij]:.4f}  n {int(total[ij])}",
            flush=True,
        )
    print(f"frames outside the κ–δ window {outside}  negative triangularity {int(neg.sum())}", flush=True)
    return paths


def _pack_split(name: str, q_pred, b_pred, ss, xs, shot_ids) -> dict:
    q_true, b_true, ip, pe = [], [], [], []
    cursor = 0
    for i, (s, x) in enumerate(zip(ss, xs)):
        if shot_ids is None:
            sl = slice(cursor, cursor + len(x))
            cursor += len(x)
        else:
            sl = shot_ids == i
        n = min(len(x), len(s), int(np.sum(sl) if shot_ids is not None else (sl.stop - sl.start)))
        # Align by leading frames. Caches and feature files share the EFIT clock.
        q_true.append(s[:n, 2])
        b_true.append(s[:n, 0])
        ip.append(x[:n, IP_COL] / 1e3)
        pe.append(_pressure(x[:n]))
        if shot_ids is None:
            q_pred_i = q_pred[sl][:n]
            b_pred_i = b_pred[sl][:n]
        else:
            q_pred_i = q_pred[sl][:n]
            b_pred_i = b_pred[sl][:n]
        if i == 0:
            qp, bp = [q_pred_i], [b_pred_i]
        else:
            qp.append(q_pred_i)
            bp.append(b_pred_i)
    return {
        "q_true": np.concatenate(q_true).astype(np.float64),
        "q_pred": np.concatenate(qp).astype(np.float64),
        "b_true": np.concatenate(b_true).astype(np.float64),
        "b_pred": np.concatenate(bp).astype(np.float64),
        "ip": np.concatenate(ip).astype(np.float64),
        "pe": np.concatenate(pe).astype(np.float64),
    }


def main() -> int:
    hold = list(p9.holdout_keys())
    print(f"holdout {len(hold)}", flush=True)
    val = _val_keys(hold)
    print("loading holdout blend from the member caches", flush=True)
    q_h, b_h, ids_h = _holdout_scalars(hold)
    psi_h, ids_psi = _holdout_psi(hold)
    if not np.array_equal(ids_h, ids_psi):
        raise SystemExit("scalar and flux caches are not on the same frames")
    xs_h, ss_h = _load_xy(hold)
    print(f"validation inference {len(val)} shots", flush=True)
    psi_v, q_v, b_v = _val_predictions(val)
    xs_v, ss_v = _load_xy(val)

    pack = {
        "holdout": _pack_split("holdout", q_h, b_h, ss_h, xs_h, ids_h),
        "validation": _pack_split("validation", q_v, b_v, ss_v, xs_v, None),
    }
    for name, d in pack.items():
        wing = _wing(d["q_pred"] - d["q_true"], d["ip"], d["q_true"])
        print(f"q95 {name} {wing}", flush=True)
        beta = _wing(d["b_pred"] - d["b_true"], d["ip"], d["q_true"])
        print(
            f"betaN {name} low-Ip median |Δ| {beta['low_ip_median_abs']:.4f}  "
            f"core {beta['core_median_abs']:.4f}  "
            f"low-Ip SSE share {beta['low_ip_sse_frac']:.3f} "
            f"on {beta['low_ip_frame_frac']:.3f} of frames",
            flush=True,
        )

    jobs = [(key, psi_h[ids_h == i]) for i, key in enumerate(hold)]
    jobs += list(zip(val, psi_v))
    print(f"deriving boundaries on {len(jobs)} shots", flush=True)
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    rows = []
    with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("spawn")) as pool:
        for n, row in enumerate(pool.map(_shot_frames, jobs, chunksize=2), 1):
            rows.append(row)
            if n % 40 == 0 or n == len(jobs):
                print(f"boundaries {n}/{len(jobs)}", flush=True)
    missing = [r["key"] for r in rows if r.get("missing")]
    if missing:
        print(f"missing score refs {len(missing)}", flush=True)
    rows = [r for r in rows if not r.get("missing")]
    hold_rows = rows[: len(hold)]
    val_rows = rows[len(hold):]
    tables = {"holdout": _pooled(hold_rows), "validation": _pooled(val_rows)}
    for split, table in tables.items():
        cons = float(np.mean([max(0.0, table[k]) for k in CONS]))
        print(f"{split} consistency {cons:.6f}", flush=True)
        for name in CONS:
            print(f"  {name:8} R2 {table[name]:.6f}  drag { _drag(table[name]):.6f}", flush=True)
        group = hold_rows if split == "holdout" else val_rows
        dist = np.concatenate([r["D"] for r in group])
        dist = dist[np.isfinite(dist)]
        shot_d = [float(np.nanmean(r["D"])) for r in group if np.isfinite(r["D"]).any()]
        print(
            f"  mean D_LCFS frame {float(dist.mean()):.6f}  "
            f"shot {float(np.mean(shot_d)):.6f}  frames {dist.size}",
            flush=True,
        )

    paths = []
    paths += _fig_scalars(pack)
    paths += _fig_cons(tables)
    paths += _fig_heat(rows)
    for path in paths:
        print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
