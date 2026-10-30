#!/usr/bin/env python3
"""S30.7/S30.9: can the psi-MSE tail be trained away?

The chain that leads here: Consistency is a slave variable of psi error (S30.6), psi
MSE is decided by a 1% tail of frames (S30.7), the per-frame oracle over existing
members is hindsight and unreachable (S30.8), and the tail is largely a *per-shot
constant offset* worth 39.8% of total MSE -- but neither the vacuum field nor Thomson
flux-surface consistency can detect that offset without labels (S30.9).

That leaves the training side. In training the labels ARE available, so the tail can be
targeted directly: fine-tune the parent with each frame weighted by the parent's own
error, w_i = (e_i / mean e)^p, mean-normalised. p=0 is an ordinary fine-tune and is the
control that separates "fine-tuning helps" from "weighting helps".

S34.17: public R2_qβ is the second half of each shot. `--late` multiplies those
error-weights by (1 + late * frac), frac = 0 at the first frame of a shot and 1 at
the last. late=0 is the S30 control; late=2 makes the last frame 3× the first.
This is a time weight, not the error-magnitude weight that hurt at p>0 (S30.10).
Also reports second-half holdout MSE (the public qβ fold).

Scored the way S30.6 says to score: plain coefficient MSE on held-out shots, which is
exactly psi MSE in the frozen orthonormal basis. No scorer in the loop.

  .venv/bin/python -m catskills.scripts.psi_tailfit --power 1.0 --tag T1
  .venv/bin/python -m catskills.scripts.psi_tailfit --power 0.0 --tag T0   # control
  .venv/bin/python -m catskills.scripts.psi_tailfit --power 0.0 --late 2 --tag LATE
  .venv/bin/python -m catskills.scripts.psi_tailfit --power 0.0 --lambda-w 1 --tag W1

PASS  >=15% holdout psi-MSE cut against the B8+F9 pair (S30.6 gate).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from catskills.coeff_cache import build_coeff_matrix_from_keys  # noqa: E402
from catskills.feature_cache import sample_cache_keys  # noqa: E402
from catskills.lab_paths import artifacts_dir, results_dir  # noqa: E402
from catskills.pinn_charcoal import (  # noqa: E402
    _clip_features, augment_temporal, load_charcoal,
)
from catskills.scripts.psi_decorrelate import _activate  # noqa: E402

PARENTS = {
    "B8": "CRC_FEQ_brown_B8_zgrid_e1", "F9": "CRC_FEQ_forest_F9_unfreeze_edge",
    "B7": "CRC_FEQ_brown_B7_zgrid_dualctx", "B11": "CRC_FEQ_brown_B11_thzr_dualctx",
    "B13": "CRC_FEQ_brown_B13_vac_in", "Y2": "CRC_FEQ_yellow_Y2_parent",
    "S1": "CRC_FEQ_sky_S1_tokens",     "O1": "CRC_FEQ_olive_O1_dualctx",
    "O1G": "CRC_FEQ_olive_geom_O1G_place",
    "F1P": "CRC_FEQ_flux1_F1_TH1_PLACE",
    "F2": "CRC_FEQ_flux1_F2_PLACE",
}
N_HOLD = 80


def shot_frac(ids: np.ndarray) -> np.ndarray:
    """0 at the first frame of each shot, 1 at the last (S34.17 public qβ fold)."""
    frac = np.empty(len(ids), dtype=np.float64)
    for s in np.unique(ids):
        m = ids == s
        n = int(m.sum())
        frac[m] = np.arange(n, dtype=np.float64) / max(n - 1, 1)
    return frac


def _fit_width(X: np.ndarray, n_raw: int) -> np.ndarray:
    """Cache width is the max over the batch, so a small batch is narrower (S30.5)."""
    if X.shape[1] == n_raw:
        return X
    if X.shape[1] < n_raw:
        return np.hstack([X, np.zeros((X.shape[0], n_raw - X.shape[1]), dtype=X.dtype)])
    return X[:, :n_raw]


def build(keys, bundle, label, n_raw):
    # A bundle's features are NOT interchangeable across families: the cache key must
    # carry the layout and widths or one parent silently reads another's features (S30.8).
    fam = f"{bundle.thomson_layout or 'th1'}_{n_raw}_{len(bundle.feature_mean)}"
    cache = results_dir() / "brown" / f"tailfit_{label}_{fam}"
    if (cache.with_suffix(".npz")).is_file():
        d = np.load(cache.with_suffix(".npz"))
        if d["X"].shape[1] == len(bundle.feature_mean):
            print(f"[build] cache hit {cache.name} {d['X'].shape}", flush=True)
            return d["X"], d["Z"], d["ids"]
    X, Z, _S, ids, _st = build_coeff_matrix_from_keys(keys, k=60, include_thomson=True)
    X = augment_temporal(_fit_width(X, n_raw), ids, bundle.temporal_window,
                         bundle.temporal_mode, seq_window=int(bundle.seq_window or 0))
    X = _clip_features(((X - bundle.feature_mean) / bundle.feature_std).astype(np.float32),
                       bundle.feature_clip, label=label)
    Z = Z.astype(np.float32)
    np.savez(cache.with_suffix(".npz"), X=X, Z=Z, ids=ids)
    return X, Z, ids


def load_warp(keys: list[str], ids: np.ndarray) -> np.ndarray | None:
    """(n_frames, 60, 5) orthonormal bases; missing shots -> identity column omitted."""
    root = results_dir() / "brown" / "warp_basis"
    if not root.is_dir():
        return None
    B = np.zeros((len(ids), 60, 5), np.float32)
    miss = 0
    for i, key in enumerate(keys):
        p = root / f"{key}.npz"
        if not p.is_file():
            miss += 1
            continue
        b = np.load(p)["B"].astype(np.float32)
        B[ids == i] = b
    if miss:
        print(f"[warp] missing basis for {miss}/{len(keys)} shots", flush=True)
    if miss == len(keys):
        return None
    return B


@torch.no_grad()
def infer(model, xs, device, bs=4096):
    model.eval()
    out = []
    for s in range(0, len(xs), bs):
        out.append(model(xs[s:s + bs].to(device, non_blocking=True))[0].detach().float())
    return torch.cat(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--power", type=float, default=1.0,
                    help="w = (e/mean e)^power; 0 = uniform fine-tune (control)")
    ap.add_argument("--cap", type=float, default=50.0, help="max weight after normalising")
    ap.add_argument("--parent", default="B8", choices=sorted(PARENTS))
    ap.add_argument("--tag", default="T1")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42,
                    help="init + shuffling + val split; O1/U and O1/L differ only by run "
                         "noise and still mix (S30.10), so seeds are a real lever")
    ap.add_argument("--late", type=float, default=0.0,
                    help="S34.17: multiply weights by (1 + late * frac); 0 = off")
    ap.add_argument("--lambda-w", type=float, default=0.0,
                    help="S34.12 Arm A: upweight coeff error in GT warp span")
    ap.add_argument("--split-json", default="",
                    help="probe9 P9: JSON with 'hold' and 'train' key lists (and optional "
                         "'name'); replaces the default 7080/seed-42 split. Feature caches "
                         "are keyed by the split name so folds cannot collide")
    ap.add_argument("--metric-g", default="",
                    help="probe9 P3b: path to a (60,60) SPD geometry metric (npy or npz "
                         "with key 'G'). Loss becomes err^T G err instead of plain coeff "
                         "MSE; validation stays plain psi-MSE so selection is unchanged")
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.backends.cudnn.enabled = False
    t0 = time.perf_counter()
    ck = artifacts_dir() / f"{PARENTS[args.parent]}.pt"
    model, bundle, _ck = load_charcoal(ck, device=device)
    # S30.8: features MUST be rebuilt under the bundle's own family/layout env or the
    # normalisation is silently meaningless (B8 is zgrid -> |z| ~ 1e21, 47% of cells clipped).
    _activate(bundle)
    print(f"### {args.tag} parent={args.parent} power={args.power} late={args.late} "
          f"lambda_w={args.lambda_w} lr={args.lr} layout={bundle.thomson_layout!r}", flush=True)

    keys = sample_cache_keys(7000 + N_HOLD, 42, include_thomson=True)
    hold_keys, train_keys = keys[:N_HOLD], keys[N_HOLD:]
    hold_label, train_label = "hold", "train"
    if args.split_json:
        spl = json.loads(Path(args.split_json).read_text())
        hold_keys, train_keys = list(spl["hold"]), list(spl["train"])
        name = str(spl.get("name") or Path(args.split_json).stem)
        # `build` caches by label, so a different fold MUST carry a different label or it
        # silently reads the legacy 80's feature matrix back.
        hold_label, train_label = f"hold_{name}", f"train_{name}"
        print(f"[{args.tag}] split {name}: hold {len(hold_keys)} train {len(train_keys)}",
              flush=True)
    n_raw = int(_ck.get("n_raw_features") or len(bundle.feature_mean) // 3)
    X, Z, ids = build(train_keys, bundle, train_label, n_raw)
    xs = torch.from_numpy(np.ascontiguousarray(X))
    zc = torch.from_numpy(Z)
    print(f"[{args.tag}] train {X.shape[0]:,} frames x {X.shape[1]}", flush=True)

    # parent error per frame -> the weights
    z0 = infer(model, xs, device)
    e = ((z0.cpu() - zc) ** 2).mean(1).numpy().astype(np.float64)
    base_mse = float(e.mean())
    w = np.ones_like(e) if args.power == 0 else (e / max(e.mean(), 1e-12)) ** args.power
    if args.late != 0.0:
        frac = shot_frac(ids)
        w = w * (1.0 + args.late * frac)
        print(f"[{args.tag}] late={args.late}  frac mean {frac.mean():.3f}  "
              f"w_first {float(w[frac < 0.1].mean()):.3f}  "
              f"w_second {float(w[frac >= 0.5].mean()):.3f}", flush=True)
    w = np.minimum(w / w.mean(), args.cap)
    w = w / w.mean()
    print(f"[{args.tag}] parent train MSE {base_mse:.4e}  weights: "
          f"median {np.median(w):.3f} p99 {np.percentile(w, 99):.2f} max {w.max():.2f}", flush=True)
    wt = torch.from_numpy(w.astype(np.float32))

    warp = None
    if args.lambda_w > 0:
        B = load_warp(train_keys, ids)
        if B is None:
            raise SystemExit("lambda-w > 0 but warp_basis cache missing; run warp_coeff_cache")
        warp = torch.from_numpy(B)
        print(f"[{args.tag}] warp bases loaded  shape {tuple(B.shape)}", flush=True)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    shots = np.unique(ids)
    rng.shuffle(shots)
    val_m = np.isin(ids, shots[: max(1, int(0.1 * len(shots)))])
    tr_i = np.flatnonzero(~val_m)
    va_i = np.flatnonzero(val_m)

    # holdout, scored exactly as S30.7 does
    Xh, Zh, idh = build(hold_keys, bundle, hold_label, n_raw)
    xh = torch.from_numpy(np.ascontiguousarray(Xh))
    zh = torch.from_numpy(Zh)
    sup = results_dir() / "brown" / "st_support_cache"
    # The support cache and the B8+F9 reference coefficients exist only for the legacy 80.
    # On any other fold the mask is all-true and the gate is read against the parent.
    have_sup = all((sup / f"{k}.npz").is_file() for k in hold_keys)
    okh = (np.concatenate([(lambda d: d["ok_mean"] & d["ok_gt"])(np.load(sup / f"{k}.npz"))
                           for k in hold_keys])
           if have_sup else np.ones(len(Zh), dtype=bool))
    D = results_dir() / "brown" / "decorr_z"
    zgt_h = zh.numpy()[okh].astype(np.float64)
    pair_mse = float("nan")
    if have_sup:
        pair = 0.5 * (np.load(D / "B8.npy") + np.load(D / "F9.npy")).astype(np.float64)
        if pair.shape == zgt_h.shape:
            pair_mse = float(((pair - zgt_h) ** 2).mean())

    frac_h = shot_frac(idh)[okh]
    late_h = frac_h >= 0.5

    def hold_mse(m, mask=None):
        zp = infer(m, xh, device).cpu().numpy().astype(np.float64)[okh]
        d = (zp - zgt_h) ** 2
        if mask is None:
            return float(d.mean())
        return float(d[mask].mean())

    parent_hm = hold_mse(model)
    print(f"[{args.tag}] holdout: parent {parent_hm:.4e}  "
          f"second-half {hold_mse(model, late_h):.4e}  B8+F9 pair {pair_mse:.4e}", flush=True)

    metric_g = None
    if args.metric_g:
        gp = Path(args.metric_g)
        raw = np.load(gp)
        Gm = np.asarray(raw["G"] if hasattr(raw, "files") else raw, np.float64)
        if Gm.shape != (60, 60):
            raise SystemExit(f"--metric-g must be (60, 60), got {Gm.shape}")
        # Normalise to unit mean eigenvalue so lr and the psi floor keep their meaning.
        Gm = Gm / max(np.trace(Gm) / 60.0, 1e-30)
        metric_g = torch.from_numpy(Gm.astype(np.float32)).to(device)
        print(f"[{args.tag}] geometry metric from {gp} "
              f"cond={np.linalg.cond(Gm):.3e}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best = {"val": 1e9, "ep": -1, "state": None}
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(tr_i)
        for s in range(0, len(perm), args.batch):
            idx = torch.from_numpy(perm[s:s + args.batch].astype(np.int64))
            xb = xs[idx].to(device, non_blocking=True)
            zb = zc[idx].to(device, non_blocking=True)
            wb = wt[idx].to(device, non_blocking=True)
            zp = model(xb)[0]
            err = zp - zb
            if metric_g is not None:
                sq = torch.einsum("bk,kl,bl->b", err, metric_g, err) / err.shape[1]
            elif warp is not None:
                Bb = warp[idx].to(device, non_blocking=True)
                # err (batch,60), Bb (batch,60,5)
                c = torch.einsum("bk,bkf->bf", err, Bb)
                e_par = torch.einsum("bf,bkf->bk", c, Bb)
                sq = (err * err).mean(1) + args.lambda_w * (e_par * e_par).mean(1)
            else:
                sq = (err * err).mean(1)
            loss = (wb * sq).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        zv = infer(model, xs[torch.from_numpy(va_i.astype(np.int64))], device).cpu()
        val = float(((zv - zc[torch.from_numpy(va_i.astype(np.int64))]) ** 2).mean())
        flag = ""
        if val < best["val"]:
            best = {"val": val, "ep": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            flag = " *"
        print(f"[{args.tag}] ep {ep:3d}  val psi-MSE {val:.5e}{flag}", flush=True)
        if ep - best["ep"] >= args.patience:
            print(f"[{args.tag}] early stop", flush=True)
            break

    model.load_state_dict(best["state"])
    hm = hold_mse(model)
    hm_late = hold_mse(model, late_h)
    # On a fold without the B8+F9 reference the cut is read against the parent instead,
    # and the 15% gate does not apply -- that number is calibrated to the pair.
    against_pair = bool(np.isfinite(pair_mse))
    ref_mse = pair_mse if against_pair else parent_hm
    cut = 1 - hm / ref_mse
    print(f"\n[{args.tag}] best ep {best['ep']}  holdout psi-MSE {hm:.4e}  "
          f"second-half {hm_late:.4e}")
    print(f"[{args.tag}] vs {'B8+F9 pair' if against_pair else 'parent'} {ref_mse:.4e}  "
          f"cut {100 * cut:.2f}%  ({time.perf_counter() - t0:.0f}s)")
    verdict = "PASS" if (cut >= 0.15 if against_pair else cut > 0.0) else "FAIL"
    print(f"[{args.tag}] {verdict}  (gate: {'>=15% vs pair' if against_pair else '>0% vs parent'})")
    out = {"tag": args.tag, "power": args.power, "late": args.late,
           "lambda_w": args.lambda_w, "metric_g": args.metric_g or None,
           "split_json": args.split_json or None, "cut_against": "pair" if against_pair
           else "parent",
           "verdict": verdict, "cut": cut,
           "holdout_mse": hm, "holdout_mse_second_half": hm_late,
           "parent_holdout_mse": parent_hm,
           "pair_mse": pair_mse, "best_epoch": best["ep"],
           "val": best["val"]}
    (results_dir() / "brown" / f"P30_TAILFIT_{args.tag}.json").write_text(json.dumps(out, indent=2))
    torch.save({**torch.load(ck, map_location="cpu", weights_only=False),
                "model_state": best["state"]},
               artifacts_dir() / f"CRC_FEQ_brown_TF_{args.tag}.pt")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
