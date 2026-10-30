#!/usr/bin/env python3
"""S33: error bars for the per-scalar Consistency R2, and a test of the S31.6 mechanism.

Every Consistency term is a POOLED R2, so its denominator is the ground-truth spread of the
fold being scored (S31.6). We have been reading local per-scalar R2 to three decimals off a
single 80-shot draw, with no error bar, and shipping on differences of ~0.001.

The public detailed results for the shipped hybrid (916155) say the local->public move is not
uniform: `li` -0.0166 and `Z_axis` -0.0102, but `kappa` +0.0061 and `tri_bot` +0.0043. Mixed
signs rule out "the public fold is harder" and point at fold spread. This script measures how
much of that swing a fold draw can produce on its own:

  1. per-frame official scalars for the shipped mix and for the labels, cached once
  2. bootstrap / subsample folds of shots, pooled R2 per scalar per fold
  3. regress each scalar's R2 on that fold's GT spread -- S31.6's mechanism, quantified
  4. RMSE per scalar, which is fold-INDEPENDENT and is what to optimise if (3) holds

Scored with the official fusion_scoring derive through `local_score.build_reference`, and with
the scorer's own nan semantics: a non-finite prediction is mean-substituted (it scores as a
constant predictor), a non-finite label is dropped.

  .venv/bin/python -m catskills.scripts.cons_foldvar
  .venv/bin/python -m catskills.scripts.cons_foldvar --stride 4      # quick
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
sys.path.insert(0, str(ROOT / "fusion_scoring"))

from catskills.feature_cache import sample_cache_keys  # noqa: E402
from catskills.lab_paths import artifacts_dir, results_dir  # noqa: E402
from catskills.operator_targets import CONS_NAMES, official_grid  # noqa: E402
from catskills.pinn_charcoal import load_charcoal  # noqa: E402
from catskills.scripts.psi_decorrelate import _activate  # noqa: E402
from catskills.scripts.psi_tailfit import build, infer  # noqa: E402
from local_score import build_reference  # noqa: E402

# The shipped hybrid mix (S30.10 / S30.11, public 916155) and the baseline it replaced
# (B8+F9, public 912438). Both are scored on the same 80 holdout shots, which is what makes
# the paired comparison legal.
MIX = ["CRC_FEQ_brown_TF_L_B8", "CRC_FEQ_brown_TF_U_O1", "CRC_FEQ_brown_TF_L_O1"]
CANDIDATES = {"hybrid": MIX, "pair": ["B8", "F9"]}      # "pair" reads results/brown/decorr_z
# S34: each mix member on its own, to separate "the average is more accurate" from "the average
# is a coherent equilibrium". Averaging K fields lowers coefficient MSE by construction; whether
# it also lowers the geometry error is the question, and the answer is not free.
CANDIDATES.update({f"solo_{t}": [m] for t, m in
                   zip(("B8L", "O1U", "O1L"), MIX)})
# S34: the shipped three plus the long-budget O1 seeds. P34_MIX_COHERENCE screens `o1x5` at
# +0.00081 Cons over the shipped mix for nothing on psi; this is the paired confirmation.
CANDIDATES["o1x4"] = MIX + ["CRC_FEQ_brown_TF_LS_O1_99"]
CANDIDATES["o1x5"] = MIX + ["CRC_FEQ_brown_TF_LS_O1_99", "CRC_FEQ_brown_TF_S_O1_7"]
# S34.18 Arm D: second O1 draw winner is LS_O1_42 (29.6% cut; still below LS_O1_99's 32.5%).
CANDIDATES["o1x6"] = CANDIDATES["o1x5"] + ["CRC_FEQ_brown_TF_LS_O1_42"]
CANDIDATES["o1x5_42for7"] = MIX + ["CRC_FEQ_brown_TF_LS_O1_99", "CRC_FEQ_brown_TF_LS_O1_42"]
# Flux1 mix PASS: B8/L slot → F1P seed 123. Cons already gated in FLUX1_MIX_GATE.json.
CANDIDATES["F1P"] = ["CRC_FEQ_flux1_F1_TH1_PLACE"]
CANDIDATES["F2"] = ["CRC_FEQ_flux1_F2_PLACE"]
CANDIDATES["swap_b8"] = [
    "CRC_FEQ_brown_TF_L_F1P_123",
    MIX[1],
    MIX[2],
    "CRC_FEQ_brown_TF_LS_O1_99",
    "CRC_FEQ_brown_TF_LS_O1_42",
]
CANDIDATES["o1x4_42"] = MIX + ["CRC_FEQ_brown_TF_LS_O1_42"]
# Third O1 seed draw: one-for-one swap of the weaker LS42 member only.
CANDIDATES["o1x5_s5for42"] = MIX + ["CRC_FEQ_brown_TF_LS_O1_99", "CRC_FEQ_brown_TF_LS_O1_5"]
CANDIDATES["o1x5_s29for42"] = MIX + ["CRC_FEQ_brown_TF_LS_O1_99", "CRC_FEQ_brown_TF_LS_O1_29"]
CANDIDATES["o1x5_s101for42"] = MIX + ["CRC_FEQ_brown_TF_LS_O1_99", "CRC_FEQ_brown_TF_LS_O1_101"]
CANDIDATES["o1x5_s314for42"] = MIX + ["CRC_FEQ_brown_TF_LS_O1_99", "CRC_FEQ_brown_TF_LS_O1_314"]
# S34.17 Arm B: swap LS_O1_42 for the late-weighted twin (same seed/budget).
CANDIDATES["o1x5_late42"] = MIX + ["CRC_FEQ_brown_TF_LS_O1_99", "CRC_FEQ_brown_TF_LS_O1_LATE42"]
# S34 Arm A: swap LS_O1_42 for warp-projected twin (lambda_w=1).
CANDIDATES["o1x5_w1for42"] = MIX + ["CRC_FEQ_brown_TF_LS_O1_99", "CRC_FEQ_brown_TF_LS_O1_W1"]
# O1 geometry rebuild (§34.19): seed-42 control/place in CRC_FEQ_olive_geom namespace.
O1G_CONTROL = "CRC_FEQ_olive_geom_O1G_control"
O1G_PLACE = "CRC_FEQ_olive_geom_O1G_place"
O1G_EARLY = "CRC_FEQ_olive_geom_O1G_early"
O1G_EARLY_B2048 = "CRC_FEQ_olive_geom_O1G_early_b2048"
CANDIDATES["O1G_control"] = [O1G_CONTROL]
CANDIDATES["O1G_place"] = [O1G_PLACE]
CANDIDATES["O1G_early"] = [O1G_EARLY]
CANDIDATES["O1G_early_b2048"] = [O1G_EARLY_B2048]
CANDIDATES["o1x5_geom42"] = MIX + ["CRC_FEQ_brown_TF_LS_O1_99", O1G_PLACE]
CANDIDATES["o1x5_early42"] = MIX + ["CRC_FEQ_brown_TF_LS_O1_99", O1G_EARLY]
CANDIDATES["o1x5_early42_b2048"] = MIX + ["CRC_FEQ_brown_TF_LS_O1_99", O1G_EARLY_B2048]
# Public Consistency for the two, so a local delta can be checked against the only delta that
# was ever paid for: hybrid 916155 minus baseline 912438.
PUBLIC_CONS = {"hybrid": 0.975861, "pair": 0.970717}
N_HOLD = 80
GRID = 65

# Public per-scalar Consistency for 916155, from the organizer's detailed_results.html
# (api/submissions/916155/get_details/ -> scoring_result.zip). 874 shots.
PUBLIC = {"R_axis": 0.974809, "Z_axis": 0.980159, "kappa": 0.959625, "tri_top": 0.989477,
          "tri_bot": 0.987845, "volume": 0.980936, "li": 0.958178}


def pooled_r2(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    """The scorer's pooled R2 for one scalar on one fold, plus that fold's GT sd and the RMSE.

    Mirrors `local_score.score_shot`: labels that are nan are dropped, predictions that are nan
    are replaced by the fold mean, so a failed derivation scores as a constant predictor.
    """
    m = np.isfinite(gt)
    if m.sum() < 8:
        return float("nan"), float("nan"), float("nan")
    g, p = gt[m], pred[m]
    p = np.where(np.isfinite(p), p, g.mean())
    ss_tot = float(((g - g.mean()) ** 2).sum())
    if ss_tot <= 0:
        return float("nan"), float(g.std()), float("nan")
    return (1.0 - float(((g - p) ** 2).sum()) / ss_tot,
            float(g.std()), float(np.sqrt(((g - p) ** 2).mean())))


def _holdout_z(name: str, device: torch.device):
    """(pred coeffs, gt coeffs, shot index, pca mean, pca components) on the 80 holdout shots."""
    keys = sample_cache_keys(7000 + N_HOLD, 42, include_thomson=True)[:N_HOLD]
    sup = results_dir() / "brown" / "st_support_cache"
    ok = np.concatenate([(lambda z: z["ok_mean"] & z["ok_gt"])(np.load(sup / f"{k}.npz"))
                         for k in keys])
    shot = np.concatenate([np.full(int(np.load(sup / f"{k}.npz")["ok_mean"].shape[0]), i)
                           for i, k in enumerate(keys)])[ok]

    if name == "pair":
        # Already inferred and ok-masked by psi_decorrelate; no model load needed.
        D = results_dir() / "brown" / "decorr_z"
        z_pred = np.mean([np.load(D / f"{m}.npy").astype(np.float64)
                          for m in CANDIDATES[name]], axis=0)
        ck = torch.load(artifacts_dir() / f"{MIX[0]}.pt", map_location="cpu", weights_only=False)
        _, bundle, _ = load_charcoal(artifacts_dir() / f"{MIX[0]}.pt", device=device)
        _activate(bundle)
        n_raw = int(ck.get("n_raw_features") or len(bundle.feature_mean) // 3)
        _, Zh, _ = build(keys, bundle, "hold", n_raw)
        return (z_pred, Zh.astype(np.float64)[ok], shot,
                np.asarray(ck["pca_mean"], np.float64),
                np.asarray(ck["pca_components"], np.float64))

    zs, zgt, comp, mean = [], None, None, None
    for nm in CANDIDATES[name]:
        model, bundle, ck = load_charcoal(artifacts_dir() / f"{nm}.pt", device=device)
        _activate(bundle)
        n_raw = int(ck.get("n_raw_features") or len(bundle.feature_mean) // 3)
        Xh, Zh, _ = build(keys, bundle, "hold", n_raw)
        z = infer(model, torch.from_numpy(np.ascontiguousarray(Xh)), device)
        zs.append(z.cpu().numpy().astype(np.float64)[ok])
        if zgt is None:
            zgt = Zh.astype(np.float64)[ok]
            mean = np.asarray(ck["pca_mean"], np.float64)
            comp = np.asarray(ck["pca_components"], np.float64)
        del model
        torch.cuda.empty_cache()
        print(f"[foldvar] {nm} inferred", flush=True)
    return np.mean(zs, axis=0), zgt, shot, mean, comp


def per_frame_scalars(stride: int, device: torch.device, name: str = "hybrid"):
    """(gt [N,7], pred [N,7], shot index [N]) through the OFFICIAL derive, cached per candidate."""
    cache = results_dir() / "brown" / (f"P33_FOLDVAR_frames_s{stride}.npz" if name == "hybrid"
                                       else f"P33_FOLDVAR_frames_{name}_s{stride}.npz")
    if cache.is_file():
        d = np.load(cache)
        print(f"[foldvar] cache hit {cache.name} {d['pred'].shape}", flush=True)
        return d["gt"], d["pred"], d["shot"]

    z_pred, zgt, shot, mean, comp = _holdout_z(name, device)
    sel = np.arange(0, len(z_pred), max(1, stride))
    z_pred, zgt, shot = z_pred[sel], zgt[sel], shot[sel]
    R, Z, mask = official_grid()
    grid = (R, Z, mask, mask.astype(np.float64))
    out = []
    for tag, z in (("gt", zgt), ("pred", z_pred)):
        t = time.perf_counter()
        psi = (z @ comp + mean).reshape(-1, GRID, GRID)
        c = np.empty((len(psi), 7))
        for s in range(0, len(psi), 512):
            c[s:s + 512] = build_reference(psi[s:s + 512], *grid)[1]
            print(f"  [{name}/{tag}] {min(s + 512, len(psi)):6d}/{len(psi)}  "
                  f"{time.perf_counter() - t:.0f}s", flush=True)
        out.append(c)
    gt_c, pred_c = out
    np.savez_compressed(cache, gt=gt_c, pred=pred_c, shot=shot)
    return gt_c, pred_c, shot


def consistency(gt: np.ndarray, pred: np.ndarray, fi: np.ndarray) -> tuple[float, np.ndarray]:
    """The scorer's Consistency on one fold: mean of the seven pooled R2, each clamped at 0."""
    each = np.array([max(0.0, pooled_r2(gt[fi, j], pred[fi, j])[0]) for j in range(7)])
    return float(each.mean()), each


def compare(names: list[str], stride: int, boot: int, seed: int, device,
            subset: np.ndarray | None = None, label: str = "") -> dict:
    """Paired bootstrap: both candidates scored on the SAME resampled fold every draw.

    An A/B decision does not need each model's absolute error bar, it needs the error bar on
    the DIFFERENCE -- and most of the fold-spread variance that dominates the absolute numbers
    (S31.6) is common to both models and cancels. That cancellation is the whole instrument.
    """
    a, b = names
    gt, pa, shot = per_frame_scalars(stride, device, a)
    gt_b, pb, shot_b = per_frame_scalars(stride, device, b)
    if not (np.array_equal(shot, shot_b) and np.allclose(np.nan_to_num(gt), np.nan_to_num(gt_b))):
        raise RuntimeError("candidates are not on the same frames; paired test is invalid")

    shots = np.unique(shot) if subset is None else np.asarray(sorted(set(subset)))
    keep = np.flatnonzero(np.isin(shot, shots))
    idx_of = {s: np.flatnonzero(shot == s) for s in shots}
    rng = np.random.default_rng(seed)
    full_a, each_a = consistency(gt, pa, keep)
    full_b, each_b = consistency(gt, pb, keep)
    if label:
        print(f"\n=== {label}: {len(shots)} shots, {len(keep):,} frames")

    ca = np.empty(boot)
    cb = np.empty(boot)
    de = np.empty((boot, 7))
    for k in range(boot):
        fi = np.concatenate([idx_of[s] for s in rng.choice(shots, len(shots), replace=True)])
        ca[k], ea = consistency(gt, pa, fi)
        cb[k], eb = consistency(gt, pb, fi)
        de[k] = ea - eb
    d = ca - cb
    pub_d = PUBLIC_CONS.get(a, np.nan) - PUBLIC_CONS.get(b, np.nan)

    print(f"\n{'':14s} {'full-80':>9} {'boot mean':>10} {'1 sigma':>9}")
    print(f"{a:14s} {full_a:9.5f} {ca.mean():10.5f} {ca.std():9.5f}")
    print(f"{b:14s} {full_b:9.5f} {cb.mean():10.5f} {cb.std():9.5f}")
    unp = float(np.hypot(ca.std(), cb.std()))
    print(f"\nPAIRED  delta Cons ({a} - {b})")
    print(f"  full-80            {full_a - full_b:+.5f}")
    print(f"  bootstrap mean     {d.mean():+.5f}")
    print(f"  paired 1 sigma     {d.std():.5f}      <- the instrument")
    print(f"  unpaired 1 sigma   {unp:.5f}      ({unp / max(d.std(), 1e-12):.1f}x wider)")
    print(f"  5-95               [{np.percentile(d, 5):+.5f}, {np.percentile(d, 95):+.5f}]")
    print(f"  P(better)          {100 * float((d > 0).mean()):.1f}%")
    print(f"  effect size        {d.mean() / max(d.std(), 1e-12):+.2f} sigma")
    print(f"  PUBLIC delta       {pub_d:+.5f}   "
          f"({'inside' if abs(pub_d - d.mean()) < 2 * d.std() else 'OUTSIDE'} the paired 2 sigma)")

    print(f"\nper scalar, paired  ({a} - {b})")
    print(f"{'scalar':8s} {'full':>9} {'paired sd':>10} {'sigma':>7}  {'P(better)':>9}")
    rows = {}
    for j, n in enumerate(CONS_NAMES):
        dj = de[:, j]
        rows[n] = {"full": float(each_a[j] - each_b[j]), "paired_sd": float(dj.std()),
                   "sigma": float(dj.mean() / dj.std()) if dj.std() > 0 else float("nan"),
                   "p_better": float((dj > 0).mean())}
        print(f"{n:8s} {each_a[j] - each_b[j]:+9.5f} {dj.std():10.5f} "
              f"{rows[n]['sigma']:+7.2f}  {100 * rows[n]['p_better']:8.1f}%")
    return {"a": a, "b": b, "full_a": full_a, "full_b": full_b,
            "delta_full": full_a - full_b, "delta_mean": float(d.mean()),
            "paired_sd": float(d.std()), "unpaired_sd": unp,
            "p_better": float((d > 0).mean()), "public_delta": float(pub_d),
            "per_scalar": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stride", type=int, default=1, help="frame stride (1 = all 18,573)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--sizes", default="20,40,80", help="fold sizes in shots to resample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compare", default="",
                    help=f"paired A/B on the same folds, e.g. 'hybrid,pair' "
                         f"(candidates: {sorted(CANDIDATES)})")
    ap.add_argument("--subset", default="all", choices=["all", "worst40", "best40"],
                    help="S33.7: the random 80 overstates the board by the whole coverage "
                         "offset; 'worst40' is the fold that brackets the public score")
    args = ap.parse_args()

    if args.compare:
        names = [s.strip() for s in args.compare.split(",")]
        sub = None
        if args.subset != "all":
            cov = json.loads((results_dir() / "brown" / "P33_COVERAGE.json").read_text())
            order = cov["rank_worst_first"]
            picked = order[:40] if args.subset == "worst40" else order[-40:]
            keys = sample_cache_keys(7000 + N_HOLD, 42, include_thomson=True)[:N_HOLD]
            sub = np.array(sorted(keys.index(k) for k in picked))
        out = compare(names, args.stride, args.boot, args.seed, torch.device("cuda"),
                      subset=sub, label=args.subset)
        stride_tag = "" if args.stride == 1 else f"_s{args.stride}"
        dest = (results_dir() / "brown"
                / f"P33_PAIRED_{names[0]}_vs_{names[1]}"
                  f"{'' if args.subset == 'all' else '_' + args.subset}{stride_tag}.json")
        out["subset"] = args.subset
        dest.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {dest}")
        return 0

    gt, pred, shot = per_frame_scalars(args.stride, torch.device("cuda"))
    shots = np.unique(shot)
    idx_of = {s: np.flatnonzero(shot == s) for s in shots}
    rng = np.random.default_rng(args.seed)

    full = {n: pooled_r2(gt[:, j], pred[:, j]) for j, n in enumerate(CONS_NAMES)}
    print(f"\n[foldvar] {len(gt):,} frames, {len(shots)} shots\n")

    # ---------------------------------------------------------------- bootstrap
    res: dict[int, dict[str, np.ndarray]] = {}
    for size in [int(v) for v in args.sizes.split(",")]:
        r2 = np.full((args.boot, 7), np.nan)
        sd = np.full((args.boot, 7), np.nan)
        for b in range(args.boot):
            pick = rng.choice(shots, size=size, replace=True)
            fi = np.concatenate([idx_of[s] for s in pick])
            for j in range(7):
                r2[b, j], sd[b, j], _ = pooled_r2(gt[fi, j], pred[fi, j])
        res[size] = {"r2": r2, "sd": sd}

    big = res[max(res)]
    print(f"{'scalar':8s} {'full-80':>8} {'RMSE':>9} | {'boot mean':>9} {'sd':>7} "
          f"{'p5':>8} {'p95':>8} | {'public':>8} {'pctile':>7}")
    rows = {}
    for j, n in enumerate(CONS_NAMES):
        r2b = big["r2"][:, j]
        pct = 100.0 * float((r2b < PUBLIC[n]).mean())
        rows[n] = {"full": full[n][0], "gt_sd": full[n][1], "rmse": full[n][2],
                   "boot_mean": float(np.nanmean(r2b)), "boot_sd": float(np.nanstd(r2b)),
                   "p5": float(np.nanpercentile(r2b, 5)), "p95": float(np.nanpercentile(r2b, 95)),
                   "public": PUBLIC[n], "public_pctile": pct}
        print(f"{n:8s} {full[n][0]:8.5f} {full[n][2]:9.3e} | {np.nanmean(r2b):9.5f} "
              f"{np.nanstd(r2b):7.5f} {np.nanpercentile(r2b, 5):8.5f} "
              f"{np.nanpercentile(r2b, 95):8.5f} | {PUBLIC[n]:8.5f} {pct:6.1f}%")

    # ------------------------------------------- S31.6: does R2 track the fold's GT spread?
    print(f"\n[S31.6 mechanism] pooled R2 vs that fold's GT spread, {args.boot} bootstrap folds "
          f"of {max(res)} shots. z = (public - full) / boot sd: |z| < 2 means a fold draw of "
          f"this size explains the public move on its own.")
    print(f"{'scalar':8s} {'corr(R2,sd)':>11} {'public-full':>11} {'z':>7}  reading")
    for j, n in enumerate(CONS_NAMES):
        r2b, sdb = big["r2"][:, j], big["sd"][:, j]
        m = np.isfinite(r2b) & np.isfinite(sdb)
        c = (float(np.corrcoef(r2b[m], sdb[m])[0, 1])
             if m.sum() > 8 and sdb[m].std() > 0 else float("nan"))
        move = PUBLIC[n] - full[n][0]
        s = rows[n]["boot_sd"]
        z = move / s if s > 0 else float("nan")
        rows[n].update({"corr_r2_sd": c, "public_minus_full": move, "z": z})
        print(f"{n:8s} {c:+11.3f} {move:+11.5f} {z:+7.2f}  "
              f"{'fold draw explains it' if abs(z) < 2 else 'REAL -- beyond fold noise'}")

    out = {"frames": int(len(gt)), "shots": int(len(shots)), "boot": args.boot,
           "stride": args.stride, "mix": MIX, "per_scalar": rows,
           "by_size": {str(k): {"r2_sd": np.nanstd(v["r2"], axis=0).tolist()}
                       for k, v in res.items()}}
    dest = results_dir() / "brown" / "P33_FOLDVAR.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
