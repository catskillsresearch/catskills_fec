#!/usr/bin/env python3
"""Train / score ``CRC_FEQ_charcoal`` (teal's PCA MLP, trained in the scorer's norm).

The split logic is copied from the teal runner on purpose: same ``sample_cache_keys`` call, same
seed, same eval-shot count, so a charcoal probe and a teal ladder unit see the identical train and
holdout shots and the comparison is like-for-like.

Usage:
  poetry run python -m catskills.run_crc_feq_pinn_charcoal --from-feature-cache --max-train-shots 128
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from catskills.feature_cache import sample_cache_keys
from catskills.lab_paths import artifacts_dir, results_dir
from catskills.local_benchmark import official_score
from catskills.run_lock import ConcurrentRunError, RunLock
from catskills.pinn_charcoal import (
    MODEL_ID,
    artifact_name,
    find_checkpoint,
    load_charcoal,
    predict_row,
    require_cuda,
    train_charcoal,
)

CODE_URL = "https://github.com/catskillsresearch/fusion-equilibrium-challenge"


def main() -> int:
    ap = argparse.ArgumentParser(description=f"End-to-end {MODEL_ID} train + score")
    ap.add_argument("--from-feature-cache", action="store_true")
    ap.add_argument("--max-train-shots", type=int, default=128)
    ap.add_argument("--eval-shots", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init-seed", type=int, default=None,
                    help="weight init + batch order only; leaves the data split on --seed so "
                         "replications stay comparable on the same holdout")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-frames", type=int, default=32768)
    ap.add_argument("--coeff-targets", action="store_true",
                    help="read targets from the coefficient cache (maroon M0) instead of the grid "
                         "cache; 374 MB instead of 26.4 GB, which is what lets --max-frames exceed "
                         "~250k on a 12 GB card")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-pca", type=int, default=60)
    ap.add_argument("--no-thomson", action="store_true")
    ap.add_argument("--thomson-zgrid", action="store_true",
                    help="Brown B7: Te/ne on a fixed Z grid. Writes cache suffix thz, never th1.")
    ap.add_argument("--thomson-thzr", action="store_true",
                    help="Brown B11: core on Z-grid, tangential on R-grid. Suffix thzr, never th1/thz.")
    ap.add_argument("--vac-in", action="store_true",
                    help="B13: concat 4 PCA coeffs of analytic ψ_vac onto X. Does not write a new cache.")

    ap.add_argument("--temporal-window", type=int, default=5,
                    help="odd window for the moving-average / difference channels (0 or 1 = off)")
    ap.add_argument("--lambda-qb", type=float, default=1.0)
    ap.add_argument("--lambda-geom", type=float, default=-1.0,
                    help="raw geometry weight; <0 means derive it from --geom-emphasis (0 = teal)")
    ap.add_argument("--geom-emphasis", type=float, default=30.0,
                    help="geometry weight in score units: how many times its true S weight")
    ap.add_argument("--lambda-h1", type=float, default=0.1)
    ap.add_argument("--lambda-disp", type=float, default=0.1)
    ap.add_argument("--lambda-coeff-white", type=float, default=0.0,
                    help="weight of the per-mode-rescaled coefficient term (0 = off, pre-C5 behaviour)")
    ap.add_argument("--coeff-white-power", type=float, default=1.0,
                    help="divisor is sd**power; 1.0 = full whitening, 0.5 = tempered")
    ap.add_argument("--coeff-white-clamp", type=int, default=15,
                    help="floor the per-mode divisor at this mode's sd, so the noise tail cannot explode")
    ap.add_argument("--edge-weight", choices=["off", "grad", "band"], default="off",
                    help="C8: reweight L_data in (R,Z). off = uniform MSE (every run to date). "
                         "grad = |∇ψ_gt|^power. band = donut on normalized ψ. Weights are "
                         "mean-normalized so --lambda-coeff-white stays calibrated.")
    ap.add_argument("--edge-mix", type=float, default=0.6,
                    help="convex weight on the edge map vs uniform 1: w = (1-mix) + mix*(s/mean(s))")
    ap.add_argument("--edge-power", type=float, default=1.0,
                    help="with --edge-weight grad, raise |∇ψ| to this power before normalizing")
    ap.add_argument("--edge-band-lo", type=float, default=0.70,
                    help="with --edge-weight band, inner edge of the donut on normalized ψ")
    ap.add_argument("--edge-band-hi", type=float, default=1.05,
                    help="with --edge-weight band, outer edge of the donut on normalized ψ")
    ap.add_argument("--val-geom-frames", type=int, default=0,
                    help="val frames given geometry targets so the selector sees Consistency with the "
                         "geometry loss off (0 = pre-C5 behaviour, select on psi MSE alone)")
    ap.add_argument("--geom-sd-chunk", type=int, default=256,
                    help="SoftDerive frames per backward chunk (dualshot geom)")
    ap.add_argument("--probe-epochs", type=int, default=0,
                    help="after resume, run this many epochs then keep .partial.pt and exit")
    ap.add_argument("--geom-frac", type=float, default=1.0,
                    help="fraction of each batch used for the geometry loss")
    ap.add_argument("--geom-warmup", type=int, default=10,
                    help="epochs to train without the geometry term first")
    ap.add_argument("--geom-huber", type=float, default=3.0,
                    help="Huber knee for the geometry residual, in sigmas (0 = plain squared)")
    ap.add_argument("--geom-weights", choices=["uniform", "deficit", "stack", "placement",
                                               "placement_early", "explicit"],
                    default="uniform",
                    help="P6b: 'deficit' weights the seven scalars by their remaining PUBLIC gap. "
                         "'stack' (Brown) puts the mass on kappa, Z_axis, volume — the F15/F32/F57 knobs. "
                         "'placement' (O1 geom rebuild) promotes kappa, R_axis, Z_axis only. "
                         "'placement_early' adds protected ℓ_i and a 0.5 floor elsewhere")
    ap.add_argument("--geom-weight-target", type=float, default=0.95,
                    help="per-scalar R2 that counts as done when deriving --geom-weights deficit")
    ap.add_argument("--geom-weight-floor", type=float, default=0.25,
                    help="share a finished scalar keeps, so it cannot drift back down")
    ap.add_argument("--geom-kappa-mult", type=float, default=1.0,
                    help="multiply the kappa geometry weight, then renormalize to mean 1")
    ap.add_argument("--geom-raxis-mult", type=float, default=1.0,
                    help="multiply the R_axis geometry weight, then renormalize to mean 1")
    ap.add_argument("--geom-zaxis-mult", type=float, default=1.0,
                    help="multiply the Z_axis geometry weight, then renormalize to mean 1")
    ap.add_argument("--geom-weight-vec", type=str, default="",
                    help="seven comma-separated weights in R_axis,Z_axis,kappa,tri_top,"
                         "tri_bot,volume,li order. Used when --geom-weights explicit. "
                         "Not renormalized.")
    ap.add_argument("--lr-eta-min", type=float, default=0.0,
                    help="cosine floor. 0 keeps the historical schedule, which ends at 0")
    ap.add_argument("--inherit-feature-stats", action="store_true",
                    help="warm-start uses the checkpoint's feature mean and std")
    ap.add_argument("--lcfs-emphasis", type=float, default=0.0,
                    help="P6a: weight of the soft symmetric-Hausdorff D_LCFS term, in score units "
                         "(0 = off, which reproduces P4)")
    ap.add_argument("--lambda-lcfs", type=float, default=-1.0,
                    help="raw D_LCFS weight; <0 means derive it from --lcfs-emphasis")
    ap.add_argument("--lcfs-cap", type=float, default=2.0,
                    help="training clamp on the soft D_LCFS; the scorer's own clamp is 1.0")
    ap.add_argument("--geom-emphasis-final", type=float, default=1.0,
                    help="P6c: decay both geometry weights to this multiple over the tail of the "
                         "run, letting the data term finish (1.0 = no anneal, as P4)")
    ap.add_argument("--geom-anneal-tail", type=float, default=0.25,
                    help="fraction of epochs over which --geom-emphasis-final is reached")
    ap.add_argument("--qb-normalize", action="store_true",
                    help="P6d: divide q95/betaN residuals by their own corpus std, so L_qb is "
                         "exactly the R2_qb deficit and --qb-emphasis reads in score units")
    ap.add_argument("--qb-emphasis", type=float, default=1.0,
                    help="with --qb-normalize and --lambda-qb -1, weight of the scalar heads in "
                         "score units")
    ap.add_argument("--tau-hi", type=float, default=0.05)
    ap.add_argument("--tau-lo", type=float, default=0.005)
    ap.add_argument("--tau-li-hi", type=float, default=0.08)
    ap.add_argument("--tau-li-lo", type=float, default=0.02)
    ap.add_argument("--anneal-frac", type=float, default=0.5)
    ap.add_argument("--band-lo", type=float, default=0.85)
    ap.add_argument("--band-hi", type=float, default=1.05)

    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--early-stop-patience", type=int, default=8)
    ap.add_argument("--kill-r2psi", type=float, default=0.0,
                    help="abort if val_R2ψ falls below this after --kill-r2psi-after (0 = off)")
    ap.add_argument("--kill-r2psi-after", type=int, default=8,
                    help="first epoch the --kill-r2psi floor is enforced")
    ap.add_argument("--init-checkpoint", type=Path, default=None)
    ap.add_argument("--optimizer", type=str, default="adamw",
                    choices=("adam", "adamw", "sgd", "radam", "nadam"))
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--hidden", type=str, default="256,128")
    ap.add_argument("--activation", type=str, default="tanh",
                    choices=("tanh", "gelu", "silu", "relu"))
    ap.add_argument("--feature-clip", type=float, default=0.0,
                    help="R2: clamp standardized inputs to +-this many sigma (0 = off). The clamp "
                         "is saved in the checkpoint and reapplied at inference")
    # R1. "mlp" is every model shipped to date and leaves the three trunk flags inert.
    ap.add_argument("--arch", type=str, default="mlp",
                    choices=("mlp", "resnet", "dualctx", "grand", "seq", "dualseq", "shotseq", "dualshot"))
    ap.add_argument("--trunk-width", type=int, default=512)
    ap.add_argument("--trunk-blocks", type=int, default=4)
    ap.add_argument("--trunk-dropout", type=float, default=0.05)
    # dualctx only: the second encoder that reads the temporal context block. Its output enters the
    # main trunk through a scalar gate initialised to ZERO, so the run starts as a plain resnet on the
    # present frame and the optimizer has to open the gate for context to matter at all.
    ap.add_argument("--ctx-width", type=int, default=512)
    ap.add_argument("--ctx-blocks", type=int, default=3)
    # seq only (D1): GRU size over the stacked raw-frame window (--temporal-mode stack).
    ap.add_argument("--seq-hidden", type=int, default=256)
    ap.add_argument("--seq-layers", type=int, default=2)
    ap.add_argument("--seq-window", type=int, default=0,
                    help="D2: GRU stack width; 0 = same as --temporal-window. Dualctx smoothdiff "
                         "always uses --temporal-window so a C7b warm-start stays aligned.")
    ap.add_argument("--freeze-dualctx", action="store_true",
                    help="D2a: train only gru/seq_proj/gate_seq; freeze the C7b-named modules")
    ap.add_argument("--shot-bidirectional", action=argparse.BooleanOptionalAction, default=True,
                    help="forest: bidirectional GRU over the shot (default on)")
    ap.add_argument("--shot-mlp-blocks", type=int, default=2,
                    help="forest: residual MLP blocks on h_t before the heads")
    ap.add_argument("--token-budget", type=int, default=1024,
                    help="forest: max frames per optimizer step (replaces --batch-size)")
    # R6: the two stabilizers R5 lacked when it diverged at 17M parameters.
    ap.add_argument("--grad-clip", type=float, default=0.0,
                    help="clip total gradient norm before each step (0 = off)")
    ap.add_argument("--lr-warmup", type=int, default=0,
                    help="linear lr ramp over this many epochs before the cosine (0 = plain cosine)")
    ap.add_argument("--temporal-mode",
                    choices=["avgdiff", "smoothdiff", "stack", "smoothdiff+stack"],
                    default="avgdiff",
                    help="how the temporal blocks are built: avgdiff differences the RAW signal "
                         "one frame apart (what R2a/R4 ran), smoothdiff differences the smoothed "
                         "signal over the window (far fewer outlier z-scores), stack lays out the "
                         "raw frames [x_{t-h}..x_{t+h}] themselves for the seq trunk (D1)")
    ap.add_argument("--resume", type=str, default="",
                    help="continue a partial checkpoint: a path, or 'auto' for this tag's own "
                         "<out>.partial.pt. Restores optimizer moments, cosine position, RNG and "
                         "selection state; refuses if the config no longer matches")
    ap.add_argument("--lbfgs-steps", type=int, default=15)
    ap.add_argument("--lbfgs-max-frames", type=int, default=8192)
    ap.add_argument("--smooth-window", type=int, default=0,
                    help="probe 0c output smoothing at inference (0 = off)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=25,
                    help="write a loadable partial checkpoint every N epochs (0 = off). "
                         "A 5h run that segfaults at epoch 370 otherwise leaves nothing.")
    ap.add_argument("--loss-report", action="store_true",
                    help="print per-term loss magnitudes each epoch (for weight calibration)")
    ap.add_argument("--geom-ramp", type=int, default=0,
                    help="Olive: linear λ_geom ramp over this many epochs after --geom-warmup "
                         "(0 = step on, which is every earlier run)")
    ap.add_argument("--gradnorm", action="store_true",
                    help="Olive: GradNorm on (L_data, L_stack) at h_fuse")
    ap.add_argument("--gradnorm-alpha", type=float, default=0.5)
    ap.add_argument("--gradnorm-w-from", type=int, default=10,
                    help="first epoch that Adam-updates w_g (unit weights until then)")
    ap.add_argument("--pcgrad", action="store_true",
                    help="Olive O1.1: PCGrad instead of GradNorm")
    ap.add_argument("--stack-li", action="store_true",
                    help="Olive: promote ℓ_i in the stack focus. Off = B8's three knobs.")
    ap.add_argument("--lp-floor", type=float, default=0.0,
                    help="Olive: SoftDerive ℓ_p floor in the ℓ_i denominator (0 = eps only)")
    ap.add_argument("--kill-cons", type=float, default=0.0,
                    help="abort if val Cons falls below this after --kill-cons-after (0 = off)")
    ap.add_argument("--kill-cons-after", type=int, default=8)
    ap.add_argument("--adaptive-warmup", action="store_true",
                    help="Olive O1b: hold λ_geom at 0 until val R²ψ and Cons "
                         "clear --warmup-r2psi / --warmup-cons, then ramp. "
                         "Kill if still unhealthy at --warmup-cap.")
    ap.add_argument("--warmup-cap", type=int, default=25,
                    help="last data-only epoch allowed under --adaptive-warmup")
    ap.add_argument("--warmup-r2psi", type=float, default=0.999)
    ap.add_argument("--warmup-cons", type=float, default=0.96)
    ap.add_argument("--psi-collapse-r2psi", type=float, default=0.0,
                    help="after the geometry gate, abort if val R²ψ stays below this for "
                         "--psi-collapse-streak consecutive epochs (0 = off)")
    ap.add_argument("--psi-collapse-streak", type=int, default=3)
    ap.add_argument("--nondim", action="store_true",
                    help="Red: non-dimensionalise raw X (I/Ip, Bt/B0, Te keV, "
                         "ne a²/|Ip|) after the feature cache, before the window.")
    ap.add_argument("--gray", action="store_true",
                    help="Gray: collapse magnetics to 4 shared roles "
                         "(Ip, sol, TF, E_shape) after the feature cache.")
    ap.add_argument("--sky", action="store_true",
                    help="Sky: replace 21 named coils with normalized "
                         "ψ_vac grid + Ip + TF (not B13 4-PCA).")
    ap.add_argument("--sky-tokens", action="store_true",
                    help="Sky S1: replace 21 names with (r,z,I/|Ip|) per PF coil.")
    ap.add_argument("--yellow", action="store_true",
                    help="Yellow: S1 tokens + ψ_vac/B_vac at Thomson Z (not S2 grid).")
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--train-keys-json", type=Path, default=None,
                    help="Exact ordered allowlist {keys, keys_sha256}. Rejects "
                         "duplicates, missing cache keys, canonical-80 leak, hash mismatch.")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--allow-concurrent", action="store_true",
                    help="start even if another run holds the lock (they will share the GPU)")
    args = ap.parse_args()

    try:
        lock = RunLock(MODEL_ID, tag=args.tag or "run",
                       allow_concurrent=args.allow_concurrent).__enter__()
    except ConcurrentRunError as exc:
        print(f"\n[{MODEL_ID}] refusing to start:\n{exc}\n", file=sys.stderr)
        return 2

    try:
        return _run(args)
    finally:
        lock.__exit__(None, None, None)


def _run(args: argparse.Namespace) -> int:
    include_thomson = not args.no_thomson
    if getattr(args, "thomson_thzr", False):
        os.environ["CRC_FEQ_THOMSON_LAYOUT"] = "thzr"
        print("[CRC_FEQ] thomson layout=thzr  cache suffix=thzr (not th1/thz)", flush=True)
    if getattr(args, "vac_in", False):
        os.environ["CRC_FEQ_VAC_IN"] = "1"
        print("[CRC_FEQ] vac-in=4  Green ψ_vac PCA appended at load (thzr cache unchanged)", flush=True)
    if getattr(args, "yellow", False):
        os.environ["CRC_FEQ_YELLOW"] = "1"
        print("[CRC_FEQ] yellow=1  S1 tokens + vac-aug thz (not S2 grid)", flush=True)
    if getattr(args, "thomson_zgrid", False) and not getattr(args, "vac_in", False):
        os.environ["CRC_FEQ_THOMSON_LAYOUT"] = "zgrid"
        print("[CRC_FEQ] thomson layout=zgrid  cache suffix=thz (not th1)", flush=True)
    device = require_cuda()
    import torch

    print("=" * 64)
    print(f"  Model:       {MODEL_ID}")
    print(f"  Device:      {device} ({torch.cuda.get_device_name(0)})")
    if args.arch == "resnet":
        trunk = (f"resnet width={args.trunk_width} blocks={args.trunk_blocks} "
                 f"dropout={args.trunk_dropout}")
    elif args.arch == "dualctx":
        trunk = (f"dualctx main={args.trunk_width}x{args.trunk_blocks} "
                 f"ctx={args.ctx_width}x{args.ctx_blocks} dropout={args.trunk_dropout} gate=0")
    elif args.arch == "grand":
        trunk = (f"grand main={args.trunk_width}x{args.trunk_blocks} "
                 f"ctx={args.ctx_width}x{args.ctx_blocks} dropout={args.trunk_dropout} "
                 f"qb_mlp=3 nbi={os.environ.get('CRC_FEQ_NBI_COLS', '0')}")
    elif args.arch == "seq":
        trunk = (f"seq gru={args.seq_hidden}x{args.seq_layers} "
                 f"dropout={args.trunk_dropout} (window from --temporal-window, mode=stack)")
    elif args.arch == "dualseq":
        trunk = (f"dualseq gru={args.seq_hidden}x{args.seq_layers} "
                 f"seq_window={args.seq_window or args.temporal_window} "
                 f"freeze={int(args.freeze_dualctx)} gate_seq=0")
    elif args.arch == "shotseq":
        trunk = (f"shotseq gru={args.seq_hidden}x{args.seq_layers} "
                 f"{'bi' if args.shot_bidirectional else 'uni'} "
                 f"mlp={args.shot_mlp_blocks} token_budget={args.token_budget}")
    elif args.arch == "dualshot":
        trunk = (f"dualshot gru={args.seq_hidden}x{args.seq_layers} "
                 f"{'bi' if args.shot_bidirectional else 'uni'} "
                 f"freeze={int(args.freeze_dualctx)} token_budget={args.token_budget} gate_seq=0")
    else:
        trunk = f"mlp hidden={args.hidden}"
    print(f"  n_pca:       {args.n_pca}   trunk={trunk} act={args.activation}")
    print(f"  temporal:    window={args.temporal_window} mode={args.temporal_mode}")
    print(f"  stability:   grad_clip={args.grad_clip} lr_warmup={args.lr_warmup} lr={args.lr}")
    print(f"  lambdas:     geom={args.lambda_geom} (emphasis={args.geom_emphasis}) "
          f"h1={args.lambda_h1} disp={args.lambda_disp} qb={args.lambda_qb}")
    if args.lambda_coeff_white > 0:
        print(f"  coeff white: lambda={args.lambda_coeff_white} power={args.coeff_white_power} "
              f"clamp=mode{args.coeff_white_clamp}")
    if args.edge_weight != "off":
        print(f"  edge weight: {args.edge_weight} mix={args.edge_mix} "
              f"power={args.edge_power} band=[{args.edge_band_lo},{args.edge_band_hi}]")
    print(f"  tau:         {args.tau_hi} -> {args.tau_lo}  (tau_li {args.tau_li_hi} -> "
          f"{args.tau_li_lo}, over {args.anneal_frac:.0%} of run)")
    print("=" * 64)

    if not args.from_feature_cache:
        raise SystemExit(f"{MODEL_ID} runner requires --from-feature-cache")

    n_train = args.max_train_shots
    holdout_keys: list[str] = []
    if args.train_keys_json is not None:
        from catskills.d3d_analog_followups import load_train_keys_json
        cache_keys = load_train_keys_json(args.train_keys_json, seed=42,
                                          include_thomson=include_thomson)
        print(f"Feature-cache split: train={len(cache_keys)}  holdout=0 "
              f"(train-keys-json {args.train_keys_json})", flush=True)
        if not args.skip_score:
            print("[CRC_FEQ] train-keys-json implies --skip-score (exact allowlist)",
                  flush=True)
            args.skip_score = True
    elif not args.skip_score:
        n_pool = n_train + max(args.eval_shots, 1)
        keys = sample_cache_keys(n_pool, args.seed, include_thomson=include_thomson)
        n_eval = min(args.eval_shots, max(1, len(keys) // 5))
        if n_eval >= len(keys):
            n_eval = max(1, len(keys) // 10)
        holdout_keys = keys[:n_eval]
        cache_keys = keys[n_eval:]
        print(f"Feature-cache split: train={len(cache_keys)}  holdout={len(holdout_keys)}",
              flush=True)
    else:
        cache_keys = sample_cache_keys(n_train, args.seed, include_thomson=include_thomson)

    tag = args.tag or "run"
    ckpt_path = artifacts_dir() / f"{artifact_name(tag)}.pt"

    resume_path = None
    if args.resume:
        # ``auto`` goes through the resolver so a run started under an earlier revision's prefix is
        # still restartable after this one claims a new namespace -- a partial that exists but cannot
        # be found by name is the same lost GPU-hours as no partial at all.
        resume_path = (find_checkpoint(tag, partial=True)
                       if args.resume == "auto" else Path(args.resume))
        if not resume_path.is_file():
            raise SystemExit(f"--resume {resume_path} not found")
        print(f"  resume:      {resume_path.name}")

    print(f"\n=== Training {MODEL_ID} ===", flush=True)
    t0 = time.perf_counter()
    train_charcoal(
        cache_keys=cache_keys,
        coeff_targets=args.coeff_targets,
        include_thomson=include_thomson,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_frames=args.max_frames,
        n_pca=args.n_pca,
        temporal_window=args.temporal_window,
        lambda_qb=args.lambda_qb,
        lambda_geom=args.lambda_geom,
        geom_emphasis=args.geom_emphasis,
        lambda_h1=args.lambda_h1,
        lambda_disp=args.lambda_disp,
        lambda_coeff_white=args.lambda_coeff_white,
        coeff_white_power=args.coeff_white_power,
        coeff_white_clamp=args.coeff_white_clamp,
        edge_weight=args.edge_weight,
        edge_mix=args.edge_mix,
        edge_power=args.edge_power,
        edge_band_lo=args.edge_band_lo,
        edge_band_hi=args.edge_band_hi,
        val_geom_frames=args.val_geom_frames,
        geom_frac=args.geom_frac,
        geom_warmup=args.geom_warmup,
        geom_sd_chunk=args.geom_sd_chunk,
        probe_epochs=args.probe_epochs,
        geom_huber=args.geom_huber,
        geom_weights=args.geom_weights,
        geom_weight_target=args.geom_weight_target,
        geom_weight_floor=args.geom_weight_floor,
        geom_kappa_mult=args.geom_kappa_mult,
        geom_raxis_mult=args.geom_raxis_mult,
        geom_zaxis_mult=args.geom_zaxis_mult,
        geom_weight_vec=tuple(
            float(x) for x in args.geom_weight_vec.split(",") if x.strip()
        ) or None,
        lr_eta_min=args.lr_eta_min,
        inherit_feature_stats=bool(args.inherit_feature_stats),
        lcfs_emphasis=args.lcfs_emphasis,
        lambda_lcfs=args.lambda_lcfs,
        lcfs_cap=args.lcfs_cap,
        geom_emphasis_final=args.geom_emphasis_final,
        geom_anneal_tail=args.geom_anneal_tail,
        qb_normalize=args.qb_normalize,
        qb_emphasis=args.qb_emphasis,
        tau_hi=args.tau_hi,
        tau_lo=args.tau_lo,
        tau_li_hi=args.tau_li_hi,
        tau_li_lo=args.tau_li_lo,
        anneal_frac=args.anneal_frac,
        band_lo=args.band_lo,
        band_hi=args.band_hi,
        val_fraction=args.val_fraction,
        early_stop_patience=args.early_stop_patience,
        kill_r2psi=args.kill_r2psi,
        kill_r2psi_after=args.kill_r2psi_after,
        init_checkpoint=args.init_checkpoint,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        hidden=args.hidden,
        activation=args.activation,
        arch=args.arch,
        feature_clip=args.feature_clip,
        trunk_width=args.trunk_width,
        ctx_width=args.ctx_width,
        ctx_blocks=args.ctx_blocks,
        seq_hidden=args.seq_hidden,
        seq_layers=args.seq_layers,
        seq_window=args.seq_window,
        freeze_dualctx=args.freeze_dualctx,
        shot_bidirectional=args.shot_bidirectional,
        shot_mlp_blocks=args.shot_mlp_blocks,
        token_budget=args.token_budget,
        trunk_blocks=args.trunk_blocks,
        trunk_dropout=args.trunk_dropout,
        grad_clip=args.grad_clip,
        lr_warmup=args.lr_warmup,
        temporal_mode=args.temporal_mode,
        resume=resume_path,
        lbfgs_steps=args.lbfgs_steps,
        lbfgs_max_frames=args.lbfgs_max_frames,
        seed=args.seed,
        init_seed=args.init_seed,
        output=ckpt_path,
        log_every=args.log_every,
        ckpt_every=args.ckpt_every,
        loss_report=args.loss_report,
        geom_ramp=args.geom_ramp,
        gradnorm=args.gradnorm,
        gradnorm_alpha=args.gradnorm_alpha,
        gradnorm_w_from=args.gradnorm_w_from,
        pcgrad=args.pcgrad,
        stack_li=args.stack_li,
        lp_floor=args.lp_floor,
        kill_cons=args.kill_cons,
        kill_cons_after=args.kill_cons_after,
        adaptive_warmup=bool(args.adaptive_warmup),
        warmup_cap=int(args.warmup_cap),
        warmup_r2psi=float(args.warmup_r2psi),
        warmup_cons=float(args.warmup_cons),
        psi_collapse_r2psi=float(args.psi_collapse_r2psi),
        psi_collapse_streak=int(args.psi_collapse_streak),
        nondim=bool(args.nondim),
        gray=bool(args.gray),
        sky=bool(args.sky),
        sky_tokens=bool(args.sky_tokens),
        yellow=bool(args.yellow),
    )
    train_s = time.perf_counter() - t0
    if args.skip_score or int(args.probe_epochs) > 0:
        print(f"\nDone (scoring skipped). train_s={train_s:.1f}s", flush=True)
        return 0
    model, bundle, ckpt = load_charcoal(ckpt_path)

    from tqdm.auto import tqdm

    from catskills.warm_feature_cache import row_from_parquet, train_parquet_dir
    from experiments import load_shot_from_hf_row

    pq_dir = train_parquet_dir()
    print("\n=== Holdout parquet (post-train) ===", flush=True)
    holdout_entries = []
    for key in tqdm(holdout_keys, desc="holdout parquet", unit="shot", leave=True):
        row = row_from_parquet(pq_dir / f"{key}.parquet")
        holdout_entries.append(
            {"row": row, "shot": load_shot_from_hf_row(row), "cache_key": key}
        )

    print("\n=== Inference on holdout ===", flush=True)
    if args.nondim:
        os.environ["CRC_FEQ_NONDIM"] = "1"
    if args.gray:
        os.environ["CRC_FEQ_GRAY"] = "1"
    if args.sky:
        os.environ["CRC_FEQ_SKY"] = "1"
    if args.sky_tokens:
        os.environ["CRC_FEQ_SKY_TOKENS"] = "1"
    if args.yellow:
        os.environ["CRC_FEQ_YELLOW"] = "1"
    t1 = time.perf_counter()
    preds = [
        predict_row(bundle, e["row"], "DIII-D", model=model, smooth_window=args.smooth_window)
        for e in holdout_entries
    ]
    infer_s = time.perf_counter() - t1

    print("\n=== Official-metric score on holdout ===", flush=True)
    # Take the resolved lambdas from the checkpoint, not from argv: --lambda-geom defaults to the
    # sentinel -1.0 meaning "derive me from geom_emphasis and Var(psi)", and recording the sentinel
    # made every emphasis run look identically configured.
    tp = ckpt.get("training_params", {})
    scores = official_score(holdout_entries, preds)
    scores.update(
        {
            "train_s": train_s,
            "infer_holdout_s": infer_s,
            "n_train": len(cache_keys),
            "n_eval": len(holdout_entries),
            "method_name": MODEL_ID,
            "code_url": CODE_URL,
            "model_version": bundle.model_version,
            "n_pca": bundle.n_pca,
            "temporal_window": bundle.temporal_window,
            "lambda_geom": tp.get("lambda_geom", args.lambda_geom),
            "geom_emphasis": tp.get("geom_emphasis", args.geom_emphasis),
            "seed": args.seed,
            "init_seed": tp.get("init_seed", args.seed),
            "lambda_h1": args.lambda_h1,
            "lambda_disp": args.lambda_disp,
            "lambda_coeff_white": args.lambda_coeff_white,
            "coeff_white_power": args.coeff_white_power,
            "coeff_white_clamp": args.coeff_white_clamp,
            "edge_weight": args.edge_weight,
            "edge_mix": args.edge_mix,
            "edge_power": args.edge_power,
            "edge_band_lo": args.edge_band_lo,
            "edge_band_hi": args.edge_band_hi,
            # Which objective the reported weights were SELECTED on, not just what they were trained
            # on. C5 established these are different questions: with this at 0 the selector maximises
            # -val_mse and best_selection prints -0.0000, and with it set it maximises the S proxy.
            # Comparing two scores without knowing this is comparing two different experiments.
            "val_geom_frames": args.val_geom_frames,
            "geom_warmup": args.geom_warmup,
            "grad_clip": args.grad_clip,
            "lr_warmup": args.lr_warmup,
            "geom_weights": tp.get("geom_weights", args.geom_weights),
            "geom_weight_vec": tp.get("geom_weight_vec"),
            "lambda_lcfs": tp.get("lambda_lcfs", args.lambda_lcfs),
            "lcfs_emphasis": tp.get("lcfs_emphasis", args.lcfs_emphasis),
            "val_fraction": args.val_fraction,
            "val_split": "shot",
            "smooth_window": args.smooth_window,
            "epochs": args.epochs,
            "tag": tag,
            "best_epoch": tp.get("best_epoch"),
            "best_selection": tp.get("best_selection"),
            "gate_seq": tp.get("gate_seq"),
            "arch": tp.get("arch", args.arch),
        }
    )
    score_path = results_dir() / f"{MODEL_ID}_{tag}_score.json"
    score_path.write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    # Second copy under results/charcoal/<tag>.json. Despite the name this is a fixed HANDOFF INBOX
    # shared by every revision, not a namespace: the ladder picks the run up here and publishes it into
    # its own color's directory (results/maroon, results/purple), which is what each board globs. It
    # keeps the literal name so a ladder and the runner it spawned cannot disagree about the drop point.
    phase_dir = results_dir() / "charcoal"
    phase_dir.mkdir(parents=True, exist_ok=True)
    (phase_dir / f"{tag}.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    print(f"  S            = {scores['S']:.4f}")
    print(f"  R2_psi       = {scores['r2_psi']:.4f}")
    print(f"  R2_qb        = {scores['r2_qb']:.4f}")
    print(f"  1-D_LCFS     = {scores['one_minus_dlcfs']:.4f}")
    print(f"  Consistency  = {scores['consistency']:.4f}")
    r2c = scores.get("r2_cons_each") or {}
    if r2c:
        print("  r2_cons_each: " + ", ".join(
            f"{k}={('n/a' if v is None else f'{float(v):+.3f}')}" for k, v in r2c.items()
        ))
    print(f"  wrote {score_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
