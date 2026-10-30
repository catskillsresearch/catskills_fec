#!/usr/bin/env python3
"""O1 geometry-aware rebuild — controlled κ/R/Z placement during plastic training.

    CRC_FEQ_MODEL_ID=CRC_FEQ_olive_geom python -m catskills.scripts.o1_geom_train contract
    CRC_FEQ_MODEL_ID=CRC_FEQ_olive_geom python -m catskills.scripts.o1_geom_train smoke
    CRC_FEQ_MODEL_ID=CRC_FEQ_olive_geom python -m catskills.scripts.o1_geom_train control
    CRC_FEQ_MODEL_ID=CRC_FEQ_olive_geom python -m catskills.scripts.o1_geom_train place
    CRC_FEQ_MODEL_ID=CRC_FEQ_olive_geom python -m catskills.scripts.o1_geom_train early
    CRC_FEQ_MODEL_ID=CRC_FEQ_olive_geom python -m catskills.scripts.o1_geom_train early --batch 2048

One GPU job at a time. Tags ``O1G_*`` live in the ``CRC_FEQ_olive_geom`` namespace so no
historical O1 or shipped checkpoint can be overwritten.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CRC_FEQ_MODEL_ID", "CRC_FEQ_olive_geom")

from catskills.lab_paths import find_checkpoint, find_score, results_dir  # noqa: E402
from catskills.olive_loss import (  # noqa: E402
    GEOM_RAMP,
    GRADNORM_ALPHA,
    PLACEMENT_EARLY_FLOOR,
    PLACEMENT_EARLY_FOCUS,
    PLACEMENT_STACK_FOCUS,
    WARMUP_CONS,
    WARMUP_R2PSI,
    geom_ramp_mult,
    warmup_healthy,
)
from catskills.pinn_charcoal import SCALAR_NAMES, derive_stack_geom_weights  # noqa: E402
from catskills.scripts import maroon_ladder as ml  # noqa: E402
from catskills.scripts.purple_ladder import DUALCTX_WIN9, STABLE_17M  # noqa: E402

GEOM_DIR = results_dir() / "olive_geom"
MODEL_ID = "CRC_FEQ_olive_geom"

COMMON = [
    *STABLE_17M,
    *DUALCTX_WIN9,
    "--thomson-zgrid",
    "--token-budget", "4096",
    "--epochs", "88",
    "--lr", "3e-4",
    "--lambda-coeff-white", "1.35",
    "--coeff-white-power", "1.0",
    "--coeff-white-clamp", "12",
    "--val-geom-frames", "20000",
    "--lbfgs-steps", "0",
    "--ckpt-every", "5",
]

PLACEMENT = [
    "--adaptive-warmup",
    "--warmup-cap", "40",
    "--warmup-r2psi", str(WARMUP_R2PSI),
    "--warmup-cons", str(WARMUP_CONS),
    "--geom-ramp", str(GEOM_RAMP),
    "--geom-weights", "placement",
    "--geom-emphasis", "1.0",
    "--gradnorm",
    "--gradnorm-alpha", str(GRADNORM_ALPHA),
    "--psi-collapse-r2psi", "0.9985",
    "--psi-collapse-streak", "3",
]

PLACEMENT_EARLY = [
    "--adaptive-warmup",
    "--warmup-cap", "40",
    "--warmup-r2psi", "0.9985",
    "--warmup-cons", "0.92",
    "--geom-ramp", str(GEOM_RAMP),
    "--geom-weights", "placement_early",
    "--geom-emphasis", "1.0",
    "--lcfs-emphasis", "1.0",
    "--gradnorm",
    "--gradnorm-alpha", str(GRADNORM_ALPHA),
    "--psi-collapse-r2psi", "0.9985",
    "--psi-collapse-streak", "3",
]


def _override_epochs(extra: list[str], epochs: int | None) -> list[str]:
    """Replace or append ``--epochs`` when a short probe is requested."""
    if epochs is None:
        return list(extra)
    out: list[str] = []
    skip = False
    for a in extra:
        if skip:
            skip = False
            continue
        if a == "--epochs":
            skip = True
            continue
        out.append(a)
    out.extend(["--epochs", str(int(epochs))])
    return out


def _run(tag: str, extra: list[str], *, epochs: int | None = None) -> int:
    extra = _override_epochs(extra, epochs)
    print(f"\n{'#' * 70}\n### {tag}\n{'#' * 70}", flush=True)
    t0 = __import__("time").perf_counter()
    proc = subprocess.Popen(
        [sys.executable, "-m", "catskills.run_crc_feq_pinn_charcoal", *extra, "--tag", tag],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={
            **os.environ,
            "TQDM_DISABLE": "1",
            "PYTHONUNBUFFERED": "1",
            "CRC_FEQ_MODEL_ID": MODEL_ID,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
    )
    assert proc.stdout is not None
    keep = ml.KEEP + (
        "shotseq", "dualshot", "token_budget", "gate_seq",
        "geometry", "stack", "placement", "lambda_geom", "geom_emphasis",
        "GradNorm", "w_g", "PCGrad", "COLLAPSE", "ℓ_p", "lp_floor",
        "KILLED", "kill_r2psi", "val_Cons", "adaptive warmup", "post-gate",
        "RESUMED", "partial checkpoint", "CHECKPOINT", "partial kept",
        "limited mixup", "stratified", "diff cons",
        "not finite", "Traceback", "CUDA error",
    )
    for raw in proc.stdout:
        line = raw.split("\r")[-1].rstrip()
        if any(k in line for k in keep):
            print(line, flush=True)
    rc = proc.wait()
    elapsed = (__import__("time").perf_counter() - t0) / 60.0
    print(f"### {tag} exit={rc}  {elapsed:.1f} min", flush=True)
    GEOM_DIR.mkdir(parents=True, exist_ok=True)
    rec = {"tag": tag, "exit": rc, "minutes": elapsed, "extra": extra}
    for src_fn in (find_score(tag), results_dir() / "charcoal" / f"{tag}.json"):
        if src_fn.is_file():
            rec["score"] = json.loads(src_fn.read_text(encoding="utf-8"))
            dest = GEOM_DIR / f"{tag}.json"
            dest.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
            break
    return rc


def _contract() -> int:
    """Verify placement weights, adaptive gate helper, and unchanged stack weights."""
    from catskills.olive_loss import OLIVE_STACK_FOCUS

    checks: dict[str, bool] = {}
    w_place = derive_stack_geom_weights(focus=PLACEMENT_STACK_FOCUS)
    w_stack = derive_stack_geom_weights(focus=OLIVE_STACK_FOCUS)
    floor = 0.15
    promoted = {"kappa", "R_axis", "Z_axis"}
    floor_only = {"tri_top", "tri_bot", "volume", "li"}
    floor_vals = [float(w_place[SCALAR_NAMES.index(n)]) for n in floor_only]
    for name in SCALAR_NAMES:
        idx = SCALAR_NAMES.index(name)
        val = float(w_place[idx])
        if name in promoted:
            checks[f"placement_{name}_promoted"] = val > floor + 0.01
    checks["placement_floor_equal"] = max(floor_vals) - min(floor_vals) < 0.01
    checks["placement_floor_below_promoted"] = min(floor_vals) < min(
        float(w_place[SCALAR_NAMES.index(n)]) for n in promoted
    )
    checks["placement_kappa_heaviest"] = (
        float(w_place[SCALAR_NAMES.index("kappa")])
        >= max(float(w_place[SCALAR_NAMES.index(n)]) for n in ("R_axis", "Z_axis"))
    )
    checks["stack_unchanged_li"] = (
        float(w_stack[SCALAR_NAMES.index("li")])
        > float(w_stack[SCALAR_NAMES.index("R_axis")])
    )
    checks["warmup_healthy_gate"] = warmup_healthy(0.999, 0.96)
    checks["warmup_unhealthy"] = not warmup_healthy(0.998, 0.96)
    checks["ramp_at_gate_zero"] = geom_ramp_mult(10, 10, GEOM_RAMP) == 0.0
    checks["ramp_after_gate"] = abs(geom_ramp_mult(11, 10, GEOM_RAMP) - 0.25) < 1e-9
    ok = all(checks.values())
    rec = {
        "tag": "O1G_contract",
        "ok": ok,
        "checks": checks,
        "placement_weights": {n: float(w_place[i]) for i, n in enumerate(SCALAR_NAMES)},
        "stack_weights": {n: float(w_stack[i]) for i, n in enumerate(SCALAR_NAMES)},
    }
    GEOM_DIR.mkdir(parents=True, exist_ok=True)
    dest = GEOM_DIR / "O1G_contract.json"
    dest.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    print(f"### O1G_contract ok={ok}  {dest}", flush=True)
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
    return 0 if ok else 2


def _smoke(*, epochs: int | None = None) -> int:
    skip = {
        "--warmup-r2psi", str(WARMUP_R2PSI),
        "--warmup-cons", str(WARMUP_CONS),
        "--psi-collapse-r2psi", "0.9985",
        "--psi-collapse-streak", "3",
    }
    placement_smoke = [a for a in PLACEMENT if a not in skip]
    extra = [
        *COMMON,
        *placement_smoke,
        "--max-frames", "60000",
        "--epochs", "20",
        # 60k-frame subset: lower both gates so adaptive ramp + geom grads execute.
        "--warmup-r2psi", "0.992",
        "--warmup-cons", "0.65",
    ]
    return _run("O1G_smoke_gate", extra, epochs=epochs)


def _control(*, epochs: int | None = None) -> int:
    extra = [*COMMON, "--geom-emphasis", "0"]
    return _run("O1G_control", extra, epochs=epochs)


def _place(*, epochs: int | None = None) -> int:
    extra = [*COMMON, *PLACEMENT]
    return _run("O1G_place", extra, epochs=epochs)


def _early(*, batch: int = 256, epochs: int | None = None) -> int:
    if not find_checkpoint("O1G_control").is_file():
        print("early needs O1G_control checkpoint for the paired eval", file=sys.stderr)
        return 2
    tag = "O1G_early" if batch == 256 else f"O1G_early_b{batch}"
    extra = [*COMMON, *PLACEMENT_EARLY, "--batch-size", str(batch)]
    return _run(tag, extra, epochs=epochs)


def main() -> int:
    ap = argparse.ArgumentParser(description="O1 geometry-aware rebuild launcher")
    ap.add_argument(
        "job",
        choices=["contract", "smoke", "control", "place", "early", "all"],
        help="contract=plumbing; smoke=60k/20ep; control=88ep geom-off; "
             "place=late gate; early=plastic gate + protected ℓ_i",
    )
    ap.add_argument("--batch", type=int, default=256,
                    help="batch size for early job (default 256; tag O1G_early_b<N> when != 256)")
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs (e.g. 2 for an initial-convergence probe)",
    )
    args = ap.parse_args()
    GEOM_DIR.mkdir(parents=True, exist_ok=True)
    log = GEOM_DIR / "o1_geom_train.log"
    sys.stdout = ml._Tee(sys.stdout, log)
    sys.stderr = sys.stdout
    print(
        f"### o1_geom_train start  job={args.job}"
        + (f"  epochs={args.epochs}" if args.epochs is not None else ""),
        flush=True,
    )

    if args.job == "contract":
        return _contract()
    if args.job == "smoke":
        rc = _contract()
        if rc != 0:
            return rc
        return _smoke(epochs=args.epochs)
    if args.job == "control":
        return _control(epochs=args.epochs)
    if args.job == "place":
        if not find_checkpoint("O1G_control").is_file():
            print("place needs O1G_control checkpoint first", file=sys.stderr)
            return 2
        return _place(epochs=args.epochs)
    if args.job == "early":
        return _early(batch=int(args.batch), epochs=args.epochs)
    # all: contract → smoke → control → place (sequential)
    for fn in (_contract, lambda: _smoke(epochs=args.epochs),
               lambda: _control(epochs=args.epochs),
               lambda: _place(epochs=args.epochs)):
        rc = fn()
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
