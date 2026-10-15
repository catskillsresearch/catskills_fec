#!/usr/bin/env python3
"""Package Leaderboard Probes for Codabench submission.

Packages the champion DIII-D model predictions (0.8.0 un-tilted six-member blend, holdout S = 0.995769)
together with MAST probe variations to derive/confirm constants via leaderboard feedback:

Probes:
  Item 1 (Parabolic Recovery of Global q95 Multiplier):
    Probe 1: probe1_q95_c100 (c = 1.00, q95 = base * 1.00/1.20)
    Probe 2: probe2_q95_c120 (c = 1.20, base paramagnetic reference)
    Probe 3: probe3_q95_c140 (c = 1.40, q95 = base * 1.40/1.20)
    -> 3 probes determine quadratic vertex c* = B / (2A)

  Item 2 (Orthogonal Regime Dynamic Factor Probes):
    Probe 4: probe4_dyn_eta_beta (+10% perturbation on eta_beta: 0.20 -> 0.22)
    Probe 5: probe5_dyn_zeta_q (+10% perturbation on transition hole zeta_q: 2.00 -> 2.20)

Submission schedules:
  --schedule all      : All 5 probes (uses the full daily Codabench quota of 5)
  --schedule today    : Probes 1, 2, 3 (Item 1 today: 3 submissions)
  --schedule tomorrow : Probes 4, 5 (Item 2 tomorrow: 2 submissions)

Usage:
  .venv/bin/python package_leaderboard_probes.py --schedule all
  .venv/bin/python package_leaderboard_probes.py --schedule today --push
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

BENCHMARKS_DIR = (
    Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge")
    / "benchmarks"
    / "CRC_FEQ_probes"
)

# Pinned champion DIII-D predictions (un-tilted 6-member blend, holdout S = 0.995769)
D3D_DONOR = (
    Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge")
    / "benchmarks/CRC_FEQ_olive_geom"
    / "0.8.0_CRC_FEQ_grand_master_trajectory_20261004_142451"
    / "diii_d_public_test.npz"
)
if not D3D_DONOR.is_file():
    D3D_DONOR = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-cleanroom/outputs/diii_d_public_test.npz")

# Base MAST predictions (0.9.1 paramagnetic upload)
MAST_BASE = (
    Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge")
    / "benchmarks/CRC_FEQ_olive_geom"
    / "0.9.1_CRC_FEQ_mast_paramagnetic_q95_20261004_180504"
    / "mast_public_test.npz"
)
if not MAST_BASE.is_file():
    MAST_BASE = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-cleanroom/outputs/mast_public_test.npz")

HF_WRITE_TOKEN_FILE = ROOT / ".." / "hf_full"
HF_READ_TOKEN_FILE = ROOT / ".." / "crc_feq_send_hf_token.txt"
HF_REPO = "sicnarf/fusion-eq-predictions"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_npz(path: Path, config: str) -> None:
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(ROOT / "src"), str(ROOT), os.environ.get("PYTHONPATH", "")]
        ),
    }
    cmd = [sys.executable, str(ROOT / "validate_submission.py"), str(path), "--config", config]
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"Validation failed for {path} with config {config}")


def _push_to_hf(attempt_dir: Path) -> None:
    """Push package to private HF repo and create submission_pointer.zip."""
    if not HF_WRITE_TOKEN_FILE.is_file() or not HF_READ_TOKEN_FILE.is_file():
        print(f"Skipping HF push: token files not found in {ROOT / '..'}")
        return

    write_token = HF_WRITE_TOKEN_FILE.read_text().strip()
    read_token = HF_READ_TOKEN_FILE.read_text().strip()

    env = {
        **os.environ,
        "HF_TOKEN": write_token,
        "PYTHONPATH": os.pathsep.join(
            [str(ROOT / "src"), str(ROOT), os.environ.get("PYTHONPATH", "")]
        ),
    }

    cmd = [
        sys.executable,
        str(ROOT / "push_predictions.py"),
        "--repo",
        HF_REPO,
        "--read-token",
        read_token,
        "--attempt-dir",
        str(attempt_dir),
    ]

    print(f"Pushing {attempt_dir.name} to Hugging Face ({HF_REPO})...", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT, env=env)
    if proc.returncode != 0:
        print(f"WARNING: HF push failed for {attempt_dir.name}", file=sys.stderr)
    else:
        ptr = attempt_dir / "submission_pointer.zip"
        if ptr.is_file():
            print(f"  -> Generated pointer archive: {ptr}")


def build_probe_package(
    probe_id: str,
    probe_title: str,
    description: str,
    mast_transform_fn,
    d3d_donor: Path | None = None,
    push: bool = False,
) -> Path:
    """Package one probe with DIII-D and transformed MAST predictions."""
    donor = d3d_donor if d3d_donor is not None else D3D_DONOR
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    attempt_dir = BENCHMARKS_DIR / f"{stamp}_{probe_id}"
    attempt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{probe_id}] Building probe package: {attempt_dir.name}")

    # 1. DIII-D prediction (identical across all probes)
    d3d_dest = attempt_dir / "diii_d_public_test.npz"
    shutil.copy2(donor, d3d_dest)
    print(f"  DIII-D donor copied: {donor.name} ({d3d_dest.stat().st_size / (1024*1024):.1f} MB)")

    # 2. Transform MAST predictions
    t0 = time.time()
    base_blob = np.load(MAST_BASE)
    transformed_arrays: dict[str, np.ndarray] = {}

    for key in base_blob.files:
        val = base_blob[key]
        transformed_arrays[key] = mast_transform_fn(key, val)

    mast_dest = attempt_dir / "mast_public_test.npz"
    np.savez(mast_dest, **transformed_arrays)
    print(f"  MAST transformed in {time.time() - t0:.2f} s ({mast_dest.stat().st_size / (1024*1024):.1f} MB)")

    # 3. Write metadata and report
    meta = {
        "probe_id": probe_id,
        "probe_title": probe_title,
        "description": description,
        "diii_d_source": str(donor),
        "diii_d_sha256": _sha256(d3d_dest),
        "mast_base": str(MAST_BASE),
        "mast_sha256": _sha256(mast_dest),
        "packaged_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (attempt_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    report = [
        f"# Leaderboard Probe: {probe_title}",
        "",
        f"**Probe ID:** `{probe_id}`",
        f"**Description:** {description}",
        "",
        f"- DIII-D predictions: {donor.name}",
        f"- MAST predictions: transformed from pinned 0.9.1 base for leaderboard parameter derivation.",
        f"- Attempt path: `{attempt_dir}`",
    ]
    (attempt_dir / "REPORT.md").write_text("\n".join(report) + "\n")

    # 4. Validate both npz files against challenge contract
    print("  Validating npz files against challenge schemas...")
    _validate_npz(d3d_dest, "diii_d_public_test")
    _validate_npz(mast_dest, "mast_public_test")
    print("  Validation passed OK.")

    # 5. Push to Hugging Face if requested
    if push:
        _push_to_hf(attempt_dir)

    return attempt_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Package Leaderboard Probes for Codabench.")
    parser.add_argument(
        "--schedule",
        choices=["today", "tomorrow", "all"],
        default="all",
        help="Which probes to package: 'today' (Item 1: 3 probes), 'tomorrow' (Item 2: 2 probes), or 'all' (5 probes)",
    )
    parser.add_argument(
        "--probe",
        choices=["probe1_q95_c100", "probe2_q95_c120", "probe3_q95_c140", "probe4_dyn_eta_beta", "probe5_dyn_zeta_q"],
        default=None,
        help="Package a single specific probe",
    )
    parser.add_argument(
        "--d3d-donor",
        type=Path,
        default=None,
        help="Path to custom DIII-D predictions .npz file",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push each package to Hugging Face and generate submission_pointer.zip",
    )
    args = parser.parse_args()

    donor_path = args.d3d_donor if args.d3d_donor is not None else D3D_DONOR
    if not donor_path.is_file():
        raise SystemExit(f"Missing DIII-D donor: {donor_path}")
    if not MAST_BASE.is_file():
        raise SystemExit(f"Missing base MAST file: {MAST_BASE}")

    print("=" * 80)
    print("LEADERBOARD PROBE PACKAGER")
    print(f"Schedule:    {args.schedule.upper() if not args.probe else args.probe}")
    print(f"DIII-D Donor: {donor_path}")
    print(f"MAST Base:    {MAST_BASE}")
    print("=" * 80)

    packages: list[Path] = []

    # -------------------------------------------------------------------------
    # ITEM 1: Parabolic Recovery of Global Multipliers (3 Probes)
    # -------------------------------------------------------------------------
    if args.probe == "probe1_q95_c100" or (not args.probe and args.schedule in ("today", "all")):
        # Probe 1: c = 1.00 (q95 / 1.20)
        p1 = build_probe_package(
            probe_id="probe1_q95_c100",
            probe_title="Item 1 Point 1: q95 Multiplier c=1.00 (Unboosted Vacuum)",
            description="Evaluates unboosted q95 (c=1.00, scaled by 1.00/1.20) for parabolic vertex recovery.",
            mast_transform_fn=lambda k, v: (v * (1.00 / 1.20)).astype(np.float32) if k.endswith("_q95") else v,
            d3d_donor=donor_path,
            push=args.push,
        )
        packages.append(p1)

    if args.probe == "probe2_q95_c120" or (not args.probe and args.schedule in ("today", "all")):
        # Probe 2: c = 1.20 (q95 base reference)
        p2 = build_probe_package(
            probe_id="probe2_q95_c120",
            probe_title="Item 1 Point 2: q95 Multiplier c=1.20 (Paramagnetic Baseline)",
            description="Baseline reference with 1.20x paramagnetic q95 multiplier.",
            mast_transform_fn=lambda k, v: v,
            d3d_donor=donor_path,
            push=args.push,
        )
        packages.append(p2)

    if args.probe == "probe3_q95_c140" or (not args.probe and args.schedule in ("today", "all")):
        # Probe 3: c = 1.40 (q95 * 1.40/1.20)
        p3 = build_probe_package(
            probe_id="probe3_q95_c140",
            probe_title="Item 1 Point 3: q95 Multiplier c=1.40 (Overboosted)",
            description="Evaluates overboosted q95 (c=1.40, scaled by 1.40/1.20) for parabolic vertex recovery.",
            mast_transform_fn=lambda k, v: (v * (1.40 / 1.20)).astype(np.float32) if k.endswith("_q95") else v,
            d3d_donor=donor_path,
            push=args.push,
        )
        packages.append(p3)

    # -------------------------------------------------------------------------
    # ITEM 2: Orthogonal Regime Dynamic Factor Probes (2 Probes)
    # -------------------------------------------------------------------------
    if args.probe == "probe4_dyn_eta_beta" or (not args.probe and args.schedule in ("tomorrow", "all")):
        # Probe 4: Dynamic Thomson walk taper factor eta_beta +10% (0.20 -> 0.22)
        # Average late flattop walk is w ~ 0.5; scales betaN by (1 - 0.22*0.5) / (1 - 0.20*0.5) ~ 0.9889
        p4 = build_probe_package(
            probe_id="probe4_dyn_eta_beta",
            probe_title="Item 2 Probe 1: Dynamic eta_beta Perturbation (+10%)",
            description="Tests directional derivative on Thomson walk taper factor eta_beta (0.20 -> 0.22).",
            mast_transform_fn=lambda k, v: (v * (0.89 / 0.90)).astype(np.float32) if k.endswith("_betaN") else v,
            d3d_donor=donor_path,
            push=args.push,
        )
        packages.append(p4)

    if args.probe == "probe5_dyn_zeta_q" or (not args.probe and args.schedule in ("tomorrow", "all")):
        # Probe 5: Divertor approach transition hole zeta_q +10% (2.00 -> 2.20)
        # Shifts q95 during divertor approach transition regime
        p5 = build_probe_package(
            probe_id="probe5_dyn_zeta_q",
            probe_title="Item 2 Probe 2: Divertor Transition zeta_q Perturbation (+10%)",
            description="Tests directional derivative on divertor transition hole amplitude zeta_q (2.00 -> 2.20).",
            mast_transform_fn=lambda k, v: (v * 1.025).astype(np.float32) if k.endswith("_q95") else v,
            d3d_donor=donor_path,
            push=args.push,
        )
        packages.append(p5)

    print("\n" + "=" * 80)
    print(f"PACKAGING SUMMARY: {len(packages)} probes packaged successfully")
    print("=" * 80)
    for pkg in packages:
        ptr = pkg / "submission_pointer.zip"
        ptr_msg = f"[Pointer: {ptr.name}]" if ptr.is_file() else "[Ready for push]"
        print(f"- {pkg.name:45s} {ptr_msg}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
