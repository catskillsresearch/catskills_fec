"""Outputs and artifacts on the lab disk. The cleanroom git checkout is not this path."""
from pathlib import Path

ROOT = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-cleanroom")
OUTPUTS = ROOT / "outputs"
ARTIFACTS = ROOT / "artifacts"
