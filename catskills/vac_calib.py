"""catskills/vac_calib.py — Analytic vacuum Green basis provider.

Computes the exact Biot-Savart signed Green basis for the 19 poloidal shaping
coils (ECOILA + F1A-F9B) across the 65x65 DIII-D computational grid.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

_CACHE: dict[tuple[str, bool], np.ndarray] = {}


def source_basis(machine: str = "d3", calibrated: bool = False) -> np.ndarray:
    """Return the raw analytic Biot-Savart vacuum Green basis (19, 4225) for DIII-D in Wb/kA."""
    if machine not in ("d3", "d3d"):
        raise ValueError(f"Unknown machine: {machine}")
    key = (machine, calibrated)
    if key in _CACHE:
        return _CACHE[key]

    from catskills.dataset_local import dataset_mirror_dir
    from src.feq.data import load_shot
    from catskills.vacuum_green import load_coil_geom, build_green

    # Read coil geometry and grid coordinates from train split
    train_dir = dataset_mirror_dir() / "data" / "diii_d_train"
    parquets = sorted(train_dir.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(f"No parquets found in {train_dir}")

    row = load_shot(parquets[0])
    coils_dict = load_coil_geom(row)
    sig_map = {c["sig"]: c for c in coils_dict}
    order = ["ECOILA"] + [f"F{i}{side}" for i in range(1, 10) for side in ("A", "B")]
    ordered_coils = [sig_map[s] for s in order]

    R = np.asarray(row["efit_grid_R"], dtype=np.float64)
    Z = np.asarray(row["efit_grid_Z"], dtype=np.float64)
    G_si = build_green(ordered_coils, R, Z) # (19, 4225) Wb/A

    # Scale from Wb/A to Wb/kA and apply poloidal Lenz sign convention
    G_raw = (G_si * -1000.0).astype(np.float32)

    _CACHE[key] = G_raw
    return G_raw
