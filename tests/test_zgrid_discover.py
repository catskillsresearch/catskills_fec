"""Phase-3 ZGRID discovery must recover the Dev public thz feature law."""
from __future__ import annotations

import numpy as np

from catskills.dataset_local import dataset_mirror_dir
from catskills.features import (
    interpolate_magnetics_to_efit_masked,
    interpolate_thomson_zgrid,
    shot_from_row_inputs_only,
)
from catskills.lab_paths import lab_root
from catskills.warm_feature_cache import row_from_parquet, train_parquet_dir
from catskills.zgrid_discover import (
    apply_zgrid_env,
    clear_zgrid_env,
    discover_zgrid,
    write_zgrid_json,
)


def test_discover_matches_dev_thz_cache(tmp_path):
    train_dir = train_parquet_dir()
    # Stride keeps CI/runtime light while covering chord extrema.
    payload = discover_zgrid(train_dir, stride=50)
    assert payload["z_lo"] == -1.2
    assert payload["z_hi"] == 1.2
    assert payload["n_bins"] == 32
    out = write_zgrid_json(tmp_path / "zgrid.json", payload)
    assert out.is_file()

    apply_zgrid_env(payload)
    try:
        pub = sorted((dataset_mirror_dir() / "data" / "diii_d_public_test").glob("d3d_shot_*.parquet"))[0]
        row = row_from_parquet(pub)
        shot = shot_from_row_inputs_only(row)
        th = interpolate_thomson_zgrid(shot)
        X = np.hstack([interpolate_magnetics_to_efit_masked(shot), th]).astype(np.float32)
        cache = np.load(
            lab_root() / "cache" / "public_test_features_v1" / "diii_d_public_test_thz" / "shot_0000.npz"
        )["X"]
        assert X.shape == cache.shape
        assert np.max(np.abs(X.astype(np.float64) - cache.astype(np.float64))) == 0.0
    finally:
        clear_zgrid_env()
