"""Train and serve challenge-style sklearn+PCA baselines (Ridge / MLP) for packaging.

These are **challenge baselines** (``baseline:ridge_pca`` / ``baseline:mlp_pca``), not
Catskills-original models. Catskills owns the train/package harness and attempt reports.
"""
from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from catskills.features import features_from_shot, shot_from_row_inputs_only
from catskills.model_version import make_model_version
from experiments import EFIT_SCALAR_LABELS, ExperimentConfig, build_feature_matrix, load_shots


@dataclass
class SklearnBundle:
    feature_scaler: StandardScaler
    pca: PCA
    flux_model: object
    q95_model: RidgeCV
    betaN_model: RidgeCV
    n_features: int
    include_thomson: bool
    method: str
    model_version: str = field(default="")


def train_sklearn_baseline(
    method: str = "ridge_pca",
    n_shots: int = 20,
    include_thomson: bool = False,
    output: Path = Path("catskills_artifacts/ridge_pca.pkl"),
    shots: list[dict] | None = None,
) -> SklearnBundle:
    if shots is None:
        cfg = ExperimentConfig(
            n_shots=n_shots,
            include_thomson=include_thomson,
            n_pca_components=40,
            output_dir=Path("results/catskills_sklearn"),
        )
        shots = load_shots(cfg)
    X, Y, S, _ = build_feature_matrix(shots, include_thomson=include_thomson)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=min(40, Xs.shape[0] - 1), random_state=42)
    Zc = pca.fit_transform(Y.reshape(len(Y), -1))

    if method == "mlp_pca":
        flux_model = MLPRegressor(
            hidden_layer_sizes=(256, 128), max_iter=400,
            random_state=42, early_stopping=True,
        )
    else:
        method = "ridge_pca"
        flux_model = RidgeCV(alphas=np.logspace(-3, 3, 13))
    flux_model.fit(Xs, Zc)

    i_q = EFIT_SCALAR_LABELS.index("efit_q95")
    i_b = EFIT_SCALAR_LABELS.index("efit_beta_n")
    q95, betaN = S[:, i_q], S[:, i_b]
    q95_model = RidgeCV(alphas=np.logspace(-3, 3, 13))
    betaN_model = RidgeCV(alphas=np.logspace(-3, 3, 13))
    m_q, m_b = np.isfinite(q95), np.isfinite(betaN)
    q95_model.fit(Xs[m_q] if m_q.sum() > 10 else Xs, q95[m_q] if m_q.sum() > 10 else np.zeros(len(Xs)))
    betaN_model.fit(Xs[m_b] if m_b.sum() > 10 else Xs, betaN[m_b] if m_b.sum() > 10 else np.zeros(len(Xs)))

    bundle = SklearnBundle(
        feature_scaler=scaler,
        pca=pca,
        flux_model=flux_model,
        q95_model=q95_model,
        betaN_model=betaN_model,
        n_features=Xs.shape[1],
        include_thomson=include_thomson,
        method=f"baseline:{method}",
        model_version=make_model_version(method),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[{method}] wrote {output}")
    return bundle


def load_bundle(path: Path) -> SklearnBundle:
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_row(bundle: SklearnBundle, row: dict, source: str) -> dict:
    shot = shot_from_row_inputs_only(row)
    X = features_from_shot(shot, include_thomson=bundle.include_thomson)
    if X.shape[1] < bundle.n_features:
        X = np.hstack([X, np.zeros((len(X), bundle.n_features - X.shape[1]), dtype=np.float32)])
    elif X.shape[1] > bundle.n_features:
        X = X[:, : bundle.n_features]
    Xs = bundle.feature_scaler.transform(X)
    psi = bundle.pca.inverse_transform(bundle.flux_model.predict(Xs)).reshape(-1, 65, 65)
    return {
        "psirz": psi.astype(np.float32),
        "q95": bundle.q95_model.predict(Xs).astype(np.float32),
        "betaN": bundle.betaN_model.predict(Xs).astype(np.float32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["ridge_pca", "mlp_pca"], default="ridge_pca")
    ap.add_argument("--n-shots", type=int, default=20)
    ap.add_argument("--include-thomson", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or Path(f"catskills_artifacts/{args.method}.pkl")
    train_sklearn_baseline(args.method, args.n_shots, args.include_thomson, out)


if __name__ == "__main__":
    main()
