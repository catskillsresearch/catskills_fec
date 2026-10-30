"""Catskills hybrid / data-driven ψ reconstruction — ``CRC_FEQ_black``.

First luminance revision (black). See ``MODEL_DESCRIPTION_CRC_FEQ_black.md``
and ``CRC_FEQ_COLORS.md``. Public model ID is opaque (no method keyword).

Hybrid loss intent (Fernandes-style, adapted to Grad-Shafranov):

    L ≈ L_data(ψ̂, ψ) + λ_GS ||r_GS||² + L_scalars

where r_GS is the discrete Grad-Shafranov residual from ``catskills.physics``.
In black, GS is applied as an inference-time amplitude nudge, not full train-time residual backprop.

Heads: one flux head (``--flux-head ridge|mlp``) + two fixed ``RidgeCV`` scalar heads.
"""
from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from catskills.features import features_from_shot, shot_from_row_inputs_only
from catskills.model_version import make_model_version
from catskills.physics import default_grid_from_row, gs_residual_loss, MU0
from experiments import ExperimentConfig, build_feature_matrix, load_shots

MODEL_ID = "CRC_FEQ_black"
FluxHead = Literal["ridge", "mlp"]


@dataclass
class PINNModelBundle:
    feature_scaler: StandardScaler
    pca: PCA
    flux_model: object
    q95_model: RidgeCV
    betaN_model: RidgeCV
    n_features: int
    gs_lambda: float
    coeffs_p: np.ndarray
    coeffs_ff: np.ndarray
    include_thomson: bool
    flux_head: str = "ridge"
    method: str = MODEL_ID
    model_version: str = field(default="")


def _fit_profiles_for_shot(
    psi: np.ndarray, R: np.ndarray, Z: np.ndarray, axis_sign: float = -1.0
) -> tuple[np.ndarray, np.ndarray]:
    from catskills.physics import grad_shafranov_star

    delta = grad_shafranov_star(psi, R, Z)
    deg = 2
    core = (slice(1, -1), slice(1, -1))
    psi_c = psi[core]
    delta_c = delta[core]
    Rr = np.meshgrid(R, Z, indexing="xy")[0]
    R_c = Rr[core]
    if axis_sign < 0:
        psi_axis, psi_edge = float(psi.min()), float(psi.max())
    else:
        psi_axis, psi_edge = float(psi.max()), float(psi.min())
    psi_n = (psi_c - psi_axis) / max(abs(psi_edge - psi_axis), 1e-8)
    if axis_sign > 0:
        psi_n = 1.0 - psi_n
    psi_n = np.clip(psi_n, 0.0, 1.0).ravel()
    R2 = R_c.ravel() ** 2

    cols = []
    for k in range(deg + 1):
        cols.append((-MU0 * R2 * (psi_n ** k)).reshape(-1, 1))
    for k in range(deg + 1):
        cols.append((-(psi_n ** k)).reshape(-1, 1))
    A = np.concatenate(cols, axis=1)
    y = delta_c.ravel()
    ATA = A.T @ A + 1e-6 * np.eye(A.shape[1])
    coeffs = np.linalg.solve(ATA, A.T @ y)
    return coeffs[: deg + 1], coeffs[deg + 1 :]


def _make_flux_model(flux_head: FluxHead):
    if flux_head == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(256, 128),
            max_iter=400,
            random_state=42,
            early_stopping=True,
        )
    return RidgeCV(alphas=np.logspace(-3, 3, 13))


def train_pinn_baseline(
    n_shots: int = 20,
    include_thomson: bool = True,
    gs_lambda: float = 1e-3,
    flux_head: FluxHead = "ridge",
    use_mlp: Optional[bool] = None,
    output: Path = Path("catskills_artifacts/CRC_FEQ_black.pkl"),
    shots: list[dict] | None = None,
) -> PINNModelBundle:
    if use_mlp is not None:
        flux_head = "mlp" if use_mlp else "ridge"
    if flux_head not in ("ridge", "mlp"):
        raise ValueError(f"flux_head must be 'ridge' or 'mlp', got {flux_head!r}")

    if shots is None:
        cfg = ExperimentConfig(
            n_shots=n_shots,
            include_thomson=include_thomson,
            n_pca_components=40,
            output_dir=Path("results/catskills_pinn"),
        )
        shots = load_shots(cfg)
    X, Y, S, _shot_ids = build_feature_matrix(shots, include_thomson=include_thomson)
    Y_flat = Y.reshape(len(Y), -1)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n_pca = min(40, max(1, Xs.shape[0] - 1))
    pca = PCA(n_components=n_pca, random_state=42)
    Zc = pca.fit_transform(Y_flat)

    print(f"[{MODEL_ID}] flux_head={flux_head}  thomson={include_thomson}  "
          f"X={Xs.shape}  PCA={n_pca}")
    flux_model = _make_flux_model(flux_head)
    flux_model.fit(Xs, Zc)

    R = np.linspace(0.8, 2.5, 65)
    Zg = np.linspace(-1.5, 1.5, 65)
    recon = pca.inverse_transform(flux_model.predict(Xs[: min(64, len(Xs))])).reshape(-1, 65, 65)
    coeffs_list_p, coeffs_list_ff, losses = [], [], []
    for i in range(len(recon)):
        cp, cf = _fit_profiles_for_shot(recon[i], R, Zg, axis_sign=-1.0)
        coeffs_list_p.append(cp)
        coeffs_list_ff.append(cf)
        losses.append(
            gs_residual_loss(recon[i], R, Zg, coeffs_p=cp, coeffs_ff=cf, axis_sign=-1.0)
        )
    coeffs_p = np.mean(np.stack(coeffs_list_p), axis=0)
    coeffs_ff = np.mean(np.stack(coeffs_list_ff), axis=0)
    print(f"[{MODEL_ID}] mean GS residual on train recon: {np.mean(losses):.4e}")
    print(f"[{MODEL_ID}] gs_lambda={gs_lambda}")

    from experiments import EFIT_SCALAR_LABELS

    i_q = EFIT_SCALAR_LABELS.index("efit_q95")
    i_b = EFIT_SCALAR_LABELS.index("efit_beta_n")
    q95 = S[:, i_q]
    betaN = S[:, i_b]
    # Fixed scalar heads — always RidgeCV (not switched by --flux-head).
    q95_model = RidgeCV(alphas=np.logspace(-3, 3, 13))
    betaN_model = RidgeCV(alphas=np.logspace(-3, 3, 13))
    m_q = np.isfinite(q95)
    m_b = np.isfinite(betaN)
    q95_model.fit(Xs[m_q] if m_q.sum() > 10 else Xs, q95[m_q] if m_q.sum() > 10 else np.zeros(len(Xs)))
    betaN_model.fit(Xs[m_b] if m_b.sum() > 10 else Xs, betaN[m_b] if m_b.sum() > 10 else np.zeros(len(Xs)))

    method_key = f"{MODEL_ID}_{flux_head}"
    bundle = PINNModelBundle(
        feature_scaler=scaler,
        pca=pca,
        flux_model=flux_model,
        q95_model=q95_model,
        betaN_model=betaN_model,
        n_features=Xs.shape[1],
        gs_lambda=gs_lambda,
        coeffs_p=coeffs_p,
        coeffs_ff=coeffs_ff,
        include_thomson=include_thomson,
        flux_head=flux_head,
        method=MODEL_ID,
        model_version=make_model_version(method_key),
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[{MODEL_ID}] wrote {output}  version={bundle.model_version}")
    return bundle


def load_bundle(path: Path) -> PINNModelBundle:
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_row(bundle: PINNModelBundle, row: dict, source: str) -> dict:
    shot = shot_from_row_inputs_only(row)
    X = features_from_shot(shot, include_thomson=bundle.include_thomson)
    if X.shape[1] < bundle.n_features:
        X = np.hstack([X, np.zeros((len(X), bundle.n_features - X.shape[1]), dtype=np.float32)])
    elif X.shape[1] > bundle.n_features:
        X = X[:, : bundle.n_features]
    Xs = bundle.feature_scaler.transform(X)
    Zc = bundle.flux_model.predict(Xs)
    psi = bundle.pca.inverse_transform(Zc).reshape(-1, 65, 65)

    R, Zg = default_grid_from_row(row, source)
    axis_sign = -1.0 if "DIII" in source.upper() else 1.0
    if bundle.gs_lambda > 0:
        scales = np.linspace(0.7, 1.3, 7)
        nudged = psi.copy()
        for t in range(len(psi)):
            best_s, best_l = 1.0, 1e99
            for s in scales:
                cand = psi[t] * s
                loss = gs_residual_loss(
                    cand, R, Zg,
                    coeffs_p=bundle.coeffs_p,
                    coeffs_ff=bundle.coeffs_ff,
                    axis_sign=axis_sign,
                )
                data_pen = float(np.mean((cand - psi[t]) ** 2))
                total = data_pen + bundle.gs_lambda * loss
                if total < best_l:
                    best_l, best_s = total, s
            nudged[t] = psi[t] * best_s
        psi = nudged

    return {
        "psirz": psi.astype(np.float32),
        "q95": bundle.q95_model.predict(Xs).astype(np.float32),
        "betaN": bundle.betaN_model.predict(Xs).astype(np.float32),
    }


def main():
    ap = argparse.ArgumentParser(description=f"Train {MODEL_ID} (Catskills CRC_FEQ luminance line)")
    ap.add_argument("--n-shots", type=int, default=20)
    ap.add_argument(
        "--no-thomson",
        action="store_true",
        help="drop Thomson features (default: on — same as full baseline:mlp_pca)",
    )
    ap.add_argument("--gs-lambda", type=float, default=1e-3)
    ap.add_argument(
        "--flux-head",
        choices=["ridge", "mlp"],
        default="ridge",
        help="flux→PCA head only (scalar q95/βN heads stay RidgeCV). Default: ridge",
    )
    ap.add_argument(
        "--mlp",
        action="store_true",
        help="alias for --flux-head mlp",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    flux_head: FluxHead = "mlp" if args.mlp else args.flux_head
    out = args.out or Path(f"catskills_artifacts/{MODEL_ID}_{flux_head}.pkl")
    train_pinn_baseline(
        n_shots=args.n_shots,
        include_thomson=not args.no_thomson,
        gs_lambda=args.gs_lambda,
        flux_head=flux_head,
        output=out,
    )


if __name__ == "__main__":
    main()
