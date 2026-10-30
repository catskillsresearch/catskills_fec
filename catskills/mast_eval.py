"""Measure MAST, locally, with the official functionals.

Every cross-machine idea in this repo was graded on the DIII-D 80-shot holdout and
then shipped to Codabench to find out what it did on MAST. That is why the ledger
carries a dozen "MAST never scored" entries: `local_score.MACHINE` was the string
``"DIII-D"``, `diii_d_train` is 7,041 DIII-D shots and 0 MAST shots, and the
1,206-shot `mast_public_test` split is inputs only.

Two measurements are possible without the withheld MAST labels, and neither had
been wired up.

Labelled holdout
----------------
``parquet_data/mast_shot_{28348,28350,28351}.parquet`` carry real ``efit_psirz``:
115 frames, 485,875 pixels, on the MAST envelope (R in [0.06, 2.00] m, Z in
[-2.00, 2.00] m). `CHARCOAL_PLAN.md` section 13 noted these exist and nothing
ever scored against them. 115 frames is thin on *shot* diversity but ample for
R2_psi, which pools over pixels -- and it detects the 0.83 -> 0.10 style collapse
that is the actual failure mode, instantly.

These shots are 283xx while the scored public split is 23xxx-27xxx, so treat the
number as a collapse detector and a relative ranker between checkpoints, not as a
prediction of public MAST S.

Plausibility, no labels at all
------------------------------
Two of the scorer's own failure counters need only the prediction:

* whether `extract_lcfs` can find a closed flux surface at all -- the public run
  reported ``lcfs_fail_frac`` 0.999899 on MAST, and that is computable here over
  all 69,075 frames;
* where `magnetic_axis` puts the O-point, against the R ~ 0.89 m / Z ~ 0.01 m the
  labelled frames actually show.

The LCFS probe deliberately passes ``axis_sign=None`` so `extract_lcfs_with_sign`
tries *both* orientations. That is more generous than the scorer, which spends one
fold-global sign bit chosen from psi residuals alone; a failure here therefore
means no closed surface exists under either convention, which is unambiguous.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "fusion_scoring") not in sys.path:
    sys.path.insert(0, str(ROOT / "fusion_scoring"))

from catskills.machine_env import load_env

# The three demo shots are the only MAST rows anywhere in the corpus with labels.
HOLDOUT_SHOTS = ("mast_shot_28348", "mast_shot_28350", "mast_shot_28351")
MACHINE = "MAST"


def holdout_dir() -> Path:
    return ROOT / "parquet_data"


@dataclass
class MastShot:
    """One labelled MAST shot: inputs the model reads, plus the withheld truth."""

    key: str
    vac: np.ndarray  # (T, 65, 65) Psi_vac, rectangle Green on MAST's own coil table
    ip: np.ndarray  # (T,)
    psi: np.ndarray  # (T, 65, 65) efit_psirz ground truth
    q95: np.ndarray  # (T,)
    betaN: np.ndarray  # (T,)
    times: np.ndarray  # (T,)

    @property
    def T(self) -> int:
        return int(self.psi.shape[0])


def _column(row: dict, name: str, T: int) -> np.ndarray:
    v = row.get(name)
    if v is None:
        return np.full(T, np.nan, dtype=np.float64)
    return np.asarray(v, dtype=np.float64).reshape(-1)[:T]


def load_holdout(keys: tuple[str, ...] = HOLDOUT_SHOTS) -> list[MastShot]:
    """Read the labelled MAST shots and build Psi_vac / I_p exactly as inference does."""
    from catskills.psivac import derived_psi, ip_on_efit
    from catskills.warm_feature_cache import row_from_parquet

    env = load_env("mast")
    out: list[MastShot] = []
    for key in keys:
        path = holdout_dir() / f"{key}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing; the labelled MAST demo shots ship in-repo")
        row = row_from_parquet(path)
        psi = np.asarray(row["efit_psirz"], dtype=np.float64)
        if psi.ndim == 1:
            # Stored flat; rows are Z and columns are R, matching mask_coarse.
            if psi.size % (65 * 65):
                raise RuntimeError(f"{key}: efit_psirz size {psi.size} is not a whole 65x65 stack")
            psi = psi.reshape(-1, 65, 65)
        elif psi.ndim == 3:
            psi = psi.astype(np.float64)
        else:
            raise RuntimeError(f"{key}: unexpected efit_psirz ndim {psi.ndim}")
        T = int(psi.shape[0])

        R = np.asarray(row["efit_grid_R"], dtype=np.float64).reshape(-1)
        Z = np.asarray(row["efit_grid_Z"], dtype=np.float64).reshape(-1)
        if not (np.allclose(R, env.grid_R, atol=1e-6) and np.allclose(Z, env.grid_Z, atol=1e-6)):
            raise RuntimeError(
                f"{key}: shot grid does not match the vendored MAST envelope; scoring it against "
                "mast_envelope.npz would compare different metres"
            )

        vac = np.asarray(derived_psi(row), dtype=np.float32)
        ip = np.asarray(ip_on_efit(row), dtype=np.float32)
        if vac.shape[0] != T:
            raise RuntimeError(f"{key}: Psi_vac has {vac.shape[0]} frames but psi_gt has {T}")
        out.append(
            MastShot(
                key=key,
                vac=vac,
                ip=ip,
                psi=psi,
                q95=_column(row, "efit_q95", T),
                betaN=_column(row, "efit_beta_n", T),
                times=_column(row, "efit_times", T),
            )
        )
    return out


def score_holdout(shots: list[MastShot], preds: list[dict], *, workers: int | None = None) -> dict:
    """Run the official composite on the labelled MAST frames.

    Reuses `local_score`'s reduction so the functionals are the vendored ones and the
    flux sign is a single fold-global bit, exactly as `metrics.finalize_machine` does it.
    """
    sys.path.insert(0, str(ROOT))
    from common import N_CONS, N_SCALARS, PSI_SIGNS, SCALARS
    from metrics import Accum, finalize_machine

    from local_score import build_reference, psi_residuals, score_shots

    env = load_env("mast")
    R, Z = env.grid_R, env.grid_Z
    mask_coarse = env.mask_coarse
    mask_f = mask_coarse.astype(np.float64)

    gts = [{"psi": s.psi, "q95": s.q95, "betaN": s.betaN} for s in shots]

    refs: list = []
    psi_sum = psi_sumsq = psi_n = 0.0
    scal_sum, scal_sumsq, scal_n = (np.zeros(N_SCALARS) for _ in range(3))
    cons_sum, cons_sumsq, cons_n = (np.zeros(N_CONS) for _ in range(3))
    for gt in gts:
        ref = build_reference(gt["psi"], R, Z, mask_coarse, mask_f, MACHINE)
        refs.append(ref)
        g = gt["psi"].astype(np.float64)
        fin = np.isfinite(g)
        psi_sum += float(g[fin].sum())
        psi_sumsq += float((g[fin] ** 2).sum())
        psi_n += int(fin.sum())
        for j, name in enumerate(SCALARS):
            v = np.asarray(gt[name], dtype=np.float64)
            v = v[np.isfinite(v)]
            scal_sum[j] += v.sum()
            scal_sumsq[j] += (v**2).sum()
            scal_n[j] += v.size
        cons_gt, cmask = ref[1], ref[2]
        for j in range(N_CONS):
            v = cons_gt[cmask[:, j], j]
            cons_sum[j] += v.sum()
            cons_sumsq[j] += (v**2).sum()
            cons_n[j] += v.size

    ref_stats = {
        "psi_sum": psi_sum, "psi_sumsq": psi_sumsq, "psi_n": psi_n,
        "scal_sum": scal_sum, "scal_sumsq": scal_sumsq, "scal_n": scal_n,
        "cons_sum": cons_sum, "cons_sumsq": cons_sumsq, "cons_n": cons_n,
    }
    mean_psi = psi_sum / psi_n if psi_n else 0.0
    means = (
        mean_psi,
        np.where(scal_n > 0, scal_sum / np.where(scal_n > 0, scal_n, 1), 0.0),
        np.where(cons_n > 0, cons_sum / np.where(cons_n > 0, cons_n, 1), 0.0),
    )

    totals = {s: 0.0 for s in PSI_SIGNS}
    for gt, p in zip(gts, preds):
        rr, _, _ = psi_residuals(gt["psi"], p["psirz"], mean_psi)
        for sg in PSI_SIGNS:
            totals[sg] += rr[sg]
    psi_sign = min(PSI_SIGNS, key=lambda sg: totals[sg])

    acc = Accum()
    acc.psi_sign = psi_sign
    for part in score_shots(
        gts, refs, preds, R, Z, mask_coarse, mask_f, psi_sign, means,
        machine=MACHINE, workers=workers, desc="mast",
    ):
        acc.add(part)
    res = finalize_machine(acc, ref_stats)
    res["n_frames"] = int(sum(s.T for s in shots))
    res["n_shots_holdout"] = len(shots)
    return res


# --------------------------------------------------------------------- label-free plausibility

# What the labelled frames actually show, so a predicted axis has something to miss.
AXIS_REF_R = 0.89
AXIS_REF_Z = 0.01


def plausibility(psi: np.ndarray, *, stride: int = 1, return_axes: bool = False) -> dict:
    """LCFS-closure rate and O-point placement for predicted MAST psi. No truth needed."""
    from derive import magnetic_axis
    from lcfs import extract_lcfs_with_sign

    env = load_env("mast")
    R, Z = env.grid_R, env.grid_Z
    mask_coarse = env.mask_coarse
    mask_f = mask_coarse.astype(np.float64)

    psi = np.asarray(psi, dtype=np.float64)
    idx = range(0, int(psi.shape[0]), max(1, int(stride)))
    n = 0
    n_closed = 0
    r_ax: list[float] = []
    z_ax: list[float] = []
    for k in idx:
        frame = psi[k]
        n += 1
        if not np.isfinite(frame).all():
            continue
        # axis_sign=None -> try both orientations; a miss here is orientation-independent.
        contour, sign = extract_lcfs_with_sign(
            frame, R, Z, MACHINE, mask_coarse, mask_f, n_points=512, axis_sign=None
        )
        if contour is not None and len(contour) >= 8:
            n_closed += 1
        ra, za, _, _ = magnetic_axis(frame, R, Z, MACHINE, mask_coarse, sign)
        if np.isfinite(ra) and np.isfinite(za):
            r_ax.append(float(ra))
            z_ax.append(float(za))

    def _q(v: list[float], name: str) -> dict:
        if not v:
            return {f"{name}_p50": float("nan"), f"{name}_p10": float("nan"), f"{name}_p90": float("nan")}
        a = np.asarray(v)
        return {
            f"{name}_p50": float(np.percentile(a, 50)),
            f"{name}_p10": float(np.percentile(a, 10)),
            f"{name}_p90": float(np.percentile(a, 90)),
        }

    out = {
        "frames": n,
        "lcfs_closed": n_closed,
        "lcfs_fail_frac": float(1.0 - n_closed / n) if n else float("nan"),
        "axis_found": len(r_ax),
    }
    out.update(_q(r_ax, "axis_R"))
    out.update(_q(z_ax, "axis_Z"))
    if r_ax:
        out["axis_R_err_p50"] = float(np.percentile(np.abs(np.asarray(r_ax) - AXIS_REF_R), 50))
        out["axis_Z_err_p50"] = float(np.percentile(np.abs(np.asarray(z_ax) - AXIS_REF_Z), 50))
    if return_axes:
        out["axis_R"] = r_ax
        out["axis_Z"] = z_ax
    return out
