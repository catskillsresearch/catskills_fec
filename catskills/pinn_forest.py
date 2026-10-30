"""Forest: shot-length recurrent encoder (``--arch shotseq``).

See MODEL_DESCRIPTION_CRC_FEQ_forest.md. The GRU sees one discharge; heads read h_t, not h_last.
"""
from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from catskills.lab_paths import MODEL_ID, results_dir
from catskills.model_version import git_full_sha, make_model_version
from catskills.pinn_charcoal import (
    CharcoalBundle,
    _ResBlock,
    _act,
    _clip_features,
    _geom_loss,
    _plausible_mask,
    derive_geom_weights,
    derive_stack_geom_weights,
)
from catskills.soft_derive import SCALAR_NAMES, SoftDerive, SoftDeriveConfig
from catskills.pinn_teal import make_optimizer

def _revision_dir() -> Path:
    return results_dir() / MODEL_ID.lower().removeprefix("crc_feq_")


def _gpu_log_path() -> Path:
    return _revision_dir() / "loop" / "gpu_log.jsonl"


def _nv_snapshot() -> dict:
    """One nvidia-smi + torch allocator sample. Cheap enough for a few times per epoch."""
    rec: dict = {"t": time.strftime("%H:%M:%S")}
    if torch.cuda.is_available():
        rec["alloc_gi"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
        rec["peak_gi"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
             "--format=csv,noheader,nounits"],
            text=True, timeout=2,
        ).strip().split(",")
        rec["util"] = int(float(out[0]))
        rec["vram_gi"] = round(float(out[1]) / 1024, 2)
        rec["vram_tot"] = round(float(out[2]) / 1024, 2)
        rec["pwr"] = round(float(out[3]), 0)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    return rec


def _log_gpu(tag: str, ep: int, **extra) -> dict:
    rec = _nv_snapshot()
    rec.update({"tag": tag, "epoch": int(ep), **extra})
    path = _gpu_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    bits = [f"gpu ep={ep}"]
    if extra.get("pack") is not None:
        bits.append(f"pack={extra['pack']}/{extra.get('n_pack', '?')}")
    if "util" in rec:
        bits.append(f"util={rec['util']}%")
    if "vram_gi" in rec:
        bits.append(f"vram={rec['vram_gi']}/{rec.get('vram_tot', '?')}GiB")
    if "pwr" in rec:
        bits.append(f"pwr={rec['pwr']:.0f}W")
    if extra.get("chunk") is not None:
        bits.append(f"chunk={extra['chunk']}")
    print(f"[{MODEL_ID}] " + "  ".join(bits), flush=True)
    return rec


class CoeffShotSeq(torch.nn.Module):
    """Bidirectional GRU over a whole shot, MLP + charcoal heads on each frame."""

    def __init__(
        self,
        n_in: int,
        n_pca: int,
        hidden: int = 1024,
        layers: int = 3,
        dropout: float = 0.05,
        bidirectional: bool = True,
        mlp_blocks: int = 2,
        activation: str = "relu",
    ):
        super().__init__()
        self.n_in = int(n_in)
        self.hidden_n = int(hidden)
        self.layers_n = int(layers)
        self.bidirectional = bool(bidirectional)
        self.dropout_p = float(dropout)
        self.mlp_blocks_n = int(mlp_blocks)
        self.activation_name = activation
        self.gru = torch.nn.GRU(
            self.n_in, self.hidden_n, num_layers=self.layers_n, batch_first=True,
            dropout=self.dropout_p if self.layers_n > 1 else 0.0,
            bidirectional=self.bidirectional,
        )
        h_out = self.hidden_n * (2 if self.bidirectional else 1)
        self.h_out = h_out
        blocks = [_ResBlock(h_out, self.dropout_p, activation) for _ in range(self.mlp_blocks_n)]
        self.mlp = torch.nn.Sequential(*blocks) if blocks else torch.nn.Identity()
        self.coeff_head = torch.nn.Linear(h_out, n_pca)
        self.q95_head = torch.nn.Linear(h_out, 1)
        self.beta_head = torch.nn.Linear(h_out, 1)
        self.r_axis_head = torch.nn.Linear(h_out, 1)
        self.z_axis_head = torch.nn.Linear(h_out, 1)
        self.li_head = torch.nn.Linear(h_out, 1)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``x`` is ``(T, F)`` one shot or ``(B, T, F)`` padded. Lengths are true T per row."""
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)
        if lengths is None:
            lengths = torch.full((x.shape[0],), x.shape[1], device=x.device, dtype=torch.long)
        lengths = lengths.to(dtype=torch.long, device="cpu")
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False,
        )
        packed_out, _ = self.gru(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.shape[1],
        )
        h = self.mlp(out)
        z = self.coeff_head(h)
        q = self.q95_head(h).squeeze(-1)
        b = self.beta_head(h).squeeze(-1)
        ra = self.r_axis_head(h).squeeze(-1)
        za = self.z_axis_head(h).squeeze(-1)
        li = self.li_head(h).squeeze(-1)
        if squeeze:
            return z[0], q[0], b[0], ra[0], za[0], li[0]
        return z, q, b, ra, za, li


def peak_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated()) / (1024 ** 3)


def try_fill(
    *,
    n_in: int,
    n_pca: int,
    hidden: int,
    layers: int,
    bidirectional: bool,
    mlp_blocks: int,
    T: int,
    token_budget: int,
    device: torch.device,
) -> tuple[bool, float, int, str]:
    """One AdamW step on a padded batch that includes a length-``T`` shot. Returns ok, peak GB, n_params, err."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = CoeffShotSeq(
        n_in, n_pca, hidden=hidden, layers=layers, bidirectional=bidirectional,
        mlp_blocks=mlp_blocks,
    ).to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Pack: one long shot plus as many 241-frame shots as the budget allows.
    lengths = [T]
    left = max(int(token_budget) - T, 0)
    while left >= 64:
        take = min(241, left)
        lengths.append(take)
        left -= take
    B = len(lengths)
    Tm = max(lengths)
    x = torch.randn(B, Tm, n_in, device=device)
    z_gt = torch.randn(B, Tm, n_pca, device=device)
    lens = torch.tensor(lengths, dtype=torch.long)
    try:
        model.train()
        z, q, b, *_ = model(x, lens)
        mask = torch.zeros(B, Tm, dtype=torch.bool, device=device)
        for i, L in enumerate(lengths):
            mask[i, :L] = True
        loss = ((z[mask] - z_gt[mask]) ** 2).mean() + 0.01 * (q[mask].square().mean() + b[mask].square().mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize()
        mem = peak_gb()
        del model, opt, x, z_gt, z, q, b, loss
        torch.cuda.empty_cache()
        return True, mem, n_params, ""
    except RuntimeError as exc:
        err = str(exc).split("\n", 1)[0]
        del model
        torch.cuda.empty_cache()
        return False, peak_gb(), n_params, err


def _group_shots(
    Xs: np.ndarray, Zc: np.ndarray, q95: np.ndarray, beta: np.ndarray, shot_ids: np.ndarray,
) -> list[dict]:
    shots: list[dict] = []
    for s in np.unique(shot_ids):
        sl = np.flatnonzero(shot_ids == s)
        shots.append({
            "X": np.ascontiguousarray(Xs[sl]),
            "Z": np.ascontiguousarray(Zc[sl]),
            "q": np.ascontiguousarray(q95[sl]),
            "b": np.ascontiguousarray(beta[sl]),
        })
    return shots


def _packs(order: list[int], lengths: list[int], budget: int) -> list[list[int]]:
    packs: list[list[int]] = []
    cur: list[int] = []
    tok = 0
    for i in order:
        L = int(lengths[i])
        if cur and tok + L > budget:
            packs.append(cur)
            cur, tok = [], 0
        cur.append(i)
        tok += L
    if cur:
        packs.append(cur)
    return packs


def _pad_pack(shots: list[dict], idx: list[int], device: torch.device):
    Ts = [int(shots[i]["X"].shape[0]) for i in idx]
    T = max(Ts)
    B = len(idx)
    F = int(shots[idx[0]]["X"].shape[1])
    K = int(shots[idx[0]]["Z"].shape[1])
    X = torch.zeros(B, T, F, device=device)
    Z = torch.zeros(B, T, K, device=device)
    q = torch.zeros(B, T, device=device)
    b = torch.zeros(B, T, device=device)
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    for row, i in enumerate(idx):
        t = Ts[row]
        X[row, :t] = torch.from_numpy(shots[i]["X"]).to(device, non_blocking=True)
        Z[row, :t] = torch.from_numpy(shots[i]["Z"]).to(device, non_blocking=True)
        q[row, :t] = torch.from_numpy(shots[i]["q"]).to(device, non_blocking=True)
        b[row, :t] = torch.from_numpy(shots[i]["b"]).to(device, non_blocking=True)
        mask[row, :t] = True
    lengths = torch.tensor(Ts, dtype=torch.long)
    return X, Z, q, b, mask, lengths


def train_shotseq_from_frames(
    *,
    Xs: np.ndarray,
    Zc: np.ndarray,
    q95: np.ndarray,
    beta: np.ndarray,
    frame_shot_idx: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    pca,
    n_raw_feat: int,
    n_comp: int,
    device: torch.device,
    out: Path,
    hidden: int,
    layers: int,
    bidirectional: bool,
    mlp_blocks: int,
    token_budget: int,
    dropout: float,
    activation: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    optimizer: str,
    lr_warmup: int,
    grad_clip: float,
    feature_clip: float,
    seed: int,
    init_seed: int | None,
    val_fraction: float,
    early_stop_patience: int,
    log_every: int,
    ckpt_every: int,
    lambda_qb: float,
    qb_normalize: bool,
    qb_emphasis: float,
    lambda_coeff_white: float,
    coeff_white_power: float,
    coeff_white_clamp: int,
    include_thomson: bool,
    cache_keys,
    cache_stats: dict,
    var_psi: float | None = None,
) -> CharcoalBundle:
    """Shot-batched AdamW loop. Geometry and L-BFGS are off — see the forest description §4."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed if init_seed is None else init_seed))
    uniq = np.unique(frame_shot_idx)
    shots = _group_shots(Xs, Zc, q95, beta, frame_shot_idx)
    shot_key = [int(s) for s in uniq]
    key_to_i = {k: i for i, k in enumerate(shot_key)}
    lengths = [int(s["X"].shape[0]) for s in shots]
    n_shots = len(shots)
    n_total = int(sum(lengths))

    val_ids: list[int] = []
    if val_fraction > 0 and n_shots > 1:
        order = np.random.default_rng(int(seed)).permutation(uniq)
        target = n_total * float(val_fraction)
        counts = {int(s): int((frame_shot_idx == s).sum()) for s in uniq}
        taken = 0
        held: list[int] = []
        for s in order:
            if taken >= target:
                break
            held.append(int(s))
            taken += counts[int(s)]
        val_ids = [key_to_i[s] for s in held if s in key_to_i]
    train_ids = [i for i in range(n_shots) if i not in set(val_ids)]
    print(
        f"[{MODEL_ID}] shotseq  shots={n_shots} frames={n_total}  "
        f"train={len(train_ids)} val={len(val_ids)}  "
        f"T=[{min(lengths)},{max(lengths)}] median={int(np.median(lengths))}  "
        f"token_budget={int(token_budget)}",
        flush=True,
    )

    model = CoeffShotSeq(
        int(Xs.shape[1]), n_comp, hidden=hidden, layers=layers, dropout=dropout,
        bidirectional=bidirectional, mlp_blocks=mlp_blocks, activation=activation,
    ).to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    print(
        f"[{MODEL_ID}] trunk=shotseq params={n_params:,}  "
        f"gru={hidden}x{layers} {'bi' if bidirectional else 'uni'} "
        f"mlp_blocks={mlp_blocks} split={n_raw_feat}",
        flush=True,
    )
    opt = make_optimizer(list(model.parameters()), name=optimizer, lr=lr, weight_decay=weight_decay)
    if lr_warmup > 0:
        w = min(int(lr_warmup), max(epochs - 1, 1))

        def _lr_at(ep: int) -> float:
            if ep < w:
                return (ep + 1) / float(w + 1)
            frac = (ep - w) / max(epochs - w, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(frac, 1.0)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_at)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    pca_mean = torch.from_numpy(np.asarray(pca.mean_, dtype=np.float32)).to(device)
    pca_comp = torch.from_numpy(np.asarray(pca.components_, dtype=np.float32)).to(device)
    z_all = torch.from_numpy(Zc)
    coeff_sd = z_all.std(dim=0).clamp_min(1e-12).to(device)
    _clamp_i = min(max(int(coeff_white_clamp), 1), int(coeff_sd.numel())) - 1
    coeff_scale = coeff_sd.clamp_min(coeff_sd[_clamp_i]) ** float(coeff_white_power)
    if lambda_coeff_white > 0:
        ww = (coeff_scale[0] / coeff_scale) ** 2
        print(
            f"[{MODEL_ID}] tempered coeff term: lambda={lambda_coeff_white:g} "
            f"power={coeff_white_power:g} clamp=mode{_clamp_i + 1}  "
            f"relative mode weight: 1 -> {float(ww[0]):.1f}, "
            f"{_clamp_i + 1} -> {float(ww[_clamp_i]):.1f}",
            flush=True,
        )
    sig_q = sig_b = 1.0
    if qb_normalize:
        sig_q = float(torch.from_numpy(q95).std().clamp_min(1e-6))
        sig_b = float(torch.from_numpy(beta).std().clamp_min(1e-6))
        print(f"[{MODEL_ID}] qb normalized: sigma_q95={sig_q:.4g} sigma_betaN={sig_b:.4g}",
              flush=True)

    # Var(psi) from a sample of reconstructed grids so the score-unit log matches charcoal.
    if var_psi is None:
        with torch.no_grad():
            sl = torch.randperm(min(len(Zc), 8192))[:4096]
            psi = (z_all[sl].to(device) @ pca_comp + pca_mean).view(-1, 65, 65)
            var_psi = float(psi.var())
    print(f"[{MODEL_ID}] Var(psi)={var_psi:.4g}  geom_emphasis=0  ->  lambda_geom=0", flush=True)
    print(f"[{MODEL_ID}] L-BFGS skipped for arch=shotseq: cuDNN GRU cannot backward in eval",
          flush=True)

    def _batch_tensors(ids: list[int]):
        return _pad_pack(shots, ids, device)

    def _loss(xb, zb, qb, bb, mask, lengths):
        z_hat, q_hat, b_hat, *_ = model(xb, lengths)
        z_m, zt = z_hat[mask], zb[mask]
        psi_h = (z_m @ pca_comp + pca_mean).view(-1, 65, 65)
        psi_t = (zt @ pca_comp + pca_mean).view(-1, 65, 65)
        L_data = ((psi_h - psi_t) ** 2).mean()
        L_coeff = ((z_m - zt) ** 2).mean()
        if lambda_coeff_white > 0:
            L_cw = (((z_m - zt) / coeff_scale) ** 2).mean()
        else:
            L_cw = z_m.new_zeros(())
        L_qb = 0.5 * (((q_hat[mask] - qb[mask]) / sig_q) ** 2).mean() + \
               0.5 * (((b_hat[mask] - bb[mask]) / sig_b) ** 2).mean()
        loss = L_data + 0.25 * L_coeff + lambda_coeff_white * L_cw + max(lambda_qb, 0.0) * L_qb
        return loss, L_data, float(((psi_h - psi_t) ** 2).mean())

    @torch.no_grad()
    def _validate() -> tuple[float, float]:
        if not val_ids:
            return float("nan"), float("nan")
        model.eval()
        tot, n = 0.0, 0
        for pack in _packs(val_ids, lengths, int(token_budget)):
            xb, zb, qb, bb, mask, lens = _batch_tensors(pack)
            z_hat = model(xb, lens)[0]
            z_m, zt = z_hat[mask], zb[mask]
            psi_h = (z_m @ pca_comp + pca_mean).view(-1, 65, 65)
            psi_t = (zt @ pca_comp + pca_mean).view(-1, 65, 65)
            tot += float(((psi_h - psi_t) ** 2).sum())
            n += int(mask.sum())
        model.train()
        mse = tot / max(n, 1)
        r2 = 1.0 - mse / max(var_psi, 1e-12)
        return mse, r2

    best_sel = -float("inf")
    best_state = None
    best_epoch = 0
    history: list[dict] = []
    t0 = time.perf_counter()
    partial_path = Path(str(out) + ".partial") if not str(out).endswith(".pt") else \
        Path(str(out).replace(".pt", ".partial.pt"))

    for ep in range(1, epochs + 1):
        model.train()
        order = train_ids[:]
        rng.shuffle(order)
        packs = _packs(order, lengths, int(token_budget))
        running = 0.0
        n_step = 0
        t_ep = time.perf_counter()
        for pack in packs:
            xb, zb, qb, bb, mask, lens = _batch_tensors(pack)
            opt.zero_grad(set_to_none=True)
            loss, L_data, _ = _loss(xb, zb, qb, bb, mask, lens)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()
            running += float(loss.detach())
            n_step += 1
        sched.step()
        val_mse, val_r2 = _validate()
        # Selector is 0.55 * R2_psi when geometry is off — same units as charcoal without Cons.
        sel = 0.55 * (val_r2 if np.isfinite(val_r2) else 0.0)
        if sel > best_sel:
            best_sel = sel
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append({"epoch": ep, "loss": running / max(n_step, 1),
                        "val_psi_mse": val_mse, "val_r2_psi": val_r2, "selection": sel})
        sec = time.perf_counter() - t_ep
        if ep == 1 or ep == epochs or ep % max(int(log_every), 1) == 0:
            eta = ""
            if ep < epochs and sec > 0:
                left = (epochs - ep) * sec
                eta = f"  eta {time.strftime('%H:%M', time.localtime(time.time() + left))}"
            print(
                f"[{MODEL_ID}] {time.strftime('%H:%M:%S')} epoch {ep}/{epochs}  "
                f"loss={running / max(n_step, 1):.4e}  val_ψ={val_mse:.4e}  "
                f"val_R2ψ={val_r2:.4f}  {sec:.1f}s/ep{eta}",
                flush=True,
            )
        if ckpt_every > 0 and ep % int(ckpt_every) == 0 and best_state is not None:
            torch.save({"model_state": best_state, "arch": "shotseq", "partial": True,
                        "epoch": ep}, partial_path)

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[{MODEL_ID}] restored best selection={best_sel:.4f} @ epoch {best_epoch}",
              flush=True)

    train_s = time.perf_counter() - t0
    tp = {
        "n_train": len(cache_keys) if cache_keys is not None else n_shots,
        "n_train_frames": n_total,
        "n_train_shots": n_shots,
        "epochs": int(epochs),
        "epochs_ran": int(epochs),
        "token_budget": int(token_budget),
        "lr": float(lr),
        "optimizer": str(optimizer),
        "arch": "shotseq",
        "seq_hidden": int(hidden),
        "seq_layers": int(layers),
        "shot_bidirectional": bool(bidirectional),
        "shot_mlp_blocks": int(mlp_blocks),
        "n_params": n_params,
        "n_pca": int(n_comp),
        "temporal_window": 0,
        "lambda_geom": 0.0,
        "geom_emphasis": 0.0,
        "lambda_coeff_white": float(lambda_coeff_white),
        "coeff_white_power": float(coeff_white_power),
        "coeff_white_clamp": int(coeff_white_clamp),
        "var_psi": float(var_psi),
        "grad_clip": float(grad_clip),
        "lr_warmup": int(lr_warmup),
        "val_fraction": float(val_fraction),
        "best_epoch": int(best_epoch),
        "best_selection": float(best_sel),
        "lbfgs_steps_ran": 0,
        "seed": int(seed),
        "init_seed": int(seed if init_seed is None else init_seed),
        "device": torch.cuda.get_device_name(0),
        "train_s": float(train_s),
        "feature_cache_hits": int((cache_stats or {}).get("hits", 0)),
        "git_commit": git_full_sha(),
    }
    ckpt = {
        "model_state": model.state_dict(),
        "residual_state": None,
        "n_features": int(Xs.shape[1]),
        "n_raw_features": int(n_raw_feat),
        "feature_mean": mean,
        "feature_std": std,
        "include_thomson": include_thomson,
        "n_pca": int(n_comp),
        "use_residual": False,
        "temporal_window": 0,
        "temporal_mode": "none",
        "hidden": [int(hidden)],
        "activation": activation,
        "arch": "shotseq",
        "arch_config": {
            "hidden": int(hidden), "layers": int(layers),
            "bidirectional": bool(bidirectional), "mlp_blocks": int(mlp_blocks),
            "dropout": float(dropout), "n_split": int(n_raw_feat),
            "seq_hidden": int(hidden), "seq_layers": int(layers),
        },
        "feature_clip": float(feature_clip),
        "pca_mean": pca.mean_.astype(np.float32),
        "pca_components": pca.components_.astype(np.float32),
        "training_params": tp,
        "history": history,
        "method": MODEL_ID,
        "model_version": make_model_version(MODEL_ID),
        "git_commit": tp["git_commit"],
    }
    torch.save(ckpt, out)
    Path(partial_path).unlink(missing_ok=True)
    print(f"[{MODEL_ID}] wrote {out}  train_s={train_s:.1f}s", flush=True)
    return CharcoalBundle(
        checkpoint=Path(out),
        n_features=int(Xs.shape[1]),
        feature_mean=mean,
        feature_std=std,
        include_thomson=include_thomson,
        n_pca=int(n_comp),
        device_name=tp["device"],
        n_raw_features=int(n_raw_feat),
        temporal_window=0,
        temporal_mode="none",
        seq_window=0,
        training_params=tp,
        model_version=ckpt["model_version"],
        pca_mean=ckpt["pca_mean"],
        pca_components=ckpt["pca_components"],
        hidden=(int(hidden),),
        activation=activation,
        arch="shotseq",
        feature_clip=float(feature_clip),
    )


def _group_dual_shots(
    Xs_dual: np.ndarray, Xs_raw: np.ndarray, Zc: np.ndarray,
    q95: np.ndarray, beta: np.ndarray, shot_ids: np.ndarray,
    G: np.ndarray | None = None,
    OK: np.ndarray | None = None,
) -> list[dict]:
    shots: list[dict] = []
    for s in np.unique(shot_ids):
        sl = np.flatnonzero(shot_ids == s)
        rec = {
            "D": np.ascontiguousarray(Xs_dual[sl]),
            "R": np.ascontiguousarray(Xs_raw[sl]),
            "Z": np.ascontiguousarray(Zc[sl]),
            "q": np.ascontiguousarray(q95[sl]),
            "b": np.ascontiguousarray(beta[sl]),
        }
        if G is not None:
            rec["G"] = np.ascontiguousarray(G[sl])
            rec["OK"] = np.ascontiguousarray(OK[sl])
        shots.append(rec)
    return shots


def _row_copy(dst: torch.Tensor, src, t: int) -> None:
    if torch.is_tensor(src):
        dst[:t] = src[:t]
    else:
        dst[:t] = torch.from_numpy(src).to(dst.device, non_blocking=True)


def _shots_to_device(shots: list[dict], device: torch.device) -> None:
    """Keep the corpus on GPU. 16 GB holds features+coeffs; 12 GB did not."""
    keys = ("D", "R", "Z", "q", "b", "G", "OK", "X")
    nbytes = 0
    for rec in shots:
        for k in keys:
            if k not in rec:
                continue
            t = rec[k]
            if not torch.is_tensor(t):
                t = torch.from_numpy(np.ascontiguousarray(t))
            rec[k] = t.to(device, non_blocking=True)
            nbytes += rec[k].nbytes
    print(f"[{MODEL_ID}] shots on {device} ({nbytes / 1e9:.2f} GB)", flush=True)


def _pad_dual(shots: list[dict], idx: list[int], device: torch.device):
    Ts = [int(shots[i]["D"].shape[0]) for i in idx]
    T = max(Ts)
    B = len(idx)
    Fd = int(shots[idx[0]]["D"].shape[1])
    Fr = int(shots[idx[0]]["R"].shape[1])
    K = int(shots[idx[0]]["Z"].shape[1])
    D = torch.zeros(B, T, Fd, device=device)
    R = torch.zeros(B, T, Fr, device=device)
    Z = torch.zeros(B, T, K, device=device)
    q = torch.zeros(B, T, device=device)
    b = torch.zeros(B, T, device=device)
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    for row, i in enumerate(idx):
        t = Ts[row]
        _row_copy(D[row], shots[i]["D"], t)
        _row_copy(R[row], shots[i]["R"], t)
        _row_copy(Z[row], shots[i]["Z"], t)
        _row_copy(q[row], shots[i]["q"], t)
        _row_copy(b[row], shots[i]["b"], t)
        mask[row, :t] = True
    if "G" not in shots[idx[0]]:
        return D, R, Z, q, b, mask, torch.tensor(Ts, dtype=torch.long), None, None
    Ng = int(shots[idx[0]]["G"].shape[1])
    G = torch.zeros(B, T, Ng, device=device)
    OK = torch.zeros(B, T, Ng, dtype=torch.bool, device=device)
    for row, i in enumerate(idx):
        t = Ts[row]
        _row_copy(G[row], shots[i]["G"], t)
        _row_copy(OK[row], shots[i]["OK"], t)
    return D, R, Z, q, b, mask, torch.tensor(Ts, dtype=torch.long), G, OK


def train_dualshot_from_frames(
    *,
    Xs_dual: np.ndarray,
    Xs_raw: np.ndarray,
    Zc: np.ndarray,
    q95: np.ndarray,
    beta: np.ndarray,
    frame_shot_idx: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    raw_mean: np.ndarray,
    raw_std: np.ndarray,
    pca,
    n_raw_feat: int,
    n_comp: int,
    device: torch.device,
    out: Path,
    hidden: int,
    layers: int,
    bidirectional: bool,
    token_budget: int,
    dropout: float,
    activation: str,
    trunk_width: int,
    trunk_blocks: int,
    ctx_width: int,
    ctx_blocks: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    optimizer: str,
    lr_warmup: int,
    grad_clip: float,
    feature_clip: float,
    seed: int,
    init_seed: int | None,
    val_fraction: float,
    log_every: int,
    ckpt_every: int,
    lambda_qb: float,
    qb_normalize: bool,
    lambda_coeff_white: float,
    coeff_white_power: float,
    coeff_white_clamp: int,
    include_thomson: bool,
    cache_keys,
    cache_stats: dict,
    init_ck: dict | None,
    freeze_dualctx: bool,
    temporal_window: int,
    temporal_mode: str,
    lambda_geom: float = -1.0,
    geom_emphasis: float = 0.0,
    geom_weights: str = "uniform",
    geom_weight_target: float = 0.95,
    geom_weight_floor: float = 0.25,
    geom_kappa_mult: float = 1.0,
    geom_raxis_mult: float = 1.0,
    geom_zaxis_mult: float = 1.0,
    geom_emphasis_final: float = 1.0,
    geom_anneal_tail: float = 0.25,
    geom_huber: float = 3.0,
    geom_frac: float = 1.0,
    geom_warmup: int = 10,
    geom_sd_chunk: int = 256,
    edge_weight: str = "off",
    edge_mix: float = 0.6,
    edge_power: float = 1.0,
    edge_band_lo: float = 0.70,
    edge_band_hi: float = 1.05,
    probe_epochs: int = 0,
    tau_hi: float = 0.05,
    tau_lo: float = 0.005,
    tau_li_hi: float = 0.08,
    tau_li_lo: float = 0.02,
    anneal_frac: float = 0.5,
    resume: Path | None = None,
    geom_ramp: int = 0,
    gradnorm: bool = False,
    gradnorm_alpha: float = 0.5,
    gradnorm_w_from: int = 10,
    pcgrad: bool = False,
    stack_li: bool = False,
    lp_floor: float = 0.0,
    kill_r2psi: float = 0.0,
    kill_r2psi_after: int = 8,
    kill_cons: float = 0.0,
    kill_cons_after: int = 8,
) -> CharcoalBundle:
    """Shot-batched C7b + gated shot-GRU. L-BFGS stays off. Geometry is C9a-style when emphasis > 0."""
    from catskills.pinn_charcoal import CoeffDualShot

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed if init_seed is None else init_seed))
    uniq = np.unique(frame_shot_idx)
    shots = _group_dual_shots(Xs_dual, Xs_raw, Zc, q95, beta, frame_shot_idx)
    key_to_i = {int(s): i for i, s in enumerate(uniq)}
    lengths = [int(s["D"].shape[0]) for s in shots]
    n_shots = len(shots)
    n_total = int(sum(lengths))

    val_ids: list[int] = []
    if val_fraction > 0 and n_shots > 1:
        order = np.random.default_rng(int(seed)).permutation(uniq)
        target = n_total * float(val_fraction)
        counts = {int(s): int((frame_shot_idx == s).sum()) for s in uniq}
        taken = 0
        held: list[int] = []
        for s in order:
            if taken >= target:
                break
            held.append(int(s))
            taken += counts[int(s)]
        val_ids = [key_to_i[s] for s in held if s in key_to_i]
    train_ids = [i for i in range(n_shots) if i not in set(val_ids)]
    print(
        f"[{MODEL_ID}] dualshot  shots={n_shots} frames={n_total}  "
        f"train={len(train_ids)} val={len(val_ids)}  "
        f"T=[{min(lengths)},{max(lengths)}] median={int(np.median(lengths))}  "
        f"token_budget={int(token_budget)}",
        flush=True,
    )

    model = CoeffDualShot(
        int(Xs_dual.shape[1]), n_comp, n_raw_feat,
        width=trunk_width, blocks=trunk_blocks, ctx_width=ctx_width, ctx_blocks=ctx_blocks,
        dropout=dropout, activation=activation,
        seq_hidden=hidden, seq_layers=layers, bidirectional=bidirectional,
    ).to(device)
    if init_ck is not None and init_ck.get("model_state") is not None:
        missing, unexpected = model.load_state_dict(init_ck["model_state"], strict=False)
        print(f"[{MODEL_ID}] init missing keys {list(missing)[:8]}…", flush=True)
        if unexpected:
            print(f"[{MODEL_ID}] init unexpected keys {list(unexpected)[:6]}…", flush=True)
    if freeze_dualctx:
        keep = {"gru", "seq_proj", "gate_seq"}
        for name, p in model.named_parameters():
            if name.split(".", 1)[0] not in keep:
                p.requires_grad = False
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        print(f"[{MODEL_ID}] freeze-dualctx: training {n_tr:,} / {n_all:,} params", flush=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = int(sum(p.numel() for p in model.parameters()))
    print(
        f"[{MODEL_ID}] trunk=dualshot params={n_params:,}  "
        f"gru={hidden}x{layers} {'bi' if bidirectional else 'uni'} gate_seq=0",
        flush=True,
    )
    opt = make_optimizer(trainable, name=optimizer, lr=lr, weight_decay=weight_decay)
    if lr_warmup > 0:
        w = min(int(lr_warmup), max(epochs - 1, 1))

        def _lr_at(ep: int) -> float:
            if ep < w:
                return (ep + 1) / float(w + 1)
            frac = (ep - w) / max(epochs - w, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(frac, 1.0)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_at)
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    pca_mean = torch.from_numpy(np.asarray(pca.mean_, dtype=np.float32)).to(device)
    pca_comp = torch.from_numpy(np.asarray(pca.components_, dtype=np.float32)).to(device)
    z_all = torch.from_numpy(Zc)
    coeff_sd = z_all.std(dim=0).clamp_min(1e-12).to(device)
    _clamp_i = min(max(int(coeff_white_clamp), 1), int(coeff_sd.numel())) - 1
    coeff_scale = coeff_sd.clamp_min(coeff_sd[_clamp_i]) ** float(coeff_white_power)
    sig_q = sig_b = 1.0
    if qb_normalize:
        sig_q = float(torch.from_numpy(q95).std().clamp_min(1e-6))
        sig_b = float(torch.from_numpy(beta).std().clamp_min(1e-6))
    with torch.no_grad():
        sl = torch.randperm(min(len(Zc), 8192))[:4096]
        psi = (z_all[sl].to(device) @ pca_comp + pca_mean).view(-1, 65, 65)
        var_psi = float(psi.var())
    W_PSI, W_CONS = 0.55, 0.20
    if lambda_geom < 0:
        lambda_geom = float(geom_emphasis * (W_CONS / W_PSI) * var_psi)
    use_geom = lambda_geom > 0
    print(
        f"[{MODEL_ID}] Var(psi)={var_psi:.4g}  geom_emphasis={geom_emphasis:g}"
        f"  ->  lambda_geom={lambda_geom:.4g}",
        flush=True,
    )
    print(f"[{MODEL_ID}] L-BFGS skipped for arch=dualshot: cuDNN GRU cannot backward in eval",
          flush=True)
    if edge_weight != "off" and edge_mix > 0:
        print(
            f"[{MODEL_ID}] edge weight: {edge_weight} mix={edge_mix:g} "
            f"power={edge_power:g} band=[{edge_band_lo:g},{edge_band_hi:g}]",
            flush=True,
        )

    sd = None
    sigma = None
    geom_w = None
    if use_geom:
        root = Path(__file__).resolve().parents[1]
        env = np.load(root / "fusion_scoring" / "masks" / "d3d_envelope.npz")
        sd = SoftDerive(
            env["grid_R"], env["grid_Z"], env["mask_coarse"].astype(bool),
            cfg=SoftDeriveConfig(lp_floor=float(lp_floor)),
        ).to(device)
        g_chunks = []
        with torch.no_grad():
            for start in range(0, int(Zc.shape[0]), 512):
                z = torch.from_numpy(Zc[start:start + 512]).to(device)
                psi = (z @ pca_comp + pca_mean).view(-1, 65, 65)
                g = sd(psi)
                g_chunks.append(torch.stack([g[k] for k in SCALAR_NAMES], dim=1))
        g_gt = torch.cat(g_chunks, dim=0)
        ok_gt = _plausible_mask(g_gt)
        sigma = torch.stack(
            [
                g_gt[ok_gt[:, j], j].std() if int(ok_gt[:, j].sum()) > 1
                else torch.ones((), device=device)
                for j in range(g_gt.shape[1])
            ]
        ).clamp_min(1e-6)
        if geom_weights == "deficit":
            geom_w = torch.tensor(
                derive_geom_weights(geom_weight_target, geom_weight_floor), device=device
            )
            if geom_kappa_mult != 1.0:
                ik = SCALAR_NAMES.index("kappa")
                geom_w = geom_w.clone()
                geom_w[ik] = geom_w[ik] * float(geom_kappa_mult)
                geom_w = geom_w / geom_w.mean()
            if geom_raxis_mult != 1.0:
                ir = SCALAR_NAMES.index("R_axis")
                geom_w = geom_w.clone()
                geom_w[ir] = geom_w[ir] * float(geom_raxis_mult)
                geom_w = geom_w / geom_w.mean()
            if geom_zaxis_mult != 1.0:
                iz = SCALAR_NAMES.index("Z_axis")
                geom_w = geom_w.clone()
                geom_w[iz] = geom_w[iz] * float(geom_zaxis_mult)
                geom_w = geom_w / geom_w.mean()
            print(
                f"[{MODEL_ID}] geometry weights (target {geom_weight_target:g}, "
                f"floor {geom_weight_floor:g}, kappa×{geom_kappa_mult:g}, "
                f"R_axis×{geom_raxis_mult:g}, Z_axis×{geom_zaxis_mult:g}): "
                + "  ".join(f"{n}={float(geom_w[j]):.3f}" for j, n in enumerate(SCALAR_NAMES)),
                flush=True,
            )
        elif geom_weights == "stack":
            from catskills.olive_loss import OLIVE_STACK_FOCUS
            focus = OLIVE_STACK_FOCUS if stack_li else None
            geom_w = torch.tensor(derive_stack_geom_weights(focus=focus), device=device)
            label = "stack κ/Z_axis/volume/ℓ_i" if stack_li else "stack κ/Z_axis/volume"
            print(
                f"[{MODEL_ID}] geometry weights ({label}): "
                + "  ".join(f"{n}={float(geom_w[j]):.3f}" for j, n in enumerate(SCALAR_NAMES)),
                flush=True,
            )
        elif geom_weights != "uniform":
            raise ValueError(
                f"geom_weights must be 'uniform', 'deficit' or 'stack', got {geom_weights!r}"
            )
        G_np = g_gt.detach().cpu().numpy().astype(np.float32)
        OK_np = ok_gt.detach().cpu().numpy()
        for s, rec in zip(np.unique(frame_shot_idx), shots):
            sl = np.flatnonzero(frame_shot_idx == s)
            rec["G"] = np.ascontiguousarray(G_np[sl])
            rec["OK"] = np.ascontiguousarray(OK_np[sl])
        del g_gt, ok_gt, g_chunks, G_np, OK_np
        print(f"[{MODEL_ID}] geometry targets on {n_total} PCA-decoded frames", flush=True)
    _shots_to_device(shots, device)

    def _geom_scale(ep: int) -> float:
        from catskills.olive_loss import geom_ramp_mult
        ramp = geom_ramp_mult(ep, int(geom_warmup), int(geom_ramp))
        if geom_emphasis_final >= 1.0 or geom_anneal_tail <= 0 or epochs <= 1:
            return ramp
        start = (1.0 - geom_anneal_tail) * epochs
        if ep <= start:
            return ramp
        f = min(1.0, (ep - start) / max(1.0, geom_anneal_tail * epochs))
        return ramp * float(geom_emphasis_final ** f)

    def _taus(ep: int) -> tuple[float, float]:
        if epochs <= 1 or anneal_frac <= 0:
            return tau_lo, tau_li_lo
        f = min(1.0, (ep - 1) / max(1.0, anneal_frac * epochs))
        return (
            float(tau_hi * (tau_lo / tau_hi) ** f),
            float(tau_li_hi * (tau_li_lo / tau_li_hi) ** f),
        )

    def _loss(xd, xr, zb, qb, bb, mask, lengths, gb, okb, want_geom, gscale, tau, tau_li):
        z_hat, q_hat, b_hat, *_ = model(xd, xr, lengths)
        z_m, zt = z_hat[mask], zb[mask]
        psi_h = (z_m @ pca_comp + pca_mean).view(-1, 65, 65)
        psi_t = (zt @ pca_comp + pca_mean).view(-1, 65, 65)
        err2 = (psi_h - psi_t) ** 2
        if edge_weight != "off" and edge_mix > 0:
            with torch.no_grad():
                if edge_weight == "grad":
                    gz = torch.zeros_like(psi_t)
                    gz[:, 1:-1, :] = (psi_t[:, 2:, :] - psi_t[:, :-2, :]) * 0.5
                    gr = torch.zeros_like(psi_t)
                    gr[:, :, 1:-1] = (psi_t[:, :, 2:] - psi_t[:, :, :-2]) * 0.5
                    s = (gz.square() + gr.square()).sqrt().clamp_min(0.0)
                    s = s.pow(float(edge_power))
                    s = s / s.mean().clamp_min(1e-12)
                elif edge_weight == "band":
                    psi_min_w = psi_t.amin(dim=(1, 2), keepdim=True)
                    psi_max_w = psi_t.amax(dim=(1, 2), keepdim=True)
                    psi_n_w = (psi_t - psi_min_w) / (psi_max_w - psi_min_w).clamp_min(1e-9)
                    s = ((psi_n_w >= float(edge_band_lo))
                         & (psi_n_w <= float(edge_band_hi))).to(psi_t.dtype)
                    s = s / s.mean().clamp_min(1e-12)
                else:
                    raise ValueError(f"edge_weight={edge_weight!r}; expected off|grad|band")
                w_edge = (1.0 - float(edge_mix)) + float(edge_mix) * s
                w_edge = w_edge / w_edge.mean().clamp_min(1e-12)
            L_data = (err2 * w_edge).mean()
        else:
            L_data = err2.mean()
        L_coeff = ((z_m - zt) ** 2).mean()
        if lambda_coeff_white > 0:
            L_cw = (((z_m - zt) / coeff_scale) ** 2).mean()
        else:
            L_cw = z_m.new_zeros(())
        L_qb = 0.5 * (((q_hat[mask] - qb[mask]) / sig_q) ** 2).mean() + \
               0.5 * (((b_hat[mask] - bb[mask]) / sig_b) ** 2).mean()
        loss = L_data + 0.25 * L_coeff + lambda_coeff_white * L_cw + max(lambda_qb, 0.0) * L_qb
        if want_geom and gb is not None and sd is not None:
            g_ref, ok_m = gb[mask], okb[mask]
            if geom_frac < 1.0 and int(psi_h.shape[0]) > 1:
                k = max(1, int(round(psi_h.shape[0] * geom_frac)))
                sub = torch.randperm(psi_h.shape[0], device=psi_h.device)[:k]
                psi_g, g_ref, ok_m = psi_h[sub], g_ref[sub], ok_m[sub]
            else:
                psi_g = psi_h
            # One backward through the GRU. Chunk only the SoftDerive forward workspace.
            chunk = max(int(geom_sd_chunk), 1)
            n_g = int(psi_g.shape[0])
            acc = psi_g.new_zeros(())
            for start in range(0, n_g, chunk):
                sl = slice(start, min(start + chunk, n_g))
                g = sd(psi_g[sl], tau=tau, tau_li=tau_li)
                g_hat = torch.stack([g[k] for k in SCALAR_NAMES], dim=1)
                acc = acc + _geom_loss(g_hat, g_ref[sl], ok_m[sl], sigma, geom_huber, geom_w) * g_hat.shape[0]
            loss = loss + gscale * lambda_geom * (acc / max(n_g, 1))
        return loss, float(((psi_h - psi_t) ** 2).mean())

    @torch.no_grad()
    def _validate() -> tuple[float, float]:
        if not val_ids:
            return float("nan"), float("nan")
        model.eval()
        tot, n = 0.0, 0
        for pack in _packs(val_ids, lengths, int(token_budget)):
            xd, xr, zb, qb, bb, mask, lens, *_ = _pad_dual(shots, pack, device)
            z_hat = model(xd, xr, lens)[0]
            z_m, zt = z_hat[mask], zb[mask]
            psi_h = (z_m @ pca_comp + pca_mean).view(-1, 65, 65)
            psi_t = (zt @ pca_comp + pca_mean).view(-1, 65, 65)
            tot += float(((psi_h - psi_t) ** 2).sum())
            n += int(mask.sum())
        model.train()
        mse = tot / max(n, 1)
        return mse, 1.0 - mse / max(var_psi, 1e-12)

    best_sel = -float("inf")
    best_state = None
    best_epoch = 0
    history: list[dict] = []
    t0 = time.perf_counter()
    partial_path = Path(str(out)).with_name(Path(out).stem + ".partial.pt")
    start_epoch = 1

    def _bundle_fields() -> dict:
        return {
            "residual_state": None,
            "n_features": int(Xs_dual.shape[1]),
            "n_raw_features": int(n_raw_feat),
            "feature_mean": mean,
            "feature_std": std,
            "raw_feature_mean": raw_mean,
            "raw_feature_std": raw_std,
            "include_thomson": include_thomson,
            "n_pca": int(n_comp),
            "use_residual": False,
            "temporal_window": int(temporal_window),
            "temporal_mode": str(temporal_mode),
            "hidden": [int(hidden)],
            "activation": activation,
            "arch": "dualshot",
            "arch_config": {
                "width": int(trunk_width), "blocks": int(trunk_blocks),
                "dropout": float(dropout),
                "ctx_width": int(ctx_width), "ctx_blocks": int(ctx_blocks),
                "seq_hidden": int(hidden), "seq_layers": int(layers),
                "bidirectional": bool(bidirectional),
                "n_split": int(n_raw_feat),
            },
            "feature_clip": float(feature_clip),
            "pca_mean": pca.mean_.astype(np.float32),
            "pca_components": pca.components_.astype(np.float32),
            "method": MODEL_ID,
        }

    def _save_partial(ep: int) -> None:
        if best_state is None:
            return
        tmp = partial_path.with_suffix(".tmp")
        torch.save(
            {
                **_bundle_fields(),
                "model_state": best_state,
                "history": list(history),
                "partial": True,
                "resume": {
                    "epoch": int(ep),
                    "live_state": {k: v.detach().cpu().clone()
                                   for k, v in model.state_dict().items()},
                    "opt_state": opt.state_dict(),
                    "sched_state": sched.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": (torch.cuda.get_rng_state(device)
                                 if device.type == "cuda" else None),
                    "numpy_rng": rng.bit_generator.state,
                    "best_selection": float(best_sel),
                    "best_epoch": int(best_epoch),
                    "best_state": best_state,
                    "guard": {
                        "seed": int(seed),
                        "epochs": int(epochs),
                        "token_budget": int(token_budget),
                        "n_pca": int(n_comp),
                        "arch": "dualshot",
                        "seq_hidden": int(hidden),
                        "seq_layers": int(layers),
                        "geom_emphasis": float(geom_emphasis),
                        "lr": float(lr),
                    },
                },
            },
            tmp,
        )
        tmp.replace(partial_path)
        print(f"[{MODEL_ID}] partial checkpoint @ ep {ep} -> {partial_path.name}", flush=True)

    if resume is not None:
        rp = Path(resume)
        if not rp.is_file():
            raise FileNotFoundError(f"--resume {rp} does not exist")
        rck = torch.load(rp, map_location="cpu", weights_only=False)
        rs = rck.get("resume")
        if not rs:
            raise ValueError(
                f"{rp.name} has no resume state: it can be scored only if it is a full bundle. "
                "Start the run again."
            )
        want = {
            "seed": int(seed), "epochs": int(epochs), "token_budget": int(token_budget),
            "n_pca": int(n_comp), "arch": "dualshot",
            "seq_hidden": int(hidden), "seq_layers": int(layers),
            "geom_emphasis": float(geom_emphasis), "lr": float(lr),
        }
        bad = {k2: (v, want[k2]) for k2, v in (rs.get("guard") or {}).items()
               if k2 in want and v != want[k2]}
        if bad:
            raise ValueError(
                f"refusing to resume {rp.name}: " + ", ".join(
                    f"{k2} was {a} now {b}" for k2, (a, b) in bad.items())
                + ". A resume that changes the schedule is a new run."
            )
        model.load_state_dict(rs["live_state"])
        opt.load_state_dict(rs["opt_state"])
        sched.load_state_dict(rs["sched_state"])
        torch.set_rng_state(rs["torch_rng"].cpu() if hasattr(rs["torch_rng"], "cpu")
                            else rs["torch_rng"])
        if rs.get("cuda_rng") is not None and device.type == "cuda":
            torch.cuda.set_rng_state(rs["cuda_rng"].cpu(), device)
        if rs.get("numpy_rng") is not None:
            rng.bit_generator.state = rs["numpy_rng"]
        history = list(rck.get("history") or [])
        best_sel = float(rs.get("best_selection", -float("inf")))
        best_epoch = int(rs.get("best_epoch", 0))
        best_state = rs.get("best_state")
        start_epoch = int(rs["epoch"]) + 1
        print(
            f"[{MODEL_ID}] RESUMED {rp.name} at epoch {rs['epoch']} -> continuing from "
            f"{start_epoch}/{epochs}  lr={opt.param_groups[0]['lr']:.3e}  "
            f"best selection {best_sel:.4f} @ ep {best_epoch}",
            flush=True,
        )

    progress_path = _revision_dir() / "loop" / "progress.json"
    geom_armed = False
    last_sec = 0.0

    def _write_progress(ep: int, sec: float, want_geom: bool) -> None:
        import json
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        eta = None
        if ep < epochs and sec > 0:
            eta = time.strftime("%H:%M", time.localtime(time.time() + (epochs - ep) * sec))
        progress_path.write_text(json.dumps({
            "tag": Path(out).stem.replace(f"{MODEL_ID}_", ""),
            "epoch": int(ep),
            "total": int(epochs),
            "s_per_ep": float(sec),
            "eta": eta,
            "geom": bool(want_geom),
            "token_budget": int(token_budget),
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }) + "\n")

    for ep in range(start_epoch, epochs + 1):
        model.train()
        order = train_ids[:]
        rng.shuffle(order)
        running, n_step = 0.0, 0
        t_ep = time.perf_counter()
        tau, tau_li = _taus(ep)
        gscale = _geom_scale(ep)
        want_geom = use_geom and ep > geom_warmup
        if want_geom and not geom_armed:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            print(
                f"[{MODEL_ID}] geometry ON at epoch {ep} "
                f"(SoftDerive chunk={max(int(geom_sd_chunk), 1)})",
                flush=True,
            )
            geom_armed = True
        _write_progress(ep, last_sec, want_geom)
        packs = _packs(order, lengths, int(token_budget))
        n_pack = len(packs)
        sample_at = {1, max(1, n_pack // 4), max(1, n_pack // 2), max(1, (3 * n_pack) // 4), n_pack}
        job_tag = Path(out).stem.replace(f"{MODEL_ID}_", "")
        for pi, pack in enumerate(packs, start=1):
            if pi in sample_at:
                _log_gpu(job_tag, ep, pack=pi, n_pack=n_pack,
                         chunk=max(int(geom_sd_chunk), 1) if want_geom else 0,
                         geom=bool(want_geom), token_budget=int(token_budget))
            xd, xr, zb, qb, bb, mask, lens, gb, okb = _pad_dual(shots, pack, device)
            opt.zero_grad(set_to_none=True)
            loss, _ = _loss(xd, xr, zb, qb, bb, mask, lens, gb, okb, want_geom, gscale, tau, tau_li)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, float(grad_clip))
            opt.step()
            running += float(loss.detach())
            n_step += 1
        sched.step()
        val_mse, val_r2 = _validate()
        if (
            kill_r2psi > 0
            and ep >= int(kill_r2psi_after)
            and np.isfinite(val_r2)
            and val_r2 < float(kill_r2psi)
        ):
            print(
                f"[{MODEL_ID}] KILLED at epoch {ep}: val_R2ψ={val_r2:.4f} "
                f"< {float(kill_r2psi):g}",
                flush=True,
            )
            break
        sel = 0.55 * (val_r2 if np.isfinite(val_r2) else 0.0)
        if sel > best_sel:
            best_sel, best_epoch = sel, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append({"epoch": ep, "loss": running / max(n_step, 1),
                        "val_psi_mse": val_mse, "val_r2_psi": val_r2})
        sec = time.perf_counter() - t_ep
        last_sec = sec
        _write_progress(ep, sec, want_geom)
        if ep == 1 or ep == epochs or ep % max(int(log_every), 1) == 0 or want_geom:
            gate = float(model.gate_seq.detach().mean())
            eta = ""
            if ep < epochs and sec > 0:
                left = (epochs - ep) * sec
                eta = f"  eta {time.strftime('%H:%M', time.localtime(time.time() + left))}"
            peak = ""
            if device.type == "cuda":
                peak = f"  peak_vram={torch.cuda.max_memory_allocated() / 1024**3:.2f}GiB"
            print(
                f"[{MODEL_ID}] {time.strftime('%H:%M:%S')} epoch {ep}/{epochs}  "
                f"loss={running / max(n_step, 1):.4e}  val_ψ={val_mse:.4e}  "
                f"val_R2ψ={val_r2:.4f}  gate_seq={gate:+.4f}  {sec:.1f}s/ep{eta}{peak}",
                flush=True,
            )
        if ckpt_every > 0 and ep % int(ckpt_every) == 0:
            _save_partial(ep)
        if int(probe_epochs) > 0 and (ep - start_epoch + 1) >= int(probe_epochs):
            _save_partial(ep)
            print(
                f"[{MODEL_ID}] probe done after {probe_epochs} epoch(s) "
                f"chunk={max(int(geom_sd_chunk), 1)}  last {sec:.1f}s/ep  "
                f"partial kept at {partial_path.name}",
                flush=True,
            )
            return CharcoalBundle(
                checkpoint=partial_path,
                n_features=int(Xs_dual.shape[1]),
                feature_mean=mean,
                feature_std=std,
                include_thomson=include_thomson,
                n_pca=int(n_comp),
                device_name=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                n_raw_features=int(n_raw_feat),
                temporal_window=int(temporal_window),
                temporal_mode=str(temporal_mode),
                raw_feature_mean=raw_mean,
                raw_feature_std=raw_std,
                training_params={"probe_epochs": int(probe_epochs), "geom_sd_chunk": int(geom_sd_chunk)},
                model_version=make_model_version(MODEL_ID),
                pca_mean=pca.mean_.astype(np.float32),
                pca_components=pca.components_.astype(np.float32),
                hidden=(int(hidden),),
                activation=activation,
                arch="dualshot",
                feature_clip=float(feature_clip),
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"[{MODEL_ID}] restored best selection={best_sel:.4f} @ epoch {best_epoch}  "
            f"gate_seq={float(model.gate_seq.detach().mean()):+.4f}",
            flush=True,
        )

    train_s = time.perf_counter() - t0
    tp = {
        "n_train": len(cache_keys) if cache_keys is not None else n_shots,
        "n_train_frames": n_total,
        "epochs": int(epochs),
        "token_budget": int(token_budget),
        "arch": "dualshot",
        "seq_hidden": int(hidden),
        "seq_layers": int(layers),
        "shot_bidirectional": bool(bidirectional),
        "freeze_dualctx": bool(freeze_dualctx),
        "gate_seq": float(model.gate_seq.detach().mean()),
        "n_params": n_params,
        "n_pca": int(n_comp),
        "temporal_window": int(temporal_window),
        "lambda_geom": float(lambda_geom),
        "geom_emphasis": float(geom_emphasis),
        "geom_weights": str(geom_weights),
        "geom_emphasis_final": float(geom_emphasis_final),
        "geom_anneal_tail": float(geom_anneal_tail),
        "lambda_coeff_white": float(lambda_coeff_white),
        "edge_weight": str(edge_weight),
        "edge_mix": float(edge_mix),
        "edge_power": float(edge_power),
        "var_psi": float(var_psi),
        "best_epoch": int(best_epoch),
        "best_selection": float(best_sel),
        "lbfgs_steps_ran": 0,
        "seed": int(seed),
        "init_seed": int(seed if init_seed is None else init_seed),
        "device": torch.cuda.get_device_name(0),
        "train_s": float(train_s),
        "git_commit": git_full_sha(),
    }
    ckpt = {
        "model_state": model.state_dict(),
        "residual_state": None,
        "n_features": int(Xs_dual.shape[1]),
        "n_raw_features": int(n_raw_feat),
        "feature_mean": mean,
        "feature_std": std,
        "raw_feature_mean": raw_mean,
        "raw_feature_std": raw_std,
        "include_thomson": include_thomson,
        "n_pca": int(n_comp),
        "use_residual": False,
        "temporal_window": int(temporal_window),
        "temporal_mode": str(temporal_mode),
        "hidden": [int(hidden)],
        "activation": activation,
        "arch": "dualshot",
        "arch_config": {
            "width": int(trunk_width), "blocks": int(trunk_blocks),
            "dropout": float(dropout),
            "ctx_width": int(ctx_width), "ctx_blocks": int(ctx_blocks),
            "seq_hidden": int(hidden), "seq_layers": int(layers),
            "bidirectional": bool(bidirectional),
            "n_split": int(n_raw_feat),
        },
        "feature_clip": float(feature_clip),
        "pca_mean": pca.mean_.astype(np.float32),
        "pca_components": pca.components_.astype(np.float32),
        "training_params": tp,
        "history": history,
        "method": MODEL_ID,
        "model_version": make_model_version(MODEL_ID),
        "git_commit": tp["git_commit"],
    }
    torch.save(ckpt, out)
    Path(partial_path).unlink(missing_ok=True)
    print(f"[{MODEL_ID}] wrote {out}  train_s={train_s:.1f}s", flush=True)
    return CharcoalBundle(
        checkpoint=Path(out),
        n_features=int(Xs_dual.shape[1]),
        feature_mean=mean,
        feature_std=std,
        include_thomson=include_thomson,
        n_pca=int(n_comp),
        device_name=tp["device"],
        n_raw_features=int(n_raw_feat),
        temporal_window=int(temporal_window),
        temporal_mode=str(temporal_mode),
        raw_feature_mean=raw_mean,
        raw_feature_std=raw_std,
        training_params=tp,
        model_version=ckpt["model_version"],
        pca_mean=ckpt["pca_mean"],
        pca_components=ckpt["pca_components"],
        hidden=(int(hidden),),
        activation=activation,
        arch="dualshot",
        feature_clip=float(feature_clip),
    )
