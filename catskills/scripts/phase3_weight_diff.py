#!/usr/bin/env python3
"""Diff shipped Phase-2 ``.pt`` weights vs Reproduce-Phase retrained weights.

Prints a console summary, writes ``weight_diff.json``, and a visual smell-check
report (``weight_diff.html`` + ``weight_diff.md`` + PNG figures).
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def _state_dict(ck: dict) -> dict[str, torch.Tensor]:
    for key in ("model_state", "model_state_dict", "state_dict"):
        raw = ck.get(key)
        if isinstance(raw, dict) and raw:
            return {k: v for k, v in raw.items() if torch.is_tensor(v)}
    raise KeyError("checkpoint has no model_state / state_dict tensors")


def _machine(stem: str) -> str:
    if "mast" in stem.lower() or stem.startswith("CRC_FEQ_mast_"):
        return "MAST"
    return "DIII-D"


def _short_label(label: str) -> str:
    s = label.replace("CRC_FEQ_olive_geom_", "").replace("CRC_FEQ_mast_", "")
    s = s.replace("CRC_FEQ_", "")
    if "↔" in s:
        a, b = s.split("↔", 1)
        return f"{a.strip()[-24:]}↔{b.strip()[-24:]}"
    return s[-40:]


def compare_state(
    a: dict[str, torch.Tensor],
    b: dict[str, torch.Tensor],
    *,
    top_tensors: int = 25,
) -> dict[str, Any]:
    keys_a, keys_b = set(a), set(b)
    shared = sorted(keys_a & keys_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)
    max_abs = 0.0
    sum_abs = 0.0
    sum_sq = 0.0
    sum_sq_ref = 0.0
    n_elem = 0
    shape_mismatch: list[str] = []
    worst_key = ""
    per_tensor: list[dict[str, Any]] = []
    for k in shared:
        ta = a[k].detach().float().cpu().reshape(-1)
        tb = b[k].detach().float().cpu().reshape(-1)
        if ta.numel() != tb.numel():
            shape_mismatch.append(k)
            continue
        d = (ta - tb).abs()
        m = float(d.max()) if d.numel() else 0.0
        mean_t = float(d.mean()) if d.numel() else 0.0
        per_tensor.append(
            {"name": k, "max_abs": m, "mean_abs": mean_t, "n": int(ta.numel())}
        )
        if m >= max_abs:
            max_abs = m
            worst_key = k
        sum_abs += float(d.sum())
        sum_sq += float(((ta - tb) ** 2).sum())
        sum_sq_ref += float((ta**2).sum())
        n_elem += int(ta.numel())
    mean_abs = (sum_abs / n_elem) if n_elem else float("nan")
    rel_l2 = math.sqrt(sum_sq / sum_sq_ref) if sum_sq_ref > 0 else float("nan")
    identical = (
        n_elem > 0
        and max_abs == 0.0
        and not only_a
        and not only_b
        and not shape_mismatch
    )
    per_tensor.sort(key=lambda x: x["max_abs"], reverse=True)
    return {
        "n_tensors_shared": len(shared) - len(shape_mismatch),
        "n_params_compared": n_elem,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rel_l2": rel_l2,
        "worst_tensor": worst_key,
        "identical": identical,
        "only_in_shipped": only_a,
        "only_in_repro": only_b,
        "shape_mismatch": shape_mismatch,
        "top_tensors": per_tensor[:top_tensors],
    }


def compare_files(shipped: Path, repro: Path) -> dict[str, Any]:
    ck_s = torch.load(shipped, map_location="cpu", weights_only=False)
    ck_r = torch.load(repro, map_location="cpu", weights_only=False)
    if not isinstance(ck_s, dict) or not isinstance(ck_r, dict):
        raise TypeError("expected dict checkpoints")
    stats = compare_state(_state_dict(ck_s), _state_dict(ck_r))
    stats["shipped"] = str(shipped)
    stats["repro"] = str(repro)
    meta = {}
    for label, ck in (("shipped", ck_s), ("repro", ck_r)):
        meta[label] = {
            k: ck.get(k)
            for k in ("best_epoch", "epochs", "val_mse", "init_seed", "seed", "train_seconds")
            if k in ck
        }
    stats["meta"] = meta
    return stats


def collect_stems(directory: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not directory.is_dir():
        return out
    for p in sorted(directory.glob("*.pt")):
        out[p.stem] = p
    return out


def _fig_to_png_bytes() -> bytes:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close()
    buf.seek(0)
    return buf.read()


def _save_png(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _smell(max_abs: float, identical: bool) -> tuple[str, str]:
    if identical or max_abs == 0.0:
        return "identical", "Same tensors (bit match on compared params)."
    if max_abs < 1e-5:
        return "noise", "Tiny drift (<1e-5) — likely float noise / nondeterminism."
    if max_abs < 1e-2:
        return "small", "Small drift — check if intentional (partial train / seed)."
    return "large", "Large drift — expected for from-scratch retrain; smell-check S, not bit-id."


def build_figures(rows: list[dict[str, Any]], fig_dir: Path) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    ok = [r for r in rows if "error" not in r]

    # 1) Per-checkpoint max_abs bars (log scale)
    if ok:
        labels = [_short_label(r["label"]) for r in ok]
        vals = [max(float(r["max_abs"]), 1e-16) for r in ok]
        colors = []
        for r in ok:
            smell, _ = _smell(float(r["max_abs"]), bool(r["identical"]))
            colors.append(
                {"identical": "#2a9d8f", "noise": "#8ab17d", "small": "#e9c46a", "large": "#e76f51"}[
                    smell
                ]
            )
        fig, ax = plt.subplots(figsize=(10, max(3.5, 0.35 * len(ok) + 1.5)))
        y = np.arange(len(ok))
        ax.barh(y, vals, color=colors, log=True)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("max |Δw| (log)")
        ax.set_title("Shipped vs repro — max abs weight delta by checkpoint")
        ax.axvline(1e-5, color="#555", ls="--", lw=0.8, label="1e-5")
        ax.axvline(1e-2, color="#999", ls=":", lw=0.8, label="1e-2")
        ax.legend(loc="lower right", fontsize=8)
        for machine in ("DIII-D", "MAST"):
            pass
        paths["max_abs_bars"] = _save_png(fig_dir / "max_abs_by_checkpoint.png", _fig_to_png_bytes())

    # 2) Machine rollup: mean of max_abs + identical counts
    machines = ["DIII-D", "MAST"]
    id_counts, diff_counts, worst = [], [], []
    for m in machines:
        g = [r for r in ok if r.get("machine") == m]
        id_counts.append(sum(1 for r in g if r["identical"]))
        diff_counts.append(sum(1 for r in g if not r["identical"]))
        worst.append(max((float(r["max_abs"]) for r in g), default=0.0))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    x = np.arange(len(machines))
    axes[0].bar(x - 0.2, id_counts, width=0.4, label="identical", color="#2a9d8f")
    axes[0].bar(x + 0.2, diff_counts, width=0.4, label="diff", color="#e76f51")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(machines)
    axes[0].set_ylabel("# checkpoints")
    axes[0].set_title("Identical vs different")
    axes[0].legend(fontsize=8)
    axes[1].bar(machines, [max(v, 1e-16) for v in worst], color=["#457b9d", "#a8dadc"], log=True)
    axes[1].set_ylabel("worst max |Δw| (log)")
    axes[1].set_title("Worst drift by machine")
    fig.suptitle("Machine rollup", y=1.02)
    fig.tight_layout()
    paths["machine_rollup"] = _save_png(fig_dir / "machine_rollup.png", _fig_to_png_bytes())

    # 3) Top tensors for worst file per machine
    for machine in machines:
        g = [r for r in ok if r.get("machine") == machine]
        if not g:
            continue
        worst_row = max(g, key=lambda r: float(r["max_abs"]))
        tops = worst_row.get("top_tensors") or []
        if not tops:
            continue
        names = [t["name"][-42:] for t in tops[:20]][::-1]
        mvals = [max(float(t["max_abs"]), 1e-16) for t in tops[:20]][::-1]
        fig, ax = plt.subplots(figsize=(9, max(3.0, 0.28 * len(names) + 1.2)))
        ax.barh(np.arange(len(names)), mvals, color="#264653", log=True)
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("max |Δw| (log)")
        ax.set_title(f"{machine}: top tensors — {_short_label(worst_row['label'])}")
        paths[f"top_tensors_{machine.replace('-', '')}"] = _save_png(
            fig_dir / f"top_tensors_{machine.replace('-', '').lower()}.png",
            _fig_to_png_bytes(),
        )

    # 4) rel_l2 scatter-ish bar
    if ok:
        fig, ax = plt.subplots(figsize=(10, max(3.2, 0.32 * len(ok) + 1.2)))
        labels = [_short_label(r["label"]) for r in ok]
        rel = [max(float(r["rel_l2"]), 1e-16) if math.isfinite(float(r["rel_l2"])) else 1e-16 for r in ok]
        ax.barh(np.arange(len(ok)), rel, color="#6d597a", log=True)
        ax.set_yticks(np.arange(len(ok)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("relative L2 ‖Δw‖ / ‖w_shipped‖ (log)")
        ax.set_title("Relative L2 drift by checkpoint")
        paths["rel_l2_bars"] = _save_png(fig_dir / "rel_l2_by_checkpoint.png", _fig_to_png_bytes())

    return paths


def write_markdown(payload: dict[str, Any], path: Path, fig_paths: dict[str, Path]) -> Path:
    rows = [r for r in payload["comparisons"] if "error" not in r]
    lines = [
        "# Phase 3 weight diff — smell check",
        "",
        f"- **Shipped:** `{payload['shipped_dir']}`",
        f"- **Repro:** `{payload['repro_dir']}`",
        "",
        "## How to read this",
        "",
        "| Smell | max\\|Δw\\| | Meaning |",
        "|---|---|---|",
        "| identical | 0 | Bit match on compared params |",
        "| noise | < 1e-5 | Float / nondeterminism |",
        "| small | < 1e-2 | Partial train / seed drift |",
        "| large | ≥ 1e-2 | Typical from-scratch retrain — judge by S, not bit-id |",
        "",
        "## Summary table",
        "",
        "| Machine | Checkpoint | Status | Smell | max\\|Δw\\| | mean\\|Δw\\| | rel L2 | params | worst tensor |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        smell, _ = _smell(float(r["max_abs"]), bool(r["identical"]))
        status = "IDENTICAL" if r["identical"] else "DIFF"
        lines.append(
            f"| {r['machine']} | `{_short_label(r['label'])}` | {status} | {smell} | "
            f"{r['max_abs']:.3e} | {r['mean_abs']:.3e} | {r['rel_l2']:.3e} | "
            f"{r['n_params_compared']:,} | `{r['worst_tensor']}` |"
        )
    lines += ["", "## Figures", ""]
    for key, title in (
        ("max_abs_bars", "Max abs Δ by checkpoint"),
        ("machine_rollup", "Machine rollup"),
        ("rel_l2_bars", "Relative L2 by checkpoint"),
        ("top_tensors_DIIID", "DIII-D top tensors (worst file)"),
        ("top_tensors_MAST", "MAST top tensors (worst file)"),
    ):
        p = fig_paths.get(key)
        if p and p.is_file():
            rel = p.name if p.parent == path.parent / "weight_diff_figures" else str(p)
            # figures live in weight_diff_figures/ next to md
            rel = f"weight_diff_figures/{p.name}"
            lines += [f"### {title}", "", f"![{title}]({rel})", ""]
    if payload.get("shipped_only"):
        lines += ["## Shipped-only stems", "", ", ".join(f"`{s}`" for s in payload["shipped_only"][:40]), ""]
    if payload.get("repro_only"):
        lines += ["## Repro-only stems", "", ", ".join(f"`{s}`" for s in payload["repro_only"][:40]), ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_html(payload: dict[str, Any], path: Path, fig_paths: dict[str, Path]) -> Path:
    rows = [r for r in payload["comparisons"] if "error" not in r]
    err_rows = [r for r in payload["comparisons"] if "error" in r]

    def img_tag(key: str, title: str) -> str:
        p = fig_paths.get(key)
        if not p or not p.is_file():
            return ""
        b64 = _b64(p.read_bytes())
        return (
            f'<figure class="fig"><figcaption>{html.escape(title)}</figcaption>'
            f'<img src="data:image/png;base64,{b64}" alt="{html.escape(title)}"/></figure>'
        )

    table_rows = []
    for r in rows:
        smell, tip = _smell(float(r["max_abs"]), bool(r["identical"]))
        status = "IDENTICAL" if r["identical"] else "DIFF"
        table_rows.append(
            f"<tr class='smell-{smell}'>"
            f"<td>{html.escape(r['machine'])}</td>"
            f"<td><code>{html.escape(_short_label(r['label']))}</code></td>"
            f"<td>{status}</td>"
            f"<td title='{html.escape(tip)}'>{smell}</td>"
            f"<td class='num'>{r['max_abs']:.3e}</td>"
            f"<td class='num'>{r['mean_abs']:.3e}</td>"
            f"<td class='num'>{r['rel_l2']:.3e}</td>"
            f"<td class='num'>{r['n_params_compared']:,}</td>"
            f"<td><code>{html.escape(str(r['worst_tensor']))}</code></td>"
            "</tr>"
        )

    machine_cards = []
    for machine in ("DIII-D", "MAST"):
        g = [r for r in rows if r.get("machine") == machine]
        if not g:
            machine_cards.append(f"<div class='card'><h3>{machine}</h3><p>No compared checkpoints.</p></div>")
            continue
        n_id = sum(1 for r in g if r["identical"])
        worst = max(g, key=lambda x: float(x["max_abs"]))
        smell, tip = _smell(float(worst["max_abs"]), bool(worst["identical"]))
        machine_cards.append(
            f"<div class='card smell-{smell}'><h3>{machine}</h3>"
            f"<p><b>{len(g)}</b> file(s), <b>{n_id}</b> identical</p>"
            f"<p>Worst max|Δw| = <b>{worst['max_abs']:.3e}</b> "
            f"on <code>{html.escape(_short_label(worst['label']))}</code></p>"
            f"<p class='tip'>{html.escape(tip)}</p></div>"
        )

    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Phase 3 weight diff — smell check</title>
<style>
  :root {{
    --bg: #f7f4ef; --ink: #1d1a16; --muted: #5c564c;
    --card: #fffdf8; --line: #ddd4c6;
    --identical: #2a9d8f; --noise: #8ab17d; --small: #e9c46a; --large: #e76f51;
  }}
  body {{ font-family: "IBM Plex Sans", "Segoe UI", sans-serif; margin: 0;
         background: linear-gradient(160deg, #f7f4ef, #ebe4d8); color: var(--ink); }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 2rem 1.25rem 4rem; }}
  h1 {{ font-family: "IBM Plex Serif", Georgia, serif; font-weight: 600; margin: 0 0 .25rem; }}
  .sub {{ color: var(--muted); margin-bottom: 1.5rem; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: .5rem; margin: 1rem 0 1.5rem; }}
  .pill {{ border-radius: 999px; padding: .25rem .7rem; font-size: .85rem; color: #111; border: 1px solid var(--line); }}
  .pill.identical {{ background: color-mix(in srgb, var(--identical) 35%, white); }}
  .pill.noise {{ background: color-mix(in srgb, var(--noise) 35%, white); }}
  .pill.small {{ background: color-mix(in srgb, var(--small) 45%, white); }}
  .pill.large {{ background: color-mix(in srgb, var(--large) 40%, white); }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem; }}
  @media (max-width: 720px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px;
           padding: 1rem 1.1rem; box-shadow: 0 1px 0 rgba(0,0,0,.03); }}
  .card.smell-identical {{ border-left: 5px solid var(--identical); }}
  .card.smell-noise {{ border-left: 5px solid var(--noise); }}
  .card.smell-small {{ border-left: 5px solid var(--small); }}
  .card.smell-large {{ border-left: 5px solid var(--large); }}
  .tip {{ color: var(--muted); font-size: .9rem; }}
  table {{ width: 100%; border-collapse: collapse; background: var(--card);
           border: 1px solid var(--line); border-radius: 12px; overflow: hidden; font-size: .9rem; }}
  th, td {{ padding: .55rem .65rem; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
  th {{ background: #efe8dc; font-size: .8rem; text-transform: uppercase; letter-spacing: .03em; }}
  td.num {{ font-variant-numeric: tabular-nums; font-family: "IBM Plex Mono", ui-monospace, monospace; }}
  tr.smell-identical td:nth-child(4) {{ color: var(--identical); font-weight: 600; }}
  tr.smell-noise td:nth-child(4) {{ color: #4d7c0f; font-weight: 600; }}
  tr.smell-small td:nth-child(4) {{ color: #a16207; font-weight: 600; }}
  tr.smell-large td:nth-child(4) {{ color: var(--large); font-weight: 600; }}
  code {{ font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: .82em; }}
  .fig {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px;
          padding: .75rem; margin: 1rem 0 1.5rem; }}
  .fig img {{ max-width: 100%; height: auto; display: block; margin: .5rem auto 0; }}
  figcaption {{ font-weight: 600; }}
  .paths {{ font-size: .85rem; color: var(--muted); word-break: break-all; }}
</style>
</head>
<body>
<main>
  <h1>Phase 3 weight diff</h1>
  <p class="sub">Shipped Phase-2 checkpoints vs Reproduce retrain — visual smell check</p>
  <p class="paths"><b>Shipped:</b> {html.escape(payload['shipped_dir'])}<br/>
     <b>Repro:</b> {html.escape(payload['repro_dir'])}</p>
  <div class="legend">
    <span class="pill identical">identical = 0</span>
    <span class="pill noise">noise &lt; 1e-5</span>
    <span class="pill small">small &lt; 1e-2</span>
    <span class="pill large">large ≥ 1e-2 (typical from-scratch)</span>
  </div>
  <div class="grid">
    {''.join(machine_cards)}
  </div>
  <h2>Checkpoint table</h2>
  <table>
    <thead><tr>
      <th>Machine</th><th>Checkpoint</th><th>Status</th><th>Smell</th>
      <th>max|Δw|</th><th>mean|Δw|</th><th>rel L2</th><th>params</th><th>worst tensor</th>
    </tr></thead>
    <tbody>
      {''.join(table_rows) if table_rows else '<tr><td colspan="9">No comparisons.</td></tr>'}
    </tbody>
  </table>
  {img_tag('max_abs_bars', 'Max |Δw| by checkpoint (log)')}
  {img_tag('machine_rollup', 'Machine rollup')}
  {img_tag('rel_l2_bars', 'Relative L2 by checkpoint (log)')}
  {img_tag('top_tensors_DIIID', 'DIII-D — top tensors on worst file')}
  {img_tag('top_tensors_MAST', 'MAST — top tensors on worst file')}
  <h2>Coverage notes</h2>
  <p class="tip">Shipped-only: {html.escape(', '.join(payload.get('shipped_only') or []) or '(none)')}</p>
  <p class="tip">Repro-only: {html.escape(', '.join(payload.get('repro_only') or []) or '(none)')}</p>
  {('<p class="tip">Errors: ' + html.escape('; '.join(f"{e.get('label')}: {e.get('error')}" for e in err_rows)) + '</p>') if err_rows else ''}
</main>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")
    return path


def write_reports(payload: dict[str, Any], out_json: Path | None) -> dict[str, Path]:
    """Write JSON companion HTML/MD + figures. Returns written paths."""
    base = out_json.parent if out_json else Path(".")
    stem = out_json.stem if out_json else "weight_diff"
    fig_dir = base / f"{stem}_figures"
    fig_paths = build_figures(payload["comparisons"], fig_dir)
    html_path = write_html(payload, base / f"{stem}.html", fig_paths)
    md_path = write_markdown(payload, base / f"{stem}.md", fig_paths)
    return {"html": html_path, "md": md_path, **{f"fig_{k}": v for k, v in fig_paths.items()}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shipped-dir", type=Path, required=True)
    ap.add_argument("--repro-dir", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--out-html", type=Path, default=None, help="Default: alongside --out-json")
    ap.add_argument("--no-report", action="store_true", help="Skip HTML/MD/figures")
    ap.add_argument(
        "--cross",
        action="append",
        default=[],
        metavar="SHIPPED_STEM=REPRO_STEM",
        help="Extra pair when stems differ",
    )
    args = ap.parse_args(argv)

    shipped = collect_stems(args.shipped_dir)
    repro = collect_stems(args.repro_dir)
    pairs: list[tuple[str, str, str]] = []
    for stem in sorted(set(shipped) & set(repro)):
        pairs.append((stem, stem, "exact"))
    for spec in args.cross:
        if "=" not in spec:
            raise SystemExit(f"--cross needs SHIPPED=REPRO, got {spec!r}")
        left, right = spec.split("=", 1)
        pairs.append((left.strip(), right.strip(), "cross"))

    rows: list[dict[str, Any]] = []
    for ship_stem, repro_stem, kind in pairs:
        if ship_stem not in shipped:
            print(f"MISSING shipped {ship_stem}", flush=True)
            continue
        if repro_stem not in repro:
            print(f"MISSING repro {repro_stem}", flush=True)
            continue
        label = ship_stem if kind == "exact" else f"{ship_stem} ↔ {repro_stem}"
        try:
            stats = compare_files(shipped[ship_stem], repro[repro_stem])
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {label}: {exc}", flush=True)
            rows.append({"label": label, "error": str(exc), "machine": _machine(ship_stem)})
            continue
        stats["label"] = label
        stats["kind"] = kind
        stats["machine"] = _machine(ship_stem)
        rows.append(stats)

    print("=" * 72, flush=True)
    print("PHASE 3 WEIGHT DIFF — shipped vs repro", flush=True)
    print(f"shipped: {args.shipped_dir}", flush=True)
    print(f"repro:   {args.repro_dir}", flush=True)
    print("=" * 72, flush=True)

    by_machine: dict[str, list] = {"DIII-D": [], "MAST": []}
    for r in rows:
        if "error" in r:
            continue
        by_machine.setdefault(r["machine"], []).append(r)
        smell, _ = _smell(float(r["max_abs"]), bool(r["identical"]))
        flag = "IDENTICAL" if r["identical"] else "DIFF"
        print(
            f"[{r['machine']}] {r['label']}: {flag}/{smell}  "
            f"max_abs={r['max_abs']:.6e}  mean_abs={r['mean_abs']:.6e}  "
            f"rel_l2={r['rel_l2']:.6e}  params={r['n_params_compared']:,}  "
            f"worst={r['worst_tensor']}",
            flush=True,
        )

    print("-" * 72, flush=True)
    for machine, group in by_machine.items():
        if not group:
            print(f"{machine}: (no compared checkpoints)", flush=True)
            continue
        worst = max(group, key=lambda x: x["max_abs"])
        n_id = sum(1 for x in group if x["identical"])
        print(
            f"{machine} summary: {len(group)} file(s), {n_id} identical; "
            f"worst max_abs={worst['max_abs']:.6e} on {worst['label']}",
            flush=True,
        )

    only_ship = sorted(set(shipped) - set(repro))
    only_repro = sorted(set(repro) - set(shipped))
    if only_ship:
        print(
            f"shipped-only stems ({len(only_ship)}): "
            + ", ".join(only_ship[:12])
            + (" …" if len(only_ship) > 12 else ""),
            flush=True,
        )
    if only_repro:
        print(
            f"repro-only stems ({len(only_repro)}): "
            + ", ".join(only_repro[:12])
            + (" …" if len(only_repro) > 12 else ""),
            flush=True,
        )

    # Drop bulky top_tensors from JSON? Keep them — useful and small (~25×N).
    payload = {
        "shipped_dir": str(args.shipped_dir),
        "repro_dir": str(args.repro_dir),
        "comparisons": rows,
        "shipped_only": only_ship,
        "repro_only": only_repro,
    }
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out_json}", flush=True)

    if not args.no_report:
        out_json = args.out_json
        if args.out_html:
            # Force report next to explicit html path
            out_json = args.out_html.with_suffix(".json") if out_json is None else out_json
            report_base = args.out_html.parent / args.out_html.stem
            fig_dir = args.out_html.parent / f"{args.out_html.stem}_figures"
            fig_paths = build_figures(payload["comparisons"], fig_dir)
            html_path = write_html(payload, args.out_html, fig_paths)
            md_path = write_markdown(payload, args.out_html.with_suffix(".md"), fig_paths)
            written = {"html": html_path, "md": md_path}
        else:
            if out_json is None:
                out_json = Path("weight_diff.json")
                out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            written = write_reports(payload, out_json)
        print(f"wrote {written['html']}", flush=True)
        print(f"wrote {written['md']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
