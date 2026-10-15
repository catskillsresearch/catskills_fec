"""Simple branching shot browser.

    .venv/bin/python -m fec_site
    http://127.0.0.1:8770
"""
from __future__ import annotations

import json
import math
import sys
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from contourpy import contour_generator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feq.data import data_root, load_shot, shot_paths
from fec_site.score import CONS_SCALARS, _extract, load_or_compute
from fec_site.diag import load_or_compute_diag
STATIC = Path(__file__).resolve().parent / "static"
DEMO = ROOT / "parquet_data"
SOLVER_ID = "osf_0.0.12"
MODEL_IDS = {"mast": "mast_12_osf_0.0.12", "d3d": "d3d_0.0.17_fleet_full"}
MODEL_ID = MODEL_IDS["d3d"]
LAB_RESULTS_ROOT = Path("/media/catskills/CRC_FUSION_LAB/fusion-equilibrium-challenge/results")
PRED_DIR = (LAB_RESULTS_ROOT / MODEL_ID) if (LAB_RESULTS_ROOT / MODEL_ID).is_dir() else (ROOT / "results" / MODEL_ID)


def _model_id(machine: str) -> str:
    return MODEL_IDS.get(machine, MODEL_ID)


def _pred_dir(machine: str = "d3d") -> Path:
    mid = _model_id(machine)
    lab_p = LAB_RESULTS_ROOT / mid
    if lab_p.is_dir():
        return lab_p
    return ROOT / "results" / mid
HOST = "127.0.0.1"
PORT = 8770
PAGE = 80

SCORE_ROWS = (
    ("S", "S"),
    ("r2_psi", "R²_ψ"),
    ("r2_q95", "R²_q95"),
    ("r2_betaN", "R²_βN"),
    ("r2_qb", "R²_{q,β}"),
    ("one_minus_D", "1 − D_LCFS"),
    ("Consistency", "Consistency"),
)


SETS = {
    ("mast", "test"): {
        "title": "MAST — unlabelled test shots",
        "labelled": False,
        "paths": lambda: shot_paths("mast_public_test"),
    },
    ("d3d", "labelled"): {
        "title": "DIII-D — labelled shots",
        "labelled": True,
        "paths": lambda: shot_paths("diii_d_train"),
    },
}

EFIT_SCALARS = (
    "efit_beta_n",
    "efit_li",
    "efit_beta_p",
    "magnetics_dsep",
)


def _jsonable(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _html(title: str, body: str, *, wide: bool = False) -> bytes:
    cls = ' class="wide"' if wide else ""
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/>
<title>{title}</title>
<link rel="stylesheet" href="/static/style.css"/>
</head><body{cls}><main>
{body}
</main></body></html>""".encode()


def _has_pred(stem: str, machine: str = "mast") -> bool:
    dest = _pred_dir(machine).joinpath(f"{stem}.npz")
    if dest.is_file():
        return True
    if machine == "d3d":
        try:
            from fec_site.d3d_infer import ensure_d3d_pred
            return ensure_d3d_pred(stem)
        except Exception:
            return False
    return False


def _crumb(*pairs: tuple[str, str]) -> str:
    bits = ['<a href="/">Home</a>']
    for href, label in pairs:
        bits.append(f'<a href="{href}">{label}</a>')
    return '<nav class="crumb">' + " / ".join(bits) + "</nav>"


def _buttons(items: list[tuple[str, str]], *, alt: bool = False) -> str:
    cls = "btn alt" if alt else "btn"
    return '<div class="buttons">' + "".join(
        f'<a class="{cls}" href="{href}">{label}</a>' for href, label in items
    ) + "</div>"


def _set(machine: str, bucket: str) -> dict:
    key = (machine, bucket)
    if key not in SETS:
        raise KeyError(key)
    return SETS[key]


@lru_cache(maxsize=1)
def _index(machine: str, bucket: str) -> dict[str, Path]:
    spec = _set(machine, bucket)
    return {p.stem: p for p in spec["paths"]()}


def _page_list(stems: list[str], q: str, page: int) -> tuple[list[str], int]:
    if q:
        q = q.lower()
        stems = [s for s in stems if q in s.lower()]
    n = max(1, math.ceil(len(stems) / PAGE))
    page = min(max(page, 1), n)
    chunk = stems[(page - 1) * PAGE : page * PAGE]
    return chunk, n


def home() -> bytes:
    return _html(
        "Shot browser",
        "<h1>Shot browser</h1>"
        '<p class="lead">Pick a machine.</p>'
        + _buttons([("/mast", "MAST data"), ("/d3d", "DIII-D data")]),
    )


def mast_home() -> bytes:
    return _html(
        "MAST data",
        _crumb(("/mast", "MAST"))
        + "<h1>MAST data</h1>"
        + '<p class="lead">1,206 input-only public-test rows.</p>'
        + _buttons([("/mast/test", "Unlabelled test shots")]),
    )


def d3d_home() -> bytes:
    return _html(
        "DIII-D data",
        _crumb(("/d3d", "DIII-D"))
        + "<h1>DIII-D data</h1>"
        '<p class="lead">Labelled train rows. Public test is inputs-only and is not on this branch.</p>'
        + _buttons([("/d3d/labelled", "Labelled test shots")]),
    )


def _fmt_score(v, digits: int = 4) -> str:
    if v is None:
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(x):
        return "—"
    if abs(x) >= 100:
        return f"{x:.3e}"
    return f"{x:.{digits}f}"


def dummy_plots() -> str:
    return ""
def shot_browser(machine: str, bucket: str, query: dict[str, list[str]]) -> bytes:
    spec = _set(machine, bucket)
    idx = _index(machine, bucket)
    stems = sorted(idx)
    q = (query.get("q") or [""])[0]
    page = int((query.get("page") or ["1"])[0] or 1)
    chunk, n_pages = _page_list(stems, q, page)
    cards = "".join(
        f'<a class="btn small" href="/{machine}/{bucket}/{stem}">{stem}</a>'
        for stem in chunk
    )
    prev_p = max(1, page - 1)
    next_p = min(n_pages, page + 1)
    q_esc = q.replace('"', "&quot;")
    body = (
        _crumb((f"/{machine}", machine.upper()), (f"/{machine}/{bucket}", spec["title"]))
        + f"<h1>{spec['title']}</h1>"
        + f'<p class="meta">{len(stems)} shots. Labelled: {spec["labelled"]}.</p>'
        + f'<form class="search" method="get" action="/{machine}/{bucket}">'
        f'<input name="q" value="{q_esc}" placeholder="filter by filename"/>'
        f'<button class="btn small" type="submit">Find</button></form>'
        f'<div class="grid">{cards}</div>'
        f'<div class="pager">'
        f'<a class="btn ghost small" href="/{machine}/{bucket}?q={q}&page={prev_p}">Prev</a>'
        f"<span>page {page} / {n_pages}</span>"
        f'<a class="btn ghost small" href="/{machine}/{bucket}?q={q}&page={next_p}">Next</a>'
        f"</div>"
    )
    return _html(spec["title"], body)


def _times_ms(row: dict) -> np.ndarray:
    return np.asarray(row["efit_times"], dtype=float)


def shot_page(machine: str, bucket: str, stem: str) -> bytes:
    spec = _set(machine, bucket)
    idx = _index(machine, bucket)
    if stem not in idx:
        return _html("Missing shot", f"<h1>No shot {stem}</h1>"), 404
    labelled = spec["labelled"]
    if labelled and machine in MODEL_IDS and _has_pred(stem, machine):
        return shot_page_compare(machine, bucket, stem, spec)
    body = (
        _crumb(
            (f"/{machine}", machine.upper()),
            (f"/{machine}/{bucket}", spec["title"]),
            (f"/{machine}/{bucket}/{stem}", stem),
        )
        + f'<h1 id="heading">{stem}</h1>'
        + (
            '<p class="lead">One clock: <code>efit_times</code>. Slider walks every diverted EFIT frame in this shot, in time order.</p>'
            '<div class="controls">'
            '<button class="btn small" id="play" type="button">Play</button>'
            '<input id="scrub" type="range" min="0" max="0" value="0"/>'
            '<span id="stamp"></span></div>'
            '<div id="psi-wrap">'
            '<div id="psi"></div>'
            '<svg id="psi-overlay"></svg>'
            '<div id="opoint"></div>'
            '<div id="psi-legend">'
            '<div class="leg"><i class="swatch opoint"></i>O-point</div>'
            '<div class="leg"><i class="swatch lcfs"></i><span id="leg-lcfs">LCFS</span></div>'
            '<div class="leg"><i class="swatch q95"></i><span id="leg-q95">q95 surface</span></div>'
            '</div></div>'
            if labelled
            else '<p class="lead">Per-frame input scalars. No ψ on this fold.</p>'
        )
        + '<div id="scalars"></div>'
        + f"""<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script>
const BASE = "/api/{machine}/{bucket}/{stem}";
const SHOT = "{stem}";
const START = 0;
const LABELLED = {str(labelled).lower()};
let times = [];
let playing = false;
let timer = null;
let psiReady = false;
let scalarReady = false;

function finiteSpan(y) {{
  const vals = y.filter((v) => Number.isFinite(v));
  if (!vals.length) return [0, 1];
  return [Math.min(...vals), Math.max(...vals)];
}}

function layoutScalars(names) {{
  const n = names.length;
  const layout = {{
    grid: {{rows: n, columns: 1, pattern: "independent"}},
    height: Math.max(280, 140 * n),
    margin: {{t: 24, r: 20, b: 40, l: 60}},
    showlegend: false,
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
  }};
  names.forEach((name, k) => {{
    layout[k ? "yaxis" + (k + 1) : "yaxis"] = {{title: name, automargin: true}};
    layout[k ? "xaxis" + (k + 1) : "xaxis"] = {{
      title: k === n - 1 ? "t (ms)" : "",
    }};
  }});
  return layout;
}}

function scalarTraces(payload, highlight) {{
  const traces = [];
  payload.names.forEach((name, i) => {{
    const xa = i ? "x" + (i + 1) : "x";
    const ya = i ? "y" + (i + 1) : "y";
    const [lo, hi] = finiteSpan(payload.series[name]);
    traces.push({{
      x: payload.t,
      y: payload.series[name],
      xaxis: xa,
      yaxis: ya,
      mode: "lines",
      name: name,
      line: {{width: 1.6}},
    }});
    traces.push({{
      x: [payload.t[highlight], payload.t[highlight]],
      y: [lo, hi],
      xaxis: xa,
      yaxis: ya,
      mode: "lines",
      line: {{color: "#b03a2e", width: 1, dash: "dot"}},
      hoverinfo: "skip",
    }});
  }});
  return traces;
}}

function drawScalars(payload, i) {{
  if (!scalarReady) {{
    Plotly.newPlot("scalars", scalarTraces(payload, i), layoutScalars(payload.names), {{responsive: true}});
    scalarReady = true;
    return;
  }}
  const xs = payload.names.map(() => [payload.t[i], payload.t[i]]);
  const idx = payload.names.map((_, k) => 2 * k + 1);
  Plotly.restyle("scalars", {{x: xs}}, idx);
}}

function psiXY(gd, r, z) {{
  const xa = gd._fullLayout.xaxis;
  const ya = gd._fullLayout.yaxis;
  return [xa._offset + xa.l2p(r), ya._offset + ya.l2p(z)];
}}

function polyPath(gd, poly) {{
  return poly.map((p, k) => {{
    const xy = psiXY(gd, p[0], p[1]);
    return (k ? "L" : "M") + xy[0] + " " + xy[1];
  }}).join(" ") + "Z";
}}

function placeOverlays(pack, i) {{
  const el = document.getElementById("opoint");
  const legend = document.getElementById("psi-legend");
  const svg = document.getElementById("psi-overlay");
  const gd = document.getElementById("psi");
  const hide = () => {{
    if (el) el.style.display = "none";
    if (legend) legend.style.display = "none";
    if (svg) svg.innerHTML = "";
  }};
  if (!gd || !gd._fullLayout || !gd._fullLayout.xaxis || !gd._fullLayout.xaxis.l2p) {{
    hide();
    return;
  }}
  const r = pack.axis_r && pack.axis_r[i];
  const zz = pack.axis_z && pack.axis_z[i];
  if (el && Number.isFinite(r) && Number.isFinite(zz)) {{
    const xy = psiXY(gd, r, zz);
    el.style.left = xy[0] + "px";
    el.style.top = xy[1] + "px";
    el.style.display = "block";
  }} else if (el) {{
    el.style.display = "none";
  }}
  if (svg) {{
    svg.setAttribute("width", String(gd.clientWidth));
    svg.setAttribute("height", String(gd.clientHeight));
    const bits = [];
    (pack.lcfs && pack.lcfs[i] || []).forEach((poly) => {{
      if (poly && poly.length > 2) bits.push('<path class="iso-lcfs" d="' + polyPath(gd, poly) + '"/>');
    }});
    (pack.q95_surf && pack.q95_surf[i] || []).forEach((poly) => {{
      if (poly && poly.length > 2) bits.push('<path class="iso-q95" d="' + polyPath(gd, poly) + '"/>');
    }});
    svg.innerHTML = bits.join("");
  }}
  const k = pack.kappa && pack.kappa[i];
  const q = pack.q95 && pack.q95[i];
  const lcfsTxt = document.getElementById("leg-lcfs");
  const q95Txt = document.getElementById("leg-q95");
  if (lcfsTxt) lcfsTxt.textContent = Number.isFinite(k) ? "LCFS  κ = " + k.toFixed(2) : "LCFS";
  if (q95Txt) q95Txt.textContent = Number.isFinite(q) ? "q95 surface  q = " + q.toFixed(2) : "q95 surface";
  if (legend) legend.style.display = "flex";
}}

async function drawPsi(pack, i) {{
  const z = pack.psi[i];
  if (!z) return;
  const title = "ψ at t = " + pack.t[i].toFixed(1) + " ms  (frame " + i + ")";
  const zmin = pack.zmin;
  const zmax = pack.zmax;
  const xrange = pack.xrange;
  const yrange = pack.yrange;
  const axis = {{
    title: title,
    showlegend: false,
    uirevision: "psi-lock",
    dragmode: false,
    xaxis: {{title: "R (m)", range: xrange, autorange: false, fixedrange: true, constrain: "domain", scaleanchor: "y", scaleratio: 1}},
    yaxis: {{title: "Z (m)", range: yrange, autorange: false, fixedrange: true, constrain: "domain"}},
    height: 560,
    margin: {{t: 48, r: 80, b: 48, l: 56}},
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
  }};
  if (!psiReady) {{
    await Plotly.newPlot("psi", [{{
      type: "heatmap",
      x: pack.R,
      y: pack.Z,
      z: z,
      zmin: zmin,
      zmax: zmax,
      zauto: false,
      colorscale: "RdBu",
      reversescale: true,
      colorbar: {{title: "ψ"}},
    }}], axis, {{responsive: true, displayModeBar: false}});
    psiReady = true;
    placeOverlays(pack, i);
    return;
  }}
  await Plotly.restyle("psi", {{z: [z], zmin: [zmin], zmax: [zmax], zauto: [false]}}, [0]);
  await Plotly.relayout("psi", {{
    title: title,
    "xaxis.range": xrange,
    "yaxis.range": yrange,
    "xaxis.autorange": false,
    "yaxis.autorange": false,
  }});
  placeOverlays(pack, i);
}}

async function boot() {{
  const payload = await (await fetch(BASE + "/scalars")).json();
  times = payload.t;
  const scrub = document.getElementById("scrub");
  if (scrub) {{
    scrub.max = String(times.length - 1);
    scrub.value = String(START);
  }}
  let pack = null;
  if (LABELLED) pack = await (await fetch(BASE + "/psi")).json();
  const stamp = document.getElementById("stamp");
  async function go(i) {{
    i = Math.max(0, Math.min(times.length - 1, i));
    if (scrub) scrub.value = String(i);
    if (stamp) stamp.textContent = "t = " + times[i].toFixed(1) + " ms";
    const heading = document.getElementById("heading");
    const label = SHOT + " · frame " + i + " of " + times.length + " · " + times[i].toFixed(1) + " ms";
    if (heading) heading.textContent = label;
    document.title = label;
    if (pack) await drawPsi(pack, i);
    drawScalars(payload, i);
    return i;
  }}
  window.addEventListener("resize", () => {{
    if (pack && scrub) placeOverlays(pack, Number(scrub.value) || 0);
  }});
  const play = document.getElementById("play");
  if (play) play.addEventListener("click", () => {{
    playing = !playing;
    play.textContent = playing ? "Pause" : "Play";
    if (!playing) {{ clearInterval(timer); return; }}
    timer = setInterval(() => {{
      let i = Number((scrub && scrub.value) || 0) + 1;
      if (i >= times.length) i = 0;
      go(i);
    }}, 120);
  }});
  if (scrub) scrub.addEventListener("input", (ev) => go(Number(ev.target.value)));
  go(START);
}}
boot();
</script>"""
    )
    return _html(stem, body)


def shot_page_compare(machine: str, bucket: str, stem: str, spec: dict) -> bytes:
    model_id = _model_id(machine)
    body = (
        _crumb(
            (f"/{machine}", machine.upper()),
            (f"/{machine}/{bucket}", spec["title"]),
            (f"/{machine}/{bucket}/{stem}", stem),
        )
        + f'<h1 id="heading">{stem}</h1>'
        + f'<p class="lead">{model_id} on this shot’s <code>efit_times</code>. '
        "R² rows are unclipped. S still uses the official floor. "
        "The shot column is this shot only. "
        "The frame column is the outright numbers on one map: R²_ψ, q95, βN, and 1−D. "
        "R² of one q95 or βN number is not defined.</p>"
        + '<table class="score" id="score-table">'
        "<thead><tr><th>term</th><th>scored file</th><th>this shot</th><th>this frame (pred / EFIT)</th></tr></thead>"
        "<tbody>"
        + "".join(
            f'<tr><td>{label}</td><td id="v-{sid}">…</td><td id="s-{sid}">…</td><td id="f-{sid}">…</td></tr>'
            for sid, label in SCORE_ROWS
        )
        + "</tbody></table>"
        + '<div class="controls">'
        '<button class="btn small" id="play" type="button">Play</button>'
        '<input id="scrub" type="range" min="0" max="0" value="0"/>'
        '<span id="stamp"></span></div>'
        + '<p class="meta" id="frame-meta"></p>'
        + '<div class="psi-row">'
        + "".join(
            f'<div class="psi-cell"><h2>{title}</h2><div class="psi-wrap">'
            f'<div id="{hid}"></div></div></div>'
            for hid, title in (
                ("psi-efit", "EFIT"),
                ("psi-pred", model_id),
                ("psi-diff", f"{model_id} − EFIT"),
                ("psi-hist", "error hist"),
            )
        )
        + "</div>"
        + '<p class="legend-line">O-point pale green / pale blue. Diamond = X-point (LCFS tip). Purple = equal-Te pair. Error hist should spike at 0.</p>'
        + '<div class="psi-row">'
        + "".join(
            f'<div class="psi-cell"><h2>{title}</h2><div id="{hid}"></div></div>'
            for hid, title in (
                ("jphi-efit", "jφ EFIT"),
                ("jphi-pred", "jφ osf"),
                ("psi-pinch", "pinch zoom"),
                ("psi-pca", "EFIT − proj₅"),
            )
        )
        + "</div>"
        + "<h2>Outputs</h2><div id=\"outputs\"></div>"
        + "<h2>Consistency</h2><div id=\"cons\"></div>"
        + "<h2>Inputs</h2><div id=\"inputs\"></div>"
        + "<h2>Thomson Te</h2>"
        + '<div class="psi-grid"><div id="thom-core"></div><div id="thom-edge"></div></div>'
        + "<h2>Thomson ne</h2>"
        + '<div class="psi-grid"><div id="thom-core-ne"></div><div id="thom-edge-ne"></div></div>'
        + ("<h2>F-coils / ECOILA</h2>" if machine == "d3d" else "<h2>P-coils / EFPS</h2>")
        + '<div id="coils"></div>'
        + "<h2>Wall ψ(t)</h2><div id=\"wall\"></div>"
        + f"""<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script>
const BASE = "/api/{machine}/{bucket}/{stem}";
const SHOT = "{stem}";
const MODEL = "{model_id}";
const START = 0;
let times = [];
let playing = false;
let timer = null;
const ready = {{}};

const PAIRS = [
  {{title: "q95", efit: "efit_q95", pred: MODEL + "_q95"}},
  {{title: "βN", efit: "efit_beta_n", pred: MODEL + "_betaN"}},
];
const CONS = {json.dumps(list(CONS_SCALARS))};
const COL_EFIT = "#1b8a3e";
const COL_PRED = "#1a5fb4";
const COL_DIFF = "#c62828";
const COL_EFIT_PALE = "#8fd99a";
const COL_PRED_PALE = "#8ec6f2";
const COL_PAIR = "#7b2d8e";
let diag = null;
let cohort = null;

function fmt(v, digits) {{
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  const x = Number(v);
  if (Math.abs(x) >= 100) return x.toExponential(3);
  return x.toFixed(digits);
}}

function axName(k) {{
  return {{x: k ? "x" + (k + 1) : "x", y: k ? "y" + (k + 1) : "y"}};
}}

function stackLayout(titles) {{
  const n = titles.length;
  const layout = {{
    height: Math.max(280, 190 * n),
    margin: {{t: 36, r: 48, b: 40, l: 64}},
    showlegend: true,
    legend: {{orientation: "h", y: 1.08, x: 0}},
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
  }};
  for (let k = 0; k < n; k++) {{
    const top = 1 - k / n;
    const bot = 1 - (k + 1) / n + (k === n - 1 ? 0 : 0.045);
    const xa = k ? "xaxis" + (k + 1) : "xaxis";
    const ya = k ? "yaxis" + (k + 1) : "yaxis";
    const ax = axName(k);
    layout[xa] = {{
      matches: k ? "x" : undefined,
      title: k === n - 1 ? "t (ms)" : "",
      showticklabels: k === n - 1,
      domain: [0, 1],
      anchor: ax.y,
    }};
    layout[ya] = {{
      title: titles[k],
      domain: [bot, top - 0.01],
      anchor: ax.x,
      automargin: true,
      zeroline: true,
    }};
  }}
  return layout;
}}

function drawStack(id, panels, t, highlight) {{
  const titles = panels.map((p) => p.title);
  const traces = [];
  const seen = {{}};
  panels.forEach((p, k) => {{
    const ax = axName(k);
    const allY = [];
    p.traces.forEach((tr) => {{
      (tr.y || []).forEach((v) => allY.push(v));
      const first = !seen[tr.name];
      seen[tr.name] = true;
      traces.push({{
        x: t, y: tr.y, name: tr.name,
        xaxis: ax.x, yaxis: ax.y,
        line: {{color: tr.color, width: 1.8}},
        legendgroup: tr.name,
        showlegend: first,
      }});
    }});
    const vals = allY.filter((v) => Number.isFinite(v));
    const lo = vals.length ? Math.min(...vals) : 0;
    const hi = vals.length ? Math.max(...vals) : 1;
    traces.push({{
      x: [t[highlight], t[highlight]], y: [lo, hi],
      xaxis: ax.x, yaxis: ax.y, mode: "lines",
      line: {{color: "#b03a2e", width: 1, dash: "dot"}},
      hoverinfo: "skip", showlegend: false,
    }});
  }});
  if (!ready[id]) {{
    Plotly.newPlot(id, traces, stackLayout(titles), {{responsive: true, displayModeBar: false}});
    ready[id] = true;
    return;
  }}
  const xs = [];
  const idx = [];
  let off = 0;
  panels.forEach((p) => {{
    off += p.traces.length;
    xs.push([t[highlight], t[highlight]]);
    idx.push(off);
    off += 1;
  }});
  Plotly.restyle(id, {{x: xs}}, idx);
}}

function drawSheet(id, sheet, title) {{
  if (!sheet || !sheet.t || !sheet.t.length) return;
  const zT = (sheet.x || []).map((_, j) => (sheet.z || []).map((row) => row[j]));
  Plotly.newPlot(id, [{{
    type: "heatmap",
    x: sheet.t,
    y: sheet.x,
    z: zT,
    colorscale: "Hot",
    colorbar: {{len: 0.7}},
  }}], {{
    title: title,
    height: 300,
    margin: {{t: 40, r: 48, b: 40, l: 56}},
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    xaxis: {{title: "t (ms)"}},
    yaxis: {{title: "channel"}},
  }}, {{responsive: true, displayModeBar: false}});
}}

function drawShotDiags(i) {{
  if (!diag) return;
  const sh = diag.shot;
  const t = diag.t;
  drawPairs("cons", CONS.map((name) => ({{
    title: name,
    efit: (sh.cons_efit && sh.cons_efit[name]) || [],
    pred: (sh.cons_pred && sh.cons_pred[name]) || [],
  }})), t, i);
  drawStack("inputs", [
    {{title: "Ip / sol", traces: [
      {{y: sh.ip, name: "Ip", color: "#1e3a4c"}},
      {{y: sh.sol, name: "sol", color: "#3d5a3d"}},
    ]}},
    {{title: "dsep", traces: [{{y: sh.dsep, name: "dsep", color: "#1e3a4c"}}]}},
    {{title: "ℓi", traces: [{{y: sh.efit_li, name: "ℓi", color: "#1e3a4c"}}]}},
    {{title: "R axis", traces: [{{y: sh.efit_R_axis, name: "R", color: "#1e3a4c"}}]}},
    {{title: "Z axis", traces: [{{y: sh.efit_Z_axis, name: "Z", color: "#1e3a4c"}}]}},
    {{title: "MSE", traces: [{{y: sh.mse, name: "MSE", color: "#b03a2e"}}]}},
    {{title: "jump", traces: [{{y: (sh.jump || []).map((v) => v ? 1 : 0), name: "jump", color: "#d4a017"}}]}},
    {{title: "Ampère Ip̂/Ip", traces: [{{y: sh.ampere_ratio_pred || [], name: "Ip̂/Ip", color: "#1a5fb4"}}]}},
    {{title: "Te-pair valid", traces: [
      {{y: (diag.frame.pair_valid_efit || []), name: "EFIT", color: COL_EFIT}},
      {{y: (diag.frame.pair_valid_pred || []), name: "osf", color: COL_PRED}},
    ]}},
    {{title: "Te med", traces: [
      {{y: sh.thomson_core_Te || [], name: "core Te", color: "#1e3a4c"}},
      {{y: sh.thomson_edge_Te || [], name: "edge Te", color: "#3d5a3d"}},
    ]}},
    {{title: "ne med", traces: [
      {{y: sh.thomson_core_ne || [], name: "core ne", color: "#1e3a4c"}},
      {{y: sh.thomson_edge_ne || [], name: "edge ne", color: "#3d5a3d"}},
    ]}},
  ], t, i);
  drawPairs("wall", [
    {{title: "wall ψ", efit: sh.wall_efit, pred: sh.wall_pred}},
  ], t, i);
  if (sh.coils) {{
    const coilNames = Object.keys(sh.coils);
    drawStack("coils", coilNames.map((name) => ({{
      title: name,
      traces: [{{y: sh.coils[name] || [], name: name, color: "#1e3a4c"}}],
    }})), t, i);
  }}
  if (!ready.thom) {{
    drawSheet("thom-core", sh.thomson_core_Te_sheet, "Thomson core Te");
    drawSheet("thom-edge", sh.thomson_edge_Te_sheet, "Thomson edge Te");
    drawSheet("thom-core-ne", sh.thomson_core_ne_sheet, "Thomson core ne");
    drawSheet("thom-edge-ne", sh.thomson_edge_ne_sheet, "Thomson edge ne");
    ready.thom = true;
  }}
}}

function finiteSpan(y) {{
  const vals = y.filter((v) => Number.isFinite(v));
  if (!vals.length) return [0, 1];
  return [Math.min(...vals), Math.max(...vals)];
}}

function xy(index) {{
  return {{
    x: index ? "x" + (index + 1) : "x",
    y: index ? "y" + (index + 1) : "y",
    xName: index ? "xaxis" + (index + 1) : "xaxis",
    yName: index ? "yaxis" + (index + 1) : "yaxis",
  }};
}}

function layoutPairs(titles) {{
  const n = titles.length;
  const layout = {{
    height: Math.max(320, 240 * n),
    margin: {{t: 40, r: 48, b: 40, l: 56}},
    showlegend: true,
    legend: {{orientation: "h", y: 1.06, x: 0}},
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
  }};
  for (let k = 0; k < n; k++) {{
    const block = 1 / n;
    const top = 1 - k * block;
    const bot = 1 - (k + 1) * block + (k === n - 1 ? 0 : 0.04);
    const split = bot + (top - bot) * 0.32;
    const val = xy(2 * k);
    const dlt = xy(2 * k + 1);
    layout[val.xName] = {{
      matches: k ? "x" : undefined,
      showticklabels: false,
      domain: [0, 1],
      anchor: val.y,
    }};
    layout[val.yName] = {{
      title: titles[k],
      domain: [split + 0.02, top - 0.01],
      anchor: val.x,
      automargin: true,
      zeroline: false,
    }};
    layout[dlt.xName] = {{
      matches: "x",
      title: k === n - 1 ? "t (ms)" : "",
      showticklabels: k === n - 1,
      domain: [0, 1],
      anchor: dlt.y,
    }};
    layout[dlt.yName] = {{
      title: "Δ",
      domain: [bot, split],
      anchor: dlt.x,
      automargin: true,
      zeroline: true,
    }};
  }}
  return layout;
}}

function pairTraces(panels, t, highlight) {{
  const traces = [];
  panels.forEach((p, k) => {{
    const val = xy(2 * k);
    const dlt = xy(2 * k + 1);
    const efit = p.efit || [];
    const pred = p.pred || [];
    const diff = pred.map((v, i) => (Number.isFinite(v) && Number.isFinite(efit[i]) ? v - efit[i] : null));
    const [vlo, vhi] = finiteSpan(efit.concat(pred));
    const [dlo, dhi] = finiteSpan(diff);
    traces.push({{
      x: t, y: efit, xaxis: val.x, yaxis: val.y, name: "EFIT",
      line: {{color: COL_EFIT, width: 1.8}}, legendgroup: "efit", showlegend: k === 0,
    }});
    traces.push({{
      x: t, y: pred, xaxis: val.x, yaxis: val.y, name: MODEL,
      line: {{color: COL_PRED, width: 1.8}}, legendgroup: "pred", showlegend: k === 0,
    }});
    traces.push({{
      x: t, y: diff, xaxis: dlt.x, yaxis: dlt.y, name: "Δ",
      line: {{color: COL_DIFF, width: 1.6}}, legendgroup: "diff", showlegend: k === 0,
    }});
    traces.push({{
      x: [t[highlight], t[highlight]], y: [vlo, vhi],
      xaxis: val.x, yaxis: val.y, mode: "lines",
      line: {{color: "#888888", width: 1, dash: "dot"}},
      hoverinfo: "skip", showlegend: false,
    }});
    traces.push({{
      x: [t[highlight], t[highlight]], y: [dlo, dhi],
      xaxis: dlt.x, yaxis: dlt.y, mode: "lines",
      line: {{color: "#888888", width: 1, dash: "dot"}},
      hoverinfo: "skip", showlegend: false,
    }});
  }});
  return traces;
}}

function drawPairs(id, panels, t, highlight) {{
  if (!ready[id]) {{
    Plotly.newPlot(id, pairTraces(panels, t, highlight), layoutPairs(panels.map((p) => p.title)), {{responsive: true, displayModeBar: false}});
    ready[id] = true;
    return;
  }}
  const xs = [];
  const idx = [];
  panels.forEach((_, k) => {{
    xs.push([t[highlight], t[highlight]], [t[highlight], t[highlight]]);
    idx.push(5 * k + 3, 5 * k + 4);
  }});
  Plotly.restyle(id, {{x: xs}}, idx);
}}

function drawScalars(payload, i) {{
  drawPairs("outputs", PAIRS.map((pair) => ({{
    title: pair.title,
    efit: payload.series[pair.efit] || [],
    pred: payload.series[pair.pred] || [],
  }})), payload.t, i);
}}

function axisLayout(title, xrange, yrange) {{
  return {{
    title: title ? {{text: title, y: 0.99, yanchor: "top", pad: {{t: 2, b: 8}}}} : "",
    showlegend: false,
    uirevision: "psi-lock",
    dragmode: false,
    xaxis: {{title: "R (m)", range: xrange, autorange: false, fixedrange: true, constrain: "domain", scaleanchor: "y", scaleratio: 1}},
    yaxis: {{title: "Z (m)", range: yrange, autorange: false, fixedrange: true, constrain: "domain"}},
    height: 420,
    margin: {{t: 28, r: 56, b: 40, l: 138}},
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    legend: {{
      x: -0.06, y: 1.0, xanchor: "right", yanchor: "top",
      font: {{size: 10}},
      bgcolor: "rgba(255,255,255,0.95)",
      borderwidth: 0,
      itemsizing: "constant",
      itemwidth: 32,
      tracegroupgap: 10,
      valign: "middle",
    }},
  }};
}}

function barSpec(title) {{
  const spec = {{len: 0.42, y: 0.28, yanchor: "middle", thickness: 10, outlinewidth: 0, xpad: 4}};
  if (title) spec.title = {{text: title, side: "right"}};
  return spec;
}}

function med2d(z) {{
  const v = [];
  z.forEach((row) => row.forEach((x) => {{ if (Number.isFinite(x)) v.push(x); }}));
  if (!v.length) return 0;
  v.sort((a, b) => a - b);
  return v[v.length >> 1];
}}

function lcfsTrace(poly, color, name, show) {{
  if (!poly || poly.length < 3) {{
    return {{x: [], y: [], mode: "lines", line: {{color: color, width: 2}}, name: name, showlegend: show, hoverinfo: "skip"}};
  }}
  const rr = poly.map((p) => p[0]);
  const zz = poly.map((p) => p[1]);
  rr.push(rr[0]); zz.push(zz[0]);
  return {{x: rr, y: zz, mode: "lines", line: {{color: color, width: 2.2}}, name: name, showlegend: show, hoverinfo: "skip"}};
}}

function opointTrace(r, z, name, show, color) {{
  const ok = Number.isFinite(r) && Number.isFinite(z);
  return {{
    x: ok ? [r] : [],
    y: ok ? [z] : [],
    mode: "markers",
    marker: {{
      size: 14,
      color: color,
      line: {{color: color, width: 3}},
      symbol: "circle-open",
    }},
    name: name,
    showlegend: show,
    hovertemplate: name + "<br>R = %{{x:.3f}} m<br>Z = %{{y:.3f}} m<extra></extra>",
  }};
}}

function xyOk(xy) {{
  return xy && Number.isFinite(xy[0]) && Number.isFinite(xy[1]);
}}

function xpointTrace(xy, name, show, color) {{
  const ok = xyOk(xy);
  return {{
    x: ok ? [xy[0]] : [],
    y: ok ? [xy[1]] : [],
    mode: "markers",
    marker: {{size: 11, color: color, symbol: "diamond", line: {{color: "#111", width: 1}}}},
    name: name,
    showlegend: show,
    hovertemplate: name + "<br>R = %{{x:.3f}} m<br>Z = %{{y:.3f}} m<extra></extra>",
  }};
}}

function tePairTraces(pairs, show) {{
  if (!pairs || !pairs.length) return [];
  const xs = [], ys = [];
  pairs.forEach((p) => {{
    xs.push(p.r[0], p.r[1], null);
    ys.push(p.z[0], p.z[1], null);
  }});
  return [{{
    x: xs, y: ys, mode: "lines+markers",
    line: {{color: COL_PAIR, width: 1.6}},
    marker: {{size: 7, color: COL_PAIR}},
    name: "Te pair",
    showlegend: show,
    hoverinfo: "skip",
  }}];
}}

async function drawHeat(id, x, y, z, zmin, zmax, title, xrange, yrange, colorscale, reversescale, extras) {{
  extras = extras || {{}};
  const axis = axisLayout(title, xrange, yrange);
  const heat = {{
    type: "heatmap", x: x, y: y, z: z,
    zmin: zmin, zmax: zmax, zauto: false,
    colorscale: colorscale, reversescale: reversescale,
    colorbar: barSpec(extras.barTitle),
    showlegend: false,
    hoverinfo: "skip",
  }};
  const onE = extras.onE;
  const onP = extras.onP;
  axis.showlegend = true;
  const traces = [
    heat,
    lcfsTrace(onE ? extras.efitL : null, "#d4a017", "EFIT last-closed", extras.legE),
    lcfsTrace(onP ? extras.predL : null, "#1a9a8c", "osf last-closed", extras.legP),
    opointTrace(onE ? extras.eR : NaN, onE ? extras.eZ : NaN, "EFIT O-point", onE, COL_EFIT_PALE),
    opointTrace(onP ? extras.pR : NaN, onP ? extras.pZ : NaN, "osf O-point", onP, COL_PRED_PALE),
    xpointTrace(onE ? extras.eX : null, "EFIT X-point", onE, COL_EFIT_PALE),
    xpointTrace(onP ? extras.pX : null, "osf X-point", onP, COL_PRED_PALE),
  ];
  if (extras.pairs && extras.showPairs) traces.push(...tePairTraces(extras.pairs, extras.legPairs));
  await Plotly.react(id, traces, axis, {{responsive: true, displayModeBar: false}});
}}

async function drawHist(i) {{
  if (!diag) return;
  const fr = diag.frame;
  const x = fr.hist_x;
  const y = fr.hist[i];
  const layout = {{
    title: "pred − EFIT",
    height: 420, margin: {{t: 40, r: 16, b: 40, l: 40}},
    paper_bgcolor: "#ffffff", plot_bgcolor: "#ffffff",
    xaxis: {{title: "Δψ"}}, yaxis: {{title: "pixels"}},
    showlegend: false,
  }};
  const tr = {{type: "bar", x: x, y: y, marker: {{color: "#1e3a4c"}}}};
  if (!ready.hist) {{
    await Plotly.newPlot("psi-hist", [tr], layout, {{responsive: true, displayModeBar: false}});
    ready.hist = true;
    return;
  }}
  await Plotly.restyle("psi-hist", {{y: [y]}}, [0]);
}}

function minus(a, b) {{
  return a.map((row, i) => row.map((v, j) => v - b[i][j]));
}}

function lockRanges(efit, pred, i) {{
  let zmin = Infinity, zmax = -Infinity, dmax = 0;
  const a = efit.psi[i], b = pred.psi[i];
  for (let r = 0; r < a.length; r++) {{
    for (let c = 0; c < a[r].length; c++) {{
      const ev = a[r][c], pv = b[r][c];
      if (Number.isFinite(ev)) {{ zmin = Math.min(zmin, ev); zmax = Math.max(zmax, ev); }}
      if (Number.isFinite(pv)) {{ zmin = Math.min(zmin, pv); zmax = Math.max(zmax, pv); }}
      if (Number.isFinite(ev) && Number.isFinite(pv)) dmax = Math.max(dmax, Math.abs(pv - ev));
    }}
  }}
  if (!Number.isFinite(zmin) || zmin === zmax) {{ zmin = 0; zmax = 1; }}
  const pad = 0.15 * (zmax - zmin);
  return {{zmin: zmin - pad, zmax: zmax + pad, dmax: dmax || 1}};
}}

function jphiMap(R, Z, psi) {{
  const MU0 = 4e-7 * Math.PI;
  const nZ = psi.length, nR = psi[0].length;
  const dR = R.length > 1 ? R[1] - R[0] : 0.03;
  const dZ = Z.length > 1 ? Z[1] - Z[0] : 0.05;
  const out = [];
  for (let iz = 0; iz < nZ; iz++) {{
    const row = [];
    for (let ir = 0; ir < nR; ir++) {{
      const r = Math.max(R[ir], 1e-6);
      const im = Math.max(0, iz - 1), ip = Math.min(nZ - 1, iz + 1);
      const jm = Math.max(0, ir - 1), jp = Math.min(nR - 1, ir + 1);
      const d2r = (psi[iz][jp] - 2 * psi[iz][ir] + psi[iz][jm]) / (dR * dR);
      const d2z = (psi[ip][ir] - 2 * psi[iz][ir] + psi[im][ir]) / (dZ * dZ);
      const d1r = (psi[iz][jp] - psi[iz][jm]) / ((jp - jm) * dR || 1e-9);
      row.push((d2r - d1r / r + d2z) / (MU0 * r));
    }}
    out.push(row);
  }}
  return out;
}}

function absMax(z) {{
  const vals = [];
  for (let i = 2; i < z.length - 2; i++) for (let j = 2; j < z[i].length - 2; j++) {{
    const v = z[i][j];
    if (Number.isFinite(v)) vals.push(Math.abs(v));
  }}
  if (!vals.length) return 1;
  vals.sort((a, b) => a - b);
  return vals[Math.floor(0.98 * (vals.length - 1))] || 1;
}}

function cropTips(R, Z, psi, a, b, pad) {{
  const pts = [a, b].filter(xyOk);
  if (!pts.length) {{
    return {{R: R, Z: Z, z: psi, xrange: [R[0], R[R.length - 1]], yrange: [Z[0], Z[Z.length - 1]]}};
  }}
  const rmin = Math.min(...pts.map((p) => p[0])) - pad;
  const rmax = Math.max(...pts.map((p) => p[0])) + pad;
  const zmin = Math.min(...pts.map((p) => p[1])) - pad;
  const zmax = Math.max(...pts.map((p) => p[1])) + pad;
  const rKeep = [], zKeep = [];
  R.forEach((v, i) => {{ if (v >= rmin && v <= rmax) rKeep.push([v, i]); }});
  Z.forEach((v, i) => {{ if (v >= zmin && v <= zmax) zKeep.push([v, i]); }});
  if (rKeep.length < 3 || zKeep.length < 3) {{
    return {{R: R, Z: Z, z: psi, xrange: [rmin, rmax], yrange: [zmin, zmax]}};
  }}
  const rr = rKeep.map((p) => p[0]);
  const zz = zKeep.map((p) => p[0]);
  const zc = zKeep.map((zp) => rKeep.map((rp) => psi[zp[1]][rp[1]]));
  return {{R: rr, Z: zz, z: zc, xrange: [rr[0], rr[rr.length - 1]], yrange: [zz[0], zz[zz.length - 1]]}};
}}

function pcaResidual(efit, i) {{
  const pca = cohort && cohort.pca;
  if (!pca || !pca.mean || !pca.mean.length) return null;
  const z = efit.psi[i];
  const nZ = z.length, nR = z[0].length;
  const flat = [];
  for (let r = 0; r < nZ; r++) for (let c = 0; c < nR; c++) flat.push(z[r][c]);
  if (flat.length !== pca.mean.length) return null;
  const C = flat.map((v, k) => v - pca.mean[k]);
  const proj = pca.mean.slice();
  (pca.comps || []).forEach((comp) => {{
    let coeff = 0;
    for (let k = 0; k < C.length; k++) coeff += C[k] * comp[k];
    for (let k = 0; k < C.length; k++) proj[k] += coeff * comp[k];
  }});
  const out = [];
  let k = 0;
  for (let r = 0; r < nZ; r++) {{
    const row = [];
    for (let c = 0; c < nR; c++) row.push(flat[k] - proj[k++]);
    out.push(row);
  }}
  return out;
}}

async function drawExtraMaps(efit, pred, i, extras) {{
  const t = efit.t[i].toFixed(1);
  const je = jphiMap(efit.R, efit.Z, efit.psi[i]);
  const jp = jphiMap(pred.R, pred.Z, pred.psi[i]);
  const jmax = Math.max(absMax(je), absMax(jp));
  const amp = diag && diag.frame.ampere_ratio_pred ? diag.frame.ampere_ratio_pred[i] : null;
  await drawHeat("jphi-efit", efit.R, efit.Z, je, -jmax, jmax, "jφ EFIT  t = " + t + " ms", efit.xrange, efit.yrange, "RdBu", true, Object.assign({{}}, extras, {{showPairs: false, onE: true, onP: false, legE: true, legP: false}}));
  await drawHeat("jphi-pred", pred.R, pred.Z, jp, -jmax, jmax, "jφ osf  Ip̂/Ip = " + fmt(amp, 3), pred.xrange, pred.yrange, "RdBu", true, Object.assign({{}}, extras, {{showPairs: false, onE: false, onP: true, legE: false, legP: true}}));
  const crop = cropTips(efit.R, efit.Z, efit.psi[i], extras.eX, extras.pX, 0.25);
  const pinchAxis = axisLayout("pinch  t = " + t + " ms", crop.xrange, crop.yrange);
  pinchAxis.showlegend = true;
  await Plotly.react("psi-pinch", [
    {{type: "heatmap", x: crop.R, y: crop.Z, z: crop.z, colorscale: "RdBu", reversescale: true, colorbar: barSpec(), showlegend: false}},
    {{type: "contour", x: efit.R, y: efit.Z, z: efit.psi[i], showscale: false, ncontours: 14, line: {{color: "#333", width: 0.8}}, contours: {{coloring: "none"}}, hoverinfo: "skip", showlegend: false}},
    xpointTrace(extras.eX, "EFIT tip", true, COL_EFIT_PALE),
    xpointTrace(extras.pX, "osf tip", true, COL_PRED_PALE),
  ], pinchAxis, {{responsive: true, displayModeBar: false}});
  const res = pcaResidual(efit, i);
  const share = cohort && cohort.residual_share && cohort.residual_share[SHOT] ? cohort.residual_share[SHOT][i] : null;
  const kept = cohort && cohort.pca ? cohort.pca.share_kept : null;
  if (res) {{
    const rmax = absMax(res);
    await drawHeat("psi-pca", efit.R, efit.Z, res, -rmax, rmax, "EFIT − proj₅  share = " + fmt(share, 4) + "  kept = " + fmt(kept, 4), efit.xrange, efit.yrange, "RdBu", true, Object.assign({{}}, extras, {{showPairs: false, onE: true, onP: false, legE: true, legP: false}}));
  }}
}}

function setCellTitle(id, text) {{
  const el = document.getElementById(id);
  const h2 = el && el.closest(".psi-cell") && el.closest(".psi-cell").querySelector("h2");
  if (h2) h2.textContent = text;
}}

async function drawTriple(efit, pred, i, lock) {{
  const t = efit.t[i].toFixed(1);
  const diff = minus(pred.psi[i], efit.psi[i]);
  const eL = diag && diag.frame.lcfs_efit ? diag.frame.lcfs_efit[i] : null;
  const pL = diag && diag.frame.lcfs_pred ? diag.frame.lcfs_pred[i] : null;
  const extras = {{
    efitL: eL, predL: pL,
    eR: efit.axis_r && efit.axis_r[i], eZ: efit.axis_z && efit.axis_z[i],
    pR: pred.axis_r && pred.axis_r[i], pZ: pred.axis_z && pred.axis_z[i],
    eX: diag && diag.frame.xpoint_efit ? diag.frame.xpoint_efit[i] : null,
    pX: diag && diag.frame.xpoint_pred ? diag.frame.xpoint_pred[i] : null,
    pairs: diag && diag.frame.te_pairs ? diag.frame.te_pairs[i] : null,
  }};
  setCellTitle("psi-efit", "EFIT  t = " + t + " ms");
  setCellTitle("psi-pred", MODEL + "  t = " + t + " ms");
  const med = med2d(diff);
  const local = diff.map((row) => row.map((v) => v - med));
  const rlim = 0.5 * (lock.zmax - lock.zmin);
  setCellTitle("psi-diff", "pred − EFIT − median  t = " + t + " ms");
  await drawHeat("psi-efit", efit.R, efit.Z, efit.psi[i], lock.zmin, lock.zmax, "", efit.xrange, efit.yrange, "RdBu", true, Object.assign({{}}, extras, {{onE: true, onP: false, showPairs: true, legE: true, legP: false, legPairs: true}}));
  await drawHeat("psi-pred", pred.R, pred.Z, pred.psi[i], lock.zmin, lock.zmax, "", pred.xrange, pred.yrange, "RdBu", true, Object.assign({{}}, extras, {{onE: false, onP: true, showPairs: true, legE: false, legP: true, legPairs: false}}));
  await drawHeat("psi-diff", efit.R, efit.Z, local, -rlim, rlim, "", efit.xrange, efit.yrange, "RdBu", true, Object.assign({{}}, extras, {{onE: true, onP: true, showPairs: false, legE: false, legP: false, barTitle: "Δψ"}}));
  await drawHist(i);
  await drawExtraMaps(efit, pred, i, extras);
  const meta = document.getElementById("frame-meta");
  if (meta && diag) {{
    const fr = diag.frame;
    const share = cohort && cohort.residual_share && cohort.residual_share[SHOT] ? cohort.residual_share[SHOT][i] : null;
    meta.textContent = "frame MSE = " + fmt(fr.mse[i], 5)
      + " · jump (≥ " + (diag.jump_mm * 1000).toFixed(0) + " mm) = " + (fr.jump[i] ? "yes" : "no")
      + " · Ampère Ip̂/Ip = " + fmt(fr.ampere_ratio_pred && fr.ampere_ratio_pred[i], 3)
      + " · Te-pair valid EFIT/osf = " + fmt(fr.pair_valid_efit && fr.pair_valid_efit[i], 2) + " / " + fmt(fr.pair_valid_pred && fr.pair_valid_pred[i], 2)
      + " · PCA residual share = " + fmt(share, 4);
  }}
}}

async function boot() {{
  const payload = await (await fetch(BASE + "/scalars")).json();
  times = payload.t;
  const scrub = document.getElementById("scrub");
  if (scrub) {{
    scrub.max = String(times.length - 1);
    scrub.value = String(START);
  }}
  const efit = await (await fetch(BASE + "/psi")).json();
  const pred = await (await fetch(BASE + "/pred")).json();
  try {{
    const r = await fetch(BASE + "/diag");
    if (r.ok) diag = await r.json();
  }} catch (e) {{}}
  try {{
    const r = await fetch("/api/mast/validation/cohort");
    if (r.ok) cohort = await r.json();
  }} catch (e) {{}}
  let lock = lockRanges(efit, pred, START);
  const stamp = document.getElementById("stamp");
  let scores = null;
  function pair(pred, efit) {{
    return fmt(pred, 3) + " / " + fmt(efit, 3);
  }}
  function fillCol(prefix, col) {{
    if (!col) return;
    ["S", "r2_psi", "r2_q95", "r2_betaN", "r2_qb", "one_minus_D", "Consistency"].forEach((id) => {{
      const el = document.getElementById(prefix + id);
      if (el) el.textContent = fmt(col[id], 4);
    }});
  }}
  function fillFrame(i) {{
    if (!scores) return;
    const fr = scores.frame;
    const cells = {{
      S: "—",
      r2_psi: fmt(fr.r2_psi[i], 4),
      r2_q95: pair(fr.q95 && fr.q95[i], fr.q95_efit && fr.q95_efit[i]),
      r2_betaN: pair(fr.betaN && fr.betaN[i], fr.betaN_efit && fr.betaN_efit[i]),
      r2_qb: "—",
      one_minus_D: fmt(fr.one_minus_D[i], 4),
      Consistency: "—",
    }};
    Object.keys(cells).forEach((id) => {{
      const el = document.getElementById("f-" + id);
      if (el) el.textContent = cells[id];
    }});
  }}
  async function go(i) {{
    i = Math.max(0, Math.min(times.length - 1, i));
    if (scrub) scrub.value = String(i);
    if (stamp) stamp.textContent = "t = " + times[i].toFixed(1) + " ms";
    const heading = document.getElementById("heading");
    const label = SHOT + " · " + MODEL + " · frame " + i + " of " + times.length + " · " + times[i].toFixed(1) + " ms";
    if (heading) heading.textContent = label;
    document.title = label;
    fillFrame(i);
    lock = lockRanges(efit, pred, i);
    await drawTriple(efit, pred, i, lock);
    drawScalars(payload, i);
    drawShotDiags(i);
    return i;
  }}
  fetch(BASE + "/score").then((r) => r.ok ? r.json() : null).then((pack) => {{
    if (!pack) return;
    scores = pack;
    fillCol("v-", pack.file);
    fillCol("s-", pack.shot);
    if (scrub) fillFrame(Number(scrub.value) || 0);
  }}).catch(() => {{}});
  const play = document.getElementById("play");
  if (play) play.addEventListener("click", () => {{
    playing = !playing;
    play.textContent = playing ? "Pause" : "Play";
    if (!playing) {{ clearInterval(timer); return; }}
    timer = setInterval(() => {{
      let i = Number((scrub && scrub.value) || 0) + 1;
      if (i >= times.length) i = 0;
      go(i);
    }}, 120);
  }});
  if (scrub) scrub.addEventListener("input", (ev) => go(Number(ev.target.value)));
  go(START);
}}
boot();
</script>"""
    )
    return _html(stem, body, wide=True)


@lru_cache(maxsize=8)
def _load(path: Path) -> dict:
    return load_shot(path)


def _interp(t_src, y, t_dst) -> list[float]:
    t_src = np.asarray(t_src, dtype=float)
    y = np.asarray(y, dtype=float)
    t_dst = np.asarray(t_dst, dtype=float)
    if t_src.size == 0 or y.size == 0:
        return [float("nan")] * int(t_dst.size)
    order = np.argsort(t_src)
    return np.interp(t_dst, t_src[order], y[order], left=np.nan, right=np.nan).tolist()


def api_scalars(machine: str, bucket: str, stem: str) -> dict:
    spec = _set(machine, bucket)
    path = _index(machine, bucket)[stem]
    row = _load(path)
    t = _times_ms(row)
    series = {}
    if spec["labelled"]:
        keys = list(EFIT_SCALARS)
        if machine in MODEL_IDS and _has_pred(stem, machine):
            keys = ("efit_q95", "efit_beta_n")
        for key in keys:
            if key in row and row[key] is not None:
                y = np.asarray(row[key], dtype=float)
                if y.shape == t.shape:
                    series[key] = y.tolist()
        pred_path = _pred_dir(machine) / f"{stem}.npz"
        if pred_path.is_file():
            pred = np.load(pred_path)
            mid = _model_id(machine)
            if pred["q95"].shape == t.shape:
                series[f"{mid}_q95"] = pred["q95"].astype(float).tolist()
            if pred["betaN"].shape == t.shape:
                series[f"{mid}_betaN"] = pred["betaN"].astype(float).tolist()
        return {"t": t.tolist(), "names": list(series), "series": series}
    mag_t = np.asarray(row.get("magnetics_time") or [], dtype=float)
    ip_t = np.asarray(row.get("magnetics_plasma_current_times") or mag_t, dtype=float)
    for key, val in row.items():
        if not key.startswith("magnetics_"):
            continue
        if key.endswith("_times") or key in {"magnetics_time", "magnetics_dsep"}:
            continue
        y = np.asarray(val, dtype=float)
        clock = ip_t if key == "magnetics_plasma_current" else mag_t
        if y.ndim != 1 or clock.size == 0:
            continue
        series[key] = _interp(clock, y, t)
    return {"t": t.tolist(), "names": list(series), "series": series}


def _col1(row: dict, key: str, n: int) -> np.ndarray:
    if key not in row or row[key] is None:
        return np.full(n, np.nan)
    y = np.asarray(row[key], dtype=float).reshape(-1)
    if y.size != n:
        return np.full(n, np.nan)
    return y


def _sample_psi(R: np.ndarray, Z: np.ndarray, psi2: np.ndarray, r: float, z: float) -> float:
    if not (np.isfinite(r) and np.isfinite(z)):
        return float("nan")
    ir = int(np.clip(np.searchsorted(R, r) - 1, 0, R.size - 2))
    iz = int(np.clip(np.searchsorted(Z, z) - 1, 0, Z.size - 2))
    tr = (r - R[ir]) / (R[ir + 1] - R[ir] + 1e-30)
    tz = (z - Z[iz]) / (Z[iz + 1] - Z[iz] + 1e-30)
    v00 = psi2[iz, ir]
    v10 = psi2[iz, ir + 1]
    v01 = psi2[iz + 1, ir]
    v11 = psi2[iz + 1, ir + 1]
    return float((1 - tz) * ((1 - tr) * v00 + tr * v10) + tz * ((1 - tr) * v01 + tr * v11))


def _longest_isoline(R: np.ndarray, Z: np.ndarray, psi2: np.ndarray, level: float) -> list[list[list[float]]]:
    if not np.isfinite(level):
        return []
    try:
        segs = contour_generator(x=R, y=Z, z=psi2).lines(float(level))
    except Exception:
        return []
    best = None
    for seg in segs:
        arr = np.asarray(seg, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 8:
            continue
        if best is None or arr.shape[0] > best.shape[0]:
            best = arr
    if best is None:
        return []
    return [np.round(best, 5).tolist()]


def _kappa_from_poly(poly: list[list[float]]) -> float:
    if len(poly) < 8:
        return float("nan")
    a = np.asarray(poly, dtype=float)
    dr = float(a[:, 0].max() - a[:, 0].min())
    dz = float(a[:, 1].max() - a[:, 1].min())
    if dr <= 0:
        return float("nan")
    return dz / dr


def api_psi(machine: str, bucket: str, stem: str) -> dict:
    spec = _set(machine, bucket)
    if not spec["labelled"]:
        raise KeyError("no psi")
    path = _index(machine, bucket)[stem]
    row = _load(path)
    psi = np.asarray(row["efit_psirz"], dtype=float)
    R = np.asarray(row["efit_grid_R"], dtype=float).reshape(-1)
    Z = np.asarray(row["efit_grid_Z"], dtype=float).reshape(-1)
    finite = psi[np.isfinite(psi)]
    zmin = float(np.min(finite)) if finite.size else 0.0
    zmax = float(np.max(finite)) if finite.size else 1.0
    if zmin == zmax:
        zmax = zmin + 1.0
    t = _times_ms(row)
    n = int(t.size)
    axis_r = _col1(row, "efit_r_axis", n)
    axis_z = _col1(row, "efit_z_axis", n)
    axis_psi = _col1(row, "efit_psi_axis", n)
    psi_b = _col1(row, "efit_psi_boundary", n)
    q95 = _col1(row, "efit_q95", n)
    kappa = _col1(row, "efit_elongation", n)
    lcfs: list[list[list[list[float]]]] = []
    q95_surf: list[list[list[list[float]]]] = []
    for i in range(n):
        frame = np.asarray(psi[i], dtype=float)
        axis_sign = -1.0 if machine == "d3d" else 1.0
        c_e, _ = _extract(frame, R, Z, axis_sign, machine="DIII-D" if machine == "d3d" else None)
        poly = [np.round(c_e, 5).tolist()] if c_e is not None else []
        if not poly and np.isfinite(psi_b[i]):
            poly = _longest_isoline(R, Z, frame, float(psi_b[i]))
        lcfs.append(poly)
        pa = float(axis_psi[i])
        pb = float(psi_b[i])
        if not np.isfinite(pa):
            pa = _sample_psi(R, Z, frame, float(axis_r[i]), float(axis_z[i]))
        if not np.isfinite(pb) and poly:
            samples = [_sample_psi(R, Z, frame, p[0], p[1]) for p in poly[0][:: max(1, len(poly[0]) // 16)]]
            samples = [s for s in samples if np.isfinite(s)]
            pb = float(np.mean(samples)) if samples else float("nan")
        lev95 = pa + 0.95 * (pb - pa) if np.isfinite(pa) and np.isfinite(pb) else float("nan")
        q95_surf.append(_longest_isoline(R, Z, frame, lev95))
        if not np.isfinite(kappa[i]) and poly:
            kappa[i] = _kappa_from_poly(poly[0])
    return {
        "t": t.tolist(),
        "R": R.tolist(),
        "Z": Z.tolist(),
        "psi": psi.tolist(),
        "zmin": zmin,
        "zmax": zmax,
        "xrange": [float(R.min()), float(R.max())],
        "yrange": [float(Z.min()), float(Z.max())],
        "axis_r": axis_r.tolist(),
        "axis_z": axis_z.tolist(),
        "axis_psi": axis_psi.tolist(),
        "lcfs": lcfs,
        "q95_surf": q95_surf,
        "q95": q95.tolist(),
        "kappa": kappa.tolist(),
    }


@lru_cache(maxsize=8)
def _load_pred(stem: str, machine: str = "mast") -> dict:
    path = _pred_dir(machine) / f"{stem}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    pack = np.load(path)
    return {key: pack[key] for key in pack.files}


def api_pred(machine: str, bucket: str, stem: str) -> dict:
    if machine not in MODEL_IDS or not _has_pred(stem, machine):
        raise KeyError("no pred")
    pack = _load_pred(stem, machine)
    psi = np.asarray(pack["psirz"], dtype=float)
    r = np.asarray(pack["efit_grid_R"], dtype=float).reshape(-1)
    z = np.asarray(pack["efit_grid_Z"], dtype=float).reshape(-1)
    finite = psi[np.isfinite(psi)]
    zmin = float(np.min(finite)) if finite.size else 0.0
    zmax = float(np.max(finite)) if finite.size else 1.0
    if zmin == zmax:
        zmax = zmin + 1.0
    t = np.asarray(pack["efit_times"], dtype=float)
    return {
        "t": t.tolist(),
        "R": r.tolist(),
        "Z": z.tolist(),
        "psi": psi.tolist(),
        "zmin": zmin,
        "zmax": zmax,
        "xrange": [float(r.min()), float(r.max())],
        "yrange": [float(z.min()), float(z.max())],
        "axis_r": np.asarray(pack["axis_r"], dtype=float).tolist(),
        "axis_z": np.asarray(pack["axis_z"], dtype=float).tolist(),
        "q95": np.asarray(pack["q95"], dtype=float).tolist(),
        "betaN": np.asarray(pack["betaN"], dtype=float).tolist(),
        "model_id": _model_id(machine),
    }


def api_score(machine: str, bucket: str, stem: str) -> dict:
    if machine not in MODEL_IDS or not _has_pred(stem, machine):
        raise KeyError("no score")
    path = _index(machine, bucket)[stem]
    row = _load(path)
    pred = _load_pred(stem, machine)
    pred_dir = _pred_dir(machine)
    shot_pack = load_or_compute(pred_dir, stem, row, pred)
    if machine == "mast":
        file_pack = load_or_compute_file(PRED_DIR)
        file_terms = file_pack.get("file")
        sign = file_pack.get("sign", shot_pack.get("sign"))
    else:
        file_terms = shot_pack["shot"]
        sign = shot_pack.get("sign")
    return {
        "file": file_terms,
        "shot": shot_pack["shot"],
        "frame": shot_pack["frame"],
        "sign": sign,
    }


def api_diag(machine: str, bucket: str, stem: str) -> dict:
    if machine not in MODEL_IDS or not _has_pred(stem, machine):
        raise KeyError("no diag")
    path = _index(machine, bucket)[stem]
    row = _load(path)
    pred = _load_pred(stem, machine)
    pred_dir = _pred_dir(machine)
    score = load_or_compute(pred_dir, stem, row, pred)
    return load_or_compute_diag(pred_dir, stem, row, pred, score)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} {fmt % args}")

    def _send(self, body: bytes, code: int = 200, ctype: str = "text/html; charset=utf-8") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(json.dumps(_jsonable(payload), allow_nan=False).encode(), code, "application/json")

    def _redirect(self, location: str) -> None:
        self.send_response(302)
        self.send_header("Location", location)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)
        try:
            if path == "/":
                return self._send(home())
            if path == "/static/style.css":
                return self._send(STATIC.joinpath("style.css").read_bytes(), 200, "text/css")
            if path == "/mast":
                return self._send(mast_home())
            if path == "/d3d":
                return self._send(d3d_home())
            parts = [p for p in path.split("/") if p]
            if len(parts) == 2 and (parts[0], parts[1]) in SETS:
                return self._send(shot_browser(parts[0], parts[1], query))
            if parts[:1] == ["api"] and len(parts) >= 5:
                _, machine, bucket, stem, kind = parts[:5]
                if kind == "scalars":
                    return self._json(api_scalars(machine, bucket, stem))
                if kind == "psi":
                    return self._json(api_psi(machine, bucket, stem))
                if kind == "pred":
                    return self._json(api_pred(machine, bucket, stem))
                if kind == "score":
                    return self._json(api_score(machine, bucket, stem))
                if kind == "diag":
                    return self._json(api_diag(machine, bucket, stem))
                return self._send(b"unknown api", 404)
            if len(parts) == 3 and (parts[0], parts[1]) in SETS:
                page = shot_page(parts[0], parts[1], parts[2])
                if isinstance(page, tuple):
                    return self._send(page[0], page[1])
                return self._send(page)
            if len(parts) == 4 and (parts[0], parts[1]) in SETS:
                return self._redirect(f"/{parts[0]}/{parts[1]}/{parts[2]}")
            return self._send(_html("Not found", "<h1>Not found</h1>"), 404)
        except FileNotFoundError as exc:
            return self._send(_html("No data", f"<h1>No data</h1><p>{exc}</p>"), 503)
        except (KeyError, ValueError, IndexError) as exc:
            return self._send(_html("Bad request", f"<h1>Bad request</h1><p>{exc}</p>"), 400)


def main() -> None:
    print(f"shot browser  http://{HOST}:{PORT}   {SOLVER_ID}   data {data_root()}", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
