#!/usr/bin/env python3
# Copyright (c) 2026 Lars Warren Ericson.
# Licensed under the Apache License, Version 2.0. See LICENSE and NOTICE.
"""Build docs/FEC.pdf from the top-level fec.md wiki tree.

Source of truth is ``fec.md``. Top-level TOC bullets set section order; nested
TOC bullets are subsections of the linked chapter file. Included chapters are
heading-demoted so their internal ``##`` peers become subsections of that
chapter. ``fec.md`` itself is only the TOC index (no inline body).

Usage (repo root)::

    python3 scripts/build_fec_pdf.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INDEX = ROOT / "fec.md"
OUT_MD = ROOT / "docs" / "FEC_assembled.md"
OUT_TEX = ROOT / "docs" / "FEC.tex"
OUT_PDF = ROOT / "docs" / "FEC.pdf"
PREAMBLE = HERE / "tex_preamble_fec.tex"

AUTHOR = "Lars Warren Ericson"
COMPANY = "Catskills Research Company"
GITHUB_URL = r"https://github.com/catskillsresearch/catskills_fec"
RELEASE_URL = (
    r"https://github.com/catskillsresearch/catskills_fec/"
    r"releases/tag/phase3-reproduce-03becd2"
)
ORCID = "0000-0001-8299-9361"
EMAIL = "lars.ericson@catskillsresearch.com"

TOC_ITEM = re.compile(r"^( *)- \[([^\]]+)\]\(([^)]+)\)\s*$")
HEADING = re.compile(r"^(#+)\s+(.*)$")

EXTRA_LITERATE = r"""    {ä}{{\"{a}}}1
    {é}{{\'{e}}}1
    {±}{{\ensuremath{\pm}}}1
    {→}{{\ensuremath{\rightarrow}}}1
    {×}{{\ensuremath{\times}}}1
    {·}{{\ensuremath{\cdot}}}1
    {ψ}{{\ensuremath{\psi}}}1
    {β}{{\ensuremath{\beta}}}1
    {≤}{{\ensuremath{\leq}}}1
    {≥}{{\ensuremath{\geq}}}1
    {−}{{\ensuremath{-}}}1
    {…}{{\ldots}}1
    {ε}{{\ensuremath{\varepsilon}}}1
    {κ}{{\ensuremath{\kappa}}}1
    {δ}{{\ensuremath{\delta}}}1
    {μ}{{\ensuremath{\mu}}}1
    {φ}{{\ensuremath{\varphi}}}1
    {ℓ}{{\ensuremath{\ell}}}1
    {∘}{{\ensuremath{\circ}}}1
"""

EXTRA_UNICODECHAR = r"""
\newunicodechar{ä}{\"{a}}
\newunicodechar{é}{\'{e}}
\newunicodechar{±}{\ensuremath{\pm}}
\newunicodechar{→}{\ensuremath{\rightarrow}}
\newunicodechar{×}{\ensuremath{\times}}
\newunicodechar{·}{\ensuremath{\cdot}}
\newunicodechar{ψ}{\ensuremath{\psi}}
\newunicodechar{β}{\ensuremath{\beta}}
\newunicodechar{≤}{\ensuremath{\leq}}
\newunicodechar{≥}{\ensuremath{\geq}}
\newunicodechar{−}{\ensuremath{-}}
\newunicodechar{…}{\ldots}
\newunicodechar{ε}{\ensuremath{\varepsilon}}
\newunicodechar{κ}{\ensuremath{\kappa}}
\newunicodechar{δ}{\ensuremath{\delta}}
\newunicodechar{μ}{\ensuremath{\mu}}
\newunicodechar{φ}{\ensuremath{\varphi}}
\newunicodechar{ℓ}{\ensuremath{\ell}}
\newunicodechar{∘}{\ensuremath{\circ}}
\newunicodechar{═}{=}
\newunicodechar{─}{-}
\newunicodechar{│}{|}
\newunicodechar{┌}{+}
\newunicodechar{┐}{+}
\newunicodechar{└}{+}
\newunicodechar{┘}{+}
\newunicodechar{├}{+}
\newunicodechar{┤}{+}
\newunicodechar{┬}{+}
\newunicodechar{┴}{+}
\newunicodechar{┼}{+}
\newunicodechar{▼}{\ensuremath{\downarrow}}
\newunicodechar{▲}{\ensuremath{\uparrow}}
"""

FENCE_RE = re.compile(r"^```[^\n]*\n(.*?)^```[ \t]*$", re.M | re.S)
PLACEHOLDER = "FECCODEBLOCK{}ENDBLOCK"
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")
# ASCII-only replacements so pandoc does not double-escape injected TeX.
_PLAIN = {
    "—": "---",
    "–": "--",
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
    "…": "...",
    "×": " x ",
    "±": "+/-",
    "→": "->",
    "≤": "<=",
    "≥": ">=",
    "−": "-",
    "·": ".",
    "∈": " in ",
    "ℝ": "R",
    "Σ": "Sigma",
    "Δ": "Delta",
    "ψ": "psi",
    "β": "beta",
    "φ": "phi",
    "θ": "theta",
    "γ": "gamma",
    "µ": "u",
    "μ": "mu",
    "è": "e",
    "¹": "^1",
    "²": "^2",
    "³": "^3",
    "⁸": "^8",
    "⁹": "^9",
    "⁻": "^-",
    "▼": "v",
    "▲": "^",
    "─": "-",
    "│": "|",
    "┌": "+",
    "┐": "+",
    "└": "+",
    "┘": "+",
    "├": "+",
    "┤": "+",
    "┬": "+",
    "┴": "+",
    "┼": "+",
    "═": "=",
}


def _collapse_sup_runs(md: str) -> str:
    """Turn runs like 10¹⁸ into 10^{18} (TeX math already in the wiki for real formulae)."""
    return re.sub(
        r"([0-9A-Za-z\)])([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)",
        lambda m: m.group(1) + "^{" + m.group(2).translate(_SUP) + "}",
        md,
    )


def sanitize(md: str) -> str:
    """Make wiki markdown safer for pandoc → pdflatex."""
    # Prize purse dollars are not math.
    md = md.replace("$20,000", "USD 20,000")
    md = md.replace("$20.000", "USD 20,000")
    # Pandoc+tex_math_dollars mishandles LaTeX \\[ \\] as literal brackets.
    md = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", md, flags=re.S)
    md = _collapse_sup_runs(md)
    for src, dst in _PLAIN.items():
        md = md.replace(src, dst)
    md = md.replace("\u0304", "")
    def fix_img(m: re.Match[str]) -> str:
        alt, path = m.group(1), m.group(2)
        if path.startswith(("http://", "https://", "/", "../")):
            return m.group(0)
        return f"![{alt}](../{path})"

    md = IMG_RE.sub(fix_img, md)

    def fix_link(m: re.Match[str]) -> str:
        text, href = m.group(1), m.group(2)
        if href.startswith(("http://", "https://", "#")):
            return m.group(0)
        if href.endswith(".md") or ".md#" in href:
            return text
        return m.group(0)

    return LINK_RE.sub(fix_link, md)


def github_slug(title: str) -> str:
    """Approximate GitHub heading slug (for matching TOC #anchors)."""
    s = title.strip().lower().replace("`", "")
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)
    return re.sub(r"[\s_]+", "-", s).strip("-")


def parse_toc_tree(toc_text: str) -> list[dict]:
    """Parse fec.md nested list into [{title, href, children: [...]}].

    Only the first two list levels are kept (PDF tocdepth=2).
    """
    tops: list[dict] = []
    for line in toc_text.splitlines():
        m = TOC_ITEM.match(line)
        if not m:
            continue
        indent, title, href = len(m.group(1)), m.group(2).strip(), m.group(3).strip()
        if indent == 0:
            tops.append({"title": title, "href": href, "children": []})
        elif indent >= 2 and tops:
            # Nested bullet under the current top-level entry.
            tops[-1]["children"].append({"title": title, "href": href})
    return tops


def nest_included_file(
    md: str,
    toc_title: str,
    children: list[dict],
) -> str:
    """Include a chapter so PDF levels 1–2 match the fec.md TOC.

    - Chapter title → ``##`` (LaTeX \\section).
    - Headings whose slug matches a TOC child → ``###`` with the TOC wording.
    - All other headings → ``####`` or deeper (hidden when tocdepth=2).
    - If a TOC child points at the file's opening heading (e.g. General
      Parquet Format), emit that ``###`` right after the chapter title.
    """
    child_by_slug = {}
    for ch in children:
        href = ch["href"]
        slug = href.split("#", 1)[1] if "#" in href else github_slug(ch["title"])
        child_by_slug[slug] = ch["title"]

    lines = md.splitlines()
    # Detect first-heading slug before rewriting.
    first_slug = None
    for line in lines:
        m = HEADING.match(line)
        if m:
            first_slug = github_slug(m.group(2))
            break

    out: list[str] = []
    seen_first = False
    in_fence = False
    used_child_slugs: set[str] = set()

    for line in lines:
        if line.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        m = HEADING.match(line)
        if not m:
            out.append(line)
            continue
        depth, heading_title = len(m.group(1)), m.group(2)
        slug = github_slug(heading_title)

        if not seen_first:
            seen_first = True
            out.append(f"## {toc_title}")
            # Opening-heading TOC child (content under the chapter title).
            if first_slug and first_slug in child_by_slug:
                out.append("")
                out.append(f"### {child_by_slug[first_slug]}")
                used_child_slugs.add(first_slug)
            continue

        if slug in child_by_slug:
            out.append(f"### {child_by_slug[slug]}")
            used_child_slugs.add(slug)
            continue

        # Not a TOC level-2 entry: keep body structure but below tocdepth.
        out.append("#" * max(depth + 2, 4) + " " + heading_title)

    if not seen_first:
        out.insert(0, f"## {toc_title}")

    missing = [s for s in child_by_slug if s not in used_child_slugs]
    if missing:
        raise RuntimeError(
            f"{toc_title!r}: TOC children not found as headings: {missing}"
        )
    return "\n".join(out).rstrip() + "\n"


def assemble() -> str:
    index = INDEX.read_text(encoding="utf-8")
    toc_part = index.split("\n---\n", 1)[0]

    title_line = next(
        (ln[2:].strip() for ln in toc_part.splitlines() if ln.startswith("# ")),
        "Fusion equilibrium challenge",
    )
    tops = parse_toc_tree(toc_part)

    chunks: list[str] = [
        f"# {title_line}",
        "",
        "**Abstract.**",
        "Methods notebook for the Catskills Research Company (`catskills`) "
        "NeurIPS 2026 Fusion Equilibrium Challenge entries. Structure follows "
        "the table of contents in `fec.md`: theory, parquet specs, scoring, "
        "One-Shot Forward (OSF) with MAST/DIII-D calibrations, DIII-D labelled "
        "ML fit, then Phase 2 Final and Phase 3 Reproduce. Pinned hand-over: "
        "GitHub Release `phase3-reproduce-03becd2`.",
        "",
        "---",
        "",
    ]

    seen_files: set[str] = set()
    for entry in tops:
        toc_title, href = entry["title"], entry["href"]
        if href.startswith("#"):
            raise RuntimeError(
                f"TOC entry {toc_title!r} still uses inline anchor {href!r}; "
                "break it out to a .md file and link that file from fec.md"
            )

        file_name = href.split("#", 1)[0]
        if file_name in seen_files:
            continue
        seen_files.add(file_name)
        path = ROOT / file_name
        if not path.is_file():
            raise FileNotFoundError(path)
        chunks.append(
            sanitize(
                nest_included_file(
                    path.read_text(encoding="utf-8"),
                    toc_title,
                    entry["children"],
                )
            )
        )
        chunks.append("")

    return "\n".join(chunks)


def _ascii_listing(code: str) -> str:
    """listings + pdflatex cannot host arbitrary Unicode; flatten code fences."""
    import unicodedata

    out: list[str] = []
    for ch in code:
        if ord(ch) < 128:
            out.append(ch)
            continue
        name = unicodedata.normalize("NFKD", ch)
        ascii_part = "".join(c for c in name if ord(c) < 128)
        out.append(ascii_part if ascii_part else "?")
    return "".join(out)


def extract_fences(md: str) -> tuple[str, list[str]]:
    blocks: list[str] = []

    def take(match: re.Match[str]) -> str:
        blocks.append(_ascii_listing(match.group(1).rstrip("\n")))
        return PLACEHOLDER.format(len(blocks) - 1)

    return FENCE_RE.sub(take, md), blocks


def splice_fences(latex: str, blocks: list[str]) -> str:
    for i, code in enumerate(blocks):
        listing = "\\begin{lstlisting}\n" + code + "\n\\end{lstlisting}"
        token = PLACEHOLDER.format(i)
        if token not in latex:
            raise RuntimeError(f"code placeholder {i} vanished during conversion")
        latex = latex.replace(token, listing)
    return latex


def pandoc(markdown: str, shift: bool) -> str:
    cmd = [
        "pandoc",
        "-f",
        "markdown+tex_math_dollars+raw_tex+smart",
        "-t",
        "latex",
        "--wrap=preserve",
    ]
    if shift:
        cmd.append("--shift-heading-level-by=-1")
    proc = subprocess.run(cmd, input=markdown, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError("pandoc failed")
    return proc.stdout


def split_front_matter(md: str) -> tuple[str, str, str]:
    lines = md.splitlines()
    title = ""
    front: list[str] = []
    body_start = 0
    for i, line in enumerate(lines):
        if line.startswith("# ") and not title:
            title = line[2:].strip()
            continue
        if line.strip() == "---":
            body_start = i + 1
            break
        front.append(line)
    abstract = "\n".join(front).strip()
    abstract = re.sub(r"^\*\*Abstract\.\*\*\s*", "", abstract)
    body = "\n".join(lines[body_start:]).lstrip("\n")
    return title, abstract, body


def break_texttt_paths(latex: str) -> str:
    def fix(match: re.Match[str]) -> str:
        inner = match.group(1)
        if "/" not in inner:
            return match.group(0)
        return "\\texttt{" + inner.replace("/", "/\\allowbreak{}") + "}"

    return re.sub(r"\\texttt\{([^{}]*)\}", fix, latex)


def tidy(latex: str) -> str:
    latex = latex.replace("\\pandocbounded{", "{")
    latex = re.sub(r"\\tightlist\n", "", latex)
    # Soften overfull image includes: constrain max width.
    latex = re.sub(
        r"\\includegraphics\{",
        r"\\includegraphics[width=\\linewidth,height=0.42\\textheight,keepaspectratio]{",
        latex,
    )
    return break_texttt_paths(latex)


def build_preamble() -> str:
    text = PREAMBLE.read_text(encoding="utf-8")
    if "literate=\n" not in text:
        raise RuntimeError("could not find the `literate=` table in the shared preamble")
    text = text.replace("literate=\n", "literate=\n" + EXTRA_LITERATE, 1)
    text = text.replace(
        r"\lstset{style=leanbox}",
        r"\lstset{style=leanbox,language=bash}",
        1,
    )
    return text + EXTRA_UNICODECHAR + "\n\\providecommand{\\passthrough}[1]{#1}\n"


def build_title_page(title_tex: str, abstract_tex: str) -> str:
    return "\n".join(
        [
            r"\title{\textbf{" + title_tex + "}}",
            "",
            r"\author[1]{\textbf{" + AUTHOR + "}}",
            r"\affil[1]{ORCID: " + ORCID + "}",
            r"\affil[1]{" + COMPANY + "}",
            r"\affil[1]{\texttt{" + EMAIL + "}}",
            "",
            r"\date{\today}",
            "",
            r"\begin{document}",
            r"\maketitle",
            "",
            r"\begin{center}",
            r"  \small",
            r"  \textbf{Github:} \url{" + GITHUB_URL + r"} \\",
            r"  \textbf{Phase~3 Release:} \url{" + RELEASE_URL + "}",
            r"\end{center}",
            "",
            r"\begin{abstract}",
            abstract_tex,
            r"\end{abstract}",
        ]
    )


def main() -> int:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    assembled = assemble()
    OUT_MD.write_text(assembled, encoding="utf-8")
    print(f"wrote {OUT_MD.relative_to(ROOT)} ({OUT_MD.stat().st_size:,} bytes)")

    title_md, abstract_md, body_md = split_front_matter(assembled)
    title_tex = tidy(pandoc(title_md, shift=False)).strip()
    abstract_tex = tidy(pandoc(abstract_md, shift=False)).strip()
    stripped_body, blocks = extract_fences(body_md)
    body_tex = splice_fences(tidy(pandoc(stripped_body, shift=True)), blocks)

    document = "\n".join(
        [
            build_preamble(),
            build_title_page(title_tex, abstract_tex),
            "",
            # Match fec.md nested list: section + subsection only.
            r"\setcounter{tocdepth}{2}",
            r"\tableofcontents",
            r"\newpage",
            "",
            body_tex,
            "",
            r"\end{document}",
            "",
        ]
    )
    OUT_TEX.write_text(document, encoding="utf-8")
    print(f"wrote {OUT_TEX.relative_to(ROOT)} ({OUT_TEX.stat().st_size:,} bytes)")

    proc = subprocess.run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            OUT_TEX.name,
        ],
        cwd=OUT_TEX.parent,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0 or not OUT_PDF.is_file():
        log = OUT_TEX.with_suffix(".log")
        sys.stderr.write(proc.stdout[-4000:] if proc.stdout else "")
        sys.stderr.write(proc.stderr[-2000:] if proc.stderr else "")
        if log.is_file():
            sys.stderr.write("\n--- tail of LaTeX log ---\n")
            sys.stderr.write("\n".join(log.read_text(errors="replace").splitlines()[-60:]))
        return 1

    pages = subprocess.run(
        ["pdfinfo", OUT_PDF.name],
        cwd=OUT_TEX.parent,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    n_pages = next((l.split()[1] for l in pages.splitlines() if l.startswith("Pages:")), "?")
    print(f"wrote {OUT_PDF.relative_to(ROOT)} ({OUT_PDF.stat().st_size:,} bytes, {n_pages} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
