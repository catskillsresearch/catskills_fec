#!/usr/bin/env python3
"""Pull own submission records from the Codabench participate tab into submits/.

Uses the authenticated Codabench session from Google Chrome on Linux.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import secretstorage

def find_repo_root() -> Path:
    p = Path(__file__).resolve().parent
    for _ in range(5):
        if (p / ".git").exists() or (p / "submits").exists():
            return p
        p = p.parent
    return Path.cwd()


REPO_ROOT = find_repo_root()
SUBMITS_DIR = REPO_ROOT / "submits"

PHASES = {
    "Development": 29580,
    "Final": 29581,
}

SCORE_COLS = [
    "d3d_S",
    "g_ratio",
    "mast_S",
    "d3d_r2_psi",
    "d3d_r2_qb",
    "d3d_dlcfs",
    "d3d_consistency",
    "mast_r2_psi",
    "mast_r2_qb",
    "mast_dlcfs",
    "mast_consistency",
]


def get_chrome_codabench_cookies() -> dict[str, str]:
    """Extract and decrypt Codabench cookies from Google Chrome on Linux."""
    bus = secretstorage.dbus_init()
    col = secretstorage.get_default_collection(bus)
    secret = None
    for item in col.get_all_items():
        attrs = item.get_attributes() or {}
        if attrs.get("application") == "chrome" and "chrome_libsecret" in (
            attrs.get("xdg:schema") or ""
        ):
            secret = item.get_secret()
            break

    if not secret:
        raise RuntimeError("Could not retrieve Chrome master password from keyring")

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA1(), length=16, salt=b"saltysalt", iterations=1
    )
    key = kdf.derive(secret)

    src = Path.home() / ".config/google-chrome/Default/Cookies"
    tmp = tempfile.mktemp(suffix=".db")
    shutil.copy2(src, tmp)

    con = sqlite3.connect(tmp)
    rows = list(
        con.execute(
            "SELECT name, encrypted_value FROM cookies WHERE host_key='www.codabench.org'"
        )
    )
    con.close()
    os.remove(tmp)

    cookies = {}
    for name, blob in rows:
        if blob[:3] in (b"v10", b"v11"):
            cipher = Cipher(algorithms.AES(key), modes.CBC(b" " * 16))
            decryptor = cipher.decryptor()
            plain = decryptor.update(blob[3:]) + decryptor.finalize()
            pad = plain[-1]
            body = plain[:-pad] if (1 <= pad <= 16) else plain
            # Chrome prepends 32 bytes of MAC/signature before the cookie plaintext
            cookies[name] = body[32:].decode("utf-8", errors="replace")

    return cookies


def fetch_submissions(phase_id: int, cookie_header: str) -> dict:
    url = f"https://www.codabench.org/api/submissions/?phase={phase_id}&page_size=all"
    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "Cookie": cookie_header,
        },
    )
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def scores_of(r: dict) -> dict[str, str]:
    return {s["column_key"]: s["score"] for s in r.get("scores", [])}


def fmt(x: str | float | None) -> str:
    if x is None or x == "":
        return ""
    try:
        return f"{float(x):.6f}"
    except (ValueError, TypeError):
        return str(x)


def pull_and_save(out_dir: Path = SUBMITS_DIR) -> None:
    cookies = get_chrome_codabench_cookies()
    if "sessionid" not in cookies:
        raise RuntimeError("No active Codabench sessionid cookie found in Chrome.")

    cookie_header = "; ".join(f"{k}={v}" for k, v in cookies.items())
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = fetch_submissions(PHASES["Development"], cookie_header)
    final = fetch_submissions(PHASES["Final"], cookie_header)

    rows = dev.get("results", [])
    final_rows = final.get("results", [])

    owners = {r["owner"] for r in rows}
    owner = list(owners)[0] if len(owners) == 1 else "catskills"
    today = datetime.now().strftime("%Y-%m-%d")

    # 1. submissions.json
    payload = {
        "source": "https://www.codabench.org/competitions/17739/#/participate-tab",
        "api": "https://www.codabench.org/api/submissions/?phase={phase}&page_size=all",
        "owner": owner,
        "downloaded": today,
        "phases": {
            "Development": {
                "phase_id": PHASES["Development"],
                "count": dev.get("count", len(rows)),
                "results": rows,
            },
            "Final": {
                "phase_id": PHASES["Final"],
                "count": final.get("count", len(final_rows)),
                "results": final_rows,
            },
        },
    }
    json_path = out_dir / "submissions.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    # 2. submissions.csv
    fieldnames = [
        "id",
        "created_when",
        "phase",
        "status",
        "on_leaderboard",
        "is_public",
        "filename",
        "status_details",
        *SCORE_COLS,
    ]
    with (out_dir / "submissions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            sc = scores_of(r)
            w.writerow(
                {
                    "id": r["id"],
                    "created_when": r["created_when"],
                    "phase": r.get("phase_name", "Development"),
                    "status": r["status"],
                    "on_leaderboard": r.get("on_leaderboard", False),
                    "is_public": r.get("is_public", False),
                    "filename": r.get("filename", ""),
                    "status_details": r.get("status_details", ""),
                    **{c: sc.get(c, "") for c in SCORE_COLS},
                }
            )

    # 3. index.md
    on_lb = [r["id"] for r in rows if r.get("on_leaderboard")]
    lb_text = f"submission **{on_lb[0]}**" if on_lb else "none"

    lines = [
        "# Codabench submissions — catskills",
        "",
        "Source: https://www.codabench.org/competitions/17739/#/participate-tab",
        f"Downloaded: {today}",
        "",
        f"Development (phase {PHASES['Development']}): **{len(rows)}** submissions.",
        f"Final (phase {PHASES['Final']}): **{len(final_rows)}** submissions.",
        f"One row is on the public leaderboard (19740): {lb_text}.",
        "",
        "Full API objects: `submissions.json`. Flat table: `submissions.csv`.",
        "",
        "| id | created (UTC) | on board | d3d_S | g_ratio | mast_S | d3d_r2_psi | d3d_r2_qb | d3d_dlcfs | d3d_consistency | mast_r2_psi | mast_r2_qb | mast_dlcfs | mast_consistency |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in rows:
        sc = scores_of(r)
        board = "yes" if r.get("on_leaderboard") else ""
        when = r.get("created_when", "").replace("T", " ").replace("Z", "")[:19]
        cells = [str(r["id"]), when, board] + [fmt(sc.get(c)) for c in SCORE_COLS]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    (out_dir / "index.md").write_text("\n".join(lines))
    print(f"Updated {out_dir}: {len(rows)} Development, {len(final_rows)} Final submissions.")
    if on_lb:
        print(f"Leaderboard row: {lb_text}")


if __name__ == "__main__":
    pull_and_save()
