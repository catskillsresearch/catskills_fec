"""Single-writer lock for GPU training runs.

Two charcoal runs on one RTX 3060 do not fail loudly -- they interleave, each takes two to three
times as long, and every wall-clock number in the resulting score json is wrong. Worse, both write
``artifacts/CRC_FEQ_charcoal_<tag>.pt`` and ``results/charcoal/<tag>.json``, so if the tags collide
the second finisher silently overwrites the first.

A lock file under ``lab/tmp/locks`` holds the owning pid. Liveness is checked through ``/proc``
rather than by trusting the file, so a lock left behind by a killed run is reclaimed automatically.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from catskills.lab_paths import lab_root


def locks_dir() -> Path:
    d = lab_root() / "tmp" / "locks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").is_dir()


def _cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return " ".join(p for p in raw.decode("utf-8", "replace").split("\0") if p)


def read_locks() -> list[dict]:
    """Every live lock, stale entries pruned. Safe to call from the dashboard."""
    out = []
    for p in sorted(locks_dir().glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        pid = int(d.get("pid", -1))
        if pid > 0 and _alive(pid):
            d["age_s"] = time.time() - float(d.get("started", time.time()))
            d["cmdline"] = _cmdline(pid)
            out.append(d)
        else:
            p.unlink(missing_ok=True)
    return out


class ConcurrentRunError(RuntimeError):
    pass


@dataclass
class RunLock:
    """Context manager. ``name`` is the resource, usually the model id."""

    name: str
    tag: str = ""
    allow_concurrent: bool = False
    path: Path | None = None

    def __enter__(self) -> "RunLock":
        self.path = locks_dir() / f"{self.name}.json"
        held = [d for d in read_locks() if d.get("name") == self.name]
        if held and not self.allow_concurrent:
            other = held[0]
            raise ConcurrentRunError(
                f"{self.name} is already running as pid {other.get('pid')} "
                f"(tag {other.get('tag') or '?'}, {other.get('age_s', 0) / 60:.1f} min in).\n"
                f"  {other.get('cmdline', '')[:300]}\n"
                "Two runs share one GPU and can overwrite each other's checkpoint. "
                "Wait for it, kill it, or pass --allow-concurrent if you really mean it."
            )
        self.path.write_text(
            json.dumps(
                {"name": self.name, "tag": self.tag, "pid": os.getpid(), "started": time.time()},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return self

    def __exit__(self, *exc) -> None:
        # Only clear our own lock: with --allow-concurrent the file belongs to whoever wrote last.
        try:
            if self.path and json.loads(self.path.read_text())["pid"] == os.getpid():
                self.path.unlink(missing_ok=True)
        except Exception:
            pass
