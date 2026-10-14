"""Frozen stem-disjoint train/dev/holdout assignment for the final campaign."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .data import shot_paths

SALT = "final-honest-v1"
ROLES = ("train", "dev", "holdout")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assignment_bucket(stem: str, *, salt: str = SALT) -> int:
    digest = hashlib.sha256(f"{salt}:{stem}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 100


def role_for_stem(stem: str, *, salt: str = SALT) -> str:
    bucket = assignment_bucket(stem, salt=salt)
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "dev"
    return "holdout"


def assign_paths(paths: list[Path], *, salt: str = SALT) -> dict[str, list[Path]]:
    assigned = {role: [] for role in ROLES}
    for path in paths:
        assigned[role_for_stem(path.stem, salt=salt)].append(path)
    return assigned


def build_manifest(
    *,
    config: str = "diii_d_train",
    salt: str = SALT,
    command: str = "",
) -> dict:
    paths = shot_paths(config)
    assigned = assign_paths(paths, salt=salt)
    records = []
    for role in ROLES:
        for path in assigned[role]:
            records.append(
                {
                    "stem": path.stem,
                    "path": str(path),
                    "role": role,
                    "sha256": file_sha256(path),
                }
            )
    stems = {role: [path.stem for path in assigned[role]] for role in ROLES}
    return {
        "salt": salt,
        "config": config,
        "command": command,
        "n_total": len(paths),
        "counts": {role: len(stems[role]) for role in ROLES},
        "stems": stems,
        "records": records,
    }


def load_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def stems_for_role(manifest: dict, role: str) -> list[str]:
    if role not in ROLES:
        raise KeyError(f"unknown role {role!r}")
    return list(manifest["stems"][role])


def paths_for_role(manifest: dict, role: str) -> list[Path]:
    lookup = {record["stem"]: Path(record["path"]) for record in manifest["records"]}
    return [lookup[stem] for stem in stems_for_role(manifest, role)]


def shot_stem(path: Path) -> str:
    """Return the parquet stem, stripping optional ``_w0000`` window suffixes."""
    name = Path(path).stem
    head, sep, tail = name.rpartition("_w")
    if sep and tail.isdigit():
        return head
    return name


def filter_files(files: list[Path], manifest: dict, role: str) -> list[Path]:
    allowed = set(stems_for_role(manifest, role))
    return [path for path in files if shot_stem(path) in allowed]


def audit_manifest(manifest: dict) -> list[str]:
    errors = []
    stems = {role: list(manifest["stems"][role]) for role in ROLES}
    seen = []
    for role, values in stems.items():
        if len(values) != len(set(values)):
            errors.append(f"duplicate stems in {role}")
        if values != sorted(values) and values != [record["stem"] for record in manifest["records"] if record["role"] == role]:
            errors.append(f"unstable {role} order")
        seen.extend(values)
    if len(seen) != len(set(seen)):
        errors.append("overlapping roles")
    expected = {record["stem"] for record in manifest["records"]}
    if set(seen) != expected:
        errors.append("manifest records do not match role lists")
    if sum(manifest["counts"].values()) != manifest["n_total"]:
        errors.append("counts do not sum to n_total")
    for role in ROLES:
        rebuilt = [stem for stem in stems[role] if role_for_stem(stem, salt=manifest["salt"]) != role]
        if rebuilt:
            errors.append(f"salt mismatch for {role}: {rebuilt[:3]}")
    return errors
