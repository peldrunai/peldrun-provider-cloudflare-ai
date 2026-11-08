#!/usr/bin/env python3
# scripts/bump_version.py
"""
Bump the ``version`` field in ``manifest.json`` following Semantic Versioning.

Usage:
    python scripts/bump_version.py patch    # 1.0.0 -> 1.0.1
    python scripts/bump_version.py minor    # 1.0.1 -> 1.1.0
    python scripts/bump_version.py major    # 1.1.0 -> 2.0.0
    python scripts/bump_version.py 1.2.3    # explicit set (must be > current)

The script only touches ``manifest.json``. It does not create a git commit,
tag, or modify any other file — those steps belong to the release workflow.

Exit codes:
    0 — success
    1 — invalid arguments / current version is not semver
    2 — manifest missing or unreadable
    3 — target version is not strictly greater than current
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


_SEMVER_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)\."
    r"(?P<minor>0|[1-9]\d*)\."
    r"(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>[0-9A-Za-z.-]+))?"
    r"(?:\+(?P<build>[0-9A-Za-z.-]+))?$"
)

_MANIFEST_PATH = Path(__file__).resolve().parent.parent / "manifest.json"


def _parse_semver(value: str) -> tuple[int, int, int]:
    """Parse the numeric core of a semver string. Raises ValueError."""
    match = _SEMVER_RE.match(value.strip())
    if not match:
        raise ValueError(f"{value!r} is not a valid semver string.")
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
    )


def _format_semver(major: int, minor: int, patch: int) -> str:
    return f"{major}.{minor}.{patch}"


def _compute_bump(current: str, spec: str) -> str:
    """
    Compute the next version string from ``current`` and a bump specification.

    ``spec`` is one of: ``patch``, ``minor``, ``major``, or an explicit
    semver string. Explicit versions must be strictly greater than
    ``current`` on the numeric tuple — we do not compare pre-release tags.
    """
    current_tuple = _parse_semver(current)

    if spec == "patch":
        major, minor, patch = current_tuple
        return _format_semver(major, minor, patch + 1)

    if spec == "minor":
        major, minor, _ = current_tuple
        return _format_semver(major, minor + 1, 0)

    if spec == "major":
        major, _, _ = current_tuple
        return _format_semver(major + 1, 0, 0)

    # Explicit version
    target_tuple = _parse_semver(spec)
    if target_tuple <= current_tuple:
        raise ValueError(
            f"Target version {spec!r} must be strictly greater than "
            f"current {current!r}."
        )
    return spec


def _write_manifest(path: Path, data: dict) -> None:
    """
    Write ``manifest.json`` atomically (tmp file + os.replace).

    Preserves existing field order and 2-space indentation to match
    the rest of the repository.
    """
    serialized = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    tmp_path = path.with_suffix(".json.tmp")
    try:
        tmp_path.write_text(serialized, encoding="utf-8")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(
            "Usage: bump_version.py <patch|minor|major|X.Y.Z>",
            file=sys.stderr,
        )
        return 1

    spec = argv[1].strip()

    if not _MANIFEST_PATH.is_file():
        print(
            f"error: manifest.json not found at {_MANIFEST_PATH}",
            file=sys.stderr,
        )
        return 2

    try:
        data = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"error: could not read manifest.json: {exc}", file=sys.stderr)
        return 2

    current = data.get("version")
    if not isinstance(current, str) or not current:
        print(
            "error: manifest.json is missing a valid 'version' field.",
            file=sys.stderr,
        )
        return 1

    try:
        new_version = _compute_bump(current, spec)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        # Distinguish "target not greater" (3) from "invalid spec" (1).
        if "must be strictly greater" in str(exc):
            return 3
        return 1

    data["version"] = new_version

    try:
        _write_manifest(_MANIFEST_PATH, data)
    except OSError as exc:
        print(f"error: could not write manifest.json: {exc}", file=sys.stderr)
        return 2

    print(f"Bumped version: {current} -> {new_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))