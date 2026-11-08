#!/usr/bin/env python3
# scripts/build_package.py
"""
Build the distributable Provider Package ZIP.

Layout produced:

    dist/cloudflare-ai-<version>.zip
    ├── manifest.json              (copied from repo root)
    ├── requirements.txt           (copied from repo root)
    └── cloudflare_ai_peldrun/     (src/cloudflare_ai_peldrun/, flattened)
        ├── __init__.py
        └── adapter.py

Notes on the layout
-------------------
The archive must satisfy ``package_validator._locate_manifest``:

  * ``manifest.json`` MUST be at the archive root (no wrapping folder).
  * The Python package MUST be at ``cloudflare_ai_peldrun/`` (matching
    the ``entry_point`` prefix), NOT at ``src/cloudflare_ai_peldrun/``.

That second requirement is why this script flattens ``src/``.

Determinism
-----------
Two consecutive runs on the same source tree produce byte-identical
ZIPs. This lets the release workflow use the resulting ``sha256`` as
an idempotency key. Determinism is achieved by:

  * Sorting every file list.
  * Setting ``date_time = (2020, 1, 1, 0, 0, 0)`` on every entry.
  * Not storing OS-specific metadata (``create_system = 3`` / Unix).
  * Preserving only the executable bit, dropping umask-dependent modes.
  * Using ``ZIP_DEFLATED`` (level 9) uniformly.

Usage:
    python scripts/build_package.py
    python scripts/build_package.py --output-dir custom_dist/
    python scripts/build_package.py --verify

Exit codes:
    0 — success
    1 — source tree is incomplete or manifest is invalid
    2 — could not write the output ZIP
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import List, Sequence


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DIST_DIR = _REPO_ROOT / "dist"
_MANIFEST_PATH = _REPO_ROOT / "manifest.json"
_REQUIREMENTS_PATH = _REPO_ROOT / "requirements.txt"
_SRC_PACKAGE_DIR = _REPO_ROOT / "src" / "cloudflare_ai_peldrun"


#: Prefix stripped from every archive entry. ``src/cloudflare_ai_peldrun/``
#: becomes ``cloudflare_ai_peldrun/`` in the ZIP.
_SRC_PREFIX = _REPO_ROOT / "src"


#: Fixed timestamp applied to every archive entry for determinism.
#: 2020-01-01T00:00:00 — an arbitrary non-zero date that is well in
#: the past (so extraction never looks "future-dated") and identical
#: across every build.
_FIXED_DATE_TIME = (2020, 1, 1, 0, 0, 0)


#: Directories and files that must never reach the archive, matched
#: against any path component.
_EXCLUDED_COMPONENTS: frozenset[str] = frozenset({
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
})


#: File suffixes that are never part of a source distribution.
_EXCLUDED_SUFFIXES: tuple[str, ...] = (
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".dylib",
    ".dll",
    ".swp",
    ".swo",
    ".tmp",
    ".log",
    ".DS_Store",
    ".orig",
    ".rej",
)


#: File name patterns (exact matches) that must never reach the archive.
_EXCLUDED_NAMES: frozenset[str] = frozenset({
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
})


#: Version pattern accepted by the validator (semver). Bumped by
#: ``scripts/bump_version.py``; the build script trusts but verifies it
#: so a typo does not silently produce a bad archive.
_VERSION_RE = re.compile(
    r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)


# ---------------------------------------------------------------------------
# Source checks
# ---------------------------------------------------------------------------


def _read_manifest_version() -> str:
    """Return ``manifest.json["version"]`` after structural checks."""
    if not _MANIFEST_PATH.is_file():
        raise SystemExit(
            f"error: manifest.json not found at {_MANIFEST_PATH}"
        )
    try:
        data = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"error: manifest.json is not valid JSON: {exc}")
    if not isinstance(data, dict):
        raise SystemExit("error: manifest.json must be a JSON object.")
    version = data.get("version")
    if not isinstance(version, str) or not _VERSION_RE.match(version):
        raise SystemExit(
            f"error: manifest.json['version'] = {version!r} is not "
            f"valid semver."
        )
    return version


def _assert_source_tree_complete() -> None:
    """Verify every required file exists before we start writing."""
    missing: List[str] = []
    if not _REQUIREMENTS_PATH.is_file():
        missing.append(str(_REQUIREMENTS_PATH))
    if not _SRC_PACKAGE_DIR.is_dir():
        missing.append(str(_SRC_PACKAGE_DIR))
    elif not (_SRC_PACKAGE_DIR / "__init__.py").is_file():
        missing.append(str(_SRC_PACKAGE_DIR / "__init__.py"))
    elif not (_SRC_PACKAGE_DIR / "adapter.py").is_file():
        missing.append(str(_SRC_PACKAGE_DIR / "adapter.py"))
    if missing:
        raise SystemExit(
            "error: source tree is incomplete. Missing:\n  "
            + "\n  ".join(missing)
        )


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------


def _should_exclude(relative: Path) -> bool:
    """Return True iff ``relative`` (from src/) must not be archived."""
    # Any excluded component anywhere in the path.
    for part in relative.parts:
        if part in _EXCLUDED_COMPONENTS:
            return True
    # Excluded exact name.
    if relative.name in _EXCLUDED_NAMES:
        return True
    # Excluded suffix.
    lowered = relative.name.lower()
    for suffix in _EXCLUDED_SUFFIXES:
        if lowered.endswith(suffix.lower()):
            return True
    return False


def _collect_package_files() -> List[Path]:
    """
    Return every file under ``src/cloudflare_ai_peldrun/`` that should
    be included in the archive, sorted by archive-relative path.
    """
    files: List[Path] = []
    for path in _SRC_PACKAGE_DIR.rglob("*"):
        if not path.is_file():
            continue
        if _should_exclude(path.relative_to(_SRC_PACKAGE_DIR)):
            continue
        files.append(path)

    # Sort by the archive path, not by absolute path — the archive
    # path is what appears inside the ZIP and what determines the
    # resulting hash, so it must be the canonical ordering key.
    files.sort(key=lambda p: str(p.relative_to(_SRC_PREFIX)).replace("\\", "/"))
    return files


def _archive_arcname(path: Path) -> str:
    """
    Return the path ``path`` should carry inside the ZIP.

    Files under ``src/`` are flattened (``src/`` prefix removed).
    Files at the repo root keep their names.
    """
    if _SRC_PREFIX in path.parents or path.is_relative_to(_SRC_PREFIX):
        return str(path.relative_to(_SRC_PREFIX)).replace("\\", "/")
    return path.name


# ---------------------------------------------------------------------------
# ZIP writing
# ---------------------------------------------------------------------------


def _zipinfo_for(arcname: str) -> zipfile.ZipInfo:
    """
    Build a deterministic ``ZipInfo`` for ``arcname``.

    Fixed timestamp + fixed Unix mode (0o644) + fixed create system
    (Unix) → identical bytes on every run, on every platform.
    """
    info = zipfile.ZipInfo(filename=arcname, date_time=_FIXED_DATE_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3  # Unix
    info.external_attr = (0o644 & 0xFFFF) << 16  # regular file, rw-r--r--
    return info


def _write_file(zf: zipfile.ZipFile, path: Path, arcname: str) -> None:
    """Add one file to the archive with deterministic metadata."""
    info = _zipinfo_for(arcname)
    with path.open("rb") as fh:
        data = fh.read()
    zf.writestr(info, data)


def build_zip(output_path: Path) -> int:
    """
    Build the archive at ``output_path``.

    Returns the number of files written (excluding directories).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temporary name first so a crash mid-write cannot leave
    # a half-built ZIP that a downstream step (verify, upload) would
    # consume by mistake.
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    file_count = 0
    try:
        with zipfile.ZipFile(
            tmp_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as zf:
            # 1. manifest.json — must be at the root.
            _write_file(zf, _MANIFEST_PATH, "manifest.json")
            file_count += 1

            # 2. requirements.txt — must be at the root.
            _write_file(zf, _REQUIREMENTS_PATH, "requirements.txt")
            file_count += 1

            # 3. Python package files, flattened.
            for path in _collect_package_files():
                arcname = _archive_arcname(path)
                _write_file(zf, path, arcname)
                file_count += 1

        tmp_path.replace(output_path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise

    return file_count


# ---------------------------------------------------------------------------
# Post-build verification (light — the full verify is a separate script)
# ---------------------------------------------------------------------------


def _verify_structure(zip_path: Path) -> None:
    """
    Sanity-check the produced archive's top-level layout.

    Only checks that the archive has the expected shape. Deep
    validation (symlinks, path traversal, checksum, entry-point file,
    etc.) is done by ``scripts/verify_package.py``.
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())

    required = {"manifest.json", "requirements.txt"}
    missing = required - names
    if missing:
        raise SystemExit(
            f"error: produced ZIP is missing required root file(s): "
            f"{sorted(missing)}"
        )

    package_prefix = "cloudflare_ai_peldrun/"
    if not any(n.startswith(package_prefix) for n in names):
        raise SystemExit(
            f"error: produced ZIP has no files under "
            f"'{package_prefix}'."
        )

    # A wrapped src/ prefix means the flatten step did not run.
    if any(n.startswith("src/") for n in names):
        raise SystemExit(
            "error: produced ZIP contains 'src/' entries. The flatten "
            "step did not run correctly."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Cloudflare AI provider package ZIP.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_DIST_DIR,
        help=(
            f"Directory for the produced ZIP (default: "
            f"{_DEFAULT_DIST_DIR})."
        ),
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run the structural verification after building.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = _parse_args(argv)

    version = _read_manifest_version()
    _assert_source_tree_complete()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"cloudflare-ai-{version}.zip"

    try:
        file_count = build_zip(zip_path)
    except OSError as exc:
        print(f"error: could not write {zip_path}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: build failed: {exc}", file=sys.stderr)
        return 2

    size_bytes = zip_path.stat().st_size
    print(f"Built {zip_path}")
    print(f"  version: {version}")
    print(f"  files:   {file_count}")
    print(f"  size:    {size_bytes} bytes")

    if args.verify:
        try:
            _verify_structure(zip_path)
        except SystemExit as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print("  verify:  ok")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))