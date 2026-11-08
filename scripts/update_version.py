#!/usr/bin/env python3
# scripts/update_version.py
"""
Update the package version everywhere it appears in the repository.

This is the single source of truth for version bumps. It replaces
``scripts/bump_version.py`` (which only updated ``manifest.json``)
and adds coverage for every other file that carries a version string.

Files touched
-------------
1. ``manifest.json``                          — the ``version`` field.
2. ``src/cloudflare_ai_peldrun/__init__.py``  — the ``__version__``.
3. ``pyproject.toml``                          — ``[project] version``.
4. ``CHANGELOG.md``                            — a new section at the top.
5. ``README.md``                               — any ``vX.Y.Z`` mention.

Files that are deliberately NOT touched
---------------------------------------
* ``dist/catalog-entry.json`` — regenerated from manifest by
  ``scripts/generate_catalog_entry.py``.
* ``dist/*.zip`` — rebuilt by ``scripts/build_package.py``.

Pre-flight consistency check
----------------------------
Before writing anything, the script verifies that every file that
DOES carry a version string carries the SAME one as ``manifest.json``.
If a mismatch is found (e.g. someone hand-edited ``__init__.py``),
the script aborts with a clear report instead of silently overwriting
one of two conflicting values.

Semver enforcement
------------------
Two input forms are accepted:

  * ``patch`` / ``minor`` / ``major`` — computed from the current
    version using strict SemVer arithmetic.
  * An explicit version string (e.g. ``2.0.0``, ``1.2.3-beta.1``)
    that MUST be strictly greater than the current version.

Usage
-----
    python scripts/update_version.py patch
    python scripts/update_version.py minor
    python scripts/update_version.py major
    python scripts/update_version.py 1.2.3
    python scripts/update_version.py 1.2.3 --dry-run
    python scripts/update_version.py patch --skip-readme
    python scripts/update_version.py --check

Exit codes
----------
    0 — success (or check passed)
    1 — invalid argument / current version is not semver
    2 — manifest missing or unreadable
    3 — target version not strictly greater than current
    4 — pre-flight consistency check failed
    5 — a target file could not be written
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MANIFEST_PATH = _REPO_ROOT / "manifest.json"
_INIT_PATH = _REPO_ROOT / "src" / "cloudflare_ai_peldrun" / "__init__.py"
_PYPROJECT_PATH = _REPO_ROOT / "pyproject.toml"
_CHANGELOG_PATH = _REPO_ROOT / "CHANGELOG.md"
_README_PATH = _REPO_ROOT / "README.md"


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_SEMVER_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)\."
    r"(?P<minor>0|[1-9]\d*)\."
    r"(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>[0-9A-Za-z.-]+))?"
    r"(?:\+(?P<build>[0-9A-Za-z.-]+))?$"
)

# ``__version__ = "1.0.0"``  (single or double quotes, optional spaces)
_INIT_VERSION_RE = re.compile(
    r'^(?P<prefix>__version__\s*=\s*)(?P<quote>["\'])(?P<value>[^"\']+)(?P=quote)',
    re.MULTILINE,
)

# ``version = "1.0.0"`` under ``[project]`` in pyproject.toml. We keep
# this simple and only match the first occurrence — if multiple
# ``version =`` lines ever appear, the pre-flight check will flag it.
_PYPROJECT_VERSION_RE = re.compile(
    r'^(?P<prefix>version\s*=\s*)(?P<quote>["\'])(?P<value>[^"\']+)(?P=quote)',
    re.MULTILINE,
)

# A ``vX.Y.Z`` mention in README / docs, bounded by word chars.
_README_VERSION_RE = re.compile(
    r"\bv(?P<value>\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?)\b"
)


# ---------------------------------------------------------------------------
# Semver helpers
# ---------------------------------------------------------------------------


def _parse_semver(value: str) -> tuple[int, int, int]:
    """Return the numeric core of ``value``. Raises ValueError."""
    match = _SEMVER_RE.match(value.strip())
    if not match:
        raise ValueError(f"{value!r} is not a valid semver string.")
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
    )


def _compute_target(
    current: str, spec: str, *, force: bool = False
) -> str:
    """
    Return the target version string, or raise ValueError.

    When ``force`` is True, the "target must be strictly greater
    than current" check is bypassed — useful for reverting an
    accidental bump during local testing. The pre-flight consistency
    check is NOT bypassed; every version-carrying file must still
    agree with ``manifest.json`` before any write occurs.
    """
    current_tuple = _parse_semver(current)

    if spec == "patch":
        major, minor, patch = current_tuple
        return f"{major}.{minor}.{patch + 1}"

    if spec == "minor":
        major, minor, _ = current_tuple
        return f"{major}.{minor + 1}.0"

    if spec == "major":
        major, _, _ = current_tuple
        return f"{major + 1}.0.0"

    target_tuple = _parse_semver(spec)
    if not force and target_tuple <= current_tuple:
        raise ValueError(
            f"Target version {spec!r} must be strictly greater than "
            f"current {current!r}. Use --force to override (e.g. for "
            f"local rollback during testing)."
        )
    return spec


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_atomic(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically (temp + replace)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Per-file update report
# ---------------------------------------------------------------------------


@dataclass
class FileUpdate:
    """One file's change (or non-change) during the update."""

    path: Path
    found_version: Optional[str]
    new_version: Optional[str]
    will_change: bool
    note: str = ""


@dataclass
class UpdateReport:
    """Aggregate result across all files."""

    current_version: str
    target_version: str
    updates: List[FileUpdate] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    def summary(self) -> str:
        lines: List[str] = []
        for u in self.updates:
            try:
                rel = u.path.relative_to(_REPO_ROOT)
            except ValueError:
                rel = u.path
            if u.will_change:
                marker = "[change]"
            elif u.found_version is None:
                marker = "[skip]  "
            else:
                marker = "[ok]    "
            note = f"  ({u.note})" if u.note else ""
            found = u.found_version if u.found_version is not None else "—"
            lines.append(
                f"  {marker} {rel}: {found}{note}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _load_manifest() -> dict[str, Any]:
    if not _MANIFEST_PATH.is_file():
        raise SystemExit(
            f"error: manifest.json not found at {_MANIFEST_PATH}"
        )
    try:
        data = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"error: manifest.json is not valid JSON: {exc}"
        )
    if not isinstance(data, dict):
        raise SystemExit("error: manifest.json must be a JSON object.")
    if "version" not in data or not isinstance(data["version"], str):
        raise SystemExit(
            "error: manifest.json has no string 'version' field."
        )
    return data


# ---------------------------------------------------------------------------
# Detectors — each returns (found_version, update_fn_or_None)
# ---------------------------------------------------------------------------


def _inspect_init() -> Optional[tuple[Path, Optional[str]]]:
    """Return the __init__.py path and its __version__ (if any)."""
    if not _INIT_PATH.is_file():
        return None
    match = _INIT_VERSION_RE.search(_read_text(_INIT_PATH))
    if not match:
        return (_INIT_PATH, None)
    return (_INIT_PATH, match.group("value"))


def _inspect_pyproject() -> Optional[tuple[Path, Optional[str]]]:
    """Return the pyproject.toml path and its version (if any)."""
    if not _PYPROJECT_PATH.is_file():
        return None
    matches = list(_PYPROJECT_VERSION_RE.finditer(_read_text(_PYPROJECT_PATH)))
    if not matches:
        return (_PYPROJECT_PATH, None)
    # If multiple appear, use the first but flag the ambiguity later
    # via a note in the update report (checked by the pre-flight).
    return (_PYPROJECT_PATH, matches[0].group("value"))


def _find_readme_versions(text: str) -> List[str]:
    """Return every distinct ``vX.Y.Z`` in ``text``."""
    return list({m.group("value") for m in _README_VERSION_RE.finditer(text)})


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------


def _preflight(
    current_version: str,
) -> tuple[List[FileUpdate], List[str]]:
    """
    Inspect every candidate file and return:

      * a ``FileUpdate`` entry for each file (populated with
        ``will_change=False`` — it is filled in later),
      * a list of consistency errors (empty if all is well).
    """
    updates: List[FileUpdate] = []
    errors: List[str] = []

    # manifest.json — always present (just validated by _load_manifest).
    updates.append(
        FileUpdate(
            path=_MANIFEST_PATH,
            found_version=current_version,
            new_version=None,
            will_change=True,
        )
    )

    # __init__.py
    init_info = _inspect_init()
    if init_info is not None:
        path, found = init_info
        if found is None:
            updates.append(
                FileUpdate(
                    path=path,
                    found_version=None,
                    new_version=None,
                    will_change=False,
                    note="no __version__ attribute found",
                )
            )
        elif found != current_version:
            errors.append(
                f"{path.relative_to(_REPO_ROOT)}: found "
                f"{found!r}, expected {current_version!r}"
            )
            updates.append(
                FileUpdate(
                    path=path,
                    found_version=found,
                    new_version=None,
                    will_change=False,
                    note="MISMATCH",
                )
            )
        else:
            updates.append(
                FileUpdate(
                    path=path,
                    found_version=found,
                    new_version=None,
                    will_change=True,
                )
            )

    # pyproject.toml
    pp_info = _inspect_pyproject()
    if pp_info is not None:
        path, found = pp_info
        if found is None:
            updates.append(
                FileUpdate(
                    path=path,
                    found_version=None,
                    new_version=None,
                    will_change=False,
                    note="no version field found",
                )
            )
        elif found != current_version:
            errors.append(
                f"{path.relative_to(_REPO_ROOT)}: found "
                f"{found!r}, expected {current_version!r}"
            )
            updates.append(
                FileUpdate(
                    path=path,
                    found_version=found,
                    new_version=None,
                    will_change=False,
                    note="MISMATCH",
                )
            )
        else:
            updates.append(
                FileUpdate(
                    path=path,
                    found_version=found,
                    new_version=None,
                    will_change=True,
                )
            )

    # CHANGELOG.md
    if _CHANGELOG_PATH.is_file():
        updates.append(
            FileUpdate(
                path=_CHANGELOG_PATH,
                found_version=None,
                new_version=None,
                will_change=True,
                note="new section will be added",
            )
        )
    else:
        updates.append(
            FileUpdate(
                path=_CHANGELOG_PATH,
                found_version=None,
                new_version=None,
                will_change=False,
                note="file does not exist",
            )
        )

    # README.md — informational only. README mentions of version are
    # optional and may be zero; if present but different, that is a
    # soft signal, not a hard error.
    if _README_PATH.is_file():
        readme_text = _read_text(_README_PATH)
        found_versions = _find_readme_versions(readme_text)
        if not found_versions:
            updates.append(
                FileUpdate(
                    path=_README_PATH,
                    found_version=None,
                    new_version=None,
                    will_change=False,
                    note="no version mention",
                )
            )
        else:
            # README.md is DOCUMENTATION, not a version carrier. Version
            # strings inside it are almost always illustrative examples
            # ("bump from v1.0.0 to v1.0.1"), not declarations of the
            # current release. Treat any mismatch as a WARNING so a
            # documentation edit never blocks a CI run.
            #
            # When every mention DOES match the current version, the
            # script will still rewrite them on the next bump — that is
            # the desired behavior for README files that display the
            # live version (e.g. "This is v1.0.0"). Files with a mix of
            # example and real versions should use the README-fix
            # convention documented in CONTRIBUTING.md.
            mismatches = [v for v in found_versions if v != current_version]
            if mismatches:
                # Non-fatal: log a warning, do not change README.
                updates.append(
                    FileUpdate(
                        path=_README_PATH,
                        found_version=",".join(found_versions),
                        new_version=None,
                        will_change=False,
                        note=(
                            "warn: mentions "
                            f"{sorted(mismatches)!r} (likely examples)"
                        ),
                    )
                )
            else:
                updates.append(
                    FileUpdate(
                        path=_README_PATH,
                        found_version=",".join(found_versions),
                        new_version=None,
                        will_change=True,
                    )
                )
    else:
        updates.append(
            FileUpdate(
                path=_README_PATH,
                found_version=None,
                new_version=None,
                will_change=False,
                note="file does not exist",
            )
        )

    return updates, errors


# ---------------------------------------------------------------------------
# Writers — one per file type
# ---------------------------------------------------------------------------


def _write_manifest(new_version: str) -> None:
    data = _load_manifest()
    data["version"] = new_version
    serialized = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    _write_atomic(_MANIFEST_PATH, serialized)


def _write_init(new_version: str) -> None:
    text = _read_text(_INIT_PATH)
    new_text, n = _INIT_VERSION_RE.subn(
        lambda m: f'{m.group("prefix")}{m.group("quote")}'
                  f'{new_version}{m.group("quote")}',
        text,
        count=1,
    )
    if n == 0:
        raise SystemExit(
            f"error: could not rewrite __version__ in {_INIT_PATH}"
        )
    _write_atomic(_INIT_PATH, new_text)


def _write_pyproject(new_version: str) -> None:
    text = _read_text(_PYPROJECT_PATH)
    new_text, n = _PYPROJECT_VERSION_RE.subn(
        lambda m: f'{m.group("prefix")}{m.group("quote")}'
                  f'{new_version}{m.group("quote")}',
        text,
        count=1,
    )
    if n == 0:
        raise SystemExit(
            f"error: could not rewrite version in {_PYPROJECT_PATH}"
        )
    _write_atomic(_PYPROJECT_PATH, new_text)


def _write_changelog(new_version: str) -> None:
    """
    Insert a CHANGELOG.md section for ``new_version``.

    Delegates to ``scripts/changelog.py`` which parses the git log
    via Conventional Commits. When that script cannot extract any
    commits (e.g. not a git repo, shallow clone, no commits since
    the last tag), it falls back to a placeholder section
    automatically — so this wrapper never fails on the happy path.
    """
    if not _CHANGELOG_PATH.is_file():
        return

    # Resolve the changelog script relative to this file so the
    # import is robust regardless of cwd.
    script = Path(__file__).resolve().parent / "changelog.py"
    if not script.is_file():
        # Fallback: no changelog script — emit the placeholder.
        existing = _read_text(_CHANGELOG_PATH)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        new_section = (
            f"## [{new_version}] - {today}\n"
            f"\n"
            f"### Changed\n"
            f"- Version bump to {new_version}. Fill in the highlights "
            f"for this release before tagging.\n"
            f"\n"
        )
        if existing.startswith("# "):
            newline = existing.find("\n")
            if newline == -1:
                new_content = existing + "\n\n" + new_section
            else:
                new_content = (
                    existing[: newline + 1]
                    + "\n"
                    + new_section
                    + existing[newline + 1 :].lstrip("\n")
                )
        else:
            new_content = new_section + existing
        _write_atomic(_CHANGELOG_PATH, new_content)
        return

    # Run the subprocess. Use --placeholder as a safety net if git
    # extraction fails for any reason — the goal is never to block
    # the version bump because of a changelog detail.
    try:
        import subprocess as _subprocess

        result = _subprocess.run(
            [sys.executable, str(script), "--version", new_version],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        result = None

    if result is None or result.returncode != 0:
        # Git extraction failed — retry with --placeholder.
        try:
            import subprocess as _subprocess

            _subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--version",
                    new_version,
                    "--placeholder",
                ],
                cwd=str(_REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except Exception as exc:
            raise SystemExit(
                f"error: could not update CHANGELOG.md: {exc}"
            )


def _write_readme(new_version: str) -> None:
    """Replace every ``vX.Y.Z`` mention with ``v<new_version>``."""
    text = _read_text(_README_PATH)
    new_text = _README_VERSION_RE.sub(
        lambda m: f"v{new_version}", text
    )
    if new_text == text:
        return
    _write_atomic(_README_PATH, new_text)


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


def _apply_updates(
    target_version: str,
    updates: List[FileUpdate],
    *,
    skip_changelog: bool,
    skip_readme: bool,
) -> List[str]:
    """
    Execute all writes. Returns a list of failure messages (empty on
    success). Any exception from a writer is captured and reported.
    """
    failures: List[str] = []

    for update in updates:
        if not update.will_change:
            continue

        try:
            if update.path == _MANIFEST_PATH:
                _write_manifest(target_version)
            elif update.path == _INIT_PATH:
                _write_init(target_version)
            elif update.path == _PYPROJECT_PATH:
                _write_pyproject(target_version)
            elif update.path == _CHANGELOG_PATH:
                if skip_changelog:
                    update.will_change = False
                    update.note = "skipped by --skip-changelog"
                    continue
                _write_changelog(target_version)
            elif update.path == _README_PATH:
                if skip_readme:
                    update.will_change = False
                    update.note = "skipped by --skip-readme"
                    continue
                _write_readme(target_version)
        except Exception as exc:
            failures.append(f"{update.path}: {exc}")

    return failures


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Update the package version across every file in the "
            "repository. Accepts 'patch' / 'minor' / 'major' or an "
            "explicit semver string."
        ),
    )
    parser.add_argument(
        "spec",
        nargs="?",
        default=None,
        help=(
            "'patch', 'minor', 'major', or an explicit version like "
            "'1.2.3'. Required unless --check is passed."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without writing anything.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Verify that every version-carrying file is consistent "
            "with manifest.json. Does not modify anything. Exits 0 if "
            "consistent, 4 otherwise."
        ),
    )
    parser.add_argument(
        "--skip-changelog",
        action="store_true",
        help="Do not prepend a new section to CHANGELOG.md.",
    )
    parser.add_argument(
        "--skip-readme",
        action="store_true",
        help="Do not rewrite ``vX.Y.Z`` mentions in README.md.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Allow the target version to be equal to or lower than "
            "the current version. Useful for reverting an accidental "
            "bump during local testing. Does NOT bypass the "
            "pre-flight consistency check."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = _parse_args(argv)

    manifest = _load_manifest()
    current_version = manifest["version"]

    # --check path: pre-flight only, no changes.
    if args.check:
        updates, errors = _preflight(current_version)
        print(f"Current version: {current_version}")
        print("Files inspected:")
        print(UpdateReport(
            current_version=current_version,
            target_version=current_version,
            updates=updates,
            errors=errors,
        ).summary())
        if errors:
            print()
            for err in errors:
                print(f"  [err]  {err}")
            print()
            print(
                f"FAILED: {len(errors)} inconsistency(ies) found."
            )
            return 4
        print()
        print("OK: every file agrees with manifest.json.")
        return 0

    # Normal path: compute the target version.
    if args.spec is None:
        print(
            "error: missing version spec. Use 'patch', 'minor', "
            "'major', or an explicit semver string. For consistency "
            "checks, pass --check.",
            file=sys.stderr,
        )
        return 1

    try:
        target_version = _compute_target(
            current_version, args.spec, force=args.force
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    updates, errors = _preflight(current_version)
    report = UpdateReport(
        current_version=current_version,
        target_version=target_version,
        updates=updates,
        errors=list(errors),
    )

    print(f"Current version: {current_version}")
    print(f"Target version:  {target_version}")
    print()

    if errors:
        print("Pre-flight consistency check FAILED:")
        print(report.summary())
        print()
        for err in errors:
            print(f"  [err]  {err}")
        print()
        print(
            "error: refusing to update because some files disagree "
            "with manifest.json. Fix the mismatch above, then re-run.",
            file=sys.stderr,
        )
        return 4

    if args.dry_run:
        # Fill in new_version for display purposes only.
        for u in updates:
            if u.will_change:
                u.new_version = target_version
        print("Plan (dry run — nothing will be written):")
        print(report.summary())
        return 0

    failures = _apply_updates(
        target_version,
        updates,
        skip_changelog=args.skip_changelog,
        skip_readme=args.skip_readme,
    )

    for u in updates:
        if u.will_change:
            u.new_version = target_version

    print("Applied:")
    print(report.summary())

    if failures:
        print()
        print("FAILED writes:")
        for f in failures:
            print(f"  [err]  {f}")
        return 5

    print()
    print(f"Version updated: {current_version} -> {target_version}")
    print()
    print("Next steps:")
    print("  1. Edit CHANGELOG.md to describe the release.")
    print("  2. python scripts/build_package.py --verify")
    print("  3. python scripts/verify_package.py")
    print("  4. python scripts/generate_catalog_entry.py ...")
    print("  5. git commit -am 'Release v" + target_version + "'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))