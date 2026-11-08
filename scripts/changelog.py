#!/usr/bin/env python3
# scripts/changelog.py
"""
Generate a CHANGELOG.md section from git commits.

Parses Conventional Commits between two refs and emits a Markdown
section compatible with Keep a Changelog:

    feat(scope): message   ->  ### Added
    fix(scope): message    ->  ### Fixed
    perf: message          ->  ### Performance
    refactor: message      ->  ### Changed
    docs: message          ->  ### Documentation
    feat!: message         ->  ### Changed  (with "BREAKING" marker)
    test/style/chore/ci    ->  skipped

The generated section is inserted at the top of CHANGELOG.md,
directly below the "# Changelog" heading and above any previous
version section.

Design constraints
------------------
* **Pure stdlib + git CLI.** No `GitPython`, no `git-cliff`, no
  third-party dependency. Runs in a bare CI container.
* **Offline-safe.** All work is local; nothing is fetched.
* **Graceful degradation.** When git is unavailable, when the
  repository is a shallow clone, or when there are no commits
  between the refs, the script falls back to a placeholder section
  that instructs the developer to fill in the highlights manually.
  The fallback is identical to the previous behavior of
  ``update_version.py``.

Usage
-----
    # Auto-detect the previous tag and emit a section for the
    # current version in manifest.json:
    python scripts/changelog.py

    # Explicit refs:
    python scripts/changelog.py --version 1.1.0 --from v1.0.0

    # Preview the section without writing to CHANGELOG.md:
    python scripts/changelog.py --version 1.1.0 --from v1.0.0 --dry-run

    # Skip the git extraction and use a placeholder instead:
    python scripts/changelog.py --version 1.1.0 --from v1.0.0 --placeholder

Exit codes
----------
    0 — section generated and inserted (or preview shown in --dry-run)
    1 — git unavailable or HEAD is not a git repository, and
        --placeholder was not passed
    2 — CHANGELOG.md could not be read/written
    3 — no commits between the refs and --placeholder was not passed
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MANIFEST_PATH = _REPO_ROOT / "manifest.json"
_CHANGELOG_PATH = _REPO_ROOT / "CHANGELOG.md"


# ---------------------------------------------------------------------------
# Conventional Commits parsing
# ---------------------------------------------------------------------------

# ``feat(scope)!: subject`` / ``fix: subject`` / ``docs(readme): ...``
_CONVENTIONAL_RE = re.compile(
    r"^(?P<type>[a-z]+)"
    r"(?:\((?P<scope>[^)]+)\))?"
    r"(?P<breaking>!)?"
    r":\s*(?P<subject>.+)$"
)

#: Commit types that add noise without adding signal to the CHANGELOG.
_SKIP_TYPES = frozenset({"chore", "test", "style", "ci"})

#: Conventional type -> Keep a Changelog category.
_TYPE_TO_CATEGORY: Dict[str, str] = {
    "feat": "Added",
    "fix": "Fixed",
    "perf": "Performance",
    "refactor": "Changed",
    "docs": "Documentation",
    "build": "Changed",
    "revert": "Removed",
}

#: Emission order — mirrors Keep a Changelog's recommended order.
_CATEGORY_ORDER: Tuple[str, ...] = (
    "Added",
    "Changed",
    "Deprecated",
    "Removed",
    "Fixed",
    "Security",
    "Performance",
    "Documentation",
)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _run_git(args: Sequence[str], *, check: bool = True) -> str:
    """
    Run a git command and return stdout.

    Raises ``RuntimeError`` on non-zero exit when ``check=True``.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "git executable not found in PATH."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"git command timed out: git {' '.join(args)}"
        ) from exc

    if check and result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed (exit "
            f"{result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout


def _is_git_repo() -> bool:
    """Return True if _REPO_ROOT is inside a git work tree."""
    try:
        output = _run_git(["rev-parse", "--is-inside-work-tree"], check=False)
    except RuntimeError:
        return False
    return output.strip().lower() == "true"


def _has_shallow_clone() -> bool:
    """
    Return True if the repository is a shallow clone.

    Shallow clones lack the full history — extraction may miss commits
    that predate the clone point. Not fatal, but worth a warning.
    """
    try:
        output = _run_git(
            ["rev-parse", "--is-shallow-repository"], check=False
        )
    except RuntimeError:
        return False
    return output.strip().lower() == "true"


def _latest_tag() -> Optional[str]:
    """
    Return the most recent reachable tag, or ``None`` when there is
    none (first release, or a branch without tags).
    """
    try:
        output = _run_git(
            ["describe", "--tags", "--abbrev=0"], check=False
        )
    except RuntimeError:
        return None
    tag = output.strip()
    return tag if tag else None


# ---------------------------------------------------------------------------
# Commit record
# ---------------------------------------------------------------------------


@dataclass
class Commit:
    """One git commit, normalized for CHANGELOG purposes."""

    sha: str
    subject: str
    category: str  # e.g. "Added", "Fixed", ...
    message: str  # the human-readable part after the type prefix
    breaking: bool = False
    scope: Optional[str] = None


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _fetch_commits(since_ref: Optional[str]) -> List[Tuple[str, str]]:
    """
    Return ``[(sha, subject)]`` for every commit since ``since_ref``.

    When ``since_ref`` is None, return every commit reachable from
    HEAD (useful for the very first release).
    """
    # NUL delimiter between fields, newline between commits. Using a
    # non-printable separator avoids breakage when commit subjects
    # contain tabs, pipes, or newlines (git guarantees no NUL).
    fmt = "%H\x1f%s"
    args = ["log", f"--pretty=format:{fmt}"]
    if since_ref:
        args.append(f"{since_ref}..HEAD")
    try:
        raw = _run_git(args, check=False)
    except RuntimeError:
        return []

    commits: List[Tuple[str, str]] = []
    for line in raw.splitlines():
        if "\x1f" not in line:
            continue
        sha, _, subject = line.partition("\x1f")
        if sha and subject:
            commits.append((sha.strip(), subject.strip()))
    return commits


def _classify(subject: str) -> Optional[Commit]:
    """
    Parse a subject line and return a ``Commit``, or ``None`` when
    the commit should be skipped (not a Conventional Commit, or a
    type we do not want in the CHANGELOG).

    Non-Conventional-Commit subjects are treated as ``Changed`` —
    keeping them is more useful than silently dropping work.
    """
    match = _CONVENTIONAL_RE.match(subject)
    if not match:
        # Non-conforming commit: keep it, treat as generic "Changed".
        return Commit(
            sha="",  # filled by caller
            subject=subject,
            category="Changed",
            message=subject,
        )

    commit_type = match.group("type").lower()
    if commit_type in _SKIP_TYPES:
        return None

    category = _TYPE_TO_CATEGORY.get(commit_type, "Changed")
    return Commit(
        sha="",
        subject=subject,
        category=category,
        message=match.group("subject").strip(),
        breaking=bool(match.group("breaking")),
        scope=match.group("scope"),
    )


def _collect_commits(since_ref: Optional[str]) -> List[Commit]:
    """Fetch, classify, and return the final list of commits."""
    raw = _fetch_commits(since_ref)
    result: List[Commit] = []
    seen: set[str] = set()
    for sha, subject in raw:
        if sha in seen:
            continue
        seen.add(sha)

        parsed = _classify(subject)
        if parsed is None:
            continue
        parsed.sha = sha[:7]
        result.append(parsed)
    return result


# ---------------------------------------------------------------------------
# Section rendering
# ---------------------------------------------------------------------------


def _render_section(
    version: str,
    commits: List[Commit],
    *,
    date: Optional[str] = None,
) -> str:
    """
    Render a Markdown CHANGELOG section for ``version``.

    Breaking changes are pulled to the top as a dedicated
    "### Breaking changes" block, even if they were ``feat`` or
    ``fix``. This is the most reliable signal for downstream
    consumers when they scan release notes.
    """
    if not commits:
        return _render_placeholder(version, date=date)

    today = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Group by category.
    grouped: Dict[str, List[Commit]] = {}
    breaking: List[Commit] = []

    for commit in commits:
        if commit.breaking:
            breaking.append(commit)
        grouped.setdefault(commit.category, []).append(commit)

    lines: List[str] = []
    lines.append(f"## [{version}] - {today}")
    lines.append("")

    if breaking:
        lines.append("### Breaking changes")
        lines.append("")
        for commit in breaking:
            scope = f"**{commit.scope}**: " if commit.scope else ""
            lines.append(
                f"- {scope}{commit.message} ({commit.sha})"
            )
        lines.append("")

    for category in _CATEGORY_ORDER:
        bucket = grouped.get(category)
        if not bucket:
            continue
        lines.append(f"### {category}")
        lines.append("")
        for commit in bucket:
            scope = f"**{commit.scope}**: " if commit.scope else ""
            lines.append(
                f"- {scope}{commit.message} ({commit.sha})"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _render_placeholder(
    version: str, *, date: Optional[str] = None
) -> str:
    """Fallback section used when git extraction yields nothing."""
    today = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return (
        f"## [{version}] - {today}\n"
        f"\n"
        f"### Changed\n"
        f"- Version bump to {version}. No commit log was available "
        f"(not a git repository, shallow clone, or no Conventional "
        f"Commits in the range). Fill in the highlights for this "
        f"release before tagging.\n"
        f"\n"
    )


# ---------------------------------------------------------------------------
# CHANGELOG.md insertion
# ---------------------------------------------------------------------------


def _insert_into_changelog(section: str) -> None:
    """
    Insert ``section`` into CHANGELOG.md.

    The insertion point is immediately after the "# Changelog" heading
    and any leading metadata paragraph, immediately BEFORE the first
    existing "## [" section. This preserves prior sections and keeps
    the file in descending version order.
    """
    if not _CHANGELOG_PATH.is_file():
        raise SystemExit(
            f"error: CHANGELOG.md not found at {_CHANGELOG_PATH}"
        )

    existing = _CHANGELOG_PATH.read_text(encoding="utf-8")

    # Find the first existing version section marker.
    marker_re = re.compile(r"^##\s+\[", re.MULTILINE)
    match = marker_re.search(existing)

    if match is None:
        # No existing version sections — append at the end after a
        # blank line. Unusual but safe.
        new_content = existing.rstrip() + "\n\n" + section
    else:
        head = existing[: match.start()].rstrip() + "\n"
        tail = existing[match.start():]
        new_content = head + "\n" + section + "\n" + tail

    tmp = _CHANGELOG_PATH.with_suffix(".md.tmp")
    tmp.write_text(new_content, encoding="utf-8")
    tmp.replace(_CHANGELOG_PATH)


# ---------------------------------------------------------------------------
# Manifest helper (for --version default)
# ---------------------------------------------------------------------------


def _manifest_version() -> str:
    if not _MANIFEST_PATH.is_file():
        raise SystemExit(
            f"error: manifest.json not found at {_MANIFEST_PATH}"
        )
    try:
        data = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"error: manifest.json is invalid: {exc}")
    version = data.get("version")
    if not isinstance(version, str):
        raise SystemExit(
            "error: manifest.json has no string 'version' field."
        )
    return version


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a CHANGELOG.md section from git commits, "
            "classifying them via Conventional Commits."
        ),
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help=(
            "Version for the section heading. Defaults to "
            "manifest.json['version']."
        ),
    )
    parser.add_argument(
        "--from",
        dest="from_ref",
        type=str,
        default=None,
        help=(
            "Git ref to start from (exclusive). Defaults to the "
            "most recent tag reachable from HEAD."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated section without writing the file.",
    )
    parser.add_argument(
        "--placeholder",
        action="store_true",
        help=(
            "Skip the git extraction and emit the placeholder "
            "section directly."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = _parse_args(argv)

    version = args.version or _manifest_version()
    if not version:
        print("error: --version empty and manifest has no version.",
              file=sys.stderr)
        return 1

    # --placeholder path: no git work at all.
    if args.placeholder:
        section = _render_placeholder(version)
        if args.dry_run:
            print(section)
            return 0
        try:
            _insert_into_changelog(section)
        except SystemExit as exc:
            print(str(exc), file=sys.stderr)
            return 2
        except OSError as exc:
            print(f"error: could not write CHANGELOG.md: {exc}",
                  file=sys.stderr)
            return 2
        print(f"Inserted placeholder section for {version}.")
        return 0

    # --git path.
    if not _is_git_repo():
        print(
            "error: not inside a git work tree. Run with "
            "--placeholder to emit the fallback section instead.",
            file=sys.stderr,
        )
        return 1

    if _has_shallow_clone():
        print(
            "warning: shallow clone detected — commit history may be "
            "incomplete.",
            file=sys.stderr,
        )

    since_ref = args.from_ref or _latest_tag()
    if since_ref is None:
        print(
            "info: no tags found — including every commit reachable "
            "from HEAD.",
            file=sys.stderr,
        )

    commits = _collect_commits(since_ref)

    if not commits:
        if args.dry_run:
            print(_render_placeholder(version))
            return 0
        print(
            "error: no commits between the resolved refs. "
            "Run with --placeholder to emit the fallback section.",
            file=sys.stderr,
        )
        return 3

    section = _render_section(version, commits)

    if args.dry_run:
        print(section)
        return 0

    try:
        _insert_into_changelog(section)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: could not write CHANGELOG.md: {exc}",
              file=sys.stderr)
        return 2

    print(
        f"Inserted section for {version} with {len(commits)} entries "
        f"from {since_ref or 'initial'}..HEAD."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))