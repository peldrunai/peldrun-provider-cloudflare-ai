#!/usr/bin/env python3
# scripts/generate_catalog_entry.py
"""
Generate a ``catalog-entry.json`` for a Provider Package.

The catalog entry is what a PELDRUN *catalog* consumes — it is NOT part
of the ZIP. It lets the catalog declare presentation fields (brand
color, icon) and a download URL, none of which fit inside
``manifest.json`` (whose Pydantic model rejects extra fields by
design).

Why generate it instead of hand-writing
---------------------------------------
The catalog entry MUST stay in sync with:

  * ``manifest.json`` — for id, name, version, features, permissions,
    connection_fields, author, license, tags, etc.
  * The produced ``.zip`` — for the SHA-256 checksum, which changes
    whenever any source file changes.

If the entry were hand-written, every ``update_version.py`` run would
require a coordinated manual edit — a step that is easy to forget and
produces a catalog that silently fails checksum verification at
install time. Generating it from source removes that failure mode.

Design notes
------------
* **Presentation data is passed via CLI**, not stored in
  ``manifest.json``. ``brand_color`` and ``icon`` belong to how the
  catalog *displays* the package, not to what the package *is*.
  Keeping them out of the manifest means the same ZIP can appear in
  multiple catalogs with different branding.

* **The SHA-256 is computed from the ZIP**, not from any manifest
  field. This is the only hash that the installer will compare
  against at download time.

* **``download_url`` is not hardcoded.** A package author publishes
  one ZIP once, but may host it on GitHub Releases, R2, S3, a CDN,
  or an internal mirror. The URL is passed via ``--download-url``
  (or an environment variable in CI). The generator does not assume
  a hosting provider.

* **Key order is deterministic and semantic.** The output dictionary
  is built in a fixed order (Identity → Runtime → Presentation →
  Distribution → Publisher → Misc) that matches the official PELDRUN
  catalog. Deterministic ordering (same manifest → same output
  byte-for-byte) keeps git diffs small between releases.

* **Output defaults to ``dist/catalog-entry.json``** so it sits next
  to the ZIP and can be attached to the same GitHub Release.

Usage
-----
    python scripts/generate_catalog_entry.py \\
        --download-url https://github.com/USER/REPO/releases/download/v1.0.0/cloudflare-ai-1.0.0.zip \\
        --brand-color "#F38020" \\
        --icon cloudflare

    # In CI, values can come from environment variables:
    #   PELDRUN_CATALOG_DOWNLOAD_URL=...
    #   PELDRUN_CATALOG_BRAND_COLOR=...
    #   PELDRUN_CATALOG_ICON_NAME=...

Exit codes
----------
    0 — catalog entry written successfully
    1 — manifest or ZIP is missing / invalid
    2 — output file could not be written
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MANIFEST_PATH = _REPO_ROOT / "manifest.json"
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "dist"


#: Validation patterns (subset of ``schema.py`` — this script is
#: intentionally stdlib-only so it can run in a bare CI container).
_HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_ICON_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


# ---------------------------------------------------------------------------
# Environment-variable resolution
# ---------------------------------------------------------------------------

_ENV_DOWNLOAD_URL = "PELDRUN_CATALOG_DOWNLOAD_URL"
_ENV_BRAND_COLOR = "PELDRUN_CATALOG_BRAND_COLOR"
_ENV_ICON_NAME = "PELDRUN_CATALOG_ICON_NAME"
_ENV_ICON_TYPE = "PELDRUN_CATALOG_ICON_TYPE"
_ENV_REPOSITORY_URL = "PELDRUN_CATALOG_REPOSITORY_URL"
_ENV_DOCS_URL = "PELDRUN_CATALOG_DOCS_URL"


def _resolve_str(
    cli_value: Optional[str],
    env_var: str,
    *,
    required: bool = False,
    default: Optional[str] = None,
) -> Optional[str]:
    """
    Resolve a value from CLI, then environment, then ``default``.

    Raises ``SystemExit`` if ``required`` is True and the resolved
    value is empty.
    """
    if cli_value is not None and cli_value.strip():
        return cli_value.strip()

    env_value = os.environ.get(env_var, "").strip()
    if env_value:
        return env_value

    if default is not None:
        return default

    if required:
        raise SystemExit(
            f"error: value required. Provide it via CLI or "
            f"environment variable {env_var}."
        )
    return None


# ---------------------------------------------------------------------------
# Reading the built ZIP
# ---------------------------------------------------------------------------


def _zip_path_for_version(version: str, output_dir: Path) -> Path:
    """Return the canonical ZIP path for ``version`` in ``output_dir``."""
    return output_dir / f"cloudflare-ai-{version}.zip"


def _compute_sha256(path: Path) -> str:
    """Return the lowercase hex SHA-256 of ``path``'s bytes."""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(64 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


def _load_manifest() -> Dict[str, Any]:
    """Load and structurally validate ``manifest.json``."""
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

    for required in ("id", "name", "version", "transport", "permissions"):
        if required not in data:
            raise SystemExit(
                f"error: manifest.json is missing required field "
                f"{required!r}"
            )
    return data


# ---------------------------------------------------------------------------
# Developer derivation
# ---------------------------------------------------------------------------


def _derive_developer(
    manifest: Dict[str, Any],
    repository_url: Optional[str],
) -> Optional[Dict[str, Any]]:
    """
    Build a ``developer`` block from the manifest's author fields.

    The catalog UI renders a richer publisher card when ``developer`` is
    present. Deriving it from the manifest ensures every published
    ``catalog-entry.json`` carries the same publisher metadata — no
    manual per-release editing required.
    """
    author = manifest.get("author")
    if not isinstance(author, str) or not author.strip():
        return None

    developer: Dict[str, Any] = {
        "name": author.strip(),
        "verified": False,
    }

    # Prefer author_url; fall back to repository_url if it looks like
    # a personal / organization profile.
    author_url = manifest.get("author_url")
    if isinstance(author_url, str) and author_url.strip():
        developer["url"] = author_url.strip()
    elif repository_url:
        developer["url"] = repository_url.rstrip("/")

    # Try to extract a GitHub username from either URL.
    candidate = developer.get("url") or ""
    if "github.com/" in candidate:
        tail = candidate.split("github.com/", 1)[1]
        username = tail.split("/", 1)[0].strip()
        if username and username not in {"releases", "issues", "pull"}:
            developer["github_username"] = username

    return developer


# ---------------------------------------------------------------------------
# Entry assembly
# ---------------------------------------------------------------------------


def _build_catalog_entry(
    manifest: Dict[str, Any],
    *,
    download_url: str,
    checksum_sha256: str,
    brand_color: Optional[str],
    icon: Optional[str],
    icon_type: str,
    repository_url: Optional[str],
    docs_url: Optional[str],
    changelog_url: Optional[str],
    screenshots: Optional[List[str]],
) -> Dict[str, Any]:
    """
    Build the catalog entry dict from the manifest plus presentation
    and distribution inputs.

    The key order is deliberately chosen to match the official
    PELDRUN provider catalog:

        Identity  →  Runtime  →  Presentation  →  Distribution
                  →  Publisher  →  Misc / links

    Insertion order is preserved by Python 3.7+ dicts, so no sorting
    step is needed. This gives human-readable files AND byte-for-byte
    deterministic output (same manifest → same file).
    """
    entry: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 1. Identity — required fields.
    # ------------------------------------------------------------------
    entry["id"] = manifest["id"]
    entry["name"] = manifest["name"]

    if manifest.get("display_name"):
        entry["display_name"] = manifest["display_name"]
    if manifest.get("description"):
        entry["description"] = manifest["description"]

    entry["version"] = manifest["version"]
    entry["transport"] = manifest["transport"]

    # ------------------------------------------------------------------
    # 2. Runtime / capabilities.
    # ------------------------------------------------------------------
    if "features" in manifest:
        entry["features"] = manifest["features"]
    if "permissions" in manifest:
        entry["permissions"] = manifest["permissions"]

    # ------------------------------------------------------------------
    # 3. Presentation — set only when provided.
    # ------------------------------------------------------------------
    if brand_color:
        entry["brand_color"] = brand_color
    if icon:
        entry["icon"] = icon
    entry["icon_type"] = icon_type

    # ------------------------------------------------------------------
    # 4. Distribution — required for a valid catalog entry.
    # ------------------------------------------------------------------
    entry["download_url"] = download_url
    entry["checksum_sha256"] = checksum_sha256

    # ------------------------------------------------------------------
    # 5. Publisher.
    # ------------------------------------------------------------------
    if manifest.get("author"):
        entry["author"] = manifest["author"]
    if manifest.get("author_url"):
        entry["author_url"] = manifest["author_url"]
    if manifest.get("homepage"):
        entry["homepage"] = manifest["homepage"]

    derived_developer = _derive_developer(manifest, repository_url)
    if derived_developer:
        entry["developer"] = derived_developer

    # ------------------------------------------------------------------
    # 6. Miscellaneous links and metadata.
    # ------------------------------------------------------------------
    if repository_url:
        entry["repository_url"] = repository_url
    if manifest.get("license"):
        entry["license"] = manifest["license"]
    if manifest.get("tags"):
        entry["tags"] = manifest["tags"]

    # docs_url: prefer the explicit --docs-url value, then fall back
    # to the manifest's homepage (which almost always points at the
    # provider's documentation). This means the same documentation
    # link never appears twice.
    resolved_docs = docs_url or manifest.get("homepage")
    if isinstance(resolved_docs, str) and resolved_docs:
        entry["docs_url"] = resolved_docs

    if changelog_url:
        entry["changelog_url"] = changelog_url

    # ------------------------------------------------------------------
    # 7. Package-specific payloads (connection fields, certificates).
    #    Placed at the end because they are the largest, least-scanned
    #    sections — keeping them out of the way for a quick glance.
    # ------------------------------------------------------------------
    if manifest.get("connection_fields"):
        entry["connection_fields"] = manifest["connection_fields"]
    if manifest.get("certificates"):
        entry["certificates"] = manifest["certificates"]
    if screenshots:
        entry["screenshots"] = list(screenshots)

    return entry


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write ``data`` to ``path`` atomically."""
    serialized = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(serialized, encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        raise SystemExit(f"error: could not write {path}: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a catalog-entry.json for the built Provider "
            "Package ZIP."
        ),
    )
    parser.add_argument(
        "--download-url",
        type=str,
        default=None,
        help=(
            "Public URL where the ZIP is hosted. If omitted, reads "
            f"from the {_ENV_DOWNLOAD_URL} environment variable."
        ),
    )
    parser.add_argument(
        "--brand-color",
        type=str,
        default=None,
        help=(
            "Hex color for the catalog icon (e.g. '#F38020'). "
            f"If omitted, reads from {_ENV_BRAND_COLOR}."
        ),
    )
    parser.add_argument(
        "--icon",
        type=str,
        default=None,
        help=(
            "Icon identifier. When icon_type is 'name', this is the "
            "base name of an asset in public/icon/ (e.g. 'cloudflare'). "
            f"If omitted, reads from {_ENV_ICON_NAME}."
        ),
    )
    parser.add_argument(
        "--icon-type",
        choices=("name", "svg"),
        default=None,
        help=(
            "How to interpret --icon. Default: 'name'. "
            f"If omitted, reads from {_ENV_ICON_TYPE}."
        ),
    )
    parser.add_argument(
        "--repository-url",
        type=str,
        default=None,
        help=(
            "Source repository URL (optional). If omitted, reads "
            f"from {_ENV_REPOSITORY_URL}."
        ),
    )
    parser.add_argument(
        "--docs-url",
        type=str,
        default=None,
        help=(
            "Documentation URL (optional). If omitted, falls back to "
            f"the manifest's 'homepage' field, then to {_ENV_DOCS_URL}."
        ),
    )
    parser.add_argument(
        "--changelog-url",
        type=str,
        default=None,
        help=(
            "Changelog URL (optional). Shown in the Details drawer. "
            "Typically 'https://github.com/OWNER/REPO/releases'."
        ),
    )
    parser.add_argument(
        "--screenshot",
        type=str,
        action="append",
        default=None,
        help=(
            "Screenshot URL (repeatable). Rendered in the Details "
            "gallery. Pass multiple times for multiple images."
        ),
    )
    parser.add_argument(
        "--zip-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help=(
            f"Directory containing the built ZIP "
            f"(default: {_DEFAULT_OUTPUT_DIR})."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path for the catalog entry. "
            "Default: <zip-dir>/catalog-entry.json."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = _parse_args(argv)

    manifest = _load_manifest()
    version = str(manifest["version"])

    zip_dir = Path(args.zip_dir).resolve()
    zip_path = _zip_path_for_version(version, zip_dir)
    if not zip_path.is_file():
        raise SystemExit(
            f"error: built ZIP not found at {zip_path}. Run "
            f"`python scripts/build_package.py` first."
        )

    checksum = _compute_sha256(zip_path)
    if not _SHA256_HEX_RE.match(checksum):
        raise SystemExit(
            f"error: computed SHA-256 {checksum!r} is not a valid "
            f"hex digest — this should never happen."
        )

    download_url = _resolve_str(
        args.download_url, _ENV_DOWNLOAD_URL, required=True
    )
    assert download_url is not None  # for type checkers
    if not download_url.lower().startswith(("http://", "https://")):
        raise SystemExit(
            f"error: download URL must use http(s)://, got "
            f"{download_url!r}"
        )

    brand_color = _resolve_str(args.brand_color, _ENV_BRAND_COLOR)
    if brand_color is not None and not _HEX_COLOR_RE.match(brand_color):
        raise SystemExit(
            f"error: brand color {brand_color!r} must be a #RRGGBB "
            f"hex string."
        )

    icon = _resolve_str(args.icon, _ENV_ICON_NAME)
    if icon is not None:
        icon_type_value = (
            _resolve_str(args.icon_type, _ENV_ICON_TYPE, default="name")
            or "name"
        )
        if icon_type_value == "name" and not _ICON_NAME_RE.match(icon):
            raise SystemExit(
                f"error: icon identifier {icon!r} must match "
                f"{_ICON_NAME_RE.pattern} when icon_type='name'."
            )
    else:
        icon_type_value = "name"

    repository_url = _resolve_str(
        args.repository_url, _ENV_REPOSITORY_URL
    )
    docs_url = _resolve_str(args.docs_url, _ENV_DOCS_URL)

    entry = _build_catalog_entry(
        manifest,
        download_url=download_url,
        checksum_sha256=checksum,
        brand_color=brand_color,
        icon=icon,
        icon_type=icon_type_value,
        repository_url=repository_url,
        docs_url=docs_url,
        changelog_url=args.changelog_url,
        screenshots=args.screenshot,
    )

    output_path = args.output or (zip_dir / "catalog-entry.json")
    _write_json(output_path, entry)

    # Print a compact summary — useful in CI logs.
    print(f"Wrote {output_path}")
    print(f"  id:            {entry.get('id')}")
    print(f"  version:       {entry.get('version')}")
    print(f"  download_url:  {entry.get('download_url')}")
    print(f"  checksum:      {entry.get('checksum_sha256')}")
    if brand_color:
        print(f"  brand_color:   {brand_color}")
    if icon:
        print(f"  icon:          {icon} ({icon_type_value})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))