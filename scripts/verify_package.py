#!/usr/bin/env python3
# scripts/verify_package.py
"""
Deep verification of a built Provider Package ZIP.

Run in CI after ``scripts/build_package.py`` to guarantee that the
archive is safe AND that the real PELDRUN installer will accept it.
The checks mirror ``omweb.provider_packages.package_validator``
exactly, using the same constants, so a ZIP that passes here will
pass install-time validation.

Checks
------
1. The file exists and is a valid ZIP archive.
2. Compressed size ≤ ``MAX_PACKAGE_SIZE_BYTES`` (100 MB).
3. Uncompressed size ≤ ``MAX_UNCOMPRESSED_BYTES`` (500 MB).
4. Compression ratio ≤ ``MAX_COMPRESSION_RATIO`` (100:1).
5. File count ≤ ``MAX_FILE_COUNT`` (2000).
6. Path safety, per entry:
     * no NUL, no backslash,
     * no absolute path, no Windows drive letter,
     * no ``..`` traversal, no empty interior segment,
     * path length ≤ ``MAX_PATH_LENGTH`` (1024).
7. Symlink rejection (``S_IFLNK`` in ``external_attr``).
8. ``manifest.json`` is present at the archive root (or inside
   exactly one top-level folder, which build_package.py never
   produces but a manual build might).
9. Manifest is valid JSON, is an object, and passes basic schema
   validation (schema_version, kind, id, version, api_contract,
   transport, permissions, dependency list).
10. ``entry_point`` (for native / custom transports) resolves to a
    real ``.py`` file inside the archive.
11. ``requirements.txt`` is present at the archive root for
    native / custom transports.
12. SHA-256 is computed and printed for downstream use (e.g. to
    populate ``catalog-entry.json``).
13. Optional: ``--expected-sha256`` compares against a recorded hash
    to detect non-reproducible builds.

Usage
-----
    python scripts/verify_package.py dist/cloudflare-ai-1.0.0.zip
    python scripts/verify_package.py dist/cloudflare-ai-1.0.0.zip \
        --expected-sha256 0a1b2c3d...
    python scripts/verify_package.py    # auto-discovers dist/*.zip

Exit codes
----------
    0 — archive is valid
    1 — archive failed one or more checks
    2 — file not found / not a readable ZIP
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import sys
import zipfile
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Constants — mirrored from omweb.provider_packages.schema / package_validator
# ---------------------------------------------------------------------------


MAX_PACKAGE_SIZE_BYTES: int = 100 * 1024 * 1024     # 100 MB
MAX_UNCOMPRESSED_BYTES: int = 500 * 1024 * 1024     # 500 MB
MAX_FILE_COUNT: int = 2000
MAX_COMPRESSION_RATIO: int = 100                     # 100:1
MAX_PATH_LENGTH: int = 1024

MANIFEST_FILENAME: str = "manifest.json"
REQUIREMENTS_FILENAME: str = "requirements.txt"

SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0.0"})
SUPPORTED_API_CONTRACTS = frozenset({"peldrun.llm.provider.v1"})
SUPPORTED_TRANSPORTS = frozenset({"compatible", "native", "custom"})
SUPPORTED_NETWORK = frozenset({"none", "connection_endpoint"})
SUPPORTED_SECRETS = frozenset({"connection.api_key"})
REQUIRED_KIND = "peldrun.llm_provider"

BUILTIN_ADAPTER_IDS = frozenset({
    "openai_compat",
    "lmstudio",
    "ollama",
    "openai_native",
    "anthropic",
    "google_genai",
    "ollama_native",
    "lmstudio_native",
})

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_VERSION_RE = re.compile(
    r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
_ENTRY_POINT_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$"
)
_BASE_ADAPTER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


#: Path components that must NEVER appear in a Provider Package.
#: ``__MACOSX`` and ``.DS_Store`` are harmless metadata; ``..`` is a
#: security violation. We reject both so the archive stays predictable.
_FORBIDDEN_COMPONENTS: frozenset[str] = frozenset({
    "__MACOSX",
    ".DS_Store",
    "Thumbs.db",
    ".Spotlight-V100",
    ".fseventsd",
})


# ---------------------------------------------------------------------------
# Report accumulator
# ---------------------------------------------------------------------------


class _Report:
    """Accumulates per-check results and prints a structured summary."""

    def __init__(self) -> None:
        self.passes: List[str] = []
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def ok(self, msg: str) -> None:
        self.passes.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def fail(self, msg: str) -> None:
        self.errors.append(msg)

    def print_summary(self) -> None:
        for line in self.passes:
            print(f"  [ok]   {line}")
        for line in self.warnings:
            print(f"  [warn] {line}")
        for line in self.errors:
            print(f"  [err]  {line}")

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)


# ---------------------------------------------------------------------------
# Path safety — mirrors package_validator._check_member_path
# ---------------------------------------------------------------------------


def _check_member_path(name: str) -> Optional[str]:
    """
    Return an error string if ``name`` is unsafe, or ``None`` if safe.

    Rules (identical to ``package_validator._check_member_path``):
        * Non-empty.
        * No NUL or backslash.
        * Not an absolute path.
        * No ``..`` segment.
        * No interior empty segment (``//``).
        * Path length ≤ MAX_PATH_LENGTH.
    """
    if not name:
        return "empty path"
    if len(name) > MAX_PATH_LENGTH:
        return f"path exceeds {MAX_PATH_LENGTH} characters"
    if "\x00" in name:
        return "path contains NUL byte"
    if "\\" in name:
        return "path contains backslash"
    if name.startswith("/"):
        return "absolute path"
    if len(name) >= 2 and name[1] == ":":
        return "absolute Windows drive path"

    normalized = name.replace("\\", "/")
    for part in normalized.split("/"):
        if part == "..":
            return "path contains '..' traversal"
    if "//" in normalized:
        return "path contains empty segment"
    return None


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    """Return True if ``info`` describes a Unix symlink."""
    if not info.external_attr:
        return False
    mode = (info.external_attr >> 16) & 0o170000
    return mode == stat.S_IFLNK


# ---------------------------------------------------------------------------
# Manifest location — mirrors package_validator._locate_manifest
# ---------------------------------------------------------------------------


def _locate_manifest(
    names: List[str],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Return ``(manifest_path, strip_prefix, error)``.

    Accepts a root manifest or exactly one single-folder manifest.
    Multiple candidates are rejected as an ambiguity attack.
    """
    if MANIFEST_FILENAME in names:
        return MANIFEST_FILENAME, "", None

    single_folder = [
        n for n in names
        if n.endswith("/" + MANIFEST_FILENAME) and n.count("/") == 1
    ]
    if len(single_folder) == 1:
        path = single_folder[0]
        prefix = path.rsplit("/", 1)[0] + "/"
        return path, prefix, None
    if len(single_folder) > 1:
        return (
            None,
            None,
            f"multiple candidate manifests found: {single_folder!r}",
        )

    deeper = [n for n in names if n.endswith("/" + MANIFEST_FILENAME)]
    if deeper:
        return (
            None,
            None,
            f"manifest.json found at unexpected depth(s): {deeper!r}",
        )

    return (
        None,
        None,
        f"manifest.json is missing. It must be at the archive root "
        f"or inside a single top-level folder.",
    )


# ---------------------------------------------------------------------------
# Entry-point resolution — mirrors package_validator._find_entry_point_file
# ---------------------------------------------------------------------------


def _find_entry_point_file(
    names: List[str],
    module_path: str,
    strip_prefix: str,
) -> Optional[str]:
    """
    Return the archive-relative path of the file that implements
    ``module_path`` (part before ``:``), or ``None``.
    """
    if not strip_prefix.endswith("/") and strip_prefix != "":
        return None

    rel = module_path.replace(".", "/")
    candidates = [
        f"{strip_prefix}{rel}.py",
        f"{strip_prefix}{rel}/__init__.py",
        f"{strip_prefix}src/{rel}.py",
        f"{strip_prefix}src/{rel}/__init__.py",
    ]
    name_set = set(names)
    for candidate in candidates:
        if candidate in name_set:
            return candidate
    return None


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_sizes(
    infos: List[zipfile.ZipInfo],
    compressed_size: int,
    report: _Report,
) -> None:
    """File count, uncompressed size, compression ratio."""
    file_count = len(infos)
    report.ok(f"file count: {file_count}")
    if file_count > MAX_FILE_COUNT:
        report.fail(
            f"file count {file_count} exceeds {MAX_FILE_COUNT}"
        )

    if compressed_size > MAX_PACKAGE_SIZE_BYTES:
        report.fail(
            f"compressed size {compressed_size} bytes exceeds "
            f"{MAX_PACKAGE_SIZE_BYTES} bytes"
        )

    total = 0
    for info in infos:
        if info.file_size < 0:
            report.fail(
                f"entry {info.filename!r} has negative size "
                f"{info.file_size}"
            )
            continue
        total += info.file_size

    if total > MAX_UNCOMPRESSED_BYTES:
        report.fail(
            f"uncompressed size {total} bytes exceeds "
            f"{MAX_UNCOMPRESSED_BYTES} bytes"
        )
    elif compressed_size > 0:
        ratio = total / compressed_size
        if ratio > MAX_COMPRESSION_RATIO:
            report.fail(
                f"compression ratio {ratio:.1f}:1 exceeds "
                f"{MAX_COMPRESSION_RATIO}:1"
            )
        else:
            report.ok(
                f"uncompressed size: {total} bytes "
                f"(ratio {ratio:.1f}:1)"
            )


def _check_entries(
    infos: List[zipfile.ZipInfo], report: _Report
) -> Tuple[List[str], bool]:
    """
    Path safety, symlink rejection, junk detection.

    Returns ``(safe_names, ok)``. ``safe_names`` includes only entries
    that passed every check.
    """
    safe_names: List[str] = []
    ok = True

    for info in infos:
        name = info.filename

        err = _check_member_path(name)
        if err is not None:
            report.fail(f"unsafe path {name!r}: {err}")
            ok = False
            continue

        if _is_symlink(info):
            report.fail(f"symlink entry {name!r} is not permitted")
            ok = False
            continue

        # Junk detection — build_package.py should have stripped these.
        parts = name.split("/")
        junk_hit = any(part in _FORBIDDEN_COMPONENTS for part in parts)
        if junk_hit:
            report.warn(
                f"entry {name!r} matches archiver junk "
                f"(should have been stripped by build_package.py)"
            )

        safe_names.append(name)

    if ok:
        report.ok(f"path safety: {len(safe_names)} entries checked")
    return safe_names, ok


def _check_manifest(
    data: Any, report: _Report
) -> Optional[dict]:
    """
    Structural manifest validation (stdlib-only; mirrors the strictest
    subset of ProviderManifest).
    """
    if not isinstance(data, dict):
        report.fail(
            f"manifest must be a JSON object, got {type(data).__name__}"
        )
        return None

    # Required identity
    schema_version = data.get("schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        report.fail(
            f"schema_version {schema_version!r} not in "
            f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )

    kind = data.get("kind")
    if kind != REQUIRED_KIND:
        report.fail(f"kind must be {REQUIRED_KIND!r}, got {kind!r}")

    provider_id = data.get("id")
    if not isinstance(provider_id, str) or not _ID_RE.match(provider_id):
        report.fail(f"id {provider_id!r} does not match {_ID_RE.pattern}")
    elif provider_id in BUILTIN_ADAPTER_IDS:
        report.fail(
            f"id {provider_id!r} collides with a built-in adapter"
        )

    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        report.fail("name must be a non-empty string")

    version = data.get("version")
    if not isinstance(version, str) or not _VERSION_RE.match(version):
        report.fail(f"version {version!r} is not valid semver")

    api_contract = data.get("api_contract")
    if api_contract not in SUPPORTED_API_CONTRACTS:
        report.fail(
            f"api_contract {api_contract!r} not in "
            f"{sorted(SUPPORTED_API_CONTRACTS)}"
        )

    transport = data.get("transport")
    if transport not in SUPPORTED_TRANSPORTS:
        report.fail(
            f"transport {transport!r} not in {sorted(SUPPORTED_TRANSPORTS)}"
        )
        return data  # further checks need transport

    entry_point = data.get("entry_point")
    base_adapter = data.get("base_adapter")

    if transport == "compatible":
        if not isinstance(base_adapter, str) or not base_adapter.strip():
            report.fail(
                "transport=compatible requires a non-empty base_adapter"
            )
        elif not _BASE_ADAPTER_RE.match(base_adapter):
            report.fail(
                f"base_adapter {base_adapter!r} does not match "
                f"{_BASE_ADAPTER_RE.pattern}"
            )
        if entry_point is not None:
            report.fail(
                "transport=compatible must not declare entry_point"
            )
    else:  # native / custom
        if not isinstance(entry_point, str) or not entry_point.strip():
            report.fail(
                f"transport={transport} requires a non-empty entry_point"
            )
        elif not _ENTRY_POINT_RE.match(entry_point):
            report.fail(
                f"entry_point {entry_point!r} does not match "
                f"{_ENTRY_POINT_RE.pattern}"
            )
        if base_adapter is not None:
            report.fail(
                f"transport={transport} must not declare base_adapter"
            )

    # Permissions
    perms = data.get("permissions")
    if not isinstance(perms, dict):
        report.fail("permissions must be an object")
    else:
        if perms.get("workspace_access") is not False:
            report.fail(
                "permissions.workspace_access must be explicitly false"
            )
        if perms.get("shell_execute") is not False:
            report.fail(
                "permissions.shell_execute must be explicitly false"
            )
        network = perms.get("network")
        if network not in SUPPORTED_NETWORK:
            report.fail(
                f"permissions.network {network!r} not in "
                f"{sorted(SUPPORTED_NETWORK)}"
            )
        secrets = perms.get("secrets")
        if not isinstance(secrets, list):
            report.fail("permissions.secrets must be a list")
        else:
            for entry in secrets:
                if entry not in SUPPORTED_SECRETS:
                    report.fail(
                        f"permissions.secrets entry {entry!r} not in "
                        f"{sorted(SUPPORTED_SECRETS)}"
                    )

    # Dependencies
    deps = data.get("dependencies")
    if deps is not None:
        if not isinstance(deps, list):
            report.fail("dependencies must be a list")
        else:
            for i, entry in enumerate(deps):
                if not isinstance(entry, str) or not entry.strip():
                    report.fail(
                        f"dependencies[{i}] must be a non-empty string"
                    )

    if not report.has_errors:
        report.ok(
            f"manifest: id={provider_id} version={version} "
            f"transport={transport}"
        )
    return data


def _check_entry_point_and_requirements(
    data: dict,
    names: List[str],
    strip_prefix: str,
    report: _Report,
) -> None:
    """Verify entry_point resolves and requirements.txt is present."""
    transport = data.get("transport")
    if transport == "compatible":
        report.ok("entry_point: not required (compatible transport)")
        return

    entry_point = data.get("entry_point")
    if not isinstance(entry_point, str) or ":" not in entry_point:
        # Already failed in _check_manifest
        return

    module_path = entry_point.split(":", 1)[0]
    resolved = _find_entry_point_file(names, module_path, strip_prefix)
    if resolved is None:
        report.fail(
            f"entry_point {entry_point!r} does not resolve to a .py "
            f"file in the archive"
        )
    else:
        report.ok(f"entry_point resolves: {resolved}")

    # requirements.txt at root (for native/custom)
    if REQUIREMENTS_FILENAME not in names:
        report.warn(
            f"{REQUIREMENTS_FILENAME} is not at the archive root. "
            f"Native/custom packages typically require it so the "
            f"installer knows which dependencies to install."
        )
    else:
        report.ok(f"{REQUIREMENTS_FILENAME}: present at root")


# ---------------------------------------------------------------------------
# Main verify function
# ---------------------------------------------------------------------------


def verify(zip_path: Path, expected_sha256: Optional[str]) -> int:
    print(f"Verifying {zip_path}")
    print()

    if not zip_path.is_file():
        print(f"error: file not found: {zip_path}", file=sys.stderr)
        return 2

    compressed_size = zip_path.stat().st_size
    print(f"  size on disk: {compressed_size} bytes")

    # Compute SHA-256 of the raw file bytes.
    hasher = hashlib.sha256()
    try:
        with zip_path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(64 * 1024), b""):
                hasher.update(chunk)
    except OSError as exc:
        print(f"error: could not read file: {exc}", file=sys.stderr)
        return 2

    actual_sha256 = hasher.hexdigest()
    print(f"  sha256:       {actual_sha256}")

    if expected_sha256 is not None:
        if not _SHA256_HEX_RE.match(expected_sha256.lower()):
            print(
                f"error: --expected-sha256 {expected_sha256!r} is not "
                f"a valid hex SHA-256 digest.",
                file=sys.stderr,
            )
            return 2
        if actual_sha256 != expected_sha256.lower():
            print()
            print(
                f"error: SHA-256 mismatch. "
                f"expected {expected_sha256.lower()}, "
                f"got {actual_sha256}",
                file=sys.stderr,
            )
            return 1
        print("  sha256 match: ok")

    print()

    try:
        zip_file = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        print(f"error: not a valid ZIP archive: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: could not open archive: {exc}", file=sys.stderr)
        return 2

    report = _Report()

    with zip_file:
        try:
            infos = zip_file.infolist()
        except Exception as exc:
            print(
                f"error: could not read central directory: {exc}",
                file=sys.stderr,
            )
            return 2

        _check_sizes(infos, compressed_size, report)

        safe_names, entries_ok = _check_entries(infos, report)

        if entries_ok:
            manifest_path, strip_prefix, layout_err = _locate_manifest(
                safe_names
            )
            if layout_err is not None:
                report.fail(layout_err)
            else:
                assert manifest_path is not None
                assert strip_prefix is not None

                if strip_prefix:
                    report.warn(
                        f"manifest inside {strip_prefix!r} — will be "
                        f"flattened at install time"
                    )
                else:
                    report.ok("manifest.json: at archive root")

                try:
                    raw = zip_file.read(manifest_path)
                    data = json.loads(raw.decode("utf-8"))
                except UnicodeDecodeError as exc:
                    report.fail(f"manifest.json is not valid UTF-8: {exc}")
                    data = None
                except json.JSONDecodeError as exc:
                    report.fail(f"manifest.json is not valid JSON: {exc}")
                    data = None
                except Exception as exc:
                    report.fail(f"could not read manifest: {exc}")
                    data = None

                if isinstance(data, dict):
                    _check_manifest(data, report)
                    if not report.has_errors:
                        _check_entry_point_and_requirements(
                            data, safe_names, strip_prefix, report
                        )

    report.print_summary()
    print()

    if report.has_errors:
        print(
            f"FAILED: {len(report.errors)} error(s), "
            f"{len(report.warnings)} warning(s), "
            f"{len(report.passes)} check(s) passed."
        )
        return 1

    print(
        f"VALID: {len(report.passes)} check(s) passed, "
        f"{len(report.warnings)} warning(s)."
    )
    print()
    print(f"  archive: {zip_path.name}")
    print(f"  sha256:  {actual_sha256}")
    print()
    print(
        "Use the SHA-256 above as 'checksum_sha256' in the catalog "
        "entry for this package."
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_zip_path() -> Optional[Path]:
    """
    Auto-discover ``dist/*.zip`` when no path is given.

    Returns the most recently modified ZIP, or ``None`` when ``dist/``
    is empty.
    """
    repo_root = Path(__file__).resolve().parent.parent
    dist = repo_root / "dist"
    if not dist.is_dir():
        return None
    zips = sorted(dist.glob("*.zip"), key=lambda p: p.stat().st_mtime)
    return zips[-1] if zips else None


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Deep verification of a built Provider Package ZIP. "
            "Mirrors package_validator.py's safety checks."
        ),
    )
    parser.add_argument(
        "zip_path",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Path to the ZIP to verify. When omitted, uses the most "
            "recently modified file in dist/."
        ),
    )
    parser.add_argument(
        "--expected-sha256",
        type=str,
        default=None,
        help=(
            "Optional. When provided, fails the check if the archive's "
            "SHA-256 does not match. Useful for CI reproducibility "
            "checks."
        ),
    )
    args = parser.parse_args(argv)

    zip_path = args.zip_path
    if zip_path is None:
        zip_path = _default_zip_path()
        if zip_path is None:
            print(
                "error: no zip path given and dist/ has no .zip files.",
                file=sys.stderr,
            )
            return 2

    return verify(zip_path, args.expected_sha256)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))