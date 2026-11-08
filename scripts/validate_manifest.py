#!/usr/bin/env python3
# scripts/validate_manifest.py
"""
Validate the PELDRUN provider manifest.

Runs in CI on every PR. Deliberately self-contained (stdlib only) so it
works in a bare CI environment without PELDRUN or pydantic installed.
Where PELDRUN's real ``ProviderManifest`` is importable (dev machines),
it runs as an extra validation layer.

Checks performed:

  1. JSON parses cleanly.
  2. Required identity fields are present and typed.
  3. Patterns: id, version, entry_point, base_adapter.
  4. Transport consistency: entry_point XOR base_adapter.
  5. Permissions: workspace_access=false, shell_execute=false,
     network in {none, connection_endpoint}, secrets in
     {connection.api_key}.
  6. No builtin adapter ID collision.
  7. Dependencies are non-empty strings.
  8. Entry-point module resolves to a real .py file.
  9. Optional: PELDRUN ProviderManifest schema validation
     (skipped silently if peldrun is not importable).

Usage:
    python scripts/validate_manifest.py                # uses ./manifest.json
    python scripts/validate_manifest.py path/to/manifest.json

Exit codes:
    0 — valid
    1 — validation failed
    2 — manifest missing or not valid JSON
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MANIFEST = _REPO_ROOT / "manifest.json"
_SRC_ROOT = _REPO_ROOT / "src"


# ---------------------------------------------------------------------------
# Allowed values (mirror docs/providers/04-manifest-reference.md)
# ---------------------------------------------------------------------------

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_VERSION_PATTERN = re.compile(
    r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)
_ENTRY_POINT_PATTERN = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$"
)
_BASE_ADAPTER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

_BUILTIN_ADAPTER_IDS = frozenset({
    "openai_compat",
    "lmstudio",
    "ollama",
    "openai_native",
    "anthropic",
    "google_genai",
    "ollama_native",
    "lmstudio_native",
})

_ALLOWED_TRANSPORTS = frozenset({"compatible", "native", "custom"})
_ALLOWED_NETWORK = frozenset({"none", "connection_endpoint"})
_ALLOWED_SECRETS = frozenset({"connection.api_key"})
_ALLOWED_SCHEMA_VERSIONS = frozenset({"1.0.0"})
_ALLOWED_API_CONTRACTS = frozenset({"peldrun.llm.provider.v1"})
_REQUIRED_KIND = "peldrun.llm_provider"

_KNOWN_FEATURE_FLAGS = frozenset({
    "model_discovery",
    "native_streaming",
    "buffered_streaming",
    "tool_calling",
    "parallel_tool_calls",
    "forced_tool_choice",
    "vision",
    "audio",
    "structured_output",
    "usage_reporting",
    "reasoning",
})


# ---------------------------------------------------------------------------
# Report accumulator
# ---------------------------------------------------------------------------


class _Report:
    """Accumulate per-check results and print a structured summary."""

    def __init__(self) -> None:
        self.passes: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []

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
# Field helpers
# ---------------------------------------------------------------------------


def _require(
    data: dict[str, Any],
    field: str,
    expected_type: type,
    report: _Report,
    *,
    allow_empty: bool = False,
) -> Any:
    """Verify a required field exists and has the right type."""
    if field not in data:
        report.fail(f"missing required field: {field!r}")
        return None
    value = data[field]
    if not isinstance(value, expected_type):
        report.fail(
            f"{field!r} must be {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )
        return None
    if not allow_empty and isinstance(value, str) and not value.strip():
        report.fail(f"{field!r} must be a non-empty string")
        return None
    return value


def _optional(
    data: dict[str, Any],
    field: str,
    expected_type: type,
    report: _Report,
) -> Any:
    """Verify an optional field, if present, has the right type."""
    if field not in data:
        return None
    value = data[field]
    if value is None:
        return None
    if not isinstance(value, expected_type):
        report.fail(
            f"{field!r} must be {expected_type.__name__} or null, "
            f"got {type(value).__name__}"
        )
        return None
    return value


# ---------------------------------------------------------------------------
# Individual check sections
# ---------------------------------------------------------------------------


def _check_identity(data: dict[str, Any], report: _Report) -> None:
    """Validate top-level identity and metadata fields."""

    schema_version = _require(data, "schema_version", str, report)
    if schema_version is not None:
        if schema_version not in _ALLOWED_SCHEMA_VERSIONS:
            report.fail(
                f"schema_version {schema_version!r} not in "
                f"{sorted(_ALLOWED_SCHEMA_VERSIONS)}"
            )
        else:
            report.ok(f"schema_version={schema_version}")

    kind = _require(data, "kind", str, report)
    if kind is not None:
        if kind != _REQUIRED_KIND:
            report.fail(
                f"kind must be {_REQUIRED_KIND!r}, got {kind!r}"
            )
        else:
            report.ok(f"kind={kind}")

    provider_id = _require(data, "id", str, report)
    if provider_id is not None:
        if not _ID_PATTERN.match(provider_id):
            report.fail(
                f"id {provider_id!r} does not match {_ID_PATTERN.pattern}"
            )
        elif provider_id in _BUILTIN_ADAPTER_IDS:
            report.fail(
                f"id {provider_id!r} collides with a builtin adapter. "
                f"Pick a different id."
            )
        else:
            report.ok(f"id={provider_id} (no builtin collision)")

    name = _require(data, "name", str, report)
    if name is not None:
        report.ok(f"name={name}")

    version = _require(data, "version", str, report)
    if version is not None:
        if not _VERSION_PATTERN.match(version):
            report.fail(
                f"version {version!r} is not valid semver"
            )
        else:
            report.ok(f"version={version}")

    api_contract = _require(data, "api_contract", str, report)
    if api_contract is not None:
        if api_contract not in _ALLOWED_API_CONTRACTS:
            report.fail(
                f"api_contract {api_contract!r} not in "
                f"{sorted(_ALLOWED_API_CONTRACTS)}"
            )
        else:
            report.ok(f"api_contract={api_contract}")

    for field in (
        "display_name",
        "description",
        "author",
        "author_url",
        "homepage",
        "license",
        "icon",
    ):
        _optional(data, field, str, report)

    tags = data.get("tags")
    if tags is not None:
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            report.fail("tags must be a list of strings")
        else:
            report.ok(f"tags: {len(tags)} entries")


def _check_transport(data: dict[str, Any], report: _Report) -> None:
    """Validate transport and the entry_point / base_adapter XOR rule."""

    transport = _require(data, "transport", str, report)
    if transport is None:
        return
    if transport not in _ALLOWED_TRANSPORTS:
        report.fail(
            f"transport {transport!r} not in {sorted(_ALLOWED_TRANSPORTS)}"
        )
        return

    entry_point = data.get("entry_point")
    base_adapter = data.get("base_adapter")

    if transport == "compatible":
        if entry_point is not None:
            report.fail(
                "transport='compatible' must not declare 'entry_point' "
                "(no code is shipped)"
            )
        if not isinstance(base_adapter, str) or not base_adapter.strip():
            report.fail(
                "transport='compatible' requires a non-empty 'base_adapter'"
            )
        elif not _BASE_ADAPTER_PATTERN.match(base_adapter):
            report.fail(
                f"base_adapter {base_adapter!r} does not match "
                f"{_BASE_ADAPTER_PATTERN.pattern}"
            )
        else:
            report.ok(
                f"transport=compatible base_adapter={base_adapter}"
            )

    elif transport in ("native", "custom"):
        if base_adapter is not None:
            report.fail(
                f"transport={transport!r} must not declare 'base_adapter'"
            )
        if not isinstance(entry_point, str) or not entry_point.strip():
            report.fail(
                f"transport={transport!r} requires a non-empty 'entry_point'"
            )
        elif not _ENTRY_POINT_PATTERN.match(entry_point):
            report.fail(
                f"entry_point {entry_point!r} does not match "
                f"{_ENTRY_POINT_PATTERN.pattern}"
            )
        else:
            report.ok(f"transport={transport} entry_point={entry_point}")


def _check_permissions(data: dict[str, Any], report: _Report) -> None:
    """Validate the permissions block."""

    perms = data.get("permissions")
    if not isinstance(perms, dict):
        report.fail("permissions must be an object")
        return

    workspace_access = perms.get("workspace_access")
    if workspace_access is True:
        report.fail(
            "permissions.workspace_access must be false — provider "
            "adapters never need workspace access"
        )
    elif workspace_access is not False:
        report.fail(
            "permissions.workspace_access must be explicitly false"
        )

    shell_execute = perms.get("shell_execute")
    if shell_execute is True:
        report.fail(
            "permissions.shell_execute must be false — provider "
            "adapters are not permitted to spawn processes"
        )
    elif shell_execute is not False:
        report.fail(
            "permissions.shell_execute must be explicitly false"
        )

    network = perms.get("network")
    if network not in _ALLOWED_NETWORK:
        report.fail(
            f"permissions.network {network!r} not in "
            f"{sorted(_ALLOWED_NETWORK)}"
        )

    secrets = perms.get("secrets")
    secrets_ok = False
    if not isinstance(secrets, list):
        report.fail("permissions.secrets must be a list")
    else:
        secrets_ok = True
        for entry in secrets:
            if not isinstance(entry, str):
                report.fail(
                    f"permissions.secrets entry {entry!r} must be a string"
                )
                secrets_ok = False
            elif entry not in _ALLOWED_SECRETS:
                report.fail(
                    f"permissions.secrets entry {entry!r} not in "
                    f"{sorted(_ALLOWED_SECRETS)}"
                )
                secrets_ok = False

    if (
        workspace_access is False
        and shell_execute is False
        and network in _ALLOWED_NETWORK
        and secrets_ok
    ):
        report.ok(
            f"permissions: network={network}, "
            f"secrets={len(secrets) if isinstance(secrets, list) else 'n/a'}, "
            "workspace_access=false, shell_execute=false"
        )


def _check_connection_fields(data: dict[str, Any], report: _Report) -> None:
    """Lightweight check of connection_fields keys."""

    fields = data.get("connection_fields")
    if fields is None:
        report.ok("connection_fields: implicit (base_url + api_key)")
        return
    if not isinstance(fields, list):
        report.fail("connection_fields must be a list")
        return

    seen_keys: set[str] = set()
    for index, field in enumerate(fields):
        if not isinstance(field, dict):
            report.fail(f"connection_fields[{index}] must be an object")
            continue
        key = field.get("key")
        if not isinstance(key, str) or not key.strip():
            report.fail(
                f"connection_fields[{index}].key must be a non-empty string"
            )
            continue
        if key in seen_keys:
            report.fail(
                f"connection_fields has duplicate key: {key!r}"
            )
        seen_keys.add(key)
        if key.startswith("_peldrun_"):
            report.fail(
                f"connection_fields[{index}].key {key!r} uses the "
                f"reserved prefix '_peldrun_'"
            )
        if not key.replace("_", "").isalnum() or key[0].isdigit():
            report.warn(
                f"connection_fields[{index}].key {key!r} is not a "
                f"valid Python identifier; the adapter cannot read it "
                f"via context.extra_params[{key!r}]"
            )

    report.ok(f"connection_fields: {len(fields)} field(s)")


def _check_features(data: dict[str, Any], report: _Report) -> None:
    """Ensure features is an object and warn on unknown keys."""

    features = data.get("features")
    if features is None:
        report.ok("features: all defaults (false)")
        return
    if not isinstance(features, dict):
        report.fail("features must be an object")
        return

    unknown = set(features.keys()) - _KNOWN_FEATURE_FLAGS
    if unknown:
        report.warn(
            f"features contains unknown keys: {sorted(unknown)}. "
            "This may indicate a typo or a future feature flag."
        )

    enabled = [k for k, v in features.items() if v is True]
    report.ok(
        f"features: {len(enabled)} enabled "
        f"({', '.join(enabled) or 'none'})"
    )


def _check_dependencies(data: dict[str, Any], report: _Report) -> None:
    """Verify dependencies is a list of non-empty strings."""

    deps = data.get("dependencies")
    if deps is None:
        report.ok("dependencies: not declared (empty)")
        return
    if not isinstance(deps, list):
        report.fail("dependencies must be a list")
        return

    for index, entry in enumerate(deps):
        if not isinstance(entry, str) or not entry.strip():
            report.fail(
                f"dependencies[{index}] must be a non-empty string"
            )

    report.ok(f"dependencies: {len(deps)} entr(y/ies)")


def _resolve_entry_point_file(
    data: dict[str, Any], report: _Report
) -> None:
    """
    Verify the entry_point module path resolves to a real .py file.

    Looks in both the pre-build tree (``src/``) and the post-build tree
    (repo root, mirroring the ZIP layout) so it can validate either.
    """

    transport = data.get("transport")
    if transport not in ("native", "custom"):
        return

    entry_point = data.get("entry_point")
    if not isinstance(entry_point, str) or ":" not in entry_point:
        return  # structural check already failed

    module_path, _class_name = entry_point.split(":", 1)
    rel = module_path.replace(".", "/")

    candidates = [
        _SRC_ROOT / f"{rel}.py",
        _SRC_ROOT / rel / "__init__.py",
        _REPO_ROOT / f"{rel}.py",
        _REPO_ROOT / rel / "__init__.py",
    ]

    for candidate in candidates:
        if candidate.is_file():
            try:
                shown = candidate.relative_to(_REPO_ROOT)
            except ValueError:
                shown = candidate
            report.ok(f"entry_point file resolves: {shown}")
            return

    rendered: list[str] = []
    for candidate in candidates:
        try:
            rendered.append(str(candidate.relative_to(_REPO_ROOT)))
        except ValueError:
            rendered.append(str(candidate))
    report.fail(
        f"entry_point {entry_point!r} does not resolve to a .py file. "
        f"Checked: {rendered}"
    )


def _check_peldrun_schema(
    manifest: dict[str, Any], report: _Report
) -> None:
    """
    If PELDRUN is importable in the current environment, validate the
    manifest against the real ``ProviderManifest`` Pydantic model for
    full schema coverage.

    Silently skipped when PELDRUN is not available (bare CI).
    """

    ProviderManifest: Any = None
    try:
        from omweb.provider_packages.schema import ProviderManifest  # type: ignore
    except Exception:
        try:
            from peldrun.omweb.provider_packages.schema import (  # type: ignore
                ProviderManifest,
            )
        except Exception:
            report.warn(
                "peldrun not importable; "
                "skipping ProviderManifest schema validation"
            )
            return

    try:
        ProviderManifest.model_validate(manifest)
    except Exception as exc:
        errors = getattr(exc, "errors", None)
        if callable(errors):
            for err in errors():
                loc = ".".join(str(p) for p in err.get("loc", []))
                msg = err.get("msg", str(err))
                report.fail(f"ProviderManifest[{loc}]: {msg}")
        else:
            report.fail(f"ProviderManifest validation failed: {exc}")
    else:
        report.ok("ProviderManifest validation (peldrun schema)")


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def validate(manifest_path: Path) -> int:
    print(f"Validating {manifest_path}")
    print()

    if not manifest_path.is_file():
        print(f"error: file not found: {manifest_path}", file=sys.stderr)
        return 2

    try:
        raw = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"error: cannot read file: {exc}", file=sys.stderr)
        return 2

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON: {exc}", file=sys.stderr)
        return 2

    if not isinstance(data, dict):
        print(
            f"error: manifest must be a JSON object, "
            f"got {type(data).__name__}",
            file=sys.stderr,
        )
        return 2

    report = _Report()

    _check_identity(data, report)
    _check_transport(data, report)
    _check_permissions(data, report)
    _check_connection_fields(data, report)
    _check_features(data, report)
    _check_dependencies(data, report)
    _resolve_entry_point_file(data, report)
    _check_peldrun_schema(data, report)

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
    return 0


def main(argv: list[str]) -> int:
    manifest_path = Path(argv[0]) if argv else _DEFAULT_MANIFEST
    return validate(manifest_path)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))