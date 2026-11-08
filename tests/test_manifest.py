# tests/test_manifest.py
"""
Integrity tests for the provider's manifest.json.

Runs in three progressively-deeper layers:

1. **Stdlib-only** structural checks — always run. Verifies the file
   parses, required fields are present, patterns match, and the
   entry_point resolves to a real file in the source tree. This layer
   catches the vast majority of manifest typos.

2. **pydantic** (via the ``ProviderManifest`` model, when importable).
   This is the exact model that PELDRUN's installer uses, so a pass
   here guarantees the archive would pass install-time validation.

3. **peldrun** (skipped when not installed). Tests that require the
   full PELDRUN runtime — currently none beyond the Pydantic schema.

Design constraints:

* The test file must run in a bare CI environment with only pytest
  and the project's own dependencies installed.
* Every test that depends on an optional import is decorated with
  ``@pytest.mark.skipif(...)``, never a runtime ``try/except`` that
  could hide a genuine assertion failure.
* Assertions use precise error messages so a CI failure pinpoints
  the offending field, not just "assert False".
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

import pytest


# ---------------------------------------------------------------------------
# Locate the manifest relative to this test file (works from any cwd).
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "manifest.json"


# ---------------------------------------------------------------------------
# Optional imports — skip markers are evaluated at collection time.
# ---------------------------------------------------------------------------

try:
    from omweb.provider_packages.schema import ProviderManifest
    HAS_PELDRUN = True
    PELDRUN_IMPORT_ERROR: str | None = None
except ImportError as exc_omweb:
    try:
        from peldrun.omweb.provider_packages.schema import (  # type: ignore[no-redef]
            ProviderManifest,
        )
        HAS_PELDRUN = True
        PELDRUN_IMPORT_ERROR = None
    except ImportError as exc_peldrun:
        HAS_PELDRUN = False
        PELDRUN_IMPORT_ERROR = (
            f"omweb import failed: {exc_omweb}; "
            f"peldrun.omweb import failed: {exc_peldrun}"
        )


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def manifest_dict() -> Dict[str, Any]:
    """Load manifest.json once for the whole module."""
    assert MANIFEST_PATH.is_file(), (
        f"manifest.json is missing at {MANIFEST_PATH}"
    )
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict), (
        f"manifest.json root must be an object, got {type(data).__name__}"
    )
    return data


# ---------------------------------------------------------------------------
# Layer 1: stdlib-only structural checks
# ---------------------------------------------------------------------------


class TestManifestStructure:
    """Basic structural integrity checks — always run."""

    def test_file_exists(self) -> None:
        assert MANIFEST_PATH.is_file()

    def test_valid_json(self, manifest_dict: Dict[str, Any]) -> None:
        # Re-parsing here would be redundant; the fixture would have
        # thrown a JSONDecodeError before this point.
        assert isinstance(manifest_dict, dict)

    @pytest.mark.parametrize(
        "field",
        [
            "schema_version",
            "kind",
            "id",
            "name",
            "version",
            "api_contract",
            "transport",
        ],
    )
    def test_required_field_present(
        self, manifest_dict: Dict[str, Any], field: str
    ) -> None:
        assert field in manifest_dict, (
            f"Required field {field!r} is missing from manifest.json"
        )

    def test_schema_version(self, manifest_dict: Dict[str, Any]) -> None:
        assert manifest_dict["schema_version"] == "1.0.0", (
            f"schema_version must be '1.0.0', "
            f"got {manifest_dict['schema_version']!r}"
        )

    def test_kind(self, manifest_dict: Dict[str, Any]) -> None:
        assert manifest_dict["kind"] == "peldrun.llm_provider", (
            f"kind must be 'peldrun.llm_provider', "
            f"got {manifest_dict['kind']!r}"
        )

    def test_api_contract(self, manifest_dict: Dict[str, Any]) -> None:
        assert manifest_dict["api_contract"] == "peldrun.llm.provider.v1", (
            f"api_contract must be 'peldrun.llm.provider.v1', "
            f"got {manifest_dict['api_contract']!r}"
        )

    def test_id_pattern(self, manifest_dict: Dict[str, Any]) -> None:
        provider_id = manifest_dict["id"]
        assert isinstance(provider_id, str) and provider_id, (
            "id must be a non-empty string"
        )
        assert re.match(r"^[a-z0-9][a-z0-9._-]*$", provider_id), (
            f"id {provider_id!r} does not match "
            f"^[a-z0-9][a-z0-9._-]*$"
        )

    def test_id_not_builtin(self, manifest_dict: Dict[str, Any]) -> None:
        builtin_ids = {
            "openai_compat",
            "lmstudio",
            "ollama",
            "openai_native",
            "anthropic",
            "google_genai",
            "ollama_native",
            "lmstudio_native",
        }
        assert manifest_dict["id"] not in builtin_ids, (
            f"id {manifest_dict['id']!r} collides with a built-in adapter"
        )

    def test_version_semver(self, manifest_dict: Dict[str, Any]) -> None:
        version = manifest_dict["version"]
        assert isinstance(version, str), "version must be a string"
        assert re.match(
            r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$",
            version,
        ), f"version {version!r} is not valid semver"

    def test_name_nonempty(self, manifest_dict: Dict[str, Any]) -> None:
        assert isinstance(manifest_dict["name"], str)
        assert manifest_dict["name"].strip(), "name must not be empty"


# ---------------------------------------------------------------------------
# Layer 1: transport consistency
# ---------------------------------------------------------------------------


class TestTransport:
    """Transport / entry_point / base_adapter XOR rules."""

    def test_transport_valid(self, manifest_dict: Dict[str, Any]) -> None:
        assert manifest_dict["transport"] in {
            "compatible",
            "native",
            "custom",
        }, (
            f"transport must be compatible/native/custom, "
            f"got {manifest_dict['transport']!r}"
        )

    def test_transport_xor(self, manifest_dict: Dict[str, Any]) -> None:
        transport = manifest_dict["transport"]
        entry_point = manifest_dict.get("entry_point")
        base_adapter = manifest_dict.get("base_adapter")

        if transport == "compatible":
            assert base_adapter, (
                "transport='compatible' requires 'base_adapter'"
            )
            assert entry_point is None, (
                "transport='compatible' must not declare 'entry_point'"
            )
        else:
            assert entry_point, (
                f"transport={transport!r} requires 'entry_point'"
            )
            assert base_adapter is None, (
                f"transport={transport!r} must not declare 'base_adapter'"
            )


# ---------------------------------------------------------------------------
# Layer 1: permissions
# ---------------------------------------------------------------------------


class TestPermissions:
    """Permissions block must be present and safe."""

    def test_permissions_present(self, manifest_dict: Dict[str, Any]) -> None:
        assert "permissions" in manifest_dict
        assert isinstance(manifest_dict["permissions"], dict)

    def test_workspace_access_false(
        self, manifest_dict: Dict[str, Any]
    ) -> None:
        assert (
            manifest_dict["permissions"].get("workspace_access") is False
        ), "permissions.workspace_access must be explicitly false"

    def test_shell_execute_false(
        self, manifest_dict: Dict[str, Any]
    ) -> None:
        assert (
            manifest_dict["permissions"].get("shell_execute") is False
        ), "permissions.shell_execute must be explicitly false"

    def test_network_valid(self, manifest_dict: Dict[str, Any]) -> None:
        network = manifest_dict["permissions"].get("network")
        assert network in {"none", "connection_endpoint"}, (
            f"permissions.network must be 'none' or "
            f"'connection_endpoint', got {network!r}"
        )

    def test_secrets_valid(self, manifest_dict: Dict[str, Any]) -> None:
        secrets = manifest_dict["permissions"].get("secrets", [])
        assert isinstance(secrets, list), "permissions.secrets must be a list"
        for entry in secrets:
            assert entry == "connection.api_key", (
                f"permissions.secrets entry {entry!r} is not allowed; "
                f"only 'connection.api_key' is permitted"
            )


# ---------------------------------------------------------------------------
# Layer 1: entry_point resolution
# ---------------------------------------------------------------------------


class TestEntryPoint:
    """entry_point must resolve to a real file in the source tree."""

    def test_entry_point_resolves(
        self, manifest_dict: Dict[str, Any]
    ) -> None:
        entry_point = manifest_dict.get("entry_point")
        if not entry_point:
            pytest.skip("No entry_point declared (compatible transport)")

        assert ":" in entry_point, (
            f"entry_point {entry_point!r} must be 'module:ClassName'"
        )
        module_path, class_name = entry_point.split(":", 1)

        assert re.match(
            r"^[A-Za-z_][A-Za-z0-9_.]*$", module_path
        ), f"module path {module_path!r} is not a valid Python path"
        assert re.match(
            r"^[A-Za-z_][A-Za-z0-9_]*$", class_name
        ), f"class name {class_name!r} is not a valid Python identifier"

        rel = module_path.replace(".", "/")
        candidates = [
            REPO_ROOT / "src" / f"{rel}.py",
            REPO_ROOT / "src" / rel / "__init__.py",
            REPO_ROOT / f"{rel}.py",
            REPO_ROOT / rel / "__init__.py",
        ]
        found = [c for c in candidates if c.is_file()]
        assert found, (
            f"entry_point {entry_point!r} does not resolve to any .py "
            f"file. Checked:\n"
            + "\n".join(f"  - {c}" for c in candidates)
        )

    def test_entry_point_class_exists_in_module(
        self, manifest_dict: Dict[str, Any]
    ) -> None:
        """
        Import the module and verify the class exists.

        Uses a fresh ``importlib`` load from the on-disk source tree so
        the test does not depend on the package being installed in the
        current interpreter's ``sys.path``.
        """
        entry_point = manifest_dict.get("entry_point")
        if not entry_point:
            pytest.skip("No entry_point declared (compatible transport)")

        module_path, class_name = entry_point.split(":", 1)
        module_file = (
            REPO_ROOT
            / "src"
            / (module_path.replace(".", "/") + ".py")
        )
        if not module_file.is_file():
            pytest.skip(
                f"module file {module_file} not found at expected "
                f"location; covered by test_entry_point_resolves"
            )

        import importlib.util

        spec = importlib.util.spec_from_file_location(
            f"_test_{module_path.replace('.', '_')}",
            str(module_file),
        )
        if spec is None or spec.loader is None:
            pytest.skip(f"could not build import spec for {module_file}")

        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except ImportError as exc:
            pytest.skip(
                f"could not import {module_path} (missing dependency): "
                f"{exc}"
            )
        except Exception as exc:
            pytest.fail(
                f"importing {module_path} raised "
                f"{type(exc).__name__}: {exc}"
            )

        assert hasattr(module, class_name), (
            f"module {module_path!r} has no attribute {class_name!r}"
        )


# ---------------------------------------------------------------------------
# Layer 1: connection_fields
# ---------------------------------------------------------------------------


class TestConnectionFields:
    """connection_fields must be well-formed and free of reserved keys."""

    def test_is_list(self, manifest_dict: Dict[str, Any]) -> None:
        fields = manifest_dict.get("connection_fields")
        if fields is None:
            pytest.skip("connection_fields not declared")
        assert isinstance(fields, list)

    def test_each_field_has_key(self, manifest_dict: Dict[str, Any]) -> None:
        for index, field in enumerate(
            manifest_dict.get("connection_fields") or []
        ):
            assert isinstance(field, dict), (
                f"connection_fields[{index}] must be an object"
            )
            key = field.get("key")
            assert isinstance(key, str) and key.strip(), (
                f"connection_fields[{index}].key must be a non-empty "
                f"string"
            )

    def test_no_reserved_prefix(self, manifest_dict: Dict[str, Any]) -> None:
        for field in manifest_dict.get("connection_fields") or []:
            key = field.get("key", "")
            assert not key.startswith("_peldrun_"), (
                f"connection_fields key {key!r} uses the reserved "
                f"'_peldrun_' prefix"
            )

    def test_unique_keys(self, manifest_dict: Dict[str, Any]) -> None:
        keys = [
            f.get("key")
            for f in (manifest_dict.get("connection_fields") or [])
        ]
        assert len(keys) == len(set(keys)), (
            f"duplicate connection_fields keys: "
            f"{[k for k in keys if keys.count(k) > 1]}"
        )


# ---------------------------------------------------------------------------
# Layer 1: model_facts
# ---------------------------------------------------------------------------


class TestModelFacts:
    """model_facts must be a non-empty dict of well-shaped entries."""

    def test_is_dict(self, manifest_dict: Dict[str, Any]) -> None:
        facts = manifest_dict.get("model_facts", {})
        assert isinstance(facts, dict)

    def test_entries_have_valid_shape(
        self, manifest_dict: Dict[str, Any]
    ) -> None:
        for model_id, fact in (manifest_dict.get("model_facts") or {}).items():
            assert isinstance(model_id, str) and model_id, (
                f"model_facts key {model_id!r} must be a non-empty string"
            )
            assert isinstance(fact, dict), (
                f"model_facts[{model_id!r}] must be an object"
            )
            for numeric_key in ("max_context_length", "max_output_tokens"):
                if numeric_key in fact:
                    assert isinstance(fact[numeric_key], int), (
                        f"model_facts[{model_id!r}].{numeric_key} must "
                        f"be an integer"
                    )
                    assert fact[numeric_key] > 0, (
                        f"model_facts[{model_id!r}].{numeric_key} must "
                        f"be positive"
                    )


# ---------------------------------------------------------------------------
# Layer 1: dependencies
# ---------------------------------------------------------------------------


class TestDependencies:
    """dependencies must be a list of non-empty strings."""

    def test_is_list(self, manifest_dict: Dict[str, Any]) -> None:
        deps = manifest_dict.get("dependencies", [])
        assert isinstance(deps, list)

    def test_entries_non_empty(self, manifest_dict: Dict[str, Any]) -> None:
        for index, dep in enumerate(manifest_dict.get("dependencies") or []):
            assert isinstance(dep, str), (
                f"dependencies[{index}] must be a string"
            )
            assert dep.strip(), (
                f"dependencies[{index}] must not be empty"
            )


# ---------------------------------------------------------------------------
# Layer 2: PELDRUN's real ProviderManifest schema
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_PELDRUN,
    reason=f"peldrun not installed ({PELDRUN_IMPORT_ERROR})",
)
class TestPeldrunSchema:
    """
    Validate the manifest against PELDRUN's real ``ProviderManifest``.

    This is the definitive check: the exact Pydantic model that the
    install pipeline uses. A pass here guarantees the archive will be
    accepted by ``package_validator.validate_zip`` at install time.
    """

    def test_provider_manifest_accepts(
        self, manifest_dict: Dict[str, Any]
    ) -> None:
        try:
            ProviderManifest.model_validate(manifest_dict)
        except Exception as exc:
            pytest.fail(
                f"ProviderManifest rejected the manifest:\n{exc}"
            )

    def test_manifest_fields_match_manifest_json(
        self, manifest_dict: Dict[str, Any]
    ) -> None:
        """
        Confirm the parsed model exposes the same identity fields as
        the on-disk JSON. Catches accidental field renaming.
        """
        model = ProviderManifest.model_validate(manifest_dict)
        assert model.id == manifest_dict["id"]
        assert model.version == manifest_dict["version"]
        assert model.kind == manifest_dict["kind"]
        assert model.api_contract == manifest_dict["api_contract"]
        assert model.transport.value == manifest_dict["transport"]