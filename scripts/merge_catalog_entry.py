#!/usr/bin/env python3
# scripts/merge_catalog_entry.py
"""
Merge a generated catalog-entry.json into the PELDRUN provider catalog.

The catalog file (``catalog.json``) is a versioned artifact published
by the PELDRUN team. It is a JSON document with this shape:

    {
      "schema_version": 1,
      "description": "...",
      "updated_at": "...",
      "items": [ { <item> }, ... ]
    }

This script:

  1. Reads a generated catalog entry (``dist/catalog-entry.json``).
  2. Reads an existing catalog file (``catalog.json``).
  3. Finds the item with the same ``id``:
        * If found, replaces it in place (idempotent update).
        * If not found, appends it to the ``items`` array.
  4. Bumps ``updated_at`` to the current UTC timestamp.
  5. Writes the result back atomically.

The script never deletes other items and never reorders them — an
existing catalog's item order is preserved exactly, so diffs stay
minimal and reviewable.

Usage
-----
    python scripts/merge_catalog_entry.py \\
        --catalog catalog.json \\
        --entry dist/catalog-entry.json \\
        --output catalog.json

    # Or omit --output to write in place:
    python scripts/merge_catalog_entry.py \\
        --catalog catalog.json \\
        --entry dist/catalog-entry.json

Exit codes
----------
    0 — merge succeeded
    1 — input files missing or malformed
    2 — output could not be written
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_json(path: Path, *, label: str) -> Any:
    """Load JSON from ``path``, raising SystemExit with a clear message."""
    if not path.is_file():
        raise SystemExit(f"error: {label} not found at {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"error: {label} is not valid JSON: {exc}")
    except OSError as exc:
        raise SystemExit(f"error: could not read {label}: {exc}")


def _validate_catalog(data: Any, *, path: Path) -> Dict[str, Any]:
    """Validate the top-level catalog shape."""
    if not isinstance(data, dict):
        raise SystemExit(
            f"error: catalog at {path} is not a JSON object."
        )
    if "items" not in data or not isinstance(data["items"], list):
        raise SystemExit(
            f"error: catalog at {path} has no 'items' array."
        )
    return data


def _validate_entry(data: Any, *, path: Path) -> Dict[str, Any]:
    """Validate the shape of a single catalog entry."""
    if not isinstance(data, dict):
        raise SystemExit(
            f"error: entry at {path} is not a JSON object."
        )
    if not isinstance(data.get("id"), str) or not data["id"]:
        raise SystemExit(
            f"error: entry at {path} is missing a valid 'id'."
        )
    return data


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def merge_entry(
    catalog: Dict[str, Any],
    entry: Dict[str, Any],
) -> bool:
    """
    Merge ``entry`` into ``catalog["items"]``.

    Returns True if an existing item was replaced, False if the entry
    was appended.
    """
    entry_id = entry["id"]
    items: List[Any] = catalog["items"]

    for index, existing in enumerate(items):
        if isinstance(existing, dict) and existing.get("id") == entry_id:
            items[index] = entry
            return True

    items.append(entry)
    return False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_atomic(path: Path, data: Any) -> None:
    """Write ``data`` to ``path`` atomically (temp + replace)."""
    serialized = json.dumps(
        data, indent=2, ensure_ascii=False, sort_keys=False
    ) + "\n"
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
            "Merge a generated catalog entry into a PELDRUN provider "
            "catalog."
        ),
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        required=True,
        help="Path to the catalog.json file to update.",
    )
    parser.add_argument(
        "--entry",
        type=Path,
        required=True,
        help="Path to the catalog-entry.json to merge.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path. Defaults to --catalog (in-place update). "
            "Use a separate path to preview before committing."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print what would change without writing anything. "
            "Useful for CI validation."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = _parse_args(argv)

    catalog_path = Path(args.catalog).resolve()
    entry_path = Path(args.entry).resolve()

    catalog = _validate_catalog(
        _load_json(catalog_path, label="catalog"),
        path=catalog_path,
    )
    entry = _validate_entry(
        _load_json(entry_path, label="catalog entry"),
        path=entry_path,
    )

    replaced = merge_entry(catalog, entry)
    catalog["updated_at"] = _now_iso()

    action = "replaced" if replaced else "added"
    print(
        f"Catalog entry {entry['id']!r} {action}. "
        f"Total items: {len(catalog['items'])}."
    )

    if args.dry_run:
        print("Dry run — no files were written.")
        return 0

    output_path = Path(args.output).resolve() if args.output else catalog_path
    _write_atomic(output_path, catalog)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))