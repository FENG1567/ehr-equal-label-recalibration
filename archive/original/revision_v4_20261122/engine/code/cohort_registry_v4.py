"""Versioned cohort path registry for v4 source data.

Legacy cohorts keep their historical processed paths unless a revision-owned
JSON registry explicitly overrides one or more sources. New cohort IDs must
provide their own sequence and split paths; they never fall back silently to a
legacy cohort's files.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
PROC = ROOT / "data/processed"
DEFAULT_REGISTRY_PATH = REV / "data_registry.json"
DEFAULT_COHORTS = (
    "mimic", "eicu", "nwicu", "sicdb", "hirid", "c2012", "zigong",
)
PATH_FIELDS = ("sequence_path", "split_path", "tabular_path", "vocab_path")
COHORT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


class CohortRegistryError(ValueError):
    """Raised when a cohort path registry is malformed or incomplete."""


@dataclass(frozen=True)
class CohortPaths:
    sequence_path: Path
    split_path: Path
    tabular_path: Path | None
    vocab_path: Path | None

    def as_metadata(self) -> dict[str, str | None]:
        return {
            "sequence_path": str(self.sequence_path),
            "split_path": str(self.split_path),
            "tabular_path": str(self.tabular_path) if self.tabular_path else None,
            "vocab_path": str(self.vocab_path) if self.vocab_path else None,
        }


def _registry_entries(registry_path: Path | None = None) -> dict[str, dict[str, Any]]:
    path = DEFAULT_REGISTRY_PATH if registry_path is None else Path(registry_path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CohortRegistryError(f"cannot read cohort registry {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CohortRegistryError("cohort registry root must be a JSON object")
    entries = payload.get("cohorts", payload)
    if not isinstance(entries, dict):
        raise CohortRegistryError("cohort registry 'cohorts' entry must be a JSON object")
    result: dict[str, dict[str, Any]] = {}
    for cohort, entry in entries.items():
        if not isinstance(cohort, str) or not COHORT_RE.fullmatch(cohort):
            raise CohortRegistryError(f"invalid cohort ID in registry: {cohort!r}")
        if not isinstance(entry, dict):
            raise CohortRegistryError(f"{cohort}: registry entry must be a JSON object")
        unknown_paths = set(entry) & set(PATH_FIELDS)
        if any(value is not None and not isinstance(value, str) for value in
               (entry.get(field) for field in unknown_paths)):
            raise CohortRegistryError(f"{cohort}: configured paths must be strings")
        result[cohort] = entry
    return result


def registered_cohorts(registry_path: Path | None = None) -> tuple[str, ...]:
    """Return legacy and explicitly registered cohort IDs in stable order."""
    entries = _registry_entries(registry_path)
    return tuple(sorted(set(DEFAULT_COHORTS) | set(entries)))


def _default_path(cohort: str, field: str) -> Path | None:
    if cohort not in DEFAULT_COHORTS:
        return None
    filename = {
        "sequence_path": f"seq_{cohort}.npz",
        "split_path": f"split_{cohort}.npy",
        "tabular_path": f"tab_{cohort}.parquet",
        "vocab_path": f"meta_{cohort}.json",
    }[field]
    return PROC / filename


def _resolve_path(value: str | None, default: Path | None,
                  registry_path: Path | None) -> Path | None:
    if value is None:
        return default
    configured = Path(value).expanduser()
    if not configured.is_absolute():
        base = (DEFAULT_REGISTRY_PATH if registry_path is None
                else Path(registry_path)).parent
        configured = base / configured
    return configured.resolve()


def resolve_cohort_paths(db: str, registry_path: Path | None = None,
                         *, verify_exists: bool = False) -> CohortPaths:
    """Resolve actual sequence, split, raw target, and vocabulary files.

    Relative registry paths are relative to the registry file's directory.
    Legacy paths are only used for the seven original cohort IDs.
    """
    entries = _registry_entries(registry_path)
    if db not in entries and db not in DEFAULT_COHORTS:
        raise CohortRegistryError(
            f"{db}: cohort is not registered; refusing legacy-path fallback")
    entry = entries.get(db, {})
    values: dict[str, Path | None] = {}
    for field in PATH_FIELDS:
        default = _default_path(db, field)
        configured = entry.get(field)
        values[field] = _resolve_path(configured, default, registry_path)
    for required in ("sequence_path", "split_path"):
        if values[required] is None:
            raise CohortRegistryError(
                f"{db}: registry must define {required} for a new cohort")
    paths = CohortPaths(
        sequence_path=values["sequence_path"],  # type: ignore[arg-type]
        split_path=values["split_path"],  # type: ignore[arg-type]
        tabular_path=values["tabular_path"],
        vocab_path=values["vocab_path"],
    )
    if verify_exists:
        missing = [str(path) for path in (
            paths.sequence_path, paths.split_path, paths.tabular_path, paths.vocab_path
        ) if path is not None and not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{db}: registered source files are missing: {missing}")
    return paths
