"""Repaired identity-bound numeric decoding for the v4 EHR-FM revision.

The module implements two distinct estimands:

* ``source_transfer`` reuses the frozen MIMIC split-1 alpha, refits Ridge on
  MIMIC split 0, checks its MIMIC split-2 point metrics against the previous
  aggregate, and evaluates the fixed model on eICU semantic v4.2.
* ``target_local_nested_cv`` tunes and evaluates a decoder within each target
  cohort using five outer and three inner patient-group folds. Target labels
  are therefore used for model selection and fitting in this separate task.

Only aggregate results and provenance are written. Patient-level predictions
and identifiers stay in memory on the analysis server.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
ENGINE = REV / "engine"
DEFAULT_MANIFEST = ENGINE / "code_snapshots/embedding_manifest_recovered_v4_2_merged41_b2eec1f7aca1.json"
DEFAULT_ELIGIBILITY = (
    REV / "temporal/results/eligibility/recovered_reviewed_v4_2"
)
RESULTS = REV / "decoding_repaired/results"
SOURCE_REFERENCE_SUMMARY = (
    REV / "decoding/results/primary_status_consistent_20261123/numeric_decoding_summary.csv"
)

SOURCE_DB = "mimic"
TARGET_DBS = ("eicu_semantic_v4_2",)
VARS = (
    "creatinine", "lactate", "wbc", "hgb", "glucose", "hr", "sbp",
    "spo2", "bun", "platelet",
)
ALPHAS = (1.0, 10.0, 100.0, 1000.0)
OUTER_FOLDS = 5
INNER_FOLDS = 3
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20261122
EXPECTED_REPAIRED_EMBEDDING_MANIFEST_SHA256 = (
    "b2eec1f7aca1857bda440ddb5ec48e477b2a20f753d5dea28b7715f7a1a243fe"
)
EXPECTED_REPAIRED_EMBEDDING_MODEL_COUNT = 41
EXPECTED_REPAIRED_ELIGIBILITY_MANIFEST_SHA256 = (
    "d302ce0a493121f9336f40fd58c5dc63a75a663492293c0153b4ba05efbb8e8f"
)
EXPECTED_SOURCE_REFERENCE_SHA256 = (
    "0ff14db563f0832740cb38c2b4722182ef11b3b2c583031e83ebac33c8014cae"
)
EXPECTED_REPAIRED_COHORTS = {
    "mimic": {
        "n_eligible": 73936,
        "eligibility_table_sha256": "c2774cd59e4df8706960e4fa58680ea6ec72b4af22adddf82b2389dc5b74749f",
    },
    "eicu_semantic_v4_2": {
        "n_eligible": 130544,
        "eligibility_table_sha256": "fb14b4a31c924b123923df0ed76d1c3c87c8495b230fa5c8584b67a974b180a8",
    },
}
CORE_ANALYSIS_IDS = (
    "main20k_real",
    "main20k_constant",
    "abl_const20k_constant",
)


class DecoderInputError(ValueError):
    """Raised when an input is stale, ambiguous, or fails identity checks."""


@dataclass(frozen=True)
class EmbeddingCache:
    db: str
    analysis_id: str
    manifest_entry: dict[str, Any]
    rows: np.ndarray
    stay_id: np.ndarray
    subject: np.ndarray
    embedding: np.ndarray
    metadata: dict[str, Any]
    cache_sha256: str


@dataclass(frozen=True)
class AlignedPopulation:
    """Same stays and target availability aligned across all requested inputs."""

    stay_id: np.ndarray
    subject: np.ndarray
    rows: np.ndarray
    embedding_by_analysis: dict[str, np.ndarray]
    targets_by_variable: dict[str, np.ndarray]
    split: np.ndarray
    n_eligible_stays: int
    cache_rows_by_analysis: dict[str, int]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ordered_identity(db: str, rows: np.ndarray,
                            stay_id: np.ndarray, subject: np.ndarray) -> str:
    """Match the identity digest used by ``v4_engine.py``."""
    digest = hashlib.sha256(str(db).encode("utf-8"))
    digest.update(np.asarray(rows, dtype=np.int64).tobytes())
    for values in (np.asarray(stay_id).astype(str), np.asarray(subject).astype(str)):
        for value in values:
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    return digest.hexdigest()


def _valid_identifier(values: np.ndarray, name: str, db: str) -> None:
    bad = {"", "nan", "none", "null", "-1"}
    text = np.asarray(values).astype(str)
    if np.isin(np.char.lower(text), list(bad)).any():
        raise DecoderInputError(f"{db}: invalid {name} in embedding cache")


def _metadata_projection(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Fields expected to be duplicated in the cache's metadata_json."""
    fields = (
        "db", "model_tag", "input_condition", "checkpoint_sha256",
        "input_mask_sha256", "row_identity_sha256", "source_npz_sha256",
        "split_sha256", "encoder_code_sha256", "runtime_identity",
        "runtime_identity_sha256", "encoding_scope", "input_mask_spec",
        "condition_definition_sha256",
    )
    return {key: entry[key] for key in fields if key in entry}


def load_embedding_cache(entry: Mapping[str, Any], db: str,
                         analysis_id: str) -> EmbeddingCache:
    """Load and validate one cache using its current embedding-manifest row."""
    path_value = entry.get("cache")
    if not path_value:
        raise DecoderInputError(f"{db}/{analysis_id}: manifest row has no cache path")
    path = Path(str(path_value))
    if not path.is_file():
        raise FileNotFoundError(f"{db}/{analysis_id}: cache is missing: {path}")

    required_manifest = (
        "db", "model_tag", "input_condition", "n_rows", "checkpoint_sha256",
        "input_mask_sha256", "row_identity_sha256", "source_npz_sha256",
        "split_sha256", "encoder_code_sha256", "runtime_identity",
        "runtime_identity_sha256", "encoding_scope",
    )
    missing_manifest = [key for key in required_manifest if entry.get(key) in (None, "")]
    if missing_manifest:
        raise DecoderInputError(
            f"{db}/{analysis_id}: embedding manifest row lacks {missing_manifest}")
    if str(entry["db"]) != db:
        raise DecoderInputError(f"{db}/{analysis_id}: manifest db identity mismatch")
    expected_id = f"{entry['model_tag']}_{entry['input_condition']}"
    if expected_id != analysis_id:
        raise DecoderInputError(f"{db}/{analysis_id}: manifest analysis identity is {expected_id}")

    with np.load(path, allow_pickle=False) as archive:
        required_arrays = {"rows", "stay_id", "subject", "source_y", "embedding", "metadata_json"}
        if not required_arrays.issubset(set(archive.files)):
            raise DecoderInputError(
                f"{db}/{analysis_id}: cache lacks arrays {sorted(required_arrays-set(archive.files))}")
        rows = np.asarray(archive["rows"], dtype=np.int64)
        stay_id = archive["stay_id"].astype(str)
        subject = archive["subject"].astype(str)
        embedding = np.asarray(archive["embedding"], dtype=np.float32)
        metadata = json.loads(str(archive["metadata_json"].item()))
        # source_y is deliberately never read: it is the mortality label, not a
        # numeric target. Numeric labels come only from the joined tabular file.

    n_rows = int(entry["n_rows"])
    if (rows.ndim != 1 or stay_id.ndim != 1 or subject.ndim != 1
            or len(rows) != n_rows or len(stay_id) != n_rows or len(subject) != n_rows
            or embedding.ndim != 2 or embedding.shape[0] != n_rows
            or embedding.shape[1] < 1):
        raise DecoderInputError(f"{db}/{analysis_id}: cache array dimensions do not match manifest")
    if len(np.unique(rows)) != n_rows or np.any(rows < 0):
        raise DecoderInputError(f"{db}/{analysis_id}: cache source rows are duplicated or negative")
    if len(np.unique(stay_id)) != n_rows:
        raise DecoderInputError(f"{db}/{analysis_id}: cache stay_id values are not unique")
    _valid_identifier(stay_id, "stay_id", db)
    _valid_identifier(subject, "subject", db)
    if not np.isfinite(embedding).all():
        raise DecoderInputError(f"{db}/{analysis_id}: embedding cache contains nonfinite values")

    actual_identity = sha256_ordered_identity(db, rows, stay_id, subject)
    if actual_identity != str(entry["row_identity_sha256"]):
        raise DecoderInputError(f"{db}/{analysis_id}: ordered row identity differs from manifest")
    if not isinstance(metadata, dict):
        raise DecoderInputError(f"{db}/{analysis_id}: cache metadata_json is not an object")
    expected_metadata = _metadata_projection(entry)
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise DecoderInputError(f"{db}/{analysis_id}: cache metadata differs from manifest row")

    return EmbeddingCache(
        db=db, analysis_id=analysis_id, manifest_entry=dict(entry), rows=rows,
        stay_id=stay_id, subject=subject, embedding=embedding,
        metadata=metadata, cache_sha256=sha256_file(path),
    )


def load_embedding_manifest(path: str | Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Return the manifest and one unique row per ``db/analysis_id``."""
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("mode") != "encode" or not isinstance(manifest.get("models"), list):
        raise DecoderInputError(f"not a v4 encoding manifest: {manifest_path}")
    entries: dict[str, dict[str, Any]] = {}
    for row in manifest["models"]:
        if not isinstance(row, dict):
            raise DecoderInputError("embedding manifest contains a non-object model row")
        db = str(row.get("db", ""))
        model = str(row.get("model_tag", ""))
        condition = str(row.get("input_condition", ""))
        if not db or not model or not condition:
            raise DecoderInputError("embedding manifest row has an incomplete cache identity")
        key = f"{db}/{model}_{condition}"
        if key in entries:
            raise DecoderInputError(f"duplicate embedding manifest row: {key}")
        entries[key] = row
    return manifest, entries


def validate_checkpoint_hashes(entries: Mapping[str, Mapping[str, Any]],
                               checkpoint_paths: Mapping[str, Path]) -> dict[str, str]:
    """Verify each cache checkpoint against its current frozen model file."""
    expected_by_model: dict[str, set[str]] = {}
    for row in entries.values():
        model_tag = str(row["model_tag"])
        expected_by_model.setdefault(model_tag, set()).add(str(row.get("checkpoint_sha256", "")))
    actual_by_model: dict[str, str] = {}
    for model_tag, expected_values in expected_by_model.items():
        if len(expected_values) != 1:
            raise DecoderInputError(f"{model_tag}: manifest cache rows disagree on checkpoint hash")
        expected = next(iter(expected_values))
        if len(expected) != 64 or any(char not in "0123456789abcdefABCDEF" for char in expected):
            raise DecoderInputError(f"{model_tag}: checkpoint SHA256 is malformed")
        path = checkpoint_paths.get(model_tag)
        if path is None or not Path(path).is_file():
            raise FileNotFoundError(f"{model_tag}: frozen checkpoint is missing: {path}")
        actual = sha256_file(path)
        if actual.lower() != expected.lower():
            raise DecoderInputError(f"{model_tag}: current checkpoint SHA256 differs from cache manifest")
        actual_by_model[model_tag] = actual
    return actual_by_model


def load_sequence_identity(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read ordered stay, patient, and mortality identity arrays from a source NPZ."""
    with np.load(path, allow_pickle=False) as source:
        required = {"stay_id", "subject", "y_mort"}
        if not required.issubset(set(source.files)):
            raise DecoderInputError(f"sequence NPZ lacks identity arrays: {sorted(required-set(source.files))}")
        stay_id = source["stay_id"].astype(str)
        subject = source["subject"].astype(str)
        y_mort = np.asarray(source["y_mort"], dtype=float)
    if (stay_id.ndim != 1 or subject.ndim != 1 or y_mort.ndim != 1
            or len(stay_id) != len(subject) or len(stay_id) != len(y_mort)):
        raise DecoderInputError("sequence NPZ identity/outcome arrays do not align")
    if len(np.unique(stay_id)) != len(stay_id):
        raise DecoderInputError("sequence NPZ stay_id values are not unique")
    _valid_identifier(stay_id, "stay_id", Path(path).name)
    _valid_identifier(subject, "subject", Path(path).name)
    return stay_id, subject, y_mort


def verify_cache_sequence_identity(cache: EmbeddingCache,
                                   sequence_stay_id: np.ndarray,
                                   sequence_subject: np.ndarray) -> None:
    """Confirm every cached identity maps to the same row in the source NPZ."""
    if cache.rows.size and int(cache.rows.max()) >= len(sequence_stay_id):
        raise DecoderInputError(f"{cache.db}/{cache.analysis_id}: cache row exceeds sequence NPZ")
    if len(sequence_stay_id) != len(sequence_subject):
        raise DecoderInputError(f"{cache.db}: sequence NPZ identity arrays have different lengths")
    if (not np.array_equal(sequence_stay_id[cache.rows], cache.stay_id)
            or not np.array_equal(sequence_subject[cache.rows], cache.subject)):
        raise DecoderInputError(
            f"{cache.db}/{cache.analysis_id}: cache IDs do not match source NPZ rows")


def normalize_stay_id(value: Any) -> str:
    """Normalize numeric integral Parquet IDs without changing string IDs."""
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return str(value)


def align_tabular_frame(frame: pd.DataFrame, stay_ids: Sequence[str],
                        variables: Sequence[str] = VARS) -> pd.DataFrame:
    """Join target columns to cache identity in the cache's requested order."""
    needed = {"stay_id", *variables}
    missing = sorted(needed - set(frame.columns))
    if missing:
        raise DecoderInputError(f"numeric target table lacks columns {missing}")
    table = frame.loc[:, ["stay_id", *variables]].copy()
    table["stay_id"] = table["stay_id"].map(normalize_stay_id)
    if table["stay_id"].eq("").any():
        raise DecoderInputError("numeric target table contains missing stay_id values")
    if table["stay_id"].duplicated().any():
        raise DecoderInputError("numeric target table has duplicate stay_id values")
    ids = np.asarray(stay_ids).astype(str)
    if len(np.unique(ids)) != len(ids):
        raise DecoderInputError("requested cache stay_id values are not unique")
    indexed = table.set_index("stay_id", drop=False)
    absent = ~pd.Index(ids).isin(indexed.index)
    if absent.any():
        raise DecoderInputError(
            f"numeric target table lacks {int(absent.sum())} embedding-cache stay_id values")
    aligned = indexed.reindex(ids).reset_index(drop=True)
    if not np.array_equal(aligned["stay_id"].to_numpy(dtype=str), ids):
        raise DecoderInputError("stay_id join did not preserve embedding-cache order")
    for variable in variables:
        aligned[variable] = pd.to_numeric(aligned[variable], errors="coerce")
    return aligned


def build_common_population(caches: Mapping[str, EmbeddingCache],
                            target_frame: pd.DataFrame,
                            split: np.ndarray,
                            eligibility_frame: pd.DataFrame | None = None,
                            variables: Sequence[str] = VARS) -> AlignedPopulation:
    """Intersect cache identities and join numeric labels on ``stay_id``.

    The resulting target availability and rows are therefore identical across
    all requested conditions. Source mortality labels are not consulted.
    """
    if not caches:
        raise DecoderInputError("at least one cache is required")
    analysis_ids = list(caches)
    first = caches[analysis_ids[0]]
    all_ids = [set(cache.stay_id.astype(str)) for cache in caches.values()]
    shared = set.intersection(*all_ids)
    eligibility_subject_by_id: dict[str, str] = {}
    n_eligible_stays = len(shared)
    if eligibility_frame is not None:
        needed = {"stay_id", "subject"}
        missing = needed - set(eligibility_frame.columns)
        if missing:
            raise DecoderInputError(
                f"{first.db}: current eligibility rows lack {sorted(missing)}")
        eligible_ids = eligibility_frame["stay_id"].map(normalize_stay_id).to_numpy(dtype=str)
        eligible_subjects = eligibility_frame["subject"].astype(str).to_numpy()
        if len(np.unique(eligible_ids)) != len(eligible_ids):
            raise DecoderInputError(f"{first.db}: current eligibility has duplicate stay_id values")
        _valid_identifier(eligible_ids, "eligible stay_id", first.db)
        _valid_identifier(eligible_subjects, "eligible subject", first.db)
        eligibility_subject_by_id = dict(zip(eligible_ids, eligible_subjects))
        n_eligible_stays = len(eligible_ids)
        shared.intersection_update(eligibility_subject_by_id)
    common_ids = np.asarray(sorted(shared), dtype=str)
    if not len(common_ids):
        raise DecoderInputError(f"{first.db}: requested conditions have no shared stay_id values")
    aligned_targets = align_tabular_frame(target_frame, common_ids, variables)
    target_by_id = aligned_targets.set_index("stay_id", drop=False)

    row_lookup: dict[str, tuple[int, str]] = {}
    embedding_by_analysis: dict[str, np.ndarray] = {}
    for analysis_id, cache in caches.items():
        positions = {stay: idx for idx, stay in enumerate(cache.stay_id.astype(str))}
        index = np.asarray([positions[stay] for stay in common_ids], dtype=np.int64)
        cache_rows = cache.rows[index]
        cache_subjects = cache.subject[index]
        if analysis_id == analysis_ids[0]:
            row_lookup = {
                stay: (int(row), str(subject))
                for stay, row, subject in zip(common_ids, cache_rows, cache_subjects)
            }
        else:
            for stay, row, subject in zip(common_ids, cache_rows, cache_subjects):
                if row_lookup[stay] != (int(row), str(subject)):
                    raise DecoderInputError(
                        f"{first.db}: condition caches disagree on row/subject identity for stay_id={stay}")
        if eligibility_subject_by_id:
            mismatched_subjects = [
                stay for stay, subject in zip(common_ids, cache_subjects)
                if eligibility_subject_by_id[stay] != str(subject)
            ]
            if mismatched_subjects:
                raise DecoderInputError(
                    f"{first.db}/{analysis_id}: eligibility subject identity mismatch for "
                    f"{len(mismatched_subjects)} stays")
        embedding_by_analysis[analysis_id] = cache.embedding[index]

    rows = np.asarray([row_lookup[stay][0] for stay in common_ids], dtype=np.int64)
    subjects = np.asarray([row_lookup[stay][1] for stay in common_ids], dtype=str)
    if rows.size and (rows.min() < 0 or rows.max() >= len(split)):
        raise DecoderInputError(f"{first.db}: cache source row index exceeds current split array")
    target_by_variable: dict[str, np.ndarray] = {}
    for variable in variables:
        target_by_variable[variable] = pd.to_numeric(
            target_by_id.loc[common_ids, variable], errors="coerce").to_numpy(dtype=float)
    return AlignedPopulation(
        stay_id=common_ids, subject=subjects, rows=rows,
        embedding_by_analysis=embedding_by_analysis,
        targets_by_variable=target_by_variable,
        split=np.asarray(split)[rows],
        n_eligible_stays=int(n_eligible_stays),
        cache_rows_by_analysis={analysis_id: len(caches[analysis_id].rows)
                                for analysis_id in analysis_ids},
    )


def make_group_splits(groups: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray(groups).astype(str)
    n_groups = len(np.unique(groups))
    if n_groups < n_splits:
        raise DecoderInputError(f"need at least {n_splits} distinct patients; found {n_groups}")
    folds = list(GroupKFold(n_splits=n_splits).split(np.zeros((len(groups), 1)), groups=groups))
    for train, test in folds:
        if set(groups[train]).intersection(groups[test]):
            raise AssertionError("patient overlap across GroupKFold split")
    return folds


def _ridge_pipeline() -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if len(y) == 0 or len(prediction) != len(y):
        raise DecoderInputError("metric input arrays must be nonempty and have matching lengths")
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise DecoderInputError("metric arrays contain nonfinite values")
    r2 = (float(r2_score(y, prediction, force_finite=False)) if len(y) >= 2
          else float("nan"))
    return {
        "r2": r2,
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "mae": float(mean_absolute_error(y, prediction)),
    }


def paired_bootstrap_seed(db: str, variable: str, mode: str,
                          seed: int = BOOTSTRAP_SEED) -> int:
    """Derive the same patient-resample seed for all requested conditions."""
    payload = f"{int(seed)}\0{db}\0{variable}\0{mode}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def patient_cluster_bootstrap_metrics(
    y: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Percentile intervals from fixed predictions and patient-cluster draws.

    Per-patient sufficient statistics keep memory proportional to the number
    of patients; no 1000-by-stay weight matrix is built. Reusing the same seed
    and group order across conditions gives paired patient resamples.
    """
    if int(n_boot) < BOOTSTRAP_REPLICATES:
        raise DecoderInputError(f"patient decoder intervals require at least {BOOTSTRAP_REPLICATES} draws")
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    groups = np.asarray(groups).astype(str)
    if (len(y) == 0 or len(prediction) != len(y) or len(groups) != len(y)
            or not np.isfinite(y).all() or not np.isfinite(prediction).all()):
        raise DecoderInputError("bootstrap y, prediction, and patient arrays must align and be finite")
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        return {
            "r2_lo": float("nan"), "r2_hi": float("nan"),
            "rmse_lo": float("nan"), "rmse_hi": float("nan"),
            "mae_lo": float("nan"), "mae_hi": float("nan"),
            "bootstrap_valid": 0, "bootstrap_replicates": int(n_boot),
            "bootstrap_cluster_unit": "patient",
        }

    count_by_patient = np.bincount(inverse).astype(np.float64)
    sum_y_by_patient = np.bincount(inverse, weights=y)
    sum_y2_by_patient = np.bincount(inverse, weights=y * y)
    squared_error = np.square(y - prediction)
    absolute_error = np.abs(y - prediction)
    sum_sqerr_by_patient = np.bincount(inverse, weights=squared_error)
    sum_abserr_by_patient = np.bincount(inverse, weights=absolute_error)

    rng = np.random.default_rng(int(seed))
    r2_values: list[float] = []
    rmse_values: list[float] = []
    mae_values: list[float] = []
    n_patients = len(unique)
    for _ in range(int(n_boot)):
        sampled = rng.integers(0, n_patients, size=n_patients)
        n_rows = float(count_by_patient[sampled].sum())
        if n_rows <= 0:
            continue
        sum_y = float(sum_y_by_patient[sampled].sum())
        sum_y2 = float(sum_y2_by_patient[sampled].sum())
        sum_sqerr = float(sum_sqerr_by_patient[sampled].sum())
        sum_abserr = float(sum_abserr_by_patient[sampled].sum())
        mean_y = sum_y / n_rows
        total_sum_squares = sum_y2 - 2.0 * mean_y * sum_y + mean_y * mean_y * n_rows
        rmse_values.append(float(np.sqrt(sum_sqerr / n_rows)))
        mae_values.append(float(sum_abserr / n_rows))
        if total_sum_squares > 0:
            r2_values.append(float(1.0 - sum_sqerr / total_sum_squares))

    def interval(values: list[float]) -> tuple[float, float]:
        if not values:
            return float("nan"), float("nan")
        lower, upper = np.percentile(np.asarray(values, dtype=float), [2.5, 97.5])
        return float(lower), float(upper)

    r2_lo, r2_hi = interval(r2_values)
    rmse_lo, rmse_hi = interval(rmse_values)
    mae_lo, mae_hi = interval(mae_values)
    return {
        "r2_lo": r2_lo, "r2_hi": r2_hi,
        "rmse_lo": rmse_lo, "rmse_hi": rmse_hi,
        "mae_lo": mae_lo, "mae_hi": mae_hi,
        "bootstrap_valid": int(len(rmse_values)),
        "bootstrap_valid_r2": int(len(r2_values)),
        "bootstrap_replicates": int(n_boot),
        "bootstrap_cluster_unit": "patient",
        "bootstrap_seed": int(seed),
    }


def fit_source_ridge(X_train: np.ndarray, y_train: np.ndarray,
                     groups_train: np.ndarray, X_validation: np.ndarray,
                     y_validation: np.ndarray, groups_validation: np.ndarray,
                     alphas: Sequence[float] = ALPHAS) -> tuple[Pipeline, dict[str, Any]]:
    """Select alpha on source split 1, then fit scaler and Ridge on split 0."""
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=float)
    groups_train = np.asarray(groups_train).astype(str)
    X_validation = np.asarray(X_validation, dtype=np.float32)
    y_validation = np.asarray(y_validation, dtype=float)
    groups_validation = np.asarray(groups_validation).astype(str)
    if (X_train.ndim != 2 or X_validation.ndim != 2
            or X_train.shape[1] != X_validation.shape[1]
            or len(X_train) != len(y_train) or len(y_train) != len(groups_train)
            or len(X_validation) != len(y_validation)
            or len(y_validation) != len(groups_validation)):
        raise DecoderInputError("source train/validation X, y, and group dimensions do not align")
    if not len(y_train) or not len(y_validation):
        raise DecoderInputError("source training and validation rows must both be nonempty")
    if not np.isfinite(X_train).all() or not np.isfinite(y_train).all():
        raise DecoderInputError("source training data contain nonfinite values")
    if not np.isfinite(X_validation).all() or not np.isfinite(y_validation).all():
        raise DecoderInputError("source validation data contain nonfinite values")
    if set(groups_train).intersection(groups_validation):
        raise DecoderInputError("source split 0 and split 1 contain overlapping patients")

    validation_scaler = StandardScaler().fit(X_train)
    X_train_scaled = validation_scaler.transform(X_train)
    X_validation_scaled = validation_scaler.transform(X_validation)
    validation_mse: dict[float, float] = {}
    for alpha in alphas:
        candidate = Ridge(alpha=float(alpha)).fit(X_train_scaled, y_train)
        prediction = candidate.predict(X_validation_scaled)
        validation_mse[float(alpha)] = float(mean_squared_error(y_validation, prediction))
    # Preserve the declared alpha order for deterministic ties.
    selected_alpha = min((float(alpha) for alpha in alphas),
                         key=lambda alpha: (validation_mse[alpha],
                                            list(map(float, alphas)).index(alpha)))
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=selected_alpha))])
    model.fit(X_train, y_train)
    metadata = {
        "alpha": selected_alpha,
        "alpha_validation_mse": validation_mse[selected_alpha],
        "alpha_validation_mse_by_alpha": validation_mse,
        "n_source_train_rows": int(len(y_train)),
        "n_source_train_patients": int(len(np.unique(groups_train))),
        "n_source_validation_rows": int(len(y_validation)),
        "n_source_validation_patients": int(len(np.unique(groups_validation))),
    }
    return model, metadata


def fit_source_ridge_fixed_alpha(X_train: np.ndarray, y_train: np.ndarray,
                                 groups_train: np.ndarray,
                                 alpha: float) -> tuple[Pipeline, dict[str, Any]]:
    """Refit scaler and Ridge on source split 0 using an already frozen alpha."""
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=float)
    groups_train = np.asarray(groups_train).astype(str)
    alpha = float(alpha)
    if alpha not in tuple(map(float, ALPHAS)):
        raise DecoderInputError(f"frozen source alpha {alpha!r} is outside {list(ALPHAS)}")
    if (X_train.ndim != 2 or len(X_train) != len(y_train)
            or len(y_train) != len(groups_train)):
        raise DecoderInputError("fixed-alpha source train X, y, and group dimensions do not align")
    if not len(y_train):
        raise DecoderInputError("fixed-alpha source training data must be nonempty")
    if not np.isfinite(X_train).all() or not np.isfinite(y_train).all():
        raise DecoderInputError("fixed-alpha source training data contain nonfinite values")
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
    model.fit(X_train, y_train)
    return model, {
        "alpha": alpha,
        "n_source_train_rows": int(len(y_train)),
        "n_source_train_patients": int(len(np.unique(groups_train))),
    }


def grouped_nested_ridge(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                         alphas: Sequence[float] = ALPHAS,
                         n_outer: int = OUTER_FOLDS,
                         n_inner: int = INNER_FOLDS) -> tuple[np.ndarray, dict[str, Any]]:
    """Return one OOF prediction per row from patient-group nested Ridge CV."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups).astype(str)
    if X.ndim != 2 or len(X) != len(y) or len(y) != len(groups):
        raise DecoderInputError("nested X/y/group dimensions do not align")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise DecoderInputError("nested CV data contain nonfinite values")
    outer = make_group_splits(groups, n_outer)
    oof = np.full(len(y), np.nan, dtype=float)
    selected_alphas: list[float] = []
    for train, test in outer:
        inner_groups = groups[train]
        inner = make_group_splits(inner_groups, n_inner)
        search = GridSearchCV(
            _ridge_pipeline(), {"ridge__alpha": list(alphas)},
            scoring="neg_mean_squared_error", cv=inner, n_jobs=1,
            refit=True, error_score="raise",
        )
        search.fit(X[train], y[train])
        oof[test] = search.predict(X[test])
        selected_alphas.append(float(search.best_params_["ridge__alpha"]))
    if not np.isfinite(oof).all():
        raise AssertionError("nested grouped CV did not assign one finite OOF prediction per row")
    summary = _metrics(y, oof)
    summary.update({
        "alpha_median": float(np.median(selected_alphas)),
        "alpha_counts": dict(Counter(selected_alphas)),
        "n_outer_folds": int(n_outer),
        "n_inner_folds": int(n_inner),
        "n_patients": int(len(np.unique(groups))),
    })
    return oof, summary


def _resolve_cohort_paths(db: str,
                          resolver: Callable[[str], Any] | None = None
                          ) -> tuple[Path, Path, Path]:
    if resolver is None:
        engine_code_dir = Path(__file__).resolve().parents[1] / "engine" / "code"
        if str(engine_code_dir) not in sys.path:
            sys.path.insert(0, str(engine_code_dir))
        try:
            from cohort_registry_v4 import resolve_cohort_paths as resolver  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "cohort_registry_v4.resolve_cohort_paths(db) is required by numeric decoding") from exc
    resolved = resolver(db)
    tabular_path = getattr(resolved, "tabular_path", None)
    if tabular_path is None:
        raise DecoderInputError("resolve_cohort_paths(db) must expose .tabular_path")
    sequence_path = getattr(resolved, "sequence_path", None)
    if sequence_path is None:
        raise DecoderInputError("resolve_cohort_paths(db) must expose .sequence_path")
    tabular_path = Path(tabular_path)
    split_path = getattr(resolved, "split_path", None)
    if split_path is None:
        raise DecoderInputError("resolve_cohort_paths(db) must expose .split_path")
    return tabular_path, Path(split_path), Path(sequence_path)


def _load_ready_eligibility(db: str, eligibility_dir: Path,
                            loader: Callable[[str, Path], tuple[pd.DataFrame, dict[str, Any]]] | None
                            ) -> tuple[pd.DataFrame, dict[str, Any]]:
    if loader is None:
        engine_code_dir = Path(__file__).resolve().parents[1] / "engine" / "code"
        if str(engine_code_dir) not in sys.path:
            sys.path.insert(0, str(engine_code_dir))
        try:
            from v4_engine import load_ready_eligibility as loader  # type: ignore
        except ImportError as exc:
            raise RuntimeError("v4_engine.load_ready_eligibility is required for decoding") from exc
    return loader(db, eligibility_dir)


def verify_current_eligibility(
    db: str,
    eligibility_dir: str | Path,
    eligible: pd.DataFrame,
    entry: Mapping[str, Any],
    sequence_sha256: str,
    sequence_stay_id: np.ndarray,
    sequence_subject: np.ndarray,
    sequence_y_mort: np.ndarray,
    *,
    expected_manifest_sha256: str | None = EXPECTED_REPAIRED_ELIGIBILITY_MANIFEST_SHA256,
    expected_endpoint: str = "hospital",
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Bind current eligibility scope, files, endpoint, labels, and patient keys."""
    eligibility_dir = Path(eligibility_dir)
    manifest_path = eligibility_dir / "eligibility_manifest.json"
    table_path = eligibility_dir / f"{db}.parquet"
    if not manifest_path.is_file() or not table_path.is_file():
        raise FileNotFoundError(f"{db}: current eligibility manifest/table is missing")
    manifest_hash = sha256_file(manifest_path)
    table_hash = sha256_file(table_path)
    if entry.get("eligibility_manifest_sha256") != manifest_hash:
        raise DecoderInputError(f"{db}: loader eligibility-manifest hash is stale")
    if entry.get("eligibility_table_sha256") != table_hash:
        raise DecoderInputError(f"{db}: loader eligibility-table hash is stale")
    if expected_manifest_sha256 is not None and manifest_hash.lower() != expected_manifest_sha256.lower():
        raise DecoderInputError("eligibility manifest SHA256 is not the root-cleared primary manifest")
    if entry.get("retrospective_landmark_prediction_allowed") is not True:
        raise DecoderInputError(f"{db}: current eligibility entry does not authorize landmark analysis")

    current_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current_entry = current_manifest.get("cohorts", {}).get(db)
    if not isinstance(current_entry, dict):
        raise DecoderInputError(f"{db}: current eligibility manifest has no cohort entry")
    for field in ("retrospective_landmark_prediction_allowed", "evidence", "status", "sourcehashes"):
        if entry.get(field) != current_entry.get(field):
            raise DecoderInputError(f"{db}: loader entry differs from current manifest field {field}")
    if current_entry.get("retrospective_landmark_prediction_allowed") is not True:
        raise DecoderInputError(f"{db}: current manifest entry blocks landmark analysis")
    evidence = str(current_entry.get("evidence", "")).lower()
    if evidence not in {"exact", "operational_proxy"}:
        raise DecoderInputError(f"{db}: unsupported current eligibility evidence {evidence!r}")
    if str(current_entry.get("status", "")).lower() in {"unavailable", "not_ready", "blocked", "failed"}:
        raise DecoderInputError(f"{db}: current eligibility status blocks analysis")
    sourcehashes = current_entry.get("sourcehashes", {})
    attested_sequence_hash = sourcehashes.get("sequence_npz") if isinstance(sourcehashes, dict) else None
    if not attested_sequence_hash or str(attested_sequence_hash).lower() != sequence_sha256.lower():
        raise DecoderInputError(f"{db}: eligibility manifest source NPZ hash differs from current sequence")
    if str(entry.get("sequence_npz_sha256", "")).lower() != sequence_sha256.lower():
        raise DecoderInputError(f"{db}: loader sequence NPZ hash differs from current sequence")

    required = {"stay_id", "subject", "y_mort", "endpoint"}
    missing = required - set(eligible.columns)
    if missing:
        raise DecoderInputError(f"{db}: loaded eligibility rows lack {sorted(missing)}")
    frame = eligible.copy()
    if "db" in frame and frame["db"].astype(str).ne(db).any():
        raise DecoderInputError(f"{db}: loaded eligibility rows include another cohort")
    frame["stay_id"] = frame["stay_id"].map(normalize_stay_id)
    frame["subject"] = frame["subject"].astype(str)
    if frame["stay_id"].eq("").any() or frame["subject"].eq("").any():
        raise DecoderInputError(f"{db}: loaded eligibility rows contain empty IDs")
    if frame["stay_id"].duplicated().any():
        raise DecoderInputError(f"{db}: loaded eligibility rows contain duplicate stays")
    if frame["endpoint"].astype(str).nunique() != 1 or str(frame["endpoint"].iloc[0]) != expected_endpoint:
        raise DecoderInputError(f"{db}: current endpoint differs from requested {expected_endpoint!r}")
    audited_y = pd.to_numeric(frame["y_mort"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(audited_y).all() or not np.isin(audited_y, (0.0, 1.0)).all():
        raise DecoderInputError(f"{db}: current eligibility mortality labels are not binary")
    pos_by_stay = {str(stay): i for i, stay in enumerate(sequence_stay_id.astype(str))}
    missing_ids = [stay for stay in frame["stay_id"] if stay not in pos_by_stay]
    if missing_ids:
        raise DecoderInputError(f"{db}: {len(missing_ids)} eligible stays are absent from source NPZ")
    positions = np.asarray([pos_by_stay[stay] for stay in frame["stay_id"]], dtype=np.int64)
    observed_subjects = sequence_subject[positions].astype(str)
    observed_y = np.asarray(sequence_y_mort[positions], dtype=float)
    if not np.array_equal(observed_subjects, frame["subject"].to_numpy(dtype=str)):
        raise DecoderInputError(f"{db}: current eligibility patient IDs differ from source NPZ mapping")
    if not np.array_equal(observed_y, audited_y):
        raise DecoderInputError(f"{db}: current eligibility mortality labels differ from source NPZ")

    expected_cohort = EXPECTED_REPAIRED_COHORTS.get(db) if expected_manifest_sha256 is not None else None
    if expected_cohort is not None:
        if table_hash.lower() != expected_cohort["eligibility_table_sha256"].lower():
            raise DecoderInputError(f"{db}: eligibility table SHA256 differs from root clearance")
        if len(frame) != int(expected_cohort["n_eligible"]):
            raise DecoderInputError(f"{db}: eligible row count differs from root clearance")
    return frame, {
        "eligibility_manifest_sha256": manifest_hash,
        "eligibility_table_sha256": table_hash,
        "eligibility_evidence": evidence,
        "eligibility_status": str(current_entry.get("status", "")),
    }


def _read_target_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() != ".parquet":
        raise DecoderInputError(f"numeric target source must be Parquet: {path}")
    return pd.read_parquet(path)


def _load_checked_split(path: Path, expected_hash: str, db: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"{db}: split file is missing: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != str(expected_hash):
        raise DecoderInputError(f"{db}: split file SHA256 differs from embedding manifest")
    split = np.load(path, allow_pickle=False)
    if split.ndim != 1 or not np.issubdtype(split.dtype, np.integer):
        raise DecoderInputError(f"{db}: split array must be a one-dimensional integer vector")
    return split


def _population_hash(population: AlignedPopulation, mask: np.ndarray,
                     db: str) -> str:
    selected = np.asarray(mask, dtype=bool)
    return sha256_ordered_identity(
        db, population.rows[selected], population.stay_id[selected],
        population.subject[selected],
    )


def validate_source_patient_partitions(population: AlignedPopulation,
                                      source_db: str) -> None:
    """Reject patient leakage across source development, validation, and test."""
    groups_by_split = {
        value: set(population.subject[population.split == value].astype(str))
        for value in (0, 1, 2)
    }
    for left, right in ((0, 1), (0, 2), (1, 2)):
        overlap = groups_by_split[left].intersection(groups_by_split[right])
        if overlap:
            raise DecoderInputError(
                f"{source_db}: patient overlap across split {left} and split {right}")


def _result_row(*, analysis_id: str, mode: str, source_db: str, target_db: str,
                variable: str, population: AlignedPopulation, mask: np.ndarray,
                target_usage: str, status: str, metrics: Mapping[str, Any] | None = None,
                **extra: Any) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    labels = population.targets_by_variable[variable]
    output = {
        "analysis_id": analysis_id,
        "mode": mode,
        "source_db": source_db,
        "target_db": target_db,
        "variable": variable,
        "target_label_usage": target_usage,
        "n_eligible_stays": int(population.n_eligible_stays),
        "n_common_condition_stays": int(len(population.stay_id)),
        "cache_rows_by_condition_json": json.dumps(
            population.cache_rows_by_analysis, sort_keys=True),
        "n_target_available": int(np.isfinite(labels).sum()),
        "n_rows": int(selected.sum()),
        "n_patients": int(len(np.unique(population.subject[selected]))),
        "population_identity_sha256": _population_hash(population, selected, target_db),
        "status": status,
    }
    output.update(dict(metrics or {}))
    output.update(extra)
    return output


def load_frozen_source_reference(
    summary_path: str | Path,
    analysis_ids: Sequence[str] = CORE_ANALYSIS_IDS,
    *,
    expected_sha256: str | None = EXPECTED_SOURCE_REFERENCE_SHA256,
) -> tuple[dict[tuple[str, str], dict[str, Any]], str]:
    """Load old MIMIC source-transfer rows as frozen alpha/metric provenance."""
    path = Path(summary_path)
    if not path.is_file():
        raise FileNotFoundError(f"frozen source-reference summary is missing: {path}")
    actual_hash = sha256_file(path)
    if expected_sha256 is not None and actual_hash.lower() != expected_sha256.lower():
        raise DecoderInputError("frozen MIMIC source-reference summary SHA256 differs from clearance")
    frame = pd.read_csv(path)
    required = {
        "analysis_id", "mode", "source_db", "target_db", "variable", "status",
        "alpha", "source_alpha_validation_mse", "source_alpha_validation_mse_by_alpha_json",
        "n_source_train_rows", "n_source_train_patients",
        "n_source_validation_rows", "n_source_validation_patients", "n_source_test_rows",
        "source_train_population_identity_sha256",
        "source_validation_population_identity_sha256", "population_identity_sha256",
        "r2", "rmse", "mae",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DecoderInputError(f"frozen source-reference summary lacks columns {missing}")
    selected = frame.loc[
        frame["mode"].eq("source_transfer")
        & frame["source_db"].eq(SOURCE_DB)
        & frame["target_db"].eq(SOURCE_DB)
        & frame["analysis_id"].isin(list(analysis_ids))
        & frame["variable"].isin(VARS)
    ].copy()
    if selected.duplicated(["analysis_id", "variable"]).any():
        raise DecoderInputError("frozen source-reference summary has duplicate MIMIC source rows")
    expected_keys = {(str(analysis_id), variable)
                     for analysis_id in analysis_ids for variable in VARS}
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in selected.iterrows():
        key = (str(row["analysis_id"]), str(row["variable"]))
        rows[key] = row.to_dict()
    if set(rows) != expected_keys:
        missing_keys = sorted(expected_keys - set(rows))
        extra_keys = sorted(set(rows) - expected_keys)
        raise DecoderInputError(
            f"frozen source-reference rows do not match requested analysis: "
            f"missing={missing_keys}, extra={extra_keys}")
    for key, row in rows.items():
        if str(row["status"]) not in {"ok", "ok_r2_undefined"}:
            raise DecoderInputError(f"{key}: prior MIMIC source metric status is not successful")
        alpha = pd.to_numeric(pd.Series([row["alpha"]]), errors="coerce").iloc[0]
        if not np.isfinite(alpha) or float(alpha) not in tuple(map(float, ALPHAS)):
            raise DecoderInputError(f"{key}: prior MIMIC alpha is missing or outside {list(ALPHAS)}")
        if not np.isfinite(pd.to_numeric(
                pd.Series([row["source_alpha_validation_mse"]]), errors="coerce").iloc[0]):
            raise DecoderInputError(f"{key}: prior MIMIC split-1 alpha-selection provenance is missing")
    return rows, actual_hash


def validate_source_reference_identities(
    population: AlignedPopulation,
    reference_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    analysis_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Check split counts and ordered stay/patient identities against old MIMIC rows."""
    records: list[dict[str, Any]] = []
    for analysis_id in analysis_ids:
        for variable in VARS:
            reference = reference_rows[(analysis_id, variable)]
            y = population.targets_by_variable[variable]
            masks = {
                "train": (population.split == 0) & np.isfinite(y),
                "validation": (population.split == 1) & np.isfinite(y),
                "test": (population.split == 2) & np.isfinite(y),
            }
            observed = {
                "n_source_train_rows": int(masks["train"].sum()),
                "n_source_train_patients": int(len(np.unique(population.subject[masks["train"]]))),
                "n_source_validation_rows": int(masks["validation"].sum()),
                "n_source_validation_patients": int(len(np.unique(population.subject[masks["validation"]]))),
                "n_source_test_rows": int(masks["test"].sum()),
                "source_train_population_identity_sha256": _population_hash(
                    population, masks["train"], SOURCE_DB),
                "source_validation_population_identity_sha256": _population_hash(
                    population, masks["validation"], SOURCE_DB),
                "population_identity_sha256": _population_hash(
                    population, masks["test"], SOURCE_DB),
            }
            count_keys = (
                "n_source_train_rows", "n_source_train_patients",
                "n_source_validation_rows", "n_source_validation_patients", "n_source_test_rows",
            )
            for field in count_keys:
                expected = pd.to_numeric(pd.Series([reference[field]]), errors="coerce").iloc[0]
                if not np.isfinite(expected) or int(expected) != observed[field]:
                    raise DecoderInputError(
                        f"{analysis_id}/{variable}: MIMIC {field} differs from frozen reference")
            hash_keys = (
                "source_train_population_identity_sha256",
                "source_validation_population_identity_sha256", "population_identity_sha256",
            )
            for field in hash_keys:
                if str(reference[field]).lower() != observed[field].lower():
                    raise DecoderInputError(
                        f"{analysis_id}/{variable}: MIMIC {field} differs from frozen reference")
            records.append({"analysis_id": analysis_id, "variable": variable, **observed})
    return records


def compare_source_test_metrics(
    observed: Mapping[str, float],
    reference: Mapping[str, Any],
    *,
    analysis_id: str,
    variable: str,
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> dict[str, Any]:
    """Reject a refit that fails to reproduce prior MIMIC split-2 point metrics."""
    record: dict[str, Any] = {"analysis_id": analysis_id, "variable": variable}
    for metric in ("r2", "rmse", "mae"):
        expected = pd.to_numeric(pd.Series([reference[metric]]), errors="coerce").iloc[0]
        actual = float(observed[metric])
        if not np.isfinite(expected) or not np.isclose(
                actual, float(expected), rtol=rtol, atol=atol):
            raise DecoderInputError(
                f"{analysis_id}/{variable}: refit MIMIC split-2 {metric} differs from frozen reference")
        record[f"{metric}_observed"] = actual
        record[f"{metric}_reference"] = float(expected)
        record[f"{metric}_absolute_difference"] = abs(actual - float(expected))
    record["alpha"] = float(reference["alpha"])
    record["status"] = "matched"
    return record


def run_decoding(manifest_path: str | Path = DEFAULT_MANIFEST,
                 analysis_ids: Sequence[str] = CORE_ANALYSIS_IDS,
                 target_dbs: Sequence[str] = TARGET_DBS,
                 source_db: str = SOURCE_DB,
                 resolver: Callable[[str], Any] | None = None,
                 target_reader: Callable[[Path], pd.DataFrame] = _read_target_frame,
                 checkpoint_resolver: Callable[[str], Path] | None = None,
                 eligibility_dir: str | Path = DEFAULT_ELIGIBILITY,
                 eligibility_loader: Callable[[str, Path], tuple[pd.DataFrame, dict[str, Any]]] | None = None,
                 source_reference_path: str | Path = SOURCE_REFERENCE_SUMMARY,
                 expected_embedding_manifest_sha256: str | None = EXPECTED_REPAIRED_EMBEDDING_MANIFEST_SHA256,
                 expected_eligibility_manifest_sha256: str | None = EXPECTED_REPAIRED_ELIGIBILITY_MANIFEST_SHA256,
                 expected_source_reference_sha256: str | None = EXPECTED_SOURCE_REFERENCE_SHA256,
                 endpoint: str = "hospital",
                 n_bootstrap: int = BOOTSTRAP_REPLICATES,
                 seed: int = BOOTSTRAP_SEED,
                 preflight_only: bool = False,
                 output_dir: str | Path = RESULTS) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run eICU-only repaired evaluation with frozen MIMIC source alphas."""
    if not analysis_ids or len(set(analysis_ids)) != len(analysis_ids):
        raise DecoderInputError("analysis_ids must be nonempty and unique")
    requested_ids = list(analysis_ids)
    requested_targets = list(target_dbs)
    if not requested_targets or len(set(requested_targets)) != len(requested_targets):
        raise DecoderInputError("target_dbs must be nonempty and unique")
    if source_db in requested_targets:
        raise DecoderInputError("source_db cannot also be an evaluation target")
    if n_bootstrap < BOOTSTRAP_REPLICATES and not preflight_only:
        raise DecoderInputError(f"decoder requires at least {BOOTSTRAP_REPLICATES} patient bootstrap draws")

    gate_active = any((expected_embedding_manifest_sha256,
                       expected_eligibility_manifest_sha256,
                       expected_source_reference_sha256))
    if gate_active:
        if tuple(requested_ids) != CORE_ANALYSIS_IDS:
            raise DecoderInputError(
                f"repaired decoder requires exactly {list(CORE_ANALYSIS_IDS)}")
        if source_db != SOURCE_DB or tuple(requested_targets) != TARGET_DBS:
            raise DecoderInputError(
                "repaired scope is MIMIC source and eicu_semantic_v4_2 target only")

    manifest, entries = load_embedding_manifest(manifest_path)
    manifest_path = Path(manifest_path)
    manifest_hash = sha256_file(manifest_path)
    if (expected_embedding_manifest_sha256 is not None
            and manifest_hash.lower() != expected_embedding_manifest_sha256.lower()):
        raise DecoderInputError("embedding manifest SHA256 differs from repaired-run clearance")
    if (expected_embedding_manifest_sha256 is not None
            and len(manifest.get("models", [])) != EXPECTED_REPAIRED_EMBEDDING_MODEL_COUNT):
        raise DecoderInputError("embedding manifest does not contain the frozen 41 cache rows")

    selected_dbs = [source_db, *requested_targets]
    if gate_active and set(selected_dbs) != set(EXPECTED_REPAIRED_COHORTS):
        raise DecoderInputError("repaired eligibility gate has an unexpected cohort scope")
    selected_entries = {
        f"{db}/{analysis_id}": entries[f"{db}/{analysis_id}"]
        for db in selected_dbs for analysis_id in requested_ids
        if f"{db}/{analysis_id}" in entries
    }
    if len(selected_entries) != len(selected_dbs) * len(requested_ids):
        missing = [f"{db}/{analysis_id}" for db in selected_dbs
                   for analysis_id in requested_ids if f"{db}/{analysis_id}" not in entries]
        raise DecoderInputError(f"embedding manifest lacks requested caches: {missing}")

    reference_rows, source_reference_hash = load_frozen_source_reference(
        source_reference_path, requested_ids,
        expected_sha256=expected_source_reference_sha256,
    )
    checkpoint_path_resolver = checkpoint_resolver or (
        lambda model_tag: ROOT / "models" / f"{model_tag}.pt")
    model_tags = sorted({str(row["model_tag"]) for row in selected_entries.values()})
    checkpoint_paths = {model_tag: Path(checkpoint_path_resolver(model_tag))
                        for model_tag in model_tags}
    checkpoint_hashes = validate_checkpoint_hashes(selected_entries, checkpoint_paths)

    caches: dict[str, dict[str, EmbeddingCache]] = {}
    target_frames: dict[str, pd.DataFrame] = {}
    target_paths: dict[str, Path] = {}
    target_hashes: dict[str, str] = {}
    split_hashes: dict[str, str] = {}
    sequence_hashes: dict[str, str] = {}
    splits: dict[str, np.ndarray] = {}
    cache_hashes: dict[str, dict[str, str]] = {}
    sequence_identities: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    eligibility_frames: dict[str, pd.DataFrame] = {}
    eligibility_records: dict[str, dict[str, str]] = {}
    eligibility_counts: dict[str, int] = {}
    eligibility_dir = Path(eligibility_dir)
    for db in selected_dbs:
        tabular_path, split_path, sequence_path = _resolve_cohort_paths(db, resolver)
        if not tabular_path.is_file():
            raise FileNotFoundError(f"{db}: numeric target table is missing: {tabular_path}")
        if not sequence_path.is_file():
            raise FileNotFoundError(f"{db}: source sequence NPZ is missing: {sequence_path}")
        eligible, eligibility_entry = _load_ready_eligibility(
            db, eligibility_dir, eligibility_loader)
        expected_sequence_hashes = {
            str(selected_entries[f"{db}/{analysis_id}"]["source_npz_sha256"])
            for analysis_id in requested_ids
        }
        if len(expected_sequence_hashes) != 1:
            raise DecoderInputError(f"{db}: condition caches disagree on source NPZ SHA256")
        expected_sequence_hash = next(iter(expected_sequence_hashes))
        actual_sequence_hash = sha256_file(sequence_path)
        if actual_sequence_hash.lower() != expected_sequence_hash.lower():
            raise DecoderInputError(f"{db}: current source sequence NPZ SHA256 differs from cache manifest")
        sequence_hashes[db] = actual_sequence_hash
        sequence_identities[db] = load_sequence_identity(sequence_path)
        eligible, eligibility_record = verify_current_eligibility(
            db, eligibility_dir, eligible, eligibility_entry, actual_sequence_hash,
            *sequence_identities[db], expected_manifest_sha256=expected_eligibility_manifest_sha256,
            expected_endpoint=endpoint,
        )
        eligibility_frames[db] = eligible
        eligibility_records[db] = eligibility_record
        eligibility_counts[db] = int(len(eligible))

        caches[db] = {}
        cache_hashes[db] = {}
        for analysis_id in requested_ids:
            row = entries[f"{db}/{analysis_id}"]
            cache = load_embedding_cache(row, db, analysis_id)
            verify_cache_sequence_identity(cache, *sequence_identities[db][:2])
            caches[db][analysis_id] = cache
            cache_hashes[db][analysis_id] = cache.cache_sha256
        target_frames[db] = target_reader(tabular_path)
        target_paths[db] = tabular_path
        target_hashes[db] = sha256_file(tabular_path)
        expected_split_hashes = {
            str(caches[db][analysis_id].manifest_entry["split_sha256"])
            for analysis_id in requested_ids
        }
        if len(expected_split_hashes) != 1:
            raise DecoderInputError(f"{db}: condition caches disagree on split SHA256")
        split_hash = next(iter(expected_split_hashes))
        splits[db] = _load_checked_split(split_path, split_hash, db)
        split_hashes[db] = split_hash

    populations: dict[str, AlignedPopulation] = {
        db: build_common_population(
            caches[db], target_frames[db], splits[db], eligibility_frames[db])
        for db in selected_dbs
    }
    source_population = populations[source_db]
    if not {0, 1, 2}.issubset(set(np.unique(source_population.split).tolist())):
        raise DecoderInputError(
            f"{source_db}: split must include development 0, validation 1, and test 2")
    validate_source_patient_partitions(source_population, source_db)
    source_reference_identity = validate_source_reference_identities(
        source_population, reference_rows, requested_ids)

    preflight_counts = {
        "eligible_count_by_db": eligibility_counts,
        "condition_intersection_count_by_db": {
            db: int(len(populations[db].stay_id)) for db in selected_dbs
        },
        "cache_rows_by_condition_by_db": {
            db: populations[db].cache_rows_by_analysis for db in selected_dbs
        },
        "finite_target_rows_by_db_variable": {
            db: {variable: int(np.isfinite(populations[db].targets_by_variable[variable]).sum())
                 for variable in VARS}
            for db in selected_dbs
        },
        "mimic_source_split_rows_by_variable": {
            variable: {
                str(split_id): int(((source_population.split == split_id)
                                    & np.isfinite(source_population.targets_by_variable[variable])).sum())
                for split_id in (0, 1, 2)
            }
            for variable in VARS
        },
    }
    preflight_manifest = {
        "status": "preflight_complete",
        "module": Path(__file__).name,
        "module_sha256": sha256_file(Path(__file__)),
        "embedding_manifest": str(manifest_path.resolve()),
        "embedding_manifest_sha256": manifest_hash,
        "embedding_manifest_model_count": int(len(manifest.get("models", []))),
        "source_reference_summary": str(Path(source_reference_path).resolve()),
        "source_reference_summary_sha256": source_reference_hash,
        "source_db": source_db,
        "target_dbs": requested_targets,
        "loaded_cohorts": selected_dbs,
        "analysis_ids": requested_ids,
        "eligibility_directory": str(eligibility_dir.resolve()),
        "eligibility_manifest_sha256_by_db": {
            db: eligibility_records[db]["eligibility_manifest_sha256"] for db in selected_dbs
        },
        "eligibility_table_sha256_by_db": {
            db: eligibility_records[db]["eligibility_table_sha256"] for db in selected_dbs
        },
        "source_sequence_npz_sha256_by_db": sequence_hashes,
        "split_sha256_by_db": split_hashes,
        "checkpoint_sha256_by_model": checkpoint_hashes,
        "embedding_cache_sha256_by_db_and_analysis": cache_hashes,
        "source_reference_identity_validation": source_reference_identity,
        "preflight_counts": preflight_counts,
    }
    if preflight_only:
        return pd.DataFrame(), preflight_manifest

    output_rows: list[dict[str, Any]] = []
    source_metric_validation: list[dict[str, Any]] = []
    for analysis_id in requested_ids:
        print(f"V4_REPAIRED_ANALYSIS_START {analysis_id}", flush=True)
        source_embedding = source_population.embedding_by_analysis[analysis_id]
        for variable in VARS:
            source_y = source_population.targets_by_variable[variable]
            source_train_mask = (source_population.split == 0) & np.isfinite(source_y)
            source_validation_mask = (source_population.split == 1) & np.isfinite(source_y)
            source_test_mask = (source_population.split == 2) & np.isfinite(source_y)
            reference = reference_rows[(analysis_id, variable)]
            model, fit_meta = fit_source_ridge_fixed_alpha(
                source_embedding[source_train_mask], source_y[source_train_mask],
                source_population.subject[source_train_mask], float(reference["alpha"]),
            )
            mimic_test_prediction = model.predict(source_embedding[source_test_mask])
            mimic_test_metrics = _metrics(source_y[source_test_mask], mimic_test_prediction)
            source_metric_validation.append(compare_source_test_metrics(
                mimic_test_metrics, reference, analysis_id=analysis_id, variable=variable))

            for target_db in requested_targets:
                population = populations[target_db]
                y = population.targets_by_variable[variable]
                test_mask = np.isfinite(y)
                target_embedding = population.embedding_by_analysis[analysis_id]
                metrics: dict[str, Any] = {}
                status = "ok"
                if test_mask.sum() == 0:
                    status = "no_finite_target_values"
                else:
                    prediction = model.predict(target_embedding[test_mask])
                    metrics = _metrics(y[test_mask], prediction)
                    bootstrap_seed = paired_bootstrap_seed(
                        target_db, variable, "source_transfer", seed)
                    metrics.update(patient_cluster_bootstrap_metrics(
                        y[test_mask], prediction, population.subject[test_mask],
                        n_boot=n_bootstrap, seed=bootstrap_seed,
                    ))
                    metrics["bootstrap_interval_type"] = "marginal_patient_cluster_percentile"
                    if not np.isfinite(metrics["r2"]):
                        status = "ok_r2_undefined"
                output_rows.append(_result_row(
                    analysis_id=analysis_id, mode="source_transfer",
                    source_db=source_db, target_db=target_db, variable=variable,
                    population=population, mask=test_mask,
                    target_usage="frozen_mimic_split1_alpha;source_split0_refit;target_labels_metric_only",
                    status=status, metrics=metrics,
                    alpha=fit_meta["alpha"],
                    source_alpha_validation_mse=float(reference["source_alpha_validation_mse"]),
                    source_alpha_validation_mse_by_alpha_json=str(
                        reference["source_alpha_validation_mse_by_alpha_json"]),
                    n_source_train_rows=int(source_train_mask.sum()),
                    n_source_train_patients=int(
                        len(np.unique(source_population.subject[source_train_mask]))),
                    n_source_validation_rows=int(source_validation_mask.sum()),
                    n_source_validation_patients=int(
                        len(np.unique(source_population.subject[source_validation_mask]))),
                    n_source_test_rows=int(source_test_mask.sum()),
                    source_train_population_identity_sha256=_population_hash(
                        source_population, source_train_mask, source_db),
                    source_validation_population_identity_sha256=_population_hash(
                        source_population, source_validation_mask, source_db),
                    source_mimic_split2_metrics_match_reference=True,
                ))

            for target_db in requested_targets:
                population = populations[target_db]
                y = population.targets_by_variable[variable]
                nested_mask = np.isfinite(y)
                target_embedding = population.embedding_by_analysis[analysis_id]
                target_usage = "target_outer_train_and_inner_model_selection"
                n_patients = int(len(np.unique(population.subject[nested_mask])))
                try:
                    if nested_mask.sum() == 0:
                        raise DecoderInputError("no finite target labels")
                    oof, metrics = grouped_nested_ridge(
                        target_embedding[nested_mask], y[nested_mask],
                        population.subject[nested_mask],
                    )
                    bootstrap_seed = paired_bootstrap_seed(
                        target_db, variable, "target_local_nested_cv", seed)
                    metrics.update(patient_cluster_bootstrap_metrics(
                        y[nested_mask], oof, population.subject[nested_mask],
                        n_boot=n_bootstrap, seed=bootstrap_seed,
                    ))
                    metrics["bootstrap_interval_type"] = "marginal_patient_cluster_percentile"
                    status = "ok" if np.isfinite(metrics["r2"]) else "ok_r2_undefined"
                    output_rows.append(_result_row(
                        analysis_id=analysis_id, mode="target_local_nested_cv",
                        source_db=source_db, target_db=target_db, variable=variable,
                        population=population, mask=nested_mask,
                        target_usage=target_usage, status=status,
                        metrics={key: value for key, value in metrics.items()
                                 if key not in {"n_patients"}},
                        alpha_median=metrics["alpha_median"],
                        alpha_counts_json=json.dumps(metrics["alpha_counts"], sort_keys=True),
                        n_outer_folds=metrics["n_outer_folds"],
                        n_inner_folds=metrics["n_inner_folds"],
                    ))
                    del oof
                except (DecoderInputError, ValueError) as exc:
                    output_rows.append(_result_row(
                        analysis_id=analysis_id, mode="target_local_nested_cv",
                        source_db=source_db, target_db=target_db, variable=variable,
                        population=population, mask=nested_mask,
                        target_usage=target_usage,
                        status=f"nested_fit_unavailable:{type(exc).__name__}:{exc}",
                        n_outer_folds=OUTER_FOLDS, n_inner_folds=INNER_FOLDS,
                        n_patients=n_patients,
                    ))
            print(f"V4_REPAIRED_PROGRESS {analysis_id} {variable}", flush=True)

    result = pd.DataFrame(output_rows)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "numeric_decoding_summary.csv"
    manifest_out_path = out_dir / "numeric_decoding_manifest.json"
    result.to_csv(results_path, index=False)
    run_manifest = {
        **preflight_manifest,
        "status": "complete",
        "variables": list(VARS),
        "alphas": list(ALPHAS),
        "source_alpha_provenance": (
            "MIMIC split-1-selected alphas reused from frozen source summary; no reselection"
        ),
        "source_transfer_fit": "scaler and Ridge refit on MIMIC split 0 only",
        "source_mimic_split2_metric_validation": source_metric_validation,
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "bootstrap_replicates": int(n_bootstrap),
        "bootstrap_cluster_unit": "patient",
        "bootstrap_seed_base": int(seed),
        "bootstrap_interval_type": "marginal patient-cluster percentile intervals",
        "bootstrap_pairing": (
            "derived patient-resample seeds are shared across the three conditions; "
            "reported intervals are marginal and no paired R2-difference intervals were computed"
        ),
        "numeric_target_source_by_db": {
            db: {"path": str(target_paths[db]), "tabular_sha256": target_hashes[db]}
            for db in selected_dbs
        },
        "source_label_usage": "raw numeric targets only; embedding source_y mortality array is not read",
        "target_availability": (
            "finite numeric target after unique stay_id join, on the intersection of stays "
            "present in every selected analysis condition for each loaded cohort"
        ),
        "source_transfer_target_label_usage": (
            "source alpha frozen from MIMIC split 1; scaler and Ridge refit on MIMIC split 0; "
            "eicu_semantic_v4_2 labels used only for scoring"
        ),
        "target_local_label_usage": "eicu_semantic_v4_2 labels used within five outer and three inner patient-group folds",
        "interval_scope": "conditional on fixed MIMIC source fits or fixed eICU nested-CV OOF predictions",
        "results_csv": str(results_path.resolve()),
        "results_csv_sha256": sha256_file(results_path),
    }
    manifest_out_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    return result, run_manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--eligibility-dir", type=Path, default=DEFAULT_ELIGIBILITY)
    parser.add_argument("--source-reference", type=Path, default=SOURCE_REFERENCE_SUMMARY)
    parser.add_argument("--source-db", default=SOURCE_DB)
    parser.add_argument("--target-dbs", nargs="+", default=list(TARGET_DBS))
    parser.add_argument("--analysis-ids", nargs="+", default=list(CORE_ANALYSIS_IDS))
    parser.add_argument("--endpoint", default="hospital", choices=("hospital", "icu"))
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    parser.add_argument(
        "--preflight", action="store_true",
        help="verify frozen inputs and source/target identities without fitting models",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="run fixed-alpha source transfer and target-local nested decoding",
    )
    args = parser.parse_args(argv)
    if args.execute == args.preflight:
        parser.error("choose exactly one of --preflight or --execute")
    _, run_manifest = run_decoding(
        manifest_path=args.manifest,
        analysis_ids=args.analysis_ids,
        target_dbs=args.target_dbs,
        source_db=args.source_db,
        source_reference_path=args.source_reference,
        eligibility_dir=args.eligibility_dir,
        endpoint=args.endpoint,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        preflight_only=args.preflight,
        output_dir=args.output_dir,
    )
    if args.preflight:
        print(json.dumps(run_manifest, sort_keys=True), flush=True)
        print("V4_NUMERIC_DECODING_PREFLIGHT_COMPLETE", flush=True)
    else:
        print(f"V4_NUMERIC_DECODING_COMPLETE {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
