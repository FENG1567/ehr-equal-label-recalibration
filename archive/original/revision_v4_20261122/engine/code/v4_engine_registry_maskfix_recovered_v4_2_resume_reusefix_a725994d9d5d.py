"""Identity-bound frozen-encoder inference for the 2026-11-22 revision.

This module deliberately separates embedding generation from clinical
prediction.  ``--mode encode`` is allowed before temporal eligibility metadata
is ready; ``--mode predict`` fails closed unless every requested cohort has an
exact, verified eligibility manifest and row-level table.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import pickle
import platform
import re
import sys
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any

from cohort_registry_v4 import registered_cohorts, resolve_cohort_paths

import numpy as np
import pandas as pd

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
ENGINE = REV / "engine"
PROC = ROOT / "data/processed"
MODELS = ROOT / "models"
TEMPORAL = REV / "temporal/results/eligibility"
DBS = ("mimic", "eicu", "nwicu", "sicdb", "hirid", "c2012", "zigong")
SPECS = (
    ("main20k", "real"),
    ("main20k", "constant"),
    ("main20k", "common8_mask"),
    ("main20k", "no_h0_therapy"),
    ("abl_const20k", "real"),
    ("abl_const20k", "constant"),
    ("common20k", "common8_mask"),
    ("big50m", "real"),
    ("big50m_full", "real"),
    ("big50m_full", "common8_mask"),
)
REQUIRED_ELIGIBILITY = {
    "db", "stay_id", "subject", "eligible_24h", "temporal_verified",
    "y_mort", "endpoint", "reason",
}
INPUT_SCOPE = "retrospective_measurement_time"
NO_H0_THERAPY_CANONICAL_VARIABLES = ("vaso", "rrt", "sedation")
NO_H0_THERAPY_RULE = {
    "condition": "no_h0_therapy",
    "canonical_variable_names": list(NO_H0_THERAPY_CANONICAL_VARIABLES),
    "hour": 0,
    "masked_features": ["V", "B", "H"],
    "mask_action": "zero",
    "stay_retention_rule": "retain iff any nonzero variable token remains",
}
CLINICAL_INPUT_AVAILABILITY_VERIFIED = False
INPUT_AVAILABILITY_LIMITATION = (
    "Original sequence NPZ is retrospective: raw event storetime was dropped from the "
    "processed inputs and negative therapy start times were clipped during preprocessing. "
    "Real-time clinical input availability is unverified."
)


class ClinicalReadinessError(RuntimeError):
    """Raised when temporal eligibility evidence does not authorize prediction."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_identity_string(value: Any) -> str:
    """Normalize mixed numeric/byte identifiers to their textual identity."""
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    return str(value)


def sha256_ids(db: str, rows: np.ndarray, stay_id: np.ndarray,
               subject: np.ndarray) -> str:
    digest = hashlib.sha256(db.encode("utf-8"))
    digest.update(np.asarray(rows, dtype=np.int64).tobytes())
    for values in (stay_id, subject):
        for value in np.asarray(values):
            encoded = canonical_identity_string(value).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    return digest.hexdigest()


def sha256_ordered_prediction_rows(frame: pd.DataFrame) -> str:
    """Hash ordered prediction identity, labels, and retrospective scope."""
    columns = ("db", "analysis_id", "stay_id", "subject", "rows", "split", "y",
               "p", "endpoint", "role", "prediction_scope", "input_scope",
               "eligibility_manifest_sha256", "eligibility_table_sha256")
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"prediction digest missing columns: {sorted(missing)}")
    digest = hashlib.sha256()
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
        record = [
            str(row[0]), str(row[1]),
            canonical_identity_string(row[2]), canonical_identity_string(row[3]),
            int(row[4]), int(row[5]), int(row[6]), format(float(row[7]), ".17g"),
            str(row[8]), str(row[9]), str(row[10]), str(row[11]),
            str(row[12]), str(row[13]),
        ]
        digest.update(json.dumps(record, separators=(",", ":"),
                                ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def source_probe_artifact_key(metadata: dict[str, Any]) -> str:
    """Identify a fitted probe by checkpoint, cache runtime, eligibility, and scope."""
    fields = (
        "run_id", "model_tag", "input_condition", "checkpoint_sha256", "input_mask_sha256",
        "eligibility_manifest_sha256", "eligibility_table_sha256",
        "source_npz_sha256", "source_train_identity_sha256",
        "source_train_label_sha256", "prediction_scope",
        "runtime_identity_sha256",
    )
    missing = [field for field in fields if metadata.get(field) in (None, "")]
    if missing:
        raise ValueError(f"source probe metadata missing {missing}")
    identity = {field: metadata[field] for field in fields}
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_probe_artifact_path(metadata: dict[str, Any],
                               root: Path | None = None) -> Path:
    digest = source_probe_artifact_key(metadata)
    base = ENGINE / "results/source_probes" if root is None else Path(root)
    return base / (f"{metadata['model_tag']}_{metadata['input_condition']}_"
                   f"{digest[:20]}.pkl")


def atomic_pickle_dump(payload: Any, path: Path) -> None:
    """Persist an audit artifact atomically so interrupted writes are never reused."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                          dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    """Persist prediction audit JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                          dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_reusable_source_probe(metadata: dict[str, Any], origin_run_id: str,
                               expected_feature_dim: int) -> dict[str, Any]:
    """Load a prior run's probe only when its scientific identity is identical."""
    origin_metadata = dict(metadata)
    origin_metadata["run_id"] = origin_run_id
    artifact_path = source_probe_artifact_path(origin_metadata)
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"audited source probe for origin run {origin_run_id!r} is missing: "
            f"{artifact_path}")
    resolved = artifact_path.resolve(strict=True)
    source_probe_root = (ENGINE / "results/source_probes").resolve()
    if resolved.parent != source_probe_root:
        raise ValueError(f"source probe artifact escapes its results directory: {resolved}")
    try:
        with resolved.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:
        raise ValueError(f"cannot load saved source probe {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"saved source probe must be an object: {resolved}")
    mismatched = [
        field for field, expected in origin_metadata.items()
        if payload.get(field) != expected
    ]
    if mismatched:
        raise ValueError(
            f"saved source probe scientific identity mismatch for "
            f"{mismatched}: {resolved}")
    if source_probe_artifact_key(payload) != source_probe_artifact_key(origin_metadata):
        raise ValueError(f"saved source probe key does not match its metadata: {resolved}")
    scaler = payload.get("scaler")
    probe = payload.get("probe")
    if not callable(getattr(scaler, "transform", None)) or not callable(
            getattr(probe, "predict_proba", None)):
        raise ValueError(f"saved source probe lacks usable scaler/classifier: {resolved}")
    mean = np.asarray(getattr(scaler, "mean_", []))
    coefficients = np.asarray(getattr(probe, "coef_", []))
    if (mean.shape != (expected_feature_dim,) or coefficients.ndim != 2
            or coefficients.shape[1] != expected_feature_dim
            or not np.isfinite(mean).all() or not np.isfinite(coefficients).all()):
        raise ValueError(f"saved source probe feature shape is invalid: {resolved}")
    artifact_sha256 = sha256_file(resolved)
    return {
        **payload,
        "artifact_path": str(resolved),
        "artifact_sha256": artifact_sha256,
        "reused_from_run_id": origin_run_id,
    }


def canonical_json_sha256(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def input_mask_hash(condition: str, variable_ids: tuple[int, ...] = ()) -> str:
    return canonical_json_sha256({"condition": condition,
                                  "variable_ids": list(variable_ids)})


def no_h0_therapy_definition_sha256() -> str:
    return canonical_json_sha256(NO_H0_THERAPY_RULE)


def no_h0_therapy_input_mask_spec(db: str,
                                  variable_ids: tuple[int, ...]) -> dict[str, Any]:
    if len(variable_ids) != len(NO_H0_THERAPY_CANONICAL_VARIABLES):
        raise ValueError("no_h0_therapy requires exactly three canonical variable IDs")
    if len(set(variable_ids)) != len(NO_H0_THERAPY_CANONICAL_VARIABLES):
        raise ValueError("no_h0_therapy requires three distinct variable IDs")
    return {
        **NO_H0_THERAPY_RULE,
        "canonical_variable_names": list(NO_H0_THERAPY_CANONICAL_VARIABLES),
        "variable_ids": [int(value) for value in variable_ids],
        "condition_definition_sha256": no_h0_therapy_definition_sha256(),
        "db": str(db),
    }


def source_probe_mask_compatible(fitted: dict[str, Any], condition: str,
                                 target_mask_sha256: str,
                                 target_mask_spec: dict[str, Any] | None = None) -> bool:
    """Check cross-cohort source-probe mask compatibility.

    Ordinary conditions require an identical mask hash. The no-H0 condition
    has cohort-specific token IDs, so compare the shared canonical rule and
    require each cohort's mapping to contain the same three distinct tokens.
    """
    if condition != "no_h0_therapy":
        return fitted.get("input_mask_sha256") == target_mask_sha256

    expected_names = list(NO_H0_THERAPY_CANONICAL_VARIABLES)

    def valid_spec(spec: Any, mask_sha256: Any) -> bool:
        if not isinstance(spec, dict):
            return False
        ids = spec.get("variable_ids")
        expected_keys = set(NO_H0_THERAPY_RULE) | {
            "variable_ids", "condition_definition_sha256", "db",
        }
        if (set(spec) != expected_keys
                or any(spec.get(key) != value
                       for key, value in NO_H0_THERAPY_RULE.items())
                or spec.get("condition_definition_sha256") != no_h0_therapy_definition_sha256()
                or not isinstance(spec.get("db"), str)
                or not spec["db"]
                or not isinstance(ids, (list, tuple))
                or len(ids) != len(expected_names)):
            return False
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
               for value in ids):
            return False
        return (len(set(ids)) == len(expected_names)
                and canonical_json_sha256(spec) == mask_sha256)

    source_spec = fitted.get("input_mask_spec")
    return (valid_spec(source_spec, fitted.get("input_mask_sha256"))
            and valid_spec(target_mask_spec, target_mask_sha256))


def model_parameter_dtypes(model) -> list[str]:
    if not hasattr(model, "parameters"):
        return ["unknown"]
    values = sorted({str(parameter.dtype) for parameter in model.parameters()})
    return values or ["no_parameters"]


def runtime_identity(torch, model, device: str, torch_num_threads: int) -> dict[str, Any]:
    cuda_device = None
    cuda_capability = None
    if device == "cuda":
        index = int(torch.cuda.current_device())
        cuda_device = {
            "index": index,
            "name": str(torch.cuda.get_device_name(index)),
        }
        capability = torch.cuda.get_device_capability(index)
        cuda_capability = [int(capability[0]), int(capability[1])]
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_machine": platform.machine(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "encoding_device": str(device),
        "cuda_device": cuda_device,
        "cuda_compute_capability": cuda_capability,
        "model_parameter_dtypes": model_parameter_dtypes(model),
        "encoding_dtype": "float32",
        "torch_num_threads": int(torch_num_threads),
    }


def encoder_code_hash() -> str:
    """Hash the encoder implementation and its model construction dependencies."""
    functions = (canonical_json_sha256, input_mask_hash,
                 no_h0_therapy_definition_sha256, no_h0_therapy_input_mask_spec,
                 model_parameter_dtypes, runtime_identity, prepare_batch, prepare_inputs,
                 load_common_ids, load_no_h0_therapy_ids, validate_sequence,
                 sequence_rows, condition_rows, load_sequence, encode_rows)
    pieces = [json.dumps(SPECS, separators=(",", ":")),
              json.dumps(NO_H0_THERAPY_RULE, sort_keys=True, separators=(",", ":"))]
    for function in functions:
        pieces.append(inspect.getsource(function))
    dependency_paths = (ROOT / "code/fm.py",
                        ROOT / "revision_v2_20261117/code/decoding_motor_v2.py")
    for path in dependency_paths:
        pieces.append(path.name + ":" + sha256_file(path))
    return hashlib.sha256("\n".join(pieces).encode("utf-8")).hexdigest()


def common_source_hashes(root: Path = ROOT) -> dict[str, str]:
    return {
        "common_variable_names_sha256": sha256_file(root / "results/tables/common_vars.npy"),
        "variable_vocabulary_sha256": sha256_file(root / "data/processed/meta_mimic.json"),
    }


def prepare_batch(ds: dict[str, np.ndarray], rows: np.ndarray,
                  condition: str, common_ids: tuple[int, ...] = ()) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare model inputs while preserving each condition's token-level rule."""
    rows = np.asarray(rows, dtype=np.int64)
    if not rows.size:
        raise ValueError("cannot prepare an empty batch")
    width = max(1, int(np.max(ds["L"][rows])))
    original_v = np.asarray(ds["V"][rows, :width])
    original_h = np.asarray(ds["H"][rows, :width])
    V, B, H = (np.array(ds[key][rows, :width], dtype=np.int64, copy=True)
               for key in ("V", "B", "H"))
    if condition == "constant":
        padding = V == 0
        B[padding] = 0
        B[~padding] = 1
    elif condition == "common8_mask":
        if len(common_ids) != 8 or len(set(common_ids)) != 8:
            raise ValueError("common8_mask requires exactly eight distinct variable IDs")
        keep = np.isin(V, np.asarray(common_ids, dtype=np.int64)) & (V != 0)
        if np.any(~keep.any(axis=1)):
            raise ValueError("common8_mask batch contains an empty eligible input mask")
        V[~keep] = 0
        B[~keep] = 0
        H[~keep] = 0
    elif condition == "no_h0_therapy":
        if (len(common_ids) != len(NO_H0_THERAPY_CANONICAL_VARIABLES)
                or len(set(common_ids)) != len(NO_H0_THERAPY_CANONICAL_VARIABLES)):
            raise ValueError("no_h0_therapy requires exactly three distinct variable IDs")
        remove = ((H == 0) &
                  np.isin(V, np.asarray(common_ids, dtype=np.int64)) & (V != 0))
        V[remove] = 0
        B[remove] = 0
        H[remove] = 0
        if np.any(~(V != 0).any(axis=1)):
            raise ValueError("no_h0_therapy batch contains a stay with no remaining inputs")
    elif condition != "real":
        raise ValueError(f"unsupported input condition: {condition}")
    if condition not in {"common8_mask", "no_h0_therapy"}:
        if not np.array_equal(V, original_v):
            raise AssertionError("input condition changed variable IDs")
        if not np.array_equal(H, original_h):
            raise AssertionError("input condition changed time IDs")
    elif condition == "no_h0_therapy":
        remove = ((original_h == 0) &
                  np.isin(original_v, np.asarray(common_ids, dtype=np.int64)) &
                  (original_v != 0))
        expected_v = np.array(original_v, dtype=np.int64, copy=True)
        expected_h = np.array(original_h, dtype=np.int64, copy=True)
        expected_v[remove] = 0
        expected_h[remove] = 0
        if not np.array_equal(V, expected_v) or not np.array_equal(H, expected_h):
            raise AssertionError("no_h0_therapy changed inputs outside its H==0 token mask")
    return V, B, H


def prepare_inputs(ds: dict[str, np.ndarray], rows: np.ndarray,
                  condition: str, variable_ids: tuple[int, ...] = ()) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Public condition-aware input helper shared with baseline fitting."""
    return prepare_batch(ds, rows, condition, variable_ids)


def load_common_ids(root: Path = ROOT) -> tuple[tuple[int, ...], str]:
    """Load the original common8 variable names and map them to token IDs."""
    names_path = root / "results/tables/common_vars.npy"
    meta_path = root / "data/processed/meta_mimic.json"
    if not names_path.exists() or not meta_path.exists():
        raise FileNotFoundError("common8 variable names or MIMIC vocabulary metadata missing")
    names = tuple(str(x) for x in np.load(names_path, allow_pickle=True).tolist())
    meta = json.loads(meta_path.read_text())
    variables = meta.get("vars")
    if not isinstance(variables, list):
        raise ValueError("meta_mimic.json does not define the variable vocabulary")
    missing = sorted(set(names) - set(variables))
    if len(names) != 8 or missing:
        raise ValueError(f"common8 vocabulary mismatch (n={len(names)}, missing={missing})")
    ids = tuple(int(variables.index(name) + 1) for name in names)
    if len(set(ids)) != 8:
        raise ValueError("common8 names do not map to eight unique variable tokens")
    return ids, input_mask_hash("common8_mask", ids)


def load_no_h0_therapy_ids(root: Path = ROOT,
                           cohorts: tuple[str, ...] = DBS) -> dict[str, dict[str, int]]:
    """Map canonical MIMIC vaso/RRT/sedation variables in each cohort vocabulary."""
    mapping: dict[str, dict[str, int]] = {}
    for db in cohorts:
        meta_path = Path(root) / "data/processed" / f"meta_{db}.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"{db}: variable vocabulary metadata missing: {meta_path}")
        meta = json.loads(meta_path.read_text())
        variables = meta.get("vars") if isinstance(meta, dict) else None
        if not isinstance(variables, list):
            raise ValueError(f"{db}: metadata does not define a variable vocabulary")
        missing = [name for name in NO_H0_THERAPY_CANONICAL_VARIABLES
                   if variables.count(name) != 1]
        if missing:
            raise ValueError(
                f"{db}: canonical no_h0_therapy variables must appear once; invalid={missing}")
        mapping[db] = {
            name: int(variables.index(name) + 1)
            for name in NO_H0_THERAPY_CANONICAL_VARIABLES
        }
        if len(set(mapping[db].values())) != len(NO_H0_THERAPY_CANONICAL_VARIABLES):
            raise ValueError(f"{db}: no_h0_therapy names do not map to distinct token IDs")
    return mapping


def load_no_h0_therapy_ids_registered(
        cohorts: tuple[str, ...]) -> dict[str, dict[str, int]]:
    """Resolve per-cohort vocabularies from the revision-owned path registry."""
    mapping: dict[str, dict[str, int]] = {}
    for db in cohorts:
        vocab_path = resolve_cohort_paths(db, verify_exists=True).vocab_path
        if vocab_path is None:
            raise FileNotFoundError(
                f"{db}: no vocabulary path is registered for no_h0_therapy")
        meta = json.loads(vocab_path.read_text())
        variables = meta.get("vars") if isinstance(meta, dict) else None
        if not isinstance(variables, list):
            raise ValueError(f"{db}: metadata does not define a variable vocabulary")
        missing = [name for name in NO_H0_THERAPY_CANONICAL_VARIABLES
                   if variables.count(name) != 1]
        if missing:
            raise ValueError(
                f"{db}: canonical no_h0_therapy variables must appear once; invalid={missing}")
        mapping[db] = {
            name: int(variables.index(name) + 1)
            for name in NO_H0_THERAPY_CANONICAL_VARIABLES
        }
        if len(set(mapping[db].values())) != len(NO_H0_THERAPY_CANONICAL_VARIABLES):
            raise ValueError(f"{db}: no_h0_therapy names do not map to distinct token IDs")
    return mapping


def validate_sequence(ds: dict[str, np.ndarray], db: str) -> None:
    n = len(ds["L"])
    for key in ("V", "B", "H", "stay_id", "subject", "y_mort", "split"):
        if key not in ds or len(ds[key]) != n:
            raise ValueError(f"{db}: missing or inconsistent sequence array {key}")
    stay_values = np.asarray(ds["stay_id"])
    if pd.isna(stay_values).any():
        raise ValueError(f"{db}: missing stay IDs in sequence data")
    stay_keys = pd.Series(stay_values.astype(str))
    if stay_keys.isin(("", "nan", "None", "-1")).any():
        raise ValueError(f"{db}: invalid stay IDs in sequence data")
    if stay_keys.duplicated().any():
        raise ValueError(f"{db}: duplicate stay_id in sequence data")
    groups = np.asarray(ds["subject"]).astype(str)
    if pd.isna(groups).any() or np.isin(groups, ("", "nan", "None", "-1")).any():
        raise ValueError(f"{db}: missing subject IDs in sequence data")


def sequence_rows(ds: dict[str, np.ndarray]) -> np.ndarray:
    return np.flatnonzero(np.asarray(ds["L"]) > 0).astype(np.int64)


def condition_rows(ds: dict[str, np.ndarray], condition: str,
                   common_ids: tuple[int, ...] = ()) -> np.ndarray:
    rows = sequence_rows(ds)
    if condition == "no_h0_therapy":
        if (len(common_ids) != len(NO_H0_THERAPY_CANONICAL_VARIABLES)
                or len(set(common_ids)) != len(NO_H0_THERAPY_CANONICAL_VARIABLES)):
            raise ValueError("no_h0_therapy requires exactly three distinct variable IDs")
        V = np.asarray(ds["V"])[rows]
        H = np.asarray(ds["H"])[rows]
        remove = ((H == 0) &
                  np.isin(V, np.asarray(common_ids, dtype=np.int64)) & (V != 0))
        usable = ((V != 0) & ~remove).any(axis=1)
        return rows[usable]
    if condition != "common8_mask":
        return rows
    if len(common_ids) != 8 or len(set(common_ids)) != 8:
        raise ValueError("common8_mask requires exactly eight distinct variable IDs")
    usable = np.isin(np.asarray(ds["V"])[rows], np.asarray(common_ids, dtype=np.int64)).any(axis=1)
    return rows[usable]


def cache_path(cache_dir: Path, db: str, model_tag: str, condition: str,
               checkpoint_hash: str, identity_hash: str, mask_hash: str,
               source_hash: str, encoder_hash: str,
               runtime_identity_hash: str) -> Path:
    return cache_dir / (f"emb_{db}_{model_tag}_{condition}_"
                        f"ckpt-{checkpoint_hash[:16]}_id-{identity_hash[:16]}_"
                        f"mask-{mask_hash[:16]}_src-{source_hash[:12]}_"
                        f"enc-{encoder_hash[:12]}_rt-{runtime_identity_hash[:12]}.npz")


def cache_metadata_matches(cached: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Require matching code, model runtime, device, dtype, and encoding inputs."""
    ignored = {"engine_code_sha256"}
    required_provenance = (
        "engine_code_sha256", "runtime_identity_sha256", "runtime_identity",
        "python_version", "torch_version", "encoding_device", "encoding_dtype",
        "model_parameter_dtypes", "torch_num_threads",
    )
    functional_cached = {k: v for k, v in cached.items() if k not in ignored}
    functional_expected = {k: v for k, v in expected.items() if k not in ignored}
    provenance_present = all(cached.get(k) not in (None, "", []) for k in required_provenance)
    return provenance_present and functional_cached == functional_expected


def save_embedding_cache(path: Path, rows: np.ndarray, stay_id: np.ndarray,
                         subject: np.ndarray, source_y: np.ndarray,
                         embedding: np.ndarray,
                         metadata: dict[str, Any]) -> None:
    if embedding.ndim != 2 or len(embedding) != len(rows):
        raise ValueError("embedding dimensions do not match cached rows")
    if not np.isfinite(embedding).all():
        raise FloatingPointError("cannot cache nonfinite embeddings")
    if len(stay_id) != len(rows) or len(subject) != len(rows) or len(source_y) != len(rows):
        raise ValueError("IDs do not match cached rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(
            handle, rows=np.asarray(rows, dtype=np.int64),
            stay_id=np.asarray(stay_id).astype(str),
            subject=np.asarray(subject).astype(str),
            source_y=np.asarray(source_y, dtype=np.float32),
            embedding=np.asarray(embedding, dtype=np.float32),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
    os.replace(tmp, path)


def load_embedding_cache(path: Path, expected_rows: np.ndarray,
                         expected_stay: np.ndarray, expected_subject: np.ndarray,
                         expected_y: np.ndarray,
                         expected_metadata: dict[str, Any]) -> np.ndarray | None:
    """Reuse only a cache whose checkpoint, condition, rows, and IDs match."""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as cache:
        rows = cache["rows"]
        stays = cache["stay_id"].astype(str)
        subjects = cache["subject"].astype(str)
        source_y = cache["source_y"].astype(np.float32)
        embedding = cache["embedding"]
        metadata = json.loads(str(cache["metadata_json"].item()))
    if (not np.array_equal(rows, expected_rows)
            or not np.array_equal(stays, np.asarray(expected_stay).astype(str))
            or not np.array_equal(subjects, np.asarray(expected_subject).astype(str))
            or not np.array_equal(source_y, np.asarray(expected_y, dtype=np.float32), equal_nan=True)
            or not cache_metadata_matches(metadata, expected_metadata)):
        raise ValueError(f"identity/provenance mismatch in embedding cache: {path}")
    if embedding.ndim != 2 or len(embedding) != len(rows) or not np.isfinite(embedding).all():
        raise ValueError(f"invalid embedding matrix in cache: {path}")
    return embedding.astype(np.float32, copy=False)


def validate_eligibility_table(db: str, frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_ELIGIBILITY - set(frame.columns)
    if missing:
        raise ClinicalReadinessError(f"{db}: eligibility table missing {sorted(missing)}")
    if frame["db"].astype(str).ne(db).any():
        raise ClinicalReadinessError(f"{db}: eligibility table contains another cohort")
    if frame["stay_id"].isna().any() or frame["subject"].isna().any():
        raise ClinicalReadinessError(f"{db}: missing stay or subject identity in eligibility table")
    frame = frame.copy()
    frame["stay_id"] = frame["stay_id"].map(canonical_identity_string)
    frame["subject"] = frame["subject"].map(canonical_identity_string)
    if frame["stay_id"].duplicated().any():
        raise ClinicalReadinessError(f"{db}: duplicate stay_id in eligibility table")
    if frame["subject"].isin(("", "nan", "None", "-1")).any():
        raise ClinicalReadinessError(f"{db}: invalid subject identity in eligibility table")
    frame["y_mort"] = pd.to_numeric(frame["y_mort"], errors="coerce")
    if not frame["endpoint"].isin(("hospital", "icu")).all():
        raise ClinicalReadinessError(f"{db}: endpoint must be hospital or icu")
    def bool_values(series: pd.Series, name: str) -> pd.Series:
        if series.isna().any():
            raise ClinicalReadinessError(f"{db}: missing boolean values in {name}")
        if pd.api.types.is_bool_dtype(series.dtype):
            return series.astype(bool)
        normalized = series.astype(str).str.strip().str.lower()
        mapping = {"true": True, "1": True, "yes": True,
                   "false": False, "0": False, "no": False}
        unknown = set(normalized.unique()) - set(mapping)
        if unknown:
            raise ClinicalReadinessError(f"{db}: invalid boolean values in {name}: {sorted(unknown)}")
        return normalized.map(mapping).astype(bool)
    eligible = bool_values(frame["eligible_24h"], "eligible_24h")
    verified = bool_values(frame["temporal_verified"], "temporal_verified")
    unverified = eligible & ~verified
    if unverified.any():
        raise ClinicalReadinessError(
            f"{db}: auditor-selected rows are temporally unverified (n={int(unverified.sum())})")
    nonbinary = eligible & ~frame["y_mort"].isin((0, 1))
    if nonbinary.any():
        raise ClinicalReadinessError(
            f"{db}: auditor-selected rows lack binary outcomes (n={int(nonbinary.sum())})")
    selected = frame.loc[eligible].copy()
    if selected.empty:
        raise ClinicalReadinessError(f"{db}: no eligible, temporally verified binary outcomes")
    if selected["endpoint"].nunique() != 1:
        raise ClinicalReadinessError(f"{db}: endpoint must be analyzed in separate cohorts")
    return selected


def retrospective_landmark_allowed(entry: dict[str, Any]) -> bool:
    """Require the current explicit outcome/landmark attestation."""
    if entry.get("retrospective_landmark_prediction_allowed") is not True:
        return False
    alias = entry.get("ready_for_future24h_landmark_analysis")
    return alias is None or alias is True


def load_ready_eligibility(db: str, eligibility_dir: Path,
                           diagnostic_only: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = eligibility_dir / "eligibility_manifest.json"
    table_path = eligibility_dir / f"{db}.parquet"
    if not manifest_path.exists() or not table_path.exists():
        raise ClinicalReadinessError(f"{db}: eligibility manifest/table is missing")
    manifest = json.loads(manifest_path.read_text())
    entry = manifest.get("cohorts", {}).get(db)
    if not isinstance(entry, dict):
        raise ClinicalReadinessError(f"{db}: cohort missing from eligibility manifest")
    evidence = str(entry.get("evidence", "unavailable")).lower()
    status = str(entry.get("status", "")).lower()
    if evidence not in {"exact", "operational_proxy"} or status in {"unavailable", "not_ready", "blocked", "failed"}:
        raise ClinicalReadinessError(
            f"{db}: prediction requires exact or operational_proxy temporal evidence; "
            f"manifest says evidence={evidence}, status={status or 'unspecified'}")
    claim_ready = retrospective_landmark_allowed(entry)
    diagnostic_ready = entry.get("retrospective_diagnostic_allowed") is True
    if not claim_ready and not (diagnostic_only and diagnostic_ready):
        raise ClinicalReadinessError(
            f"{db}: manifest lacks positive future-24h landmark readiness attestation; "
            "retrospective diagnostic mode requires its separate explicit attestation")
    source_hashes = entry.get("sourcehashes")
    sequence_hash = (source_hashes.get("sequence_npz")
                     if isinstance(source_hashes, dict) else None)
    if (not isinstance(sequence_hash, str) or len(sequence_hash) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in sequence_hash)):
        raise ClinicalReadinessError(
            f"{db}: eligibility manifest lacks an audited sourcehashes.sequence_npz SHA256")
    selected = validate_eligibility_table(db, pd.read_parquet(table_path))
    entry = dict(entry)
    entry["sequence_npz_sha256"] = sequence_hash.lower()
    entry["retrospective_landmark_prediction_allowed"] = bool(claim_ready)
    entry["ready_for_future24h_clinical_claim"] = False
    entry["retrospective_diagnostic_allowed"] = bool(diagnostic_ready)
    entry["eligibility_table_sha256"] = sha256_file(table_path)
    entry["eligibility_manifest_sha256"] = sha256_file(manifest_path)
    return selected, entry


def assert_sequence_npz_attestation(db: str, eligibility_entry: dict[str, Any],
                                   actual_sequence_npz_sha256: str) -> None:
    expected = str(eligibility_entry.get("sequence_npz_sha256", "")).lower()
    actual = str(actual_sequence_npz_sha256).lower()
    if len(expected) != 64 or actual != expected:
        raise ClinicalReadinessError(
            f"{db}: source seq NPZ SHA256 does not match the temporal manifest attestation")


def align_eligible_rows(db: str, ds: dict[str, np.ndarray], eligible: pd.DataFrame,
                        rows: np.ndarray, condition: str = "real",
                        common_ids: tuple[int, ...] = ()) -> pd.DataFrame:
    source_rows = np.arange(len(ds["L"]), dtype=np.int64)
    pos_by_stay = {
        canonical_identity_string(ds["stay_id"][row]): int(row)
        for row in source_rows
    }
    subjects = np.asarray(
        [canonical_identity_string(value) for value in ds["subject"]], dtype=str)
    matched = eligible.copy()
    matched["stay_id"] = matched["stay_id"].map(canonical_identity_string)
    matched["subject"] = matched["subject"].map(canonical_identity_string)
    matched["stay_key"] = matched["stay_id"]
    matched["rows"] = matched["stay_key"].map(pos_by_stay)
    missing = matched["rows"].isna()
    if missing.any():
        raise ClinicalReadinessError(
            f"{db}: {int(missing.sum())} audited eligible stays are absent from the source sequence NPZ")
    matched["rows"] = matched["rows"].astype(np.int64)
    observed_subject = subjects[matched["rows"].to_numpy()]
    if not np.array_equal(observed_subject, matched["subject"].to_numpy(dtype=str)):
        raise ClinicalReadinessError(f"{db}: stay_id-to-subject identity mismatch")
    observed_y = np.asarray(ds["y_mort"], dtype=float)[matched["rows"].to_numpy()]
    audited_y = matched["y_mort"].to_numpy(dtype=float)
    if not np.array_equal(observed_y, audited_y, equal_nan=True):
        raise ClinicalReadinessError(f"{db}: audited and source NPZ outcome labels differ")

    condition_rows = np.asarray(rows, dtype=np.int64)
    if (condition_rows.size and
            (condition_rows.min() < 0 or condition_rows.max() >= len(source_rows))):
        raise ClinicalReadinessError(f"{db}: condition row index is outside the source sequence")
    if len(np.unique(condition_rows)) != len(condition_rows):
        raise ClinicalReadinessError(f"{db}: condition row selection contains duplicate rows")
    retained = matched["rows"].isin(set(condition_rows.tolist()))
    excluded = matched.loc[~retained]
    if not excluded.empty:
        excluded_rows = excluded["rows"].to_numpy(dtype=np.int64)
        if condition == "common8_mask":
            if len(common_ids) != 8 or len(set(common_ids)) != 8:
                raise ClinicalReadinessError("common8 exclusion requires exactly eight variable IDs")
            if np.any(np.asarray(ds["L"])[excluded_rows] <= 0):
                raise ClinicalReadinessError(
                    f"{db}: empty original sequence rows cannot be treated as common8 mask exclusions")
            has_common_token = np.isin(
                np.asarray(ds["V"])[excluded_rows],
                np.asarray(common_ids, dtype=np.int64)).any(axis=1)
            if has_common_token.any():
                raise ClinicalReadinessError(
                    f"{db}: common8 condition unexpectedly excluded rows with common-variable inputs")
        elif condition == "no_h0_therapy":
            if (len(common_ids) != len(NO_H0_THERAPY_CANONICAL_VARIABLES)
                    or len(set(common_ids)) != len(NO_H0_THERAPY_CANONICAL_VARIABLES)):
                raise ClinicalReadinessError(
                    "no_h0_therapy exclusion requires three distinct variable IDs")
            nonempty = np.asarray(ds["L"])[excluded_rows] > 0
            candidate_rows = excluded_rows[nonempty]
            if len(candidate_rows):
                V = np.asarray(ds["V"])[candidate_rows]
                H = np.asarray(ds["H"])[candidate_rows]
                removed = ((H == 0) &
                           np.isin(V, np.asarray(common_ids, dtype=np.int64)) &
                           (V != 0))
                has_remaining = ((V != 0) & ~removed).any(axis=1)
                if has_remaining.any():
                    raise ClinicalReadinessError(
                        f"{db}: no_h0_therapy excluded a row with remaining inputs")
        else:
            raise ClinicalReadinessError(
                f"{db}: {len(excluded)} audited eligible rows were unexpectedly excluded")
    return matched.loc[retained].copy()

def fit_source_probe(X: np.ndarray, y: np.ndarray, train_mask: np.ndarray,
                     seed: int = 20261122):
    """Fit scaling and mortality head on source training rows only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int8)
    train_mask = np.asarray(train_mask, dtype=bool)
    if X.ndim != 2 or len(X) != len(y) or len(y) != len(train_mask):
        raise ValueError("X, y, and source-train mask have inconsistent shapes")
    if not train_mask.any() or len(np.unique(y[train_mask])) != 2:
        raise ClinicalReadinessError("source training outcomes require both classes")
    if not np.isfinite(X[train_mask]).all():
        raise ValueError("source training embeddings contain nonfinite values")
    scaler = StandardScaler().fit(X[train_mask])
    probe = LogisticRegression(C=1.0, max_iter=3000, random_state=seed)
    probe.fit(scaler.transform(X[train_mask]), y[train_mask])
    return scaler, probe


def load_model(model_tag: str, device: str):
    import torch
    code_dir = ROOT / "revision_v2_20261117/code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    import decoding_motor_v2 as v2
    return v2.make_model(model_tag, device)


def load_sequence(db: str) -> dict[str, np.ndarray]:
    path = PROC / f"seq_{db}.npz"
    with np.load(path, allow_pickle=True) as source:
        ds = {key: source[key] for key in
              ("V", "B", "H", "L", "stay_id", "subject", "y_mort")}
    ds["split"] = np.load(PROC / f"split_{db}.npy")
    validate_sequence(ds, db)
    return ds


def load_sequence_registered(db: str, paths=None) -> dict[str, np.ndarray]:
    """Load sequence and split arrays through the revision-owned registry."""
    resolved = resolve_cohort_paths(db, verify_exists=True) if paths is None else paths
    with np.load(resolved.sequence_path, allow_pickle=True) as source:
        ds = {key: source[key] for key in
              ("V", "B", "H", "L", "stay_id", "subject", "y_mort")}
    ds["split"] = np.load(resolved.split_path)
    validate_sequence(ds, db)
    return ds


def encode_rows(model, ds: dict[str, np.ndarray], rows: np.ndarray,
                condition: str, device: str, batch_size: int,
                common_ids: tuple[int, ...]) -> np.ndarray:
    import torch
    output = np.empty((len(rows), int(model.d)), dtype=np.float32)
    started = __import__("time").time()
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            take = rows[start:start + batch_size]
            V, B, H = prepare_batch(ds, take, condition, common_ids)
            Vt, Bt, Ht = (torch.from_numpy(x).to(device) for x in (V, B, H))
            E = model.embed(Vt, Bt, Ht, Vt == 0).detach().cpu().numpy().astype(np.float32)
            if not np.isfinite(E).all():
                raise FloatingPointError(f"nonfinite embeddings at rows {start}:{start + len(take)}")
            output[start:start + len(take)] = E
            done = start + len(take)
            if start == 0 or done == len(rows) or done % (batch_size * 50) == 0:
                print(f"[encode] {done}/{len(rows)} ({__import__('time').time()-started:.1f}s)", flush=True)
    return output


def run_encode(args: argparse.Namespace) -> int:
    import torch
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    cache_dir = ENGINE / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_tags = list(dict.fromkeys(args.models))
    needs_common = any(tag in model_tags and condition == "common8_mask"
                       for tag, condition in SPECS)
    needs_no_h0 = any(tag in model_tags and condition == "no_h0_therapy"
                      for tag, condition in SPECS)
    ids, _ = load_common_ids() if needs_common else ((), "")
    common_hashes = common_source_hashes() if needs_common else {}
    no_h0_ids_by_cohort = (load_no_h0_therapy_ids_registered(tuple(args.cohorts))
                           if needs_no_h0 else {})
    source_hashes: dict[str, str] = {}
    split_hashes: dict[str, str] = {}
    source_paths_by_cohort: dict[str, dict[str, str | None]] = {}
    encoder_hash = encoder_code_hash()
    full_engine_hash = sha256_file(Path(__file__))
    torch_version = str(torch.__version__)
    summaries = []
    runtime_by_model_tag: dict[str, dict[str, Any]] = {}
    identity_sets: dict[tuple[str, str], set[str]] = {}
    for model_tag in model_tags:
        conditions = [condition for tag, condition in SPECS if tag == model_tag]
        for condition in conditions:
            model, checkpoint_hash = load_model(model_tag, device)
            # The legacy v2 model module sets 16 threads while importing fm.py.
            # Set and verify the requested 8 threads only after model construction.
            torch.set_num_threads(8)
            torch_num_threads = int(torch.get_num_threads())
            if torch_num_threads != 8:
                raise RuntimeError(
                    f"encoder thread count is {torch_num_threads}; expected 8 after model import")
            runtime = runtime_identity(torch, model, device, torch_num_threads)
            runtime_hash = canonical_json_sha256(runtime)
            runtime_by_model_tag[model_tag] = {
                "runtime_identity": runtime,
                "runtime_identity_sha256": runtime_hash,
            }
            for db in args.cohorts:
                paths = resolve_cohort_paths(db, verify_exists=True)
                if db not in source_hashes:
                    source_hashes[db] = sha256_file(paths.sequence_path)
                    split_hashes[db] = sha256_file(paths.split_path)
                    source_paths_by_cohort[db] = paths.as_metadata()
                ds = load_sequence_registered(db, paths)
                variable_ids = (
                    ids if condition == "common8_mask" else
                    tuple(no_h0_ids_by_cohort[db][name]
                          for name in NO_H0_THERAPY_CANONICAL_VARIABLES)
                    if condition == "no_h0_therapy" else ()
                )
                rows = condition_rows(ds, condition, variable_ids)
                excluded_empty_mask = len(sequence_rows(ds)) - len(rows)
                stays = np.asarray(ds["stay_id"])[rows]
                subjects = np.asarray(ds["subject"]).astype(str)[rows]
                source_y = np.asarray(ds["y_mort"], dtype=np.float32)[rows]
                identity_hash = sha256_ids(db, rows, stays, subjects)
                input_mask_spec = (no_h0_therapy_input_mask_spec(db, variable_ids)
                                   if condition == "no_h0_therapy" else None)
                mask_hash = (canonical_json_sha256(input_mask_spec)
                             if input_mask_spec is not None
                             else input_mask_hash(condition, variable_ids))
                encoding_scope = (
                    "nonempty original rows with at least one common8 token"
                    if condition == "common8_mask" else
                    "nonempty original rows after removing vaso/rrt/sedation tokens with H==0"
                    if condition == "no_h0_therapy" else
                    "all nonempty original sequence rows"
                )
                metadata = {
                    "db": db, "model_tag": model_tag,
                    "input_scope": INPUT_SCOPE,
                    "clinical_input_availability_verified": CLINICAL_INPUT_AVAILABILITY_VERIFIED,
                    "input_availability_limitation": INPUT_AVAILABILITY_LIMITATION,
                    "checkpoint_sha256": checkpoint_hash,
                    "input_condition": condition,
                    "input_mask_sha256": mask_hash,
                    "row_identity_sha256": identity_hash,
                    "source_npz_sha256": source_hashes[db],
                    "split_sha256": split_hashes[db],
                    "encoder_code_sha256": encoder_hash,
                    "engine_code_sha256": full_engine_hash,
                    "runtime_identity": runtime,
                    "runtime_identity_sha256": runtime_hash,
                    "python_version": runtime["python_version"],
                    "torch_version": runtime["torch_version"],
                    "encoding_device": runtime["encoding_device"],
                    "encoding_dtype": runtime["encoding_dtype"],
                    "model_parameter_dtypes": runtime["model_parameter_dtypes"],
                    "torch_num_threads": runtime["torch_num_threads"],
                    **(common_hashes if condition == "common8_mask" else {}),
                    **({"input_mask_spec": input_mask_spec,
                        "condition_definition_sha256":
                            input_mask_spec["condition_definition_sha256"]}
                       if input_mask_spec is not None else {}),
                    "encoding_scope": encoding_scope,
                    "excluded_empty_input_mask_rows": int(excluded_empty_mask),
                }
                path = cache_path(cache_dir, db, model_tag, condition,
                                  checkpoint_hash, identity_hash, mask_hash,
                                  source_hashes[db], encoder_hash, runtime_hash)
                row_metadata = {
                    "db": db, "model_tag": model_tag,
                    "input_condition": condition,
                    "source_npz_sha256": source_hashes[db],
                    "split_sha256": split_hashes[db],
                    "source_path_provenance": source_paths_by_cohort[db],
                    "n_rows": int(len(rows)),
                    "excluded_empty_input_mask_rows": int(excluded_empty_mask),
                    "checkpoint_sha256": checkpoint_hash,
                    "input_mask_sha256": mask_hash,
                    "row_identity_sha256": identity_hash,
                    "encoder_code_sha256": encoder_hash,
                    "runtime_identity": runtime,
                    "runtime_identity_sha256": runtime_hash,
                    "encoding_scope": encoding_scope,
                    **({"input_mask_spec": input_mask_spec,
                        "condition_definition_sha256":
                            input_mask_spec["condition_definition_sha256"]}
                       if input_mask_spec is not None else {}),
                }
                if not len(rows):
                    summaries.append({
                        **row_metadata,
                        "status": "no_rows_with_nonempty_input_mask",
                    })
                    identity_sets[(db, f"{model_tag}_{condition}")] = set()
                    del ds
                    continue
                embedding = load_embedding_cache(path, rows, stays, subjects,
                                                 source_y, metadata)
                if embedding is None:
                    print(f"[encode-start] {model_tag}/{condition}/{db} n={len(rows)} "
                          f"checkpoint={checkpoint_hash} mask={mask_hash} "
                          f"runtime={runtime_hash}", flush=True)
                    embedding = encode_rows(model, ds, rows, condition, device,
                                            args.batch_size, variable_ids)
                    save_embedding_cache(path, rows, stays, subjects, source_y,
                                         embedding, metadata)
                summaries.append({**row_metadata, "cache": str(path)})
                identity_sets[(db, f"{model_tag}_{condition}")] = set(stays.astype(str))
                del embedding, ds
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
    manifest = {
        "mode": "encode", "clinical_ready": False,
        "note": "Embeddings only; no clinical prediction or performance was emitted.",
        "input_scope": INPUT_SCOPE,
        "clinical_input_availability_verified": CLINICAL_INPUT_AVAILABILITY_VERIFIED,
        "input_availability_limitation": INPUT_AVAILABILITY_LIMITATION,
        "encoder_code_sha256": encoder_hash,
        "engine_code_sha256": full_engine_hash,
        "torch_version": torch_version,
        "encoding_device": device,
        "runtime_identity_by_model_tag": runtime_by_model_tag,
        "source_npz_sha256_by_cohort": source_hashes,
        "split_sha256_by_cohort": split_hashes,
        "source_paths_by_cohort": source_paths_by_cohort,
        "models": summaries,
    }
    if needs_no_h0:
        manifest["no_h0_therapy_mapping"] = {
            "canonical_variable_names": list(NO_H0_THERAPY_CANONICAL_VARIABLES),
            "variable_ids_by_cohort": no_h0_ids_by_cohort,
            "condition_definition_sha256": no_h0_therapy_definition_sha256(),
        }
    paired_rows = []
    for db in args.cohorts:
        cohort_sets = {analysis_id: ids_set for (set_db, analysis_id), ids_set in identity_sets.items()
                       if set_db == db}
        if len(cohort_sets) > 1:
            for (analysis_a, ids_a), (analysis_b, ids_b) in combinations(sorted(cohort_sets.items()), 2):
                shared = ids_a & ids_b
                paired_rows.append({"db": db, "analysis_id_a": analysis_a,
                                    "analysis_id_b": analysis_b,
                                    "n_usable_a": len(ids_a), "n_usable_b": len(ids_b),
                                    "n_paired_intersection": len(shared),
                                    "n_excluded_a": len(ids_a - shared),
                                    "n_excluded_b": len(ids_b - shared)})
    manifest["paired_condition_intersection"] = paired_rows
    out = ENGINE / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "embedding_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("V4_ENCODING_COMPLETE", flush=True)
    return 0

def model_conditions(model_tag: str) -> list[str]:
    return [condition for tag, condition in SPECS if tag == model_tag]


def load_encoding_manifest(path: Path | None = None) -> tuple[dict[str, Any], dict[tuple[str, str, str], dict[str, Any]]]:
    manifest_path = (ENGINE / "results/embedding_manifest.json"
                     if path is None else Path(path))
    if not manifest_path.exists():
        raise FileNotFoundError(f"embedding manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("mode") != "encode" or not isinstance(manifest.get("models"), list):
        raise ValueError(f"invalid embedding manifest: {manifest_path}")
    entries: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in manifest["models"]:
        if not isinstance(row, dict):
            raise ValueError("embedding manifest contains a non-object model row")
        key = (str(row.get("db", "")), str(row.get("model_tag", "")),
               str(row.get("input_condition", "")))
        if key in entries:
            raise ValueError(f"embedding manifest contains duplicate cache identity {key}")
        entries[key] = row
    return manifest, entries


def prediction_scope(evidence: str, *, diagnostic_only: bool = False,
                     landmark_ready: bool = True) -> str:
    evidence = str(evidence).lower()
    if diagnostic_only and not landmark_ready:
        return "retrospective_association_unverified_landmark"
    if evidence in {"exact", "operational_proxy"}:
        return "retrospective_landmark_prediction_measurement_time"
    raise ClinicalReadinessError(f"unsupported temporal evidence scope: {evidence}")


def build_prediction_run_protocol(args: argparse.Namespace, eligibility_dir: Path,
                                  encoder_hash: str, engine_hash: str) -> dict[str, Any]:
    model_tags = list(dict.fromkeys(args.models))
    if not args.run_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.run_id):
        raise ValueError("--run-id is required and must use 1-64 letters, digits, dots, underscores, or hyphens")
    reuse_source_probe_run_id = getattr(args, "reuse_source_probes_from_run_id", None)
    if reuse_source_probe_run_id is not None:
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}",
                             reuse_source_probe_run_id)
                or reuse_source_probe_run_id == args.run_id):
            raise ValueError("source-probe reuse requires a distinct valid origin run ID")
    manifest_path = Path(eligibility_dir) / "eligibility_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"temporal eligibility manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    embedding_manifest_path = ENGINE / "results/embedding_manifest.json"
    if not embedding_manifest_path.exists():
        raise FileNotFoundError(
            f"identity-bound embedding manifest is missing: {embedding_manifest_path}")
    embedding_manifest_sha256 = sha256_file(embedding_manifest_path)
    cohorts = manifest.get("cohorts", {})
    table_hashes = {}
    source_hashes = {}
    source_paths_by_cohort = {}
    source_file_hashes_by_cohort = {}
    for db in args.cohorts:
        paths = resolve_cohort_paths(db, verify_exists=True)
        source_paths_by_cohort[db] = paths.as_metadata()
        source_file_hashes_by_cohort[db] = {
            "sequence_npz_sha256": sha256_file(paths.sequence_path),
            "split_npy_sha256": sha256_file(paths.split_path),
        }
        table_path = Path(eligibility_dir) / f"{db}.parquet"
        table_hashes[db] = sha256_file(table_path) if table_path.exists() else None
        entry = cohorts.get(db, {}) if isinstance(cohorts, dict) else {}
        hashes = entry.get("sourcehashes", {}) if isinstance(entry, dict) else {}
        source_hashes[db] = hashes.get("sequence_npz") if isinstance(hashes, dict) else None
    protocol = {
        "run_id": args.run_id,
        "mode": args.mode,
        "models": model_tags,
        "cohorts": list(args.cohorts),
        "encoder_code_sha256": encoder_hash,
        "engine_code_sha256": engine_hash,
        "eligibility_manifest_sha256": sha256_file(manifest_path),
        "embedding_manifest_sha256": embedding_manifest_sha256,
        "eligibility_table_sha256_by_cohort": table_hashes,
        "source_npz_sha256_attested_by_cohort": source_hashes,
        "source_paths_by_cohort": source_paths_by_cohort,
        "source_file_sha256_by_cohort": source_file_hashes_by_cohort,
        "source_probe_reuse_run_id": reuse_source_probe_run_id,
        "checkpoint_sha256_by_model": {
            model_tag: sha256_file(MODELS / f"{model_tag}.pt")
            for model_tag in model_tags
        },
        "input_scope": INPUT_SCOPE,
        "clinical_input_availability_verified": CLINICAL_INPUT_AVAILABILITY_VERIFIED,
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    protocol["run_protocol_sha256"] = hashlib.sha256(canonical).hexdigest()
    return protocol


def prediction_run_directory(protocol: dict[str, Any],
                             results_root: Path | None = None) -> Path:
    run_id = str(protocol.get("run_id", ""))
    digest = str(protocol.get("run_protocol_sha256", ""))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", run_id):
        raise ValueError("invalid prediction run ID")
    if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
        raise ValueError("prediction protocol requires a SHA256 identity")
    base = ENGINE / "results/prediction_runs" if results_root is None else Path(results_root)
    return base / f"{run_id}-{digest[:16].lower()}"


def run_predict(args: argparse.Namespace) -> int:
    if args.eligibility_dir is None:
        raise ClinicalReadinessError("--mode predict requires --eligibility-dir")
    eligibility_dir = Path(args.eligibility_dir)
    diagnostic_only = args.mode == "diagnostic"
    if any(db != "mimic" for db in args.cohorts) and args.cohorts[0] != "mimic":
        raise ClinicalReadinessError("MIMIC must be first when external predictions are requested")
    reuse_source_probe_run_id = getattr(args, "reuse_source_probes_from_run_id", None)
    if reuse_source_probe_run_id is not None:
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}",
                             reuse_source_probe_run_id)
                or reuse_source_probe_run_id == args.run_id):
            raise ValueError("source-probe reuse requires a distinct valid origin run ID")
    manifest_rows = []
    prediction_ids: dict[tuple[str, str], dict[str, tuple[str, int]]] = {}
    prediction_frames: dict[str, list[pd.DataFrame]] = {}
    fitted_source_probes: dict[tuple[str, str], dict[str, Any]] = {}
    readiness_rows = []
    model_tags = list(dict.fromkeys(args.models))
    needs_common = any(tag in model_tags and condition == "common8_mask"
                       for tag, condition in SPECS)
    needs_no_h0 = any(tag in model_tags and condition == "no_h0_therapy"
                      for tag, condition in SPECS)
    ids, _ = load_common_ids() if needs_common else ((), "")
    common_hashes = common_source_hashes() if needs_common else {}
    no_h0_ids_by_cohort = (load_no_h0_therapy_ids_registered(tuple(args.cohorts))
                           if needs_no_h0 else {})
    embedding_manifest, embedding_entries = load_encoding_manifest()
    encoder_hash = encoder_code_hash()
    full_engine_hash = sha256_file(Path(__file__))
    run_protocol = build_prediction_run_protocol(
        args, eligibility_dir, encoder_hash, full_engine_hash)
    run_protocol_hash = run_protocol["run_protocol_sha256"]
    prediction_dir = prediction_run_directory(run_protocol)
    if prediction_dir.exists() and any(prediction_dir.iterdir()):
        raise FileExistsError(
            f"prediction run output already exists and will not be overwritten: {prediction_dir}; "
            "choose a new --run-id")
    prediction_dir.mkdir(parents=True, exist_ok=True)
    source_probe_reuse_records = []
    source_probe_reuse_manifest_path = (
        prediction_dir / "source_probe_reuse_manifest.json"
        if reuse_source_probe_run_id is not None else None
    )
    if source_probe_reuse_manifest_path is not None:
        atomic_json_dump({
            "run_id": args.run_id,
            "run_protocol_sha256": run_protocol_hash,
            "source_probe_reuse_run_id": reuse_source_probe_run_id,
            "embedding_manifest_sha256": sha256_file(
                ENGINE / "results/embedding_manifest.json"),
            "artifacts": source_probe_reuse_records,
        }, source_probe_reuse_manifest_path)
    for db in args.cohorts:
        # Validate readiness before loading model outcomes or writing any table.
        try:
            eligible, eligibility_entry = load_ready_eligibility(
                db, eligibility_dir, diagnostic_only=diagnostic_only)
        except ClinicalReadinessError as exc:
            readiness_rows.append({"db": db, "status": "not_ready", "n_predictions": 0,
                                   "reason": str(exc)[:300]})
            continue
        source_paths = resolve_cohort_paths(db, verify_exists=True)
        source_npz_hash = sha256_file(source_paths.sequence_path)
        assert_sequence_npz_attestation(db, eligibility_entry, source_npz_hash)
        landmark_ready = bool(eligibility_entry.get("retrospective_landmark_prediction_allowed"))
        scope = prediction_scope(eligibility_entry.get("evidence"),
                                diagnostic_only=diagnostic_only,
                                landmark_ready=landmark_ready)
        readiness_rows.append({"db": db, "status": "ready",
                               "temporal_evidence": eligibility_entry.get("evidence"),
                               "prediction_scope": scope,
                               "input_scope": INPUT_SCOPE,
                               "clinical_input_availability_verified": CLINICAL_INPUT_AVAILABILITY_VERIFIED,
                               "retrospective_landmark_prediction_ready": landmark_ready,
                               "retrospective_association_ready": bool(
                                   eligibility_entry.get("retrospective_diagnostic_allowed"))})
        ds = load_sequence_registered(db, source_paths)
        split = np.asarray(ds["split"], dtype=np.int8)
        split_hash = sha256_file(source_paths.split_path)
        for model_tag in model_tags:
            for condition in model_conditions(model_tag):
                checkpoint_hash = sha256_file(MODELS / f"{model_tag}.pt")
                variable_ids = (
                    ids if condition == "common8_mask" else
                    tuple(no_h0_ids_by_cohort[db][name]
                          for name in NO_H0_THERAPY_CANONICAL_VARIABLES)
                    if condition == "no_h0_therapy" else ()
                )
                input_mask_spec = (no_h0_therapy_input_mask_spec(db, variable_ids)
                                   if condition == "no_h0_therapy" else None)
                mask_hash = (canonical_json_sha256(input_mask_spec)
                             if input_mask_spec is not None
                             else input_mask_hash(condition, variable_ids))
                rows = condition_rows(ds, condition, variable_ids)
                matched = align_eligible_rows(db, ds, eligible, rows,
                                              condition=condition,
                                              common_ids=variable_ids)
                matched["split"] = split[matched["rows"].to_numpy()]
                source_label_changed = 0
                source_label_missing_before = 0
                condition_excluded = int(len(eligible) - len(matched))
                analysis_id = f"{model_tag}_{condition}"
                if not len(matched):
                    manifest_rows.append({
                        "analysis_id": analysis_id, "db": db,
                        "mode": args.mode,
                        "run_id": args.run_id,
                        "run_protocol_sha256": run_protocol_hash,
                        "model_tag": model_tag, "input_condition": condition,
                        "checkpoint_sha256": checkpoint_hash,
                        "input_mask_sha256": mask_hash,
                        "eligibility_source_sha256": eligibility_entry["sequence_npz_sha256"],
                        "eligibility_table_sha256": eligibility_entry.get("eligibility_table_sha256"),
                        "eligibility_manifest_sha256": eligibility_entry.get("eligibility_manifest_sha256"),
                        "source_npz_sha256": source_npz_hash,
                        "temporal_evidence": eligibility_entry.get("evidence"),
                        "evidence_basis": eligibility_entry.get("evidence_basis"),
                        "input_scope": INPUT_SCOPE,
                        "clinical_input_availability_verified": CLINICAL_INPUT_AVAILABILITY_VERIFIED,
                        "input_availability_limitation": INPUT_AVAILABILITY_LIMITATION,
                        "prediction_scope": scope, "n_predictions": 0,
                        "n_prediction_rows": 0,
                        "n_eligible_excluded_empty_input_mask": condition_excluded,
                        "n_source_label_changed": source_label_changed,
                        "n_source_label_missing_before": source_label_missing_before,
                        "clinical_ready": False,
                        "retrospective_landmark_prediction_allowed": landmark_ready and not diagnostic_only,
                        "retrospective_landmark_prediction_ready": landmark_ready,
                        "retrospective_association_ready": bool(
                            eligibility_entry.get("retrospective_diagnostic_allowed")),
                        "diagnostic_ready": bool(eligibility_entry.get("retrospective_diagnostic_allowed"))
                                           and diagnostic_only,
                        "status": "no_eligible_rows_with_nonempty_input_mask",
                    })
                    prediction_ids[(db, analysis_id)] = {}
                    continue
                if db == "mimic" and (not (matched["split"] == 0).any()
                                       or not (matched["split"] == 2).any()):
                    raise ClinicalReadinessError("mimic: eligible source train and test rows are required")
                identity_hash = sha256_ids(db, rows, np.asarray(ds["stay_id"])[rows],
                                           np.asarray(ds["subject"]).astype(str)[rows])
                encoding_entry = embedding_entries.get((db, model_tag, condition))
                if not isinstance(encoding_entry, dict):
                    raise FileNotFoundError(
                        f"embedding manifest has no cache row for {db}/{model_tag}/{condition}")
                runtime = encoding_entry.get("runtime_identity")
                runtime_hash = str(encoding_entry.get("runtime_identity_sha256", ""))
                if (not isinstance(runtime, dict)
                        or canonical_json_sha256(runtime) != runtime_hash):
                    raise ValueError("embedding manifest runtime identity/hash is invalid")
                if (encoding_entry.get("checkpoint_sha256") != checkpoint_hash
                        or encoding_entry.get("input_mask_sha256") != mask_hash
                        or encoding_entry.get("row_identity_sha256") != identity_hash
                        or encoding_entry.get("source_npz_sha256") != source_npz_hash
                        or encoding_entry.get("split_sha256") != split_hash
                        or encoding_entry.get("encoder_code_sha256") != encoder_hash):
                    raise ValueError("embedding manifest cache identity does not match current inputs")
                cache = Path(str(encoding_entry.get("cache", "")))
                expected_cache = cache_path(ENGINE / "cache", db, model_tag, condition,
                                            checkpoint_hash, identity_hash, mask_hash,
                                            source_npz_hash, encoder_hash, runtime_hash)
                if cache.resolve() != expected_cache.resolve():
                    raise ValueError("embedding manifest cache path does not match its runtime identity")
                if cache.parent.resolve() != (ENGINE / "cache").resolve():
                    raise ValueError("embedding manifest cache path escapes engine/cache")
                if not cache.exists():
                    raise FileNotFoundError(f"ID-bound embedding cache missing: {cache}")
                with np.load(cache, allow_pickle=False) as z:
                    E = z["embedding"].astype(np.float32, copy=False)
                    cache_rows = z["rows"]
                    cache_stay = z["stay_id"].astype(str)
                    cache_subject = z["subject"].astype(str)
                    cache_source_y = z["source_y"].astype(np.float32)
                    cache_meta = json.loads(str(z["metadata_json"].item()))
                encoding_scope = (
                    "nonempty original rows with at least one common8 token"
                    if condition == "common8_mask" else
                    "nonempty original rows after removing vaso/rrt/sedation tokens with H==0"
                    if condition == "no_h0_therapy" else
                    "all nonempty original sequence rows"
                )
                expected_meta = {
                    "db": db, "model_tag": model_tag,
                    "input_scope": INPUT_SCOPE,
                    "clinical_input_availability_verified": CLINICAL_INPUT_AVAILABILITY_VERIFIED,
                    "input_availability_limitation": INPUT_AVAILABILITY_LIMITATION,
                    "checkpoint_sha256": checkpoint_hash,
                    "input_condition": condition,
                    "input_mask_sha256": mask_hash,
                    "row_identity_sha256": identity_hash,
                    "source_npz_sha256": source_npz_hash,
                    "split_sha256": split_hash,
                    "encoder_code_sha256": encoder_hash,
                    "runtime_identity": runtime,
                    "runtime_identity_sha256": runtime_hash,
                    "python_version": runtime["python_version"],
                    "torch_version": runtime["torch_version"],
                    "encoding_device": runtime["encoding_device"],
                    "encoding_dtype": runtime["encoding_dtype"],
                    "model_parameter_dtypes": runtime["model_parameter_dtypes"],
                    "torch_num_threads": runtime["torch_num_threads"],
                    **(common_hashes if condition == "common8_mask" else {}),
                    **({"input_mask_spec": input_mask_spec,
                        "condition_definition_sha256":
                            input_mask_spec["condition_definition_sha256"]}
                       if input_mask_spec is not None else {}),
                    "encoding_scope": encoding_scope,
                    "excluded_empty_input_mask_rows": int(len(sequence_rows(ds)) - len(rows)),
                }
                if (not np.array_equal(cache_rows, rows)
                        or not np.array_equal(cache_stay, np.asarray(ds["stay_id"])[rows].astype(str))
                        or not np.array_equal(cache_subject, np.asarray(ds["subject"]).astype(str)[rows])
                        or not np.array_equal(cache_source_y,
                                              np.asarray(ds["y_mort"], dtype=np.float32)[rows],
                                              equal_nan=True)
                        or not cache_metadata_matches(cache_meta, expected_meta)):
                    raise ValueError(f"embedding cache provenance mismatch: {cache}")
                pos = {int(row): i for i, row in enumerate(rows)}
                aligned_E = E[np.asarray([pos[int(r)] for r in matched["rows"]], dtype=np.int64)]
                y = matched["y_mort"].to_numpy(dtype=np.int8)
                groups = matched["subject"].to_numpy(dtype=str)
                if db == "mimic":
                    train_mask = matched["split"].to_numpy() == 0
                    test_mask = matched["split"].to_numpy() == 2
                    train_groups = set(groups[train_mask])
                    test_groups = set(groups[test_mask])
                    if train_groups & test_groups:
                        raise ClinicalReadinessError("mimic: source train/test patients overlap")
                else:
                    train_mask = np.zeros(len(matched), dtype=bool)
                    test_mask = np.ones(len(matched), dtype=bool)
                if db == "mimic":
                    if len(np.unique(y[train_mask])) != 2:
                        raise ClinicalReadinessError("mimic: source training outcomes require both classes")
                    source_train_rows = matched.loc[train_mask, "rows"].to_numpy(dtype=np.int64)
                    source_train_identity_hash = sha256_ids(
                        db, source_train_rows,
                        np.asarray(ds["stay_id"])[source_train_rows],
                        np.asarray(ds["subject"]).astype(str)[source_train_rows])
                    source_train_label_hash = hashlib.sha256(
                        np.asarray(y[train_mask], dtype=np.int8).tobytes()).hexdigest()
                    probe_metadata = {
                        "run_id": args.run_id,
                        "model_tag": model_tag,
                        "input_condition": condition,
                        "checkpoint_sha256": checkpoint_hash,
                        "input_mask_sha256": mask_hash,
                        **({"input_mask_spec": input_mask_spec}
                           if input_mask_spec is not None else {}),
                        "eligibility_manifest_sha256": eligibility_entry["eligibility_manifest_sha256"],
                        "eligibility_table_sha256": eligibility_entry["eligibility_table_sha256"],
                        "source_npz_sha256": source_npz_hash,
                        "source_train_identity_sha256": source_train_identity_hash,
                        "source_train_label_sha256": source_train_label_hash,
                        "runtime_identity_sha256": runtime_hash,
                        "source_train_n": int(train_mask.sum()),
                        "source_train_patient_n": len(train_groups),
                        "prediction_scope": scope,
                        "temporal_evidence": eligibility_entry.get("evidence"),
                        "input_scope": INPUT_SCOPE,
                        "clinical_input_availability_verified": CLINICAL_INPUT_AVAILABILITY_VERIFIED,
                    }
                    if reuse_source_probe_run_id is not None:
                        fitted = load_reusable_source_probe(
                            probe_metadata, reuse_source_probe_run_id,
                            expected_feature_dim=aligned_E.shape[1])
                        scaler, probe = fitted["scaler"], fitted["probe"]
                        probe_path = Path(fitted["artifact_path"])
                        source_probe_reuse_records.append({
                            "origin_run_id": reuse_source_probe_run_id,
                            "model_tag": model_tag,
                            "input_condition": condition,
                            "artifact_path": str(probe_path),
                            "artifact_sha256": fitted["artifact_sha256"],
                            "checkpoint_sha256": checkpoint_hash,
                            "input_mask_sha256": mask_hash,
                            "eligibility_manifest_sha256":
                                probe_metadata["eligibility_manifest_sha256"],
                            "eligibility_table_sha256":
                                probe_metadata["eligibility_table_sha256"],
                            "source_npz_sha256": source_npz_hash,
                            "source_train_identity_sha256":
                                source_train_identity_hash,
                            "source_train_label_sha256":
                                source_train_label_hash,
                            "runtime_identity_sha256": runtime_hash,
                            "prediction_scope": scope,
                        })
                        atomic_json_dump({
                            "run_id": args.run_id,
                            "run_protocol_sha256": run_protocol_hash,
                            "source_probe_reuse_run_id":
                                reuse_source_probe_run_id,
                            "embedding_manifest_sha256":
                                run_protocol["embedding_manifest_sha256"],
                            "artifacts": source_probe_reuse_records,
                        }, source_probe_reuse_manifest_path)
                        print(
                            f"[source_probe] reused {model_tag}/{condition} "
                            f"from {reuse_source_probe_run_id} "
                            f"sha256={fitted['artifact_sha256']}", flush=True)
                    else:
                        scaler, probe = fit_source_probe(aligned_E, y, train_mask)
                        probe_path = source_probe_artifact_path(probe_metadata)
                        fitted = {"scaler": scaler, "probe": probe, **probe_metadata}
                        atomic_pickle_dump(fitted, probe_path)
                        fitted = {
                            **fitted,
                            "artifact_path": str(probe_path),
                            "artifact_sha256": sha256_file(probe_path),
                            "reused_from_run_id": args.run_id,
                        }
                    fitted_source_probes[(model_tag, condition)] = fitted
                else:
                    fitted = fitted_source_probes.get((model_tag, condition))
                    if fitted is None:
                        manifest_rows.append({
                            "analysis_id": analysis_id, "db": db,
                            "mode": args.mode,
                            "run_id": args.run_id,
                            "run_protocol_sha256": run_protocol_hash,
                            "model_tag": model_tag, "input_condition": condition,
                            "checkpoint_sha256": checkpoint_hash,
                            "input_mask_sha256": mask_hash,
                            "input_scope": INPUT_SCOPE,
                            "clinical_input_availability_verified": CLINICAL_INPUT_AVAILABILITY_VERIFIED,
                            "prediction_scope": scope,
                            "n_predictions": 0, "n_prediction_rows": 0,
                            "clinical_ready": False,
                            "retrospective_landmark_prediction_allowed": landmark_ready and not diagnostic_only,
                            "retrospective_landmark_prediction_ready": landmark_ready,
                            "retrospective_association_ready": bool(
                                eligibility_entry.get("retrospective_diagnostic_allowed")),
                            "n_source_label_changed": source_label_changed,
                            "n_source_label_missing_before": source_label_missing_before,
                            "status": "blocked_current_run_has_no_attested_mimic_source_probe",
                        })
                        prediction_ids[(db, analysis_id)] = {}
                        del E, aligned_E
                        continue
                    expected_probe_run_id = (
                        reuse_source_probe_run_id or args.run_id)
                    if (fitted.get("run_id") != expected_probe_run_id
                            or fitted["checkpoint_sha256"] != checkpoint_hash
                            or fitted.get("runtime_identity_sha256") != runtime_hash
                            or not source_probe_mask_compatible(
                                fitted, condition, mask_hash, input_mask_spec)):
                        raise ValueError(
                            "source probe run/checkpoint/condition/runtime/mask mismatch")
                    scaler, probe = fitted["scaler"], fitted["probe"]
                    probe_path = Path(fitted["artifact_path"])
                p = probe.predict_proba(scaler.transform(aligned_E[test_mask]))[:, 1]
                test = matched.loc[test_mask].copy()
                test_rows = test["rows"].to_numpy(dtype=np.int64)
                output = pd.DataFrame({
                    "db": db,
                    "analysis_id": analysis_id,
                    "stay_id": test["stay_id"].to_numpy(),
                    "subject": test["subject"].to_numpy(dtype=str),
                    "rows": test_rows,
                    "split": test["split"].to_numpy(dtype=np.int8),
                    "y": test["y_mort"].to_numpy(dtype=np.int8),
                    "p": p,
                    "model_tag": model_tag,
                    "input_condition": condition,
                    "endpoint": test["endpoint"].to_numpy(dtype=str),
                    "role": "evaluation",
                    "prediction_scope": scope,
                    "input_scope": INPUT_SCOPE,
                    "clinical_input_availability_verified": CLINICAL_INPUT_AVAILABILITY_VERIFIED,
                    "input_availability_limitation": INPUT_AVAILABILITY_LIMITATION,
                    "runtime_identity_sha256": runtime_hash,
                    "eligibility_manifest_sha256": eligibility_entry["eligibility_manifest_sha256"],
                    "eligibility_table_sha256": eligibility_entry["eligibility_table_sha256"],
                    "temporal_evidence": eligibility_entry.get("evidence"),
                    "evidence_basis": eligibility_entry.get("evidence_basis", eligibility_entry.get("evidence")),
                })
                prediction_frames.setdefault(analysis_id, []).append(output)
                manifest_rows.append({
                    "analysis_id": analysis_id, "db": db,
                    "mode": args.mode,
                    "run_id": args.run_id,
                    "run_protocol_sha256": run_protocol_hash,
                    "model_tag": model_tag, "input_condition": condition,
                    "checkpoint_sha256": checkpoint_hash,
                    "input_mask_sha256": mask_hash,
                    "eligibility_source_sha256": eligibility_entry["sequence_npz_sha256"],
                    "temporal_evidence": eligibility_entry.get("evidence"),
                    "evidence_basis": eligibility_entry.get("evidence_basis"),
                    "input_scope": INPUT_SCOPE,
                    "clinical_input_availability_verified": CLINICAL_INPUT_AVAILABILITY_VERIFIED,
                    "input_availability_limitation": INPUT_AVAILABILITY_LIMITATION,
                    "eligibility_table_sha256": eligibility_entry.get("eligibility_table_sha256"),
                    "eligibility_manifest_sha256": eligibility_entry.get("eligibility_manifest_sha256"),
                    "source_npz_sha256": source_npz_hash,
                    "split_sha256": split_hash,
                    "encoder_code_sha256": encoder_hash,
                    "engine_code_sha256": full_engine_hash,
                    "cache_engine_code_sha256": cache_meta.get("engine_code_sha256"),
                    "cache_torch_version": cache_meta.get("torch_version"),
                    "cache_encoding_device": cache_meta.get("encoding_device"),
                    "cache_encoding_dtype": cache_meta.get("encoding_dtype"),
                    "cache_model_parameter_dtypes": cache_meta.get("model_parameter_dtypes"),
                    "cache_torch_num_threads": cache_meta.get("torch_num_threads"),
                    "cache_runtime_identity_sha256": runtime_hash,
                    "cache_runtime_identity": runtime,
                    "input_mask_spec": input_mask_spec,
                    "prediction_scope": scope,
                    "role": "evaluation",
                    "n_predictions": int(len(output)),
                    "n_prediction_rows": int(len(output)),
                    "n_eligible_excluded_empty_input_mask": condition_excluded,
                    "n_source_label_changed": source_label_changed,
                    "n_source_label_missing_before": source_label_missing_before,
                    "n_prediction_ids_sha256": sha256_ids(
                        db, test_rows, output["stay_id"].to_numpy(),
                        output["subject"].to_numpy()),
                    "clinical_ready": False,
                    "real_time_available_landmark_prediction": False,
                    "retrospective_landmark_prediction_allowed": landmark_ready and not diagnostic_only,
                    "retrospective_landmark_prediction_ready": landmark_ready,
                    "retrospective_association_ready": bool(
                        eligibility_entry.get("retrospective_diagnostic_allowed")),
                    "diagnostic_ready": bool(eligibility_entry.get("retrospective_diagnostic_allowed"))
                                        and diagnostic_only,
                    "source_probe_artifact": str(probe_path),
                    "source_probe_sha256": sha256_file(probe_path),
                    "source_probe_artifact_sha256": fitted["artifact_sha256"],
                    "source_probe_origin_run_id":
                        fitted.get("reused_from_run_id", args.run_id),
                    "source_probe_reused_from_prior_run":
                        fitted.get("reused_from_run_id", args.run_id) != args.run_id,
                    "pretraining_target_exposed": model_tag in {"big50m", "big50m_full"},
                })
                prediction_ids[(db, analysis_id)] = {
                    str(row.stay_id): (str(row.subject), int(row.y))
                    for row in output.itertuples(index=False)
                }
                print(f"[predictions] {analysis_id}/{db}: n={len(output)}", flush=True)
                del E, aligned_E
        del ds
    for analysis_id, frames in prediction_frames.items():
        path = prediction_dir / f"{analysis_id}.parquet"
        combined = pd.concat(frames, ignore_index=True)
        combined.to_parquet(path, index=False)
        file_hash = sha256_file(path)
        for row in manifest_rows:
            if row.get("analysis_id") == analysis_id and row.get("n_predictions", 0) > 0:
                row["prediction_file"] = str(path.resolve())
                row["prediction_file_sha256"] = file_hash
                cohort_rows = combined.loc[combined["db"].astype(str) == str(row["db"])]
                row["n_prediction_rows"] = int(len(cohort_rows))
                row["ordered_row_digest_sha256"] = sha256_ordered_prediction_rows(cohort_rows)
    manifest_path = prediction_dir / "prediction_manifest.json"
    paired = []
    landmark_ready_by_cohort = {}
    diagnostic_ready_by_cohort = {}
    for db in args.cohorts:
        condition_maps = {analysis_id: id_map for (map_db, analysis_id), id_map in prediction_ids.items()
                          if map_db == db}
        db_manifest = [entry for entry in manifest_rows if entry["db"] == db]
        landmark_ready_by_cohort[db] = (bool(db_manifest)
                                        and all(x.get("retrospective_landmark_prediction_ready", False)
                                                for x in db_manifest))
        diagnostic_ready_by_cohort[db] = (bool(db_manifest)
                                           and all(x.get("diagnostic_ready", False)
                                                   for x in db_manifest))
        if len(condition_maps) > 1:
            for (analysis_a, values_a), (analysis_b, values_b) in combinations(sorted(condition_maps.items()), 2):
                common = set(values_a) & set(values_b)
                labels_match = all(values_a[stay] == values_b[stay] for stay in common)
                paired.append({"db": db, "analysis_id_a": analysis_a,
                               "analysis_id_b": analysis_b,
                               "n_predictions_a": len(values_a),
                               "n_predictions_b": len(values_b),
                               "n_paired_intersection": len(common),
                               "n_excluded_a": len(set(values_a) - common),
                               "n_excluded_b": len(set(values_b) - common),
                               "paired_subject_and_label_identity_verified": labels_match})
    prediction_manifest = {
        "encoder_code_sha256": encoder_hash,
        "engine_code_sha256": full_engine_hash,
        "run_id": args.run_id,
        "run_protocol_sha256": run_protocol_hash,
        "run_protocol": run_protocol,
        "mode": args.mode,
        "input_scope": INPUT_SCOPE,
        "clinical_input_availability_verified": CLINICAL_INPUT_AVAILABILITY_VERIFIED,
        "input_availability_limitation": INPUT_AVAILABILITY_LIMITATION,
        "real_time_available_landmark_prediction_by_cohort": {
            db: False for db in args.cohorts
        },
        "retrospective_landmark_prediction_by_cohort": {
            db: landmark_ready_by_cohort.get(db, False) for db in args.cohorts
        },
        "retrospective_diagnostic_by_cohort": ({db: diagnostic_ready_by_cohort.get(db, False)
                                                  for db in args.cohorts}
                                                  if diagnostic_only else {}),
        "readiness": readiness_rows,
        "predictions": manifest_rows,
        "paired_condition_intersection": paired,
        "source_probe_reuse_manifest": (
            str(source_probe_reuse_manifest_path)
            if source_probe_reuse_manifest_path is not None else None
        ),
    }
    manifest_path.write_text(json.dumps(prediction_manifest, indent=2))
    print(f"V4_PREDICTIONS_COMPLETE {manifest_path}", flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("encode", "predict", "diagnostic"), required=True)
    parser.add_argument("--models", nargs="+", choices=sorted({x[0] for x in SPECS}),
                        default=list(dict.fromkeys(x[0] for x in SPECS)))
    parser.add_argument("--cohorts", nargs="+", choices=registered_cohorts(),
                        default=["mimic"])
    parser.add_argument("--eligibility-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=None,
                        help="required for prediction modes; outputs are immutable and run-scoped")
    parser.add_argument(
        "--reuse-source-probes-from-run-id", default=None,
        help="reuse source-fit probes only after validating their full scientific identity")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if args.mode == "encode":
        return run_encode(args)
    return run_predict(args)


if __name__ == "__main__":
    raise SystemExit(main())
