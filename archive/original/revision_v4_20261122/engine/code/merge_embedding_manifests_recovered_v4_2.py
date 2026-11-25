#!/usr/bin/env python3
"""Validate and atomically merge the frozen 20-row and reviewed 21-row manifests."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
ENGINE = REV / "engine"
CACHE_ROOT = ENGINE / "cache"
SNAPSHOTS = ENGINE / "code_snapshots"
BASE_MANIFEST = SNAPSHOTS / "embedding_manifest_mimic_eicu_core_plus_big20_20261123.json"
NEW_COMPONENT_SNAPSHOT = SNAPSHOTS / "embedding_manifest_recovered_core_newids_v4_2_20261123.json"
LIVE_MANIFEST = ENGINE / "results/embedding_manifest.json"
BASE_MANIFEST_SHA256 = "3ccc3927e7bf7e6a9d31be1136c2870ad8c9fd51bcbb2240424b2a4ff005bb58"
EXPECTED_ENCODER_SHA256 = "59c3f4059662fb9e1d81baf20120c8668a761a926f5cab5ff4e654f7c16c1abc"
EXPECTED_INTEGRATED_ENGINE_SHA256 = "ef70b4e857ce4a02480b7453a514bdfbf4f2a5e7aa4dd033455851919e269768"
NEW_COHORTS = (
    "nwicu_discharge_v4_1", "sicdb_zero_offset_v4_2", "eicu_semantic_v4_2",
)
CORE_SPECS = (
    ("main20k", "real"),
    ("main20k", "constant"),
    ("main20k", "common8_mask"),
    ("main20k", "no_h0_therapy"),
    ("abl_const20k", "real"),
    ("abl_const20k", "constant"),
    ("common20k", "common8_mask"),
)
ALL_BASE_SPECS = CORE_SPECS + (
    ("big50m", "real"),
    ("big50m_full", "real"),
    ("big50m_full", "common8_mask"),
)
METADATA_FIELDS = (
    "db", "model_tag", "input_condition", "checkpoint_sha256",
    "input_mask_sha256", "row_identity_sha256", "source_npz_sha256",
    "split_sha256", "encoder_code_sha256", "runtime_identity_sha256",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_key(row: dict) -> tuple[str, str, str]:
    return (str(row.get("db", "")), str(row.get("model_tag", "")),
            str(row.get("input_condition", "")))


def verify_component_rows(manifest: dict, expected_keys: set[tuple[str, str, str]],
                          label: str) -> list[dict]:
    rows = manifest.get("models")
    if not isinstance(rows, list):
        raise ValueError(f"{label}: manifest models must be a list")
    keyed: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{label}: manifest row must be an object")
        key = identity_key(row)
        if key in keyed:
            raise ValueError(f"{label}: duplicate manifest key {key}")
        if not row.get("cache") or row.get("status") not in (None, "complete"):
            raise ValueError(f"{label}: cache row is incomplete for {key}")
        if int(row.get("n_rows", 0)) <= 0:
            raise ValueError(f"{label}: cache row has no encoded rows for {key}")
        if row.get("encoder_code_sha256") != EXPECTED_ENCODER_SHA256:
            raise ValueError(f"{label}: encoder hash mismatch for {key}")
        keyed[key] = row
    actual = set(keyed)
    if actual != expected_keys:
        missing = sorted(expected_keys - actual)
        unexpected = sorted(actual - expected_keys)
        raise ValueError(f"{label}: key set mismatch; missing={missing}, unexpected={unexpected}")
    return rows


def verify_cache_file(row: dict, component_sha256: str) -> dict:
    key = identity_key(row)
    cache_path = Path(str(row["cache"]))
    if not cache_path.is_absolute():
        raise ValueError(f"cache path must be absolute for {key}: {cache_path}")
    resolved = cache_path.resolve(strict=True)
    if not resolved.is_relative_to(CACHE_ROOT.resolve()):
        raise ValueError(f"cache path escapes engine/cache for {key}: {resolved}")
    with np.load(resolved, allow_pickle=False) as arrays:
        if "metadata_json" not in arrays.files:
            raise ValueError(f"cache metadata is missing for {key}: {resolved}")
        metadata = json.loads(str(arrays["metadata_json"].item()))
    for field in METADATA_FIELDS:
        if metadata.get(field) != row.get(field):
            raise ValueError(f"cache/manifest mismatch for {key}: {field}")
    if row.get("input_condition") == "no_h0_therapy":
        if (metadata.get("input_mask_spec") != row.get("input_mask_spec")
                or metadata.get("condition_definition_sha256")
                != row.get("condition_definition_sha256")):
            raise ValueError(f"cache/manifest no-H0 rule mismatch for {key}")
    cache_sha256 = sha256_file(resolved)
    return {
        "key": list(key),
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": cache_sha256,
        "component_manifest_sha256": component_sha256,
    }


def merge_dicts(old: dict, new: dict, label: str) -> dict:
    merged = dict(old)
    for key, value in new.items():
        if key in merged and merged[key] != value:
            raise ValueError(f"{label}: conflicting values for {key}")
        merged[key] = value
    return merged


def atomic_write(path: Path, data: bytes, *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                          dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


if sha256_file(BASE_MANIFEST) != BASE_MANIFEST_SHA256:
    raise ValueError("frozen 20-row component SHA256 changed")
base_sha256 = sha256_file(BASE_MANIFEST)
if not NEW_COMPONENT_SNAPSHOT.is_file():
    raise FileNotFoundError(f"new 21-row component snapshot is missing: {NEW_COMPONENT_SNAPSHOT}")
new_component_sha256 = sha256_file(NEW_COMPONENT_SNAPSHOT)
if sha256_file(LIVE_MANIFEST) != new_component_sha256:
    raise ValueError("live manifest is not the frozen new 21-row component")
base_manifest = json.loads(BASE_MANIFEST.read_text())
new_manifest = json.loads(NEW_COMPONENT_SNAPSHOT.read_text())
if (base_manifest.get("encoder_code_sha256") != EXPECTED_ENCODER_SHA256
        or new_manifest.get("encoder_code_sha256") != EXPECTED_ENCODER_SHA256):
    raise ValueError("component encoder hashes do not match the frozen encoder identity")
if new_manifest.get("engine_code_sha256") != EXPECTED_INTEGRATED_ENGINE_SHA256:
    raise ValueError("new component was not produced by the verified integrated engine snapshot")
for field in (
    "mode", "input_scope", "clinical_input_availability_verified",
    "input_availability_limitation", "encoder_code_sha256", "torch_version",
    "encoding_device",
):
    if base_manifest.get(field) != new_manifest.get(field):
        raise ValueError(f"manifest component metadata differs: {field}")

base_keys = {
    (db, model, condition)
    for db in ("mimic", "eicu")
    for model, condition in ALL_BASE_SPECS
}
new_keys = {
    (db, model, condition)
    for db in NEW_COHORTS
    for model, condition in CORE_SPECS
}
base_rows = verify_component_rows(base_manifest, base_keys, "frozen 20-row component")
new_rows = verify_component_rows(new_manifest, new_keys, "new 21-row component")
all_keys = [identity_key(row) for row in base_rows + new_rows]
if len(all_keys) != len(set(all_keys)):
    raise ValueError("duplicate cache identity across manifest components")
if len(all_keys) != 41:
    raise ValueError(f"expected 41 component rows, found {len(all_keys)}")

base_runtime = base_manifest.get("runtime_identity_by_model_tag", {})
new_runtime = new_manifest.get("runtime_identity_by_model_tag", {})
for model in sorted(set(base_runtime) & set(new_runtime)):
    if base_runtime[model] != new_runtime[model]:
        raise ValueError(f"runtime identity differs across components for {model}")
runtime_by_model = merge_dicts(base_runtime, new_runtime, "runtime identity")

source_hashes = merge_dicts(
    base_manifest.get("source_npz_sha256_by_cohort", {}),
    new_manifest.get("source_npz_sha256_by_cohort", {}),
    "source NPZ hash",
)
base_no_h0 = base_manifest.get("no_h0_therapy_mapping", {})
new_no_h0 = new_manifest.get("no_h0_therapy_mapping", {})
if (base_no_h0.get("canonical_variable_names")
        != new_no_h0.get("canonical_variable_names")
        or base_no_h0.get("condition_definition_sha256")
        != new_no_h0.get("condition_definition_sha256")):
    raise ValueError("no-H0 canonical mapping definitions differ across components")
no_h0_by_cohort = merge_dicts(
    base_no_h0.get("variable_ids_by_cohort", {}),
    new_no_h0.get("variable_ids_by_cohort", {}),
    "no-H0 variable mapping",
)

base_pairs = base_manifest.get("paired_condition_intersection", [])
new_pairs = new_manifest.get("paired_condition_intersection", [])
pair_keys = [
    (str(row.get("db")), str(row.get("analysis_id_a")), str(row.get("analysis_id_b")))
    for row in base_pairs + new_pairs
]
if len(pair_keys) != len(set(pair_keys)):
    raise ValueError("duplicate paired-condition intersection key across components")
expected_pair_keys = set()
for db, specs in (
    (("mimic", ALL_BASE_SPECS), ("eicu", ALL_BASE_SPECS))
    + tuple((cohort, CORE_SPECS) for cohort in NEW_COHORTS)
):
    analysis_ids = sorted(f"{model}_{condition}" for model, condition in specs)
    expected_pair_keys.update(
        (db, analysis_a, analysis_b)
        for analysis_a, analysis_b in combinations(analysis_ids, 2)
    )
if set(pair_keys) != expected_pair_keys:
    raise ValueError("paired-condition intersection key set is incomplete")

cache_records = []
seen_cache_paths = set()
for row, component_sha in (
    [(row, base_sha256) for row in base_rows]
    + [(row, new_component_sha256) for row in new_rows]
):
    cache_record = verify_cache_file(row, component_sha)
    if cache_record["path"] in seen_cache_paths:
        raise ValueError(f"multiple manifest rows point to one cache: {cache_record['path']}")
    seen_cache_paths.add(cache_record["path"])
    cache_records.append(cache_record)

merged = dict(new_manifest)
merged["models"] = base_rows + new_rows
merged["runtime_identity_by_model_tag"] = runtime_by_model
merged["source_npz_sha256_by_cohort"] = source_hashes
merged["paired_condition_intersection"] = base_pairs + new_pairs
merged["no_h0_therapy_mapping"] = {
    "canonical_variable_names": base_no_h0["canonical_variable_names"],
    "variable_ids_by_cohort": no_h0_by_cohort,
    "condition_definition_sha256": base_no_h0["condition_definition_sha256"],
}
merged["manifest_components"] = {
    "frozen_mimic_eicu_core_plus_big20": {
        "path": str(BASE_MANIFEST),
        "manifest_sha256": base_sha256,
        "engine_code_sha256": base_manifest.get("engine_code_sha256"),
        "row_count": len(base_rows),
    },
    "reviewed_new_cohorts_core21": {
        "path": str(NEW_COMPONENT_SNAPSHOT),
        "manifest_sha256": new_component_sha256,
        "engine_code_sha256": new_manifest.get("engine_code_sha256"),
        "row_count": len(new_rows),
    },
}

serialized = (json.dumps(merged, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
merged_sha256 = hashlib.sha256(serialized).hexdigest()
snapshot_path = SNAPSHOTS / f"embedding_manifest_recovered_v4_2_merged41_{merged_sha256[:12]}.json"
snapshot_sha_path = Path(str(snapshot_path) + ".sha256")
audit_path = SNAPSHOTS / f"embedding_manifest_merge_audit_recovered_v4_2_{merged_sha256[:12]}.json"
if snapshot_path.exists() or snapshot_sha_path.exists() or audit_path.exists():
    raise FileExistsError("a merged manifest artifact with this SHA already exists")

audit = {
    "base_component": {
        "path": str(BASE_MANIFEST), "sha256": base_sha256, "row_count": len(base_rows),
    },
    "new_component": {
        "path": str(NEW_COMPONENT_SNAPSHOT), "sha256": new_component_sha256, "row_count": len(new_rows),
    },
    "merged_manifest_snapshot": str(snapshot_path),
    "merged_manifest_sha256": merged_sha256,
    "merged_row_count": len(merged["models"]),
    "encoder_code_sha256": EXPECTED_ENCODER_SHA256,
    "duplicate_keys": 0,
    "cache_metadata_identity_checks": len(cache_records),
    "cache_file_sha256_checks": len(cache_records),
    "cache_files": cache_records,
    "prediction_run_id": "recovered_external_primary_v4_2",
}
audit_bytes = (json.dumps(audit, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
atomic_write(snapshot_path, serialized, replace=False)
atomic_write(snapshot_sha_path,
             f"{merged_sha256}  {snapshot_path}\n".encode("utf-8"), replace=False)
atomic_write(audit_path, audit_bytes, replace=False)

if sha256_file(LIVE_MANIFEST) != new_component_sha256:
    raise RuntimeError("live manifest changed during merge validation; refusing publication")
atomic_write(LIVE_MANIFEST, serialized, replace=True)
if sha256_file(LIVE_MANIFEST) != merged_sha256:
    raise RuntimeError("atomic merged-manifest publication SHA256 mismatch")
print(f"MERGED_MANIFEST={LIVE_MANIFEST}", flush=True)
print(f"MERGED_MANIFEST_SHA256={merged_sha256}", flush=True)
print(f"MERGED_MANIFEST_SNAPSHOT={snapshot_path}", flush=True)
print(f"MERGED_MANIFEST_AUDIT={audit_path}", flush=True)
print("MERGED_ROWS=41 CACHE_FILES_SHA256_VERIFIED=41", flush=True)
