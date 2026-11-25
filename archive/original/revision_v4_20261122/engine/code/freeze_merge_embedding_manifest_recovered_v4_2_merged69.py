#!/usr/bin/env python3
"""Validate and freeze the 69-row embedding component from immutable snapshots."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
ENGINE = REV / "engine"
CACHE_ROOT = ENGINE / "cache"
SNAPSHOTS = ENGINE / "code_snapshots"
EXPECTED_ENCODER_SHA256 = "59c3f4059662fb9e1d81baf20120c8668a761a926f5cab5ff4e654f7c16c1abc"
COMPONENTS = (
    {
        "name": "source_headreuse_merged41",
        "path": SNAPSHOTS / "embedding_manifest_recovered_v4_2_merged41_b2eec1f7aca1.json",
        "sha256": "b2eec1f7aca1857bda440ddb5ec48e477b2a20f753d5dea28b7715f7a1a243fe",
        "rows": 41,
        "engine_sha256": "ef70b4e857ce4a02480b7453a514bdfbf4f2a5e7aa4dd033455851919e269768",
    },
    {
        "name": "inspire_20",
        "path": SNAPSHOTS / "embedding_manifest_inspire_v4_1_20row_1b100d82660d.json",
        "sha256": "1b100d82660dc56bdcdfc0512387d174605368305e5a79e5ba3d4c314000ca2d",
        "rows": 20,
        "engine_sha256": "a725994d9d5dd33d8a2b23ba42afb029c98c5f530658ddda20e76f259a4b9528",
    },
    {
        "name": "availability_8",
        "path": SNAPSHOTS / "embedding_manifest_availability_v4_1_8row_21d95e2cd521.json",
        "sha256": "21d95e2cd521a816a23624c6dd07d36cdd746d12c7b9ce0537150c8075f93e80",
        "rows": 8,
        "engine_sha256": "21aae5dd6b013a6fc9a4943c50799607e30690c611e322cb5799bfa21a8e95da",
    },
)
COMMON_FIELDS = (
    "mode",
    "input_scope",
    "clinical_input_availability_verified",
    "clinical_ready",
    "input_availability_limitation",
    "encoder_code_sha256",
    "torch_version",
    "encoding_device",
    "note",
)
METADATA_FIELDS = (
    "db",
    "model_tag",
    "input_condition",
    "checkpoint_sha256",
    "input_mask_sha256",
    "row_identity_sha256",
    "source_npz_sha256",
    "split_sha256",
    "encoder_code_sha256",
    "runtime_identity_sha256",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
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


def identity_key(row: dict) -> tuple[str, str, str]:
    return (str(row.get("db", "")), str(row.get("model_tag", "")),
            str(row.get("input_condition", "")))


def expected_pair_keys(rows: list[dict]) -> set[tuple[str, str, str]]:
    analyses_by_db: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        analyses_by_db[str(row["db"])].add(
            f"{row['model_tag']}_{row['input_condition']}"
        )
    return {
        (db, left, right)
        for db, analyses in analyses_by_db.items()
        for left, right in combinations(sorted(analyses), 2)
    }


def actual_pair_keys(manifest: dict, label: str) -> set[tuple[str, str, str]]:
    pairs = manifest.get("paired_condition_intersection")
    if not isinstance(pairs, list):
        raise ValueError(f"{label}: paired-condition intersections must be a list")
    keys = []
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError(f"{label}: paired-condition intersection row is invalid")
        keys.append((
            str(pair.get("db", "")),
            str(pair.get("analysis_id_a", "")),
            str(pair.get("analysis_id_b", "")),
        ))
    if len(keys) != len(set(keys)):
        raise ValueError(f"{label}: duplicate paired-condition intersection keys")
    return set(keys)


def merge_disjoint(mappings: list[dict], label: str) -> dict:
    merged = {}
    for mapping in mappings:
        for key, value in mapping.items():
            if key in merged and merged[key] != value:
                raise ValueError(f"{label}: conflicting values for {key}")
            merged[key] = value
    return merged


def verify_component(component: dict) -> tuple[dict, list[dict]]:
    path = component["path"]
    if not path.is_file():
        raise FileNotFoundError(f"{component['name']}: missing frozen snapshot {path}")
    digest = sha256_file(path)
    if digest != component["sha256"]:
        raise ValueError(f"{component['name']}: frozen manifest SHA256 mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if len(manifest.get("models", [])) != component["rows"]:
        raise ValueError(f"{component['name']}: expected {component['rows']} rows")
    if manifest.get("engine_code_sha256") != component["engine_sha256"]:
        raise ValueError(f"{component['name']}: engine snapshot hash mismatch")
    if manifest.get("encoder_code_sha256") != EXPECTED_ENCODER_SHA256:
        raise ValueError(f"{component['name']}: encoder identity mismatch")
    if manifest.get("mode") != "encode":
        raise ValueError(f"{component['name']}: component is not an encoder manifest")

    rows = manifest["models"]
    keyed = {}
    for row in rows:
        key = identity_key(row)
        if key in keyed:
            raise ValueError(f"{component['name']}: duplicate cache identity {key}")
        if row.get("status") not in (None, "complete") or int(row.get("n_rows", 0)) <= 0:
            raise ValueError(f"{component['name']}: incomplete cache row {key}")
        if row.get("encoder_code_sha256") != EXPECTED_ENCODER_SHA256:
            raise ValueError(f"{component['name']}: row encoder hash mismatch {key}")
        keyed[key] = row

    expected_pairs = expected_pair_keys(rows)
    observed_pairs = actual_pair_keys(manifest, component["name"])
    if observed_pairs != expected_pairs:
        raise ValueError(
            f"{component['name']}: paired-condition intersection mismatch "
            f"(missing={len(expected_pairs - observed_pairs)}, "
            f"unexpected={len(observed_pairs - expected_pairs)})"
        )
    return manifest, rows


def verify_cache(row: dict, component_sha256: str, seen_paths: set[str]) -> dict:
    key = identity_key(row)
    cache_path = Path(str(row.get("cache", "")))
    if not cache_path.is_absolute():
        raise ValueError(f"cache path must be absolute for {key}: {cache_path}")
    resolved = cache_path.resolve(strict=True)
    if not resolved.is_relative_to(CACHE_ROOT.resolve()):
        raise ValueError(f"cache path escapes engine/cache for {key}: {resolved}")
    resolved_text = str(resolved)
    if resolved_text in seen_paths:
        raise ValueError(f"multiple rows point to the same cache: {resolved}")
    seen_paths.add(resolved_text)

    with np.load(resolved, allow_pickle=False) as arrays:
        if "metadata_json" not in arrays.files:
            raise ValueError(f"cache metadata is missing for {key}: {resolved}")
        metadata = json.loads(str(arrays["metadata_json"].item()))
    for field in METADATA_FIELDS:
        if metadata.get(field) != row.get(field):
            raise ValueError(f"cache/manifest mismatch for {key}: {field}")
    if row.get("input_condition") == "no_h0_therapy" and (
        metadata.get("input_mask_spec") != row.get("input_mask_spec")
        or metadata.get("condition_definition_sha256")
        != row.get("condition_definition_sha256")
    ):
        raise ValueError(f"cache/manifest no-H0 rule mismatch for {key}")
    return {
        "key": list(key),
        "path": resolved_text,
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "component_manifest_sha256": component_sha256,
    }


def main() -> None:
    manifests = []
    rows_by_component = []
    component_records = []
    for component in COMPONENTS:
        manifest, rows = verify_component(component)
        for field in COMMON_FIELDS:
            if not manifests and field not in manifest:
                raise ValueError(f"{component['name']}: missing common metadata field {field}")
            if manifests and manifest.get(field) != manifests[0].get(field):
                raise ValueError(f"{component['name']}: component metadata differs for {field}")
        manifests.append(manifest)
        rows_by_component.append(rows)
        component_records.append({
            "path": str(component["path"]),
            "manifest_sha256": component["sha256"],
            "engine_code_sha256": component["engine_sha256"],
            "row_count": component["rows"],
        })

    all_rows = [row for rows in rows_by_component for row in rows]
    all_keys = [identity_key(row) for row in all_rows]
    if len(all_rows) != 69 or len(set(all_keys)) != 69:
        raise ValueError(f"expected 69 unique rows; found {len(all_rows)} rows and {len(set(all_keys))} keys")

    runtime = merge_disjoint(
        [manifest.get("runtime_identity_by_model_tag", {}) for manifest in manifests],
        "runtime identity",
    )
    source_hashes = merge_disjoint(
        [manifest.get("source_npz_sha256_by_cohort", {}) for manifest in manifests],
        "source NPZ hashes",
    )
    source_paths = merge_disjoint(
        [manifest.get("source_paths_by_cohort", {}) for manifest in manifests],
        "source paths",
    )
    split_hashes = merge_disjoint(
        [manifest.get("split_sha256_by_cohort", {}) for manifest in manifests],
        "split hashes",
    )
    base_no_h0 = manifests[0].get("no_h0_therapy_mapping", {})
    for manifest in manifests[1:]:
        current = manifest.get("no_h0_therapy_mapping", {})
        if (current.get("canonical_variable_names") != base_no_h0.get("canonical_variable_names")
                or current.get("condition_definition_sha256")
                != base_no_h0.get("condition_definition_sha256")):
            raise ValueError("no-H0 canonical mapping definitions differ across components")
    no_h0_by_cohort = merge_disjoint(
        [manifest.get("no_h0_therapy_mapping", {}).get("variable_ids_by_cohort", {})
         for manifest in manifests],
        "no-H0 variable mappings",
    )

    merged_pairs = [
        pair
        for manifest in manifests
        for pair in manifest["paired_condition_intersection"]
    ]
    merged_pair_keys = [
        (str(pair["db"]), str(pair["analysis_id_a"]), str(pair["analysis_id_b"]))
        for pair in merged_pairs
    ]
    if len(merged_pair_keys) != len(set(merged_pair_keys)):
        raise ValueError("duplicate paired-condition intersection key across components")
    if set(merged_pair_keys) != expected_pair_keys(all_rows):
        raise ValueError("merged paired-condition intersection key set is incomplete")

    cache_records = []
    seen_cache_paths: set[str] = set()
    for component, rows in zip(COMPONENTS, rows_by_component):
        for row in rows:
            cache_records.append(
                verify_cache(row, component["sha256"], seen_cache_paths)
            )

    merged = dict(manifests[-1])
    merged["models"] = all_rows
    merged["runtime_identity_by_model_tag"] = runtime
    merged["source_npz_sha256_by_cohort"] = source_hashes
    merged["source_paths_by_cohort"] = source_paths
    merged["split_sha256_by_cohort"] = split_hashes
    merged["paired_condition_intersection"] = merged_pairs
    merged["no_h0_therapy_mapping"] = {
        "canonical_variable_names": base_no_h0["canonical_variable_names"],
        "variable_ids_by_cohort": no_h0_by_cohort,
        "condition_definition_sha256": base_no_h0["condition_definition_sha256"],
    }
    merged["manifest_components"] = {
        "source_headreuse_merged41": {
            **component_records[0],
            "nested_components": manifests[0].get("manifest_components", {}),
        },
        "inspire_20": component_records[1],
        "availability_8": component_records[2],
    }

    serialized = (json.dumps(merged, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    merged_sha256 = hashlib.sha256(serialized).hexdigest()
    snapshot_path = SNAPSHOTS / (
        f"embedding_manifest_recovered_v4_2_merged69_{merged_sha256[:12]}.json"
    )
    sha_path = Path(str(snapshot_path) + ".sha256")
    audit_path = SNAPSHOTS / (
        f"embedding_manifest_merge_audit_recovered_v4_2_merged69_{merged_sha256[:12]}.json"
    )
    if snapshot_path.exists() or sha_path.exists() or audit_path.exists():
        raise FileExistsError("a merged69 artifact with this SHA already exists")

    audit = {
        "source_components": [
            {"name": component["name"], **record}
            for component, record in zip(COMPONENTS, component_records)
        ],
        "merged_manifest_snapshot": str(snapshot_path),
        "merged_manifest_sha256": merged_sha256,
        "merged_row_count": len(all_rows),
        "merged_pair_count": len(merged_pairs),
        "encoder_code_sha256": EXPECTED_ENCODER_SHA256,
        "cache_metadata_identity_checks": len(cache_records),
        "cache_file_sha256_checks": len(cache_records),
        "cache_files": cache_records,
        "merge_script_sha256": sha256_file(Path(__file__)),
        "live_manifest_modified": False,
    }
    audit_bytes = (json.dumps(audit, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    atomic_write(snapshot_path, serialized)
    atomic_write(sha_path, f"{merged_sha256}  {snapshot_path}\n".encode("utf-8"))
    atomic_write(audit_path, audit_bytes)
    print(f"MERGED69_SNAPSHOT={snapshot_path}", flush=True)
    print(f"MERGED69_SHA256={merged_sha256}", flush=True)
    print(f"MERGED69_AUDIT={audit_path}", flush=True)
    print(f"MERGED69_ROWS={len(all_rows)} PAIRS={len(merged_pairs)} CACHE_SHA256_VERIFIED={len(cache_records)}", flush=True)
    print("MERGED69_FREEZE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
