#!/usr/bin/env python3
"""Static AST and on-disk preflight for the recovered-cohort resume3 runner."""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path


REV = Path("/data/admin/fengshuo/ehr_fm_20261102/revision_v4_20261122")
RUNNER = REV / "engine/code/run_recovered_external_primary_v4_2_cpu_resume3.py"
EXPECTED_ENGINE_FILE = (
    "code/v4_engine_registry_maskfix_recovered_v4_2_resume_"
    "reusefix_a725994d9d5d.py"
)
EXPECTED_RUN_ID = "recovered_external_primary_v4_2_resume3"
EXPECTED_SOURCE_PROBE_RUN_ID = "recovered_external_primary_v4_2_vocabfix1"
EXPECTED_HASH_NAMES = {
    "EXPECTED_ENGINE_SHA256",
    "EXPECTED_EMBEDDING_SHA256",
    "EXPECTED_ELIGIBILITY_SHA256",
    "EXPECTED_ENCODER_SHA256",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def top_level_literals(tree: ast.Module) -> dict[str, object]:
    values: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    return values


def main() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(RUNNER))
    literals = top_level_literals(tree)
    missing = EXPECTED_HASH_NAMES - set(literals)
    if missing:
        raise ValueError(f"runner is missing static SHA constants: {sorted(missing)}")
    hashes = {name: literals[name] for name in EXPECTED_HASH_NAMES}
    invalid = {
        name: value for name, value in hashes.items()
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
    }
    if invalid:
        raise ValueError(f"runner SHA constants must be 64 lowercase hex characters: {invalid}")
    if literals.get("RUN_ID") != EXPECTED_RUN_ID:
        raise ValueError(f"unexpected prediction namespace: {literals.get('RUN_ID')!r}")
    if literals.get("SOURCE_PROBE_REUSE_RUN_ID") != EXPECTED_SOURCE_PROBE_RUN_ID:
        raise ValueError("runner does not reuse the preserved vocabfix1 source probes")
    if literals.get("RUN_ID") == literals.get("SOURCE_PROBE_REUSE_RUN_ID"):
        raise ValueError("prediction run and source-probe origin namespaces must differ")
    if tuple(literals.get("MODELS", ())) != ("main20k", "abl_const20k", "common20k"):
        raise ValueError("runner model set differs from the authorized core-three scope")
    if tuple(literals.get("COHORTS", ())) != (
            "mimic", "nwicu_discharge_v4_1", "sicdb_zero_offset_v4_2",
            "eicu_semantic_v4_2"):
        raise ValueError("runner cohort set/order differs from the recovered-cohort scope")

    engine_script_node = next(
        node.value for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ENGINE_SCRIPT"
                for target in node.targets))
    engine_constants = [node.value for node in ast.walk(engine_script_node)
                        if isinstance(node, ast.Constant)
                        and isinstance(node.value, str)
                        and node.value.startswith("code/")]
    if engine_constants != [EXPECTED_ENGINE_FILE]:
        raise ValueError(f"runner engine path mismatch: {engine_constants}")

    engine_path = REV / "engine" / EXPECTED_ENGINE_FILE
    embedding_path = REV / "engine/results/embedding_manifest.json"
    frozen_embedding_path = (
        REV / "engine/code_snapshots/embedding_manifest_recovered_v4_2_merged41_"
        "b2eec1f7aca1.json"
    )
    merge_audit_path = (
        REV / "engine/code_snapshots/embedding_manifest_merge_audit_recovered_v4_2_"
        "b2eec1f7aca1.json"
    )
    eligibility_path = (
        REV / "temporal/results/eligibility/recovered_reviewed_v4_2/"
        "eligibility_manifest.json"
    )
    for label, path in (
            ("engine snapshot", engine_path),
            ("live embedding manifest", embedding_path),
            ("frozen embedding manifest", frozen_embedding_path),
            ("manifest merge audit", merge_audit_path),
            ("eligibility manifest", eligibility_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} missing: {path}")

    actual_hashes = {
        "EXPECTED_ENGINE_SHA256": sha256_file(engine_path),
        "EXPECTED_EMBEDDING_SHA256": sha256_file(embedding_path),
        "EXPECTED_ELIGIBILITY_SHA256": sha256_file(eligibility_path),
    }
    mismatches = {
        name: {"expected": hashes[name], "actual": actual}
        for name, actual in actual_hashes.items() if hashes[name] != actual
    }
    if mismatches:
        raise ValueError(f"runner input SHA mismatch: {mismatches}")
    live_bytes = embedding_path.read_bytes()
    if live_bytes != frozen_embedding_path.read_bytes():
        raise ValueError("live embedding manifest differs from the frozen 41-row snapshot")
    embedding = json.loads(live_bytes)
    if len(embedding.get("models", [])) != 41:
        raise ValueError("expected exactly 41 rows in the frozen embedding manifest")
    if embedding.get("encoder_code_sha256") != hashes["EXPECTED_ENCODER_SHA256"]:
        raise ValueError("encoder hash does not match the runner's static expectation")
    merge_audit = json.loads(merge_audit_path.read_text(encoding="utf-8"))
    if (merge_audit.get("merged_manifest_sha256") != hashes["EXPECTED_EMBEDDING_SHA256"]
            or merge_audit.get("merged_row_count") != 41
            or merge_audit.get("cache_file_sha256_checks") != 41):
        raise ValueError("frozen embedding merge audit is incomplete or mismatched")

    run_id = str(literals["RUN_ID"])
    runs_dir = REV / "engine/results/prediction_runs"
    if any(runs_dir.glob(f"{run_id}-*")):
        raise FileExistsError(f"prediction output namespace already exists for {run_id}")
    runtime_sidecar = runs_dir / f"{run_id}_cpu_consumer_runtime.json"
    if runtime_sidecar.exists():
        raise FileExistsError(f"consumer runtime sidecar already exists: {runtime_sidecar}")

    source_probe_dir = REV / "engine/results/source_probes"
    probe_files = list(source_probe_dir.glob("*.pkl")) if source_probe_dir.is_dir() else []
    runner_hash = sha256_file(RUNNER)
    print(json.dumps({
        "status": "static_runner_preflight_passed",
        "runner": str(RUNNER),
        "runner_sha256": runner_hash,
        "run_id": run_id,
        "source_probe_reuse_run_id": literals["SOURCE_PROBE_REUSE_RUN_ID"],
        "engine_snapshot": str(engine_path),
        "engine_snapshot_sha256": actual_hashes["EXPECTED_ENGINE_SHA256"],
        "embedding_manifest_sha256": actual_hashes["EXPECTED_EMBEDDING_SHA256"],
        "eligibility_manifest_sha256": actual_hashes["EXPECTED_ELIGIBILITY_SHA256"],
        "encoder_sha256": hashes["EXPECTED_ENCODER_SHA256"],
        "frozen_embedding_rows": len(embedding["models"]),
        "source_probe_store_files_seen": len(probe_files),
        "prediction_namespace_clear": True,
    }, indent=2, sort_keys=True), flush=True)
    print("RECOVERED_RESUME3_STATIC_PREFLIGHT_PASS", flush=True)


if __name__ == "__main__":
    main()
