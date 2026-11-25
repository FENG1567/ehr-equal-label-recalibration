"""Read-only release gate for the frozen repaired numeric-decoding aggregate.

This audit does not fit models or alter decoder outputs. It reads the pinned
aggregate and provenance, then hashes the original payloads on the analysis
server before writing its report beside this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
RELEASE_DIR = REV / "decoding_repaired/release_audit"
SUMMARY_PATH = REV / "decoding_repaired/results/numeric_decoding_summary.csv"
RUN_MANIFEST_PATH = REV / "decoding_repaired/results/numeric_decoding_manifest.json"
EMBEDDING_MANIFEST_PATH = (
    REV / "engine/code_snapshots/embedding_manifest_recovered_v4_2_merged41_b2eec1f7aca1.json"
)
CACHE_AUDIT_MANIFEST_PATH = (
    REV / "engine/code_snapshots/embedding_manifest_merge_audit_recovered_v4_2_b2eec1f7aca1.json"
)
ELIGIBILITY_DIR = REV / "temporal/results/eligibility/recovered_reviewed_v4_2"
SOURCE_REFERENCE_PATH = (
    REV / "decoding/results/primary_status_consistent_20261123/numeric_decoding_summary.csv"
)
PRIMARY_MANIFEST_PATH = (
    REV / "decoding/results/primary_status_consistent_20261123/numeric_decoding_manifest.json"
)
DATASET_MANIFEST_PATH = REV / "datasets/eicu_semantic_v4_2/dataset_manifest.json"
REGISTRY_PATH = REV / "data_registry.json"
EXECUTED_MODULE_PATH = REV / "decoding_repaired/numeric_decoding_repaired.py"

EXPECTED_SUMMARY_SHA256 = "60bf90cd07ef32cacdf276be2bc27f3ff6ffda75f4e4b2f37df869888e94960e"
EXPECTED_RUN_MANIFEST_SHA256 = "e7afa8d7d3a652c85e1d036a905fbfaa51af13148763880c99feae7f78800b11"
EXPECTED_EMBEDDING_MANIFEST_SHA256 = "b2eec1f7aca1857bda440ddb5ec48e477b2a20f753d5dea28b7715f7a1a243fe"
EXPECTED_CACHE_AUDIT_MANIFEST_SHA256 = "ba792d7c600c134bbaa79deebba123183217b4bf474a5102c7100ac65f09740d"
EXPECTED_ELIGIBILITY_MANIFEST_SHA256 = "d302ce0a493121f9336f40fd58c5dc63a75a663492293c0153b4ba05efbb8e8f"
EXPECTED_SOURCE_REFERENCE_SHA256 = "0ff14db563f0832740cb38c2b4722182ef11b3b2c583031e83ebac33c8014cae"
EXPECTED_PRIMARY_MANIFEST_SHA256 = "3fc2d12330002e8df3df5a01f53a163b448f1c01183db766b0a6f61647318446"
EXPECTED_DATASET_MANIFEST_SHA256 = "7148b23c135ec0bd97a105058d334e15a15e2ecbcfe0e775ee65f355b97a0059"
EXPECTED_REGISTRY_SHA256 = "e42ee431df5117f6cf399bf601445fbddbc16571ca83ae5e5f5a2678b3ebdacc"
EXPECTED_EXECUTED_MODULE_SHA256 = "9b129514f1a470e5315ac8c9950fe161f404884ad0eae04a8fe4344f7e42a459"

SOURCE_DB = "mimic"
TARGET_DB = "eicu_semantic_v4_2"
ANALYSIS_IDS = ("main20k_real", "main20k_constant", "abl_const20k_constant")
MODES = ("source_transfer", "target_local_nested_cv")
VARIABLES = (
    "creatinine", "lactate", "wbc", "hgb", "glucose", "hr", "sbp",
    "spo2", "bun", "platelet",
)
ALPHA_BY_ANALYSIS = {
    "main20k_real": ("main20k", "real"),
    "main20k_constant": ("main20k", "constant"),
    "abl_const20k_constant": ("abl_const20k", "constant"),
}


class ReleaseAuditError(ValueError):
    """Raised when a frozen release artifact or required payload fails audit."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_hash_match(label: str, declared: str, trusted: str,
                      actual: str) -> dict[str, str]:
    """Require recorded, trusted-manifest, and on-disk payload hashes to match."""
    values = {"declared": str(declared).lower(), "trusted": str(trusted).lower(),
              "actual": str(actual).lower()}
    if any(len(value) != 64 for value in values.values()):
        raise ReleaseAuditError(f"{label}: an expected SHA256 is missing or malformed")
    if len(set(values.values())) != 1:
        raise ReleaseAuditError(f"{label}: recorded, trusted, and payload SHA256 values differ")
    return {"label": label, **values, "status": "matched"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseAuditError(message)


def _number(value: Any, label: str) -> float:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if not np.isfinite(number):
        raise ReleaseAuditError(f"{label}: expected a finite numeric value")
    return float(number)


def _identity_signature(group: pd.DataFrame, label: str) -> tuple[Any, ...]:
    fields = (
        "population_identity_sha256", "n_eligible_stays", "n_common_condition_stays",
        "n_target_available", "n_rows", "n_patients", "cache_rows_by_condition_json",
    )
    signatures: list[Any] = []
    for field in fields:
        values = group[field].drop_duplicates().tolist()
        _require(len(values) == 1, f"{label}: target identity/count field {field} differs across cells")
        signatures.append(values[0])
    return tuple(signatures)


def audit_aggregate(summary: pd.DataFrame, run_manifest: Mapping[str, Any],
                    source_reference: pd.DataFrame) -> dict[str, Any]:
    """Verify full grid, successful cells, marginal intervals, and target identities."""
    required = {
        "analysis_id", "mode", "source_db", "target_db", "variable", "status",
        "population_identity_sha256", "n_eligible_stays", "n_common_condition_stays",
        "n_target_available", "n_rows", "n_patients", "cache_rows_by_condition_json",
        "r2", "rmse", "mae", "r2_lo", "r2_hi", "rmse_lo", "rmse_hi", "mae_lo", "mae_hi",
        "bootstrap_valid", "bootstrap_valid_r2", "bootstrap_replicates", "bootstrap_cluster_unit",
        "bootstrap_interval_type", "bootstrap_seed", "alpha", "source_alpha_validation_mse",
        "source_mimic_split2_metrics_match_reference", "n_outer_folds", "n_inner_folds",
    }
    missing_columns = sorted(required - set(summary.columns))
    _require(not missing_columns, f"aggregate lacks required columns: {missing_columns}")
    forbidden_columns = {"stay_id", "subject", "prediction", "y_mort"}.intersection(summary.columns)
    _require(not forbidden_columns, f"aggregate exposes patient-level fields: {sorted(forbidden_columns)}")
    _require(len(summary) == 60, f"aggregate must contain 60 rows; found {len(summary)}")
    _require(run_manifest.get("status") == "complete", "run manifest is not complete")
    _require(run_manifest.get("source_db") == SOURCE_DB, "run manifest has an unexpected source cohort")
    _require(run_manifest.get("target_dbs") == [TARGET_DB], "run manifest target scope is not eICU semantic v4.2 only")
    _require(run_manifest.get("analysis_ids") == list(ANALYSIS_IDS), "run manifest conditions differ from the frozen grid")
    _require(run_manifest.get("loaded_cohorts") == [SOURCE_DB, TARGET_DB], "run manifest loaded cohort scope differs")
    _require(run_manifest.get("bootstrap_replicates") == 1000,
             "run manifest bootstrap replicate count differs")
    _require("marginal" in str(run_manifest.get("bootstrap_interval_type", "")).lower(),
             "run manifest does not label bootstrap intervals marginal")

    observed = set(zip(summary["analysis_id"].astype(str), summary["mode"].astype(str),
                       summary["variable"].astype(str), summary["target_db"].astype(str)))
    expected = {(analysis_id, mode, variable, TARGET_DB)
                for analysis_id in ANALYSIS_IDS for mode in MODES for variable in VARIABLES}
    duplicates = summary.duplicated(["analysis_id", "mode", "variable", "target_db"], keep=False)
    _require(not duplicates.any(), "aggregate contains duplicate grid cells")
    _require(observed == expected,
             f"aggregate grid mismatch: missing={sorted(expected-observed)}, extra={sorted(observed-expected)}")
    _require(summary["source_db"].astype(str).eq(SOURCE_DB).all(), "aggregate source cohort differs")
    _require(summary["status"].astype(str).eq("ok").all(), "aggregate has failed or non-success cells")

    for row_index, row in summary.iterrows():
        label = f"row {row_index} {row['analysis_id']}/{row['mode']}/{row['variable']}"
        _require(int(_number(row["bootstrap_replicates"], label)) == 1000,
                 f"{label}: expected 1,000 bootstrap replicates")
        _require(int(_number(row["bootstrap_valid"], label)) == 1000,
                 f"{label}: expected 1,000 valid patient-cluster replicates")
        _require(int(_number(row["bootstrap_valid_r2"], label)) == 1000,
                 f"{label}: expected 1,000 valid R-squared replicates")
        _require(str(row["bootstrap_cluster_unit"]) == "patient",
                 f"{label}: bootstrap unit is not patient")
        _require(str(row["bootstrap_interval_type"]) == "marginal_patient_cluster_percentile",
                 f"{label}: interval type is not labeled marginal")
        values = {metric: _number(row[metric], label) for metric in
                  ("r2", "rmse", "mae", "r2_lo", "r2_hi", "rmse_lo", "rmse_hi", "mae_lo", "mae_hi")}
        _require(values["r2_lo"] <= values["r2_hi"], f"{label}: invalid R-squared interval")
        _require(values["rmse_lo"] <= values["rmse_hi"], f"{label}: invalid RMSE interval")
        _require(values["mae_lo"] <= values["mae_hi"], f"{label}: invalid MAE interval")
        _require(values["rmse"] >= 0 and values["mae"] >= 0,
                 f"{label}: error metric cannot be negative")

    _require((pd.to_numeric(summary["r2"], errors="coerce") < 0).any(),
             "aggregate contains no negative R-squared values; verify they were not clipped")
    _require(summary["n_eligible_stays"].astype(int).eq(130544).all(),
             "aggregate eligible target count differs from the recovered cohort")
    _require(summary["n_common_condition_stays"].astype(int).eq(130544).all(),
             "aggregate common-condition target count differs from the recovered cohort")
    for variable, group in summary.groupby("variable", sort=False):
        _identity_signature(group, f"{TARGET_DB}/{variable}")
        _require(group["n_rows"].astype(int).eq(group["n_target_available"].astype(int)).all(),
                 f"{variable}: scored target rows differ from finite target availability")

    expected_cache_rows = run_manifest.get("preflight_counts", {}).get(
        "cache_rows_by_condition_by_db", {}).get(TARGET_DB)
    _require(isinstance(expected_cache_rows, dict), "manifest lacks target cache-row provenance")
    _require(set(expected_cache_rows) == set(ANALYSIS_IDS), "manifest cache-row conditions differ")
    target_identity_records = []
    for variable, group in summary.groupby("variable", sort=False):
        cache_jsons = group["cache_rows_by_condition_json"].drop_duplicates().tolist()
        _require(len(cache_jsons) == 1, f"{variable}: cache identity provenance differs across cells")
        try:
            parsed = json.loads(cache_jsons[0])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ReleaseAuditError(f"{variable}: malformed cache-row identity JSON") from exc
        _require(parsed == expected_cache_rows, f"{variable}: aggregate cache-row counts differ from preflight")
        target_identity_records.append({
            "variable": variable,
            "population_identity_sha256": str(group["population_identity_sha256"].iloc[0]),
            "n_target_rows": int(group["n_rows"].iloc[0]),
            "n_target_patients": int(group["n_patients"].iloc[0]),
        })

    nested = summary.loc[summary["mode"].eq("target_local_nested_cv")]
    _require(nested["n_outer_folds"].astype(int).eq(5).all(), "target-local rows must use five outer folds")
    _require(nested["n_inner_folds"].astype(int).eq(3).all(), "target-local rows must use three inner folds")

    source_rows = summary.loc[summary["mode"].eq("source_transfer")].copy()
    reference_rows = source_reference.loc[
        source_reference["mode"].eq("source_transfer")
        & source_reference["source_db"].eq(SOURCE_DB)
        & source_reference["target_db"].eq(SOURCE_DB)
    ].copy()
    _require(len(reference_rows) == 30, "frozen MIMIC source summary does not contain 30 source rows")
    reference_index = reference_rows.set_index(["analysis_id", "variable"])
    for _, row in source_rows.iterrows():
        key = (str(row["analysis_id"]), str(row["variable"]))
        _require(key in reference_index.index, f"source-transfer cell {key} lacks frozen MIMIC reference")
        frozen = reference_index.loc[key]
        _require(np.isclose(float(row["alpha"]), float(frozen["alpha"]), rtol=0, atol=0),
                 f"{key}: source alpha differs from frozen MIMIC summary")
        _require(np.isclose(float(row["source_alpha_validation_mse"]),
                            float(frozen["source_alpha_validation_mse"]), rtol=1e-12, atol=1e-12),
                 f"{key}: MIMIC split-1 alpha-selection provenance differs")
        _require(str(row["source_mimic_split2_metrics_match_reference"]).lower() == "true",
                 f"{key}: MIMIC split-2 metric comparison was not successful")
    source_validation = run_manifest.get("source_mimic_split2_metric_validation", [])
    _require(len(source_validation) == 30 and all(
        record.get("status") == "matched" for record in source_validation),
        "run manifest lacks 30 successful MIMIC split-2 point-metric checks")
    max_source_metric_delta = max(
        float(record[f"{metric}_absolute_difference"])
        for record in source_validation for metric in ("r2", "rmse", "mae")
    )
    _require(max_source_metric_delta <= 1e-10,
             "MIMIC split-2 point-metric validation exceeds numerical tolerance")

    return {
        "status": "aggregate_pass",
        "row_count": int(len(summary)),
        "successful_cells": int(summary["status"].eq("ok").sum()),
        "conditions": list(ANALYSIS_IDS),
        "estimands": list(MODES),
        "variables": list(VARIABLES),
        "target_db": TARGET_DB,
        "all_rows_have_1000_valid_patient_bootstrap_draws": True,
        "intervals": "marginal patient-cluster percentile intervals",
        "negative_r2_row_count": int((pd.to_numeric(summary["r2"], errors="coerce") < 0).sum()),
        "target_identity_by_variable": target_identity_records,
        "source_reference_rows": int(len(reference_rows)),
        "source_mimic_split2_checks": int(len(source_validation)),
        "maximum_source_split2_metric_absolute_difference": max_source_metric_delta,
    }


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReleaseAuditError(f"expected a JSON object at {path}")
    return value


def _registered_path(db: str, field: str, registry: Mapping[str, Any]) -> Path:
    entries = registry.get("cohorts", registry)
    entry = entries.get(db, {})
    raw = entry.get(field) if isinstance(entry, dict) else None
    if raw is None:
        prefix = {"sequence_path": "seq", "split_path": "split", "tabular_path": "tab",
                  "vocab_path": "meta"}[field]
        path = ROOT / "data/processed" / f"{prefix}_{db}.{ {'sequence_path':'npz','split_path':'npy','tabular_path':'parquet','vocab_path':'json'}[field] }"
    else:
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = REGISTRY_PATH.parent / path
    return path.resolve()


def audit_upstream_payloads(run_manifest: Mapping[str, Any]) -> list[dict[str, str]]:
    """Compare recorded hashes to trusted upstream manifests and server payloads."""
    for path, expected, label in (
        (EMBEDDING_MANIFEST_PATH, EXPECTED_EMBEDDING_MANIFEST_SHA256, "frozen embedding manifest"),
        (CACHE_AUDIT_MANIFEST_PATH, EXPECTED_CACHE_AUDIT_MANIFEST_SHA256, "frozen cache audit manifest"),
        (ELIGIBILITY_DIR / "eligibility_manifest.json", EXPECTED_ELIGIBILITY_MANIFEST_SHA256, "recovered eligibility manifest"),
        (SOURCE_REFERENCE_PATH, EXPECTED_SOURCE_REFERENCE_SHA256, "frozen MIMIC source summary"),
        (PRIMARY_MANIFEST_PATH, EXPECTED_PRIMARY_MANIFEST_SHA256, "prior primary run manifest"),
        (DATASET_MANIFEST_PATH, EXPECTED_DATASET_MANIFEST_SHA256, "eICU semantic dataset manifest"),
        (REGISTRY_PATH, EXPECTED_REGISTRY_SHA256, "cohort data registry"),
        (EXECUTED_MODULE_PATH, EXPECTED_EXECUTED_MODULE_SHA256, "executed decoder module"),
    ):
        _require(path.is_file(), f"{label} is missing: {path}")
        actual = sha256_file(path)
        if actual.lower() != expected.lower():
            raise ReleaseAuditError(f"{label} SHA256 differs from frozen trust value")

    embedding_manifest = _read_json(EMBEDDING_MANIFEST_PATH)
    cache_audit_manifest = _read_json(CACHE_AUDIT_MANIFEST_PATH)
    eligibility_manifest = _read_json(ELIGIBILITY_DIR / "eligibility_manifest.json")
    primary_manifest = _read_json(PRIMARY_MANIFEST_PATH)
    dataset_manifest = _read_json(DATASET_MANIFEST_PATH)
    registry = _read_json(REGISTRY_PATH)
    upstream_rows = embedding_manifest.get("models", [])
    _require(len(upstream_rows) == 41, "frozen embedding manifest does not have 41 rows")
    _require(cache_audit_manifest.get("merged_manifest_sha256") == EXPECTED_EMBEDDING_MANIFEST_SHA256,
             "cache audit manifest does not attest the frozen embedding manifest")
    _require(cache_audit_manifest.get("merged_row_count") == 41
             and cache_audit_manifest.get("cache_file_sha256_checks") == 41,
             "cache audit manifest does not cover all 41 frozen cache files")
    trusted_cache_files = {
        tuple(str(part) for part in item["key"]): item
        for item in cache_audit_manifest.get("cache_files", [])
        if isinstance(item, dict) and isinstance(item.get("key"), list)
    }
    _require(run_manifest.get("embedding_manifest_sha256") == EXPECTED_EMBEDDING_MANIFEST_SHA256,
             "run manifest embedding hash differs from frozen manifest")
    _require(run_manifest.get("source_reference_summary_sha256") == EXPECTED_SOURCE_REFERENCE_SHA256,
             "run manifest source summary hash differs from frozen MIMIC source summary")
    _require(Path(str(run_manifest.get("embedding_manifest", ""))).resolve()
             == EMBEDDING_MANIFEST_PATH.resolve(),
             "run manifest points to a different embedding manifest")
    _require(Path(str(run_manifest.get("source_reference_summary", ""))).resolve()
             == SOURCE_REFERENCE_PATH.resolve(),
             "run manifest points to a different MIMIC source summary")
    _require(run_manifest.get("module_sha256") == EXPECTED_EXECUTED_MODULE_SHA256,
             "run manifest decoder module digest differs from the executed module")
    _require(run_manifest.get("eligibility_directory") == str(ELIGIBILITY_DIR),
             "run manifest eligibility directory differs from frozen recovered scope")

    payload_checks: list[dict[str, str]] = []
    recorded_target_sources = run_manifest.get("numeric_target_source_by_db", {})
    _require(set(recorded_target_sources) == {SOURCE_DB, TARGET_DB},
             "run manifest target-table provenance has an unexpected cohort scope")
    dataset_entry = dataset_manifest.get("registry_entry", {})
    dataset_hashes = dataset_manifest.get("output_hashes", {})
    dataset_target_path = Path(str(dataset_entry.get("tabular_path", ""))).resolve()
    registered_target_path = _registered_path(TARGET_DB, "tabular_path", registry)
    _require(dataset_target_path == registered_target_path,
             "eICU semantic path in data registry differs from dataset manifest")
    dataset_target_hash = dataset_hashes.get("tab_eicu_semantic_v4_2.parquet")
    _require(isinstance(dataset_target_hash, str), "dataset manifest lacks the repaired target table hash")

    for db in (SOURCE_DB, TARGET_DB):
        record = recorded_target_sources[db]
        path = _registered_path(db, "tabular_path", registry)
        _require(Path(str(record.get("path", ""))).resolve() == path,
                 f"{db}: recorded target-table path differs from registry")
        if db == SOURCE_DB:
            primary_table = primary_manifest.get("numeric_target_source_by_db", {}).get(SOURCE_DB, {})
            _require(Path(str(primary_table.get("path", ""))).resolve() == path,
                     "MIMIC target-table path differs from prior primary manifest")
            trusted_table_hash = primary_table.get("tabular_sha256")
        else:
            trusted_table_hash = dataset_target_hash
        _require(isinstance(trusted_table_hash, str), f"{db}: no upstream target-table hash")
        payload_checks.append(assert_hash_match(
            f"{db} numeric target table", record.get("tabular_sha256", ""),
            trusted_table_hash, sha256_file(path),
        ))

    recorded_eligibility_dir = Path(str(run_manifest["eligibility_directory"])).resolve()
    eligibility_path = recorded_eligibility_dir / "eligibility_manifest.json"
    actual_eligibility_hash = sha256_file(eligibility_path)
    _require(actual_eligibility_hash == EXPECTED_ELIGIBILITY_MANIFEST_SHA256,
             "current eligibility manifest differs from frozen recovered manifest")
    recorded_eligibility_manifest_hashes = run_manifest.get("eligibility_manifest_sha256_by_db", {})
    recorded_eligibility_table_hashes = run_manifest.get("eligibility_table_sha256_by_db", {})
    eligibility_cohorts = eligibility_manifest.get("cohorts", {})
    for db in (SOURCE_DB, TARGET_DB):
        entry = eligibility_cohorts.get(db, {})
        table_path = recorded_eligibility_dir / f"{db}.parquet"
        table_hash = entry.get("table_sha256") or entry.get("eligibility_table_sha256")
        _require(isinstance(table_hash, str), f"{db}: recovered eligibility manifest lacks table hash")
        payload_checks.append(assert_hash_match(
            f"{db} eligibility manifest", recorded_eligibility_manifest_hashes.get(db, ""),
            EXPECTED_ELIGIBILITY_MANIFEST_SHA256, actual_eligibility_hash,
        ))
        payload_checks.append(assert_hash_match(
            f"{db} eligibility table", recorded_eligibility_table_hashes.get(db, ""),
            table_hash, sha256_file(table_path),
        ))
        eligible_count = run_manifest.get("preflight_counts", {}).get(
            "eligible_count_by_db", {}).get(db)
        manifest_eligible_count = entry.get("n_eligible_24h")
        if manifest_eligible_count is None:
            manifest_eligible_count = entry.get("counts", {}).get("n_eligible_24h")
        _require(int(eligible_count) == int(manifest_eligible_count or -1),
                 f"{db}: eligible row count differs from recovered manifest")

    recorded_cache_hashes = run_manifest.get("embedding_cache_sha256_by_db_and_analysis", {})
    recorded_checkpoint_hashes = run_manifest.get("checkpoint_sha256_by_model", {})
    recorded_sequence_hashes = run_manifest.get("source_sequence_npz_sha256_by_db", {})
    recorded_split_hashes = run_manifest.get("split_sha256_by_db", {})
    for db in (SOURCE_DB, TARGET_DB):
        for analysis_id, (model_tag, condition) in ALPHA_BY_ANALYSIS.items():
            matching = [row for row in upstream_rows
                        if row.get("db") == db and row.get("model_tag") == model_tag
                        and row.get("input_condition") == condition]
            _require(len(matching) == 1, f"{db}/{analysis_id}: frozen embedding row missing or duplicated")
            row = matching[0]
            cache_path = Path(str(row.get("cache", "")))
            cache_audit_row = trusted_cache_files.get((db, model_tag, condition))
            _require(isinstance(cache_audit_row, dict),
                     f"{db}/{analysis_id}: trusted cache audit digest is missing")
            _require(Path(str(cache_audit_row.get("path", ""))).resolve() == cache_path.resolve(),
                     f"{db}/{analysis_id}: cache path differs from trusted cache audit")
            actual_cache_hash = sha256_file(cache_path)
            payload_checks.append(assert_hash_match(
                f"{db}/{analysis_id} embedding cache",
                recorded_cache_hashes.get(db, {}).get(analysis_id, ""),
                str(cache_audit_row.get("sha256", "")), actual_cache_hash,
            ))
            upstream_checkpoint_hash = str(row.get("checkpoint_sha256", ""))
            checkpoint_path = ROOT / "models" / f"{model_tag}.pt"
            payload_checks.append(assert_hash_match(
                f"{model_tag} checkpoint", recorded_checkpoint_hashes.get(model_tag, ""),
                upstream_checkpoint_hash, sha256_file(checkpoint_path),
            ))
            source_hash = str(row.get("source_npz_sha256", ""))
            split_hash = str(row.get("split_sha256", ""))
            _require(recorded_sequence_hashes.get(db) == source_hash,
                     f"{db}: run manifest sequence hash differs from embedding manifest")
            _require(recorded_split_hashes.get(db) == split_hash,
                     f"{db}: run manifest split hash differs from embedding manifest")

        sequence_path = _registered_path(db, "sequence_path", registry)
        split_path = _registered_path(db, "split_path", registry)
        trusted_sequence_hash = str(recorded_sequence_hashes.get(db, ""))
        trusted_split_hash = str(recorded_split_hashes.get(db, ""))
        if db == TARGET_DB:
            dataset_output = dataset_manifest.get("output_hashes", {})
            _require(dataset_output.get("seq_eicu_semantic_v4_2.npz") == trusted_sequence_hash,
                     "eICU sequence hash differs from repaired dataset manifest")
            _require(dataset_output.get("split_eicu_semantic_v4_2.npy") == trusted_split_hash,
                     "eICU split hash differs from repaired dataset manifest")
        payload_checks.append(assert_hash_match(
            f"{db} sequence NPZ", trusted_sequence_hash, trusted_sequence_hash,
            sha256_file(sequence_path),
        ))
        payload_checks.append(assert_hash_match(
            f"{db} split array", trusted_split_hash, trusted_split_hash,
            sha256_file(split_path),
        ))

    return payload_checks


def run_release_audit(*, summary_path: Path = SUMMARY_PATH,
                      run_manifest_path: Path = RUN_MANIFEST_PATH,
                      report_path: Path | None = None) -> dict[str, Any]:
    """Run aggregate and server-payload checks; write only a new audit report."""
    for path, expected, label in (
        (summary_path, EXPECTED_SUMMARY_SHA256, "frozen aggregate summary"),
        (run_manifest_path, EXPECTED_RUN_MANIFEST_SHA256, "frozen run manifest"),
        (SOURCE_REFERENCE_PATH, EXPECTED_SOURCE_REFERENCE_SHA256, "frozen MIMIC source summary"),
    ):
        _require(path.is_file(), f"{label} is missing: {path}")
        if sha256_file(path).lower() != expected.lower():
            raise ReleaseAuditError(f"{label} SHA256 differs from pinned release value")

    run_manifest = _read_json(run_manifest_path)
    summary = pd.read_csv(summary_path)
    source_reference = pd.read_csv(SOURCE_REFERENCE_PATH)
    aggregate_checks = audit_aggregate(summary, run_manifest, source_reference)
    payload_checks = audit_upstream_payloads(run_manifest)
    report = {
        "status": "PASS",
        "audit_script_sha256": sha256_file(Path(__file__)),
        "trusted_artifacts": {
            "summary_sha256": sha256_file(summary_path),
            "run_manifest_sha256": sha256_file(run_manifest_path),
            "embedding_manifest_sha256": sha256_file(EMBEDDING_MANIFEST_PATH),
            "cache_audit_manifest_sha256": sha256_file(CACHE_AUDIT_MANIFEST_PATH),
            "eligibility_manifest_sha256": sha256_file(ELIGIBILITY_DIR / "eligibility_manifest.json"),
            "source_reference_sha256": sha256_file(SOURCE_REFERENCE_PATH),
        },
        "aggregate_checks": aggregate_checks,
        "payload_check_count": int(len(payload_checks)),
        "payload_checks": payload_checks,
        "payload_scope": "all original payload hashing was performed on the analysis server",
        "refits_performed": False,
        "original_artifacts_modified": False,
    }
    destination = report_path or (RELEASE_DIR / "numeric_decoding_release_audit.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--run-manifest", type=Path, default=RUN_MANIFEST_PATH)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        report = run_release_audit(
            summary_path=args.summary, run_manifest_path=args.run_manifest,
            report_path=args.report,
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"V4_NUMERIC_DECODING_RELEASE_AUDIT_FAIL {type(exc).__name__}: {exc}", flush=True)
        return 1
    print(json.dumps(report, sort_keys=True), flush=True)
    print("V4_NUMERIC_DECODING_RELEASE_AUDIT_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
