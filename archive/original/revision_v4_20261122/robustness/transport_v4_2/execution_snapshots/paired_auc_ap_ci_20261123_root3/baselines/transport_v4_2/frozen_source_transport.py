"""Fit frozen source baseline arms, verify original predictions, then score targets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
BASELINE_CODE = REV / "baselines"
ROBUSTNESS_CODE = REV / "robustness"
SOURCE_RESULTS = REV / "baselines/results/primary_status_consistent_20261123/results"
SOURCE_MANIFEST = SOURCE_RESULTS / "baseline_manifest.json"
SOURCE_PREDICTION_MANIFEST = SOURCE_RESULTS / "baseline_prediction_manifest.json"
SOURCE_GATE = REV / "temporal/results/eligibility/status_consistent_v4_1/landmark_v4_1"
RECOVERED_RUN = REV / "engine/results/prediction_runs/recovered_external_primary_v4_2_resume3-04b21bfc05943b7f"
RECOVERED_MANIFEST = RECOVERED_RUN / "prediction_manifest.json"
RECOVERED_GATE = REV / "temporal/results/eligibility/recovered_reviewed_v4_2"
IDENTITY_ATTESTATION_PATH = REV / "admin/subject_identity_attestations_v4_2.json"
IDENTITY_ATTESTATION_SHA256 = "199cdf9dfeb277ef49c30f4a4648a6b06bbf1a65958cd1cf84b36f986bb9cd55"
PROC = ROOT / "data/processed"
VARIABLE_META = PROC / "meta_mimic.json"
DEFAULT_OUT = REV / "baselines/transport_v4_2"
TARGET_COHORTS = (
    "nwicu_discharge_v4_1", "sicdb_zero_offset_v4_2", "eicu_semantic_v4_2",
)
SOURCE_COHORTS = ("mimic", "eicu")
ANALYSIS_IDS = (
    "conventional_lr_pattern_only",
    "conventional_hgb_pattern_only",
    "conventional_lr_value_plus_pattern",
    "conventional_hgb_value_plus_pattern",
)
EXPECTED_PARAMS = {
    "conventional_lr_pattern_only": {"C": 0.01},
    "conventional_hgb_pattern_only": {
        "learning_rate": 0.1, "max_iter": 200, "early_stopping": False,
        "max_leaf_nodes": 15, "l2_regularization": 1.0, "min_samples_leaf": 20,
    },
    "conventional_lr_value_plus_pattern": {"C": 0.01},
    "conventional_hgb_value_plus_pattern": {
        "learning_rate": 0.1, "max_iter": 200, "early_stopping": False,
        "max_leaf_nodes": 15, "l2_regularization": 10.0, "min_samples_leaf": 20,
    },
}

for _path in (BASELINE_CODE, ROBUSTNESS_CODE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

_SNAPSHOT_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(_SNAPSHOT_ROOT / "robustness"),
    str(_SNAPSHOT_ROOT / "baselines" / "transport_v4_2"),
    str(_SNAPSHOT_ROOT / "baselines"),
]

from conventional_baselines import (  # noqa: E402
    _engine_module, _identity_hash, _new_estimator, _summarize_conditioned_inputs,
)
from paired_baselines import (  # noqa: E402
    ordered_row_digest, sha256_file, validate_complete_fm_prediction_manifest,
)
from robustness_common import load_checked, read_frame  # noqa: E402
import robustness_common as _robustness_common  # noqa: E402


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _write_frame_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _validate_subject_identity_attestations(
    *, sidecar_path: Path, eligibility_dir: Path, cohorts: tuple[str, ...],
    expected_sidecar_sha256: str | None = IDENTITY_ATTESTATION_SHA256,
) -> tuple[dict[str, dict[str, Any]], str]:
    """Validate transport-only identity supplements against every bound source artifact."""
    sidecar_path = sidecar_path.resolve()
    actual_sidecar_sha = sha256_file(sidecar_path)
    if expected_sidecar_sha256 is not None and actual_sidecar_sha != expected_sidecar_sha256:
        raise ValueError("subject-identity sidecar SHA256 differs from the accepted root attestation")
    document = _json(sidecar_path)
    if document.get("schema") != "ehr_fm_v4_2_explicit_subject_identity_attestations":
        raise ValueError("subject-identity sidecar schema is not recognized")
    if document.get("status") != "root_accepted_within_source_patient_keys_only":
        raise ValueError("subject-identity sidecar is not root-accepted")
    if not isinstance(document.get("cohorts"), dict):
        raise ValueError("subject-identity sidecar lacks cohort records")
    if "RecordID" in str(document.get("forbidden_inference", "")) and any(
            "c2012" in cohort.lower() or "c_2012" in cohort.lower() for cohort in cohorts):
        raise ValueError("RecordID identity evidence is explicitly forbidden")

    gate_dir = eligibility_dir.resolve()
    manifest_path = (gate_dir / "eligibility_manifest.json").resolve()
    manifest = _json(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    validated: dict[str, dict[str, Any]] = {}
    for db in cohorts:
        record = document["cohorts"].get(db)
        if not isinstance(record, dict):
            raise ValueError(f"{db}: subject-identity sidecar record is absent")
        gate_cohort = manifest.get("cohorts", {}).get(db)
        if not isinstance(gate_cohort, dict):
            raise ValueError(f"{db}: original eligibility cohort record is absent")
        table_path = (gate_dir / f"{db}.parquet").resolve()
        sequence_path = Path(str(record.get("sequence_path", ""))).resolve()
        record_path = Path(str(record.get("scope_record_path", ""))).resolve()
        bindings = {
            "eligibility_manifest_path": str(manifest_path),
            "eligibility_manifest_sha256": manifest_sha,
            "eligibility_table_sha256": sha256_file(table_path),
            "sequence_npz_sha256": sha256_file(sequence_path),
            "scope_record_sha256": sha256_file(record_path),
        }
        if str(Path(str(record.get("eligibility_manifest_path", ""))).resolve()) != bindings[
                "eligibility_manifest_path"]:
            raise ValueError(f"{db}: sidecar names a different eligibility manifest")
        if str(record.get("eligibility_manifest_sha256", "")).lower() != bindings[
                "eligibility_manifest_sha256"].lower():
            raise ValueError(f"{db}: sidecar eligibility manifest hash is stale")
        if str(record.get("eligibility_table_sha256", "")).lower() != bindings[
                "eligibility_table_sha256"].lower():
            raise ValueError(f"{db}: sidecar eligibility table hash is stale")
        if str(record.get("sequence_npz_sha256", "")).lower() != bindings[
                "sequence_npz_sha256"].lower():
            raise ValueError(f"{db}: sidecar sequence NPZ hash is stale")
        if str(record.get("scope_record_sha256", "")).lower() != bindings[
                "scope_record_sha256"].lower():
            raise ValueError(f"{db}: sidecar scope-record hash is stale")
        gate_sequence_sha = _robustness_common._source_sequence_hash(gate_cohort)
        if not gate_sequence_sha or gate_sequence_sha.lower() != bindings[
                "sequence_npz_sha256"].lower():
            raise ValueError(f"{db}: sidecar sequence hash differs from the original gate")
        if str(gate_cohort.get("scope_record_sha256", "")).lower() != bindings[
                "scope_record_sha256"].lower():
            raise ValueError(f"{db}: sidecar scope-record hash differs from the original gate")
        for key, value in (("source_patient_identity_verified", True),
                           ("patient_disjoint_claim_allowed", True)):
            if gate_cohort.get(key) is not value:
                raise ValueError(f"{db}: original gate lacks explicit {key}=true")
        original_basis = gate_cohort.get("subject_identity_basis")
        if isinstance(original_basis, dict) and original_basis.get(
                "patient_disjoint_claim_allowed") is False:
            raise ValueError(f"{db}: original gate explicitly forbids patient-disjoint inference")
        if record.get("metadata_relation") != "explicit identity supplement; original gate and prediction bytes untouched":
            raise ValueError(f"{db}: sidecar does not preserve the original gate/prediction artifacts")
        basis = record.get("subject_identity_basis")
        if not isinstance(basis, dict) or not str(basis.get("basis", "")).strip():
            raise ValueError(f"{db}: sidecar patient identity basis is missing")
        if basis.get("patient_disjoint_claim_allowed") is not True:
            raise ValueError(f"{db}: sidecar does not explicitly allow within-source patient clustering")
        if basis.get("resampling_unit") != "patient":
            raise ValueError(f"{db}: sidecar resampling unit is not patient")
        if basis.get("identifier_scope") != "within this original data source":
            raise ValueError(f"{db}: sidecar identity scope is not limited to the original source")
        if basis.get("cross_source_patient_independence_verified") is not False:
            raise ValueError(f"{db}: cross-source patient independence must remain unverified")
        if basis.get("cross_scope_patient_independence_verified") is not False:
            raise ValueError(f"{db}: cross-scope patient independence must remain unverified")
        validated[db] = {
            **record,
            "eligibility_manifest_path": str(manifest_path),
            "eligibility_manifest_sha256": manifest_sha,
            "eligibility_table_path": str(table_path),
            "sequence_path": str(sequence_path),
            "scope_record_path": str(record_path),
            "original_subject_identity_basis": original_basis,
            "original_subject_identity_verified": gate_cohort.get("source_patient_identity_verified"),
        }
    return validated, actual_sidecar_sha


def _source_choice_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    choices = manifest.get("choices")
    if not isinstance(choices, list):
        raise ValueError("original baseline manifest lacks the completed frozen source choices")
    mapped: dict[str, dict[str, Any]] = {}
    for row in choices:
        if not isinstance(row, dict):
            continue
        analysis_id = str(row.get("analysis_id") or
                          f"{row.get('model_tag')}_{row.get('feature_type')}")
        if analysis_id in mapped:
            raise ValueError(f"original baseline manifest repeats choice {analysis_id}")
        mapped[analysis_id] = row
    if set(mapped) != set(ANALYSIS_IDS):
        raise ValueError(f"original baseline choice set differs from the four registered arms: {sorted(mapped)}")
    for analysis_id, choice in mapped.items():
        params = choice.get("selected_parameters")
        if params != EXPECTED_PARAMS[analysis_id]:
            raise ValueError(f"{analysis_id}: selected parameters differ from the registered frozen values")
        if choice.get("final_model_fit_scope") != "MIMIC source split 0 only; split 1 is selection-only and not refit":
            raise ValueError(f"{analysis_id}: source-fit scope is not MIMIC split 0 only")
        if choice.get("selected_parameters_recovered_from_run_log") is not True:
            raise ValueError(f"{analysis_id}: selected parameter evidence is not bound to the completed run log")
    return mapped


def _load_target_fm_frames() -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]], dict[str, str]]:
    fm_doc = _json(RECOVERED_MANIFEST)
    validate_complete_fm_prediction_manifest(fm_doc)
    fm_path = RECOVERED_RUN / "main20k_real.parquet"
    raw = read_frame(fm_path)
    identity_attestations, identity_attestation_sha = _validate_subject_identity_attestations(
        sidecar_path=IDENTITY_ATTESTATION_PATH, eligibility_dir=RECOVERED_GATE,
        cohorts=TARGET_COHORTS,
    )
    endpoint_by_db: dict[str, str] = {}
    for db in TARGET_COHORTS:
        table = pd.read_parquet(RECOVERED_GATE / f"{db}.parquet", columns=["endpoint", "eligible_24h", "temporal_verified"])
        selected = table.loc[
            table["eligible_24h"].astype(bool) & table["temporal_verified"].astype(bool), "endpoint"
        ].astype(str).unique()
        if len(selected) != 1:
            raise ValueError(f"{db}: strict eligible rows do not have exactly one endpoint")
        endpoint_by_db[db] = str(selected[0])
    loaded: dict[str, pd.DataFrame] = {}
    gates: dict[str, dict[str, Any]] = {}
    original_identity_adapter = _robustness_common.manifest_subject_identity

    def _attested_subject_identity(path: str | Path, db: str) -> tuple[str, bool, str]:
        original_basis, original_ok, original_label = original_identity_adapter(path, db)
        if original_ok:
            return original_basis, original_ok, original_label
        attestation = identity_attestations.get(str(db))
        if attestation is None:
            return original_basis, original_ok, original_label
        basis = attestation["subject_identity_basis"]
        return str(basis["basis"]), True, "patient"

    _robustness_common.manifest_subject_identity = _attested_subject_identity
    try:
        for endpoint in sorted(set(endpoint_by_db.values())):
            part, part_gates = load_checked(
                fm_path, RECOVERED_GATE, model_tag="main20k", analysis_id="main20k_real",
                input_condition="real", endpoint=endpoint,
                prediction_manifest_path=RECOVERED_MANIFEST,
            )
            for db in TARGET_COHORTS:
                if endpoint_by_db[db] == endpoint:
                    if db not in part or db not in part_gates:
                        raise ValueError(f"recovered FM strict-manifest adapter omitted {db}")
                    loaded[db], gates[db] = part[db], part_gates[db]
    finally:
        _robustness_common.manifest_subject_identity = original_identity_adapter
    if set(loaded) != set(TARGET_COHORTS):
        raise ValueError(f"recovered FM target adapter returned wrong cohorts: {sorted(loaded)}")
    for db, frame in loaded.items():
        gate = gates[db]
        if gate.get("bootstrap_cluster_unit") != "patient" or gate.get(
                "independent_patient_clustering_verified") is not True:
            raise ValueError(f"{db}: patient-cluster evaluation is not explicitly verified")
        if frame.empty:
            raise ValueError(f"{db}: recovered FM target frame is empty")
        if frame["stay_id"].duplicated().any():
            raise ValueError(f"{db}: recovered FM artifact contains duplicate stays")
        attestation = identity_attestations[db]
        gate["original_subject_identity_basis"] = attestation["original_subject_identity_basis"]
        gate["subject_identity_attestation"] = {
            "path": str(IDENTITY_ATTESTATION_PATH.resolve()),
            "sha256": identity_attestation_sha,
            "cohort_record": attestation,
            "scope": "within-source patient clustering and split grouping only; no cross-source or cross-scope independence",
        }
        gate["independent_patient_clustering_verified"] = True
        gate["bootstrap_cluster_unit"] = "patient"
    print(json.dumps({
        "stage": "recovered_manifest_adapter_preflight_complete",
        "manifest_sha256": sha256_file(RECOVERED_MANIFEST),
        "gate_manifest_sha256": sha256_file(RECOVERED_GATE / "eligibility_manifest.json"),
        "subject_identity_attestation_sha256": identity_attestation_sha,
        "target_counts": {
            db: {
                "stays": int(len(frame)), "patients": int(frame["subject"].nunique()),
                "events": int(frame["y"].sum()), "nonevents": int(len(frame) - frame["y"].sum()),
                "endpoint": endpoint_by_db[db],
                "eligibility_table_sha256": gates[db]["eligibility_table_sha256"],
                "sequence_npz_sha256": identity_attestations[db]["sequence_npz_sha256"],
            }
            for db, frame in loaded.items()
        },
    }, sort_keys=True), flush=True)
    return loaded, gates, endpoint_by_db


def write_transport_plan(*, out_dir: Path) -> dict[str, Any]:
    source_manifest = _json(SOURCE_MANIFEST)
    choices = _source_choice_map(source_manifest)
    fm_doc = _json(RECOVERED_MANIFEST)
    validate_complete_fm_prediction_manifest(fm_doc)
    plan = {
        "status": "preflight_pending_source_reproduction",
        "scope": "frozen MIMIC source baseline transport; no target tuning or source parameter selection",
        "source_baseline_manifest": str(SOURCE_MANIFEST),
        "source_baseline_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "source_prediction_manifest": str(SOURCE_PREDICTION_MANIFEST),
        "source_prediction_manifest_sha256": sha256_file(SOURCE_PREDICTION_MANIFEST),
        "source_gate": str(SOURCE_GATE),
        "source_gate_manifest_sha256": sha256_file(SOURCE_GATE / "eligibility_manifest.json"),
        "source_training": "fit selected model parameters on MIMIC eligible split 0 only; split 1 is selection-only; no retuning or split-1 refit",
        "frozen_arms": {
            analysis_id: {
                "model_tag": choices[analysis_id]["model_tag"],
                "feature_type": choices[analysis_id]["feature_type"],
                "selected_parameters": choices[analysis_id]["selected_parameters"],
            }
            for analysis_id in ANALYSIS_IDS
        },
        "source_reproduction_gate": {
            "cohorts": ["mimic", "eicu"],
            "identity_fields": ["db", "stay_id", "subject", "rows", "split", "y"],
            "maximum_absolute_probability_difference": 1e-8,
            "target_scoring_allowed_only_after_all_four_arms_pass": True,
        },
        "recovered_target_manifest": str(RECOVERED_MANIFEST),
        "recovered_target_manifest_sha256": sha256_file(RECOVERED_MANIFEST),
        "recovered_gate": str(RECOVERED_GATE),
        "recovered_gate_manifest_sha256": sha256_file(RECOVERED_GATE / "eligibility_manifest.json"),
        "target_cohorts": list(TARGET_COHORTS),
        "target_scoring": "same registered sequence arrays and binned V/B/H input summaries; score only after source reproduction; no target fitting",
        "paired_comparisons": "main20k_real minus each of the four frozen baselines; exact stay/patient/row/split/label identity; 1000 shared patient-cluster draws",
        "shared_calibration": {
            "cohorts": list(TARGET_COHORTS), "budgets_requested": [50, 200, 500, 2000],
            "repeats": 200, "patient_disjoint_adaptation_assessment": True,
            "one_outcome_blind_random_adaptation_stay_per_patient": True,
            "arms": ["main20k_real", *ANALYSIS_IDS],
            "methods": ["Jeffreys-smoothed prevalence", "offset", "Platt"],
            "freeze_sampling_plans_before_metric_evaluation": True,
            "budgets_above_available_adaptation_patients": "record unsupported; do not resample or relax the gate",
        },
        "excluded": [
            "C calibration: RecordID is not verified as a patient identifier",
            "any candidate-grid search on target cohorts",
            "any alteration of the original baseline prediction artifacts",
        ],
        "clinical_input_availability_verified": False,
        "clinical_promotion_allowed": False,
        "patient_level_exported_to_local": False,
    }
    _write_json_atomic(out_dir / "plan.json", plan)
    return plan


def _load_source_frames() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, pd.DataFrame]], Any, int]:
    source_manifest = _json(SOURCE_MANIFEST)
    baseline_doc = _json(SOURCE_PREDICTION_MANIFEST)
    if source_manifest.get("status") != "complete" or baseline_doc.get("status") != "complete":
        raise ValueError("source baseline manifests must both be complete")
    choices = _source_choice_map(source_manifest)
    if sha256_file(SOURCE_GATE / "eligibility_manifest.json") != source_manifest.get(
            "eligibility_manifest_sha256"):
        raise ValueError("source eligibility manifest SHA256 differs from the original baseline manifest")
    engine = _engine_module()
    meta = _json(VARIABLE_META)
    variables = meta.get("vars")
    if not isinstance(variables, list) or not variables:
        raise ValueError("source variable vocabulary is absent")
    n_variables = len(variables)

    source_features: dict[str, dict[str, Any]] = {}
    for db in SOURCE_COHORTS:
        eligible, entry = engine.load_ready_eligibility(db, SOURCE_GATE)
        if entry.get("retrospective_landmark_prediction_allowed") is not True:
            raise ValueError(f"{db}: source cohort no longer passes its cleared primary gate")
        eligible = eligible.loc[eligible["endpoint"].astype(str) == "hospital"].copy()
        sequence_path = PROC / f"seq_{db}.npz"
        sequence_sha = sha256_file(sequence_path)
        if str(entry.get("sequence_npz_sha256", entry.get("sourcehashes", {}).get("sequence_npz", ""))).lower() != sequence_sha.lower():
            raise ValueError(f"{db}: source sequence hash differs from original eligibility attestation")
        ds = engine.load_sequence(db)
        rows = engine.condition_rows(ds, "real", ())
        matched = engine.align_eligible_rows(db, ds, eligible, rows)
        if matched.empty:
            raise ValueError(f"{db}: cleared source frame has no eligible sequence rows")
        matched["split"] = np.asarray(ds["split"], dtype=np.int8)[
            matched["rows"].to_numpy(dtype=np.int64)]
        matched["y"] = matched["y_mort"].to_numpy(dtype=np.int8)
        matched["subject"] = matched["subject"].astype(str)
        feature_value, feature_pattern = _summarize_conditioned_inputs(
            ds, matched["rows"].to_numpy(dtype=np.int64), n_variables,
            condition="real", variable_ids=(), engine=engine,
        )
        source_features[db] = {
            "frame": matched,
            "features": {"value_plus_pattern": feature_value, "pattern_only": feature_pattern},
            "sequence_sha256": sequence_sha,
            "eligibility_table_sha256": sha256_file(SOURCE_GATE / f"{db}.parquet"),
        }
        print(json.dumps({
            "stage": "frozen_source_inputs_ready", "db": db,
            "eligible_rows": int(len(matched)),
            "split0_rows": int((matched["split"] == 0).sum()),
            "split1_rows": int((matched["split"] == 1).sum()),
            "split2_rows": int((matched["split"] == 2).sum()),
            "sequence_sha256": sequence_sha,
        }, sort_keys=True), flush=True)

    source_baselines: dict[str, dict[str, pd.DataFrame]] = {}
    for analysis_id in ANALYSIS_IDS:
        choice = choices[analysis_id]
        records = [row for row in baseline_doc.get("predictions", [])
                   if row.get("analysis_id") == analysis_id and row.get("db") == "mimic"]
        if len(records) != 1:
            raise ValueError(f"{analysis_id}: source prediction manifest entry is absent or duplicated")
        prediction_path = Path(records[0]["prediction_file"])
        frames, gates = load_checked(
            prediction_path, SOURCE_GATE,
            model_tag=str(choice["model_tag"]), analysis_id=analysis_id,
            input_condition="real", endpoint="hospital",
            prediction_manifest_path=SOURCE_PREDICTION_MANIFEST,
        )
        if set(frames) != set(SOURCE_COHORTS):
            raise ValueError(f"{analysis_id}: strict source artifact has unexpected cohorts {sorted(frames)}")
        source_baselines[analysis_id] = frames
        print(json.dumps({
            "stage": "frozen_source_prediction_adapter_verified",
            "analysis_id": analysis_id,
            "rows_by_cohort": {db: int(len(frame)) for db, frame in frames.items()},
            "source_prediction_manifest_sha256": sha256_file(SOURCE_PREDICTION_MANIFEST),
        }, sort_keys=True), flush=True)
    return choices, source_baselines, source_features, n_variables


def _exact_prediction_match(generated: pd.DataFrame, reference: pd.DataFrame,
                            *, analysis_id: str, max_abs_tolerance: float = 1e-8) -> dict[str, Any]:
    identity = ["db", "stay_id", "subject", "rows", "split", "y"]
    left = generated[identity + ["p"]].copy()
    right = reference[identity + ["p"]].copy()
    left["stay_id"] = left["stay_id"].astype(str)
    right["stay_id"] = right["stay_id"].astype(str)
    joined = left.merge(right, on=["db", "stay_id"], how="outer", validate="one_to_one",
                        suffixes=("_new", "_saved"), indicator=True)
    if joined["_merge"].ne("both").any() or len(left) != len(right):
        raise ValueError(f"{analysis_id}: reproduced source predictions differ in exact db/stay IDs")
    for field in identity[2:]:
        if not joined[f"{field}_new"].astype(str).equals(joined[f"{field}_saved"].astype(str)):
            raise ValueError(f"{analysis_id}: reproduced source predictions differ in {field}")
    maximum = float(np.max(np.abs(joined["p_new"].to_numpy(float)
                                  - joined["p_saved"].to_numpy(float))))
    if not np.isfinite(maximum) or maximum > max_abs_tolerance:
        raise ValueError(
            f"{analysis_id}: frozen source refit differs from original by {maximum:.17g} "
            f"(maximum allowed {max_abs_tolerance:.1e}); target scoring is blocked"
        )
    return {"status": "pass", "rows": int(len(joined)),
            "n_mimic_rows": int(joined["db"].eq("mimic").sum()),
            "n_eicu_rows": int(joined["db"].eq("eicu").sum()),
            "max_abs_probability_difference": maximum,
            "tolerance": max_abs_tolerance,
            "exact_identity_and_label_match": True}


def _exact_target_identity_match(sequence_rows: pd.DataFrame, fm_rows: pd.DataFrame,
                                 *, db: str) -> None:
    """Require exact target identity while normalizing numeric/string ID dtypes."""
    identity = ["db", "stay_id", "subject", "rows", "split", "y"]
    left = sequence_rows[identity].copy()
    right = fm_rows[identity].copy()
    invalid = {"", "nan", "none", "null", "-1"}
    for frame in (left, right):
        raw_keys = frame[["db", "stay_id"]]
        if raw_keys.isna().any().any() or raw_keys.duplicated().any():
            raise ValueError(f"{db}: duplicate or missing raw target identity keys")
        frame["db"] = frame["db"].astype("string").str.strip()
        frame["stay_id"] = frame["stay_id"].astype("string").str.strip()
        normalized_keys = frame[["db", "stay_id"]]
        if (normalized_keys.isna().any().any()
                or normalized_keys["db"].str.lower().isin(invalid).any()
                or normalized_keys["stay_id"].str.lower().isin(invalid).any()):
            raise ValueError(f"{db}: target identity key normalization produced an invalid identifier")
        if normalized_keys.duplicated().any():
            raise ValueError(f"{db}: duplicate target identity keys after string normalization")
    merged = left.merge(right, on=["db", "stay_id"], how="outer", validate="one_to_one",
                        suffixes=("_sequence", "_fm"), indicator=True)
    if merged["_merge"].ne("both").any() or len(left) != len(right):
        raise ValueError(f"{db}: frozen source target frame does not match exact FM stay IDs")
    for field in identity[2:]:
        if not merged[f"{field}_sequence"].astype(str).equals(
                merged[f"{field}_fm"].astype(str)):
            raise ValueError(f"{db}: frozen source target identity/label mismatch in {field}")


def fit_verify_and_freeze(*, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = write_transport_plan(out_dir=out_dir)
    recovered_fm_doc = _json(RECOVERED_MANIFEST)
    validate_complete_fm_prediction_manifest(recovered_fm_doc)
    plan["status"] = "source_artifact_adapter_preflight_passed"
    plan["source_artifact_adapter_preflight"] = (
        "original MIMIC/eICU baseline prediction artifacts load through the strict current baseline manifest adapter; "
        "recovered FM manifest run protocol is valid; target eligibility adapter is run before any target score"
    )
    _write_json_atomic(out_dir / "plan.json", plan)
    choices, references, source, n_variables = _load_source_frames()
    source_train = source["mimic"]["frame"]
    train_mask = source_train["split"].to_numpy(dtype=int) == 0
    validation_mask = source_train["split"].to_numpy(dtype=int) == 1
    test_mask = source_train["split"].to_numpy(dtype=int) == 2
    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        raise ValueError("MIMIC source rows must contain the original split 0/1/2 partitions")
    train_identity = _identity_hash("mimic", source_train.loc[train_mask])
    for choice in choices.values():
        if choice.get("source_train_identity_sha256") != train_identity:
            raise ValueError("current MIMIC split-0 row identity differs from the original source fit")
        if choice.get("source_fit_rows") != int(train_mask.sum()):
            raise ValueError("MIMIC source train row count differs from the original source fit")

    frozen_models: dict[str, dict[str, Any]] = {}
    reproduction: dict[str, dict[str, Any]] = {}
    for analysis_id in ANALYSIS_IDS:
        choice = choices[analysis_id]
        model_tag, feature_type = str(choice["model_tag"]), str(choice["feature_type"])
        X = source["mimic"]["features"][feature_type]
        y = source_train["y"].to_numpy(dtype=np.int8)
        scaler: StandardScaler | None = None
        X_train = X[train_mask]
        if model_tag == "conventional_lr":
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
        model = _new_estimator(model_tag, choice["selected_parameters"], seed=20261122)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X_train, y[train_mask])
        convergence = any(issubclass(w.category, ConvergenceWarning) for w in caught)
        generated_parts: list[pd.DataFrame] = []
        for db in SOURCE_COHORTS:
            source_frame = source[db]["frame"]
            eval_mask = source_frame["split"].to_numpy(dtype=int) == 2 if db == "mimic" else np.ones(
                len(source_frame), dtype=bool)
            if not eval_mask.any():
                raise ValueError(f"{analysis_id}/{db}: source reproduction cohort is empty")
            X_eval = source[db]["features"][feature_type][eval_mask]
            if scaler is not None:
                X_eval = scaler.transform(X_eval)
            probabilities = model.predict_proba(X_eval)[:, 1]
            rows = source_frame.loc[eval_mask]
            generated_parts.append(pd.DataFrame({
                "db": db, "stay_id": rows["stay_id"].astype(str).to_numpy(),
                "subject": rows["subject"].astype(str).to_numpy(),
                "rows": rows["rows"].to_numpy(dtype=np.int64),
                "split": rows["split"].to_numpy(dtype=np.int8),
                "y": rows["y"].to_numpy(dtype=np.int8), "p": probabilities,
            }))
        generated = pd.concat(generated_parts, ignore_index=True)
        ref = pd.concat([references[analysis_id][db] for db in SOURCE_COHORTS], ignore_index=True)
        reproduction[analysis_id] = _exact_prediction_match(
            generated, ref, analysis_id=analysis_id, max_abs_tolerance=1e-8,
        )
        frozen_models[analysis_id] = {
            "model_tag": model_tag, "feature_type": feature_type,
            "selected_parameters": choice["selected_parameters"],
            "scaler": scaler, "estimator": model,
        }
        print(json.dumps({
            "stage": "frozen_source_prediction_reproduction",
            "analysis_id": analysis_id, **reproduction[analysis_id],
            "fit_seconds": time.perf_counter() - started,
            "convergence_warning": bool(convergence),
        }, sort_keys=True), flush=True)

    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = model_dir / "frozen_source_models.pkl"
    if bundle_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen source model bundle: {bundle_path}")
    bundle = {
        "status": "source_reproduction_passed",
        "models": frozen_models,
        "seed": 20261122,
        "n_variables": int(n_variables),
        "variable_vocab_sha256": sha256_file(VARIABLE_META),
    }
    bundle_tmp = bundle_path.with_name(f".{bundle_path.name}.tmp")
    with bundle_tmp.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    bundle_tmp.replace(bundle_path)
    source_manifest = _json(SOURCE_MANIFEST)
    source_prediction_manifest = _json(SOURCE_PREDICTION_MANIFEST)
    fm_doc = _json(RECOVERED_MANIFEST)
    manifest = {
        "status": "source_reproduction_passed",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "analysis_scope": "frozen MIMIC source refits at original selected parameters; target scoring is gated on exact reproduction",
        "source_baseline_manifest": str(SOURCE_MANIFEST),
        "source_baseline_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "source_baseline_prediction_manifest": str(SOURCE_PREDICTION_MANIFEST),
        "source_baseline_prediction_manifest_sha256": sha256_file(SOURCE_PREDICTION_MANIFEST),
        "source_gate_manifest_sha256": sha256_file(SOURCE_GATE / "eligibility_manifest.json"),
        "source_sequence_npz_sha256": {
            db: source[db]["sequence_sha256"] for db in SOURCE_COHORTS
        },
        "mimic_split0_identity_sha256": train_identity,
        "variable_vocab_sha256": sha256_file(VARIABLE_META),
        "baseline_code_sha256": sha256_file(BASELINE_CODE / "conventional_baselines.py"),
        "feature_code_sha256": sha256_file(BASELINE_CODE / "baseline_features.py"),
        "engine_code_sha256": sha256_file(REV / "engine/code/v4_engine.py"),
        "runtime": {
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        },
        "source_fits": {
            analysis_id: {
                "model_tag": choices[analysis_id]["model_tag"],
                "feature_type": choices[analysis_id]["feature_type"],
                "selected_parameters": choices[analysis_id]["selected_parameters"],
                "source_fit_scope": "MIMIC eligible split 0 only; split 1 is not refit",
                "source_train_rows": int(train_mask.sum()),
                "source_train_identity_sha256": train_identity,
                "reproduction": reproduction[analysis_id],
            }
            for analysis_id in ANALYSIS_IDS
        },
        "source_model_bundle": str(bundle_path),
        "source_model_bundle_sha256": sha256_file(bundle_path),
        "source_model_classes_used_for_selection": {
            "selected_parameters_recovered_from_run_log": all(
                row.get("selected_parameters_recovered_from_run_log") is True
                for row in source_manifest["choices"]
            ),
            "original_prediction_manifest_status": source_prediction_manifest.get("status"),
            "original_baseline_manifest_status": source_manifest.get("status"),
        },
        "target_scoring_gate": "all four exact MIMIC split-2/eICU original predictions reproduced with row ID, split, label identity and max probability error <= 1e-8",
        "recovered_fm_manifest_sha256_at_preflight": sha256_file(RECOVERED_MANIFEST),
        "recovered_target_prediction_preflight": "run protocol validated; strict target eligibility adapter still required before any target scoring",
        "clinical_input_availability_verified": False,
        "clinical_promotion_allowed": False,
        "patient_level_exported_to_local": False,
    }
    manifest_path = out_dir / "source_reproduction_manifest.json"
    _write_json_atomic(manifest_path, manifest)
    plan["status"] = "source_reproduction_passed"
    plan["source_reproduction_manifest_sha256"] = sha256_file(manifest_path)
    _write_json_atomic(out_dir / "plan.json", plan)
    print(json.dumps({
        "status": "source_reproduction_passed",
        "source_model_bundle": str(bundle_path),
        "source_model_bundle_sha256": manifest["source_model_bundle_sha256"],
        "target_scoring_unlocked": True,
    }, sort_keys=True), flush=True)
    return manifest


def _load_source_bundle(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = out_dir / "source_reproduction_manifest.json"
    manifest = _json(manifest_path)
    if manifest.get("status") != "source_reproduction_passed":
        raise ValueError("target scoring is blocked until the four source predictions reproduce")
    bundle_path = Path(manifest["source_model_bundle"])
    if not bundle_path.is_file() or sha256_file(bundle_path) != manifest.get("source_model_bundle_sha256"):
        raise ValueError("frozen source model bundle is absent or has changed")
    if sha256_file(SOURCE_MANIFEST) != manifest.get("source_baseline_manifest_sha256"):
        raise ValueError("original selected baseline manifest changed after source verification")
    if sha256_file(SOURCE_PREDICTION_MANIFEST) != manifest.get("source_baseline_prediction_manifest_sha256"):
        raise ValueError("original baseline prediction manifest changed after source verification")
    if sha256_file(RECOVERED_MANIFEST) != manifest.get("recovered_fm_manifest_sha256_at_preflight"):
        raise ValueError("recovered FM prediction manifest changed after source verification")
    with bundle_path.open("rb") as handle:
        bundle = pickle.load(handle)
    if bundle.get("status") != "source_reproduction_passed" or set(bundle.get("models", {})) != set(ANALYSIS_IDS):
        raise ValueError("frozen source model bundle does not contain the four reproduced arms")
    return manifest, bundle


def score_recovered_targets(*, out_dir: Path,
                            cohorts: tuple[str, ...] = TARGET_COHORTS) -> dict[str, Any]:
    if not cohorts or len(set(cohorts)) != len(cohorts) or any(db not in TARGET_COHORTS for db in cohorts):
        raise ValueError(f"requested recovered target cohorts must be a unique subset of {TARGET_COHORTS}")
    source_manifest, bundle = _load_source_bundle(out_dir)
    plan = _json(out_dir / "plan.json")
    target_frames, target_gates, endpoint_by_db = _load_target_fm_frames()
    identity_sidecar = target_gates[TARGET_COHORTS[0]]["subject_identity_attestation"]
    plan["subject_identity_attestation"] = {
        "path": identity_sidecar["path"],
        "sha256": identity_sidecar["sha256"],
        "scope": identity_sidecar["scope"],
        "validated_cohorts": list(TARGET_COHORTS),
    }
    plan["actual_recovered_target_adapter_preflight"] = {
        "status": "complete",
        "cohorts_requested_for_scoring": list(cohorts),
        "target_cohort_counts": {
            db: {"n_stays": int(len(frame)), "n_patients": int(frame["subject"].nunique()),
                 "events": int(frame["y"].sum()), "endpoint": endpoint_by_db[db]}
            for db, frame in target_frames.items() if db in cohorts
        },
        "cluster_unit_by_cohort": {
            db: gate["bootstrap_cluster_unit"] for db, gate in target_gates.items() if db in cohorts
        },
    }
    _write_json_atomic(out_dir / "plan.json", plan)
    fm_doc = _json(RECOVERED_MANIFEST)
    records = {
        str(row["db"]): row for row in fm_doc.get("predictions", [])
        if row.get("analysis_id") == "main20k_real"
    }
    engine = _engine_module()
    n_variables = int(bundle["n_variables"])
    scored_dir = out_dir / "checkpoints"
    scored_dir.mkdir(parents=True, exist_ok=True)
    all_predictions: list[pd.DataFrame] = []
    checkpoint_records: dict[str, dict[str, Any]] = {}
    prediction_manifest_sha = sha256_file(RECOVERED_MANIFEST)
    model_sha = str(source_manifest["source_model_bundle_sha256"])
    for db in cohorts:
        gate = target_gates[db]
        eligible, entry = engine.load_ready_eligibility(db, RECOVERED_GATE)
        endpoint = endpoint_by_db[db]
        eligible = eligible.loc[eligible["endpoint"].astype(str) == endpoint].copy()
        if entry.get("retrospective_landmark_prediction_allowed") is not True:
            raise ValueError(f"{db}: recovered target gate no longer permits primary retrospective evaluation")
        identity_record = gate["subject_identity_attestation"]["cohort_record"]
        sequence_path = Path(identity_record["sequence_path"])
        sequence_sha = sha256_file(sequence_path)
        ds = engine.load_sequence_registered(db)
        rows = engine.condition_rows(ds, "real", ())
        matched = engine.align_eligible_rows(db, ds, eligible, rows)
        matched["split"] = np.asarray(ds["split"], dtype=np.int8)[
            matched["rows"].to_numpy(dtype=np.int64)]
        matched["y"] = matched["y_mort"].to_numpy(dtype=np.int8)
        matched["subject"] = matched["subject"].astype(str)
        fm = target_frames[db].copy()
        try:
            _exact_target_identity_match(matched, fm, db=db)
        except ValueError:
            sequence_ids = set(matched["stay_id"].astype(str))
            fm_ids = set(fm["stay_id"].astype(str))
            sequence_keys = set(zip(matched["db"].astype(str), matched["stay_id"].astype(str)))
            fm_keys = set(zip(fm["db"].astype(str), fm["stay_id"].astype(str)))
            left = matched[["db", "stay_id"]].copy()
            right = fm[["db", "stay_id"]].copy()
            for frame in (left, right):
                frame["db"] = frame["db"].astype(str)
                frame["stay_id"] = frame["stay_id"].astype(str)
            merged = left.merge(right, on=["db", "stay_id"], how="outer",
                                validate="one_to_one", indicator=True)
            common_ids = sorted(sequence_ids & fm_ids)
            if common_ids:
                sequence_by_id = matched.assign(stay_id=matched["stay_id"].astype(str)).set_index("stay_id")
                fm_by_id = fm.assign(stay_id=fm["stay_id"].astype(str)).set_index("stay_id")
                identity_mismatches = {
                    field: int(sum(str(sequence_by_id.loc[stay_id, field]) !=
                                   str(fm_by_id.loc[stay_id, field]) for stay_id in common_ids))
                    for field in ("subject", "rows", "split", "y")
                }
            else:
                identity_mismatches = {field: 0 for field in ("subject", "rows", "split", "y")}
            mismatch_record = {
                "stage": "recovered_target_identity_mismatch",
                "db": db,
                "sequence_rows": int(len(matched)), "sequence_unique_stays": int(len(sequence_ids)),
                "fm_rows": int(len(fm)), "fm_unique_stays": int(len(fm_ids)),
                "sequence_only_stays": int(len(sequence_ids - fm_ids)),
                "fm_only_stays": int(len(fm_ids - sequence_ids)),
                "common_stays": int(len(common_ids)),
                "sequence_db_values": sorted(matched["db"].astype(str).unique().tolist()),
                "fm_db_values": sorted(fm["db"].astype(str).unique().tolist()),
                "sequence_only_db_stay_keys": int(len(sequence_keys - fm_keys)),
                "fm_only_db_stay_keys": int(len(fm_keys - sequence_keys)),
                "merged_rows": int(len(merged)),
                "outer_merge_left_only": int(merged["_merge"].eq("left_only").sum()),
                "outer_merge_right_only": int(merged["_merge"].eq("right_only").sum()),
                "identity_mismatches_among_common_stays": identity_mismatches,
                "sequence_path": str(sequence_path),
                "sequence_sha256": sequence_sha,
                "fm_prediction_manifest_sha256": prediction_manifest_sha,
                "eligibility_manifest_sha256": gate["eligibility_manifest_sha256"],
                "eligibility_table_sha256": gate["eligibility_table_sha256"],
            }
            print(json.dumps(mismatch_record, sort_keys=True), flush=True)
            raise
        feature_value, feature_pattern = _summarize_conditioned_inputs(
            ds, matched["rows"].to_numpy(dtype=np.int64), n_variables,
            condition="real", variable_ids=(), engine=engine,
        )
        features = {"value_plus_pattern": feature_value, "pattern_only": feature_pattern}
        for analysis_id in ANALYSIS_IDS:
            checkpoint_path = scored_dir / f"{db}__{analysis_id}.parquet"
            sidecar_path = checkpoint_path.with_suffix(".manifest.json")
            expected_sidecar = {
                "db": db, "analysis_id": analysis_id,
                "source_model_bundle_sha256": model_sha,
                "fm_prediction_manifest_sha256": prediction_manifest_sha,
                "eligibility_table_sha256": gate["eligibility_table_sha256"],
                "sequence_npz_sha256": sequence_sha,
                "subject_identity_attestation_sha256": identity_sidecar["sha256"],
                "n_rows": int(len(fm)),
            }
            if checkpoint_path.is_file() and sidecar_path.is_file():
                saved_sidecar = _json(sidecar_path)
                if any(saved_sidecar.get(key) != value for key, value in expected_sidecar.items()):
                    raise ValueError(f"stale recovered transport checkpoint: {checkpoint_path}")
                if sha256_file(checkpoint_path) != saved_sidecar.get("prediction_file_sha256"):
                    raise ValueError(f"recovered transport checkpoint hash mismatch: {checkpoint_path}")
                scored = read_frame(checkpoint_path)
                if len(scored) != len(fm):
                    raise ValueError(f"recovered transport checkpoint row count mismatch: {checkpoint_path}")
                identity = ["db", "stay_id", "subject", "rows", "split", "y"]
                joined_checkpoint = scored[identity].merge(
                    fm[identity], on=["db", "stay_id"], how="outer",
                    suffixes=("_saved", "_fm"), validate="one_to_one", indicator=True,
                )
                if joined_checkpoint["_merge"].ne("both").any():
                    raise ValueError(f"recovered transport checkpoint ID set differs: {checkpoint_path}")
                for field in identity[2:]:
                    if not joined_checkpoint[f"{field}_saved"].astype(str).equals(
                            joined_checkpoint[f"{field}_fm"].astype(str)):
                        raise ValueError(f"recovered transport checkpoint identity differs: {checkpoint_path}")
            else:
                model_spec = bundle["models"][analysis_id]
                X = features[model_spec["feature_type"]]
                if model_spec["scaler"] is not None:
                    X = model_spec["scaler"].transform(X)
                probability = model_spec["estimator"].predict_proba(X)[:, 1]
                fm_record = records[db]
                scored = pd.DataFrame({
                    "db": db, "analysis_id": analysis_id,
                    "stay_id": matched["stay_id"].astype(str).to_numpy(),
                    "subject": matched["subject"].astype(str).to_numpy(),
                    "rows": matched["rows"].to_numpy(dtype=np.int64),
                    "arrayindex": matched["rows"].to_numpy(dtype=np.int64),
                    "split": matched["split"].to_numpy(dtype=np.int8),
                    "y": matched["y"].to_numpy(dtype=np.int8),
                    "p": probability,
                    "model_tag": model_spec["model_tag"],
                    "feature_type": model_spec["feature_type"],
                    "input_condition": "real",
                    "input_mask_sha256": fm_record["input_mask_sha256"],
                    "endpoint": matched["endpoint"].astype(str).to_numpy(),
                    "role": "evaluation",
                    "prediction_scope": fm_record["prediction_scope"],
                    "input_scope": fm_record["input_scope"],
                    "clinical_input_availability_verified": False,
                    "eligibility_manifest_sha256": fm_record["eligibility_manifest_sha256"],
                    "eligibility_table_sha256": fm_record["eligibility_table_sha256"],
                    "evidence_basis": entry.get("evidence_basis", entry.get("evidence")),
                })
                _write_frame_atomic(checkpoint_path, scored)
                sidecar = {
                    **expected_sidecar,
                    "prediction_file_sha256": sha256_file(checkpoint_path),
                    "ordered_row_digest_sha256": ordered_row_digest(scored),
                    "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                _write_json_atomic(sidecar_path, sidecar)
            all_predictions.append(scored)
            checkpoint_records[f"{db}/{analysis_id}"] = {
                **expected_sidecar,
                "prediction_file": str(checkpoint_path),
                "prediction_file_sha256": sha256_file(checkpoint_path),
                "ordered_row_digest_sha256": ordered_row_digest(scored),
            }
            print(json.dumps({
                "stage": "recovered_target_model_checkpoint_complete",
                "db": db, "analysis_id": analysis_id, "rows": int(len(scored)),
                "endpoint": endpoint,
            }, sort_keys=True), flush=True)

    pred_dir = out_dir / "results/predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    cohort_suffix = "" if set(cohorts) == set(TARGET_COHORTS) else "__" + "__".join(cohorts)
    target_pred_path = pred_dir / f"recovered_frozen_baseline_predictions{cohort_suffix}.parquet"
    if target_pred_path.exists():
        raise FileExistsError(f"refusing to overwrite recovered baseline predictions: {target_pred_path}")
    combined = pd.concat(all_predictions, ignore_index=True)
    _write_frame_atomic(target_pred_path, combined)
    prediction_sha = sha256_file(target_pred_path)
    readiness = {
        db: {
            "retrospective_landmark_prediction_allowed": True,
            "ready_for_future24h_clinical_claim": False,
            "clinical_input_availability_verified": False,
            "eligibility_manifest_sha256": target_gates[db]["eligibility_manifest_sha256"],
            "eligibility_table_sha256": target_gates[db]["eligibility_table_sha256"],
            "sequence_npz_sha256": target_gates[db]["subject_identity_attestation"][
                "cohort_record"]["sequence_npz_sha256"],
            "subject_identity_attestation_sha256": identity_sidecar["sha256"],
            "prediction_scope": target_gates[db]["prediction_scope"],
            "input_scope": target_gates[db]["input_scope"],
            "endpoint": endpoint_by_db[db],
        }
        for db in cohorts
    }
    prediction_records: list[dict[str, Any]] = []
    for analysis_id in ANALYSIS_IDS:
        choice = bundle["models"][analysis_id]
        for db in cohorts:
            frame = combined.loc[(combined["analysis_id"] == analysis_id)
                                 & (combined["db"] == db)]
            fm_record = records[db]
            prediction_records.append({
                "analysis_id": analysis_id, "db": db,
                "model_tag": choice["model_tag"],
                "prediction_file": str(target_pred_path),
                "prediction_file_sha256": prediction_sha,
                "ordered_row_digest_sha256": ordered_row_digest(frame),
                "n_prediction_rows": int(len(frame)),
                "role": "evaluation", "prediction_scope": fm_record["prediction_scope"],
                "input_scope": fm_record["input_scope"],
                "clinical_input_availability_verified": False,
                "retrospective_landmark_prediction_allowed": True,
                "input_condition": "real",
                "input_mask_sha256": fm_record["input_mask_sha256"],
                "input_mask_spec": fm_record.get("input_mask_spec"),
                "fm_analysis_id": "main20k_real",
                "eligibility_manifest_sha256": fm_record["eligibility_manifest_sha256"],
                "eligibility_table_sha256": fm_record["eligibility_table_sha256"],
                "source_npz_sha256": fm_record["source_npz_sha256"],
                "transport_source_model_sha256": model_sha,
                "frozen_selected_parameters": choice["selected_parameters"],
            })
    baseline_doc = {
        "status": "complete", "endpoint": "per-cohort; see cohort_readiness.endpoint",
        "analysis_stage": "frozen_source_model_transport_only",
        "seed": int(bundle["seed"]), "cohorts": list(cohorts),
        "input_conditions": ["real"],
        "cohort_readiness": readiness,
        "predictions": prediction_records,
        "source_reproduction_manifest": str(out_dir / "source_reproduction_manifest.json"),
        "source_reproduction_manifest_sha256": sha256_file(out_dir / "source_reproduction_manifest.json"),
        "source_model_bundle_sha256": model_sha,
        "target_input_prediction_manifest": str(RECOVERED_MANIFEST),
        "target_input_prediction_manifest_sha256": prediction_manifest_sha,
        "subject_identity_attestation_path": identity_sidecar["path"],
        "subject_identity_attestation_sha256": identity_sidecar["sha256"],
        "within_source_patient_clustering_only": True,
        "cross_source_patient_independence_verified": False,
        "cross_scope_patient_independence_verified": False,
        "no_target_model_fit": True,
        "clinical_input_availability_verified": False,
        "clinical_promotion_allowed": False,
        "patient_level_exported_to_local": False,
    }
    # Validate the generated artifact through the same strict baseline/FΜ adapter before returning.
    manifest_path = out_dir / f"results/baseline_prediction_manifest{cohort_suffix}.json"
    _write_json_atomic(manifest_path, baseline_doc)
    manifest_path_sha = sha256_file(manifest_path)
    scored_cohorts = set(plan.get("scored_target_cohorts", [])) | set(cohorts)
    plan["scored_target_cohorts"] = [db for db in TARGET_COHORTS if db in scored_cohorts]
    plan["status"] = ("recovered_target_scoring_complete"
                       if scored_cohorts == set(TARGET_COHORTS)
                       else "recovered_target_scoring_partial")
    plan["baseline_prediction_manifest_sha256"] = manifest_path_sha
    plan["baseline_prediction_file_sha256"] = prediction_sha
    existing_checkpoints = plan.get("per_cohort_model_checkpoints", {})
    if not isinstance(existing_checkpoints, dict):
        existing_checkpoints = {}
    existing_checkpoints.update(checkpoint_records)
    plan["per_cohort_model_checkpoints"] = existing_checkpoints
    _write_json_atomic(out_dir / "plan.json", plan)
    print(json.dumps({
        "status": plan["status"],
        "output_prediction_file": str(target_pred_path),
        "baseline_prediction_manifest": str(manifest_path),
        "rows_by_cohort": {
            db: {analysis_id: int(len(combined.loc[
                (combined["db"] == db) & (combined["analysis_id"] == analysis_id)
            ])) for analysis_id in ANALYSIS_IDS}
            for db in cohorts
        },
    }, sort_keys=True), flush=True)
    return baseline_doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("source-verify", "score-recovered"), required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cohorts", nargs="+", choices=TARGET_COHORTS,
                        help="subset of recovered target cohorts to score; default is all")
    args = parser.parse_args(argv)
    if args.phase == "source-verify":
        fit_verify_and_freeze(out_dir=args.out_dir)
    elif args.phase == "score-recovered":
        score_recovered_targets(out_dir=args.out_dir,
                                cohorts=tuple(args.cohorts) if args.cohorts else TARGET_COHORTS)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__,
                          "error": str(exc)}, sort_keys=True), flush=True)
        raise
