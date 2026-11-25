"""Paired metrics and shared-patient calibration for frozen recovered transports."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
BASELINE_TRANSPORT = REV / "baselines/transport_v4_2"
ROBUSTNESS_CODE = REV / "robustness"
RECOVERED_RUN = REV / "engine/results/prediction_runs/recovered_external_primary_v4_2_resume3-04b21bfc05943b7f"
RECOVERED_MANIFEST = RECOVERED_RUN / "prediction_manifest.json"
RECOVERED_GATE = REV / "temporal/results/eligibility/recovered_reviewed_v4_2"
IDENTITY_ATTESTATION_PATH = REV / "admin/subject_identity_attestations_v4_2.json"
ORIGINAL_FM_RUN = REV / "engine/results/prediction_runs/status_consistent_primary_core_fix2-7da980ffcbd8f00b"
ORIGINAL_FM_MANIFEST = ORIGINAL_FM_RUN / "prediction_manifest.json"
ORIGINAL_ELIGIBILITY_GATE = REV / "temporal/results/eligibility/status_consistent_v4_1/landmark_v4_1"
OLD_EICU_CALIBRATION_MANIFEST = (
    REV / "robustness/results/primary_status_consistent_shared_calibration_20261123"
    / "main20k_real/recalibration_run_manifest.json"
)
OLD_EICU_CALIBRATION_DRAWS = OLD_EICU_CALIBRATION_MANIFEST.parent / "recalibration_draws.csv"
SOURCE_REPRODUCTION_MANIFEST = BASELINE_TRANSPORT / "source_reproduction_manifest.json"
BASELINE_PREDICTION_MANIFEST = BASELINE_TRANSPORT / "results/baseline_prediction_manifest.json"
DEFAULT_OUT = REV / "robustness/results/recovered_transport_v4_2_20261123_reuse"
SEED = 20261122
BUDGETS = (50, 200, 500, 2000)
REPEATS = 200
N_BOOT = 1000
BOOTSTRAP_BATCH = 16

for _path in (BASELINE_TRANSPORT, ROBUSTNESS_CODE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import robustness_common as _robustness_common  # noqa: E402
from frozen_source_transport import (  # noqa: E402
    ANALYSIS_IDS, IDENTITY_ATTESTATION_SHA256, TARGET_COHORTS,
    _exact_target_identity_match, _json, _load_target_fm_frames,
    _validate_subject_identity_attestations,
)
from hospital_heterogeneity import metric_row  # noqa: E402
from paired_baselines import sha256_file  # noqa: E402
from recalibration import (  # noqa: E402
    _identity_digest, _make_sampling_plan, adaptation_assessment_pools, run_db,
    verify_shared_population_frames,
)
from robustness_common import load_checked  # noqa: E402

MODEL_IDS = ("main20k_real", *ANALYSIS_IDS)
MODEL_TAGS = {
    "main20k_real": "main20k",
    "conventional_lr_pattern_only": "conventional_lr",
    "conventional_hgb_pattern_only": "conventional_hgb",
    "conventional_lr_value_plus_pattern": "conventional_lr",
    "conventional_hgb_value_plus_pattern": "conventional_hgb",
}
PLAN_COMPARE_FIELDS = (
    "fixed_population_identity_sha256", "adaptation_index_patient_stay_sha256",
    "adaptation_patient_set_sha256", "assessment_population_identity_sha256",
    "assessment_patient_set_sha256", "sampling_plan_sha256",
)


def _stable_sha256(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _write_rows_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


@contextmanager
def _sidecar_identity_adapter(records: dict[str, dict[str, Any]]) -> Iterator[None]:
    """Use a validated local sidecar only when the unchanged generic loader lacks identity."""
    original = _robustness_common.manifest_subject_identity

    def adapted(path: str | Path, db: str) -> tuple[str, bool, str]:
        basis, verified, label = original(path, db)
        if verified or str(db) not in records:
            return basis, verified, label
        supplement = records[str(db)]["subject_identity_basis"]
        return str(supplement["basis"]), True, "patient"

    _robustness_common.manifest_subject_identity = adapted
    try:
        yield
    finally:
        _robustness_common.manifest_subject_identity = original


def _load_all_model_frames() -> tuple[
    dict[str, dict[str, pd.DataFrame]], dict[str, dict[str, Any]],
    dict[str, str], dict[str, Any],
]:
    identity_records, identity_sidecar_sha = _validate_subject_identity_attestations(
        sidecar_path=IDENTITY_ATTESTATION_PATH,
        eligibility_dir=RECOVERED_GATE,
        cohorts=TARGET_COHORTS,
        expected_sidecar_sha256=IDENTITY_ATTESTATION_SHA256,
    )
    fm_frames, fm_gates, endpoint_by_db = _load_target_fm_frames()
    baseline_manifest = _json(BASELINE_PREDICTION_MANIFEST)
    if baseline_manifest.get("status") != "complete":
        raise ValueError("recovered baseline target prediction manifest is not complete")
    if set(baseline_manifest.get("cohorts", [])) != set(TARGET_COHORTS):
        raise ValueError("recovered baseline manifest does not cover the three registered target cohorts")
    records = baseline_manifest.get("predictions", [])
    if not isinstance(records, list):
        raise ValueError("recovered baseline manifest prediction records are absent")
    files = {str(row.get("prediction_file")) for row in records if isinstance(row, dict)}
    file_hashes = {str(row.get("prediction_file_sha256")) for row in records if isinstance(row, dict)}
    if len(files) != 1 or len(file_hashes) != 1:
        raise ValueError("recovered baseline predictions must use one immutable combined file")
    prediction_path = Path(next(iter(files)))
    if not prediction_path.is_absolute():
        prediction_path = (REV / prediction_path).resolve()
    prediction_path_sha = sha256_file(prediction_path)
    if prediction_path_sha != next(iter(file_hashes)):
        raise ValueError("recovered baseline prediction file hash differs from its manifest")

    frames: dict[str, dict[str, pd.DataFrame]] = {"main20k_real": fm_frames}
    gates = dict(fm_gates)
    with _sidecar_identity_adapter(identity_records):
        for analysis_id in ANALYSIS_IDS:
            analysis_records = [row for row in records
                                if row.get("analysis_id") == analysis_id]
            if len(analysis_records) != len(TARGET_COHORTS):
                raise ValueError(f"{analysis_id}: recovered baseline manifest coverage is incomplete")
            arm: dict[str, pd.DataFrame] = {}
            for endpoint in sorted(set(endpoint_by_db.values())):
                part, part_gates = load_checked(
                    prediction_path, RECOVERED_GATE,
                    model_tag=MODEL_TAGS[analysis_id], analysis_id=analysis_id,
                    input_condition="real", endpoint=endpoint,
                    prediction_manifest_path=BASELINE_PREDICTION_MANIFEST,
                )
                for db in TARGET_COHORTS:
                    if endpoint_by_db[db] == endpoint:
                        if db not in part or db not in part_gates:
                            raise ValueError(f"{analysis_id}: strict adapter omitted {db}")
                        arm[db] = part[db]
            if set(arm) != set(TARGET_COHORTS):
                raise ValueError(f"{analysis_id}: strict target adapter returned the wrong cohort set")
            frames[analysis_id] = arm

    for db in TARGET_COHORTS:
        if set(frames) != set(MODEL_IDS):
            raise ValueError("transport analysis requires FM plus all four frozen conventional arms")
        for analysis_id in ANALYSIS_IDS:
            _exact_target_identity_match(frames["main20k_real"][db], frames[analysis_id][db], db=db)
        signature = verify_shared_population_frames({
            analysis_id: frames[analysis_id][db] for analysis_id in MODEL_IDS
        })
        fm_gate = gates[db]
        if (fm_gate.get("bootstrap_cluster_unit") != "patient"
                or fm_gate.get("independent_patient_clustering_verified") is not True):
            raise ValueError(f"{db}: patient-cluster uncertainty is not explicitly verified")
        fm_gate["shared_population_identity_sha256"] = signature

    provenance = {
        "fm_prediction_manifest_path": str(RECOVERED_MANIFEST),
        "fm_prediction_manifest_sha256": sha256_file(RECOVERED_MANIFEST),
        "baseline_prediction_manifest_path": str(BASELINE_PREDICTION_MANIFEST),
        "baseline_prediction_manifest_sha256": sha256_file(BASELINE_PREDICTION_MANIFEST),
        "baseline_prediction_file_path": str(prediction_path),
        "baseline_prediction_file_sha256": prediction_path_sha,
        "source_reproduction_manifest_path": str(SOURCE_REPRODUCTION_MANIFEST),
        "source_reproduction_manifest_sha256": sha256_file(SOURCE_REPRODUCTION_MANIFEST),
        "source_model_bundle_sha256": _json(SOURCE_REPRODUCTION_MANIFEST)["source_model_bundle_sha256"],
        "eligibility_manifest_sha256": sha256_file(RECOVERED_GATE / "eligibility_manifest.json"),
        "subject_identity_attestation_path": str(IDENTITY_ATTESTATION_PATH),
        "subject_identity_attestation_sha256": identity_sidecar_sha,
        "identity_scope": "within-source patient clustering only; cross-source and cross-scope independence unverified",
        "clinical_input_availability_verified": False,
        "clinical_promotion_allowed": False,
        "patient_level_exported_to_local": False,
    }
    return frames, gates, endpoint_by_db, provenance


def _sampling_plan_for_frame(frame: pd.DataFrame, *, db: str) -> tuple[dict[str, Any], dict[str, int]]:
    usable = frame.loc[frame["eligible_24h"].astype(bool)
                       & frame["temporal_verified"].astype(bool)].copy()
    if usable.empty or not usable["y"].isin([0, 1]).all():
        raise ValueError(f"{db}: shared calibration plan requires eligible binary outcomes")
    adaptation, assessment, stratified = adaptation_assessment_pools(usable, SEED)
    if adaptation.empty or assessment.empty:
        raise ValueError(f"{db}: insufficient patients for disjoint adaptation and assessment pools")
    sampling, _ = _make_sampling_plan(
        usable, adaptation, assessment, repeats=REPEATS, seed=SEED, budgets=BUDGETS,
    )
    counts = {
        "n_eligible_stays": int(len(usable)),
        "n_patients": int(usable["subject"].nunique()),
        "n_adaptation_patients": int(adaptation["subject"].nunique()),
        "n_assessment_patients": int(assessment["subject"].nunique()),
        "n_assessment_stays": int(len(assessment)),
        "n_assessment_events": int(assessment["y"].sum()),
        "n_assessment_nonevents": int(len(assessment) - assessment["y"].sum()),
        "patient_overlap": int(len(set(adaptation["subject"]) & set(assessment["subject"]))),
        "one_outcome_blind_index_stay_per_adaptation_patient": True,
        "pool_split_stratified_on_outcomes": False,
        "index_stay_selection_uses_outcomes": False,
        "stratified_patient_index_rule_used": bool(stratified),
        "supported_budgets": [int(budget) for budget in BUDGETS if budget <= len(adaptation)],
        "unsupported_budgets": [int(budget) for budget in BUDGETS if budget > len(adaptation)],
    }
    return {**sampling, **counts}, counts


def _load_original_eicu_frame() -> pd.DataFrame:
    original = _json(ORIGINAL_FM_MANIFEST)
    ready_eicu = [row for row in original.get("readiness", [])
                  if row.get("db") == "eicu"]
    if (original.get("run_id") != "status_consistent_primary_core_fix2"
            or len(ready_eicu) != 1
            or ready_eicu[0].get("status") != "ready"
            or ready_eicu[0].get("retrospective_landmark_prediction_ready") is not True):
        raise ValueError("original eICU FM prediction run is not complete")
    records = [row for row in original.get("predictions", [])
               if row.get("analysis_id") == "main20k_real" and row.get("db") == "eicu"]
    if len(records) != 1:
        raise ValueError("original FM run does not have exactly one registered eICU real prediction")
    record = records[0]
    if (record.get("mode") != "predict"
            or record.get("input_condition") != "real"
            or record.get("model_tag") != "main20k"
            or record.get("retrospective_landmark_prediction_allowed") is not True
            or record.get("retrospective_landmark_prediction_ready") is not True):
        raise ValueError("original FM eICU prediction record is not a ready main20k real prediction")
    prediction_path = Path(record["prediction_file"])
    if not prediction_path.is_absolute():
        prediction_path = (REV / prediction_path).resolve()
    frames, _ = load_checked(
        prediction_path, ORIGINAL_ELIGIBILITY_GATE,
        model_tag="main20k", analysis_id="main20k_real", input_condition="real",
        endpoint="hospital", prediction_manifest_path=ORIGINAL_FM_MANIFEST,
    )
    if "eicu" not in frames:
        raise ValueError("strict original FM adapter returned no eICU frame")
    return frames["eicu"]


def _compare_eicu_row_identity(original: pd.DataFrame, repaired: pd.DataFrame) -> dict[str, Any]:
    fields = ["stay_id", "subject", "rows", "split", "y"]
    left = original[fields].copy()
    right = repaired[fields].copy()
    for frame in (left, right):
        frame["stay_id"] = frame["stay_id"].astype("string").str.strip()
        if frame[["stay_id", "subject", "rows", "split", "y"]].isna().any().any():
            raise ValueError("original/repaired eICU identity comparison has missing IDs or labels")
        if frame["stay_id"].duplicated().any():
            raise ValueError("original/repaired eICU identity comparison has duplicate stays")
    merged = left.merge(right, on="stay_id", how="outer", validate="one_to_one",
                        suffixes=("_original", "_repaired"), indicator=True)
    mismatch_by_field = {
        field: int((merged.loc[merged["_merge"] == "both", f"{field}_original"].astype(str)
                    != merged.loc[merged["_merge"] == "both", f"{field}_repaired"].astype(str)).sum())
        for field in ("subject", "rows", "split", "y")
    }
    original_patients = set(original["subject"].astype(str))
    repaired_patients = set(repaired["subject"].astype(str))
    original_digest = _identity_digest(original, ("stay_id", "subject", "rows", "split", "y"))
    repaired_digest = _identity_digest(repaired, ("stay_id", "subject", "rows", "split", "y"))
    exact = bool(
        len(original) == len(repaired)
        and merged["_merge"].eq("both").all()
        and all(value == 0 for value in mismatch_by_field.values())
        and original_digest == repaired_digest
    )
    return {
        "exact_stay_subject_row_split_label_identity": exact,
        "n_original_stays": int(len(original)),
        "n_repaired_stays": int(len(repaired)),
        "original_only_stays": int(merged["_merge"].eq("left_only").sum()),
        "repaired_only_stays": int(merged["_merge"].eq("right_only").sum()),
        "n_original_patients": int(len(original_patients)),
        "n_repaired_patients": int(len(repaired_patients)),
        "patient_set_sha256_equal": hashlib.sha256(
            "\n".join(sorted(original_patients)).encode("utf-8")).hexdigest()
        == hashlib.sha256("\n".join(sorted(repaired_patients)).encode("utf-8")).hexdigest(),
        "row_identity_sha256_original": original_digest,
        "row_identity_sha256_repaired": repaired_digest,
        "identity_mismatches_by_field": mismatch_by_field,
    }


def _compare_eicu_draw_hashes(plan: dict[str, Any]) -> dict[str, Any]:
    if not OLD_EICU_CALIBRATION_DRAWS.is_file():
        return {"exact_draw_and_index_stay_hash_match": False,
                "status": "original_calibration_draw_table_unavailable",
                "n_draws_compared": 0}
    old = pd.read_csv(OLD_EICU_CALIBRATION_DRAWS)
    required = {"method", "budget", "repeat", "adaptation_sample_index_sha256",
                "adaptation_sample_patient_index_stay_sha256"}
    if not required.issubset(old.columns):
        return {"exact_draw_and_index_stay_hash_match": False,
                "status": "original_calibration_draw_table_lacks_hash_fields",
                "n_draws_compared": 0}
    old = old.loc[old["method"].astype(str).eq("raw")
                  & pd.to_numeric(old["budget"], errors="coerce").isin(BUDGETS)].copy()
    old_records: dict[tuple[int, int], tuple[str, str]] = {}
    for row in old.itertuples(index=False):
        key = (int(row.budget), int(row.repeat))
        if key in old_records:
            return {"exact_draw_and_index_stay_hash_match": False,
                    "status": "duplicate_original_draw_hash_rows", "n_draws_compared": 0}
        old_records[key] = (str(row.adaptation_sample_index_sha256),
                            str(row.adaptation_sample_patient_index_stay_sha256))
    new_records = {
        (int(row["budget"]), int(row["repeat"])): (
            str(row["draw_index_sha256"]), str(row["patient_index_stay_sha256"]))
        for row in plan.get("draws", [])
    }
    same_keys = set(old_records) == set(new_records)
    same_hashes = same_keys and all(old_records[key] == new_records[key] for key in old_records)
    return {
        "exact_draw_and_index_stay_hash_match": bool(same_hashes),
        "status": "exact_draw_and_index_stay_hash_match" if same_hashes else "draw_or_index_stay_hash_differs",
        "n_original_draws": int(len(old_records)),
        "n_repaired_draws": int(len(new_records)),
        "n_draws_compared": int(len(set(old_records) & set(new_records))),
        "draw_key_sets_equal": bool(same_keys),
    }


def _original_eicu_plan_match(frame: pd.DataFrame,
                              plan: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    result = {
        "reused_original_plan": False,
        "cross_version_curve_comparison_allowed": False,
        "original_plan_manifest_path": str(OLD_EICU_CALIBRATION_MANIFEST),
        "original_plan_manifest_sha256": None,
        "comparison_mismatched_fields": [],
        "original_fm_manifest_path": str(ORIGINAL_FM_MANIFEST),
        "original_fm_manifest_sha256": None,
        "original_eligibility_manifest_sha256": None,
    }
    if not OLD_EICU_CALIBRATION_MANIFEST.is_file():
        result["comparison_status"] = "original_plan_manifest_unavailable"
        return result, None
    original = _json(OLD_EICU_CALIBRATION_MANIFEST)
    result["original_plan_manifest_sha256"] = sha256_file(OLD_EICU_CALIBRATION_MANIFEST)
    if (original.get("status") != "complete"
            or int(original.get("seed", -1)) != SEED
            or int(original.get("repeats", -1)) != REPEATS
            or tuple(original.get("budgets", ())) != BUDGETS):
        result["comparison_status"] = "original_plan_configuration_differs"
        return result, None
    expected = original.get("cohort_sampling_plans", {}).get("eicu")
    if not isinstance(expected, dict):
        result["comparison_status"] = "original_eicu_plan_missing"
        return result, None
    if not ORIGINAL_FM_MANIFEST.is_file():
        result["comparison_status"] = "original_fm_manifest_unavailable"
        return result, None
    result["original_fm_manifest_sha256"] = sha256_file(ORIGINAL_FM_MANIFEST)
    result["original_eligibility_manifest_sha256"] = sha256_file(
        ORIGINAL_ELIGIBILITY_GATE / "eligibility_manifest.json")
    original_frame = _load_original_eicu_frame()
    identity_check = _compare_eicu_row_identity(original_frame, frame)
    result["original_repaired_row_identity"] = identity_check
    canonical_frame = frame.copy()
    canonical_frame["db"] = "eicu"
    canonical_plan, _ = _sampling_plan_for_frame(canonical_frame, db="eicu")
    mismatch = [field for field in PLAN_COMPARE_FIELDS
                if expected.get(field) != canonical_plan.get(field)]
    result["comparison_mismatched_fields"] = mismatch
    draw_check = _compare_eicu_draw_hashes(canonical_plan)
    result["original_repaired_actual_draw_check"] = draw_check
    if (not identity_check["exact_stay_subject_row_split_label_identity"]
            or not draw_check["exact_draw_and_index_stay_hash_match"] or mismatch):
        result["comparison_status"] = "repaired_eicu_plan_differs_new_plan_frozen"
        return result, None
    result.update({
        "reused_original_plan": True,
        "cross_version_curve_comparison_allowed": True,
        "comparison_status": "exact_original_eicu_plan_reused",
        "original_eicu_sampling_plan_sha256": expected["sampling_plan_sha256"],
        "plan_db_alias": "eicu",
    })
    return result, canonical_plan


def freeze_sampling_plans(*, out_dir: Path) -> dict[str, Any]:
    frames, gates, endpoints, provenance = _load_all_model_frames()
    plans: dict[str, dict[str, Any]] = {}
    plan_dir = out_dir / "sampling_plans"
    for db in TARGET_COHORTS:
        sampling, counts = _sampling_plan_for_frame(frames["main20k_real"][db], db=db)
        original_comparison: dict[str, Any] | None = None
        row = {
            "endpoint": endpoints[db],
            "cluster_unit": gates[db]["bootstrap_cluster_unit"],
            "population_identity_sha256": gates[db]["shared_population_identity_sha256"],
            "subject_identity_attestation_sha256": provenance["subject_identity_attestation_sha256"],
            "cohort_counts": counts,
            **sampling,
        }
        if db == "eicu_semantic_v4_2":
            original_comparison, canonical_plan = _original_eicu_plan_match(
                frames["main20k_real"][db], sampling,
            )
            if canonical_plan is not None:
                sampling = canonical_plan
                row["identity_db_alias_for_sampling_plan"] = "eicu"
            row["original_eicu_comparison"] = original_comparison
            row.update(sampling)
        else:
            row.update(sampling)
        plans[db] = row
        _write_json_atomic(plan_dir / f"{db}.json", {
            "status": "complete", "db": db, "seed": SEED, "budgets": list(BUDGETS),
            "repeats": REPEATS, "cohort_sampling_plan": row,
        })
        print(json.dumps({
            "stage": "transport_sampling_plan_frozen", "db": db,
            "sampling_plan_sha256": sampling["sampling_plan_sha256"],
            "n_adaptation_patients": counts["n_adaptation_patients"],
            "n_assessment_patients": counts["n_assessment_patients"],
            "supported_budgets": counts["supported_budgets"],
            "original_eicu_plan": row.get("original_eicu_comparison", {}).get("comparison_status"),
        }, sort_keys=True), flush=True)
    document = {
        "status": "complete",
        "schema": "ehr_fm_recovered_transport_shared_calibration_plan_v4_2",
        "seed": SEED, "budgets": list(BUDGETS), "repeats": REPEATS,
        "model_ids": list(MODEL_IDS), "cohorts": list(TARGET_COHORTS),
        "sampling": "one outcome-blind adaptation stay per patient; patient-disjoint pools; natural without-replacement patient draws; no outcome balancing",
        "plans_frozen_before_metric_evaluation": True,
        "cohort_sampling_plans": plans,
        "provenance": provenance,
        "cohort_gate_provenance": {
            db: {
                "eligibility_manifest_sha256": gates[db]["eligibility_manifest_sha256"],
                "eligibility_table_sha256": gates[db]["eligibility_table_sha256"],
                "sequence_npz_sha256": gates[db]["subject_identity_attestation"][
                    "cohort_record"]["sequence_npz_sha256"],
                "subject_identity_attestation_sha256": provenance["subject_identity_attestation_sha256"],
                "within_source_patient_clustering_verified": True,
                "cross_source_patient_independence_verified": False,
                "cross_scope_patient_independence_verified": False,
            }
            for db in TARGET_COHORTS
        },
    }
    document["plan_manifest_sha256"] = _stable_sha256(document)
    plan_path = out_dir / "sampling_plan_manifest.json"
    if plan_path.exists():
        prior = _json(plan_path)
        if prior.get("plan_manifest_sha256") != document["plan_manifest_sha256"]:
            raise FileExistsError(f"refusing to replace a different frozen sampling plan: {plan_path}")
    else:
        _write_json_atomic(plan_path, document)
    print(json.dumps({
        "stage": "transport_sampling_plan_manifest_complete",
        "path": str(plan_path), "sha256": sha256_file(plan_path),
        "plan_manifest_sha256": document["plan_manifest_sha256"],
    }, sort_keys=True), flush=True)
    return document


def _read_and_validate_sampling_plan(path: Path, provenance: dict[str, Any],
                                     frames: dict[str, dict[str, pd.DataFrame]]) -> dict[str, Any]:
    document = _json(path)
    stored_sha = document.get("plan_manifest_sha256")
    unhashed = dict(document)
    unhashed.pop("plan_manifest_sha256", None)
    if document.get("status") != "complete" or _stable_sha256(unhashed) != stored_sha:
        raise ValueError("transport sampling plan manifest is incomplete or its hash is invalid")
    if document.get("provenance") != provenance:
        raise ValueError("transport sampling plan provenance differs from current prediction/gate artifacts")
    if (int(document.get("seed", -1)) != SEED
            or tuple(document.get("budgets", ())) != BUDGETS
            or int(document.get("repeats", -1)) != REPEATS
            or document.get("plans_frozen_before_metric_evaluation") is not True):
        raise ValueError("transport sampling plan configuration differs from the registered calibration design")
    for db in TARGET_COHORTS:
        frozen = document.get("cohort_sampling_plans", {}).get(db, {})
        if (db == "eicu_semantic_v4_2"
                and frozen.get("original_eicu_comparison", {}).get("reused_original_plan") is True):
            comparison, canonical = _original_eicu_plan_match(
                frames["main20k_real"][db], {},
            )
            if canonical is None or comparison.get("reused_original_plan") is not True:
                raise ValueError("repaired eICU no longer passes the original plan identity/draw attestation")
            current = canonical
        else:
            current, _ = _sampling_plan_for_frame(frames["main20k_real"][db], db=db)
        if any(current.get(key) != frozen.get(key) for key in PLAN_COMPARE_FIELDS):
            raise ValueError(f"{db}: current patient/stay sampling plan differs from the frozen plan")
    return document


def _loss_arrays(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probability = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    labels = np.asarray(y, dtype=float)
    return (labels - probability) ** 2, -(labels * np.log(probability)
                                            + (1 - labels) * np.log(1 - probability))


def _paired_target_predictions(left: pd.DataFrame, right: pd.DataFrame,
                               *, db: str) -> pd.DataFrame:
    _exact_target_identity_match(left, right, db=db)
    keys = ["db", "stay_id"]
    identity = ["subject", "rows", "split", "y"]
    l = left[[*keys, *identity, "p"]].copy()
    r = right[[*keys, *identity, "p"]].copy()
    for frame in (l, r):
        frame["db"] = frame["db"].astype("string").str.strip()
        frame["stay_id"] = frame["stay_id"].astype("string").str.strip()
    paired = l.merge(r, on=keys, how="inner", suffixes=("_fm", "_baseline"),
                      validate="one_to_one")
    for field in identity:
        if not paired[f"{field}_fm"].astype(str).equals(paired[f"{field}_baseline"].astype(str)):
            raise ValueError(f"{db}: paired FM/baseline identity mismatch in {field}")
    return pd.DataFrame({
        "db": paired["db"], "stay_id": paired["stay_id"],
        "subject": paired["subject_fm"].astype(str),
        "rows": paired["rows_fm"], "split": paired["split_fm"], "y": paired["y_fm"],
        "p_fm": paired["p_fm"].astype(float),
        "p_baseline": paired["p_baseline"].astype(float),
    })


def _cluster_bootstrap_loss_delta(pairs: dict[str, pd.DataFrame], *, seed: int,
                                  n_boot: int = N_BOOT) -> dict[str, dict[str, Any]]:
    if n_boot < 1000:
        raise ValueError("paired patient-cluster intervals require at least 1000 draws")
    names = list(pairs)
    first = pairs[names[0]]
    subjects = first["subject"].astype(str).to_numpy()
    unique_subjects, inverse = np.unique(subjects, return_inverse=True)
    if len(unique_subjects) < 2:
        raise ValueError("paired patient-cluster interval requires at least two patients")
    patient_counts = np.bincount(inverse).astype(float)
    deltas: dict[tuple[str, str], np.ndarray] = {}
    for name, frame in pairs.items():
        if not frame["subject"].astype(str).equals(first["subject"].astype(str)):
            raise ValueError("shared paired bootstrap requires identical stay ordering across comparators")
        brier_fm, logloss_fm = _loss_arrays(frame["y"].to_numpy(), frame["p_fm"].to_numpy())
        brier_base, logloss_base = _loss_arrays(frame["y"].to_numpy(), frame["p_baseline"].to_numpy())
        deltas[(name, "brier")] = brier_fm - brier_base
        deltas[(name, "logloss")] = logloss_fm - logloss_base
    keys = list(deltas)
    patient_sums = np.vstack([
        np.bincount(inverse, weights=deltas[key], minlength=len(unique_subjects))
        for key in keys
    ])
    estimates = np.empty((n_boot, len(keys)), dtype=float)
    rng = np.random.default_rng(seed)
    for start in range(0, n_boot, BOOTSTRAP_BATCH):
        size = min(BOOTSTRAP_BATCH, n_boot - start)
        sampled = rng.integers(0, len(unique_subjects), size=(size, len(unique_subjects)),
                               dtype=np.int32)
        weights = np.stack([
            np.bincount(row, minlength=len(unique_subjects)) for row in sampled
        ]).astype(float, copy=False)
        denominator = weights @ patient_counts
        estimates[start:start + size] = (weights @ patient_sums.T) / denominator[:, None]
    result: dict[str, dict[str, Any]] = {}
    for index, (name, metric) in enumerate(keys):
        lo, hi = np.percentile(estimates[:, index], [2.5, 97.5])
        result.setdefault(name, {})[metric] = {
            "ci_95_lo": float(lo), "ci_95_hi": float(hi),
            "bootstrap_draws": int(n_boot), "valid_draws": int(np.isfinite(estimates[:, index]).sum()),
        }
    return result


def _performance_for_cohort(db: str, models: dict[str, dict[str, pd.DataFrame]],
                            gates: dict[str, dict[str, Any]], provenance: dict[str, Any],
                            out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cohort_dir = out_dir / "performance" / db
    metrics_path = cohort_dir / "performance_metrics.csv"
    paired_path = cohort_dir / "paired_deltas.csv"
    manifest_path = cohort_dir / "manifest.json"
    expected_inputs = {
        "fm_prediction_manifest_sha256": provenance["fm_prediction_manifest_sha256"],
        "baseline_prediction_manifest_sha256": provenance["baseline_prediction_manifest_sha256"],
        "identity_attestation_sha256": provenance["subject_identity_attestation_sha256"],
        "eligibility_manifest_sha256": gates[db]["eligibility_manifest_sha256"],
        "eligibility_table_sha256": gates[db]["eligibility_table_sha256"],
        "shared_population_identity_sha256": gates[db]["shared_population_identity_sha256"],
    }
    if manifest_path.is_file():
        existing = _json(manifest_path)
        if existing.get("status") != "complete" or existing.get("inputs") != expected_inputs:
            raise ValueError(f"{db}: saved paired metrics are stale or incomplete")
        if sha256_file(metrics_path) != existing.get("performance_metrics_sha256"):
            raise ValueError(f"{db}: saved performance table hash mismatch")
        if sha256_file(paired_path) != existing.get("paired_deltas_sha256"):
            raise ValueError(f"{db}: saved paired delta table hash mismatch")
        return pd.read_csv(metrics_path).to_dict("records"), pd.read_csv(paired_path).to_dict("records")

    metric_rows: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        row = metric_row(models[model_id][db], model_id=model_id, scope="all_eligible_stays")
        row["db"] = db
        metric_rows.append(row)
    metric_by_model = {str(row["model_id"]): row for row in metric_rows}
    paired_frames = {
        baseline_id: _paired_target_predictions(
            models["main20k_real"][db], models[baseline_id][db], db=db,
        )
        for baseline_id in ANALYSIS_IDS
    }
    boot_seed = SEED + (TARGET_COHORTS.index(db) + 1) * 1000
    boot = _cluster_bootstrap_loss_delta(paired_frames, seed=boot_seed, n_boot=N_BOOT)
    paired_rows: list[dict[str, Any]] = []
    for baseline_id in ANALYSIS_IDS:
        fm_metric = metric_by_model["main20k_real"]
        base_metric = metric_by_model[baseline_id]
        pair = paired_frames[baseline_id]
        brier_fm, logloss_fm = _loss_arrays(pair["y"].to_numpy(), pair["p_fm"].to_numpy())
        brier_base, logloss_base = _loss_arrays(pair["y"].to_numpy(), pair["p_baseline"].to_numpy())
        paired_rows.append({
            "db": db,
            "comparison_id": f"main20k_real_minus_{baseline_id}",
            "fm_model_id": "main20k_real", "baseline_model_id": baseline_id,
            "n_stays": int(len(pair)), "n_patients": int(pair["subject"].nunique()),
            "events": int(pair["y"].sum()), "nonevents": int(len(pair) - pair["y"].sum()),
            "delta_auroc": float(fm_metric["auroc"] - base_metric["auroc"]),
            "auroc_pair_status": "ok" if np.isfinite(fm_metric["auroc"] + base_metric["auroc"])
            else "unestimable_single_class",
            "delta_average_precision": float(
                fm_metric["average_precision"] - base_metric["average_precision"]),
            "average_precision_pair_status": (
                "ok" if np.isfinite(fm_metric["average_precision"] + base_metric["average_precision"])
                else "unestimable_no_events"),
            "delta_brier": float(np.mean(brier_fm - brier_base)),
            "delta_brier_ci_95_lo": boot[baseline_id]["brier"]["ci_95_lo"],
            "delta_brier_ci_95_hi": boot[baseline_id]["brier"]["ci_95_hi"],
            "delta_logloss": float(np.mean(logloss_fm - logloss_base)),
            "delta_logloss_ci_95_lo": boot[baseline_id]["logloss"]["ci_95_lo"],
            "delta_logloss_ci_95_hi": boot[baseline_id]["logloss"]["ci_95_hi"],
            "delta_calibration_intercept": float(
                fm_metric["calibration_intercept"] - base_metric["calibration_intercept"]),
            "delta_calibration_slope": float(
                fm_metric["calibration_slope"] - base_metric["calibration_slope"]),
            "bootstrap_cluster_unit": "patient",
            "bootstrap_draws": N_BOOT,
            "bootstrap_valid_draws": boot[baseline_id]["brier"]["valid_draws"],
            "ci_scope": "paired full-cohort stay-weighted losses; shared patient-cluster draws",
            "auroc_average_precision_ci": "not_bootstrapped; paired point deltas only",
        })
    _write_rows_atomic(metrics_path, metric_rows)
    _write_rows_atomic(paired_path, paired_rows)
    _write_json_atomic(manifest_path, {
        "status": "complete", "db": db, "inputs": expected_inputs,
        "analysis_ids": list(MODEL_IDS), "paired_comparators": list(ANALYSIS_IDS),
        "paired_ci_draws": N_BOOT, "paired_ci_seed": boot_seed,
        "paired_ci_unit": "patient", "brier_logloss_ci": True,
        "auroc_average_precision_ci": False,
        "clinical_input_availability_verified": False, "clinical_promotion_allowed": False,
        "patient_level_exported_to_local": False,
        "performance_metrics_sha256": sha256_file(metrics_path),
        "paired_deltas_sha256": sha256_file(paired_path),
    })
    return metric_rows, paired_rows


def _load_or_run_calibration(db: str, model_id: str,
                             frame: pd.DataFrame, gate: dict[str, Any],
                             plan_row: dict[str, Any], provenance: dict[str, Any],
                             out_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    run_dir = out_dir / "calibration" / db / model_id
    draws_path = run_dir / "recalibration_draws.csv"
    summary_path = run_dir / "recalibration_summary.csv"
    paired_path = run_dir / "recalibration_paired_patient_ci.csv"
    manifest_path = run_dir / "manifest.json"
    plan = plan_row
    expected_inputs = {
        "fm_prediction_manifest_sha256": provenance["fm_prediction_manifest_sha256"],
        "baseline_prediction_manifest_sha256": provenance["baseline_prediction_manifest_sha256"],
        "identity_attestation_sha256": provenance["subject_identity_attestation_sha256"],
        "eligibility_manifest_sha256": gate["eligibility_manifest_sha256"],
        "eligibility_table_sha256": gate["eligibility_table_sha256"],
        "sampling_plan_sha256": plan["sampling_plan_sha256"],
    }
    if manifest_path.is_file():
        existing = _json(manifest_path)
        if existing.get("status") != "complete" or existing.get("inputs") != expected_inputs:
            raise ValueError(f"{db}/{model_id}: saved calibration output is stale or incomplete")
        for path, key in ((draws_path, "draws_sha256"), (summary_path, "summary_sha256"),
                          (paired_path, "paired_ci_sha256")):
            if sha256_file(path) != existing.get(key):
                raise ValueError(f"{db}/{model_id}: saved calibration table hash mismatch")
        return (pd.read_csv(draws_path).to_dict("records"),
                pd.read_csv(summary_path).to_dict("records"),
                pd.read_csv(paired_path).to_dict("records"))

    print(json.dumps({"stage": "transport_calibration_arm_started", "db": db,
                      "analysis_id": model_id,
                      "sampling_plan_sha256": plan["sampling_plan_sha256"]}, sort_keys=True), flush=True)
    original_plan = plan.get("original_eicu_comparison", {})
    calibration_db = db
    calibration_frame = frame
    if (db == "eicu_semantic_v4_2"
            and original_plan.get("reused_original_plan") is True
            and original_plan.get("plan_db_alias") == "eicu"):
        calibration_db = "eicu"
        calibration_frame = frame.copy()
        calibration_frame["db"] = calibration_db
    draws, summary, ci_rows = run_db(
        calibration_db, calibration_frame, repeats=REPEATS, n_boot=N_BOOT, seed=SEED, budgets=BUDGETS,
        cluster_unit="patient", expected_shared_plan=plan,
    )
    model_tag = MODEL_TAGS[model_id]
    for collection in (draws, summary, ci_rows):
        for row in collection:
            if str(row.get("db")) == calibration_db:
                row["db"] = db
            row["analysis_id"] = model_id
            row["model_tag"] = model_tag
    for row in summary:
        row["reused_original_eicu_sampling_plan"] = bool(
            original_plan.get("reused_original_plan", False))
        row["cross_version_curve_comparison_allowed"] = bool(
            original_plan.get("cross_version_curve_comparison_allowed", False))
    _write_rows_atomic(draws_path, draws)
    _write_rows_atomic(summary_path, summary)
    _write_rows_atomic(paired_path, ci_rows)
    _write_json_atomic(manifest_path, {
        "status": "complete", "db": db, "analysis_id": model_id, "model_tag": model_tag,
        "inputs": expected_inputs,
        "n_draw_rows": len(draws), "n_summary_rows": len(summary), "n_paired_ci_rows": len(ci_rows),
        "sampling_plan": {
            key: plan.get(key) for key in PLAN_COMPARE_FIELDS
        },
        "original_eicu_comparison": original_plan,
        "cluster_unit": "patient", "n_patient_cluster_bootstrap_draws": N_BOOT,
        "clinical_input_availability_verified": False, "clinical_promotion_allowed": False,
        "patient_level_exported_to_local": False,
        "draws_sha256": sha256_file(draws_path),
        "summary_sha256": sha256_file(summary_path),
        "paired_ci_sha256": sha256_file(paired_path),
    })
    print(json.dumps({"stage": "transport_calibration_arm_complete", "db": db,
                      "analysis_id": model_id, "draw_rows": len(draws),
                      "summary_rows": len(summary), "paired_ci_rows": len(ci_rows)},
                     sort_keys=True), flush=True)
    return draws, summary, ci_rows


def evaluate_transport(*, out_dir: Path,
                       cohorts: tuple[str, ...] = TARGET_COHORTS) -> dict[str, Any]:
    if not cohorts or len(set(cohorts)) != len(cohorts) or any(db not in TARGET_COHORTS for db in cohorts):
        raise ValueError(f"requested evaluation cohorts must be a unique subset of {TARGET_COHORTS}")
    frames, gates, endpoints, provenance = _load_all_model_frames()
    plan_path = out_dir / "sampling_plan_manifest.json"
    plan_document = _read_and_validate_sampling_plan(plan_path, provenance, frames)
    for db in cohorts:
        metric_rows, paired_rows = _performance_for_cohort(
            db, frames, gates, provenance, out_dir,
        )
        del metric_rows, paired_rows
        plan_row = plan_document["cohort_sampling_plans"][db]
        for model_id in MODEL_IDS:
            _load_or_run_calibration(
                db, model_id, frames[model_id][db], gates[db], plan_row,
                provenance, out_dir,
            )
    metrics_all: list[dict[str, Any]] = []
    paired_all: list[dict[str, Any]] = []
    draws_all: list[dict[str, Any]] = []
    summary_all: list[dict[str, Any]] = []
    calibration_ci_all: list[dict[str, Any]] = []
    checkpoint_manifests: dict[str, dict[str, str]] = {}
    complete_cohorts: list[str] = []
    for db in TARGET_COHORTS:
        performance_manifest = out_dir / "performance" / db / "manifest.json"
        if performance_manifest.is_file():
            metrics_all.extend(pd.read_csv(performance_manifest.parent / "performance_metrics.csv").to_dict("records"))
            paired_all.extend(pd.read_csv(performance_manifest.parent / "paired_deltas.csv").to_dict("records"))
            checkpoint_manifests[f"performance/{db}"] = {
                "manifest_sha256": sha256_file(performance_manifest),
            }
        cohort_complete = performance_manifest.is_file()
        for model_id in MODEL_IDS:
            run_dir = out_dir / "calibration" / db / model_id
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.is_file():
                cohort_complete = False
                continue
            draws_all.extend(pd.read_csv(run_dir / "recalibration_draws.csv").to_dict("records"))
            summary_all.extend(pd.read_csv(run_dir / "recalibration_summary.csv").to_dict("records"))
            calibration_ci_all.extend(
                pd.read_csv(run_dir / "recalibration_paired_patient_ci.csv").to_dict("records"))
            checkpoint_manifests[f"calibration/{db}/{model_id}"] = {
                "manifest_sha256": sha256_file(manifest_path),
            }
        if cohort_complete:
            complete_cohorts.append(db)
    aggregate_paths = {
        "performance_metrics": out_dir / "transport_performance_metrics.csv",
        "paired_deltas": out_dir / "transport_paired_deltas.csv",
        "calibration_draws": out_dir / "transport_calibration_draws.csv",
        "calibration_summary": out_dir / "transport_calibration_summary.csv",
        "calibration_paired_patient_ci": out_dir / "transport_calibration_paired_patient_ci.csv",
    }
    _write_rows_atomic(aggregate_paths["performance_metrics"], metrics_all)
    _write_rows_atomic(aggregate_paths["paired_deltas"], paired_all)
    _write_rows_atomic(aggregate_paths["calibration_draws"], draws_all)
    _write_rows_atomic(aggregate_paths["calibration_summary"], summary_all)
    _write_rows_atomic(aggregate_paths["calibration_paired_patient_ci"], calibration_ci_all)
    manifest = {
        "status": "complete" if set(complete_cohorts) == set(TARGET_COHORTS) else "partial",
        "schema": "ehr_fm_recovered_transport_analysis_v4_2",
        "cohorts": list(TARGET_COHORTS), "cohorts_complete": complete_cohorts,
        "cohorts_evaluated_this_run": list(cohorts), "model_ids": list(MODEL_IDS),
        "endpoint_by_cohort": endpoints,
        "sampling_plan_manifest": str(plan_path),
        "sampling_plan_manifest_sha256": sha256_file(plan_path),
        "provenance": provenance,
        "target_counts": {
            db: {
                "n_stays": int(len(frames["main20k_real"][db])),
                "n_patients": int(frames["main20k_real"][db]["subject"].nunique()),
                "events": int(frames["main20k_real"][db]["y"].sum()),
            }
            for db in complete_cohorts
        },
        "paired_performance_ci": {
            "n_bootstrap": N_BOOT, "cluster_unit": "patient",
            "metrics_with_ci": ["brier", "logloss"],
            "auroc_average_precision_ci": False,
        },
        "shared_calibration": {
            "budgets": list(BUDGETS), "repeats": REPEATS,
            "methods": ["Jeffreys-smoothed adaptation prevalence", "offset", "Platt"],
            "plan_frozen_before_metric_evaluation": True,
            "patient_disjoint_adaptation_assessment": True,
            "models_use_same_plan_per_cohort": True,
            "assessment_cluster_bootstrap_draws": N_BOOT,
        },
        "cohort_model_checkpoints": checkpoint_manifests,
        "aggregate_tables": {
            key: {"path": str(path), "sha256": sha256_file(path),
                  "rows": int(len(pd.read_csv(path)))}
            for key, path in aggregate_paths.items()
        },
        "clinical_input_availability_verified": False,
        "clinical_promotion_allowed": False,
        "patient_level_exported_to_local": False,
    }
    _write_json_atomic(out_dir / "transport_analysis_manifest.json", manifest)
    print(json.dumps({
        "status": "recovered_transport_analysis_complete",
        "manifest": str(out_dir / "transport_analysis_manifest.json"),
        "aggregate_tables": manifest["aggregate_tables"],
    }, sort_keys=True), flush=True)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("freeze-plans", "evaluate"), required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cohorts", nargs="+", choices=TARGET_COHORTS,
                        help="subset of recovered cohorts to evaluate; default is all")
    args = parser.parse_args(argv)
    os.chdir(REV)
    if args.phase == "freeze-plans":
        freeze_sampling_plans(out_dir=args.out_dir)
    else:
        evaluate_transport(out_dir=args.out_dir,
                           cohorts=tuple(args.cohorts) if args.cohorts else TARGET_COHORTS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
