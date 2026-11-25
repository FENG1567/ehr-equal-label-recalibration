"""Validate and finalize the completed primary baseline prediction artifacts.

This is a recovery path for the 2026-11-23 unpaired source fit whose worker
exited after writing all four Parquet files but before the aggregate outputs.
It does not fit models or change any prediction rows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from conventional_baselines import (
    INPUT_SCOPE, PRIMARY_SCOPE, _candidate_specs, _effective_prediction_scope,
    _identity_hash, _sha256, evaluation_metrics,
)
from paired_baselines import (
    compare_artifacts, ordered_row_digest, sha256_file,
    validate_complete_fm_prediction_manifest,
)

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
DBS = ("mimic", "eicu")
SEED = 20261122
N_BOOT = 1000
CLEARED_ELIGIBILITY_SHA = "ea2b55bc3cf071d48b019d4ba97b4a3dc53d5d7fb3f6ac49a9e7c26007064cb5"
CLEARED_TABLE_SHA = {
    "mimic": "c2774cd59e4df8706960e4fa58680ea6ec72b4af22adddf82b2389dc5b74749f",
    "eicu": "d138fd4bf718f051af3aa39298958dad44ccdf2462b583d554e28d214028f04d",
}
EXPECTED = {
    "conventional_lr_pattern_only": ("conventional_lr", "pattern_only"),
    "conventional_hgb_pattern_only": ("conventional_hgb", "pattern_only"),
    "conventional_lr_value_plus_pattern": ("conventional_lr", "value_plus_pattern"),
    "conventional_hgb_value_plus_pattern": ("conventional_hgb", "value_plus_pattern"),
}
FIT_CODE_SHA256 = "16eb056648e5fd30929795322d4ad6402554b322d318761a8019937be03a9fb4"


def _engine_module():
    engine_dir = str(REV / "engine/code")
    if engine_dir not in sys.path:
        sys.path.insert(0, engine_dir)
    import v4_engine
    return v4_engine


def _read_selected_parameters(log_path: Path) -> dict[str, dict[str, Any]]:
    recovered: dict[str, dict[str, Any]] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        analysis_id = str(row.get("analysis_id", ""))
        if (analysis_id not in EXPECTED or row.get("input_condition") != "real"
                or not isinstance(row.get("selected_parameters"), dict)):
            continue
        if analysis_id in recovered:
            raise ValueError(f"run log repeats selected parameters for {analysis_id}")
        recovered[analysis_id] = row["selected_parameters"]
    if set(recovered) != set(EXPECTED):
        raise ValueError(
            f"run log does not attest all four completed source selections: {sorted(recovered)}")
    grids = _candidate_specs()
    for analysis_id, (model_tag, _) in EXPECTED.items():
        if recovered[analysis_id] not in grids[model_tag]:
            raise ValueError(f"{analysis_id}: logged selected parameters are outside the registered grid")
    return recovered


def _assert_same_order(actual: pd.Series, expected: pd.Series, label: str) -> None:
    left = actual.astype(str).to_numpy()
    right = expected.astype(str).to_numpy()
    if not np.array_equal(left, right):
        raise ValueError(f"{label}: prediction identity/order differs from the current eligible source rows")


def _assert_same_binary_labels(actual: pd.Series, expected: pd.Series, label: str) -> None:
    left = actual.to_numpy(dtype=np.int8)
    right = expected.to_numpy(dtype=np.int8)
    if not np.array_equal(left, right):
        raise ValueError(f"{label}: prediction outcomes differ from the current eligible labels")


def _verify_prediction_frame(
    frame: pd.DataFrame,
    *,
    analysis_id: str,
    model_tag: str,
    feature_type: str,
    expected_by_db: dict[str, pd.DataFrame],
    entry_by_db: dict[str, dict[str, Any]],
    mask_hash: str,
) -> None:
    required = {
        "db", "analysis_id", "stay_id", "subject", "rows", "arrayindex",
        "split", "y", "p", "model_tag", "feature_type", "input_condition",
        "input_mask_sha256", "endpoint", "role", "prediction_scope",
        "input_scope", "clinical_input_availability_verified",
        "eligibility_manifest_sha256", "eligibility_table_sha256",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{analysis_id}: prediction file lacks columns {sorted(missing)}")
    if frame.empty or frame.duplicated(["db", "stay_id"]).any():
        raise ValueError(f"{analysis_id}: prediction file is empty or has duplicate stay IDs")
    if set(frame["db"].astype(str).unique()) != set(DBS):
        raise ValueError(f"{analysis_id}: prediction file does not contain exactly MIMIC and eICU")
    for value, expected in (
        (frame["analysis_id"], analysis_id), (frame["model_tag"], model_tag),
        (frame["feature_type"], feature_type), (frame["input_condition"], "real"),
        (frame["input_mask_sha256"], mask_hash), (frame["endpoint"], "hospital"),
        (frame["role"], "evaluation"), (frame["prediction_scope"], PRIMARY_SCOPE),
        (frame["input_scope"], INPUT_SCOPE),
    ):
        if value.astype(str).ne(str(expected)).any():
            raise ValueError(f"{analysis_id}: inconsistent prediction metadata for {value.name}")
    if frame["clinical_input_availability_verified"].isna().any() or frame[
            "clinical_input_availability_verified"].ne(False).any():
        raise ValueError(f"{analysis_id}: clinical input availability is not explicitly unverified")
    if not np.isfinite(frame["p"].to_numpy(dtype=float)).all():
        raise ValueError(f"{analysis_id}: probabilities contain nonfinite values")

    for db in DBS:
        part = frame.loc[frame["db"].astype(str) == db].copy().reset_index(drop=True)
        expected = expected_by_db[db].reset_index(drop=True)
        entry = entry_by_db[db]
        if len(part) != len(expected):
            raise ValueError(f"{analysis_id}/{db}: row count differs from the cleared eligible evaluation pool")
        for actual_col, expected_col in (
            ("stay_id", "stay_id"), ("subject", "subject"),
            ("rows", "rows"), ("arrayindex", "rows"), ("split", "split"),
            ("endpoint", "endpoint"),
        ):
            _assert_same_order(part[actual_col], expected[expected_col],
                               f"{analysis_id}/{db}/{actual_col}")
        _assert_same_binary_labels(part["y"], expected["y_mort"],
                                   f"{analysis_id}/{db}/y")
        for col, expected_value in (
            ("eligibility_manifest_sha256", entry["eligibility_manifest_sha256"]),
            ("eligibility_table_sha256", entry["eligibility_table_sha256"]),
        ):
            if part[col].astype(str).ne(str(expected_value)).any():
                raise ValueError(f"{analysis_id}/{db}: {col} differs from current eligibility attestation")
        if part["p"].lt(0).any() or part["p"].gt(1).any():
            raise ValueError(f"{analysis_id}/{db}: prediction probability is outside [0, 1]")


def finalize(
    eligibility_dir: Path,
    out_dir: Path,
    run_log: Path,
    fm_manifest_path: Path,
    expected_fm_prediction_path: Path,
) -> dict[str, Any]:
    eligibility_dir = eligibility_dir.resolve()
    out_dir = out_dir.resolve()
    run_log = run_log.resolve()
    fm_manifest_path = fm_manifest_path.resolve()
    expected_fm_prediction_path = expected_fm_prediction_path.resolve()
    if not run_log.is_file():
        raise FileNotFoundError(f"interrupted baseline run log is missing: {run_log}")
    if sha256_file(eligibility_dir / "eligibility_manifest.json") != CLEARED_ELIGIBILITY_SHA:
        raise ValueError("eligibility manifest is not the root-cleared primary manifest")

    fm_doc = json.loads(fm_manifest_path.read_text(encoding="utf-8"))
    validate_complete_fm_prediction_manifest(fm_doc)
    if fm_doc.get("run_id") != "status_consistent_primary_core_fix2":
        raise ValueError("FM manifest is not the currently authorized core-fix2 prediction run")
    fm_records = [row for row in fm_doc.get("predictions", [])
                  if row.get("model_tag") == "main20k"
                  and row.get("input_condition") == "real"
                  and row.get("db") in DBS]
    if {str(row.get("db")) for row in fm_records} != set(DBS):
        raise ValueError("current FM manifest lacks the cleared real-condition MIMIC/eICU entries")
    paths = {Path(str(row["prediction_file"])).resolve() for row in fm_records}
    if paths != {expected_fm_prediction_path}:
        raise ValueError("current FM manifest does not name the expected core-fix2 artifact")

    selected = _read_selected_parameters(run_log)
    engine = _engine_module()
    eligibility_manifest_hash = sha256_file(eligibility_dir / "eligibility_manifest.json")
    entries: dict[str, dict[str, Any]] = {}
    expected_by_db: dict[str, pd.DataFrame] = {}
    source_splits: dict[str, pd.DataFrame] = {}
    sequence_hashes: dict[str, str] = {}
    for db in DBS:
        table_path = eligibility_dir / f"{db}.parquet"
        if sha256_file(table_path) != CLEARED_TABLE_SHA[db]:
            raise ValueError(f"{db}: eligibility table differs from the cleared manifest")
        eligible, entry = engine.load_ready_eligibility(db, eligibility_dir)
        eligible = eligible.loc[eligible["endpoint"].astype(str) == "hospital"].copy()
        expected_eligible = {"mimic": 73936, "eicu": 130544}[db]
        if len(eligible) != expected_eligible:
            raise ValueError(f"{db}: eligible row count differs from the cleared primary population")
        if (entry.get("retrospective_landmark_prediction_allowed") is not True
                or _effective_prediction_scope(entry, engine) != PRIMARY_SCOPE
                or entry.get("input_scope") != INPUT_SCOPE
                or entry.get("clinical_input_availability_verified") is not False
                or entry.get("eligibility_manifest_sha256") != eligibility_manifest_hash
                or entry.get("eligibility_table_sha256") != CLEARED_TABLE_SHA[db]):
            raise ValueError(f"{db}: current eligibility attestation fails the registered run scope")
        sequence_path = ROOT / f"data/processed/seq_{db}.npz"
        sequence_hashes[db] = sha256_file(sequence_path)
        if sequence_hashes[db] != entry.get("sequence_npz_sha256"):
            raise ValueError(f"{db}: current sequence NPZ differs from the cleared source attestation")
        ds = engine.load_sequence(db)
        selected_rows = engine.condition_rows(ds, "real", ())
        matched = engine.align_eligible_rows(db, ds, eligible, selected_rows)
        matched["split"] = np.asarray(ds["split"], dtype=np.int8)[
            matched["rows"].to_numpy(dtype=np.int64)]
        matched["y"] = matched["y_mort"].to_numpy(dtype=np.int8)
        matched["subject"] = matched["subject"].astype(str)
        matched["stay_id"] = matched["stay_id"].astype(str)
        if db == "mimic":
            evaluation = matched.loc[matched["split"] == 2].copy()
            source_splits["split0"] = matched.loc[matched["split"] == 0].copy()
            source_splits["split1"] = matched.loc[matched["split"] == 1].copy()
            source_splits["split2"] = matched.loc[matched["split"] == 2].copy()
            split_groups = [set(source_splits[f"split{n}"]["subject"]) for n in range(3)]
            if any(split_groups[i] & split_groups[j]
                   for i in range(3) for j in range(i + 1, 3)):
                raise ValueError("MIMIC source split patient groups overlap")
            for split in (0, 1):
                if source_splits[f"split{split}"]["y"].nunique() != 2:
                    raise ValueError(f"MIMIC source split {split} lacks both outcome classes")
        else:
            evaluation = matched.copy()
        expected_by_db[db] = evaluation.reset_index(drop=True)
        entries[db] = entry
        print(json.dumps({"preflight": db, "eligible": len(eligible),
                          "evaluation_rows": len(evaluation),
                          "evaluation_patients": int(evaluation["subject"].nunique()),
                          "eligibility_sha256": entry["eligibility_table_sha256"],
                          "sequence_sha256": sequence_hashes[db]}), flush=True)

    mask_hash = engine.input_mask_hash("real", ())
    metrics_rows: list[dict[str, Any]] = []
    choices: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    prediction_dir = out_dir / "results/predictions"
    for analysis_id, (model_tag, feature_type) in EXPECTED.items():
        path = prediction_dir / f"{analysis_id}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"completed run prediction artifact is missing: {path}")
        frame = pd.read_parquet(path)
        _verify_prediction_frame(
            frame, analysis_id=analysis_id, model_tag=model_tag,
            feature_type=feature_type, expected_by_db=expected_by_db,
            entry_by_db=entries, mask_hash=mask_hash,
        )
        file_hash = sha256_file(path)
        for db in DBS:
            part = frame.loc[frame["db"].astype(str) == db].copy().reset_index(drop=True)
            entry = entries[db]
            identity = entry.get("subject_identity_basis", {})
            cluster_unit = ("patient" if isinstance(identity, dict)
                            and identity.get("patient_disjoint_claim_allowed") is True
                            else "supplied_subject_identifier")
            metrics = evaluation_metrics(
                part, n_boot=N_BOOT,
                seed=SEED + sum(ord(ch) for ch in db), cluster_unit=cluster_unit,
            )
            metrics_rows.append({"analysis_id": analysis_id,
                                 "input_condition": "real", "db": db, **metrics})
            prediction_records.append({
                "analysis_id": analysis_id, "db": db,
                "prediction_file": str(path), "prediction_file_sha256": file_hash,
                "ordered_row_digest_sha256": ordered_row_digest(part),
                "n_prediction_rows": int(len(part)), "role": "evaluation",
                "prediction_scope": PRIMARY_SCOPE, "input_scope": INPUT_SCOPE,
                "clinical_input_availability_verified": False,
                "retrospective_landmark_prediction_allowed": True,
                "input_condition": "real", "input_mask_sha256": mask_hash,
                "input_mask_spec": None, "fm_analysis_id": "main20k_real",
                "eligibility_manifest_sha256": entry["eligibility_manifest_sha256"],
                "eligibility_table_sha256": entry["eligibility_table_sha256"],
                "source_npz_sha256": sequence_hashes[db],
            })
        split0, split1, split2 = (
            source_splits["split0"], source_splits["split1"], source_splits["split2"])
        params = selected[analysis_id]
        choices.append({
            "model_tag": model_tag, "feature_type": feature_type,
            "input_condition": "real", "input_mask_sha256": mask_hash,
            "fm_analysis_id": "main20k_real", "selected_parameters": params,
            "selected_parameters_recovered_from_run_log": True,
            "selected_validation_brier": None,
            "selected_validation_brier_recovered": False,
            "tie_break": "lowest MIMIC split 1 Brier; exact ties choose simpler parameterization",
            "final_model_fit_scope": "MIMIC source split 0 only; split 1 is selection-only and not refit",
            "source_fit_rows": int(len(split0)),
            "source_train_identity_sha256": _identity_hash("mimic", split0),
            "source_validation_identity_sha256": _identity_hash("mimic", split1),
            "source_final_fit_identity_sha256": _identity_hash("mimic", split0),
            "source_train_patients": int(split0["subject"].nunique()),
            "source_validation_patients": int(split1["subject"].nunique()),
            "source_test_patients": int(split2["subject"].nunique()),
            "sequence_source_sha256": sequence_hashes["mimic"],
            "selected_parameters_evidence": "completed worker JSON progress line",
        })
        print(json.dumps({"validated_artifact": analysis_id,
                          "rows": len(frame), "sha256": file_hash,
                          "selected_parameters": params}), flush=True)

    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    baseline_prediction_manifest_path = results_dir / "baseline_prediction_manifest.json"
    baseline_prediction_manifest = {
        "status": "complete", "endpoint": "hospital", "seed": SEED,
        "input_conditions": ["real"],
        "source_fit_run_mode": "explicit_unpaired_real_only",
        "patient_level_data_exported_to_local": False,
        "cohort_readiness": {db: {
            "retrospective_landmark_prediction_allowed": True,
            "ready_for_future24h_clinical_claim": False,
            "eligibility_manifest_sha256": entries[db]["eligibility_manifest_sha256"],
            "eligibility_table_sha256": entries[db]["eligibility_table_sha256"],
            "sequence_npz_sha256": sequence_hashes[db],
            "prediction_scope": PRIMARY_SCOPE,
            "input_scope": INPUT_SCOPE,
            "clinical_input_availability_verified": False,
        } for db in DBS},
        "predictions": prediction_records,
    }
    baseline_prediction_manifest_path.write_text(
        json.dumps(baseline_prediction_manifest, indent=2), encoding="utf-8")
    pd.DataFrame(metrics_rows).to_csv(results_dir / "baseline_evaluation_metrics.csv", index=False)

    pairing, paired_metrics, pairing_summary = compare_artifacts(
        baseline_prediction_manifest_path, fm_manifest_path,
        expected_fm_prediction_path, n_boot=N_BOOT, seed=SEED,
    )
    pd.DataFrame(pairing).to_csv(results_dir / "baseline_pairing_audit.csv", index=False)
    pd.DataFrame(paired_metrics).to_csv(results_dir / "baseline_paired_metrics.csv", index=False)
    current_manifest_hash = sha256_file(fm_manifest_path)
    names_path = ROOT / "data/processed/meta_mimic.json"
    manifest = {
        "status": "complete",
        "analysis_stage": "source_validation_selection_and_retrospective_landmark_prediction",
        "seed": SEED, "cohorts": list(DBS), "endpoint": "hospital",
        "source_fit_run_mode": "explicit_unpaired_real_only_then_paired_to_current_fm",
        "paired_comparison_deferred": False,
        "paired_comparison_performed_after_source_fit": True,
        "recovered_from_interrupted_worker": True,
        "worker_exit_cause": "eligibility manifest omitted prediction_scope although registered operational_proxy maps to the primary scope",
        "cohort_readiness": baseline_prediction_manifest["cohort_readiness"],
        "feature_contract": {
            "vocabulary": "canonical MIMIC token vocabulary shared by V/B/H/L inputs",
            "value_plus_pattern": "observed,count/128,mean/min/max/last bin /10,first/last/mean hour /23",
            "pattern_only": "observed,count/128,first/last/mean hour /23; no bin-value summaries",
            "last_bin_tie_break": "maximum hour, then last source token order",
            "missing_continuous_summaries": 0,
            "feature_construction_uses_outcomes": False,
            "input_condition_specs": {"real": {db: None for db in DBS}},
        },
        "selection": "MIMIC split 0 only fits candidates; MIMIC split 1 Brier selects; ties choose simpler; selected model remains fit on split 0 only; no split 1 refit; MIMIC split 2 and external cohorts are evaluation only",
        "grids": _candidate_specs(), "choices": choices,
        "candidate_trials": [],
        "candidate_trials_recovered": False,
        "candidate_trial_loss_note": "Original worker exited before serializing candidate validation scores, convergence warnings, and fit seconds. Selected parameters were recovered from its flushed progress records; no model was refit during recovery.",
        "feature_matrix_dimensions_and_bytes": {},
        "feature_matrix_metadata_recovered": False,
        "evaluation_metrics_file": str(results_dir / "baseline_evaluation_metrics.csv"),
        "baseline_prediction_manifest": str(baseline_prediction_manifest_path.resolve()),
        "baseline_prediction_manifest_sha256": sha256_file(baseline_prediction_manifest_path),
        "fm_prediction_manifest": str(fm_manifest_path),
        "fm_prediction_manifest_sha256": current_manifest_hash,
        "pairing_to_fm": pairing_summary,
        "paired_metrics_file": str(results_dir / "baseline_paired_metrics.csv"),
        "target_recalibration": "use the same actual patient/index-stay draw hashes after exact matched-population verification; compare runs with verify_shared_recalibration_manifests before pooling",
        "patient_level_data_exported_to_local": False,
        "fit_wall_time_recorded": False,
        "convergence_warning_recorded": False,
        "baseline_fit_code_sha256": FIT_CODE_SHA256,
        "baseline_code_sha256_at_finalization": _sha256(REV / "baselines/conventional_baselines.py"),
        "feature_code_sha256": _sha256(REV / "baselines/baseline_features.py"),
        "variable_vocab_sha256": _sha256(names_path),
        "source_npz_sha256_by_db": sequence_hashes,
        "eligibility_manifest_sha256": eligibility_manifest_hash,
        "eligibility_table_sha256_by_db": CLEARED_TABLE_SHA,
        "baseline_fit_worker_log": str(run_log),
    }
    manifest_path = results_dir / "baseline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "baseline_manifest": str(manifest_path),
                      "baseline_manifest_sha256": sha256_file(manifest_path),
                      "fm_manifest_sha256": current_manifest_hash,
                      "paired_audit_rows": len(pairing),
                      "paired_metric_rows": len(paired_metrics)}), flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eligibility-dir", type=Path,
                        default=REV / "temporal/results/eligibility/status_consistent_v4_1/landmark_v4_1")
    parser.add_argument("--out-dir", type=Path,
                        default=REV / "baselines/results/primary_status_consistent_20261123")
    parser.add_argument("--run-log", type=Path,
                        default=REV / "baselines/logs/conventional_baselines_primary_status_consistent_20261123.log")
    parser.add_argument("--fm-manifest", type=Path, required=True)
    parser.add_argument("--fm-prediction", type=Path, required=True)
    args = parser.parse_args()
    finalize(args.eligibility_dir, args.out_dir, args.run_log,
             args.fm_manifest, args.fm_prediction)
    print("BASELINE_RECOVERY_FINALIZATION_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
