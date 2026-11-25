"""Score frozen INSPIRE baselines and evaluate the locked 500-patient Platt update.

This job never refits the source models or redraws the locked patient roles. The
four conventional models are loaded from their hash-bound MIMIC source bundle;
the three density-study FM heads are read from the saved no-fit score artifact.
Only Platt calibration uses INSPIRE adaptation labels. All reported target
metrics use the frozen assessment role, with conditional patient bootstrap CIs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
ROBUSTNESS = REV / "robustness"
BASELINES = REV / "baselines"
BASELINE_TRANSPORT = BASELINES / "transport_v4_2"
SOURCE_REPRO = BASELINE_TRANSPORT / "source_reproduction_manifest.json"
SOURCE_BUNDLE = BASELINE_TRANSPORT / "models/frozen_source_models.pkl"
SOURCE_BASELINE_MANIFEST = BASELINES / "results/primary_status_consistent_20261123/results/baseline_manifest.json"
SOURCE_BASELINE_PRED_MANIFEST = BASELINES / "results/primary_status_consistent_20261123/results/baseline_prediction_manifest.json"
GATE = REV / "temporal/results/eligibility/inspire_exploratory_locked_v4_1"
ROLE_LOCK = GATE / "assessment_lock.json"
IDENTITY = REV / "admin/subject_identity_attestations_v4_2.json"
DENSITY_DIR = REV / "density_repair/inspire_frozen_scoring_v4_2"
DENSITY_PREDICTIONS = DENSITY_DIR / "frozen_density_inspire_predictions.parquet"
DENSITY_MANIFEST = DENSITY_DIR / "frozen_density_inspire_scoring_manifest.json"
DENSITY_SOURCE = REV / "density_repair/run_source_20261123_03"
DENSITY_SCORER = REV / "manuscript/score_frozen_density_inspire_v4.py"
DENSITY_FITTED_PROBES = DENSITY_SOURCE / "fitted_probes.pkl"
CORE_RUN = REV / "engine/results/prediction_runs/inspire_core_source_reuse_m69_20261123-39444bed8e30c9db"
BIG_RUN = REV / "engine/results/prediction_runs/inspire_big_source_reuse_m69_20261123-834854b4b4f4d7a9"
COHORTS = ("inspire_postop_v4_1", "inspire_broad_surgery_associated_v4_1")
FM_ARMS = ("original_ordinary_C1", "ordinary_tuned", "density_augmented")
BASELINE_ARMS = (
    "conventional_lr_pattern_only",
    "conventional_hgb_pattern_only",
    "conventional_lr_value_plus_pattern",
    "conventional_hgb_value_plus_pattern",
)
ALL_ARMS = FM_ARMS + BASELINE_ARMS
EXPECTED_COUNTS = {
    "inspire_postop_v4_1": {
        "adaptation_patients": 500, "adaptation_events": 21,
        "assessment_patients": 3358, "assessment_events": 106,
        "assessment_nonevents": 3252,
    },
    "inspire_broad_surgery_associated_v4_1": {
        "adaptation_patients": 500, "adaptation_events": 24,
        "assessment_patients": 5213, "assessment_events": 186,
        "assessment_nonevents": 5027,
    },
}
EXPECTED_GATE_SHA = "97c7aeacacb2af9cfce63086d6dc5b9ce9838292a4197e43d3ec98d7f533c019"
EXPECTED_ROLE_LOCK_SHA = "3ac72b0d60dc47c81d5fb894f827f573735f32e220f430852281947c10152ffd"
EXPECTED_DENSITY_LOCK_SHA = "fc69bcbe2e6433fe5d6e055e4e288eba9634857eec642822aaf9d5d2b1da5f1d"
EXPECTED_DENSITY_AUDIT_SHA = "4e3467901d4ac021096a51232e35a165845cab7cb03e36c81eacd42020247c5b"
EXPECTED_DENSITY_PRED_SHA = "347a21e0a79c37d290fccf1f07afdd09a8de4cd8c8094f078f4ad48dc6af3d8a"
EXPECTED_SOURCE_BUNDLE_SHA = "e9a09363585f082a92e21628979830e47f35221584d3a9d60a267ae4b782b0b6"
EXPECTED_DENSITY_SCORER_SHA = "e634d29b179aac6553ca27349bb1d8e051ad0568ae562ab8dd99a722f9aa60ca"
SEED = 20261123
N_BOOT = 1000
PLATT_C = 1e6

for _path in (BASELINES, ROBUSTNESS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from conventional_baselines import _engine_module, _summarize_conditioned_inputs  # noqa: E402
from paired_baselines import sha256_file  # noqa: E402


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def _write_frame(path: Path, frame: pd.DataFrame, parquet: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name("." + path.name + ".tmp")
    if parquet:
        frame.to_parquet(temp, index=False)
    else:
        frame.to_csv(temp, index=False)
    temp.replace(path)


def _fail(message: str) -> None:
    raise ValueError(message)


def _verify_no_fit_reuse_run(run_dir: Path, expected_analyses: set[str]) -> dict[str, Any]:
    manifest_path = run_dir / "prediction_manifest.json"
    completion_path = run_dir / "target_source_probe_reuse_completion.json"
    manifest = _json(manifest_path)
    completion = _json(completion_path)
    if manifest.get("mode") != "predict" or manifest.get("source_probe_refit_forbidden") is not True:
        _fail(f"{run_dir.name}: prediction manifest does not forbid source-probe refitting")
    if completion.get("status") != "target_source_probe_reuse_complete" or int(
            completion.get("fit_source_probe_calls", -1)) != 0:
        _fail(f"{run_dir.name}: no-fit source-probe completion record is absent")
    rows = [r for r in manifest.get("predictions", []) if isinstance(r, dict)
            and r.get("db") in COHORTS]
    analyses = {str(r.get("analysis_id")) for r in rows}
    if analyses != expected_analyses or len(rows) != 2 * len(expected_analyses):
        _fail(f"{run_dir.name}: INSPIRE no-fit prediction coverage differs from frozen analysis set")
    for row in rows:
        path = Path(str(row.get("prediction_file", "")))
        if not path.is_file() or sha256_file(path) != row.get("prediction_file_sha256"):
            _fail(f"{run_dir.name}: no-fit prediction file/hash mismatch for {row.get('analysis_id')}/{row.get('db')}")
        if row.get("input_scope") != "retrospective_measurement_time" or row.get(
                "clinical_input_availability_verified") is not False:
            _fail(f"{run_dir.name}: target input scope/availability statement is invalid")
    return {
        "run_id": manifest.get("run_id"),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "reuse_completion_path": str(completion_path),
        "reuse_completion_sha256": sha256_file(completion_path),
        "fit_source_probe_calls": 0,
        "n_prediction_entries": len(rows),
        "analysis_ids": sorted(analyses),
        "prediction_file_hashes_verified": len(rows),
        "used_for_seven_arm_calibration": False,
    }


def _verify_locked_inputs() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    for path in (ROLE_LOCK, GATE / "eligibility_manifest.json", IDENTITY,
                 DENSITY_PREDICTIONS, DENSITY_MANIFEST, SOURCE_REPRO, SOURCE_BUNDLE,
                 SOURCE_BASELINE_MANIFEST, SOURCE_BASELINE_PRED_MANIFEST,
                 DENSITY_SOURCE / "freeze_lock.json", DENSITY_SOURCE / "supplemental_paired_metrics_audit.json",
                 DENSITY_SCORER, DENSITY_FITTED_PROBES):
        if not path.is_file():
            _fail(f"required frozen INSPIRE calibration input is missing: {path}")
    role_lock_sha = sha256_file(ROLE_LOCK)
    gate_manifest_sha = sha256_file(GATE / "eligibility_manifest.json")
    if role_lock_sha != EXPECTED_ROLE_LOCK_SHA or gate_manifest_sha != EXPECTED_GATE_SHA:
        _fail("fixed INSPIRE gate or assessment lock hash changed")
    if sha256_file(DENSITY_SOURCE / "freeze_lock.json") != EXPECTED_DENSITY_LOCK_SHA:
        _fail("source-only density freeze lock hash changed")
    if sha256_file(DENSITY_SOURCE / "supplemental_paired_metrics_audit.json") != EXPECTED_DENSITY_AUDIT_SHA:
        _fail("source-only density saved-prediction audit hash changed")
    if sha256_file(DENSITY_PREDICTIONS) != EXPECTED_DENSITY_PRED_SHA:
        _fail("frozen density INSPIRE score file hash changed")
    if sha256_file(DENSITY_SCORER) != EXPECTED_DENSITY_SCORER_SHA:
        _fail("frozen density scoring code hash changed")
    source_manifest = _json(SOURCE_REPRO)
    if source_manifest.get("status") != "source_reproduction_passed":
        _fail("four baseline source-reproduction manifest is not passed")
    if sha256_file(SOURCE_BUNDLE) != EXPECTED_SOURCE_BUNDLE_SHA or source_manifest.get(
            "source_model_bundle_sha256") != EXPECTED_SOURCE_BUNDLE_SHA:
        _fail("frozen conventional source model bundle hash differs")
    if sha256_file(SOURCE_BASELINE_MANIFEST) != source_manifest.get("source_baseline_manifest_sha256"):
        _fail("source baseline model-selection manifest changed after source reproduction")
    if sha256_file(SOURCE_BASELINE_PRED_MANIFEST) != source_manifest.get(
            "source_baseline_prediction_manifest_sha256"):
        _fail("source baseline prediction manifest changed after source reproduction")

    role_doc = _json(ROLE_LOCK)
    if (role_doc.get("seed") != 20261122 or role_doc.get("budget") != 500
            or role_doc.get("no_model_performance_used_for_role_assignment") is not True
            or role_doc.get("no_redraw_or_threshold_relaxation") is not True
            or role_doc.get("metric_population") != "fixed assessment only for all model/calibration comparisons"):
        _fail("locked INSPIRE role design no longer matches the performance-blind fixed-500 protocol")
    gate_doc = _json(GATE / "eligibility_manifest.json")
    if (gate_doc.get("gate_version") != GATE.name
            or gate_doc.get("analysis_role") != "exploratory_external_only"
            or gate_doc.get("source_development_lock_sha256") != EXPECTED_DENSITY_LOCK_SHA
            or gate_doc.get("source_development_audit_sha256") != EXPECTED_DENSITY_AUDIT_SHA
            or gate_doc.get("independent_data_review_sha256") != "adcd041bb10b9f8683e9eae574b4eabf68ab3d3f8e468dc8b958fe3902a329ed"):
        _fail("locked INSPIRE eligibility manifest provenance changed")
    density_doc = _json(DENSITY_MANIFEST)
    if (density_doc.get("status") != "frozen_density_inspire_scoring_complete"
            or int(density_doc.get("source_fit_calls", -1)) != 0
            or density_doc.get("assessment_metrics_computed") is not False
            or density_doc.get("source_freeze_lock_sha256") != EXPECTED_DENSITY_LOCK_SHA
            or density_doc.get("source_saved_audit_sha256") != EXPECTED_DENSITY_AUDIT_SHA
            or density_doc.get("source_fitted_probes_sha256") != sha256_file(DENSITY_FITTED_PROBES)
            or density_doc.get("prediction_file_sha256") != EXPECTED_DENSITY_PRED_SHA
            or density_doc.get("heads") != list(FM_ARMS)
            or set(density_doc.get("targets", [])) != set(COHORTS)):
        _fail("frozen density score manifest does not match the no-fit expected contract")
    core_reuse = _verify_no_fit_reuse_run(CORE_RUN, {
        "main20k_real", "main20k_constant", "main20k_common8_mask", "main20k_no_h0_therapy",
        "abl_const20k_real", "abl_const20k_constant", "common20k_common8_mask",
    })
    big_reuse = _verify_no_fit_reuse_run(BIG_RUN, {
        "big50m_real", "big50m_full_real", "big50m_full_common8_mask",
    })

    identity_doc = _json(IDENTITY)
    identity_cohorts = identity_doc.get("cohorts", {})
    role_tables: dict[str, pd.DataFrame] = {}
    counts_summary: dict[str, Any] = {}
    for db in COHORTS:
        entry = gate_doc.get("cohorts", {}).get(db)
        expected = EXPECTED_COUNTS[db]
        ident = identity_cohorts.get(db, {})
        table_path = GATE / f"{db}.parquet"
        table_sha = sha256_file(table_path)
        if (not isinstance(entry, dict) or entry.get("fixed_patient_role_counts") != expected
                or entry.get("table_sha256") != table_sha
                or entry.get("confirmation_allowed") is not False
                or entry.get("pragmatic_confirmation_count_gate_met") is not False
                or entry.get("source_patient_identity_verified") is not True
                or entry.get("patient_disjoint_claim_allowed") is not True):
            _fail(f"{db}: frozen role counts/identity/confirmation restrictions changed")
        basis = ident.get("subject_identity_basis", {})
        if (ident.get("eligibility_manifest_sha256") != gate_manifest_sha
                or ident.get("eligibility_table_sha256") != table_sha
                or ident.get("patient_disjoint_claim_allowed") is False
                or basis.get("patient_disjoint_claim_allowed") is not True
                or basis.get("resampling_unit") != "patient"
                or basis.get("cross_scope_patient_independence_verified") is not False
                or basis.get("cross_source_patient_independence_verified") is not False):
            _fail(f"{db}: fixed-role identity attestation differs from the locked cohort")
        table = pd.read_parquet(table_path)
        required = {"db", "stay_id", "subject", "y_mort", "split_role"}
        if not required.issubset(table.columns):
            _fail(f"{db}: locked role table lacks required identity/outcome/role columns")
        table = table.copy()
        for col in ("db", "stay_id", "subject"):
            table[col] = table[col].astype(str).str.strip()
        table["y_mort"] = pd.to_numeric(table["y_mort"], errors="raise").astype(int)
        if (table["db"].astype(str).ne(db).any()
                or table["stay_id"].duplicated().any()
                or table["subject"].duplicated().any()
                or not table["y_mort"].isin([0, 1]).all()
                or not table["split_role"].isin(["adaptation", "assessment"]).all()):
            _fail(f"{db}: invalid IDs, labels, or role values in locked table")
        if table.groupby("split_role", observed=True)["subject"].nunique().to_dict() != {
                "adaptation": expected["adaptation_patients"],
                "assessment": expected["assessment_patients"]}:
            _fail(f"{db}: role counts differ from the lock; no redraw is permitted")
        if int(table.loc[table.split_role.eq("adaptation"), "y_mort"].sum()) != expected["adaptation_events"]:
            _fail(f"{db}: fixed adaptation event count differs from the lock")
        assess = table.loc[table.split_role.eq("assessment")]
        if (int(assess["y_mort"].sum()) != expected["assessment_events"]
                or int(len(assess) - assess["y_mort"].sum()) != expected["assessment_nonevents"]):
            _fail(f"{db}: fixed assessment event count differs from the lock")
        role_tables[db] = table
        counts_summary[db] = {
            "locked_table_sha256": table_sha,
            "n_patients": int(len(table)),
            "adaptation_patients": int(table.split_role.eq("adaptation").sum()),
            "adaptation_events": int(table.loc[table.split_role.eq("adaptation"), "y_mort"].sum()),
            "assessment_patients": int(assess["subject"].nunique()),
            "assessment_events": int(assess["y_mort"].sum()),
            "assessment_nonevents": int(len(assess) - assess["y_mort"].sum()),
        }

    frozen_fm = pd.read_parquet(DENSITY_PREDICTIONS)
    for col in ("db", "stay_id", "subject"):
        if col not in frozen_fm:
            _fail(f"frozen density score artifact lacks {col}")
        frozen_fm[col] = frozen_fm[col].astype(str).str.strip()
    if int(density_doc.get("output_rows", -1)) != len(frozen_fm):
        _fail("frozen density score row count differs from its manifest")
    required_pred = {"analysis_id", "db", "stay_id", "subject", "y", "p", "rows", "split"}
    if not required_pred.issubset(frozen_fm.columns):
        _fail("frozen density score artifact lacks ID/label/probability fields")
    if set(frozen_fm["analysis_id"].astype(str).unique()) != set(FM_ARMS):
        _fail("frozen density score artifact does not contain exactly the three locked FM heads")
    for db in COHORTS:
        ref = role_tables[db].copy()
        pred_db = frozen_fm.loc[frozen_fm["db"].astype(str).eq(db)].copy()
        for arm in FM_ARMS:
            arm_df = pred_db.loc[pred_db.analysis_id.astype(str).eq(arm)].copy()
            if len(arm_df) != len(ref) or arm_df["stay_id"].astype(str).duplicated().any():
                _fail(f"{db}/{arm}: frozen score count or stay uniqueness differs from locked roles")
            checked = ref[["db", "stay_id", "subject", "y_mort", "split_role"]].merge(
                arm_df[["db", "stay_id", "subject", "y", "p"]].assign(
                    stay_id=lambda d: d.stay_id.astype(str), subject=lambda d: d.subject.astype(str)),
                on=["db", "stay_id"], how="outer", validate="one_to_one", indicator=True,
                suffixes=("_role", "_pred"))
            if checked["_merge"].ne("both").any():
                _fail(f"{db}/{arm}: frozen score IDs differ from locked role table")
            for field in ("subject_role", "subject_pred"):
                if checked[field].astype(str).duplicated().any():
                    # One patient per source cohort is required; two scopes may overlap.
                    _fail(f"{db}/{arm}: duplicate patient key in matched predictions")
            if not np.array_equal(checked["y_mort"].to_numpy(dtype=int), checked["y"].to_numpy(dtype=int)):
                _fail(f"{db}/{arm}: frozen labels differ from locked outcome labels")
            if not np.isfinite(arm_df["p"].to_numpy(dtype=float)).all() or not arm_df["p"].between(0, 1).all():
                _fail(f"{db}/{arm}: frozen predictions are nonfinite or outside [0,1]")
            if arm_df["p"].isna().any():
                _fail(f"{db}/{arm}: frozen prediction contains missing probabilities")
        if len(pred_db) != len(ref) * len(FM_ARMS):
            _fail(f"{db}: unexpected extra/missing frozen density prediction rows")

    provenance = {
        "gate_manifest_path": str(GATE / "eligibility_manifest.json"),
        "gate_manifest_sha256": gate_manifest_sha,
        "assessment_lock_path": str(ROLE_LOCK),
        "assessment_lock_sha256": role_lock_sha,
        "subject_identity_attestation_path": str(IDENTITY),
        "subject_identity_attestation_sha256": sha256_file(IDENTITY),
        "density_prediction_path": str(DENSITY_PREDICTIONS),
        "density_prediction_sha256": sha256_file(DENSITY_PREDICTIONS),
        "density_scoring_manifest_path": str(DENSITY_MANIFEST),
        "density_scoring_manifest_sha256": sha256_file(DENSITY_MANIFEST),
        "density_source_freeze_lock_sha256": EXPECTED_DENSITY_LOCK_SHA,
        "density_saved_source_audit_sha256": EXPECTED_DENSITY_AUDIT_SHA,
        "density_source_fit_calls_for_target_scoring": 0,
        "source_reproduction_manifest_path": str(SOURCE_REPRO),
        "source_reproduction_manifest_sha256": sha256_file(SOURCE_REPRO),
        "frozen_source_model_bundle_path": str(SOURCE_BUNDLE),
        "frozen_source_model_bundle_sha256": sha256_file(SOURCE_BUNDLE),
        "source_baseline_manifest_sha256": sha256_file(SOURCE_BASELINE_MANIFEST),
        "source_baseline_prediction_manifest_sha256": sha256_file(SOURCE_BASELINE_PRED_MANIFEST),
        "no_fit_engine_reuse_runs": [core_reuse, big_reuse],
        "fixed_role_counts": counts_summary,
        "calibration_scope": "fixed 500-patient adaptation role; fixed assessment role only",
        "confirmation_allowed": False,
        "cross_scope_patient_independence_verified": False,
        "clinical_input_availability_verified": False,
    }
    return role_tables, frozen_fm, provenance


def _score_frozen_baselines(role_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    import conventional_baselines as conventional

    source_manifest = _json(SOURCE_REPRO)
    bundle = pickle.loads(SOURCE_BUNDLE.read_bytes())
    if bundle.get("status") != "source_reproduction_passed" or set(bundle.get("models", {})) != set(BASELINE_ARMS):
        _fail("frozen source model bundle does not contain the exact four reproduced baselines")
    if int(bundle.get("n_variables", 0)) < 1:
        _fail("frozen source baseline bundle is missing its source feature dimension")
    engine = _engine_module()
    rows: list[pd.DataFrame] = []
    for db in COHORTS:
        eligible, entry = engine.load_ready_eligibility(db, GATE)
        if entry.get("retrospective_landmark_prediction_allowed") is not True:
            _fail(f"{db}: locked exploratory gate no longer permits measurement-time scoring")
        paths = engine.resolve_cohort_paths(db, verify_exists=True)
        sequence_sha = sha256_file(Path(paths.sequence_path))
        engine.assert_sequence_npz_attestation(db, entry, sequence_sha)
        ds = engine.load_sequence_registered(db, paths=paths)
        target_rows = engine.condition_rows(ds, "real", ())
        matched = engine.align_eligible_rows(db, ds, eligible, target_rows)
        if len(matched) != len(role_tables[db]):
            _fail(f"{db}: registered sequence rows differ from fixed role-table count")
        role = role_tables[db][["db", "stay_id", "subject", "y_mort", "split_role"]].copy()
        matched["stay_id"] = matched["stay_id"].astype(str)
        matched["subject"] = matched["subject"].astype(str)
        check = role.merge(
            matched[["db", "stay_id", "subject", "y_mort", "rows"]],
            on=["db", "stay_id"], how="outer", validate="one_to_one", indicator=True,
            suffixes=("_role", "_seq"))
        if check["_merge"].ne("both").any():
            _fail(f"{db}: registered sequence IDs differ from locked roles")
        for col in ("subject", "y_mort"):
            if not check[f"{col}_role"].astype(str).equals(check[f"{col}_seq"].astype(str)):
                _fail(f"{db}: sequence {col} differs from fixed role map")
        positions = check["rows"].to_numpy(dtype=np.int64)
        if len(np.unique(positions)) != len(positions):
            _fail(f"{db}: sequence row positions are not unique")
        value, pattern = _summarize_conditioned_inputs(
            ds, positions, int(bundle["n_variables"]), condition="real",
            variable_ids=(), engine=engine)
        features = {"value_plus_pattern": value, "pattern_only": pattern}
        for analysis_id in BASELINE_ARMS:
            model_spec = bundle["models"][analysis_id]
            X = features[model_spec["feature_type"]]
            if model_spec["scaler"] is not None:
                X = model_spec["scaler"].transform(X)
            estimator = model_spec["estimator"]
            if not hasattr(estimator, "predict_proba"):
                _fail(f"{analysis_id}: frozen model lacks predict_proba")
            p = np.asarray(estimator.predict_proba(X)[:, 1], dtype=float)
            if p.shape != (len(check),) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                _fail(f"{db}/{analysis_id}: frozen baseline predictions are invalid")
            frame = check[["db", "stay_id", "subject_role", "y_mort_role", "split_role", "rows"]].copy()
            frame = frame.rename(columns={"subject_role": "subject", "y_mort_role": "y"})
            frame["analysis_id"] = analysis_id
            frame["p"] = p
            frame["model_tag"] = str(model_spec["model_tag"])
            frame["feature_type"] = str(model_spec["feature_type"])
            frame["role"] = "evaluation"
            frame["model_source"] = "frozen MIMIC split-0 source bundle; no target/source fit in this scoring run"
            frame["source_model_bundle_sha256"] = source_manifest["source_model_bundle_sha256"]
            frame["input_scope"] = "retrospective_measurement_time"
            frame["clinical_input_availability_verified"] = False
            frame["source_fit_calls_in_target_scoring_run"] = 0
            rows.append(frame)
            print(json.dumps({"stage": "frozen_baseline_target_scores_complete",
                              "db": db, "analysis_id": analysis_id,
                              "rows": int(len(frame)), "source_fit_calls": 0}, sort_keys=True), flush=True)
    result = pd.concat(rows, ignore_index=True)
    for db in COHORTS:
        target_ids = set(role_tables[db]["stay_id"].astype(str))
        for arm in BASELINE_ARMS:
            got = result.loc[(result.db == db) & (result.analysis_id == arm)]
            if len(got) != len(target_ids) or set(got.stay_id.astype(str)) != target_ids:
                _fail(f"{db}/{arm}: scored baseline identity set differs from fixed role map")
    return result


def _fit_platt(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=np.int8)
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("fixed adaptation pool is single-class")
    lp = logit(np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)).reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = LogisticRegression(C=PLATT_C, max_iter=2000, tol=1e-8, solver="lbfgs")
        model.fit(lp, y)
    a, b = float(model.coef_[0, 0]), float(model.intercept_[0])
    if not np.isfinite(a + b):
        raise ValueError("nonfinite Platt coefficients")
    return a, b


def _loss_matrix(y: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, dtype=float)[:, None]
    return np.stack(((y - p) ** 2,
                     -(y * np.log(p) + (1 - y) * np.log1p(-p))), axis=2)


def _point_auc_ap(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    if np.unique(y).size != 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


def _calibrate_and_summarize(
    role_tables: dict[str, pd.DataFrame], all_predictions: pd.DataFrame,
    *, n_boot: int = N_BOOT, seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if n_boot < 1000:
        raise ValueError("INSPIRE conditional patient bootstrap requires at least 1000 draws")
    summary: list[dict[str, Any]] = []
    fit_audit: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for db_index, db in enumerate(COHORTS):
        pred = all_predictions.loc[all_predictions.db.astype(str).eq(db)].copy()
        # Exact common rows are checked before calibration; the role map is the sole source of split labels.
        role_ref = role_tables[db][["db", "stay_id", "subject", "y_mort", "split_role"]].copy()
        arm_frames: dict[str, pd.DataFrame] = {}
        for arm in ALL_ARMS:
            arm_frame = pred.loc[pred.analysis_id.astype(str).eq(arm)].copy()
            arm_frame["stay_id"] = arm_frame["stay_id"].astype(str)
            arm_frame["subject"] = arm_frame["subject"].astype(str)
            if len(arm_frame) != len(role_ref) or arm_frame["stay_id"].duplicated().any():
                _fail(f"{db}/{arm}: arm does not cover one exact row per fixed-role stay")
            joined = role_ref.merge(arm_frame[["db", "stay_id", "subject", "y", "p"]],
                                   on=["db", "stay_id"], how="outer", validate="one_to_one",
                                   indicator=True, suffixes=("_role", "_pred"))
            if joined["_merge"].ne("both").any():
                _fail(f"{db}/{arm}: arm IDs differ from fixed role map")
            if (not joined["subject_role"].astype(str).equals(joined["subject_pred"].astype(str))
                    or not np.array_equal(joined["y_mort"].to_numpy(dtype=int), joined["y"].to_numpy(dtype=int))):
                _fail(f"{db}/{arm}: arm subject or label differs from fixed role map")
            if joined["subject_role"].duplicated().any():
                _fail(f"{db}/{arm}: this locked scope must contain one index episode per patient")
            arm_frames[arm] = joined.sort_values("stay_id", kind="stable").reset_index(drop=True)
        adaptation = role_ref.loc[role_ref.split_role.eq("adaptation")].copy()
        assessment = role_ref.loc[role_ref.split_role.eq("assessment")].copy()
        adapt_ids = set(adaptation.stay_id.astype(str))
        assess_ids = set(assessment.stay_id.astype(str))
        if adapt_ids & assess_ids:
            _fail(f"{db}: fixed adaptation and assessment stay roles overlap")
        base = arm_frames[ALL_ARMS[0]]
        order = base["stay_id"].astype(str).tolist()
        y_all = base["y_mort"].to_numpy(dtype=np.int8)
        role_all = base["split_role"].astype(str).to_numpy()
        adapt_mask, assess_mask = role_all == "adaptation", role_all == "assessment"
        if (int(adapt_mask.sum()) != EXPECTED_COUNTS[db]["adaptation_patients"]
                or int(assess_mask.sum()) != EXPECTED_COUNTS[db]["assessment_patients"]):
            _fail(f"{db}: actual role masks changed after preflight")
        adapt_y, assess_y = y_all[adapt_mask], y_all[assess_mask]
        assess_subjects = base.loc[assess_mask, "subject_role"].astype(str).to_numpy()
        q_jeffreys = float((int(adapt_y.sum()) + 0.5) / (len(adapt_y) + 1.0))
        methods = ("raw", "fixed500_platt", "jeffreys_prevalence")
        prob_columns: list[np.ndarray] = []
        model_method_keys: list[tuple[str, str]] = []
        for arm in ALL_ARMS:
            joined = arm_frames[arm]
            if joined["stay_id"].astype(str).tolist() != order:
                _fail(f"{db}/{arm}: internal alignment order differs across arms")
            raw = joined["p"].to_numpy(dtype=float)
            if not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 1)):
                _fail(f"{db}/{arm}: probability values are invalid")
            try:
                a, b = _fit_platt(adapt_y, raw[adapt_mask])
                p_platt = expit(a * logit(np.clip(raw, 1e-6, 1 - 1e-6)) + b)
                fit_status = "fitted_on_fixed_500_adaptation_only"
                fallback = False
            except Exception as exc:
                a, b = float("nan"), float("nan")
                p_platt = np.full(len(raw), q_jeffreys, dtype=float)
                fit_status = f"failed_fallback_jeffreys:{type(exc).__name__}:{str(exc)[:120]}"
                fallback = True
            fit_audit.append({
                "db": db, "analysis_id": arm, "adaptation_n": int(adapt_mask.sum()),
                "adaptation_events": int(adapt_y.sum()), "platt_status": fit_status,
                "platt_intercept": b, "platt_slope": a,
                "platt_C": PLATT_C, "fallback_used": fallback,
                "fallback_probability": q_jeffreys if fallback else "",
                "adaptation_role_map_sha256": EXPECTED_ROLE_LOCK_SHA,
            })
            values = {
                "raw": raw[assess_mask],
                "fixed500_platt": p_platt[assess_mask],
                "jeffreys_prevalence": np.full(int(assess_mask.sum()), q_jeffreys, dtype=float),
            }
            for method in methods:
                p_assess = values[method]
                if not np.isfinite(p_assess).all() or np.any((p_assess < 0) | (p_assess > 1)):
                    _fail(f"{db}/{arm}/{method}: invalid assessment probabilities")
                prob_columns.append(p_assess)
                model_method_keys.append((arm, method))

        P = np.column_stack(prob_columns)
        assessment_losses = _loss_matrix(assess_y, P)
        point_loss = assessment_losses.mean(axis=0)
        boot_losses = np.empty((n_boot, len(model_method_keys), 2), dtype=float)
        rng = np.random.default_rng(seed + db_index)
        n_assess = len(assess_y)
        for draw_i in range(n_boot):
            sampled = rng.integers(0, n_assess, size=n_assess)
            weights = np.bincount(sampled, minlength=n_assess).astype(float)
            boot_losses[draw_i] = np.einsum("n,nkm->km", weights, assessment_losses) / float(n_assess)
            if draw_i == 0 or draw_i + 1 == n_boot or (draw_i + 1) % 250 == 0:
                print(json.dumps({"stage": "inspire_fixed_assessment_bootstrap_progress",
                                  "db": db, "draw": draw_i + 1, "draws": n_boot}, sort_keys=True), flush=True)
        lower = np.percentile(boot_losses, 2.5, axis=0)
        upper = np.percentile(boot_losses, 97.5, axis=0)
        loss_index = {key: i for i, key in enumerate(model_method_keys)}
        for i, (arm, method) in enumerate(model_method_keys):
            p_assess = P[:, i]
            auc, ap = _point_auc_ap(assess_y, p_assess)
            summary.append({
                "db": db, "analysis_id": arm, "calibration_method": method,
                "n_adaptation_patients": int(adapt_mask.sum()),
                "adaptation_events": int(adapt_y.sum()),
                "n_assessment_patients": int(assess_mask.sum()),
                "assessment_events": int(assess_y.sum()),
                "assessment_nonevents": int(len(assess_y) - assess_y.sum()),
                "jeffreys_adaptation_prevalence": q_jeffreys,
                "platt_status": next(row["platt_status"] for row in fit_audit
                                     if row["db"] == db and row["analysis_id"] == arm),
                "prevalence": float(assess_y.mean()),
                "auroc": auc, "average_precision": ap,
                "brier": float(point_loss[i, 0]), "brier_ci_low": float(lower[i, 0]),
                "brier_ci_high": float(upper[i, 0]),
                "logloss": float(point_loss[i, 1]), "logloss_ci_low": float(lower[i, 1]),
                "logloss_ci_high": float(upper[i, 1]),
                "bootstrap_draws": int(n_boot), "valid_bootstrap_draws": int(n_boot),
                "ci_scope": "assessment-patient bootstrap conditional on frozen scores and fixed adaptation Platt fit",
                "analysis_role": "exploratory_external_only",
                "confirmation_allowed": False,
                "clinical_input_availability_verified": False,
            })
        ref_arm = "original_ordinary_C1"
        for arm in ALL_ARMS:
            if arm == ref_arm:
                continue
            for method in ("raw", "fixed500_platt"):
                idx, ref_idx = loss_index[(arm, method)], loss_index[(ref_arm, method)]
                point_delta = point_loss[idx] - point_loss[ref_idx]
                boot_delta = boot_losses[:, idx, :] - boot_losses[:, ref_idx, :]
                ci = np.percentile(boot_delta, [2.5, 97.5], axis=0)
                for m, metric in enumerate(("brier", "logloss")):
                    paired.append({
                        "db": db, "comparison": f"{arm} minus {ref_arm}",
                        "delta_direction": f"{arm} - {ref_arm}",
                        "analysis_id": arm, "reference_analysis_id": ref_arm,
                        "calibration_method": method, "metric": metric,
                        "delta": float(point_delta[m]), "ci_low": float(ci[0, m]),
                        "ci_high": float(ci[1, m]), "bootstrap_draws": int(n_boot),
                        "valid_bootstrap_draws": int(n_boot),
                        "n_assessment_patients": int(n_assess),
                        "assessment_events": int(assess_y.sum()),
                        "assessment_nonevents": int(n_assess - assess_y.sum()),
                        "primary_contrast": bool(arm == "density_augmented"
                                                  and method == "fixed500_platt"),
                        "paired_unit": "same fixed assessment patients; shared bootstrap weights",
                        "ci_scope": "conditional on frozen scores and fixed adaptation Platt fits",
                        "analysis_role": "exploratory_external_only",
                        "confirmation_allowed": False,
                    })
        density_id = loss_index[("density_augmented", "fixed500_platt")]
        tuned_id = loss_index[("ordinary_tuned", "fixed500_platt")]
        density_tuned_delta = point_loss[density_id] - point_loss[tuned_id]
        density_tuned_boot = boot_losses[:, density_id, :] - boot_losses[:, tuned_id, :]
        density_tuned_ci = np.percentile(density_tuned_boot, [2.5, 97.5], axis=0)
        for m, metric in enumerate(("brier", "logloss")):
            paired.append({
                "db": db,
                "comparison": "density_augmented fixed500_platt minus ordinary_tuned fixed500_platt",
                "delta_direction": "density_augmented fixed500_platt - ordinary_tuned fixed500_platt",
                "analysis_id": "density_augmented", "reference_analysis_id": "ordinary_tuned",
                "calibration_method": "fixed500_platt", "metric": metric,
                "delta": float(density_tuned_delta[m]), "ci_low": float(density_tuned_ci[0, m]),
                "ci_high": float(density_tuned_ci[1, m]), "bootstrap_draws": int(n_boot),
                "valid_bootstrap_draws": int(n_boot),
                "n_assessment_patients": int(n_assess),
                "assessment_events": int(assess_y.sum()),
                "assessment_nonevents": int(n_assess - assess_y.sum()),
                "primary_contrast": True,
                "paired_unit": "same fixed assessment patients; shared bootstrap weights",
                "ci_scope": "conditional on frozen scores and fixed adaptation Platt fits",
                "analysis_role": "exploratory_external_only", "confirmation_allowed": False,
            })
        print(json.dumps({"stage": "inspire_fixed500_cohort_complete", "db": db,
                          "arms": len(ALL_ARMS), "assessment_patients": int(n_assess),
                          "bootstrap_draws": int(n_boot), "jeffreys_prevalence": q_jeffreys,
                          "primary_contrasts": [
                              "density_augmented fixed500_platt minus original_ordinary_C1 fixed500_platt",
                              "density_augmented fixed500_platt minus ordinary_tuned fixed500_platt",
                          ]}, sort_keys=True), flush=True)
    return pd.DataFrame(summary), pd.DataFrame(paired), pd.DataFrame(fit_audit)


def run(out_dir: Path, *, preflight_only: bool = False, n_boot: int = N_BOOT) -> dict[str, Any]:
    role_tables, frozen_fm, provenance = _verify_locked_inputs()
    print(json.dumps({"stage": "inspire_fixed500_seven_arm_preflight_complete",
                      "cohorts": provenance["fixed_role_counts"],
                      "fm_heads": list(FM_ARMS), "baseline_arms_to_score_without_fitting": list(BASELINE_ARMS),
                      "no_fit_reuse_runs": provenance["no_fit_engine_reuse_runs"],
                      "output_rows_frozen_density": int(len(frozen_fm))}, sort_keys=True), flush=True)
    if preflight_only:
        return {"status": "preflight_passed", **provenance}
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite INSPIRE calibration output: {out_dir}")
    baseline_scores = _score_frozen_baselines(role_tables)
    output = []
    for db in COHORTS:
        fm = frozen_fm.loc[frozen_fm.db.astype(str).eq(db)].copy()
        baselines = baseline_scores.loc[baseline_scores.db.astype(str).eq(db)].copy()
        role = role_tables[db][["db", "stay_id", "subject", "y_mort", "split_role"]].copy()
        fm = fm.drop(columns=[c for c in ("split_role",) if c in fm.columns])
        fm = fm.merge(role[["db", "stay_id", "split_role"]], on=["db", "stay_id"],
                      how="left", validate="many_to_one")
        if fm.split_role.isna().any():
            _fail(f"{db}: density scores failed to pick up frozen role labels")
        if "source_probe_refitted" not in fm.columns:
            fm["source_probe_refitted"] = False
        if fm["source_probe_refitted"].astype(bool).any():
            _fail(f"{db}: frozen density scores claim source probe refitting")
        fm["model_source"] = "frozen source readout from density study; no target fit"
        fm["source_fit_calls_in_target_scoring_run"] = 0
        output.extend([fm, baselines])
    all_predictions = pd.concat(output, ignore_index=True, sort=False)
    metrics, paired, fit_audit = _calibrate_and_summarize(role_tables, all_predictions, n_boot=n_boot)
    pred_dir = out_dir / "predictions"
    _write_frame(pred_dir / "inspire_fixed500_seven_arm_predictions.parquet", all_predictions, parquet=True)
    _write_frame(out_dir / "inspire_fixed500_metrics.csv", metrics)
    _write_frame(out_dir / "inspire_fixed500_paired_loss_ci.csv", paired)
    _write_frame(out_dir / "inspire_fixed500_platt_fit_audit.csv", fit_audit)
    files = [pred_dir / "inspire_fixed500_seven_arm_predictions.parquet",
             out_dir / "inspire_fixed500_metrics.csv",
             out_dir / "inspire_fixed500_paired_loss_ci.csv",
             out_dir / "inspire_fixed500_platt_fit_audit.csv"]
    manifest = {
        "schema": "ehr_fm_inspire_fixed500_paired_calibration_v1",
        "status": "complete",
        "analysis_role": "exploratory_external_only",
        "confirmation_allowed": False,
        "pragmatic_confirmation_min_events": 200,
        "fixed_role_seed": 20261122,
        "fixed_adaptation_budget": 500,
        "roles_redrawn": False,
        "adaptation_role_selection_performance_blind": True,
        "assessment_only_metrics": True,
        "cohorts": list(COHORTS),
        "analysis_ids": list(ALL_ARMS),
        "methods": ["raw", "fixed500_platt", "jeffreys_prevalence"],
        "platt_fit": {"training_role": "fixed 500-patient adaptation role only",
                      "predictor": "logit(clipped frozen probability)", "C": PLATT_C,
                      "failed_fit_fallback": "Jeffreys adaptation prevalence (events+0.5)/(n+1)"},
        "uncertainty": {"draws": int(n_boot), "unit": "within-scope patient",
                        "type": "conditional percentile bootstrap on fixed assessment predictions and fixed adaptation Platt fits",
                        "shared_patient_weights_across_arms_and_methods": True,
                        "paired_contrasts": "each arm versus original_ordinary_C1 under raw and fixed500_platt; primary contrasts are density_augmented versus original_ordinary_C1 and ordinary_tuned under fixed500_platt"},
        "source_model_fit_calls_in_target_scoring_job": 0,
        "source_probe_refit_calls_for_density_heads": 0,
        "target_fit_calls": int(len(fit_audit)),
        "target_fit_scope": "Platt calibrators only; one per arm and cohort, fit on fixed adaptation labels",
        "cross_scope_patient_independence_verified": False,
        "clinical_input_availability_verified": False,
        "patient_level_exported_to_local": False,
        "provenance": provenance,
        "output_files": {str(p.relative_to(out_dir)): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                         for p in files},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(out_dir / "inspire_fixed500_calibration_manifest.json", manifest)
    print(json.dumps({"stage": "inspire_fixed500_calibration_complete",
                      "manifest": str(out_dir / "inspire_fixed500_calibration_manifest.json"),
                      "metrics_rows": int(len(metrics)), "paired_rows": int(len(paired)),
                      "fit_audit_rows": int(len(fit_audit)), "output_sha256": sha256_file(
                          out_dir / "inspire_fixed500_calibration_manifest.json")}, sort_keys=True), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=N_BOOT)
    args = parser.parse_args()
    result = run(args.out_dir, preflight_only=args.preflight_only, n_boot=args.bootstrap)
    if args.preflight_only:
        print("INSPIRE_FIXED500_PREFLIGHT_PASSED", flush=True)
    else:
        print("INSPIRE_FIXED500_CALIBRATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
