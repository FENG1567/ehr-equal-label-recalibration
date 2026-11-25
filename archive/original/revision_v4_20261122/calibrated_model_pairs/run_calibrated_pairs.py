"""Direct calibrated FM-minus-conventional loss contrasts on frozen plans.

Patient-level inputs and the per-stay loss archive are intended to remain on the
analysis server. This script replays only offset and Platt fits over saved,
frozen prediction probabilities and validates parity with retained per-draw
outputs before writing inferential summaries.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
OUT = REV / "calibrated_model_pairs" / "results" / "direct_calibrated_fm_baseline_v1"
SEED = 20261122
BUDGETS = (50, 200, 500, 2000)
REPEATS = 200
N_BOOT = 1000
BOOTSTRAP_BATCH = 16
MODEL_IDS = (
    "main20k_real",
    "conventional_lr_pattern_only",
    "conventional_hgb_pattern_only",
    "conventional_lr_value_plus_pattern",
    "conventional_hgb_value_plus_pattern",
)
BASELINE_IDS = MODEL_IDS[1:]
METHODS = ("raw", "offset_operational", "platt_operational")
METRICS = ("brier", "logloss")
SOURCE_OUT = REV / "robustness/results/primary_status_consistent_shared_calibration_20261123"
TRANSPORT_OUT = REV / "robustness/results/recovered_transport_v4_2_20261123_reuse_r2"
ORIGINAL_FM_RUN = REV / "engine/results/prediction_runs/status_consistent_primary_core_fix2-7da980ffcbd8f00b"
ORIGINAL_GATE = REV / "temporal/results/eligibility/status_consistent_v4_1/landmark_v4_1"
SOURCE_BASELINE_RUN = REV / "baselines/results/primary_status_consistent_20261123/results"

sys.path.insert(0, str(REV / "robustness"))
sys.path.insert(0, str(REV / "robustness/transport_v4_2"))
sys.path.insert(0, str(REV / "baselines/transport_v4_2"))

import robustness_common as rc  # noqa: E402
import recalibration as recal  # noqa: E402
import transport_analysis as transport  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _same_identity_rows(reference: pd.DataFrame, candidate: pd.DataFrame, label: str) -> None:
    fields = ["db", "stay_id", "subject", "rows", "split", "y"]
    if not reference[fields].astype(str).reset_index(drop=True).equals(
            candidate[fields].astype(str).reset_index(drop=True)):
        raise ValueError(f"{label}: row-wise cohort/stay/patient/label identity differs")


def _load_source_mimic() -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Strictly load frozen source predictions from the original current manifests."""
    fm_manifest = ORIGINAL_FM_RUN / "prediction_manifest.json"
    fm_path = ORIGINAL_FM_RUN / "main20k_real.parquet"
    baseline_manifest = SOURCE_BASELINE_RUN / "baseline_prediction_manifest.json"
    fm, _ = rc.load_checked(
        fm_path, ORIGINAL_GATE, model_tag="main20k", analysis_id="main20k_real",
        input_condition="real", endpoint="hospital", prediction_manifest_path=fm_manifest,
    )
    if "mimic" not in fm:
        raise ValueError("frozen source FM artifact lacks MIMIC")
    frames = {"main20k_real": fm["mimic"].copy()}
    manifest = _json(baseline_manifest)
    if manifest.get("status") != "complete":
        raise ValueError("source conventional prediction manifest is not complete")
    for model_id in BASELINE_IDS:
        records = [row for row in manifest.get("predictions", [])
                   if row.get("analysis_id") == model_id and row.get("db") == "mimic"]
        if len(records) != 1:
            raise ValueError(f"source baseline manifest has {len(records)} {model_id}/mimic rows")
        record = records[0]
        prediction_path = Path(record["prediction_file"])
        baseline_frames, _ = rc.load_checked(
            prediction_path, ORIGINAL_GATE,
            model_tag=transport.MODEL_TAGS[model_id], analysis_id=model_id,
            input_condition="real", endpoint="hospital", prediction_manifest_path=baseline_manifest,
        )
        if "mimic" not in baseline_frames:
            raise ValueError(f"frozen source baseline artifact lacks MIMIC: {model_id}")
        frames[model_id] = baseline_frames["mimic"].copy()
    for model_id in MODEL_IDS[1:]:
        _same_identity_rows(frames["main20k_real"], frames[model_id], f"source {model_id}")
    inputs = {
        "fm_prediction_manifest_sha256": _sha256(fm_manifest),
        "fm_prediction_file_sha256": _sha256(fm_path),
        "baseline_prediction_manifest_sha256": _sha256(baseline_manifest),
    }
    return frames, inputs


def _target_frames() -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, str]]:
    frames, gates, _, provenance = transport._load_all_model_frames()
    if set(frames) != set(MODEL_IDS):
        raise ValueError(f"target frame model coverage differs: {sorted(frames)}")
    for db in transport.TARGET_COHORTS:
        if gates[db].get("bootstrap_cluster_unit") != "patient" or not gates[db].get(
                "independent_patient_clustering_verified"):
            raise ValueError(f"{db}: patient-cluster bootstrap is not attested")
        for model_id in BASELINE_IDS:
            _same_identity_rows(frames["main20k_real"][db], frames[model_id][db],
                                f"{db}/{model_id}")
    plan_path = TRANSPORT_OUT / "sampling_plan_manifest.json"
    analysis_path = TRANSPORT_OUT / "transport_analysis_manifest.json"
    if not plan_path.is_file() or not analysis_path.is_file():
        raise FileNotFoundError("repaired-r2 transport plan/aggregate manifest is absent")
    plan_doc = _json(plan_path)
    plan_unhashed = dict(plan_doc)
    recorded_plan_manifest_hash = plan_unhashed.pop("plan_manifest_sha256", None)
    canonical = json.dumps(plan_unhashed, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False).encode("utf-8")
    if (plan_doc.get("status") != "complete"
            or hashlib.sha256(canonical).hexdigest() != recorded_plan_manifest_hash):
        raise ValueError("repaired-r2 frozen sampling-plan manifest is incomplete or hash-invalid")
    if plan_doc.get("provenance") != provenance:
        raise ValueError("repaired-r2 plan provenance differs from currently loaded frozen predictions")
    if (int(plan_doc.get("seed", -1)) != SEED
            or int(plan_doc.get("repeats", -1)) != REPEATS
            or tuple(plan_doc.get("budgets", ())) != BUDGETS
            or set(plan_doc.get("model_ids", ())) != set(MODEL_IDS)):
        raise ValueError("repaired-r2 plan configuration/model roster differs")
    outputs = {
        "fm_prediction_manifest_sha256": provenance["fm_prediction_manifest_sha256"],
        "baseline_prediction_manifest_sha256": provenance["baseline_prediction_manifest_sha256"],
        "subject_identity_attestation_sha256": provenance["subject_identity_attestation_sha256"],
        "sampling_plan_manifest_sha256": _sha256(plan_path),
        "transport_analysis_manifest_sha256": _sha256(analysis_path),
    }
    return frames, outputs


def _assert_plan_matches(frame: pd.DataFrame, db: str,
                         expected: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame,
                                                           dict[int, list[np.ndarray]],
                                                           dict[str, Any]]:
    usable = frame.loc[frame["eligible_24h"].astype(bool)
                       & frame["temporal_verified"].astype(bool)].copy()
    adaptation, assessment, _ = recal.adaptation_assessment_pools(usable, SEED)
    plan, draw_indices = recal._make_sampling_plan(
        usable, adaptation, assessment, repeats=REPEATS, seed=SEED, budgets=BUDGETS,
    )
    fields = ("fixed_population_identity_sha256", "adaptation_index_patient_stay_sha256",
              "adaptation_patient_set_sha256", "assessment_population_identity_sha256",
              "assessment_patient_set_sha256", "sampling_plan_sha256")
    mismatches = [field for field in fields if plan.get(field) != expected.get(field)]
    if mismatches:
        raise ValueError(f"{db}: exact frozen plan mismatch in {mismatches}")
    if "draws" in expected:
        if plan.get("draws") != expected.get("draws"):
            raise ValueError(f"{db}: actual frozen adaptation/index-stay hashes differ")
    if int(plan.get("repeats", -1)) != REPEATS or tuple(plan.get("budgets", ())) != BUDGETS:
        raise ValueError(f"{db}: draw plan repeats/budgets differ")
    return usable, assessment, draw_indices, plan


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if str(value).strip().lower() in {"true", "1", "yes"}:
        return True
    if str(value).strip().lower() in {"false", "0", "no"}:
        return False
    raise ValueError(f"unrecognized boolean cell: {value!r}")


def _losses(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probability = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    label = np.asarray(y, dtype=float)
    return (label - probability) ** 2, -(label * np.log(probability)
                                           + (1 - label) * np.log(1 - probability))


def _fit_operational(arm: str, draw_y: np.ndarray, draw_lg: np.ndarray,
                     assessment_lg: np.ndarray, budget: int) -> dict[str, Any]:
    """Replay one calibration fit and its frozen Jeffreys operational fallback."""
    smooth_prev = float((int(np.asarray(draw_y).sum()) + 0.5) / (budget + 1.0))
    fit_fn = recal.fit_offset if arm == "offset" else recal.fit_platt
    try:
        a, b = fit_fn(draw_y, draw_lg)
        p = np.clip(recal.expit(a * assessment_lg + b), 1e-6, 1 - 1e-6)
        return {"fit_ok": True, "fit_status": "ok", "slope": float(a),
                "intercept": float(b), "fallback_probability": np.nan, "p": p}
    except (ValueError, RuntimeError, FloatingPointError, recal.ConvergenceWarning) as exc:
        return {"fit_ok": False,
                "fit_status": f"{type(exc).__name__}:{str(exc)[:100]}",
                "slope": np.nan, "intercept": np.nan,
                "fallback_probability": smooth_prev,
                "p": np.full(len(assessment_lg), smooth_prev, dtype=float)}


def _expected_group(draws: pd.DataFrame, db: str, model_id: str) -> pd.DataFrame:
    if "analysis_id" in draws.columns:
        selected = draws.loc[draws["analysis_id"].astype(str).eq(model_id)
                             & draws["db"].astype(str).eq(db)].copy()
    else:
        selected = draws.loc[draws["db"].astype(str).eq(db)].copy()
    if selected.empty:
        raise ValueError(f"existing draw table has no rows for {db}/{model_id}")
    return selected


def _verify_plan_draw_rows(expected: pd.DataFrame, plan: dict[str, Any], db: str,
                           model_id: str) -> dict[tuple[int, int], dict[str, Any]]:
    selected = expected.loc[expected["method"].astype(str).eq("raw")].copy()
    wanted = {(int(row["budget"]), int(row["repeat"])):
              (str(row["draw_index_sha256"]), str(row["patient_index_stay_sha256"]))
              for row in plan["draws"]}
    got: dict[tuple[int, int], dict[str, Any]] = {}
    for row in selected.to_dict("records"):
        key = (int(row["budget"]), int(row["repeat"]))
        values = (str(row["adaptation_sample_index_sha256"]),
                  str(row["adaptation_sample_patient_index_stay_sha256"]))
        if key in got:
            if (got[key]["adaptation_sample_index_sha256"],
                    got[key]["adaptation_sample_patient_index_stay_sha256"]) != values:
                raise ValueError(f"{db}/{model_id}: conflicted existing repeat hashes {key}")
        got[key] = {
            "adaptation_sample_index_sha256": values[0],
            "adaptation_sample_patient_index_stay_sha256": values[1],
        }
    if set(got) != set(wanted):
        raise ValueError(f"{db}/{model_id}: draw repeat key set differs ({len(got)} vs {len(wanted)})")
    for key, value in got.items():
        if (value["adaptation_sample_index_sha256"],
                value["adaptation_sample_patient_index_stay_sha256"]) != wanted[key]:
            raise ValueError(f"{db}/{model_id}: retained adaptation hash differs at {key}")
    return got


def _validate_existing_draw_grid(expected: pd.DataFrame, plan: dict[str, Any],
                                 db: str, model_id: str) -> dict[str, int]:
    """Check complete repeat keys, hashes, fit/fallback states and conditional rows."""
    draw_hashes = _verify_plan_draw_rows(expected, plan, db, model_id)
    if not expected["sampling_plan_sha256"].astype(str).eq(plan["sampling_plan_sha256"]).all():
        raise ValueError(f"{db}/{model_id}: retained output contains a different sampling-plan SHA")
    count = {"raw_rows": 0, "operational_rows": 0, "conditional_rows": 0,
             "fit_status_cells": 0, "draw_hash_cells": 0}
    for budget in BUDGETS:
        for repeat in range(REPEATS):
            raw = expected.loc[expected["budget"].astype(int).eq(budget)
                               & expected["repeat"].astype(int).eq(repeat)
                               & expected["method"].astype(str).eq("raw")]
            if len(raw) != 1:
                raise ValueError(f"{db}/{model_id}/{budget}/{repeat}/raw: retained count={len(raw)}")
            raw_row = raw.iloc[0]
            draw_key = (budget, repeat)
            if (str(raw_row["adaptation_sample_index_sha256"])
                    != draw_hashes[draw_key]["adaptation_sample_index_sha256"]
                    or str(raw_row["adaptation_sample_patient_index_stay_sha256"])
                    != draw_hashes[draw_key]["adaptation_sample_patient_index_stay_sha256"]):
                raise ValueError(f"{db}/{model_id}/{budget}/{repeat}: raw row hash conflict")
            if (not _as_bool(raw_row["fit_ok"])
                    or str(raw_row["fit_status"]) != "not_applicable"
                    or _as_bool(raw_row["is_operational_fallback"])):
                raise ValueError(f"{db}/{model_id}/{budget}/{repeat}/raw: raw fit state is invalid")
            count["raw_rows"] += 1
            count["draw_hash_cells"] += 2
            for arm in ("offset", "platt"):
                operational = expected.loc[
                    expected["budget"].astype(int).eq(budget)
                    & expected["repeat"].astype(int).eq(repeat)
                    & expected["method"].astype(str).eq(f"{arm}_operational")
                ]
                conditional = expected.loc[
                    expected["budget"].astype(int).eq(budget)
                    & expected["repeat"].astype(int).eq(repeat)
                    & expected["method"].astype(str).eq(f"{arm}_conditional")
                ]
                if len(operational) != 1:
                    raise ValueError(f"{db}/{model_id}/{budget}/{repeat}/{arm}_operational: retained count={len(operational)}")
                op = operational.iloc[0]
                fit_ok = _as_bool(op["fit_ok"])
                fit_status = str(op["fit_status"])
                expected_fallback = not fit_ok
                if _as_bool(op["is_operational_fallback"]) != expected_fallback:
                    raise ValueError(f"{db}/{model_id}/{budget}/{repeat}/{arm}: operational fallback flag conflicts with fit status")
                if not np.isfinite(float(op["brier"])) or not np.isfinite(float(op["logloss"])):
                    raise ValueError(f"{db}/{model_id}/{budget}/{repeat}/{arm}: operational score is nonfinite")
                if len(conditional) != int(fit_ok):
                    raise ValueError(f"{db}/{model_id}/{budget}/{repeat}/{arm}_conditional: row count disagrees with fit status")
                if fit_ok:
                    con = conditional.iloc[0]
                    if (_as_bool(con["fit_ok"]) is not True
                            or str(con["fit_status"]) != fit_status
                            or _as_bool(con["is_operational_fallback"])):
                        raise ValueError(f"{db}/{model_id}/{budget}/{repeat}/{arm}: conditional fit state conflicts")
                    if not np.isfinite(float(con["brier"])) or not np.isfinite(float(con["logloss"])):
                        raise ValueError(f"{db}/{model_id}/{budget}/{repeat}/{arm}: conditional score is nonfinite")
                    count["conditional_rows"] += 1
                for row in (op, *([conditional.iloc[0]] if fit_ok else [])):
                    if (str(row["adaptation_sample_index_sha256"])
                            != draw_hashes[draw_key]["adaptation_sample_index_sha256"]
                            or str(row["adaptation_sample_patient_index_stay_sha256"])
                            != draw_hashes[draw_key]["adaptation_sample_patient_index_stay_sha256"]):
                        raise ValueError(f"{db}/{model_id}/{budget}/{repeat}/{arm}: calibration hash differs from raw")
                    count["draw_hash_cells"] += 2
                count["operational_rows"] += 1
                count["fit_status_cells"] += 1 + int(fit_ok)
    if count["raw_rows"] != REPEATS * len(BUDGETS) or count["operational_rows"] != 2 * REPEATS * len(BUDGETS):
        raise AssertionError(f"{db}/{model_id}: unexpected retained grid counts: {count}")
    return count


def _preflight_all() -> dict[str, Any]:
    """Read-only all-cohort/model/plan and retained-draw identity preflight."""
    source_frames, _ = _load_source_mimic()
    source_plans, source_draws, source_hashes = _source_plan_and_draws()
    target_frames, target_inputs = _target_frames()
    target_plans, target_draws, target_hashes = _transport_plans_and_draws()
    entries: dict[str, Any] = {}
    cohorts = {"mimic": (source_frames, source_draws, source_plans["mimic"])}
    cohorts.update({db: ({model_id: target_frames[model_id][db] for model_id in MODEL_IDS},
                         target_draws, target_plans[db])
                    for db in transport.TARGET_COHORTS})
    expected_fields = [
        "db", "budget", "repeat", "method", "adaptation_sample_index_sha256",
        "adaptation_sample_patient_index_stay_sha256", "fixed_population_identity_sha256",
        "adaptation_index_patient_stay_sha256", "assessment_population_identity_sha256",
        "sampling_plan_sha256", "fit_ok", "fit_status", "is_operational_fallback", "brier", "logloss",
    ]
    for db, (frames, draws_by_model, expected_plan) in cohorts.items():
        usable, assessment, _, current_plan = _assert_plan_matches(frames["main20k_real"], db, expected_plan)
        cohort_result = {"sampling_plan_sha256": current_plan["sampling_plan_sha256"],
                         "fixed_population_identity_sha256": current_plan["fixed_population_identity_sha256"],
                         "adaptation_index_patient_stay_sha256": current_plan["adaptation_index_patient_stay_sha256"],
                         "assessment_population_identity_sha256": current_plan["assessment_population_identity_sha256"],
                         "n_eligible_stays": len(usable), "n_assessment_patients": int(assessment["subject"].nunique()),
                         "n_assessment_stays": len(assessment), "models": {}}
        for model_id in MODEL_IDS:
            _same_identity_rows(frames["main20k_real"], frames[model_id], f"preflight {db}/{model_id}")
            expected = _expected_group(draws_by_model[model_id], db, model_id)
            missing = sorted(set(expected_fields) - set(expected.columns))
            if missing:
                raise ValueError(f"{db}/{model_id}: retained draw table lacks {missing}")
            grid = _validate_existing_draw_grid(expected, current_plan, db, model_id)
            cohort_result["models"][model_id] = grid
        entries[db] = cohort_result
        print(json.dumps({"stage": "paired_calibration_preflight_cohort_complete", "db": db,
                          "models": list(cohort_result["models"]),
                          "sampling_plan_sha256": current_plan["sampling_plan_sha256"]},
                         sort_keys=True), flush=True)
    manifest = {
        "status": "preflight_complete", "schema": "ehr_fm_direct_calibrated_model_pairs_preflight_v1",
        "cohorts": entries, "expected_parity_fields": expected_fields,
        "source_artifacts": source_hashes, "target_artifacts": {**target_inputs, **target_hashes},
        "patient_level_exported_to_local": False,
    }
    path = OUT / "preflight_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(path)
    manifest["manifest_sha256"] = _sha256(path)
    print(json.dumps({"stage": "paired_calibration_preflight_complete", "manifest": str(path),
                      "manifest_sha256": manifest["manifest_sha256"]}, sort_keys=True), flush=True)
    return manifest


def _check_metric(expected: Any, actual: float, *, label: str) -> None:
    if pd.isna(expected):
        if np.isfinite(actual):
            raise ValueError(f"{label}: replay finite but retained metric is missing")
        return
    if not np.isfinite(actual) or not np.isclose(float(expected), actual, rtol=1e-9, atol=1e-11):
        raise ValueError(f"{label}: replay {actual:.17g} differs from retained {float(expected):.17g}")


def _cluster_bootstrap_contrasts(assessment: pd.DataFrame, contrasts: np.ndarray,
                                 *, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap stay-weighted paired means using common patient resamples."""
    groups = assessment["subject"].astype(str).to_numpy()
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2 or contrasts.shape[0] != len(groups) or not np.isfinite(contrasts).all():
        raise ValueError("cluster bootstrap needs finite stay contrasts and >=2 patient clusters")
    patient_counts = np.bincount(inverse, minlength=len(unique)).astype(float)
    patient_sums = np.vstack([
        np.bincount(inverse, weights=contrasts[:, j], minlength=len(unique))
        for j in range(contrasts.shape[1])
    ])
    point = contrasts.mean(axis=0)
    estimates = np.empty((N_BOOT, contrasts.shape[1]), dtype=float)
    rng = np.random.default_rng(seed)
    for start in range(0, N_BOOT, BOOTSTRAP_BATCH):
        size = min(BOOTSTRAP_BATCH, N_BOOT - start)
        sampled = rng.integers(0, len(unique), size=(size, len(unique)), dtype=np.int32)
        weights = np.stack([np.bincount(row, minlength=len(unique)) for row in sampled]).astype(float)
        denominator = weights @ patient_counts
        estimates[start:start + size] = (weights @ patient_sums.T) / denominator[:, None]
    low, high = np.percentile(estimates, [2.5, 97.5], axis=0)
    return point, low, high


def _replay_cohort(db: str, frames: dict[str, pd.DataFrame], expected_by_model: dict[str, pd.DataFrame],
                   plan_expected: dict[str, Any], *, out_cohort: Path,
                   cohort_index: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    usable, assessment, draw_indices, plan = _assert_plan_matches(
        frames["main20k_real"], db, plan_expected,
    )
    if len(frames) != len(MODEL_IDS):
        raise ValueError(f"{db}: missing frozen model frame")
    for model_id in MODEL_IDS[1:]:
        _same_identity_rows(frames["main20k_real"], frames[model_id], f"{db}/{model_id}")
    for model_id in MODEL_IDS:
        expected_hash = expected_by_model[model_id]
        expected_hash = expected_hash.loc[expected_hash["db"].astype(str).eq(db)]
        if expected_hash.empty or not expected_hash["sampling_plan_sha256"].astype(str).eq(
                plan["sampling_plan_sha256"]).all():
            raise ValueError(f"{db}/{model_id}: retained draw rows carry another plan SHA")
    expected_groups = {
        model_id: _expected_group(expected_by_model[model_id], db, model_id)
        for model_id in MODEL_IDS
    }
    draw_hashes = {
        model_id: _verify_plan_draw_rows(expected_groups[model_id], plan, db, model_id)
        for model_id in MODEL_IDS
    }
    y_assess = assessment["y"].to_numpy(dtype=int)
    subject = assessment["subject"].astype(str).to_numpy()
    stay_ids = assessment["stay_id"].astype(str).to_numpy()
    stay_identity_sha = recal._identity_digest(assessment)
    n_stays = len(assessment)
    loss_means: dict[tuple[str, int, str, str], np.ndarray] = {}
    fit_rows: list[dict[str, Any]] = []
    parity_counts = {"rows_checked": 0, "metric_cells_checked": 0, "fit_status_cells_checked": 0}
    operational_methods = ("offset_operational", "platt_operational")

    # Raw predictions are fixed by model and assessment stay, independent of budget.
    for model_id in MODEL_IDS:
        p_raw = np.clip(frames[model_id].set_index("stay_id").loc[stay_ids, "p"].to_numpy(float),
                        1e-6, 1 - 1e-6)
        raw_brier, raw_ll = _losses(y_assess, p_raw)
        for budget in BUDGETS:
            loss_means[(model_id, budget, "raw", "brier")] = raw_brier.copy()
            loss_means[(model_id, budget, "raw", "logloss")] = raw_ll.copy()
            expected = expected_groups[model_id]
            rows = expected.loc[expected["budget"].astype(int).eq(budget)
                                & expected["method"].astype(str).eq("raw")]
            if len(rows) != REPEATS:
                raise ValueError(f"{db}/{model_id}/{budget}/raw: expected {REPEATS} retained rows, found {len(rows)}")
            for row in rows.itertuples(index=False):
                _check_metric(row.brier, float(raw_brier.mean()), label=f"{db}/{model_id}/{budget}/{row.repeat}/raw/brier")
                _check_metric(row.logloss, float(raw_ll.mean()), label=f"{db}/{model_id}/{budget}/{row.repeat}/raw/logloss")
                parity_counts["metric_cells_checked"] += 2
                parity_counts["rows_checked"] += 1
                parity_counts["fit_status_cells_checked"] += 1
                if not _as_bool(row.fit_ok) or str(row.fit_status) != "not_applicable":
                    raise ValueError(f"{db}/{model_id}/{budget}/{row.repeat}/raw: retained fit state differs")

    for model_id in MODEL_IDS:
        frame = frames[model_id]
        frame_stay = frame.set_index("stay_id")
        p_assess = np.clip(frame_stay.loc[stay_ids, "p"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
        lg_assess = recal.logit(p_assess)
        usable_arm = frame.loc[frame["eligible_24h"].astype(bool)
                               & frame["temporal_verified"].astype(bool)].copy()
        adaptation, arm_assessment, _ = recal.adaptation_assessment_pools(usable_arm, SEED)
        _same_identity_rows(assessment, arm_assessment, f"{db}/{model_id}/assessment")
        adapt_y = adaptation["y"].to_numpy(dtype=int)
        adapt_lg = recal.logit(np.clip(adaptation["p"].to_numpy(dtype=float), 1e-6, 1 - 1e-6))
        expected = expected_groups[model_id]
        for budget in BUDGETS:
            idx_repeats = draw_indices[budget]
            for method in operational_methods:
                arm = method.split("_")[0]
                sums = {"brier": np.zeros(n_stays, dtype=float),
                        "logloss": np.zeros(n_stays, dtype=float)}
                n_success = 0
                for repeat, draw_idx in enumerate(idx_repeats):
                    draw_y = adapt_y[draw_idx]
                    events = int(draw_y.sum())
                    selected_lg = adapt_lg[draw_idx]
                    fit = _fit_operational(arm, draw_y, selected_lg, lg_assess, budget)
                    success, fit_status = fit["fit_ok"], fit["fit_status"]
                    a, b = fit["slope"], fit["intercept"]
                    p_eval = fit["p"]
                    fit_rows.append({
                        "db": db, "analysis_id": model_id, "budget": budget, "repeat": repeat,
                        "method": arm, "fit_ok": success, "fit_status": fit_status,
                        "slope": float(a) if success else np.nan,
                        "intercept": float(b) if success else np.nan,
                        "operational_fallback_probability": fit["fallback_probability"],
                        "events": events, "adaptation_sample_index_sha256":
                            draw_hashes[model_id][(budget, repeat)]["adaptation_sample_index_sha256"],
                        "adaptation_sample_patient_index_stay_sha256": draw_hashes[model_id][
                            (budget, repeat)]["adaptation_sample_patient_index_stay_sha256"],
                        "sampling_plan_sha256": plan["sampling_plan_sha256"],
                    })
                    brier_loss, ll_loss = _losses(y_assess, p_eval)
                    sums["brier"] += brier_loss
                    sums["logloss"] += ll_loss
                    n_success += int(success)
                    for suffix, actual in (("conditional", p_eval if success else None),
                                           ("operational", p_eval)):
                        retained_method = f"{arm}_{suffix}"
                        rows = expected.loc[expected["budget"].astype(int).eq(budget)
                                            & expected["repeat"].astype(int).eq(repeat)
                                            & expected["method"].astype(str).eq(retained_method)]
                        expected_n = 1 if suffix == "operational" or success else 0
                        if len(rows) != expected_n:
                            raise ValueError(f"{db}/{model_id}/{budget}/{repeat}/{retained_method}: retained row count={len(rows)}")
                        if expected_n == 0:
                            continue
                        row = rows.iloc[0]
                        fit_expected = _as_bool(row["fit_ok"])
                        if fit_expected != success or str(row["fit_status"]) != fit_status:
                            raise ValueError(f"{db}/{model_id}/{budget}/{repeat}/{retained_method}: replay fit state differs from retained")
                        fallback_expected = _as_bool(row["is_operational_fallback"])
                        fallback_actual = bool(suffix == "operational" and not success)
                        if fallback_expected != fallback_actual:
                            raise ValueError(f"{db}/{model_id}/{budget}/{repeat}/{retained_method}: fallback flag mismatch")
                        parity_counts["fit_status_cells_checked"] += 1
                        parity_counts["rows_checked"] += 1
                        for metric, actual_loss in (("brier", brier_loss), ("logloss", ll_loss)):
                            actual_metric = float(np.mean(actual_loss)) if actual is not None else float("nan")
                            _check_metric(row[metric], actual_metric,
                                          label=f"{db}/{model_id}/{budget}/{repeat}/{retained_method}/{metric}")
                            parity_counts["metric_cells_checked"] += 1
                for metric in METRICS:
                    loss_means[(model_id, budget, method, metric)] = sums[metric] / REPEATS
                print(json.dumps({"stage": "paired_calibration_replay_arm_budget_complete",
                                  "db": db, "analysis_id": model_id, "budget": budget,
                                  "method": arm, "fit_success": n_success,
                                  "fit_fail": REPEATS - n_success}, sort_keys=True), flush=True)

    # Form all planned FM-minus-baseline contrasts, then use one common set of
    # patient-cluster bootstrap resamples per cohort across every contrast.
    contrast_labels: list[dict[str, Any]] = []
    contrast_columns: list[np.ndarray] = []
    for baseline_id in BASELINE_IDS:
        for budget in BUDGETS:
            for method in METHODS:
                for metric in METRICS:
                    contrast_labels.append({"baseline_model_id": baseline_id, "budget": budget,
                                            "method": method, "metric": metric})
                    contrast_columns.append(
                        loss_means[("main20k_real", budget, method, metric)]
                        - loss_means[(baseline_id, budget, method, metric)]
                    )
    contrasts = np.column_stack(contrast_columns)
    seed = SEED + (10000 * (cohort_index + 1))
    points, lows, highs = _cluster_bootstrap_contrasts(assessment, contrasts, seed=seed)
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(contrast_labels):
        rows.append({
            "db": db, **label,
            "comparison_id": f"main20k_real_minus_{label['baseline_model_id']}",
            "estimand": "FM_minus_baseline_mean_assessment_stay_loss_after_within_stay_repeat_average",
            "delta_fm_minus_baseline": float(points[index]),
            "assessment_patient_cluster_ci_95_lo": float(lows[index]),
            "assessment_patient_cluster_ci_95_hi": float(highs[index]),
            "n_assessment_patients": int(assessment["subject"].nunique()),
            "n_assessment_stays": int(n_stays), "assessment_bootstrap_draws": N_BOOT,
            "bootstrap_cluster_unit": "patient",
            "adaptation_repeats": REPEATS,
            "operational_scope": "all 200 repeats; Jeffreys-smoothed adaptation prevalence after fit failure",
            "fit_success_n_fm": int(sum(row["fit_ok"] for row in fit_rows
                                          if row["analysis_id"] == "main20k_real"
                                          and row["budget"] == label["budget"]
                                          and row["method"] == label["method"].split("_")[0]))
                if label["method"] != "raw" else REPEATS,
            "fit_success_n_baseline": int(sum(row["fit_ok"] for row in fit_rows
                                               if row["analysis_id"] == label["baseline_model_id"]
                                               and row["budget"] == label["budget"]
                                               and row["method"] == label["method"].split("_")[0]))
                if label["method"] != "raw" else REPEATS,
            "within_cohort_assessment_identity_sha256": stay_identity_sha,
            "sampling_plan_sha256": plan["sampling_plan_sha256"],
            "uncertainty_scope": "frozen predictors/plan and realized adaptation fits; patient-cluster resampling of paired stay losses",
        })

    out_cohort.mkdir(parents=True, exist_ok=True)
    archive: dict[str, np.ndarray] = {
        "stay_id": stay_ids.astype("U"), "subject": subject.astype("U"),
        "y": y_assess.astype(np.int8),
    }
    for (model_id, budget, method, metric), vector in loss_means.items():
        safe = f"{model_id}__budget{budget}__{method}__{metric}"
        archive[safe] = np.asarray(vector, dtype=np.float32)
    per_stay_path = out_cohort / "server_only_per_stay_mean_losses.npz"
    np.savez_compressed(per_stay_path, **archive)
    fit_path = out_cohort / "server_only_fit_parameters.csv"
    pd.DataFrame(fit_rows).to_csv(fit_path, index=False)
    cohort_doc = {
        "status": "parity_verified", "db": db,
        "n_assessment_stays": int(n_stays),
        "n_assessment_patients": int(assessment["subject"].nunique()),
        "assessment_identity_sha256": stay_identity_sha,
        "sampling_plan_sha256": plan["sampling_plan_sha256"],
        "parity": parity_counts,
        "fit_parameter_rows": len(fit_rows),
        "per_stay_loss_archive_path": str(per_stay_path),
        "per_stay_loss_archive_sha256": _sha256(per_stay_path),
        "fit_parameters_path": str(fit_path), "fit_parameters_sha256": _sha256(fit_path),
        "direct_contrast_rows": len(rows),
        "bootstrap_seed": seed, "bootstrap_draws": N_BOOT,
        "patient_level_exported_to_local": False,
    }
    (out_cohort / "cohort_manifest.json").write_text(
        json.dumps(cohort_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rows, cohort_doc


def _source_plan_and_draws() -> tuple[dict[str, dict[str, Any]], dict[str, pd.DataFrame], dict[str, str]]:
    plan_path = SOURCE_OUT / "main20k_real" / "recalibration_run_manifest.json"
    expected_by_model: dict[str, pd.DataFrame] = {}
    manifests = []
    manifest_paths = []
    hashes: dict[str, str] = {}
    for model_id in MODEL_IDS:
        path = SOURCE_OUT / model_id / "recalibration_run_manifest.json"
        draws_path = SOURCE_OUT / model_id / "recalibration_draws.csv"
        if not path.is_file() or not draws_path.is_file():
            raise FileNotFoundError(f"source retained draw output absent for {model_id}")
        manifest = _json(path)
        if (manifest.get("status") != "complete" or int(manifest.get("seed", -1)) != SEED
                or int(manifest.get("repeats", -1)) != REPEATS
                or tuple(manifest.get("budgets", ())) != BUDGETS):
            raise ValueError(f"source {model_id} manifest is not complete/current")
        manifests.append(manifest)
        manifest_paths.append(path)
        expected_by_model[model_id] = pd.read_csv(draws_path, low_memory=False)
        hashes[f"{model_id}_draws_sha256"] = _sha256(draws_path)
        hashes[f"{model_id}_run_manifest_sha256"] = _sha256(path)
    shared = recal.verify_shared_recalibration_manifests(manifest_paths)
    if shared.get("status") != "verified" or "mimic" not in shared.get("cohorts", {}):
        raise ValueError("source MIMIC manifests do not attest one exact shared sampling plan")
    return {"mimic": manifests[0]["cohort_sampling_plans"]["mimic"]}, expected_by_model, hashes


def _transport_plans_and_draws() -> tuple[dict[str, dict[str, Any]], dict[str, pd.DataFrame], dict[str, str]]:
    old_out = REV / "robustness/results/recovered_transport_v4_2_20261123"
    old_plan_path = old_out / "sampling_plan_manifest.json"
    old_analysis_path = old_out / "transport_analysis_manifest.json"
    old_draw_path = old_out / "transport_calibration_draws.csv"
    r2_plan_path = TRANSPORT_OUT / "sampling_plan_manifest.json"
    r2_analysis_path = TRANSPORT_OUT / "transport_analysis_manifest.json"
    r2_draw_path = TRANSPORT_OUT / "transport_calibration_draws.csv"
    old_plan, r2_plan = _json(old_plan_path), _json(r2_plan_path)
    old_analysis, r2_analysis = _json(old_analysis_path), _json(r2_analysis_path)
    for label, path, document in (("NW/SIC", old_plan_path, old_plan),
                                  ("repaired-r2", r2_plan_path, r2_plan)):
        unhashed = dict(document)
        stored = unhashed.pop("plan_manifest_sha256", None)
        canonical = json.dumps(unhashed, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False).encode("utf-8")
        if (document.get("status") != "complete"
                or hashlib.sha256(canonical).hexdigest() != stored):
            raise ValueError(f"{label} sampling plan manifest is incomplete or hash-invalid")
    old_hash, r2_hash = _sha256(old_draw_path), _sha256(r2_draw_path)
    if old_analysis.get("aggregate_tables", {}).get("calibration_draws", {}).get("sha256") != old_hash:
        raise ValueError("NW/SIC aggregate calibration draw file differs from its own manifest")
    if r2_analysis.get("aggregate_tables", {}).get("calibration_draws", {}).get("sha256") != r2_hash:
        raise ValueError("repaired-r2 aggregate calibration draw file differs from its own manifest")
    old_draws = pd.read_csv(old_draw_path, low_memory=False)
    r2_draws = pd.read_csv(r2_draw_path, low_memory=False)
    for label, draws in (("NW/SIC", old_draws), ("repaired-r2", r2_draws)):
        if not {"analysis_id", "model_tag", "sampling_plan_sha256"}.issubset(draws.columns):
            raise ValueError(f"{label} calibration draw table lacks model/plan identity columns")
    target_dbs = ("nwicu_discharge_v4_1", "sicdb_zero_offset_v4_2", "eicu_semantic_v4_2")
    plans = {db: old_plan["cohort_sampling_plans"][db] for db in target_dbs[:2]}
    plans[target_dbs[2]] = r2_plan["cohort_sampling_plans"][target_dbs[2]]
    all_draws = pd.concat([
        old_draws.loc[old_draws["db"].astype(str).isin(target_dbs[:2])],
        r2_draws.loc[r2_draws["db"].astype(str).eq(target_dbs[2])],
    ], ignore_index=True)
    expected = {model_id: all_draws.loc[all_draws["analysis_id"].astype(str).eq(model_id)].copy()
                for model_id in MODEL_IDS}
    if any(frame.empty for frame in expected.values()):
        raise ValueError("retained NW/SIC plus repaired-r2 draw tables lack a model arm")
    return plans, expected, {
        "nw_sic_calibration_draws_sha256": old_hash,
        "nw_sic_plan_manifest_sha256": _sha256(old_plan_path),
        "nw_sic_analysis_manifest_sha256": _sha256(old_analysis_path),
        "repaired_r2_calibration_draws_sha256": r2_hash,
        "repaired_r2_plan_manifest_sha256": _sha256(r2_plan_path),
        "repaired_r2_analysis_manifest_sha256": _sha256(r2_analysis_path),
    }


def main() -> int:
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true",
                        help="validate every saved plan and draw/model identity without fitting")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        _preflight_all()
        return 0
    preflight_path = OUT / "preflight_manifest.json"
    if not preflight_path.is_file() or _json(preflight_path).get("status") != "preflight_complete":
        raise ValueError("all-cohort/model frozen-plan preflight must pass before replay")
    print(json.dumps({"stage": "paired_calibration_inventory_started", "out": str(OUT)}, sort_keys=True), flush=True)
    source_frames, source_inputs = _load_source_mimic()
    source_plans, source_draws, source_draw_hashes = _source_plan_and_draws()
    target_frames, target_inputs = _target_frames()
    target_plans, target_draws, target_draw_hashes = _transport_plans_and_draws()
    # The repaired eICU plan is intentionally treated as its own frozen within-version plan;
    # this analysis does not assert old-vs-repaired cross-version draw identity.
    cohorts: dict[str, tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, Any]]] = {}
    cohorts["mimic"] = (source_frames, source_draws, source_plans["mimic"])
    for db in transport.TARGET_COHORTS:
        cohorts[db] = ({model_id: target_frames[model_id][db] for model_id in MODEL_IDS},
                       target_draws, target_plans[db])
    all_rows: list[dict[str, Any]] = []
    cohort_manifests: dict[str, Any] = {}
    for index, (db, (frames, expected_by_model, plan)) in enumerate(cohorts.items()):
        print(json.dumps({"stage": "paired_calibration_cohort_started", "db": db,
                          "sampling_plan_sha256": plan["sampling_plan_sha256"]}, sort_keys=True), flush=True)
        rows, cohort_manifest = _replay_cohort(
            db, frames, expected_by_model, plan, out_cohort=OUT / "cohorts" / db,
            cohort_index=index,
        )
        all_rows.extend(rows)
        cohort_manifests[db] = cohort_manifest
        print(json.dumps({"stage": "paired_calibration_cohort_complete", "db": db,
                          "direct_contrast_rows": len(rows), "parity": cohort_manifest["parity"]},
                         sort_keys=True), flush=True)
    if len(all_rows) != 4 * len(cohorts) * len(BASELINE_IDS) * len(METHODS) * len(METRICS):
        raise AssertionError(f"unexpected direct contrast row grid: {len(all_rows)}")
    table_path = OUT / "direct_calibrated_fm_minus_baseline_patient_ci.csv"
    pd.DataFrame(all_rows).to_csv(table_path, index=False)
    inputs = {
        "source": {**source_inputs, **source_draw_hashes},
        "transport": {**target_inputs, **target_draw_hashes},
        "recalibration_code_sha256": _sha256(REV / "robustness/recalibration.py"),
        "transport_analysis_code_sha256": _sha256(REV / "robustness/transport_v4_2/transport_analysis.py"),
    }
    manifest = {
        "status": "complete", "schema": "ehr_fm_direct_calibrated_model_pairs_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cohorts": list(cohorts), "models": list(MODEL_IDS),
        "baselines": list(BASELINE_IDS), "methods": list(METHODS),
        "budgets": list(BUDGETS), "adaptation_repeats": REPEATS,
        "assessment_patient_cluster_bootstrap_draws": N_BOOT,
        "estimand": "For each model/method/repeat, compute assessment-stay Brier/log-loss; average each stay's loss across 200 realized repeats, then take the stay-weighted FM-minus-baseline mean. Bootstrap assessment patients (all their stays) with common resamples across contrasts.",
        "scope": "all 200 operational repeats; offset/Platt use Jeffreys-smoothed adaptation prevalence after fit failure; raw is frozen source-p prediction",
        "replay_status": "calibration-only replay on frozen source probabilities and exact frozen adaptation/assessment plans; no source head, encoder, predictor, target model, or sampling plan refit/redraw",
        "replay_parity": "all retained per-repeat raw/offset/Platt score rows, fit statuses, fallback flags, and actual adaptation/index-stay hashes matched before CI table publication",
        "preflight_manifest_path": str(preflight_path),
        "preflight_manifest_sha256": _sha256(preflight_path),
        "eicu_scope": "within repaired-r2 frozen plan only; no old-vs-repaired draw identity or cross-version curve claim",
        "inputs": inputs, "cohort_manifests": cohort_manifests,
        "direct_ci_table_path": str(table_path), "direct_ci_table_sha256": _sha256(table_path),
        "direct_ci_rows": len(all_rows), "patient_level_exported_to_local": False,
        "analysis_code_sha256": _sha256(Path(__file__)),
        "elapsed_seconds": round(time.time() - start, 2),
    }
    manifest_path = OUT / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "paired_calibration_complete", "manifest": str(manifest_path),
                      "manifest_sha256": _sha256(manifest_path), "rows": len(all_rows),
                      "elapsed_seconds": manifest["elapsed_seconds"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
