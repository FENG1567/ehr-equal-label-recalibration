#!/usr/bin/env python3
"""V5 calibration and hospital adaptation diagnostics.

Patient-level inputs and outputs remain on the analysis server.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV4 = ROOT / "revision_v4_20261122"
REV5 = ROOT / "revision_v5_20261123"
ROBUSTNESS = REV4 / "robustness"
for _path in (ROBUSTNESS / "transport_v4_2", ROBUSTNESS,
              REV4 / "baselines/transport_v4_2"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import transport_analysis as ta  # noqa: E402
from v5_curve_tools import calibration_curve_cluster, paired_cluster_mean_ci  # noqa: E402
from hospital_heterogeneity import metric_row  # noqa: E402
from paired_baselines import sha256_file  # noqa: E402
from recalibration import (  # noqa: E402
    _identity_digest, adaptation_assessment_pools, fit_offset, fit_platt,
    one_index_per_patient,
)

RESOURCE_SCOPE = os.environ.get("V5_RESOURCE_SCOPE", "unspecified")


def current_cgroup_path() -> str:
    for row in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
        if row.startswith("0::"):
            return row.split("::", 1)[1]
    raise RuntimeError("unified cgroup path is unavailable")

SEED = 20261123
GLOBAL_ROLE_SEED = 20261122
N_BOOT = 1000
REPEATS = 200
BUDGETS = (50, 200, 500, 2000)
SITE_BUDGETS = (50, 200, 500)
COHORTS = ("eicu_semantic_v4_2", "nwicu_discharge_v4_1")
MODELS = ("main20k_real", "conventional_lr_value_plus_pattern")
MODEL_LABELS = {"main20k_real": "FM", "conventional_lr_value_plus_pattern": "V+P LR"}
DIRECT_ROOT = REV4 / "calibrated_model_pairs/results/direct_calibrated_fm_baseline_v1/cohorts"
GATE_ROOT = REV4 / "temporal/results/eligibility/recovered_reviewed_v4_2"
RUN_ID = "v5_20261123"
CAL_OUT = REV5 / "calibration" / RUN_ID
SITE_OUT = REV5 / "sites" / RUN_ID
LEGACY_DIRECT_REUSE_LOG = (
    REV5 / "logs" / "ehrv5-calibration-sites-r2-20261123T154516Z-3077415.service.log"
)
CALIBRATION_CHECKPOINT_TABLES = {
    "global_recalibration_repeat_scores": "repeats",
    "paired_loss_bootstrap_draws": "gap_draws",
    "paired_loss_gap_bootstrap": "gap_ci",
    "per_stay_repeat_mean_parity": "parity",
    "calibration_curve_points": "curves",
    "calibration_reliability_bins": "bins",
    "calibration_curve_risk_support": "risks",
}
EPS = 1e-6
PENALTY = 1e-6
V4_HOSPITAL_METRICS = ROBUSTNESS / "results/hospital_primary_v4_20261123_0515/eicu_hospital_metrics.csv"
V4_HOSPITAL_MANIFEST = ROBUSTNESS / "results/hospital_primary_v4_20261123_0515/hospital_analysis_manifest.json"
V4_HOSPITAL_METRICS_SHA256 = "b055744e6674b27c287bae7b90a2926e11a07e519442b83bbac1b5d8d0a393f4"
V4_HOSPITAL_MANIFEST_SHA256 = "0c227dd9c19c471763b328381e1bf4502f2bfbabfcd042592c3763c97bc2e56e"
HGBT_SUMMARY_SOURCES = {
    "eicu_semantic_v4_2": (
        REV4 / "publication_assembly/table2_recalibration_summary_eicu_allmethods.csv",
        "6d05d787ac33163f0f757fa208518e9415b80ab8b50a53385b3a0232a0d36fb7",
    ),
    "nwicu_discharge_v4_1": (
        REV4 / "publication_assembly/table2_recalibration_summary_nw_sic_allmethods.csv",
        "e0cc66e7945c6af10a4d05168bc2ff48be6df1636ea88590c3f6e9f519b7cc49",
    ),
}
HGBT_SUMMARY_METHODS = (
    "raw", "adaptation_prevalence", "smoothed_adaptation_prevalence",
    "offset_conditional", "offset_operational", "platt_conditional",
    "platt_operational",
)
SITE_METHODS = (
    "no_update", "prevalence", "jeffreys_prevalence", "intercept", "platt",
)
FITTED_SITE_METHODS = {"intercept", "platt"}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _progress_event(stage: str, **fields: Any) -> None:
    print(json.dumps({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stage": stage,
        **fields,
    }, sort_keys=True, default=str), flush=True)


def _verify_original_v4_site_reference() -> dict[str, Any]:
    """Verify the historical site counts against their original v4 artifacts."""
    metrics_hash = sha256_file(V4_HOSPITAL_METRICS)
    manifest_hash = sha256_file(V4_HOSPITAL_MANIFEST)
    if metrics_hash != V4_HOSPITAL_METRICS_SHA256:
        raise ValueError("original v4 eICU hospital metrics hash changed")
    if manifest_hash != V4_HOSPITAL_MANIFEST_SHA256:
        raise ValueError("original v4 eICU hospital manifest hash changed")
    manifest = _json(V4_HOSPITAL_MANIFEST)
    if manifest.get("status") != "complete":
        raise ValueError("original v4 eICU hospital analysis is not complete")
    metrics = pd.read_csv(V4_HOSPITAL_METRICS)
    fm = metrics.loc[
        (metrics["scope"] == "selected_hospital_all_stays")
        & (metrics["model_id"] == "main20k_real")
    ]
    slope_points = int(fm["calibration_slope"].notna().sum())
    stable_sites = int(
        fm["site_stability_flag_ge10_events_and_nonevents"].fillna(False).sum()
    )
    manifest_slope_status = manifest["within_site_calibration_slope_ci"][
        "site_slope_ci_status_counts_by_model"]["main20k_real"]
    expected_slope_status = {
        "diagnostic_not_for_manuscript_below_10_events_or_nonevents": 48,
        "ok": 147,
        "unestimable_single_class": 11,
    }
    if (len(fm) != 206 or fm["hospitalid"].nunique() != 206
            or stable_sites != 147 or slope_points != 193
            or int(manifest["selected_hospital_metric_rows_by_model"]["main20k_real"]) != 206
            or int(manifest["within_site_calibration_slope_ci"]["stable_site_counts_by_model"]["main20k_real"]) != 147
            or manifest_slope_status != expected_slope_status):
        raise ValueError("original v4 eICU hospital counts do not match the verified summary")
    _progress_event(
        "original_v4_site_reference_verified",
        hospital_denominator=int(len(fm)), finite_slope_points=slope_points,
        stable_sites_ge10_events_nonevents=stable_sites,
        metrics_sha256=metrics_hash, manifest_sha256=manifest_hash,
    )
    return {
        "status": "verified_original_v4_reference",
        "source_metrics_path_internal": str(V4_HOSPITAL_METRICS),
        "source_metrics_sha256": metrics_hash,
        "source_manifest_path_internal": str(V4_HOSPITAL_MANIFEST),
        "source_manifest_sha256": manifest_hash,
        "source_status": manifest["status"],
        "source_scope": "selected_hospital_all_stays; original v4 eICU predictions",
        "model_id": "main20k_real",
        "hospital_denominator": int(len(fm)),
        "finite_calibration_slope_points": slope_points,
        "stable_sites_ge10_events_and_nonevents": stable_sites,
        "stable_site_rule": ">=10 events and >=10 nonevents; display/stability rule only",
        "counts_source": "original v4 aggregate and manifest; not repaired-v5 site counts",
    }


def _load_hgbt_calibration_supplement(
    endpoints: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and validate v4 HGBT aggregate rows as an explicitly historical supplement."""
    supplements = []
    source_manifest: dict[str, Any] = {}
    expected_endpoint_scope = "retrospective_landmark_prediction_measurement_time"
    for db in COHORTS:
        if db not in HGBT_SUMMARY_SOURCES:
            raise ValueError(f"no verified v4 HGBT summary is registered for {db}")
        path, expected_hash = HGBT_SUMMARY_SOURCES[db]
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            raise ValueError(f"v4 HGBT summary hash changed for {db}")
        table = pd.read_csv(path)
        cohort_rows = table.loc[
            (table["db"] == db)
            & (table["analysis_id"] == "conventional_hgb_value_plus_pattern")
        ].copy()
        meta = cohort_rows.loc[cohort_rows["budget"].isna()]
        draws = cohort_rows.loc[cohort_rows["budget"].notna()].copy()
        if len(meta) != 1 or len(draws) != 28:
            raise ValueError(f"v4 HGBT summary has unexpected row counts for {db}")
        metadata = meta.iloc[0]
        main_meta = table.loc[
            (table["db"] == db)
            & (table["analysis_id"] == "main20k_real")
            & table["budget"].isna()
        ]
        claim_scope = str(metadata.get("cohort_status", ""))
        endpoint = endpoints.get(db, {})
        if isinstance(endpoint, str):
            if endpoint != "hospital":
                raise ValueError(f"HGBT hospital-death supplement has incompatible endpoint for {db}: {endpoint}")
            # The upstream transport loader returns endpoint names after its
            # strict manifest, scope and population checks.
            current_claim_scope = expected_endpoint_scope
        elif isinstance(endpoint, dict):
            current_claim_scope = endpoint.get("analysis_claim_scope", claim_scope)
        else:
            raise TypeError(f"unrecognized endpoint metadata type for {db}: {type(endpoint).__name__}")
        shared_role_fields = (
            "assessment_population_identity_sha256",
            "assessment_patient_set_sha256",
            "adaptation_index_patient_stay_sha256",
            "sampling_plan_sha256",
            "seed", "repeats",
        )
        if (len(main_meta) != 1
                or any(str(metadata.get(field)) != str(main_meta.iloc[0].get(field))
                       for field in shared_role_fields)
                or str(metadata.get("status")) != "ok"
                or claim_scope != expected_endpoint_scope
                or current_claim_scope != claim_scope
                or str(metadata.get("model_tag")) != "conventional_hgb"
                or int(metadata.get("repeats")) != REPEATS
                or int(metadata.get("seed")) != GLOBAL_ROLE_SEED
                or str(metadata.get("bootstrap_cluster_unit")) != "patient"):
            raise ValueError(f"v4 HGBT provenance or endpoint scope is not verified for {db}")
        if (set(draws["budget"].astype(int)) != set(BUDGETS)
                or set(draws["method"].astype(str)) != set(HGBT_SUMMARY_METHODS)
                or draws.groupby(["budget", "method"], dropna=False).size().ne(1).any()
                or not draws["n_repeats"].eq(REPEATS).all()):
            raise ValueError(f"v4 HGBT summary does not cover all four 200-draw budgets for {db}")
        full_draw_methods = {
            "raw", "adaptation_prevalence", "smoothed_adaptation_prevalence",
            "offset_operational", "platt_operational",
        }
        full_draw_rows = draws.loc[draws["method"].isin(full_draw_methods)]
        if (len(full_draw_rows) != len(full_draw_methods) * len(BUDGETS)
                or not full_draw_rows["n_valid_repeats"].eq(REPEATS).all()
                or not draws["brier_mean"].notna().all()
                or not draws["logloss_mean"].notna().all()):
            raise ValueError(f"v4 HGBT operational rows are not complete for {db}")
        draws["source_version"] = "v4"
        draws["source_archive_method"] = draws["method"].astype(str)
        draws["method"] = draws["method"].replace({
            "offset_conditional": "intercept_conditional",
            "offset_operational": "intercept",
        })
        draws["source_artifact_path_internal"] = str(path)
        draws["source_artifact_sha256"] = observed_hash
        draws["source_endpoint_scope"] = claim_scope
        draws["supplement_scope"] = (
            "historical pooled calibration summary; all 200 adaptation draws; "
            "not recomputed on repaired-v5 scores or site-level"
        )
        supplements.append(draws)
        source_manifest[db] = {
            "path_internal": str(path),
            "sha256": observed_hash,
            "status": str(metadata["status"]),
            "analysis_id": str(metadata["analysis_id"]),
            "model_tag": str(metadata["model_tag"]),
            "endpoint_scope": claim_scope,
            "repeats": int(metadata["repeats"]),
            "adaptation_seed": int(metadata["seed"]),
            "assessment_patient_set_sha256": str(metadata["assessment_patient_set_sha256"]),
            "assessment_population_identity_sha256": str(metadata["assessment_population_identity_sha256"]),
            "adaptation_index_patient_stay_sha256": str(metadata["adaptation_index_patient_stay_sha256"]),
            "sampling_plan_sha256": str(metadata["sampling_plan_sha256"]),
            "assessment_patients": int(metadata["n_assessment_patients"]),
            "assessment_stays": int(metadata["n_assessment_stays"]),
            "rows_retained": int(len(draws)),
            "operational_rows_with_200_valid_draws": int(len(full_draw_rows)),
        }
    return pd.concat(supplements, ignore_index=True), source_manifest


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def _canon_id(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8")
    if pd.isna(value):
        raise ValueError("missing ID in frozen scoring artifact")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return False if pd.isna(value) else str(value).strip().lower() in {"true", "1", "yes"}


def _losses(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yy, pp = np.asarray(y, dtype=float), _clip(p)
    return (yy - pp) ** 2, -(yy * np.log(pp) + (1.0 - yy) * np.log1p(-pp))


def _metrics(
    y: np.ndarray, p: np.ndarray, subjects: np.ndarray | None = None,
    inverse: np.ndarray | None = None, counts: np.ndarray | None = None,
) -> dict[str, float]:
    brier, logloss = _losses(y, p)
    if inverse is None or counts is None:
        if subjects is None:
            raise ValueError("patient IDs or precomputed patient groups are required")
        _, inverse = np.unique(np.asarray(subjects).astype(str), return_inverse=True)
        counts = np.bincount(inverse).astype(float)
    patient_brier = np.bincount(inverse, weights=brier) / counts
    patient_logloss = np.bincount(inverse, weights=logloss) / counts
    return {
        "brier": float(brier.mean()),
        "logloss": float(logloss.mean()),
        "brier_patient_weighted": float(patient_brier.mean()),
        "logloss_patient_weighted": float(patient_logloss.mean()),
        "mean_predicted_probability": float(np.mean(p)),
        "observed_event_rate": float(np.mean(y)),
    }


def _validate_pair(fm: pd.DataFrame, lr: pd.DataFrame, db: str) -> None:
    ta._exact_target_identity_match(fm, lr, db=db)
    if ("hospitalid" in fm.columns) != ("hospitalid" in lr.columns):
        raise ValueError(f"{db}: hospital ID metadata is absent from one frozen model arm")
    if "hospitalid" in fm.columns and "hospitalid" in lr.columns:
        left = fm.set_index("stay_id")["hospitalid"]
        right = lr.set_index("stay_id")["hospitalid"]
        left, right = left.sort_index(), right.sort_index()
        if not left.index.equals(right.index):
            raise ValueError(f"{db}: stay IDs differ between frozen model arms")
        left_missing, right_missing = left.isna(), right.isna()
        if not left_missing.equals(right_missing):
            raise ValueError(f"{db}: hospital ID missingness differs between frozen model arms")
        known = ~left_missing
        if not left.loc[known].map(_canon_id).equals(right.loc[known].map(_canon_id)):
            raise ValueError(f"{db}: hospital IDs differ between frozen model arms")


def _load_direct(db: str, frame: pd.DataFrame, model_frames: dict[str, pd.DataFrame]):
    folder = DIRECT_ROOT / db
    manifest_path = folder / "cohort_manifest.json"
    params_path = folder / "server_only_fit_parameters.csv"
    losses_path = folder / "server_only_per_stay_mean_losses.npz"
    manifest = _json(manifest_path)
    if manifest.get("status") != "parity_verified":
        raise ValueError(f"{db}: direct calibration artifact is not parity_verified")
    if sha256_file(params_path) != manifest.get("fit_parameters_sha256"):
        raise ValueError(f"{db}: fit parameter hash differs from direct manifest")
    if sha256_file(losses_path) != manifest.get("per_stay_loss_archive_sha256"):
        raise ValueError(f"{db}: per-stay loss hash differs from direct manifest")
    params = pd.read_csv(params_path)
    required = {
        "db", "analysis_id", "budget", "repeat", "method", "fit_ok",
        "fit_status", "slope", "intercept", "operational_fallback_probability",
        "events", "sampling_plan_sha256",
    }
    if not required.issubset(params.columns):
        raise ValueError(f"{db}: direct fit parameter schema is incomplete")
    usable = frame.loc[
        frame["eligible_24h"].astype(bool) & frame["temporal_verified"].astype(bool)
    ].copy()
    sampling, _counts = ta._sampling_plan_for_frame(frame, db=db)
    if sampling["sampling_plan_sha256"] != manifest.get("sampling_plan_sha256"):
        raise ValueError(f"{db}: direct sampling plan differs from repaired frozen scores")
    _adapt, assess, _ = adaptation_assessment_pools(usable, GLOBAL_ROLE_SEED)
    if len(assess) != int(manifest.get("n_assessment_stays", -1)):
        raise ValueError(f"{db}: assessment stay count differs from direct manifest")
    if assess["subject"].nunique() != int(manifest.get("n_assessment_patients", -1)):
        raise ValueError(f"{db}: assessment patient count differs from direct manifest")
    if len(params) != int(manifest.get("fit_parameter_rows", -1)):
        raise ValueError(f"{db}: fit parameter row count differs from direct manifest")
    if not params["sampling_plan_sha256"].astype(str).eq(
        str(manifest["sampling_plan_sha256"])
    ).all():
        raise ValueError(f"{db}: fit parameter rows do not share the validated plan")
    _progress_event("direct_loss_archive_read_start", cohort=db, path=str(losses_path))
    with np.load(losses_path, allow_pickle=False) as archive:
        if not {"stay_id", "subject", "y"}.issubset(archive.files):
            raise ValueError(f"{db}: per-stay archive lacks identity arrays")
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    _progress_event(
        "direct_loss_archive_read_done", cohort=db, arrays=len(arrays),
        assessment_stays=int(len(arrays["stay_id"])),
    )
    archive_ids = [_canon_id(value) for value in arrays["stay_id"]]
    if len(archive_ids) != len(set(archive_ids)):
        raise ValueError(f"{db}: per-stay archive contains duplicate stay IDs")
    assess = assess.copy()
    assess["_stay_key"] = assess["stay_id"].map(_canon_id)
    if set(archive_ids) != set(assess["_stay_key"]):
        raise ValueError(f"{db}: archived assessment roster differs from reconstructed roles")
    keyed = assess.set_index("_stay_key")
    for i, stay in enumerate(archive_ids):
        row = keyed.loc[stay]
        if _canon_id(arrays["subject"][i]) != _canon_id(row["subject"]):
            raise ValueError(f"{db}: assessment patient ID mismatch in per-stay archive")
        if int(arrays["y"][i]) != int(row["y"]):
            raise ValueError(f"{db}: assessment label mismatch in per-stay archive")
    ordered = keyed.loc[archive_ids]
    for model in MODELS:
        scored = model_frames[model].set_index("stay_id").loc[ordered["stay_id"]]
        yy = ordered["y"].to_numpy(dtype=int)
        raw_brier, raw_logloss = _losses(yy, scored["p"].to_numpy(dtype=float))
        for budget in BUDGETS:
            for metric, computed in (("brier", raw_brier), ("logloss", raw_logloss)):
                key = f"{model}__budget{budget}__raw__{metric}"
                if key not in arrays or not np.allclose(
                    arrays[key], computed, rtol=2e-5, atol=2e-6
                ):
                    raise ValueError(f"{db}: frozen raw score parity failed for {key}")
    provenance = {
        "cohort_manifest_path_internal": str(manifest_path),
        "cohort_manifest_sha256": sha256_file(manifest_path),
        "fit_parameters_path_internal": str(params_path),
        "fit_parameters_sha256": sha256_file(params_path),
        "per_stay_loss_path_internal": str(losses_path),
        "per_stay_loss_sha256": sha256_file(losses_path),
        "sampling_plan_sha256": sampling["sampling_plan_sha256"],
        "assessment_patients": int(assess["subject"].nunique()),
        "assessment_stays": int(len(assess)),
        "assessment_events": int(assess["y"].sum()),
        "per_stay_roster_identity_verified": True,
    }
    return arrays, params, assess, provenance


def _parameter_map(params: pd.DataFrame, db: str, model: str):
    subset = params.loc[(params["db"] == db) & (params["analysis_id"] == model)]
    out = {}
    for row in subset.to_dict("records"):
        key = (int(row["budget"]), int(row["repeat"]), str(row["method"]))
        if key in out:
            raise ValueError(f"{db}/{model}: duplicate fit parameter row {key}")
        out[key] = row
    for budget in BUDGETS:
        for repeat in range(REPEATS):
            for method in ("offset", "platt"):
                if (budget, repeat, method) not in out:
                    raise ValueError(f"{db}/{model}: missing {budget}/{repeat}/{method} fit")
    return out


def _predict_fit(raw: np.ndarray, fit: dict[str, Any], budget: int):
    if _bool(fit["fit_ok"]):
        pred = expit(float(fit["slope"]) * logit(_clip(raw)) + float(fit["intercept"]))
        return _clip(pred), True, str(fit.get("fit_status", "ok")), False
    fallback = fit.get("operational_fallback_probability")
    if pd.isna(fallback):
        fallback = (int(fit["events"]) + 0.5) / (budget + 1.0)
    return np.full(len(raw), float(fallback)), False, str(fit.get("fit_status", "failed")), True


def _global_outputs(db, arrays, params, assessment, frames):
    y = assessment["y"].to_numpy(dtype=int)
    subjects = assessment["subject"].astype(str).to_numpy()
    _, inverse = np.unique(subjects, return_inverse=True)
    patient_counts = np.bincount(inverse).astype(float)
    model_assessment = {}
    raw_predictions = {}
    for model in MODELS:
        use = frames[model][db].loc[
            frames[model][db]["eligible_24h"].astype(bool)
            & frames[model][db]["temporal_verified"].astype(bool)
        ]
        aligned = use.set_index("stay_id").loc[assessment["stay_id"]]
        if not np.array_equal(aligned["y"].to_numpy(dtype=int), y):
            raise ValueError(f"{db}/{model}: global assessment labels differ")
        model_assessment[model] = aligned
        raw_predictions[model] = _clip(aligned["p"].to_numpy(dtype=float))
    fits = {model: _parameter_map(params, db, model) for model in MODELS}
    for budget in BUDGETS:
        for repeat in range(REPEATS):
            draw_rows = [
                fits[model][(budget, repeat, method)]
                for model in MODELS for method in ("offset", "platt")
            ]
            for field in (
                "adaptation_sample_index_sha256",
                "adaptation_sample_patient_index_stay_sha256",
                "events",
            ):
                if field in draw_rows[0]:
                    values = {str(row.get(field)) for row in draw_rows}
                    if len(values) != 1:
                        raise ValueError(
                            f"{db}/{budget}/{repeat}: fit arms did not share labels/sample ({field})"
                        )
    raw_loss = {model: _losses(y, raw_predictions[model]) for model in MODELS}
    archive_ids = [_canon_id(value) for value in arrays["stay_id"]]
    current_ids = [_canon_id(value) for value in assessment["stay_id"]]
    current_position = {key: i for i, key in enumerate(current_ids)}
    archive_order = np.asarray([current_position[key] for key in archive_ids], dtype=int)
    score_rows, gap_draw_rows, parity_rows = [], [], []

    for budget in BUDGETS:
        mean_post = {
            model: {"offset": {"brier": np.zeros(len(y)), "logloss": np.zeros(len(y))},
                    "platt": {"brier": np.zeros(len(y)), "logloss": np.zeros(len(y))}}
            for model in MODELS
        }
        for repeat in range(REPEATS):
            predictions, fit_info, events_by_model = {}, {}, {}
            for model in MODELS:
                offset_fit = fits[model][(budget, repeat, "offset")]
                platt_fit = fits[model][(budget, repeat, "platt")]
                if int(offset_fit["events"]) != int(platt_fit["events"]):
                    raise ValueError(f"{db}/{model}/{budget}/{repeat}: offset and Platt labels differ")
                offset, offset_ok, offset_status, offset_fallback = _predict_fit(
                    raw_predictions[model], offset_fit, budget
                )
                platt, platt_ok, platt_status, platt_fallback = _predict_fit(
                    raw_predictions[model], platt_fit, budget
                )
                events = int(platt_fit["events"])
                events_by_model[model] = events
                prevalence = events / float(budget)
                jeffreys = (events + 0.5) / (budget + 1.0)
                predictions[model] = {
                    "no_update": raw_predictions[model],
                    "prevalence": np.full(len(y), prevalence),
                    "jeffreys_prevalence": np.full(len(y), jeffreys),
                    "intercept": offset,
                    "platt": platt,
                }
                fit_info[model] = {
                    "intercept": (offset_ok, offset_status, offset_fallback),
                    "platt": (platt_ok, platt_status, platt_fallback),
                }
                for method, pred in predictions[model].items():
                    values = _metrics(y, pred, inverse=inverse, counts=patient_counts)
                    ok, status, fallback = fit_info[model].get(
                        method, (True, "not_applicable", False)
                    )
                    score_rows.append({
                        "cohort": db, "model": MODEL_LABELS[model], "model_id": model,
                        "budget": int(budget), "repeat": int(repeat), "method": method,
                        "events": events, "nonevents": int(budget - events),
                        "fit_ok": bool(ok), "fit_status": status,
                        "fallback_used": bool(fallback),
                        "fallback_probability": (events + 0.5) / (budget + 1.0) if fallback else math.nan,
                        **values,
                    })
                    archive_method = "offset" if method == "intercept" else method
                    if archive_method in {"offset", "platt"}:
                        brier_loss, logloss_loss = _losses(y, pred)
                        mean_post[model][archive_method]["brier"] += brier_loss / REPEATS
                        mean_post[model][archive_method]["logloss"] += logloss_loss / REPEATS
            if len(set(events_by_model.values())) != 1:
                raise ValueError(f"{db}/{budget}/{repeat}: FM and LR adaptation labels differ")

            for metric, ix in (("brier", 0), ("logloss", 1)):
                raw_delta = raw_loss["main20k_real"][ix] - raw_loss["conventional_lr_value_plus_pattern"][ix]
                for method in ("platt",):
                    post_delta = (
                        _losses(y, predictions["main20k_real"][method])[ix]
                        - _losses(y, predictions["conventional_lr_value_plus_pattern"][method])[ix]
                    )
                    for contrast, delta in (
                        ("raw_gap", raw_delta),
                        ("platt_gap", post_delta),
                        ("gap_change", post_delta - raw_delta),
                    ):
                        patient_mean = np.bincount(inverse, weights=delta) / patient_counts
                        for weighting, estimate in (
                            ("stay_weighted", float(delta.mean())),
                            ("patient_weighted", float(patient_mean.mean())),
                        ):
                            gap_draw_rows.append({
                                "cohort": db, "budget": int(budget), "repeat": int(repeat),
                                "metric": metric, "contrast": contrast,
                                "weighting": weighting, "estimate": estimate,
                            })
        for model in MODELS:
            for method in ("intercept", "platt"):
                archive_method = "offset" if method == "intercept" else method
                for metric in ("brier", "logloss"):
                    key = f"{model}__budget{budget}__{archive_method}_operational__{metric}"
                    got = mean_post[model][archive_method][metric][archive_order]
                    expected = np.asarray(arrays[key], dtype=float)
                    if not np.allclose(expected, got, rtol=2e-5, atol=2e-6):
                        raise ValueError(f"{db}/{model}/{budget}/{method}/{metric}: repeat mean parity failed")
                    parity_rows.append({
                        "cohort": db, "model": MODEL_LABELS[model], "model_id": model,
                        "budget": int(budget), "method": method, "metric": metric,
                        "parity_ok": True, "max_abs_difference": float(np.max(np.abs(expected - got))),
                    })
        print(json.dumps({"stage": "global_repeat_budget_complete", "cohort": db,
                          "budget": int(budget), "repeats": REPEATS}, sort_keys=True), flush=True)

    # Bootstrap paired per-stay mean losses at patient level, sharing each
    # resample across raw gap, Platt gap and gap change.
    npz_subjects = np.asarray([_canon_id(v) for v in arrays["subject"]], dtype=str)
    gap_rows = []
    draw_frame = pd.DataFrame(gap_draw_rows)
    for budget in BUDGETS:
        for metric in ("brier", "logloss"):
            fm_raw = np.asarray(arrays[f"main20k_real__budget{budget}__raw__{metric}"], dtype=float)
            lr_raw = np.asarray(arrays[f"conventional_lr_value_plus_pattern__budget{budget}__raw__{metric}"], dtype=float)
            fm_post = np.asarray(arrays[f"main20k_real__budget{budget}__platt_operational__{metric}"], dtype=float)
            lr_post = np.asarray(arrays[f"conventional_lr_value_plus_pattern__budget{budget}__platt_operational__{metric}"], dtype=float)
            vectors = {
                "raw_gap": fm_raw - lr_raw,
                "platt_gap": fm_post - lr_post,
            }
            vectors["gap_change"] = vectors["platt_gap"] - vectors["raw_gap"]
            columns = np.column_stack([vectors[k] for k in ("raw_gap", "platt_gap", "gap_change")])
            seed = SEED + (COHORTS.index(db) + 1) * 10000 + budget
            ci_stay = paired_cluster_mean_ci(
                npz_subjects, columns, draws=N_BOOT, seed=seed, patient_weighted=False,
            )
            ci_patient = paired_cluster_mean_ci(
                npz_subjects, columns, draws=N_BOOT, seed=seed, patient_weighted=True,
            )
            for j, contrast in enumerate(("raw_gap", "platt_gap", "gap_change")):
                values = vectors[contrast]
                ci_stay_values = {
                    key: ci_stay[key][j] for key in ("point", "low", "high")
                }
                ci_patient_values = {
                    key: ci_patient[key][j] for key in ("point", "low", "high")
                }
                for weighting, interval in (
                    ("stay_weighted", ci_stay_values),
                    ("patient_weighted", ci_patient_values),
                ):
                    fit_draws = draw_frame.loc[
                        (draw_frame["cohort"] == db) & (draw_frame["budget"] == budget)
                        & (draw_frame["metric"] == metric)
                        & (draw_frame["contrast"] == contrast)
                        & (draw_frame["weighting"] == weighting), "estimate"
                    ].to_numpy(dtype=float)
                    gap_rows.append({
                        "cohort": db, "budget": int(budget), "metric": metric,
                        "contrast": contrast, "weighting": weighting,
                        "estimate": float(interval["point"]),
                        "bootstrap_ci_95_lower": float(interval["low"]),
                        "bootstrap_ci_95_upper": float(interval["high"]),
                        "bootstrap_requested": int(N_BOOT),
                        "bootstrap_valid": int(ci_stay["valid_draws"]),
                        "assessment_stays": int(ci_stay["n_stays"]),
                        "assessment_patients": int(ci_stay["n_patients"]),
                        "adaptation_draws": int(len(fit_draws)),
                        "adaptation_draw_p05": float(np.percentile(fit_draws, 5)),
                        "adaptation_draw_p95": float(np.percentile(fit_draws, 95)),
                        "adaptation_draw_mean": float(np.mean(fit_draws)),
                        "interval_scope": "200 repeat per-stay means; assessment patient-cluster bootstrap conditional on frozen scores",
                    })
    return (
        pd.DataFrame(score_rows), draw_frame, pd.DataFrame(gap_rows),
        pd.DataFrame(parity_rows),
    )


def _site_seed(site: str) -> int:
    token = hashlib.sha256(site.encode("utf-8")).hexdigest()[:8]
    return int((SEED + int(token, 16)) % (2**32 - 1))


def _site_inventory(fm: pd.DataFrame, lr: pd.DataFrame):
    eligible = fm.loc[
        fm["eligible_24h"].astype(bool) & fm["temporal_verified"].astype(bool)
    ].copy()
    lr_eligible = lr.loc[
        lr["eligible_24h"].astype(bool) & lr["temporal_verified"].astype(bool)
    ].copy()
    if "hospitalid" not in eligible or eligible["hospitalid"].isna().any():
        raise ValueError("repaired eICU eligible FM predictions lack hospital IDs")
    if "hospitalid" not in lr_eligible or lr_eligible["hospitalid"].isna().any():
        raise ValueError("repaired eICU eligible LR predictions lack hospital IDs")
    eligible["hospitalid"] = eligible["hospitalid"].map(_canon_id)
    lr_eligible["hospitalid"] = lr_eligible["hospitalid"].map(_canon_id)
    patient_counts = eligible.groupby("hospitalid", observed=True)["subject"].nunique().to_dict()
    sites = sorted(
        patient_counts,
        key=lambda site: (-int(patient_counts[site]), int(site) if site.isdigit() else site),
    )
    if len(sites) != 206:
        raise ValueError(f"expected 206 repaired eICU hospitals; found {len(sites)}")
    selected = set(sites[:24])
    roster = pd.DataFrame([{
        "hospitalid": site,
        "eligible_unique_patients": int(patient_counts[site]),
        "eligible_stays": int((eligible["hospitalid"] == site).sum()),
        "selected_top24": bool(site in selected),
        "eligible_patient_rank_largest_first": int(rank),
        "tie_break": "ascending canonical hospital ID",
    } for rank, site in enumerate(sites, 1)])
    performance_rows = []
    for model, data in (
        ("main20k_real", eligible),
        ("conventional_lr_value_plus_pattern", lr_eligible),
    ):
        for site in sites:
            part = data.loc[data["hospitalid"] == site].copy()
            row = metric_row(
                part, model_id=model, scope="all_eligible_frozen_scores",
                hospitalid=site,
            )
            row["model"] = MODEL_LABELS[model]
            row["metric_status"] = "ok" if len(part) else "no_eligible_frozen_rows"
            row["mean_predicted_probability"] = float(part["p"].mean()) if len(part) else math.nan
            row["observed_event_rate"] = float(part["y"].mean()) if len(part) else math.nan
            performance_rows.append(row)
    performance = pd.DataFrame(performance_rows)
    fm_rows = performance.loc[performance["model_id"] == "main20k_real"]
    audit = {
        "all_eligible_source_hospitals": int(len(sites)),
        "selected_hospitals": 24,
        "selection_rule": "largest eligible unique-patient counts; ascending canonical hospital ID tie-break; outcome/performance blind",
        "finite_calibration_slope_point_hospitals_fm": int(fm_rows["calibration_slope"].notna().sum()),
        "stable_site_threshold_hospitals_fm": int(
            fm_rows["site_stability_flag_ge10_events_and_nonevents"].fillna(False).sum()
        ),
        "stable_site_threshold": ">=10 events and >=10 nonevents; interval/stability rule only, not exclusion from point estimates or hospital selection",
        "all_hospital_performance_denominator": int(len(sites)),
        "calibration_slope_point_coverage_by_model": {
            model: int(performance.loc[
                performance["model_id"] == model, "calibration_slope"
            ].notna().sum())
            for model in MODELS
        },
        "stable_threshold_coverage_by_model": {
            model: int(performance.loc[
                performance["model_id"] == model,
                "site_stability_flag_ge10_events_and_nonevents",
            ].fillna(False).sum())
            for model in MODELS
        },
    }
    return roster, performance, audit


def _across_site_summary(
    site_draws: pd.DataFrame,
    site_loss_sums: dict[tuple[str, int, str, str, str], np.ndarray],
    site_patient_info: dict[str, dict[str, Any]],
    feasible_sites_by_budget: dict[int, list[str]],
    selected_site_count: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Summarize feasible sites and bootstrap globally shared patient clusters."""
    output_rows = []
    feasibility = {}
    metrics = ("brier", "logloss")
    aggregations = ("micro", "macro")
    weightings = ("stay_weighted", "patient_weighted")

    for budget in SITE_BUDGETS:
        sites = list(feasible_sites_by_budget.get(budget, []))
        feasibility[str(budget)] = {
            "selected_site_count": int(selected_site_count),
            "feasible_site_count": int(len(sites)),
            "infeasible_site_count": int(selected_site_count - len(sites)),
        }
        if not sites:
            for model in MODELS:
                for method in SITE_METHODS:
                    method_fit_counts = (
                        {"fit_attempts": 0, "fit_successes": 0, "fit_failures": 0,
                         "fit_failure_rate": math.nan}
                        if method in FITTED_SITE_METHODS
                        else {"fit_attempts": math.nan, "fit_successes": math.nan,
                              "fit_failures": math.nan, "fit_failure_rate": math.nan}
                    )
                    for metric in metrics:
                        for aggregation in aggregations:
                            for weighting in weightings:
                                output_rows.append({
                                    "budget": int(budget), "model": MODEL_LABELS[model],
                                    "model_id": model, "method": method,
                                    "metric": metric, "aggregation": aggregation,
                                    "weighting": weighting, "status": "no_feasible_sites",
                                    "selected_site_count": int(selected_site_count),
                                    "feasible_site_count": 0,
                                    "infeasible_site_count": int(selected_site_count),
                                    "feasible_assessment_stays": 0,
                                    "feasible_unique_assessment_patients": 0,
                                    "feasible_site_patient_memberships": 0,
                                    "global_cross_site_assessment_patients": 0,
                                    "adaptation_repeats_averaged": REPEATS,
                                    "estimate": math.nan,
                                    "bootstrap_ci_95_lower": math.nan,
                                    "bootstrap_ci_95_upper": math.nan,
                                    "bootstrap_requested": N_BOOT,
                                    "bootstrap_valid": 0,
                                    "bootstrap_cluster_unit": "global patient; shared across sites",
                                    **method_fit_counts,
                                })
            continue

        global_subjects = sorted({
            subject
            for site in sites
            for subject in site_patient_info[site]["subjects"]
        })
        subject_position = {subject: index for index, subject in enumerate(global_subjects)}
        site_patient_indexes: list[np.ndarray] = []
        stay_matrix = np.zeros((len(sites), len(global_subjects)), dtype=float)
        for site_index, site in enumerate(sites):
            subjects = np.asarray(site_patient_info[site]["subjects"], dtype=str)
            patient_indexes = np.asarray(
                [subject_position[value] for value in subjects], dtype=int,
            )
            site_patient_indexes.append(patient_indexes)
            stay_matrix[site_index, patient_indexes] = np.asarray(
                site_patient_info[site]["stay_counts"], dtype=float,
            )
        site_patient_mask = stay_matrix > 0
        site_patient_denominators = site_patient_mask.sum(axis=1).astype(float)
        site_stay_denominators = stay_matrix.sum(axis=1)
        global_stays = int(stay_matrix.sum())
        global_patient_count = int(len(global_subjects))
        site_patient_memberships = int(site_patient_mask.sum())
        cross_site_patients = int((site_patient_mask.sum(axis=0) > 1).sum())
        denominator_fields = {
            "selected_site_count": int(selected_site_count),
            "feasible_site_count": int(len(sites)),
            "infeasible_site_count": int(selected_site_count - len(sites)),
            "feasible_assessment_stays": global_stays,
            "feasible_unique_assessment_patients": global_patient_count,
            "feasible_site_patient_memberships": site_patient_memberships,
            "global_cross_site_assessment_patients": cross_site_patients,
        }
        feasibility[str(budget)].update({
            "feasible_assessment_stays": global_stays,
            "feasible_unique_assessment_patients": global_patient_count,
            "feasible_site_patient_memberships": site_patient_memberships,
            "global_cross_site_assessment_patients": cross_site_patients,
        })

        for model in MODELS:
            for method in SITE_METHODS:
                if method in FITTED_SITE_METHODS:
                    fit_rows = site_draws.loc[
                        site_draws["hospitalid"].isin(sites)
                        & (site_draws["budget"] == budget)
                        & (site_draws["model_id"] == model)
                        & (site_draws["method"] == method)
                    ]
                    fit_attempts = int(len(fit_rows))
                    fit_successes = int(fit_rows["fit_ok"].sum())
                    fit_failures = fit_attempts - fit_successes
                    method_fit_counts = {
                        "fit_attempts": fit_attempts,
                        "fit_successes": fit_successes,
                        "fit_failures": fit_failures,
                        "fit_failure_rate": (
                            float(fit_failures / fit_attempts) if fit_attempts else math.nan
                        ),
                    }
                else:
                    method_fit_counts = {
                        "fit_attempts": math.nan, "fit_successes": math.nan,
                        "fit_failures": math.nan, "fit_failure_rate": math.nan,
                    }

                for metric in metrics:
                    loss_matrix = np.zeros_like(stay_matrix)
                    for site_index, site in enumerate(sites):
                        key = (site, int(budget), model, method, metric)
                        patient_loss = site_loss_sums.get(key)
                        if patient_loss is None:
                            raise ValueError(f"missing across-site loss accumulator {key}")
                        loss_matrix[site_index, site_patient_indexes[site_index]] = patient_loss

                    point_site_stay = loss_matrix.sum(axis=1) / (
                        REPEATS * site_stay_denominators
                    )
                    point_site_patient_losses = np.divide(
                        loss_matrix, stay_matrix, out=np.zeros_like(loss_matrix),
                        where=stay_matrix > 0,
                    ) / REPEATS
                    point_site_patient = (
                        point_site_patient_losses.sum(axis=1) / site_patient_denominators
                    )
                    point_global_patient_losses = (
                        loss_matrix.sum(axis=0)
                        / (REPEATS * stay_matrix.sum(axis=0))
                    )
                    point_estimates = {
                        ("micro", "stay_weighted"): float(
                            loss_matrix.sum() / (REPEATS * stay_matrix.sum())
                        ),
                        ("micro", "patient_weighted"): float(
                            point_global_patient_losses.mean()
                        ),
                        ("macro", "stay_weighted"): float(point_site_stay.mean()),
                        ("macro", "patient_weighted"): float(point_site_patient.mean()),
                    }

                    rng = np.random.default_rng(SEED + 50000 + int(budget))
                    bootstrap_values = {
                        (aggregation, weighting): []
                        for aggregation in aggregations for weighting in weightings
                    }
                    for _ in range(N_BOOT):
                        sampled = rng.integers(
                            0, global_patient_count, size=global_patient_count,
                            dtype=np.int32,
                        )
                        multiplicity = np.bincount(
                            sampled, minlength=global_patient_count,
                        ).astype(float)
                        sampled_site_stays = stay_matrix @ multiplicity
                        if np.any(sampled_site_stays <= 0):
                            continue
                        sampled_site_loss = (loss_matrix @ multiplicity) / REPEATS
                        sampled_site_patients = site_patient_mask @ multiplicity
                        if np.any(sampled_site_patients <= 0):
                            continue
                        sampled_global_patient_losses = (
                            loss_matrix.sum(axis=0)
                            / (REPEATS * stay_matrix.sum(axis=0))
                        )
                        bootstrap_values[("micro", "stay_weighted")].append(
                            float(sampled_site_loss.sum() / sampled_site_stays.sum())
                        )
                        bootstrap_values[("micro", "patient_weighted")].append(
                            float(np.dot(sampled_global_patient_losses, multiplicity)
                                  / multiplicity.sum())
                        )
                        bootstrap_values[("macro", "stay_weighted")].append(
                            float(np.mean(sampled_site_loss / sampled_site_stays))
                        )
                        sampled_site_patient_loss = point_site_patient_losses @ multiplicity
                        bootstrap_values[("macro", "patient_weighted")].append(
                            float(np.mean(sampled_site_patient_loss / sampled_site_patients))
                        )

                    for aggregation in aggregations:
                        for weighting in weightings:
                            values = np.asarray(
                                bootstrap_values[(aggregation, weighting)], dtype=float,
                            )
                            valid = int(len(values))
                            output_rows.append({
                                "budget": int(budget), "model": MODEL_LABELS[model],
                                "model_id": model, "method": method, "metric": metric,
                                "aggregation": aggregation, "weighting": weighting,
                                "status": (
                                    "complete_1000_global_patient_cluster_draws"
                                    if valid == N_BOOT
                                    else "incomplete_global_patient_cluster_draws"
                                ),
                                **denominator_fields,
                                "adaptation_repeats_averaged": REPEATS,
                                "estimate": point_estimates[(aggregation, weighting)],
                                "bootstrap_ci_95_lower": (
                                    float(np.percentile(values, 2.5)) if valid else math.nan
                                ),
                                "bootstrap_ci_95_upper": (
                                    float(np.percentile(values, 97.5)) if valid else math.nan
                                ),
                                "bootstrap_requested": N_BOOT,
                                "bootstrap_valid": valid,
                                "bootstrap_cluster_unit": "global patient; shared across sites",
                                **method_fit_counts,
                            })

    return pd.DataFrame(output_rows), feasibility


def _site_adaptation(fm: pd.DataFrame, lr: pd.DataFrame, roster: pd.DataFrame):
    usable = fm.loc[
        fm["eligible_24h"].astype(bool) & fm["temporal_verified"].astype(bool)
    ].copy()
    adapt_pool, assess_pool, _ = adaptation_assessment_pools(
        usable, GLOBAL_ROLE_SEED,
    )
    adapt_subjects = set(adapt_pool["subject"].astype(str))
    assess_subjects = set(assess_pool["subject"].astype(str))
    if adapt_subjects.intersection(assess_subjects):
        raise ValueError("global adaptation and assessment patient roles overlap")
    fm_use = usable.copy()
    lr_use = lr.loc[
        lr["eligible_24h"].astype(bool) & lr["temporal_verified"].astype(bool)
    ].copy()
    fm_use["hospitalid"] = fm_use["hospitalid"].map(_canon_id)
    lr_use["hospitalid"] = lr_use["hospitalid"].map(_canon_id)
    cross_site = fm_use.groupby("subject", observed=True)["hospitalid"].nunique()
    crossing_patients = int((cross_site > 1).sum())
    lr_index = lr_use.assign(_key=lr_use["stay_id"].map(_canon_id)).set_index("_key")
    draw_rows, summary_rows, role_rows = [], [], []
    site_loss_sums: dict[tuple[str, int, str, str, str], np.ndarray] = {}
    site_patient_info: dict[str, dict[str, Any]] = {}
    feasible_sites_by_budget: dict[int, list[str]] = {
        int(budget): [] for budget in SITE_BUDGETS
    }
    top_sites = roster.loc[roster["selected_top24"]].sort_values(
        "eligible_patient_rank_largest_first"
    )
    for site_row in top_sites.to_dict("records"):
        site = str(site_row["hospitalid"])
        adapt_all = fm_use.loc[
            (fm_use["hospitalid"] == site)
            & fm_use["subject"].astype(str).isin(adapt_subjects)
        ].copy()
        assess = fm_use.loc[
            (fm_use["hospitalid"] == site)
            & fm_use["subject"].astype(str).isin(assess_subjects)
        ].copy().reset_index(drop=True)
        if adapt_all.empty or assess.empty:
            role_rows.append({
                "hospitalid": site,
                "rank_by_eligible_patients": int(site_row["eligible_patient_rank_largest_first"]),
                "eligible_unique_patients_all": int(site_row["eligible_unique_patients"]),
                "global_adaptation_patients_in_site": int(adapt_all["subject"].nunique()),
                "global_assessment_patients_in_site": int(assess["subject"].nunique()),
                "global_assessment_stays_in_site": int(len(assess)),
                "patient_role_overlap": 0,
                "patients_crossing_hospitals_global_eicu": crossing_patients,
                "status": "no_patient_disjoint_adaptation_or_assessment_pool",
                "time_design": "random within-site adaptation; no temporal claim",
            })
            for model in MODELS:
                for budget in SITE_BUDGETS:
                    for method in SITE_METHODS:
                        for weighting in ("stay_weighted", "patient_weighted"):
                            fitted = method in FITTED_SITE_METHODS
                            summary_rows.append({
                                "hospitalid": site, "model": MODEL_LABELS[model],
                                "model_id": model, "budget": budget, "method": method,
                                "weighting": weighting,
                                "status": "no_patient_disjoint_adaptation_or_assessment_pool",
                                "feasible_site": False,
                                "available_adaptation_patients": int(adapt_all["subject"].nunique()),
                                "assessment_patients": int(assess["subject"].nunique()),
                                "assessment_stays": int(len(assess)),
                                "assessment_site_patient_memberships": int(assess["subject"].nunique()),
                                "fit_attempts": 0 if fitted else math.nan,
                                "fit_successes": 0 if fitted else math.nan,
                                "fit_failures": 0 if fitted else math.nan,
                                "fit_failure_rate": 0.0 if fitted else math.nan,
                            })
            continue
        if set(adapt_all["subject"].astype(str)).intersection(assess["subject"].astype(str)):
            raise ValueError(f"{site}: global patient roles overlap within site")
        index_stays = one_index_per_patient(adapt_all, _site_seed(site) + 101)
        index_stays = index_stays.sort_values(
            ["subject", "stay_id"], kind="stable"
        ).reset_index(drop=True)
        assess["_key"] = assess["stay_id"].map(_canon_id)
        lr_assess = lr_index.loc[assess["_key"]]
        if not np.array_equal(assess["y"].to_numpy(dtype=int), lr_assess["y"].to_numpy(dtype=int)):
            raise ValueError(f"{site}: FM/LR site assessment labels differ")
        if not np.array_equal(assess["subject"].astype(str), lr_assess["subject"].astype(str)):
            raise ValueError(f"{site}: FM/LR site assessment patients differ")
        y_eval = assess["y"].to_numpy(dtype=int)
        subjects_eval = assess["subject"].astype(str).to_numpy()
        _, site_inverse = np.unique(subjects_eval, return_inverse=True)
        site_patient_counts = np.bincount(site_inverse).astype(float)
        site_subjects = np.unique(subjects_eval)
        site_patient_info[site] = {
            "subjects": site_subjects,
            "stay_counts": site_patient_counts.astype(int),
        }
        raw_eval = {
            "main20k_real": _clip(assess["p"].to_numpy(dtype=float)),
            "conventional_lr_value_plus_pattern": _clip(lr_assess["p"].to_numpy(dtype=float)),
        }
        role_rows.append({
            "hospitalid": site,
            "rank_by_eligible_patients": int(site_row["eligible_patient_rank_largest_first"]),
            "eligible_unique_patients_all": int(site_row["eligible_unique_patients"]),
            "global_adaptation_patients_in_site": int(index_stays["subject"].nunique()),
            "global_assessment_patients_in_site": int(assess["subject"].nunique()),
            "global_assessment_stays_in_site": int(len(assess)),
            "patient_role_overlap": 0,
            "patients_crossing_hospitals_global_eicu": crossing_patients,
            "adaptation_index_stay": "one outcome-blind stay per adaptation patient within site",
            "assessment_rule": "all eligible stays of globally held-out patients",
            "index_stay_seed": _site_seed(site) + 101,
            "sampling_seed_by_budget": json.dumps({
                str(value): _site_seed(site) + value for value in SITE_BUDGETS
            }, sort_keys=True),
            "time_design": "random within-site adaptation; no temporal claim",
        })
        for budget in SITE_BUDGETS:
            if budget > len(index_stays):
                for model in MODELS:
                    for method in SITE_METHODS:
                        for weighting in ("stay_weighted", "patient_weighted"):
                            fitted = method in FITTED_SITE_METHODS
                            summary_rows.append({
                                "hospitalid": site, "model": MODEL_LABELS[model],
                                "model_id": model, "budget": budget, "method": method,
                                "weighting": weighting,
                                "status": "budget_exceeds_site_adaptation_pool",
                                "feasible_site": False,
                                "available_adaptation_patients": int(len(index_stays)),
                                "assessment_patients": int(assess["subject"].nunique()),
                                "assessment_stays": int(len(assess)),
                                "assessment_site_patient_memberships": int(assess["subject"].nunique()),
                                "fit_attempts": 0 if fitted else math.nan,
                                "fit_successes": 0 if fitted else math.nan,
                                "fit_failures": 0 if fitted else math.nan,
                                "fit_failure_rate": 0.0 if fitted else math.nan,
                            })
                continue
            feasible_sites_by_budget[int(budget)].append(site)
            for model in MODELS:
                for method in SITE_METHODS:
                    for metric in ("brier", "logloss"):
                        site_loss_sums[(site, int(budget), model, method, metric)] = np.zeros(
                            len(site_patient_counts), dtype=float,
                        )
            rng = np.random.default_rng(_site_seed(site) + budget)
            results = []
            successes = {model: {"intercept": 0, "platt": 0} for model in MODELS}
            failures = {model: {"intercept": 0, "platt": 0} for model in MODELS}
            for repeat in range(REPEATS):
                # Same patient sample for each model and method in this repeat.
                selected = index_stays.iloc[rng.permutation(len(index_stays))[:budget]]
                y_adapt = selected["y"].to_numpy(dtype=int)
                event_count = int(y_adapt.sum())
                prevalence = event_count / float(budget)
                jeffreys = (event_count + 0.5) / (budget + 1.0)
                stay_keys = selected["stay_id"].map(_canon_id)
                sample_hash = hashlib.sha256("\n".join(sorted(
                    selected["subject"].astype(str)
                )).encode("utf-8")).hexdigest()
                index_stay_hash = _identity_digest(
                    selected,
                    tuple(col for col in ("db", "subject", "stay_id", "rows", "split")
                          if col in selected.columns),
                )
                for model in MODELS:
                    p_eval = raw_eval[model]
                    p_adapt = (
                        _clip(selected["p"].to_numpy(dtype=float))
                        if model == "main20k_real"
                        else _clip(lr_index.loc[stay_keys, "p"].to_numpy(dtype=float))
                    )
                    predictions = {
                        "no_update": (p_eval, True, "not_applicable", False),
                        "prevalence": (np.full(len(y_eval), prevalence), True, "not_applicable", False),
                        "jeffreys_prevalence": (np.full(len(y_eval), jeffreys), True, "not_applicable", False),
                    }
                    for method, fit_fn in (("intercept", fit_offset), ("platt", fit_platt)):
                        try:
                            slope, intercept = fit_fn(
                                y_adapt, logit(p_adapt),
                            )
                            post = _clip(expit(slope * logit(p_eval) + intercept))
                            predictions[method] = (post, True, "ok", False)
                            successes[model][method] += 1
                        except (ValueError, RuntimeError, FloatingPointError, OverflowError, Warning) as exc:
                            failures[model][method] += 1
                            predictions[method] = (
                                np.full(len(y_eval), jeffreys), False,
                                f"{type(exc).__name__}:{str(exc)[:100]}", True,
                            )
                    for method, (pred, fit_ok, fit_status, fallback) in predictions.items():
                        scores = _metrics(
                            y_eval, pred, inverse=site_inverse, counts=site_patient_counts,
                        )
                        brier_losses, logloss_losses = _losses(y_eval, pred)
                        site_loss_sums[(site, int(budget), model, method, "brier")] += np.bincount(
                            site_inverse, weights=brier_losses,
                            minlength=len(site_patient_counts),
                        )
                        site_loss_sums[(site, int(budget), model, method, "logloss")] += np.bincount(
                            site_inverse, weights=logloss_losses,
                            minlength=len(site_patient_counts),
                        )
                        results.append({
                            "hospitalid": site,
                            "rank_by_eligible_patients": int(site_row["eligible_patient_rank_largest_first"]),
                            "model": MODEL_LABELS[model], "model_id": model,
                            "budget": int(budget), "repeat": int(repeat), "method": method,
                            "events": event_count, "nonevents": int(budget - event_count),
                            "fit_ok": bool(fit_ok), "fit_status": fit_status,
                            "fallback_used": bool(fallback),
                            "fallback_probability": jeffreys if fallback else math.nan,
                            "sample_patient_set_sha256": sample_hash,
                            "sample_patient_index_stay_sha256": index_stay_hash,
                            "sampling_seed": int(_site_seed(site) + budget),
                            "n_adaptation_patients": int(budget),
                            "assessment_patients": int(assess["subject"].nunique()),
                            "assessment_stays": int(len(assess)),
                            "assessment_events": int(y_eval.sum()),
                            **scores,
                        })
            draw_rows.extend(results)
            for model in MODELS:
                for method in SITE_METHODS:
                    subset = [row for row in results if row["model_id"] == model and row["method"] == method]
                    fitted = method in FITTED_SITE_METHODS
                    fit_successes = int(successes[model][method]) if fitted else math.nan
                    fit_failures = int(failures[model][method]) if fitted else math.nan
                    for weighting, bkey, lkey in (
                        ("stay_weighted", "brier", "logloss"),
                        ("patient_weighted", "brier_patient_weighted", "logloss_patient_weighted"),
                    ):
                        brier = np.asarray([row[bkey] for row in subset], dtype=float)
                        logloss = np.asarray([row[lkey] for row in subset], dtype=float)
                        summary_rows.append({
                            "hospitalid": site,
                            "rank_by_eligible_patients": int(site_row["eligible_patient_rank_largest_first"]),
                            "model": MODEL_LABELS[model], "model_id": model,
                            "budget": int(budget), "method": method, "weighting": weighting,
                            "status": "complete_200_repeats", "feasible_site": True,
                            "available_adaptation_patients": int(len(index_stays)),
                            "assessment_patients": int(assess["subject"].nunique()),
                            "assessment_stays": int(len(assess)),
                            "assessment_site_patient_memberships": int(assess["subject"].nunique()),
                            "assessment_events": int(y_eval.sum()),
                            "assessment_nonevents": int(len(y_eval) - y_eval.sum()),
                            "fit_attempts": int(REPEATS) if fitted else math.nan,
                            "fit_successes": fit_successes,
                            "fit_failures": fit_failures,
                            "fit_failure_rate": (
                                float(fit_failures / REPEATS) if fitted else math.nan
                            ),
                            "brier_mean": float(brier.mean()),
                            "brier_p05": float(np.percentile(brier, 5)),
                            "brier_p95": float(np.percentile(brier, 95)),
                            "logloss_mean": float(logloss.mean()),
                            "logloss_p05": float(np.percentile(logloss, 5)),
                            "logloss_p95": float(np.percentile(logloss, 95)),
                            "adaptation_draws": int(REPEATS),
                            "adaptation_event_p05": float(np.percentile([r["events"] for r in subset], 5)),
                            "adaptation_event_p95": float(np.percentile([r["events"] for r in subset], 95)),
                        })
            print(json.dumps({
                "stage": "site_budget_complete", "hospitalid": site,
                "budget": int(budget), "repeats": int(REPEATS),
                "adaptation_patients": int(len(index_stays)),
            }, sort_keys=True), flush=True)
    draw_frame = pd.DataFrame(draw_rows)
    summary_frame = pd.DataFrame(summary_rows)
    across_summary, across_feasibility = _across_site_summary(
        draw_frame, site_loss_sums, site_patient_info, feasible_sites_by_budget,
        selected_site_count=int(len(top_sites)),
    )
    roles = {
        "global_adaptation_role_sha256": hashlib.sha256(
            "\n".join(sorted(adapt_subjects)).encode("utf-8")
        ).hexdigest(),
        "global_assessment_role_sha256": hashlib.sha256(
            "\n".join(sorted(assess_subjects)).encode("utf-8")
        ).hexdigest(),
        "global_role_seed_reused_from_v4": int(GLOBAL_ROLE_SEED),
        "global_role_split": "patient-level random half split; no outcome stratification",
        "patients_crossing_hospitals": crossing_patients,
        "selected_site_flow": role_rows,
        "across_site_feasibility": across_feasibility,
        "across_site_uncertainty_scope": (
            "1000 global patient-cluster draws shared across feasible sites; "
            "cross-site patients retain one shared bootstrap multiplicity; "
            "all 200 adaptation repeats are averaged before resampling"
        ),
        "time_design": "random within-site adaptation; no temporal validation claim",
        "temporal_claim_status": "not performed; no comparable cross-patient calendar time in frozen prediction roles",
    }
    return draw_frame, summary_frame, across_summary, roles


def _curve_result(cohort, model, method, y, p, subjects, seed, fallback=False):
    def progress(done, total, valid):
        print(json.dumps({
            "stage": "calibration_curve_bootstrap_progress", "cohort": cohort,
            "model": MODEL_LABELS[model], "method": method,
            "completed": int(done), "total": int(total), "valid": int(valid),
        }, sort_keys=True), flush=True)

    result = calibration_curve_cluster(
        subjects, y, p, draws=N_BOOT, seed=seed, grid_points=61, progress=progress,
    )
    status = str(result.get("status", "unknown"))
    support = result.get("probability_support", [])
    if not support:
        support = [float(np.quantile(p, .01)), float(np.quantile(p, .99))]
    risk = {
        "cohort": cohort, "model": MODEL_LABELS[model], "model_id": model,
        "method": method, "curve_status": status,
        "n_stays": int(result.get("n_stays", len(y))),
        "n_patients": int(result.get("n_patients", len(np.unique(subjects)))),
        "events": int(result.get("events", np.sum(y))),
        "nonevents": int(len(y) - np.sum(y)),
        "mean_predicted_probability": float(result.get("mean_predicted", np.mean(p))),
        "observed_event_rate": float(result.get("observed", np.mean(y))),
        "brier": float(np.mean((np.asarray(y) - np.asarray(p)) ** 2)),
        "logloss": float(np.mean(_losses(np.asarray(y), np.asarray(p))[1])),
        "p_min": float(np.min(p)), "p01": float(np.quantile(p, .01)),
        "p05": float(np.quantile(p, .05)), "p25": float(np.quantile(p, .25)),
        "p50": float(np.quantile(p, .5)), "p75": float(np.quantile(p, .75)),
        "p95": float(np.quantile(p, .95)), "p99": float(np.quantile(p, .99)),
        "p_max": float(np.max(p)),
        "smoother_predictor_df": result.get("predictor_df", 2),
        "smoother_coefficient_count": result.get("coefficient_count", 3),
        "smoother_coefficients": json.dumps(result.get("beta", [])),
        "knots_logit": json.dumps(result.get("knots_logit", [])),
        "probability_support": json.dumps(support),
        "bootstrap_requested": int(result.get("bootstrap_draws", N_BOOT)),
        "bootstrap_valid": int(result.get("valid_bootstrap", 0)),
        "bootstrap_invalid": int(result.get("invalid_bootstrap", N_BOOT)),
        "bootstrap_failures": json.dumps(result.get("failures", {}), sort_keys=True),
        "fixed_calibrator_fallback": bool(fallback),
        "assessment_scope": result.get("scope", "fixed assessment scores and patient-cluster bootstrap"),
    }
    bins = [{
        "cohort": cohort, "model": MODEL_LABELS[model], "model_id": model,
        "method": method, **row,
    } for row in result.get("risk_bins", [])]
    if not bins:
        bins.append({
            "cohort": cohort, "model": MODEL_LABELS[model], "model_id": model,
            "method": method, "bin": 0, "n_stays": int(len(y)),
            "events": int(np.sum(y)), "mean_predicted": float(np.mean(p)),
            "observed": float(np.mean(y)),
        })
    curve_rows = []
    grid = result.get("grid", [])
    curve = result.get("curve", [])
    low = result.get("low", [])
    high = result.get("high", [])
    if len(grid) and len(curve) == len(grid):
        for i, value in enumerate(grid):
            curve_rows.append({
                "cohort": cohort, "model": MODEL_LABELS[model], "model_id": model,
                "method": method, "curve_type": "restricted_cubic_logistic",
                "point_index": int(i), "predicted_probability": float(value),
                "smoothed_observed_risk": float(curve[i]),
                "ci95_lower": float(low[i]) if len(low) == len(grid) else math.nan,
                "ci95_upper": float(high[i]) if len(high) == len(grid) else math.nan,
                "risk_support_low": float(support[0]), "risk_support_high": float(support[1]),
                "smoother_predictor_df": result.get("predictor_df", 2),
                "smoother_coefficient_count": result.get("coefficient_count", 3),
                "knot_logit_q10": float(result.get("knots_logit", [math.nan] * 3)[0]),
                "knot_logit_q50": float(result.get("knots_logit", [math.nan] * 3)[1]),
                "knot_logit_q90": float(result.get("knots_logit", [math.nan] * 3)[2]),
                "bootstrap_requested": int(result.get("bootstrap_draws", N_BOOT)),
                "bootstrap_valid": int(result.get("valid_bootstrap", 0)),
                "bootstrap_invalid": int(result.get("invalid_bootstrap", N_BOOT)),
                "curve_fit_status": status,
            })
    elif status in {"constant_or_insufficient_score_support", "degenerate_knots"} or fallback:
        point_ci = paired_cluster_mean_ci(
            np.asarray(subjects), np.asarray(y, dtype=float), draws=N_BOOT,
            seed=seed + 500000, patient_weighted=False,
        )
        point_status = "jeffreys_fallback_reference" if fallback else "constant_or_degenerate_reference"
        curve_rows.append({
            "cohort": cohort, "model": MODEL_LABELS[model], "model_id": model,
            "method": method, "curve_type": point_status, "point_index": 0,
            "predicted_probability": float(np.mean(p)),
            "smoothed_observed_risk": float(np.mean(y)),
            "ci95_lower": float(point_ci["low"]), "ci95_upper": float(point_ci["high"]),
            "risk_support_low": float(np.min(p)), "risk_support_high": float(np.max(p)),
            "smoother_predictor_df": 0, "smoother_coefficient_count": 0,
            "knot_logit_q10": math.nan, "knot_logit_q50": math.nan,
            "knot_logit_q90": math.nan,
            "bootstrap_requested": int(N_BOOT),
            "bootstrap_valid": int(point_ci["valid_draws"]),
            "bootstrap_invalid": int(N_BOOT - point_ci["valid_draws"]),
            "curve_fit_status": point_status,
        })
    else:
        curve_rows.append({
            "cohort": cohort, "model": MODEL_LABELS[model], "model_id": model,
            "method": method, "curve_type": "unavailable", "point_index": 0,
            "predicted_probability": math.nan, "smoothed_observed_risk": math.nan,
            "ci95_lower": math.nan, "ci95_upper": math.nan,
            "risk_support_low": float(support[0]), "risk_support_high": float(support[1]),
            "smoother_predictor_df": result.get("predictor_df", 2),
            "smoother_coefficient_count": result.get("coefficient_count", 3),
            "knot_logit_q10": math.nan, "knot_logit_q50": math.nan,
            "knot_logit_q90": math.nan, "bootstrap_requested": N_BOOT,
            "bootstrap_valid": int(result.get("valid_bootstrap", 0)),
            "bootstrap_invalid": int(N_BOOT - result.get("valid_bootstrap", 0)),
            "curve_fit_status": status,
        })
    return curve_rows, bins, risk


def _write_table(frame: pd.DataFrame, path: Path, description: str, schema: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)
    schema[path.name] = {
        "description": description,
        "rows": int(len(frame)),
        "sha256": sha256_file(path),
        "columns": [{"name": str(col), "dtype": str(frame[col].dtype)} for col in frame.columns],
    }


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    _write_text_atomic(
        path, json.dumps(document, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )


def _write_calibration_checkpoint(
    db: str,
    repeats: pd.DataFrame,
    gap_draws: pd.DataFrame,
    gap_ci: pd.DataFrame,
    parity: pd.DataFrame,
    curve_rows: list[dict[str, Any]],
    bin_rows: list[dict[str, Any]],
    risk_rows: list[dict[str, Any]],
    direct_meta: dict[str, Any],
) -> None:
    """Atomically save aggregate results after each cohort's curves finish."""
    checkpoint_dir = CAL_OUT / "checkpoints" / db
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _progress_event("calibration_checkpoint_start", cohort=db)
    schema: dict[str, Any] = {}
    outputs = [
        ("global_recalibration_repeat_scores", repeats,
         "Cohort-level adaptation repeat metrics; aggregate only."),
        ("paired_loss_bootstrap_draws", gap_draws,
         "Aggregate repeat-level raw, Platt, and gap-change loss contrasts."),
        ("paired_loss_gap_bootstrap", gap_ci,
         "Aggregate patient-cluster intervals for paired loss contrasts."),
        ("per_stay_repeat_mean_parity", parity,
         "Aggregate checks that saved loss means match frozen fit replay."),
        ("calibration_curve_points", pd.DataFrame(curve_rows),
         "Aggregate calibration curve points and patient-cluster intervals."),
        ("calibration_reliability_bins", pd.DataFrame(bin_rows),
         "Aggregate calibration-bin support and observed risks."),
        ("calibration_curve_risk_support", pd.DataFrame(risk_rows),
         "Aggregate probability support and curve-fit diagnostics."),
    ]
    for name, frame, description in outputs:
        _write_table(
            frame, checkpoint_dir / f"{name}.csv", description, schema,
        )
    manifest = {
        "status": "cohort_complete_checkpoint",
        "stage": "global_calibration_cohort_complete",
        "cohort": db,
        "models": list(MODELS),
        "global_sampling_plan_sha256": direct_meta["sampling_plan_sha256"],
        "contains_patient_level_identifiers": False,
        "files": schema,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    path = checkpoint_dir / "checkpoint_manifest.json"
    _write_json_atomic(path, manifest)
    _progress_event(
        "calibration_checkpoint_done", cohort=db,
        manifest=str(path), outputs=len(outputs),
    )


def _load_calibration_checkpoint(
    db: str, expected_manifest_sha256: str, direct_meta: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Load only an explicitly authorized, intact checkpoint for this direct input."""
    checkpoint_dir = CAL_OUT / "checkpoints" / db
    manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError(f"{db}: checkpoint manifest hash differs from explicit resume authorization")
    manifest = _json(manifest_path)
    expected_files = {f"{name}.csv" for name in CALIBRATION_CHECKPOINT_TABLES}
    files = manifest.get("files")
    if (manifest.get("status") != "cohort_complete_checkpoint"
            or manifest.get("stage") != "global_calibration_cohort_complete"
            or manifest.get("cohort") != db
            or manifest.get("models") != list(MODELS)
            or manifest.get("contains_patient_level_identifiers") is not False
            or manifest.get("global_sampling_plan_sha256") != direct_meta["sampling_plan_sha256"]
            or not isinstance(files, dict) or set(files) != expected_files):
        raise ValueError(f"{db}: checkpoint manifest metadata is incomplete or incompatible")
    actual_files = {path.name for path in checkpoint_dir.glob("*.csv")}
    if actual_files != expected_files:
        raise ValueError(f"{db}: checkpoint CSV inventory differs from its seven-file schema")

    restored: dict[str, pd.DataFrame] = {}
    for table_name in CALIBRATION_CHECKPOINT_TABLES:
        filename = f"{table_name}.csv"
        path = checkpoint_dir / filename
        entry = files[filename]
        expected_sha = entry.get("sha256") if isinstance(entry, dict) else None
        expected_rows = entry.get("rows") if isinstance(entry, dict) else None
        if (not isinstance(expected_sha, str)
                or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None
                or type(expected_rows) is not int or expected_rows < 0
                or sha256_file(path) != expected_sha):
            raise ValueError(f"{db}: checkpoint table hash or row metadata is invalid for {filename}")
        frame = pd.read_csv(path)
        column_schema = entry.get("columns")
        if (len(frame) != expected_rows or not isinstance(column_schema, list)
                or [str(item.get("name")) for item in column_schema
                    if isinstance(item, dict)] != [str(col) for col in frame.columns]
                or sha256_file(path) != expected_sha):
            raise ValueError(f"{db}: checkpoint table schema, rows, or stable hash failed for {filename}")
        if "cohort" in frame.columns and not frame.empty:
            if not frame["cohort"].astype(str).eq(db).all():
                raise ValueError(f"{db}: checkpoint table contains another cohort in {filename}")
        restored[table_name] = frame

    last_event = None
    for line in LEGACY_DIRECT_REUSE_LOG.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (isinstance(event, dict)
                and event.get("stage") == "direct_calibration_reuse_complete"
                and event.get("cohort") == db):
            last_event = event
    if (last_event is None
            or last_event.get("fit_parameter_hash") != direct_meta["fit_parameters_sha256"]
            or last_event.get("per_stay_loss_hash") != direct_meta["per_stay_loss_sha256"]):
        raise ValueError(f"{db}: legacy completion log does not attest the current direct inputs")
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError(f"{db}: checkpoint manifest changed while it was being loaded")
    return restored


def _readme(title: str, schema: dict[str, Any]) -> str:
    lines = [
        f"# V5 {title} analysis",
        "",
        "All patient-level scores and identifiers remain on the server. CSVs contain cohort-, model-, site- or repeat-level aggregates and no patient-level records.",
        "",
        "Calibration curves use raw scores and the exact server-only Platt fit at 500 labels, repeat 0, on the reused v4.2 patient roles. The diagnostic smoother has three knots at assessment logit-score q10/q50/q90, two predictor degrees of freedom plus intercept (three coefficients), and 1000 patient-cluster bootstrap draws. Curves are limited to empirical p01-p99 support. Constant scores and Jeffreys fallbacks appear as a point reference.",
        "",
        "Brier and logloss are both reported. P05/P95 across 200 adaptation draws describe repeat variability on the fixed assessment pool, separately from the patient bootstrap confidence interval. Paired gaps use the per-stay loss means across all 200 fits, followed by a shared patient-cluster bootstrap; patient-weighted estimates are a sensitivity analysis.",
        "",
        "Cohort-level atomic checkpoints are retained under checkpoints/. The tables above are canonical only when the section jobmanifest reports status=complete.",
        "",
        f"Runtime limits: 2 CPU threads, 200% CPU quota, {int(os.environ.get('V5_HOST_MEMORY_MAX_BYTES', str(3 * 1024 ** 3))) // 1024**3} GiB unit MemoryMax, and 0 swap. Resource scope: {RESOURCE_SCOPE}; actual cgroup path: {current_cgroup_path()}. The global revision-v5 science lock serializes analyses.",
        "",
        "Hospitals are selected only by eligible unique-patient count (largest 24; ascending canonical ID breaks ties). Global patient roles are locked before within-site sampling. Site results represent random within-site adaptation and make no temporal validation claim.",
        "",
        "For the 206-hospital basic performance table, finite calibration-slope point coverage is distinct from the >=10-event and >=10-nonevent interval-stability rule. The rule does not exclude sites from point estimates or site selection.",
        "",
        "## CSV schemas",
        "",
    ]
    if title == "calibration":
        lines[8:8] = [
            "The HGBT supplementary table is retained from verified v4 all-200-draw aggregate outputs for eICU and NWICU. Offset methods display as intercept; the original archive method is preserved in source_archive_method. These are historical summaries and were not recomputed on repaired-v5 scores.",
            "",
        ]
    if title == "sites":
        lines[8:8] = [
            "Across-site estimates use the same feasible site set within each budget. Micro summaries pool stays or unique global patients; macro summaries give each feasible site equal weight. Intervals resample global eICU patient clusters, so a patient who appears at multiple hospitals receives one shared bootstrap multiplicity.",
            "",
            "Each across-site row reports feasible-site, assessment-stay, unique-patient and site-patient-membership denominators. Infeasible site cells retain both stay- and patient-weighting rows. The historical v4 site coverage reference is sourced from the original verified v4 metrics and manifest.",
            "",
        ]
    for name, entry in schema.items():
        lines.extend([
            f"### {name}", "", entry["description"], "",
            f"Rows: {entry['rows']}; SHA256: {entry['sha256']}.", "",
            "| Column | Type |", "|---|---|",
        ])
        lines.extend([f"| {col['name']} | {col['dtype']} |" for col in entry["columns"]])
        lines.append("")
    return "\n".join(lines) + "\n"


def run(resume_checkpoints: dict[str, str] | None = None) -> None:
    """Run global calibration diagnostics and fixed hospital adaptation."""
    resume_checkpoints = {} if resume_checkpoints is None else resume_checkpoints
    started = time.time()
    CAL_OUT.mkdir(parents=True, exist_ok=True)
    SITE_OUT.mkdir(parents=True, exist_ok=True)
    _progress_event(
        "analysis_started",
        seed=SEED,
        global_role_seed_reused_from_v4=GLOBAL_ROLE_SEED,
    )
    frames, gates, endpoints, provenance = ta._load_all_model_frames()
    if provenance.get("patient_level_exported_to_local") is not False:
        raise ValueError("source provenance does not attest server-only patient-level inputs")
    hgbt_supplement, hgbt_sources = _load_hgbt_calibration_supplement(endpoints)
    original_v4_site_reference = _verify_original_v4_site_reference()
    _progress_event(
        "historical_v4_supplements_verified",
        original_v4_site_counts={
            "hospital_denominator": original_v4_site_reference["hospital_denominator"],
            "finite_slope_points": original_v4_site_reference["finite_calibration_slope_points"],
            "stable_sites": original_v4_site_reference[
                "stable_sites_ge10_events_and_nonevents"
            ],
        },
        hgbt_rows=int(len(hgbt_supplement)),
        hgbt_source_hashes={db: item["sha256"] for db, item in hgbt_sources.items()},
    )
    _progress_event(
        "frozen_scores_loaded", cohorts=list(endpoints),
        row_counts={model: {db: int(len(frame)) for db, frame in cohort_frames.items()}
                    for model, cohort_frames in frames.items()},
    )

    calibration_schema: dict[str, Any] = {}
    site_schema: dict[str, Any] = {}
    all_repeats, all_gaps, all_gap_ci, all_parity = [], [], [], []
    all_curves, all_bins, all_risks = [], [], []
    direct_artifacts: dict[str, Any] = {}

    for db in COHORTS:
        _progress_event("cohort_calibration_start", cohort=db)
        fm = frames["main20k_real"][db]
        lr = frames["conventional_lr_value_plus_pattern"][db]
        _validate_pair(fm, lr, db)
        model_frames = {model: frames[model][db] for model in MODELS}
        _progress_event("direct_calibration_reuse_start", cohort=db)
        arrays, params, assessment, direct_meta = _load_direct(db, fm, model_frames)
        hgbt_source = hgbt_sources[db]
        if (hgbt_source["sampling_plan_sha256"] != direct_meta["sampling_plan_sha256"]
                or hgbt_source["assessment_patients"] != direct_meta["assessment_patients"]
                or hgbt_source["assessment_stays"] != direct_meta["assessment_stays"]):
            raise ValueError(f"{db}: v4 HGBT supplement roles differ from the frozen v5 assessment plan")
        _progress_event(
            "direct_calibration_reuse_loaded", cohort=db,
            assessment_patients=direct_meta["assessment_patients"],
            assessment_stays=direct_meta["assessment_stays"],
            fit_parameter_rows=int(len(params)),
        )
        direct_artifacts[db] = direct_meta
        if db in resume_checkpoints:
            restored = _load_calibration_checkpoint(
                db, resume_checkpoints[db], direct_meta,
            )
            all_repeats.append(restored["global_recalibration_repeat_scores"])
            all_gaps.append(restored["paired_loss_bootstrap_draws"])
            all_gap_ci.append(restored["paired_loss_gap_bootstrap"])
            all_parity.append(restored["per_stay_repeat_mean_parity"])
            all_curves.extend(
                restored["calibration_curve_points"].to_dict(orient="records")
            )
            all_bins.extend(
                restored["calibration_reliability_bins"].to_dict(orient="records")
            )
            all_risks.extend(
                restored["calibration_curve_risk_support"].to_dict(orient="records")
            )
            _progress_event(
                "checkpoint_reused", cohort=db,
                manifest_sha256=resume_checkpoints[db],
                input_fit_parameter_hash=direct_meta["fit_parameters_sha256"],
                input_per_stay_loss_hash=direct_meta["per_stay_loss_sha256"],
                sampling_plan_sha256=direct_meta["sampling_plan_sha256"],
                tables=len(CALIBRATION_CHECKPOINT_TABLES),
            )
            del arrays, params
            continue
        _progress_event("global_repeat_analysis_start", cohort=db)
        repeats, gap_draws, gap_ci, parity = _global_outputs(
            db, arrays, params, assessment, frames,
        )
        _progress_event(
            "global_repeat_analysis_done", cohort=db,
            repeat_rows=int(len(repeats)), gap_draw_rows=int(len(gap_draws)),
            parity_rows=int(len(parity)),
        )
        all_repeats.append(repeats)
        all_gaps.append(gap_draws)
        all_gap_ci.append(gap_ci)
        all_parity.append(parity)

        # Main curves use the exact saved Platt coefficients at 500 labels,
        # repeat 0; the curve helper never refits the adaptation calibrator.
        maps = {model: _parameter_map(params, db, model) for model in MODELS}
        y = assessment["y"].to_numpy(dtype=int)
        subject = assessment["subject"].astype(str).to_numpy()
        for model in MODELS:
            use = frames[model][db].loc[
                frames[model][db]["eligible_24h"].astype(bool)
                & frames[model][db]["temporal_verified"].astype(bool)
            ]
            assess_model = use.set_index("stay_id").loc[assessment["stay_id"]]
            raw = _clip(assess_model["p"].to_numpy(dtype=float))
            fit = maps[model][(500, 0, "platt")]
            post, fit_ok, fit_status, fallback = _predict_fit(raw, fit, 500)
            for method, probabilities, used_fallback in (
                ("raw", raw, False),
                ("platt_500_repeat0", post, fallback),
            ):
                point_rows, bin_rows, risk = _curve_result(
                    db, model, method, y, probabilities, subject,
                    SEED + (COHORTS.index(db) + 1) * 1000
                    + MODELS.index(model) * 100
                    + (0 if method == "raw" else 1),
                    fallback=used_fallback,
                )
                risk.update({
                    "fixed_calibrator_budget": 0 if method == "raw" else 500,
                    "fixed_calibrator_repeat": -1 if method == "raw" else 0,
                    "platt_fit_ok": True if method == "raw" else bool(fit_ok),
                    "platt_fit_status": "not_applicable" if method == "raw" else fit_status,
                    "global_sampling_plan_sha256": direct_meta["sampling_plan_sha256"],
                })
                all_curves.extend(point_rows)
                all_bins.extend(bin_rows)
                all_risks.append(risk)
            _progress_event(
                "calibration_model_complete", cohort=db,
                model=MODEL_LABELS[model], platt500_repeat0_fit_ok=bool(fit_ok),
                platt500_repeat0_fallback=bool(fallback),
            )
        _write_calibration_checkpoint(
            db, repeats, gap_draws, gap_ci, parity,
            [row for row in all_curves if row.get("cohort") == db],
            [row for row in all_bins if row.get("cohort") == db],
            [row for row in all_risks if row.get("cohort") == db],
            direct_meta,
        )
        _progress_event(
            "direct_calibration_reuse_complete", cohort=db,
            assessment_patients=direct_meta["assessment_patients"],
            assessment_stays=direct_meta["assessment_stays"],
            fit_parameter_hash=direct_meta["fit_parameters_sha256"],
            per_stay_loss_hash=direct_meta["per_stay_loss_sha256"],
            repeat_mean_parity_rows=int(len(parity)),
        )
        del arrays, params

    repeat_frame = pd.concat(all_repeats, ignore_index=True)
    gap_draw_frame = pd.concat(all_gaps, ignore_index=True)
    gap_ci_frame = pd.concat(all_gap_ci, ignore_index=True)
    parity_frame = pd.concat(all_parity, ignore_index=True)
    _write_table(
        repeat_frame, CAL_OUT / "global_recalibration_repeat_scores.csv",
        "All 200 repeat-level raw, prevalence, Jeffreys, intercept and Platt results for each available budget; Brier/logloss and patient-weighted sensitivity.",
        calibration_schema,
    )
    summary_50_500 = (
        repeat_frame.loc[repeat_frame["budget"].isin([50, 500])]
        .groupby(["cohort", "model", "model_id", "budget", "method"], dropna=False)
        .agg(
            attempts=("repeat", "nunique"),
            fit_successes=("fit_ok", "sum"),
            fallback_count=("fallback_used", "sum"),
            events_p05=("events", lambda x: float(np.percentile(x, 5))),
            events_p95=("events", lambda x: float(np.percentile(x, 95))),
            brier_mean=("brier", "mean"),
            brier_p05=("brier", lambda x: float(np.percentile(x, 5))),
            brier_p95=("brier", lambda x: float(np.percentile(x, 95))),
            logloss_mean=("logloss", "mean"),
            logloss_p05=("logloss", lambda x: float(np.percentile(x, 5))),
            logloss_p95=("logloss", lambda x: float(np.percentile(x, 95))),
        ).reset_index()
    )
    fitted_summary_rows = summary_50_500["method"].isin({"intercept", "platt"})
    summary_50_500["fit_attempts"] = np.where(
        fitted_summary_rows, summary_50_500["attempts"], math.nan,
    )
    summary_50_500["fit_failures"] = np.where(
        fitted_summary_rows,
        summary_50_500["attempts"] - summary_50_500["fit_successes"],
        math.nan,
    )
    summary_50_500["fit_failure_rate"] = np.where(
        fitted_summary_rows,
        summary_50_500["fit_failures"] / summary_50_500["fit_attempts"],
        math.nan,
    )
    summary_50_500["fit_successes"] = summary_50_500["fit_successes"].where(
        fitted_summary_rows, math.nan,
    )
    _write_table(
        summary_50_500, CAL_OUT / "global_50_500_summary.csv",
        "Complete 200-draw results at 50 and 500 labels, both proper scores, adaptation event P05/P95 and failed-fit/fallback counts.",
        calibration_schema,
    )
    _write_table(
        hgbt_supplement, CAL_OUT / "hgbt_supplementary_calibration_v4.csv",
        "Verified v4 pooled eICU/NWICU HGBT calibration summaries across 200 adaptation draws; offset is displayed as intercept and preserved in source_archive_method. Historical aggregate only, not recomputed on repaired-v5 scores or site-level.",
        calibration_schema,
    )
    _write_table(
        gap_ci_frame, CAL_OUT / "paired_loss_gap_bootstrap.csv",
        "Raw gap, Platt gap, and post-gap minus raw-gap for Brier/logloss using saved per-stay means across all 200 fits; 1000 paired patient-cluster bootstraps and patient-weighted sensitivity.",
        calibration_schema,
    )
    _write_table(
        parity_frame, CAL_OUT / "per_stay_repeat_mean_parity.csv",
        "Parity checks that saved fit parameters recreate each server-only per-stay mean-loss array across 200 fits.",
        calibration_schema,
    )
    _write_table(
        pd.DataFrame(all_curves), CAL_OUT / "calibration_curve_points.csv",
        "Smooth risk curves and pointwise patient-cluster intervals for raw predictions and the fixed 500-label Platt repeat-0 fit.",
        calibration_schema,
    )
    _write_table(
        pd.DataFrame(all_bins), CAL_OUT / "calibration_risk_bins.csv",
        "Empirical score-bin support, event counts, mean predictions and observed event fractions underlying the smooth curves.",
        calibration_schema,
    )
    _write_table(
        pd.DataFrame(all_risks), CAL_OUT / "calibration_risk_summary.csv",
        "Risk distribution, mean predicted and observed risks, Brier/logloss, curve status and fixed-fit provenance per model/cohort.",
        calibration_schema,
    )
    _write_text_atomic(CAL_OUT / "README.md", _readme("calibration", calibration_schema))
    schema_path_cal = CAL_OUT / "schema.json"
    _write_json_atomic(schema_path_cal, calibration_schema)

    eicu_fm = frames["main20k_real"]["eicu_semantic_v4_2"]
    eicu_lr = frames["conventional_lr_value_plus_pattern"]["eicu_semantic_v4_2"]
    _validate_pair(eicu_fm, eicu_lr, "eicu_semantic_v4_2")
    roster, site_performance, site_audit = _site_inventory(eicu_fm, eicu_lr)
    site_audit["original_v4_site_reference"] = original_v4_site_reference
    original_v4_reference_row = {
        key: value for key, value in original_v4_site_reference.items()
        if not key.endswith("_path_internal")
    }
    _progress_event(
        "site_inventory_complete", eligible_hospitals=int(len(roster)),
        selected_hospitals=int(site_audit["selected_hospitals"]),
    )
    _write_table(
        roster, SITE_OUT / "site_roster.csv",
        "All 206 repaired eICU hospitals ranked by eligible unique-patient counts; the largest 24 are selected with an ascending hospital ID tie-break.",
        site_schema,
    )
    _write_table(
        site_performance, SITE_OUT / "site_basic_performance.csv",
        "Recomputed basic FM and V+P LR metrics at all 206 repaired eICU hospitals; all point estimates retained regardless of event count.",
        site_schema,
    )
    _write_table(
        pd.DataFrame([original_v4_reference_row]),
        SITE_OUT / "original_v4_site_reference.csv",
        "Historical eICU hospital denominator, finite calibration-slope points and stability-flag count from the verified original v4 metrics and manifest.",
        site_schema,
    )
    site_checkpoint = SITE_OUT / "checkpoints" / "site_inventory_checkpoint.json"
    _write_json_atomic(site_checkpoint, {
        "status": "site_inventory_checkpoint",
        "stage": "site_inventory_complete_before_adaptation_repeats",
        "eligible_hospitals": int(len(roster)),
        "selected_hospitals": int(site_audit["selected_hospitals"]),
        "original_v4_site_reference": original_v4_site_reference,
        "tables": {
            "site_roster.csv": site_schema["site_roster.csv"],
            "site_basic_performance.csv": site_schema["site_basic_performance.csv"],
            "original_v4_site_reference.csv": site_schema["original_v4_site_reference.csv"],
        },
        "contains_patient_level_identifiers": False,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    })
    _progress_event(
        "site_inventory_checkpoint_done", manifest=str(site_checkpoint),
    )
    _progress_event("site_adaptation_start", selected_hospitals=int(site_audit["selected_hospitals"]))
    site_draws, site_summary, across_site_summary, site_roles = _site_adaptation(
        eicu_fm, eicu_lr, roster,
    )
    _progress_event(
        "site_adaptation_done", draw_rows=int(len(site_draws)),
        summary_rows=int(len(site_summary)), across_site_rows=int(len(across_site_summary)),
    )
    _write_table(
        site_draws, SITE_OUT / "site_adaptation_repeat_scores.csv",
        "Full repeat-level 50/200/500 patient results for selected hospitals, both models, no-update/prevalence/Jeffreys/intercept/Platt, and stay- plus patient-weighted Brier/logloss.",
        site_schema,
    )
    _write_table(
        site_summary, SITE_OUT / "site_adaptation_summary.csv",
        "Per-site, model, budget, method and weighting summary; infeasible cells retain both weighting rows and explicit adaptation/assessment denominators. Fit counts apply only to offset/intercept and Platt fits.",
        site_schema,
    )
    _write_table(
        across_site_summary, SITE_OUT / "across_site_adaptation_summary.csv",
        "Micro and macro stay- and patient-weighted losses over the same feasible site set at each budget, with explicit site/stay/patient denominators and 1000 shared global patient-cluster intervals.",
        site_schema,
    )
    role_frame = pd.DataFrame(site_roles["selected_site_flow"])
    _write_table(
        role_frame, SITE_OUT / "site_role_flow.csv",
        "Selected-site counts showing globally locked patient roles, adaptation and assessment support, and no temporal design.",
        site_schema,
    )
    _write_text_atomic(SITE_OUT / "README.md", _readme("sites", site_schema))
    schema_path_site = SITE_OUT / "schema.json"
    _write_json_atomic(schema_path_site, site_schema)

    script_path = Path(__file__).resolve()
    inputs_internal = {
        "v4_loader_provenance": provenance,
        "direct_server_only_artifacts": direct_artifacts,
        "verified_hgbt_calibration_supplement_v4": hgbt_sources,
        "eligibility_eicu_path": str(GATE_ROOT / "eicu.parquet"),
        "eligibility_eicu_sha256": sha256_file(GATE_ROOT / "eicu.parquet"),
        "eligibility_eicu_semantic_path": str(GATE_ROOT / "eicu_semantic_v4_2.parquet"),
        "eligibility_eicu_semantic_sha256": sha256_file(GATE_ROOT / "eicu_semantic_v4_2.parquet"),
        "endpoint_by_cohort": endpoints,
        "script_path_internal": str(script_path),
        "script_sha256": sha256_file(script_path),
        "runner_path_internal": str(REV5 / "scripts" / "v5_calibration_sites_runner.sh"),
        "runner_sha256": sha256_file(REV5 / "scripts" / "v5_calibration_sites_runner.sh"),
        "launcher_path_internal": str(REV5 / "scripts" / "v5_calibration_sites_launch.sh"),
        "launcher_sha256": sha256_file(REV5 / "scripts" / "v5_calibration_sites_launch.sh"),
    }
    site_audit.update({
        "global_adaptation_role_sha256": site_roles["global_adaptation_role_sha256"],
        "global_assessment_role_sha256": site_roles["global_assessment_role_sha256"],
        "global_role_seed": GLOBAL_ROLE_SEED,
        "patients_crossing_hospitals": site_roles["patients_crossing_hospitals"],
        "temporal_claim_status": site_roles["temporal_claim_status"],
    })
    for folder, schema, section in (
        (CAL_OUT, calibration_schema, "calibration"),
        (SITE_OUT, site_schema, "sites"),
    ):
        readme_path = folder / "README.md"
        schema_path = folder / "schema.json"
        manifest = {
            "status": "complete",
            "schema_version": "ehr_fm_v5_calibration_sites_1",
            "run_id": RUN_ID, "section": section,
            "seed": SEED, "global_role_seed_reused_from_v4": GLOBAL_ROLE_SEED,
            "inputs_internal": inputs_internal,
            "tables": schema,
            "documentation": {
                "readme_path": str(readme_path), "readme_sha256": sha256_file(readme_path),
                "schema_path": str(schema_path), "schema_sha256": sha256_file(schema_path),
            },
            "patient_level_exported_to_local": False,
            "private_source_paths_only_in_internal_manifest": True,
            "calibration_protocol": {
                "cohorts": list(COHORTS), "models": list(MODELS),
                "fixed_fit": "server-only Platt fit, budget 500, repeat 0",
                "proper_scores": ["brier", "logloss"],
                "curve_cluster_bootstrap_draws": N_BOOT,
                "curve_cluster_unit": "patient; all assessment stays retained",
                "curve_knots": "assessment logit score q10/q50/q90",
                "curve_parameters": "2 predictor degrees of freedom plus unpenalized intercept; 3 coefficients",
                "curve_support": "assessment prediction p01-p99",
                "constant_fallback": "single-point reference; no fabricated curve",
                "paired_gap_bootstrap_draws": N_BOOT,
                "gap_scope": "per-stay loss means over all 200 fits; patient-cluster assessment bootstrap",
                "adaptation_p05_p95": "empirical 200-repeat fit distribution, separate from bootstrap CI",
                "hgbt_supplement": "verified v4 eICU/NWICU aggregate; 200 draws; not recomputed on repaired-v5 scores",
            },
            "site_protocol": {
                **site_audit,
                "models": list(MODELS), "budgets": list(SITE_BUDGETS),
                "repeats": REPEATS,
                "methods": ["no_update", "prevalence", "jeffreys_prevalence", "intercept", "platt"],
                "fit_failure_fallback": "Jeffreys prevalence (events+0.5)/(budget+1)",
                "primary_assessment_weighting": "all eligible stays from globally held-out patients",
                "patient_weighted_sensitivity": "mean per-patient stay loss, then equal patient mean",
                "across_site_estimands": {
                    "micro_stay_weighted": "pooled assessment stay loss over the feasible sites",
                    "micro_patient_weighted": "mean of global patient mean stay losses; cross-site patients appear once",
                    "macro_stay_weighted": "equal mean of feasible-site stay-weighted losses",
                    "macro_patient_weighted": "equal mean of feasible-site patient-weighted losses",
                    "bootstrap": "1000 global patient-cluster draws shared across sites, conditional on mean loss over 200 adaptation repeats",
                },
                "global_site_role_flow": site_roles["selected_site_flow"],
            },
            "resource_limits": {
                "cpu_threads": min(4, max(1, int(os.environ.get("V5_THREADS", "2")))),
                "cpu_quota_percent": 200,
                "memory_max_bytes": int(os.environ.get('V5_HOST_MEMORY_MAX_BYTES', str(3*1024**3))),
                "memory_swap_max_bytes": 0,
                "account_slice_memory_high_bytes": 182536110080,
                "account_slice_memory_max_bytes": 186705375232,
                "environment_reinstalled": False,
            },
            "resource_scope": RESOURCE_SCOPE,
            "cgroup_path": current_cgroup_path(),
            "runtime_seconds": float(time.time() - started),
            "host": os.uname().nodename,
        }
        manifest_path = folder / "jobmanifest.json"
        _write_json_atomic(manifest_path, manifest)
        _write_text_atomic(
            folder / "jobmanifest.sha256",
            f"{sha256_file(manifest_path)}  jobmanifest.json\n",
        )
    _progress_event(
        "v5_calibration_sites_complete",
        calibration_rows=int(len(repeat_frame)), curve_points=int(len(all_curves)),
        paired_gap_rows=int(len(gap_ci_frame)),
        site_hospitals=int(site_audit["all_eligible_source_hospitals"]),
        selected_sites=int(site_audit["selected_hospitals"]),
        site_repeat_rows=int(len(site_draws)),
        finite_site_slope_points=int(site_audit["finite_calibration_slope_point_hospitals_fm"]),
        stable_sites_ge10_events_nonevents=int(site_audit["stable_site_threshold_hospitals_fm"]),
        runtime_seconds=round(time.time() - started, 2),
    )


def _parse_resume_checkpoints(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        db, separator, digest = value.partition("=")
        if (not separator or db not in COHORTS
                or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None
                or db in parsed):
            raise ValueError(
                "--resume-checkpoint must be a unique COHORT=64-hex-manifest-SHA256 entry"
            )
        parsed[db] = digest.lower()
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume-checkpoint", action="append", default=[], metavar="COHORT=SHA256",
        help="explicitly reuse one verified cohort checkpoint; may be repeated",
    )
    args = parser.parse_args()
    run(_parse_resume_checkpoints(args.resume_checkpoint))


if __name__ == "__main__":
    main()
