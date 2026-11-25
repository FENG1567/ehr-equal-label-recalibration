"""Patient-disjoint, natural-draw recalibration robustness analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from robustness_common import (
    brier, hindsight_bss, load_checked, logloss, write_rows,
)

SEED = 20261122
BUDGETS = (50, 200, 500, 2000)


def one_index_per_patient(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Choose one stay at random per patient without consulting outcome labels."""
    ordered = df.sort_values(["subject", "stay_id"], kind="stable").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    selected = []
    for _, group in ordered.groupby("subject", sort=True, observed=True):
        selected.append(group.iloc[int(rng.integers(0, len(group)))])
    if not selected:
        return ordered.iloc[0:0].copy()
    out = pd.DataFrame(selected).reset_index(drop=True)
    if out["subject"].duplicated().any():
        raise AssertionError("index-stay selection failed to preserve one stay per patient")
    return out


def split_patient_pool(indexed: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    y = indexed["y"].to_numpy(dtype=int)
    stratify = y if np.unique(y).size == 2 and int(np.bincount(y).min()) >= 2 else None
    adapt, assess = train_test_split(
        np.arange(len(indexed)), test_size=0.5, random_state=seed, stratify=stratify
    )
    adaptation = indexed.iloc[adapt].reset_index(drop=True)
    assessment = indexed.iloc[assess].reset_index(drop=True)
    overlap = set(adaptation["subject"]) & set(assessment["subject"])
    if overlap:
        raise AssertionError(f"patient overlap in adaptation/assessment: {len(overlap)}")
    if adaptation["subject"].duplicated().any() or assessment["subject"].duplicated().any():
        raise AssertionError("more than one index stay per patient")
    return adaptation, assessment, stratify is not None


def adaptation_assessment_pools(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Randomly split patients; one outcome-blind adaptation stay, all assessment stays."""
    unique = np.unique(df["subject"].astype(str).to_numpy())
    adapt_idx, assess_idx = train_test_split(
        np.arange(len(unique)), test_size=0.5, random_state=seed, stratify=None
    )
    adapt_subjects = set(unique[adapt_idx])
    assess_subjects = set(unique[assess_idx])
    adaptation_all = df.loc[df["subject"].astype(str).isin(adapt_subjects)].copy()
    assessment = df.loc[df["subject"].astype(str).isin(assess_subjects)].copy().reset_index(drop=True)
    if adapt_subjects & assess_subjects:
        raise AssertionError("patient overlap in adaptation/assessment pools")
    adaptation = one_index_per_patient(adaptation_all, seed)
    if len(adaptation) != len(adapt_subjects):
        raise AssertionError("adaptation index selection did not keep exactly one stay per patient")
    return adaptation, assessment, False


def _identity_digest(df: pd.DataFrame, fields: tuple[str, ...] | None = None) -> str:
    """Hash actual cohort, patient, stay, source-row, split, and label identities."""
    if fields is None:
        fields = tuple(col for col in ("db", "stay_id", "subject", "rows", "split", "y")
                       if col in df.columns)
    required = {"stay_id", "subject"}
    if not required.issubset(fields):
        raise ValueError(f"identity digest requires {sorted(required)}")
    missing = set(fields) - set(df.columns)
    if missing:
        raise ValueError(f"identity digest fields are absent: {sorted(missing)}")
    if df[list(fields)].isna().any().any():
        raise ValueError("identity digest cannot include missing identifiers or labels")
    if "y" in fields and not df["y"].isin([0, 1]).all():
        raise ValueError("identity digest requires binary labels")
    key_fields = [col for col in ("db", "stay_id") if col in fields]
    if df.duplicated(key_fields).any():
        raise ValueError("identity digest input contains duplicate cohort/stay keys")
    ordered = df.sort_values(list(fields), kind="stable")
    digest = hashlib.sha256()
    for row in ordered[list(fields)].itertuples(index=False, name=None):
        normalized = []
        for col, value in zip(fields, row):
            if col in {"rows", "split", "y"}:
                normalized.append(int(value))
            else:
                normalized.append(str(value).strip())
        digest.update(json.dumps(normalized, separators=(",", ":"),
                                ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_shared_population_frames(frames: dict[str, pd.DataFrame]) -> str:
    """Require exact actual row, patient, split, and label identity across arms."""
    if len(frames) < 2:
        raise ValueError("shared-pool verification requires at least two model arms")
    if any(frame.empty for frame in frames.values()):
        raise ValueError("shared-pool verification does not accept empty model arms")
    signature_fields = tuple(col for col in ("db", "stay_id", "subject", "rows", "split", "y")
                             if col in next(iter(frames.values())).columns)
    for name, frame in frames.items():
        present = tuple(col for col in ("db", "stay_id", "subject", "rows", "split", "y")
                        if col in frame.columns)
        if present != signature_fields:
            raise ValueError(f"{name}: identity field set differs from other model arms")
    signatures = {name: _identity_digest(frame, signature_fields) for name, frame in frames.items()}
    if len(set(signatures.values())) != 1:
        raise ValueError(f"model arms do not have one exact matched population: {signatures}")
    return next(iter(signatures.values()))


def _make_sampling_plan(usable: pd.DataFrame, adaptation: pd.DataFrame,
                        assessment: pd.DataFrame, *, repeats: int, seed: int,
                        budgets: tuple[int, ...]) -> tuple[dict[str, Any], dict[int, list[np.ndarray]]]:
    n_adapt = len(adaptation)
    rng = np.random.default_rng(seed + 1)
    draw_indices: dict[int, list[np.ndarray]] = {}
    draw_records: list[dict[str, Any]] = []
    for budget in budgets:
        if budget > n_adapt:
            continue
        draw_indices[budget] = []
        for repeat in range(repeats):
            indices = rng.choice(n_adapt, size=budget, replace=False)
            draw_indices[budget].append(indices)
            selected = adaptation.iloc[indices]
            draw_records.append({
                "budget": int(budget), "repeat": int(repeat),
                "draw_index_sha256": hashlib.sha256(
                    np.sort(indices).astype("<i8").tobytes()).hexdigest(),
                "patient_index_stay_sha256": _identity_digest(
                    selected, tuple(col for col in ("db", "subject", "stay_id", "rows", "split")
                                    if col in selected.columns)),
            })
    plan: dict[str, Any] = {
        "fixed_population_identity_sha256": _identity_digest(usable),
        "adaptation_index_patient_stay_sha256": _identity_digest(
            adaptation, tuple(col for col in ("db", "subject", "stay_id", "rows", "split")
                              if col in adaptation.columns)),
        "adaptation_patient_set_sha256": hashlib.sha256(
            "\n".join(sorted(adaptation["subject"].astype(str).unique())).encode("utf-8")
        ).hexdigest(),
        "assessment_population_identity_sha256": _identity_digest(assessment),
        "assessment_patient_set_sha256": hashlib.sha256(
            "\n".join(sorted(assessment["subject"].astype(str).unique())).encode("utf-8")
        ).hexdigest(),
        "draws": draw_records,
        "seed": int(seed), "repeats": int(repeats), "budgets": [int(x) for x in budgets],
    }
    plan["sampling_plan_sha256"] = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return plan, draw_indices


def verify_shared_recalibration_manifests(paths: list[str | Path]) -> dict[str, Any]:
    """Fail unless separate model runs attest the same real patient/stay draw plan."""
    if len(paths) < 2:
        raise ValueError("at least two recalibration manifests are required")
    manifests = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    if any(row.get("status") != "complete" for row in manifests):
        raise ValueError("all shared-plan recalibration manifests must be complete")
    db_sets = [set(row.get("cohort_sampling_plans", {})) for row in manifests]
    if any(db_set != db_sets[0] for db_set in db_sets[1:]):
        raise ValueError("recalibration manifests contain different cohort sets")
    comparison_fields = (
        "fixed_population_identity_sha256", "adaptation_index_patient_stay_sha256",
        "adaptation_patient_set_sha256", "assessment_population_identity_sha256",
        "assessment_patient_set_sha256", "sampling_plan_sha256",
    )
    verified = {}
    for db in sorted(db_sets[0]):
        reference = manifests[0]["cohort_sampling_plans"][db]
        for index, candidate_manifest in enumerate(manifests[1:], start=1):
            candidate = candidate_manifest["cohort_sampling_plans"][db]
            if any(candidate.get(field) != reference.get(field) for field in comparison_fields):
                raise ValueError(f"{db}: shared recalibration plan differs in manifest {index + 1}")
        verified[db] = {field: reference[field] for field in comparison_fields}
    return {"status": "verified", "n_manifests": len(manifests), "cohorts": verified}


def fit_offset(y: np.ndarray, lg: np.ndarray) -> tuple[float, float]:
    if np.unique(y).size < 2:
        raise ValueError("single_class_no_finite_offset_MLE")
    intercept = brentq(
        lambda b: float(np.sum(expit(lg + b) - y)), -60.0, 60.0, xtol=1e-12
    )
    return 1.0, float(intercept)


def fit_platt(y: np.ndarray, lg: np.ndarray) -> tuple[float, float]:
    if np.unique(y).size < 2:
        raise ValueError("single_class_no_finite_Platt_MLE")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = LogisticRegression(C=1e6, max_iter=2000, tol=1e-8, solver="lbfgs")
        model.fit(lg.reshape(-1, 1), y)
    a, b = float(model.coef_[0, 0]), float(model.intercept_[0])
    if not np.isfinite(a + b):
        raise ValueError("nonfinite_Platt_coefficients")
    return a, b


def score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "brier": brier(y, p),
        "logloss": logloss(y, p),
        "bss_assessment_prevalence_hindsight_descriptive": hindsight_bss(y, p),
    }


def _cluster_bootstrap_ci(values_by_stay: np.ndarray, groups: np.ndarray,
                          n_boot: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values_by_stay, dtype=float)
    groups = np.asarray(groups).astype(str)
    if (n_boot < 1000 or len(values) == 0 or len(values) != len(groups)
            or not np.isfinite(values).all()):
        return float("nan"), float("nan")
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        draw = rng.integers(0, len(unique), size=len(unique))
        weights = np.bincount(draw, minlength=len(unique)).astype(float)[inverse]
        estimates[b] = float(np.average(values, weights=weights))
    return tuple(float(x) for x in np.percentile(estimates, [2.5, 97.5]))


def run_db(
    db: str, df: pd.DataFrame, *, repeats: int, n_boot: int, seed: int,
    budgets: tuple[int, ...] = BUDGETS, cluster_unit: str = "supplied_subject_identifier",
    expected_shared_plan: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if repeats < 1 or n_boot < 1000:
        raise ValueError("recalibration requires positive repeats and at least 1000 cluster-bootstrap draws")
    if not budgets or any(int(budget) < 1 for budget in budgets) or len(set(budgets)) != len(budgets):
        raise ValueError("recalibration budgets must be unique positive integers")
    usable = df.loc[df["eligible_24h"] & df["temporal_verified"]].copy()
    if usable.empty:
        return [], [{"db": db, "status": "no_temporally_eligible_rows"}], []
    if not usable["y"].isin([0, 1]).all():
        raise ValueError(f"{db}: eligible rows contain nonbinary labels")
    adaptation, assessment, label_stratified = adaptation_assessment_pools(usable, seed)
    if adaptation.empty or assessment.empty:
        return [], [{"db": db, "status": "insufficient_patients_for_disjoint_split"}], []
    sampling_plan, planned_draw_indices = _make_sampling_plan(
        usable, adaptation, assessment, repeats=repeats, seed=seed, budgets=budgets)
    if expected_shared_plan is not None:
        for key in ("fixed_population_identity_sha256", "adaptation_index_patient_stay_sha256",
                    "assessment_population_identity_sha256", "sampling_plan_sha256"):
            if expected_shared_plan.get(key) != sampling_plan.get(key):
                raise ValueError(f"{db}: recalibration population/plan differs from shared attestation ({key})")
    y_assess = assessment["y"].to_numpy(dtype=int)
    assessment_groups = assessment["subject"].astype(str).to_numpy()
    p_assess = np.clip(assessment["p"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    lg_assess = logit(p_assess)
    draw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    ci_rows: list[dict[str, Any]] = []
    split_record = {
        "db": db,
        "status": "ok",
        "cohort_status": str(df["cohort_status"].iloc[0]),
        "n_eligible_stays": int(len(usable)),
        "n_index_patients": int(usable["subject"].nunique()),
        "n_adaptation_patients": int(len(adaptation)),
        "n_assessment_patients": int(assessment["subject"].nunique()),
        "n_assessment_stays": int(len(assessment)),
        "n_assessment_events": int(y_assess.sum()),
        "n_assessment_nonevents": int((1 - y_assess).sum()),
        "assessment_precision_feasibility_200_events_200_nonevents": bool(
            int(y_assess.sum()) >= 200 and int((1 - y_assess).sum()) >= 200
        ),
        "primary_budget_500_feasible_for_confirmation": bool(
            len(adaptation) >= 500 and int(y_assess.sum()) >= 200 and int((1 - y_assess).sum()) >= 200
        ),
        "primary_budget_feasibility_is_performance_blind": True,
        "patient_overlap": 0,
        "bootstrap_cluster_unit": cluster_unit,
        "independent_patient_clustering_verified": cluster_unit == "patient",
        "one_random_index_stay_per_adaptation_patient": True,
        "index_stay_selection_uses_outcome": False,
        "stratified_by_patient_any_event": bool(label_stratified),
        "patient_pool_split_uses_outcomes": False,
        "assessment_weighting": "all eligible assessment stays; patient-cluster uncertainty",
        "repeat_uncertainty_scope": "adaptation-draw distribution conditional on the fixed assessment pool",
        "bootstrap_uncertainty_scope": f"1000+ {cluster_unit} cluster bootstrap of stay-weighted paired assessment losses conditional on observed adaptation draws",
        **{key: value for key, value in sampling_plan.items() if key != "draws"},
    }
    summary_rows.append(split_record)
    adapt_y = adaptation["y"].to_numpy(dtype=int)
    adapt_lg = logit(np.clip(adaptation["p"].to_numpy(dtype=float), 1e-6, 1 - 1e-6))
    n_adapt = len(adaptation)
    for budget in budgets:
        if budget > n_adapt:
            summary_rows.append({
                "db": db, "budget": budget, "status": "budget_exceeds_patient_disjoint_adaptation_pool",
                "available_adaptation_patients": n_adapt,
            })
            continue
        method_names = (
            "raw", "adaptation_prevalence", "smoothed_adaptation_prevalence",
            "offset_conditional", "offset_operational", "platt_conditional", "platt_operational",
        )
        losses = {
            (method, metric): np.full((repeats, len(assessment)), np.nan, dtype=float)
            for method in method_names for metric in ("brier", "logloss")
        }
        fit_failures = {"offset": 0, "platt": 0}
        fit_success = {"offset": np.zeros(repeats, dtype=bool),
                       "platt": np.zeros(repeats, dtype=bool)}
        fit_status_counts: dict[str, dict[str, int]] = {"offset": {}, "platt": {}}
        repeat_metrics: dict[str, list[dict[str, float]]] = {m: [] for m in method_names}
        events_by_repeat: list[int] = []
        for rep in range(repeats):
            draw_idx = planned_draw_indices[budget][rep]
            yd = adapt_y[draw_idx]
            lpd = adapt_lg[draw_idx]
            events = int(yd.sum())
            events_by_repeat.append(events)
            sample_hash = hashlib.sha256(np.sort(draw_idx).astype("<i8").tobytes()).hexdigest()
            selected_index_stays = adaptation.iloc[draw_idx]
            sample_patient_stay_hash = _identity_digest(
                selected_index_stays,
                tuple(col for col in ("db", "subject", "stay_id", "rows", "split")
                      if col in selected_index_stays.columns),
            )
            adapt_prev = float(np.mean(yd))
            smooth_prev = float((events + 0.5) / (budget + 1.0))
            predictions: dict[str, np.ndarray] = {
                "raw": p_assess,
                "adaptation_prevalence": np.full(len(assessment), adapt_prev),
                "smoothed_adaptation_prevalence": np.full(len(assessment), smooth_prev),
            }
            fits: dict[str, tuple[bool, str]] = {}
            for arm, fit_function in (("offset", fit_offset), ("platt", fit_platt)):
                try:
                    a, b = fit_function(yd, lpd)
                    fitted = np.clip(expit(a * lg_assess + b), 1e-6, 1 - 1e-6)
                    predictions[f"{arm}_conditional"] = fitted
                    predictions[f"{arm}_operational"] = fitted
                    fits[arm] = (True, "ok")
                    fit_success[arm][rep] = True
                except (ValueError, RuntimeError, FloatingPointError, ConvergenceWarning) as exc:
                    fit_failures[arm] += 1
                    reason = f"{type(exc).__name__}:{str(exc)[:100]}"
                    fit_status_counts[arm][reason] = fit_status_counts[arm].get(reason, 0) + 1
                    predictions[f"{arm}_conditional"] = np.full(len(assessment), np.nan)
                    predictions[f"{arm}_operational"] = np.full(len(assessment), smooth_prev)
                    fits[arm] = (False, reason)
            baseline = predictions["adaptation_prevalence"]
            smoothed_baseline = predictions["smoothed_adaptation_prevalence"]
            raw = predictions["raw"]
            baseline_scores = {
                "adaptation_prevalence": score(y_assess, baseline),
                "smoothed_adaptation_prevalence": score(y_assess, smoothed_baseline),
                "raw": score(y_assess, raw),
            }
            for method, p_eval in predictions.items():
                if np.isnan(p_eval).any():
                    continue
                method_score = score(y_assess, p_eval)
                repeat_metrics[method].append(method_score)
                losses[(method, "brier")][rep] = (y_assess - p_eval) ** 2
                peval = np.clip(p_eval, 1e-12, 1 - 1e-12)
                losses[(method, "logloss")][rep] = -(y_assess * np.log(peval) + (1 - y_assess) * np.log(1 - peval))
                draw_rows.append({
                    "db": db, "budget": budget, "repeat": rep, "method": method,
                    "events": events, "nonevents": int(budget - events),
                    "n_budget_patients": int(budget),
                    "n_budget_stays": int(len(selected_index_stays)),
                    "zero_class_draw": bool(events == 0),
                    "one_class_draw": bool(events == budget),
                    "adaptation_sample_index_sha256": sample_hash,
                    "adaptation_sample_patient_index_stay_sha256": sample_patient_stay_hash,
                    "fixed_population_identity_sha256": sampling_plan["fixed_population_identity_sha256"],
                    "adaptation_index_patient_stay_sha256": sampling_plan[
                        "adaptation_index_patient_stay_sha256"],
                    "assessment_population_identity_sha256": sampling_plan[
                        "assessment_population_identity_sha256"],
                    "sampling_plan_sha256": sampling_plan["sampling_plan_sha256"],
                    "fit_ok": fits.get(method.split("_")[0], (True, "not_applicable"))[0],
                    "fit_status": fits.get(method.split("_")[0], (True, "not_applicable"))[1],
                    "is_operational_fallback": bool(
                        method.endswith("_operational")
                        and not fits.get(method.split("_")[0], (True, "not_applicable"))[0]),
                    "brier": method_score["brier"], "logloss": method_score["logloss"],
                    "bss_assessment_prevalence_hindsight_descriptive": method_score[
                        "bss_assessment_prevalence_hindsight_descriptive"
                    ],
                    "delta_brier_vs_same_budget_prevalence": method_score["brier"] - baseline_scores["adaptation_prevalence"]["brier"],
                    "delta_logloss_vs_same_budget_prevalence": method_score["logloss"] - baseline_scores["adaptation_prevalence"]["logloss"],
                    "delta_brier_vs_smoothed_prevalence": method_score["brier"] - baseline_scores["smoothed_adaptation_prevalence"]["brier"],
                    "delta_logloss_vs_smoothed_prevalence": method_score["logloss"] - baseline_scores["smoothed_adaptation_prevalence"]["logloss"],
                    "delta_brier_vs_raw": method_score["brier"] - baseline_scores["raw"]["brier"],
                    "delta_logloss_vs_raw": method_score["logloss"] - baseline_scores["raw"]["logloss"],
                    "bss_reference": "assessment_prevalence_hindsight_descriptive_only",
                    "fallback": "Jeffreys-smoothed adaptation prevalence after offset/Platt fit failure",
                })
            progress_interval = max(1, repeats // 20)
            if (rep + 1) % progress_interval == 0 or rep + 1 == repeats:
                print(json.dumps({
                    "stage": "recalibration_repeat_progress", "db": db,
                    "budget": int(budget), "repeat_completed": int(rep + 1),
                    "repeats": int(repeats),
                    "events_min_to_current": [int(min(events_by_repeat)), int(max(events_by_repeat))],
                    "zero_class_draws": int(sum(value == 0 for value in events_by_repeat)),
                    "one_class_draws": int(sum(value == budget for value in events_by_repeat)),
                }, sort_keys=True), flush=True)
        fit_rate_rows = {
            arm: {
                "fit_failure_rate": fit_failures[arm] / repeats,
                "n_fit_fail": fit_failures[arm],
                "n_fit_attempts": repeats,
                "fit_failure_reasons": json.dumps(fit_status_counts[arm], sort_keys=True),
            }
            for arm in ("offset", "platt")
        }
        for method in method_names:
            values = repeat_metrics[method]
            successful = len(values)
            if values:
                briers = np.asarray([x["brier"] for x in values])
                lls = np.asarray([x["logloss"] for x in values])
                bss_vals = np.asarray([x["bss_assessment_prevalence_hindsight_descriptive"] for x in values])
            else:
                briers = lls = bss_vals = np.array([], dtype=float)
            arm_name = method.split("_")[0]
            failure = fit_rate_rows.get(arm_name, {"fit_failure_rate": 0.0, "n_fit_fail": 0,
                                                       "n_fit_attempts": repeats, "fit_failure_reasons": "{}"})
            summary_rows.append({
                "db": db, "budget": budget, "method": method, "status": "ok" if successful else "no_successful_fit",
                "n_repeats": repeats, "n_valid_repeats": successful,
                "n_budget_patients": int(budget),
                "n_budget_stays": int(budget),
                "n_zero_class_draws": int(sum(value == 0 for value in events_by_repeat)),
                "n_one_class_draws": int(sum(value == budget for value in events_by_repeat)),
                "n_mixed_class_draws": int(sum(0 < value < budget for value in events_by_repeat)),
                "events_min": int(min(events_by_repeat)) if events_by_repeat else 0,
                "events_median": float(np.median(events_by_repeat)) if events_by_repeat else np.nan,
                "events_max": int(max(events_by_repeat)) if events_by_repeat else 0,
                "nonevents_min": int(budget - max(events_by_repeat)) if events_by_repeat else 0,
                "nonevents_median": float(budget - np.median(events_by_repeat)) if events_by_repeat else np.nan,
                "nonevents_max": int(budget - min(events_by_repeat)) if events_by_repeat else 0,
                "brier_mean": float(np.mean(briers)) if successful else np.nan,
                "brier_p05": float(np.percentile(briers, 5)) if successful else np.nan,
                "brier_p95": float(np.percentile(briers, 95)) if successful else np.nan,
                "logloss_mean": float(np.mean(lls)) if successful else np.nan,
                "logloss_p05": float(np.percentile(lls, 5)) if successful else np.nan,
                "logloss_p95": float(np.percentile(lls, 95)) if successful else np.nan,
                "negative_bss_rate_descriptive": float(np.mean(bss_vals < 0)) if successful else np.nan,
                "fit_failure_rate": failure["fit_failure_rate"],
                "n_fit_fail": failure["n_fit_fail"],
                "n_fit_success": (int(fit_success[arm_name].sum()) if arm_name in fit_success else repeats),
                "n_fit_attempts": failure["n_fit_attempts"],
                "fit_failure_reasons": failure["fit_failure_reasons"],
                "conditional_on_assessment_patients": True,
                "assessment_prevalence_bss_is_deployable_reference": False,
            })
        # Patient-cluster bootstrap of paired assessment losses; adaptation draws
        # are averaged within patient first, so these intervals condition on the
        # realized natural-draw set. Draw-to-draw spread is reported separately.
        for metric in ("brier", "logloss"):
            for method in (
                "raw", "adaptation_prevalence", "smoothed_adaptation_prevalence",
                "offset_conditional", "offset_operational",
                "platt_conditional", "platt_operational",
            ):
                for baseline in ("adaptation_prevalence", "smoothed_adaptation_prevalence", "raw"):
                    if method == baseline:
                        continue
                    mat_a = losses[(method, metric)]
                    mat_b = losses[(baseline, metric)]
                    diff_mat = mat_a - mat_b
                    valid_draws = np.isfinite(diff_mat).all(axis=1)
                    if not valid_draws.any():
                        continue
                    assessment_delta = np.mean(diff_mat[valid_draws], axis=0)
                    lo, hi = _cluster_bootstrap_ci(assessment_delta, assessment_groups,
                                                   n_boot, seed + budget)
                    draw_deltas = np.mean(diff_mat[valid_draws], axis=1)
                    ci_rows.append({
                        "db": db, "budget": budget, "metric": metric,
                        "method": method, "comparator": baseline,
                        "analysis_scope": (
                            "successful_fit_draws" if method.endswith("_conditional")
                            else "all_repeats"),
                        "delta_method_minus_comparator": float(np.mean(assessment_delta)),
                        "assessment_cluster_ci_95_lo": lo,
                        "assessment_cluster_ci_95_hi": hi,
                        "bootstrap_cluster_unit": cluster_unit,
                        "independent_patient_clustering_verified": cluster_unit == "patient",
                        "assessment_bootstrap_draws": n_boot,
                        "n_assessment_patients": int(assessment["subject"].nunique()),
                        "n_assessment_stays": int(len(assessment)),
                        "assessment_cluster_ci_status": (
                            "ok" if assessment["subject"].nunique() >= 2
                            else "insufficient_assessment_clusters"),
                        "n_valid_adaptation_draws": int(valid_draws.sum()),
                        "adaptation_draw_delta_p05_conditional": float(np.percentile(draw_deltas, 5)),
                        "adaptation_draw_delta_p95_conditional": float(np.percentile(draw_deltas, 95)),
                        "uncertainty_note": "patient CI conditions on realized adaptation draws; p05/p95 describe adaptation draws conditional on fixed assessment pool",
                    })
            for arm in ("offset", "platt"):
                fallback_draws = ~fit_success[arm]
                if not fallback_draws.any():
                    continue
                fallback_diff = (
                    losses[("smoothed_adaptation_prevalence", metric)][fallback_draws]
                    - losses[("adaptation_prevalence", metric)][fallback_draws]
                )
                valid_fallback = np.isfinite(fallback_diff).all(axis=1)
                if not valid_fallback.any():
                    continue
                assessment_delta = np.mean(fallback_diff[valid_fallback], axis=0)
                lo, hi = _cluster_bootstrap_ci(
                    assessment_delta, assessment_groups, n_boot, seed + budget)
                draw_deltas = np.mean(fallback_diff[valid_fallback], axis=1)
                ci_rows.append({
                    "db": db, "budget": budget, "metric": metric,
                    "method": f"{arm}_failure_fallback_smoothed_prevalence",
                    "comparator": "adaptation_prevalence",
                    "analysis_scope": "fallback_only",
                    "delta_method_minus_comparator": float(np.mean(assessment_delta)),
                    "assessment_cluster_ci_95_lo": lo,
                    "assessment_cluster_ci_95_hi": hi,
                    "bootstrap_cluster_unit": cluster_unit,
                    "independent_patient_clustering_verified": cluster_unit == "patient",
                    "assessment_bootstrap_draws": n_boot,
                    "n_assessment_patients": int(assessment["subject"].nunique()),
                    "n_assessment_stays": int(len(assessment)),
                    "assessment_cluster_ci_status": (
                        "ok" if assessment["subject"].nunique() >= 2
                        else "insufficient_assessment_clusters"),
                    "n_valid_adaptation_draws": int(valid_fallback.sum()),
                    "adaptation_draw_delta_p05_conditional": float(np.percentile(draw_deltas, 5)),
                    "adaptation_draw_delta_p95_conditional": float(np.percentile(draw_deltas, 95)),
                    "uncertainty_note": "Jeffreys fallback versus unsmoothed prevalence, restricted to draws where the offset/Platt fit failed",
                })
    return draw_rows, summary_rows, ci_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions")
    parser.add_argument("--eligibility-dir")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--model-tag", default="main20k")
    parser.add_argument("--analysis-id", help="registered prediction analysis ID; defaults to <model-tag>_<input-condition>")
    parser.add_argument("--input-condition", default="real")
    parser.add_argument("--endpoint", default="hospital")
    parser.add_argument("--prediction-manifest")
    parser.add_argument("--diagnostic-only", action="store_true",
                         help="allow separately attested retrospective association predictions; outputs remain descriptive")
    parser.add_argument("--shared-plan-manifest",
                         help="require exact per-cohort population and patient/index-stay draw-plan hashes")
    parser.add_argument("--verify-shared-run-manifests", nargs="+",
                         help="verify that multiple model runs used identical patient/index-stay draws, then exit")
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    if args.verify_shared_run_manifests:
        result = verify_shared_recalibration_manifests(args.verify_shared_run_manifests)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    if not args.predictions or not args.eligibility_dir:
        parser.error("--predictions and --eligibility-dir are required for recalibration")
    if args.repeats < 1 or args.bootstrap < 1000:
        parser.error("--repeats must be positive and --bootstrap must be at least 1000")
    shared_plan_doc = None
    if args.shared_plan_manifest:
        shared_plan_doc = json.loads(Path(args.shared_plan_manifest).read_text(encoding="utf-8"))
        if shared_plan_doc.get("status") != "complete":
            parser.error("--shared-plan-manifest must be a complete sampling-plan manifest")
        if (int(shared_plan_doc.get("seed", -1)) != args.seed
                or int(shared_plan_doc.get("repeats", -1)) != args.repeats
                or tuple(shared_plan_doc.get("budgets", ())) != BUDGETS):
            parser.error("shared-plan seed/repeats/budgets do not match this run")
    data, gates = load_checked(
        args.predictions, args.eligibility_dir, model_tag=args.model_tag,
        analysis_id=args.analysis_id,
        input_condition=args.input_condition, endpoint=args.endpoint,
        prediction_manifest_path=args.prediction_manifest,
        diagnostic_only=args.diagnostic_only,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_draws: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []
    all_ci: list[dict[str, Any]] = []
    cohort_sampling_plans: dict[str, dict[str, Any]] = {}
    for db, df in data.items():
        expected_plan = (shared_plan_doc.get("cohort_sampling_plans", {}).get(db)
                         if shared_plan_doc is not None else None)
        if shared_plan_doc is not None and expected_plan is None:
            raise ValueError(f"shared-plan manifest lacks cohort {db}")
        dr, su, ci_rows = run_db(db, df, repeats=args.repeats, n_boot=args.bootstrap,
                                 seed=args.seed, cluster_unit=gates[db]["bootstrap_cluster_unit"],
                                 expected_shared_plan=expected_plan)
        all_draws.extend(dr)
        all_summary.extend(su)
        all_ci.extend(ci_rows)
        plan_row = next((row for row in su if row.get("sampling_plan_sha256")), None)
        if plan_row is not None:
            cohort_sampling_plans[db] = {
                key: plan_row[key] for key in (
                    "fixed_population_identity_sha256", "adaptation_index_patient_stay_sha256",
                    "adaptation_patient_set_sha256", "assessment_population_identity_sha256",
                    "assessment_patient_set_sha256", "sampling_plan_sha256",
                )
            }
        print(json.dumps({"db": db, "ready": gates[db]["retrospective_landmark_prediction_allowed"],
                          "draw_rows": len(dr), "summary_rows": len(su), "paired_ci_rows": len(ci_rows)}), flush=True)
    write_rows(out_dir / "recalibration_draws.csv", all_draws)
    write_rows(out_dir / "recalibration_summary.csv", all_summary)
    write_rows(out_dir / "recalibration_paired_patient_ci.csv", all_ci)
    (out_dir / "recalibration_run_manifest.json").write_text(json.dumps({
        "status": "complete", "seed": args.seed, "budgets": BUDGETS,
        "model_tag": args.model_tag,
        "analysis_id": args.analysis_id or f"{args.model_tag}_{args.input_condition}",
        "input_condition": args.input_condition,
        "repeats": args.repeats, "assessment_cluster_bootstrap": args.bootstrap,
        "cohort_gates": gates,
        "cohort_sampling_plans": cohort_sampling_plans,
        "prediction_scope": "retrospective association diagnostic only" if args.diagnostic_only else "retrospective landmark prediction at recorded measurement time",
        "clinical_input_availability_verified": False,
        "clinical_promotion_allowed": False,
        "sampling": "one outcome-blind index stay per patient; patient-disjoint 50/50 adaptation/assessment; natural without-replacement patient draws; no forced events",
        "sampling_plan_verification": "population, adaptation index-stay, assessment identity, and all actual sampled patient/index-stay hashes are bound into the plan SHA256; use --verify-shared-run-manifests across model arms",
        "fallback": "Jeffreys-smoothed adaptation prevalence (events+0.5)/(budget+1) for single-class or failed offset/Platt fit",
        "proposed_primary_budget": 500,
        "primary_budget_rule": "use 500 only if adaptation size >=500 and assessment has >=200 events and >=200 nonevents before comparative performance is opened; otherwise no confirmation interpretation",
        "bss_reference": "assessment prevalence is hindsight descriptive only, never an oracle deployment comparator",
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

