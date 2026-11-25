"""Current-manifest eICU hospital metrics and paired site heterogeneity."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.special import expit

from robustness_common import (
    load_checked, logloss, sha256_file, write_rows,
)

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
FM_RUN = REV / "engine/results/prediction_runs/status_consistent_primary_core_fix2-7da980ffcbd8f00b"
BASELINE_RESULTS = REV / "baselines/results/primary_status_consistent_20261123/results"
ELIGIBILITY_DIR = REV / "temporal/results/eligibility/status_consistent_v4_1/landmark_v4_1"
N_BOOT = 1000
MIN_SITE_EVENTS = 10
SLOPE_BOOTSTRAP_BATCH_SIZE = 32
SLOPE_BOOTSTRAP_MAX_ABS = 50.0


def _site_label(series: pd.Series) -> pd.Series:
    value = series.astype("string").str.strip()
    missing = value.str.casefold().isin(["", "nan", "none", "null", "<na>"])
    return value.mask(missing)


def _complete_separation(y: np.ndarray, logits: np.ndarray) -> bool:
    y = np.asarray(y, dtype=int)
    logits = np.asarray(logits, dtype=float)
    if np.unique(y).size < 2:
        return False
    x0, x1 = logits[y == 0], logits[y == 1]
    return bool(np.max(x0) < np.min(x1) or np.max(x1) < np.min(x0))


def _calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float, str]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    if np.unique(y).size < 2:
        return float("nan"), float("nan"), "unestimable_single_class"
    x = np.log(p / (1 - p))
    if float(np.std(x)) <= 1e-10:
        return float("nan"), float("nan"), "unestimable_no_probability_variation"
    if _complete_separation(y, x):
        return float("nan"), float("nan"), "unestimable_complete_separation"
    logits = x.reshape(-1, 1)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model = LogisticRegression(C=1e6, max_iter=2000, tol=1e-8, solver="lbfgs")
            model.fit(logits, y)
        return float(model.intercept_[0]), float(model.coef_[0, 0]), "ok"
    except (ValueError, RuntimeError, FloatingPointError, ConvergenceWarning) as exc:
        return float("nan"), float("nan"), f"unestimable_{type(exc).__name__}"


def _solve_weighted_logistic_slope_batch(
    y: np.ndarray, logits: np.ndarray, weights: np.ndarray, *,
    max_iter: int = 100, tol: float = 1e-8, regularization: float = 1e-6,
) -> tuple[np.ndarray, dict[str, int]]:
    """Solve 2-parameter calibration logits in batches with explicit failures."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(logits, dtype=float)
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or w.shape[1] != len(y) or len(x) != len(y):
        raise ValueError("bootstrap weights must be [draw, stay] and match y/logits")
    if not np.isfinite(w).all() or (w < 0).any() or not np.isfinite(x).all():
        raise ValueError("calibration bootstrap inputs contain invalid weights or logits")
    n_draws = len(w)
    n_events = w @ y
    total = w.sum(axis=1)
    mixed = (n_events > 0) & (n_events < total)
    single_class = ~mixed

    has_zero = np.any((w > 0) & (y[None, :] == 0), axis=1)
    has_one = np.any((w > 0) & (y[None, :] == 1), axis=1)
    x0_max = np.max(np.where((w > 0) & (y[None, :] == 0), x, -np.inf), axis=1)
    x1_min = np.min(np.where((w > 0) & (y[None, :] == 1), x, np.inf), axis=1)
    x1_max = np.max(np.where((w > 0) & (y[None, :] == 1), x, -np.inf), axis=1)
    x0_min = np.min(np.where((w > 0) & (y[None, :] == 0), x, np.inf), axis=1)
    separated = mixed & has_zero & has_one & ((x0_max < x1_min) | (x1_max < x0_min))

    mean_x = np.divide(w @ x, total, out=np.zeros(n_draws, dtype=float), where=total > 0)
    sum_x2 = w @ (x * x)
    centered_ss = sum_x2 - total * mean_x * mean_x
    no_variation = mixed & (centered_ss <= np.maximum(total, 1.0) * 1e-12)
    eligible = mixed & ~separated & ~no_variation

    intercept = np.zeros(n_draws, dtype=float)
    slope = np.zeros(n_draws, dtype=float)
    converged = np.zeros(n_draws, dtype=bool)
    singular = np.zeros(n_draws, dtype=bool)
    active = eligible.copy()
    for _ in range(max_iter):
        if not active.any():
            break
        eta = intercept[:, None] + slope[:, None] * x[None, :]
        # Stable sigmoid evaluation only; fitted coefficients are never clipped.
        probability = expit(np.clip(eta, -60.0, 60.0))
        residual = w * (y[None, :] - probability)
        variance = w * probability * (1.0 - probability)
        # Match LogisticRegression(lbfgs): the slope is L2-regularized, while
        # the intercept is unpenalized for this solver.
        g0 = residual.sum(axis=1)
        g1 = residual @ x - regularization * slope
        h00 = variance.sum(axis=1)
        h01 = variance @ x
        h11 = variance @ (x * x) + regularization
        determinant = h00 * h11 - h01 * h01
        good = active & np.isfinite(determinant) & (determinant > 1e-18)
        singular |= active & ~good
        delta0 = np.zeros(n_draws, dtype=float)
        delta1 = np.zeros(n_draws, dtype=float)
        delta0[good] = (g0[good] * h11[good] - g1[good] * h01[good]) / determinant[good]
        delta1[good] = (g1[good] * h00[good] - g0[good] * h01[good]) / determinant[good]
        intercept[good] += delta0[good]
        slope[good] += delta1[good]
        step_size = np.maximum(np.abs(delta0), np.abs(delta1))
        scale = 1.0 + np.maximum(np.abs(intercept), np.abs(slope))
        just_converged = good & (step_size <= tol * scale)
        converged |= just_converged
        active = good & ~just_converged

    finite = np.isfinite(intercept) & np.isfinite(slope)
    large_slope = converged & finite & (np.abs(slope) > SLOPE_BOOTSTRAP_MAX_ABS)
    nonconverged = eligible & ~converged & ~singular
    valid = eligible & converged & finite & ~large_slope
    estimates = np.where(valid, slope, np.nan)
    failures = {
        "single_class": int(single_class.sum()),
        "complete_separation": int(separated.sum()),
        "no_probability_variation": int(no_variation.sum()),
        "singular_hessian": int(singular.sum()),
        "nonconverged": int(nonconverged.sum()),
        "nonfinite": int((eligible & converged & ~finite).sum()),
        "large_slope": int(large_slope.sum()),
        "insufficient_patient_clusters": 0,
    }
    return estimates, failures


def _patient_cluster_slope_bootstrap(
    y: np.ndarray, p: np.ndarray, subjects: np.ndarray, *, n_boot: int, seed: int,
    batch_size: int = SLOPE_BOOTSTRAP_BATCH_SIZE,
    progress_context: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    if n_boot < 1000:
        raise ValueError("within-site calibration slope CIs require at least 1000 patient-cluster draws")
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    subjects = np.asarray(subjects).astype(str)
    if len(y) == 0 or len(y) != len(p) or len(y) != len(subjects):
        raise ValueError("slope bootstrap requires aligned nonempty labels, probabilities, and patients")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
        raise ValueError("slope bootstrap requires binary labels and finite probabilities")
    unique, inverse = np.unique(subjects, return_inverse=True)
    draws = np.full(n_boot, np.nan, dtype=float)
    failures = {key: 0 for key in (
        "single_class", "complete_separation", "no_probability_variation",
        "singular_hessian", "nonconverged", "nonfinite", "large_slope",
        "insufficient_patient_clusters",
    )}
    if len(unique) < 2:
        status = "insufficient_patient_clusters"
        failures["insufficient_patient_clusters"] = n_boot
        failures["single_class"] = n_boot if np.unique(y).size < 2 else 0
    elif np.unique(y).size < 2:
        status = "unestimable_single_class"
        failures["single_class"] = n_boot
    else:
        logits = np.log(p / (1 - p))
        if float(np.std(logits)) <= 1e-10:
            status = "unestimable_no_probability_variation"
            failures["no_probability_variation"] = n_boot
        else:
            rng = np.random.default_rng(seed)
            progress_interval = max(1, n_boot // 10)
            last_progress = 0
            for start in range(0, n_boot, batch_size):
                stop = min(n_boot, start + batch_size)
                batch_draws = stop - start
                sampled_clusters = rng.integers(0, len(unique), size=(batch_draws, len(unique)))
                offsets = np.arange(batch_draws, dtype=np.int64)[:, None] * len(unique)
                cluster_counts = np.bincount(
                    (sampled_clusters + offsets).ravel(), minlength=batch_draws * len(unique)
                ).reshape(batch_draws, len(unique))
                row_weights = cluster_counts[:, inverse]
                estimates, batch_failures = _solve_weighted_logistic_slope_batch(
                    y, logits, row_weights,
                )
                draws[start:stop] = estimates
                for key in failures:
                    failures[key] += batch_failures[key]
                if progress_context is not None and (
                    stop - last_progress >= progress_interval or stop == n_boot
                ):
                    print(json.dumps({
                        "stage": "hospital_site_slope_bootstrap_progress",
                        **progress_context,
                        "draw_completed": int(stop), "draws_requested": int(n_boot),
                    }, sort_keys=True), flush=True)
                    last_progress = stop
            valid = int(np.isfinite(draws).sum())
            min_valid = max(200, int(math.ceil(0.8 * n_boot)))
            status = "ok" if valid >= min_valid else "unstable_insufficient_valid_draws"
    valid = int(np.isfinite(draws).sum())
    lo, hi = (np.percentile(draws[np.isfinite(draws)], [2.5, 97.5])
              if status == "ok" else (float("nan"), float("nan")))
    result = {
        "calibration_slope_ci_95_lo": float(lo),
        "calibration_slope_ci_95_hi": float(hi),
        "calibration_slope_ci_status": status,
        "calibration_slope_ci_bootstrap_draws_requested": int(n_boot),
        "calibration_slope_ci_bootstrap_draws_valid": valid,
        "calibration_slope_ci_bootstrap_patient_clusters": int(len(unique)),
        "calibration_slope_ci_bootstrap_draws_invalid_single_class": failures["single_class"],
        "calibration_slope_ci_bootstrap_draws_invalid_separation": failures["complete_separation"],
        "calibration_slope_ci_bootstrap_draws_invalid_no_variation": failures["no_probability_variation"],
        "calibration_slope_ci_bootstrap_draws_invalid_singular": failures["singular_hessian"],
        "calibration_slope_ci_bootstrap_draws_invalid_nonconverged": failures["nonconverged"],
        "calibration_slope_ci_bootstrap_draws_invalid_nonfinite": failures["nonfinite"],
        "calibration_slope_ci_bootstrap_draws_invalid_large_slope": failures["large_slope"],
        "calibration_slope_ci_bootstrap_draws_invalid_insufficient_clusters": failures[
            "insufficient_patient_clusters"],
        "calibration_slope_ci_bootstrap_draws_invalid_not_attempted": 0,
        "calibration_slope_ci_scope": "all eligible stays within hospital; patient-cluster resampling",
    }
    return result, draws


def _site_cluster_digest(subjects: np.ndarray) -> str:
    digest = hashlib.sha256()
    for subject in sorted(set(map(str, np.asarray(subjects)))):
        encoded = subject.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _site_prediction_digest(frame: pd.DataFrame, *, probability_column: str) -> str:
    ordered = frame.sort_values(["subject", "stay_id"], kind="stable")
    digest = hashlib.sha256()
    for stay_id, subject, y, probability in ordered[
        ["stay_id", "subject", "y", probability_column]
    ].itertuples(index=False, name=None):
        record = [str(stay_id), str(subject), int(y), format(float(probability), ".17g")]
        digest.update(json.dumps(record, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _checkpoint_path(checkpoint_dir: Path, *, model_id: str, hospitalid: str) -> Path:
    site_key = hashlib.sha256(f"{model_id}\0{hospitalid}".encode("utf-8")).hexdigest()[:24]
    return checkpoint_dir / f"site_slope_ci_{site_key}.json"


def _json_safe_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (float(value) if np.isfinite(value) else None)
        if isinstance(value, (float, np.floating)) else value
        for key, value in summary.items()
    }


def _site_slope_ci(
    frame: pd.DataFrame, *, model_id: str, hospitalid: str, point_status: str,
    n_boot: int, seed: int, checkpoint_dir: Path,
) -> tuple[dict[str, Any], np.ndarray, str]:
    ordered = frame.sort_values(["subject", "stay_id"], kind="stable").reset_index(drop=True)
    y = ordered["y"].to_numpy(dtype=int)
    p = ordered["p"].to_numpy(dtype=float)
    subjects = ordered["subject"].astype(str).to_numpy()
    data_digest = _site_prediction_digest(ordered, probability_column="p")
    cluster_digest = _site_cluster_digest(subjects)
    checkpoint = _checkpoint_path(checkpoint_dir, model_id=model_id, hospitalid=hospitalid)
    if checkpoint.is_file():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        expected = {
            "model_id": model_id, "hospitalid": hospitalid,
            "n_boot": int(n_boot), "seed": int(seed),
            "site_prediction_sha256": data_digest,
            "patient_cluster_set_sha256": cluster_digest,
        }
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError(f"stale or mismatched site calibration checkpoint: {checkpoint}")
        estimates = np.array([
            np.nan if value is None else float(value)
            for value in saved["bootstrap_slope_estimates"]
        ], dtype=float)
        if len(estimates) != n_boot:
            raise ValueError(f"site calibration checkpoint has the wrong draw count: {checkpoint}")
        result = {
            key: (float("nan") if value is None and key.endswith(("_lo", "_hi")) else value)
            for key, value in saved["summary"].items()
        }
        return result, estimates, cluster_digest

    events = int(y.sum())
    nonevents = int(len(y) - events)
    n_patients = int(pd.Series(subjects).nunique())
    eligible = events >= MIN_SITE_EVENTS and nonevents >= MIN_SITE_EVENTS
    estimates = np.full(n_boot, np.nan, dtype=float)
    site_seed = seed + int.from_bytes(hashlib.sha256(hospitalid.encode("utf-8")).digest()[:4], "little")
    attempted = False
    if np.unique(y).size < 2:
        result = {
            "calibration_slope_ci_95_lo": float("nan"),
            "calibration_slope_ci_95_hi": float("nan"),
            "calibration_slope_ci_status": "unestimable_single_class",
            "calibration_slope_ci_bootstrap_draws_requested": int(n_boot),
            "calibration_slope_ci_bootstrap_draws_valid": 0,
            "calibration_slope_ci_bootstrap_patient_clusters": n_patients,
        }
    elif not eligible:
        result = {
            "calibration_slope_ci_95_lo": float("nan"),
            "calibration_slope_ci_95_hi": float("nan"),
            "calibration_slope_ci_status": "diagnostic_not_for_manuscript_below_10_events_or_nonevents",
            "calibration_slope_ci_bootstrap_draws_requested": int(n_boot),
            "calibration_slope_ci_bootstrap_draws_valid": 0,
            "calibration_slope_ci_bootstrap_patient_clusters": n_patients,
        }
    elif point_status != "ok":
        result = {
            "calibration_slope_ci_95_lo": float("nan"),
            "calibration_slope_ci_95_hi": float("nan"),
            "calibration_slope_ci_status": f"diagnostic_not_for_manuscript_point_{point_status}",
            "calibration_slope_ci_bootstrap_draws_requested": int(n_boot),
            "calibration_slope_ci_bootstrap_draws_valid": 0,
            "calibration_slope_ci_bootstrap_patient_clusters": n_patients,
        }
    else:
        attempted = True
        result, estimates = _patient_cluster_slope_bootstrap(
            y, p, subjects, n_boot=n_boot, seed=site_seed,
            progress_context={"model_id": model_id, "hospitalid": hospitalid},
        )
    summary = {
        **result,
        "calibration_slope_ci_seed": int(site_seed),
        "calibration_slope_ci_eligibility": (
            "ge_10_events_and_nonevents" if eligible else "below_10_events_or_nonevents"
        ),
        "calibration_slope_ci_n_stays": int(len(ordered)),
        "calibration_slope_ci_n_patients": n_patients,
        "calibration_slope_ci_patient_cluster_set_sha256": cluster_digest,
        "calibration_slope_ci_site_prediction_sha256": data_digest,
        "calibration_slope_ci_bootstrap_draws_invalid_not_attempted": 0 if attempted else int(n_boot),
    }
    checkpoint_record = {
        "model_id": model_id, "hospitalid": hospitalid,
        "n_boot": int(n_boot), "seed": int(seed),
        "site_prediction_sha256": data_digest,
        "patient_cluster_set_sha256": cluster_digest,
        "summary": _json_safe_summary(summary),
        "bootstrap_slope_estimates": [
            float(value) if np.isfinite(value) else None for value in estimates
        ],
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(checkpoint_record, separators=(",", ":"), allow_nan=False),
                         encoding="utf-8")
    temporary.replace(checkpoint)
    return summary, estimates, cluster_digest


def _attach_site_slope_cis(
    models: dict[str, pd.DataFrame], site_rows: list[dict[str, Any]], *,
    n_boot: int, seed: int, checkpoint_dir: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    site_row_by_key = {
        (str(row["model_id"]), str(row["hospitalid"])): row
        for row in site_rows if row.get("scope") == "selected_hospital_all_stays"
    }
    results: dict[tuple[str, str], dict[str, Any]] = {}
    for model_id, frame in models.items():
        eicu = frame.loc[frame["db"].astype(str) == "eicu"].copy()
        eicu["hospitalid"] = _site_label(eicu["hospitalid"])
        for hospitalid, group in eicu.dropna(subset=["hospitalid"]).groupby("hospitalid", sort=True):
            hospitalid = str(hospitalid)
            site_row = site_row_by_key[(model_id, hospitalid)]
            summary, estimates, cluster_digest = _site_slope_ci(
                group, model_id=model_id, hospitalid=hospitalid,
                point_status=str(site_row["calibration_status"]),
                n_boot=n_boot, seed=seed, checkpoint_dir=checkpoint_dir,
            )
            site_row.update(summary)
            results[(model_id, hospitalid)] = {
                "summary": summary, "draws": estimates,
                "patient_cluster_set_sha256": cluster_digest,
            }
            print(json.dumps({
                "stage": "hospital_site_slope_checkpoint_complete",
                "model_id": model_id, "hospitalid": hospitalid,
                "status": summary["calibration_slope_ci_status"],
                "n_patients": summary["calibration_slope_ci_n_patients"],
                "draws_requested": summary["calibration_slope_ci_bootstrap_draws_requested"],
                "draws_valid": summary["calibration_slope_ci_bootstrap_draws_valid"],
            }, sort_keys=True), flush=True)
    for row in site_rows:
        if row.get("scope") == "hospitalid_missing_aggregate":
            row.update({
                "calibration_slope_ci_95_lo": float("nan"),
                "calibration_slope_ci_95_hi": float("nan"),
                "calibration_slope_ci_status": "not_applicable_missing_hospital_id_aggregate",
                "calibration_slope_ci_bootstrap_draws_requested": int(n_boot),
                "calibration_slope_ci_bootstrap_draws_valid": 0,
                "calibration_slope_ci_bootstrap_draws_invalid_not_attempted": int(n_boot),
                "calibration_slope_ci_eligibility": "not_a_selected_hospital",
            })
    return results


def _attach_paired_slope_cis(
    pair_rows: list[dict[str, Any]], pairs: dict[str, pd.DataFrame],
    slope_results: dict[tuple[str, str], dict[str, Any]], *, n_boot: int,
) -> None:
    model_by_comparison = {
        "main20k_real_minus_main20k_constant": "main20k_constant",
        "main20k_real_minus_conventional_lr_value_plus_pattern": "conventional_lr_value_plus_pattern",
    }
    for comparison_id, frame in pairs.items():
        eicu = frame.loc[frame["db"].astype(str) == "eicu"].copy()
        eicu["hospitalid"] = _site_label(eicu["hospitalid"])
        expected_sites = set(eicu["hospitalid"].dropna().astype(str))
        reported_sites = {
            str(row["hospitalid"]) for row in pair_rows
            if row["comparison_id"] == comparison_id
        }
        if expected_sites != reported_sites:
            raise ValueError(
                f"{comparison_id}: paired site slope rows do not cover the exact mapped site set "
                f"(expected={len(expected_sites)}, reported={len(reported_sites)})"
            )
    for comparison_id, comparator_id in model_by_comparison.items():
        for row in pair_rows:
            if row["comparison_id"] != comparison_id:
                continue
            hospitalid = str(row["hospitalid"])
            real_result = slope_results[("main20k_real", hospitalid)]
            comparator_result = slope_results[(comparator_id, hospitalid)]
            real_summary = real_result["summary"]
            comparator_summary = comparator_result["summary"]
            row.update({
                "real_calibration_slope_ci_95_lo": real_summary["calibration_slope_ci_95_lo"],
                "real_calibration_slope_ci_95_hi": real_summary["calibration_slope_ci_95_hi"],
                "real_calibration_slope_ci_status": real_summary["calibration_slope_ci_status"],
                "comparator_calibration_slope_ci_95_lo": comparator_summary["calibration_slope_ci_95_lo"],
                "comparator_calibration_slope_ci_95_hi": comparator_summary["calibration_slope_ci_95_hi"],
                "comparator_calibration_slope_ci_status": comparator_summary["calibration_slope_ci_status"],
            })
            if (real_result["patient_cluster_set_sha256"]
                    != comparator_result["patient_cluster_set_sha256"]):
                raise ValueError(f"{comparison_id}/{hospitalid}: patient cluster sets differ across arms")
            real_draws = real_result["draws"]
            comparator_draws = comparator_result["draws"]
            if len(real_draws) != n_boot or len(comparator_draws) != n_boot:
                raise ValueError(f"{comparison_id}/{hospitalid}: shared slope bootstrap draw counts differ")
            valid = np.isfinite(real_draws) & np.isfinite(comparator_draws)
            delta = real_draws[valid] - comparator_draws[valid]
            required = max(200, int(math.ceil(0.8 * n_boot)))
            if not row["stability_flag_ge10_events_and_nonevents"]:
                lo = hi = float("nan")
                status = "diagnostic_not_for_manuscript_below_10_events_or_nonevents"
            elif (real_summary["calibration_slope_ci_status"] != "ok"
                  or comparator_summary["calibration_slope_ci_status"] != "ok"):
                lo = hi = float("nan")
                status = "diagnostic_not_for_manuscript_component_slope_unestimable"
            elif len(delta) >= required:
                lo, hi = (float(x) for x in np.percentile(delta, [2.5, 97.5]))
                status = "ok_shared_patient_cluster_draws"
            else:
                lo = hi = float("nan")
                status = "unstable_insufficient_paired_valid_draws"
            row.update({
                "delta_calibration_slope_ci_95_lo": lo,
                "delta_calibration_slope_ci_95_hi": hi,
                "delta_calibration_slope_ci_status": status,
                "delta_calibration_slope_ci_bootstrap_draws_requested": int(n_boot),
                "delta_calibration_slope_ci_bootstrap_draws_valid": int(len(delta)),
                "delta_calibration_slope_ci_scope": "paired site stays; shared patient-cluster resampling across frozen prediction arms",
            })


def metric_row(frame: pd.DataFrame, *, model_id: str, scope: str,
               hospitalid: str | None = None) -> dict[str, Any]:
    y = frame["y"].to_numpy(dtype=int)
    p = frame["p"].to_numpy(dtype=float)
    n_events = int(y.sum())
    n_nonevents = int(len(y) - n_events)
    if np.unique(y).size == 2:
        auc = float(roc_auc_score(y, p))
        auc_status = "ok"
    else:
        auc = float("nan")
        auc_status = "unestimable_single_class"
    if n_events > 0:
        ap = float(average_precision_score(y, p))
        ap_status = "ok"
    else:
        ap = float("nan")
        ap_status = "unestimable_no_events"
    intercept, slope, cal_status = _calibration(y, p)
    return {
        "scope": scope, "model_id": model_id, "hospitalid": hospitalid,
        "n_stays": int(len(frame)), "n_patients": int(frame["subject"].nunique()),
        "events": n_events, "nonevents": n_nonevents,
        "prevalence": float(np.mean(y)),
        "auroc": auc, "auroc_status": auc_status,
        "average_precision": ap, "average_precision_status": ap_status,
        "brier": float(np.mean((y - p) ** 2)), "logloss": logloss(y, p),
        "calibration_intercept": intercept, "calibration_slope": slope,
        "calibration_status": cal_status,
        "site_stability_flag_ge10_events_and_nonevents": bool(n_events >= 10 and n_nonevents >= 10),
        "site_stability_rule_excludes_from_cohort": False,
    }


def patient_hospital_crossing(frame: pd.DataFrame) -> dict[str, Any]:
    mapped = frame.copy()
    mapped["hospitalid"] = _site_label(mapped["hospitalid"])
    mapped = mapped.dropna(subset=["hospitalid"])
    sites_per_patient = mapped.groupby("subject", sort=False)["hospitalid"].nunique()
    crossing = sites_per_patient[sites_per_patient > 1]
    return {
        "n_mapped_patients": int(sites_per_patient.size),
        "n_patients_crossing_hospitals": int(len(crossing)),
        "n_hospital_patient_pairs": int(len(mapped[["subject", "hospitalid"]].drop_duplicates())),
        "patients_are_nested_within_hospital": bool(crossing.empty),
    }


def paired_frame(left: pd.DataFrame, right: pd.DataFrame, *, label: str) -> pd.DataFrame:
    keys = ["db", "stay_id"]
    for name, frame in (("left", left), ("right", right)):
        if frame.duplicated(keys).any():
            raise ValueError(f"{label}: {name} has duplicate cohort/stay keys")
    lkeys = set(map(tuple, left[keys].itertuples(index=False, name=None)))
    rkeys = set(map(tuple, right[keys].itertuples(index=False, name=None)))
    if lkeys != rkeys:
        raise ValueError(f"{label}: exact eligible ID sets differ (left={len(lkeys)}, right={len(rkeys)})")
    identity = ["subject", "rows", "split", "y", "hospitalid"]
    merged = left.merge(right, on=keys, how="inner", suffixes=("_left", "_right"), validate="one_to_one")
    for field in identity:
        a = merged[f"{field}_left"].astype("string").fillna("<missing>")
        b = merged[f"{field}_right"].astype("string").fillna("<missing>")
        if not a.equals(b):
            raise ValueError(f"{label}: paired stay identity differs in {field}")
    endpoint_fields = [
        field for field in ("endpoint_prediction", "endpoint_eligibility")
        if f"{field}_left" in merged.columns and f"{field}_right" in merged.columns
    ]
    if not endpoint_fields and "endpoint_left" in merged.columns and "endpoint_right" in merged.columns:
        endpoint_fields = ["endpoint"]
    if not endpoint_fields:
        raise ValueError(f"{label}: paired predictions lack endpoint identity columns")
    for field in endpoint_fields:
        a = merged[f"{field}_left"].astype("string").fillna("<missing>")
        b = merged[f"{field}_right"].astype("string").fillna("<missing>")
        if not a.equals(b):
            raise ValueError(f"{label}: paired stay identity differs in {field}")
    endpoint_source = (
        "endpoint_prediction_left" if "endpoint_prediction_left" in merged.columns
        else "endpoint_left"
    )
    out = pd.DataFrame({
        "db": merged["db"], "stay_id": merged["stay_id"],
        "subject": merged["subject_left"], "rows": merged["rows_left"],
        "split": merged["split_left"], "y": merged["y_left"],
        "endpoint": merged[endpoint_source],
        "hospitalid": _site_label(merged["hospitalid_left"]),
        "p_real": merged["p_left"].astype(float),
        "p_comparator": merged["p_right"].astype(float),
    })
    return out


def _site_metric_table(frames: dict[str, pd.DataFrame]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    for model_id, frame in frames.items():
        eicu = frame.loc[frame["db"].astype(str) == "eicu"].copy()
        if eicu.empty:
            raise ValueError(f"{model_id}: current strict loader returned no eICU rows")
        cohort_rows.append(metric_row(eicu, model_id=model_id, scope="eicu_all_eligible_stays"))
        eicu["hospitalid"] = _site_label(eicu["hospitalid"])
        for hospitalid, group in eicu.dropna(subset=["hospitalid"]).groupby("hospitalid", sort=True):
            metrics.append(metric_row(group, model_id=model_id,
                                      scope="selected_hospital_all_stays",
                                      hospitalid=str(hospitalid)))
        missing = eicu.loc[eicu["hospitalid"].isna()]
        if not missing.empty:
            metrics.append(metric_row(missing, model_id=model_id,
                                      scope="hospitalid_missing_aggregate",
                                      hospitalid=None))
    return cohort_rows, metrics


def _selected_site_metric_rows(site_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return mapped selected hospitals, excluding the missing-ID aggregate."""
    return [row for row in site_rows if row.get("scope") == "selected_hospital_all_stays"]


def _site_count_preflight(models: dict[str, pd.DataFrame], *, n_boot: int) -> dict[str, Any]:
    """Summarize strict-prediction site coverage and CI workload without fitting CIs."""
    if not models:
        raise ValueError("site-count preflight requires at least one strict prediction model")
    sites_by_model: dict[str, set[str]] = {}
    summary_by_model: dict[str, dict[str, int]] = {}
    missing_by_model: dict[str, int] = {}
    for model_id, frame in models.items():
        eicu = frame.loc[frame["db"].astype(str) == "eicu"].copy()
        if eicu.empty:
            raise ValueError(f"{model_id}: current strict loader returned no eICU rows")
        eicu["hospitalid"] = _site_label(eicu["hospitalid"])
        missing_by_model[model_id] = int(eicu["hospitalid"].isna().sum())
        mapped = eicu.dropna(subset=["hospitalid"])
        sites_by_model[model_id] = set(mapped["hospitalid"].astype(str).unique())
        n_all = len(sites_by_model[model_id])
        n_stable = 0
        n_single_class = 0
        for _, group in mapped.groupby("hospitalid", sort=False):
            events = int(group["y"].sum())
            nonevents = int(len(group) - events)
            if events >= MIN_SITE_EVENTS and nonevents >= MIN_SITE_EVENTS:
                n_stable += 1
            if events == 0 or nonevents == 0:
                n_single_class += 1
        summary_by_model[model_id] = {
            "selected_hospitals": n_all,
            "sites_ge_10_events_and_nonevents": n_stable,
            "sites_below_10_events_or_nonevents": n_all - n_stable,
            "single_class_sites": n_single_class,
            "hospital_id_missing_prediction_stays": missing_by_model[model_id],
        }
    canonical = next(iter(sites_by_model.values()))
    mismatched = {
        model_id: len(sites.symmetric_difference(canonical))
        for model_id, sites in sites_by_model.items()
        if sites != canonical
    }
    if mismatched:
        raise ValueError(f"strict prediction models have different selected hospital sets: {mismatched}")
    n_models = len(models)
    selected_site_model_jobs = len(canonical) * n_models
    stable_site_model_upper_bound = sum(
        values["sites_ge_10_events_and_nonevents"] for values in summary_by_model.values()
    )
    return {
        "status": "complete",
        "scope": "strict-manifest aggregate-only site-count preflight; no calibration fits or confidence intervals",
        "selected_hospital_ids_with_current_predictions": len(canonical),
        "selected_hospital_metric_rows_by_model": {
            model_id: values["selected_hospitals"] for model_id, values in summary_by_model.items()
        },
        "site_counts_by_model": summary_by_model,
        "workload": {
            "models": n_models,
            "site_model_checkpoint_jobs": selected_site_model_jobs,
            "patient_cluster_bootstrap_draws_requested_per_site_model": int(n_boot),
            "event_threshold_eligible_site_model_bootstrap_upper_bound": stable_site_model_upper_bound,
            "maximum_patient_cluster_bootstrap_draws_by_event_threshold": int(
                stable_site_model_upper_bound * n_boot
            ),
            "note": "Actual bootstrap count can be lower when a site calibration point fit is unestimable.",
        },
    }


def preflight_current_manifests(*, n_boot: int = N_BOOT, seed: int = 20261122) -> dict[str, Any]:
    """Load only the registered current prediction arms and report site counts."""
    fm_manifest = FM_RUN / "prediction_manifest.json"
    baseline_manifest = BASELINE_RESULTS / "baseline_prediction_manifest.json"
    print(json.dumps({
        "stage": "hospital_site_preflight_manifest_load_start",
        "current_fm_manifest": str(fm_manifest),
        "current_baseline_manifest": str(baseline_manifest),
    }, sort_keys=True), flush=True)
    real, gates_real = load_checked(
        FM_RUN / "main20k_real.parquet", ELIGIBILITY_DIR, model_tag="main20k",
        input_condition="real", prediction_manifest_path=fm_manifest,
    )
    print(json.dumps({
        "stage": "hospital_site_preflight_model_loaded", "model_id": "main20k_real",
        "rows_by_cohort": {db: int(len(frame)) for db, frame in real.items()},
    }, sort_keys=True), flush=True)
    constant, gates_constant = load_checked(
        FM_RUN / "main20k_constant.parquet", ELIGIBILITY_DIR, model_tag="main20k",
        input_condition="constant", prediction_manifest_path=fm_manifest,
    )
    print(json.dumps({
        "stage": "hospital_site_preflight_model_loaded", "model_id": "main20k_constant",
        "rows_by_cohort": {db: int(len(frame)) for db, frame in constant.items()},
    }, sort_keys=True), flush=True)
    lr, gates_lr = load_checked(
        BASELINE_RESULTS / "predictions/conventional_lr_value_plus_pattern.parquet",
        ELIGIBILITY_DIR, model_tag="conventional_lr",
        analysis_id="conventional_lr_value_plus_pattern", input_condition="real",
        prediction_manifest_path=baseline_manifest,
    )
    print(json.dumps({
        "stage": "hospital_site_preflight_model_loaded",
        "model_id": "conventional_lr_value_plus_pattern",
        "rows_by_cohort": {db: int(len(frame)) for db, frame in lr.items()},
    }, sort_keys=True), flush=True)
    if set(real) != {"mimic", "eicu"} or set(constant) != set(real) or set(lr) != set(real):
        raise ValueError("current real, constant, and LR strict loaders returned different cohort sets")
    real_eicu, constant_eicu, lr_eicu = real["eicu"], constant["eicu"], lr["eicu"]
    comparisons = {
        "main20k_real_minus_main20k_constant": paired_frame(
            real_eicu, constant_eicu, label="FM real minus constant"),
        "main20k_real_minus_conventional_lr_value_plus_pattern": paired_frame(
            real_eicu, lr_eicu, label="FM real minus LR value-plus-pattern"),
    }
    models = {
        "main20k_real": real_eicu,
        "main20k_constant": constant_eicu,
        "conventional_lr_value_plus_pattern": lr_eicu,
    }
    result = _site_count_preflight(models, n_boot=n_boot)
    result.update({
        "seed": int(seed),
        "current_fm_prediction_manifest": str(fm_manifest),
        "current_fm_prediction_manifest_sha256": sha256_file(fm_manifest),
        "current_baseline_prediction_manifest": str(baseline_manifest),
        "current_baseline_prediction_manifest_sha256": sha256_file(baseline_manifest),
        "eligibility_manifest_sha256": gates_real["eicu"]["eligibility_manifest_sha256"],
        "eligibility_table_sha256": gates_real["eicu"]["eligibility_table_sha256"],
        "paired_id_checks": {
            name: {"status": "exact", "n_matched_eicu_stays": int(len(frame)),
                   "n_matched_patients": int(frame["subject"].nunique())}
            for name, frame in comparisons.items()
        },
        "cohort_gates": {
            "main20k_real": gates_real["eicu"],
            "main20k_constant": gates_constant["eicu"],
            "conventional_lr_value_plus_pattern": gates_lr["eicu"],
        },
    })
    return result


def _paired_site_rows(comparisons: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comparison_id, frame in comparisons.items():
        eicu = frame.loc[frame["db"].astype(str) == "eicu"].copy()
        eicu["hospitalid"] = _site_label(eicu["hospitalid"])
        for hospitalid, group in eicu.dropna(subset=["hospitalid"]).groupby("hospitalid", sort=True):
            left = group.assign(p=group["p_real"])
            right = group.assign(p=group["p_comparator"])
            a = metric_row(left, model_id=f"{comparison_id}:real", scope="paired_hospital")
            b = metric_row(right, model_id=f"{comparison_id}:comparator", scope="paired_hospital")
            rows.append({
                "comparison_id": comparison_id, "hospitalid": str(hospitalid),
                "n_stays": int(len(group)), "n_patients": int(group["subject"].nunique()),
                "events": int(group["y"].sum()), "nonevents": int(len(group) - group["y"].sum()),
                **{f"real_{key}": a[key] for key in (
                    "auroc", "average_precision", "brier", "logloss",
                    "calibration_intercept", "calibration_slope")},
                **{f"comparator_{key}": b[key] for key in (
                    "auroc", "average_precision", "brier", "logloss",
                    "calibration_intercept", "calibration_slope")},
                **{f"delta_{key}": (a[key] - b[key] if np.isfinite(a[key]) and np.isfinite(b[key]) else np.nan)
                   for key in ("auroc", "average_precision", "brier", "logloss",
                               "calibration_intercept", "calibration_slope")},
                "auroc_pair_status": "ok" if np.isfinite(a["auroc"]) and np.isfinite(b["auroc"]) else "unestimable_single_class",
                "average_precision_pair_status": "ok" if np.isfinite(a["average_precision"]) and np.isfinite(b["average_precision"]) else "unestimable_no_events",
                "stability_flag_ge10_events_and_nonevents": bool(
                    group["y"].sum() >= 10 and (1 - group["y"]).sum() >= 10),
                "site_retained_in_overall_cohort": True,
            })
    return rows


def _site_bootstrap_ci(metric_rows: list[dict[str, Any]], *, group_field: str,
                       metrics: tuple[str, ...], n_boot: int, seed: int,
                       crossing: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in metric_rows:
        groups.setdefault(str(row[group_field]), []).append(row)
    scope = ("site-resampling-only; patients cross hospitals; no patient-independence claim"
             if crossing else "hospital-cluster bootstrap over selected hospitals; patients are nested")
    progress_interval = max(1, n_boot // 10)
    for group_key, rows in groups.items():
        frame = pd.DataFrame(rows)
        for metric in metrics:
            column = metric if metric in frame.columns else f"delta_{metric}"
            if column not in frame.columns:
                continue
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values) < 2:
                out.append({group_field: group_key, "metric": metric, "n_hospitals_estimable": int(len(values)),
                            "status": "insufficient_estimable_hospitals", "ci_scope": scope})
                continue
            rng = np.random.default_rng(seed + sum(ord(ch) for ch in f"{group_key}:{metric}"))
            estimates = np.empty(n_boot, dtype=float)
            for draw in range(n_boot):
                estimates[draw] = float(np.mean(values[rng.integers(0, len(values), size=len(values))]))
                if (draw + 1) % progress_interval == 0 or draw + 1 == n_boot:
                    print(json.dumps({
                        "stage": "hospital_site_bootstrap_progress", group_field: group_key,
                        "metric": metric, "draw_completed": draw + 1,
                        "draws_requested": n_boot,
                    }, sort_keys=True), flush=True)
            lo, hi = np.percentile(estimates, [2.5, 97.5])
            out.append({
                group_field: group_key, "metric": metric,
                "n_hospitals_estimable": int(len(values)), "estimate_equal_site_mean": float(values.mean()),
                "equal_site_mean_ci_95_lo": float(lo), "equal_site_mean_ci_95_hi": float(hi),
                "site_resampling_draws": n_boot, "status": "ok", "ci_scope": scope,
            })
    return out


def run(*, n_boot: int = N_BOOT, seed: int = 20261122,
        checkpoint_dir: Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    fm_manifest = FM_RUN / "prediction_manifest.json"
    baseline_manifest = BASELINE_RESULTS / "baseline_prediction_manifest.json"
    eligibility_dir = ELIGIBILITY_DIR
    fm_real_path = FM_RUN / "main20k_real.parquet"
    fm_constant_path = FM_RUN / "main20k_constant.parquet"
    lr_path = BASELINE_RESULTS / "predictions/conventional_lr_value_plus_pattern.parquet"
    print(json.dumps({
        "stage": "hospital_manifest_load_start",
        "bootstrap_draws_per_site_model": int(n_boot),
        "fm_manifest": str(fm_manifest),
        "baseline_manifest": str(baseline_manifest),
    }, sort_keys=True), flush=True)
    real, gates_real = load_checked(
        fm_real_path, eligibility_dir, model_tag="main20k", input_condition="real",
        prediction_manifest_path=fm_manifest)
    print(json.dumps({
        "stage": "hospital_model_loaded", "model_id": "main20k_real",
        "rows_by_cohort": {db: int(len(frame)) for db, frame in real.items()},
    }, sort_keys=True), flush=True)
    constant, gates_constant = load_checked(
        fm_constant_path, eligibility_dir, model_tag="main20k", input_condition="constant",
        prediction_manifest_path=fm_manifest)
    print(json.dumps({
        "stage": "hospital_model_loaded", "model_id": "main20k_constant",
        "rows_by_cohort": {db: int(len(frame)) for db, frame in constant.items()},
    }, sort_keys=True), flush=True)
    lr, gates_lr = load_checked(
        lr_path, eligibility_dir, model_tag="conventional_lr",
        analysis_id="conventional_lr_value_plus_pattern", input_condition="real",
        prediction_manifest_path=baseline_manifest)
    print(json.dumps({
        "stage": "hospital_model_loaded",
        "model_id": "conventional_lr_value_plus_pattern",
        "rows_by_cohort": {db: int(len(frame)) for db, frame in lr.items()},
    }, sort_keys=True), flush=True)
    if set(real) != {"mimic", "eicu"} or set(constant) != set(real) or set(lr) != set(real):
        raise ValueError("current real, constant, and LR strict loaders returned different cohort sets")

    real_eicu = real["eicu"].copy()
    constant_eicu = constant["eicu"].copy()
    lr_eicu = lr["eicu"].copy()
    pairs = {
        "main20k_real_minus_main20k_constant": paired_frame(real_eicu, constant_eicu,
                                                            label="FM real minus constant"),
        "main20k_real_minus_conventional_lr_value_plus_pattern": paired_frame(
            real_eicu, lr_eicu, label="FM real minus LR value-plus-pattern"),
    }
    models = {
        "main20k_real": real_eicu,
        "main20k_constant": constant_eicu,
        "conventional_lr_value_plus_pattern": lr_eicu,
    }
    cohort_rows, site_rows = _site_metric_table(models)
    pair_rows = _paired_site_rows(pairs)
    real_eicu["hospitalid"] = _site_label(real_eicu["hospitalid"])
    site_count = int(real_eicu["hospitalid"].dropna().nunique())
    selected_site_rows = _selected_site_metric_rows(site_rows)
    model_site_counts = {
        model_id: len({row["hospitalid"] for row in selected_site_rows
                       if row["model_id"] == model_id})
        for model_id in models
    }
    duplicate_site_rows = len(selected_site_rows) != len({
        (row["model_id"], row["hospitalid"]) for row in selected_site_rows
    })
    if duplicate_site_rows or any(count != site_count for count in model_site_counts.values()):
        raise ValueError(
            "selected hospital metric coverage differs across models or from the paired FM real cohort "
            f"(expected={site_count}, by_model={model_site_counts}, duplicate_rows={duplicate_site_rows})"
        )
    print(json.dumps({
        "stage": "hospital_site_inventory_ready",
        "selected_hospitals": site_count,
        "selected_hospital_metric_rows_by_model": model_site_counts,
        "site_model_slope_jobs": site_count * len(models),
        "stable_site_counts_by_model": {
            model_id: sum(1 for row in selected_site_rows
                          if row["model_id"] == model_id
                          and row["site_stability_flag_ge10_events_and_nonevents"])
            for model_id in models
        },
        "patient_cluster_draws_requested_per_site_model": int(n_boot),
    }, sort_keys=True), flush=True)
    checkpoint_dir = checkpoint_dir or (REV / "robustness/results/hospital_primary_v4/site_slope_ci_checkpoints")
    slope_results = _attach_site_slope_cis(
        models, site_rows, n_boot=n_boot, seed=seed, checkpoint_dir=checkpoint_dir,
    )
    _attach_paired_slope_cis(pair_rows, pairs, slope_results, n_boot=n_boot)
    # Read the current full eligibility table only for the source-hospital inventory count.
    eligibility_path = eligibility_dir / "eicu.parquet"
    eligibility_hash = sha256_file(eligibility_path)
    if eligibility_hash != gates_real["eicu"]["eligibility_table_sha256"]:
        raise ValueError("eICU source hospital inventory table hash differs from current loaded gate")
    eligibility_meta = pd.read_parquet(eligibility_path, columns=["hospitalid", "eligible_24h", "temporal_verified"])
    source_subset = eligibility_meta.loc[eligibility_meta["eligible_24h"].astype(bool)
                                         & eligibility_meta["temporal_verified"].astype(bool)]
    source_site_count = int(_site_label(source_subset["hospitalid"]).nunique())
    crossing = patient_hospital_crossing(real_eicu)
    ci_rows = _site_bootstrap_ci(selected_site_rows, group_field="model_id",
                                 metrics=("auroc", "average_precision", "brier",
                                          "calibration_slope"), n_boot=n_boot,
                                 seed=seed, crossing=bool(crossing["n_patients_crossing_hospitals"]))
    paired_ci = _site_bootstrap_ci(pair_rows, group_field="comparison_id",
                                   metrics=("auroc", "average_precision", "brier",
                                            "calibration_slope"), n_boot=n_boot,
                                   seed=seed + 10000,
                                   crossing=bool(crossing["n_patients_crossing_hospitals"]))
    ci_rows.extend(paired_ci)
    site_slope_ci_status_counts = {
        model_id: {
            status: sum(1 for row in selected_site_rows
                        if row["model_id"] == model_id
                        and row["calibration_slope_ci_status"] == status)
            for status in sorted({row["calibration_slope_ci_status"] for row in selected_site_rows
                                  if row["model_id"] == model_id})
        }
        for model_id in models
    }
    stable_site_counts = {
        model_id: sum(1 for row in selected_site_rows
                      if row["model_id"] == model_id
                      and row["site_stability_flag_ge10_events_and_nonevents"])
        for model_id in models
    }
    checkpoint_files = list(checkpoint_dir.glob("site_slope_ci_*.json"))
    if len(slope_results) != site_count * len(models):
        raise ValueError(
            f"site slope bootstrap coverage incomplete: got {len(slope_results)} rows "
            f"for {site_count} selected sites and {len(models)} models"
        )
    manifest = {
        "status": "complete", "seed": int(seed), "bootstrap_draws": int(n_boot),
        "current_fm_prediction_manifest": str(fm_manifest),
        "current_fm_prediction_manifest_sha256": sha256_file(fm_manifest),
        "current_baseline_prediction_manifest": str(baseline_manifest),
        "current_baseline_prediction_manifest_sha256": sha256_file(baseline_manifest),
        "eligibility_manifest_sha256": gates_real["eicu"]["eligibility_manifest_sha256"],
        "eligibility_table_sha256": eligibility_hash,
        "source_hospital_ids_in_current_eligible_metadata": source_site_count,
        "selected_hospital_ids_with_current_predictions": site_count,
        "selected_hospital_metric_rows_by_model": model_site_counts,
        "selected_hospital_metric_rows_per_model": site_count,
        "within_site_calibration_slope_ci": {
            "estimand": "per selected hospital calibration slope; all eligible stays, patient-cluster resampling within hospital",
            "bootstrap_draws_requested": int(n_boot),
            "site_stability_eligibility": "at least 10 events and 10 nonevents; other hospital rows remain present with explicit diagnostic status",
            "minimum_valid_draws_for_interval": max(200, int(math.ceil(0.8 * n_boot))),
            "solver": "batched two-parameter logistic Newton solver matching LogisticRegression(lbfgs,C=1e6): slope L2=1e-6 and unpenalized intercept; no coefficient clipping; separation, no variation, nonconvergence, nonfinite and absolute slope >50 are counted as invalid draws",
            "paired_comparisons_use_shared_patient_cluster_draws": True,
            "stable_site_counts_by_model": stable_site_counts,
            "site_slope_ci_status_counts_by_model": site_slope_ci_status_counts,
            "site_model_checkpoints_completed": len(slope_results),
            "site_model_checkpoint_files_present": len(checkpoint_files),
            "checkpoint_directory": str(checkpoint_dir),
        },
        "hospital_id_missing_prediction_stays": int(real_eicu["hospitalid"].isna().sum()),
        "patient_hospital_crossing": crossing,
        "paired_id_checks": {
            name: {"status": "exact", "n_matched_eicu_stays": int(len(frame)),
                   "n_matched_patients": int(frame["subject"].nunique())}
            for name, frame in pairs.items()},
        "cohort_gates": {"main20k_real": gates_real, "main20k_constant": gates_constant,
                         "conventional_lr_value_plus_pattern": gates_lr},
        "site_scope": "all selected eICU hospitals are retained; >=10 events and >=10 nonevents is a display/stability flag only",
        "calibration": "intercept and slope fitted as diagnostics to frozen predictions; per-site slope CIs use patient-cluster bootstrap; no clinical predictor model is refit",
        "bootstrap_scope": ci_rows[0]["ci_scope"] if ci_rows else "no estimable hospitals",
        "prediction_scope": "retrospective landmark prediction at recorded measurement time",
        "clinical_input_availability_verified": False,
        "clinical_promotion_allowed": False,
        "patient_level_exported": False,
    }
    return cohort_rows, site_rows, pair_rows, {"manifest": manifest, "bootstrap_rows": ci_rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(REV / "robustness/results/hospital_primary_v4"))
    parser.add_argument("--bootstrap", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--preflight", action="store_true",
                        help="validate current manifests and count site-CI workload without fitting CIs")
    parser.add_argument("--preflight-output", type=Path,
                        help="optional aggregate JSON output path for --preflight")
    args = parser.parse_args(argv)
    if args.preflight:
        result = preflight_current_manifests(n_boot=args.bootstrap, seed=args.seed)
        encoded = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)
        if args.preflight_output is not None:
            args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
            args.preflight_output.write_text(encoded + "\n", encoding="utf-8")
        print(encoded, flush=True)
        return 0
    if args.bootstrap < 1000:
        parser.error("--bootstrap must be at least 1000 site-resampling draws")
    out_dir = Path(args.out_dir)
    completed_manifest = out_dir / "hospital_analysis_manifest.json"
    if completed_manifest.is_file():
        parser.error(f"refusing to overwrite completed hospital output: {completed_manifest}")
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "site_slope_ci_checkpoints"
    cohorts, sites, pairs, aux = run(
        n_boot=args.bootstrap, seed=args.seed, checkpoint_dir=checkpoint_dir,
    )
    write_rows(out_dir / "eicu_overall_performance.csv", cohorts)
    write_rows(out_dir / "eicu_hospital_metrics.csv", sites)
    write_rows(out_dir / "eicu_hospital_paired_deltas.csv", pairs)
    write_rows(out_dir / "eicu_hospital_site_bootstrap_ci.csv", aux["bootstrap_rows"])
    (out_dir / "hospital_analysis_manifest.json").write_text(
        json.dumps(aux["manifest"], indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "status": "complete", "overall_rows": len(cohorts), "hospital_metric_rows": len(sites),
        "paired_delta_rows": len(pairs), "bootstrap_ci_rows": len(aux["bootstrap_rows"]),
        "selected_hospitals": aux["manifest"]["selected_hospital_ids_with_current_predictions"],
        "patient_hospital_crossing": aux["manifest"]["patient_hospital_crossing"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
