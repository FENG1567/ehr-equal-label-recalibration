"""Paired patient-cluster AUROC/AP intervals for fixed recovered predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from paired_baselines import sha256_file
from transport_analysis import (
    ANALYSIS_IDS, MODEL_IDS, N_BOOT, REV, SEED, TARGET_COHORTS,
    _load_all_model_frames, _paired_target_predictions, _write_json_atomic,
    _write_rows_atomic,
)

DEFAULT_OUT = REV / "robustness/results/recovered_transport_v4_2_auroc_ap_ci_20261123"
DEFAULT_POINT_REFERENCE = (
    REV / "robustness/results/recovered_transport_v4_2_20261123"
    / "transport_paired_deltas.csv"
)
BOOTSTRAP_BATCH = 16


def _weighted_rank_metric_draws(
    labels: np.ndarray, scores: np.ndarray, row_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute weighted AUROC/AP for each row of cluster-bootstrap weights.

    Scores are sorted once and tied scores are evaluated as a single threshold,
    matching sklearn's weighted AUROC and average-precision definitions.
    """
    y = np.asarray(labels, dtype=np.int8).reshape(-1)
    p = np.asarray(scores, dtype=float).reshape(-1)
    weights = np.asarray(row_weights, dtype=float)
    if (weights.ndim != 2 or weights.shape[1] != len(y) or len(y) != len(p)
            or len(y) == 0 or not np.isin(y, [0, 1]).all()
            or not np.isfinite(p).all() or not np.isfinite(weights).all()
            or (weights < 0).any()):
        raise ValueError("weighted ranking inputs have invalid dimensions or values")

    order = np.argsort(-p, kind="mergesort")
    sorted_p = p[order]
    group_starts = np.r_[0, np.flatnonzero(sorted_p[1:] != sorted_p[:-1]) + 1]
    sorted_y = y[order].astype(float, copy=False)
    sorted_weights = weights[:, order]
    positive_by_score = np.add.reduceat(
        sorted_weights * sorted_y[None, :], group_starts, axis=1,
    )
    negative_by_score = np.add.reduceat(
        sorted_weights * (1.0 - sorted_y)[None, :], group_starts, axis=1,
    )
    total_positive = positive_by_score.sum(axis=1)
    total_negative = negative_by_score.sum(axis=1)
    valid = (total_positive > 0) & (total_negative > 0)

    auc = np.full(weights.shape[0], np.nan, dtype=float)
    average_precision = np.full(weights.shape[0], np.nan, dtype=float)
    if valid.any():
        cumulative_positive = np.cumsum(positive_by_score, axis=1)
        cumulative_negative = np.cumsum(negative_by_score, axis=1)
        lower_negative = total_negative[:, None] - cumulative_negative
        auc_numerator = np.sum(
            positive_by_score * (lower_negative + 0.5 * negative_by_score), axis=1,
        )
        auc[valid] = auc_numerator[valid] / (total_positive[valid] * total_negative[valid])
        precision = np.divide(
            cumulative_positive, cumulative_positive + cumulative_negative,
            out=np.zeros_like(cumulative_positive),
            where=(cumulative_positive + cumulative_negative) > 0,
        )
        ap_values = np.sum(
            np.divide(positive_by_score, total_positive[:, None],
                      out=np.zeros_like(positive_by_score),
                      where=total_positive[:, None] > 0) * precision,
            axis=1,
        )
        average_precision[valid] = ap_values[valid]
    return auc, average_precision


def _interval(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        lo = hi = float("nan")
    else:
        lo, hi = np.percentile(finite, [2.5, 97.5])
    return {
        "ci_95_lo": float(lo), "ci_95_hi": float(hi),
        "valid_draws": int(len(finite)),
    }


def _cluster_bootstrap_auc_ap(
    pairs: dict[str, pd.DataFrame], *, seed: int, n_boot: int = N_BOOT,
    batch_size: int = BOOTSTRAP_BATCH,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Use common patient-resampling weights for FM and every comparator."""
    if n_boot < 1000 or batch_size < 1:
        raise ValueError("paired rank-metric patient intervals require at least 1000 draws")
    if not pairs:
        raise ValueError("paired rank-metric bootstrap requires at least one comparator")
    first = next(iter(pairs.values()))
    identity_fields = ("db", "stay_id", "subject", "rows", "split", "y")
    for name, frame in pairs.items():
        for field in identity_fields:
            if not frame[field].astype(str).equals(first[field].astype(str)):
                raise ValueError(f"shared rank bootstrap has different stay ordering in {name}/{field}")

    subjects = first["subject"].astype(str).to_numpy()
    unique_subjects, inverse = np.unique(subjects, return_inverse=True)
    if len(unique_subjects) < 2:
        raise ValueError("patient-cluster ranking intervals require at least two patients")
    y = first["y"].to_numpy(dtype=np.int8)
    model_scores: dict[str, np.ndarray] = {
        "main20k_real": first["p_fm"].to_numpy(dtype=float),
    }
    for analysis_id, frame in pairs.items():
        current_fm = frame["p_fm"].to_numpy(dtype=float)
        if not np.array_equal(current_fm, model_scores["main20k_real"]):
            raise ValueError("shared rank bootstrap has different FM predictions across comparators")
        model_scores[analysis_id] = frame["p_baseline"].to_numpy(dtype=float)

    draws = {
        model_id: {
            "auroc": np.full(n_boot, np.nan, dtype=float),
            "average_precision": np.full(n_boot, np.nan, dtype=float),
        }
        for model_id in model_scores
    }
    rng = np.random.default_rng(seed)
    one_class_draws = 0
    for start in range(0, n_boot, batch_size):
        size = min(batch_size, n_boot - start)
        sampled = rng.integers(
            0, len(unique_subjects), size=(size, len(unique_subjects)), dtype=np.int32,
        )
        patient_weights = np.stack([
            np.bincount(row, minlength=len(unique_subjects)) for row in sampled
        ]).astype(float, copy=False)
        row_weights = patient_weights[:, inverse]
        event_weight = row_weights @ y
        total_weight = row_weights.sum(axis=1)
        one_class_draws += int(((event_weight <= 0) | (event_weight >= total_weight)).sum())
        for model_id, scores in model_scores.items():
            auc_values, ap_values = _weighted_rank_metric_draws(y, scores, row_weights)
            draws[model_id]["auroc"][start:start + size] = auc_values
            draws[model_id]["average_precision"][start:start + size] = ap_values
        if progress is not None:
            progress(start + size, n_boot)

    model_intervals = {
        model_id: {metric: _interval(values) for metric, values in metrics.items()}
        for model_id, metrics in draws.items()
    }
    comparison_intervals: dict[str, dict[str, dict[str, Any]]] = {}
    fm_draws = draws["main20k_real"]
    for analysis_id in pairs:
        comparison_intervals[analysis_id] = {}
        for metric in ("auroc", "average_precision"):
            comparison_intervals[analysis_id][metric] = _interval(
                fm_draws[metric] - draws[analysis_id][metric]
            )
    return {
        "draws": int(n_boot), "seed": int(seed),
        "cluster_unit": "patient", "one_class_draws": int(one_class_draws),
        "models": model_intervals, "comparisons": comparison_intervals,
    }


def _point_rank_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["y"].to_numpy(dtype=int)
    p = frame["p"].to_numpy(dtype=float)
    if np.unique(y).size != 2:
        raise ValueError("AUROC/AP point estimates require both outcome classes")
    return {
        "auroc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
    }


def _validate_saved_points(path: Path, point_deltas: list[dict[str, Any]]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"completed NW/SIC point-delta reference is absent: {path}")
    saved = pd.read_csv(path)
    required = {"db", "comparison_id", "delta_auroc", "delta_average_precision"}
    if not required.issubset(saved.columns):
        raise ValueError("completed point-delta reference lacks AUROC/AP fields")
    saved_by_key = {
        (str(row.db), str(row.comparison_id)): row
        for row in saved.itertuples(index=False)
    }
    checked = 0
    for row in point_deltas:
        key = (str(row["db"]), str(row["comparison_id"]))
        if key not in saved_by_key:
            if row["db"] in TARGET_COHORTS[:2]:
                raise ValueError(f"completed point-delta reference is missing {key}")
            continue
        prior = saved_by_key[key]
        for field in ("delta_auroc", "delta_average_precision"):
            if not np.isclose(float(getattr(prior, field)), float(row[field]), rtol=0, atol=1e-12):
                raise ValueError(f"recomputed fixed-prediction point estimate differs from saved {key}/{field}")
        checked += 1
    return {
        "path": str(path), "sha256": sha256_file(path),
        "rows_verified": int(checked),
    }


def evaluate_paired_auc_ap_ci(
    *, out_dir: Path = DEFAULT_OUT, point_reference: Path = DEFAULT_POINT_REFERENCE,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    output_path = out_dir / "transport_paired_auroc_ap_patient_ci.csv"
    manifest_path = out_dir / "transport_paired_auroc_ap_ci_manifest.json"
    frames, gates, endpoints, provenance = _load_all_model_frames()
    expected_inputs = {
        "fm_prediction_manifest_sha256": provenance["fm_prediction_manifest_sha256"],
        "baseline_prediction_manifest_sha256": provenance["baseline_prediction_manifest_sha256"],
        "baseline_prediction_file_sha256": provenance["baseline_prediction_file_sha256"],
        "source_reproduction_manifest_sha256": provenance["source_reproduction_manifest_sha256"],
        "eligibility_manifest_sha256": provenance["eligibility_manifest_sha256"],
        "subject_identity_attestation_sha256": provenance["subject_identity_attestation_sha256"],
        "saved_point_reference_sha256": sha256_file(point_reference),
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") != "complete" or existing.get("inputs") != expected_inputs:
            raise ValueError("saved AUROC/AP supplement is stale or incomplete")
        if sha256_file(output_path) != existing.get("aggregate_table_sha256"):
            raise ValueError("saved AUROC/AP supplement table hash mismatch")
        return existing
    if output_path.exists():
        raise FileExistsError(f"refusing to replace an unmanifested AUROC/AP supplement: {output_path}")

    rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    cohort_summaries: dict[str, dict[str, Any]] = {}
    for db in TARGET_COHORTS:
        paired = {
            analysis_id: _paired_target_predictions(
                frames["main20k_real"][db], frames[analysis_id][db], db=db,
            )
            for analysis_id in ANALYSIS_IDS
        }
        fm_metrics = _point_rank_metrics(frames["main20k_real"][db])
        point_by_model = {"main20k_real": fm_metrics}
        for model_id in ANALYSIS_IDS:
            point_by_model[model_id] = _point_rank_metrics(frames[model_id][db])
        for model_id in ANALYSIS_IDS:
            point_rows.append({
                "db": db,
                "comparison_id": f"main20k_real_minus_{model_id}",
                "delta_auroc": point_by_model["main20k_real"]["auroc"]
                - point_by_model[model_id]["auroc"],
                "delta_average_precision": point_by_model["main20k_real"]["average_precision"]
                - point_by_model[model_id]["average_precision"],
            })

        seed = SEED + 500_000 + (TARGET_COHORTS.index(db) + 1) * 1000
        boot = _cluster_bootstrap_auc_ap(
            paired, seed=seed,
            progress=lambda complete, total, cohort=db: print(json.dumps({
                "stage": "paired_auroc_ap_bootstrap_progress", "db": cohort,
                "repeat_completed": complete, "repeats": total,
            }, sort_keys=True), flush=True),
        )
        y = frames["main20k_real"][db]["y"].to_numpy(dtype=int)
        n_stays = int(len(y))
        n_events = int(y.sum())
        n_patients = int(frames["main20k_real"][db]["subject"].nunique())
        for model_id in ANALYSIS_IDS:
            comparison = boot["comparisons"][model_id]
            fm_ci = boot["models"]["main20k_real"]
            baseline_ci = boot["models"][model_id]
            rows.append({
                "db": db,
                "comparison_id": f"main20k_real_minus_{model_id}",
                "fm_model_id": "main20k_real", "baseline_model_id": model_id,
                "n_stays": n_stays, "n_patients": n_patients,
                "events": n_events, "nonevents": n_stays - n_events,
                "fm_auroc": point_by_model["main20k_real"]["auroc"],
                "fm_auroc_ci_95_lo": fm_ci["auroc"]["ci_95_lo"],
                "fm_auroc_ci_95_hi": fm_ci["auroc"]["ci_95_hi"],
                "baseline_auroc": point_by_model[model_id]["auroc"],
                "baseline_auroc_ci_95_lo": baseline_ci["auroc"]["ci_95_lo"],
                "baseline_auroc_ci_95_hi": baseline_ci["auroc"]["ci_95_hi"],
                "delta_auroc": point_by_model["main20k_real"]["auroc"]
                - point_by_model[model_id]["auroc"],
                "delta_auroc_ci_95_lo": comparison["auroc"]["ci_95_lo"],
                "delta_auroc_ci_95_hi": comparison["auroc"]["ci_95_hi"],
                "fm_average_precision": point_by_model["main20k_real"]["average_precision"],
                "fm_average_precision_ci_95_lo": fm_ci["average_precision"]["ci_95_lo"],
                "fm_average_precision_ci_95_hi": fm_ci["average_precision"]["ci_95_hi"],
                "baseline_average_precision": point_by_model[model_id]["average_precision"],
                "baseline_average_precision_ci_95_lo": baseline_ci["average_precision"]["ci_95_lo"],
                "baseline_average_precision_ci_95_hi": baseline_ci["average_precision"]["ci_95_hi"],
                "delta_average_precision": point_by_model["main20k_real"]["average_precision"]
                - point_by_model[model_id]["average_precision"],
                "delta_average_precision_ci_95_lo": comparison["average_precision"]["ci_95_lo"],
                "delta_average_precision_ci_95_hi": comparison["average_precision"]["ci_95_hi"],
                "bootstrap_cluster_unit": "patient",
                "bootstrap_draws": boot["draws"],
                "bootstrap_valid_draws": comparison["auroc"]["valid_draws"],
                "bootstrap_one_class_draws": boot["one_class_draws"],
                "bootstrap_seed": boot["seed"],
                "ci_scope": "paired full-cohort stay-weighted ranking metrics; shared patient-cluster weights; fixed predictions",
            })
        cohort_summaries[db] = {
            "endpoint": endpoints[db], "n_stays": n_stays,
            "n_patients": n_patients, "events": n_events,
            "nonevents": n_stays - n_events,
            "bootstrap_seed": boot["seed"], "bootstrap_draws": boot["draws"],
            "bootstrap_one_class_draws": boot["one_class_draws"],
            "shared_population_identity_sha256": gates[db]["shared_population_identity_sha256"],
            "eligibility_manifest_sha256": gates[db]["eligibility_manifest_sha256"],
            "eligibility_table_sha256": gates[db]["eligibility_table_sha256"],
        }

    point_reference_provenance = _validate_saved_points(point_reference, point_rows)
    _write_rows_atomic(output_path, rows)
    manifest = {
        "status": "complete",
        "schema": "ehr_fm_recovered_transport_paired_auroc_ap_ci_v1",
        "analysis_scope": "fixed saved predictions; no model or calibration fits",
        "cohorts": list(TARGET_COHORTS),
        "model_ids": list(MODEL_IDS),
        "paired_comparators": list(ANALYSIS_IDS),
        "metrics": ["auroc", "average_precision"],
        "ci_method": "percentile bootstrap of patient clusters with replacement; all stays for a sampled patient receive that draw's multiplicity",
        "paired_weighting": "the same patient multiplicities are shared by the FM model and all four conventional baselines within each cohort draw",
        "n_bootstrap": N_BOOT, "cluster_unit": "patient",
        "all_conventional_comparators_reported": True,
        "performance_based_selection": False,
        "cohort_summaries": cohort_summaries,
        "inputs": expected_inputs,
        "input_provenance": provenance,
        "saved_point_reference": point_reference_provenance,
        "aggregate_table": str(output_path),
        "aggregate_table_sha256": sha256_file(output_path),
        "aggregate_rows": int(len(rows)),
        "patient_level_exported_to_local": False,
        "clinical_input_availability_verified": False,
        "clinical_promotion_allowed": False,
    }
    _write_json_atomic(manifest_path, manifest)
    print(json.dumps({
        "status": "paired_auroc_ap_ci_complete",
        "manifest": str(manifest_path),
        "aggregate_table": str(output_path), "rows": len(rows),
    }, sort_keys=True), flush=True)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--point-reference", type=Path, default=DEFAULT_POINT_REFERENCE)
    args = parser.parse_args(argv)
    result = evaluate_paired_auc_ap_ci(out_dir=args.out_dir, point_reference=args.point_reference)
    return 0 if result.get("status") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
