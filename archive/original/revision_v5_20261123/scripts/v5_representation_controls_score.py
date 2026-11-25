#!/usr/bin/env python3
"""Score repaired-eICU fixed-head representation controls on the matched roster.

The GPU producer writes only embeddings. This CPU stage loads the historical
MIMIC C=1 source head, verifies every control against the repaired-eICU real
cache and reviewed target roster, and exports aggregate metrics only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import v5_representation_controls_gpu as gpu  # noqa: E402
import v5_representation_controls_readout_r4 as readout  # noqa: E402

ROOT = gpu.ROOT
R4 = gpu.R4
R5 = gpu.R5
DB = gpu.DB
OUT = (gpu.OUT / "scores" if gpu.RUN_ID else
       R5 / "controls" / "repaired_eicu_fixed_head_scores_20261123")
CONTROL_EMBEDDINGS = gpu.EMBEDDINGS
CONTROL_GPU_MANIFEST = gpu.GPU_MANIFEST
PARITY_REPORT = gpu.PARITY_REPORT


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite score evidence: {path}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite score evidence: {path}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _identity_hash(rows: np.ndarray, stays: np.ndarray, subjects: np.ndarray) -> str:
    return gpu.identity_hash(rows, stays, subjects)


def _point_metrics(y: np.ndarray, p: np.ndarray, subjects: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=float)
    subjects = np.asarray(subjects).astype(str)
    if len(y) == 0 or len(y) != len(p) or len(y) != len(subjects):
        raise ValueError("scoring arrays must have matching nonempty dimensions")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("risk probabilities must be finite and within [0, 1]")
    if len(np.unique(y)) != 2:
        raise ValueError("fixed-head scoring requires both outcome classes on the locked roster")
    p_loss = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "n_stays": int(len(y)), "n_patients": int(len(np.unique(subjects))),
        "events": int(y.sum()), "prevalence": float(y.mean()),
        "mean_predicted": float(p.mean()),
        "mean_predicted_minus_observed": float(np.mean(p - y)),
        "auroc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p_loss)),
        "logloss": float(log_loss(y, p_loss, labels=[0, 1])),
    }


def _weighted_rank_metric_draws(
    labels: np.ndarray, scores: np.ndarray, row_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match v4's weighted tie-aware AUROC/AP patient-bootstrap calculation."""
    y = np.asarray(labels, dtype=np.int8).reshape(-1)
    p = np.asarray(scores, dtype=float).reshape(-1)
    weights = np.asarray(row_weights, dtype=float)
    if (weights.ndim != 2 or weights.shape[1] != len(y) or len(y) != len(p)
            or not len(y) or not np.isin(y, [0, 1]).all()
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


def _paired_rank_metric_bootstrap(
    y: np.ndarray, subjects: np.ndarray, scores_by_arm: dict[str, np.ndarray],
    *, seed: int, draws: int = 1000, batch_size: int = 16,
) -> tuple[dict[str, dict[str, np.ndarray]], int]:
    y = np.asarray(y, dtype=np.int8)
    subjects = np.asarray(subjects).astype(str)
    if len(y) != len(subjects) or not len(y) or len(np.unique(y)) != 2:
        raise ValueError("paired AUROC/AP intervals require an aligned two-class roster")
    if any(len(np.asarray(p)) != len(y) for p in scores_by_arm.values()):
        raise ValueError("paired AUROC/AP score vectors do not share the roster")
    unique, inverse = np.unique(subjects, return_inverse=True)
    if len(unique) < 2:
        return ({arm: {metric: np.full(draws, np.nan) for metric in ("auroc", "average_precision")}
                 for arm in scores_by_arm}, draws)
    result = {arm: {metric: np.full(draws, np.nan, dtype=float)
                    for metric in ("auroc", "average_precision")}
              for arm in scores_by_arm}
    rng = np.random.default_rng(seed)
    one_class_draws = 0
    for start in range(0, draws, batch_size):
        size = min(batch_size, draws - start)
        sampled = rng.integers(0, len(unique), size=(size, len(unique)), dtype=np.int32)
        patient_weights = np.stack([
            np.bincount(row, minlength=len(unique)) for row in sampled
        ]).astype(float, copy=False)
        row_weights = patient_weights[:, inverse]
        event_weight = row_weights @ y
        total_weight = row_weights.sum(axis=1)
        one_class_draws += int(((event_weight <= 0) | (event_weight >= total_weight)).sum())
        for arm, probabilities in scores_by_arm.items():
            auc, ap = _weighted_rank_metric_draws(y, probabilities, row_weights)
            result[arm]["auroc"][start:start + size] = auc
            result[arm]["average_precision"][start:start + size] = ap
    return result, one_class_draws


def _fraction_from_counts(counts: np.ndarray, active: np.ndarray,
                          positions: np.ndarray) -> tuple[int, int, float | None]:
    numerator = int(np.asarray(counts, dtype=np.int64)[positions].sum())
    denominator = int(np.asarray(active, dtype=np.int64)[positions].sum())
    return numerator, denominator, (float(numerator / denominator) if denominator else None)


def _predict_in_batches(scaler, probe, embedding: np.ndarray,
                        positions: np.ndarray, *, batch_size: int = 8192) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.int64)
    if (embedding.ndim != 2 or not len(positions)
            or positions.min() < 0 or positions.max() >= len(embedding)):
        raise ValueError("control embedding positions are empty or outside the cache")
    if embedding.shape[1] != int(getattr(scaler, "n_features_in_", -1)):
        raise ValueError("control embedding width differs from the frozen source scaler")
    probability = np.empty(len(positions), dtype=np.float64)
    for start in range(0, len(positions), batch_size):
        stop = min(start + batch_size, len(positions))
        features = np.asarray(embedding[positions[start:stop]], dtype=np.float32)
        if not np.isfinite(features).all():
            raise ValueError("control embeddings contain nonfinite values")
        probability[start:stop] = probe.predict_proba(scaler.transform(features))[:, 1]
    if not np.isfinite(probability).all():
        raise FloatingPointError("fixed source head produced nonfinite probabilities")
    return probability


def run() -> None:
    started = time.monotonic()
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite existing score output: {OUT}")
    if not CONTROL_EMBEDDINGS.is_file() or not CONTROL_GPU_MANIFEST.is_file():
        raise FileNotFoundError("GPU control embeddings and their manifest must be complete first")
    OUT.mkdir(parents=True)

    gpu_manifest = json.loads(CONTROL_GPU_MANIFEST.read_text(encoding="utf-8"))
    if gpu_manifest.get("status") != "complete":
        raise ValueError("GPU control manifest is not complete")
    if gpu_manifest.get("readout_fit_or_selection_performed") is not False:
        raise ValueError("GPU stage unexpectedly fit or selected a readout")
    if gpu_manifest.get("outcome_labels_read_by_gpu_producer") is not False:
        raise ValueError("GPU stage unexpectedly read target labels")
    if gpu_manifest.get("frozen_pretrained_parameters_verified_unchanged") is not True:
        raise ValueError("GPU stage did not verify frozen encoder parameters")
    if readout.sha256_file(CONTROL_EMBEDDINGS) != gpu_manifest.get("intermediate_sha256"):
        raise ValueError("GPU control embedding archive hash differs from its manifest")
    parity = json.loads(PARITY_REPORT.read_text(encoding="utf-8"))
    if parity.get("status") != "passed" or parity.get("full_control_inference_started") is not False:
        raise ValueError("observed runtime parity must pass before control scores are accepted")
    if int(parity.get("sample_rows", -1)) != int(gpu.OBSERVED_SAMPLE_SIZE):
        raise ValueError("observed runtime parity did not use the locked 128-row sample")
    parity_sha256 = readout.sha256_file(PARITY_REPORT)
    if gpu_manifest.get("observed_runtime_parity_report_sha256") != parity_sha256:
        raise ValueError("GPU manifest does not bind the exact observed-runtime parity report")

    decoder, transport, _recalibration, _source_transport = readout.load_modules()
    manifest, entries = decoder.load_embedding_manifest(gpu.MANIFEST)
    if readout.sha256_file(gpu.MANIFEST) != gpu.EXPECTED_MANIFEST:
        raise ValueError("frozen v4.2 embedding manifest hash differs")
    if len(manifest.get("models", [])) != 41:
        raise ValueError("expected the frozen 41-row v4.2 embedding manifest")
    real_entry = entries[f"{DB}/main20k_real"]
    constant_entry = entries[f"{DB}/main20k_constant"]
    cache_pair_attestation = gpu.cache_validator.validate_real_constant_pair(
        gpu.MANIFEST, real_entry, constant_entry, DB,
    )
    if (gpu_manifest.get("cache_pair_attestation") != cache_pair_attestation
            or parity.get("cache_pair_attestation") != cache_pair_attestation):
        raise ValueError("GPU manifest/parity report do not bind the verified real/constant cache pair")
    real_cache = decoder.load_embedding_cache(real_entry, DB, "main20k_real")
    constant_cache = decoder.load_embedding_cache(constant_entry, DB, "main20k_constant")
    sequence_stay, sequence_subject, sequence_y = decoder.load_sequence_identity(gpu.SEQ)
    decoder.verify_cache_sequence_identity(real_cache, sequence_stay, sequence_subject)
    decoder.verify_cache_sequence_identity(constant_cache, sequence_stay, sequence_subject)
    constant_provenance = gpu.verify_constant_cache(
        real_entry, constant_entry, real_cache.rows, real_cache.stay_id,
        real_cache.subject, cache_pair_attestation,
    )
    if (not np.array_equal(real_cache.rows, constant_cache.rows)
            or not np.array_equal(real_cache.stay_id.astype(str), constant_cache.stay_id.astype(str))
            or not np.array_equal(real_cache.subject.astype(str), constant_cache.subject.astype(str))):
        raise ValueError("observed and constant cache identities differ")

    gpu_archive = np.load(CONTROL_EMBEDDINGS, allow_pickle=False)
    try:
        control_rows = np.asarray(gpu_archive["rows"], dtype=np.int64)
        control_stays = np.asarray(gpu_archive["stay_id"]).astype(str)
        control_subjects = np.asarray(gpu_archive["subject"]).astype(str)
        control_identity = _identity_hash(control_rows, control_stays, control_subjects)
        if control_identity != gpu_manifest.get("row_identity_sha256"):
            raise ValueError("GPU embedding row identities differ from the GPU manifest")
        if (not np.array_equal(control_rows, real_cache.rows)
                or not np.array_equal(control_stays, real_cache.stay_id.astype(str))
                or not np.array_equal(control_subjects, real_cache.subject.astype(str))):
            raise ValueError("GPU control embeddings do not match the observed real-cache roster")

        active_counts = np.asarray(gpu_archive["active_token_counts"], dtype=np.int64)
        constant_changed_counts = np.asarray(
            gpu_archive["constant_changed_token_counts"], dtype=np.int64)
        within_changed_counts = np.asarray(
            gpu_archive["within_changed_token_counts"], dtype=np.int64)
        donor_changed_counts = np.asarray(
            gpu_archive["donor_changed_token_counts"], dtype=np.int64)
        for name, counts in (("constant", constant_changed_counts),
                             ("within", within_changed_counts),
                             ("donor", donor_changed_counts)):
            if (counts.shape != active_counts.shape or np.any(counts < 0)
                    or np.any(counts > active_counts)):
                raise ValueError(f"GPU private {name} changed-token counts are invalid")

        frames, _gates, _endpoints, transport_provenance = transport._load_all_model_frames()
        if DB not in frames.get("main20k_real", {}):
            raise ValueError("strict frozen-score loader lacks repaired-eICU main20k_real")
        reference = frames["main20k_real"][DB].copy().reset_index(drop=True)
        if reference.empty or reference["stay_id"].astype(str).duplicated().any():
            raise ValueError("reviewed eICU reference roster is empty or contains duplicate stays")
        cache_positions = {str(stay): index for index, stay in enumerate(real_cache.stay_id)}
        positions = np.asarray([cache_positions.get(stay, -1)
                                for stay in reference["stay_id"].astype(str)], dtype=np.int64)
        if np.any(positions < 0):
            raise ValueError("reviewed eICU score roster has stays absent from control embeddings")
        if (not np.array_equal(real_cache.rows[positions], reference["rows"].to_numpy(dtype=np.int64))
                or not np.array_equal(real_cache.subject[positions].astype(str),
                                      reference["subject"].astype(str).to_numpy())):
            raise ValueError("reviewed target roster row/patient identities differ from control cache")
        target_rows = reference["rows"].to_numpy(dtype=np.int64)
        if (target_rows.min() < 0 or target_rows.max() >= len(sequence_y)
                or not np.array_equal(reference["y"].to_numpy(dtype=np.int8),
                                      sequence_y[target_rows].astype(np.int8))):
            raise ValueError("reviewed target labels differ from current sequence labels")
        if (not np.isin(reference["y"].to_numpy(dtype=np.int8), [0, 1]).all()
                or not np.isfinite(real_cache.embedding[positions]).all()
                or not np.isfinite(constant_cache.embedding[positions]).all()):
            raise ValueError("reviewed target labels or cached embeddings are invalid")

        parity_tolerance = parity.get("probability_tolerance", {})
        parity_atol = float(parity_tolerance.get("atol", float("nan")))
        parity_rtol = float(parity_tolerance.get("rtol", float("nan")))
        if not (np.isfinite(parity_atol) and np.isfinite(parity_rtol)
                and parity_atol >= 0 and parity_rtol >= 0):
            raise ValueError("observed-runtime parity report has no valid probability tolerance")

        scaler, probe = gpu.load_historical_source_probe()
        arms: list[tuple[str, np.ndarray, str]] = [
            ("observed_real_cache", real_cache.embedding, "historical main20k C=1 source head"),
            ("constant_main20k_cache", constant_cache.embedding,
             "historical main20k C=1 source head"),
        ]
        for arm_name in ("within_embedding", "donor_embedding"):
            embedding = np.asarray(gpu_archive[arm_name], dtype=np.float32)
            if embedding.shape != real_cache.embedding.shape:
                raise ValueError(f"{arm_name}: embedding dimensions differ from the real cache")
            if not np.isfinite(embedding).all():
                raise ValueError(f"{arm_name}: control embeddings contain nonfinite values")
            score_name = ("within_variable_shuffle" if arm_name == "within_embedding"
                          else "distinct_patient_donor")
            arms.append((score_name, embedding, "historical main20k C=1 source head"))

        y = reference["y"].to_numpy(dtype=np.int8)
        subject = reference["subject"].astype(str).to_numpy()
        metrics_rows: list[dict[str, Any]] = []
        probabilities_by_arm: dict[str, np.ndarray] = {}
        change_counts = {
            "observed_real_cache": np.zeros_like(active_counts),
            "constant_main20k_cache": constant_changed_counts,
            "within_variable_shuffle": within_changed_counts,
            "distinct_patient_donor": donor_changed_counts,
        }
        for arm_name, embedding, head in arms:
            probability = _predict_in_batches(scaler, probe, embedding, positions)
            probabilities_by_arm[arm_name] = probability
            changed_n, active_n, changed_fraction = _fraction_from_counts(
                change_counts[arm_name], active_counts, positions,
            )
            metrics_rows.append({
                "cohort": DB, "scope": "reviewed_external_roster",
                "arm_id": arm_name, "fixed_head": head,
                **_point_metrics(y, probability, subject),
                "changed_token_fraction": changed_fraction,
                "changed_token_count": changed_n,
                "active_token_count": active_n,
                "changed_token_fraction_scope": "reviewed eICU roster; all retained nonpadding tokens",
                "input_condition": (
                    "real" if arm_name == "observed_real_cache" else
                    "constant" if arm_name == "constant_main20k_cache" else arm_name
                ),
            })

        reference_probability = reference["p"].to_numpy(dtype=float)
        observed_probability = probabilities_by_arm["observed_real_cache"]
        full_roster_probability_max_abs = float(
            np.max(np.abs(observed_probability - reference_probability)))
        if not np.allclose(observed_probability, reference_probability,
                           atol=parity_atol, rtol=parity_rtol):
            raise ValueError(
                "recomputed observed-cache probabilities differ from frozen v4 scores over "
                f"the full reviewed roster: max_abs_difference={full_roster_probability_max_abs:.10g}, "
                f"atol={parity_atol:.10g}, rtol={parity_rtol:.10g}"
            )

        contrast_rows: list[dict[str, Any]] = []
        for arm_name in ("constant_main20k_cache", "within_variable_shuffle",
                         "distinct_patient_donor"):
            p_new = np.clip(probabilities_by_arm[arm_name], 1e-6, 1 - 1e-6)
            p_ref = np.clip(probabilities_by_arm["observed_real_cache"], 1e-6, 1 - 1e-6)
            losses_new = np.column_stack([
                np.square(y - p_new),
                -(y * np.log(p_new) + (1 - y) * np.log1p(-p_new)),
            ])
            losses_ref = np.column_stack([
                np.square(y - p_ref),
                -(y * np.log(p_ref) + (1 - y) * np.log1p(-p_ref)),
            ])
            for patient_weighted in (False, True):
                result = __import__("v5_curve_tools").paired_cluster_mean_ci(
                    subject, losses_new - losses_ref, draws=1000,
                    seed=20261123 + 1000 * (1 + ("constant_main20k_cache",
                                                 "within_variable_shuffle",
                                                 "distinct_patient_donor").index(arm_name)),
                    patient_weighted=patient_weighted,
                )
                for metric_index, metric_name in enumerate(("brier", "logloss")):
                    contrast_rows.append({
                        "cohort": DB, "candidate_arm": arm_name,
                        "reference_arm": "observed_real_cache", "metric": metric_name,
                        "difference_candidate_minus_reference": float(result["point"][metric_index]),
                        "ci_95_low": float(result["low"][metric_index]),
                        "ci_95_high": float(result["high"][metric_index]),
                        "weighting": result["weighting"],
                        "n_stays": int(result["n_stays"]),
                        "n_patients": int(result["n_patients"]),
                        "bootstrap_draws": int(result["bootstrap_draws"]),
                        "ci_scope": result["scope"],
                        "negative_loss_difference_favors_candidate_arm": True,
                    })

        rank_scores = {name: np.asarray(probability, dtype=float)
                       for name, probability in probabilities_by_arm.items()}
        rank_draws, rank_one_class_draws = _paired_rank_metric_bootstrap(
            y, subject, rank_scores, seed=20261123 + 7000,
        )
        for arm_index, arm_name in enumerate(("constant_main20k_cache",
                                              "within_variable_shuffle",
                                              "distinct_patient_donor")):
            for metric in ("auroc", "average_precision"):
                candidate_draws = rank_draws[arm_name][metric]
                reference_draws = rank_draws["observed_real_cache"][metric]
                delta = candidate_draws - reference_draws
                valid = delta[np.isfinite(delta)]
                low, high = (np.percentile(valid, [2.5, 97.5]) if len(valid)
                             else (float("nan"), float("nan")))
                point_candidate = (roc_auc_score(y, probabilities_by_arm[arm_name])
                                   if metric == "auroc" else
                                   average_precision_score(y, probabilities_by_arm[arm_name]))
                point_reference = (roc_auc_score(y, probabilities_by_arm["observed_real_cache"])
                                   if metric == "auroc" else
                                   average_precision_score(y, probabilities_by_arm["observed_real_cache"]))
                contrast_rows.append({
                    "cohort": DB, "candidate_arm": arm_name,
                    "reference_arm": "observed_real_cache", "metric": metric,
                    "difference_candidate_minus_reference": float(point_candidate - point_reference),
                    "ci_95_low": float(low), "ci_95_high": float(high),
                    "n_stays": int(len(y)), "n_patients": int(len(np.unique(subject))),
                    "bootstrap_draws": 1000, "bootstrap_valid_draws": int(len(valid)),
                    "bootstrap_one_class_draws": int(rank_one_class_draws),
                    "bootstrap_cluster_unit": "patient",
                    "bootstrap_seed": 20261123 + 7000,
                    "ci_scope": "paired full-roster ranking metrics; shared patient-cluster multiplicities; fixed predictions",
                    "positive_difference_favors_candidate_arm": True,
                })

        metrics_path = OUT / "fixed_head_control_metrics.csv"
        contrasts_path = OUT / "fixed_head_control_paired_contrasts.csv"
        _write_csv_atomic(pd.DataFrame(metrics_rows), metrics_path)
        _write_csv_atomic(pd.DataFrame(contrast_rows), contrasts_path)
        manifest_payload = {
            "status": "complete",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_id": gpu.RUN_ID or "standalone_20261123",
            "execution_resource_limits": (gpu.RESOURCE_LIMITS if gpu.RUN_ID else
                                           {"status": "standalone_limits_not_attested"}),
            "cohort": DB,
            "scope": "repaired eICU reviewed main20k_real roster",
            "fixed_head": "historical frozen MIMIC main20k C=1 source scaler and logistic probe",
            "historical_source_probe_path": str(gpu.PROBE),
            "historical_source_probe_sha256": readout.sha256_file(gpu.PROBE),
            "reviewed_roster_identity_sha256": _identity_hash(
                reference["rows"].to_numpy(dtype=np.int64),
                reference["stay_id"].astype(str).to_numpy(), subject,
            ),
            "n_stays": int(len(reference)),
            "n_patients": int(len(np.unique(subject))),
            "embedding_manifest_sha256": readout.sha256_file(gpu.MANIFEST),
            "real_cache_sha256": real_cache.cache_sha256,
            "constant_cache_sha256": constant_cache.cache_sha256,
            "constant_cache_provenance": constant_provenance,
            "cache_pair_attestation": cache_pair_attestation,
            "observed_runtime_probability_parity": {
                "reference_probability_source": "frozen v4 main20k_real scores on the full reviewed roster",
                "max_abs_difference": full_roster_probability_max_abs,
                "atol": parity_atol, "rtol": parity_rtol,
                "within_tolerance": True,
                "roster_identity_sha256": _identity_hash(
                    reference["rows"].to_numpy(dtype=np.int64),
                    reference["stay_id"].astype(str).to_numpy(), subject,
                ),
            },
            "gpu_embedding_archive_sha256": readout.sha256_file(CONTROL_EMBEDDINGS),
            "gpu_manifest_sha256": readout.sha256_file(CONTROL_GPU_MANIFEST),
            "observed_runtime_parity_report_sha256": parity_sha256,
            "transformations": gpu_manifest.get("transformations"),
            "transport_provenance": transport_provenance,
            "metrics_csv": str(metrics_path),
            "metrics_csv_sha256": readout.sha256_file(metrics_path),
            "paired_contrasts_csv": str(contrasts_path),
            "paired_contrasts_csv_sha256": readout.sha256_file(contrasts_path),
            "patient_level_predictions_exported": False,
            "runtime_seconds": round(time.monotonic() - started, 3),
        }
        _write_json_atomic(OUT / "fixed_head_control_score_manifest.json", manifest_payload)
        print(f"FIXED_HEAD_CONTROL_SCORE_COMPLETE n={len(reference)} output={OUT}", flush=True)
    finally:
        gpu_archive.close()


if __name__ == "__main__":
    run()
