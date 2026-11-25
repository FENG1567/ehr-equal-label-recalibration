#!/usr/bin/env python3
"""Decode same-window numeric targets from frozen FM and direct token summaries.

This v5 sensitivity reuses the repaired v4 MIMIC/eICU rosters and source or
patient-grouped nested-CV procedures. The direct reference summarizes only the
ten decoded variables' retained V/B/H tokens (90 value-plus-pattern features).
Patient-level predictions remain in memory and aggregate results are exported.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_var] = "2"

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from threadpoolctl import threadpool_info, threadpool_limits

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV4 = ROOT / "revision_v4_20261122"
REV5 = ROOT / "revision_v5_20261123"
for _module_path in (SCRIPT_DIR, REV4 / "decoding_repaired", REV4 / "baselines"):
    if str(_module_path) not in sys.path:
        sys.path.insert(0, str(_module_path))

import numeric_decoding_repaired as decoder  # noqa: E402
from baseline_features import VALUE_PATTERN_NAMES, summarize_inputs  # noqa: E402
import v5_cache_validator as cache_validator  # noqa: E402

DBS = ("mimic", "eicu_semantic_v4_2")
SOURCE_DB = "mimic"
TARGET_DB = "eicu_semantic_v4_2"
ANALYSIS_ID = "main20k_real"
RUN_ID = os.environ.get("V5_CONTROL_RUN_ID", "")
if RUN_ID and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", RUN_ID):
    raise ValueError("V5_CONTROL_RUN_ID contains unsafe path characters")
OUT = (REV5 / "decoding" / f"direct_token_summary_{RUN_ID}" if RUN_ID else
       REV5 / "decoding" / "direct_token_summary_20261123")
MANIFEST_PATH = REV4 / "engine/code_snapshots/embedding_manifest_recovered_v4_2_merged41_b2eec1f7aca1.json"
EXPECTED_MANIFEST = decoder.EXPECTED_REPAIRED_EMBEDDING_MANIFEST_SHA256
ELIGIBILITY_DIR = decoder.DEFAULT_ELIGIBILITY
SOURCE_REFERENCE = decoder.SOURCE_REFERENCE_SUMMARY
TOKEN_META = ROOT / "data/processed/meta_mimic.json"
SEED = 20261123
N_BOOT = 1000
RESOURCE_SCOPE = os.environ.get("V5_RESOURCE_SCOPE", "unspecified")


def current_cgroup_path() -> str:
    for row in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
        if row.startswith("0::"):
            return row.split("::", 1)[1]
    raise RuntimeError("unified cgroup path is unavailable")


RESOURCE_LIMITS = {
    "runner_memory_max_bytes": int(os.environ.get('V5_HOST_MEMORY_MAX_BYTES', str(3*1024**3))),
    "runner_memory_swap_max_bytes": 0,
    "runner_cpu_quota_percent": 200,
    "cpu_threads": 2,
    "resource_scope": RESOURCE_SCOPE,
    "global_serial_lock": "R5/admin/v5_serial_science.lock",
}
VISIBILITY_GROUPS = ("all", "visible", "not_visible")
FEATURE_ID = "direct_token_summary_90"
# Reproduce the frozen v4.2 source embedding head with the BLAS runtime used to
# create its reference metrics. The process-level CPU quota and thread defaults
# remain two; this scoped limit applies only to the fixed-alpha source head.
SOURCE_FM_BLAS_THREADS = 8


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite numeric decoder evidence: {path}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite numeric decoder evidence: {path}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _load_population(db: str, entries: dict[str, dict[str, Any]]) -> tuple[Any, dict[str, Any], Path]:
    entry = entries[f"{db}/{ANALYSIS_ID}"]
    constant_key = f"{db}/{entry.get('model_tag', '')}_constant"
    if str(entry.get("input_condition", "")) == "real" and constant_key in entries:
        cache_attestation = cache_validator.validate_real_constant_pair(
            MANIFEST_PATH, entry, entries[constant_key], db,
        )
    else:
        cache_attestation = cache_validator.validate_cache_entry(
            MANIFEST_PATH, entry, db,
        )
    cache = decoder.load_embedding_cache(entry, db, ANALYSIS_ID)
    tabular_path, split_path, sequence_path = decoder._resolve_cohort_paths(db)
    sequence_hash = decoder.sha256_file(sequence_path)
    if sequence_hash.lower() != str(entry["source_npz_sha256"]).lower():
        raise ValueError(f"{db}: numeric sequence SHA256 differs from embedding manifest")
    sequence_stay, sequence_subject, sequence_y = decoder.load_sequence_identity(sequence_path)
    decoder.verify_cache_sequence_identity(cache, sequence_stay, sequence_subject)
    eligible, eligibility_entry = decoder._load_ready_eligibility(db, ELIGIBILITY_DIR, None)
    eligible, eligibility_provenance = decoder.verify_current_eligibility(
        db, ELIGIBILITY_DIR, eligible, eligibility_entry, sequence_hash,
        sequence_stay, sequence_subject, sequence_y,
        expected_manifest_sha256=decoder.EXPECTED_REPAIRED_ELIGIBILITY_MANIFEST_SHA256,
        expected_endpoint="hospital",
    )
    target_frame = decoder._read_target_frame(tabular_path)
    split = decoder._load_checked_split(split_path, str(entry["split_sha256"]), db)
    population = decoder.build_common_population(
        {ANALYSIS_ID: cache}, target_frame, split, eligible,
    )
    if len(population.stay_id) == 0:
        raise ValueError(f"{db}: repaired numeric population is empty")
    if not np.isfinite(population.embedding_by_analysis[ANALYSIS_ID]).all():
        raise ValueError(f"{db}: numeric embedding population contains nonfinite values")
    return population, {
        "sequence_path": str(sequence_path),
        "sequence_sha256": sequence_hash,
        "split_sha256": str(entry["split_sha256"]),
        "tabular_path": str(tabular_path),
        "tabular_sha256": decoder.sha256_file(tabular_path),
        "eligibility": eligibility_provenance,
        "cache_attestation": cache_attestation,
        "cache_sha256": cache.cache_sha256,
        "cache_rows": int(len(cache.rows)),
        "common_population_rows": int(len(population.stay_id)),
        "common_population_identity_sha256": decoder.sha256_ordered_identity(
            db, population.rows, population.stay_id, population.subject,
        ),
    }, sequence_path


def _token_summary_for_population(sequence_path: Path, rows: np.ndarray,
                                  vocabulary: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if len(vocabulary) != len(set(vocabulary)):
        raise ValueError("MIMIC token vocabulary contains duplicate variable names")
    missing = [name for name in decoder.VARS if vocabulary.count(name) != 1]
    if missing:
        raise ValueError(f"token vocabulary lacks unique numeric target variables: {missing}")
    n_vocab = len(vocabulary)
    source_token_ids = np.asarray([vocabulary.index(name) + 1 for name in decoder.VARS],
                                  dtype=np.int64)
    if len(set(source_token_ids.tolist())) != len(decoder.VARS):
        raise ValueError("numeric target variables do not map to distinct token IDs")

    with np.load(sequence_path, allow_pickle=False) as sequence:
        v = np.asarray(sequence["V"][rows], dtype=np.int64)
        b = np.asarray(sequence["B"][rows], dtype=np.int64)
        h = np.asarray(sequence["H"][rows], dtype=np.int64)
        lengths = np.asarray(sequence["L"][rows], dtype=np.int64)
    if (v.ndim != 2 or b.shape != v.shape or h.shape != v.shape
            or lengths.shape != (len(rows),)):
        raise ValueError("V/B/H/L numeric sequence arrays do not align")
    if np.any(v < 0) or np.any(v > n_vocab):
        raise ValueError("sequence token IDs exceed the frozen MIMIC variable vocabulary")
    if np.any(lengths < 0) or np.any(lengths > v.shape[1]):
        raise ValueError("sequence lengths exceed V/B/H width")

    token_map = np.zeros(n_vocab + 1, dtype=np.int16)
    token_map[source_token_ids] = np.arange(1, len(decoder.VARS) + 1, dtype=np.int16)
    selected_v = token_map[v]
    keep = selected_v > 0
    compact = {
        "V": selected_v,
        "B": np.where(keep, b, 0),
        "H": np.where(keep, h, 0),
        "L": lengths,
    }
    features, _pattern_only = summarize_inputs(
        compact, np.arange(len(rows), dtype=np.int64), len(decoder.VARS),
    )
    expected_width = len(decoder.VARS) * len(VALUE_PATTERN_NAMES)
    if features.shape != (len(rows), expected_width):
        raise AssertionError("direct token summary has an unexpected feature width")
    if not np.isfinite(features).all():
        raise FloatingPointError("direct token summary contains nonfinite values")
    visible = features[:, np.arange(len(decoder.VARS)) * len(VALUE_PATTERN_NAMES)] > 0
    return features, visible, {
        "n_features": int(features.shape[1]),
        "n_variables": int(len(decoder.VARS)),
        "variable_names": list(decoder.VARS),
        "feature_names_per_variable": list(VALUE_PATTERN_NAMES),
        "variable_token_ids_1_based": {
            name: int(token_id) for name, token_id in zip(decoder.VARS, source_token_ids)
        },
        "visible_definition": "target variable occurs at least once in retained main20k V/B/H tokens for that stay",
        "visible_is_measurement_retention_proof": False,
        "input_scope": "V/B/H token-summary for the same retained [0,24) sequence used by main20k_real",
        "target_values_used_to_construct_features": False,
    }


def _nested_oof_pair(X_fm: np.ndarray, X_direct: np.ndarray, y: np.ndarray,
                     groups: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups).astype(str)
    X_by_arm = {"main20k_real": np.asarray(X_fm, dtype=np.float32),
                FEATURE_ID: np.asarray(X_direct, dtype=np.float32)}
    if (len(y) != len(groups)
            or any(x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all()
                   for x in X_by_arm.values())):
        raise ValueError("nested decoder arrays are nonfinite or misaligned")
    outer_splits = decoder.make_group_splits(groups, decoder.OUTER_FOLDS)
    oof = {arm: np.full(len(y), np.nan, dtype=float) for arm in X_by_arm}
    alphas = {arm: [] for arm in X_by_arm}
    fold_provenance = []
    for fold_index, (train, test) in enumerate(outer_splits):
        inner_splits = decoder.make_group_splits(groups[train], decoder.INNER_FOLDS)
        if set(groups[train]).intersection(groups[test]):
            raise AssertionError("outer nested decoder folds split a patient")
        fold_provenance.append({
            "outer_fold": int(fold_index),
            "train_patient_count": int(len(np.unique(groups[train]))),
            "test_patient_count": int(len(np.unique(groups[test]))),
            "train_rows": int(len(train)), "test_rows": int(len(test)),
            "inner_fold_count": int(len(inner_splits)),
        })
        for arm, features in X_by_arm.items():
            search = GridSearchCV(
                decoder._ridge_pipeline(),
                param_grid={"ridge__alpha": list(decoder.ALPHAS)},
                scoring="neg_mean_squared_error", cv=inner_splits,
                refit=True, n_jobs=1,
            )
            search.fit(features[train], y[train])
            oof[arm][test] = search.predict(features[test])
            alphas[arm].append(float(search.best_params_["ridge__alpha"]))
    if any(not np.isfinite(prediction).all() for prediction in oof.values()):
        raise AssertionError("nested folds did not assign every row an out-of-fold prediction")
    return oof, {
        "outer_folds": int(len(outer_splits)),
        "inner_folds": int(decoder.INNER_FOLDS),
        "same_outer_and_inner_patient_group_splits_for_both_feature_arms": True,
        "alpha_by_outer_fold": {arm: values for arm, values in alphas.items()},
        "fold_population": fold_provenance,
    }


def _metric_values(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return decoder._metrics(y, prediction)


def _paired_bootstrap_differences(y: np.ndarray, prediction_fm: np.ndarray,
                                 prediction_direct: np.ndarray, groups: np.ndarray,
                                 *, draws: int, seed: int) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    fm = np.asarray(prediction_fm, dtype=float)
    direct = np.asarray(prediction_direct, dtype=float)
    groups = np.asarray(groups).astype(str)
    if (not len(y) or len(y) != len(fm) or len(y) != len(direct)
            or len(y) != len(groups) or not np.isfinite(y).all()
            or not np.isfinite(fm).all() or not np.isfinite(direct).all()):
        raise ValueError("paired numeric decoder bootstrap inputs are invalid")
    unique, inverse = np.unique(groups, return_inverse=True)
    n_patient = len(unique)
    counts = np.bincount(inverse, minlength=n_patient).astype(float)
    sums_y = np.bincount(inverse, weights=y, minlength=n_patient)
    sums_y2 = np.bincount(inverse, weights=y * y, minlength=n_patient)
    losses = {}
    for arm, prediction in (("fm", fm), ("direct", direct)):
        sq = np.square(y - prediction)
        ab = np.abs(y - prediction)
        losses[arm] = {
            "sq": np.bincount(inverse, weights=sq, minlength=n_patient),
            "ab": np.bincount(inverse, weights=ab, minlength=n_patient),
        }

    def point(weights: np.ndarray, arm: str) -> tuple[float, float, float]:
        n_rows = float(np.dot(weights, counts))
        if n_rows <= 0:
            return float("nan"), float("nan"), float("nan")
        sse = float(np.dot(weights, losses[arm]["sq"]))
        sae = float(np.dot(weights, losses[arm]["ab"]))
        sy = float(np.dot(weights, sums_y))
        sy2 = float(np.dot(weights, sums_y2))
        total_variance = sy2 - (sy * sy / n_rows)
        r2 = 1 - sse / total_variance if total_variance > 0 else float("nan")
        return r2, float(np.sqrt(sse / n_rows)), float(sae / n_rows)

    base_weights = np.ones(n_patient, dtype=float)
    direct_point = np.asarray(point(base_weights, "direct"), dtype=float)
    fm_point = np.asarray(point(base_weights, "fm"), dtype=float)
    estimate = direct_point - fm_point
    if n_patient < 2:
        return {
            "intervals": {
                metric: {
                    "difference_direct_minus_fm": float(estimate[index]),
                    "ci_95_low": float("nan"), "ci_95_high": float("nan"),
                    "bootstrap_valid": 0, "bootstrap_draws": int(draws),
                    "ci_status": "unavailable_fewer_than_two_patients",
                }
                for index, metric in enumerate(("r2", "rmse", "mae"))
            },
            "n_patients": int(n_patient), "n_stays": int(len(y)),
            "bootstrap_cluster_unit": "patient",
            "uncertainty_scope": "paired patient-cluster percentile bootstrap conditional on fitted source heads or shared target group folds",
        }
    rng = np.random.default_rng(seed)
    boot = np.full((draws, 3), np.nan, dtype=float)
    for replicate in range(draws):
        weights = rng.multinomial(n_patient, np.full(n_patient, 1 / n_patient))
        boot[replicate] = np.asarray(point(weights, "direct")) - np.asarray(point(weights, "fm"))
    intervals = {}
    for index, metric in enumerate(("r2", "rmse", "mae")):
        valid = boot[:, index][np.isfinite(boot[:, index])]
        low, high = (np.percentile(valid, [2.5, 97.5]) if len(valid)
                     else (float("nan"), float("nan")))
        intervals[metric] = {
            "difference_direct_minus_fm": float(estimate[index]),
            "ci_95_low": float(low), "ci_95_high": float(high),
            "bootstrap_valid": int(len(valid)),
            "bootstrap_draws": int(draws),
            "ci_status": (
                "available" if len(valid) else
                "unavailable_constant_target" if np.unique(y).size < 2 else
                "unavailable_no_valid_draws"
            ),
        }
    return {
        "intervals": intervals,
        "n_patients": int(n_patient), "n_stays": int(len(y)),
        "bootstrap_cluster_unit": "patient",
        "uncertainty_scope": "paired patient-cluster percentile bootstrap conditional on fitted source heads or shared target group folds",
    }


def _append_metrics_and_contrasts(
    metrics_rows: list[dict[str, Any]], contrasts_rows: list[dict[str, Any]],
    *, cohort: str, mode: str, variable: str, y: np.ndarray,
    predictions: dict[str, np.ndarray], groups: np.ndarray, visible: np.ndarray,
    seed: int, alpha_by_arm: dict[str, Any], fold_meta: dict[str, Any] | None,
) -> None:
    for visibility_index, visibility in enumerate(VISIBILITY_GROUPS):
        mask = (np.ones(len(y), dtype=bool) if visibility == "all" else
                np.asarray(visible, dtype=bool) if visibility == "visible" else
                ~np.asarray(visible, dtype=bool))
        if not mask.any():
            for arm in predictions:
                metrics_rows.append({
                    "cohort": cohort, "mode": mode, "variable": variable,
                    "feature_id": arm, "visibility": visibility,
                    "status": "no_rows_in_visibility_stratum",
                    "alpha": alpha_by_arm.get(arm),
                })
            continue
        y_subset = np.asarray(y, dtype=float)[mask]
        group_subset = np.asarray(groups).astype(str)[mask]
        target_constant = np.unique(y_subset).size < 2
        patient_count = int(len(np.unique(group_subset)))
        fm_prediction = np.asarray(predictions["main20k_real"], dtype=float)[mask]
        direct_prediction = np.asarray(predictions[FEATURE_ID], dtype=float)[mask]
        for arm, prediction in (("main20k_real", fm_prediction),
                                (FEATURE_ID, direct_prediction)):
            if len(y_subset) < 1 or len(prediction) != len(y_subset):
                raise ValueError("numeric decoder visibility metric arrays do not align")
            metric = _metric_values(y_subset, prediction)
            interval = decoder.patient_cluster_bootstrap_metrics(
                y_subset, prediction, group_subset, n_boot=N_BOOT,
                seed=seed + visibility_index,
            ) if patient_count >= 2 else {
                "r2_lo": float("nan"), "r2_hi": float("nan"),
                "rmse_lo": float("nan"), "rmse_hi": float("nan"),
                "mae_lo": float("nan"), "mae_hi": float("nan"),
                "bootstrap_valid": 0, "bootstrap_valid_r2": 0,
                "bootstrap_replicates": int(N_BOOT),
                "bootstrap_cluster_unit": "patient",
            }
            metrics_rows.append({
                "cohort": cohort, "mode": mode, "variable": variable,
                "feature_id": arm, "visibility": visibility, "status": "complete",
                "visibility_definition": "at least one retained token for the decoded target variable",
                "n_stays": int(len(y_subset)),
                "n_patients": int(len(np.unique(group_subset))),
                "target_visible_stays": int(np.sum(visible & mask)),
                "alpha": alpha_by_arm.get(arm),
                "alpha_by_outer_fold": (fold_meta or {}).get("alpha_by_outer_fold", {}).get(arm),
                **metric,
                **interval,
                "r2_status": "unavailable_constant_target" if target_constant else "available",
                "ci_status": ("available" if patient_count >= 2 else
                              "unavailable_fewer_than_two_patients"),
                "same_label_roster_and_folds": True,
            })
        paired = _paired_bootstrap_differences(
            y_subset, fm_prediction, direct_prediction, group_subset,
            draws=N_BOOT, seed=seed + 1000 + visibility_index,
        )
        for metric, result in paired["intervals"].items():
            contrasts_rows.append({
                "cohort": cohort, "mode": mode, "variable": variable,
                "candidate_feature": FEATURE_ID, "reference_feature": "main20k_real",
                "visibility": visibility, "metric": metric,
                **result,
                "n_stays": paired["n_stays"],
                "n_patients": paired["n_patients"],
                "bootstrap_cluster_unit": paired["bootstrap_cluster_unit"],
                "uncertainty_scope": paired["uncertainty_scope"],
                "r2_status": ("unavailable_constant_target" if target_constant else "available"),
                "direction": ("higher_is_better" if metric == "r2" else "lower_is_better"),
            })


def run() -> None:
    started = time.monotonic()
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite existing numeric decoder output: {OUT}")
    OUT.mkdir(parents=True)
    manifest, entries = decoder.load_embedding_manifest(MANIFEST_PATH)
    manifest_sha = decoder.sha256_file(MANIFEST_PATH)
    if manifest_sha.lower() != EXPECTED_MANIFEST.lower():
        raise ValueError("frozen v4.2 embedding manifest SHA256 differs")
    if len(manifest.get("models", [])) != decoder.EXPECTED_REPAIRED_EMBEDDING_MODEL_COUNT:
        raise ValueError("expected the frozen repaired v4.2 embedding manifest rows")
    source_reference_rows, source_reference_sha = decoder.load_frozen_source_reference(
        SOURCE_REFERENCE, [ANALYSIS_ID],
        expected_sha256=decoder.EXPECTED_SOURCE_REFERENCE_SHA256,
    )
    token_meta = json.loads(TOKEN_META.read_text(encoding="utf-8"))
    vocabulary = token_meta.get("vars") if isinstance(token_meta, dict) else None
    if not isinstance(vocabulary, list) or not vocabulary:
        raise ValueError("frozen MIMIC token metadata does not define a variable vocabulary")
    token_vocab_sha = decoder.sha256_file(TOKEN_META)

    populations: dict[str, Any] = {}
    provenance_by_db: dict[str, Any] = {}
    sequences: dict[str, Path] = {}
    token_features: dict[str, np.ndarray] = {}
    token_visible: dict[str, np.ndarray] = {}
    feature_provenance: dict[str, Any] = {}
    for db in DBS:
        print(f"NUMERIC_TOKEN_POPULATION_START db={db}", flush=True)
        population, provenance, sequence_path = _load_population(db, entries)
        populations[db] = population
        provenance_by_db[db] = provenance
        sequences[db] = sequence_path
        token_features[db], token_visible[db], feature_provenance[db] = (
            _token_summary_for_population(sequence_path, population.rows, vocabulary)
        )
        print(f"NUMERIC_TOKEN_POPULATION_READY db={db} rows={len(population.rows)}", flush=True)

    source_population = populations[SOURCE_DB]
    decoder.validate_source_patient_partitions(source_population, SOURCE_DB)
    source_reference_identity = decoder.validate_source_reference_identities(
        source_population, source_reference_rows, [ANALYSIS_ID],
    )
    source_metric_validation = []
    source_fm_blas_pools: list[dict[str, Any]] | None = None
    metrics_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    source_models: dict[str, dict[str, Any]] = {}
    source_alphas: dict[str, dict[str, float]] = {arm: {} for arm in ("main20k_real", FEATURE_ID)}

    for variable_index, variable in enumerate(decoder.VARS):
        source_y = source_population.targets_by_variable[variable]
        train = (source_population.split == 0) & np.isfinite(source_y)
        validation = (source_population.split == 1) & np.isfinite(source_y)
        source_test = (source_population.split == 2) & np.isfinite(source_y)
        if not train.any() or not validation.any() or not source_test.any():
            raise ValueError(f"MIMIC/{variable}: split0/1/2 numeric decoder roster is incomplete")
        reference = source_reference_rows[(ANALYSIS_ID, variable)]
        # Fit/scaling and both source and transfer predictions must share the
        # reference BLAS limit. The diagnostic showed that the frozen Hgb
        # metrics reproduce at 8 threads, while the default 2-thread result
        # misses the unchanged 1e-10 comparison threshold.
        with threadpool_limits(limits=SOURCE_FM_BLAS_THREADS):
            fm_model, fm_meta = decoder.fit_source_ridge_fixed_alpha(
                source_population.embedding_by_analysis[ANALYSIS_ID][train], source_y[train],
                source_population.subject[train], float(reference["alpha"]),
            )
            fm_source_prediction = fm_model.predict(
                source_population.embedding_by_analysis[ANALYSIS_ID][source_test]
            )
            if source_fm_blas_pools is None:
                source_fm_blas_pools = [
                    {key: pool.get(key) for key in ("user_api", "internal_api", "num_threads", "prefix")}
                    for pool in threadpool_info()
                ]
        direct_model, direct_meta = decoder.fit_source_ridge(
            token_features[SOURCE_DB][train], source_y[train],
            source_population.subject[train], token_features[SOURCE_DB][validation],
            source_y[validation], source_population.subject[validation],
        )
        direct_source_prediction = direct_model.predict(token_features[SOURCE_DB][source_test])
        source_metric_validation.append(decoder.compare_source_test_metrics(
            _metric_values(source_y[source_test], fm_source_prediction),
            reference, analysis_id=ANALYSIS_ID, variable=variable,
        ))
        source_alpha = {
            "main20k_real": float(fm_meta["alpha"]),
            FEATURE_ID: float(direct_meta["alpha"]),
        }
        source_alphas["main20k_real"][variable] = source_alpha["main20k_real"]
        source_alphas[FEATURE_ID][variable] = source_alpha[FEATURE_ID]
        source_visibility = token_visible[SOURCE_DB][source_test, variable_index]
        _append_metrics_and_contrasts(
            metrics_rows, contrast_rows, cohort=SOURCE_DB, mode="source_internal_test",
            variable=variable, y=source_y[source_test],
            predictions={"main20k_real": fm_source_prediction,
                         FEATURE_ID: direct_source_prediction},
            groups=source_population.subject[source_test], visible=source_visibility,
            seed=decoder.paired_bootstrap_seed(
                SOURCE_DB, variable, "source_internal_test", SEED,
            ),
            alpha_by_arm=source_alpha, fold_meta=None,
        )

        target = populations[TARGET_DB]
        target_y = target.targets_by_variable[variable]
        target_mask = np.isfinite(target_y)
        if not target_mask.any():
            raise ValueError(f"{TARGET_DB}/{variable}: no finite same-window targets")
        with threadpool_limits(limits=SOURCE_FM_BLAS_THREADS):
            target_fm_prediction = fm_model.predict(
                target.embedding_by_analysis[ANALYSIS_ID][target_mask]
            )
        target_direct_prediction = direct_model.predict(token_features[TARGET_DB][target_mask])
        _append_metrics_and_contrasts(
            metrics_rows, contrast_rows, cohort=TARGET_DB,
            mode="mimic_to_eicu_frozen_source_transfer", variable=variable,
            y=target_y[target_mask],
            predictions={"main20k_real": target_fm_prediction,
                         FEATURE_ID: target_direct_prediction},
            groups=target.subject[target_mask],
            visible=token_visible[TARGET_DB][target_mask, variable_index],
            seed=decoder.paired_bootstrap_seed(
                TARGET_DB, variable, "mimic_to_eicu_frozen_source_transfer", SEED,
            ),
            alpha_by_arm=source_alpha, fold_meta=None,
        )

        target_y = target.targets_by_variable[variable]
        target_mask = np.isfinite(target_y)
        if not target_mask.any():
            raise ValueError(f"{TARGET_DB}/{variable}: no finite same-window targets")
        target_groups = target.subject[target_mask]
        target_fm = target.embedding_by_analysis[ANALYSIS_ID][target_mask]
        target_direct = token_features[TARGET_DB][target_mask]
        target_visible = token_visible[TARGET_DB][target_mask, variable_index]
        target_oof, fold_meta = _nested_oof_pair(
            target_fm, target_direct, target_y[target_mask], target_groups,
        )
        _append_metrics_and_contrasts(
            metrics_rows, contrast_rows, cohort=TARGET_DB,
            mode="target_local_nested_cv", variable=variable,
            y=target_y[target_mask], predictions=target_oof,
            groups=target_groups, visible=target_visible,
            seed=decoder.paired_bootstrap_seed(
                TARGET_DB, variable, "target_local_nested_cv", SEED,
            ),
            alpha_by_arm={}, fold_meta=fold_meta,
        )
        source_models[variable] = {
            "main20k_real_source_alpha": float(fm_meta["alpha"]),
            "direct_token_summary_source_alpha": float(direct_meta["alpha"]),
            "direct_token_summary_source_validation_mse": float(direct_meta["alpha_validation_mse"]),
            "source_train_rows": int(train.sum()),
            "source_train_patients": int(len(np.unique(source_population.subject[train]))),
            "source_validation_rows": int(validation.sum()),
            "source_validation_patients": int(len(np.unique(source_population.subject[validation]))),
            "source_test_rows": int(source_test.sum()),
            "source_test_patients": int(len(np.unique(source_population.subject[source_test]))),
            "target_local_fold_provenance": fold_meta,
        }
        print(f"NUMERIC_TOKEN_VARIABLE_COMPLETE variable={variable}", flush=True)

    metrics_path = OUT / "numeric_token_decoder_metrics.csv"
    contrasts_path = OUT / "numeric_token_decoder_paired_contrasts.csv"
    _write_csv_atomic(pd.DataFrame(metrics_rows), metrics_path)
    _write_csv_atomic(pd.DataFrame(contrast_rows), contrasts_path)
    manifest_out = {
        "status": "complete",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schema": "ehr_fm_v5_direct_token_numeric_decoder_1",
        "run_id": RUN_ID or "standalone_20261123",
        "execution_resource_limits": (RESOURCE_LIMITS if RUN_ID else
                                       {"status": "standalone_limits_not_attested"}),
        "resource_scope": RESOURCE_SCOPE,
        "cgroup_path": current_cgroup_path(),
        "source_db": SOURCE_DB, "target_db": TARGET_DB,
        "analysis_id": ANALYSIS_ID,
        "embedding_manifest_path": str(MANIFEST_PATH),
        "embedding_manifest_sha256": manifest_sha,
        "source_reference_summary_path": str(SOURCE_REFERENCE),
        "source_reference_summary_sha256": source_reference_sha,
        "source_reference_identity_validation": source_reference_identity,
        "token_vocabulary_path": str(TOKEN_META),
        "token_vocabulary_sha256": token_vocab_sha,
        "token_feature_definition": feature_provenance,
        "target_variables": list(decoder.VARS),
        "source_alpha_grid": list(decoder.ALPHAS),
        "source_internal_test": "MIMIC split2 is the source-internal test partition; source heads are selected/fitted using MIMIC split0/1 only",
        "source_fit": "main20k_real reuses frozen MIMIC split1 alpha and refits on split0; direct token summary selects alpha on MIMIC split1 and refits scaler/Ridge on split0",
        "mimic_to_eicu_frozen_source_transfer": "MIMIC split0/1-fitted heads, with alpha fixed before target scoring, are scored on the shared finite-label eICU roster; no eICU fitting or selection is used",
        "target_local_fit": "5 outer / 3 inner patient-group Ridge CV; the same outer and inner group folds are used for main20k_real and direct token summary",
        "visibility_strata": {
            "visible": "decoded target variable occurs at least once in retained V/B/H tokens for that stay",
            "not_visible": "decoded target variable does not occur in retained V/B/H tokens for that stay",
            "fit_scope": "all finite target rows; visibility is applied after source or shared-fold OOF predictions are generated",
            "measurement_retention_proof": False,
        },
        "uncertainty": "1000 patient-cluster draws; paired direct-minus-FM RMSE, MAE, and R2 contrasts use shared draws within each roster and visibility stratum",
        "source_roster_provenance": provenance_by_db[SOURCE_DB],
        "target_roster_provenance": provenance_by_db[TARGET_DB],
        "cache_attestation_by_db": {
            db: provenance_by_db[db]["cache_attestation"] for db in DBS
        },
        "source_test_metric_reproduction": source_metric_validation,
        "source_fm_numeric_runtime": {
            "blas_threads_requested": SOURCE_FM_BLAS_THREADS,
            "blas_pools_observed_during_fit_and_source_prediction": source_fm_blas_pools,
            "scope": "main20k_real fixed-alpha MIMIC source Ridge fit (including scaler fit), MIMIC split-2 prediction, and eICU transfer prediction (including scaler transforms)",
            "process_thread_defaults_preserved": {name: os.environ.get(name) for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )},
            "cpu_quota_percent": RESOURCE_LIMITS["runner_cpu_quota_percent"],
            "global_cpu_threads": RESOURCE_LIMITS["cpu_threads"],
        },
        "source_fit_by_variable": source_models,
        "source_alphas_by_variable": source_alphas,
        "metrics_csv": str(metrics_path),
        "metrics_csv_sha256": decoder.sha256_file(metrics_path),
        "paired_contrasts_csv": str(contrasts_path),
        "paired_contrasts_csv_sha256": decoder.sha256_file(contrasts_path),
        "patient_level_predictions_exported": False,
        "runtime_seconds": round(time.monotonic() - started, 3),
    }
    _write_json_atomic(OUT / "numeric_token_decoder_manifest.json", manifest_out)
    print(f"NUMERIC_TOKEN_DECODER_COMPLETE output={OUT}", flush=True)


if __name__ == "__main__":
    run()
