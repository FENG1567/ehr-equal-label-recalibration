#!/usr/bin/env python3
"""Source-only main20k C selection, paired target losses, and shared 500-label Platt."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
R4 = ROOT / "revision_v4_20261122"
R5 = ROOT / "revision_v5_20261123"
RUN_ID = os.environ.get("V5_CONTROL_RUN_ID", "")
if RUN_ID and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", RUN_ID):
    raise ValueError("V5_CONTROL_RUN_ID contains unsafe path characters")
OUT = (R5 / "readout" / f"source_c_selection_{RUN_ID}" if RUN_ID else
       R5 / "readout" / "source_c_selection_20261123_r4")
ELIG = R4 / "temporal/results/eligibility/recovered_reviewed_v4_2"
EMBED_MANIFEST = R4 / "engine/code_snapshots/embedding_manifest_recovered_v4_2_merged41_b2eec1f7aca1.json"
EXPECTED_EMBED_MANIFEST = "b2eec1f7aca1857bda440ddb5ec48e477b2a20f753d5dea28b7715f7a1a243fe"
PROBE = R4 / "engine/results/source_probes/main20k_real_39321cc8e6e2ad96c18d.pkl"
EXPECTED_PROBE = "25b6c9015cf73d5e0156b76197cf1101d6f6e4649407b3ba15b48b2f90a00707"
SOURCE_PREDICTION_RUN = R4 / "engine/results/prediction_runs/status_consistent_primary_core_fix2-7da980ffcbd8f00b"
R4_TRANSPORT = R4 / "robustness/results/recovered_transport_v4_2_20261123_reuse_r2"
R4_PLAN = R4_TRANSPORT / "sampling_plan_manifest.json"
DIRECT_CALIBRATION_ROOT = (
    R4 / "calibrated_model_pairs/results/direct_calibrated_fm_baseline_v1/cohorts"
)
DBS = ("mimic", "eicu_semantic_v4_2", "nwicu_discharge_v4_1")
TARGET_DBS = ("eicu_semantic_v4_2", "nwicu_discharge_v4_1")
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
SEED = 20261123
SOURCE_MODEL_SEED = 20261122
C1_REPRODUCTION_TOLERANCE = 2e-6
NUMERICAL_READOUT_THREADS = 8
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
BOOTSTRAPS = 1000
BUDGET = 500
REPEATS = 200
ANALYSIS_IDS = ("main20k_real", "conventional_lr_value_plus_pattern",
                "conventional_hgb_value_plus_pattern")
FRAME_IDS = ("main20k_real", "conventional_lr_value_plus_pattern",
             "conventional_hgb_value_plus_pattern")


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    if temporary.exists() or path.exists():
        raise FileExistsError(f"refusing to overwrite JSON evidence: {path}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    if temporary.exists() or path.exists():
        raise FileExistsError(f"refusing to overwrite CSV evidence: {path}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def load_modules():
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    sys.path.insert(0, str(R4 / "decoding_repaired"))
    sys.path.insert(0, str(R4 / "engine/code"))
    sys.path.insert(0, str(R4 / "robustness/transport_v4_2"))
    sys.path.insert(0, str(R4 / "robustness"))
    sys.path.insert(0, str(R4 / "baselines/transport_v4_2"))
    sys.path.insert(0, str(R4 / "baselines"))
    import numeric_decoding_repaired as decoder
    import transport_analysis as transport
    import recalibration
    import frozen_source_transport as source_transport
    return decoder, transport, recalibration, source_transport


def load_cache_cohort(decoder, db: str, manifest: dict, entries: dict,
                      analysis_id: str = "main20k_real") -> dict:
    log(f"COHORT_INPUT_START db={db} analysis_id={analysis_id}")
    entry = entries[f"{db}/{analysis_id}"]
    import v5_cache_validator as cache_validator
    cache_attestation = cache_validator.validate_cache_entry(
        EMBED_MANIFEST, entry, db,
    )
    cache = decoder.load_embedding_cache(entry, db, analysis_id)
    tab_path, split_path, sequence_path = decoder._resolve_cohort_paths(db)
    sequence_hash = sha256_file(sequence_path)
    if sequence_hash.lower() != str(entry["source_npz_sha256"]).lower():
        raise ValueError(f"{db}: cache source sequence hash mismatch")
    if sequence_hash.lower() != str(entry.get("source_npz_sha256", "")).lower():
        raise ValueError(f"{db}: sequence source hash differs from manifest")
    split = decoder._load_checked_split(split_path, str(entry["split_sha256"]), db)
    stay_id, subject, sequence_y = decoder.load_sequence_identity(sequence_path)
    if cache.rows.max() >= len(stay_id):
        raise ValueError(f"{db}: cache row exceeds sequence")
    if (not np.array_equal(stay_id[cache.rows].astype(str), cache.stay_id.astype(str))
            or not np.array_equal(subject[cache.rows].astype(str), cache.subject.astype(str))):
        raise ValueError(f"{db}: cache identities do not match current sequence")
    if len(sequence_y) != len(split):
        raise ValueError(f"{db}: mortality, split, and sequence dimensions differ")
    eligible, loader_entry = decoder._load_ready_eligibility(db, ELIG, None)
    eligible, eligibility_provenance = decoder.verify_current_eligibility(
        db, ELIG, eligible, loader_entry, sequence_hash,
        stay_id, subject, sequence_y,
        expected_manifest_sha256=decoder.EXPECTED_REPAIRED_ELIGIBILITY_MANIFEST_SHA256,
        expected_endpoint="hospital",
    )
    eligible = eligible.copy()
    eligible["stay_id"] = eligible["stay_id"].astype(str)
    eligible["subject"] = eligible["subject"].astype(str)
    if eligible["stay_id"].duplicated().any():
        raise ValueError(f"{db}: eligibility contains duplicate stay IDs")
    cache_position = {str(stay): i for i, stay in enumerate(cache.stay_id)}
    positions = np.asarray([cache_position.get(stay, -1) for stay in eligible["stay_id"]],
                           dtype=np.int64)
    if np.any(positions < 0):
        raise ValueError(f"{db}: eligible rows are absent from the embedding cache")
    rows = cache.rows[positions]
    subjects = cache.subject[positions].astype(str)
    labels = pd.to_numeric(eligible["y_mort"], errors="coerce").to_numpy(dtype=np.int8)
    if not np.array_equal(eligible["subject"].to_numpy(dtype=str), subjects):
        raise ValueError(f"{db}: cache patient IDs disagree with eligibility")
    if not np.array_equal(labels, sequence_y[rows].astype(np.int8)):
        raise ValueError(f"{db}: eligibility labels disagree with sequence labels")
    eligible["rows"] = rows.astype(np.int64)
    eligible["split"] = split[rows].astype(np.int8)
    eligible["y"] = labels
    eligible["db"] = db
    row_frame = eligible
    if row_frame["stay_id"].duplicated().any():
        raise ValueError(f"{db}: duplicate cache stay IDs")
    if not np.isfinite(cache.embedding[positions]).all():
        raise ValueError(f"{db}: selected cache embeddings contain nonfinite values")
    log(f"COHORT_INPUT_READY db={db} eligible_rows={len(row_frame)} cache_rows={len(cache.rows)}")
    return {
        "entry": entry, "cache": cache, "cache_positions": positions,
        "cache_attestation": cache_attestation,
        "embedding": cache.embedding[positions], "row_frame": row_frame.reset_index(drop=True),
        "eligible": eligible.reset_index(drop=True), "split_all": split,
        "sequence_y": sequence_y, "sequence_path": sequence_path,
        "tabular_path": tab_path, "eligibility_provenance": eligibility_provenance,
    }


def source_identity_hash(decoder, frame: pd.DataFrame) -> str:
    return decoder.sha256_ordered_identity(
        "mimic", frame["rows"].to_numpy(dtype=np.int64),
        frame["stay_id"].to_numpy(dtype=str), frame["subject"].to_numpy(dtype=str),
    )


def point_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.int8)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "n_rows": int(len(y)), "n_patients": None,
        "prevalence": float(np.mean(y)), "mean_predicted": float(np.mean(p)),
        "mean_predicted_minus_observed": float(np.mean(p - y)),
        "auroc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
    }


def calibration_fit(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=np.int8)
    x = logit(np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)).reshape(-1, 1)
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000, tol=1e-10)
    model.fit(x, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def _identity_match(left: pd.DataFrame, right: pd.DataFrame, db: str) -> None:
    fields = ["db", "stay_id", "subject", "rows", "split", "y"]
    a, b = left[fields].copy(), right[fields].copy()
    for frame in (a, b):
        frame["db"] = frame["db"].astype(str)
        frame["stay_id"] = frame["stay_id"].astype(str)
        if frame[["db", "stay_id"]].duplicated().any():
            raise ValueError(f"{db}: duplicate model identity keys")
    merged = a.merge(b, on=["db", "stay_id"], how="outer", suffixes=("_a", "_b"),
                     validate="one_to_one", indicator=True)
    if merged["_merge"].ne("both").any() or len(a) != len(b):
        raise ValueError(f"{db}: model arms do not share an exact stay roster")
    for field in fields[2:]:
        if not merged[f"{field}_a"].astype(str).equals(merged[f"{field}_b"].astype(str)):
            raise ValueError(f"{db}: model-arm {field} identities differ")


def _align_scores(reference: pd.DataFrame, scored: pd.DataFrame, db: str) -> pd.DataFrame:
    _identity_match(reference, scored, db)
    keys = ["db", "stay_id"]
    order = reference[keys].copy()
    left = order.merge(reference[["db", "stay_id", "subject", "rows", "split", "y", "p"]],
                       on=keys, validate="one_to_one")
    right = order.merge(scored[["db", "stay_id", "p"]], on=keys, validate="one_to_one",
                        suffixes=("_reference", "_new"))
    joined = left.merge(right, on=keys, validate="one_to_one",
                        suffixes=("_reference", "_new"))
    return joined


def _point_metrics(y: np.ndarray, p: np.ndarray, subject: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("risk probabilities must be finite and within [0, 1]")
    p_loss = np.clip(p, 1e-6, 1 - 1e-6)
    if len(y) == 0 or len(np.unique(y)) != 2:
        raise ValueError("risk comparison requires a nonempty roster with both classes")
    return {
        "n_stays": int(len(y)), "n_patients": int(len(np.unique(subject))),
        "events": int(y.sum()), "prevalence": float(y.mean()),
        "mean_predicted": float(p.mean()),
        "auroc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p_loss)),
        "logloss": float(log_loss(y, p_loss, labels=[0, 1])),
    }


def _paired_loss_ci(y: np.ndarray, p_new: np.ndarray, p_ref: np.ndarray,
                    subject: np.ndarray, *, seed: int, ci_helper) -> list[dict[str, Any]]:
    yy = np.asarray(y, dtype=float)
    a = np.clip(np.asarray(p_new, dtype=float), 1e-6, 1 - 1e-6)
    b = np.clip(np.asarray(p_ref, dtype=float), 1e-6, 1 - 1e-6)
    differences = np.column_stack([
        (yy - a) ** 2 - (yy - b) ** 2,
        -(yy * np.log(a) + (1 - yy) * np.log1p(-a))
        + (yy * np.log(b) + (1 - yy) * np.log1p(-b)),
    ])
    output = []
    for weighting, patient_weighted in (("stay_weighted", False), ("patient_weighted", True)):
        result = ci_helper(subject, differences, draws=1000, seed=seed,
                           patient_weighted=patient_weighted)
        for j, metric in enumerate(("brier", "logloss")):
            output.append({
                "metric": metric, "weighting": weighting,
                "difference_new_minus_reference": float(result["point"][j]),
                "ci_95_low": float(result["low"][j]),
                "ci_95_high": float(result["high"][j]),
                "n_patients": int(result["n_patients"]),
                "n_stays": int(result["n_stays"]),
                "bootstrap_draws": int(result["bootstrap_draws"]),
                "ci_scope": result["scope"],
            })
    return output


def _source_train_identity_hash(decoder, frame: pd.DataFrame) -> str:
    return decoder.sha256_ordered_identity(
        "mimic", frame["rows"].to_numpy(dtype=np.int64),
        frame["stay_id"].to_numpy(dtype=str), frame["subject"].to_numpy(dtype=str),
    )


def _fit_source_grid(data: dict, out_dir: Path, decoder) -> tuple[dict[float, Any], dict[float, Any], dict]:
    frame = data["row_frame"]
    X = np.asarray(data["embedding"], dtype=np.float32)
    y = frame["y"].to_numpy(dtype=np.int8)
    split = frame["split"].to_numpy(dtype=np.int8)
    train, validation = split == 0, split == 1
    if not train.any() or not validation.any():
        raise ValueError("MIMIC split 0 and split 1 must both be nonempty")
    if set(frame.loc[train, "subject"]) & set(frame.loc[validation, "subject"]):
        raise ValueError("MIMIC patient IDs overlap between source split 0 and split 1")
    scaler = StandardScaler().fit(X[train])
    X_train, X_val = scaler.transform(X[train]), scaler.transform(X[validation])
    y_train, y_val = y[train], y[validation]
    if len(np.unique(y_train)) != 2 or len(np.unique(y_val)) != 2:
        raise ValueError("MIMIC source train and validation both need two outcome classes")
    candidate_models, candidate_scalers, grid_rows = {}, {}, []
    for c_value in C_GRID:
        model = LogisticRegression(C=float(c_value), max_iter=3000,
                                   random_state=SOURCE_MODEL_SEED,
                                   solver="lbfgs")
        with threadpool_limits(limits=NUMERICAL_READOUT_THREADS):
            model.fit(X_train, y_train)
        val_probability = model.predict_proba(X_val)[:, 1]
        brier = float(brier_score_loss(y_val, val_probability))
        candidate_models[float(c_value)] = model
        candidate_scalers[float(c_value)] = scaler
        grid_rows.append({
            "C": float(c_value), "validation_brier": brier,
            "validation_logloss": float(log_loss(y_val, val_probability, labels=[0, 1])),
            "n_iter": int(np.max(model.n_iter_)),
            "n_source_train_rows": int(train.sum()),
            "n_source_train_patients": int(frame.loc[train, "subject"].nunique()),
            "n_source_validation_rows": int(validation.sum()),
            "n_source_validation_patients": int(frame.loc[validation, "subject"].nunique()),
            "fit_partition": "MIMIC split0 only",
            "selection_partition": "MIMIC split1 only",
        })
        log(f"SOURCE_C_CANDIDATE_COMPLETE C={c_value} validation_brier={brier:.8f}")
    selected = min(grid_rows, key=lambda row: (row["validation_brier"], row["C"]))
    grid_frame = pd.DataFrame(grid_rows)
    grid_path = out_dir / "source_c_validation_grid.csv"
    _write_csv_atomic(grid_frame, grid_path)
    train_identity = _source_train_identity_hash(decoder, frame.loc[train])
    lock = {
        "status": "source_selection_frozen",
        "schema": "ehr_fm_v5_source_c_selection_1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_db": "mimic", "source_endpoint": "hospital",
        "embedding_manifest_sha256": sha256_file(EMBED_MANIFEST),
        "source_sequence_sha256": sha256_file(data["sequence_path"]),
        "source_split_sha256": str(data["entry"]["split_sha256"]),
        "eligibility_manifest_sha256": data["eligibility_provenance"]["eligibility_manifest_sha256"],
        "eligibility_table_sha256": data["eligibility_provenance"]["eligibility_table_sha256"],
        "source_split0_identity_sha256": train_identity,
        "c_grid": list(C_GRID),
        "numerical_readout_threading": {
            "blas_workers_per_grid_fit": NUMERICAL_READOUT_THREADS,
            "scope": "all six source-grid LogisticRegression.fit calls only",
            "reason": "reproduce the historical C=1 source-head numerical path",
            "dedicated_cgroup_cpu_quota_percent": RESOURCE_LIMITS["runner_cpu_quota_percent"],
            "dedicated_cgroup_cpu_quota_cores": RESOURCE_LIMITS["cpu_threads"],
            "cpu_quota_or_global_thread_policy_changed": False,
        },
        "selection_rule": "minimum MIMIC split1 Brier; exact ties choose smaller C",
        "selected_C": float(selected["C"]),
        "selected_validation_brier": float(selected["validation_brier"]),
        "source_fit_partition": "split0 only; scaler and LogisticRegression fit on split0",
        "source_validation_partition": "split1 selection only; not refit",
        "target_outcomes_opened_before_lock": False,
        "grid_csv": str(grid_path), "grid_csv_sha256": sha256_file(grid_path),
    }
    lock_path = out_dir / "source_choice_lock.json"
    _write_json_atomic(lock_path, lock)
    return candidate_models, {c: scaler for c in candidate_scalers}, lock


def _load_historical_fm_probe():
    if not PROBE.is_file() or sha256_file(PROBE) != EXPECTED_PROBE:
        raise ValueError("historical C=1 source probe is missing or has changed")
    with PROBE.open("rb") as handle:
        probe = pickle.load(handle)
    if not isinstance(probe, dict) or not {"scaler", "probe"}.issubset(probe):
        raise ValueError("historical C=1 probe artifact schema is incomplete")
    return probe


def _score_frame(frame: pd.DataFrame, probability: np.ndarray) -> pd.DataFrame:
    out = frame[["db", "stay_id", "subject", "rows", "split", "y"]].copy()
    out["db"] = out["db"].astype(str)
    out["stay_id"] = out["stay_id"].astype(str)
    out["p"] = np.asarray(probability, dtype=float)
    if len(out) != len(probability) or not np.isfinite(out["p"]).all():
        raise ValueError("scored probability vector is invalid")
    return out


def _selected_c_calibration_500(
    db: str, reference: pd.DataFrame, selected: pd.DataFrame, model_frames: dict,
    selected_c: float, transport, recalibration, *, ci_helper,
) -> tuple[list[dict], list[dict], dict]:
    """Replay the complete v4 500-label plan for the source-selected FM head.

    Only repeat-level aggregates and paired, per-stay-averaged loss summaries
    leave this function. The assessment predictions and identifiers remain in
    memory for pairing against the verified v4 loss archive.
    """
    required = {"db", "stay_id", "subject", "rows", "split", "y", "p",
                "eligible_24h", "temporal_verified"}
    if not required.issubset(reference.columns):
        raise ValueError(f"{db}: calibration reference frame lacks {sorted(required - set(reference.columns))}")
    if not required.issubset(selected.columns):
        raise ValueError(f"{db}: selected-C frame lacks {sorted(required - set(selected.columns))}")
    _identity_match(reference, selected, db)

    direct_dir = DIRECT_CALIBRATION_ROOT / db
    direct_manifest_path = direct_dir / "cohort_manifest.json"
    direct_manifest = json.loads(direct_manifest_path.read_text(encoding="utf-8"))
    if direct_manifest.get("status") != "parity_verified":
        raise ValueError(f"{db}: existing v4 paired calibration archive is not parity_verified")
    direct_archive_path = direct_dir / "server_only_per_stay_mean_losses.npz"
    if sha256_file(direct_archive_path) != direct_manifest.get("per_stay_loss_archive_sha256"):
        raise ValueError(f"{db}: existing v4 per-stay loss archive hash differs from its manifest")

    usable = reference.loc[
        reference["eligible_24h"].astype(bool)
        & reference["temporal_verified"].astype(bool)
    ].copy()
    adaptation, assessment, stratified = recalibration.adaptation_assessment_pools(
        usable, int(transport.SEED)
    )
    if adaptation.empty or assessment.empty:
        raise ValueError(f"{db}: v4 calibration roles have an empty adaptation or assessment pool")
    sampling_plan, draw_indices = recalibration._make_sampling_plan(
        usable, adaptation, assessment, repeats=int(transport.REPEATS),
        seed=int(transport.SEED), budgets=tuple(transport.BUDGETS),
    )
    if sampling_plan.get("sampling_plan_sha256") != direct_manifest.get("sampling_plan_sha256"):
        raise ValueError(f"{db}: reconstructed sample plan differs from parity-verified v4 plan")
    if 500 not in draw_indices or len(draw_indices[500]) != int(transport.REPEATS):
        raise ValueError(f"{db}: existing v4 plan does not contain all 500-label repetitions")

    selected_by_stay = selected.copy()
    selected_by_stay["stay_id"] = selected_by_stay["stay_id"].astype(str)
    selected_by_stay = selected_by_stay.set_index("stay_id", drop=False)
    selected_assess = selected_by_stay.loc[assessment["stay_id"].astype(str)]
    selected_adapt = selected_by_stay.loc[adaptation["stay_id"].astype(str)]
    if (not np.array_equal(selected_assess["y"].to_numpy(dtype=np.int8),
                           assessment["y"].to_numpy(dtype=np.int8))
            or not np.array_equal(selected_adapt["y"].to_numpy(dtype=np.int8),
                                  adaptation["y"].to_numpy(dtype=np.int8))):
        raise ValueError(f"{db}: selected-C labels do not align with the frozen v4 calibration roles")

    adapt_y = adaptation["y"].to_numpy(dtype=np.int8)
    adapt_lg = logit(np.clip(selected_adapt["p"].to_numpy(dtype=float), 1e-6, 1 - 1e-6))
    assess_y = assessment["y"].to_numpy(dtype=np.int8)
    assess_subject = assessment["subject"].astype(str).to_numpy()
    assess_lg = logit(np.clip(selected_assess["p"].to_numpy(dtype=float), 1e-6, 1 - 1e-6))
    assess_raw = np.clip(selected_assess["p"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    methods = ("raw", "offset_operational", "platt_operational")
    metrics = ("brier", "logloss")
    sums = {
        (method, metric): np.zeros(len(assessment), dtype=np.float64)
        for method in methods for metric in metrics
    }
    repeat_rows: list[dict] = []
    plan_draws = [row for row in sampling_plan["draws"]
                  if int(row["budget"]) == 500]
    if len(plan_draws) != int(transport.REPEATS):
        raise ValueError(f"{db}: v4 500-label plan has an unexpected number of draws")
    for repeat, draw_idx in enumerate(draw_indices[500]):
        selected_patients = adaptation.iloc[draw_idx]
        record = plan_draws[repeat]
        draw_hash = hashlib.sha256(np.sort(draw_idx).astype("<i8").tobytes()).hexdigest()
        patient_stay_hash = recalibration._identity_digest(
            selected_patients,
            tuple(col for col in ("db", "subject", "stay_id", "rows", "split")
                  if col in selected_patients.columns),
        )
        if (draw_hash != record.get("draw_index_sha256")
                or patient_stay_hash != record.get("patient_index_stay_sha256")):
            raise ValueError(f"{db}: reconstructed v4 draw identities differ at repeat {repeat}")
        yd = adapt_y[draw_idx]
        events = int(yd.sum())
        repeat_row = {
            "cohort": db, "model_id": "main20k_C_selected", "budget": 500,
            "selected_C": float(selected_c), "repeat": int(repeat),
            "events": events, "nonevents": int(500 - events),
            "adaptation_sample_index_sha256": draw_hash,
            "adaptation_sample_patient_index_stay_sha256": patient_stay_hash,
            "sampling_plan_sha256": sampling_plan["sampling_plan_sha256"],
            "stratified_patient_pool_split": bool(stratified),
        }
        for arm, method in (("offset", "offset_operational"),
                            ("platt", "platt_operational")):
            fallback_probability = float((events + 0.5) / 501.0)
            fit_function = recalibration.fit_offset if arm == "offset" else recalibration.fit_platt
            try:
                slope, intercept = fit_function(yd, adapt_lg[draw_idx])
                probability = np.clip(expit(slope * assess_lg + intercept), 1e-6, 1 - 1e-6)
                fit_ok, fit_status = True, "ok"
            except (ValueError, RuntimeError, FloatingPointError,
                    recalibration.ConvergenceWarning) as exc:
                probability = np.full(len(assessment), fallback_probability, dtype=float)
                slope, intercept = np.nan, np.nan
                fit_ok = False
                fit_status = f"{type(exc).__name__}:{str(exc)[:100]}"
            brier_loss = np.square(assess_y - probability)
            logloss_loss = -(assess_y * np.log(probability)
                             + (1 - assess_y) * np.log1p(-probability))
            sums[(method, "brier")] += brier_loss
            sums[(method, "logloss")] += logloss_loss
            repeat_row.update({
                f"{arm}_fit_ok": bool(fit_ok),
                f"{arm}_fit_status": fit_status,
                f"{arm}_slope": float(slope) if fit_ok else None,
                f"{arm}_intercept": float(intercept) if fit_ok else None,
                f"{arm}_fallback_probability": None if fit_ok else fallback_probability,
                f"{arm}_operational_brier": float(np.mean(brier_loss)),
                f"{arm}_operational_logloss": float(np.mean(logloss_loss)),
            })
        raw_brier = np.square(assess_y - assess_raw)
        raw_logloss = -(assess_y * np.log(assess_raw)
                        + (1 - assess_y) * np.log1p(-assess_raw))
        sums[("raw", "brier")] += raw_brier
        sums[("raw", "logloss")] += raw_logloss
        repeat_row["raw_brier"] = float(np.mean(raw_brier))
        repeat_row["raw_logloss"] = float(np.mean(raw_logloss))
        repeat_rows.append(repeat_row)

    loss_means = {key: value / float(transport.REPEATS) for key, value in sums.items()}
    archive_arrays = np.load(direct_archive_path, allow_pickle=False)
    try:
        archive_stays = np.asarray(archive_arrays["stay_id"]).astype(str)
        archive_subjects = np.asarray(archive_arrays["subject"]).astype(str)
        archive_y = np.asarray(archive_arrays["y"], dtype=np.int8)
        if len(np.unique(archive_stays)) != len(archive_stays):
            raise ValueError(f"{db}: v4 per-stay loss archive contains duplicate stays")
        assess_index = assessment.set_index(assessment["stay_id"].astype(str), drop=False)
        if set(archive_stays) != set(assess_index.index):
            raise ValueError(f"{db}: v4 per-stay loss archive assessment roster differs")
        archive_assess = assess_index.loc[archive_stays]
        if (not np.array_equal(archive_subjects, archive_assess["subject"].astype(str).to_numpy())
                or not np.array_equal(archive_y, archive_assess["y"].to_numpy(dtype=np.int8))):
            raise ValueError(f"{db}: v4 per-stay loss archive assessment identities or labels differ")
        reindexed_subject = archive_assess["subject"].astype(str).to_numpy()
        archive_raw_models = (
            "main20k_real", "conventional_lr_value_plus_pattern",
            "conventional_hgb_value_plus_pattern",
        )
        positions_in_assessment = assessment.set_index(
            assessment["stay_id"].astype(str)
        ).index.get_indexer(archive_stays)
        if np.any(positions_in_assessment < 0):
            raise ValueError(f"{db}: cannot align selected-C losses to the v4 assessment order")
        for reference_model in archive_raw_models:
            if reference_model not in model_frames or db not in model_frames[reference_model]:
                raise ValueError(f"{db}: frozen calibration scores lack {reference_model}")
            ref_frame = model_frames[reference_model][db]
            _identity_match(reference, ref_frame, db)
            ref_by_stay = ref_frame.copy()
            ref_by_stay["stay_id"] = ref_by_stay["stay_id"].astype(str)
            ref_by_stay = ref_by_stay.set_index("stay_id", drop=False)
            ref_assess = ref_by_stay.loc[archive_stays]
            if (not np.array_equal(ref_assess["subject"].astype(str).to_numpy(), archive_subjects)
                    or not np.array_equal(ref_assess["y"].to_numpy(dtype=np.int8), archive_y)):
                raise ValueError(f"{db}/{reference_model}: frozen score archive identity differs")
            ref_probability = np.clip(ref_assess["p"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
            raw_brier = np.square(archive_y.astype(float) - ref_probability)
            raw_logloss = -(archive_y * np.log(ref_probability)
                            + (1 - archive_y) * np.log1p(-ref_probability))
            for metric, computed in (("brier", raw_brier), ("logloss", raw_logloss)):
                raw_key = f"{reference_model}__budget500__raw__{metric}"
                if raw_key not in archive_arrays.files or not np.allclose(
                    archive_arrays[raw_key], computed, rtol=2e-5, atol=2e-6
                ):
                    raise ValueError(f"{db}/{reference_model}: direct v4 raw-score parity failed for {metric}")
        # Compare each model with the exact v4 same-draw, within-stay loss means.
        contrast_rows: list[dict] = []
        summary_rows: list[dict] = []
        for method in methods:
            for metric in metrics:
                candidate_vector = loss_means[(method, metric)][positions_in_assessment]
                summary_rows.append({
                    "cohort": db, "model_id": "main20k_C_selected",
                    "selected_C": float(selected_c), "budget": 500,
                    "method": method, "metric": metric,
                    "assessment_mean_loss": float(np.mean(candidate_vector)),
                    "loss_scope": ("raw point loss" if method == "raw"
                                   else "per-stay loss averaged over all 200 fit-specific probabilities"),
                    "n_assessment_stays": int(len(archive_stays)),
                    "n_assessment_patients": int(len(np.unique(reindexed_subject))),
                    "repeats": int(transport.REPEATS),
                    "sampling_plan_sha256": sampling_plan["sampling_plan_sha256"],
                })
                for reference_model in archive_raw_models:
                    key = f"{reference_model}__budget500__{method}__{metric}"
                    if key not in archive_arrays.files:
                        raise ValueError(f"{db}: v4 loss archive lacks {key}")
                    reference_vector = np.asarray(archive_arrays[key], dtype=float)
                    if len(reference_vector) != len(archive_stays) or not np.isfinite(reference_vector).all():
                        raise ValueError(f"{db}: v4 loss archive has invalid vector {key}")
                    difference = candidate_vector - reference_vector
                    for patient_weighted in (False, True):
                        interval = ci_helper(
                            reindexed_subject, difference, draws=1000,
                            seed=20261123 + 10000 * (0 if db == "eicu_semantic_v4_2" else 1)
                            + 100 * methods.index(method) + metrics.index(metric)
                            + archive_raw_models.index(reference_model),
                            patient_weighted=patient_weighted,
                        )
                        contrast_rows.append({
                            "cohort": db, "candidate_model": "main20k_C_selected",
                            "selected_C": float(selected_c),
                            "reference_model": reference_model, "budget": 500,
                            "method": method, "metric": metric,
                            "difference_candidate_minus_reference": float(interval["point"]),
                            "ci_95_low": float(interval["low"]),
                            "ci_95_high": float(interval["high"]),
                            "weighting": interval["weighting"],
                            "n_assessment_stays": int(interval["n_stays"]),
                            "n_assessment_patients": int(interval["n_patients"]),
                            "bootstrap_draws": int(interval["bootstrap_draws"]),
                            "uncertainty_scope": interval["scope"],
                            "sampling_plan_sha256": sampling_plan["sampling_plan_sha256"],
                        })
    finally:
        archive_arrays.close()

    provenance = {
        "cohort": db,
        "status": "complete",
        "selected_C": float(selected_c),
        "budget": 500,
        "repeats": int(transport.REPEATS),
        "sampling_plan_sha256": sampling_plan["sampling_plan_sha256"],
        "v4_sampling_plan_sha256": direct_manifest["sampling_plan_sha256"],
        "v4_archive_sha256": sha256_file(direct_archive_path),
        "v4_archive_manifest_sha256": sha256_file(direct_manifest_path),
        "n_adaptation_patients": int(adaptation["subject"].nunique()),
        "n_assessment_patients": int(assessment["subject"].nunique()),
        "n_assessment_stays": int(len(assessment)),
        "n_assessment_events": int(assess_y.sum()),
        "fit_success_by_arm": {
            arm: int(sum(bool(row[f"{arm}_fit_ok"]) for row in repeat_rows))
            for arm in ("offset", "platt")
        },
        "fit_failure_by_arm": {
            arm: int(sum(not bool(row[f"{arm}_fit_ok"]) for row in repeat_rows))
            for arm in ("offset", "platt")
        },
        "patient_overlap": len(
            set(adaptation["subject"].astype(str)).intersection(
                set(assessment["subject"].astype(str))
            )
        ),
        "roles_reused": "v4 20261122 random patient roles; one outcome-blind adaptation index stay per patient; all eligible assessment stays",
        "draws_reused": "all v4 500-label draws (200 repetitions), each checked against the plan's index and patient/stay hashes",
        "fallback": "Jeffreys-smoothed adaptation prevalence after offset/Platt fit failure",
        "loss_aggregation": "mean of the 200 fit-specific per-stay losses, followed by patient-cluster paired bootstrap",
        "patient_level_predictions_exported": False,
    }
    return repeat_rows, contrast_rows, {"provenance": provenance,
                                      "summary_rows": summary_rows}


def run_source_selection() -> None:
    started = time.monotonic()
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite existing v5 readout output: {OUT}")
    OUT.mkdir(parents=True)
    log("source-only C selection starting")
    decoder, transport, recalibration, source_transport = load_modules()
    manifest, entries = decoder.load_embedding_manifest(EMBED_MANIFEST)
    manifest_hash = sha256_file(EMBED_MANIFEST)
    if manifest_hash.lower() != EXPECTED_EMBED_MANIFEST.lower():
        raise ValueError("embedding manifest SHA256 differs from the frozen v4.2 manifest")
    if len(manifest.get("models", [])) != 41:
        raise ValueError("expected the frozen 41-row v4.2 embedding manifest")
    mimic = load_cache_cohort(decoder, "mimic", manifest, entries)
    models, scalers, lock = _fit_source_grid(mimic, OUT, decoder)
    selected_c = float(lock["selected_C"])
    selected_model = models[selected_c]
    # The source-selection lock is written before any external labels or scores are loaded.
    log(f"SOURCE_SELECTION_LOCKED C={selected_c} validation_brier={lock['selected_validation_brier']:.8f}")

    source_frame = mimic["row_frame"]
    source_test = source_frame.loc[source_frame["split"].eq(2)].copy()
    if source_test.empty:
        raise ValueError("MIMIC split2 source test roster is empty")
    X_test = np.asarray(mimic["embedding"], dtype=np.float32)[source_frame["split"].to_numpy() == 2]
    historical_artifact = _load_historical_fm_probe()
    historical_meta = {key: value for key, value in historical_artifact.items()
                       if key not in ("scaler", "probe")}
    historical_scaler = historical_artifact["scaler"]
    historical_probe = historical_artifact["probe"]
    source_train = source_frame.loc[source_frame["split"].eq(0)].copy()
    source_train_identity = _source_train_identity_hash(decoder, source_train)
    source_train_label_hash = hashlib.sha256(
        source_train["y"].to_numpy(dtype=np.int8).tobytes()).hexdigest()
    if historical_meta.get("source_train_identity_sha256") != source_train_identity:
        raise ValueError("historical C=1 probe source-train identity differs from current MIMIC split0")
    if historical_meta.get("source_train_label_sha256") != source_train_label_hash:
        raise ValueError("historical C=1 probe source-train labels differ from current MIMIC split0")
    if int(historical_meta.get("source_train_n", -1)) != len(source_train):
        raise ValueError("historical C=1 probe source-train row count differs from current MIMIC split0")
    if int(historical_meta.get("source_train_patient_n", -1)) != int(source_train["subject"].nunique()):
        raise ValueError("historical C=1 probe source-train patient count differs from current MIMIC split0")
    historical_params = historical_probe.get_params()
    if (float(historical_params.get("C", float("nan"))) != 1.0
            or historical_params.get("random_state") != SOURCE_MODEL_SEED
            or historical_params.get("class_weight") is not None):
        raise ValueError("historical probe is not the frozen C=1, seed=20261122 source head")
    if getattr(historical_scaler, "n_samples_seen_", None) != len(source_train):
        raise ValueError("historical probe scaler was not fit on the current MIMIC split0 row count")
    p_c1_refit = models[1.0].predict_proba(scalers[1.0].transform(X_test))[:, 1]
    p_selected = selected_model.predict_proba(scalers[selected_c].transform(X_test))[:, 1]
    p_historical = historical_probe.predict_proba(
        historical_scaler.transform(X_test)
    )[:, 1]
    c1_reproduction_maxdiff = float(np.max(np.abs(p_c1_refit - p_historical)))
    if not np.isfinite(c1_reproduction_maxdiff) or c1_reproduction_maxdiff > C1_REPRODUCTION_TOLERANCE:
        raise ValueError(
            "historical C=1 source-head reproduction failed before target score loading: "
            f"max_abs_difference={c1_reproduction_maxdiff:.10g}, "
            f"tolerance={C1_REPRODUCTION_TOLERANCE:.10g}"
        )
    _write_json_atomic(OUT / "source_c1_reproduction_guard.json", {
        "status": "passed",
        "historical_probe_sha256": sha256_file(PROBE),
        "source_train_identity_sha256": source_train_identity,
        "source_train_label_sha256": source_train_label_hash,
        "source_train_rows": int(len(source_train)),
        "source_train_patients": int(source_train["subject"].nunique()),
        "historical_head_C": float(historical_params["C"]),
        "historical_head_random_state": int(historical_params["random_state"]),
        "historical_head_class_weight": historical_params["class_weight"],
        "max_abs_probability_difference": c1_reproduction_maxdiff,
        "tolerance": C1_REPRODUCTION_TOLERANCE,
        "checked_before_external_prediction_frames": True,
    })
    source_new_c = _score_frame(source_test, p_selected)
    source_old_c1 = _score_frame(source_test, p_historical)

    # Load and validate the existing frozen conventional source predictions.
    choices, source_refs, _source_features, _nvar = source_transport._load_source_frames()
    source_results = []
    source_contrasts = []
    source_y = source_test["y"].to_numpy(dtype=np.int8)
    source_subject = source_test["subject"].astype(str).to_numpy()
    for model_id, score in (("main20k_C1_historical", source_old_c1),
                            ("main20k_C_selected", source_new_c)):
        values = _point_metrics(source_y, score["p"].to_numpy(), source_subject)
        source_results.append({"cohort": "mimic", "scope": "split2_source_test",
                               "model_id": model_id, **values})
    for analysis_id in ("conventional_lr_value_plus_pattern",
                        "conventional_hgb_value_plus_pattern"):
        reference = source_refs[analysis_id]["mimic"].copy()
        reference = reference.loc[reference["split"].astype(int).eq(2)].copy()
        _identity_match(source_new_c, reference, "mimic")
        values = _point_metrics(reference["y"].to_numpy(dtype=np.int8),
                                reference["p"].to_numpy(dtype=float),
                                reference["subject"].astype(str).to_numpy())
        source_results.append({"cohort": "mimic", "scope": "split2_source_test",
                               "model_id": analysis_id, **values})
    for model_id, reference in (
        ("main20k_C1_historical", source_old_c1),
        ("conventional_lr_value_plus_pattern",
         source_refs["conventional_lr_value_plus_pattern"]["mimic"].query("split == 2")),
        ("conventional_hgb_value_plus_pattern",
         source_refs["conventional_hgb_value_plus_pattern"]["mimic"].query("split == 2")),
    ):
        reference = reference.copy()
        if "p" not in reference:
            raise ValueError(f"{model_id}: source reference lacks prediction probabilities")
        aligned = _align_scores(reference, source_new_c, "mimic")
        source_contrasts.extend({
            "cohort": "mimic", "scope": "split2_source_test",
            "candidate_model": "main20k_C_selected", "reference_model": model_id,
            **row,
        } for row in _paired_loss_ci(
            aligned["y"].to_numpy(), aligned["p_new"].to_numpy(),
            aligned["p_reference"].to_numpy(), aligned["subject"].astype(str).to_numpy(),
            seed=SEED + 3000 + len(source_contrasts), ci_helper=__import__(
                "v5_curve_tools").paired_cluster_mean_ci,
        ))

    source_metrics_path = OUT / "source_c_mimic_split2_metrics.csv"
    source_contrasts_path = OUT / "source_c_mimic_split2_loss_contrasts.csv"
    source_metrics_frame = pd.DataFrame(source_results)
    source_contrasts_frame = pd.DataFrame(source_contrasts)
    _write_csv_atomic(source_metrics_frame, source_metrics_path)
    _write_csv_atomic(source_contrasts_frame, source_contrasts_path)
    log("SOURCE_MIMIC_SPLIT2_AGGREGATES_SAVED before external prediction frames")

    # Only after the lock: read target scores and evaluate the selected frozen source head.
    frames, gates, endpoint_by_db, transport_provenance = transport._load_all_model_frames()
    target_results = []
    target_contrasts = []
    target_rosters = {}
    target_cache_attestations = {}
    selected_calibration_repeats = []
    selected_calibration_summaries = []
    selected_calibration_contrasts = []
    selected_calibration_provenance = {}
    for db in TARGET_DBS:
        log(f"TARGET_SCORE_START db={db}")
        cohort = load_cache_cohort(decoder, db, manifest, entries)
        target_cache_attestations[db] = cohort["cache_attestation"]
        probabilities = selected_model.predict_proba(
            scalers[selected_c].transform(np.asarray(cohort["embedding"], dtype=np.float32))
        )[:, 1]
        new_scores = _score_frame(cohort["row_frame"], probabilities)
        fm_reference = frames["main20k_real"][db]
        new_target_scores = new_scores.loc[
            new_scores["stay_id"].isin(fm_reference["stay_id"].astype(str))
        ].copy()
        _identity_match(fm_reference, new_target_scores, db)
        # Keep the exact reviewed target score roster used by the frozen v4 comparison.
        new_target = _align_scores(fm_reference, new_target_scores, db)
        aligned_new = fm_reference[["db", "stay_id", "subject", "rows", "split", "y"]].copy()
        aligned_new["p"] = new_target["p_new"].to_numpy(dtype=float)
        selected_full = fm_reference.copy()
        selected_full["p"] = new_target["p_new"].to_numpy(dtype=float)
        calibration_repeats, calibration_contrasts, calibration_payload = (
            _selected_c_calibration_500(
                db, fm_reference, selected_full, frames, selected_c,
                transport, recalibration,
                ci_helper=__import__("v5_curve_tools").paired_cluster_mean_ci,
            )
        )
        selected_calibration_repeats.extend(calibration_repeats)
        selected_calibration_summaries.extend(calibration_payload["summary_rows"])
        selected_calibration_contrasts.extend(calibration_contrasts)
        selected_calibration_provenance[db] = calibration_payload["provenance"]
        target_rosters[db] = {
            "n_stays": int(len(aligned_new)),
            "n_patients": int(aligned_new["subject"].nunique()),
            "identity_sha256": decoder.sha256_ordered_identity(
                db, aligned_new["rows"].to_numpy(dtype=np.int64),
                aligned_new["stay_id"].astype(str).to_numpy(),
                aligned_new["subject"].astype(str).to_numpy()),
            "roster_source": "exact reviewed main20k_real frozen v4 target score roster",
        }
        for model_id in ("main20k_real", "conventional_lr_value_plus_pattern",
                         "conventional_hgb_value_plus_pattern"):
            reference = frames[model_id][db]
            _identity_match(fm_reference, reference, db)
            values = _point_metrics(reference["y"].to_numpy(dtype=np.int8),
                                    reference["p"].to_numpy(dtype=float),
                                    reference["subject"].astype(str).to_numpy())
            target_results.append({"cohort": db, "scope": "reviewed_external_roster",
                                   "model_id": model_id, **values})
        values = _point_metrics(aligned_new["y"].to_numpy(dtype=np.int8),
                                aligned_new["p"].to_numpy(dtype=float),
                                aligned_new["subject"].astype(str).to_numpy())
        target_results.append({"cohort": db, "scope": "reviewed_external_roster",
                               "model_id": "main20k_C_selected", **values})
        for model_id in ("main20k_real", "conventional_lr_value_plus_pattern",
                         "conventional_hgb_value_plus_pattern"):
            reference = frames[model_id][db]
            aligned = _align_scores(reference, aligned_new, db)
            target_contrasts.extend({
                "cohort": db, "scope": "reviewed_external_roster",
                "candidate_model": "main20k_C_selected", "reference_model": model_id,
                **row,
            } for row in _paired_loss_ci(
                aligned["y"].to_numpy(), aligned["p_new"].to_numpy(),
                aligned["p_reference"].to_numpy(), aligned["subject"].astype(str).to_numpy(),
                seed=SEED + 5000 + 1000 * TARGET_DBS.index(db) + len(target_contrasts),
                ci_helper=__import__("v5_curve_tools").paired_cluster_mean_ci,
            ))
        log(f"TARGET_SCORE_COMPLETE db={db} stays={len(aligned_new)}")

    metrics_frame = pd.DataFrame(source_results + target_results)
    contrasts_frame = pd.DataFrame(source_contrasts + target_contrasts)
    _write_csv_atomic(metrics_frame, OUT / "source_c_paired_roster_metrics.csv")
    _write_csv_atomic(contrasts_frame, OUT / "source_c_paired_loss_contrasts.csv")
    calibration_repeats_path = OUT / "source_c_selected_500_calibration_repeat_metrics.csv"
    calibration_summary_path = OUT / "source_c_selected_500_calibration_loss_summary.csv"
    calibration_contrasts_path = OUT / "source_c_selected_500_calibration_paired_contrasts.csv"
    _write_csv_atomic(pd.DataFrame(selected_calibration_repeats), calibration_repeats_path)
    _write_csv_atomic(pd.DataFrame(selected_calibration_summaries), calibration_summary_path)
    _write_csv_atomic(pd.DataFrame(selected_calibration_contrasts), calibration_contrasts_path)
    result_path = OUT / "source_c_paired_roster_metrics.csv"
    contrast_path = OUT / "source_c_paired_loss_contrasts.csv"
    summary = {
        "status": "complete",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "MIMIC source split2 plus repaired eICU and NWICU only",
        "run_id": RUN_ID or "standalone_r4",
        "raw_probability_loss_clip": [1e-6, 1 - 1e-6],
        "resource_scope": RESOURCE_SCOPE,
        "cgroup_path": current_cgroup_path(),
        "execution_resource_limits": (RESOURCE_LIMITS if RUN_ID else
                                       {"status": "standalone_limits_not_attested"}),
        "numerical_readout_threading": {
            "blas_workers_per_grid_fit": NUMERICAL_READOUT_THREADS,
            "scope": "all six source-grid LogisticRegression.fit calls only",
            "reason": "reproduce the historical C=1 source-head numerical path",
            "dedicated_cgroup_cpu_quota_percent": RESOURCE_LIMITS["runner_cpu_quota_percent"],
            "dedicated_cgroup_cpu_quota_cores": RESOURCE_LIMITS["cpu_threads"],
            "cpu_quota_or_global_thread_policy_changed": False,
        },
        "source_choice_lock": str(OUT / "source_choice_lock.json"),
        "source_choice_lock_sha256": sha256_file(OUT / "source_choice_lock.json"),
        "source_c1_reproduction_guard": str(OUT / "source_c1_reproduction_guard.json"),
        "source_c1_reproduction_guard_sha256": sha256_file(OUT / "source_c1_reproduction_guard.json"),
        "source_mimic_split2_metrics_csv": str(source_metrics_path),
        "source_mimic_split2_metrics_csv_sha256": sha256_file(source_metrics_path),
        "source_mimic_split2_contrasts_csv": str(source_contrasts_path),
        "source_mimic_split2_contrasts_csv_sha256": sha256_file(source_contrasts_path),
        "source_c1_refit_vs_historical_max_abs_probability_difference": c1_reproduction_maxdiff,
        "source_c1_refit_matches_historical_probe_within_2e-6": bool(c1_reproduction_maxdiff <= 2e-6),
        "embedding_manifest_sha256": manifest_hash,
        "cache_attestations": {
            "mimic": mimic["cache_attestation"],
            **{db: target_cache_attestations[db] for db in TARGET_DBS},
        },
        "transport_sampling_plan_sha256": sha256_file(R4_PLAN),
        "transport_provenance": transport_provenance,
        "target_endpoints": {db: endpoint_by_db[db] for db in TARGET_DBS},
        "source_split_counts": {
            str(k): int((source_frame["split"] == k).sum()) for k in (0, 1, 2)
        },
        "target_evaluated": list(TARGET_DBS),
        "evaluated_rosters": {
            "mimic": {
                "split": 2,
                "n_stays": int(len(source_test)),
                "n_patients": int(source_test["subject"].nunique()),
                "identity_sha256": decoder.sha256_ordered_identity(
                    "mimic", source_test["rows"].to_numpy(dtype=np.int64),
                    source_test["stay_id"].astype(str).to_numpy(),
                    source_test["subject"].astype(str).to_numpy()),
                "roster_source": "eligible MIMIC split2 source test partition",
            },
            **target_rosters,
        },
        "sicdb_scored": False,
        "input_cache_rows_used": {
            "mimic": int(len(mimic["cache"].rows)),
            **{db: int(entries[f"{db}/main20k_real"]["n_rows"]) for db in TARGET_DBS},
        },
        "models": [
            "historical main20k C=1",
            "MIMIC split1-Brier-selected main20k C",
            "existing source-selected value-plus-pattern logistic regression",
            "existing source-selected value-plus-pattern histogram gradient boosting",
        ],
        "model_ids": {
            "selected_candidate": "main20k_C_selected",
            "historical_reference": "main20k_C1_historical",
            "external_references": ["main20k_real", "conventional_lr_value_plus_pattern",
                                    "conventional_hgb_value_plus_pattern"],
        },
        "same_roster_identity_checks": "all source and target model arms passed one-to-one stay/patient/source-row/split/label identity checks",
        "contrast_sign": "candidate minus reference; negative Brier/logloss differences favor the candidate",
        "bootstrap_unit": "patient; all stays from each sampled patient are carried together",
        "paired_ci": "1000 patient-cluster draws for paired Brier and logloss loss differences; stay- and patient-weighted",
        "selected_c_calibration_500": {
            "status": "complete",
            "budget": 500,
            "repeats": REPEATS,
            "provenance_by_cohort": selected_calibration_provenance,
            "repeat_metrics_csv": str(calibration_repeats_path),
            "repeat_metrics_csv_sha256": sha256_file(calibration_repeats_path),
            "loss_summary_csv": str(calibration_summary_path),
            "loss_summary_csv_sha256": sha256_file(calibration_summary_path),
            "paired_contrasts_csv": str(calibration_contrasts_path),
            "paired_contrasts_csv_sha256": sha256_file(calibration_contrasts_path),
            "comparison_scope": "selected-C raw, offset, and Platt operational losses paired against the same v4 500-label / 200-draw losses for historical C1, V+P LR, and V+P HGBT",
            "fallback": "Jeffreys-smoothed adaptation prevalence after offset/Platt fit failure",
            "loss_aggregation": "per-stay loss averaged over all 200 fit-specific probabilities before patient-cluster uncertainty",
        },
        "csv_outputs": {
            "source_c_mimic_split2_metrics.csv": {
                "purpose": "point metrics for selected C, historical C=1, and frozen conventional references on MIMIC split2",
                "columns": list(source_metrics_frame.columns),
            },
            "source_c_mimic_split2_loss_contrasts.csv": {
                "purpose": "paired selected-C minus historical/conventional Brier and logloss differences on MIMIC split2",
                "columns": list(source_contrasts_frame.columns),
            },
            "source_c_paired_roster_metrics.csv": {
                "purpose": "point metrics for every scored arm on the matched MIMIC split2 and reviewed external rosters",
                "columns": list(metrics_frame.columns),
            },
            "source_c_paired_loss_contrasts.csv": {
                "purpose": "paired selected-C minus historical/frozen-reference Brier and logloss contrasts with cluster intervals",
                "columns": list(contrasts_frame.columns),
            },
            "source_c_selected_500_calibration_repeat_metrics.csv": {
                "purpose": "repeat-level aggregate events, fit status, fallback, and assessment scores for all matched 500-label selected-C offset and Platt fits",
                "columns": list(pd.DataFrame(selected_calibration_repeats).columns),
            },
            "source_c_selected_500_calibration_loss_summary.csv": {
                "purpose": "selected-C raw point losses and within-stay losses averaged over the complete 200-repeat 500-label fit plan",
                "columns": list(pd.DataFrame(selected_calibration_summaries).columns),
            },
            "source_c_selected_500_calibration_paired_contrasts.csv": {
                "purpose": "selected-C minus historical C1, V+P LR, and V+P HGBT loss differences on exactly matched v4 500-label draws and assessment stays",
                "columns": list(pd.DataFrame(selected_calibration_contrasts).columns),
            },
        },
        "patient_level_predictions_exported": False,
        "metrics_csv": str(result_path),
        "metrics_csv_sha256": sha256_file(result_path),
        "paired_contrasts_csv": str(contrast_path),
        "paired_contrasts_csv_sha256": sha256_file(contrast_path),
        "runtime_seconds": round(time.monotonic() - started, 3),
    }
    _write_json_atomic(OUT / "readout_manifest.json", summary)
    log(f"READOUT_SOURCE_C_SELECTION_COMPLETE C={selected_c} output={OUT}")


if __name__ == "__main__":
    run_source_selection()
