"""MIMIC-source-tuned conventional summaries of the v4 V/B/H/L inputs.

The run path checks every temporal eligibility manifest before loading cohort
arrays or fitting a source model. It only tunes against MIMIC split 1, keeps
the selected model fit on split 0 only, and predicts MIMIC split 2 plus
explicitly ready external cohorts. Target labels are only attached as
evaluation labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from baseline_features import summarize_inputs
from paired_baselines import (
    INPUT_SCOPE, PRIMARY_SCOPE, compare_artifacts, ordered_row_digest,
    validate_complete_fm_prediction_manifest,
)

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
ENGINE_CODE = REV / "engine/code"
PROC = ROOT / "data/processed"
OUT = REV / "baselines"
SEED = 20261122


def _engine_module():
    engine_dir = str(ENGINE_CODE)
    if engine_dir not in sys.path:
        sys.path.insert(0, engine_dir)
    import v4_engine
    return v4_engine


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _identity_hash(db: str, frame: pd.DataFrame) -> str:
    h = hashlib.sha256(db.encode("utf-8"))
    for row in frame[["rows", "stay_id", "subject"]].itertuples(index=False, name=None):
        h.update(str(int(row[0])).encode("ascii"))
        for value in row[1:]:
            encoded = str(value).encode("utf-8")
            h.update(len(encoded).to_bytes(8, "little"))
            h.update(encoded)
    return h.hexdigest()


def _candidate_specs() -> dict[str, list[dict[str, Any]]]:
    lr = [{"C": C} for C in (0.01, 0.1, 1.0, 10.0)]
    hgb = [
        {"learning_rate": 0.1, "max_iter": 200, "early_stopping": False,
         "max_leaf_nodes": leaves, "l2_regularization": reg, "min_samples_leaf": 20}
        for leaves in (15, 31) for reg in (10.0, 1.0)
    ]
    return {"conventional_lr": lr, "conventional_hgb": hgb}


def _new_estimator(model_tag: str, params: dict[str, Any], seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    if model_tag == "conventional_lr":
        return LogisticRegression(C=params["C"], max_iter=3000,
                                  class_weight=None, random_state=seed,
                                  solver="lbfgs")
    if model_tag == "conventional_hgb":
        return HistGradientBoostingClassifier(**params, random_state=seed)
    raise ValueError(f"unknown model_tag: {model_tag}")


def _complexity_key(model_tag: str, params: dict[str, Any]) -> tuple[Any, ...]:
    if model_tag == "conventional_lr":
        # Lower C means stronger shrinkage and is the predeclared simpler tie-break.
        return (float(params["C"]),)
    return (int(params["max_leaf_nodes"]), -float(params["l2_regularization"]))


def final_source_fit_mask(train_mask: np.ndarray, validation_mask: np.ndarray,
                          test_mask: np.ndarray | None = None) -> np.ndarray:
    """Keep the final conventional fit on split 0; validation/test stay held out."""
    train_mask = np.asarray(train_mask, dtype=bool)
    validation_mask = np.asarray(validation_mask, dtype=bool)
    if len(train_mask) != len(validation_mask) or np.any(train_mask & validation_mask):
        raise ValueError("source train and validation masks overlap or have different lengths")
    if test_mask is not None:
        test_mask = np.asarray(test_mask, dtype=bool)
        if len(test_mask) != len(train_mask) or np.any(train_mask & test_mask) or np.any(validation_mask & test_mask):
            raise ValueError("source train/validation/test masks overlap or have different lengths")
    return train_mask.copy()


def tune_source(
    model_tag: str, feature_type: str, X: np.ndarray, y: np.ndarray,
    train_mask: np.ndarray, validation_mask: np.ndarray, *, seed: int,
) -> tuple[Any, Any, dict[str, Any], list[dict[str, Any]]]:
    """Choose by source validation Brier; final model uses source train only."""
    from sklearn.metrics import brier_score_loss
    from sklearn.preprocessing import StandardScaler
    from sklearn.exceptions import ConvergenceWarning

    if not train_mask.any() or not validation_mask.any():
        raise ValueError("source train and MIMIC validation rows must both be nonempty")
    if len(np.unique(y[train_mask])) != 2 or len(np.unique(y[validation_mask])) != 2:
        raise ValueError("source train and validation must each contain both classes")
    if set(X.shape[:1]) != {len(y)}:
        raise ValueError("feature and source-label dimensions differ")

    scale = None
    if model_tag == "conventional_lr":
        scale = StandardScaler().fit(X[train_mask])
        X_train = scale.transform(X[train_mask])
        X_val = scale.transform(X[validation_mask])
    else:
        X_train, X_val = X[train_mask], X[validation_mask]
    trials: list[dict[str, Any]] = []
    fitted: list[tuple[tuple[float, tuple[Any, ...]], Any, dict[str, Any]]] = []
    for rank, params in enumerate(_candidate_specs()[model_tag]):
        estimator = _new_estimator(model_tag, params, seed)
        started = time.perf_counter()
        convergence_warning = False
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                estimator.fit(X_train, y[train_mask])
                convergence_warning = any(issubclass(w.category, ConvergenceWarning) for w in caught)
            pred = estimator.predict_proba(X_val)[:, 1]
            val_brier = float(brier_score_loss(y[validation_mask], pred))
            status = "fit_with_convergence_warning" if convergence_warning else "ok"
            fit_seconds = time.perf_counter() - started
            trials.append({
                "model_tag": model_tag, "feature_type": feature_type,
                "parameters": params, "candidate_order": rank,
                "validation_brier": val_brier, "convergence_warning": convergence_warning,
                "fit_seconds": fit_seconds, "status": status,
                "n_source_train": int(train_mask.sum()),
                "n_source_validation": int(validation_mask.sum()),
            })
            fitted.append(((val_brier, _complexity_key(model_tag, params)), estimator, params))
        except Exception as exc:
            trials.append({
                "model_tag": model_tag, "feature_type": feature_type,
                "parameters": params, "candidate_order": rank,
                "validation_brier": None, "convergence_warning": convergence_warning,
                "fit_seconds": time.perf_counter() - started,
                "status": f"failed:{type(exc).__name__}:{str(exc)[:180]}",
                "n_source_train": int(train_mask.sum()),
                "n_source_validation": int(validation_mask.sum()),
            })
    if not fitted:
        raise RuntimeError(f"all source candidates failed for {model_tag}/{feature_type}")
    _, final_estimator, selected_params = min(fitted, key=lambda item: item[0])
    fit_mask = final_source_fit_mask(train_mask, validation_mask)
    selected_trial = next(row for row in trials if row["parameters"] == selected_params
                          and row["validation_brier"] is not None)
    choice = {
        "model_tag": model_tag, "feature_type": feature_type,
        "selected_parameters": selected_params,
        "selected_validation_brier": min(fitted, key=lambda item: item[0])[0][0],
        "tie_break": "lowest MIMIC validation Brier; exact ties choose simpler parameterization",
        "source_fit_rows": int(fit_mask.sum()),
        "source_train_rows_for_selection": int(train_mask.sum()),
        "source_validation_rows_for_selection": int(validation_mask.sum()),
        "final_model_fit_scope": "MIMIC source split 0 only; split 1 is selection-only and not refit",
        "final_model_fit_seconds": selected_trial["fit_seconds"],
        "final_model_convergence_warning": selected_trial["convergence_warning"],
    }
    return scale, final_estimator, choice, trials


def _summarize_conditioned_inputs(
    ds: dict[str, np.ndarray], rows: np.ndarray, n_variables: int, *,
    condition: str, variable_ids: tuple[int, ...], engine: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Summarize the exact V/B/H arrays sent to the frozen encoder condition."""
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or not len(rows):
        raise ValueError("conditioned feature construction requires nonempty source rows")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("conditioned feature rows must be unique")
    if np.any(rows < 0) or np.any(rows >= len(ds["L"])):
        raise ValueError("conditioned feature rows are outside the sequence arrays")
    prepare_inputs = getattr(engine, "prepare_inputs", None)
    if prepare_inputs is None:
        prepare_inputs = engine.prepare_batch
    V, B, H = prepare_inputs(ds, rows, condition, variable_ids)
    compact = {
        "V": np.asarray(V), "B": np.asarray(B), "H": np.asarray(H),
        "L": np.asarray(ds["L"], dtype=np.int64)[rows],
    }
    expected_shape = compact["V"].shape
    if compact["V"].ndim != 2 or any(
        compact[key].shape != expected_shape for key in ("B", "H")
    ) or expected_shape[0] != len(rows):
        raise ValueError("engine condition returned inconsistent V/B/H batch dimensions")
    if np.any(compact["L"] > expected_shape[1]):
        raise ValueError("engine condition batch width is shorter than a source sequence length")
    return summarize_inputs(compact, np.arange(len(rows), dtype=np.int64), n_variables)


def _calibration_parameters(y: np.ndarray, p: np.ndarray,
                           weights: np.ndarray | None = None) -> tuple[float, float]:
    from scipy.special import logit
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y, dtype=np.int8)
    if np.unique(y).size != 2:
        return float("nan"), float("nan")
    lg = logit(np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, max_iter=3000, tol=1e-8, solver="lbfgs")
    model.fit(lg, y, sample_weight=weights)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def evaluation_metrics(frame: pd.DataFrame, *, n_boot: int, seed: int,
                       cluster_unit: str) -> dict[str, Any]:
    """Point metrics plus conditional patient/subject-cluster bootstrap CIs."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = frame["y"].to_numpy(dtype=np.int8)
    p = np.clip(frame["p"].to_numpy(dtype=float), 1e-12, 1 - 1e-12)
    groups = frame["subject"].astype(str).to_numpy()
    unique, inverse = np.unique(groups, return_inverse=True)
    if np.unique(y).size < 2:
        return {"status": "single_class_evaluation", "n_stays": len(y),
                "n_subject_clusters": len(unique), "bootstrap_cluster_unit": cluster_unit}
    intercept, slope = _calibration_parameters(y, p)
    point = {
        "prevalence": float(np.mean(y)),
        "average_precision": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)),
        "brier": float(np.mean((y - p) ** 2)),
        "logloss": float(np.mean(-y * np.log(p) - (1-y) * np.log1p(-p))),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }
    values = {key: [] for key in point}
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        sample = rng.integers(0, len(unique), size=len(unique))
        weights = np.bincount(sample, minlength=len(unique)).astype(float)[inverse]
        if weights.sum() == 0 or np.unique(y[weights > 0]).size < 2:
            continue
        values["prevalence"].append(float(np.average(y, weights=weights)))
        values["average_precision"].append(float(average_precision_score(y, p, sample_weight=weights)))
        values["auroc"].append(float(roc_auc_score(y, p, sample_weight=weights)))
        values["brier"].append(float(np.average((y - p) ** 2, weights=weights)))
        values["logloss"].append(float(np.average(-y * np.log(p) - (1-y) * np.log1p(-p), weights=weights)))
        try:
            b_intercept, b_slope = _calibration_parameters(y, p, weights)
            values["calibration_intercept"].append(b_intercept)
            values["calibration_slope"].append(b_slope)
        except (ValueError, RuntimeError, FloatingPointError):
            continue
    row: dict[str, Any] = {
        "status": "ok", "n_stays": int(len(y)),
        "n_subject_clusters": int(len(unique)),
        "bootstrap_cluster_unit": cluster_unit,
        "independent_patient_clustering_verified": cluster_unit == "patient",
        "bootstrap_draws_requested": n_boot,
        "ci_scope": "patient/subject-cluster bootstrap conditional on fixed source-selected model and prediction cohort",
    }
    for name, value in point.items():
        draws = np.asarray(values[name], dtype=float)
        row[name] = value
        row[f"{name}_ci_95_lo"] = float(np.percentile(draws, 2.5)) if len(draws) >= n_boot // 2 else np.nan
        row[f"{name}_ci_95_hi"] = float(np.percentile(draws, 97.5)) if len(draws) >= n_boot // 2 else np.nan
        row[f"{name}_valid_bootstrap_draws"] = int(len(draws))
    return row


def _source_feature_set(features: dict[str, dict[str, dict[str, np.ndarray]]],
                        condition: str, feature_type: str) -> np.ndarray:
    return features["mimic"][condition][feature_type]


def _hash_json(document: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdefABCDEF" for char in text)


def _condition_specs_from_manifest(
    document: dict[str, Any], cohorts: tuple[str, ...],
) -> tuple[list[str], dict[str, dict[str, dict[str, Any]]]]:
    """Select real and optional no-H0 FM conditions as baseline mask contracts."""
    validate_complete_fm_prediction_manifest(document)
    if document.get("mode") != "predict":
        raise ValueError("baseline requires a complete retrospective prediction manifest")
    predictions = document.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("FM prediction manifest lacks per-analysis prediction records")

    def records_for(condition: str, db: str) -> list[dict[str, Any]]:
        return [row for row in predictions
                if isinstance(row, dict)
                and str(row.get("model_tag")) == "main20k"
                and str(row.get("input_condition")) == condition
                and str(row.get("db")) == db
                and str(row.get("mode")) == "predict"]

    selected: dict[str, dict[str, dict[str, Any]]] = {"real": {}}
    for db in cohorts:
        matches = records_for("real", db)
        if len(matches) != 1:
            raise ValueError(f"{db}: expected one main20k/real FM prediction record; found {len(matches)}")
        record = matches[0]
        _validate_condition_record(record, "real", db)
        selected["real"][db] = {
            "analysis_id": str(record["analysis_id"]),
            "input_condition": "real",
            "input_mask_sha256": str(record["input_mask_sha256"]).lower(),
            "input_mask_spec": None,
            "variable_ids": (),
        }

    no_h0_records = [row for row in predictions
                     if isinstance(row, dict)
                     and str(row.get("model_tag")) == "main20k"
                     and str(row.get("input_condition")) == "no_h0_therapy"
                     and str(row.get("mode")) == "predict"]
    conditions = ["real"]
    if no_h0_records:
        mapping = document.get("no_h0_therapy_mapping")
        expected_names = ["vaso", "rrt", "sedation"]
        if not isinstance(mapping, dict):
            raise ValueError("no_h0_therapy FM manifest lacks the cohort token mapping")
        if mapping.get("canonical_variable_names") != expected_names:
            raise ValueError("no_h0_therapy canonical variable mapping differs from the registered order")
        ids_by_cohort = mapping.get("variable_ids_by_cohort")
        condition_hash = mapping.get("condition_definition_sha256")
        if not isinstance(ids_by_cohort, dict) or not _is_sha256(condition_hash):
            raise ValueError("no_h0_therapy mapping lacks cohort IDs or a condition-definition SHA256")
        selected["no_h0_therapy"] = {}
        for db in cohorts:
            matches = records_for("no_h0_therapy", db)
            if len(matches) != 1:
                raise ValueError(
                    f"{db}: no_h0_therapy is present in the FM run but has "
                    f"{len(matches)} current main20k prediction records")
            record = matches[0]
            _validate_condition_record(record, "no_h0_therapy", db)
            mask_spec = record.get("input_mask_spec")
            if not isinstance(mask_spec, dict):
                raise ValueError(f"{db}: no_h0_therapy prediction lacks input_mask_spec")
            expected_ids = _cohort_therapy_ids(ids_by_cohort, db, expected_names)
            spec_ids = mask_spec.get("variable_ids")
            if (mask_spec.get("condition") != "no_h0_therapy"
                    or mask_spec.get("db") != db
                    or mask_spec.get("canonical_variable_names") != expected_names
                    or spec_ids != list(expected_ids)
                    or mask_spec.get("hour") != 0
                    or mask_spec.get("masked_features") != ["V", "B", "H"]
                    or mask_spec.get("mask_action") != "zero"
                    or mask_spec.get("stay_retention_rule") !=
                    "retain iff any nonzero variable token remains"
                    or mask_spec.get("condition_definition_sha256") != condition_hash):
                raise ValueError(f"{db}: no_h0_therapy mask definition does not match its manifest mapping")
            if _hash_json(mask_spec) != str(record.get("input_mask_sha256", "")).lower():
                raise ValueError(f"{db}: no_h0_therapy input mask hash does not match its definition")
            selected["no_h0_therapy"][db] = {
                "analysis_id": str(record["analysis_id"]),
                "input_condition": "no_h0_therapy",
                "input_mask_sha256": str(record["input_mask_sha256"]).lower(),
                "input_mask_spec": mask_spec,
                "variable_ids": expected_ids,
            }
        conditions.append("no_h0_therapy")
    return conditions, selected


def _unpaired_real_condition_specs(
    engine: Any, cohorts: tuple[str, ...],
) -> tuple[list[str], dict[str, dict[str, dict[str, Any]]]]:
    """Build the registered unmasked condition contract for independent fits."""
    selected: dict[str, dict[str, dict[str, Any]]] = {"real": {}}
    for db in cohorts:
        selected["real"][db] = {
            "analysis_id": "main20k_real",
            "input_condition": "real",
            "input_mask_sha256": engine.input_mask_hash("real", ()),
            "input_mask_spec": None,
            "variable_ids": (),
        }
    return ["real"], selected


def _cohort_therapy_ids(mapping: dict[str, Any], db: str,
                        names: list[str]) -> tuple[int, ...]:
    entry = mapping.get(db)
    if not isinstance(entry, dict) or set(entry) != set(names):
        raise ValueError(f"{db}: no_h0_therapy token IDs do not match the canonical names")
    values = []
    for name in names:
        value = entry[name]
        if isinstance(value, bool):
            raise ValueError(f"{db}: boolean is not a valid token ID for {name}")
        try:
            token_id = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{db}: invalid token ID for {name}") from exc
        if token_id <= 0 or str(token_id) != str(value):
            raise ValueError(f"{db}: token ID for {name} must be a positive integer")
        values.append(token_id)
    if len(set(values)) != len(values):
        raise ValueError(f"{db}: no_h0_therapy names map to duplicate token IDs")
    return tuple(values)


def _validate_condition_record(record: dict[str, Any], condition: str, db: str) -> None:
    required = ("analysis_id", "input_mask_sha256", "n_prediction_rows",
                "role", "prediction_scope", "input_scope",
                "clinical_input_availability_verified",
                "retrospective_landmark_prediction_allowed")
    if any(record.get(key) in (None, "") for key in required):
        raise ValueError(f"{db}/{condition}: FM manifest record lacks required scope/provenance")
    if int(record["n_prediction_rows"]) <= 0:
        raise ValueError(f"{db}/{condition}: FM condition has no evaluation predictions")
    if (record.get("role") != "evaluation"
            or record.get("prediction_scope") != PRIMARY_SCOPE
            or record.get("input_scope") != INPUT_SCOPE
            or record.get("clinical_input_availability_verified") is not False
            or record.get("retrospective_landmark_prediction_allowed") is not True
            or not _is_sha256(record.get("input_mask_sha256"))):
        raise ValueError(f"{db}/{condition}: FM condition is outside the approved prediction scope")


def _effective_prediction_scope(entry: dict[str, Any], engine: Any) -> str:
    """Use the engine's registered scope mapping when an attestation omits it."""
    scope = entry.get("prediction_scope")
    if scope not in (None, ""):
        return str(scope)
    return str(engine.prediction_scope(entry.get("evidence", "unavailable")))


def run(cohorts: tuple[str, ...], eligibility_dir: Path, out_dir: Path,
        endpoint: str = "hospital", seed: int = SEED,
        fm_prediction_manifest: Path | None = None,
        allow_unpaired: bool = False) -> dict[str, Any]:
    """Run only after every requested cohort passes the retrospective landmark gate."""
    if "mimic" not in cohorts:
        raise ValueError("MIMIC must be included as the source train/validation cohort")
    if fm_prediction_manifest is None and not allow_unpaired:
        raise ValueError(
            "the current run-scoped FM prediction manifest is required unless "
            "the explicit unpaired source-fit mode is enabled")
    engine = _engine_module()
    if fm_prediction_manifest is None:
        fm_manifest_path = None
        fm_manifest_doc: dict[str, Any] = {}
        conditions, condition_specs = _unpaired_real_condition_specs(engine, cohorts)
    else:
        fm_manifest_path = Path(fm_prediction_manifest).resolve()
        fm_manifest_doc = json.loads(fm_manifest_path.read_text(encoding="utf-8"))
        conditions, condition_specs = _condition_specs_from_manifest(fm_manifest_doc, cohorts)

    # Gate every cohort before sequence loading, feature construction, or fitting.
    ready: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}
    for db in cohorts:
        eligible, entry = engine.load_ready_eligibility(db, eligibility_dir)
        eligible = eligible.loc[eligible["endpoint"].astype(str) == endpoint].copy()
        if eligible.empty:
            raise engine.ClinicalReadinessError(f"{db}: no ready eligible rows for endpoint {endpoint}")
        if entry.get("retrospective_landmark_prediction_allowed") is not True:
            raise engine.ClinicalReadinessError(
                f"{db}: retrospective_landmark_prediction_allowed is not true")
        ready[db] = (eligible, entry)

    names_path = ROOT / "data/processed/meta_mimic.json"
    meta = json.loads(names_path.read_text(encoding="utf-8"))
    variables = meta.get("vars")
    if not isinstance(variables, list) or not variables:
        raise ValueError("meta_mimic.json must define the canonical variable vocabulary")
    n_variables = len(variables)

    datasets: dict[str, dict[str, dict[str, Any]]] = {}
    features: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    sequence_source_hashes: dict[str, str] = {}
    for db in cohorts:
        sequence_path = PROC / f"seq_{db}.npz"
        sequence_source_hashes[db] = _sha256(sequence_path)
        attested_source_hash = str(ready[db][1].get("sequence_npz_sha256", "")).lower()
        if attested_source_hash != sequence_source_hashes[db].lower():
            raise engine.ClinicalReadinessError(
                f"{db}: current sequence NPZ SHA256 differs from the temporal attestation")
        ds = engine.load_sequence(db)
        datasets[db] = {}
        features[db] = {}
        for condition in conditions:
            variable_ids = tuple(condition_specs[condition][db]["variable_ids"])
            rows = engine.condition_rows(ds, condition, variable_ids)
            matched = engine.align_eligible_rows(db, ds, ready[db][0], rows)
            if matched.empty:
                raise engine.ClinicalReadinessError(
                    f"{db}/{condition}: eligible rows have no nonempty V/B/H/L inputs")
            matched["split"] = np.asarray(ds["split"], dtype=np.int8)[
                matched["rows"].to_numpy(dtype=np.int64)]
            matched["y"] = matched["y_mort"].to_numpy(dtype=np.int8)
            matched["subject"] = matched["subject"].astype(str)
            features[db][condition] = dict(zip(
                ("value_plus_pattern", "pattern_only"),
                _summarize_conditioned_inputs(
                    ds, matched["rows"].to_numpy(dtype=np.int64), n_variables,
                    condition=condition, variable_ids=variable_ids, engine=engine,
                ),
            ))
            datasets[db][condition] = {"ds": ds, "matched": matched}

    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = out_dir / "results/predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    choices: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    prediction_frames: dict[str, pd.DataFrame] = {}
    metric_rows: list[dict[str, Any]] = []
    feature_dimensions: dict[str, dict[str, dict[str, Any]]] = {}
    for db in cohorts:
        feature_dimensions[db] = {}
        for condition in conditions:
            feature_dimensions[db][condition] = {
                key: {"shape": list(value.shape), "dtype": str(value.dtype),
                      "bytes": int(value.nbytes)}
                for key, value in features[db][condition].items()
            }
    for condition in conditions:
        source = datasets["mimic"][condition]["matched"]
        train_mask = source["split"].to_numpy() == 0
        validation_mask = source["split"].to_numpy() == 1
        test_mask = source["split"].to_numpy() == 2
        if not train_mask.any() or not validation_mask.any() or not test_mask.any():
            raise engine.ClinicalReadinessError(
                f"MIMIC/{condition} requires eligible nonempty split 0/1/2 rows")
        sets = [set(source.loc[mask, "subject"]) for mask in
                (train_mask, validation_mask, test_mask)]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise engine.ClinicalReadinessError(
                f"MIMIC/{condition} source train/validation/test patient groups overlap")
        y_source = source["y"].to_numpy(dtype=np.int8)
        if any(np.unique(y_source[mask]).size != 2
               for mask in (train_mask, validation_mask)):
            raise engine.ClinicalReadinessError(
                f"MIMIC/{condition} source train and validation each require both classes")
        for feature_type in ("pattern_only", "value_plus_pattern"):
            X_source = _source_feature_set(features, condition, feature_type)
            for model_tag in ("conventional_lr", "conventional_hgb"):
                scaler, model, choice, model_trials = tune_source(
                    model_tag, feature_type, X_source, y_source,
                    train_mask, validation_mask, seed=seed,
                )
                condition_spec = condition_specs[condition]["mimic"]
                choice.update({
                    "input_condition": condition,
                    "input_mask_sha256": condition_spec["input_mask_sha256"],
                    "fm_analysis_id": condition_spec["analysis_id"],
                    "source_train_identity_sha256": _identity_hash("mimic", source.loc[train_mask]),
                    "source_validation_identity_sha256": _identity_hash("mimic", source.loc[validation_mask]),
                    "source_final_fit_identity_sha256": _identity_hash("mimic", source.loc[train_mask]),
                    "source_train_patients": int(source.loc[train_mask, "subject"].nunique()),
                    "source_validation_patients": int(source.loc[validation_mask, "subject"].nunique()),
                    "source_test_patients": int(source.loc[test_mask, "subject"].nunique()),
                    "variable_vocab_sha256": _sha256(names_path),
                    "sequence_source_sha256": sequence_source_hashes["mimic"],
                })
                choices.append(choice)
                for trial in model_trials:
                    trial["input_condition"] = condition
                trials.extend(model_trials)
                analysis_id = f"{model_tag}_{feature_type}"
                if condition != "real":
                    analysis_id = f"{analysis_id}_{condition}"
                all_frames = []
                for db in cohorts:
                    current = datasets[db][condition]["matched"]
                    if db == "mimic":
                        eval_mask = current["split"].to_numpy() == 2
                    else:
                        eval_mask = np.ones(len(current), dtype=bool)
                    if not eval_mask.any():
                        continue
                    X = features[db][condition][feature_type][eval_mask]
                    X_scaled = scaler.transform(X) if scaler is not None else X
                    prob = model.predict_proba(X_scaled)[:, 1]
                    eval_rows = current.loc[eval_mask].copy().reset_index(drop=True)
                    entry = ready[db][1]
                    db_condition_spec = condition_specs[condition][db]
                    out = pd.DataFrame({
                        "db": db, "analysis_id": analysis_id,
                        "stay_id": eval_rows["stay_id"].astype(str),
                        "subject": eval_rows["subject"].astype(str),
                        "rows": eval_rows["rows"].astype(np.int64),
                        "arrayindex": eval_rows["rows"].astype(np.int64),
                        "split": eval_rows["split"].astype(np.int8),
                        "y": eval_rows["y"].astype(np.int8), "p": prob,
                        "model_tag": model_tag, "feature_type": feature_type,
                        "input_condition": condition,
                        "input_mask_sha256": db_condition_spec["input_mask_sha256"],
                        "endpoint": eval_rows["endpoint"].astype(str),
                        "role": "evaluation",
                        "prediction_scope": _effective_prediction_scope(entry, engine),
                        "input_scope": entry.get("input_scope"),
                        "clinical_input_availability_verified": entry.get("clinical_input_availability_verified"),
                        "eligibility_manifest_sha256": entry.get("eligibility_manifest_sha256"),
                        "eligibility_table_sha256": entry.get("eligibility_table_sha256"),
                        "evidence_basis": entry.get("evidence_basis", entry.get("evidence")),
                    })
                    all_frames.append(out)
                if not all_frames:
                    raise RuntimeError(f"{analysis_id}: no evaluation predictions were generated")
                full = pd.concat(all_frames, ignore_index=True)
                path = prediction_dir / f"{analysis_id}.parquet"
                full.to_parquet(path, index=False)
                prediction_frames[analysis_id] = full
                choice["prediction_file"] = str(path)
                choice["n_predictions_by_db"] = {
                    db: int(n) for db, n in full.groupby("db").size().to_dict().items()
                }
                for db, evaluation_frame in full.groupby("db", sort=True):
                    entry = ready[str(db)][1]
                    identity = entry.get("subject_identity_basis", {})
                    cluster_unit = ("patient" if isinstance(identity, dict)
                                    and identity.get("patient_disjoint_claim_allowed") is True
                                    else "supplied_subject_identifier")
                    metrics = evaluation_metrics(
                        evaluation_frame, n_boot=1000,
                        seed=seed + sum(ord(ch) for ch in str(db)),
                        cluster_unit=cluster_unit,
                    )
                    metric_rows.append({"analysis_id": analysis_id,
                                        "input_condition": condition,
                                        "db": db, **metrics})
                print(json.dumps({"analysis_id": analysis_id,
                                  "input_condition": condition,
                                  "prediction_counts": choice["n_predictions_by_db"],
                                  "selected_parameters": choice["selected_parameters"]}), flush=True)

    baseline_prediction_manifest_path = out_dir / "results/baseline_prediction_manifest.json"
    prediction_artifacts: list[dict[str, Any]] = []
    for analysis_id, frame in prediction_frames.items():
        path = (prediction_dir / f"{analysis_id}.parquet").resolve()
        file_sha256 = _sha256(path)
        for db, db_frame in frame.groupby("db", sort=True):
            entry = ready[str(db)][1]
            condition = str(db_frame["input_condition"].iloc[0])
            db_condition_spec = condition_specs[condition][str(db)]
            if (_effective_prediction_scope(entry, engine) != PRIMARY_SCOPE
                    or entry.get("input_scope") != INPUT_SCOPE
                    or entry.get("clinical_input_availability_verified") is not False):
                raise engine.ClinicalReadinessError(
                    f"{db}: prediction metadata does not meet the retrospective landmark measurement-time scope")
            record = {
                "analysis_id": analysis_id, "db": str(db),
                "prediction_file": str(path), "prediction_file_sha256": file_sha256,
                "ordered_row_digest_sha256": ordered_row_digest(db_frame),
                "n_prediction_rows": int(len(db_frame)),
                "role": "evaluation", "prediction_scope": PRIMARY_SCOPE,
                "input_scope": INPUT_SCOPE,
                "clinical_input_availability_verified": False,
                "retrospective_landmark_prediction_allowed": True,
                "input_condition": condition,
                "input_mask_sha256": db_condition_spec["input_mask_sha256"],
                "input_mask_spec": db_condition_spec["input_mask_spec"],
                "fm_analysis_id": db_condition_spec["analysis_id"],
                "eligibility_manifest_sha256": str(entry["eligibility_manifest_sha256"]),
                "eligibility_table_sha256": str(entry["eligibility_table_sha256"]),
                "source_npz_sha256": sequence_source_hashes[str(db)],
            }
            prediction_artifacts.append(record)
    baseline_prediction_manifest = {
        "status": "complete", "endpoint": endpoint, "seed": seed,
        "input_conditions": conditions,
        "no_h0_therapy_mapping": fm_manifest_doc.get("no_h0_therapy_mapping"),
        "analysis_ids": sorted(prediction_frames),
        "cohort_readiness": {db: {
            "retrospective_landmark_prediction_allowed": entry.get("retrospective_landmark_prediction_allowed"),
            "ready_for_future24h_clinical_claim": False,
            "eligibility_manifest_sha256": entry.get("eligibility_manifest_sha256"),
            "eligibility_table_sha256": entry.get("eligibility_table_sha256"),
            "sequence_npz_sha256": sequence_source_hashes[db],
            "prediction_scope": _effective_prediction_scope(entry, engine),
            "input_scope": entry.get("input_scope"),
            "clinical_input_availability_verified": entry.get("clinical_input_availability_verified"),
        } for db, (_, entry) in ready.items()},
        "predictions": prediction_artifacts,
    }
    baseline_prediction_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_prediction_manifest_path.write_text(
        json.dumps(baseline_prediction_manifest, indent=2), encoding="utf-8")
    if fm_manifest_path is None:
        pairing: list[dict[str, Any]] = []
        paired_metrics: list[dict[str, Any]] = []
        pairing_summary = {
            "status": "deferred_no_fm_prediction_manifest",
            "reason": "independent source fits completed before the current FM prediction manifest",
        }
    else:
        pairing, paired_metrics, pairing_summary = compare_artifacts(
            baseline_prediction_manifest_path, fm_manifest_path,
            n_boot=1000, seed=seed,
        )
    manifest = {
        "status": "complete", "analysis_stage": "source_validation_selection_and_retrospective_landmark_prediction",
        "seed": seed, "cohorts": list(cohorts), "endpoint": endpoint,
        "cohort_readiness": {db: {
            "evidence": entry.get("evidence"),
            "evidence_basis": entry.get("evidence_basis"),
            "retrospective_landmark_prediction_allowed": entry.get("retrospective_landmark_prediction_allowed"),
            "ready_for_future24h_clinical_claim": False,
            "eligibility_table_sha256": entry.get("eligibility_table_sha256"),
            "eligibility_manifest_sha256": entry.get("eligibility_manifest_sha256"),
            "prediction_scope": _effective_prediction_scope(entry, engine),
            "input_scope": entry.get("input_scope"),
            "clinical_input_availability_verified": False,
        } for db, (_, entry) in ready.items()},
        "feature_contract": {
            "vocabulary": "canonical MIMIC token vocabulary shared by V/B/H/L inputs",
            "value_plus_pattern": "observed,count/128,mean/min/max/last bin /10,first/last/mean hour /23",
            "pattern_only": "observed,count/128,first/last/mean hour /23; no bin-value summaries",
            "last_bin_tie_break": "maximum hour, then last source token order",
            "missing_continuous_summaries": 0,
            "feature_construction_uses_outcomes": False,
            "input_condition_specs": {
                condition: {
                    db: condition_specs[condition][db]["input_mask_spec"]
                    for db in cohorts
                }
                for condition in conditions
            },
        },
        "selection": "MIMIC split 0 only fits candidates; MIMIC split 1 Brier selects; ties choose simpler; selected model remains fit on split 0 only; no split 1 refit; MIMIC split 2 and external cohorts are evaluation only",
        "grids": _candidate_specs(), "choices": choices,
        "candidate_trials": trials,
        "feature_matrix_dimensions_and_bytes": feature_dimensions,
        "evaluation_metrics_file": str(out_dir / "results/baseline_evaluation_metrics.csv"),
        "baseline_prediction_manifest": str(baseline_prediction_manifest_path.resolve()),
        "baseline_prediction_manifest_sha256": _sha256(baseline_prediction_manifest_path),
        "fm_prediction_manifest": (str(fm_manifest_path)
                                    if fm_manifest_path is not None else None),
        "pairing_to_fm": pairing_summary,
        "paired_comparison_deferred": fm_manifest_path is None,
        "paired_metrics_file": str(out_dir / "results/baseline_paired_metrics.csv"),
        "target_recalibration": "use the same actual patient/index-stay draw hashes after exact matched-population verification; compare runs with verify_shared_recalibration_manifests before pooling",
        "patient_level_data_exported_to_local": False,
        "fit_wall_time_recorded": True,
        "convergence_warning_recorded": True,
        "baseline_code_sha256": _sha256(Path(__file__)),
        "feature_code_sha256": _sha256(Path(__file__).with_name("baseline_features.py")),
        "variable_vocab_sha256": _sha256(names_path),
    }
    manifest_path = out_dir / "results/baseline_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(pairing).to_csv(out_dir / "results/baseline_pairing_audit.csv", index=False)
    pd.DataFrame(paired_metrics).to_csv(out_dir / "results/baseline_paired_metrics.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(out_dir / "results/baseline_evaluation_metrics.csv", index=False)
    return manifest


def _self_test() -> None:
    ds = {
        "L": np.array([3, 2]),
        "V": np.array([[1, 1, 2], [2, 0, 0]]),
        "B": np.array([[1, 3, 10], [5, 0, 0]]),
        "H": np.array([[5, 5, 23], [4, 0, 0]]),
    }
    value, pattern = summarize_inputs(ds, np.array([0, 1]), 2)
    assert value.shape == (2, 18)
    assert pattern.shape == (2, 10)
    # Variable 1 at tied maximum hour uses the last source token's bin (3).
    assert np.isclose(value[0, 5], .3)
    assert np.isclose(value[0, 0], 1.0)
    assert np.isclose(value[0, 1], 2 / 128)
    # Missing value summaries remain zero, while unobserved stays explicit.
    assert np.all(value[1, 0:9] == 0)
    assert np.all(pattern[1, 0:5] == 0)
    specs = _candidate_specs()
    assert len(specs["conventional_lr"]) == 4
    assert len(specs["conventional_hgb"]) == 4
    train = np.array([True, True, False, False])
    validation = np.array([False, False, True, False])
    test = np.array([False, False, False, True])
    final = final_source_fit_mask(train, validation, test)
    assert np.array_equal(final, train) and not np.any(final & (validation | test))
    print("CONVENTIONAL_BASELINE_SELF_TESTS_PASS", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("run", "self-test"), default="run")
    ap.add_argument("--eligibility-dir", default=str(REV / "temporal/results/eligibility"))
    ap.add_argument("--cohorts", default="mimic,eicu")
    ap.add_argument("--endpoint", default="hospital", choices=("hospital", "icu"))
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--fm-prediction-manifest", type=Path,
                    help="current run-scoped FM manifest; artifact paths and hashes are verified before pairing")
    ap.add_argument("--allow-unpaired", action="store_true",
                    help="fit only the unmasked real condition before FM predictions exist; defer all pairing")
    args = ap.parse_args(argv)
    if args.mode == "self-test":
        _self_test()
        return 0
    if args.fm_prediction_manifest is None and not args.allow_unpaired:
        ap.error("--fm-prediction-manifest is required unless --allow-unpaired is explicit")
    cohorts = tuple(x.strip() for x in args.cohorts.split(",") if x.strip())
    run(cohorts, Path(args.eligibility_dir), Path(args.out_dir), args.endpoint, args.seed,
        fm_prediction_manifest=args.fm_prediction_manifest,
        allow_unpaired=args.allow_unpaired)
    print("CONVENTIONAL_BASELINE_RUN_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

