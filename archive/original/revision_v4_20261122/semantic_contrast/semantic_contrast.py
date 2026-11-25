#!/usr/bin/env python3
"""Paired descriptive contrasts from the saved original and recovered runs.

This job only reads bound prediction manifests, prediction parquet files,
source-probe artifacts, and the prior aggregate preview. It does not read raw
clinical tables, fit models, encode cohorts, or export patient-level rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
OLD_RUN = REV / "engine/results/prediction_runs/status_consistent_primary_core_fix2-7da980ffcbd8f00b"
NEW_RUN = REV / "engine/results/prediction_runs/recovered_external_primary_v4_2_resume3-04b21bfc05943b7f"
OLD_SERVER_MANIFEST = OLD_RUN / "prediction_manifest.json"
NEW_SERVER_MANIFEST = NEW_RUN / "prediction_manifest.json"
PREVIEW = REV / "admin/recovered_main_preview_v4_2/main20k_real_point_metrics.csv"
EXPECTED_ROWS = 130_544
EXPECTED_PATIENTS = 102_957
EXPECTED_EVENTS = 11_943
EXPECTED_MANIFEST_SHA256 = {
    "old": "b2818a3443c8cf5fe34ef500beb79f3874c2729fa59c0397433d40be29d56971",
    "new": "4a52de662bf1e28c165c0387d8a95fa3208afc46e255c1a57e6526ee61d78340",
}
ARM_SPECS = {
    "original_real": ("old", "eicu", "main20k_real", "real"),
    "original_constant": ("old", "eicu", "main20k_constant", "constant"),
    "repaired_real": ("new", "eicu_semantic_v4_2", "main20k_real", "real"),
    "repaired_constant": ("new", "eicu_semantic_v4_2", "main20k_constant", "constant"),
}
METRIC_NAMES = ("auroc", "average_precision", "brier", "logloss")
CONTRASTS = {
    "repaired_real_minus_original_real": ("repaired_real", "original_real"),
    "repaired_constant_minus_original_constant": ("repaired_constant", "original_constant"),
}
SEED = 20261123
BOOTSTRAP_REPS = 1000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _sorted_score_groups(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-np.asarray(p, dtype=np.float64), kind="mergesort")
    sorted_p = np.asarray(p, dtype=np.float64)[order]
    starts = np.r_[0, np.flatnonzero(sorted_p[1:] != sorted_p[:-1]) + 1]
    return order, starts


def weighted_metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray,
                     groups: tuple[np.ndarray, np.ndarray] | None = None) -> dict[str, float]:
    """Compute weighted binary metrics, grouping equal scores for AUC and AP."""
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or w.ndim != 1 or not (len(y) == len(p) == len(w)):
        raise ValueError("y, p, and w must be one-dimensional arrays of equal length")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must contain only binary labels")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("p must contain finite probabilities in [0, 1]")
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("w must contain finite nonnegative weights")
    total = float(w.sum())
    pos_total = float(w[y == 1].sum())
    neg_total = float(w[y == 0].sum())
    if total <= 0 or pos_total <= 0 or neg_total <= 0:
        return {name: math.nan for name in METRIC_NAMES}
    order, starts = groups if groups is not None else _sorted_score_groups(p)
    pos = np.add.reduceat(w[order] * y[order], starts)
    neg = np.add.reduceat(w[order] * (1 - y[order]), starts)
    cum_pos = np.cumsum(pos)
    cum_neg = np.cumsum(neg)
    auroc = float(np.sum(pos * (neg_total - cum_neg + 0.5 * neg)) / (pos_total * neg_total))
    precision = cum_pos / np.maximum(cum_pos + cum_neg, 1e-300)
    average_precision = float(np.sum((pos / pos_total) * precision))
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    brier = float(np.sum(w * (p - y) ** 2) / total)
    logloss = float(-np.sum(w * (y * np.log(clipped) + (1 - y) * np.log1p(-clipped))) / total)
    return {"auroc": auroc, "average_precision": average_precision,
            "brier": brier, "logloss": logloss}


def paired_contrasts(metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    out = {
        name: {metric: metrics[right][metric] - metrics[left][metric]
               for metric in METRIC_NAMES}
        for name, (right, left) in CONTRASTS.items()
    }
    out["difference_in_differences"] = {
        metric: ((metrics["repaired_real"][metric] - metrics["repaired_constant"][metric])
                 - (metrics["original_real"][metric] - metrics["original_constant"][metric]))
        for metric in METRIC_NAMES
    }
    return out


def get_manifest_entry(manifest: dict, analysis_id: str, db: str) -> dict:
    matches = [row for row in manifest.get("predictions", [])
               if row.get("analysis_id") == analysis_id and row.get("db") == db
               and int(row.get("n_predictions", 0)) > 0]
    if len(matches) != 1:
        raise ValueError(f"expected one positive manifest entry for {analysis_id}/{db}, got {len(matches)}")
    return matches[0]


def load_prediction(entry: dict, cache: dict[Path, object], validate_predictions,
                    sha256_ordered_prediction_rows):
    import pandas as pd

    analysis_id, db = str(entry["analysis_id"]), str(entry["db"])
    path = Path(entry["prediction_file"])
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_file_hash = sha256_file(path)
    if actual_file_hash != str(entry.get("prediction_file_sha256", "")).lower():
        raise ValueError(f"{analysis_id}/{db}: prediction parquet SHA256 mismatch")
    if path not in cache:
        cache[path] = pd.read_parquet(path)
    full = cache[path]
    if "db" not in full:
        raise ValueError(f"{analysis_id}: prediction parquet has no db column")
    part = full.loc[full["db"].astype(str).eq(db)].copy()
    part = validate_predictions(part, analysis_id)
    if len(part) != int(entry.get("n_predictions", -1)) or len(part) != int(entry.get("n_prediction_rows", -1)):
        raise ValueError(f"{analysis_id}/{db}: row count differs from manifest")
    if entry.get("role") != "evaluation" or not part["role"].astype(str).eq("evaluation").all():
        raise ValueError(f"{analysis_id}/{db}: non-evaluation rows found")
    if sha256_ordered_prediction_rows(part) != str(entry.get("ordered_row_digest_sha256", "")).lower():
        raise ValueError(f"{analysis_id}/{db}: ordered row digest mismatch")
    return part


def _canonical_identity(frame):
    out = frame.copy()
    out["_stay_key"] = out["stay_id"].map(str)
    out["_subject_key"] = out["subject"].map(str)
    if out["_stay_key"].duplicated().any():
        raise ValueError("duplicate stay identities")
    return out


def verify_four_arm_eicu(frames: dict[str, object]) -> tuple[object, dict, dict[str, np.ndarray]]:
    import pandas as pd

    identity_cols = ("_subject_key", "rows", "split", "y", "endpoint")
    keyed = {name: _canonical_identity(frame) for name, frame in frames.items()}
    base = keyed["original_real"].sort_values("_stay_key", kind="mergesort").reset_index(drop=True)
    base_ids = base["_stay_key"].to_numpy()
    if len(base) != EXPECTED_ROWS:
        raise ValueError(f"eICU stay count mismatch: expected {EXPECTED_ROWS}, found {len(base)}")
    for name, candidate in keyed.items():
        candidate = candidate.sort_values("_stay_key", kind="mergesort").reset_index(drop=True)
        if not np.array_equal(base_ids, candidate["_stay_key"].to_numpy()):
            missing = int(len(set(base_ids) - set(candidate["_stay_key"])))
            added = int(len(set(candidate["_stay_key"]) - set(base_ids)))
            raise ValueError(f"{name}: eICU roster mismatch (missing={missing}, added={added})")
        for column in identity_cols:
            if not base[column].reset_index(drop=True).equals(candidate[column].reset_index(drop=True)):
                raise ValueError(f"{name}: eICU identity/label field mismatch for {column}")
    subjects = base["_subject_key"].astype(str).to_numpy()
    y = base["y"].to_numpy(dtype=np.int8)
    n_patients, events = int(pd.Series(subjects).nunique()), int(y.sum())
    if n_patients != EXPECTED_PATIENTS or events != EXPECTED_EVENTS:
        raise ValueError(f"eICU patient/event mismatch: patients={n_patients}, events={events}")
    aligned = {}
    for name, frame in keyed.items():
        part = frame.sort_values("_stay_key", kind="mergesort").reset_index(drop=True)
        aligned[name] = part["p"].to_numpy(dtype=np.float64)
    return base, {"status": "exact_four_arm_roster", "stays": len(base),
                  "patients": n_patients, "events": events, "missing_or_added_ids": 0,
                  "identity_fields_equal": list(identity_cols), "arm_count": len(frames)}, aligned


def compare_mimic(old_frame, new_frame, condition: str) -> dict:
    old = _canonical_identity(old_frame)
    new = _canonical_identity(new_frame)
    cols = ("_subject_key", "rows", "split", "y", "endpoint", "role",
            "input_scope", "prediction_scope")
    old = old.sort_values("_stay_key", kind="mergesort").reset_index(drop=True)
    new = new.sort_values("_stay_key", kind="mergesort").reset_index(drop=True)
    if not np.array_equal(old["_stay_key"].to_numpy(), new["_stay_key"].to_numpy()):
        raise ValueError(f"MIMIC {condition}: saved prediction stay roster changed")
    equal = {column: old[column].reset_index(drop=True).equals(new[column].reset_index(drop=True))
             for column in cols}
    if not all(equal.values()):
        raise ValueError(f"MIMIC {condition}: saved prediction identity mismatch: {equal}")
    delta = np.abs(old["p"].to_numpy(dtype=np.float64) - new["p"].to_numpy(dtype=np.float64))
    if not np.array_equal(old["p"].to_numpy(dtype=np.float64), new["p"].to_numpy(dtype=np.float64)):
        raise ValueError(f"MIMIC {condition}: source-cohort prediction values differ")
    return {"condition": condition, "status": "exact_identity_labels_and_predictions",
            "n_predictions": int(len(old)), "n_unpaired": 0,
            "max_abs_prediction_difference": float(delta.max()) if len(delta) else None,
            "identity_checks": equal}


def compare_probe_pair(old_entry: dict, new_entry: dict, condition: str) -> dict:
    old_path, new_path = Path(old_entry["source_probe_artifact"]), Path(new_entry["source_probe_artifact"])
    old_hash, new_hash = sha256_file(old_path), sha256_file(new_path)
    old_expected = str(old_entry.get("source_probe_sha256", "")).lower()
    new_expected = str(new_entry.get("source_probe_artifact_sha256")
                       or new_entry.get("source_probe_sha256", "")).lower()
    if old_hash != old_expected or new_hash != new_expected:
        raise ValueError(f"{condition}: source-probe artifact hash mismatch")
    with old_path.open("rb") as handle:
        old = pickle.load(handle)
    with new_path.open("rb") as handle:
        new = pickle.load(handle)
    metadata_fields = ("model_tag", "input_condition", "checkpoint_sha256", "input_mask_sha256",
                       "source_train_identity_sha256", "source_train_label_sha256",
                       "source_train_n", "source_train_patient_n")
    metadata_equal = {field: old.get(field) == new.get(field) for field in metadata_fields}
    array_fields = {
        "scaler_mean": (old["scaler"].mean_, new["scaler"].mean_),
        "scaler_scale": (old["scaler"].scale_, new["scaler"].scale_),
        "scaler_var": (old["scaler"].var_, new["scaler"].var_),
        "probe_coef": (old["probe"].coef_, new["probe"].coef_),
        "probe_intercept": (old["probe"].intercept_, new["probe"].intercept_),
        "probe_classes": (old["probe"].classes_, new["probe"].classes_),
    }
    arrays_equal = {name: bool(np.array_equal(np.asarray(left), np.asarray(right)))
                    for name, (left, right) in array_fields.items()}
    if not all(metadata_equal.values()) or not all(arrays_equal.values()):
        raise ValueError(f"{condition}: source-head fit population/parameters changed")
    return {"condition": condition, "status": "fit_population_and_coefficients_exact",
            "fit_population_identity_equal": True, "source_train_n": int(old["source_train_n"]),
            "source_train_patient_n": int(old["source_train_patient_n"]),
            "metadata_equal": metadata_equal, "parameter_arrays_equal": arrays_equal,
            "old_probe_sha256": old_hash, "new_probe_sha256": new_hash}


def verify_preview(metrics: dict[str, dict[str, float]]) -> dict:
    import pandas as pd

    if not PREVIEW.is_file():
        raise FileNotFoundError(PREVIEW)
    preview = pd.read_csv(PREVIEW)
    rows = {
        "repaired_real": preview.loc[(preview["version"] == "recovered_v4_2")
                                      & (preview["db"] == "eicu_semantic_v4_2")
                                      & (preview["analysis_id"] == "main20k_real")],
        "original_real": preview.loc[(preview["version"] == "original_core")
                                     & (preview["db"] == "eicu")
                                     & (preview["analysis_id"] == "main20k_real")],
    }
    expected_counts = {"repaired_real": (EXPECTED_ROWS, EXPECTED_PATIENTS, EXPECTED_EVENTS),
                       "original_real": (EXPECTED_ROWS, EXPECTED_PATIENTS, EXPECTED_EVENTS)}
    checks = {}
    for name, selected in rows.items():
        if len(selected) != 1:
            raise ValueError(f"preview does not contain one {name} row")
        row = selected.iloc[0]
        observed_counts = (int(row["n_stays"]), int(row["n_patients"]), int(row["events"]))
        if observed_counts != expected_counts[name]:
            raise ValueError(f"preview count mismatch for {name}: {observed_counts}")
        metric_checks = {}
        for metric in METRIC_NAMES:
            actual = float(metrics[name][metric])
            reference = float(row[metric])
            metric_checks[metric] = {"analysis": actual, "preview": reference,
                                     "absolute_difference": abs(actual - reference)}
            if not math.isclose(actual, reference, rel_tol=0, abs_tol=2e-12):
                raise ValueError(f"{name} {metric} does not match the saved preview")
        checks[name] = {"status": "point_metrics_match_saved_preview", "metrics": metric_checks}
    return {"preview_file": str(PREVIEW), "preview_sha256": sha256_file(PREVIEW), "checks": checks}


def bootstrap_contrasts(y: np.ndarray, subjects: np.ndarray,
                        probabilities: dict[str, np.ndarray], out_dir: Path,
                        reps: int = BOOTSTRAP_REPS, seed: int = SEED) -> tuple[dict, int]:
    unique, inverse = np.unique(np.asarray(subjects).astype(str), return_inverse=True)
    if len(unique) != EXPECTED_PATIENTS or reps != BOOTSTRAP_REPS:
        raise ValueError("unexpected patient count or bootstrap replicate count")
    groups = {name: _sorted_score_groups(p) for name, p in probabilities.items()}
    contrast_names = ("repaired_real_minus_original_real",
                      "repaired_constant_minus_original_constant",
                      "difference_in_differences")
    values = np.full((len(contrast_names), len(METRIC_NAMES), reps), np.nan, dtype=np.float64)
    rng = np.random.default_rng(seed)
    degenerate = 0
    progress_path = out_dir / "progress.json"
    checkpoint_path = out_dir / "bootstrap_checkpoint.npz"
    for draw in range(reps):
        patient_draws = rng.integers(0, len(unique), size=len(unique))
        patient_counts = np.bincount(patient_draws, minlength=len(unique))
        weights = patient_counts[inverse].astype(np.float64, copy=False)
        arm_metrics = {name: weighted_metrics(y, probability, weights, groups[name])
                       for name, probability in probabilities.items()}
        if any(not all(math.isfinite(arm_metrics[name][metric]) for metric in METRIC_NAMES)
               for name in probabilities):
            degenerate += 1
        else:
            contrast_values = paired_contrasts(arm_metrics)
            for c_index, contrast in enumerate(contrast_names):
                for m_index, metric in enumerate(METRIC_NAMES):
                    values[c_index, m_index, draw] = contrast_values[contrast][metric]
        done = draw + 1
        if done % 100 == 0:
            atomic_npz(checkpoint_path, completed_draws=np.asarray(done, dtype=np.int64),
                       contrast_names=np.asarray(contrast_names), metric_names=np.asarray(METRIC_NAMES),
                       samples=values[:, :, :done])
            atomic_json(progress_path, {"status": "running", "completed_draws": done,
                                       "requested_draws": reps, "valid_draws": done - degenerate,
                                       "degenerate_draws": degenerate, "seed": seed})
            print(f"BOOTSTRAP_PROGRESS draws={done}/{reps} valid={done-degenerate} degenerate={degenerate}",
                  flush=True)
    valid_n = reps - degenerate
    summary = {}
    for c_index, contrast in enumerate(contrast_names):
        summary[contrast] = {}
        for m_index, metric in enumerate(METRIC_NAMES):
            sample = values[c_index, m_index, :]
            sample = sample[np.isfinite(sample)]
            summary[contrast][metric] = {
                "point_estimate": None,
                "ci_95_percentile": ([float(np.quantile(sample, .025)), float(np.quantile(sample, .975))]
                                      if len(sample) else None),
                "valid_replicates": int(len(sample)),
                "degenerate_replicates": int(degenerate),
            }
    atomic_npz(checkpoint_path, completed_draws=np.asarray(reps, dtype=np.int64),
               contrast_names=np.asarray(contrast_names), metric_names=np.asarray(METRIC_NAMES), samples=values)
    atomic_json(progress_path, {"status": "complete", "completed_draws": reps,
                               "requested_draws": reps, "valid_draws": valid_n,
                               "degenerate_draws": degenerate, "seed": seed})
    return summary, degenerate


def run(output_dir: Path) -> None:
    import pandas as pd
    sys.path.insert(0, str(REV / "engine/code"))
    from paired_metrics_v4 import sha256_ordered_prediction_rows, validate_predictions

    script_dir = Path(__file__).resolve().parent
    copy_paths = {"old": script_dir / "old_prediction_manifest.json",
                  "new": script_dir / "new_prediction_manifest.json"}
    server_paths = {"old": OLD_SERVER_MANIFEST, "new": NEW_SERVER_MANIFEST}
    expected_manifests, manifest_hashes = {}, {}
    for name in ("old", "new"):
        copied_hash = sha256_file(copy_paths[name])
        server_hash = sha256_file(server_paths[name])
        if copied_hash != EXPECTED_MANIFEST_SHA256[name] or server_hash != copied_hash:
            raise ValueError(f"{name}: copied and server prediction manifests are not the bound versions")
        expected_manifests[name] = json.loads(copy_paths[name].read_text(encoding="utf-8"))
        manifest_hashes[name] = copied_hash
    old_manifest, new_manifest = expected_manifests["old"], expected_manifests["new"]
    if (old_manifest.get("run_id") != "status_consistent_primary_core_fix2"
            or new_manifest.get("run_id") != "recovered_external_primary_v4_2_resume3"):
        raise ValueError("unexpected old/new run IDs")
    entries = {}
    cache = {}
    frames = {}
    for arm, (version, db, analysis_id, condition) in ARM_SPECS.items():
        entry = get_manifest_entry(expected_manifests[version], analysis_id, db)
        if entry.get("input_condition") != condition or entry.get("analysis_id") != analysis_id:
            raise ValueError(f"{arm}: manifest condition mismatch")
        entries[arm] = entry
        frames[arm] = load_prediction(entry, cache, validate_predictions, sha256_ordered_prediction_rows)
    base, identity_summary, probabilities = verify_four_arm_eicu(frames)
    y = base["y"].to_numpy(dtype=np.int8)
    subjects = base["_subject_key"].astype(str).to_numpy()
    point_metrics = {name: weighted_metrics(y, probabilities[name], np.ones(len(y)))
                     for name in ARM_SPECS}
    preview_check = verify_preview(point_metrics)

    mimic_checks, probe_checks = [], []
    for condition, analysis_id in (("real", "main20k_real"), ("constant", "main20k_constant")):
        old_entry = get_manifest_entry(old_manifest, analysis_id, "mimic")
        new_entry = get_manifest_entry(new_manifest, analysis_id, "mimic")
        old_mimic = load_prediction(old_entry, cache, validate_predictions, sha256_ordered_prediction_rows)
        new_mimic = load_prediction(new_entry, cache, validate_predictions, sha256_ordered_prediction_rows)
        mimic_checks.append(compare_mimic(old_mimic, new_mimic, condition))
        old_eicu_entry = get_manifest_entry(old_manifest, analysis_id, "eicu")
        new_eicu_entry = get_manifest_entry(new_manifest, analysis_id, "eicu_semantic_v4_2")
        probe_checks.append(compare_probe_pair(old_eicu_entry, new_eicu_entry, condition))

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    boot_summary, degenerate = bootstrap_contrasts(y, subjects, probabilities, output_dir)
    point_contrasts = paired_contrasts(point_metrics)
    for contrast in boot_summary:
        for metric in METRIC_NAMES:
            boot_summary[contrast][metric]["point_estimate"] = float(point_contrasts[contrast][metric])
    point_rows = []
    for name, (version, db, analysis_id, condition) in ARM_SPECS.items():
        point_rows.append({"arm": name, "run_version": version, "db": db,
                           "analysis_id": analysis_id, "input_condition": condition,
                           "n_stays": len(y), "n_patients": identity_summary["patients"],
                           "events": int(y.sum()), **point_metrics[name]})
    pd.DataFrame(point_rows).to_csv(output_dir / "point_metrics.csv", index=False)
    contrast_rows = []
    for contrast, metrics in boot_summary.items():
        for metric, summary in metrics.items():
            ci = summary["ci_95_percentile"]
            contrast_rows.append({"contrast": contrast, "metric": metric,
                                  "point_estimate": summary["point_estimate"],
                                  "ci_95_low": ci[0] if ci else None,
                                  "ci_95_high": ci[1] if ci else None,
                                  "bootstrap_replicates": BOOTSTRAP_REPS,
                                  "valid_replicates": summary["valid_replicates"],
                                  "degenerate_replicates": summary["degenerate_replicates"],
                                  "resampling_unit": "patient", "weights_shared_across_all_four_arms": True})
    pd.DataFrame(contrast_rows).to_csv(output_dir / "paired_contrasts.csv", index=False)
    manifest = {
        "status": "complete",
        "analysis_type": "post_hoc_full_pipeline_input_version_descriptive_contrasts",
        "scope": "Saved prediction outputs only; no refitting, encoding, or raw clinical table access.",
        "interpretation_limit": ("The recovered input package jointly changes aliases, observation patterns, values, "
                                 "and token resampling. Contrasts do not identify causal effects or variable-specific "
                                 "mechanisms and do not support an information-fragmentation (IF) claim."),
        "prediction_manifests": {name: {"path": str(server_paths[name]), "sha256": manifest_hashes[name],
                                         "copied_manifest_path": str(copy_paths[name])}
                                 for name in ("old", "new")},
        "population_identity": identity_summary,
        "mimic_prediction_parity": mimic_checks,
        "source_head_fit_population_and_coefficients": probe_checks,
        "root_preview_verification": preview_check,
        "arms": {name: {"db": spec[1], "analysis_id": spec[2], "condition": spec[3],
                        "prediction_file": entries[name]["prediction_file"],
                        "prediction_file_sha256": entries[name]["prediction_file_sha256"],
                        "ordered_row_digest_sha256": entries[name]["ordered_row_digest_sha256"]}
                 for name, spec in ARM_SPECS.items()},
        "bootstrap": {"repetitions": BOOTSTRAP_REPS, "seed": SEED,
                      "unit": "patient", "shared_weights_across_all_four_arms": True,
                      "degenerate_replicates": degenerate,
                      "percentile_ci": [0.025, 0.975], "checkpoint_interval": 100,
                      "progress_path": str(output_dir / "progress.json"),
                      "aggregate_checkpoint_path": str(output_dir / "bootstrap_checkpoint.npz")},
        "point_metrics": point_metrics,
        "paired_contrasts": boot_summary,
        "output_contains_patient_level_rows_or_ids": False,
        "runtime": {"python": sys.version.split()[0], "implementation": platform.python_implementation(),
                    "numpy": np.__version__, "pandas": pd.__version__},
        "script_sha256": sha256_file(Path(__file__)),
    }
    atomic_json(output_dir / "analysis_manifest.json", manifest)
    print(json.dumps({"status": "complete", "output_dir": str(output_dir),
                      "population": identity_summary, "degenerate_replicates": degenerate,
                      "paired_contrasts": boot_summary}, allow_nan=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
