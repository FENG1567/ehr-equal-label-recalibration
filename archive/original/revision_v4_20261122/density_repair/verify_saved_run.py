#!/usr/bin/env python3
"""Audit run-3 provenance and compare saved frozen-probe metrics without fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
PROC = ROOT / "data/processed"
ELIGIBILITY = REV / "temporal/results/eligibility/status_consistent_v4_1/landmark_v4_1"
SNAPSHOT_SHA256 = "a636f5f38ba7376dcb632cc62ba20592413376da366f105711ce412b32d1fdc9"
MANIFEST_SHA256 = "ea2b55bc3cf071d48b019d4ba97b4a3dc53d5d7fb3f6ac49a9e7c26007064cb5"
TABLE_SHA256 = "c2774cd59e4df8706960e4fa58680ea6ec72b4af22adddf82b2389dc5b74749f"
SOURCE_SHA256 = "9bd20bf1394a34ba112a4de9a8d7e8971821bef90558a9987be4b9da5ae06824"
SPLIT_SHA256 = "ed5bd94933d22daaa4cd5fdff82ce48b36b72ea50d2db7e4642906036c043c1a"
CHECKPOINT_SHA256 = "0719db57b91b261869aed119cee95bf8980eb14411684c2040ba8588b45460bd"
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20261122 + 9000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ids(db: str, rows: np.ndarray, stay_id: np.ndarray,
               subject: np.ndarray) -> str:
    digest = hashlib.sha256(db.encode("utf-8"))
    digest.update(np.asarray(rows, dtype=np.int64).tobytes())
    for array in (np.asarray(stay_id).astype(str), np.asarray(subject).astype(str)):
        for value in array:
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    return digest.hexdigest()


def score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import log_loss, roc_auc_score

    y = np.asarray(y, dtype=np.int8)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return {"brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "auroc": float(roc_auc_score(y, p))}


def weighted_score(y: np.ndarray, p: np.ndarray,
                   weight: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=np.int8)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    weight = np.asarray(weight, dtype=np.float64)
    use = weight > 0
    if len(np.unique(y[use])) != 2:
        raise ValueError("patient resample did not contain both outcomes")
    total = float(weight.sum())
    return {
        "brier": float(np.sum(weight * (p - y) ** 2) / total),
        "log_loss": float(-np.sum(weight * (y * np.log(p) +
                                               (1 - y) * np.log1p(-p))) / total),
        "auroc": float(roc_auc_score(y, p, sample_weight=weight)),
    }


def paired_cluster_deltas(y: np.ndarray, groups: np.ndarray,
                          predictions: dict[str, np.ndarray]) -> dict:
    groups = np.asarray(groups).astype(str)
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons = ("ordinary_tuned", "original_ordinary_C1")
    metrics = ("brier", "log_loss", "auroc")
    values = {name: {metric: [] for metric in metrics} for name in comparisons}
    invalid = 0
    for _ in range(N_BOOTSTRAP):
        sampled = rng.integers(0, len(unique), size=len(unique))
        group_weights = np.bincount(sampled, minlength=len(unique))
        weights = group_weights[inverse]
        try:
            scores = {name: weighted_score(y, probability, weights)
                      for name, probability in predictions.items()}
        except ValueError:
            invalid += 1
            continue
        for comparator in comparisons:
            for metric in metrics:
                values[comparator][metric].append(
                    scores["density_augmented"][metric] - scores[comparator][metric])
    if invalid or any(len(values[name][metric]) != N_BOOTSTRAP
                      for name in comparisons for metric in metrics):
        raise RuntimeError(f"paired bootstrap had invalid draws: {invalid}")
    point = {name: {metric: score(y, predictions["density_augmented"])[metric]
                    - score(y, predictions[name])[metric] for metric in metrics}
             for name in comparisons}
    intervals = {}
    for name in comparisons:
        intervals[name] = {
            metric: {"point_difference_density_minus_comparator": point[name][metric],
                     "lower_95": float(np.quantile(values[name][metric], 0.025)),
                     "upper_95": float(np.quantile(values[name][metric], 0.975)),
                     "valid_draws": int(len(values[name][metric]))}
            for metric in metrics
        }
    return {"comparison_direction": "density_augmented minus comparator",
            "bootstrap_unit": "MIMIC patient/subject cluster",
            "draws": N_BOOTSTRAP, "shared_patient_draws": True,
            "seed": BOOTSTRAP_SEED, "patient_clusters": int(len(unique)),
            "invalid_draws": int(invalid), "comparisons": intervals}


def _assert_close(saved: dict, current: dict, label: str) -> None:
    for metric in ("brier", "log_loss", "auroc"):
        if not np.isclose(float(saved[metric]), float(current[metric]),
                          rtol=0.0, atol=1e-12):
            raise RuntimeError(f"frozen metric reconstruction mismatch: {label}.{metric}")


def run_audit(run_dir: Path, output_path: Path) -> dict:
    import pandas as pd
    import sklearn

    started = time.time()
    run_dir = Path(run_dir).resolve(strict=True)
    identity_path = run_dir / "input_identity.json"
    selection_path = run_dir / "selection.npz"
    view_path = run_dir / "view_embeddings.npz"
    lock_path = run_dir / "freeze_lock.json"
    fitted_path = run_dir / "fitted_probes.pkl"
    report_path = run_dir / "source_report.json"
    snapshot_path = run_dir / "embedding_manifest_core14_20261123.json"
    identity = json.loads(identity_path.read_text())
    lock = json.loads(lock_path.read_text())
    report = json.loads(report_path.read_text())
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite supplemental output {output_path}")

    expected_inputs = {
        "source_npz_sha256": (PROC / "seq_mimic.npz", SOURCE_SHA256),
        "split_sha256": (PROC / "split_mimic.npy", SPLIT_SHA256),
        "checkpoint_sha256": (ROOT / "models/main20k.pt", CHECKPOINT_SHA256),
        "fm_code_sha256": (ROOT / "code/fm.py", identity.get("fm_code_sha256")),
        "decoder_code_sha256": (ROOT / "revision_v2_20261117/code/decoding_motor_v2.py",
                                 identity.get("decoder_code_sha256")),
        "bins_sha256": (PROC / "bins.json", identity.get("bins_sha256")),
        "variable_metadata_sha256": (PROC / "meta_mimic.json",
                                      identity.get("variable_metadata_sha256")),
        "eligibility_manifest_sha256": (ELIGIBILITY / "eligibility_manifest.json",
                                         MANIFEST_SHA256),
        "eligibility_table_sha256": (ELIGIBILITY / "mimic.parquet", TABLE_SHA256),
        "frozen_embedding_manifest_sha256": (snapshot_path, SNAPSHOT_SHA256),
        "embedding_cache_sha256": (Path(identity["embedding_cache_path"]),
                                    identity["embedding_cache_sha256"]),
    }
    for key, (path, expected) in expected_inputs.items():
        actual = sha256_file(path)
        if actual != expected or identity.get(key) != expected:
            raise RuntimeError(f"provenance hash audit failed for {key}")
    cache_path = Path(identity["embedding_cache_path"]).resolve()
    if cache_path.parent != (REV / "engine/cache").resolve():
        raise RuntimeError("real embedding cache path is outside the immutable engine cache")

    frozen = json.loads(snapshot_path.read_text())
    cache_rows = [row for row in frozen["models"] if row.get("db") == "mimic"
                  and row.get("model_tag") == "main20k"
                  and row.get("input_condition") == "real"]
    if len(cache_rows) != 1:
        raise RuntimeError("frozen manifest has no unique MIMIC main20k real cache row")
    cache_record = cache_rows[0]
    with np.load(PROC / "seq_mimic.npz", allow_pickle=True) as z:
        source = {key: z[key] for key in ("V", "B", "H", "L", "stay_id",
                                           "subject", "y_mort")}
    split_all = np.load(PROC / "split_mimic.npy")
    source_stays = np.asarray(source["stay_id"]).astype(str)
    source_subjects = np.asarray(source["subject"]).astype(str)
    nonempty = np.flatnonzero(np.asarray(source["L"]) > 0).astype(np.int64)
    row_identity = sha256_ids("mimic", nonempty, source_stays[nonempty],
                              source_subjects[nonempty])
    if row_identity != cache_record["row_identity_sha256"]:
        raise RuntimeError("current source row identity differs from frozen cache manifest")
    with np.load(cache_path, allow_pickle=False) as cache:
        if (not np.array_equal(cache["rows"], nonempty)
                or not np.array_equal(cache["stay_id"].astype(str), source_stays[nonempty])
                or not np.array_equal(cache["subject"].astype(str), source_subjects[nonempty])
                or not np.array_equal(cache["source_y"],
                                      np.asarray(source["y_mort"], dtype=np.float32)[nonempty],
                                      equal_nan=True)):
            raise RuntimeError("current real cache row/stay/patient/label identity mismatch")

    frame = pd.read_parquet(ELIGIBILITY / "mimic.parquet")
    eligible = frame.loc[frame["eligible_24h"].astype(bool)
                         & frame["temporal_verified"].astype(bool)].copy()
    pos = {stay: i for i, stay in enumerate(source_stays)}
    eligible_rows = eligible["stay_id"].astype(str).map(pos).to_numpy(dtype=np.int64)
    order = np.argsort(eligible_rows)
    eligible_rows = eligible_rows[order]
    eligible_subject = eligible["subject"].astype(str).to_numpy()[order]
    eligible_y = eligible["y_mort"].to_numpy(dtype=np.int8)[order]
    if len(eligible_rows) != 73936 or int(eligible_y.sum()) != 8143:
        raise RuntimeError("current corrected MIMIC gate population differs from clearance")

    if sha256_file(selection_path) != identity["selection_npz_sha256"]:
        raise RuntimeError("run-3 selection hash mismatch")
    if sha256_file(view_path) != report["artifacts"]["view_embeddings_sha256"]:
        raise RuntimeError("run-3 view embedding hash mismatch")
    if lock.get("provenance") != identity:
        raise RuntimeError("freeze lock provenance differs from run-3 prepared identity")
    if (lock.get("external_outcomes_or_INSPIRE_scores_read") is not False
            or report.get("source_scope") != "MIMIC only"):
        raise RuntimeError("run-3 report is not certified as source-only")
    code_hashes = {
        "fit_readout_sha256": run_dir / "fit_readout.py",
        "encode_views_sha256": run_dir / "encode_views.py",
        "density_repair_helpers_sha256": run_dir / "density_repair.py",
        "launcher_sha256": run_dir / "run_density_repair.sh",
        "fm_code_sha256": ROOT / "code/fm.py",
        "decoder_code_sha256": ROOT / "revision_v2_20261117/code/decoding_motor_v2.py",
    }
    for key, path in code_hashes.items():
        if sha256_file(path) != lock["provenance_hashes"][key]:
            raise RuntimeError(f"run-3 code hash differs from freeze lock: {key}")
    expected_out = 'density_repair/run_source_20261123_03'
    for script in (run_dir / "fit_readout.py", run_dir / "encode_views.py",
                   run_dir / "run_density_repair.sh"):
        if expected_out not in script.read_text():
            raise RuntimeError(f"run-3 stage does not bind its own shared path: {script.name}")

    with np.load(selection_path, allow_pickle=False) as selection:
        rows = np.asarray(selection["rows"], dtype=np.int64)
        stays = np.asarray(selection["stay_id"]).astype(str)
        subjects = np.asarray(selection["subject"]).astype(str)
        y = np.asarray(selection["y"], dtype=np.int8)
        split = np.asarray(selection["split"], dtype=np.int8)
        token_count = np.asarray(selection["token_count"], dtype=np.int32)
        real = np.asarray(selection["real"], dtype=np.float32)
    if (not np.array_equal(rows, eligible_rows)
            or not np.array_equal(stays, source_stays[rows])
            or not np.array_equal(subjects, eligible_subject)
            or not np.array_equal(y, eligible_y)
            or not np.array_equal(split, split_all[rows].astype(np.int8))):
        raise RuntimeError("run-3 selection does not match corrected gate and frozen source rows")
    if hashlib.sha256(rows.tobytes()).hexdigest() != identity["selected_rows_sha256"]:
        raise RuntimeError("selected source row digest mismatch")
    with np.load(view_path, allow_pickle=False) as views:
        if not np.array_equal(views["rows"], rows):
            raise RuntimeError("view embeddings do not use the same source row ordering")
        train_aug = np.asarray(views["train_aug"], dtype=np.float32)
        val_aug = np.asarray(views["val_aug"], dtype=np.float32)
        test_ret = np.asarray(views["test_ret"], dtype=np.float32)
        kept_test = np.asarray(views["kept_test"], dtype=np.int32)
        gpu_runtime = json.loads(str(views["gpu_runtime_json"].item()))
    n_train, n_val, n_test = (int(np.sum(split == i)) for i in (0, 1, 2))
    embedding_dim = int(real.shape[1])
    if (train_aug.shape != (3, n_train, embedding_dim)
            or val_aug.shape != (3, n_val, embedding_dim)
            or test_ret.shape != (3, n_test, embedding_dim)
            or kept_test.shape != (3, n_test)
            or not all(np.isfinite(array).all()
                       for array in (real, train_aug, val_aug, test_ret))):
        raise RuntimeError("saved view embeddings do not align to the frozen split rows")

    with fitted_path.open("rb") as handle:
        fitted = pickle.load(handle)
    if fitted["freeze_lock_sha256"] != sha256_file(lock_path):
        raise RuntimeError("fitted probes are not bound to the validation freeze lock")
    if fitted["provenance_hashes"] != lock["provenance_hashes"]:
        raise RuntimeError("fitted probes and lock have different code provenance")
    candidates = ("density_augmented", "ordinary_tuned", "original_ordinary_C1")
    test = split == 2
    y_test = y[test]
    groups = subjects[test]
    X_test = real[test]
    predictions = {}
    for name in candidates:
        artifact = fitted[name]
        if float(artifact["C"]) != float(lock["validation_summary"][name]["selected_C"]):
            raise RuntimeError(f"fitted {name} C differs from freeze lock")
        predictions[name] = artifact["probe"].predict_proba(
            artifact["scaler"].transform(X_test))[:, 1]
    reconstructed = {name: score(y_test, predictions[name]) for name in candidates}
    for name in candidates:
        _assert_close(report["report_split_full_input_metrics"][name],
                      reconstructed[name], name)

    paired = paired_cluster_deltas(y_test, groups, predictions)
    retention_rows = [{"retention": 1.0, "kept_counts": token_count[test]}]
    for i, fraction in enumerate((0.75, 0.5, 0.25)):
        retention_rows.append({"retention": fraction,
                               "kept_counts": kept_test[i],
                               "embeddings": test_ret[i]})
    retention = {}
    original_counts = token_count[test]
    for name in candidates:
        artifact = fitted[name]
        full_p = predictions[name]
        full_clipped = np.clip(full_p, 1e-7, 1-1e-7)
        full_logit = np.log(full_clipped / (1-full_clipped))
        model_rows = []
        for item in retention_rows:
            fraction = float(item["retention"])
            if fraction == 1.0:
                probability = full_p
                kept = np.asarray(item["kept_counts"], dtype=np.int32)
            else:
                index = {0.75: 0, 0.5: 1, 0.25: 2}[fraction]
                probability = artifact["probe"].predict_proba(
                    artifact["scaler"].transform(item["embeddings"]))[:, 1]
                kept = np.asarray(item["kept_counts"], dtype=np.int32)
            clipped = np.clip(probability, 1e-7, 1-1e-7)
            logits = np.log(clipped / (1-clipped))
            usable = kept > 0
            event = y_test == 1
            item_out = {
                "retention": fraction,
                "usable_stays": int(usable.sum()),
                "usable_fraction": float(usable.mean()),
                "event_stays_with_any_token": int(np.sum(event & usable)),
                "event_retention_fraction": float(np.sum(event & usable) / event.sum()),
                "retained_token_fraction_all": float(kept.sum() / original_counts.sum()),
                "retained_token_fraction_events": float(kept[event].sum() /
                                                          original_counts[event].sum()),
                "mean_absolute_probability_change_vs_full":
                    float(np.mean(np.abs(probability - full_p))),
                "mean_absolute_logit_change_vs_full":
                    float(np.mean(np.abs(logits - full_logit))),
                **score(y_test, probability),
            }
            if name == "density_augmented":
                _assert_close(report["controlled_retention_density_candidate"][
                    int(round((1.0-fraction)/0.25))], item_out, f"retention {fraction}")
            model_rows.append(item_out)
        retention[name] = model_rows

    result = {
        "audit_status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(run_dir),
        "source_scope": "MIMIC only; no external cohort outcomes or INSPIRE scores read",
        "method": "recomputed prediction metrics from saved split-0 probes and split-2 embeddings; no model fitting, encoder training, or hyperparameter selection",
        "run3_source_job": {"pid": 1088143,
                            "log": str(REV / "density_repair/logs/density_repair_source_20261123_run3.log"),
                            "exit_code": 0},
        "gate_and_identity": {
            "eligibility_manifest_sha256": MANIFEST_SHA256,
            "eligibility_table_sha256": TABLE_SHA256,
            "source_npz_sha256": SOURCE_SHA256,
            "split_sha256": SPLIT_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "bin_file_sha256": identity["bins_sha256"],
            "variable_metadata_sha256": identity["variable_metadata_sha256"],
            "embedding_cache_sha256": identity["embedding_cache_sha256"],
            "embedding_cache_row_identity_sha256": row_identity,
            "selected_stays": int(len(rows)), "selected_events": int(y.sum()),
            "split_counts": {str(i): int(np.sum(split == i)) for i in (0, 1, 2)},
            "report_patients": int(len(np.unique(groups)),),
        },
        "validation_lock": {
            "path": str(lock_path), "sha256": sha256_file(lock_path),
            "locked_at_utc": lock["locked_at_utc"],
            "base_seed": lock["base_seed"], "c_grid": lock["c_grid"],
            "selection_objective": lock["selection_objective"],
            "validation_summary": lock["validation_summary"],
            "per_C_grid_objectives_recorded": False,
        },
        "stage_paths_and_hashes": {
            "input_identity_path": str(identity_path),
            "input_identity_sha256": sha256_file(identity_path),
            "selection_path": str(selection_path),
            "selection_sha256": sha256_file(selection_path),
            "view_embeddings_path": str(view_path),
            "view_embeddings_sha256": sha256_file(view_path),
            "fitted_probes_path": str(fitted_path),
            "fitted_probes_sha256": sha256_file(fitted_path),
            "source_report_path": str(report_path),
            "source_report_sha256": sha256_file(report_path),
            "code_hashes": lock["provenance_hashes"],
        },
        "reproduced_report_split_metrics_from_saved_probes": reconstructed,
        "saved_marginal_patient_cluster_cis": report["report_split_full_input_metrics"],
        "paired_patient_cluster_deltas": paired,
        "retention_metrics_all_saved_probes": retention,
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__,
                    "scikit_learn": sklearn.__version__, "platform": platform.platform(),
                    "saved_gpu_view_encoder_runtime": gpu_runtime,
                    "audit_elapsed_seconds": round(time.time() - started, 3),
                    "max_rss_kb": int(__import__("resource").getrusage(
                        __import__("resource").RUSAGE_SELF).ru_maxrss)},
        "interpretation_limit": "The density-augmented probe remains the validation-frozen exploratory candidate. Source split-2 results were already observed; this audit adds no candidate selection or external validation claim.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    result["supplemental_output_sha256"] = sha256_file(output_path)
    print("SAVED_RUN_AUDIT_PASS", json.dumps({
        "output": str(output_path),
        "sha256": result["supplemental_output_sha256"],
        "paired": paired,
        "retention_models": list(retention),
        "reproduced_metrics": reconstructed,
    }, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_audit(args.run_dir, args.output)


if __name__ == "__main__":
    main()
