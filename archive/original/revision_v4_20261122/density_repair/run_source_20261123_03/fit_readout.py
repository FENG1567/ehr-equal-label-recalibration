#!/usr/bin/env python3
"""Prepare and fit the MIMIC-only density-augmented frozen-encoder readout."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from density_repair import (BASE_SEED, equal_per_stay_view_weights,
                            fit_training_scaler, partition_masks,
                            select_min_validation, write_selection_npz)

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
OUT = REV / "density_repair/run_source_20261123_03"
PROC = ROOT / "data/processed"
ELIGIBILITY = REV / "temporal/results/eligibility/status_consistent_v4_1/landmark_v4_1"
SNAPSHOT = OUT / "embedding_manifest_core14_20261123.json"
CHECKPOINT = ROOT / "models/main20k.pt"
FM_CODE = ROOT / "code/fm.py"
DECODER_CODE = ROOT / "revision_v2_20261117/code/decoding_motor_v2.py"
SOURCE_NPZ = PROC / "seq_mimic.npz"
SPLIT_NPY = PROC / "split_mimic.npy"
BIN_FILE = PROC / "bins.json"
META_FILE = PROC / "meta_mimic.json"
MANIFEST_EXPECTED = "ea2b55bc3cf071d48b019d4ba97b4a3dc53d5d7fb3f6ac49a9e7c26007064cb5"
TABLE_EXPECTED = "c2774cd59e4df8706960e4fa58680ea6ec72b4af22adddf82b2389dc5b74749f"
SNAPSHOT_EXPECTED = "a636f5f38ba7376dcb632cc62ba20592413376da366f105711ce412b32d1fdc9"
SOURCE_EXPECTED = "9bd20bf1394a34ba112a4de9a8d7e8971821bef90558a9987be4b9da5ae06824"
SPLIT_EXPECTED = "ed5bd94933d22daaa4cd5fdff82ce48b36b72ea50d2db7e4642906036c043c1a"
CHECKPOINT_EXPECTED = "0719db57b91b261869aed119cee95bf8980eb14411684c2040ba8588b45460bd"
ENCODER_EXPECTED = "59c3f4059662fb9e1d81baf20120c8668a761a926f5cab5ff4e654f7c16c1abc"
C_GRID = (0.01, 0.1, 1.0, 10.0)
RETENTION_GRID = (0.75, 0.5, 0.25)
N_BOOTSTRAP = 1000


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


def _source_arrays() -> dict[str, np.ndarray]:
    if sha256_file(SOURCE_NPZ) != SOURCE_EXPECTED:
        raise RuntimeError("MIMIC source NPZ hash differs from the cleared source")
    if sha256_file(SPLIT_NPY) != SPLIT_EXPECTED:
        raise RuntimeError("MIMIC split hash differs from the frozen split")
    with np.load(SOURCE_NPZ, allow_pickle=True) as source:
        data = {key: source[key] for key in
                ("V", "B", "H", "L", "stay_id", "subject", "y_mort")}
    data["split"] = np.load(SPLIT_NPY)
    n = len(data["L"])
    if any(len(data[key]) != n for key in
           ("V", "B", "H", "stay_id", "subject", "y_mort", "split")):
        raise RuntimeError("sequence arrays have inconsistent row counts")
    if len(np.unique(np.asarray(data["stay_id"]).astype(str))) != n:
        raise RuntimeError("source stay IDs are not unique")
    return data


def prepare() -> None:
    import pandas as pd

    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    identity_path = OUT / "input_identity.json"
    selection_path = OUT / "selection.npz"
    if identity_path.exists() or selection_path.exists():
        raise FileExistsError("this density_repair run directory already contains preparation outputs")
    if sha256_file(SNAPSHOT) != SNAPSHOT_EXPECTED:
        raise RuntimeError("frozen core14 manifest snapshot hash mismatch")
    if sha256_file(ELIGIBILITY / "eligibility_manifest.json") != MANIFEST_EXPECTED:
        raise RuntimeError("cleared eligibility manifest hash mismatch")
    table_path = ELIGIBILITY / "mimic.parquet"
    if sha256_file(table_path) != TABLE_EXPECTED:
        raise RuntimeError("cleared MIMIC eligibility table hash mismatch")

    top = json.loads(SNAPSHOT.read_text())
    matches = [row for row in top.get("models", []) if
               row.get("db") == "mimic" and row.get("model_tag") == "main20k"
               and row.get("input_condition") == "real"]
    if len(matches) != 1:
        raise RuntimeError("immutable core14 manifest lacks one MIMIC/main20k/real row")
    cache_record = matches[0]
    expected_record = {
        "source_npz_sha256": SOURCE_EXPECTED,
        "split_sha256": SPLIT_EXPECTED,
        "checkpoint_sha256": CHECKPOINT_EXPECTED,
        "encoder_code_sha256": ENCODER_EXPECTED,
        "input_mask_sha256": "3f453e2f4b91e5d9ec42871d91461861c3ba583797b64479ff1384150ca01421",
        "row_identity_sha256": "009c81e89cf7b3ad50f067da2e650546a39672eef3165c72e6676cd92f903f87",
    }
    for key, expected in expected_record.items():
        if cache_record.get(key) != expected:
            raise RuntimeError(f"frozen cache manifest mismatch for {key}")
    cache_path = Path(cache_record["cache"])
    if not cache_path.is_file() or cache_path.parent != REV / "engine/cache":
        raise RuntimeError(f"expected read-only real embedding cache missing: {cache_path}")

    data = _source_arrays()
    source_stays = np.asarray(data["stay_id"]).astype(str)
    source_subjects = np.asarray(data["subject"]).astype(str)
    nonempty_rows = np.flatnonzero(np.asarray(data["L"]) > 0).astype(np.int64)
    computed_identity = sha256_ids("mimic", nonempty_rows,
                                   source_stays[nonempty_rows],
                                   source_subjects[nonempty_rows])
    if computed_identity != cache_record["row_identity_sha256"]:
        raise RuntimeError("source row identity hash does not match the immutable cache manifest")

    with np.load(cache_path, allow_pickle=False) as cache:
        cache_rows = np.asarray(cache["rows"], dtype=np.int64)
        cache_stays = np.asarray(cache["stay_id"]).astype(str)
        cache_subjects = np.asarray(cache["subject"]).astype(str)
        cache_y = np.asarray(cache["source_y"], dtype=np.float32)
        embeddings = np.asarray(cache["embedding"], dtype=np.float32)
        cache_metadata = json.loads(str(cache["metadata_json"].item()))
    if (not np.array_equal(cache_rows, nonempty_rows)
            or not np.array_equal(cache_stays, source_stays[nonempty_rows])
            or not np.array_equal(cache_subjects, source_subjects[nonempty_rows])
            or not np.array_equal(cache_y,
                                  np.asarray(data["y_mort"], dtype=np.float32)[nonempty_rows],
                                  equal_nan=True)
            or embeddings.ndim != 2 or len(embeddings) != len(nonempty_rows)
            or not np.isfinite(embeddings).all()):
        raise RuntimeError("real embedding cache row identity or values failed validation")
    runtime = cache_record["runtime_identity"]
    required_meta = {
        "db": "mimic", "model_tag": "main20k", "input_condition": "real",
        "input_scope": top["input_scope"],
        "clinical_input_availability_verified": top["clinical_input_availability_verified"],
        "input_availability_limitation": top["input_availability_limitation"],
        "checkpoint_sha256": CHECKPOINT_EXPECTED,
        "input_mask_sha256": expected_record["input_mask_sha256"],
        "row_identity_sha256": computed_identity,
        "source_npz_sha256": SOURCE_EXPECTED,
        "split_sha256": SPLIT_EXPECTED,
        "encoder_code_sha256": ENCODER_EXPECTED,
        "runtime_identity": runtime,
        "runtime_identity_sha256": cache_record["runtime_identity_sha256"],
        "python_version": runtime["python_version"],
        "torch_version": runtime["torch_version"],
        "encoding_device": "cuda", "encoding_dtype": "float32",
        "model_parameter_dtypes": ["torch.float32"], "torch_num_threads": 8,
        "encoding_scope": "all nonempty original sequence rows",
        "excluded_empty_input_mask_rows": 0,
    }
    if top.get("engine_code_sha256"):
        required_meta["engine_code_sha256"] = top["engine_code_sha256"]
    mismatches = [key for key, value in required_meta.items()
                  if cache_metadata.get(key) != value]
    if mismatches:
        raise RuntimeError(f"embedding cache metadata mismatch: {mismatches}")

    frame = pd.read_parquet(table_path)
    required_columns = {"db", "stay_id", "subject", "eligible_24h",
                        "temporal_verified", "y_mort", "endpoint"}
    if not required_columns.issubset(frame.columns):
        raise RuntimeError("eligibility table is missing required audited columns")
    if frame["db"].astype(str).ne("mimic").any() or frame["stay_id"].astype(str).duplicated().any():
        raise RuntimeError("eligibility table has wrong cohort or duplicate stays")
    selected = frame.loc[frame["eligible_24h"].astype(bool)
                         & frame["temporal_verified"].astype(bool)].copy()
    if len(selected) != 73936 or int(selected["y_mort"].sum()) != 8143:
        raise RuntimeError("MIMIC selected population disagrees with root clearance")
    pos = {stay: i for i, stay in enumerate(source_stays)}
    selected_rows = selected["stay_id"].astype(str).map(pos)
    if selected_rows.isna().any():
        raise RuntimeError("cleared eligible stay is absent from sequence NPZ")
    selected_rows = selected_rows.to_numpy(dtype=np.int64)
    selected_subjects = selected["subject"].astype(str).to_numpy()
    selected_y = selected["y_mort"].to_numpy(dtype=np.int8)
    if (not np.array_equal(source_subjects[selected_rows], selected_subjects)
            or not np.array_equal(np.asarray(data["y_mort"], dtype=np.int8)[selected_rows], selected_y)
            or not np.isin(selected_y, (0, 1)).all()):
        raise RuntimeError("eligibility stay, patient, or outcome identity mismatch")
    selected_split = np.asarray(data["split"], dtype=np.int8)[selected_rows]
    order = np.argsort(selected_rows)
    selected_rows = selected_rows[order]
    selected_subjects = np.asarray(selected_subjects[order], dtype=str)
    selected_y = selected_y[order]
    selected_split = selected_split[order]
    selected_stays = np.asarray(source_stays[selected_rows], dtype=str)
    if not np.array_equal(np.unique(selected_split), np.array([0, 1, 2], dtype=np.int8)):
        raise RuntimeError("eligible MIMIC rows do not cover frozen splits 0, 1, and 2")
    for split_id in (0, 1, 2):
        take = selected_split == split_id
        if len(np.unique(selected_y[take])) != 2:
            raise RuntimeError(f"eligible split {split_id} lacks one outcome class")
    groups_by_split = {split_id: set(selected_subjects[selected_split == split_id])
                       for split_id in (0, 1, 2)}
    if any(groups_by_split[a] & groups_by_split[b]
           for a, b in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("source patient appears across the frozen MIMIC splits")
    cache_pos = {int(row): i for i, row in enumerate(cache_rows)}
    cache_take = np.asarray([cache_pos[int(row)] for row in selected_rows], dtype=np.int64)
    selected_real = embeddings[cache_take]
    token_counts = np.asarray([
        int(np.count_nonzero(np.asarray(data["V"])[row, :int(data["L"][row])] != 0))
        for row in selected_rows
    ], dtype=np.int32)
    if np.any(token_counts < 1):
        raise RuntimeError("selected eligible stay has no usable token")
    write_selection_npz(selection_path, rows=selected_rows, stay_id=selected_stays,
                        subject=selected_subjects, y=selected_y, split=selected_split,
                        token_count=token_counts, real=selected_real)
    identities = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "MIMIC only; split 0 train, split 1 tune, split 2 report",
        "external_outcomes_or_INSPIRE_scores_read": False,
        "source_npz_sha256": SOURCE_EXPECTED,
        "split_sha256": SPLIT_EXPECTED,
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "checkpoint_sha256_expected": CHECKPOINT_EXPECTED,
        "fm_code_sha256": sha256_file(FM_CODE),
        "decoder_code_sha256": sha256_file(DECODER_CODE),
        "bins_sha256": sha256_file(BIN_FILE),
        "variable_metadata_sha256": sha256_file(META_FILE),
        "frozen_embedding_manifest_sha256": SNAPSHOT_EXPECTED,
        "embedding_cache_path": str(cache_path),
        "embedding_cache_sha256": sha256_file(cache_path),
        "embedding_cache_row_identity_sha256": computed_identity,
        "embedding_cache_runtime_identity": runtime,
        "embedding_cache_runtime_identity_sha256": cache_record["runtime_identity_sha256"],
        "eligibility_manifest_sha256": MANIFEST_EXPECTED,
        "eligibility_table_sha256": TABLE_EXPECTED,
        "selected_stays": int(len(selected_rows)),
        "selected_events": int(selected_y.sum()),
        "split_counts": {str(i): int(np.sum(selected_split == i)) for i in (0, 1, 2)},
        "selected_rows_sha256": hashlib.sha256(selected_rows.tobytes()).hexdigest(),
        "selection_npz_sha256": sha256_file(selection_path),
        "preparation_runtime": {"python": sys.version.split()[0],
                                "numpy": np.__version__, "pandas": pd.__version__},
        "preparation_elapsed_seconds": round(time.time() - start, 3),
        "retrospective_input_scope": top.get("input_scope"),
        "clinical_input_availability_verified": top.get("clinical_input_availability_verified"),
        "input_availability_limitation": top.get("input_availability_limitation"),
    }
    temporary_identity = identity_path.with_suffix(".json.tmp")
    temporary_identity.write_text(json.dumps(identities, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_identity, identity_path)
    print("PREPARE_PASS", json.dumps({k: identities[k] for k in
          ("selected_stays", "selected_events", "split_counts", "embedding_cache_sha256")},
          sort_keys=True), flush=True)


def _scores(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import log_loss, roc_auc_score

    y = np.asarray(y, dtype=np.int8)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    if len(np.unique(y)) != 2:
        raise ValueError("risk metric cohort must contain both outcome classes")
    return {"brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "auroc": float(roc_auc_score(y, p))}


def _weighted_scores(y: np.ndarray, p: np.ndarray,
                     weights: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=np.int8)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    w = np.asarray(weights, dtype=np.float64)
    positive = w > 0
    if not positive.any() or len(np.unique(y[positive])) != 2:
        raise ValueError("bootstrap draw lacks both outcome classes")
    denom = float(w.sum())
    return {
        "brier": float(np.sum(w * (p - y) ** 2) / denom),
        "log_loss": float(-np.sum(w * (y * np.log(p) + (1-y) * np.log1p(-p))) / denom),
        "auroc": float(roc_auc_score(y, p, sample_weight=w)),
    }


def _cluster_ci(y: np.ndarray, p: np.ndarray, groups: np.ndarray,
                draws: int, seed: int) -> dict[str, dict[str, float | int]]:
    y = np.asarray(y, dtype=np.int8)
    groups = np.asarray(groups).astype(str)
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    values = {name: [] for name in ("brier", "log_loss", "auroc")}
    invalid = 0
    for _ in range(int(draws)):
        sampled = rng.integers(0, len(unique_groups), size=len(unique_groups))
        counts = np.bincount(sampled, minlength=len(unique_groups))
        weight = counts[inverse]
        try:
            score = _weighted_scores(y, p, weight)
        except ValueError:
            invalid += 1
            continue
        for name in values:
            values[name].append(score[name])
    output = {}
    for name, draws_values in values.items():
        if len(draws_values) < int(draws) * 0.95:
            raise RuntimeError(f"too few valid patient-cluster bootstrap draws for {name}")
        output[name] = {"lower_95": float(np.quantile(draws_values, 0.025)),
                        "upper_95": float(np.quantile(draws_values, 0.975)),
                        "valid_draws": int(len(draws_values)),
                        "invalid_class_draws": int(invalid)}
    return output


def _fit_probe(X: np.ndarray, y: np.ndarray, weights: np.ndarray,
               C: float):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=float(C), max_iter=3000, random_state=BASE_SEED,
                               solver="lbfgs")
    model.fit(X, y, sample_weight=weights)
    if not model.n_iter_.max() < model.max_iter:
        raise RuntimeError(f"logistic fit failed to converge for C={C}")
    return model


def fit() -> None:
    import sklearn

    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    identity_path = OUT / "input_identity.json"
    selection_path = OUT / "selection.npz"
    embedding_path = OUT / "view_embeddings.npz"
    lock_path = OUT / "freeze_lock.json"
    probe_path = OUT / "fitted_probes.pkl"
    report_path = OUT / "source_report.json"
    if not identity_path.is_file() or not selection_path.is_file() or not embedding_path.is_file():
        raise FileNotFoundError("prepare and GPU encoding stages must complete before fit")
    if lock_path.exists() or probe_path.exists() or report_path.exists():
        raise FileExistsError("density repair fit outputs already exist; refusing overwrite")
    identity = json.loads(identity_path.read_text())
    if sha256_file(selection_path) != identity["selection_npz_sha256"]:
        raise RuntimeError("selection artifact hash changed after preparation")
    if identity.get("external_outcomes_or_INSPIRE_scores_read") is not False:
        raise RuntimeError("external outcome use is not allowed for source development")
    with np.load(selection_path, allow_pickle=False) as selection:
        rows = np.asarray(selection["rows"], dtype=np.int64)
        stay_id = np.asarray(selection["stay_id"]).astype(str)
        subject = np.asarray(selection["subject"]).astype(str)
        y = np.asarray(selection["y"], dtype=np.int8)
        split = np.asarray(selection["split"], dtype=np.int8)
        token_count = np.asarray(selection["token_count"], dtype=np.int32)
        real = np.asarray(selection["real"], dtype=np.float32)
    if sha256_file(embedding_path) == "":
        raise RuntimeError("failed to hash encoded view artifact")
    with np.load(embedding_path, allow_pickle=False) as encoded:
        encoded_rows = np.asarray(encoded["rows"], dtype=np.int64)
        train_aug = np.asarray(encoded["train_aug"], dtype=np.float32)
        val_aug = np.asarray(encoded["val_aug"], dtype=np.float32)
        test_ret = np.asarray(encoded["test_ret"], dtype=np.float32)
        kept_train = np.asarray(encoded["kept_train"], dtype=np.int32)
        kept_val = np.asarray(encoded["kept_val"], dtype=np.int32)
        kept_test = np.asarray(encoded["kept_test"], dtype=np.int32)
        gpu_runtime = json.loads(str(encoded["gpu_runtime_json"].item()))
    if not np.array_equal(rows, encoded_rows):
        raise RuntimeError("encoded rows differ from validated selection rows")
    train, valid, report = partition_masks(split)
    masks = {0: train, 1: valid, 2: report}
    n_train, n_val, n_test = (int(masks[i].sum()) for i in (0, 1, 2))
    d = real.shape[1]
    if (train_aug.shape != (3, n_train, d) or val_aug.shape != (3, n_val, d)
            or test_ret.shape != (3, n_test, d)
            or kept_train.shape != (3, n_train) or kept_val.shape != (3, n_val)
            or kept_test.shape != (3, n_test)):
        raise RuntimeError("encoded view matrices have unexpected shapes")
    if not all(np.isfinite(value).all() for value in (real, train_aug, val_aug, test_ret)):
        raise RuntimeError("nonfinite embeddings in readout inputs")
    y_train, y_val = y[train], y[valid]
    real_train, real_val = real[train], real[valid]
    val_views = [real_val] + [val_aug[i] for i in range(3)]
    aug_train_views = [real_train] + [train_aug[i] for i in range(3)]
    aug_train = np.concatenate(aug_train_views, axis=0)
    aug_y = np.tile(y_train, 4)
    aug_weights = equal_per_stay_view_weights(n_train, 4)
    if not np.allclose(aug_weights.reshape(4, n_train).sum(axis=0), 1.0):
        raise RuntimeError("training view weights do not sum to one per stay")
    aug_scaler = fit_training_scaler(
        aug_train, np.ones(len(aug_train), dtype=bool), aug_weights)
    ordinary_scaler = fit_training_scaler(real, train)
    aug_train_scaled = aug_scaler.transform(aug_train)
    ordinary_train_scaled = ordinary_scaler.transform(real_train)
    aug_val_scaled = [aug_scaler.transform(view) for view in val_views]
    ordinary_val_scaled = [ordinary_scaler.transform(view) for view in val_views]

    candidates: dict[str, dict[float, dict]] = {"density_augmented": {}, "ordinary_tuned": {}}
    for C in C_GRID:
        for name, Xfit, yfit, fit_weights, validation_scaled in (
            ("density_augmented", aug_train_scaled, aug_y, aug_weights, aug_val_scaled),
            ("ordinary_tuned", ordinary_train_scaled, y_train,
             np.ones(n_train, dtype=np.float64), ordinary_val_scaled),
        ):
            model = _fit_probe(Xfit, yfit, fit_weights, C)
            briers = []
            val_scores = []
            for Xval in validation_scaled:
                probability = model.predict_proba(Xval)[:, 1]
                metrics = _scores(y_val, probability)
                briers.append(metrics["brier"])
                val_scores.append(metrics)
            candidates[name][C] = {"model": model, "mean_validation_brier": float(np.mean(briers)),
                                   "validation_brier_by_view": briers,
                                   "validation_metrics_by_view": val_scores}
    selected_C = select_min_validation(
        C_GRID, {C: candidates["density_augmented"][C]["mean_validation_brier"]
                 for C in C_GRID})
    ordinary_C = select_min_validation(
        C_GRID, {C: candidates["ordinary_tuned"][C]["mean_validation_brier"]
                 for C in C_GRID})
    original_model = candidates["ordinary_tuned"][1.0]["model"]
    density_model = candidates["density_augmented"][selected_C]["model"]
    ordinary_tuned_model = candidates["ordinary_tuned"][ordinary_C]["model"]
    validation_summary = {}
    for name, selected_c in (("density_augmented", selected_C),
                             ("ordinary_tuned", ordinary_C),
                             ("original_ordinary_C1", 1.0)):
        record = candidates["density_augmented" if name == "density_augmented"
                            else "ordinary_tuned"][selected_c]
        validation_summary[name] = {
            "selected_C": float(selected_c),
            "mean_validation_brier_real_and_three_heldout_50pct_views":
                record["mean_validation_brier"],
            "validation_brier_by_view_real_then_three_heldout_50pct":
                record["validation_brier_by_view"],
            "full_real_validation_brier": record["validation_brier_by_view"][0],
        }
    code_hashes = {
        "density_repair_helpers_sha256": sha256_file(Path(__file__).with_name("density_repair.py")),
        "fit_readout_sha256": sha256_file(Path(__file__)),
        "encode_views_sha256": sha256_file(OUT / "encode_views.py"),
        "launcher_sha256": sha256_file(OUT / "run_density_repair.sh"),
        "fm_code_sha256": identity["fm_code_sha256"],
        "decoder_code_sha256": identity["decoder_code_sha256"],
    }
    lock = {
        "lock_id": "mimic_main20k_density_augmented_readout_v1",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_scope": "MIMIC source only; split 0 train, split 1 tune, split 2 report",
        "external_outcomes_or_INSPIRE_scores_read": False,
        "candidate_recipe": "real plus three deterministic outcome-independent 50% token-retention views; per-stay total training weight one",
        "base_seed": int(BASE_SEED),
        "seed_namespaces": ["train_aug", "validation_heldout", "report_retention"],
        "c_grid": list(C_GRID),
        "density_augmented_selected_C": float(selected_C),
        "ordinary_tuned_selected_C": float(ordinary_C),
        "selection_objective": "minimum mean validation Brier across real and three held-out-seed 50% views; ties choose smaller C",
        "validation_summary": validation_summary,
        "training_weight_rule": "each stay contributes four views at 0.25 each; weights sum to one per stay",
        "scalers_fit_on": {"density_augmented": "pooled split-0 training views only, with per-view sample weights",
                           "ordinary": "split-0 real training rows only"},
        "final_probe_fit_partition": "MIMIC split 0 only",
        "provenance": identity,
        "provenance_hashes": code_hashes,
        "frozen_embedding_producer_runtime": identity["embedding_cache_runtime_identity"],
        "density_view_encoder_consumer_runtime": gpu_runtime,
        "consumer_fit_runtime": {"python": sys.version.split()[0], "numpy": np.__version__,
                                 "scikit_learn": sklearn.__version__,
                                 "platform": platform.platform()},
        "encoder_training_performed": False,
        "claim_limit": "This source-only experiment tests a fixed readout augmentation; it does not establish novelty, benefit, or clinical usefulness.",
    }
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print("VALIDATION_FREEZE_LOCK_WRITTEN", lock_path, "C=", selected_C,
          "ordinary_tuned_C=", ordinary_C, flush=True)

    # Only after writing the source-validation lock do we read split-2 labels for metrics.
    y_test = y[report]
    groups_test = subject[report]
    real_test = real[report]
    if len(np.unique(y_test)) != 2:
        raise RuntimeError("report split lacks both classes")
    models = {
        "density_augmented": (aug_scaler, density_model),
        "ordinary_tuned": (ordinary_scaler, ordinary_tuned_model),
        "original_ordinary_C1": (ordinary_scaler, original_model),
    }
    test_metrics = {}
    test_probabilities = {}
    for name, (scaler, model) in models.items():
        probability = model.predict_proba(scaler.transform(real_test))[:, 1]
        test_probabilities[name] = probability
        test_metrics[name] = {
            **_scores(y_test, probability),
            "patient_cluster_95pct_ci": _cluster_ci(
                y_test, probability, groups_test, N_BOOTSTRAP,
                BASE_SEED + 1000 + list(models).index(name)),
        }

    # Controlled token retention for the validation-selected density candidate.
    density_scaler, density_candidate = models["density_augmented"]
    p_full = test_probabilities["density_augmented"]
    logit_full = np.log(np.clip(p_full, 1e-7, 1-1e-7) /
                        (1 - np.clip(p_full, 1e-7, 1-1e-7)))
    sensitivity = [{"retention": 1.0, "usable_stays": int(n_test),
                    "usable_fraction": 1.0,
                    "retained_token_fraction_all": 1.0,
                    "retained_token_fraction_events": 1.0,
                    "event_stays_with_any_token": int(y_test.sum()),
                    "event_retention_fraction": 1.0,
                    "mean_absolute_probability_change_vs_full": 0.0,
                    "mean_absolute_logit_change_vs_full": 0.0,
                    **_scores(y_test, p_full)}]
    for i, retention in enumerate(RETENTION_GRID):
        X = test_ret[i]
        probability = density_candidate.predict_proba(density_scaler.transform(X))[:, 1]
        clipped = np.clip(probability, 1e-7, 1-1e-7)
        logits = np.log(clipped / (1-clipped))
        counts = kept_test[i]
        usable = counts > 0
        event_mask = y_test == 1
        sensitivity.append({
            "retention": float(retention),
            "usable_stays": int(usable.sum()),
            "usable_fraction": float(usable.mean()),
            "retained_token_fraction_all": float(np.sum(counts) / np.sum(token_count[report])),
            "retained_token_fraction_events": float(np.sum(counts[event_mask]) /
                                                        np.sum(token_count[report][event_mask])),
            "event_stays_with_any_token": int(np.sum(event_mask & usable)),
            "event_retention_fraction": float(np.sum(event_mask & usable) / event_mask.sum()),
            "mean_absolute_probability_change_vs_full": float(np.mean(np.abs(probability - p_full))),
            "mean_absolute_logit_change_vs_full": float(np.mean(np.abs(logits - logit_full))),
            **_scores(y_test, probability),
        })
    fitted = {
        "density_augmented": {"C": float(selected_C), "scaler": aug_scaler,
                              "probe": density_model},
        "ordinary_tuned": {"C": float(ordinary_C), "scaler": ordinary_scaler,
                           "probe": ordinary_tuned_model},
        "original_ordinary_C1": {"C": 1.0, "scaler": ordinary_scaler,
                                 "probe": original_model},
        "provenance_hashes": code_hashes,
        "freeze_lock_sha256": sha256_file(lock_path),
    }
    with probe_path.open("xb") as handle:
        pickle.dump(fitted, handle, protocol=pickle.HIGHEST_PROTOCOL)
    report = {
        "source_scope": "MIMIC only",
        "split_roles": {"0": "train", "1": "tune", "2": "report"},
        "selected_stays": int(len(y)), "selected_events": int(y.sum()),
        "split_counts": {str(i): int(masks[i].sum()) for i in (0, 1, 2)},
        "validation_selection": validation_summary,
        "report_split_full_input_metrics": test_metrics,
        "report_split_patient_cluster_bootstrap_draws": N_BOOTSTRAP,
        "report_split_patient_cluster_count": int(len(np.unique(groups_test))),
        "controlled_retention_density_candidate": sensitivity,
        "runtime": {"producer_real_embedding": identity["embedding_cache_runtime_identity"],
                    "consumer_view_encoder": gpu_runtime,
                    "consumer_readout_fit": {"python": sys.version.split()[0],
                                             "numpy": np.__version__,
                                             "scikit_learn": sklearn.__version__,
                                             "elapsed_seconds": round(time.time() - start, 3)}},
        "memory": {"readout_max_rss_kb": int(__import__("resource").getrusage(
            __import__("resource").RUSAGE_SELF).ru_maxrss)},
        "artifacts": {"freeze_lock": str(lock_path), "fitted_probes": str(probe_path),
                      "embedding_cache_sha256": identity["embedding_cache_sha256"],
                      "view_embeddings_sha256": sha256_file(embedding_path)},
        "limitations": [
            "All outcomes are MIMIC source outcomes; no external cohort or INSPIRE performance was used.",
            "The source report split is a held-out internal partition from the same retrospective source.",
            "Input availability at real-time clinical decision points remains unverified.",
            "Density augmentation is evaluated as a fixed-readout candidate and is not asserted to be novel or beneficial.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("FIT_PASS", json.dumps({
        "selected_C": selected_C, "ordinary_tuned_C": ordinary_C,
        "test": {name: {metric: values[metric] for metric in ("brier", "log_loss", "auroc")}
                 for name, values in test_metrics.items()},
        "report": str(report_path), "lock_sha256": sha256_file(lock_path),
    }, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "fit"))
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        fit()


if __name__ == "__main__":
    main()
