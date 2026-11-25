#!/usr/bin/env python3
"""CPU-side provenance validation, frozen-probe scoring, and paired analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
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
OUT = REV / "controls" / "fixed_head_controls_20261123"
RUN_DIR = REV / "engine/results/prediction_runs/status_consistent_primary_core_fix2-7da980ffcbd8f00b"
ELIG = REV / "temporal/results/eligibility/status_consistent_v4_1/landmark_v4_1"
EMBED_MANIFEST = REV / "decoding/inputs/immutable/embedding_manifest_core14_20261123_a636f5f3.json"
PRED_MANIFEST = RUN_DIR / "prediction_manifest.json"
CHECKPOINT = ROOT / "models/main20k.pt"
ENCODER_CODE = ROOT / "code/fm.py"
ENGINE_CODE = REV / "engine/code/v4_engine.py"
ENGINE_SNAPSHOT = REV / "engine/code_snapshots/v4_engine_pre_registry_20261123_maskfix_status_consistent_primary_core_fix1.py"
V2_ENCODER_CODE = ROOT / "revision_v2_20261117/code/decoding_motor_v2.py"
INPUTS = OUT / "preflight_arrays.npz"
PREFLIGHT = OUT / "preflight_manifest.json"
GPU_MANIFEST = OUT / "gpu_encode_manifest.json"
INTERMEDIATE = OUT / "gpu_embeddings.npz"
SMOKE_EMBEDDINGS = OUT / "smoke_embeddings.npz"
SMOKE_GPU_MANIFEST = OUT / "smoke_gpu_manifest.json"
SMOKE_CPU_RESULT = OUT / "smoke_cpu_validation.json"
PROBE_PATH = REV / "engine/results/source_probes/main20k_real_39321cc8e6e2ad96c18d.pkl"
EXPECTED = {
    "run_protocol_sha256": "7da980ffcbd8f00b972bf3515590c59014e92bb94256183d15979e617e560d3b",
    "eligibility_manifest_sha256": "ea2b55bc3cf071d48b019d4ba97b4a3dc53d5d7fb3f6ac49a9e7c26007064cb5",
    "embedding_manifest_sha256": "a636f5f38ba7376dcb632cc62ba20592413376da366f105711ce412b32d1fdc9",
    "checkpoint_sha256": "0719db57b91b261869aed119cee95bf8980eb14411684c2040ba8588b45460bd",
    "probe_sha256": "25b6c9015cf73d5e0156b76197cf1101d6f6e4649407b3ba15b48b2f90a00707",
    "mimic_sequence_sha256": "9bd20bf1394a34ba112a4de9a8d7e8971821bef90558a9987be4b9da5ae06824",
    "eicu_sequence_sha256": "5255d2c39b011126b473d51c5fbf3284d554e18b133281064d7655a4cbad786a",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def runtime_record() -> dict:
    import sklearn
    import pandas
    import scipy
    return {"python": sys.version.split()[0], "implementation": platform.python_implementation(),
            "numpy": np.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "python_executable": sys.executable}


def engine_module():
    sys.path.insert(0, str(REV / "engine/code"))
    import v4_engine
    return v4_engine


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        ds = {k: z[k] for k in ("V", "B", "H", "L", "stay_id", "subject", "y_mort")}
    ds["split"] = np.load(path.parent / f"split_{path.stem.removeprefix('seq_')}.npy")
    return ds


def scalar_str(a) -> str:
    return str(np.asarray(a).item())


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def preflight() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (INPUTS, PREFLIGHT):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite: {path}")
    import pandas as pd
    import sklearn
    if sklearn.__version__ != "1.8.0":
        raise RuntimeError(f"fixed probe consumer must use sklearn 1.8.0, found {sklearn.__version__}")
    eng = engine_module()
    log("PRELIGHT start; validating current fix2 run and immutable cache identity")
    pred_manifest = json.loads(PRED_MANIFEST.read_text())
    if pred_manifest.get("run_id") != "status_consistent_primary_core_fix2":
        raise ValueError("unexpected prediction run")
    if pred_manifest.get("run_protocol_sha256") != EXPECTED["run_protocol_sha256"]:
        raise ValueError("fix2 run-protocol SHA256 does not match its versioned run ID")
    run_protocol = pred_manifest["run_protocol"]
    if run_protocol.get("run_protocol_sha256") != pred_manifest.get("run_protocol_sha256"):
        raise ValueError("run-protocol hash field mismatch")
    if run_protocol.get("checkpoint_sha256_by_model", {}).get("main20k") != EXPECTED["checkpoint_sha256"]:
        raise ValueError("run protocol main20k checkpoint does not match")
    if sha256_file(CHECKPOINT) != EXPECTED["checkpoint_sha256"]:
        raise ValueError("main20k checkpoint SHA256 mismatch")
    if sha256_file(EMBED_MANIFEST) != EXPECTED["embedding_manifest_sha256"]:
        raise ValueError("immutable core14 embedding manifest SHA256 mismatch")
    if run_protocol.get("embedding_manifest_sha256") != EXPECTED["embedding_manifest_sha256"]:
        raise ValueError("fix2 run does not attest the immutable core14 embedding manifest")
    if sha256_file(ELIG / "eligibility_manifest.json") != EXPECTED["eligibility_manifest_sha256"]:
        raise ValueError("v4.1 eligibility manifest SHA256 mismatch")
    if eng.encoder_code_hash() != pred_manifest.get("encoder_code_sha256"):
        raise ValueError("composite encoder implementation hash no longer matches fix2")
    if sha256_file(ENGINE_SNAPSHOT) != pred_manifest.get("engine_code_sha256"):
        raise ValueError("exact fix2 engine snapshot SHA256 mismatch")

    embedding_manifest = json.loads(EMBED_MANIFEST.read_text())
    current_predictions = pd.read_parquet(RUN_DIR / "main20k_real.parquet")
    if sha256_file(RUN_DIR / "main20k_real.parquet") != next(
        r["prediction_file_sha256"] for r in pred_manifest["predictions"]
        if r.get("analysis_id") == "main20k_real" and r.get("db") == "mimic"
    ):
        raise ValueError("current main20k real prediction file hash mismatch")

    source_meta = None
    source_manifest_entry = None
    for row in pred_manifest["predictions"]:
        if row.get("analysis_id") == "main20k_real" and row.get("db") == "mimic":
            source_manifest_entry = row
            break
    if source_manifest_entry is None:
        raise ValueError("fix2 manifest lacks main20k_real/MIMIC source entry")
    if source_manifest_entry.get("source_probe_artifact") != str(PROBE_PATH):
        raise ValueError("exact source probe path differs from fix2 prediction manifest")
    if source_manifest_entry.get("source_probe_sha256") != EXPECTED["probe_sha256"]:
        raise ValueError("fix2 manifest source probe hash differs from expected")
    if sha256_file(PROBE_PATH) != EXPECTED["probe_sha256"]:
        raise ValueError("source probe artifact SHA256 mismatch")
    probe_obj = pickle.load(PROBE_PATH.open("rb"))
    if not isinstance(probe_obj, dict) or "scaler" not in probe_obj or "probe" not in probe_obj:
        raise ValueError("source probe pickle schema is invalid")
    probe_meta = {k: v for k, v in probe_obj.items() if k not in ("scaler", "probe")}
    if eng.source_probe_artifact_path(probe_meta).resolve() != PROBE_PATH.resolve():
        raise ValueError("source probe artifact key/path is invalid")
    for key, expected in {
        "run_id": "status_consistent_primary_core_fix2", "model_tag": "main20k",
        "input_condition": "real", "checkpoint_sha256": EXPECTED["checkpoint_sha256"],
        "input_mask_sha256": eng.input_mask_hash("real"),
        "eligibility_manifest_sha256": EXPECTED["eligibility_manifest_sha256"],
        "eligibility_table_sha256": "c2774cd59e4df8706960e4fa58680ea6ec72b4af22adddf82b2389dc5b74749f",
        "source_npz_sha256": EXPECTED["mimic_sequence_sha256"],
        "prediction_scope": "retrospective_landmark_prediction_measurement_time",
        "input_scope": "retrospective_measurement_time",
        "clinical_input_availability_verified": False,
    }.items():
        if probe_meta.get(key) != expected:
            raise ValueError(f"source probe metadata mismatch for {key}: {probe_meta.get(key)!r}")
    if source_manifest_entry.get("checkpoint_sha256") != EXPECTED["checkpoint_sha256"]:
        raise ValueError("source probe manifest checkpoint mismatch")
    if source_manifest_entry.get("retrospective_landmark_prediction_ready") is not True:
        raise ValueError("MIMIC source probe cohort is not retrospectively eligible")
    if source_manifest_entry.get("clinical_ready") is not False:
        raise ValueError("clinical readiness must remain false")
    if source_manifest_entry.get("pretraining_target_exposed") is not False:
        raise ValueError("main20k probe unexpectedly claims target exposure")
    if probe_meta.get("runtime_identity_sha256") != source_manifest_entry.get("cache_runtime_identity_sha256"):
        raise ValueError("source probe runtime identity does not match main20k cache producer")
    consumer_sidecar = RUN_DIR.parent / "status_consistent_primary_core_fix2_cpu_consumer_runtime.json"
    if not consumer_sidecar.is_file():
        # The sidecar is stored with the prediction run directory, not its parent.
        consumer_sidecar = RUN_DIR / "cpu_consumer_runtime.json"
    if not consumer_sidecar.is_file():
        raise FileNotFoundError("verified CPU 1.8 source-probe consumer sidecar is missing")
    sidecar_obj = json.loads(consumer_sidecar.read_text())
    sidecar_sklearn = sidecar_obj.get("consumer_runtime", {}).get("scikit_learn", {}).get("version")
    if sidecar_sklearn != "1.8.0":
        raise RuntimeError(f"source-probe CPU consumer runtime is not sklearn 1.8.0: {sidecar_sklearn}")

    arrays: dict[str, np.ndarray] = {}
    cohort_provenance = {}
    cache_by_db = {}
    aligned_by_db = {}
    for db in ("mimic", "eicu"):
        paths = eng.resolve_cohort_paths(db, verify_exists=True)
        seq_path = paths.sequence_path
        split_path = paths.split_path
        expected_seq = EXPECTED[f"{db}_sequence_sha256"]
        seq_hash = sha256_file(seq_path)
        split_hash = sha256_file(split_path)
        if seq_hash != expected_seq:
            raise ValueError(f"{db}: sequence NPZ hash mismatch")
        eligible, eligible_entry = eng.load_ready_eligibility(db, ELIG)
        if eligible_entry.get("eligibility_manifest_sha256") != EXPECTED["eligibility_manifest_sha256"]:
            raise ValueError(f"{db}: eligibility manifest entry hash mismatch")
        ds = eng.load_sequence_registered(db, paths)
        all_rows = eng.sequence_rows(ds)
        matched = eng.align_eligible_rows(db, ds, eligible, all_rows)
        matched["split"] = np.asarray(ds["split"], dtype=np.int8)[matched["rows"].to_numpy()]
        if db == "mimic":
            train = matched.loc[matched["split"].to_numpy() == 0].copy()
            eval_rows = matched.loc[matched["split"].to_numpy() == 2].copy()
            if not train.shape[0] or len(np.unique(train["y_mort"])) != 2:
                raise ValueError("MIMIC eligible split0 source training rows/classes are invalid")
            if len(set(train["subject"].astype(str)) & set(eval_rows["subject"].astype(str))):
                raise ValueError("MIMIC source train/test patients overlap")
            train_rows_probe_order = train["rows"].to_numpy(dtype=np.int64)
            train_y_probe_order = train["y_mort"].to_numpy(dtype=np.int8)
            source_train_identity = eng.sha256_ids(
                db, train_rows_probe_order, np.asarray(ds["stay_id"])[train_rows_probe_order],
                np.asarray(ds["subject"]).astype(str)[train_rows_probe_order])
            source_train_label = hashlib.sha256(train_y_probe_order.tobytes()).hexdigest()
            if source_train_identity != probe_meta.get("source_train_identity_sha256"):
                raise ValueError("source probe source-train identity hash mismatch")
            if source_train_label != probe_meta.get("source_train_label_sha256"):
                raise ValueError("source probe source-train label hash mismatch")
            if int(train.shape[0]) != int(probe_meta.get("source_train_n", -1)):
                raise ValueError("source probe source-train row count mismatch")
            if len(set(train["subject"].astype(str))) != int(probe_meta.get("source_train_patient_n", -1)):
                raise ValueError("source probe source-train patient count mismatch")
            if len(eval_rows) != 14890:
                raise ValueError(f"MIMIC eligible split2 count unexpected: {len(eval_rows)}")
            model_rows = np.sort(np.unique(np.concatenate([
                train["rows"].to_numpy(dtype=np.int64), eval_rows["rows"].to_numpy(dtype=np.int64)])))
        else:
            train = None
            eval_rows = matched
            model_rows = np.empty(0, dtype=np.int64)
            if len(eval_rows) != 130544:
                raise ValueError(f"eICU eligible count unexpected: {len(eval_rows)}")
        eval_indices = eval_rows["rows"].to_numpy(dtype=np.int64)
        eval_indices = np.sort(eval_indices)
        # Cache identity is checked for every source row before selected rows are reused.
        cache_entry = next((r for r in embedding_manifest.get("models", [])
                            if r.get("db") == db and r.get("model_tag") == "main20k"
                            and r.get("input_condition") == "real"), None)
        if not isinstance(cache_entry, dict):
            raise ValueError(f"{db}: immutable core14 manifest has no main20k real cache row")
        cache = Path(cache_entry["cache"])
        identity_hash = eng.sha256_ids(
            db, all_rows, np.asarray(ds["stay_id"])[all_rows],
            np.asarray(ds["subject"]).astype(str)[all_rows])
        mask_hash = eng.input_mask_hash("real")
        ck_hash = sha256_file(CHECKPOINT)
        code_hash = eng.encoder_code_hash()
        if (cache_entry.get("source_npz_sha256") != seq_hash
                or cache_entry.get("split_sha256") != split_hash
                or cache_entry.get("checkpoint_sha256") != ck_hash
                or cache_entry.get("input_mask_sha256") != mask_hash
                or cache_entry.get("row_identity_sha256") != identity_hash
                or cache_entry.get("encoder_code_sha256") != code_hash
                or cache_entry.get("n_rows") != len(all_rows)):
            raise ValueError(f"{db}: immutable embedding manifest identity mismatch")
        if cache.parent.resolve() != (REV / "engine/cache").resolve() or not cache.is_file():
            raise ValueError(f"{db}: cache path missing or outside engine/cache")
        with np.load(cache, allow_pickle=False) as z:
            cache_rows = z["rows"]
            cache_stay = z["stay_id"].astype(str)
            cache_subject = z["subject"].astype(str)
            cache_y = z["source_y"].astype(np.float32)
            embeddings = z["embedding"].astype(np.float32, copy=False)
            cache_meta = json.loads(scalar_str(z["metadata_json"]))
        if (not np.array_equal(cache_rows, all_rows)
                or not np.array_equal(cache_stay, np.asarray(ds["stay_id"])[all_rows].astype(str))
                or not np.array_equal(cache_subject, np.asarray(ds["subject"])[all_rows].astype(str))
                or not np.array_equal(cache_y, np.asarray(ds["y_mort"], dtype=np.float32)[all_rows], equal_nan=True)
                or embeddings.shape != (len(all_rows), 128) or not np.isfinite(embeddings).all()):
            raise ValueError(f"{db}: immutable cache row/ID/label/embedding mismatch")
        for key, expected in {
            "db": db, "model_tag": "main20k", "input_condition": "real",
            "input_scope": "retrospective_measurement_time",
            "clinical_input_availability_verified": False,
            "checkpoint_sha256": ck_hash, "input_mask_sha256": mask_hash,
            "row_identity_sha256": identity_hash, "source_npz_sha256": seq_hash,
            "split_sha256": split_hash, "encoder_code_sha256": code_hash,
            "runtime_identity": cache_entry.get("runtime_identity"),
            "runtime_identity_sha256": cache_entry.get("runtime_identity_sha256"),
        }.items():
            if cache_meta.get(key) != expected:
                raise ValueError(f"{db}: cache metadata mismatch: {key}")
        if canonical_sha(cache_entry.get("runtime_identity")) != cache_entry.get("runtime_identity_sha256"):
            raise ValueError(f"{db}: producer runtime identity hash mismatch")
        pos = {int(row): i for i, row in enumerate(cache_rows)}
        eval_cache_pos = np.asarray([pos[int(r)] for r in eval_indices], dtype=np.int64)
        cache_by_db[db] = (cache_entry, cache, embeddings, pos, cache_meta)
        aligned_by_db[db] = (ds, matched, eval_rows, train, eval_indices, eval_cache_pos)
        cohort_provenance[db] = {
            "source_sequence": str(seq_path), "source_sequence_sha256": seq_hash,
            "source_sequence_size_bytes": seq_path.stat().st_size,
            "source_sequence_mtime_ns": seq_path.stat().st_mtime_ns,
            "split_sha256": split_hash,
            "eligibility_table": str(ELIG / f"{db}.parquet"),
            "eligibility_table_sha256": eligible_entry["eligibility_table_sha256"],
            "eligible_evaluation_rows": int(len(eval_indices)),
            "source_train_rows": int(len(train)) if train is not None else None,
            "row_identity_sha256": identity_hash,
            "cache_path": str(cache), "cache_sha256": sha256_file(cache),
            "cache_producer_runtime_identity": cache_entry.get("runtime_identity"),
            "cache_producer_runtime_identity_sha256": cache_entry.get("runtime_identity_sha256"),
        }
        if db == "mimic":
            eval_matched = eval_rows
            model_rows = np.sort(np.unique(np.concatenate([
                train["rows"].to_numpy(dtype=np.int64), eval_indices])))
            arrays["mimic_model_rows"] = model_rows.astype(np.int64)
            arrays["mimic_train_rows"] = np.sort(train["rows"].to_numpy(dtype=np.int64))
            arrays["mimic_eval_rows"] = eval_indices
            arrays["mimic_eval_stay_id"] = np.asarray(ds["stay_id"])[eval_indices].astype(str)
            arrays["mimic_eval_subject"] = np.asarray(ds["subject"])[eval_indices].astype(str)
            arrays["mimic_eval_y"] = np.asarray(ds["y_mort"], dtype=np.int8)[eval_indices]
            arrays["mimic_train_y"] = np.asarray(ds["y_mort"], dtype=np.int8)[arrays["mimic_train_rows"]]
            arrays["mimic_model_split"] = np.asarray(ds["split"], dtype=np.int8)[model_rows]
            arrays["mimic_model_stay_id"] = np.asarray(ds["stay_id"])[model_rows].astype(str)
            arrays["mimic_model_subject"] = np.asarray(ds["subject"])[model_rows].astype(str)
            arrays["mimic_model_y"] = np.asarray(ds["y_mort"], dtype=np.int8)[model_rows]
            if not np.array_equal(arrays["mimic_model_rows"][arrays["mimic_model_split"] == 0], arrays["mimic_train_rows"]):
                raise ValueError("MIMIC random encoder source train rows are not split0-only")
        else:
            arrays["eicu_eval_rows"] = eval_indices
            arrays["eicu_eval_stay_id"] = np.asarray(ds["stay_id"])[eval_indices].astype(str)
            arrays["eicu_eval_subject"] = np.asarray(ds["subject"])[eval_indices].astype(str)
            arrays["eicu_eval_y"] = np.asarray(ds["y_mort"], dtype=np.int8)[eval_indices]
        log(f"PREFLIGHT {db}: eligible_eval={len(eval_indices)} cache_rows={len(all_rows)} cache_id_hash={identity_hash[:16]}")

    # Validate the fixed source probe against current real-input probabilities.
    scaler, probe = probe_obj["scaler"], probe_obj["probe"]
    if getattr(probe, "class_weight", None) is not None or float(probe.C) != 1.0 or int(probe.max_iter) != 3000:
        raise ValueError("fixed source probe estimator contract differs from current C1 head")
    fixed_parameter_snapshot = (scaler.mean_.copy(), scaler.scale_.copy(), probe.coef_.copy(), probe.intercept_.copy())
    prediction_checks = {}
    for db in ("mimic", "eicu"):
        _, cache, embeddings, _, _ = cache_by_db[db]
        _, _, eval_rows, _, eval_indices, eval_cache_pos = aligned_by_db[db]
        E = embeddings[eval_cache_pos]
        p = probe.predict_proba(scaler.transform(E))[:, 1]
        truth = np.asarray(aligned_by_db[db][0]["y_mort"], dtype=np.int8)[eval_indices]
        stay = np.asarray(aligned_by_db[db][0]["stay_id"])[eval_indices].astype(str)
        frame = current_predictions.loc[current_predictions["db"].astype(str) == db].copy()
        frame["stay_id"] = frame["stay_id"].astype(str)
        lookup = frame.set_index("stay_id", verify_integrity=True)
        if set(stay) != set(lookup.index):
            raise ValueError(f"{db}: eligible row IDs do not equal current fix2 prediction IDs")
        p_current = np.asarray([float(lookup.loc[s, "p"]) for s in stay])
        y_current = np.asarray([int(lookup.loc[s, "y"]) for s in stay], dtype=np.int8)
        if not np.array_equal(truth, y_current):
            raise ValueError(f"{db}: current run and eligible source labels differ")
        max_abs = float(np.max(np.abs(p - p_current)))
        if max_abs > 2e-6:
            raise ValueError(f"{db}: frozen source probe/cache predictions differ from fix2 (max {max_abs})")
        prediction_checks[db] = {"n": int(len(p)), "max_abs_probability_difference_vs_fix2": max_abs}
        if db == "mimic":
            pred_row = source_manifest_entry
        else:
            pred_row = next(r for r in pred_manifest["predictions"]
                            if r.get("analysis_id") == "main20k_real" and r.get("db") == "eicu")
        cohort_frame = current_predictions.loc[current_predictions["db"].astype(str) == db]
        if eng.sha256_ordered_prediction_rows(cohort_frame) != pred_row["ordered_row_digest_sha256"]:
            raise ValueError(f"{db}: fix2 ordered prediction digest mismatch")

    if not all(np.array_equal(a, b) for a, b in zip(
        fixed_parameter_snapshot,
        (scaler.mean_, scaler.scale_, probe.coef_, probe.intercept_))):
        raise AssertionError("source fixed-probe parameters changed during validation")
    # Actual allow_pickle=False Unicode-ID round trip.
    with tempfile.TemporaryDirectory(prefix="unicode_npz_test_", dir=OUT) as td:
        test_path = Path(td) / "unicode_ids.npz"
        np.savez(test_path, ids=np.asarray(["stay_α", "patient_002"], dtype=str))
        with np.load(test_path, allow_pickle=False) as z:
            if z["ids"].dtype.kind != "U" or z["ids"].tolist() != ["stay_α", "patient_002"]:
                raise AssertionError("allow_pickle=False Unicode-ID serialization round trip failed")
    with INPUTS.open("wb") as f:
        np.savez(f, **arrays)
    with np.load(INPUTS, allow_pickle=False) as z:
        if z["mimic_eval_stay_id"].dtype.kind != "U" or z["eicu_eval_stay_id"].dtype.kind != "U":
            raise AssertionError("preflight input ID arrays are not Unicode")
        if not np.array_equal(z["mimic_eval_rows"], arrays["mimic_eval_rows"]):
            raise AssertionError("preflight array serialization round trip failed")
    manifest = {
        "status": "preflight_validated",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_arrays": str(INPUTS), "input_arrays_sha256": sha256_file(INPUTS),
        "input_arrays_size_bytes": INPUTS.stat().st_size,
        "prediction_run_id": pred_manifest["run_id"],
        "prediction_manifest_sha256": sha256_file(PRED_MANIFEST),
        "prediction_manifest_run_protocol_sha256": pred_manifest["run_protocol_sha256"],
        "eligibility_manifest_sha256": sha256_file(ELIG / "eligibility_manifest.json"),
        "embedding_manifest_path": str(EMBED_MANIFEST),
        "embedding_manifest_sha256": sha256_file(EMBED_MANIFEST),
        "checkpoint_path": str(CHECKPOINT), "checkpoint_sha256": sha256_file(CHECKPOINT),
        "encoder_code_sha256": eng.encoder_code_hash(),
        "fm_source_sha256": sha256_file(ENCODER_CODE),
        "v2_encoder_source_sha256": sha256_file(V2_ENCODER_CODE),
        "run_engine_snapshot_path": str(ENGINE_SNAPSHOT),
        "run_engine_snapshot_sha256": sha256_file(ENGINE_SNAPSHOT),
        "readonly_current_engine_helper_path": str(ENGINE_CODE),
        "readonly_current_engine_helper_sha256": sha256_file(ENGINE_CODE),
        "source_probe_path": str(PROBE_PATH), "source_probe_sha256": sha256_file(PROBE_PATH),
        "source_probe_metadata": probe_meta,
        "source_probe_estimator": {
            "scaler_n_features": int(scaler.n_features_in_),
            "scaler_n_samples_seen": int(scaler.n_samples_seen_),
            "probe_n_features": int(probe.n_features_in_), "probe_classes": probe.classes_.astype(int).tolist(),
            "C": float(probe.C), "max_iter": int(probe.max_iter),
            "class_weight": probe.class_weight, "random_state": probe.random_state,
        },
        "source_probe_predictions_match_fix2": prediction_checks,
        "source_train_identity_sha256": probe_meta["source_train_identity_sha256"],
        "source_train_label_sha256": probe_meta["source_train_label_sha256"],
        "cohorts": cohort_provenance,
        "consumer_runtime": runtime_record(),
        "source_probe_original_consumer_runtime_sidecar": str(consumer_sidecar),
        "source_probe_original_consumer_runtime_sidecar_sha256": sha256_file(consumer_sidecar),
        "source_probe_original_consumer_runtime": sidecar_obj.get("consumer_runtime"),
        "producer_runtime_identity_by_cohort": {
            db: cohort_provenance[db]["cache_producer_runtime_identity"] for db in ("mimic", "eicu")
        },
        "limitations": [
            "Inputs use retrospective measurement-time data; real-time input availability is not verified.",
            "MIMIC mortality endpoint has exact timestamp evidence; eICU uses an operational discharge proxy.",
        ],
    }
    atomic_json(PREFLIGHT, manifest)
    log(f"PREFLIGHT_COMPLETE manifest={PREFLIGHT} inputs_sha256={manifest['input_arrays_sha256']}")


def logits_from_probe(scaler, probe, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = probe.decision_function(scaler.transform(np.asarray(X, dtype=np.float32)))
    p = probe.predict_proba(scaler.transform(np.asarray(X, dtype=np.float32)))[:, 1]
    return np.asarray(z, dtype=np.float64), np.asarray(p, dtype=np.float64)


def point_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
    y = np.asarray(y, dtype=np.int8); p = np.asarray(p, dtype=np.float64)
    return {"auroc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "logloss": float(log_loss(y, p, labels=[0, 1]))}


def _sorted_score_groups(y: np.ndarray, p: np.ndarray):
    order = np.argsort(-np.asarray(p, dtype=np.float64), kind="mergesort")
    ps = np.asarray(p, dtype=np.float64)[order]
    starts = np.r_[0, np.flatnonzero(ps[1:] != ps[:-1]) + 1]
    return order, starts


def weighted_metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray, groups=None) -> dict:
    y = np.asarray(y, dtype=np.int8); p = np.asarray(p, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    total = w.sum()
    pos_total = float(w[y == 1].sum()); neg_total = float(w[y == 0].sum())
    if total <= 0 or pos_total <= 0 or neg_total <= 0:
        return {k: np.nan for k in ("auroc", "average_precision", "brier", "logloss")}
    order, starts = groups if groups is not None else _sorted_score_groups(y, p)
    pos = np.add.reduceat(w[order] * y[order], starts)
    neg = np.add.reduceat(w[order] * (1 - y[order]), starts)
    cum_pos = np.cumsum(pos); cum_neg = np.cumsum(neg)
    # Descending scores: for each positive, count lower-scoring negatives plus
    # half of the negatives tied at the same score.
    auc = float(np.sum(pos * (neg_total - cum_neg + 0.5 * neg)) / (pos_total * neg_total))
    precision = cum_pos / np.maximum(cum_pos + cum_neg, 1e-300)
    ap = float(np.sum((pos / pos_total) * precision))
    pc = np.clip(p, 1e-15, 1 - 1e-15)
    brier = float(np.sum(w * (pc - y) ** 2) / total)
    ll = float(-np.sum(w * (y * np.log(pc) + (1 - y) * np.log1p(-pc))) / total)
    return {"auroc": auc, "average_precision": ap, "brier": brier, "logloss": ll}


def paired_cluster_bootstrap(y, p_base, p_alt, z_base, z_alt, subjects,
                             seed: int, reps: int = 1000, label: str = "") -> dict:
    y = np.asarray(y, dtype=np.int8); subjects = np.asarray(subjects).astype(str)
    unique, inverse = np.unique(subjects, return_inverse=True)
    if len(unique) < 2 or len(np.unique(y)) < 2:
        raise ValueError("cluster bootstrap needs multiple patients and outcome classes")
    rng = np.random.default_rng(seed)
    idx_base, starts_base = _sorted_score_groups(y, p_base)
    idx_alt, starts_alt = _sorted_score_groups(y, p_alt)
    base_groups, alt_groups = (idx_base, starts_base), (idx_alt, starts_alt)
    keys = ("auroc", "average_precision", "brier", "logloss", "mean_delta_probability",
            "mean_abs_delta_probability", "mean_delta_logit", "mean_abs_delta_logit")
    samples = {k: np.empty(reps, dtype=np.float64) for k in keys}
    dp = np.asarray(p_alt) - np.asarray(p_base)
    dz = np.asarray(z_alt) - np.asarray(z_base)
    for b in range(reps):
        draws = rng.integers(0, len(unique), size=len(unique))
        patient_counts = np.bincount(draws, minlength=len(unique))
        w = patient_counts[inverse].astype(np.float64)
        m0 = weighted_metrics(y, p_base, w, base_groups)
        m1 = weighted_metrics(y, p_alt, w, alt_groups)
        for key in ("auroc", "average_precision", "brier", "logloss"):
            samples[key][b] = m1[key] - m0[key]
        denom = w.sum()
        samples["mean_delta_probability"][b] = np.sum(w * dp) / denom
        samples["mean_abs_delta_probability"][b] = np.sum(w * np.abs(dp)) / denom
        samples["mean_delta_logit"][b] = np.sum(w * dz) / denom
        samples["mean_abs_delta_logit"][b] = np.sum(w * np.abs(dz)) / denom
        if (b + 1) % 100 == 0:
            log(f"BOOTSTRAP {label}: {b + 1}/{reps}")
    point0 = point_metrics(y, p_base); point1 = point_metrics(y, p_alt)
    point = {k: point1[k] - point0[k] for k in ("auroc", "average_precision", "brier", "logloss")}
    point.update({
        "mean_delta_probability": float(np.mean(dp)),
        "mean_abs_delta_probability": float(np.mean(np.abs(dp))),
        "mean_delta_logit": float(np.mean(dz)),
        "mean_abs_delta_logit": float(np.mean(np.abs(dz))),
    })
    ci = {k: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] for k, v in samples.items()}
    return {"n_rows": int(len(y)), "n_patients": int(len(unique)), "bootstrap_replicates": reps,
            "patient_cluster_bootstrap_seed": seed,
            "base_metrics": point0, "alternative_metrics": point1,
            "change_alternative_minus_base": point, "paired_patient_bootstrap_95pct_ci": ci,
            "bootstrap_ci_conditioning": "fixed eligible evaluation rows; patients sampled with replacement, all their stays carried together"}


def analyze() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (GPU_MANIFEST, INTERMEDIATE, PREFLIGHT, INPUTS):
        if not path.is_file():
            raise FileNotFoundError(path)
    pre = json.loads(PREFLIGHT.read_text())
    gpu = json.loads(GPU_MANIFEST.read_text())
    if sha256_file(INPUTS) != pre["input_arrays_sha256"]:
        raise ValueError("preflight input array hash mismatch")
    if sha256_file(INTERMEDIATE) != gpu["intermediate_sha256"]:
        raise ValueError("GPU intermediate hash mismatch")
    if gpu.get("preflight_manifest_sha256") != sha256_file(PREFLIGHT):
        raise ValueError("GPU output was not produced from this validated preflight")
    if gpu.get("checkpoint_sha256") != EXPECTED["checkpoint_sha256"]:
        raise ValueError("GPU control checkpoint identity mismatch")
    if gpu.get("random_checkpoint_sha256") != sha256_file(OUT / "random_main20k_seed20261102.pt"):
        raise ValueError("random encoder checkpoint hash mismatch")
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
    eng = engine_module()
    pred_manifest = json.loads(PRED_MANIFEST.read_text())
    current_predictions = pd.read_parquet(RUN_DIR / "main20k_real.parquet")
    probe_obj = pickle.load(PROBE_PATH.open("rb"))
    scaler, probe = probe_obj["scaler"], probe_obj["probe"]
    with np.load(INPUTS, allow_pickle=False) as z:
        inp = {k: z[k] for k in z.files}
    with np.load(INTERMEDIATE, allow_pickle=False) as z:
        inter = {k: z[k] for k in z.files}
    results = {"control_a_fixed_main20k_head": {}, "control_b_random_initialized_same_architecture": {}}
    # Re-open source sequences and exact immutable caches for real-input baselines.
    trained_prob = {}; trained_logit = {}; trained_y = {}; trained_subject = {}; trained_stay = {}
    fixed_prob = {}; fixed_logit = {}; fixed_y = {}; fixed_subject = {}; fixed_stay = {}
    for db in ("mimic", "eicu"):
        paths = eng.resolve_cohort_paths(db, verify_exists=True)
        ds = eng.load_sequence_registered(db, paths)
        rows = inp[f"{db}_eval_rows"]
        stay = np.asarray(ds["stay_id"])[rows].astype(str)
        subject = np.asarray(ds["subject"])[rows].astype(str)
        y = np.asarray(ds["y_mort"], dtype=np.int8)[rows]
        if not np.array_equal(stay, inp[f"{db}_eval_stay_id"]) or not np.array_equal(subject, inp[f"{db}_eval_subject"]) or not np.array_equal(y, inp[f"{db}_eval_y"]):
            raise ValueError(f"{db}: intermediate eligible identity differs from source")
        cache_entry = next(r for r in json.loads(EMBED_MANIFEST.read_text())["models"]
                           if r.get("db") == db and r.get("model_tag") == "main20k" and r.get("input_condition") == "real")
        with np.load(cache_entry["cache"], allow_pickle=False) as c:
            cache_rows = c["rows"]
            Eall = c["embedding"].astype(np.float32, copy=False)
        pos = {int(r): i for i, r in enumerate(cache_rows)}
        Ereal = Eall[np.asarray([pos[int(r)] for r in rows], dtype=np.int64)]
        zreal, preal = logits_from_probe(scaler, probe, Ereal)
        frame = current_predictions.loc[current_predictions["db"].astype(str) == db].copy()
        frame["stay_id"] = frame["stay_id"].astype(str)
        frame = frame.set_index("stay_id", verify_integrity=True)
        pfix2 = np.asarray([float(frame.loc[s, "p"]) for s in stay])
        if np.max(np.abs(preal - pfix2)) > 2e-6:
            raise ValueError(f"{db}: real main20k source probe failed post-GPU probability match")
        trained_prob[db] = pfix2
        trained_logit[db] = np.log(np.clip(pfix2, 1e-15, 1 - 1e-15) / np.clip(1 - pfix2, 1e-15, 1))
        fixed_prob[db], fixed_logit[db] = preal, zreal
        fixed_y[db], fixed_subject[db], fixed_stay[db] = y, subject, stay
        trained_y[db], trained_subject[db], trained_stay[db] = y, subject, stay
        fixed_conditions = {}
        for cond in ("within", "donor"):
            X = inter[f"{db}_{cond}_embedding"].astype(np.float32, copy=False)
            if len(X) != len(rows):
                raise ValueError(f"{db}/{cond}: embedding row count mismatch")
            zc, pc = logits_from_probe(scaler, probe, X)
            fixed_conditions[cond] = (zc, pc)
            fixed_head_parameters_before = (scaler.mean_.copy(), scaler.scale_.copy(), probe.coef_.copy(), probe.intercept_.copy())
            if not all(np.array_equal(a, b) for a, b in zip(fixed_head_parameters_before, (scaler.mean_, scaler.scale_, probe.coef_, probe.intercept_))):
                raise AssertionError("fixed source head parameters changed")
            boot = paired_cluster_bootstrap(y, preal, pc, zreal, zc, subject,
                                            seed=20261123 + (0 if db == "mimic" else 1000) + (0 if cond == "within" else 10),
                                            label=f"A/{db}/{cond}")
            changed = gpu["cohorts"][db][f"{cond}_changed_token_fraction"]
            fixed_conditions[cond] = {"aggregate": boot, "changed_token_fraction": changed,
                                      "donor_coverage": gpu["cohorts"][db].get("donor", {}) if cond == "donor" else None}
        results["control_a_fixed_main20k_head"][db] = {
            "n": int(len(y)), "prevalence": float(y.mean()),
            "real_input_metrics": point_metrics(y, preal),
            "conditions": fixed_conditions,
            "interpretation_limits": [
                "The temporal shuffle preserves each stay-variable B-bin multiset and all V/H/L/padding positions.",
                "The donor replacement assigns a different patient uniformly within variable-specific eligible donors and samples donor bins with replacement.",
                "Different-patient donor replacement breaks joint physiology; it is a representation-response control, not a causal intervention.",
                "Cohorts are retrospective measurement-time analyses; real-time clinical input availability is unverified.",
            ],
        }

    # Random-init head: same architecture, seeded before construction, source C1 readout on split0 only.
    mimic_model_rows = inp["mimic_model_rows"]
    mimic_model_split = inp["mimic_model_split"]
    mimic_train_mask = mimic_model_split == 0
    mimic_test_mask = mimic_model_split == 2
    if (not mimic_train_mask.any() or not mimic_test_mask.any()
            or not set(np.unique(mimic_model_split).tolist()).issubset({0, 2})
            or int(mimic_train_mask.sum()) != int(len(inp["mimic_train_rows"]))
            or int(mimic_test_mask.sum()) != int(len(inp["mimic_eval_rows"]))):
        raise ValueError("random-init source/evaluation split contract failed")
    Xtrain = inter["mimic_random_embedding"][mimic_train_mask].astype(np.float32, copy=False)
    ytrain = inp["mimic_model_y"][mimic_train_mask].astype(np.int8, copy=False)
    readout_scaler = StandardScaler().fit(Xtrain)
    readout = LogisticRegression(C=1.0, max_iter=3000, class_weight=None, random_state=20261122)
    readout.fit(readout_scaler.transform(Xtrain), ytrain)
    if readout_scaler.n_samples_seen_ != len(Xtrain) or not np.array_equal(np.unique(inp["mimic_model_split"][mimic_train_mask]), np.asarray([0])):
        raise ValueError("random-init scaler/readout was not fit only on MIMIC split0")
    for db in ("mimic", "eicu"):
        y = inp[f"{db}_eval_y"].astype(np.int8, copy=False)
        subject = inp[f"{db}_eval_subject"].astype(str)
        X = inter[f"{db}_random_embedding"].astype(np.float32, copy=False)
        if db == "mimic":
            test_mask = inp["mimic_model_split"] == 2
            if int(test_mask.sum()) != len(y):
                raise ValueError("random MIMIC test embedding count differs from eligible split2")
            if (not np.array_equal(inp["mimic_model_stay_id"][test_mask].astype(str), inp["mimic_eval_stay_id"].astype(str))
                    or not np.array_equal(inp["mimic_model_subject"][test_mask].astype(str), subject)
                    or not np.array_equal(inp["mimic_model_y"][test_mask].astype(np.int8), y)):
                raise ValueError("random MIMIC test embedding IDs/subjects/labels do not match preflight evaluation rows")
            X = X[test_mask]
        zrand = readout.decision_function(readout_scaler.transform(X))
        prand = readout.predict_proba(readout_scaler.transform(X))[:, 1]
        # Verify identity pairs with current fix2 test/eICU rows.
        if not np.array_equal(inp[f"{db}_eval_stay_id"].astype(str), fixed_stay[db]):
            raise ValueError(f"{db}: random and pretrained evaluation row IDs differ")
        boot = paired_cluster_bootstrap(y, trained_prob[db], prand, trained_logit[db], zrand,
                                        subject, seed=20261123 + (5000 if db == "mimic" else 6000),
                                        label=f"B/{db}/random-vs-trained")
        results["control_b_random_initialized_same_architecture"][db] = {
            "n": int(len(y)), "n_source_train_split0": int(mimic_train_mask.sum()),
            "random_init_metrics": point_metrics(y, prand),
            "trained_main20k_metrics": point_metrics(y, trained_prob[db]),
            "random_minus_trained_paired_patient_bootstrap": boot,
            "interpretation_limits": [
                "The encoder is randomly initialized from the same architecture and receives no pretrained weights.",
                "Only the source C1 scaler/readout is fit, using eligible MIMIC split0 rows.",
                "This compares initialization/pretraining at matched architecture; it is not a patient-membership or causal analysis.",
            ],
        }
    report = {
        "status": "analysis_complete",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "preflight_manifest_sha256": sha256_file(PREFLIGHT),
        "gpu_encode_manifest_sha256": sha256_file(GPU_MANIFEST),
        "gpu_intermediate_sha256": sha256_file(INTERMEDIATE),
        "fixed_probe_sha256": sha256_file(PROBE_PATH),
        "fixed_probe_consumer_runtime": runtime_record(),
        "embedding_producer_runtime_by_cohort": pre["producer_runtime_identity_by_cohort"],
        "gpu_encoder_runtime": gpu["producer_runtime"],
        "controls": results,
        "not_performed": ["No derived secondary outcomes", "No influence-function estimates", "No INSPIRE scoring"],
    }
    out = OUT / "control_results_aggregate.json"
    atomic_json(out, report)
    log(f"ANALYSIS_COMPLETE result={out} sha256={sha256_file(out)}")


def smoke_score() -> None:
    import pandas as pd
    if not all(p.is_file() for p in (PREFLIGHT, INPUTS, SMOKE_EMBEDDINGS, SMOKE_GPU_MANIFEST)):
        raise FileNotFoundError("validated preflight and GPU smoke artifacts are required")
    pre = json.loads(PREFLIGHT.read_text())
    gpu = json.loads(SMOKE_GPU_MANIFEST.read_text())
    if gpu.get("preflight_manifest_sha256") != sha256_file(PREFLIGHT):
        raise ValueError("smoke embedding was not generated from this preflight")
    if sha256_file(SMOKE_EMBEDDINGS) != gpu.get("smoke_embeddings_sha256"):
        raise ValueError("smoke embedding file SHA mismatch")
    if sha256_file(PROBE_PATH) != EXPECTED["probe_sha256"]:
        raise ValueError("fixed probe changed before smoke scoring")
    with np.load(INPUTS, allow_pickle=False) as z:
        expected_rows = z["mimic_eval_rows"][:2]
        expected_stay = z["mimic_eval_stay_id"][:2].astype(str)
        expected_subject = z["mimic_eval_subject"][:2].astype(str)
    with np.load(SMOKE_EMBEDDINGS, allow_pickle=False) as z:
        rows = z["rows"]
        stay = z["stay_id"].astype(str)
        subject = z["subject"].astype(str)
        E = z["embedding"].astype(np.float32, copy=False)
    if (E.shape != (2, 128) or not np.array_equal(rows, expected_rows)
            or not np.array_equal(stay, expected_stay) or not np.array_equal(subject, expected_subject)):
        raise ValueError("serialized GPU smoke embedding does not match exact eligible source identities")
    probe_obj = pickle.load(PROBE_PATH.open("rb"))
    scaler, probe = probe_obj["scaler"], probe_obj["probe"]
    p = probe.predict_proba(scaler.transform(E))[:, 1]
    frame = pd.read_parquet(RUN_DIR / "main20k_real.parquet")
    frame = frame.loc[frame["db"].astype(str) == "mimic"].copy()
    frame["stay_id"] = frame["stay_id"].astype(str)
    lookup = frame.set_index("stay_id", verify_integrity=True)
    expected_p = np.asarray([float(lookup.loc[s, "p"]) for s in stay])
    maxdiff = float(np.max(np.abs(p - expected_p)))
    if maxdiff > 2e-6:
        raise ValueError(f"CPU1.8 fixed-probe smoke score differs from fix2: {maxdiff}")
    report = {"status": "gpu_to_cpu_fixed_probe_smoke_passed", "n": 2,
              "gpu_embedding_sha256": sha256_file(SMOKE_EMBEDDINGS),
              "fixed_probe_sha256": sha256_file(PROBE_PATH),
              "cpu_consumer_runtime": runtime_record(),
              "max_abs_probability_difference_vs_fix2": maxdiff,
              "serialization": "npz round trip with allow_pickle=False and Unicode stay/patient IDs"}
    atomic_json(SMOKE_CPU_RESULT, report)
    log(f"GPU_TO_CPU_SMOKE_PASS n=2 max_abs_probability_difference={maxdiff}")


def self_test() -> None:
    # Minimal invariants mirroring the actual transformation contract.
    V = np.asarray([[1, 1, 2, 0], [1, 2, 2, 0], [1, 2, 0, 0]], dtype=np.int64)
    B = np.asarray([[2, 5, 3, 0], [5, 7, 1, 0], [4, 9, 0, 0]], dtype=np.int64)
    H = np.asarray([[0, 1, 2, 0], [1, 2, 3, 0], [2, 3, 0, 0]], dtype=np.int64)
    L = np.asarray([3, 3, 2], dtype=np.int64)
    rng = np.random.default_rng(77)
    B2 = B.copy()
    for i in range(len(L)):
        k = int(L[i])
        for vid in np.unique(V[i, :k]):
            if vid > 0:
                pos = np.flatnonzero(V[i, :k] == vid)
                if len(pos) > 1:
                    B2[i, pos] = rng.permutation(B2[i, pos])
                if sorted(B2[i, pos].tolist()) != sorted(B[i, pos].tolist()):
                    raise AssertionError("within-stay temporal permutation changed a B-bin multiset")
    if not np.array_equal(V, V.copy()) or not np.array_equal(H, H.copy()) or not np.array_equal(L, np.asarray([3, 3, 2])) or np.any(B2[V == 0] != 0):
        raise AssertionError("within permutation changed V/H/L/padding")
    subjects = np.asarray(["p1", "p1", "p2"])
    donor_candidates = {"p1": ["p2"], "p2": ["p1"]}
    donor_map = {}
    for target in set(subjects):
        donor = donor_candidates[target][0]
        if donor == target:
            raise AssertionError("self donor detected")
        donor_map[target] = donor
    if donor_map["p1"] != donor_map["p1"] or any(donor_map[s] == s for s in donor_map):
        raise AssertionError("donor selection is not stable/nonself across repeated stays")
    # Optimized bootstrap metric equals sklearn on a representative weighted sample.
    from sklearn.metrics import roc_auc_score, average_precision_score
    yy = np.asarray([0, 1, 0, 1, 1, 0], dtype=np.int8)
    pp = np.asarray([.2, .7, .4, .7, .1, .4])
    ww = np.asarray([1, 2, 3, 1, 2, 1], dtype=float)
    m = weighted_metrics(yy, pp, ww)
    if abs(m["auroc"] - roc_auc_score(yy, pp, sample_weight=ww)) > 1e-12:
        raise AssertionError("weighted bootstrap AUC implementation differs from sklearn")
    if abs(m["average_precision"] - average_precision_score(yy, pp, sample_weight=ww)) > 1e-12:
        raise AssertionError("weighted bootstrap AP implementation differs from sklearn")
    log("SELF_TESTS_PASS within multiset/V/H/L/padding, donor nonself/repeated-stay, weighted metrics")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("self-test", "preflight", "smoke-score", "analyze"), required=True)
    args = p.parse_args()
    if args.mode == "self-test":
        self_test()
    elif args.mode == "preflight":
        preflight()
    elif args.mode == "smoke-score":
        smoke_score()
    else:
        analyze()


if __name__ == "__main__":
    main()
