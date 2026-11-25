#!/usr/bin/env python3
"""Reuse frozen MIMIC source heads for INSPIRE and availability scores from merged69."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import runpy
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
ENGINE = REV / "engine"
CODE = ENGINE / "code"
SNAPSHOTS = ENGINE / "code_snapshots"
MERGED = SNAPSHOTS / "embedding_manifest_recovered_v4_2_merged69_10fa455935a8.json"
MERGED_SHA = "10fa455935a87d296562cbc1b00c9b61563b514ba3dc72667647973a65db02fa"
BASE_ENGINE = CODE / "v4_engine_registry_maskfix_recovered_v4_2_resume_vocabadapter_21aae5dd.py"
BASE_ENGINE_SHA = "21aae5dd6b013a6fc9a4943c50799607e30690c611e322cb5799bfa21a8e95da"
SOURCE_GATE = REV / "temporal/results/eligibility/status_consistent_v4_1/landmark_v4_1"
SOURCE_GATE_SHA = "ea2b55bc3cf071d48b019d4ba97b4a3dc53d5d7fb3f6ac49a9e7c26007064cb5"
INS_GATE = REV / "temporal/results/eligibility/inspire_exploratory_locked_v4_1"
INS_GATE_SHA = "97c7aeacacb2af9cfce63086d6dc5b9ce9838292a4197e43d3ec98d7f533c019"
AVAIL_GATE = REV / "temporal/results/eligibility/availability_testonly_reviewed_v4_1"
AVAIL_GATE_SHA = "4a8c73898cf770a98ece8a6bc6d79185f1b8dfb82c70fd90c751fbb17001e9ea"
ENCODER_SHA = "59c3f4059662fb9e1d81baf20120c8668a761a926f5cab5ff4e654f7c16c1abc"
CORE_ORIGIN = "status_consistent_primary_core_fix2"
BIG_ORIGIN = "big_readout_frozen20_v4_2"

GROUPS = {
    "inspire_core": {
        "gate": INS_GATE, "gate_sha": INS_GATE_SHA,
        "targets": ("inspire_postop_v4_1", "inspire_broad_surgery_associated_v4_1"),
        "models": ("main20k", "abl_const20k", "common20k"),
        "origin": CORE_ORIGIN, "run_id": "inspire_core_source_reuse_m69_20261123",
    },
    "inspire_big": {
        "gate": INS_GATE, "gate_sha": INS_GATE_SHA,
        "targets": ("inspire_postop_v4_1", "inspire_broad_surgery_associated_v4_1"),
        "models": ("big50m", "big50m_full"),
        "origin": BIG_ORIGIN, "run_id": "inspire_big_source_reuse_m69_20261123",
    },
    "availability_main": {
        "gate": AVAIL_GATE, "gate_sha": AVAIL_GATE_SHA,
        "targets": ("mimic_available_by24_v4_1", "mimic_available_by24_fixedselection_v4_1"),
        "models": ("main20k",),
        "origin": CORE_ORIGIN, "run_id": "availability_main_source_reuse_m69_20261123",
    },
}


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(data).hexdigest()


def atomic_write(path: Path, data: bytes, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(f"refusing overwrite: {path}")
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, value: dict, replace: bool = False) -> None:
    atomic_write(path, (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(),
                 replace=replace)


def pinned_engine() -> Path:
    if file_sha(MERGED) != MERGED_SHA or file_sha(BASE_ENGINE) != BASE_ENGINE_SHA:
        raise ValueError("merged69 or adapter engine SHA256 changed")
    source = BASE_ENGINE.read_text(encoding="utf-8")
    needle = 'ENGINE / "results/embedding_manifest.json"'
    if source.count(needle) != 3:
        raise ValueError("expected three live-manifest expressions in base engine")
    body = source.replace(needle, f"Path({json.dumps(str(MERGED))})").encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    target = SNAPSHOTS / f"v4_engine_registry_maskfix_vocabadapter_merged69_{digest[:12]}.py"
    if target.exists():
        if file_sha(target) != digest:
            raise ValueError(f"pinned engine path collision: {target}")
    else:
        atomic_write(target, body)
    return target


def read_gate(gate_dir: Path, expected_sha: str) -> tuple[dict, Path, str]:
    manifest_path = gate_dir / "eligibility_manifest.json"
    digest = file_sha(manifest_path)
    if digest != expected_sha:
        raise ValueError(f"gate SHA256 mismatch: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8")), manifest_path, digest


def binding(gate_dir: Path, gate_sha: str, db: str) -> dict:
    manifest, path, digest = read_gate(gate_dir, gate_sha)
    entry = manifest.get("cohorts", {}).get(db)
    table = gate_dir / f"{db}.parquet"
    if not isinstance(entry, dict) or not table.is_file():
        raise FileNotFoundError(f"{db}: missing gate entry/table in {gate_dir}")
    return {
        "cohort": db,
        "eligibility_manifest_path": str(path),
        "eligibility_manifest_sha256": digest,
        "eligibility_table_path": str(table),
        "eligibility_table_sha256": file_sha(table),
        "sourcehashes": entry.get("sourcehashes", {}),
        "cohort_entry_sha256": canonical_sha(entry),
        "cohort_entry": entry,
    }


def bridge(target_gate: Path, target_sha: str) -> dict:
    source = binding(SOURCE_GATE, SOURCE_GATE_SHA, "mimic")
    mirror = binding(target_gate, target_sha, "mimic")
    same_entry = source["cohort_entry"] == mirror["cohort_entry"]
    same_table = source["eligibility_table_sha256"] == mirror["eligibility_table_sha256"]
    same_npz = source["sourcehashes"].get("sequence_npz") == mirror["sourcehashes"].get("sequence_npz")
    if not (same_entry and same_table and same_npz):
        raise ValueError("MIMIC source/target-gate bridge differs beyond container")
    return {
        "source_cohort": "mimic",
        "source_gate_path": source["eligibility_manifest_path"],
        "source_gate_sha256": source["eligibility_manifest_sha256"],
        "source_table_path": source["eligibility_table_path"],
        "source_table_sha256": source["eligibility_table_sha256"],
        "source_mimic_entry_sha256": source["cohort_entry_sha256"],
        "target_gate_container_path": mirror["eligibility_manifest_path"],
        "target_gate_container_sha256": mirror["eligibility_manifest_sha256"],
        "target_mimic_mirror_table_path": mirror["eligibility_table_path"],
        "target_mimic_mirror_table_sha256": mirror["eligibility_table_sha256"],
        "target_mimic_mirror_entry_sha256": mirror["cohort_entry_sha256"],
        "target_mimic_mirror_used_for_source": False,
        "mimic_entry_exact": same_entry,
        "mimic_table_exact": same_table,
        "mimic_sequence_npz_hash_exact": same_npz,
        "source_sequence_npz_sha256": source["sourcehashes"].get("sequence_npz"),
        "only_container_sha_differs": (
            source["eligibility_manifest_sha256"] != mirror["eligibility_manifest_sha256"]),
    }


def coef_sha(fitted: dict) -> str:
    digest = hashlib.sha256()
    for name, value in (
        ("mean", fitted["scaler"].mean_), ("scale", fitted["scaler"].scale_),
        ("var", fitted["scaler"].var_), ("coef", fitted["probe"].coef_),
        ("intercept", fitted["probe"].intercept_), ("classes", fitted["probe"].classes_),
    ):
        arr = np.ascontiguousarray(np.asarray(value))
        digest.update(name.encode() + b"\0" + str(arr.dtype).encode() + b"\0")
        digest.update(json.dumps(list(arr.shape)).encode() + b"\0" + arr.tobytes())
    return digest.hexdigest()


def preflight_probe(engine: dict, entries: dict, tag: str, condition: str,
                    origin: str, eligible: pd.DataFrame, gate_entry: dict,
                    ds: dict) -> dict:
    source_paths = engine["resolve_cohort_paths"]("mimic", verify_exists=True)
    source_sha = file_sha(source_paths.sequence_path)
    engine["assert_sequence_npz_attestation"]("mimic", gate_entry, source_sha)
    split = np.asarray(ds["split"], dtype=np.int8)
    common_ids = engine["load_common_ids"]()[0] if condition == "common8_mask" else ()
    no_h0 = (engine["load_no_h0_therapy_ids_registered"](("mimic",))
             if condition == "no_h0_therapy" else {})
    variable_ids = (
        tuple(common_ids) if condition == "common8_mask" else
        tuple(no_h0["mimic"][name] for name in engine["NO_H0_THERAPY_CANONICAL_VARIABLES"])
        if condition == "no_h0_therapy" else ()
    )
    mask_spec = (engine["no_h0_therapy_input_mask_spec"]("mimic", variable_ids)
                 if condition == "no_h0_therapy" else None)
    mask_sha = (engine["canonical_json_sha256"](mask_spec) if mask_spec is not None
                else engine["input_mask_hash"](condition, variable_ids))
    rows = engine["condition_rows"](ds, condition, variable_ids)
    matched = engine["align_eligible_rows"](
        "mimic", ds, eligible, rows, condition=condition, common_ids=variable_ids)
    matched["split"] = split[matched["rows"].to_numpy()]
    train = matched["split"].to_numpy() == 0
    test = matched["split"].to_numpy() == 2
    groups = matched["subject"].to_numpy(dtype=str)
    train_groups, test_groups = set(groups[train]), set(groups[test])
    y = matched["y_mort"].to_numpy(dtype=np.int8)
    if (not train.any() or not test.any() or train_groups & test_groups
            or len(np.unique(y[train])) != 2):
        raise ValueError(f"invalid MIMIC source fit/test rows for {tag}/{condition}")
    source_rows = matched.loc[train, "rows"].to_numpy(dtype=np.int64)
    train_identity_sha = engine["sha256_ids"](
        "mimic", source_rows, np.asarray(ds["stay_id"])[source_rows],
        np.asarray(ds["subject"]).astype(str)[source_rows])
    train_label_sha = hashlib.sha256(np.asarray(y[train], dtype=np.int8).tobytes()).hexdigest()
    row = entries.get(("mimic", tag, condition))
    if not isinstance(row, dict) or row.get("encoder_code_sha256") != ENCODER_SHA:
        raise ValueError(f"missing or untrusted MIMIC merged69 cache row for {tag}/{condition}")
    metadata = {
        "run_id": "source_probe_reuse_preflight",
        "model_tag": tag,
        "input_condition": condition,
        "checkpoint_sha256": file_sha(engine["MODELS"] / f"{tag}.pt"),
        "input_mask_sha256": mask_sha,
        **({"input_mask_spec": mask_spec} if mask_spec is not None else {}),
        "eligibility_manifest_sha256": gate_entry["eligibility_manifest_sha256"],
        "eligibility_table_sha256": gate_entry["eligibility_table_sha256"],
        "source_npz_sha256": source_sha,
        "source_train_identity_sha256": train_identity_sha,
        "source_train_label_sha256": train_label_sha,
        "runtime_identity_sha256": str(row.get("runtime_identity_sha256", "")),
        "source_train_n": int(train.sum()),
        "source_train_patient_n": len(train_groups),
        "prediction_scope": engine["prediction_scope"](
            gate_entry.get("evidence"), diagnostic_only=False,
            landmark_ready=bool(gate_entry.get("retrospective_landmark_prediction_allowed"))),
        "temporal_evidence": gate_entry.get("evidence"),
        "input_scope": engine["INPUT_SCOPE"],
        "clinical_input_availability_verified": engine["CLINICAL_INPUT_AVAILABILITY_VERIFIED"],
    }
    origin_meta = {**metadata, "run_id": origin}
    artifact_path = engine["source_probe_artifact_path"](origin_meta)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"source probe missing for {tag}/{condition}: {artifact_path}")
    with artifact_path.open("rb") as stream:
        raw = pickle.load(stream)
    feature_dim = int(np.asarray(raw["scaler"].mean_).shape[0])
    loaded = engine["load_reusable_source_probe"](metadata, origin, feature_dim)
    return {
        "model_tag": tag,
        "input_condition": condition,
        "origin_run_id": origin,
        "artifact_path": loaded["artifact_path"],
        "artifact_sha256": loaded["artifact_sha256"],
        "source_eligibility_manifest_sha256": metadata["eligibility_manifest_sha256"],
        "source_eligibility_table_sha256": metadata["eligibility_table_sha256"],
        "source_npz_sha256": source_sha,
        "source_train_identity_sha256": train_identity_sha,
        "source_train_label_sha256": train_label_sha,
        "input_mask_sha256": mask_sha,
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "runtime_identity_sha256": metadata["runtime_identity_sha256"],
        "source_probe_coefficient_fingerprint_sha256": coef_sha(loaded),
        "feature_dim": feature_dim,
    }


def prepare(group: str):
    if group not in GROUPS:
        raise ValueError(f"unknown group {group}")
    config = GROUPS[group]
    engine_path = pinned_engine()
    sys.path.insert(0, str(CODE))
    engine = runpy.run_path(str(engine_path), run_name=f"reuse_preflight_{group}")
    manifest, entries = engine["load_encoding_manifest"]()
    if (file_sha(MERGED) != MERGED_SHA or len(manifest.get("models", [])) != 69
            or manifest.get("encoder_code_sha256") != ENCODER_SHA):
        raise ValueError("merged69 source manifest does not match the frozen identity")
    target_manifest, target_manifest_path, target_manifest_sha = read_gate(
        config["gate"], config["gate_sha"])
    source_manifest, source_manifest_path, source_manifest_sha = read_gate(
        SOURCE_GATE, SOURCE_GATE_SHA)
    bridge_record = bridge(config["gate"], config["gate_sha"])
    route = {"mimic": SOURCE_GATE}
    gate_sha_by_cohort = {"mimic": SOURCE_GATE_SHA}
    for db in config["targets"]:
        route[db] = config["gate"]
        gate_sha_by_cohort[db] = config["gate_sha"]
    bindings = {db: binding(route[db], gate_sha_by_cohort[db], db)
                for db in ("mimic", *config["targets"])}

    loader = engine["load_ready_eligibility"]
    source_eligible, source_entry = loader("mimic", SOURCE_GATE, diagnostic_only=False)
    if source_entry["eligibility_manifest_sha256"] != source_manifest_sha:
        raise ValueError("source MIMIC did not load from the original status-consistent gate")
    source_paths = engine["resolve_cohort_paths"]("mimic", verify_exists=True)
    source_ds = engine["load_sequence_registered"]("mimic", source_paths)
    engine["validate_sequence"](source_ds, "mimic")

    expected = []
    probe_records = []
    for tag in config["models"]:
        for condition in engine["model_conditions"](tag):
            expected.append(f"{tag}_{condition}")
            for db in ("mimic", *config["targets"]):
                row = entries.get((db, tag, condition))
                cache = Path(str(row.get("cache", ""))) if isinstance(row, dict) else Path("")
                if (not row or int(row.get("n_rows", 0)) <= 0 or not cache.is_absolute()
                        or not cache.is_file()
                        or not cache.resolve().is_relative_to((ENGINE / "cache").resolve())):
                    raise ValueError(f"merged69 cache row/path missing: {db}/{tag}/{condition}")
            probe_records.append(preflight_probe(
                engine, entries, tag, condition, config["origin"], source_eligible,
                source_entry, source_ds))

    target_summary = {}
    for db in config["targets"]:
        selected, entry = loader(db, config["gate"], diagnostic_only=False)
        if not len(selected):
            raise ValueError(f"{db}: target eligibility has no positive rows")
        paths = engine["resolve_cohort_paths"](db, verify_exists=True)
        engine["assert_sequence_npz_attestation"](db, entry, file_sha(paths.sequence_path))
        if (target_manifest["cohorts"][db].get("retrospective_landmark_prediction_allowed")
                is not True or entry.get("clinical_input_availability_verified") is not False):
            raise ValueError(f"{db}: target gate does not authorize this measurement-time score")
        target_summary[db] = {"eligible_rows": int(len(selected)),
                              "eligibility_manifest_sha256": entry["eligibility_manifest_sha256"],
                              "eligibility_table_sha256": entry["eligibility_table_sha256"]}

    proof = {
        "status": "source_probe_reuse_preflight_passed",
        "group": group,
        "run_id": config["run_id"],
        "models": list(config["models"]),
        "targets": list(config["targets"]),
        "expected_analyses": expected,
        "target_eligibility": target_summary,
        "embedding_manifest_path": str(MERGED),
        "embedding_manifest_sha256": MERGED_SHA,
        "pinned_engine_path": str(engine_path),
        "pinned_engine_sha256": file_sha(engine_path),
        "source_gate_path": str(source_manifest_path),
        "source_gate_sha256": source_manifest_sha,
        "target_gate_path": str(target_manifest_path),
        "target_gate_sha256": target_manifest_sha,
        "source_probe_origin_run_id": config["origin"],
        "source_probe_refit_forbidden": True,
        "source_probe_artifact_preflight": probe_records,
        "source_bridge_equivalence": bridge_record,
        "eligibility_bindings_by_cohort": {
            db: {key: value for key, value in b.items() if key != "cohort_entry"}
            for db, b in bindings.items()
        },
        "clinical_input_availability_verified": False,
        "clinical_ready": False,
    }
    aux = {"engine": engine, "config": config, "route": route,
           "gate_sha_by_cohort": gate_sha_by_cohort, "bindings": bindings,
           "bridge": bridge_record, "probe_records": probe_records,
           "target_manifest_path": target_manifest_path}
    return proof, aux


def exact_source_reproduction(engine: dict, current_manifest_path: Path,
                              origin_run_id: str, analyses: list[str]) -> dict:
    origin_dirs = list((ENGINE / "results/prediction_runs").glob(origin_run_id + "-*"))
    if len(origin_dirs) != 1:
        raise ValueError(f"{origin_run_id}: expected one frozen source run, found {len(origin_dirs)}")
    origin_manifest_path = origin_dirs[0] / "prediction_manifest.json"
    origin_manifest = json.loads(origin_manifest_path.read_text(encoding="utf-8"))
    current_manifest = json.loads(current_manifest_path.read_text(encoding="utf-8"))
    old_rows = {(r.get("db"), r.get("analysis_id")): r
                for r in origin_manifest.get("predictions", [])}
    new_rows = {(r.get("db"), r.get("analysis_id")): r
                for r in current_manifest.get("predictions", [])}
    checks = []
    identity = ["rows", "stay_id", "subject", "split", "y", "endpoint", "role"]
    for analysis in analyses:
        key = ("mimic", analysis)
        old, new = old_rows.get(key), new_rows.get(key)
        if not old or not new:
            raise ValueError(f"source reproduction row missing: {key}")
        old_df = pd.read_parquet(old["prediction_file"])
        new_df = pd.read_parquet(new["prediction_file"])
        old_df = old_df.loc[(old_df["db"].astype(str) == "mimic")
                            & (old_df["analysis_id"].astype(str) == analysis)]
        new_df = new_df.loc[(new_df["db"].astype(str) == "mimic")
                            & (new_df["analysis_id"].astype(str) == analysis)]
        sort_cols = ["rows", "stay_id", "subject"]
        old_df = old_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        new_df = new_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        if old_df.duplicated(subset=sort_cols).any() or new_df.duplicated(subset=sort_cols).any():
            raise ValueError(f"duplicate MIMIC prediction identity for {analysis}")
        if len(old_df) != len(new_df):
            raise ValueError(f"source prediction count mismatch for {analysis}: {len(old_df)} vs {len(new_df)}")
        for col in identity:
            if old_df[col].astype(str).tolist() != new_df[col].astype(str).tolist():
                raise ValueError(f"source prediction identity mismatch for {analysis}/{col}")
        old_p = old_df["p"].to_numpy(dtype=np.float64)
        new_p = new_df["p"].to_numpy(dtype=np.float64)
        if not np.array_equal(old_p, new_p, equal_nan=True):
            diff = float(np.max(np.abs(old_p - new_p))) if len(old_p) else None
            raise ValueError(f"source probabilities not bitwise reproduced for {analysis}: {diff}")
        checks.append({
            "db": "mimic", "analysis_id": analysis,
            "origin_run_id": origin_run_id,
            "origin_manifest_sha256": file_sha(origin_manifest_path),
            "origin_prediction_file_sha256": file_sha(Path(old["prediction_file"])),
            "current_prediction_file_sha256": file_sha(Path(new["prediction_file"])),
            "n_source_predictions": int(len(new_df)),
            "identity_exact": True, "probabilities_bitwise_exact": True,
        })
    return {"status": "source_probabilities_reproduced_exactly",
            "analysis_count": len(checks), "comparisons": checks}


def run(group: str, preflight_only: bool) -> None:
    proof, aux = prepare(group)
    print(json.dumps(proof, sort_keys=True), flush=True)
    print("SOURCE_PROBE_REUSE_PREFLIGHT_COMPLETE", flush=True)
    if preflight_only:
        return

    engine, config = aux["engine"], aux["config"]
    globals_dict = engine["run_predict"].__globals__
    original_loader = globals_dict["load_ready_eligibility"]
    bindings = aux["bindings"]
    protocol_holder = {}

    def routed_loader(db, _eligibility_dir, diagnostic_only=False):
        if db not in aux["route"]:
            raise ValueError(f"missing exact gate route for {db}")
        selected, entry = original_loader(
            db, aux["route"][db], diagnostic_only=diagnostic_only)
        entry["eligibility_manifest_path"] = bindings[db]["eligibility_manifest_path"]
        entry["eligibility_table_path"] = bindings[db]["eligibility_table_path"]
        return selected, entry

    original_protocol = globals_dict["build_prediction_run_protocol"]

    def bound_protocol(args, eligibility_dir, encoder_hash, engine_hash):
        protocol = original_protocol(args, eligibility_dir, encoder_hash, engine_hash)
        container_path = Path(eligibility_dir) / "eligibility_manifest.json"
        protocol["eligibility_manifest_container_path"] = str(container_path)
        protocol["eligibility_manifest_container_sha256"] = protocol["eligibility_manifest_sha256"]
        protocol["eligibility_manifest_path_by_cohort"] = {
            db: bindings[db]["eligibility_manifest_path"] for db in args.cohorts}
        protocol["eligibility_manifest_sha256_by_cohort"] = {
            db: bindings[db]["eligibility_manifest_sha256"] for db in args.cohorts}
        protocol["eligibility_table_path_by_cohort"] = {
            db: bindings[db]["eligibility_table_path"] for db in args.cohorts}
        protocol["eligibility_table_sha256_by_cohort"] = {
            db: bindings[db]["eligibility_table_sha256"] for db in args.cohorts}
        protocol["source_npz_sha256_attested_by_cohort"] = {
            db: bindings[db]["sourcehashes"].get("sequence_npz") for db in args.cohorts}
        protocol["eligibility_gate_route"] = {
            db: ("original_status_consistent_source_gate" if db == "mimic"
                 else "reviewed_target_gate") for db in args.cohorts}
        protocol["source_bridge_equivalence"] = aux["bridge"]
        protocol["source_probe_origin_run_id"] = config["origin"]
        protocol["source_probe_artifact_preflight"] = proof["source_probe_artifact_preflight"]
        protocol["source_probe_refit_forbidden"] = True
        protocol["embedding_manifest_path"] = str(MERGED)
        protocol["embedding_manifest_sha256"] = MERGED_SHA
        protocol.pop("run_protocol_sha256", None)
        protocol["run_protocol_sha256"] = canonical_sha(protocol)
        protocol_holder["protocol"] = protocol
        return protocol

    def forbidden_fit(*_args, **_kwargs):
        raise RuntimeError("source-probe fitting is forbidden in a reuse run")

    globals_dict["load_ready_eligibility"] = routed_loader
    globals_dict["build_prediction_run_protocol"] = bound_protocol
    globals_dict["fit_source_probe"] = forbidden_fit

    args = engine["parse_args"]([
        "--mode", "predict", "--models", *config["models"],
        "--cohorts", "mimic", *config["targets"],
        "--eligibility-dir", str(config["gate"]),
        "--run-id", config["run_id"],
        "--reuse-source-probes-from-run-id", config["origin"],
        "--device", "cpu", "--batch-size", "128",
    ])
    status = engine["run_predict"](args)
    if status != 0:
        raise SystemExit(status)
    protocol = protocol_holder["protocol"]
    prediction_dir = engine["prediction_run_directory"](protocol)
    manifest_path = prediction_dir / "prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest.get("predictions", []):
        db = str(row.get("db", ""))
        b = bindings[db]
        if (row.get("eligibility_manifest_sha256") != b["eligibility_manifest_sha256"]
                or row.get("eligibility_table_sha256") != b["eligibility_table_sha256"]):
            raise ValueError(f"{db}: prediction row gate hashes do not match routed files")
        row["eligibility_manifest_path"] = b["eligibility_manifest_path"]
        row["eligibility_table_path"] = b["eligibility_table_path"]
    for row in manifest.get("readiness", []):
        db = str(row.get("db", ""))
        if db in bindings:
            for field in ("eligibility_manifest_path", "eligibility_manifest_sha256",
                          "eligibility_table_path", "eligibility_table_sha256"):
                row[field] = bindings[db][field]
    manifest["eligibility_bindings_by_cohort"] = {
        db: {key: value for key, value in b.items() if key != "cohort_entry"}
        for db, b in bindings.items()
    }
    manifest["source_bridge_equivalence"] = aux["bridge"]
    manifest["embedding_manifest_path"] = str(MERGED)
    manifest["embedding_manifest_sha256"] = MERGED_SHA
    manifest["source_probe_refit_forbidden"] = True

    reuse_path = manifest.get("source_probe_reuse_manifest")
    if reuse_path:
        reuse_path = Path(reuse_path)
        reused = json.loads(reuse_path.read_text(encoding="utf-8"))
        reused["eligibility_bindings_by_cohort"] = manifest["eligibility_bindings_by_cohort"]
        reused["source_bridge_equivalence"] = aux["bridge"]
        reused["source_probe_refit_forbidden"] = True
        atomic_json(reuse_path, reused, replace=True)

    analyses = [f"{tag}_{condition}" for tag in config["models"]
                for condition in engine["model_conditions"](tag)]
    target_grid = {(db, analysis) for db in config["targets"] for analysis in analyses}
    rows = {(r.get("db"), r.get("analysis_id")): r for r in manifest.get("predictions", [])}
    positive = {key for key in target_grid
                if key in rows and int(rows[key].get("n_predictions", 0)) > 0}
    if positive != target_grid:
        raise ValueError(f"target positive grid incomplete: {sorted(target_grid - positive)}")
    source_grid = {("mimic", analysis) for analysis in analyses}
    if any(key not in rows or int(rows[key].get("n_predictions", 0)) <= 0 for key in source_grid):
        raise ValueError("MIMIC source positive grid incomplete")
    for key in source_grid | target_grid:
        row = rows[key]
        if (row.get("source_probe_origin_run_id") != config["origin"]
                or row.get("source_probe_reused_from_prior_run") is not True
                or row.get("clinical_input_availability_verified") is not False
                or row.get("clinical_ready") is not False):
            raise ValueError(f"frozen source probe/readiness provenance mismatch: {key}")

    reproduction = exact_source_reproduction(
        engine, manifest_path, config["origin"], analyses)
    manifest["frozen_source_probability_reproduction"] = reproduction
    atomic_json(manifest_path, manifest, replace=True)
    completion = {
        "status": "target_source_probe_reuse_complete",
        "group": group,
        "run_id": config["run_id"],
        "run_protocol_sha256": protocol["run_protocol_sha256"],
        "prediction_manifest_path": str(manifest_path),
        "prediction_manifest_sha256": file_sha(manifest_path),
        "embedding_manifest_path": str(MERGED),
        "embedding_manifest_sha256": MERGED_SHA,
        "models": list(config["models"]),
        "targets": list(config["targets"]),
        "source_probe_origin_run_id": config["origin"],
        "source_probe_refit_forbidden": True,
        "fit_source_probe_calls": 0,
        "positive_target_row_count": len(positive),
        "expected_positive_target_row_count": len(target_grid),
        "frozen_source_probability_reproduction": reproduction,
        "source_bridge_equivalence": aux["bridge"],
        "clinical_input_availability_verified": False,
        "clinical_ready": False,
        "scope_note": ("same-original-heldout score sensitivity only"
                       if group == "availability_main"
                       else "retrospective measurement-time readout; not real-time availability"),
    }
    completion_path = prediction_dir / "target_source_probe_reuse_completion.json"
    atomic_json(completion_path, completion)
    print(f"TARGET_SOURCE_PROBE_REUSE_COMPLETE={completion_path}", flush=True)
    print(json.dumps(completion, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=sorted(GROUPS), required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run(args.group, args.preflight_only)


if __name__ == "__main__":
    main()
