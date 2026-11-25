#!/usr/bin/env python3
"""Run recovered external paired CIs and reuse identical completed MIMIC CIs.

Only aggregate results are written. Saved predictions are validated by the
existing paired_metrics_v4 loader; no training, encoding, or raw tables occur.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import runpy
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
PRED_DIR = REV / "engine/results/prediction_runs/recovered_external_primary_v4_2_resume3-04b21bfc05943b7f"
PRED_MANIFEST = PRED_DIR / "prediction_manifest.json"
OLD_PRED_DIR = REV / "engine/results/prediction_runs/status_consistent_primary_core_fix2-7da980ffcbd8f00b"
OLD_PRED_MANIFEST = OLD_PRED_DIR / "prediction_manifest.json"
ELIG_DIR = REV / "temporal/results/eligibility/recovered_reviewed_v4_2"
ELIG_MANIFEST = ELIG_DIR / "eligibility_manifest.json"
EMBED_SNAPSHOT = REV / "engine/code_snapshots/embedding_manifest_recovered_v4_2_merged41_b2eec1f7aca1.json"
IDENTITY_ATTESTATION = REV / "admin/subject_identity_attestations_v4_2.json"
METRICS_CODE = REV / "engine/code/paired_metrics_v4.py"
CORE_METRICS_DIR = REV / "engine/results/paired_metrics/status_consistent_primary_core_fix2_1000boot"
CORE_METRICS = CORE_METRICS_DIR / "paired_core_metrics.csv"
CORE_FLOW = CORE_METRICS_DIR / "paired_core_pair_flow.csv"
CORE_CONTEXT = CORE_METRICS_DIR / "paired_core_metrics_run_context.json"
EXPECTED = {
    "prediction_manifest": "4a52de662bf1e28c165c0387d8a95fa3208afc46e255c1a57e6526ee61d78340",
    "eligibility_manifest": "d302ce0a493121f9336f40fd58c5dc63a75a663492293c0153b4ba05efbb8e8f",
    "embedding_snapshot": "b2eec1f7aca1857bda440ddb5ec48e477b2a20f753d5dea28b7715f7a1a243fe",
    "identity_attestation": "199cdf9dfeb277ef49c30f4a4648a6b06bbf1a65958cd1cf84b36f986bb9cd55",
}
ANALYSIS_IDS = (
    "main20k_real", "main20k_constant", "main20k_common8_mask",
    "main20k_no_h0_therapy", "abl_const20k_real", "abl_const20k_constant",
    "common20k_common8_mask",
)
COHORTS = ("nwicu_discharge_v4_1", "sicdb_zero_offset_v4_2", "eicu_semantic_v4_2")
EXPECTED_POPULATIONS = {
    "nwicu_discharge_v4_1": {"rows": 19972, "events": 2226, "endpoint": "hospital"},
    "sicdb_zero_offset_v4_2": {"rows": 6774, "events": 402, "endpoint": "icu"},
    "eicu_semantic_v4_2": {"rows": 130544, "events": 11943, "endpoint": "hospital"},
}
BOOTSTRAP = 1000
SEED = 20261122
METRIC_COLUMNS = ("auroc", "average_precision", "brier", "logloss",
                  "calibration_slope", "calibration_intercept", "offset_intercept")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def atomic_npz(path: Path, **arrays) -> None:
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def extract_sequence_npz_hash(gate_entry: dict) -> str:
    """Use the documented recovered gate shape; do not guess alternate names."""
    sourcehashes = gate_entry.get("sourcehashes")
    if not isinstance(sourcehashes, dict):
        raise ValueError("gate cohort entry lacks the sourcehashes object")
    value = sourcehashes.get("sequence_npz")
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("gate sourcehashes.sequence_npz must be a lowercase SHA256")
    return value


def load_and_verify_bindings() -> tuple[dict, dict, dict, dict]:
    checks = {
        "prediction_manifest": (PRED_MANIFEST, EXPECTED["prediction_manifest"]),
        "eligibility_manifest": (ELIG_MANIFEST, EXPECTED["eligibility_manifest"]),
        "embedding_snapshot": (EMBED_SNAPSHOT, EXPECTED["embedding_snapshot"]),
        "identity_attestation": (IDENTITY_ATTESTATION, EXPECTED["identity_attestation"]),
    }
    actual = {}
    for name, (path, expected) in checks.items():
        digest = sha256_file(path)
        if digest != expected:
            raise ValueError(f"{name} SHA256 mismatch: {digest}")
        actual[name] = {"path": str(path), "sha256": digest}
    manifest = json.loads(PRED_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("run_id") != "recovered_external_primary_v4_2_resume3" or manifest.get("mode") != "predict":
        raise ValueError("unexpected recovered prediction run or mode")
    gate = json.loads(ELIG_MANIFEST.read_text(encoding="utf-8"))
    sidecar = json.loads(IDENTITY_ATTESTATION.read_text(encoding="utf-8"))
    if not isinstance(gate.get("cohorts"), dict) or not isinstance(sidecar.get("cohorts"), dict):
        raise ValueError("gate or identity sidecar has unsupported cohorts schema")
    if (sidecar.get("status") != "root_accepted_within_source_patient_keys_only"
            or not isinstance(sidecar.get("scope"), str)):
        raise ValueError("subject-identity attestation status/scope is missing")
    for db in COHORTS:
        gate_row = gate["cohorts"].get(db)
        identity = sidecar["cohorts"].get(db)
        if not isinstance(gate_row, dict) or not isinstance(identity, dict):
            raise ValueError(f"missing gate or identity sidecar cohort entry: {db}")
        sequence_hash = extract_sequence_npz_hash(gate_row)
        basis = identity.get("subject_identity_basis")
        if not isinstance(basis, dict):
            raise ValueError(f"{db}: subject_identity_basis object is missing")
        if (identity.get("eligibility_manifest_sha256") != EXPECTED["eligibility_manifest"]
                or identity.get("eligibility_table_sha256") not in {gate_row.get("eligibility_table_sha256"), gate_row.get("table_sha256")}
                or identity.get("sequence_npz_sha256") != sequence_hash
                or identity.get("metadata_relation") != "explicit identity supplement; original gate and prediction bytes untouched"):
            raise ValueError(f"{db}: identity supplement does not bind to gate source/table hashes")
        if (basis.get("resampling_unit") != "patient"
                or basis.get("identifier_scope") != "within this original data source"
                or basis.get("cross_source_patient_independence_verified") is not False
                or basis.get("cross_scope_patient_independence_verified") is not False):
            raise ValueError(f"{db}: unsupported patient identity or cross-source independence claim")
    positive = [r for r in manifest.get("predictions", [])
                if r.get("analysis_id") in ANALYSIS_IDS and int(r.get("n_predictions", 0)) > 0]
    expected_pairs = {(analysis, db) for analysis in ANALYSIS_IDS for db in ("mimic", *COHORTS)}
    actual_pairs = [(str(r.get("analysis_id")), str(r.get("db"))) for r in positive]
    if len(actual_pairs) != len(set(actual_pairs)) or set(actual_pairs) != expected_pairs:
        raise ValueError("prediction manifest does not contain exactly the seven analyses for MIMIC and three new cohorts")
    for db in COHORTS:
        gate_row = gate["cohorts"][db]
        table_hash = gate_row.get("eligibility_table_sha256") or gate_row.get("table_sha256")
        for analysis in ANALYSIS_IDS:
            entry = next(r for r in positive if r["analysis_id"] == analysis and r["db"] == db)
            if (entry.get("eligibility_manifest_sha256") != EXPECTED["eligibility_manifest"]
                    or entry.get("eligibility_table_sha256") != table_hash
                    or entry.get("source_npz_sha256") != extract_sequence_npz_hash(gate_row)
                    or entry.get("role") != "evaluation"):
                raise ValueError(f"{analysis}/{db}: source/gate binding mismatch")
    return manifest, gate, sidecar, actual


def find_entry(manifest: dict, analysis_id: str, db: str) -> dict:
    rows = [r for r in manifest["predictions"] if r.get("analysis_id") == analysis_id
            and r.get("db") == db and int(r.get("n_predictions", 0)) > 0]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one {analysis_id}/{db} entry")
    return rows[0]


def row_identity_digest(frame) -> str:
    digest = hashlib.sha256()
    cols = ("stay_id", "subject", "rows", "split", "y", "p", "endpoint", "role",
            "prediction_scope", "input_scope")
    for row in frame.loc[:, cols].sort_values("stay_id", kind="mergesort").itertuples(index=False, name=None):
        values = [str(row[0]), str(row[1]), int(row[2]), int(row[3]), int(row[4]),
                  format(float(row[5]), ".17g"), *[str(value) for value in row[6:]]]
        digest.update(json.dumps(values, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def verify_mimic_and_core_reuse(manifest: dict, metrics_code_hash: str) -> tuple[list, list, dict]:
    import pandas as pd
    sys.path.insert(0, str(METRICS_CODE.parent))
    from paired_metrics_v4 import sha256_ordered_prediction_rows, validate_predictions

    old_manifest_hash = sha256_file(OLD_PRED_MANIFEST)
    old_manifest = json.loads(OLD_PRED_MANIFEST.read_text(encoding="utf-8"))
    core_context = json.loads(CORE_CONTEXT.read_text(encoding="utf-8"))
    if (core_context.get("run_id") != "status_consistent_primary_core_fix2"
            or core_context.get("prediction_manifest_sha256") != old_manifest_hash
            or core_context.get("paired_metrics_code_sha256") != metrics_code_hash
            or core_context.get("bootstrap_replicates") != BOOTSTRAP
            or core_context.get("analysis_ids") != list(ANALYSIS_IDS)):
        raise ValueError("completed core MIMIC aggregate provenance is not compatible for reuse")
    baseline = pd.read_csv(CORE_METRICS)
    flow = pd.read_csv(CORE_FLOW)
    expected_combos = {tuple(pair) for pair in itertools.combinations(sorted(ANALYSIS_IDS), 2)}
    baseline_mimic = baseline.loc[baseline["db"].astype(str).eq("mimic")].copy()
    baseline_flow = flow.loc[flow["db"].astype(str).eq("mimic")].copy()
    observed_combos = set(zip(baseline_mimic["analysis_id_a"], baseline_mimic["analysis_id_b"]))
    if (len(baseline_mimic) != len(expected_combos) * len(METRIC_COLUMNS)
            or observed_combos != expected_combos
            or not baseline_mimic["n_bootstrap"].eq(BOOTSTRAP).all()
            or not baseline_flow["paired_identity_labels_verified"].astype(str).str.lower().eq("true").all()
            or len(baseline_flow) != len(expected_combos)):
        raise ValueError("completed core run lacks the full 21-pair MIMIC aggregate grid")
    parity = []
    old_cache, new_cache = {}, {}
    identity_columns = ("stay_id", "subject", "rows", "split", "y", "endpoint", "role",
                        "prediction_scope", "input_scope")
    for analysis in ANALYSIS_IDS:
        old_entry = find_entry(old_manifest, analysis, "mimic")
        new_entry = find_entry(manifest, analysis, "mimic")
        old_path, new_path = Path(old_entry["prediction_file"]), Path(new_entry["prediction_file"])
        for path, entry in ((old_path, old_entry), (new_path, new_entry)):
            actual_hash = sha256_file(path)
            if actual_hash != str(entry.get("prediction_file_sha256", "")).lower():
                raise ValueError(f"{analysis}/mimic: prediction parquet SHA256 mismatch")
        if old_path not in old_cache:
            old_cache[old_path] = pd.read_parquet(old_path, filters=[("db", "==", "mimic")])
        if new_path not in new_cache:
            new_cache[new_path] = pd.read_parquet(new_path, filters=[("db", "==", "mimic")])
        a = validate_predictions(old_cache[old_path], analysis)
        b = validate_predictions(new_cache[new_path], analysis)
        if (len(a) != int(old_entry["n_prediction_rows"])
                or len(b) != int(new_entry["n_prediction_rows"])
                or sha256_ordered_prediction_rows(a) != old_entry["ordered_row_digest_sha256"]
                or sha256_ordered_prediction_rows(b) != new_entry["ordered_row_digest_sha256"]):
            raise ValueError(f"{analysis}/mimic: row count/digest mismatch")
        a = a.sort_values("stay_id", kind="mergesort").reset_index(drop=True)
        b = b.sort_values("stay_id", kind="mergesort").reset_index(drop=True)
        equal = {col: a[col].astype(str).equals(b[col].astype(str))
                 if col in ("stay_id", "subject", "endpoint", "role", "prediction_scope", "input_scope")
                 else a[col].reset_index(drop=True).equals(b[col].reset_index(drop=True))
                 for col in identity_columns}
        if not all(equal.values()) or not np.array_equal(a["p"].to_numpy(float), b["p"].to_numpy(float)):
            raise ValueError(f"{analysis}/mimic: recovered predictions differ from core: {equal}")
        parity.append({"analysis_id": analysis, "n_predictions": int(len(a)),
                       "identity_and_labels_equal": True, "predictions_exact": True,
                       "eligibility_manifest_digest_compared": False,
                       "row_identity_prediction_sha256": row_identity_digest(a)})
    return baseline_mimic, baseline_flow, {"core_context": core_context, "mimic_parity": parity,
                                            "mimic_pair_count": len(expected_combos),
                                            "reused_metric_rows": len(baseline_mimic),
                                            "reused_pair_flow_rows": len(baseline_flow),
                                            "core_metrics_sha256": sha256_file(CORE_METRICS),
                                            "core_pair_flow_sha256": sha256_file(CORE_FLOW),
                                            "historical_seed_protocol": "retained completed core pair-specific derived_seed(base, db, endpoint, analysis_a, analysis_b) values unchanged"}


def cohort_population_attestations(manifest: dict) -> dict:
    import pandas as pd
    attestations = {}
    for db, expected in EXPECTED_POPULATIONS.items():
        entry = find_entry(manifest, "main20k_real", db)
        path = Path(entry["prediction_file"])
        if sha256_file(path) != str(entry["prediction_file_sha256"]).lower():
            raise ValueError(f"{db}: prediction parquet SHA256 mismatch")
        data = pd.read_parquet(path, columns=["db", "analysis_id", "stay_id", "subject", "y", "endpoint"],
                               filters=[("db", "==", db), ("analysis_id", "==", "main20k_real")])
        rows, patients, events = len(data), int(data.subject.astype(str).nunique()), int(data.y.sum())
        endpoint = sorted(data.endpoint.astype(str).unique().tolist())
        if (rows != expected["rows"] or events != expected["events"]
                or endpoint != [expected["endpoint"]]
                or rows != int(entry["n_prediction_rows"])):
            raise ValueError(f"{db}: saved population counts/endpoint do not match prerequested values")
        attestations[db] = {"stays": rows, "patients": patients, "events": events,
                            "endpoint": expected["endpoint"], "status": "saved_prediction_population_verified"}
    return attestations


def build_preflight() -> dict:
    """Verify real recovered gate, prediction, row, MIMIC reuse and schema bindings."""
    import pandas as pd

    manifest, gate, sidecar, bindings = load_and_verify_bindings()
    metrics_code_hash = sha256_file(METRICS_CODE)
    scope = runpy.run_path(str(METRICS_CODE), run_name="paired_metrics_recovered_preflight")
    scope["current_eligibility_hashes"].__defaults__ = (ELIG_DIR,)
    current_hashes = {db: scope["current_eligibility_hashes"](db) for db in COHORTS}
    tables, row_checks, file_hashes = {}, [], {}
    for analysis in ANALYSIS_IDS:
        path = PRED_DIR / f"{analysis}.parquet"
        full_hash = sha256_file(path)
        file_hashes[analysis] = full_hash
        full = pd.read_parquet(path, filters=[("db", "in", list(COHORTS))])
        frame = scope["validate_predictions"](full, analysis)
        if set(frame["db"].astype(str).unique()) != set(COHORTS):
            raise ValueError(f"{analysis}: prediction parquet cohort set differs from requested new cohorts")
        tables[analysis] = frame
        for db in COHORTS:
            entry = find_entry(manifest, analysis, db)
            scope["validate_manifest_entry"](entry, analysis, db, path, current_hashes[db])
            part = frame.loc[frame["db"].astype(str).eq(db)]
            if len(part) != int(entry["n_prediction_rows"]):
                raise ValueError(f"{analysis}/{db}: row count differs from manifest")
            if scope["sha256_ordered_prediction_rows"](part) != str(entry["ordered_row_digest_sha256"]).lower():
                raise ValueError(f"{analysis}/{db}: ordered row digest mismatch")
            if (part["prediction_scope"].astype(str).ne(entry["prediction_scope"]).any()
                    or part["temporal_evidence"].astype(str).ne(entry["temporal_evidence"]).any()):
                raise ValueError(f"{analysis}/{db}: prediction scope or temporal evidence differs from manifest")
            for column in ("eligibility_manifest_sha256", "eligibility_table_sha256"):
                if part[column].astype(str).str.lower().ne(current_hashes[db][column]).any():
                    raise ValueError(f"{analysis}/{db}: row-level {column} differs from gate")
            row_checks.append({"analysis_id": analysis, "db": db, "rows": int(len(part)),
                               "ordered_row_digest_sha256": entry["ordered_row_digest_sha256"],
                               "prediction_file_sha256": full_hash, "status": "verified"})
    flow_checks = []
    expected_pairs = tuple(itertools.combinations(sorted(ANALYSIS_IDS), 2))
    for db in COHORTS:
        endpoint = EXPECTED_POPULATIONS[db]["endpoint"]
        for analysis_a, analysis_b in expected_pairs:
            paired, flow = scope["pair_rows"](tables[analysis_a], tables[analysis_b], db, endpoint)
            if not flow["paired_identity_labels_verified"] or paired.empty:
                raise ValueError(f"{db}: within-source paired identity failed for {analysis_a}/{analysis_b}")
            flow_checks.append({"db": db, "endpoint": endpoint, "analysis_id_a": analysis_a,
                                "analysis_id_b": analysis_b, "paired_stays": int(len(paired)),
                                "paired_patients": int(paired["subject"].nunique()),
                                "identity_labels_verified": True})
    populations = cohort_population_attestations(manifest)
    if len(row_checks) != len(ANALYSIS_IDS) * len(COHORTS) or len(flow_checks) != len(COHORTS) * len(expected_pairs):
        raise ValueError("preflight did not cover all 21 condition rows and 63 within-source pairs")
    mimic_rows, mimic_flow, mimic_provenance = verify_mimic_and_core_reuse(manifest, metrics_code_hash)
    return {"manifest": manifest, "gate": gate, "sidecar": sidecar, "bindings": bindings,
            "metrics_code_hash": metrics_code_hash, "mimic_rows": mimic_rows,
            "mimic_flow": mimic_flow, "mimic_provenance": mimic_provenance,
            "populations": populations, "row_checks": row_checks,
            "within_source_pair_checks": flow_checks, "external_file_hashes": file_hashes,
            "tables": tables, "metric_scope": scope}


def preflight_summary(result: dict) -> dict:
    groups, flows = build_roster_groups(result)
    pairwise_evaluations = 2 * len(COHORTS) * len(tuple(itertools.combinations(sorted(ANALYSIS_IDS), 2)))
    grouped_evaluations = sum(len(group["conditions"]) for group in groups)
    return {"status": "preflight_complete", "population_counts": result["populations"],
            "prediction_manifest": result["bindings"]["prediction_manifest"],
            "eligibility_manifest": result["bindings"]["eligibility_manifest"],
            "embedding_snapshot": result["bindings"]["embedding_snapshot"],
            "identity_sidecar": result["bindings"]["identity_attestation"],
            "verified_prediction_entries": len(result["row_checks"]),
            "verified_within_source_pairs": len(result["within_source_pair_checks"]),
            "mimic_reused_metric_rows": len(result["mimic_rows"]),
            "mimic_reused_pairs": result["mimic_provenance"]["mimic_pair_count"],
            "mimic_exact_prediction_parity_analyses": len(result["mimic_provenance"]["mimic_parity"]),
            "exact_roster_groups": [{"db": g["db"], "endpoint": g["endpoint"],
                                      "ordered_population_digest_sha256": g["population_digest"],
                                      "seed": g["seed"], "stays": len(g["base"]),
                                      "patients": int(g["base"]["subject"].astype(str).nunique()),
                                      "condition_ids": g["conditions"], "pair_count": len(g["pairs"])}
                                     for g in groups],
            "external_pair_flow_rows": len(flows),
            "pairwise_metric_evaluations_per_draw": pairwise_evaluations,
            "grouped_metric_evaluations_per_draw": grouped_evaluations,
            "estimated_evaluation_saving_factor": pairwise_evaluations / grouped_evaluations,
            "patient_ids_exported": False}


def exact_population_digest(paired) -> str:
    cols = ("stay_id", "subject", "rows", "split", "y")
    ordered = paired.loc[:, cols].copy()
    ordered["_stay_key"] = ordered["stay_id"].map(str)
    ordered = ordered.sort_values("_stay_key", kind="mergesort")
    digest = hashlib.sha256()
    for row in ordered.loc[:, cols].itertuples(index=False, name=None):
        values = [str(row[0]), str(row[1]), int(row[2]), int(row[3]), int(row[4])]
        digest.update(json.dumps(values, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def compute_shared_weighted_metrics(y, probabilities: dict[str, np.ndarray], weights,
                                    weighted_metrics_fn) -> dict[str, dict[str, float]]:
    """Evaluate every arm on one identical patient-weight vector."""
    return {analysis_id: weighted_metrics_fn(y, p, weights)
            for analysis_id, p in probabilities.items()}


def build_roster_groups(preflight: dict) -> tuple[list[dict], list[dict]]:
    scope, tables = preflight["metric_scope"], preflight["tables"]
    groups_by_key, flows = {}, []
    pair_order = tuple(itertools.combinations(sorted(ANALYSIS_IDS), 2))
    for db in sorted(COHORTS):
        endpoint = EXPECTED_POPULATIONS[db]["endpoint"]
        for analysis_a, analysis_b in pair_order:
            paired, flow = scope["pair_rows"](tables[analysis_a], tables[analysis_b], db, endpoint)
            if not flow["paired_identity_labels_verified"] or paired.empty:
                raise ValueError(f"{db}: exact-roster grouping found an invalid pair")
            population_digest = exact_population_digest(paired)
            key = (db, endpoint, population_digest)
            group = groups_by_key.setdefault(key, {
                "db": db, "endpoint": endpoint, "population_digest": population_digest,
                "base": paired[["stay_id", "subject", "rows", "split", "y"]].copy(),
                "pairs": [],
            })
            group["pairs"].append((analysis_a, analysis_b))
            flow.update(analysis_id_a=analysis_a, analysis_id_b=analysis_b,
                        n_source_rows_a=int(tables[analysis_a]["db"].astype(str).eq(db).sum()),
                        n_source_rows_b=int(tables[analysis_b]["db"].astype(str).eq(db).sum()))
            flows.append(flow)
    groups = list(groups_by_key.values())
    for group in groups:
        group["conditions"] = sorted({condition for pair in group["pairs"] for condition in pair})
        group["seed"] = scope["derive_seed"](SEED, group["db"], group["population_digest"])
    return groups, flows


def group_probabilities(group: dict, tables: dict):
    base = group["base"].copy()
    base["_stay_key"] = base["stay_id"].map(str)
    base = base.sort_values("_stay_key", kind="mergesort").reset_index(drop=True)
    expected_subject = base["subject"].astype(str).to_numpy()
    expected_rows = base["rows"].to_numpy()
    expected_split = base["split"].to_numpy()
    expected_y = base["y"].to_numpy(dtype=np.int8)
    probability_by_analysis = {}
    for analysis in group["conditions"]:
        part = tables[analysis].loc[tables[analysis]["db"].astype(str).eq(group["db"])].copy()
        part["_stay_key"] = part["stay_id"].map(str)
        if part["_stay_key"].duplicated().any():
            raise ValueError(f"{analysis}/{group['db']}: duplicate stay identity")
        indexed = part.set_index("_stay_key", drop=False)
        wanted = base["_stay_key"].tolist()
        if not all(stay in indexed.index for stay in wanted):
            raise ValueError(f"{analysis}/{group['db']}: roster group is not present in every condition")
        ordered = indexed.loc[wanted]
        checks = {
            "subject": np.array_equal(ordered["subject"].astype(str).to_numpy(), expected_subject),
            "rows": np.array_equal(ordered["rows"].to_numpy(), expected_rows),
            "split": np.array_equal(ordered["split"].to_numpy(), expected_split),
            "y": np.array_equal(ordered["y"].to_numpy(dtype=np.int8), expected_y),
        }
        if not all(checks.values()):
            raise ValueError(f"{analysis}/{group['db']}: roster identity mismatch: {checks}")
        probability_by_analysis[analysis] = ordered["p"].to_numpy(dtype=np.float64)
    return base, expected_y, expected_subject, probability_by_analysis


def run_group_bootstrap(group: dict, tables: dict, scope: dict, output_dir: Path,
                        group_index: int, group_count: int) -> tuple[list[dict], dict]:
    base, y, subjects, probability_by_analysis = group_probabilities(group, tables)
    unique, inverse = np.unique(subjects, return_inverse=True)
    if len(unique) != int(base["subject"].astype(str).nunique()):
        raise ValueError("patient cluster indexing failed")
    point = {analysis: scope["weighted_metrics"](y, p) for analysis, p in probability_by_analysis.items()}
    samples = {analysis: {metric: np.full(BOOTSTRAP, np.nan, dtype=np.float64)
                          for metric in METRIC_COLUMNS}
               for analysis in group["conditions"]}
    rng = np.random.default_rng(group["seed"])
    checkpoint_path = output_dir / "bootstrap_checkpoint.npz"
    for draw in range(BOOTSTRAP):
        sampled = rng.integers(0, len(unique), size=len(unique))
        counts = np.bincount(sampled, minlength=len(unique)).astype(np.float64)
        weights = counts[inverse]
        draw_metrics = compute_shared_weighted_metrics(y, probability_by_analysis, weights,
                                                      scope["weighted_metrics"])
        for analysis, values in draw_metrics.items():
            for metric in METRIC_COLUMNS:
                samples[analysis][metric][draw] = values[metric]
        completed = draw + 1
        if completed % 100 == 0:
            arrays = {f"c{index}_{metric}": samples[analysis][metric][:completed]
                      for index, analysis in enumerate(group["conditions"])
                      for metric in METRIC_COLUMNS}
            atomic_npz(checkpoint_path, completed_draws=np.asarray(completed),
                       condition_ids=np.asarray(group["conditions"]),
                       metric_ids=np.asarray(METRIC_COLUMNS), **arrays)
            atomic_json(output_dir / "progress.json", {
                "status": "running", "completed_roster_groups": group_index - 1,
                "roster_group_index": group_index, "roster_group_count": group_count,
                "db": group["db"], "endpoint": group["endpoint"],
                "ordered_population_digest_sha256": group["population_digest"],
                "group_seed": group["seed"], "completed_draws": completed,
                "requested_draws": BOOTSTRAP, "pair_count_in_group": len(group["pairs"]),
                "conditions_scored_per_draw": len(group["conditions"]),
            })
            print(f"BOOTSTRAP_PROGRESS group={group_index}/{group_count} db={group['db']} "
                  f"draws={completed}/{BOOTSTRAP} pairs={len(group['pairs'])} "
                  f"conditions={len(group['conditions'])}", flush=True)
    rows = []
    valid_by_condition = {}
    for condition in group["conditions"]:
        valid_by_condition[condition] = {
            metric: int(np.isfinite(samples[condition][metric]).sum()) for metric in METRIC_COLUMNS
        }
    for analysis_a, analysis_b in group["pairs"]:
        for metric in METRIC_COLUMNS:
            values_a = samples[analysis_a][metric]
            values_b = samples[analysis_b][metric]
            delta = values_b - values_a
            valid_a, valid_b, valid_delta = (values_a[np.isfinite(values_a)],
                                             values_b[np.isfinite(values_b)],
                                             delta[np.isfinite(delta)])
            def interval(values):
                return tuple(map(float, np.quantile(values, [0.025, 0.975]))) if len(values) else (math.nan, math.nan)
            ci_a, ci_b, ci_delta = interval(valid_a), interval(valid_b), interval(valid_delta)
            estimate_a, estimate_b = point[analysis_a][metric], point[analysis_b][metric]
            rows.append({
                "metric": metric, "estimate_a": estimate_a, "ci_a_low": ci_a[0], "ci_a_high": ci_a[1],
                "valid_boot_a": int(len(valid_a)), "invalid_boot_a": BOOTSTRAP-int(len(valid_a)),
                "estimate_b": estimate_b, "ci_b_low": ci_b[0], "ci_b_high": ci_b[1],
                "valid_boot_b": int(len(valid_b)), "invalid_boot_b": BOOTSTRAP-int(len(valid_b)),
                "difference_b_minus_a": (estimate_b-estimate_a
                    if np.isfinite(estimate_a) and np.isfinite(estimate_b) else math.nan),
                "difference_ci_low": ci_delta[0], "difference_ci_high": ci_delta[1],
                "valid_paired_boot": int(len(valid_delta)),
                "invalid_paired_boot": BOOTSTRAP-int(len(valid_delta)),
                "db": group["db"], "endpoint": group["endpoint"],
                "analysis_id_a": analysis_a, "analysis_id_b": analysis_b,
                "n_paired_stays": int(len(base)), "n_paired_patients": int(len(unique)),
                "n_bootstrap": BOOTSTRAP, "seed": int(group["seed"]),
                "inference_scope": "conditional on frozen encoder and fitted source probes",
            })
    group_record = {
        "db": group["db"], "endpoint": group["endpoint"],
        "ordered_population_digest_sha256": group["population_digest"], "seed": int(group["seed"]),
        "stays": int(len(base)), "patients": int(len(unique)),
        "conditions_scored_per_draw": len(group["conditions"]),
        "condition_ids": group["conditions"], "pair_count": len(group["pairs"]),
        "pairs": [list(pair) for pair in group["pairs"]],
        "valid_bootstrap_draws_by_condition_metric": valid_by_condition,
        "completed_draws": BOOTSTRAP,
    }
    atomic_json(output_dir / "progress.json", {
        "status": "roster_group_complete", "completed_roster_groups": group_index,
        "roster_group_index": group_index, "roster_group_count": group_count,
        "db": group["db"], "endpoint": group["endpoint"],
        "ordered_population_digest_sha256": group["population_digest"],
        "group_seed": group["seed"], "completed_draws": BOOTSTRAP,
        "requested_draws": BOOTSTRAP, "pair_count_in_group": len(group["pairs"]),
        "conditions_scored_per_draw": len(group["conditions"]),
    })
    return rows, group_record


def atomic_csv(frame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def install_checkpointed_bootstrap(scope: dict, output_dir: Path) -> dict:
    runner_globals = scope["run"].__globals__
    original_pair_rows = runner_globals["pair_rows"]
    original_weighted_metrics = runner_globals["weighted_metrics"]
    state = {"pair_index": 0, "context": None, "completed": 0, "checkpoint_path": output_dir / "bootstrap_checkpoint.npz"}

    def pair_rows_with_context(a, b, db, endpoint):
        pair, flow = original_pair_rows(a, b, db, endpoint)
        state["pair_index"] += 1
        state["context"] = {"pair_index": state["pair_index"], "db": db, "endpoint": endpoint,
                             "analysis_id_a": str(a.analysis_id.iloc[0]),
                             "analysis_id_b": str(b.analysis_id.iloc[0])}
        return pair, flow

    def checkpointed_bootstrap(paired, n_boot=1000, seed=20261122):
        if n_boot != BOOTSTRAP:
            raise ValueError("recovered CIs require exactly 1,000 bootstrap draws")
        y = paired["y"].to_numpy(dtype=np.int8)
        pa, pb = paired["p_a"].to_numpy(dtype=float), paired["p_b"].to_numpy(dtype=float)
        groups, group_index = np.unique(paired["subject"].astype(str), return_inverse=True)
        metrics_a = original_weighted_metrics(y, pa)
        metrics_b = original_weighted_metrics(y, pb)
        boot_a = {name: np.full(n_boot, np.nan) for name in scope["METRICS"]}
        boot_b = {name: np.full(n_boot, np.nan) for name in scope["METRICS"]}
        boot_delta = {name: np.full(n_boot, np.nan) for name in scope["METRICS"]}
        rng = np.random.default_rng(seed)
        for replicate in range(n_boot):
            sampled = rng.integers(0, len(groups), size=len(groups))
            counts = np.bincount(sampled, minlength=len(groups)).astype(float)
            row_weights = counts[group_index]
            draw_a = original_weighted_metrics(y, pa, row_weights)
            draw_b = original_weighted_metrics(y, pb, row_weights)
            for name in scope["METRICS"]:
                boot_a[name][replicate] = draw_a[name]
                boot_b[name][replicate] = draw_b[name]
                if np.isfinite(draw_a[name]) and np.isfinite(draw_b[name]):
                    boot_delta[name][replicate] = draw_b[name] - draw_a[name]
            if (replicate + 1) % 100 == 0:
                arrays = {f"a_{name}": values[:replicate + 1] for name, values in boot_a.items()}
                arrays.update({f"b_{name}": values[:replicate + 1] for name, values in boot_b.items()})
                arrays.update({f"delta_{name}": values[:replicate + 1] for name, values in boot_delta.items()})
                atomic_npz(state["checkpoint_path"], completed_draws=np.asarray(replicate + 1), **arrays)
                atomic_json(output_dir / "progress.json", {"status": "running",
                            "current_pair": state["context"], "completed_draws_for_pair": replicate + 1,
                            "requested_draws_per_pair": n_boot, "pairs_started": state["pair_index"]})
                print(f"BOOTSTRAP_PROGRESS pair={state['pair_index']} db={state['context']['db']} "
                      f"a={state['context']['analysis_id_a']} b={state['context']['analysis_id_b']} "
                      f"draws={replicate + 1}/{n_boot}", flush=True)
        rows = []
        for name in scope["METRICS"]:
            def summarize(values):
                valid = values[np.isfinite(values)]
                if not len(valid):
                    return math.nan, math.nan, 0
                lo, hi = np.quantile(valid, [0.025, 0.975])
                return float(lo), float(hi), int(len(valid))
            alo, ahi, nav = summarize(boot_a[name])
            blo, bhi, nbv = summarize(boot_b[name])
            dlo, dhi, ndv = summarize(boot_delta[name])
            rows.append({"metric": name, "estimate_a": metrics_a[name], "ci_a_low": alo,
                         "ci_a_high": ahi, "valid_boot_a": nav, "invalid_boot_a": int(n_boot-nav),
                         "estimate_b": metrics_b[name], "ci_b_low": blo, "ci_b_high": bhi,
                         "valid_boot_b": nbv, "invalid_boot_b": int(n_boot-nbv),
                         "difference_b_minus_a": (metrics_b[name]-metrics_a[name]
                            if np.isfinite(metrics_a[name]) and np.isfinite(metrics_b[name]) else math.nan),
                         "difference_ci_low": dlo, "difference_ci_high": dhi,
                         "valid_paired_boot": ndv, "invalid_paired_boot": int(n_boot-ndv)})
        protocol = {"bootstrap_unit": "subject, resample patients with replacement; retain all stays",
                    "evaluation_weighting": "stay-weighted within each patient-cluster replicate",
                    "bootstrap_replicates": int(n_boot), "seed": int(seed),
                    "inference_scope": "conditional on frozen encoder and fitted source probes",
                    "calibration_slope": "weighted logistic regression of outcome on logit(p), slope penalty 1e-6, intercept unpenalized",
                    "offset_intercept": "weighted intercept-only recalibration with logit coefficient fixed at one"}
        state["completed"] += 1
        atomic_json(output_dir / "progress.json", {"status": "pair_complete", "current_pair": state["context"],
                    "completed_draws_for_pair": n_boot, "requested_draws_per_pair": n_boot,
                    "completed_pairs": state["completed"], "pairs_started": state["pair_index"]})
        return rows, protocol

    # runpy.run_path returns a namespace mapping separate from function globals.
    # Patch the actual runner namespace as well as the returned mapping.
    runner_globals["pair_rows"] = pair_rows_with_context
    runner_globals["paired_cluster_bootstrap"] = checkpointed_bootstrap
    scope["pair_rows"] = pair_rows_with_context
    scope["paired_cluster_bootstrap"] = checkpointed_bootstrap
    if (runner_globals["pair_rows"] is not pair_rows_with_context
            or runner_globals["paired_cluster_bootstrap"] is not checkpointed_bootstrap):
        raise RuntimeError("checkpoint shim did not reach paired_metrics_v4.run globals")
    return state


def run(output_dir: Path) -> None:
    import pandas as pd

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    preflight = build_preflight()
    output_dir.mkdir(parents=True)
    sidecar, bindings = preflight["sidecar"], preflight["bindings"]
    metrics_code_hash = preflight["metrics_code_hash"]
    mimic_rows, mimic_flow = preflight["mimic_rows"], preflight["mimic_flow"]
    mimic_provenance, populations = preflight["mimic_provenance"], preflight["populations"]
    atomic_json(output_dir / "preflight_attestation.json", preflight_summary(preflight))
    groups, flow_rows = build_roster_groups(preflight)
    expected_pairs = len(tuple(itertools.combinations(sorted(ANALYSIS_IDS), 2)))
    if sum(len(group["pairs"]) for group in groups) != len(COHORTS) * expected_pairs:
        raise ValueError("exact-roster groups do not cover the full 63-pair grid")
    for db in COHORTS:
        observed = {(row["analysis_id_a"], row["analysis_id_b"]) for row in flow_rows if row["db"] == db}
        if observed != set(itertools.combinations(sorted(ANALYSIS_IDS), 2)):
            raise ValueError(f"{db}: exact-roster pair grid is incomplete")
    group_records_seed = [{"db": g["db"], "endpoint": g["endpoint"],
                           "ordered_population_digest_sha256": g["population_digest"],
                           "seed": int(g["seed"]), "stays": int(len(g["base"])),
                           "patients": int(g["base"]["subject"].astype(str).nunique()),
                           "condition_ids": g["conditions"], "pairs": [list(p) for p in g["pairs"]],
                           "conditions_scored_per_draw": len(g["conditions"])} for g in groups]
    atomic_json(output_dir / "roster_group_definitions.json", {"groups": group_records_seed})
    external_flow = pd.DataFrame(flow_rows)
    atomic_csv(external_flow, output_dir / "paired_pair_flow_new_cohorts.csv")
    metric_rows, completed_group_records = [], []
    for group_index, group in enumerate(groups, start=1):
        rows, group_record = run_group_bootstrap(group, preflight["tables"],
                                                 preflight["metric_scope"], output_dir,
                                                 group_index, len(groups))
        metric_rows.extend(rows)
        completed_group_records.append(group_record)
        atomic_csv(pd.DataFrame(metric_rows), output_dir / "paired_metrics_new_cohorts_partial.csv")
    external_metrics = pd.DataFrame(metric_rows)
    for db in COHORTS:
        cohort_rows = external_metrics.loc[external_metrics["db"].astype(str).eq(db)]
        cohort_flow = external_flow.loc[external_flow["db"].astype(str).eq(db)]
        expected_combos = set(itertools.combinations(sorted(ANALYSIS_IDS), 2))
        if len(cohort_rows) != expected_pairs * len(METRIC_COLUMNS) or len(cohort_flow) != expected_pairs:
            raise ValueError(f"{db}: completed paired metrics grid incomplete")
        if set(zip(cohort_rows.analysis_id_a, cohort_rows.analysis_id_b)) != expected_combos:
            raise ValueError(f"{db}: completed contrast pairs differ from expected grid")
    combined_metrics = pd.concat([mimic_rows, external_metrics], ignore_index=True, sort=False)
    combined_flow = pd.concat([mimic_flow, external_flow], ignore_index=True, sort=False)
    combined_metrics.insert(0, "aggregate_source", ["reused_core_exact_mimic_predictions"] * len(mimic_rows)
                            + ["recovered_resume3_external_bootstrap"] * len(external_metrics))
    combined_flow.insert(0, "aggregate_source", ["reused_core_exact_mimic_predictions"] * len(mimic_flow)
                         + ["recovered_resume3_external_bootstrap"] * len(external_flow))
    atomic_csv(combined_metrics, output_dir / "paired_metrics_all_cohorts.csv")
    atomic_csv(combined_flow, output_dir / "paired_pair_flow_all_cohorts.csv")
    atomic_csv(mimic_rows, output_dir / "mimic_metrics_reused_from_core.csv")
    atomic_csv(mimic_flow, output_dir / "mimic_pair_flow_reused_from_core.csv")
    atomic_csv(external_metrics, output_dir / "paired_metrics_new_cohorts.csv")
    progress = {"status": "complete", "completed_roster_groups": len(groups),
                "requested_roster_groups": len(groups), "completed_pairs": len(COHORTS) * expected_pairs,
                "requested_external_pairs": len(COHORTS) * expected_pairs,
                "completed_draws_per_roster_group": BOOTSTRAP, "mimic_pairs_reused": expected_pairs}
    atomic_json(output_dir / "progress.json", progress)
    protocol = {
        "status": "complete",
        "analysis_ids": list(ANALYSIS_IDS), "new_cohorts": list(COHORTS),
        "population_counts": populations,
        "prediction_manifest": bindings["prediction_manifest"],
        "eligibility_manifest": bindings["eligibility_manifest"],
        "embedding_manifest_snapshot": bindings["embedding_snapshot"],
        "within_source_patient_identity_sidecar": bindings["identity_attestation"],
        "within_source_identity_basis": {db: sidecar["cohorts"][db]["subject_identity_basis"] for db in COHORTS},
        "cross_source_patient_independence_claimed": False,
        "sourcehash_schema": "cohorts[db].sourcehashes.sequence_npz; validated exactly",
        "paired_metrics_code": {"path": str(METRICS_CODE), "sha256": metrics_code_hash},
        "bootstrap_replicates": BOOTSTRAP, "base_seed": SEED,
        "inference_scope": "conditional on the completed frozen prediction outputs and fitted source probes",
        "external_seed_protocol": "derive_seed(base_seed, db, exact_ordered_population_digest); one patient draw vector is shared across every condition in that exact pair-intersection roster",
        "external_grid": {"pairs_per_cohort": expected_pairs, "metric_rows": len(external_metrics),
                          "pair_flow_rows": len(external_flow), "bootstrap_draws_per_pair": BOOTSTRAP,
                          "shared_patient_weights_within_each_exact_roster_group": True,
                          "roster_group_count": len(groups),
                          "roster_groups": completed_group_records,
                          "pairwise_metric_evaluations_per_draw": 2 * len(COHORTS) * expected_pairs,
                          "grouped_metric_evaluations_per_draw": sum(len(g["conditions"]) for g in groups),
                          "estimated_metric_evaluation_saving_factor": (2 * len(COHORTS) * expected_pairs)
                              / sum(len(g["conditions"]) for g in groups),
                          "calibration_convergence_failures_preserved": True,
                          "invalid_bootstrap_counts_preserved_in_rows": True},
        "mimic_reuse": mimic_provenance,
        "combined_grid": {"metric_rows": len(combined_metrics), "pair_flow_rows": len(combined_flow),
                          "mimic_metric_rows_reused": len(mimic_rows),
                          "new_cohort_metric_rows_computed": len(external_metrics)},
        "progress_checkpoint": {"path": str(output_dir / "bootstrap_checkpoint.npz"),
                                 "interval": 100, "path_includes_aggregate_samples_only": True},
        "patient_level_rows_or_ids_exported": False,
        "wrapper_sha256": sha256_file(Path(__file__)),
    }
    atomic_json(output_dir / "analysis_manifest.json", protocol)
    print(json.dumps({"status": "complete", "output_dir": str(output_dir),
                      "external_metric_rows": len(external_metrics),
                      "reused_mimic_metric_rows": len(mimic_rows),
                      "combined_metric_rows": len(combined_metrics), "roster_groups": len(groups),
                      "grouped_metric_evaluations_per_draw": sum(len(g["conditions"]) for g in groups),
                      "progress": progress},
                     allow_nan=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        print(json.dumps(preflight_summary(build_preflight()), allow_nan=False), flush=True)
        return
    if args.output is None:
        parser.error("--output is required unless --preflight-only is set")
    run(args.output)


if __name__ == "__main__":
    main()
