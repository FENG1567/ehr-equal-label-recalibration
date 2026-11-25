"""Paired source-only availability score sensitivity on original MIMIC stays.

Loads only manifested frozen probabilities. Patient-level inputs remain on
the server; this program writes aggregate metric, bootstrap, and flow tables.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
ENGINE = REV / "engine"
ENGINE_CODE = ENGINE / "code"
OUT = REV / "availability_metrics" / "results" / "paired_availability_source_v1"
NEW_RUN = ENGINE / "results/prediction_runs/availability_main_source_reuse_m69_20261123-3c067fe1960b32af"
NEW_MANIFEST_PATH = NEW_RUN / "prediction_manifest.json"
NEW_COMPLETION_PATH = NEW_RUN / "target_source_probe_reuse_completion.json"
OLD_RUN = ENGINE / "results/prediction_runs/status_consistent_primary_core_fix2-7da980ffcbd8f00b"
OLD_MANIFEST_PATH = OLD_RUN / "prediction_manifest.json"
OLD_GATE = REV / "temporal/results/eligibility/status_consistent_v4_1/landmark_v4_1"
AVAIL_GATE = REV / "temporal/results/eligibility/availability_testonly_reviewed_v4_1"
FULL_DB = "mimic_available_by24_v4_1"
FIXED_DB = "mimic_available_by24_fixedselection_v4_1"
ANALYSIS_IDS = (
    "main20k_real", "main20k_constant", "main20k_common8_mask", "main20k_no_h0_therapy",
)
SOURCE_PAIR_IDS = (
    "fullreal_minus_originalreal", "fixedreal_minus_originalreal",
    "fullreal_minus_fullconstant", "fixedreal_minus_fixedconstant",
    "fullcommon8_minus_fullreal", "fixedcommon8_minus_fixedreal",
)
N_BOOT = 1000
BASE_SEED = 20261122

sys.path.insert(0, str(ENGINE_CODE))
import paired_metrics_v4 as pm  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _entries_by_analysis(manifest: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    entries: dict[str, dict[str, dict[str, Any]]] = {}
    for row in manifest.get("predictions", []):
        analysis_id, db = str(row.get("analysis_id", "")), str(row.get("db", ""))
        if analysis_id in ANALYSIS_IDS:
            if not db or db in entries.setdefault(analysis_id, {}):
                raise ValueError(f"duplicate or missing manifest prediction entry: {analysis_id}/{db}")
            entries[analysis_id][db] = row
    missing = [analysis_id for analysis_id in ANALYSIS_IDS if analysis_id not in entries]
    if missing:
        raise ValueError(f"availability manifest lacks source model entries: {missing}")
    return entries


def _validate_run_manifest(path: Path, expected_run_id: str,
                           expected_dbs: set[str], *, exact_dbs: bool = True
                           ) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    manifest = _json(path)
    if manifest.get("run_id") != expected_run_id:
        raise ValueError(f"{path}: unexpected run_id")
    entries = _entries_by_analysis(manifest)
    for analysis_id in ANALYSIS_IDS:
        observed = set(entries[analysis_id])
        if not expected_dbs.issubset(observed) or (exact_dbs and observed != expected_dbs):
            raise ValueError(f"{analysis_id}: cohort roster differs: {sorted(entries[analysis_id])}")
    return manifest, entries


def _load_analysis_frames(manifest: dict[str, Any], entries: dict[str, dict[str, dict[str, Any]]],
                          *, dbs: tuple[str, ...], gate_by_db: dict[str, Path]
                          ) -> dict[str, dict[str, pd.DataFrame]]:
    tables: dict[str, dict[str, pd.DataFrame]] = {}
    for analysis_id in ANALYSIS_IDS:
        per_db = entries[analysis_id]
        paths = {Path(str(row["prediction_file"])).resolve() for row in per_db.values()}
        hashes = {str(row.get("prediction_file_sha256", "")).lower() for row in per_db.values()}
        if len(paths) != 1 or len(hashes) != 1:
            raise ValueError(f"{analysis_id}: manifest must point to one immutable prediction file")
        prediction_path = next(iter(paths))
        frame = pd.read_parquet(prediction_path, filters=[("db", "in", list(dbs))])
        frame = pm.validate_predictions(frame, analysis_id)
        for db in dbs:
            entry = per_db.get(db)
            if entry is None:
                raise ValueError(f"{analysis_id}/{db}: current manifest entry is missing")
            gate = gate_by_db[db]
            current_hashes = pm.current_eligibility_hashes(db, gate)
            pm.validate_manifest_entry(entry, analysis_id, db, prediction_path, current_hashes)
            cohort = frame.loc[frame["db"].astype(str).eq(db)].copy()
            if len(cohort) != int(entry["n_prediction_rows"]):
                raise ValueError(f"{analysis_id}/{db}: prediction row count differs from manifest")
            if pm.sha256_ordered_prediction_rows(cohort) != str(entry["ordered_row_digest_sha256"]).lower():
                raise ValueError(f"{analysis_id}/{db}: ordered row digest differs from manifest")
            if cohort["prediction_scope"].astype(str).ne(entry["prediction_scope"]).any():
                raise ValueError(f"{analysis_id}/{db}: prediction scope differs from manifest")
            if cohort["temporal_evidence"].astype(str).ne(entry["temporal_evidence"]).any():
                raise ValueError(f"{analysis_id}/{db}: temporal evidence differs from manifest")
            for field in ("eligibility_manifest_sha256", "eligibility_table_sha256"):
                if cohort[field].astype(str).str.lower().ne(current_hashes[field]).any():
                    raise ValueError(f"{analysis_id}/{db}: stale row-level {field}")
            tables.setdefault(analysis_id, {})[db] = cohort
    return tables


def _canonical_identity(frame: pd.DataFrame) -> tuple[str, list[tuple[str, str, int, int, int]]]:
    fields = ["stay_id", "subject", "rows", "split", "y"]
    ordered = frame[fields].copy()
    ordered["stay_id"] = ordered["stay_id"].astype(str)
    ordered["subject"] = ordered["subject"].astype(str)
    ordered = ordered.sort_values("stay_id", kind="stable").reset_index(drop=True)
    if ordered["stay_id"].duplicated().any():
        raise ValueError("canonical alias identity has duplicate stay IDs")
    records = [(str(r.stay_id), str(r.subject), int(r.rows), int(r.split), int(r.y))
               for r in ordered.itertuples(index=False)]
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest(), records


def _identity_and_probability_bridge(old: dict[str, pd.DataFrame],
                                     new: dict[str, dict[str, pd.DataFrame]],
                                     completion: dict[str, Any],
                                     old_entries: dict[str, dict[str, dict[str, Any]]],
                                     new_entries: dict[str, dict[str, dict[str, Any]]]
                                     ) -> dict[str, Any]:
    comparisons = completion.get("frozen_source_probability_reproduction", {}).get("comparisons", [])
    by_model = {str(row.get("analysis_id")): row for row in comparisons
                if row.get("db") == "mimic"}
    if set(by_model) != set(ANALYSIS_IDS):
        raise ValueError("completion record does not attest all four source probability reproductions")
    source_proof: dict[str, Any] = {}
    source_ref = old["main20k_real"]
    old_run_sha = _sha256(OLD_MANIFEST_PATH)
    alias_proofs: dict[str, Any] = {}
    for model_id in ANALYSIS_IDS:
        status = by_model[model_id]
        if (status.get("identity_exact") is not True
                or status.get("probabilities_bitwise_exact") is not True
                or status.get("origin_run_id") != "status_consistent_primary_core_fix2"
                or str(status.get("origin_manifest_sha256", "")).lower() != old_run_sha):
            raise ValueError(f"completion does not prove exact original-source identity/probabilities for {model_id}")
        if (str(status.get("origin_prediction_file_sha256", "")).lower()
                != str(old_entries[model_id]["mimic"].get("prediction_file_sha256", "")).lower()
                or str(status.get("current_prediction_file_sha256", "")).lower()
                != str(new_entries[model_id]["mimic"].get("prediction_file_sha256", "")).lower()):
            raise ValueError(f"{model_id}: source-reuse completion file hashes differ from live manifests")
        new_source = new[model_id]["mimic"]
        old_frame = old[model_id]
        old_digest, old_records = _canonical_identity(old_frame)
        new_digest, new_records = _canonical_identity(new_source)
        if old_records != new_records or old_digest != new_digest:
            raise ValueError(f"{model_id}: new source alias differs from original heldout identity")
        old_by_stay = old_frame.set_index("stay_id").loc[new_source["stay_id"].astype(str)]
        new_ordered = new_source.set_index("stay_id").loc[new_source["stay_id"].astype(str)]
        if not np.array_equal(old_by_stay["p"].to_numpy(float), new_ordered["p"].to_numpy(float)):
            raise ValueError(f"{model_id}: source probabilities are not bitwise equal to original artifact")
        source_proof[model_id] = {
            "identity_sha256": old_digest, "n_stays": len(old_records),
            "probabilities_bitwise_equal_to_original": True,
            "original_prediction_file_sha256": old_entries[model_id]["mimic"]["prediction_file_sha256"],
            "reused_prediction_file_sha256": new_entries[model_id]["mimic"]["prediction_file_sha256"],
        }
    for target_db in (FULL_DB, FIXED_DB):
        model_proofs = {}
        for model_id in ANALYSIS_IDS:
            target_frame = new[model_id][target_db]
            digest, records = _canonical_identity(target_frame)
            source_digest, source_records = _canonical_identity(source_ref)
            if records != source_records or digest != source_digest:
                raise ValueError(f"{target_db}/{model_id}: target alias is not the original heldout stay identity")
            model_proofs[model_id] = {
                "identity_sha256": digest, "n_stays": len(records),
                "exact_original_heldout_stay_subject_row_split_label_identity": True,
                "target_prediction_file_sha256": new_entries[model_id][target_db]["prediction_file_sha256"],
                "target_ordered_row_digest_sha256": new_entries[model_id][target_db]["ordered_row_digest_sha256"],
            }
        alias_proofs[target_db] = model_proofs
    bridge = completion.get("source_bridge_equivalence", {})
    if not all(bridge.get(key) is True for key in (
            "mimic_entry_exact", "mimic_sequence_npz_hash_exact", "mimic_table_exact")):
        raise ValueError("completion record does not prove original MIMIC source-table bridge identity")
    return {
        "original_manifest_sha256": old_run_sha,
        "source_run_mimic_probability_bridge": source_proof,
        "target_alias_identity_to_original_mimic": alias_proofs,
        "source_gate_path": bridge.get("source_gate_path"),
        "source_gate_sha256": bridge.get("source_gate_sha256"),
        "target_gate_container_path": bridge.get("target_gate_container_path"),
        "target_gate_container_sha256": bridge.get("target_gate_container_sha256"),
        "source_mimic_sequence_npz_sha256": bridge.get("source_sequence_npz_hash"),
        "cross_version_alias_interpretation": "same original heldout patients established by exact stay, patient, source-row, split and label tuples; only cohort alias and verified eligibility container differ",
    }


def _rekey_to_mimic(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["db"] = "mimic"
    return result


def _comparison_specs(old: dict[str, pd.DataFrame],
                      new: dict[str, dict[str, pd.DataFrame]]) -> list[dict[str, Any]]:
    original = old["main20k_real"]
    full_real = new["main20k_real"][FULL_DB]
    fixed_real = new["main20k_real"][FIXED_DB]
    full_constant = new["main20k_constant"][FULL_DB]
    fixed_constant = new["main20k_constant"][FIXED_DB]
    full_common8 = new["main20k_common8_mask"][FULL_DB]
    fixed_common8 = new["main20k_common8_mask"][FIXED_DB]
    return [
        {"pair_id": SOURCE_PAIR_IDS[0], "a_label": "originalreal", "b_label": "fullreal",
         "a": original, "b": _rekey_to_mimic(full_real), "db": "mimic",
         "population": FULL_DB, "cross_version_alias": True},
        {"pair_id": SOURCE_PAIR_IDS[1], "a_label": "originalreal", "b_label": "fixedreal",
         "a": original, "b": _rekey_to_mimic(fixed_real), "db": "mimic",
         "population": FIXED_DB, "cross_version_alias": True},
        {"pair_id": SOURCE_PAIR_IDS[2], "a_label": "fullconstant", "b_label": "fullreal",
         "a": full_constant, "b": full_real, "db": FULL_DB,
         "population": FULL_DB, "cross_version_alias": False},
        {"pair_id": SOURCE_PAIR_IDS[3], "a_label": "fixedconstant", "b_label": "fixedreal",
         "a": fixed_constant, "b": fixed_real, "db": FIXED_DB,
         "population": FIXED_DB, "cross_version_alias": False},
        {"pair_id": SOURCE_PAIR_IDS[4], "a_label": "fullreal", "b_label": "fullcommon8",
         "a": full_real, "b": full_common8, "db": FULL_DB,
         "population": FULL_DB, "cross_version_alias": False},
        {"pair_id": SOURCE_PAIR_IDS[5], "a_label": "fixedreal", "b_label": "fixedcommon8",
         "a": fixed_real, "b": fixed_common8, "db": FIXED_DB,
         "population": FIXED_DB, "cross_version_alias": False},
    ]


def main() -> int:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    new_manifest, new_entries = _validate_run_manifest(
        NEW_MANIFEST_PATH, "availability_main_source_reuse_m69_20261123",
        {"mimic", FULL_DB, FIXED_DB},
    )
    completion = _json(NEW_COMPLETION_PATH)
    if (completion.get("status") != "target_source_probe_reuse_complete"
            or completion.get("clinical_input_availability_verified") is not False
            or completion.get("clinical_ready") is not False
            or completion.get("fit_source_probe_calls") != 0
            or completion.get("source_probe_refit_forbidden") is not True
            or completion.get("prediction_manifest_sha256") != _sha256(NEW_MANIFEST_PATH)):
        raise ValueError("source-reuse completion record is missing/invalid or permits refitting/clinical claims")
    old_manifest, old_entries = _validate_run_manifest(
        OLD_MANIFEST_PATH, "status_consistent_primary_core_fix2", {"mimic"}, exact_dbs=False,
    )
    current_new = {
        "mimic": OLD_GATE,
        FULL_DB: AVAIL_GATE,
        FIXED_DB: AVAIL_GATE,
    }
    new = _load_analysis_frames(new_manifest, new_entries,
                                dbs=("mimic", FULL_DB, FIXED_DB), gate_by_db=current_new)
    old_nested = _load_analysis_frames(old_manifest, old_entries, dbs=("mimic",),
                                       gate_by_db={"mimic": OLD_GATE})
    old = {analysis_id: old_nested[analysis_id]["mimic"] for analysis_id in ANALYSIS_IDS}
    bridge = _identity_and_probability_bridge(old, new, completion, old_entries, new_entries)
    all_rows: list[dict[str, Any]] = []
    flow_rows: list[dict[str, Any]] = []
    for spec in _comparison_specs(old, new):
        paired, flow = pm.pair_rows(spec["a"], spec["b"], spec["db"], "hospital")
        if len(paired) != len(spec["a"]) or len(paired) != len(spec["b"]):
            raise ValueError(f"{spec['pair_id']}: intersection excludes rows; inspect sample flow")
        if paired["subject"].nunique() != spec["a"]["subject"].nunique():
            raise ValueError(f"{spec['pair_id']}: paired patient set is incomplete")
        seed = pm.derive_seed(BASE_SEED, "availability", spec["pair_id"])
        metric_rows, protocol = pm.paired_cluster_bootstrap(paired, n_boot=N_BOOT, seed=seed)
        for row in metric_rows:
            row.update({
                "pair_id": spec["pair_id"], "comparison_direction": f"{spec['b_label']}_minus_{spec['a_label']}",
                "analysis_label_a": spec["a_label"], "analysis_label_b": spec["b_label"],
                "analysis_id_a": "main20k_real" if spec["a_label"] == "originalreal"
                    else ("main20k_constant" if "constant" in spec["a_label"] else
                          "main20k_common8_mask" if "common8" in spec["a_label"] else "main20k_real"),
                "analysis_id_b": "main20k_real" if "real" in spec["b_label"]
                    else "main20k_common8_mask" if "common8" in spec["b_label"] else "main20k_constant",
                "population_alias": spec["population"], "db_pair_key": spec["db"],
                "n_paired_patients": int(paired["subject"].nunique()),
                "n_paired_stays": int(len(paired)), "n_bootstrap": N_BOOT,
                "seed": seed, "bootstrap_unit": protocol["bootstrap_unit"],
                "clinical_input_availability_verified": False,
                "fixedselection_interpretation": (
                    "conditional on original sampled token-key selection; this is not a pure store-time effect"
                    if spec["population"] == FIXED_DB else "full H24-rebuild availability variant"),
            })
            all_rows.append(row)
        flow.update({
            "pair_id": spec["pair_id"], "analysis_label_a": spec["a_label"],
            "analysis_label_b": spec["b_label"], "population_alias": spec["population"],
            "cross_version_alias_rekeyed_to_original_mimic": bool(spec["cross_version_alias"]),
            "exact_intersection_used": True,
        })
        flow_rows.append(flow)
        print(json.dumps({"stage": "availability_pair_complete", "pair_id": spec["pair_id"],
                          "n_paired_stays": int(len(paired)),
                          "n_paired_patients": int(paired["subject"].nunique())}, sort_keys=True), flush=True)
    if len(all_rows) != len(SOURCE_PAIR_IDS) * len(pm.METRICS) or len(flow_rows) != len(SOURCE_PAIR_IDS):
        raise AssertionError("availability pair result grid is incomplete")
    metrics_path = OUT / "paired_availability_metrics.csv"
    flow_path = OUT / "paired_availability_sample_flow.csv"
    pd.DataFrame(all_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(flow_rows).to_csv(flow_path, index=False)
    manifest = {
        "status": "complete", "schema": "ehr_fm_availability_source_paired_metrics_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "new_prediction_run_id": new_manifest.get("run_id"),
        "new_prediction_manifest_path": str(NEW_MANIFEST_PATH),
        "new_prediction_manifest_sha256": _sha256(NEW_MANIFEST_PATH),
        "source_reuse_completion_path": str(NEW_COMPLETION_PATH),
        "source_reuse_completion_sha256": _sha256(NEW_COMPLETION_PATH),
        "original_prediction_run_id": old_manifest.get("run_id"),
        "original_prediction_manifest_path": str(OLD_MANIFEST_PATH),
        "original_prediction_manifest_sha256": _sha256(OLD_MANIFEST_PATH),
        "source_score_pairs": SOURCE_PAIR_IDS,
        "metrics": list(pm.METRICS), "bootstrap_replicates": N_BOOT,
        "estimand": "Paired metric difference B minus A on exact original-heldout stay intersections; patients are resampled with all their stays retained.",
        "clinical_input_availability_verified": False,
        "clinical_promotion_allowed": False,
        "refit_or_reencode_performed": False,
        "scope_notes": [
            "All predictions are retrospective measurement-time scores; clinical-time availability is unverified.",
            "Fixedselection comparisons are conditional on the original token-key selection and do not isolate a pure store-time effect.",
            "Cross-version cohort aliases are linked only after exact stay, subject, source-row, split, and label identity checks.",
        ],
        "identity_bridge": bridge,
        "source_eligibility_paths": {"mimic": str(OLD_GATE), FULL_DB: str(AVAIL_GATE), FIXED_DB: str(AVAIL_GATE)},
        "paired_metrics_path": str(metrics_path), "paired_metrics_sha256": _sha256(metrics_path),
        "paired_metric_rows": len(all_rows),
        "paired_flow_path": str(flow_path), "paired_flow_sha256": _sha256(flow_path),
        "paired_flow_rows": len(flow_rows),
        "patient_level_exported_to_local": False,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    manifest_path = OUT / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "availability_metrics_complete", "manifest": str(manifest_path),
                      "manifest_sha256": _sha256(manifest_path), "metric_rows": len(all_rows),
                      "flow_rows": len(flow_rows), "elapsed_seconds": manifest["elapsed_seconds"]},
                     sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
