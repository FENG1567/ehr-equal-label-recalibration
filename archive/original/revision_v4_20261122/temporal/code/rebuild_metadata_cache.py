#!/usr/bin/env python3
"""Rebuild row-level source metadata from the preserved v4 all-row snapshot.

Only the compact MIMIC/EICU source metadata tables are rejoined. This never
opens an event table and preserves the snapshot's verified input-window mask.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

print("metadata_cache_stage=module_import_begin", flush=True)
import pandas as pd
import repair_landmark as repair
print("metadata_cache_stage=module_import_complete", flush=True)


ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102/revision_v4_20261122/temporal")
ELIG = ROOT / "results/eligibility"
SNAPSHOT = ELIG / "v4_pre_4_1_20261122_224500"
CACHE = ELIG / "source_metadata_v4_1"
SOURCE_AUDIT = ROOT / "logs/source_audit_20261122_223100.log"
PROJECT = Path("/data/admin/fengshuo/ehr_fm_20261102")
INTERIM = PROJECT / "data/interim"
PROCESSED = PROJECT / "data/processed"


def file_sha(path: Path) -> str:
    return repair.sha256(path)


def read_jsonl_final(path: Path) -> dict[str, Any]:
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        if line.lstrip().startswith("{"):
            value = json.loads(line)
            if isinstance(value, dict):
                return value
    raise ValueError(f"no JSON record in aggregate audit log: {path}")


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def verify_snapshot_hashes() -> dict[str, str]:
    checks: dict[str, str] = {}
    for line in (SNAPSHOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, filename = line.strip().split(maxsplit=1)
        filename = filename.lstrip("* ")
        path = SNAPSHOT / filename
        actual = file_sha(path)
        if actual != expected:
            raise ValueError(f"preserved v4 snapshot hash mismatch: {filename}")
        checks[filename] = actual
    return checks


def bool_series(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        raise ValueError(f"snapshot lacks cached input audit mask {name}")
    return frame[name].fillna(False).astype(bool)


def run_source_join(db: str, snapshot_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if db == "mimic":
        joined, evidence = repair.mimic_metadata(snapshot_frame)
    elif db == "eicu":
        joined, evidence = repair.eicu_metadata(snapshot_frame)
    else:
        raise ValueError(f"unsupported source metadata cohort: {db}")
    joined["db"] = db
    joined["subject"] = joined["subject_id"].map(repair.canon_id)
    # These fields come from source_time_checks in the existing implementations.
    gap, gap_hr, gap_evidence = repair.post24_gap_summary(joined)
    joined["icu_exit_after_hospital_exit_after_24"] = gap.astype(bool)
    joined["icu_exit_after_hospital_exit_after_24_gap_hr"] = gap_hr
    evidence["post24_gap_summary"] = gap_evidence
    return joined, evidence


def main() -> None:
    print("metadata_cache_stage=snapshot_hash_check_begin", flush=True)
    snapshot_hashes = verify_snapshot_hashes()
    prior = json.loads((SNAPSHOT / "eligibility_manifest.json").read_text(encoding="utf-8"))
    audit = read_jsonl_final(SOURCE_AUDIT)
    if audit.get("audit_only") is not True or audit.get("schema_version") != "4.1":
        raise ValueError("completed aggregate-only source audit is missing or not v4.1")
    print("metadata_cache_stage=snapshot_hash_check_complete", flush=True)
    if CACHE.exists():
        raise FileExistsError(f"refusing to overwrite existing metadata cache: {CACHE}")
    stage = CACHE.with_name(".source_metadata_v4_1.staging")
    if stage.exists():
        raise FileExistsError(f"metadata staging directory exists; inspect before recovery: {stage}")
    stage.mkdir(parents=True)
    try:
        cohort_entries: dict[str, Any] = {}
        for db in ("mimic", "eicu"):
            old = prior.get("cohorts", {}).get(db)
            audit_record = audit.get("cohorts", {}).get(db)
            if not isinstance(old, dict) or not isinstance(audit_record, dict):
                raise ValueError(f"{db} missing from preserved snapshot or aggregate audit")
            expected_hashes = old.get("sourcehashes", {})
            audited_hashes = audit_record.get("source_hashes", {})
            if expected_hashes != audited_hashes:
                raise ValueError(f"{db} source hashes differ between snapshot and successful audit")

            cohort_path = INTERIM / f"cohort_{db}.parquet"
            sequence_path = PROCESSED / f"seq_{db}.npz"
            if file_sha(cohort_path) != expected_hashes.get("cohort_parquet"):
                raise ValueError(f"{db} current cohort parquet hash differs from preserved snapshot")
            if file_sha(sequence_path) != expected_hashes.get("sequence_npz"):
                raise ValueError(f"{db} current sequence NPZ hash differs from preserved snapshot")

            snapshot_path = SNAPSHOT / f"{db}.parquet"
            print(f"metadata_cache_stage={db}_snapshot_read_begin", flush=True)
            base = pd.read_parquet(snapshot_path)
            print(f"metadata_cache_stage={db}_snapshot_read_complete rows={len(base)}", flush=True)
            if len(base) != int(old.get("counts", {}).get("n_rows", -1)):
                raise ValueError(f"{db} snapshot row count differs from preserved manifest")
            if bool_series(base, "input_window_le_24h_verified").sum() != len(base) - int(old.get("counts", {}).get("n_negative_hour_input_rows", -1)):
                raise ValueError(f"{db} cached feature-negative mask count differs from preserved manifest")
            audit_negative = int(audit_record.get("counts", {}).get("n_negative_hour_input_rows", -1))
            manifest_negative = int(old.get("counts", {}).get("n_negative_hour_input_rows", -2))
            if audit_negative != manifest_negative:
                raise ValueError(f"{db} feature-negative mask count differs from aggregate audit")

            print(f"metadata_cache_stage={db}_source_join_begin", flush=True)
            joined, source_evidence = run_source_join(db, base)
            print(f"metadata_cache_stage={db}_source_join_complete rows={len(joined)}", flush=True)
            if source_evidence.get("source_join_coverage") != 1.0 or len(joined) != len(base):
                raise ValueError(f"{db} row-level source metadata join is incomplete")
            if source_evidence.get("source_hashes") != {k: v for k, v in expected_hashes.items() if k not in {"cohort_parquet", "sequence_npz"}}:
                raise ValueError(f"{db} small raw source metadata hashes differ from audited hashes")

            required = {
                "source_endpoint_known", "source_endpoint_unknown", "source_y_mort",
                "source_y_mort_match", "source_time_status_conflict",
                "source_landmark_contradiction", "icu_exit_after_hospital_exit_after_24",
                "icu_exit_after_hospital_exit_after_24_gap_hr", "icu_exit_hr",
                "hospital_exit_hr", "hospital_death_time_hr", "icu_death_time_hr",
            }
            absent = required.difference(joined.columns)
            if absent:
                raise ValueError(f"{db} source join missing required row-level fields: {sorted(absent)}")
            counts = audit_record.get("counts", {})
            unknown_rows = int(bool_series(joined, "source_endpoint_unknown").sum())
            mismatch_rows = int(joined["source_y_mort_match"].eq(False).fillna(False).sum())
            status_conflicts = int(bool_series(joined, "source_time_status_conflict").sum())
            landmark_contradictions = int(bool_series(joined, "source_landmark_contradiction").sum())
            gaps = int(bool_series(joined, "icu_exit_after_hospital_exit_after_24").sum())
            expected = {
                "source_endpoint_unknown_rows": int(counts.get("n_source_endpoint_unknown_rows", -1)),
                "source_label_mismatch_rows": int(counts.get("n_source_label_mismatch_rows", -1)),
                "source_time_status_conflicts_total": int(counts.get("n_source_time_status_conflicts_total", -1)),
                "source_landmark_contradictions": int(counts.get("n_source_landmark_contradictions", -1)),
                "post24_exit_order_gaps": int(counts.get("n_icu_exit_after_hospital_exit_after_24", -1)),
            }
            actual = {
                "source_endpoint_unknown_rows": unknown_rows,
                "source_label_mismatch_rows": mismatch_rows,
                "source_time_status_conflicts_total": status_conflicts,
                "source_landmark_contradictions": landmark_contradictions,
                "post24_exit_order_gaps": gaps,
            }
            if actual != expected:
                raise ValueError(f"{db} per-row source metadata counts disagree with aggregate audit: {actual} vs {expected}")

            table = stage / f"{db}.parquet"
            tmp = table.with_name(table.name + ".tmp")
            joined.to_parquet(tmp, index=False)
            os.replace(tmp, table)
            cohort_entries[db] = {
                "rows": int(len(joined)),
                "source_join_coverage": float(source_evidence["source_join_coverage"]),
                "sourcehashes": expected_hashes,
                "source_metadata_evidence": {k: v for k, v in source_evidence.items() if k not in {"source_hashes", "source_files"}},
                "table_sha256": file_sha(table),
                "negative_hour_input_rows_preserved": manifest_negative,
                "source_endpoint_unknown_rows": unknown_rows,
                "source_label_mismatch_rows": mismatch_rows,
                "source_time_status_conflicts": status_conflicts,
                "source_landmark_contradictions": landmark_contradictions,
                "post24_exit_order_gaps": gaps,
            }
            print(
                f"metadata_cache {db}: rows={len(joined)} unknown_endpoint={unknown_rows} "
                f"label_mismatch={mismatch_rows} landmark_contradictions={landmark_contradictions} "
                f"post24_gaps={gaps} input_mask_negative={manifest_negative} "
                f"table_sha256={cohort_entries[db]['table_sha256']}", flush=True)

        manifest = {
            "schema_version": "4.1",
            "purpose": "row-level source metadata cache reconstructed from preserved all-row v4 tables",
            "raw_tables_read": "only source timing/status metadata tables; no event tables read",
            "parent_snapshot": str(SNAPSHOT),
            "parent_snapshot_sha256s": snapshot_hashes,
            "source_audit_log": str(SOURCE_AUDIT),
            "source_audit_log_sha256": file_sha(SOURCE_AUDIT),
            "cohorts": cohort_entries,
        }
        atomic_json(stage / "metadata_cache_manifest.json", manifest)
        stage.rename(CACHE)
        print(f"metadata_cache_stage=complete output={CACHE}", flush=True)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


if __name__ == "__main__":
    main()
