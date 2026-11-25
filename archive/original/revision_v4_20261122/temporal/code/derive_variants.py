#!/usr/bin/env python3
"""Derive independent primary and strict v4.1 cohorts from cached metadata.

This intentionally reads only the preserved all-row eligibility parquet files
and the completed aggregate source-audit log. It does not reopen raw source or
event tables.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

print("derive_stage=module_import_begin", flush=True)

import numpy as np
import pandas as pd

print("derive_stage=module_import_complete", flush=True)


ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102/revision_v4_20261122/temporal")
ELIG = ROOT / "results/eligibility"
SNAPSHOT = ELIG / "v4_pre_4_1_20261122_224500"
METADATA_CACHE = ELIG / "source_metadata_v4_1"
SOURCE_AUDIT = ROOT / "logs/source_audit_20261122_223100.log"
VARIANTS = {
    "landmark_v4_1": "primary_retrospective_landmark",
    "strict_v4_1": "fully_ordered_sensitivity",
}
REQUIRED = {
    "db", "stay_id", "subject", "subject_id", "y_mort", "endpoint",
    "eligible_24h", "temporal_verified", "adult_eligible",
    "input_available_24h", "input_window_le_24h_verified",
    "input_availability_cutoff_hr", "source_endpoint_known",
    "source_endpoint_unknown", "source_y_mort", "source_y_mort_match",
    "source_time_issue", "source_time_status_conflict",
    "source_landmark_contradiction", "icu_exit_after_hospital_exit_after_24",
    "icu_exit_after_hospital_exit_after_24_gap_hr", "icu_exit_hr",
    "hospital_exit_hr", "hospital_death_time_hr", "icu_death_time_hr",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def read_jsonl_final(path: Path) -> dict[str, Any]:
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            value = json.loads(line)
            if isinstance(value, dict):
                return value
    raise ValueError(f"no final JSON audit record in {path}")


def verify_snapshot() -> tuple[dict[str, Any], dict[str, Any]]:
    sums = SNAPSHOT / "SHA256SUMS"
    for line in sums.read_text(encoding="utf-8").splitlines():
        expected, filename = line.strip().split(maxsplit=1)
        filename = filename.lstrip("* ")
        path = SNAPSHOT / filename
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"preserved snapshot hash mismatch: {filename}")
    prior = json.loads((SNAPSHOT / "eligibility_manifest.json").read_text(encoding="utf-8"))
    audit = read_jsonl_final(SOURCE_AUDIT)
    snapshot_schema = prior.get("schema_version")
    if snapshot_schema not in (None, "4.1"):
        raise ValueError(f"unsupported preserved snapshot schema version: {snapshot_schema}")
    if audit.get("audit_only") is not True or audit.get("schema_version") != "4.1":
        raise ValueError("expected completed aggregate-only v4.1 source audit")
    if not isinstance(prior.get("cohorts"), dict) or not isinstance(audit.get("cohorts"), dict):
        raise ValueError("snapshot or aggregate audit lacks cohort records")
    return prior, audit


def bool_col(frame: pd.DataFrame, name: str, default: bool = False) -> pd.Series:
    if name not in frame:
        return pd.Series(default, index=frame.index, dtype=bool)
    return frame[name].fillna(default).astype(bool)


def gap_summary(frame: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
    gap = bool_col(frame, "source_postlandmark_ordering_issue") & mask
    values = pd.to_numeric(frame.loc[gap, "icu_exit_after_hospital_exit_after_24_gap_hr"], errors="coerce")
    values = values[np.isfinite(values)]
    if len(values):
        q = np.quantile(values.to_numpy(dtype=float), [0.05, 0.5, 0.95])
        quantiles: dict[str, float | None] = {"p05": float(q[0]), "median": float(q[1]), "p95": float(q[2])}
    else:
        quantiles = {"p05": None, "median": None, "p95": None}
    labels = pd.to_numeric(frame.loc[gap, "source_y_mort"], errors="coerce")
    return {
        "rows": int(gap.sum()),
        "duration_hr_quantiles": quantiles,
        "source_label_distribution": {
            "hospital_death_1": int(labels.eq(1).sum()),
            "hospital_survival_0": int(labels.eq(0).sum()),
            "unknown_or_nonbinary": int((~labels.isin([0, 1])).sum()),
        },
    }


def derive_frame(db: str, source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    missing = REQUIRED.difference(source.columns)
    if missing:
        raise ValueError(f"{db} cached metadata missing required per-row fields: {sorted(missing)}")
    frame = source.copy()
    if frame["db"].isna().any() or not frame["db"].eq(db).all():
        raise ValueError(f"{db} cached rows have null or mismatched database keys")
    if frame[["stay_id", "subject", "subject_id"]].isna().any().any():
        raise ValueError(f"{db} cached rows have null identifiers")
    if frame.duplicated(["db", "stay_id", "subject"]).any():
        raise ValueError(f"{db} cached rows have duplicate keys")

    known = bool_col(frame, "source_endpoint_known")
    unknown = bool_col(frame, "source_endpoint_unknown", True)
    if not unknown.equals(~known):
        raise ValueError(f"{db} source endpoint known/unknown per-row flags disagree")
    source_y = pd.to_numeric(frame["source_y_mort"], errors="coerce")
    y = pd.to_numeric(frame["y_mort"], errors="coerce")
    match = bool_col(frame, "source_y_mort_match")
    binary_match = known & source_y.isin([0, 1]) & y.isin([0, 1]) & source_y.eq(y) & match

    icu_exit = pd.to_numeric(frame["icu_exit_hr"], errors="coerce")
    hospital_exit = pd.to_numeric(frame["hospital_exit_hr"], errors="coerce")
    hospital_death = pd.to_numeric(frame["hospital_death_time_hr"], errors="coerce")
    icu_death = pd.to_numeric(frame["icu_death_time_hr"], errors="coerce")
    finite_icu = pd.Series(np.isfinite(icu_exit), index=frame.index)
    finite_hospital = pd.Series(np.isfinite(hospital_exit), index=frame.index)
    finite_hospital_death = pd.Series(np.isfinite(hospital_death), index=frame.index)
    finite_icu_death = pd.Series(np.isfinite(icu_death), index=frame.index)
    death_by_24 = (finite_hospital_death & hospital_death.le(24)) | (finite_icu_death & icu_death.le(24))
    pre_index_death = (finite_hospital_death & hospital_death.lt(0)) | (finite_icu_death & icu_death.lt(0))

    post_gap = finite_icu & finite_hospital & icu_exit.gt(hospital_exit) & hospital_exit.gt(24)
    cached_gap = bool_col(frame, "icu_exit_after_hospital_exit_after_24")
    if not post_gap.equals(cached_gap):
        raise ValueError(f"{db} cached post-24h exit-order flags disagree with per-row exit times")
    frame["source_postlandmark_ordering_issue"] = post_gap.astype(bool)

    status_conflict = bool_col(frame, "source_time_status_conflict")
    legacy_issue = bool_col(frame, "source_time_issue")
    early_contradiction = bool_col(frame, "source_landmark_contradiction")
    exit_negative = (finite_icu & icu_exit.lt(0)) | (finite_hospital & hospital_exit.lt(0))
    exit_nonfinite = ((frame["icu_exit_hr"].notna() & ~finite_icu)
                      | (frame["hospital_exit_hr"].notna() & ~finite_hospital))
    death_nonfinite = ((frame["hospital_death_time_hr"].notna() & ~finite_hospital_death)
                       | (frame["icu_death_time_hr"].notna() & ~finite_icu_death))
    hospital_before_icu = finite_hospital & finite_icu & hospital_exit.lt(icu_exit) & ~post_gap
    death_after_hospital = finite_hospital_death & finite_hospital & hospital_death.gt(hospital_exit)
    # The legacy flag includes post-landmark ordering gaps. Remove only that
    # standalone category; independently detect every other source-time issue.
    landmark_issue = (
        (legacy_issue & ~post_gap) | status_conflict | early_contradiction
        | death_by_24 | pre_index_death | exit_negative | exit_nonfinite
        | death_nonfinite | hospital_before_icu | death_after_hospital
    )
    frame["source_landmark_issue"] = landmark_issue.astype(bool)

    input_verified = bool_col(frame, "input_window_le_24h_verified")
    input_available = bool_col(frame, "input_available_24h")
    adult = bool_col(frame, "adult_eligible")
    cutoff = pd.to_numeric(frame["input_availability_cutoff_hr"], errors="coerce")
    common = (
        binary_match & finite_icu & icu_exit.gt(24)
        & finite_hospital & hospital_exit.gt(24)
        & ~death_by_24 & ~pre_index_death & ~landmark_issue
        & ~status_conflict & adult & input_verified & input_available & cutoff.le(24)
    ).fillna(False).astype(bool)
    primary = common
    strict = common & ~post_gap

    # Post-landmark ordering gaps do not alter whether a patient survived and
    # remained in ICU at 24h. Restore those two derived flags for primary rows.
    for column in ("alive_at_24h", "in_icu_at_24h"):
        if column not in frame:
            frame[column] = pd.array([pd.NA] * len(frame), dtype="boolean")
        else:
            frame[column] = pd.array(frame[column], dtype="boolean")
        frame.loc[primary, column] = True
    frame["temporal_verified"] = (
        binary_match & finite_icu & finite_hospital & ~landmark_issue
        & input_verified & cutoff.le(24)
    ).fillna(False).astype(bool)
    frame["retrospective_landmark_prediction_allowed"] = True
    frame["ready_for_future24h_landmark_analysis"] = True
    frame["ready_for_retrospective_24h_measurement_analysis"] = True
    frame["ready_for_future24h_clinical_claim"] = False
    frame["inputs_temporal_scope"] = "retrospective_measurement_time"
    frame["input_scope"] = "retrospective_measurement_time"
    frame["clinical_input_availability_verified"] = False

    frame["eligible_24h"] = primary
    frame.loc[primary, "reason"] = "eligible_24h"
    frame.loc[~primary & bool_col(frame, "source_endpoint_unknown", True), "reason"] = "source_endpoint_unknown"
    mismatch = known & ~match
    frame.loc[~primary & mismatch, "reason"] = "source_outcome_label_mismatch"
    frame.loc[~primary & landmark_issue, "reason"] = "source_landmark_issue"
    frame.loc[~primary & post_gap, "reason"] = "postlandmark_ordering_issue"

    audit = {
        "n_rows": int(len(frame)),
        "n_eligible_24h": int(primary.sum()),
        "n_eligible_positive": int((primary & y.eq(1)).sum()),
        "n_eligible_negative": int((primary & y.eq(0)).sum()),
        "n_temporal_verified": int(frame["temporal_verified"].sum()),
        "n_source_endpoint_unknown_rows": int(unknown.sum()),
        "n_source_endpoint_unknown_eligible": int((unknown & primary).sum()),
        "n_source_label_mismatch_rows": int(mismatch.sum()),
        "n_source_label_mismatch_eligible": int((mismatch & primary).sum()),
        "n_source_time_status_conflicts": int(status_conflict.sum()),
        "n_source_time_status_conflicts_eligible": int((status_conflict & primary).sum()),
        "n_source_landmark_issue_rows": int(landmark_issue.sum()),
        "n_source_landmark_issue_eligible": int((landmark_issue & primary).sum()),
        "n_source_postlandmark_ordering_issue_rows": int(post_gap.sum()),
        "n_source_postlandmark_ordering_issue_eligible": int((post_gap & primary).sum()),
        "n_known_death_by_24h": int(death_by_24.sum()),
        "n_pre_index_death": int(pre_index_death.sum()),
        "n_negative_hour_input_eligible": int((~input_verified & primary).sum()),
        "n_input_unavailable_eligible": int((~input_available & primary).sum()),
        "source_postlandmark_ordering_gap_all": gap_summary(frame, pd.Series(True, index=frame.index)),
        "source_postlandmark_ordering_gap_eligible": gap_summary(frame, primary),
        "source_label_distribution_eligible": {
            "hospital_death_1": int((primary & y.eq(1)).sum()),
            "hospital_survival_0": int((primary & y.eq(0)).sum()),
        },
        "selected_keys_labels_sha256": hashlib.sha256("\n".join(sorted(
            f"{db}|{sid}|{sub}|{int(label)}"
            for sid, sub, label in zip(frame.loc[primary, "stay_id"].astype(str),
                                       frame.loc[primary, "subject"].astype(str),
                                       y.loc[primary].astype(int))
        )).encode("utf-8")).hexdigest(),
        "strict_eligible_24h": int(strict.sum()),
        "strict_eligible_positive": int((strict & y.eq(1)).sum()),
        "strict_eligible_negative": int((strict & y.eq(0)).sum()),
    }
    return frame, audit


def make_manifest_entry(db: str, old: dict[str, Any], frame: pd.DataFrame,
                        audit: dict[str, Any], variant: str, out_path: Path) -> dict[str, Any]:
    primary = variant == "landmark_v4_1"
    selected = bool_col(frame, "eligible_24h")
    labels = pd.to_numeric(frame["y_mort"], errors="coerce")
    entry = dict(old)
    entry.update({
        "variant": variant,
        "variant_role": VARIANTS[variant],
        "population_rule": (
            "known binary source endpoint matching y_mort; adult; verified 0-24h input window; "
            "finite ICU and hospital exits both after 24h; no known death by 24h, pre-index death, "
            "or source landmark issue; post-24h administrative exit-order gaps retained and flagged"
            if primary else
            "primary landmark_v4_1 rule plus exclusion of post-24h ICU-after-hospital administrative exit-order gaps"
        ),
        "ready_for_future24h_clinical_claim": False,
        "retrospective_landmark_prediction_allowed": True,
        "ready_for_future24h_landmark_analysis": True,
        "ready_for_retrospective_24h_measurement_analysis": True,
        "inputs_temporal_scope": "retrospective_measurement_time",
        "input_scope": "retrospective_measurement_time",
        "clinical_input_availability_verified": False,
        "output": str(out_path),
    })
    counts = dict(old.get("counts", {}))
    counts.update(audit)
    counts.update({
        "n_rows": int(len(frame)),
        "n_unique_stay_subject": int(frame[["stay_id", "subject_id"]].drop_duplicates().shape[0]),
        "n_eligible_24h": int(selected.sum()),
        "n_eligible_positive": int((selected & labels.eq(1)).sum()),
        "n_eligible_negative": int((selected & labels.eq(0)).sum()),
        "n_source_landmark_contradictions": int(bool_col(frame, "source_landmark_contradiction").sum()),
        "n_icu_exit_after_hospital_exit_after_24": int(bool_col(frame, "source_postlandmark_ordering_issue").sum()),
        "n_icu_exit_after_hospital_exit_after_24_eligible": int((selected & bool_col(frame, "source_postlandmark_ordering_issue")).sum()),
        "post24_icu_after_hospital_exit_gap_rows": audit["source_postlandmark_ordering_gap_all"]["rows"],
        "post24_icu_after_hospital_exit_gap_duration_hr_quantiles": audit["source_postlandmark_ordering_gap_all"]["duration_hr_quantiles"],
        "post24_icu_after_hospital_exit_gap_source_label_distribution": audit["source_postlandmark_ordering_gap_all"]["source_label_distribution"],
    })
    entry["counts"] = counts
    entry["source_issue_definitions"] = {
        "source_landmark_issue": "Source timing or status conflict that can invalidate alive/in-ICU-at-24h interpretation; excludes a post-24h ICU-exit-after-hospital-exit gap by itself.",
        "source_postlandmark_ordering_issue": "Finite ICU exit is after finite hospital exit, with hospital exit after 24h; retained in landmark_v4_1 and excluded from strict_v4_1.",
    }
    entry["selected_keys_labels_sha256"] = audit["selected_keys_labels_sha256"]
    return entry


def main() -> None:
    print("derive_stage=snapshot_verification_begin", flush=True)
    prior, source_audit = verify_snapshot()
    print("derive_stage=snapshot_verification_complete", flush=True)
    cache_manifest_path = METADATA_CACHE / "metadata_cache_manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    if cache_manifest.get("schema_version") != "4.1":
        raise ValueError("row-level source metadata cache is not schema 4.1")
    if cache_manifest.get("source_audit_log_sha256") != sha256(SOURCE_AUDIT):
        raise ValueError("row-level metadata cache is bound to a different source audit log")
    prepared: dict[str, pd.DataFrame] = {}
    audits: dict[str, dict[str, Any]] = {}
    old_entries = prior["cohorts"]
    for db in ("mimic", "eicu"):
        if db not in old_entries or db not in source_audit["cohorts"]:
            raise ValueError(f"{db} missing from preserved manifest or aggregate audit")
        old = old_entries[db]
        source_record = source_audit["cohorts"][db]
        cache_record = cache_manifest.get("cohorts", {}).get(db)
        if not isinstance(cache_record, dict):
            raise ValueError(f"{db} missing from row-level metadata cache manifest")
        table = METADATA_CACHE / f"{db}.parquet"
        audit_hashes = source_record.get("source_hashes", {})
        if old.get("sourcehashes") != audit_hashes:
            raise ValueError(f"{db} source hashes differ between snapshot and successful source audit")
        if cache_record.get("sourcehashes") != audit_hashes:
            raise ValueError(f"{db} row-level metadata cache source hashes differ from audit")
        print(f"derive_stage={db}_metadata_read_begin", flush=True)
        frame = pd.read_parquet(table)
        print(f"derive_stage={db}_metadata_read_complete rows={len(frame)}", flush=True)
        if cache_record.get("table_sha256") != sha256(table):
            raise ValueError(f"{db} row-level metadata cache table hash mismatch")
        if int(source_record.get("counts", {}).get("n_rows", -1)) != len(frame):
            raise ValueError(f"{db} row count differs from completed aggregate source audit")
        if source_record.get("source_join_coverage") != 1.0 or old.get("source_join_coverage") != 1.0:
            raise ValueError(f"{db} source join coverage is not complete")
        # Explicitly confirm row-level endpoint-status fields match the audited
        # unknown counts; aggregate totals alone are never used as a row mask.
        if "source_endpoint_known" not in frame or "source_endpoint_unknown" not in frame:
            raise ValueError(f"{db} cached parquet lacks row-level endpoint status fields")
        unknown = bool_col(frame, "source_endpoint_unknown", True)
        expected_unknown = int(source_record.get("counts", {}).get("n_source_endpoint_unknown_rows", -1))
        if int(unknown.sum()) != expected_unknown:
            raise ValueError(f"{db} row-level unknown endpoint count does not match audit")
        prepared[db], audits[db] = derive_frame(db, frame)

    for variant, role in VARIANTS.items():
        final_dir = ELIG / variant
        if final_dir.exists():
            raise FileExistsError(f"refusing to overwrite existing derived cohort directory: {final_dir}")
        stage = ELIG / f".{variant}.staging"
        if stage.exists():
            raise FileExistsError(f"staging directory already exists; inspect before recovery: {stage}")
        stage.mkdir(parents=True)
        print(f"derive_stage={variant}_write_begin", flush=True)
        try:
            cohort_entries: dict[str, Any] = {}
            strict = variant == "strict_v4_1"
            for db, full_frame in prepared.items():
                frame = full_frame.copy()
                if strict:
                    frame["eligible_24h"] = bool_col(frame, "eligible_24h") & ~bool_col(frame, "source_postlandmark_ordering_issue")
                    frame.loc[frame["eligible_24h"], "reason"] = "eligible_24h"
                    frame.loc[~frame["eligible_24h"] & bool_col(frame, "source_postlandmark_ordering_issue"), "reason"] = "postlandmark_ordering_issue"
                # Keep the entire all-row metadata table; only its variant mask differs.
                table = stage / f"{db}.parquet"
                tmp = table.with_name(table.name + ".tmp")
                frame.to_parquet(tmp, index=False)
                os.replace(tmp, table)
                print(f"derive_stage={variant}_{db}_table_written rows={len(frame)}", flush=True)
                entry = make_manifest_entry(db, old_entries[db], frame, audits[db], variant, final_dir / table.name)
                entry["eligibility_table_sha256"] = sha256(table)
                entry["counts"]["n_eligible_24h"] = int(bool_col(frame, "eligible_24h").sum())
                entry["counts"]["n_eligible_positive"] = int((bool_col(frame, "eligible_24h") & pd.to_numeric(frame["y_mort"], errors="coerce").eq(1)).sum())
                entry["counts"]["n_eligible_negative"] = int((bool_col(frame, "eligible_24h") & pd.to_numeric(frame["y_mort"], errors="coerce").eq(0)).sum())
                selected = bool_col(frame, "eligible_24h")
                post_gap = bool_col(frame, "source_postlandmark_ordering_issue")
                entry["counts"]["n_source_postlandmark_ordering_issue_eligible"] = int((selected & post_gap).sum())
                entry["counts"]["n_icu_exit_after_hospital_exit_after_24_eligible"] = int((selected & post_gap).sum())
                entry["counts"]["source_postlandmark_ordering_gap_eligible"] = gap_summary(frame, selected)
                entry["counts"]["source_label_distribution_eligible"] = {
                    "hospital_death_1": int((selected & pd.to_numeric(frame["y_mort"], errors="coerce").eq(1)).sum()),
                    "hospital_survival_0": int((selected & pd.to_numeric(frame["y_mort"], errors="coerce").eq(0)).sum()),
                }
                entry["selected_keys_labels_sha256"] = hashlib.sha256("\n".join(sorted(
                    f"{db}|{sid}|{sub}|{int(label)}"
                    for sid, sub, label in zip(frame.loc[bool_col(frame, "eligible_24h"), "stay_id"].astype(str),
                                               frame.loc[bool_col(frame, "eligible_24h"), "subject"].astype(str),
                                               pd.to_numeric(frame.loc[bool_col(frame, "eligible_24h"), "y_mort"], errors="coerce").astype(int))
                )).encode("utf-8")).hexdigest()
                cohort_entries[db] = entry
            manifest = {
                "schema_version": "4.1",
                "variant": variant,
                "variant_role": role,
                "derivation": "cached all-row metadata from preserved v4 snapshot; no raw-source or event-table rescan",
                "parent_snapshot": str(SNAPSHOT),
                "parent_snapshot_sha256s": {name: sha256(SNAPSHOT / name) for name in ("mimic.parquet", "eicu.parquet", "eligibility_manifest.json", "core_verification.json")},
                "row_level_metadata_cache": str(METADATA_CACHE),
                "row_level_metadata_cache_manifest_sha256": sha256(cache_manifest_path),
                "source_audit_log": str(SOURCE_AUDIT),
                "source_audit_log_sha256": sha256(SOURCE_AUDIT),
                "cohorts": cohort_entries,
            }
            atomic_json(stage / "eligibility_manifest.json", manifest)
            stage.rename(final_dir)
            print(f"derive_stage={variant}_write_complete", flush=True)
            print(f"variant={variant} output={final_dir}", flush=True)
            for db in ("mimic", "eicu"):
                entry = cohort_entries[db]
                c = entry["counts"]
                print(
                    f"{variant} {db}: rows={c['n_rows']} eligible={c['n_eligible_24h']} "
                    f"positive={c['n_eligible_positive']} negative={c['n_eligible_negative']} "
                    f"post24_gap_all={c['n_source_postlandmark_ordering_issue_rows']} "
                    f"post24_gap_selected={c['n_source_postlandmark_ordering_issue_eligible']} "
                    f"landmark_issues={c['n_source_landmark_issue_rows']} "
                    f"unknown_endpoint={c['n_source_endpoint_unknown_rows']} "
                    f"table_sha256={entry['eligibility_table_sha256']}", flush=True)
        except Exception:
            if stage.exists():
                shutil.rmtree(stage)
            raise


if __name__ == "__main__":
    main()
