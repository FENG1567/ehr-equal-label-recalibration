#!/usr/bin/env python3
"""Verify selected rows and hashes for primary/strict v4.1 cohort variants."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102/revision_v4_20261122/temporal")
ELIG = ROOT / "results/eligibility"
SOURCE_AUDIT = ROOT / "logs/source_audit_20261122_223100.log"
VARIANTS = ("landmark_v4_1", "strict_v4_1")


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


def bool_col(frame: pd.DataFrame, name: str, default: bool = False) -> pd.Series:
    return frame[name].fillna(default).astype(bool)


def selected_digest(db: str, frame: pd.DataFrame, selected: pd.Series) -> str:
    labels = pd.to_numeric(frame.loc[selected, "y_mort"], errors="coerce").astype(int)
    material = "\n".join(sorted(
        f"{db}|{stay}|{subject}|{label}"
        for stay, subject, label in zip(
            frame.loc[selected, "stay_id"].astype(str),
            frame.loc[selected, "subject"].astype(str), labels,
        )
    ))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def gap_summary(frame: pd.DataFrame, selected: pd.Series) -> dict[str, Any]:
    mask = bool_col(frame, "source_postlandmark_ordering_issue") & selected
    values = pd.to_numeric(frame.loc[mask, "icu_exit_after_hospital_exit_after_24_gap_hr"], errors="coerce")
    values = values[np.isfinite(values)]
    if len(values):
        q = np.quantile(values.to_numpy(dtype=float), [0.05, 0.5, 0.95])
        quantiles = {"p05": float(q[0]), "median": float(q[1]), "p95": float(q[2])}
    else:
        quantiles = {"p05": None, "median": None, "p95": None}
    labels = pd.to_numeric(frame.loc[mask, "source_y_mort"], errors="coerce")
    return {
        "rows": int(mask.sum()),
        "duration_hr_quantiles": quantiles,
        "source_label_distribution": {
            "hospital_death_1": int(labels.eq(1).sum()),
            "hospital_survival_0": int(labels.eq(0).sum()),
            "unknown_or_nonbinary": int((~labels.isin([0, 1])).sum()),
        },
    }


def verify_cohort(variant: str, db: str, entry: dict[str, Any], frame: pd.DataFrame) -> dict[str, Any]:
    required = {
        "db", "stay_id", "subject", "subject_id", "eligible_24h", "temporal_verified",
        "y_mort", "adult_eligible", "input_available_24h", "input_window_le_24h_verified",
        "input_availability_cutoff_hr", "source_endpoint_known", "source_endpoint_unknown",
        "source_y_mort", "source_y_mort_match", "source_time_status_conflict",
        "source_landmark_issue", "source_postlandmark_ordering_issue",
        "icu_exit_hr", "hospital_exit_hr", "hospital_death_time_hr", "icu_death_time_hr",
        "icu_exit_after_hospital_exit_after_24_gap_hr",
    }
    absent = required.difference(frame.columns)
    if absent:
        raise ValueError(f"{variant}/{db}: required columns absent: {sorted(absent)}")
    if frame[["db", "stay_id", "subject", "subject_id"]].isna().any().any():
        raise ValueError(f"{variant}/{db}: null keys")
    if frame.duplicated(["db", "stay_id", "subject"]).any() or not frame["db"].eq(db).all():
        raise ValueError(f"{variant}/{db}: duplicate or wrong database keys")
    if not frame["subject"].astype(str).equals(frame["subject_id"].astype(str)):
        raise ValueError(f"{variant}/{db}: subject alias mismatch")

    selected = bool_col(frame, "eligible_24h")
    endpoint_known = bool_col(frame, "source_endpoint_known")
    endpoint_unknown = bool_col(frame, "source_endpoint_unknown", True)
    if not endpoint_unknown.equals(~endpoint_known):
        raise ValueError(f"{variant}/{db}: per-row source endpoint known/unknown flags disagree")
    source_y = pd.to_numeric(frame["source_y_mort"], errors="coerce")
    y = pd.to_numeric(frame["y_mort"], errors="coerce")
    match = bool_col(frame, "source_y_mort_match")
    if (selected & (~endpoint_known | ~source_y.isin([0, 1]) | ~y.isin([0, 1]) | ~match | source_y.ne(y))).any():
        raise ValueError(f"{variant}/{db}: selected source outcome is unknown or mismatched")
    icu_exit = pd.to_numeric(frame["icu_exit_hr"], errors="coerce")
    hospital_exit = pd.to_numeric(frame["hospital_exit_hr"], errors="coerce")
    hospital_death = pd.to_numeric(frame["hospital_death_time_hr"], errors="coerce")
    icu_death = pd.to_numeric(frame["icu_death_time_hr"], errors="coerce")
    finite_icu = pd.Series(np.isfinite(icu_exit), index=frame.index)
    finite_hospital = pd.Series(np.isfinite(hospital_exit), index=frame.index)
    death_by_24 = (pd.Series(np.isfinite(hospital_death), index=frame.index) & hospital_death.le(24)) | (pd.Series(np.isfinite(icu_death), index=frame.index) & icu_death.le(24))
    pre_index_death = (pd.Series(np.isfinite(hospital_death), index=frame.index) & hospital_death.lt(0)) | (pd.Series(np.isfinite(icu_death), index=frame.index) & icu_death.lt(0))
    expected_gap = finite_icu & finite_hospital & icu_exit.gt(hospital_exit) & hospital_exit.gt(24)
    post_gap = bool_col(frame, "source_postlandmark_ordering_issue")
    if not expected_gap.equals(post_gap):
        raise ValueError(f"{variant}/{db}: post-landmark ordering flag disagrees with source timing")
    if (selected & (~finite_icu | icu_exit.le(24) | ~finite_hospital | hospital_exit.le(24))).any():
        raise ValueError(f"{variant}/{db}: selected row lacks finite ICU and hospital exits after 24h")
    if (selected & (death_by_24 | pre_index_death | bool_col(frame, "source_landmark_issue") | bool_col(frame, "source_time_status_conflict"))).any():
        raise ValueError(f"{variant}/{db}: selected row has a source landmark issue or known early death")
    if (selected & (~bool_col(frame, "adult_eligible") | ~bool_col(frame, "input_available_24h") | ~bool_col(frame, "input_window_le_24h_verified"))).any():
        raise ValueError(f"{variant}/{db}: selected row fails adult/input-window criteria")
    cutoff = pd.to_numeric(frame["input_availability_cutoff_hr"], errors="coerce")
    if (selected & cutoff.gt(24)).any():
        raise ValueError(f"{variant}/{db}: selected row has input cutoff after 24h")
    if (selected & ~bool_col(frame, "temporal_verified")).any():
        raise ValueError(f"{variant}/{db}: selected row is not temporally verified")
    if variant == "strict_v4_1" and (selected & post_gap).any():
        raise ValueError(f"{variant}/{db}: strict selection retains post-landmark exit-order issue")
    if not bool_col(frame, "retrospective_landmark_prediction_allowed").all():
        raise ValueError(f"{variant}/{db}: row-level retrospective prediction attestation missing")
    if frame["clinical_input_availability_verified"].fillna(True).astype(bool).any():
        raise ValueError(f"{variant}/{db}: clinical input availability was overstated")
    if not frame["input_scope"].eq("retrospective_measurement_time").all() or not frame["inputs_temporal_scope"].eq("retrospective_measurement_time").all():
        raise ValueError(f"{variant}/{db}: retrospective measurement scope missing")
    if entry.get("retrospective_landmark_prediction_allowed") is not True or entry.get("ready_for_future24h_clinical_claim") is not False or entry.get("clinical_input_availability_verified") is not False:
        raise ValueError(f"{variant}/{db}: manifest readiness contract is incomplete")
    if entry.get("ready_for_retrospective_24h_measurement_analysis") is not True or entry.get("input_scope") != "retrospective_measurement_time":
        raise ValueError(f"{variant}/{db}: manifest measurement-time scope is incomplete")

    counts = entry.get("counts", {})
    if int(counts.get("n_rows", -1)) != len(frame) or int(counts.get("n_eligible_24h", -1)) != int(selected.sum()):
        raise ValueError(f"{variant}/{db}: manifest row/eligibility count mismatch")
    unknown_rows = int(endpoint_unknown.sum())
    mismatch_rows = int((endpoint_known & (~match | ~source_y.isin([0, 1]) | ~y.isin([0, 1]) | source_y.ne(y))).sum())
    if int(counts.get("n_source_endpoint_unknown_rows", -1)) != unknown_rows or unknown_rows != int(counts.get("n_source_endpoint_unknown_rows", -1)):
        raise ValueError(f"{variant}/{db}: per-row unknown endpoint count mismatch")
    if mismatch_rows != int(counts.get("n_source_label_mismatch_rows", -1)):
        raise ValueError(f"{variant}/{db}: per-row source label mismatch count mismatch")
    if counts.get("source_postlandmark_ordering_gap_eligible") != gap_summary(frame, selected):
        raise ValueError(f"{variant}/{db}: manifest selected gap counts/durations/labels mismatch")
    digest = selected_digest(db, frame, selected)
    if entry.get("selected_keys_labels_sha256") != digest:
        raise ValueError(f"{variant}/{db}: selected key/label digest mismatch")
    table_path = ELIG / variant / f"{db}.parquet"
    if entry.get("eligibility_table_sha256") != sha256(table_path):
        raise ValueError(f"{variant}/{db}: eligibility table hash mismatch")
    labels = y.loc[selected]
    return {
        "rows": int(len(frame)),
        "unique_keys": int(frame[["db", "stay_id", "subject"]].drop_duplicates().shape[0]),
        "eligible_24h": int(selected.sum()),
        "positive": int(labels.eq(1).sum()),
        "negative": int(labels.eq(0).sum()),
        "source_endpoint_unknown_rows": unknown_rows,
        "source_label_mismatch_rows": mismatch_rows,
        "source_landmark_issue_rows": int(bool_col(frame, "source_landmark_issue").sum()),
        "source_landmark_issue_selected": int((selected & bool_col(frame, "source_landmark_issue")).sum()),
        "source_postlandmark_ordering_issue_rows": int(post_gap.sum()),
        "source_postlandmark_ordering_issue_selected": int((selected & post_gap).sum()),
        "selected_keys_labels_sha256": digest,
        "eligibility_table_sha256": sha256(table_path),
        "selected_gap_summary": gap_summary(frame, selected),
        "retrospective_landmark_prediction_allowed": True,
        "input_scope": "retrospective_measurement_time",
        "clinical_input_availability_verified": False,
    }


def main() -> None:
    all_results: dict[str, Any] = {}
    loaded: dict[str, dict[str, pd.DataFrame]] = {}
    for variant in VARIANTS:
        directory = ELIG / variant
        manifest_path = directory / "eligibility_manifest.json"
        print(f"verify_stage={variant}_manifest_read_begin", flush=True)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "4.1" or manifest.get("variant") != variant:
            raise ValueError(f"{variant}: manifest schema or variant identity invalid")
        if manifest.get("source_audit_log_sha256") != sha256(SOURCE_AUDIT):
            raise ValueError(f"{variant}: aggregate source-audit log hash mismatch")
        variant_results = {}
        loaded[variant] = {}
        for db in ("mimic", "eicu"):
            entry = manifest.get("cohorts", {}).get(db)
            if not isinstance(entry, dict) or entry.get("source_join_coverage") != 1.0:
                raise ValueError(f"{variant}/{db}: source-join evidence missing")
            print(f"verify_stage={variant}_{db}_metadata_read_begin", flush=True)
            frame = pd.read_parquet(directory / f"{db}.parquet")
            print(f"verify_stage={variant}_{db}_metadata_read_complete rows={len(frame)}", flush=True)
            loaded[variant][db] = frame
            variant_results[db] = verify_cohort(variant, db, entry, frame)
        all_results[variant] = variant_results

    for db in ("mimic", "eicu"):
        primary = loaded["landmark_v4_1"][db]
        strict = loaded["strict_v4_1"][db]
        primary_keys = set(primary.loc[bool_col(primary, "eligible_24h"), "stay_id"].astype(str))
        strict_keys = set(strict.loc[bool_col(strict, "eligible_24h"), "stay_id"].astype(str))
        if not strict_keys.issubset(primary_keys):
            raise ValueError(f"{db}: strict cohort is not a subset of primary")
        expected_removed = set(primary.loc[
            bool_col(primary, "eligible_24h") & bool_col(primary, "source_postlandmark_ordering_issue"),
            "stay_id",
        ].astype(str))
        if primary_keys - strict_keys != expected_removed:
            raise ValueError(f"{db}: strict-vs-primary difference is not exactly post-landmark ordering gaps")

    for variant, cohorts in all_results.items():
        report = {"schema_version": "4.1", "variant": variant, "status": "passed", "cohorts": cohorts}
        atomic_json(ELIG / variant / "core_verification.json", report)
        for db, value in cohorts.items():
            print(
                f"{variant} {db}: rows={value['rows']} eligible={value['eligible_24h']} "
                f"positive={value['positive']} negative={value['negative']} "
                f"unknown_endpoint={value['source_endpoint_unknown_rows']} "
                f"landmark_issues_selected={value['source_landmark_issue_selected']} "
                f"post24_gaps_selected={value['source_postlandmark_ordering_issue_selected']} "
                f"selected_hash={value['selected_keys_labels_sha256']} "
                f"table_hash={value['eligibility_table_sha256']}", flush=True)
        print(f"{variant}_selected_row_verification=passed", flush=True)


if __name__ == "__main__":
    main()
