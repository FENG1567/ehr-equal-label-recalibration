#!/usr/bin/env python3
"""Server-only INSPIRE operations-table intake and cohort feasibility audit."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE = Path("/data/admin/fengshuo/ehr_fm_20261102/revision_v4_20261122/inspire")
INPUT = BASE / "input" / "clean" / "operations.csv.gz"
ORIGINAL_INPUT = BASE / "input" / "failed" / "operations_local_unverified.csv.gz"
EXPECTED_OPS_SHA256 = "0c366d4945f258ecf317261f54a36b116bb170c583b2c6c59c558711b15cebf7"
OFFICIAL_PREFIX_BYTES = 5629428
OFFICIAL_PREFIX_ROWS = 130960
PROTOCOL = Path("/data/admin/fengshuo/ehr_fm_20261102/revision_v4_20261122/admin/execution_protocol_v4.md")
EXPECTED_PROTOCOL_SHA256 = "605f9440026bc0074c3ede3ee450adfd94d394219b0f8fcd5e92014776f9d9ff"
AUDIT = BASE / "audit"
NORMALIZED = BASE / "normalized" / "feasibility"
COHORT_OUTPUT = NORMALIZED / "cohort_first_icuin.csv"

ID_COLUMNS = ["op_id", "subject_id", "hadm_id"]
NUMERIC_COLUMNS = [
    "age", "weight", "height", "opstart_time", "opend_time", "admission_time", "discharge_time",
    "icuin_time", "icuout_time", "inhosp_death_time", "allcause_death_time",
]
REQUIRED_COLUMNS = set(ID_COLUMNS + ["sex"] + NUMERIC_COLUMNS)


def qstats(series: pd.Series) -> dict[str, object]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"n": 0}
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "p05": float(values.quantile(0.05)),
        "median": float(values.median()),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def count(series: pd.Series) -> int:
    return int(series.fillna(False).astype(bool).sum())


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    NORMALIZED.mkdir(parents=True, exist_ok=True)
    header = list(pd.read_csv(INPUT, compression="gzip", nrows=0).columns)
    missing_columns = sorted(REQUIRED_COLUMNS - set(header))
    integrity = {
        "source_file": str(INPUT),
        "original_received_copy": str(ORIGINAL_INPUT),
        "source_version_claim": "INSPIRE 1.4.2 per adjacent source metadata and the accompanying source note",
        "official_prefix_bytes": OFFICIAL_PREFIX_BYTES,
        "official_prefix_rows_per_parent_note": OFFICIAL_PREFIX_ROWS,
        "official_prefix_sha256_per_parent_note": EXPECTED_OPS_SHA256,
        "source_disposition": "official gzip prefix plus known trailing bytes; clean working copy is cut to the parent-specified prefix; original received copy remains unchanged",
        "header_fields": header,
        "missing_required_fields": missing_columns,
        "provenance_state": "official_prefix_accepted_from_parent_note",
    }
    (AUDIT / "operations_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    if missing_columns:
        raise RuntimeError(f"Required operations fields missing: {missing_columns}")

    usecols = [column for column in header if column in REQUIRED_COLUMNS]
    dtypes = {column: "string" for column in ID_COLUMNS}
    dtypes["sex"] = "string"
    operations = pd.read_csv(INPUT, compression="gzip", usecols=usecols, dtype=dtypes)
    for column in NUMERIC_COLUMNS:
        operations[column] = pd.to_numeric(operations[column], errors="coerce")

    adult = operations["age"].between(18, 90, inclusive="both")
    has_icu = operations["icuin_time"].notna()
    eligible = adult & has_icu & operations["subject_id"].notna() & operations["hadm_id"].notna()
    candidates = operations.loc[eligible].copy()
    candidates = candidates.sort_values(["icuin_time", "hadm_id", "op_id"], kind="mergesort")
    admission_first = candidates.drop_duplicates("hadm_id", keep="first").copy()
    subject_first = admission_first.sort_values(["icuin_time", "hadm_id", "op_id"], kind="mergesort")
    subject_first = subject_first.drop_duplicates("subject_id", keep="first").copy()

    horizon = subject_first["icuin_time"] + 1440.0
    death = subject_first["inhosp_death_time"]
    icuout = subject_first["icuout_time"]
    discharge = subject_first["discharge_time"]
    subject_first["landmark24_time"] = horizon
    subject_first["icu_exit_at_or_after_24h"] = icuout.notna() & (icuout >= horizon)
    subject_first["hospital_death_by_24h"] = death.notna() & (death <= horizon)
    subject_first["hospital_death_at_24h"] = death.notna() & (death == horizon)
    subject_first["hospital_death_after_24h"] = death.notna() & (death > horizon)
    subject_first["death_time_unknown"] = death.isna()
    subject_first["discharge_before_24h"] = discharge.notna() & (discharge < horizon)
    subject_first["discharge_at_24h"] = discharge.notna() & (discharge == horizon)
    subject_first["discharge_after_24h"] = discharge.notna() & (discharge > horizon)
    subject_first["discharge_at_or_after_24h"] = discharge.notna() & (discharge >= horizon)
    subject_first["confirmed_alive_in_icu_at_24h"] = subject_first["icu_exit_at_or_after_24h"] & subject_first["hospital_death_after_24h"]
    subject_first["icu_at_24h_death_time_unknown"] = subject_first["icu_exit_at_or_after_24h"] & subject_first["death_time_unknown"]
    subject_first["landmark_status"] = "unknown"
    subject_first.loc[subject_first["hospital_death_by_24h"], "landmark_status"] = "excluded_death_at_or_before_24h"
    subject_first.loc[subject_first["icu_exit_at_or_after_24h"] & subject_first["hospital_death_after_24h"], "landmark_status"] = "confirmed_alive_in_icu_at_24h"
    subject_first.loc[subject_first["icu_exit_at_or_after_24h"] & subject_first["death_time_unknown"], "landmark_status"] = "icu_exit_supports_24h_but_death_status_unknown"
    subject_first.loc[subject_first["discharge_before_24h"] & ~subject_first["hospital_death_by_24h"], "landmark_status"] = "excluded_hospital_discharge_before_24h"
    subject_first.loc[icuout.notna() & (icuout < horizon) & ~subject_first["hospital_death_by_24h"], "landmark_status"] = "excluded_icu_exit_before_24h"
    subject_first["patient_group"] = subject_first["subject_id"]
    subject_first["index_variable"] = "icuin_time"
    subject_first["provenance_state"] = integrity["provenance_state"]
    subject_first["measurement_time_used_as_availability_proxy"] = True
    valid_height = pd.to_numeric(subject_first["height"], errors="coerce").where(lambda x: x > 0)
    valid_weight = pd.to_numeric(subject_first["weight"], errors="coerce").where(lambda x: x > 0)
    subject_first["bmi"] = valid_weight / (valid_height / 100.0) ** 2

    delta = operations["icuin_time"] - operations["opend_time"]
    finite_delta = delta.dropna()
    postop_audit = {
        "paired_icuin_and_opend_rows": int(finite_delta.size),
        "icuin_before_opend": int((finite_delta < 0).sum()),
        "icuin_from_opend_0_to_1440_min_inclusive": int(finite_delta.between(0, 1440, inclusive="both").sum()),
        "icuin_more_than_1440_min_after_opend": int((finite_delta > 1440).sum()),
        "icuin_minus_opend_minutes": qstats(finite_delta),
        "interpretation": "Audit only; primary postoperative eligibility uses the official v1.4.2 icuin_time definition (ICU admission within 24 h of surgery).",
    }

    overlap_pairs = 0
    overlapping_subjects: set[str] = set()
    overlap_indeterminate_subjects: set[str] = set()
    for subject_id, admissions in admission_first.groupby("subject_id", dropna=True, sort=False):
        ordered = admissions.sort_values("icuin_time", kind="mergesort")
        prior_ends: list[float] = []
        for row in ordered.itertuples(index=False):
            current_start = getattr(row, "icuin_time")
            if pd.notna(current_start):
                if any(pd.isna(end) for end in prior_ends):
                    overlap_indeterminate_subjects.add(str(subject_id))
                overlaps = sum(pd.notna(end) and float(current_start) < float(end) for end in prior_ends)
                if overlaps:
                    overlap_pairs += int(overlaps)
                    overlapping_subjects.add(str(subject_id))
            prior_ends.append(getattr(row, "icuout_time"))

    risk = subject_first["confirmed_alive_in_icu_at_24h"]
    counts = {
        "operations_rows": int(len(operations)),
        "unique_op_ids": int(operations["op_id"].nunique(dropna=True)),
        "unique_hadm_ids": int(operations["hadm_id"].nunique(dropna=True)),
        "unique_subject_ids": int(operations["subject_id"].nunique(dropna=True)),
        "all_icuin_rows_all_ages": count(has_icu),
        "adult_18_90_rows": count(adult),
        "adult_icuin_rows": count(eligible),
        "adult_icuin_unique_hadm": int(admission_first["hadm_id"].nunique(dropna=True)),
        "duplicate_adult_icuin_rows_removed_within_hadm": int(len(candidates) - len(admission_first)),
        "adult_first_icuin_per_subject": int(len(subject_first)),
        "subjects_with_multiple_qualifying_hadm": int((admission_first.groupby("subject_id").size() > 1).sum()),
        "icu_exit_at_or_after_24h": count(subject_first["icu_exit_at_or_after_24h"]),
        "icu_exit_before_24h": int((icuout.notna() & (icuout < horizon)).sum()),
        "icu_exit_exactly_24h": int((icuout.notna() & (icuout == horizon)).sum()),
        "icu_exit_time_unknown": int(icuout.isna().sum()),
        "death_time_after_24h": count(subject_first["hospital_death_after_24h"]),
        "death_time_at_or_before_24h": count(subject_first["hospital_death_by_24h"]),
        "death_time_exactly_24h": count(subject_first["hospital_death_at_24h"]),
        "death_time_unknown_not_counted_as_survival": count(subject_first["death_time_unknown"]),
        "icu_at_24h_but_death_time_unknown": count(subject_first["icu_at_24h_death_time_unknown"]),
        "confirmed_alive_in_icu_at_24h_strict_lower_bound": count(risk),
        "known_post24h_hospital_deaths_in_strict_lower_bound": count(subject_first["hospital_death_after_24h"] & risk),
        "discharge_before_24h": count(subject_first["discharge_before_24h"]),
        "discharge_exactly_24h": count(subject_first["discharge_at_24h"]),
        "discharge_after_24h": count(subject_first["discharge_after_24h"]),
        "discharge_time_unknown": int(discharge.isna().sum()),
        "discharge_at_or_after_24h": count(subject_first["discharge_at_or_after_24h"]),
        "discharge_before24_but_icu_exit_at_or_after24_contradictions": count(subject_first["discharge_before_24h"] & subject_first["icu_exit_at_or_after_24h"]),
        "overlap_audit_all_qualifying_hadm": {
            "subject_pairs_with_overlapping_icu_intervals": overlap_pairs,
            "subjects_with_any_overlapping_icu_intervals": len(overlapping_subjects),
            "subjects_with_overlap_indeterminate_due_to_missing_icuout": len(overlap_indeterminate_subjects),
        },
    }
    static_coverage = {}
    for variable in ["age", "sex", "weight", "height", "bmi"]:
        observed = subject_first[variable].notna()
        static_coverage[variable] = {
            "selected_first_per_subject_with_value": count(observed),
            "selected_first_per_subject_total": int(len(subject_first)),
            "strict_confirmed_alive_in_icu_24h_with_value": count(observed & risk),
            "strict_confirmed_alive_in_icu_24h_total": count(risk),
        }

    subject_first.to_csv(COHORT_OUTPUT, index=False)
    summary = {
        "dataset": "INSPIRE",
        "version_claim": "1.4.2",
        "protocol_path": str(PROTOCOL),
        "protocol_sha256_expected": EXPECTED_PROTOCOL_SHA256,
        "source_integrity": integrity,
        "selection": {
            "adult_age_inclusive": [18, 90],
            "postoperative_icu_flag": "nonmissing operations.icuin_time; official 1.4.2 dictionary states ICU admissions in this field are within 24 h after surgery",
            "index_time": "operations.icuin_time",
            "index_window_minutes": [0, 1440],
            "repeat_operation_resolution": "earliest icuin_time per hadm_id, then earliest qualifying admission per subject_id; deterministic ties hadm_id/op_id; outcome-independent",
            "postoperative_definition": "nonmissing operations.icuin_time; the official v1.4.2 dictionary says this field includes only ICU admissions within 24 h of surgery; orstart_time is not the index",
            "patient_group": "subject_id",
            "measurement_availability": "measurement time used because acquisition/result availability time is not supplied",
        },
        "time_origin_audit": postop_audit,
        "counts": counts,
        "static_variable_coverage": static_coverage,
        "endpoint_caution": "inhosp_death_time missing is unknown, not a survivor label unless INSPIRE null semantics and complete admission/discharge follow-up are documented. ICU-specific death is not supplied and is not inferred.",
        "cohort_output": str(COHORT_OUTPUT),
        "status": "feasibility_only; official prefixes are separate working copies; no models or performance metrics run",
    }
    (AUDIT / "operations_feasibility.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "counts": counts, "audit": str(AUDIT / "operations_feasibility.json")}))


if __name__ == "__main__":
    main()
