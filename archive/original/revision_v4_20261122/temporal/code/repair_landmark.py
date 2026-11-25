#!/usr/bin/env python3
"""Build source-grounded, strict-key 24h landmark metadata for the EHR cohorts.

This script only reads preserved cohort/sequence artifacts and selected
source-level metadata tables. It never opens feature-event tables or fits a
model. Patient-level output stays on the server; stdout contains aggregates.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT = Path("/data/admin/fengshuo/ehr_fm_20261102")
OUT_ROOT = PROJECT / "revision_v4_20261122" / "temporal"
ELIG_DIR = OUT_ROOT / "results" / "eligibility"
INTERIM = PROJECT / "data" / "interim"
PROCESSED = PROJECT / "data" / "processed"
MIMIC = Path("/data/admin/fengshuo/if10_20261002/shared/raw/mimic")
EICU = Path("/data/admin/fengshuo/if10_20261002/shared/raw/eicu")
ZIGONG = PROJECT / "data" / "raw" / "zigong" / "DataTables"
CHALLENGE = PROJECT / "data" / "external" / "challenge2012"

DATABASES = ("mimic", "eicu", "zigong", "c2012", "hirid", "nwicu", "sicdb")
ENDPOINT = {
    "c2012": "hospital",
    "eicu": "hospital",
    "hirid": "icu",
    "mimic": "hospital",
    "nwicu": "hospital",
    "sicdb": "icu",
    "zigong": "icu",
}
SUBJECT_IDENTITY = {
    "c2012": {"basis": "RecordID copied to subject_id; no verified longitudinal person identifier", "patient_disjoint_claim_allowed": False},
    "eicu": {"basis": "source uniquepid", "patient_disjoint_claim_allowed": True},
    "hirid": {"basis": "patientid/stay identifier copied to subject_id; no verified longitudinal person linkage", "patient_disjoint_claim_allowed": False},
    "mimic": {"basis": "source subject_id", "patient_disjoint_claim_allowed": True},
    "nwicu": {"basis": "source subject_id", "patient_disjoint_claim_allowed": True},
    "sicdb": {"basis": "source PatientID", "patient_disjoint_claim_allowed": True},
    "zigong": {"basis": "source PATIENT_ID", "patient_disjoint_claim_allowed": True},
}
NEEDED = {
    "hirid": [
        "HiRID 1.1.1 reference/general_table.csv with patientid, admissiontime, discharge_status",
        "A source ICU discharge timestamp/duration for each patientid (general_table has no discharge time)",
    ],
    "nwicu": [
        "nw_icu/icustays.csv.gz (stay_id, subject_id, intime, outtime)",
        "nw_hosp/patients.csv.gz (subject_id, anchor_age, gender)",
        "nw_hosp/admissions.csv.gz (hadm_id, dischtime, deathtime, hospital_expire_flag, discharge_location)",
    ],
    "sicdb": [
        "SICdb cases.csv.gz (CaseID, PatientID, AgeOnAdmission, DischargeState)",
        "A documented ICU discharge timestamp/offset keyed by CaseID; event-table max offset is not a discharge time",
    ],
    "c2012": [
        "Challenge 2012 has outcomes-a.txt (RecordID, In-hospital_death), but no source ICU discharge time/death timestamp",
        "A source ICU discharge timestamp/duration keyed by RecordID to verify alive and in ICU at 24h",
    ],
}


def canon_id(value: Any) -> str | None:
    """Normalize identifier scalar types without losing leading zeroes in strings."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if not math.isfinite(float(value)):
            return None
        if float(value).is_integer():
            return str(int(value))
    return str(value).strip()


def normalize_keys(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for col in ("stay_id", "subject_id"):
        if col not in frame:
            raise ValueError(f"required key missing: {col}")
        frame[col] = frame[col].map(canon_id)
    if frame[["stay_id", "subject_id"]].isna().any().any():
        raise ValueError("null key count is nonzero")
    return frame


def assert_unique(frame: pd.DataFrame, label: str) -> None:
    dup = frame.duplicated(["stay_id", "subject_id"], keep=False)
    if bool(dup.any()):
        raise ValueError(f"{label} has {int(dup.sum())} rows with duplicate keys")


def strict_join(left: pd.DataFrame, right: pd.DataFrame, label: str) -> tuple[pd.DataFrame, float]:
    """Require one-to-one source coverage for every sequence key; never join by row order."""
    left = normalize_keys(left)
    right = normalize_keys(right)
    assert_unique(left, "sequence/cohort")
    assert_unique(right, label)
    joined = left.merge(right, on=["stay_id", "subject_id"], how="left", validate="one_to_one", indicator=True)
    matched = joined["_merge"].eq("both")
    coverage = float(matched.mean()) if len(joined) else 1.0
    if not bool(matched.all()):
        raise ValueError(f"{label} key coverage failed: matched={int(matched.sum())}, expected={len(joined)}")
    return joined.drop(columns="_merge"), coverage


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_hashes(paths: list[Path]) -> dict[str, str]:
    return {p.name: sha256(p) for p in paths if p.is_file()}


def nullable_bool(value: Any) -> Any:
    if value is None:
        return pd.NA
    try:
        if pd.isna(value):
            return pd.NA
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"dead", "expired", "death", "deceased", "1", "true", "yes"} or "expired" in low or "deceased" in low or low.endswith(" dead"):
            return True
        if low in {"alive", "survived", "0", "false", "no"} or "alive" in low or "surviv" in low:
            return False
        return pd.NA
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value) if float(value) in (0.0, 1.0) else pd.NA
    return pd.NA


def to_timestamp(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce")


def source_timing_checks(icu_exit_hr: pd.Series, hospital_exit_hr: pd.Series, exact_death_hr: pd.Series) -> dict[str, pd.Series]:
    """Flag impossible source ordering separately from valid-but-early exits."""
    icu_exit = pd.to_numeric(icu_exit_hr, errors="coerce")
    hospital_exit = pd.to_numeric(hospital_exit_hr, errors="coerce")
    death_hr = pd.to_numeric(exact_death_hr, errors="coerce")
    finite_icu_exit = pd.Series(np.isfinite(icu_exit), index=icu_exit.index)
    finite_hospital_exit = pd.Series(np.isfinite(hospital_exit), index=hospital_exit.index)
    finite_death = pd.Series(np.isfinite(death_hr), index=death_hr.index)
    landmark_contradiction = finite_death & death_hr.le(24) & finite_icu_exit & icu_exit.gt(24)
    hospital_before_icu = finite_hospital_exit & finite_icu_exit & hospital_exit.lt(icu_exit)
    death_after_hospital = finite_death & finite_hospital_exit & death_hr.gt(hospital_exit)
    nonfinite_reported_time = (icu_exit.notna() & ~finite_icu_exit) | (hospital_exit.notna() & ~finite_hospital_exit) | (death_hr.notna() & ~finite_death)
    source_time_issue = (
        (finite_icu_exit & icu_exit.lt(0))
        | (finite_hospital_exit & hospital_exit.lt(0))
        | hospital_before_icu
        | death_after_hospital
        | landmark_contradiction
        | nonfinite_reported_time
    )
    return {
        "source_time_issue": source_time_issue.fillna(False).astype(bool),
        "landmark_contradiction": landmark_contradiction.fillna(False).astype(bool),
        "hospital_exit_before_icu_exit": hospital_before_icu.fillna(False).astype(bool),
        "icu_exit_after_hospital_exit_after_24": (hospital_before_icu & hospital_exit.gt(24)).fillna(False).astype(bool),
        "death_after_hospital_exit": death_after_hospital.fillna(False).astype(bool),
        "nonfinite_reported_time": nonfinite_reported_time.fillna(False).astype(bool),
    }


def eicu_offset_hours_since_unit_admit(event_offset_minutes: pd.Series) -> pd.Series:
    """Convert eICU offsets, which use ICU-unit admission as zero, to hours."""
    return pd.to_numeric(event_offset_minutes, errors="coerce") / 60.0


def read_csv_columns(path: Path, requested: list[str]) -> pd.DataFrame:
    header = pd.read_csv(path, compression="infer", nrows=0).columns.tolist()
    columns = [c for c in requested if c in header]
    absent = [c for c in requested if c not in header]
    if not columns:
        raise ValueError(f"no expected source columns present in {path.name}")
    frame = pd.read_csv(path, usecols=columns, compression="infer", low_memory=False)
    frame.attrs["absent_columns"] = absent
    return frame


def sequence_base(db: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    cohort_path = INTERIM / f"cohort_{db}.parquet"
    sequence_path = PROCESSED / f"seq_{db}.npz"
    cohort = pd.read_parquet(cohort_path)
    if "subject_id" not in cohort or "stay_id" not in cohort:
        raise ValueError(f"{db} cohort lacks stay_id/subject_id")
    cohort = normalize_keys(cohort)
    assert_unique(cohort, f"cohort_{db}")
    if "mort_hosp" not in cohort:
        raise ValueError(f"{db} cohort lacks current y_mort source label")
    with np.load(sequence_path, allow_pickle=False) as seq:
        required = {"stay_id", "subject", "L", "H", "y_mort"}
        absent = required.difference(seq.files)
        if absent:
            raise ValueError(f"{db} sequence keys/labels missing {sorted(absent)}")
        seqdf = pd.DataFrame({
            "stay_id": seq["stay_id"],
            "subject_id": seq["subject"],
            "seq_y_mort": seq["y_mort"],
            "input_event_count_24h": seq["L"],
        })
        lengths = np.asarray(seq["L"], dtype=np.int64)
        hours = np.asarray(seq["H"])
        max_hours: list[float] = []
        window_valid: list[bool] = []
        for i, length in enumerate(lengths):
            used = hours[i, :max(0, int(length))]
            max_hours.append(float(np.max(used)) if len(used) else np.nan)
            window_valid.append(bool(np.all((used >= 0) & (used < 24))))
        seqdf["input_max_hour_24h"] = max_hours
        seqdf["input_window_le_24h_verified"] = window_valid
    seqdf = normalize_keys(seqdf)
    assert_unique(seqdf, f"seq_{db}")
    joined, _ = strict_join(cohort, seqdf, f"seq_{db}")
    y_cohort = pd.to_numeric(joined["mort_hosp"], errors="coerce")
    y_seq = pd.to_numeric(joined["seq_y_mort"], errors="coerce")
    mismatch = int((y_cohort.fillna(-999) != y_seq.fillna(-999)).sum())
    if mismatch:
        raise ValueError(f"{db} cohort/sequence y_mort mismatch count={mismatch}")
    joined["db"] = db
    joined["y_mort"] = y_cohort.astype(float)
    joined["endpoint"] = ENDPOINT[db]
    joined["input_window_start_hr"] = 0.0
    joined["input_window_end_hr"] = 24.0
    joined["input_available_24h"] = pd.to_numeric(joined["input_event_count_24h"], errors="coerce").fillna(0).gt(0)
    joined["adult_eligible"] = pd.to_numeric(joined.get("age"), errors="coerce").ge(18) if "age" in joined else pd.NA
    joined["adult_eligibility_source"] = "existing_cohort_age"
    audit = {
        "n_sequence": int(len(seqdf)),
        "n_cohort": int(len(cohort)),
        "sequence_cohort_join_coverage": 1.0,
        "cohort_sequence_y_mort_mismatches": mismatch,
        "sequence_sha256": sha256(sequence_path),
        "cohort_sha256": sha256(cohort_path),
    }
    return joined, audit


def source_label_audit(current: pd.Series, source: pd.Series, known: pd.Series) -> tuple[pd.Series, int]:
    """Compare labels only where the source endpoint is an explicit binary value."""
    current = pd.to_numeric(current, errors="coerce")
    source = pd.to_numeric(source, errors="coerce")
    known = known.fillna(False).astype(bool)
    match = pd.Series(pd.NA, index=current.index, dtype="boolean")
    valid_current = current.isin([0, 1])
    match.loc[known] = (valid_current.loc[known] & current.loc[known].eq(source.loc[known])).astype("boolean")
    mismatch = known & ~match.fillna(False).astype(bool)
    return match, int(mismatch.sum())


def mimic_metadata(base: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = [MIMIC / "icu" / "icustays.csv.gz", MIMIC / "hosp" / "patients.csv.gz", MIMIC / "hosp" / "admissions.csv.gz"]
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError("MIMIC timing tables unavailable")
    icu = read_csv_columns(paths[0], ["subject_id", "hadm_id", "stay_id", "intime", "outtime"])
    pat = read_csv_columns(paths[1], ["subject_id", "gender", "anchor_age", "anchor_year"])
    adm = read_csv_columns(paths[2], ["hadm_id", "hospital_expire_flag", "admittime", "dischtime", "deathtime", "hospital_id"])
    if not {"subject_id", "hadm_id", "stay_id", "intime", "outtime"}.issubset(icu.columns):
        raise ValueError("MIMIC icustays required columns unavailable")
    if not {"subject_id", "anchor_age", "gender"}.issubset(pat.columns):
        raise ValueError("MIMIC patients adult/sex columns unavailable")
    if not {"hadm_id", "dischtime"}.issubset(adm.columns):
        raise ValueError("MIMIC admissions discharge time unavailable")
    raw = icu.merge(pat, on="subject_id", how="left", validate="many_to_one")
    raw = raw.merge(adm, on="hadm_id", how="left", validate="many_to_one")
    raw = normalize_keys(raw.rename(columns={"anchor_age": "source_age", "gender": "source_gender"}))
    assert_unique(raw, "MIMIC source joins")
    joined, coverage = strict_join(base, raw, "MIMIC source metadata")
    start = to_timestamp(joined["intime"])
    exit_ = to_timestamp(joined["outtime"])
    hdeath = to_timestamp(joined["deathtime"]) if "deathtime" in joined else pd.Series(pd.NaT, index=joined.index)
    hexit = to_timestamp(joined["dischtime"])
    joined["icu_start_time"] = start.astype("string")
    joined["icu_exit_time"] = exit_.astype("string")
    joined["icu_exit_hr"] = (exit_ - start).dt.total_seconds() / 3600.0
    joined["hospital_exit_time"] = hexit.astype("string")
    joined["hospital_exit_hr"] = (hexit - start).dt.total_seconds() / 3600.0
    joined["hospital_death_time"] = hdeath.astype("string")
    joined["hospital_death_time_hr"] = (hdeath - start).dt.total_seconds() / 3600.0
    flag = pd.to_numeric(joined["hospital_expire_flag"], errors="coerce") if "hospital_expire_flag" in joined else pd.Series(np.nan, index=joined.index)
    endpoint_known = flag.isin([0, 1])
    source_y = flag.where(endpoint_known)
    joined["source_endpoint_known"] = endpoint_known.astype(bool)
    joined["source_endpoint_unknown"] = (~endpoint_known).astype(bool)
    joined["hospital_death"] = pd.array(source_y.eq(1).where(endpoint_known), dtype="boolean")
    joined["hospital_death_time_kind"] = np.where(hdeath.notna(), "exact_deathtime", "unavailable")
    in_icu_death = hdeath.notna() & start.notna() & exit_.notna() & hdeath.ge(start) & hdeath.le(exit_)
    joined["icu_death"] = pd.array(np.where(hdeath.notna(), in_icu_death, np.where(flag.eq(0), False, pd.NA)), dtype="boolean")
    joined["icu_death_time"] = np.where(in_icu_death, hdeath.astype("string"), pd.NA)
    joined["icu_death_time_hr"] = np.where(in_icu_death, joined["hospital_death_time_hr"], np.nan)
    joined["icu_death_time_kind"] = np.where(in_icu_death, "exact_deathtime", "unavailable")
    joined["source_time_basis"] = "timestamp_from_icu_intime"
    joined["source_timing_precision"] = "exact"
    joined["icu_exit_time_kind"] = "exact_outtime"
    anchor_age = pd.to_numeric(joined["source_age"], errors="coerce")
    if "anchor_year" in joined:
        anchor_year = pd.to_numeric(joined["anchor_year"], errors="coerce")
        age_at_icu = anchor_age + start.dt.year - anchor_year
        age_at_icu = age_at_icu.where(anchor_year.notna(), anchor_age)
        joined["age"] = age_at_icu.clip(upper=91)
        joined["age_precision"] = "approximate_to_year"
        joined["adult_eligibility_source"] = "patients.anchor_age + (intime.year - patients.anchor_year); year precision"
    else:
        joined["age"] = anchor_age.clip(upper=91)
        joined["age_precision"] = "anchor_age_proxy"
        joined["adult_eligibility_source"] = "patients.anchor_age; anchor_year unavailable"
    joined["adult_eligible"] = pd.to_numeric(joined["age"], errors="coerce").ge(18)
    joined["sex"] = joined["source_gender"].map({"M": 1, "F": 0})
    if "hospital_id" not in joined:
        joined["hospitalid"] = pd.NA
        joined["hospitalid_source"] = "no_per_admission_field_single_site_source"
    else:
        joined["hospitalid"] = joined["hospital_id"].map(canon_id)
        joined["hospitalid_source"] = "admissions.hospital_id"
    icu_exit = pd.to_numeric(joined["icu_exit_hr"], errors="coerce")
    hospital_exit = pd.to_numeric(joined["hospital_exit_hr"], errors="coerce")
    death_hr = pd.to_numeric(joined["hospital_death_time_hr"], errors="coerce")
    timing = source_timing_checks(icu_exit, hospital_exit, death_hr)
    landmark_contradiction = timing["landmark_contradiction"]
    hospital_before_icu = timing["hospital_exit_before_icu_exit"]
    invalid = timing["source_time_issue"]
    flag_time_conflict = flag.eq(0) & hdeath.notna()
    invalid |= flag_time_conflict
    joined["source_time_issue"] = invalid
    joined["source_time_status_conflict"] = flag_time_conflict
    joined["source_y_mort"] = source_y
    current_y = pd.to_numeric(joined["y_mort"], errors="coerce")
    joined["source_y_mort_match"], mismatch = source_label_audit(current_y, source_y, endpoint_known)
    joined["source_landmark_contradiction"] = landmark_contradiction.astype(bool)
    joined["icu_exit_after_hospital_exit_after_24"] = (hospital_before_icu & hospital_exit.gt(24)).astype(bool)
    return joined, {"source_join_coverage": coverage, "source_hashes": file_hashes(paths), "source_files": [p.name for p in paths], "icu_exit_source": "icustays.intime/outtime", "hospital_death_source": "admissions.hospital_expire_flag; exact deathtime retained for timing validation", "hospital_id_coverage": int(joined["hospitalid"].notna().sum()), "source_label_mismatch_count": mismatch, "source_endpoint_unknown_count": int((~endpoint_known).sum()), "source_time_status_conflicts": int(flag_time_conflict.sum()), "source_landmark_contradictions": int(landmark_contradiction.sum()), "icu_exit_after_hospital_exit_after_24": int((hospital_before_icu & hospital_exit.gt(24)).sum())}


def eicu_metadata(base: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = EICU / "patient.csv.gz"
    if not path.is_file():
        raise FileNotFoundError("EICU patient.csv.gz unavailable")
    raw = read_csv_columns(path, [
        "patientunitstayid", "uniquepid", "age", "gender", "hospitalid",
        "unitdischargeoffset", "unitdischargestatus",
        "hospitaladmitoffset", "hospitaldischargeoffset", "hospitaldischargestatus",
    ])
    required = {"patientunitstayid", "uniquepid", "unitdischargeoffset", "hospitaldischargeoffset", "hospitaldischargestatus"}
    if not required.issubset(raw.columns):
        raise ValueError("EICU patient metadata lacks required discharge columns")
    raw = raw.rename(columns={"patientunitstayid": "stay_id", "uniquepid": "subject_id", "age": "source_age"})
    raw = normalize_keys(raw)
    assert_unique(raw, "EICU patient source")
    joined, coverage = strict_join(base, raw, "EICU patient source")
    joined["icu_start_time"] = pd.NA
    joined["icu_exit_time"] = pd.NA
    joined["icu_exit_hr"] = eicu_offset_hours_since_unit_admit(joined["unitdischargeoffset"])
    joined["hospital_exit_time"] = pd.NA
    joined["hospital_exit_hr"] = eicu_offset_hours_since_unit_admit(joined["hospitaldischargeoffset"])
    unit_status = joined.get("unitdischargestatus", pd.Series(pd.NA, index=joined.index)).map(nullable_bool)
    hosp_status = joined["hospitaldischargestatus"].map(nullable_bool)
    endpoint_known = hosp_status.notna()
    joined["source_endpoint_known"] = endpoint_known.astype(bool)
    joined["source_endpoint_unknown"] = (~endpoint_known).astype(bool)
    joined["hospital_death"] = pd.array(hosp_status, dtype="boolean")
    joined["icu_death"] = pd.array(unit_status, dtype="boolean")
    # eICU has discharge offsets/statuses, not a separate death timestamp.
    joined["hospital_death_time"] = pd.NA
    joined["hospital_death_time_hr"] = np.where(hosp_status.fillna(False), joined["hospital_exit_hr"], np.nan)
    joined["hospital_death_time_kind"] = np.where(hosp_status.fillna(False), "hospital_discharge_offset_proxy", "unavailable")
    joined["icu_death_time"] = pd.NA
    joined["icu_death_time_hr"] = np.where(unit_status.fillna(False), joined["icu_exit_hr"], np.nan)
    joined["icu_death_time_kind"] = np.where(unit_status.fillna(False), "icu_discharge_offset_proxy", "unavailable")
    joined["source_time_basis"] = "minutes_since_icu_unit_admission"
    joined["source_timing_precision"] = "exact_offset"
    joined["icu_exit_time_kind"] = "exact_unitdischargeoffset"
    age = joined["source_age"].astype("string").str.replace("> 89", "90", regex=False)
    joined["age"] = pd.to_numeric(age, errors="coerce")
    joined["adult_eligible"] = joined["age"].ge(18)
    joined["age_precision"] = "source_reported_year_or_topcoded"
    joined["adult_eligibility_source"] = "patient.age"
    if "hospitalid" in joined:
        joined["hospitalid"] = joined["hospitalid"].map(canon_id)
        joined["hospitalid_source"] = "patient.hospitalid"
    else:
        joined["hospitalid"] = pd.NA
        joined["hospitalid_source"] = "column_unavailable"
    if "gender" in joined:
        joined["sex"] = joined["gender"].map({"Male": 1, "Female": 0, "M": 1, "F": 0})
    icu_exit = pd.to_numeric(joined["icu_exit_hr"], errors="coerce")
    hospital_exit = pd.to_numeric(joined["hospital_exit_hr"], errors="coerce")
    death_hr = pd.to_numeric(joined["hospital_death_time_hr"], errors="coerce")
    timing = source_timing_checks(icu_exit, hospital_exit, death_hr)
    landmark_contradiction = timing["landmark_contradiction"]
    hospital_before_icu = timing["hospital_exit_before_icu_exit"]
    invalid = timing["source_time_issue"]
    time_status_conflict = landmark_contradiction
    joined["source_time_issue"] = invalid
    joined["source_time_status_conflict"] = time_status_conflict
    joined["source_y_mort"] = pd.array(hosp_status, dtype="Float64")
    current_y = pd.to_numeric(joined["y_mort"], errors="coerce")
    source_y = pd.to_numeric(joined["source_y_mort"], errors="coerce")
    joined["source_y_mort_match"], mismatch = source_label_audit(current_y, source_y, endpoint_known)
    joined["source_landmark_contradiction"] = landmark_contradiction.astype(bool)
    joined["icu_exit_after_hospital_exit_after_24"] = (hospital_before_icu & hospital_exit.gt(24)).astype(bool)
    return joined, {"source_join_coverage": coverage, "source_hashes": file_hashes([path]), "source_files": [path.name], "icu_exit_source": "patient.unitdischargeoffset since ICU unit admission", "hospital_death_source": "patient.hospitaldischargestatus; hospitaldischargeoffset since ICU unit admission is a discharge proxy", "hospital_id_coverage": int(joined["hospitalid"].notna().sum()), "source_label_mismatch_count": mismatch, "source_endpoint_unknown_count": int((~endpoint_known).sum()), "source_time_status_conflicts": int(time_status_conflict.sum()), "source_landmark_contradictions": int(landmark_contradiction.sum()), "icu_exit_after_hospital_exit_after_24": int((hospital_before_icu & hospital_exit.gt(24)).sum())}


def zigong_metadata(base: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    baseline_path = ZIGONG / "dtBaseline.csv"
    outcome_path = ZIGONG / "dtOutCome.csv"
    if not baseline_path.is_file() or not outcome_path.is_file():
        raise FileNotFoundError("Zigong baseline/outcome tables unavailable")
    b = read_csv_columns(baseline_path, ["PATIENT_ID", "INP_NO", "Age", "SEX", "ICU_discharge_time", "DISCHARGE_DATE_TIME"])
    o = read_csv_columns(outcome_path, ["PATIENT_ID", "Death_Date"])
    if not {"PATIENT_ID", "INP_NO", "ICU_discharge_time"}.issubset(b.columns) or not {"PATIENT_ID", "Death_Date"}.issubset(o.columns):
        raise ValueError("Zigong baseline/outcome fields unavailable")
    raw = b.merge(o.drop_duplicates(["PATIENT_ID", "Death_Date"]), on="PATIENT_ID", how="left", validate="one_to_many")
    raw = raw.rename(columns={"INP_NO": "stay_id", "PATIENT_ID": "subject_id", "Age": "source_age"})
    raw = normalize_keys(raw)
    assert_unique(raw, "Zigong source metadata")
    joined, coverage = strict_join(base, raw, "Zigong source metadata")
    # The preserved extractor uses ICU_discharge_time as an hour duration. It
    # does not establish the unit/anchor of Death_Date as a timestamp.
    joined["icu_start_time"] = pd.NA
    joined["icu_exit_time"] = pd.NA
    joined["icu_exit_hr"] = pd.to_numeric(joined["ICU_discharge_time"], errors="coerce")
    joined["hospital_exit_time"] = joined.get("DISCHARGE_DATE_TIME", pd.Series(pd.NA, index=joined.index)).astype("string")
    joined["hospital_exit_hr"] = np.nan
    joined["hospital_death"] = pd.array([pd.NA] * len(joined), dtype="boolean")
    joined["icu_death"] = pd.array(pd.to_numeric(joined["y_mort"], errors="coerce").eq(1), dtype="boolean")
    joined["hospital_death_time"] = pd.NA
    joined["hospital_death_time_hr"] = np.nan
    joined["hospital_death_time_kind"] = "unavailable"
    joined["icu_death_time"] = pd.NA
    joined["icu_death_time_hr"] = np.nan
    joined["icu_death_time_kind"] = "unresolved_Death_Date_units_anchor"
    joined["source_death_date_value"] = pd.to_numeric(joined["Death_Date"], errors="coerce")
    joined["source_time_basis"] = "hours_since_icu_admission_for_exit; Death_Date_anchor_unresolved"
    joined["source_timing_precision"] = "exact_offset_icu_exit_death_time_unavailable"
    joined["icu_exit_time_kind"] = "ICU_discharge_time_duration_hours"
    joined["age"] = pd.to_numeric(joined["source_age"], errors="coerce")
    joined["adult_eligible"] = joined["age"].ge(18)
    joined["adult_eligibility_source"] = "dtBaseline.Age"
    joined["sex"] = joined.get("SEX", pd.Series(pd.NA, index=joined.index))
    joined["hospitalid"] = pd.NA
    joined["hospitalid_source"] = "not_present_in_baseline_fields"
    joined["source_time_issue"] = joined["icu_exit_hr"].notna() & joined["icu_exit_hr"].lt(0)
    death_value = pd.to_numeric(joined["Death_Date"], errors="coerce")
    source_y = (death_value.notna() & death_value.le(joined["icu_exit_hr"])).astype(float)
    current_y = pd.to_numeric(joined["y_mort"], errors="coerce")
    joined["source_y_mort"] = source_y
    joined["source_y_mort_match"] = current_y.eq(source_y) | (current_y.isna() & source_y.isna())
    mismatch = int((current_y.notna() & current_y.ne(source_y)).sum())
    return joined, {"source_join_coverage": coverage, "source_hashes": file_hashes([baseline_path, outcome_path]), "source_files": [baseline_path.name, outcome_path.name], "icu_exit_source": "dtBaseline.ICU_discharge_time (duration in hours per preserved extractor)", "icu_death_source": "preserved label Death_Date <= ICU_discharge_time; exact timestamp anchor/units unverified", "hospital_id_coverage": 0, "source_label_mismatch_count": mismatch}


def c2012_outcome(base: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = CHALLENGE / "outcomes-a.txt"
    if not path.is_file():
        raise FileNotFoundError("Challenge2012 outcomes-a.txt unavailable")
    out = read_csv_columns(path, ["RecordID", "In-hospital_death"])
    if not {"RecordID", "In-hospital_death"}.issubset(out.columns):
        raise ValueError("Challenge2012 outcome columns unavailable")
    out = out.rename(columns={"RecordID": "stay_id"})
    out["subject_id"] = out["stay_id"].map(canon_id)
    out = normalize_keys(out)
    joined, coverage = strict_join(base, out, "Challenge2012 outcome source")
    joined["hospital_death"] = pd.array(pd.to_numeric(joined["In-hospital_death"], errors="coerce").eq(1), dtype="boolean")
    joined["adult_eligibility_source"] = "existing_cohort_age_filter; source record has Age but no death/ICU exit time"
    return joined, {"source_join_coverage": coverage, "source_hashes": file_hashes([path]), "source_files": [path.name], "hospital_death_source": "outcomes-a.txt In-hospital_death", "icu_exit_source": "unavailable; record observation max is not discharge time", "hospital_id_coverage": 0}


def event_time_audit(db: str) -> dict[str, Any]:
    """Reuse the prior v4 event audit and per-stay mask without rescanning events."""
    path = INTERIM / f"events_{db}.parquet"
    if not path.is_file():
        return {"status": "unavailable", "event_table": path.name}
    manifest_path = ELIG_DIR / "eligibility_manifest.json"
    previous_path = ELIG_DIR / f"{db}.parquet"
    if not manifest_path.is_file() or not previous_path.is_file():
        raise ValueError(f"{db} event audit cache unavailable; refusing an unreviewed event rescan")
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    item = previous.get("cohorts", {}).get(db, {})
    audit = item.get("feature_time_audit", {})
    if audit.get("status") != "audited" or audit.get("event_table") != path.name:
        raise ValueError(f"{db} cached event audit does not match the expected event table")
    prior = pd.read_parquet(previous_path, columns=["stay_id", "input_window_le_24h_verified"])
    negative_stays = {canon_id(x) for x in prior.loc[~prior["input_window_le_24h_verified"].fillna(False).astype(bool), "stay_id"].dropna().unique()}
    expected_negative = audit.get("n_negative_hour_stays_in_sequence")
    expected_mask_rows = item.get("counts", {}).get("n_negative_hour_input_rows")
    if expected_negative is None or expected_mask_rows is None or len(negative_stays) != int(expected_mask_rows) or len(negative_stays) < int(expected_negative):
        raise ValueError(f"{db} cached negative-hour stay mask disagrees with prior aggregate audit: mask_rows={len(negative_stays)}, expected_mask_rows={expected_mask_rows}, audited_negative_stays={expected_negative}")
    return {**audit, "cache_provenance": "prior v4 manifest plus prior eligibility exclusion mask; event table not reread", "_negative_stay_ids": negative_stays}


def assess(base: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    icu_exit = pd.to_numeric(base.get("icu_exit_hr", pd.Series(np.nan, index=base.index)), errors="coerce")
    hospital_exit = pd.to_numeric(base.get("hospital_exit_hr", pd.Series(np.nan, index=base.index)), errors="coerce")
    death_hr = pd.to_numeric(base.get("hospital_death_time_hr", pd.Series(np.nan, index=base.index)), errors="coerce")
    finite_icu_exit = pd.Series(np.isfinite(icu_exit), index=base.index)
    finite_hospital_exit = pd.Series(np.isfinite(hospital_exit), index=base.index)
    finite_death = pd.Series(np.isfinite(death_hr), index=base.index)
    invalid = base.get("source_time_issue", pd.Series(False, index=base.index)).fillna(False).astype(bool)
    endpoint_known = base.get("source_endpoint_known", pd.Series(False, index=base.index)).fillna(False).astype(bool)
    label_match = pd.Series(pd.array(base.get("source_y_mort_match", pd.Series(pd.NA, index=base.index)), dtype="boolean"), index=base.index)
    endpoint_match = label_match.fillna(False).astype(bool)
    in_icu: list[Any] = []
    for _, row in base.iterrows():
        death_by24 = any(pd.notna(row.get(c)) and float(row.get(c)) <= 24.0 for c in ("hospital_death_time_hr", "icu_death_time_hr"))
        if death_by24:
            in_icu.append(False)
        elif pd.notna(row.get("icu_exit_hr")) and np.isfinite(float(row["icu_exit_hr"])) and not bool(row.get("source_time_issue", False)):
            in_icu.append(float(row["icu_exit_hr"]) > 24.0)
        else:
            in_icu.append(pd.NA)
    base["in_icu_at_24h"] = pd.array(in_icu, dtype="boolean")
    # In-ICU at the exact boundary is false (the ICU interval is [start, exit)).
    alive: list[Any] = []
    for i, row in base.iterrows():
        death_times = [row.get("hospital_death_time_hr"), row.get("icu_death_time_hr")]
        known_death_by24 = any(pd.notna(t) and float(t) <= 24.0 for t in death_times)
        known_death_after24 = any(pd.notna(t) and float(t) > 24.0 for t in death_times)
        if known_death_by24:
            alive.append(False)
        elif bool(row.get("source_time_issue", False)) or pd.isna(row.get("icu_exit_hr")) or not np.isfinite(float(row["icu_exit_hr"])):
            alive.append(pd.NA)
        elif float(row["icu_exit_hr"]) > 24.0:
            alive.append(True)
        elif known_death_after24:
            alive.append(True)
        elif nullable_bool(row.get("hospital_death")) is False:
            alive.append(True)
        else:
            alive.append(pd.NA)
    base["alive_at_24h"] = pd.array(alive, dtype="boolean")
    timing_available = finite_icu_exit & finite_hospital_exit
    base["temporal_verified"] = timing_available & ~invalid & endpoint_known & endpoint_match & base["input_window_le_24h_verified"].fillna(False).astype(bool)
    base["icu_exit_finite_after_24h"] = finite_icu_exit & icu_exit.gt(24.0)
    base["hospital_exit_finite_after_24h"] = finite_hospital_exit & hospital_exit.gt(24.0)
    base["hospital_exit_not_before_icu_exit"] = finite_hospital_exit & finite_icu_exit & hospital_exit.ge(icu_exit)
    exact_death_valid = ~finite_death | (death_hr.gt(24.0) & finite_hospital_exit & death_hr.le(hospital_exit))
    base["exact_death_time_valid_for_landmark"] = exact_death_valid.astype(bool)
    base["eligible_24h"] = (
        base["temporal_verified"]
        & base["adult_eligible"].fillna(False).astype(bool)
        & endpoint_known
        & endpoint_match
        & base["icu_exit_finite_after_24h"]
        & base["hospital_exit_finite_after_24h"]
        & base["hospital_exit_not_before_icu_exit"]
        & base["exact_death_time_valid_for_landmark"]
        & base["alive_at_24h"].fillna(False).astype(bool)
        & base["in_icu_at_24h"].fillna(False).astype(bool)
        & base["input_available_24h"].fillna(False).astype(bool)
    )
    reasons: list[str] = []
    for _, row in base.iterrows():
        if pd.notna(row.get("adult_eligible")) and not bool(row["adult_eligible"]):
            reasons.append("age_under_18")
        elif pd.isna(row.get("adult_eligible")):
            reasons.append("adult_eligibility_unavailable")
        elif bool(row.get("source_time_issue", False)):
            reasons.append("impossible_source_time")
        elif not bool(row.get("source_endpoint_known", False)):
            reasons.append("source_endpoint_unknown")
        elif pd.notna(row.get("source_y_mort_match")) and not bool(row.get("source_y_mort_match")):
            reasons.append("source_outcome_label_mismatch")
        elif pd.isna(row.get("icu_exit_hr")):
            reasons.append("icu_exit_time_unavailable")
        elif not np.isfinite(float(row.get("icu_exit_hr"))):
            reasons.append("icu_exit_time_nonfinite")
        elif float(row["icu_exit_hr"]) <= 24.0:
            reasons.append("not_in_icu_at_24h")
        elif pd.isna(row.get("hospital_exit_hr")):
            reasons.append("hospital_exit_time_unavailable")
        elif not np.isfinite(float(row.get("hospital_exit_hr"))):
            reasons.append("hospital_exit_time_nonfinite")
        elif float(row["hospital_exit_hr"]) <= 24.0:
            reasons.append("not_in_hospital_at_24h")
        elif float(row["hospital_exit_hr"]) < float(row["icu_exit_hr"]):
            reasons.append("hospital_exit_before_icu_exit")
        elif not bool(row.get("exact_death_time_valid_for_landmark", True)):
            reasons.append("exact_death_time_inconsistent_with_landmark")
        elif not bool(row.get("input_window_le_24h_verified", False)):
            reasons.append("input_window_not_verified_le_24h")
        elif not pd.isna(row.get("in_icu_at_24h")) and not bool(row.get("in_icu_at_24h")):
            reasons.append("not_in_icu_at_24h")
        elif not pd.isna(row.get("alive_at_24h")) and not bool(row.get("alive_at_24h")):
            reasons.append("not_alive_at_24h")
        elif pd.isna(row.get("alive_at_24h")):
            reasons.append("alive_at_24h_unverified")
        elif not bool(row.get("input_available_24h", False)):
            reasons.append("no_input_observation_by_24h")
        else:
            reasons.append("eligible_24h")
    base["reason"] = reasons
    base["input_availability_cutoff_hr"] = 24.0
    return base


def missing_timing(base: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    defaults: dict[str, Any] = {
        "icu_start_time": pd.NA, "icu_exit_time": pd.NA, "icu_exit_hr": np.nan,
        "hospital_exit_time": pd.NA, "hospital_exit_hr": np.nan,
        "hospital_death": pd.NA, "hospital_death_time": pd.NA, "hospital_death_time_hr": np.nan,
        "hospital_death_time_kind": "unavailable", "icu_death": pd.NA,
        "icu_death_time": pd.NA, "icu_death_time_hr": np.nan,
        "icu_death_time_kind": "unavailable", "source_time_basis": "unavailable",
        "source_timing_precision": "unavailable", "icu_exit_time_kind": "unavailable",
        "source_time_issue": False, "hospitalid": pd.NA, "hospitalid_source": "unavailable",
        "source_endpoint_known": False, "source_endpoint_unknown": True,
        "source_y_mort": pd.NA, "source_y_mort_match": pd.NA,
        "source_landmark_contradiction": False, "icu_exit_after_hospital_exit_after_24": False,
    }
    for key, value in defaults.items():
        if key not in base:
            base[key] = value
    return base


def summarize(db: str, frame: pd.DataFrame) -> dict[str, Any]:
    def ntrue(column: str) -> int:
        if column not in frame:
            return 0
        return int(frame[column].fillna(False).sum())

    counts = {
        "n_rows": int(len(frame)),
        "n_unique_stay_subject": int(frame[["stay_id", "subject_id"]].drop_duplicates().shape[0]),
        "n_adult_eligible": ntrue("adult_eligible"),
        "n_input_available_24h": ntrue("input_available_24h"),
        "n_temporal_verified": ntrue("temporal_verified"),
        "n_alive_at_24h": ntrue("alive_at_24h"),
        "n_in_icu_at_24h": ntrue("in_icu_at_24h"),
        "n_eligible_24h": ntrue("eligible_24h"),
        "n_hospital_death": ntrue("hospital_death"),
        "n_icu_death": ntrue("icu_death"),
        "reason_counts": {str(k): int(v) for k, v in frame["reason"].value_counts(dropna=False).items()},
        "hospitalid_coverage": int(frame["hospitalid"].notna().sum()),
    }
    return counts


def post24_gap_summary(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Aggregate post-24h ICU-after-hospital exit gaps and their outcome labels."""
    gap = frame.get("icu_exit_after_hospital_exit_after_24", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    icu_exit = pd.to_numeric(frame.get("icu_exit_hr", pd.Series(np.nan, index=frame.index)), errors="coerce")
    hospital_exit = pd.to_numeric(frame.get("hospital_exit_hr", pd.Series(np.nan, index=frame.index)), errors="coerce")
    gap_hr = (icu_exit - hospital_exit).where(gap)
    finite_gap = gap_hr[np.isfinite(gap_hr)]
    if len(finite_gap):
        quantiles = np.quantile(finite_gap.to_numpy(dtype=float), [0.05, 0.5, 0.95])
        duration_quantiles = {"p05": float(quantiles[0]), "median": float(quantiles[1]), "p95": float(quantiles[2])}
    else:
        duration_quantiles = {"p05": None, "median": None, "p95": None}
    labels = pd.to_numeric(frame.get("source_y_mort", pd.Series(np.nan, index=frame.index)), errors="coerce")
    source_label_distribution = {
        "hospital_death_1": int((gap & labels.eq(1)).sum()),
        "hospital_survival_0": int((gap & labels.eq(0)).sum()),
        "unknown_or_nonbinary": int((gap & ~labels.isin([0, 1])).sum()),
    }
    return gap, gap_hr, {
        "post24_icu_after_hospital_exit_gap_rows": int(gap.sum()),
        "post24_icu_after_hospital_exit_gap_duration_hr_quantiles": duration_quantiles,
        "post24_icu_after_hospital_exit_gap_source_label_distribution": source_label_distribution,
    }


def set_readiness_contract(frame: pd.DataFrame, allowed: bool) -> pd.DataFrame:
    """Set explicit retrospective prediction readiness without a clinical-time claim."""
    frame = frame.copy()
    frame["retrospective_landmark_prediction_allowed"] = bool(allowed)
    frame["ready_for_future24h_landmark_analysis"] = bool(allowed)
    frame["ready_for_retrospective_24h_measurement_analysis"] = bool(allowed)
    frame["ready_for_future24h_clinical_claim"] = False
    frame["inputs_temporal_scope"] = "retrospective_measurement_time"
    frame["input_scope"] = "retrospective_measurement_time"
    frame["clinical_input_availability_verified"] = False
    return frame


def process(db: str, *, audit_only: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    base, base_audit = sequence_base(db)
    methods: dict[str, Any] = {}
    status = "timing_unavailable"
    evidencelevel = "unavailable"
    if db == "mimic":
        try:
            base, methods = mimic_metadata(base)
            status, evidencelevel = "exact_timestamp_metadata", "exact"
            methods["readiness_basis"] = "exact ICU intime/outtime and hospital deathtime/dischtime source timestamps"
        except Exception as exc:
            base = missing_timing(base)
            methods = {"source_join_coverage": None, "source_hashes": {}, "source_files": [], "validation_error": type(exc).__name__, "validation_error_summary": str(exc)}
            status = "source_metadata_validation_failed"
    elif db == "eicu":
        try:
            base, methods = eicu_metadata(base)
            status, evidencelevel = "exact_icu_offsets_death_times_are_discharge_proxies", "operational_proxy"
            methods["readiness_basis"] = "exact ICU/hospital discharge offsets plus explicit discharge-status mortality; death time is a discharge proxy"
        except Exception as exc:
            base = missing_timing(base)
            methods = {"source_join_coverage": None, "source_hashes": {}, "source_files": [], "validation_error": type(exc).__name__, "validation_error_summary": str(exc)}
            status = "source_metadata_validation_failed"
    elif db == "zigong":
        try:
            base, methods = zigong_metadata(base)
            status, evidencelevel = "icu_exit_offset_endpoint_unverified", "unavailable"
            methods["timing_evidencelevel"] = "operational_proxy"
            methods["readiness_basis"] = "ICU exit duration supports still-in-ICU at 24h; endpoint label remains unverified because Death_Date anchor/units and independent ICU death status are unavailable"
        except Exception as exc:
            base = missing_timing(base)
            methods = {"source_join_coverage": None, "source_hashes": {}, "source_files": [], "validation_error": type(exc).__name__}
            status = "source_metadata_validation_failed"
    elif db == "c2012":
        try:
            base, methods = c2012_outcome(base)
            status = "outcome_available_icu_exit_unavailable"
        except Exception as exc:
            methods = {"source_join_coverage": None, "source_hashes": {}, "source_files": [], "validation_error": type(exc).__name__, "path_needed": NEEDED.get(db, [])}
            status = "outcome_or_timing_metadata_unavailable"
    else:
        methods = {"source_join_coverage": None, "source_hashes": {}, "source_files": [], "path_needed": NEEDED.get(db, [])}
    if db == "c2012":
        # Outcome status is retained separately, but no discharge/death times are in the source table.
        base = missing_timing(base)
        if "In-hospital_death" in base and "hospital_death" not in base:
            base["hospital_death"] = pd.array(pd.to_numeric(base["In-hospital_death"], errors="coerce").eq(1), dtype="boolean")
        if "hospital_death_time_kind" not in base or "In-hospital_death" in base:
            base["hospital_death_time_kind"] = "no_death_timestamp_in_challenge_outcomes"
    else:
        base = missing_timing(base) if "icu_exit_hr" not in base else base
    time_audit = event_time_audit(db)
    negative_stays = time_audit.pop("_negative_stay_ids", set())
    if negative_stays:
        negative_mask = base["stay_id"].map(canon_id).isin(negative_stays)
        base.loc[negative_mask, "input_window_le_24h_verified"] = False
        time_audit["n_negative_hour_stays_in_sequence"] = int(negative_mask.sum())
    else:
        time_audit["n_negative_hour_stays_in_sequence"] = 0
    base = assess(base)
    exit_gap, exit_gap_hr, exit_gap_audit = post24_gap_summary(base)
    base["icu_exit_after_hospital_exit_after_24"] = exit_gap
    base["icu_exit_after_hospital_exit_after_24_gap_hr"] = exit_gap_hr
    if methods.get("validation_error"):
        base["eligible_24h"] = False
        base["temporal_verified"] = False
        base["reason"] = "source_metadata_validation_failed"
    if evidencelevel == "unavailable":
        base["eligible_24h"] = False
        base["temporal_verified"] = False
    mismatch_count = int(methods.get("source_label_mismatch_count", 0))
    if mismatch_count:
        if "source_y_mort_match" in base:
            mismatch_rows = base["source_y_mort_match"].eq(False).fillna(False)
            base.loc[mismatch_rows, "reason"] = "source_outcome_label_mismatch"
            base.loc[mismatch_rows, "eligible_24h"] = False
    counts = summarize(db, base)
    counts.update(exit_gap_audit)
    source_time_issue = base.get("source_time_issue", pd.Series(False, index=base.index)).fillna(False).astype(bool)
    label_mismatch = base.get("source_y_mort_match", pd.Series(True, index=base.index)).eq(False).fillna(False)
    counts["n_source_time_issue_rows"] = int(source_time_issue.sum())
    counts["n_source_time_issue_eligible"] = int((source_time_issue & base["eligible_24h"].fillna(False)).sum())
    status_conflict = base.get("source_time_status_conflict", pd.Series(False, index=base.index)).fillna(False).astype(bool)
    counts["n_source_time_status_conflicts_total"] = int(status_conflict.sum())
    counts["n_source_time_status_conflicts_eligible"] = int((status_conflict & base["eligible_24h"].fillna(False)).sum())
    counts["n_source_label_mismatch_rows"] = int(label_mismatch.sum())
    counts["n_source_label_mismatch_eligible"] = int((label_mismatch & base["eligible_24h"].fillna(False)).sum())
    endpoint_unknown = base.get("source_endpoint_unknown", pd.Series(True, index=base.index)).fillna(True).astype(bool)
    counts["n_source_endpoint_unknown_rows"] = int(endpoint_unknown.sum())
    counts["n_source_endpoint_unknown_eligible"] = int((endpoint_unknown & base["eligible_24h"].fillna(False)).sum())
    landmark_contradiction = base.get("source_landmark_contradiction", pd.Series(False, index=base.index)).fillna(False).astype(bool)
    exit_order_gap = base.get("icu_exit_after_hospital_exit_after_24", pd.Series(False, index=base.index)).fillna(False).astype(bool)
    counts["n_source_landmark_contradictions"] = int(landmark_contradiction.sum())
    counts["n_source_landmark_contradictions_eligible"] = int((landmark_contradiction & base["eligible_24h"].fillna(False)).sum())
    counts["n_icu_exit_after_hospital_exit_after_24"] = int(exit_order_gap.sum())
    counts["n_icu_exit_after_hospital_exit_after_24_eligible"] = int((exit_order_gap & base["eligible_24h"].fillna(False)).sum())
    counts["n_negative_hour_input_rows"] = int((~base["input_window_le_24h_verified"].fillna(False)).sum())
    counts["n_negative_hour_input_eligible"] = int(((~base["input_window_le_24h_verified"].fillna(False)) & base["eligible_24h"].fillna(False)).sum())
    join_coverage = methods.get("source_join_coverage")
    readiness = (
        evidencelevel != "unavailable"
        and counts["n_eligible_24h"] > 0
        and mismatch_count == 0
        and join_coverage == 1.0
        and counts["n_source_time_issue_eligible"] == 0
        and counts["n_source_time_status_conflicts_eligible"] == 0
        and counts["n_source_label_mismatch_eligible"] == 0
        and counts["n_source_endpoint_unknown_eligible"] == 0
        and counts["n_negative_hour_input_eligible"] == 0
    )
    if mismatch_count and not audit_only:
        raise ValueError(f"{db} raw source/current mortality label mismatches={mismatch_count}; refusing output write")
    base = set_readiness_contract(base, readiness)
    base["evidencelevel"] = evidencelevel
    base["subject"] = base["subject_id"].map(canon_id)
    out_cols = [
        "db", "stay_id", "subject", "subject_id", "eligible_24h", "temporal_verified", "y_mort", "endpoint", "reason",
        "ready_for_future24h_clinical_claim", "evidencelevel",
        "retrospective_landmark_prediction_allowed", "ready_for_future24h_landmark_analysis",
        "ready_for_retrospective_24h_measurement_analysis", "inputs_temporal_scope", "input_scope",
        "clinical_input_availability_verified",
        "age", "age_precision", "sex", "hospitalid", "adult_eligible", "adult_eligibility_source",
        "source_y_mort", "source_y_mort_match",
        "source_endpoint_known", "source_endpoint_unknown",
        "source_time_issue",
        "source_time_status_conflict",
        "source_landmark_contradiction", "icu_exit_after_hospital_exit_after_24",
        "icu_exit_after_hospital_exit_after_24_gap_hr",
        "icu_start_time", "icu_exit_time", "icu_exit_hr", "icu_exit_time_kind",
        "icu_exit_finite_after_24h", "hospital_exit_finite_after_24h", "hospital_exit_not_before_icu_exit",
        "exact_death_time_valid_for_landmark",
        "hospital_exit_time", "hospital_exit_hr", "hospital_death", "hospital_death_time", "hospital_death_time_hr", "hospital_death_time_kind",
        "icu_death", "icu_death_time", "icu_death_time_hr", "icu_death_time_kind",
        "alive_at_24h", "in_icu_at_24h", "input_available_24h", "input_event_count_24h", "input_max_hour_24h",
        "input_window_start_hr", "input_window_end_hr", "input_availability_cutoff_hr", "input_window_le_24h_verified",
        "source_time_basis", "source_timing_precision", "hospitalid_source",
    ]
    if "source_death_date_value" in base:
        out_cols.append("source_death_date_value")
    out = base[[c for c in out_cols if c in base.columns]].copy()
    assert_unique(out, f"eligibility_{db}")
    if not audit_only:
        ELIG_DIR.mkdir(parents=True, exist_ok=True)
        out.to_parquet(ELIG_DIR / f"{db}.parquet", index=False)
    cohort_hashes = {"cohort_parquet": base_audit.pop("cohort_sha256"), "sequence_npz": base_audit.pop("sequence_sha256")}
    hashes = {**cohort_hashes, **methods.get("source_hashes", {})}
    manifest = {
        "db": db,
        "status": status,
        "endpoint": ENDPOINT[db],
        "evidence": evidencelevel,
        "evidencelevel": evidencelevel,
        "ready_for_future24h_clinical_claim": False,
        "retrospective_landmark_prediction_allowed": readiness,
        "ready_for_future24h_landmark_analysis": readiness,
        "ready_for_retrospective_24h_measurement_analysis": readiness,
        "inputs_temporal_scope": "retrospective_measurement_time",
        "input_scope": "retrospective_measurement_time",
        "clinical_input_availability_verified": False,
        "subject_identity_basis": SUBJECT_IDENTITY[db],
        "sourcehashes": hashes,
        "sourcefiles": methods.get("source_files", []),
        "counts": {**base_audit, **counts},
        "source_join_coverage": methods.get("source_join_coverage"),
        "source_metadata_evidence": {k: v for k, v in methods.items() if k not in {"source_hashes", "source_files"}},
        "feature_time_audit": time_audit,
        "required_source_metadata": methods.get("path_needed", NEEDED.get(db, [])),
        "output": str(ELIG_DIR / f"{db}.parquet"),
    }
    print(f"{db}: rows={counts['n_rows']} source_join={manifest['source_join_coverage']} temporal={counts['n_temporal_verified']} eligible24={counts['n_eligible_24h']} hospitalid={counts['hospitalid_coverage']} status={status}", flush=True)
    return manifest, counts


def main() -> None:
    audit_only = os.environ.get("TEMPORAL_AUDIT_ONLY", "0") == "1"
    manifests: dict[str, Any] = {}
    flows: list[dict[str, Any]] = []
    selected = tuple(x.strip() for x in os.environ.get("TEMPORAL_DATABASES", ",".join(DATABASES)).split(",") if x.strip())
    unknown = set(selected).difference(DATABASES)
    if unknown:
        raise ValueError(f"unknown databases requested: {sorted(unknown)}")
    for db in selected:
        manifest, counts = process(db, audit_only=audit_only)
        manifests[db] = manifest
        flows.append({"db": db, "endpoint": ENDPOINT[db], "status": manifest["status"], **{k: v for k, v in counts.items() if not isinstance(v, dict)}})
    if audit_only:
        report = {
            db: {
                "status": item["status"],
                "source_join_coverage": item["source_join_coverage"],
                "source_hashes": item["sourcehashes"],
                "counts": item["counts"],
                "source_metadata_evidence": item["source_metadata_evidence"],
                "retrospective_landmark_prediction_allowed": item["retrospective_landmark_prediction_allowed"],
                "ready_for_future24h_landmark_analysis": item["ready_for_future24h_landmark_analysis"],
                "ready_for_retrospective_24h_measurement_analysis": item["ready_for_retrospective_24h_measurement_analysis"],
                "input_scope": item["input_scope"],
                "clinical_input_availability_verified": item["clinical_input_availability_verified"],
            }
            for db, item in manifests.items()
        }
        print(json.dumps({"audit_only": True, "schema_version": "4.1", "cohorts": report}, sort_keys=True, allow_nan=False), flush=True)
        return
    ELIG_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = ELIG_DIR / "eligibility_manifest.json"
    manifest_path.write_text(json.dumps({"schema_version": "4.1", "cohorts": manifests}, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    (OUT_ROOT / "results").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(flows).to_csv(OUT_ROOT / "results" / "eligibility_flow.csv", index=False)
    print(f"manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
