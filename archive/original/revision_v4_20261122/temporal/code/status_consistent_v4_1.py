#!/usr/bin/env python3
"""Create corrected, separately versioned MIMIC/eICU eligibility variants.

Existing v4.1 outputs are read-only inputs. This correction only writes under
eligibility/status_consistent_v4_1 and does not reopen feature/event tables.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = PROJECT / "revision_v4_20261122"
ELIG = REV / "temporal/results/eligibility"
PROCESSED = PROJECT / "data/processed"
MIMIC_PATIENTS = Path("/data/admin/fengshuo/if10_20261002/shared/raw/mimic/hosp/patients.csv.gz")
EICU_PATIENT = Path("/data/admin/fengshuo/if10_20261002/shared/raw/eicu/patient.csv.gz")
OUT_ROOT = ELIG / "status_consistent_v4_1"
STAGE_ROOT = ELIG / ".status_consistent_v4_1.staging"
VARIANTS = ("landmark_v4_1", "strict_v4_1")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_id(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)) and math.isfinite(float(value)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def bool_col(frame: pd.DataFrame, name: str, default: bool = False) -> pd.Series:
    if name not in frame:
        return pd.Series(default, index=frame.index, dtype=bool)
    return frame[name].fillna(default).astype(bool)


def nullable_status(values: pd.Series) -> pd.Series:
    low = values.astype("string").str.strip().str.lower()
    dead = low.isin({"dead", "expired", "deceased", "death"}) | low.str.contains("expired|deceased", na=False)
    alive = low.isin({"alive", "survived", "survivor"}) | low.str.contains("alive|surviv", na=False)
    result = pd.Series(pd.NA, index=values.index, dtype="boolean")
    result.loc[dead] = True
    result.loc[alive] = False
    return result


def eicu_status_conflict(unit_status: pd.Series, hospital_status: pd.Series) -> pd.Series:
    unit_dead = nullable_status(unit_status)
    hospital_dead = nullable_status(hospital_status)
    return (unit_dead.fillna(False) & hospital_dead.eq(False).fillna(False)).astype(bool)


def reconcile_hospital_ids(legacy: pd.Series, raw: pd.Series) -> pd.Series:
    old_hosp = legacy.map(canonical_id)
    raw_hosp = raw.map(canonical_id)
    if old_hosp.isna().any() or raw_hosp.isna().any():
        raise ValueError("eICU legacy/raw hospital ID columns do not both cover every row")
    disagreement = old_hosp.ne(raw_hosp)
    if disagreement.any():
        raise ValueError(f"eICU hospital ID columns disagree for {int(disagreement.sum())} rows")
    return raw_hosp


def admit_order_flags(offset_hours: pd.Series) -> tuple[pd.Series, pd.Series]:
    offset = pd.to_numeric(offset_hours, errors="coerce")
    finite = pd.Series(np.isfinite(offset), index=offset.index)
    return finite & offset.gt(0), finite & offset.gt(24)


def mimic_dod_bounds(dod: pd.Series, icu_start: pd.Series) -> pd.DataFrame:
    day = pd.to_datetime(dod, errors="coerce").dt.normalize()
    start = pd.to_datetime(icu_start, errors="coerce")
    lower = (day - start).dt.total_seconds() / 3600.0
    upper = (day + pd.Timedelta(days=1) - start).dt.total_seconds() / 3600.0
    known = day.notna() & start.notna() & np.isfinite(lower) & np.isfinite(upper)
    return pd.DataFrame({
        "source_patient_dod_lower_bound_hr": lower.where(known),
        "source_patient_dod_upper_bound_hr": upper.where(known),
        "source_patient_dod_interval_known": known.astype(bool),
        "source_patient_dod_death_by_24h_proven": (known & upper.le(24)).astype(bool),
        "source_patient_dod_preindex_proven": (known & upper.le(0)).astype(bool),
        "source_patient_dod_interval_overlaps_24h": (known & lower.le(24) & upper.gt(24)).astype(bool),
    }, index=dod.index)


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


def gap_summary(frame: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
    gap = bool_col(frame, "source_postlandmark_ordering_issue") & mask
    values = pd.to_numeric(frame.loc[gap, "icu_exit_after_hospital_exit_after_24_gap_hr"], errors="coerce")
    values = values[np.isfinite(values)]
    if len(values):
        q = np.quantile(values.to_numpy(dtype=float), [0.05, 0.5, 0.95])
        quantiles = {"p05": float(q[0]), "median": float(q[1]), "p95": float(q[2])}
    else:
        quantiles = {"p05": None, "median": None, "p95": None}
    y = pd.to_numeric(frame.loc[gap, "source_y_mort"], errors="coerce")
    return {
        "rows": int(gap.sum()),
        "duration_hr_quantiles": quantiles,
        "source_label_distribution": {
            "hospital_death_1": int(y.eq(1).sum()),
            "hospital_survival_0": int(y.eq(0).sum()),
            "unknown_or_nonbinary": int((~y.isin([0, 1])).sum()),
        },
    }


def reason_column(frame: pd.DataFrame, db: str, selected: pd.Series,
                  dod_by24: pd.Series, variant: str) -> pd.Series:
    reason = pd.Series("not_eligible_24h", index=frame.index, dtype="string")
    endpoint_unknown = bool_col(frame, "source_endpoint_unknown", True)
    label_bad = frame["source_y_mort_match"].eq(False).fillna(False)
    landmark = bool_col(frame, "source_landmark_issue")
    status_conflict = bool_col(frame, "source_time_status_conflict")
    input_bad = ~bool_col(frame, "input_window_le_24h_verified")
    icu_exit = pd.to_numeric(frame["icu_exit_hr"], errors="coerce")
    hospital_exit = pd.to_numeric(frame["hospital_exit_hr"], errors="coerce")
    finite_icu = pd.Series(np.isfinite(icu_exit), index=frame.index)
    finite_hosp = pd.Series(np.isfinite(hospital_exit), index=frame.index)
    gap = bool_col(frame, "source_postlandmark_ordering_issue")
    conditions = [
        (status_conflict, "source_time_status_conflict"),
        (dod_by24, "source_mimic_dod_death_by_24h_proven"),
        (landmark, "source_landmark_issue"),
        (endpoint_unknown, "source_endpoint_unknown"),
        (label_bad, "source_outcome_label_mismatch"),
        (~bool_col(frame, "adult_eligible"), "not_adult_eligible"),
        (input_bad | ~bool_col(frame, "input_available_24h"), "input_window_not_verified_le_24h"),
        (~finite_icu | icu_exit.le(24), "not_in_icu_at_24h"),
        (~finite_hosp | hospital_exit.le(24), "not_in_hospital_at_24h"),
        ((variant == "strict_v4_1") & gap, "post24_exit_order_sensitivity_exclusion"),
    ]
    for mask, label in reversed(conditions):
        if not isinstance(mask, pd.Series):
            mask = pd.Series(bool(mask), index=frame.index)
        reason.loc[(~selected) & mask.fillna(False).astype(bool)] = label
    reason.loc[selected] = "eligible_24h"
    return reason


def summarize_final(db: str, frame: pd.DataFrame, selected: pd.Series) -> dict[str, Any]:
    y = pd.to_numeric(frame["y_mort"], errors="coerce")
    source_y = pd.to_numeric(frame["source_y_mort"], errors="coerce")
    status_conflict = bool_col(frame, "source_time_status_conflict")
    landmark_issue = bool_col(frame, "source_landmark_issue")
    gap = bool_col(frame, "source_postlandmark_ordering_issue")
    negative_input = ~bool_col(frame, "input_window_le_24h_verified")
    hospitalid = frame.get("hospitalid", pd.Series(pd.NA, index=frame.index)).map(canonical_id)
    if db == "mimic":
        dod_known = bool_col(frame, "source_patient_dod_interval_known")
        dod_proven = bool_col(frame, "source_patient_dod_death_by_24h_proven")
        dod_preindex = bool_col(frame, "source_patient_dod_preindex_proven")
        dod_overlap = bool_col(frame, "source_patient_dod_interval_overlaps_24h")
    else:
        dod_known = dod_proven = dod_preindex = dod_overlap = pd.Series(False, index=frame.index)
    return {
        "n_rows": int(len(frame)),
        "n_unique_stay_subject": int(frame[["stay_id", "subject_id"]].drop_duplicates().shape[0]),
        "n_adult_eligible": int(bool_col(frame, "adult_eligible").sum()),
        "n_input_available_24h": int(bool_col(frame, "input_available_24h").sum()),
        "n_negative_hour_input_rows": int(negative_input.sum()),
        "n_negative_hour_input_eligible": int((negative_input & selected).sum()),
        "n_temporal_verified": int(bool_col(frame, "temporal_verified").sum()),
        "n_alive_at_24h": int(bool_col(frame, "alive_at_24h").sum()),
        "n_alive_at_24h_eligible": int((bool_col(frame, "alive_at_24h") & selected).sum()),
        "n_in_icu_at_24h": int(bool_col(frame, "in_icu_at_24h").sum()),
        "n_in_icu_at_24h_eligible": int((bool_col(frame, "in_icu_at_24h") & selected).sum()),
        "n_eligible_24h": int(selected.sum()),
        "n_eligible_positive": int((selected & y.eq(1)).sum()),
        "n_eligible_negative": int((selected & y.eq(0)).sum()),
        "n_hospital_death": int(bool_col(frame, "hospital_death").sum()),
        "n_icu_death": int(bool_col(frame, "icu_death").sum()),
        "reason_counts": {str(k): int(v) for k, v in frame["reason"].value_counts(dropna=False).items()},
        "hospitalid_coverage": int(hospitalid.notna().sum()),
        "hospitalid_coverage_eligible": int(hospitalid.loc[selected].notna().sum()),
        "hospitalid_unique_all": int(hospitalid.dropna().nunique()),
        "hospitalid_unique_eligible": int(hospitalid.loc[selected].dropna().nunique()),
        "n_source_endpoint_unknown_rows": int(bool_col(frame, "source_endpoint_unknown", True).sum()),
        "n_source_endpoint_unknown_eligible": int((bool_col(frame, "source_endpoint_unknown", True) & selected).sum()),
        "n_source_label_mismatch_rows": int(frame["source_y_mort_match"].eq(False).fillna(False).sum()),
        "n_source_label_mismatch_eligible": int((frame["source_y_mort_match"].eq(False).fillna(False) & selected).sum()),
        "n_source_time_status_conflicts_total": int(status_conflict.sum()),
        "n_source_time_status_conflicts_eligible": int((status_conflict & selected).sum()),
        "n_eicu_unit_expired_hospital_alive": int(bool_col(frame, "source_unit_expired_hospital_alive_conflict").sum()) if db == "eicu" else 0,
        "n_source_landmark_issues": int(landmark_issue.sum()),
        "n_source_landmark_issues_eligible": int((landmark_issue & selected).sum()),
        "n_source_landmark_contradictions": int(bool_col(frame, "source_landmark_contradiction").sum()),
        "n_source_landmark_contradictions_eligible": int((bool_col(frame, "source_landmark_contradiction") & selected).sum()),
        "n_post24_exit_order_issues": int(gap.sum()),
        "n_post24_exit_order_issues_eligible": int((gap & selected).sum()),
        "post24_exit_order_all": gap_summary(frame, pd.Series(True, index=frame.index)),
        "post24_exit_order_eligible": gap_summary(frame, selected),
        "n_hospital_admit_after_icu_start": int(bool_col(frame, "source_hospital_admit_after_icu_start").sum()),
        "n_hospital_admit_after_icu_start_eligible": int((bool_col(frame, "source_hospital_admit_after_icu_start") & selected).sum()),
        "n_hospital_admit_after_24h": int(bool_col(frame, "source_hospital_admit_after_24h").sum()),
        "n_hospital_admit_after_24h_eligible": int((bool_col(frame, "source_hospital_admit_after_24h") & selected).sum()),
        "n_mimic_dod_interval_known": int(dod_known.sum()),
        "n_mimic_dod_death_by_24h_proven": int(dod_proven.sum()),
        "n_mimic_dod_death_by_24h_proven_eligible": int((dod_proven & selected).sum()),
        "n_mimic_dod_preindex_proven": int(dod_preindex.sum()),
        "n_mimic_dod_interval_overlaps_24h_uncertain": int(dod_overlap.sum()),
        "n_mimic_dod_interval_overlaps_24h_uncertain_eligible": int((dod_overlap & selected).sum()),
        "source_label_distribution_eligible": {
            "hospital_death_1": int((selected & source_y.eq(1)).sum()),
            "hospital_survival_0": int((selected & source_y.eq(0)).sum()),
            "unknown_or_nonbinary": int((selected & ~source_y.isin([0, 1])).sum()),
        },
    }


def prepare_frame(db: str, variant: str, original: pd.DataFrame,
                  mimic_dod: pd.Series | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = original.copy()
    if not {"stay_id", "subject", "subject_id", "eligible_24h", "y_mort", "source_y_mort",
            "source_y_mort_match", "source_endpoint_known", "source_endpoint_unknown",
            "source_time_status_conflict", "icu_exit_hr", "hospital_exit_hr",
            "input_window_le_24h_verified"}.issubset(frame.columns):
        raise ValueError(f"{variant}/{db}: current table is missing required source/eligibility fields")
    frame["stay_id"] = frame["stay_id"].map(canonical_id)
    frame["subject_id"] = frame["subject_id"].map(canonical_id)
    frame["subject"] = frame["subject_id"]
    if frame[["stay_id", "subject_id"]].isna().any().any() or frame.duplicated(["stay_id", "subject_id"]).any():
        raise ValueError(f"{variant}/{db}: invalid or duplicate keys")
    selected_before = bool_col(frame, "eligible_24h")

    if db == "eicu":
        for column in ("unitdischargestatus", "hospitaldischargestatus", "hospitalid_x", "hospitalid_y", "hospitaladmitoffset"):
            if column not in frame:
                raise ValueError(f"eICU metadata missing retained source field {column}")
        unit_status = frame["unitdischargestatus"].astype("string").str.strip()
        hospital_status = frame["hospitaldischargestatus"].astype("string").str.strip()
        explicit_conflict = eicu_status_conflict(unit_status, hospital_status)
        frame["source_unitdischargestatus"] = unit_status
        frame["source_hospitaldischargestatus"] = hospital_status
        frame["source_unit_expired_hospital_alive_conflict"] = explicit_conflict
        status_conflict = bool_col(frame, "source_time_status_conflict") | explicit_conflict
        frame["source_time_status_conflict"] = status_conflict

        old_hosp = frame["hospitalid_x"].map(canonical_id)
        raw_hosp = reconcile_hospital_ids(frame["hospitalid_x"], frame["hospitalid_y"])
        frame["hospitalid_legacy_snapshot"] = old_hosp
        frame["hospitalid_source_raw"] = raw_hosp
        frame["hospitalid"] = raw_hosp
        frame["hospitalid_source"] = "patient.hospitalid_raw_join"
        admit_offset_hr = pd.to_numeric(frame["hospitaladmitoffset"], errors="coerce") / 60.0
    else:
        if mimic_dod is None or len(mimic_dod) != len(frame):
            raise ValueError("MIMIC DOD source metadata was not joined one-to-one")
        bounds = mimic_dod_bounds(mimic_dod, frame["icu_start_time"])
        frame["source_patient_dod"] = mimic_dod.astype("string")
        for column in bounds:
            frame[column] = bounds[column]
        status_conflict = bool_col(frame, "source_time_status_conflict")
        if "admittime" not in frame:
            raise ValueError("MIMIC retained source admissions.admittime is missing")
        admit_time = pd.to_datetime(frame["admittime"], errors="coerce")
        icu_start = pd.to_datetime(frame["icu_start_time"], errors="coerce")
        admit_offset_hr = (admit_time - icu_start).dt.total_seconds() / 3600.0

    after_icu, after_24 = admit_order_flags(admit_offset_hr)
    frame["source_hospital_admit_offset_from_icu_start_hr"] = admit_offset_hr
    frame["source_hospital_admit_after_icu_start"] = after_icu
    frame["source_hospital_admit_after_24h"] = after_24

    icu_exit = pd.to_numeric(frame["icu_exit_hr"], errors="coerce")
    hospital_exit = pd.to_numeric(frame["hospital_exit_hr"], errors="coerce")
    post_gap = np.isfinite(icu_exit) & np.isfinite(hospital_exit) & icu_exit.gt(hospital_exit) & hospital_exit.gt(24)
    if "source_postlandmark_ordering_issue" in frame and not bool_col(frame, "source_postlandmark_ordering_issue").equals(post_gap.astype(bool)):
        raise ValueError(f"{db} existing post-landmark ordering flag disagrees with source times")
    frame["source_postlandmark_ordering_issue"] = post_gap.astype(bool)
    frame["icu_exit_after_hospital_exit_after_24"] = post_gap.astype(bool)

    if db == "mimic":
        dod_proven = bool_col(frame, "source_patient_dod_death_by_24h_proven")
        preindex = bool_col(frame, "source_patient_dod_preindex_proven")
    else:
        dod_proven = pd.Series(False, index=frame.index)
        preindex = pd.Series(False, index=frame.index)
    prior_landmark = bool_col(frame, "source_landmark_issue") if "source_landmark_issue" in frame else bool_col(frame, "source_time_issue") & ~post_gap
    landmark_issue = prior_landmark | dod_proven | preindex
    frame["source_landmark_issue"] = landmark_issue
    frame["source_time_issue"] = bool_col(frame, "source_time_issue") | landmark_issue | bool_col(frame, "source_time_status_conflict") | post_gap

    if variant == "strict_v4_1":
        if (selected_before & post_gap).any():
            raise ValueError("strict variant input unexpectedly selects post-24h exit-order rows")
    excluded_new = status_conflict | dod_proven | preindex
    selected = selected_before & ~excluded_new
    frame["eligible_24h"] = selected.astype(bool)
    frame["source_endpoint_known"] = bool_col(frame, "source_endpoint_known")
    frame["source_endpoint_unknown"] = ~frame["source_endpoint_known"]
    frame["source_y_mort_match"] = frame["source_y_mort_match"].astype("boolean")
    frame["retrospective_landmark_prediction_allowed"] = True
    frame["ready_for_future24h_landmark_analysis"] = True
    frame["ready_for_retrospective_24h_measurement_analysis"] = True
    frame["ready_for_future24h_clinical_claim"] = False
    frame["clinical_input_availability_verified"] = False
    frame["input_scope"] = "retrospective_measurement_time"
    frame["inputs_temporal_scope"] = "retrospective_measurement_time"

    # Proven early death means the patient was not alive/in ICU at the landmark.
    if "alive_at_24h" not in frame:
        frame["alive_at_24h"] = pd.array([pd.NA] * len(frame), dtype="boolean")
    else:
        frame["alive_at_24h"] = pd.array(frame["alive_at_24h"], dtype="boolean")
    if "in_icu_at_24h" not in frame:
        frame["in_icu_at_24h"] = pd.array([pd.NA] * len(frame), dtype="boolean")
    else:
        frame["in_icu_at_24h"] = pd.array(frame["in_icu_at_24h"], dtype="boolean")
    frame.loc[dod_proven | preindex, ["alive_at_24h", "in_icu_at_24h"]] = False
    frame.loc[status_conflict, ["alive_at_24h", "in_icu_at_24h"]] = pd.NA
    frame.loc[status_conflict | dod_proven | preindex, "temporal_verified"] = False
    frame.loc[selected, ["alive_at_24h", "in_icu_at_24h", "temporal_verified"]] = True
    frame["reason"] = reason_column(frame, db, selected, dod_proven | preindex, variant)

    if db == "mimic":
        frame["source_patient_dod_date_precision"] = "date_only_24h_interval"
        frame["source_patient_dod_time_kind"] = "day_granular_interval_not_exact_timestamp"
    else:
        frame["source_status_conflict_rule"] = "unitdischargestatus expired AND hospitaldischargestatus alive"

    details = {
        "prior_selected_rows": int(selected_before.sum()),
        "new_status_conflicts": int(status_conflict.sum()) if db == "eicu" else 0,
        "new_proven_dod_exclusions": int((dod_proven | preindex).sum()) if db == "mimic" else 0,
        "new_selected_rows": int(selected.sum()),
        "status_conflicts_selected": int((status_conflict & selected).sum()),
        "dod_proven_selected": int((dod_proven & selected).sum()),
        "post24_exit_order_selected": int((post_gap & selected).sum()),
        "hospital_admit_after_icu_start_all": int(frame["source_hospital_admit_after_icu_start"].sum()),
        "hospital_admit_after_icu_start_selected": int((frame["source_hospital_admit_after_icu_start"] & selected).sum()),
        "hospital_admit_after_24h_all": int(frame["source_hospital_admit_after_24h"].sum()),
        "hospital_admit_after_24h_selected": int((frame["source_hospital_admit_after_24h"] & selected).sum()),
    }
    return frame, details


def main() -> None:
    if OUT_ROOT.exists() or STAGE_ROOT.exists():
        raise FileExistsError("status_consistent_v4_1 output/staging path already exists; refusing overwrite")
    manifests = {}
    frames_by_variant: dict[str, dict[str, pd.DataFrame]] = {v: {} for v in VARIANTS}
    source_manifest = {}
    for variant in VARIANTS:
        path = ELIG / variant / "eligibility_manifest.json"
        source_manifest[variant] = json.loads(path.read_text(encoding="utf-8"))
        if source_manifest[variant].get("variant") != variant:
            raise ValueError(f"input manifest variant identity mismatch: {variant}")

    source_hashes = source_manifest["landmark_v4_1"]["cohorts"]["mimic"]["sourcehashes"]
    mimic_pat_hash = sha256(MIMIC_PATIENTS)
    if mimic_pat_hash != source_hashes.get("patients.csv.gz"):
        raise ValueError("MIMIC patients.csv.gz hash differs from the preserved source attestation")
    print("status_fix_stage=mimic_dod_read_begin", flush=True)
    pats = pd.read_csv(MIMIC_PATIENTS, usecols=["subject_id", "dod"], compression="infer", low_memory=False)
    pats["subject_key"] = pats["subject_id"].map(canonical_id)
    if pats["subject_key"].isna().any() or pats["subject_key"].duplicated().any():
        raise ValueError("MIMIC patients DOD key table has null or duplicate subject IDs")
    dod_by_subject = pats.set_index("subject_key")["dod"]
    print(f"status_fix_stage=mimic_dod_read_complete rows={len(pats)}", flush=True)

    for variant in VARIANTS:
        old_entry = source_manifest[variant]["cohorts"]["mimic"]
        if old_entry.get("sourcehashes") != source_hashes:
            raise ValueError("MIMIC source hashes differ between primary and strict inputs")
        npz_path = PROCESSED / "seq_mimic.npz"
        for db in ("mimic", "eicu"):
            input_entry = source_manifest[variant]["cohorts"][db]
            npz_path = PROCESSED / f"seq_{db}.npz"
            if sha256(npz_path) != input_entry.get("sourcehashes", {}).get("sequence_npz"):
                raise ValueError(f"{db} sequence NPZ hash changed")
            table_path = ELIG / variant / f"{db}.parquet"
            if input_entry.get("eligibility_table_sha256") != sha256(table_path):
                raise ValueError(f"{variant}/{db} input table hash mismatch")
            print(f"status_fix_stage={variant}_{db}_input_read_begin", flush=True)
            original = pd.read_parquet(table_path)
            print(f"status_fix_stage={variant}_{db}_input_read_complete rows={len(original)}", flush=True)
            if db == "mimic":
                subject_keys = original["subject_id"].map(canonical_id)
                if not subject_keys.isin(dod_by_subject.index).all():
                    raise ValueError("MIMIC patients DOD join misses source subject IDs")
                dod = subject_keys.map(dod_by_subject)
                mimic_dod = dod
            else:
                mimic_dod = None
                eicu_source_hash = input_entry.get("sourcehashes", {}).get("patient.csv.gz")
                if eicu_source_hash is None:
                    raise ValueError("eICU patient source hash absent from input manifest")
                # The row-level statuses and hospital IDs in the cached table were
                # read from this source file and are independently checked later.
            frame, details = prepare_frame(db, variant, original, mimic_dod)
            if not bool_col(frame, "source_endpoint_known").loc[bool_col(frame, "eligible_24h")].all():
                raise ValueError(f"{variant}/{db} selected unknown source endpoint")
            if not frame.loc[bool_col(frame, "eligible_24h"), "source_y_mort_match"].fillna(False).astype(bool).all():
                raise ValueError(f"{variant}/{db} selected source/current label mismatch")
            if ((~bool_col(frame, "input_window_le_24h_verified")) & bool_col(frame, "eligible_24h")).any():
                raise ValueError(f"{variant}/{db} selected invalid input mask")
            frames_by_variant[variant][db] = frame
            details["source_hashes"] = input_entry["sourcehashes"]
            print(
                f"status_fix {variant} {db}: prior={details['prior_selected_rows']} selected={details['new_selected_rows']} "
                f"new_status_conflicts={details['new_status_conflicts']} new_proven_dod={details['new_proven_dod_exclusions']} "
                f"post24_selected={details['post24_exit_order_selected']} "
                f"admit_after_icu_selected={details['hospital_admit_after_icu_start_selected']} "
                f"admit_after24_selected={details['hospital_admit_after_24h_selected']}", flush=True)

    primary_keys = {
        db: set(frames_by_variant["landmark_v4_1"][db].loc[bool_col(frames_by_variant["landmark_v4_1"][db], "eligible_24h"), "stay_id"].astype(str))
        for db in ("mimic", "eicu")
    }
    for db in ("mimic", "eicu"):
        strict_frame = frames_by_variant["strict_v4_1"][db]
        strict_keys = set(strict_frame.loc[bool_col(strict_frame, "eligible_24h"), "stay_id"].astype(str))
        if not strict_keys.issubset(primary_keys[db]):
            raise ValueError(f"{db} corrected strict cohort is not a subset of primary")

    STAGE_ROOT.mkdir(parents=True)
    try:
        current_paths = {}
        for variant in VARIANTS:
            role = "primary_retrospective_landmark" if variant == "landmark_v4_1" else "post24_exit_order_sensitivity"
            variant_stage = STAGE_ROOT / variant
            variant_stage.mkdir()
            cohorts = {}
            for db in ("mimic", "eicu"):
                frame = frames_by_variant[variant][db]
                selected = bool_col(frame, "eligible_24h")
                counts = summarize_final(db, frame, selected)
                ready = bool(selected.any() and counts["n_source_endpoint_unknown_eligible"] == 0
                             and counts["n_source_label_mismatch_eligible"] == 0
                             and counts["n_source_time_status_conflicts_eligible"] == 0
                             and counts["n_source_landmark_issues_eligible"] == 0
                             and counts["n_mimic_dod_death_by_24h_proven_eligible"] == 0
                             and counts["n_negative_hour_input_eligible"] == 0
                             and counts["n_alive_at_24h_eligible"] == int(selected.sum())
                             and counts["n_in_icu_at_24h_eligible"] == int(selected.sum()))
                if not ready:
                    raise ValueError(f"{variant}/{db} retrospective readiness gate failed from final table")
                table = variant_stage / f"{db}.parquet"
                tmp = table.with_name(table.name + ".tmp")
                frame.to_parquet(tmp, index=False)
                os.replace(tmp, table)
                previous = source_manifest[variant]["cohorts"][db]
                source_meta = dict(previous.get("source_metadata_evidence", {}))
                source_meta.update({
                    "unit_expired_hospital_alive_conflict_rows": counts["n_eicu_unit_expired_hospital_alive"],
                    "mimic_patients_dod_precision": "date_only; bounds are [dod midnight, next midnight) relative to ICU intime",
                    "mimic_dod_proven_death_by_24h_rows": counts["n_mimic_dod_death_by_24h_proven"],
                    "mimic_dod_preindex_proven_rows": counts["n_mimic_dod_preindex_proven"],
                    "mimic_dod_24h_overlap_uncertain_rows": counts["n_mimic_dod_interval_overlaps_24h_uncertain"],
                    "hospital_admit_after_icu_start_rows": counts["n_hospital_admit_after_icu_start"],
                    "hospital_admit_after_24h_rows": counts["n_hospital_admit_after_24h"],
                    "hospitalid_coverage": counts["hospitalid_coverage"],
                    "hospitalid_unique": counts["hospitalid_unique_all"],
                    "hospitalid_source": "eICU patient.hospitalid raw column" if db == "eicu" else "not available in MIMIC admission rows",
                    "source_endpoint_unknown_count": counts["n_source_endpoint_unknown_rows"],
                    "source_label_mismatch_count": counts["n_source_label_mismatch_rows"],
                    "source_time_status_conflicts": counts["n_source_time_status_conflicts_total"],
                    "source_landmark_contradictions": counts["n_source_landmark_contradictions"],
                    "icu_exit_after_hospital_exit_after_24": counts["n_post24_exit_order_issues"],
                })
                entry = {
                    "db": db,
                    "variant": variant,
                    "variant_role": role,
                    "status": previous.get("status"),
                    "endpoint": previous.get("endpoint"),
                    "evidence": previous.get("evidence"),
                    "evidencelevel": previous.get("evidencelevel"),
                    "ready_for_future24h_clinical_claim": False,
                    "retrospective_landmark_prediction_allowed": True,
                    "ready_for_future24h_landmark_analysis": True,
                    "ready_for_retrospective_24h_measurement_analysis": True,
                    "inputs_temporal_scope": "retrospective_measurement_time",
                    "input_scope": "retrospective_measurement_time",
                    "clinical_input_availability_verified": False,
                    "subject_identity_basis": previous.get("subject_identity_basis"),
                    "sourcehashes": previous.get("sourcehashes"),
                    "sourcefiles": previous.get("sourcefiles", []),
                    "source_join_coverage": previous.get("source_join_coverage"),
                    "source_metadata_evidence": source_meta,
                    "feature_time_audit": previous.get("feature_time_audit", {}),
                    "source_issue_definitions": {
                        "eicu_status_conflict": "unitdischargestatus is expired while hospitaldischargestatus is alive",
                        "mimic_dod_interval": "date-only DOD is represented as [midnight, next midnight); exclude only when interval upper bound is at or before 24h or at/before index",
                        "source_postlandmark_ordering_issue": "finite ICU exit is after finite hospital exit, with hospital exit after 24h; retained in the primary variant and excluded only in post24_exit_order_sensitivity",
                        "hospital_admit_order": "recorded as an administrative timing flag only; does not independently exclude rows",
                    },
                    "counts": counts,
                    "readiness_audit": {
                        "eligible_rows_checked": int(selected.sum()),
                        "source_endpoint_unknown_selected": counts["n_source_endpoint_unknown_eligible"],
                        "source_label_mismatch_selected": counts["n_source_label_mismatch_eligible"],
                        "source_time_status_conflicts_selected": counts["n_source_time_status_conflicts_eligible"],
                        "source_landmark_issues_selected": counts["n_source_landmark_issues_eligible"],
                        "mimic_dod_death_by_24h_proven_selected": counts["n_mimic_dod_death_by_24h_proven_eligible"],
                        "negative_hour_input_selected": counts["n_negative_hour_input_eligible"],
                        "alive_at_24h_selected": counts["n_alive_at_24h_eligible"],
                        "in_icu_at_24h_selected": counts["n_in_icu_at_24h_eligible"],
                        "post24_exit_order_selected": counts["n_post24_exit_order_issues_eligible"],
                        "retrospective_landmark_prediction_allowed": True,
                        "ready_for_future24h_clinical_claim": False,
                        "clinical_input_availability_verified": False,
                    },
                    "selected_keys_labels_sha256": selected_digest(db, frame, selected),
                    "eligibility_table_sha256": sha256(table),
                    "output": str(OUT_ROOT / variant / table.name),
                }
                cohorts[db] = entry
                print(
                    f"final {variant} {db}: rows={counts['n_rows']} selected={counts['n_eligible_24h']} "
                    f"positive={counts['n_eligible_positive']} negative={counts['n_eligible_negative']} "
                    f"status_conflicts={counts['n_source_time_status_conflicts_total']} "
                    f"dod_proven={counts['n_mimic_dod_death_by_24h_proven']} "
                    f"dod_overlap={counts['n_mimic_dod_interval_overlaps_24h_uncertain']} "
                    f"hospid_cov={counts['hospitalid_coverage']} "
                    f"admit_after_icu={counts['n_hospital_admit_after_icu_start_eligible']} "
                    f"admit_after24={counts['n_hospital_admit_after_24h_eligible']} "
                    f"table_sha256={entry['eligibility_table_sha256']}", flush=True)
            manifest = {
                "schema_version": "4.1",
                "variant": variant,
                "variant_role": role,
                "status_consistency_version": "status_consistent_v4_1",
                "derived_from": str(ELIG / variant),
                "source_audit": "row-level retained source status plus date-bounded MIMIC patients.dod; no event rescan",
                "cohorts": cohorts,
            }
            manifest_tmp = variant_stage / "eligibility_manifest.json.tmp"
            manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
            os.replace(manifest_tmp, variant_stage / "eligibility_manifest.json")
            current_paths[variant] = {
                "eligibility_dir": str(OUT_ROOT / variant),
                "manifest": str(OUT_ROOT / variant / "eligibility_manifest.json"),
                "cohort_tables": {db: str(OUT_ROOT / variant / f"{db}.parquet") for db in ("mimic", "eicu")},
                "manifest_sha256": sha256(variant_stage / "eligibility_manifest.json"),
            }
        path_file = STAGE_ROOT / "current_paths.json"
        path_file.write_text(json.dumps({"version": "status_consistent_v4_1", "variants": current_paths}, indent=2, sort_keys=True), encoding="utf-8")
        STAGE_ROOT.rename(OUT_ROOT)
        print(f"status_fix_complete output={OUT_ROOT}", flush=True)
    except Exception:
        if STAGE_ROOT.exists():
            shutil.rmtree(STAGE_ROOT)
        raise


if __name__ == "__main__":
    main()
