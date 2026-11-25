#!/usr/bin/env python3
"""Independent source-row and selected-population verification."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT = Path("/data/admin/fengshuo/ehr_fm_20261102")
ELIG = PROJECT / "revision_v4_20261122/temporal/results/eligibility"
OUT = ELIG / "status_consistent_v4_1"
PROCESSED = PROJECT / "data/processed"
MIMIC = Path("/data/admin/fengshuo/if10_20261002/shared/raw/mimic")
EICU = Path("/data/admin/fengshuo/if10_20261002/shared/raw/eicu/patient.csv.gz")
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
    if isinstance(value, (np.floating, float)) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def bool_col(frame: pd.DataFrame, name: str, default: bool = False) -> pd.Series:
    return frame[name].fillna(default).astype(bool) if name in frame else pd.Series(default, index=frame.index, dtype=bool)


def explicit_status_conflict(unit: pd.Series, hospital: pd.Series) -> pd.Series:
    u = unit.astype("string").str.strip().str.lower()
    h = hospital.astype("string").str.strip().str.lower()
    unit_expired = u.isin({"expired", "dead", "deceased", "death"}) | u.str.contains("expired|deceased", na=False)
    hospital_alive = h.isin({"alive", "survived", "survivor"}) | h.str.contains("alive|surviv", na=False)
    return (unit_expired & hospital_alive).astype(bool)


def dod_bounds(dod: pd.Series, intime: pd.Series) -> pd.DataFrame:
    day = pd.to_datetime(dod, errors="coerce").dt.normalize()
    start = pd.to_datetime(intime, errors="coerce")
    lower = (day - start).dt.total_seconds() / 3600.0
    upper = (day + pd.Timedelta(days=1) - start).dt.total_seconds() / 3600.0
    known = day.notna() & start.notna() & np.isfinite(lower) & np.isfinite(upper)
    return pd.DataFrame({
        "lower": lower.where(known),
        "upper": upper.where(known),
        "known": known.astype(bool),
        "death_by24": (known & upper.le(24)).astype(bool),
        "preindex": (known & upper.le(0)).astype(bool),
        "overlap24": (known & lower.le(24) & upper.gt(24)).astype(bool),
    }, index=dod.index)


def source_inputs(originals: dict[tuple[str, str], pd.DataFrame], manifests: dict[str, Any]) -> dict[str, Any]:
    print("independent_verify_stage=raw_metadata_read_begin", flush=True)
    mimic_pat_path = MIMIC / "hosp/patients.csv.gz"
    mimic_adm_path = MIMIC / "hosp/admissions.csv.gz"
    mimic_icu_path = MIMIC / "icu/icustays.csv.gz"
    eicu_path = EICU
    mimic_pat = pd.read_csv(mimic_pat_path, usecols=["subject_id", "dod"], compression="infer", low_memory=False)
    mimic_pat["subject_key"] = mimic_pat["subject_id"].map(canonical_id)
    if mimic_pat["subject_key"].isna().any() or mimic_pat["subject_key"].duplicated().any():
        raise ValueError("independent verifier: MIMIC patient DOD keys are invalid")
    mimic_adm = pd.read_csv(mimic_adm_path, usecols=["hadm_id", "admittime"], compression="infer", low_memory=False)
    mimic_adm["hadm_key"] = mimic_adm["hadm_id"].map(canonical_id)
    if mimic_adm["hadm_key"].isna().any() or mimic_adm["hadm_key"].duplicated().any():
        raise ValueError("independent verifier: MIMIC admission keys are invalid")
    mimic_icu = pd.read_csv(mimic_icu_path, usecols=["stay_id", "subject_id", "hadm_id", "intime"], compression="infer", low_memory=False)
    mimic_icu["stay_key"] = mimic_icu["stay_id"].map(canonical_id)
    mimic_icu["subject_key"] = mimic_icu["subject_id"].map(canonical_id)
    mimic_icu["hadm_key"] = mimic_icu["hadm_id"].map(canonical_id)
    if mimic_icu[["stay_key", "subject_key", "hadm_key"]].isna().any().any() or mimic_icu["stay_key"].duplicated().any():
        raise ValueError("independent verifier: MIMIC ICU stay keys are invalid")
    eicu = pd.read_csv(eicu_path, usecols=[
        "patientunitstayid", "uniquepid", "unitdischargestatus", "hospitaldischargestatus", "hospitalid", "hospitaladmitoffset",
    ], compression="infer", low_memory=False)
    eicu["stay_key"] = eicu["patientunitstayid"].map(canonical_id)
    eicu["subject_key"] = eicu["uniquepid"].map(canonical_id)
    eicu["hospitalid_key"] = eicu["hospitalid"].map(canonical_id)
    if eicu[["stay_key", "subject_key", "hospitalid_key"]].isna().any().any() or eicu["stay_key"].duplicated().any():
        raise ValueError("independent verifier: eICU source keys or hospital IDs are invalid")
    raw_hashes = {
        "mimic_patients": sha256(mimic_pat_path),
        "mimic_admissions": sha256(mimic_adm_path),
        "mimic_icustays": sha256(mimic_icu_path),
        "eicu_patient": sha256(eicu_path),
    }
    print("independent_verify_stage=raw_metadata_read_complete", flush=True)
    return {"mimic_pat": mimic_pat, "mimic_adm": mimic_adm, "mimic_icu": mimic_icu, "eicu": eicu, "hashes": raw_hashes}


def assert_unchanged_columns(old: pd.DataFrame, new: pd.DataFrame, db: str, variant: str) -> None:
    for col in ("stay_id", "subject_id", "y_mort", "source_y_mort", "source_y_mort_match", "input_window_le_24h_verified", "input_event_count_24h", "input_max_hour_24h"):
        if col not in old or col not in new:
            if col in {"input_event_count_24h", "input_max_hour_24h"}:
                continue
            raise ValueError(f"{variant}/{db}: unchanged field missing: {col}")
        left, right = old[col].reset_index(drop=True), new[col].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(left) or pd.api.types.is_numeric_dtype(right):
            a = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            b = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            if not np.array_equal(a, b, equal_nan=True):
                raise ValueError(f"{variant}/{db}: {col} changed")
        elif not left.astype("string").fillna("<NA>").equals(right.astype("string").fillna("<NA>")):
            raise ValueError(f"{variant}/{db}: {col} changed")


def verify_table(variant: str, db: str, original: pd.DataFrame, frame: pd.DataFrame,
                 entry: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    selected = bool_col(frame, "eligible_24h")
    if len(original) != len(frame) or frame[["db", "stay_id", "subject_id"]].isna().any().any():
        raise ValueError(f"{variant}/{db}: row/key integrity failed")
    if frame.duplicated(["db", "stay_id", "subject_id"]).any() or not frame["db"].eq(db).all():
        raise ValueError(f"{variant}/{db}: duplicate or incorrect keys")
    if not frame["subject"].astype(str).equals(frame["subject_id"].astype(str)):
        raise ValueError(f"{variant}/{db}: subject id does not match source subject_id")
    assert_unchanged_columns(original, frame, db, variant)

    status_conflict = pd.Series(False, index=frame.index)
    dod = pd.DataFrame({"death_by24": False, "preindex": False, "overlap24": False, "known": False}, index=frame.index)
    admission_offset = pd.Series(np.nan, index=frame.index)
    if db == "eicu":
        source = raw["eicu"].set_index("stay_key")
        keys = frame["stay_id"].map(canonical_id)
        if not keys.isin(source.index).all():
            raise ValueError("eICU independent raw-status join is incomplete")
        raw_rows = source.loc[keys.to_list()].reset_index(drop=True)
        if not raw_rows["subject_key"].astype(str).equals(frame["subject_id"].astype(str)):
            raise ValueError("eICU source uniquepid differs from retained subject_id")
        status_conflict = explicit_status_conflict(raw_rows["unitdischargestatus"], raw_rows["hospitaldischargestatus"])
        for output_col, raw_col in (("source_unitdischargestatus", "unitdischargestatus"),
                                    ("source_hospitaldischargestatus", "hospitaldischargestatus")):
            if not raw_rows[raw_col].astype("string").fillna("<NA>").reset_index(drop=True).equals(
                frame[output_col].astype("string").fillna("<NA>").reset_index(drop=True)):
                raise ValueError(f"eICU retained {output_col} differs from independent raw source")
        raw_hospitalid = raw_rows["hospitalid_key"].astype(str).reset_index(drop=True)
        if not frame["hospitalid_source_raw"].astype(str).reset_index(drop=True).equals(raw_hospitalid):
            raise ValueError("eICU retained raw hospitalid differs from independent source")
        if not frame["hospitalid"].astype(str).reset_index(drop=True).equals(raw_hospitalid):
            raise ValueError("eICU canonical hospitalid does not match source hospitalid")
        if not frame["hospitalid_legacy_snapshot"].astype(str).reset_index(drop=True).equals(raw_hospitalid):
            raise ValueError("eICU legacy/raw hospitalid columns disagree")
        admission_offset = pd.to_numeric(raw_rows["hospitaladmitoffset"], errors="coerce") / 60.0
        expected_endpoint_known = raw_rows["hospitaldischargestatus"].map(
            lambda x: str(x).strip().lower() in {"alive", "survived", "survivor", "expired", "dead", "deceased", "death"}
        )
        expected_source_y = raw_rows["hospitaldischargestatus"].map(
            lambda x: 1.0 if str(x).strip().lower() in {"expired", "dead", "deceased", "death"} else
            (0.0 if str(x).strip().lower() in {"alive", "survived", "survivor"} else np.nan)
        )
        if not expected_endpoint_known.reset_index(drop=True).equals(bool_col(frame, "source_endpoint_known").reset_index(drop=True)):
            raise ValueError("eICU source endpoint known flag differs from raw discharge status")
        if not np.array_equal(
            pd.Series(expected_source_y).to_numpy(dtype=float, na_value=np.nan),
            pd.to_numeric(frame["source_y_mort"], errors="coerce").to_numpy(dtype=float, na_value=np.nan),
            equal_nan=True,
        ):
            raise ValueError("eICU source mortality label differs from raw hospital discharge status")
        if int(status_conflict.sum()) != int(frame["source_unit_expired_hospital_alive_conflict"].sum()):
            raise ValueError("eICU stored status-conflict count differs from independent raw predicate")
        if (selected & status_conflict).any() or (status_conflict & ~bool_col(frame, "source_time_status_conflict")).any():
            raise ValueError("eICU selected unit-expired/hospital-alive row or missing conflict flag")
    else:
        icu = raw["mimic_icu"].set_index("stay_key")
        keys = frame["stay_id"].map(canonical_id)
        if not keys.isin(icu.index).all():
            raise ValueError("MIMIC independent ICU start join is incomplete")
        icu_rows = icu.loc[keys.to_list()].reset_index(drop=True)
        if not icu_rows["subject_key"].astype(str).equals(frame["subject_id"].astype(str)):
            raise ValueError("MIMIC raw subject_id differs from retained subject_id")
        icu_start = pd.to_datetime(icu_rows["intime"], errors="coerce")
        if not icu_start.astype("string").fillna("<NA>").equals(pd.to_datetime(frame["icu_start_time"], errors="coerce").astype("string").fillna("<NA>")):
            raise ValueError("MIMIC retained ICU start differs from raw icustays.intime")
        pat = raw["mimic_pat"].set_index("subject_key")
        subject_keys = frame["subject_id"].map(canonical_id)
        if not subject_keys.isin(pat.index).all():
            raise ValueError("MIMIC independent patients DOD join is incomplete")
        dod_values = pat.loc[subject_keys.to_list(), "dod"].reset_index(drop=True)
        dod = dod_bounds(dod_values, icu_rows["intime"].reset_index(drop=True))
        if not frame["source_patient_dod"].astype("string").fillna("<NA>").reset_index(drop=True).equals(dod_values.astype("string").fillna("<NA>")):
            raise ValueError("MIMIC retained patients.dod differs from raw source")
        mapping = {
            "lower": "source_patient_dod_lower_bound_hr",
            "upper": "source_patient_dod_upper_bound_hr",
            "known": "source_patient_dod_interval_known",
            "death_by24": "source_patient_dod_death_by_24h_proven",
            "preindex": "source_patient_dod_preindex_proven",
            "overlap24": "source_patient_dod_interval_overlaps_24h",
        }
        for key, col in mapping.items():
            if key in {"lower", "upper"}:
                a = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                b = pd.to_numeric(dod[key], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                if not np.allclose(a, b, equal_nan=True, rtol=0, atol=1e-10):
                    raise ValueError(f"MIMIC DOD {key} bounds differ from independent source calculation")
            elif not bool_col(frame, col).reset_index(drop=True).equals(dod[key].astype(bool).reset_index(drop=True)):
                raise ValueError(f"MIMIC DOD {key} flag differs from independent source calculation")
        if (selected & (dod["death_by24"] | dod["preindex"])).any():
            raise ValueError("MIMIC selected proven early-death or pre-index DOD interval")

        adm = raw["mimic_adm"].set_index("hadm_key")
        hadm_keys = frame["hadm_id"].map(canonical_id)
        if not hadm_keys.isin(adm.index).all():
            raise ValueError("MIMIC independent admission-time join is incomplete")
        adm_rows = adm.loc[hadm_keys.to_list()].reset_index(drop=True)
        admission_offset = (pd.to_datetime(adm_rows["admittime"], errors="coerce") - icu_start).dt.total_seconds() / 3600.0

    after_icu = pd.Series(np.isfinite(admission_offset), index=frame.index) & admission_offset.gt(0)
    after24 = pd.Series(np.isfinite(admission_offset), index=frame.index) & admission_offset.gt(24)
    if not after_icu.equals(bool_col(frame, "source_hospital_admit_after_icu_start")):
        raise ValueError(f"{db} hospital-admit-after-ICU flag differs from raw timing")
    if not after24.equals(bool_col(frame, "source_hospital_admit_after_24h")):
        raise ValueError(f"{db} hospital-admit-after-24h flag differs from raw timing")

    original_selected = bool_col(original, "eligible_24h")
    if db == "mimic":
        drop = dod["death_by24"].reset_index(drop=True) | dod["preindex"].reset_index(drop=True)
    else:
        drop = status_conflict.reset_index(drop=True)
    expected = original_selected.reset_index(drop=True) & ~drop
    if not expected.equals(selected.reset_index(drop=True)):
        raise ValueError(f"{variant}/{db}: corrected eligible mask differs from the independent expected mask")

    endpoint_known = bool_col(frame, "source_endpoint_known")
    endpoint_unknown = bool_col(frame, "source_endpoint_unknown", True)
    source_y = pd.to_numeric(frame["source_y_mort"], errors="coerce")
    y = pd.to_numeric(frame["y_mort"], errors="coerce")
    if not endpoint_unknown.equals(~endpoint_known):
        raise ValueError(f"{variant}/{db}: endpoint known/unknown flags disagree")
    if (selected & (~endpoint_known | ~source_y.isin([0, 1]) | ~y.isin([0, 1]) | ~bool_col(frame, "source_y_mort_match") | source_y.ne(y))).any():
        raise ValueError(f"{variant}/{db}: selected source outcome is not known/matching binary")
    icu_exit = pd.to_numeric(frame["icu_exit_hr"], errors="coerce")
    hospital_exit = pd.to_numeric(frame["hospital_exit_hr"], errors="coerce")
    if (selected & (~np.isfinite(icu_exit) | icu_exit.le(24) | ~np.isfinite(hospital_exit) | hospital_exit.le(24))).any():
        raise ValueError(f"{variant}/{db}: selected row lacks finite ICU/hospital exit after 24h")
    exact_hospital_death = pd.to_numeric(frame.get("hospital_death_time_hr", pd.Series(np.nan, index=frame.index)), errors="coerce")
    exact_icu_death = pd.to_numeric(frame.get("icu_death_time_hr", pd.Series(np.nan, index=frame.index)), errors="coerce")
    exact_early_death = (np.isfinite(exact_hospital_death) & exact_hospital_death.le(24)) | (np.isfinite(exact_icu_death) & exact_icu_death.le(24))
    exact_preindex_death = (np.isfinite(exact_hospital_death) & exact_hospital_death.lt(0)) | (np.isfinite(exact_icu_death) & exact_icu_death.lt(0))
    if (selected & (exact_early_death | exact_preindex_death)).any():
        raise ValueError(f"{variant}/{db}: selected row has exact known early/pre-index death time")
    if (selected & (bool_col(frame, "source_landmark_issue") | bool_col(frame, "source_time_status_conflict"))).any():
        raise ValueError(f"{variant}/{db}: selected source timing/status issue")
    if (selected & (~bool_col(frame, "adult_eligible") | ~bool_col(frame, "input_available_24h") | ~bool_col(frame, "input_window_le_24h_verified") | ~bool_col(frame, "temporal_verified") | ~bool_col(frame, "alive_at_24h") | ~bool_col(frame, "in_icu_at_24h"))).any():
        raise ValueError(f"{variant}/{db}: selected row fails adult/input/landmark gate")
    if variant == "strict_v4_1" and (selected & bool_col(frame, "source_postlandmark_ordering_issue")).any():
        raise ValueError("post24_exit_order_sensitivity retains a post-24h exit gap")

    counts = entry.get("counts", {})
    recomputed = {
        "n_rows": int(len(frame)),
        "n_eligible_24h": int(selected.sum()),
        "n_eligible_positive": int((selected & y.eq(1)).sum()),
        "n_eligible_negative": int((selected & y.eq(0)).sum()),
        "reason_counts": {str(k): int(v) for k, v in frame["reason"].value_counts(dropna=False).items()},
        "n_alive_at_24h": int(bool_col(frame, "alive_at_24h").sum()),
        "n_in_icu_at_24h": int(bool_col(frame, "in_icu_at_24h").sum()),
        "hospitalid_coverage": int(frame["hospitalid"].map(canonical_id).notna().sum()),
        "n_source_time_status_conflicts_total": int(bool_col(frame, "source_time_status_conflict").sum()),
        "n_source_time_status_conflicts_eligible": int((selected & bool_col(frame, "source_time_status_conflict")).sum()),
        "n_post24_exit_order_issues": int(bool_col(frame, "source_postlandmark_ordering_issue").sum()),
        "n_post24_exit_order_issues_eligible": int((selected & bool_col(frame, "source_postlandmark_ordering_issue")).sum()),
        "n_hospital_admit_after_icu_start": int(after_icu.sum()),
        "n_hospital_admit_after_icu_start_eligible": int((after_icu & selected).sum()),
        "n_hospital_admit_after_24h": int(after24.sum()),
        "n_hospital_admit_after_24h_eligible": int((after24 & selected).sum()),
    }
    for key, value in recomputed.items():
        if counts.get(key) != value:
            raise ValueError(f"{variant}/{db}: manifest count {key} differs from final table")
    if entry.get("retrospective_landmark_prediction_allowed") is not True or entry.get("ready_for_future24h_clinical_claim") is not False or entry.get("clinical_input_availability_verified") is not False:
        raise ValueError(f"{variant}/{db}: stale or incorrect readiness scope")
    if not frame["clinical_input_availability_verified"].eq(False).all() or frame["ready_for_future24h_clinical_claim"].fillna(True).astype(bool).any():
        raise ValueError(f"{variant}/{db}: clinical availability was overstated")
    if not frame["input_scope"].eq("retrospective_measurement_time").all() or not frame["inputs_temporal_scope"].eq("retrospective_measurement_time").all():
        raise ValueError(f"{variant}/{db}: retrospective scope not set on all rows")
    if "readiness_audit" not in entry or entry["readiness_audit"].get("clinical_input_availability_verified") is not False:
        raise ValueError(f"{variant}/{db}: fresh readiness audit is absent or stale")
    if entry["readiness_audit"].get("eligible_rows_checked") != int(selected.sum()):
        raise ValueError(f"{variant}/{db}: readiness audit does not match final selection")
    if entry.get("eligibility_table_sha256") != sha256(OUT / variant / f"{db}.parquet"):
        raise ValueError(f"{variant}/{db}: final table SHA256 mismatch")
    if entry.get("selected_keys_labels_sha256") != selected_digest(db, frame, selected):
        raise ValueError(f"{variant}/{db}: final selected-key digest mismatch")
    sourcehashes = entry.get("sourcehashes", {})
    if sha256(PROCESSED / f"seq_{db}.npz") != sourcehashes.get("sequence_npz"):
        raise ValueError(f"{variant}/{db}: current source NPZ hash mismatch")
    return {
        "rows": int(len(frame)),
        "eligible_24h": int(selected.sum()),
        "positive": int((selected & y.eq(1)).sum()),
        "negative": int((selected & y.eq(0)).sum()),
        "status_conflict_rows": int(bool_col(frame, "source_time_status_conflict").sum()),
        "explicit_eicu_status_conflicts": int(status_conflict.sum()) if db == "eicu" else 0,
        "dod_proven_early_rows": int((dod["death_by24"] | dod["preindex"]).sum()) if db == "mimic" else 0,
        "dod_overlap_uncertain_rows": int(dod["overlap24"].sum()) if db == "mimic" else 0,
        "post24_exit_order_selected": int((selected & bool_col(frame, "source_postlandmark_ordering_issue")).sum()),
        "hospital_admit_after_icu_start_selected": int((selected & after_icu).sum()),
        "hospital_admit_after_24h_selected": int((selected & after24).sum()),
        "hospitalid_coverage": int(frame["hospitalid"].map(canonical_id).notna().sum()),
        "hospitalid_unique": int(frame["hospitalid"].map(canonical_id).dropna().nunique()),
        "eligibility_table_sha256": sha256(OUT / variant / f"{db}.parquet"),
        "selected_keys_labels_sha256": selected_digest(db, frame, selected),
    }


def selected_digest(db: str, frame: pd.DataFrame, selected: pd.Series) -> str:
    y = pd.to_numeric(frame.loc[selected, "y_mort"], errors="coerce").astype(int)
    material = "\n".join(sorted(
        f"{db}|{sid}|{sub}|{label}"
        for sid, sub, label in zip(frame.loc[selected, "stay_id"].astype(str), frame.loc[selected, "subject"].astype(str), y)
    ))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    originals = {}
    new_frames = {}
    manifests = {}
    for variant in VARIANTS:
        manifests[variant] = json.loads((OUT / variant / "eligibility_manifest.json").read_text(encoding="utf-8"))
        for db in ("mimic", "eicu"):
            print(f"independent_verify_stage={variant}_{db}_table_read_begin", flush=True)
            originals[(variant, db)] = pd.read_parquet(ELIG / variant / f"{db}.parquet")
            new_frames[(variant, db)] = pd.read_parquet(OUT / variant / f"{db}.parquet")
            print(f"independent_verify_stage={variant}_{db}_table_read_complete rows={len(new_frames[(variant, db)])}", flush=True)
    raw = source_inputs(originals, manifests)
    results = {variant: {} for variant in VARIANTS}
    for variant in VARIANTS:
        manifest = manifests[variant]
        if manifest.get("schema_version") != "4.1" or manifest.get("status_consistency_version") != "status_consistent_v4_1":
            raise ValueError(f"{variant} new-version manifest missing")
        expected_role = "primary_retrospective_landmark" if variant == "landmark_v4_1" else "post24_exit_order_sensitivity"
        if manifest.get("variant_role") != expected_role:
            raise ValueError(f"{variant} variant role is stale")
        for db in ("mimic", "eicu"):
            entry = manifest["cohorts"][db]
            results[variant][db] = verify_table(variant, db, originals[(variant, db)], new_frames[(variant, db)], entry, raw)
    for db in ("mimic", "eicu"):
        primary = new_frames[("landmark_v4_1", db)]
        strict = new_frames[("strict_v4_1", db)]
        pkeys = set(primary.loc[bool_col(primary, "eligible_24h"), "stay_id"].astype(str))
        skeys = set(strict.loc[bool_col(strict, "eligible_24h"), "stay_id"].astype(str))
        if not skeys.issubset(pkeys):
            raise ValueError(f"{db}: post24_exit_order_sensitivity is not a primary subset")
        expected_difference = set(primary.loc[
            bool_col(primary, "eligible_24h") & bool_col(primary, "source_postlandmark_ordering_issue"), "stay_id"
        ].astype(str))
        if pkeys - skeys != expected_difference:
            raise ValueError(f"{db}: strict-primary difference is not exactly post-24h exit gaps")
    paths = json.loads((OUT / "current_paths.json").read_text(encoding="utf-8"))
    for variant in VARIANTS:
        if paths["variants"][variant]["manifest_sha256"] != sha256(OUT / variant / "eligibility_manifest.json"):
            raise ValueError(f"{variant}: current path index manifest hash mismatch")
    report = {
        "version": "status_consistent_v4_1",
        "status": "passed",
        "raw_metadata_hashes": raw["hashes"],
        "variants": results,
    }
    atomic_json(OUT / "independent_verification.json", report)
    for variant in VARIANTS:
        for db in ("mimic", "eicu"):
            v = results[variant][db]
            print(
                f"verified {variant} {db}: rows={v['rows']} eligible={v['eligible_24h']} "
                f"positive={v['positive']} negative={v['negative']} status_conflicts={v['status_conflict_rows']} "
                f"eicu_expired_alive={v['explicit_eicu_status_conflicts']} dod_proven={v['dod_proven_early_rows']} "
                f"dod_overlap={v['dod_overlap_uncertain_rows']} selected_exit_gaps={v['post24_exit_order_selected']} "
                f"admit_after_icu={v['hospital_admit_after_icu_start_selected']} admit_after24={v['hospital_admit_after_24h_selected']} "
                f"hospitalid_coverage={v['hospitalid_coverage']} hospitalid_unique={v['hospitalid_unique']} "
                f"table_sha256={v['eligibility_table_sha256']}", flush=True)
        print(f"{variant}_independent_verification=passed", flush=True)


if __name__ == "__main__":
    main()
