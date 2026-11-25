#!/usr/bin/env python3
"""Qualify Challenge 2012 set A from cached records and its official outcome file.

The study's published inclusion rule requires an adult ICU stay of at least
48 hours. This script records that as a lower-bound proof and never invents an
exact ICU/hospital exit or death timestamp. It is aggregate-only unless run
with --write, which writes a separate cohort directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT = Path("/data/admin/fengshuo/ehr_fm_20261102")
TEMPORAL = PROJECT / "revision_v4_20261122/temporal"
INTERIM = PROJECT / "data/interim"
PROCESSED = PROJECT / "data/processed"
CHALLENGE = PROJECT / "data/external/challenge2012"
OUT_DIR = TEMPORAL / "results/eligibility/c2012_a_landmark_v4_1"
INCLUSION_RULE = "PhysioNet Challenge 2012 set A inclusion: adult ICU stays of at least 48 hours"


def key(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)) and math.isfinite(float(value)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


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


def unique(frame: pd.DataFrame, cols: list[str], name: str) -> None:
    if frame[cols].isna().any().any() or frame.duplicated(cols).any():
        raise ValueError(f"{name} has null or duplicate keys")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write the separate c2012_a_landmark_v4_1 directory")
    args = parser.parse_args()
    cohort_path = INTERIM / "cohort_c2012.parquet"
    sequence_path = PROCESSED / "seq_c2012.npz"
    outcome_path = CHALLENGE / "outcomes-a.txt"

    cohort = pd.read_parquet(cohort_path)
    if not {"stay_id", "subject_id", "age", "mort_hosp"}.issubset(cohort.columns):
        raise ValueError("C2012 cached cohort lacks required keys, age, or current mortality label")
    cohort = cohort.copy()
    cohort["stay_id"] = cohort["stay_id"].map(key)
    cohort["subject_id"] = cohort["subject_id"].map(key)
    unique(cohort, ["stay_id", "subject_id"], "C2012 cohort")

    with np.load(sequence_path, allow_pickle=False) as seq:
        required = {"stay_id", "subject", "L", "H", "y_mort"}
        if not required.issubset(seq.files):
            raise ValueError(f"C2012 sequence NPZ lacks {sorted(required.difference(seq.files))}")
        lengths = np.asarray(seq["L"], dtype=np.int64)
        hours = np.asarray(seq["H"])
        seq_frame = pd.DataFrame({
            "stay_id": [key(x) for x in seq["stay_id"]],
            "subject_id": [key(x) for x in seq["subject"]],
            "sequence_y_mort": np.asarray(seq["y_mort"], dtype=float),
            "input_event_count_24h": lengths,
            "input_window_le_24h_verified": [
                bool(np.all(np.isfinite(hours[i, :int(length)])
                            & (hours[i, :int(length)] >= 0)
                            & (hours[i, :int(length)] < 24)))
                for i, length in enumerate(lengths)
            ],
        })
    unique(seq_frame, ["stay_id", "subject_id"], "C2012 sequence")
    if len(seq_frame) != len(cohort):
        raise ValueError("C2012 cohort/sequence row counts differ")
    frame = cohort.merge(seq_frame, on=["stay_id", "subject_id"], how="left", validate="one_to_one", indicator="_sequence_join")
    if not frame["_sequence_join"].eq("both").all():
        raise ValueError("C2012 sequence key join is incomplete")
    frame = frame.drop(columns="_sequence_join")

    outcome = pd.read_csv(outcome_path, usecols=["RecordID", "In-hospital_death"])
    outcome = outcome.rename(columns={"RecordID": "stay_id", "In-hospital_death": "source_y_mort"})
    outcome["stay_id"] = outcome["stay_id"].map(key)
    outcome["subject_id"] = outcome["stay_id"]
    unique(outcome, ["stay_id", "subject_id"], "C2012 outcomes-a")
    frame = frame.merge(outcome, on=["stay_id", "subject_id"], how="left", validate="one_to_one", indicator="_outcome_join")
    coverage = float(frame["_outcome_join"].eq("both").mean()) if len(frame) else 0.0
    if coverage != 1.0:
        raise ValueError(f"C2012 outcomes-a key coverage is {coverage}, expected 1.0")
    frame = frame.drop(columns="_outcome_join")

    current_y = pd.to_numeric(frame["mort_hosp"], errors="coerce")
    sequence_y = pd.to_numeric(frame["sequence_y_mort"], errors="coerce")
    source_y = pd.to_numeric(frame["source_y_mort"], errors="coerce")
    known = source_y.isin([0, 1])
    label_match = known & current_y.isin([0, 1]) & sequence_y.isin([0, 1]) & current_y.eq(source_y) & sequence_y.eq(source_y)
    adult = pd.to_numeric(frame["age"], errors="coerce").ge(18)
    input_ok = frame["input_window_le_24h_verified"].fillna(False).astype(bool) & pd.to_numeric(frame["input_event_count_24h"], errors="coerce").gt(0)
    selected = (adult & label_match & input_ok).fillna(False).astype(bool)

    frame["db"] = "c2012"
    frame["subject"] = frame["subject_id"].astype(str)
    frame["y_mort"] = current_y
    frame["endpoint"] = "hospital"
    frame["source_y_mort"] = source_y
    frame["source_endpoint_known"] = known.astype(bool)
    frame["source_endpoint_unknown"] = (~known).astype(bool)
    frame["source_y_mort_match"] = label_match.astype(bool)
    frame["adult_eligible"] = adult.fillna(False).astype(bool)
    frame["eligible_24h"] = selected
    frame["temporal_verified"] = selected
    frame["reason"] = np.where(selected, "eligible_24h", np.where(~adult.fillna(False), "adult_eligibility_unavailable_or_under_18", np.where(~known, "source_endpoint_unknown", np.where(~label_match, "source_outcome_label_mismatch", "input_window_not_verified_le_24h"))))
    frame["input_available_24h"] = pd.to_numeric(frame["input_event_count_24h"], errors="coerce").gt(0)
    frame["input_availability_cutoff_hr"] = 24.0
    frame["input_window_start_hr"] = 0.0
    frame["input_window_end_hr"] = 24.0
    frame["input_scope"] = "retrospective_measurement_time"
    frame["inputs_temporal_scope"] = "retrospective_measurement_time"
    frame["clinical_input_availability_verified"] = False
    frame["retrospective_landmark_prediction_allowed"] = True
    frame["ready_for_future24h_landmark_analysis"] = True
    frame["ready_for_retrospective_24h_measurement_analysis"] = True
    frame["ready_for_future24h_clinical_claim"] = False
    frame["official_study_minimum_48h_proof"] = True
    frame["landmark_lower_bound_hr"] = 48.0
    frame["landmark_lower_bound_kind"] = "official_study_minimum_48h"
    frame["icu_exit_time"] = pd.NA
    frame["icu_exit_hr"] = np.nan
    frame["icu_exit_time_kind"] = "unavailable_exact_time_official_48h_lower_bound_only"
    frame["hospital_exit_time"] = pd.NA
    frame["hospital_exit_hr"] = np.nan
    frame["hospital_death"] = pd.array(source_y.eq(1).where(known), dtype="boolean")
    frame["hospital_death_time"] = pd.NA
    frame["hospital_death_time_hr"] = np.nan
    frame["hospital_death_time_kind"] = "unavailable_outcomes_a_has_no_death_timestamp"
    frame["alive_at_24h"] = pd.array([True] * len(frame), dtype="boolean")
    frame["in_icu_at_24h"] = pd.array([True] * len(frame), dtype="boolean")
    frame["source_landmark_issue"] = False
    frame["source_postlandmark_ordering_issue"] = False
    frame["source_time_issue"] = False
    frame["source_time_status_conflict"] = False

    labels = current_y.loc[selected]
    selected_hash = hashlib.sha256("\n".join(sorted(
        f"c2012|{stay}|{subject}|{int(label)}"
        for stay, subject, label in zip(frame.loc[selected, "stay_id"].astype(str),
                                       frame.loc[selected, "subject"].astype(str), labels.astype(int))
    )).encode("utf-8")).hexdigest()
    source_hashes = {
        "cohort_parquet": sha256(cohort_path),
        "sequence_npz": sha256(sequence_path),
        "outcomes-a.txt": sha256(outcome_path),
    }
    summary = {
        "n_rows": int(len(frame)),
        "n_eligible_24h": int(selected.sum()),
        "n_eligible_positive": int(labels.eq(1).sum()),
        "n_eligible_negative": int(labels.eq(0).sum()),
        "n_adult_eligible": int(adult.fillna(False).sum()),
        "n_source_endpoint_unknown": int((~known).sum()),
        "n_source_label_mismatch": int((known & ~label_match).sum()),
        "n_input_window_invalid": int((~frame["input_window_le_24h_verified"].fillna(False)).sum()),
        "source_join_coverage": coverage,
        "selected_keys_labels_sha256": selected_hash,
        "exact_icu_exit_timestamp_available": False,
        "icu_exit_lower_bound_hr": 48.0,
        "lower_bound_evidence": INCLUSION_RULE,
        "patient_disjoint_claim_allowed": False,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    if not args.write:
        print("qualification_only=true; no cohort files written", flush=True)
        return
    if OUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite existing separate C2012 A output: {OUT_DIR}")
    OUT_DIR.mkdir(parents=True)
    table_path = OUT_DIR / "c2012.parquet"
    tmp = table_path.with_name(table_path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, table_path)
    manifest = {
        "schema_version": "4.1",
        "variant": "c2012_a_landmark_v4_1",
        "derivation": "Challenge 2012 set A official >=48h ICU-stay inclusion proof; no exact exit/death timestamps inferred",
        "cohorts": {
            "c2012": {
                "db": "c2012",
                "status": "study_inclusion_lower_bound_48h",
                "evidence": "operational_proxy",
                "temporal_evidence_kind": "official_study_minimum_48h",
                "endpoint": "hospital",
                "source_join_coverage": coverage,
                "sourcehashes": source_hashes,
                "sourcefiles": ["outcomes-a.txt"],
                "source_metadata_evidence": {
                    "hospital_death_source": "outcomes-a.txt In-hospital_death",
                    "icu_exit_source": "official study inclusion lower bound; no exact timestamp",
                    "icu_exit_lower_bound_hr": 48.0,
                    "inclusion_rule": INCLUSION_RULE,
                    "exact_death_time_available": False,
                },
                "subject_identity_basis": {
                    "basis": "RecordID used as admission/record proxy; no verified longitudinal person identifier",
                    "patient_disjoint_claim_allowed": False,
                    "bootstrap_unit": "record_id",
                },
                "counts": {**summary, "n_rows": int(len(frame))},
                "retrospective_landmark_prediction_allowed": True,
                "ready_for_future24h_landmark_analysis": True,
                "ready_for_retrospective_24h_measurement_analysis": True,
                "ready_for_future24h_clinical_claim": False,
                "inputs_temporal_scope": "retrospective_measurement_time",
                "input_scope": "retrospective_measurement_time",
                "clinical_input_availability_verified": False,
                "eligibility_table_sha256": sha256(table_path),
                "selected_keys_labels_sha256": selected_hash,
                "output": str(table_path),
            }
        },
    }
    atomic_json(OUT_DIR / "eligibility_manifest.json", manifest)
    print(f"separate_cohort_written={OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
