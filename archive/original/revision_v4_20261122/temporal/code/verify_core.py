"""Aggregate-only contract and strict-key verification for core cohorts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102/revision_v4_20261122/temporal")
ELIG = ROOT / "results" / "eligibility"
REQUIRED = {
    "db", "stay_id", "subject", "eligible_24h", "temporal_verified", "y_mort", "endpoint", "reason",
    "hospital_death", "icu_death", "input_window_le_24h_verified", "input_availability_cutoff_hr",
    "source_endpoint_known", "source_endpoint_unknown", "source_y_mort", "source_y_mort_match",
    "source_time_issue", "source_time_status_conflict", "source_landmark_contradiction",
    "icu_exit_after_hospital_exit_after_24", "icu_exit_after_hospital_exit_after_24_gap_hr",
    "icu_exit_hr", "hospital_exit_hr", "hospital_death_time_hr",
    "icu_exit_finite_after_24h", "hospital_exit_finite_after_24h",
    "hospital_exit_not_before_icu_exit", "exact_death_time_valid_for_landmark",
    "retrospective_landmark_prediction_allowed", "ready_for_future24h_landmark_analysis",
    "ready_for_retrospective_24h_measurement_analysis", "ready_for_future24h_clinical_claim",
    "inputs_temporal_scope", "input_scope", "clinical_input_availability_verified",
}


def main():
    manifest_path = ELIG / "eligibility_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "4.1":
        raise ValueError("manifest schema_version must be 4.1")
    if not isinstance(manifest.get("cohorts"), dict):
        raise ValueError("manifest lacks cohorts map")
    results = {}
    for db in ("mimic", "eicu"):
        item = manifest["cohorts"].get(db)
        if not isinstance(item, dict):
            raise ValueError(f"manifest cohort missing: {db}")
        frame = pd.read_parquet(ELIG / f"{db}.parquet")
        absent = REQUIRED.difference(frame.columns)
        if absent:
            raise ValueError(f"{db} output missing required columns: {sorted(absent)}")
        if frame[["db", "stay_id", "subject"]].isna().any().any():
            raise ValueError(f"{db} output has null keys")
        if frame.duplicated(["db", "stay_id", "subject"]).any():
            raise ValueError(f"{db} output has duplicate keys")
        if not frame["db"].eq(db).all():
            raise ValueError(f"{db} output contains another db key")
        if "subject_id" in frame and not frame["subject"].eq(frame["subject_id"]).all():
            raise ValueError(f"{db} subject alias mismatch")
        selected = frame["eligible_24h"].fillna(False).astype(bool)
        if (selected & ~frame["temporal_verified"].fillna(False).astype(bool)).any():
            raise ValueError(f"{db} eligible rows are not temporally verified")
        if (selected & ~frame["input_window_le_24h_verified"].fillna(False).astype(bool)).any():
            raise ValueError(f"{db} eligible rows violate input cutoff")
        if (selected & ~frame["source_endpoint_known"].fillna(False).astype(bool)).any():
            raise ValueError(f"{db} selected row has an unknown source endpoint")
        if (selected & ~frame["source_y_mort_match"].fillna(False).astype(bool)).any():
            raise ValueError(f"{db} selected row source label is unknown or mismatched")
        icu_exit = pd.to_numeric(frame["icu_exit_hr"], errors="coerce")
        hospital_exit = pd.to_numeric(frame["hospital_exit_hr"], errors="coerce")
        death_time = pd.to_numeric(frame["hospital_death_time_hr"], errors="coerce")
        finite_icu_exit = pd.Series(np.isfinite(icu_exit), index=frame.index)
        finite_hospital_exit = pd.Series(np.isfinite(hospital_exit), index=frame.index)
        finite_death_time = pd.Series(np.isfinite(death_time), index=frame.index)
        endpoint_known = frame["source_endpoint_known"].fillna(False).astype(bool)
        endpoint_unknown = frame["source_endpoint_unknown"].fillna(True).astype(bool)
        if not endpoint_unknown.equals(~endpoint_known):
            raise ValueError(f"{db} endpoint known/unknown flags disagree")
        if not frame["icu_exit_finite_after_24h"].fillna(False).astype(bool).equals(finite_icu_exit & icu_exit.gt(24)):
            raise ValueError(f"{db} ICU exit landmark flags disagree with source times")
        if not frame["hospital_exit_finite_after_24h"].fillna(False).astype(bool).equals(finite_hospital_exit & hospital_exit.gt(24)):
            raise ValueError(f"{db} hospital exit landmark flags disagree with source times")
        if not frame["hospital_exit_not_before_icu_exit"].fillna(False).astype(bool).equals(finite_hospital_exit & finite_icu_exit & hospital_exit.ge(icu_exit)):
            raise ValueError(f"{db} hospital/ICU exit ordering flags disagree with source times")
        expected_death_valid = ~finite_death_time | (death_time.gt(24) & finite_hospital_exit & death_time.le(hospital_exit))
        if not frame["exact_death_time_valid_for_landmark"].fillna(False).astype(bool).equals(expected_death_valid):
            raise ValueError(f"{db} exact death-time validity flags disagree with source times")
        expected_contradiction = finite_death_time & death_time.le(24) & finite_icu_exit & icu_exit.gt(24)
        if not frame["source_landmark_contradiction"].fillna(False).astype(bool).equals(expected_contradiction):
            raise ValueError(f"{db} early-death landmark contradiction flags disagree with source times")
        if (selected & (~pd.Series(np.isfinite(icu_exit), index=frame.index) | icu_exit.le(24))).any():
            raise ValueError(f"{db} selected row lacks finite ICU exit after 24h")
        if (selected & (~pd.Series(np.isfinite(hospital_exit), index=frame.index) | hospital_exit.le(24))).any():
            raise ValueError(f"{db} selected row lacks finite hospital exit after 24h")
        if (selected & hospital_exit.lt(icu_exit)).any():
            raise ValueError(f"{db} selected row has hospital exit before ICU exit")
        if (selected & death_time.notna() & (death_time.le(24) | death_time.gt(hospital_exit))).any():
            raise ValueError(f"{db} selected row has an invalid exact death time")
        gap = frame["icu_exit_after_hospital_exit_after_24"].fillna(False).astype(bool)
        expected_gap = (icu_exit.gt(hospital_exit) & hospital_exit.gt(24)).fillna(False).astype(bool)
        if not expected_gap.equals(gap):
            raise ValueError(f"{db} post-24h ICU-after-hospital gap flags disagree with source timing")
        gap_hours = pd.to_numeric(frame["icu_exit_after_hospital_exit_after_24_gap_hr"], errors="coerce")
        expected_gap_hours = (icu_exit - hospital_exit).where(gap)
        if (gap & (~np.isfinite(gap_hours) | ~np.isclose(gap_hours, expected_gap_hours, equal_nan=False))).any():
            raise ValueError(f"{db} post-24h gap durations disagree with source timing")
        gap_values = gap_hours[gap]
        if len(gap_values):
            q = np.quantile(gap_values.to_numpy(dtype=float), [0.05, 0.5, 0.95])
            gap_quantiles = {"p05": float(q[0]), "median": float(q[1]), "p95": float(q[2])}
        else:
            gap_quantiles = {"p05": None, "median": None, "p95": None}
        source_labels = pd.to_numeric(frame["source_y_mort"], errors="coerce")
        gap_labels = {
            "hospital_death_1": int((gap & source_labels.eq(1)).sum()),
            "hospital_survival_0": int((gap & source_labels.eq(0)).sum()),
            "unknown_or_nonbinary": int((gap & ~source_labels.isin([0, 1])).sum()),
        }
        if item.get("counts", {}).get("post24_icu_after_hospital_exit_gap_rows") != int(gap.sum()):
            raise ValueError(f"{db} post-24h gap count disagrees with manifest")
        declared_gap_quantiles = item.get("counts", {}).get("post24_icu_after_hospital_exit_gap_duration_hr_quantiles", {})
        if not isinstance(declared_gap_quantiles, dict) or any(
            (gap_quantiles[k] is None) != (declared_gap_quantiles.get(k) is None)
            or (gap_quantiles[k] is not None and not np.isclose(gap_quantiles[k], declared_gap_quantiles.get(k), rtol=1e-10, atol=1e-10))
            for k in ("p05", "median", "p95")
        ):
            raise ValueError(f"{db} post-24h gap duration quantiles disagree with manifest")
        if item.get("counts", {}).get("post24_icu_after_hospital_exit_gap_source_label_distribution") != gap_labels:
            raise ValueError(f"{db} post-24h gap label distribution disagrees with manifest")
        if not frame["inputs_temporal_scope"].eq("retrospective_measurement_time").all() or not frame["input_scope"].eq("retrospective_measurement_time").all():
            raise ValueError(f"{db} temporal scope is not retrospective measurement time")
        if frame["clinical_input_availability_verified"].fillna(True).astype(bool).any():
            raise ValueError(f"{db} clinical input availability was not independently verified")
        if frame["ready_for_future24h_clinical_claim"].fillna(True).astype(bool).any():
            raise ValueError(f"{db} claims clinical readiness without real-time availability evidence")
        if not frame["ready_for_retrospective_24h_measurement_analysis"].fillna(False).astype(bool).all():
            raise ValueError(f"{db} retrospective measurement readiness is not attested on every row")
        if not frame["retrospective_landmark_prediction_allowed"].fillna(False).astype(bool).all():
            raise ValueError(f"{db} primary retrospective prediction attestation is not true on every row")
        if not frame["ready_for_future24h_landmark_analysis"].fillna(False).astype(bool).all():
            raise ValueError(f"{db} landmark readiness alias disagrees with the primary attestation")
        if not (pd.to_numeric(frame["input_availability_cutoff_hr"], errors="coerce") == 24).all():
            raise ValueError(f"{db} input cutoff is not 24h")
        if item.get("source_join_coverage") != 1.0:
            raise ValueError(f"{db} source join is not complete")
        if item.get("counts", {}).get("n_rows") != len(frame):
            raise ValueError(f"{db} manifest row count differs")
        if item.get("evidence") not in {"exact", "operational_proxy", "unavailable"}:
            raise ValueError(f"{db} evidence level invalid")
        if item.get("ready_for_future24h_clinical_claim") is not False:
            raise ValueError(f"{db} claims future clinical readiness without verified input availability")
        if item.get("ready_for_retrospective_24h_measurement_analysis") is not True:
            raise ValueError(f"{db} retrospective measurement readiness is not attested")
        if item.get("retrospective_landmark_prediction_allowed") is not True:
            raise ValueError(f"{db} manifest lacks the primary retrospective prediction attestation")
        if item.get("ready_for_future24h_landmark_analysis") is not True:
            raise ValueError(f"{db} landmark readiness alias is not true")
        if item.get("inputs_temporal_scope") != "retrospective_measurement_time" or item.get("input_scope") != "retrospective_measurement_time" or item.get("clinical_input_availability_verified") is not False:
            raise ValueError(f"{db} manifest temporal availability scope is incomplete")
        declared_conflicts = int(item.get("source_metadata_evidence", {}).get("source_time_status_conflicts", -1))
        actual_conflicts = int(frame["source_time_status_conflict"].fillna(False).sum())
        if declared_conflicts != actual_conflicts:
            raise ValueError(f"{db} source status-conflict count disagrees with metadata audit")
        if "source_time_issue" in frame:
            issue = frame["source_time_issue"].fillna(False).astype(bool)
        else:
            issue = frame["reason"].eq("impossible_source_time")
        label_bad = frame["source_y_mort_match"].eq(False).fillna(False)
        endpoint_unknown = frame["source_endpoint_unknown"].fillna(True).astype(bool)
        n_issue_eligible = int((issue & selected).sum())
        n_label_bad_eligible = int((label_bad & selected).sum())
        n_endpoint_unknown_eligible = int((endpoint_unknown & selected).sum())
        n_bad_input_eligible = int(((~frame["input_window_le_24h_verified"].fillna(False)) & selected).sum())
        if n_issue_eligible or n_label_bad_eligible or n_endpoint_unknown_eligible or n_bad_input_eligible:
            raise ValueError(f"{db} has selected rows with a timing/label/input audit failure")
        if int(label_bad.sum()) != int(item.get("counts", {}).get("n_source_label_mismatch_rows", -1)):
            raise ValueError(f"{db} source label mismatch count disagrees with manifest")
        if int(label_bad.sum()) != 0:
            raise ValueError(f"{db} has source/current mortality label mismatches")
        hashes = item.get("sourcehashes", {})
        if not hashes or not all(len(str(v)) == 64 for v in hashes.values()):
            raise ValueError(f"{db} source hashes invalid")
        results[db] = {
            "schema_columns": len(frame.columns),
            "rows": len(frame),
            "unique_keys": int(frame[["db", "stay_id", "subject"]].drop_duplicates().shape[0]),
            "source_join_coverage": item["source_join_coverage"],
            "eligible_24h": int(selected.sum()),
            "temporal_verified": int(frame["temporal_verified"].sum()),
            "evidence": item["evidence"],
            "ready_for_future24h_clinical_claim": item["ready_for_future24h_clinical_claim"],
            "retrospective_landmark_prediction_allowed": item["retrospective_landmark_prediction_allowed"],
            "ready_for_future24h_landmark_analysis": item["ready_for_future24h_landmark_analysis"],
            "ready_for_retrospective_24h_measurement_analysis": item["ready_for_retrospective_24h_measurement_analysis"],
            "inputs_temporal_scope": item["inputs_temporal_scope"],
            "input_scope": item["input_scope"],
            "clinical_input_availability_verified": item["clinical_input_availability_verified"],
            "hash_count": len(hashes),
            "source_time_issue_rows": int(issue.sum()),
            "source_time_issue_eligible": n_issue_eligible,
            "source_time_status_conflict_rows": int(frame["source_time_status_conflict"].fillna(False).sum()),
            "source_label_mismatch_rows": int(label_bad.sum()),
            "source_label_mismatch_eligible": n_label_bad_eligible,
            "source_endpoint_unknown_rows": int(endpoint_unknown.sum()),
            "source_endpoint_unknown_eligible": n_endpoint_unknown_eligible,
            "source_landmark_contradictions": int(frame["source_landmark_contradiction"].fillna(False).sum()),
            "icu_exit_after_hospital_exit_after_24": int(frame["icu_exit_after_hospital_exit_after_24"].fillna(False).sum()),
            "post24_icu_after_hospital_exit_gap_duration_hr_quantiles": gap_quantiles,
            "post24_icu_after_hospital_exit_gap_source_label_distribution": gap_labels,
            "negative_hour_input_rows": int((~frame["input_window_le_24h_verified"].fillna(False)).sum()),
            "negative_hour_input_eligible": n_bad_input_eligible,
        }
    report = {"schema_version": "4.1", "status": "passed", "cohorts": results}
    report_path = ELIG / "core_verification.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    for db, value in results.items():
        print(f"{db}: rows={value['rows']} unique={value['unique_keys']} eligible24={value['eligible_24h']} temporal={value['temporal_verified']} evidence={value['evidence']} ready={value['ready_for_future24h_clinical_claim']}", flush=True)
    print("contract_verification=passed", flush=True)


if __name__ == "__main__":
    main()
