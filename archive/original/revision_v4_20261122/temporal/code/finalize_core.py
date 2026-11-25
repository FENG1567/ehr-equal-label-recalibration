"""Attest retrospective landmark readiness from audited selected rows only."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102/revision_v4_20261122/temporal")
ELIG = ROOT / "results" / "eligibility"
EXPECTED = {"mimic": "exact", "eicu": "operational_proxy"}


def atomic_json(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def audit_cohort(db: str, meta: dict) -> tuple[pd.DataFrame, dict]:
    path = ELIG / f"{db}.parquet"
    frame = pd.read_parquet(path)
    required = {
        "db", "stay_id", "subject", "eligible_24h", "temporal_verified",
        "input_window_le_24h_verified", "reason", "source_endpoint_known",
        "source_endpoint_unknown", "source_y_mort", "source_y_mort_match", "source_time_issue",
        "source_time_status_conflict", "source_landmark_contradiction",
        "icu_exit_after_hospital_exit_after_24", "icu_exit_after_hospital_exit_after_24_gap_hr",
        "icu_exit_hr", "hospital_exit_hr", "hospital_death_time_hr",
        "icu_exit_finite_after_24h", "hospital_exit_finite_after_24h",
        "hospital_exit_not_before_icu_exit", "exact_death_time_valid_for_landmark",
        "inputs_temporal_scope",
        "input_scope", "clinical_input_availability_verified",
        "retrospective_landmark_prediction_allowed", "ready_for_future24h_landmark_analysis",
        "ready_for_retrospective_24h_measurement_analysis",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"{db} required readiness fields missing")
    if frame[["db", "stay_id", "subject"]].isna().any().any() or frame.duplicated(["db", "stay_id", "subject"]).any():
        raise ValueError(f"{db} key audit failed")
    issue = frame["source_time_issue"].fillna(False).astype(bool)
    status_conflict = frame["source_time_status_conflict"].fillna(False).astype(bool)
    contradiction = frame["source_landmark_contradiction"].fillna(False).astype(bool)
    exit_order_gap = frame["icu_exit_after_hospital_exit_after_24"].fillna(False).astype(bool)
    label_bad = frame.get("source_y_mort_match", pd.Series(True, index=frame.index)).eq(False).fillna(False)
    endpoint_unknown = frame["source_endpoint_unknown"].fillna(True).astype(bool)
    input_bad = ~frame["input_window_le_24h_verified"].fillna(False).astype(bool)
    selected = frame["eligible_24h"].fillna(False).astype(bool)
    selected_time_issues = int((issue & selected).sum())
    selected_status_conflicts = int((status_conflict & selected).sum())
    selected_unverified_time = int((selected & ~frame["temporal_verified"].fillna(False).astype(bool)).sum())
    declared_status_conflicts = int(meta.get("source_metadata_evidence", {}).get("source_time_status_conflicts", 0))
    if int(status_conflict.sum()) != declared_status_conflicts:
        raise ValueError(f"{db} status-conflict count disagrees with source metadata audit")
    selected_label_mismatches = int((label_bad & selected).sum())
    selected_endpoint_unknown = int((endpoint_unknown & selected).sum())
    selected_input_issues = int((input_bad & selected).sum())
    total_label_mismatches = int(label_bad.sum())
    selected_missing_outcomes = int((selected & pd.to_numeric(frame["y_mort"], errors="coerce").isna()).sum())
    adult = frame.get("adult_eligible", pd.Series(pd.NA, index=frame.index))
    selected_adult_unknown_or_false = int((selected & ~adult.fillna(False).astype(bool)).sum())
    cutoff = pd.to_numeric(frame.get("input_availability_cutoff_hr", pd.Series(float("nan"), index=frame.index)), errors="coerce")
    selected_bad_cutoff = int((selected & cutoff.gt(24)).sum())
    icu_exit = pd.to_numeric(frame["icu_exit_hr"], errors="coerce")
    hospital_exit = pd.to_numeric(frame["hospital_exit_hr"], errors="coerce")
    death_time = pd.to_numeric(frame["hospital_death_time_hr"], errors="coerce")
    finite_icu_exit = pd.Series(np.isfinite(icu_exit), index=frame.index)
    finite_hospital_exit = pd.Series(np.isfinite(hospital_exit), index=frame.index)
    finite_death_time = pd.Series(np.isfinite(death_time), index=frame.index)
    if not frame["source_endpoint_unknown"].fillna(True).astype(bool).equals(~frame["source_endpoint_known"].fillna(False).astype(bool)):
        raise ValueError(f"{db} source endpoint known/unknown flags disagree")
    if not frame["icu_exit_finite_after_24h"].fillna(False).astype(bool).equals(finite_icu_exit & icu_exit.gt(24)):
        raise ValueError(f"{db} ICU exit landmark flags disagree with source times")
    if not frame["hospital_exit_finite_after_24h"].fillna(False).astype(bool).equals(finite_hospital_exit & hospital_exit.gt(24)):
        raise ValueError(f"{db} hospital exit landmark flags disagree with source times")
    if not frame["hospital_exit_not_before_icu_exit"].fillna(False).astype(bool).equals(finite_hospital_exit & finite_icu_exit & hospital_exit.ge(icu_exit)):
        raise ValueError(f"{db} hospital/ICU exit ordering flags disagree with source times")
    expected_exact_death_valid = ~finite_death_time | (death_time.gt(24) & finite_hospital_exit & death_time.le(hospital_exit))
    if not frame["exact_death_time_valid_for_landmark"].fillna(False).astype(bool).equals(expected_exact_death_valid):
        raise ValueError(f"{db} exact death-time validity flags disagree with source times")
    expected_landmark_contradiction = finite_death_time & death_time.le(24) & finite_icu_exit & icu_exit.gt(24)
    if not frame["source_landmark_contradiction"].fillna(False).astype(bool).equals(expected_landmark_contradiction):
        raise ValueError(f"{db} early-death landmark contradiction flags disagree with source times")
    selected_bad_icu_exit = int((selected & (~pd.Series(np.isfinite(icu_exit), index=frame.index) | icu_exit.le(24))).sum())
    selected_bad_hospital_exit = int((selected & (~pd.Series(np.isfinite(hospital_exit), index=frame.index) | hospital_exit.le(24))).sum())
    selected_bad_exit_order = int((selected & hospital_exit.lt(icu_exit)).sum())
    selected_bad_death_time = int((selected & death_time.notna() & (death_time.le(24) | death_time.gt(hospital_exit))).sum())
    expected_gap = icu_exit.gt(hospital_exit) & hospital_exit.gt(24)
    if not expected_gap.fillna(False).equals(exit_order_gap):
        raise ValueError(f"{db} preserved post-24h ICU-after-hospital gap flags disagree with source times")
    gap_hours = pd.to_numeric(frame["icu_exit_after_hospital_exit_after_24_gap_hr"], errors="coerce")
    expected_gap_hours = (icu_exit - hospital_exit).where(exit_order_gap)
    if (exit_order_gap & (~np.isfinite(gap_hours) | ~np.isclose(gap_hours, expected_gap_hours, equal_nan=False))).any():
        raise ValueError(f"{db} preserved post-24h exit gap durations disagree with source times")
    finite_gap_hours = gap_hours[exit_order_gap]
    if len(finite_gap_hours):
        gap_quantiles = np.quantile(finite_gap_hours.to_numpy(dtype=float), [0.05, 0.5, 0.95])
        gap_duration_summary = {"p05": float(gap_quantiles[0]), "median": float(gap_quantiles[1]), "p95": float(gap_quantiles[2])}
    else:
        gap_duration_summary = {"p05": None, "median": None, "p95": None}
    source_labels = pd.to_numeric(frame["source_y_mort"], errors="coerce")
    gap_label_distribution = {
        "hospital_death_1": int((exit_order_gap & source_labels.eq(1)).sum()),
        "hospital_survival_0": int((exit_order_gap & source_labels.eq(0)).sum()),
        "unknown_or_nonbinary": int((exit_order_gap & ~source_labels.isin([0, 1])).sum()),
    }
    manifest_gap_summary = {
        "post24_icu_after_hospital_exit_gap_rows": int(exit_order_gap.sum()),
        "post24_icu_after_hospital_exit_gap_duration_hr_quantiles": gap_duration_summary,
        "post24_icu_after_hospital_exit_gap_source_label_distribution": gap_label_distribution,
    }
    for key, value in manifest_gap_summary.items():
        if meta.get("counts", {}).get(key) != value:
            if key.endswith("quantiles"):
                old = meta.get("counts", {}).get(key, {})
                if not isinstance(old, dict) or any(
                    (value[q] is None) != (old.get(q) is None)
                    or (value[q] is not None and not np.isclose(value[q], old.get(q), rtol=1e-10, atol=1e-10))
                    for q in ("p05", "median", "p95")
                ):
                    raise ValueError(f"{db} post-24h gap quantiles disagree with manifest")
            else:
                raise ValueError(f"{db} {key} disagrees with manifest")
    if not frame["inputs_temporal_scope"].eq("retrospective_measurement_time").all() or not frame["input_scope"].eq("retrospective_measurement_time").all():
        raise ValueError(f"{db} input temporal scope is not explicit retrospective measurement time")
    if frame["clinical_input_availability_verified"].fillna(True).astype(bool).any():
        raise ValueError(f"{db} marks clinical input availability verified without raw-input lag evidence")
    join_coverage = meta.get("source_join_coverage")
    ready = (
        meta.get("evidence") == EXPECTED[db]
        and join_coverage == 1.0
        and int(selected.sum()) > 0
        and selected_time_issues == 0
        and selected_status_conflicts == 0
        and selected_unverified_time == 0
        and selected_label_mismatches == 0
        and selected_endpoint_unknown == 0
        and selected_input_issues == 0
        and selected_missing_outcomes == 0
        and selected_adult_unknown_or_false == 0
        and selected_bad_cutoff == 0
        and selected_bad_icu_exit == 0
        and selected_bad_hospital_exit == 0
        and selected_bad_exit_order == 0
        and selected_bad_death_time == 0
        and total_label_mismatches == 0
        and frame["retrospective_landmark_prediction_allowed"].fillna(False).astype(bool).all()
        and frame["ready_for_future24h_landmark_analysis"].fillna(False).astype(bool).all()
    )
    if not ready:
        raise ValueError(f"{db} selected-row readiness checks failed")
    frame["source_time_issue"] = issue.astype(bool)
    frame["source_time_status_conflict"] = status_conflict.astype(bool)
    frame["retrospective_landmark_prediction_allowed"] = ready
    frame["ready_for_future24h_landmark_analysis"] = ready
    frame["ready_for_retrospective_24h_measurement_analysis"] = ready
    frame["ready_for_future24h_clinical_claim"] = False
    frame["clinical_input_availability_verified"] = False
    audit = {
        "eligible_rows_checked": int(selected.sum()),
        "source_time_issue_rows_total": int(issue.sum()),
        "source_time_status_conflicts_total": declared_status_conflicts,
        "source_time_issue_rows_selected": selected_time_issues,
        "source_time_status_conflicts_selected": selected_status_conflicts,
        "source_landmark_contradictions_total": int(contradiction.sum()),
        "source_landmark_contradictions_selected": int((contradiction & selected).sum()),
        "icu_exit_after_hospital_exit_after_24_total": int(exit_order_gap.sum()),
        "icu_exit_after_hospital_exit_after_24_selected": int((exit_order_gap & selected).sum()),
        **manifest_gap_summary,
        "selected_rows_temporal_unverified": selected_unverified_time,
        "source_label_mismatch_rows_total": total_label_mismatches,
        "source_label_mismatch_rows_selected": selected_label_mismatches,
        "source_endpoint_unknown_rows_total": int(endpoint_unknown.sum()),
        "source_endpoint_unknown_rows_selected": selected_endpoint_unknown,
        "negative_hour_input_rows_total": int(input_bad.sum()),
        "negative_hour_input_rows_selected": selected_input_issues,
        "selected_rows_missing_y_mort": selected_missing_outcomes,
        "selected_rows_not_adult_eligible": selected_adult_unknown_or_false,
        "selected_rows_with_cutoff_after_24h": selected_bad_cutoff,
        "selected_rows_icu_exit_not_finite_or_after_24h": selected_bad_icu_exit,
        "selected_rows_hospital_exit_not_finite_or_after_24h": selected_bad_hospital_exit,
        "selected_rows_hospital_exit_before_icu_exit": selected_bad_exit_order,
        "selected_rows_exact_death_time_invalid": selected_bad_death_time,
        "selected_rows_temporally_verified": int((selected & frame["temporal_verified"].fillna(False)).sum()),
        "source_join_coverage": join_coverage,
        "schema_columns": int(len(frame.columns)),
        "evidence": meta.get("evidence"),
        "inputs_temporal_scope": "retrospective_measurement_time",
        "input_scope": "retrospective_measurement_time",
        "clinical_input_availability_verified": False,
        "retrospective_landmark_prediction_allowed": True,
        "ready_for_future24h_landmark_analysis": True,
        "ready_for_retrospective_24h_measurement_analysis": True,
        "ready_for_future24h_clinical_claim": False,
    }
    meta["ready_for_future24h_clinical_claim"] = False
    meta["ready_for_retrospective_24h_measurement_analysis"] = True
    meta["inputs_temporal_scope"] = "retrospective_measurement_time"
    meta["input_scope"] = "retrospective_measurement_time"
    meta["clinical_input_availability_verified"] = False
    meta["retrospective_landmark_prediction_allowed"] = True
    meta["ready_for_future24h_landmark_analysis"] = True
    meta["readiness_audit"] = audit
    meta.setdefault("counts", {}).update({
        "n_source_time_issue_rows": audit["source_time_issue_rows_total"],
        "n_source_time_issue_eligible": selected_time_issues,
        "n_source_time_status_conflicts_total": audit["source_time_status_conflicts_total"],
        "n_source_time_status_conflicts_eligible": selected_status_conflicts,
        "n_source_label_mismatch_rows": total_label_mismatches,
        "n_source_label_mismatch_eligible": selected_label_mismatches,
        "n_source_endpoint_unknown_rows": int(endpoint_unknown.sum()),
        "n_source_endpoint_unknown_eligible": selected_endpoint_unknown,
        "n_source_landmark_contradictions": int(contradiction.sum()),
        "n_source_landmark_contradictions_eligible": int((contradiction & selected).sum()),
        "n_icu_exit_after_hospital_exit_after_24": int(exit_order_gap.sum()),
        "n_icu_exit_after_hospital_exit_after_24_eligible": int((exit_order_gap & selected).sum()),
        **manifest_gap_summary,
        "n_negative_hour_input_rows": int(input_bad.sum()),
        "n_negative_hour_input_eligible": selected_input_issues,
    })
    return frame, audit


def main():
    manifest_path = ELIG / "eligibility_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "4.1":
        raise ValueError("manifest schema_version must be 4.1")
    cohorts = manifest.get("cohorts")
    if not isinstance(cohorts, dict):
        raise ValueError("manifest cohorts map missing")
    reports = {}
    prepared = {}
    for db in ("mimic", "eicu"):
        if db not in cohorts:
            raise ValueError(f"manifest cohort missing: {db}")
        item = cohorts[db]
        if item.get("retrospective_landmark_prediction_allowed") is not True:
            raise ValueError(f"{db} manifest lacks the primary retrospective prediction attestation")
        if item.get("ready_for_future24h_landmark_analysis") is not True:
            raise ValueError(f"{db} manifest alias disagrees with the primary readiness key")
        if item.get("ready_for_future24h_clinical_claim") is not False or item.get("clinical_input_availability_verified") is not False:
            raise ValueError(f"{db} manifest overstates real-time clinical input availability")
        frame, audit = audit_cohort(db, cohorts[db])
        reports[db] = audit
        prepared[db] = frame
    for db, frame in prepared.items():
        path = ELIG / f"{db}.parquet"
        tmp = path.with_name(path.name + ".tmp")
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    report_path = ELIG / "core_verification.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {"cohorts": {}}
    report["status"] = "passed"
    report["cohorts"] = report.get("cohorts", {})
    for db, audit in reports.items():
        report["cohorts"].setdefault(db, {}).update(audit)
    atomic_json(report_path, report)
    atomic_json(manifest_path, manifest)
    for db, audit in reports.items():
        print(f"{db}: selected={audit['eligible_rows_checked']} issue_selected={audit['source_time_issue_rows_selected']} status_conflict_selected={audit['source_time_status_conflicts_selected']} mismatch_selected={audit['source_label_mismatch_rows_selected']} endpoint_unknown_selected={audit['source_endpoint_unknown_rows_selected']} landmark_contradiction_total={audit['source_landmark_contradictions_total']} ICUout_after_hospitalout_after24={audit['icu_exit_after_hospital_exit_after_24_total']} ready_retrospective={audit['ready_for_retrospective_24h_measurement_analysis']} future_clinical_claim={audit['ready_for_future24h_clinical_claim']}", flush=True)
    print("selected_row_attestation=passed", flush=True)


if __name__ == "__main__":
    main()
