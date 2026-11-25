"""Validate finalized case-mix and completed hospital aggregate outputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

MODELS = (
    "main20k_real",
    "main20k_constant",
    "conventional_lr_value_plus_pattern",
)
COMPARISONS = (
    "main20k_real_minus_main20k_constant",
    "main20k_real_minus_conventional_lr_value_plus_pattern",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(*, case_mix_dir: Path, hospital_dir: Path, preflight_path: Path,
             checkpoint_dir: Path, out_path: Path) -> dict[str, Any]:
    case_manifest_path = case_mix_dir / "case_mix_selection_manifest.json"
    case_manifest = _json(case_manifest_path)
    standardization_columns, standardization = _csv(case_mix_dir / "age_sex_case_mix_standardized.csv")
    _require(case_manifest.get("status") == "complete", "case-mix manifest is not complete")
    _require(len(standardization) == 2
             and {row.get("db") for row in standardization} == {"eicu", "mimic"},
             "case-mix standardized table must contain eICU and MIMIC only")
    common_cells = set(case_manifest.get("common_support_cells", []))
    _require(len(common_cells) == 8, "case-mix manifest lacks all eight age-sex support cells")
    for row in standardization:
        _require(row.get("status") == "ok" and row.get("slope_ci_status") == "ok",
                 f"{row.get('db')}: standardized row or CI status is not ok")
        _require(set(row["common_age_sex_cells"].split(";")) == common_cells,
                 f"{row.get('db')}: standardized row support differs from the manifest")
        _require(int(row["slope_bootstrap_draws_requested"]) == 1000
                 and int(row["slope_bootstrap_draws_valid"]) == 1000,
                 f"{row.get('db')}: standardized slope CI does not contain 1000 valid draws")
    failed_run = case_manifest.get("failed_original_finalization_attempt", {})
    _require(failed_run.get("pid") == 1191787 and failed_run.get("exit_status") == 1
             and failed_run.get("bootstrap_completion_confirmed") is True,
             "case-mix manifest does not bind the completed computation to its path-only failure")
    _require(case_manifest.get("patient_level_exported") is False,
             "case-mix manifest must explicitly state no patient-level export")

    preflight = _json(preflight_path)
    _require(preflight.get("status") == "complete", "hospital site preflight is incomplete")
    _require(preflight.get("selected_hospital_ids_with_current_predictions") == 206,
             "hospital preflight selected-site count differs from 206")
    preflight_counts = preflight.get("site_counts_by_model", {})
    _require(set(preflight_counts) == set(MODELS), "preflight model set is incomplete")
    for model_id, counts in preflight_counts.items():
        _require(counts.get("selected_hospitals") == 206,
                 f"{model_id}: preflight site count differs from 206")
        _require(counts.get("sites_ge_10_events_and_nonevents") == 147
                 and counts.get("sites_below_10_events_or_nonevents") == 59,
                 f"{model_id}: preflight event-threshold counts are unexpected")
    _require(preflight.get("workload", {}).get("site_model_checkpoint_jobs") == 618,
             "preflight checkpoint workload does not equal 618 site-model jobs")

    hospital_manifest_path = hospital_dir / "hospital_analysis_manifest.json"
    hospital_manifest = _json(hospital_manifest_path)
    _require(hospital_manifest.get("status") == "complete", "hospital analysis manifest is not complete")
    _require(hospital_manifest.get("bootstrap_draws") == 1000,
             "hospital manifest does not record 1000 patient-cluster draws")
    _require(hospital_manifest.get("selected_hospital_ids_with_current_predictions") == 206,
             "hospital manifest selected-site count differs from 206")
    crossing = hospital_manifest.get("patient_hospital_crossing", {})
    _require(crossing.get("n_patients_crossing_hospitals", 0) > 0
             and crossing.get("patients_are_nested_within_hospital") is False,
             "hospital manifest does not record cross-hospital patient membership")
    _require("site-resampling-only" in hospital_manifest.get("bootstrap_scope", ""),
             "hospital manifest must label site-bootstrap uncertainty as site-resampling-only")
    slope_ci_manifest = hospital_manifest.get("within_site_calibration_slope_ci", {})
    _require(slope_ci_manifest.get("site_model_checkpoints_completed") == 618
             and slope_ci_manifest.get("site_model_checkpoint_files_present") == 618,
             "hospital manifest checkpoint accounting is not 618/618")

    overall_columns, overall = _csv(hospital_dir / "eicu_overall_performance.csv")
    site_columns, site_rows = _csv(hospital_dir / "eicu_hospital_metrics.csv")
    paired_columns, paired_rows = _csv(hospital_dir / "eicu_hospital_paired_deltas.csv")
    ci_columns, ci_rows = _csv(hospital_dir / "eicu_hospital_site_bootstrap_ci.csv")
    _require(len(overall) == 3 and {row.get("model_id") for row in overall} == set(MODELS),
             "hospital overall table does not contain three required models")
    _require(len(site_rows) == 618, "hospital metrics table does not contain 618 rows")
    _require(len(paired_rows) == 412, "paired hospital table does not contain 412 rows")
    _require(len(ci_rows) == 20, "hospital site-bootstrap summary does not contain 20 rows")
    for field_columns in (overall_columns, site_columns, paired_columns, ci_columns):
        _require(not ({"subject", "stay_id", "patient_id", "unique_pid"} & set(field_columns)),
                 "aggregate hospital table exposes patient or stay identifiers")

    sites_by_model: dict[str, set[str]] = {}
    slope_status_counts: dict[str, dict[str, int]] = {}
    for model_id in MODELS:
        model_rows = [row for row in site_rows if row.get("scope") == "selected_hospital_all_stays"
                      and row.get("model_id") == model_id]
        _require(len(model_rows) == 206, f"{model_id}: expected 206 hospital metric rows")
        sites_by_model[model_id] = {row["hospitalid"] for row in model_rows}
        _require(len(sites_by_model[model_id]) == 206, f"{model_id}: duplicated hospital metric rows")
        _require(all(int(row["calibration_slope_ci_bootstrap_draws_requested"]) == 1000
                     for row in model_rows), f"{model_id}: site slope draw requests are incomplete")
        counts: dict[str, int] = {}
        for row in model_rows:
            status = row.get("calibration_slope_ci_status", "")
            _require(status != "", f"{model_id}/{row.get('hospitalid')}: missing site slope CI status")
            counts[status] = counts.get(status, 0) + 1
        slope_status_counts[model_id] = counts
    _require(len(set.intersection(*sites_by_model.values())) == 206,
             "hospital metric hospital IDs differ across model arms")

    paired_sites: dict[str, set[str]] = {}
    for comparison_id in COMPARISONS:
        rows = [row for row in paired_rows if row.get("comparison_id") == comparison_id]
        _require(len(rows) == 206, f"{comparison_id}: expected 206 paired hospital rows")
        paired_sites[comparison_id] = {row["hospitalid"] for row in rows}
    _require(all(site_set == next(iter(sites_by_model.values()))
                 for site_set in paired_sites.values()),
             "paired hospital rows do not cover the model hospital set")

    checkpoint_files = list(checkpoint_dir.glob("site_slope_ci_*.json"))
    _require(len(checkpoint_files) == 618, "expected 618 per-site/model slope checkpoints")
    result = {
        "status": "complete",
        "validation_scope": "aggregate-only verification; no model fits or bootstrap recomputation",
        "case_mix": {
            "output_dir": str(case_mix_dir),
            "standardized_rows": len(standardization),
            "common_support_cells": sorted(common_cells),
            "slope_draws_valid_by_cohort": {
                row["db"]: int(row["slope_bootstrap_draws_valid"]) for row in standardization
            },
            "manifest_sha256": _sha256(case_manifest_path),
            "standardized_csv_sha256": _sha256(case_mix_dir / "age_sex_case_mix_standardized.csv"),
            "failed_run_pid": failed_run["pid"],
            "failed_run_exit_status": failed_run["exit_status"],
        },
        "hospital": {
            "output_dir": str(hospital_dir),
            "selected_hospitals": 206,
            "site_rows_by_model": {model_id: len(sites) for model_id, sites in sites_by_model.items()},
            "slope_ci_status_counts_by_model": slope_status_counts,
            "overall_rows": len(overall),
            "paired_rows": len(paired_rows),
            "site_bootstrap_ci_rows": len(ci_rows),
            "checkpoints": len(checkpoint_files),
            "cross_hospital_patients": crossing["n_patients_crossing_hospitals"],
            "bootstrap_scope": hospital_manifest["bootstrap_scope"],
            "manifest_sha256": _sha256(hospital_manifest_path),
        },
        "preflight": {
            "path": str(preflight_path),
            "sha256": _sha256(preflight_path),
            "selected_hospitals": 206,
            "stable_sites_per_model": 147,
            "below_threshold_sites_per_model": 59,
            "site_model_jobs": 618,
            "bootstrap_job_upper_bound": 441,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(json.dumps({"status": "complete", "validation_path": str(out_path),
                      "hospital_slope_ci_status_counts_by_model": slope_status_counts},
                     sort_keys=True), flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-mix-dir", type=Path, required=True)
    parser.add_argument("--hospital-dir", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    validate(case_mix_dir=args.case_mix_dir, hospital_dir=args.hospital_dir,
             preflight_path=args.preflight, checkpoint_dir=args.checkpoint_dir,
             out_path=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
