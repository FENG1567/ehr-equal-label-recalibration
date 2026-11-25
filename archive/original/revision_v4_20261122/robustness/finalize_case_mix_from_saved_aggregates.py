"""Finalize corrected case-mix provenance without rerunning any analysis."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
ROBUSTNESS = REV / "robustness"
PREVIOUS_DIR = ROBUSTNESS / "results/case_mix_selective_primary_v4"
INCOMPLETE_DIR = ROBUSTNESS / "results/case_mix_selective_primary_v4_1_incomplete_20261123_0440"
DEFAULT_OUT = ROBUSTNESS / "results/case_mix_selective_primary_v4_1"
DEFAULT_AUDIT = ROBUSTNESS / "results/case_mix_domain_audit_corrected_20261123.json"
FAILED_LOG = ROBUSTNESS / "logs/case_mix_standardization_v4_1_20261123_0440.log"
FAILED_STATUS = ROBUSTNESS / "logs/case_mix_standardization_v4_1_20261123_0440.exit_status"
RECOMPUTE_SCRIPT = ROBUSTNESS / "recompute_case_mix_standardization.py"
ANALYSIS_SCRIPT = ROBUSTNESS / "case_mix_selective.py"
MAPPING_SOURCE = REV / "temporal/code/repair_landmark.py"
EXPECTED_CELLS = {
    f"{age}|{sex}"
    for age in ("18-49", "50-64", "65-74", "75+")
    for sex in ("female", "male")
}
REUSED_FILES = (
    "age_stratum_performance_slopes.csv",
    "selective_prediction.csv",
    "age_sex_case_mix_raw.csv",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_saved_standardization(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    rows = _read_rows(path)
    if {row.get("db") for row in rows} != {"eicu", "mimic"} or len(rows) != 2:
        raise ValueError("saved corrected standardization must contain exactly eICU and MIMIC")
    common_cells: set[str] | None = None
    for row in rows:
        cells = set(row["common_age_sex_cells"].split(";"))
        if row.get("status") != "ok" or row.get("slope_ci_status") != "ok":
            raise ValueError(f"{row.get('db')}: saved standardization or slope CI is not estimable")
        if cells != EXPECTED_CELLS:
            raise ValueError(f"{row.get('db')}: corrected common support does not contain all 8 cells")
        if int(row["slope_bootstrap_draws_requested"]) != 1000:
            raise ValueError(f"{row.get('db')}: corrected standardization was not requested at 1000 draws")
        if int(row["slope_bootstrap_draws_valid"]) != 1000:
            raise ValueError(f"{row.get('db')}: corrected standardization does not have 1000 valid draws")
        if common_cells is None:
            common_cells = cells
        elif common_cells != cells:
            raise ValueError("saved standardized rows disagree on common support")
    return rows, sorted(common_cells or ())


def _validate_audit(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    audit = _load_json(path)
    if audit.get("status") != "complete" or audit.get("scope", "").startswith("aggregate-only") is False:
        raise ValueError("corrected aggregate-only age-sex audit is incomplete")
    cohorts = audit.get("cohorts", {})
    counts = {
        db: cohorts[db]["normalized_age_sex_cell_counts"]
        for db in ("eicu", "mimic")
    }
    for db, cells in counts.items():
        if set(cells) != EXPECTED_CELLS or sum(int(value) for value in cells.values()) <= 0:
            raise ValueError(f"{db}: corrected audit does not contain all nonempty age-sex cells")
    eicu_sex = cohorts["eicu"]["normalized_sex_value_counts"]
    if eicu_sex.get("female") != 47051 or eicu_sex.get("male") != 55886:
        raise ValueError("corrected audit does not show normalized eICU float sex codes")
    if eicu_sex.get("<missing>") != 20:
        raise ValueError("corrected audit eICU missing-sex count differs from the source audit")
    return audit, counts


def finalize(*, out_dir: Path, audit_path: Path, failed_run_pid: int) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite finalized case-mix output: {out_dir}")
    required = [
        *(INCOMPLETE_DIR / name for name in (*REUSED_FILES, "age_sex_case_mix_standardized.csv")),
        PREVIOUS_DIR / "case_mix_selection_manifest.json",
        audit_path,
        FAILED_LOG,
        FAILED_STATUS,
        RECOMPUTE_SCRIPT,
        ANALYSIS_SCRIPT,
        MAPPING_SOURCE,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required saved aggregate or provenance inputs are missing: {missing}")

    standardized_path = INCOMPLETE_DIR / "age_sex_case_mix_standardized.csv"
    standardized_rows, common_cells = _validate_saved_standardization(standardized_path)
    audit, cell_counts = _validate_audit(audit_path)
    exit_code = int(FAILED_STATUS.read_text(encoding="utf-8").strip())
    if exit_code != 1:
        raise ValueError(f"expected the original finalization-path failure exit 1, got {exit_code}")
    failed_log_text = FAILED_LOG.read_text(encoding="utf-8", errors="replace")
    for marker in ('"draw_completed": 1000', "FileNotFoundError", "/temporal/repair_landmark.py"):
        if marker not in failed_log_text:
            raise ValueError(f"failed run log lacks expected provenance marker {marker!r}")
    previous_manifest_path = PREVIOUS_DIR / "case_mix_selection_manifest.json"
    previous_manifest = _load_json(previous_manifest_path)

    reused_hashes: dict[str, str] = {}
    for name in REUSED_FILES:
        source = INCOMPLETE_DIR / name
        previous_hash = sha256_file(PREVIOUS_DIR / name)
        source_hash = sha256_file(source)
        if source_hash != previous_hash:
            raise ValueError(f"reused aggregate differs from completed v4 output: {name}")
        reused_hashes[name] = source_hash

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = out_dir.with_name(f".{out_dir.name}.finalizing")
    if staging.exists():
        raise FileExistsError(f"refusing to reuse an existing finalizer staging directory: {staging}")
    staging.mkdir(parents=False)
    try:
        for name in (*REUSED_FILES, "age_sex_case_mix_standardized.csv"):
            shutil.copyfile(INCOMPLETE_DIR / name, staging / name)
        shutil.copyfile(audit_path, staging / "case_mix_domain_audit_20261123.json")

        corrected_cells = {
            db: {str(cell): int(value) for cell, value in cells.items()}
            for db, cells in cell_counts.items()
        }
        (staging / "case_mix_domain_counts.json").write_text(
            json.dumps({
                "status": "complete",
                "scope": "aggregate-only corrected age-sex cell audit; no model fitting or bootstrap",
                "cell_counts_by_cohort": corrected_cells,
                "common_support_cells": common_cells,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        mapping_source_sha = sha256_file(MAPPING_SOURCE)
        manifest = {
            **previous_manifest,
            "status": "complete",
            "seed": 20261122,
            "bootstrap_draws": 1000,
            "analysis_scope": (
                "corrected age-sex standardization finalized from saved 1000-draw results; "
                "unchanged age-stratum and selective aggregates reused from the completed v4 diagnostic run; "
                "no bootstrap or model refit was performed by this finalizer"
            ),
            "sex_normalization": (
                "0/0.0=female; 1/1.0=male; source-supported M/F and Male/Female aliases; "
                "other nonmissing codes excluded as unknown"
            ),
            "sex_mapping_source": str(MAPPING_SOURCE),
            "sex_mapping_source_sha256": mapping_source_sha,
            "source_sex_semantics": (
                "repair_landmark.py maps MIMIC patients.gender M/F and eICU patient.gender "
                "Male/Female or M/F to 1=male, 0=female"
            ),
            "normalized_age_sex_cell_counts": corrected_cells,
            "common_support_cells": common_cells,
            "minimum_patients_per_cell_per_cohort": 10,
            "target_distribution_rule": (
                "equal-weight mean of cohort-specific distributions across common age-sex cells"
            ),
            "slope_uncertainty": "patient-cluster bootstrap with fixed predictions and fixed common-support weights",
            "corrected_standardized_rows": len(standardized_rows),
            "reused_aggregate_outputs_from_diagnostic_run": reused_hashes,
            "corrected_saved_standardization": {
                "source_directory": str(INCOMPLETE_DIR),
                "source_csv_sha256": sha256_file(standardized_path),
                "source_script": str(RECOMPUTE_SCRIPT),
                "source_script_sha256": sha256_file(RECOMPUTE_SCRIPT),
                "analysis_module": str(ANALYSIS_SCRIPT),
                "analysis_module_sha256": sha256_file(ANALYSIS_SCRIPT),
                "requested_patient_cluster_draws_per_cohort": 1000,
                "valid_patient_cluster_draws_per_cohort": {
                    row["db"]: int(row["slope_bootstrap_draws_valid"])
                    for row in standardized_rows
                },
            },
            "aggregate_only_corrected_cell_audit": {
                "path": str(audit_path),
                "sha256": sha256_file(audit_path),
                "prediction_manifest_sha256": audit["prediction_manifest_sha256"],
                "prediction_file_sha256": audit["prediction_file_sha256"],
                "eligibility_manifest_sha256": audit["eligibility_manifest_sha256"],
            },
            "failed_original_finalization_attempt": {
                "pid": int(failed_run_pid),
                "log_path": str(FAILED_LOG),
                "log_sha256": sha256_file(FAILED_LOG),
                "exit_status_path": str(FAILED_STATUS),
                "exit_status": exit_code,
                "reason": "computed CSV was written; provenance serialization failed while hashing the obsolete source path temporal/repair_landmark.py",
                "bootstrap_completion_confirmed": True,
            },
            "previous_zero_support_diagnostic": {
                "path": str(PREVIOUS_DIR / "age_sex_case_mix_standardized.csv"),
                "sha256": sha256_file(PREVIOUS_DIR / "age_sex_case_mix_standardized.csv"),
                "status": "insufficient_common_age_sex_support_before_float_code_normalization",
            },
            "previous_run_manifest_sha256": sha256_file(previous_manifest_path),
            "finalizer": {
                "method": "aggregate-only; copies validated saved outputs and writes provenance",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "patient_level_exported": False,
            },
            "clinical_input_availability_verified": False,
            "clinical_promotion_allowed": False,
            "patient_level_exported": False,
        }
        (staging / "case_mix_selection_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # Re-read the staged outputs and confirm all inputs survived byte-for-byte.
        for name in (*REUSED_FILES, "age_sex_case_mix_standardized.csv"):
            if sha256_file(staging / name) != sha256_file(INCOMPLETE_DIR / name):
                raise IOError(f"staged output hash mismatch after copy: {name}")
        staging.replace(out_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({
        "status": "complete",
        "output_dir": str(out_dir),
        "finalization_method": "no-refit; saved corrected CSV and aggregate audit only",
        "common_support_cells": common_cells,
        "standardized_rows": len(standardized_rows),
        "reused_files": sorted(reused_hashes),
        "original_failed_run_pid": int(failed_run_pid),
        "original_failed_run_exit_status": exit_code,
    }, sort_keys=True), flush=True)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--failed-run-pid", type=int, required=True)
    args = parser.parse_args(argv)
    finalize(out_dir=args.out_dir, audit_path=args.audit, failed_run_pid=args.failed_run_pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
