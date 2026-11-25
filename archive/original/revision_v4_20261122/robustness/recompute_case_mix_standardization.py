"""Recompute only the corrected age-sex common-support standardization."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from case_mix_selective import (
    _age_sex_cell_counts, _age_sex_standardized_rows, age_group, normalize_sex,
)
from recalibration import one_index_per_patient
from robustness_common import load_checked, sha256_file

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
FM_RUN = REV / "engine/results/prediction_runs/status_consistent_primary_core_fix2-7da980ffcbd8f00b"
ELIGIBILITY_DIR = REV / "temporal/results/eligibility/status_consistent_v4_1/landmark_v4_1"
PREVIOUS_DIR = REV / "robustness/results/case_mix_selective_primary_v4"
SEED = 20261122


def _counts(values: pd.Series) -> dict[str, int]:
    result: dict[str, int] = {}
    for value, count in values.value_counts(dropna=False, sort=True).items():
        key = "<missing>" if pd.isna(value) else str(value)
        result[key] = int(count)
    return result


def run(*, out_dir: Path, n_boot: int, seed: int) -> dict[str, Any]:
    if n_boot < 1000:
        raise ValueError("corrected age-sex standardization requires at least 1000 patient-cluster draws")
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite corrected case-mix output: {out_dir}")
    required_previous = (
        "age_stratum_performance_slopes.csv",
        "selective_prediction.csv",
        "age_sex_case_mix_raw.csv",
        "case_mix_selection_manifest.json",
        "age_sex_case_mix_standardized.csv",
    )
    missing_previous = [name for name in required_previous if not (PREVIOUS_DIR / name).is_file()]
    if missing_previous:
        raise FileNotFoundError(f"completed diagnostic case-mix outputs missing: {missing_previous}")

    prediction_manifest = FM_RUN / "prediction_manifest.json"
    prediction_path = FM_RUN / "main20k_real.parquet"
    data, gates = load_checked(
        prediction_path, ELIGIBILITY_DIR, model_tag="main20k", input_condition="real",
        prediction_manifest_path=prediction_manifest,
    )
    indexed_data: dict[str, pd.DataFrame] = {}
    for di, (db, joined) in enumerate(data.items()):
        usable = joined.loc[joined["eligible_24h"] & joined["temporal_verified"]].copy()
        if usable.empty:
            continue
        indexed = one_index_per_patient(usable, seed + di)
        if "age" in indexed:
            indexed["age_group"] = age_group(indexed["age"])
        if "sex" in indexed:
            indexed["sex_norm"] = normalize_sex(indexed["sex"])
        indexed_data[db] = indexed
        print(json.dumps({
            "stage": "corrected_case_mix_indexed_population",
            "db": db, "n_index_patients": int(len(indexed)),
            "raw_sex_dtype": str(indexed["sex"].dtype) if "sex" in indexed else "unavailable",
            "raw_sex_value_counts": _counts(indexed["sex"]) if "sex" in indexed else {},
            "normalized_sex_value_counts": _counts(indexed["sex_norm"]) if "sex_norm" in indexed else {},
        }, sort_keys=True), flush=True)

    cell_counts = _age_sex_cell_counts(indexed_data)
    standardized = _age_sex_standardized_rows(
        indexed_data, gates, n_boot=n_boot, seed=seed,
    )
    if len(standardized) != len(indexed_data) or any(row.get("status") != "ok" for row in standardized):
        raise ValueError("corrected age-sex standardization did not produce one estimable row per cohort")
    if any(row.get("slope_bootstrap_draws_requested") != n_boot for row in standardized):
        raise ValueError("corrected standardization rows do not record the requested patient-cluster draws")
    common_cells = set(standardized[0]["common_age_sex_cells"].split(";"))
    if len(common_cells) != 8:
        raise ValueError(f"expected all 8 normalized age-sex cells after source-code harmonization, got {len(common_cells)}")
    mapping_path = REV / "temporal/code/repair_landmark.py"
    mapping_sha256 = sha256_file(mapping_path)

    out_dir.mkdir(parents=True, exist_ok=False)
    reused_hashes: dict[str, str] = {}
    for name in required_previous[:3]:
        source = PREVIOUS_DIR / name
        shutil.copyfile(source, out_dir / name)
        reused_hashes[name] = sha256_file(source)
    standardized_path = out_dir / "age_sex_case_mix_standardized.csv"
    pd.DataFrame(standardized).to_csv(standardized_path, index=False)

    old_standardized_path = PREVIOUS_DIR / "age_sex_case_mix_standardized.csv"
    previous_manifest = json.loads(
        (PREVIOUS_DIR / "case_mix_selection_manifest.json").read_text(encoding="utf-8")
    )
    manifest = {
        "status": "complete",
        "seed": int(seed),
        "bootstrap_draws": int(n_boot),
        "analysis_scope": "corrected age-sex standardization only; unchanged selective and age-stratum aggregates were reused from the completed diagnostic run",
        "prediction_manifest": str(prediction_manifest),
        "prediction_manifest_sha256": sha256_file(prediction_manifest),
        "prediction_file_sha256": sha256_file(prediction_path),
        "eligibility_manifest_sha256": gates[next(iter(gates))]["eligibility_manifest_sha256"],
        "cohort_gates": gates,
        "sex_normalization": "0/0.0=female; 1/1.0=male; source-supported M/F and Male/Female aliases; other nonmissing codes excluded as unknown",
        "sex_mapping_source": str(mapping_path),
        "sex_mapping_source_sha256": mapping_sha256,
        "source_sex_semantics": "repair_landmark.py maps MIMIC patients.gender M/F and eICU patient.gender Male/Female or M/F to 1=male, 0=female",
        "normalized_age_sex_cell_counts": cell_counts,
        "common_support_cells": sorted(common_cells),
        "minimum_patients_per_cell_per_cohort": 10,
        "target_distribution_rule": "equal-weight mean of cohort-specific distributions across common age-sex cells",
        "slope_uncertainty": "patient-cluster bootstrap with fixed predictions and fixed common-support weights",
        "corrected_standardized_rows": len(standardized),
        "reused_aggregate_outputs_from_diagnostic_run": reused_hashes,
        "previous_zero_support_diagnostic": {
            "path": str(old_standardized_path),
            "sha256": sha256_file(old_standardized_path),
            "status": "insufficient_common_age_sex_support",
        },
        "previous_run_manifest_sha256": sha256_file(PREVIOUS_DIR / "case_mix_selection_manifest.json"),
        "prediction_scope": previous_manifest.get("prediction_scope"),
        "clinical_input_availability_verified": False,
        "clinical_promotion_allowed": False,
        "patient_level_exported": False,
    }
    (out_dir / "case_mix_selection_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
    )
    (out_dir / "case_mix_domain_counts.json").write_text(
        json.dumps({"status": "complete", "cell_counts_by_cohort": cell_counts,
                    "common_support_cells": sorted(common_cells)}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "complete", "output_dir": str(out_dir),
        "reused_files": sorted(reused_hashes),
        "standardized_rows": len(standardized),
        "common_support_cells": sorted(common_cells),
    }, sort_keys=True), flush=True)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(REV / "robustness/results/case_mix_selective_primary_v4_1"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    run(out_dir=Path(args.out_dir), n_boot=args.bootstrap, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
