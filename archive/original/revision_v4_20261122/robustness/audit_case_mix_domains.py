"""Aggregate-only audit of current indexed age/sex coding and support cells."""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from case_mix_selective import _age_sex_cell_counts, age_group, normalize_sex
from recalibration import one_index_per_patient
from robustness_common import load_checked, sha256_file

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
FM_RUN = REV / "engine/results/prediction_runs/status_consistent_primary_core_fix2-7da980ffcbd8f00b"
ELIGIBILITY_DIR = REV / "temporal/results/eligibility/status_consistent_v4_1/landmark_v4_1"
SEED = 20261122


def _counts(values: pd.Series) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value, count in values.value_counts(dropna=False, sort=True).items():
        key = "<missing>" if pd.isna(value) else str(value)
        counts[key] = int(count)
    return counts


def run() -> dict[str, Any]:
    prediction_manifest = FM_RUN / "prediction_manifest.json"
    prediction_path = FM_RUN / "main20k_real.parquet"
    data, gates = load_checked(
        prediction_path, ELIGIBILITY_DIR,
        model_tag="main20k", input_condition="real",
        prediction_manifest_path=prediction_manifest,
    )
    cohorts: dict[str, Any] = {}
    for di, (db, joined) in enumerate(data.items()):
        usable = joined.loc[joined["eligible_24h"] & joined["temporal_verified"]].copy()
        indexed = one_index_per_patient(usable, SEED + di)
        raw_age = indexed["age"]
        raw_sex = indexed["sex"]
        age_numeric = pd.to_numeric(raw_age, errors="coerce")
        age_labels = age_group(raw_age)
        sex_labels = normalize_sex(raw_sex)
        indexed["age_group"] = age_labels
        indexed["sex_norm"] = sex_labels
        cell_counts = _age_sex_cell_counts({db: indexed})[db]
        unrecognized_sex = raw_sex.notna() & sex_labels.isna()
        cohorts[db] = {
            "n_selected_index_patients": int(len(indexed)),
            "raw_age_dtype": str(raw_age.dtype),
            "raw_sex_dtype": str(raw_sex.dtype),
            "raw_sex_value_counts": _counts(raw_sex),
            "normalized_sex_value_counts": _counts(sex_labels),
            "age_numeric_nonmissing": int(age_numeric.notna().sum()),
            "age_group_value_counts": _counts(age_labels),
            "normalized_age_sex_cell_counts": cell_counts,
            "unrecognized_raw_sex_value_counts": _counts(raw_sex.loc[unrecognized_sex]),
            "eligible_prediction_population": {
                key: gates[db][key] for key in (
                    "n_joined", "n_eligible_24h", "n_temporal_verified",
                    "eligibility_table_sha256", "prediction_file_sha256",
                ) if key in gates[db]
            },
        }
    return {
        "status": "complete",
        "scope": "aggregate-only; same current strict population and deterministic one-stay-per-patient selection as case_mix_selective",
        "seed": SEED,
        "prediction_manifest_sha256": sha256_file(prediction_manifest),
        "prediction_file_sha256": sha256_file(prediction_path),
        "eligibility_manifest_sha256": gates[next(iter(gates))]["eligibility_manifest_sha256"],
        "cohorts": cohorts,
        "current_sex_normalization": "case_mix_selective.normalize_sex; 0/0.0=female and 1/1.0=male; unrecognized nonmissing codes are excluded",
        "source_sex_mapping_reference": str(REV / "temporal/code/repair_landmark.py"),
        "source_sex_mapping_reference_sha256": sha256_file(
            REV / "temporal/code/repair_landmark.py"),
        "source_sex_mapping_semantics": "MIMIC patients.gender M/F and eICU patient.gender Male/Female or M/F are mapped to 1=male, 0=female before eligibility output",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded, flush=True)
