"""Fail-closed attestation of original versus repaired eICU sampling draws.

This uses only aggregate draw hashes for the report. Patient identifiers and
per-stay records remain in server memory and are never written to the output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from paired_baselines import sha256_file
from transport_analysis import (
    BUDGETS, PLAN_COMPARE_FIELDS, REPEATS, SEED, TARGET_COHORTS,
    OLD_EICU_CALIBRATION_DRAWS, OLD_EICU_CALIBRATION_MANIFEST,
    _compare_eicu_row_identity, _load_all_model_frames, _load_original_eicu_frame,
    _sampling_plan_for_frame,
)

R2_OUT = (
    Path("/data/admin/fengshuo/ehr_fm_20261102/revision_v4_20261122")
    / "robustness/results/recovered_transport_v4_2_20261123_reuse_r2"
)


def _records(rows: list[dict[str, Any]], *, index_name: str, stay_name: str) -> tuple[
    dict[tuple[int, int], tuple[str, str]], int, int
]:
    records: dict[tuple[int, int], tuple[str, str]] = {}
    duplicates = 0
    missing_hashes = 0
    for row in rows:
        key = (int(row["budget"]), int(row["repeat"]))
        index_hash = str(row[index_name])
        stay_hash = str(row[stay_name])
        if not index_hash or index_hash.lower() == "nan" or not stay_hash or stay_hash.lower() == "nan":
            missing_hashes += 1
        if key in records:
            duplicates += 1
        else:
            records[key] = (index_hash, stay_hash)
    return records, duplicates, missing_hashes


def _digest_pair_mismatches(
    left: dict[tuple[int, int], tuple[str, str]],
    right: dict[tuple[int, int], tuple[str, str]],
) -> dict[str, int]:
    shared = set(left) & set(right)
    return {
        "draw_index_sha256": sum(left[key][0] != right[key][0] for key in shared),
        "patient_index_stay_sha256": sum(left[key][1] != right[key][1] for key in shared),
    }


def attest(out_path: Path, *, r2_out: Path = R2_OUT) -> dict[str, Any]:
    r2_manifest_path = r2_out / "transport_analysis_manifest.json"
    r2_plan_path = r2_out / "sampling_plans/eicu_semantic_v4_2.json"
    r2_manifest = json.loads(r2_manifest_path.read_text(encoding="utf-8"))
    r2_plan_document = json.loads(r2_plan_path.read_text(encoding="utf-8"))
    r2_plan = r2_plan_document.get("cohort_sampling_plan", {})
    original_manifest = json.loads(OLD_EICU_CALIBRATION_MANIFEST.read_text(encoding="utf-8"))
    original_plan = original_manifest.get("cohort_sampling_plans", {}).get("eicu", {})

    frames, _, _, _ = _load_all_model_frames()
    repaired = frames["main20k_real"]["eicu_semantic_v4_2"]
    original = _load_original_eicu_frame()
    identity = _compare_eicu_row_identity(original, repaired)

    canonical = repaired.copy()
    canonical["db"] = "eicu"
    canonical_plan, _ = _sampling_plan_for_frame(canonical, db="eicu")
    canonical_records, canonical_duplicates, canonical_missing = _records(
        canonical_plan.get("draws", []), index_name="draw_index_sha256",
        stay_name="patient_index_stay_sha256",
    )
    alias_records, alias_duplicates, alias_missing = _records(
        r2_plan.get("draws", []), index_name="draw_index_sha256",
        stay_name="patient_index_stay_sha256",
    )

    old = pd.read_csv(OLD_EICU_CALIBRATION_DRAWS)
    required = {
        "db", "method", "budget", "repeat", "adaptation_sample_index_sha256",
        "adaptation_sample_patient_index_stay_sha256",
    }
    missing_columns = sorted(required - set(old.columns))
    old_filtered = old.iloc[0:0].copy()
    if not missing_columns:
        old_budget = pd.to_numeric(old["budget"], errors="coerce")
        old_filtered = old.loc[
            old["db"].astype(str).eq("eicu")
            & old["method"].astype(str).eq("raw")
            & old_budget.isin(BUDGETS)
        ].copy()
    old_rows = [
        {
            "budget": int(row.budget), "repeat": int(row.repeat),
            "adaptation_sample_index_sha256": row.adaptation_sample_index_sha256,
            "adaptation_sample_patient_index_stay_sha256": row.adaptation_sample_patient_index_stay_sha256,
        }
        for row in old_filtered.itertuples(index=False)
    ]
    old_records, old_duplicates, old_missing = _records(
        old_rows, index_name="adaptation_sample_index_sha256",
        stay_name="adaptation_sample_patient_index_stay_sha256",
    )

    expected_keys = {(int(budget), int(repeat)) for budget in BUDGETS for repeat in range(REPEATS)}
    old_keys_equal = set(old_records) == expected_keys
    canonical_keys_equal = set(canonical_records) == expected_keys
    alias_keys_equal = set(alias_records) == expected_keys
    old_canonical_mismatches = _digest_pair_mismatches(old_records, canonical_records)
    alias_canonical_mismatches = _digest_pair_mismatches(alias_records, canonical_records)
    plan_mismatches = [
        field for field in PLAN_COMPARE_FIELDS
        if original_plan.get(field) != canonical_plan.get(field)
    ]
    old_other_db_raw_rows = int((
        old["method"].astype(str).eq("raw")
        & pd.to_numeric(old["budget"], errors="coerce").isin(BUDGETS)
        & ~old["db"].astype(str).eq("eicu")
    ).sum()) if {"db", "method", "budget"}.issubset(old.columns) else None

    r2_eicu_complete = (
        r2_plan_document.get("status") == "complete"
        and r2_plan_document.get("db") == "eicu_semantic_v4_2"
        and "eicu_semantic_v4_2" in r2_manifest.get("cohorts_complete", [])
    )
    exact = bool(
        not missing_columns
        and r2_eicu_complete
        and identity["exact_stay_subject_row_split_label_identity"]
        and not plan_mismatches
        and len(old_filtered) == REPEATS * len(BUDGETS)
        and old_duplicates == canonical_duplicates == alias_duplicates == 0
        and old_missing == canonical_missing == alias_missing == 0
        and old_keys_equal and canonical_keys_equal and alias_keys_equal
        and all(value == 0 for value in old_canonical_mismatches.values())
        and alias_canonical_mismatches["draw_index_sha256"] == 0
    )
    document = {
        "schema": "repaired_eicu_original_draw_hash_attestation_v1",
        "status": "exact_original_draw_hashes_attested" if exact else "draw_hash_attestation_failed",
        "cross_version_curve_comparison_allowed": exact,
        "eicu_evaluation_complete_in_r2": bool(r2_eicu_complete),
        "identity_check": identity,
        "plan_field_mismatches": plan_mismatches,
        "original_plan_sampling_plan_sha256": original_plan.get("sampling_plan_sha256"),
        "canonical_repaired_sampling_plan_sha256": canonical_plan.get("sampling_plan_sha256"),
        "executed_r2_alias_sampling_plan_sha256": r2_plan.get("sampling_plan_sha256"),
        "original_draws_path": str(OLD_EICU_CALIBRATION_DRAWS),
        "original_draws_sha256": sha256_file(OLD_EICU_CALIBRATION_DRAWS),
        "original_draw_method_filter": "db=eicu AND method=raw AND budget in [50,200,500,2000]",
        "old_other_database_raw_budget_rows_excluded": old_other_db_raw_rows,
        "expected_draw_keys": int(len(expected_keys)),
        "n_original_filtered_draw_rows": int(len(old_filtered)),
        "n_original_unique_keys": int(len(old_records)),
        "n_canonical_repaired_plan_draws": int(len(canonical_records)),
        "n_executed_r2_plan_draws": int(len(alias_records)),
        "duplicate_keys": {
            "original": old_duplicates, "canonical_repaired": canonical_duplicates,
            "executed_r2_alias": alias_duplicates,
        },
        "missing_hash_rows": {
            "original": old_missing, "canonical_repaired": canonical_missing,
            "executed_r2_alias": alias_missing,
        },
        "draw_key_sets_equal_expected_800": {
            "original": old_keys_equal, "canonical_repaired": canonical_keys_equal,
            "executed_r2_alias": alias_keys_equal,
        },
        "original_vs_canonical_repaired_hash_mismatches": old_canonical_mismatches,
        "executed_alias_vs_canonical_hash_mismatches": alias_canonical_mismatches,
        "original_plan_manifest_sha256": sha256_file(OLD_EICU_CALIBRATION_MANIFEST),
        "executed_r2_plan_path": str(r2_plan_path),
        "executed_r2_plan_sha256": sha256_file(r2_plan_path),
        "executed_r2_analysis_manifest_path": str(r2_manifest_path),
        "executed_r2_analysis_manifest_sha256": sha256_file(r2_manifest_path),
        "patient_level_exported_to_local": False,
        "attestation_note": (
            "The r2 evaluated plan encoded the cohort alias in patient/stay digests. "
            "The canonical repaired plan is regenerated from the same repaired predictions "
            "with db=eicu; exact row identity plus matching draw-index hashes binds the "
            "executed r2 indices to the canonical plan, and both canonical digests are "
            "compared with the filtered original eICU raw rows."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise FileExistsError(f"refusing to replace existing draw hash attestation: {out_path}")
    temporary = out_path.with_name(f".{out_path.name}.tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(out_path)
    print(json.dumps({
        "status": document["status"],
        "cross_version_curve_comparison_allowed": document["cross_version_curve_comparison_allowed"],
        "n_original_filtered_draw_rows": document["n_original_filtered_draw_rows"],
        "n_canonical_repaired_plan_draws": document["n_canonical_repaired_plan_draws"],
        "n_executed_r2_plan_draws": document["n_executed_r2_plan_draws"],
        "original_vs_canonical_repaired_hash_mismatches": old_canonical_mismatches,
        "output_path": str(out_path),
    }, sort_keys=True), flush=True)
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--r2-out", type=Path, default=R2_OUT)
    args = parser.parse_args(argv)
    document = attest(args.out, r2_out=args.r2_out)
    return 0 if document["status"] == "exact_original_draw_hashes_attested" else 1


if __name__ == "__main__":
    raise SystemExit(main())
