"""Synthetic-only tests for the separate frozen-output release gate."""
from __future__ import annotations

import json
import unittest

import pandas as pd

import audit_numeric_decoding_release as audit


def _synthetic_inputs():
    rows = []
    refs = []
    validation = []
    cache_rows = {analysis_id: 192805 for analysis_id in audit.ANALYSIS_IDS}
    for analysis_id in audit.ANALYSIS_IDS:
        for variable in audit.VARIABLES:
            reference = {
                "analysis_id": analysis_id,
                "mode": "source_transfer",
                "source_db": audit.SOURCE_DB,
                "target_db": audit.SOURCE_DB,
                "variable": variable,
                "alpha": 10.0,
                "source_alpha_validation_mse": 0.25,
            }
            refs.append(reference)
            validation.append({
                "analysis_id": analysis_id,
                "variable": variable,
                "status": "matched",
                "r2_absolute_difference": 0.0,
                "rmse_absolute_difference": 0.0,
                "mae_absolute_difference": 0.0,
            })
            for mode in audit.MODES:
                rows.append({
                    "analysis_id": analysis_id,
                    "mode": mode,
                    "source_db": audit.SOURCE_DB,
                    "target_db": audit.TARGET_DB,
                    "variable": variable,
                    "status": "ok",
                    "population_identity_sha256": f"{variable:0<64}"[:64],
                    "n_eligible_stays": 130544,
                    "n_common_condition_stays": 130544,
                    "n_target_available": 1000,
                    "n_rows": 1000,
                    "n_patients": 900,
                    "cache_rows_by_condition_json": json.dumps(cache_rows, sort_keys=True),
                    "r2": -0.15 if variable == "wbc" else 0.25,
                    "rmse": 0.8,
                    "mae": 0.6,
                    "r2_lo": -0.35,
                    "r2_hi": 0.4,
                    "rmse_lo": 0.7,
                    "rmse_hi": 0.9,
                    "mae_lo": 0.5,
                    "mae_hi": 0.7,
                    "bootstrap_valid": 1000,
                    "bootstrap_valid_r2": 1000,
                    "bootstrap_replicates": 1000,
                    "bootstrap_cluster_unit": "patient",
                    "bootstrap_interval_type": "marginal_patient_cluster_percentile",
                    "bootstrap_seed": 123,
                    "alpha": 10.0 if mode == "source_transfer" else None,
                    "source_alpha_validation_mse": 0.25 if mode == "source_transfer" else None,
                    "source_mimic_split2_metrics_match_reference": (
                        True if mode == "source_transfer" else None
                    ),
                    "n_outer_folds": 5 if mode == "target_local_nested_cv" else None,
                    "n_inner_folds": 3 if mode == "target_local_nested_cv" else None,
                })
    run_manifest = {
        "status": "complete",
        "source_db": audit.SOURCE_DB,
        "target_dbs": [audit.TARGET_DB],
        "loaded_cohorts": [audit.SOURCE_DB, audit.TARGET_DB],
        "analysis_ids": list(audit.ANALYSIS_IDS),
        "bootstrap_replicates": 1000,
        "bootstrap_interval_type": "marginal patient-cluster percentile intervals",
        "preflight_counts": {"cache_rows_by_condition_by_db": {audit.TARGET_DB: cache_rows}},
        "source_mimic_split2_metric_validation": validation,
    }
    return pd.DataFrame(rows), run_manifest, pd.DataFrame(refs)


class NumericDecodingReleaseAuditTests(unittest.TestCase):
    def test_complete_60_cell_grid_passes(self):
        summary, run_manifest, source_reference = _synthetic_inputs()
        report = audit.audit_aggregate(summary, run_manifest, source_reference)
        self.assertEqual(report["row_count"], 60)
        self.assertEqual(report["successful_cells"], 60)
        self.assertEqual(report["source_mimic_split2_checks"], 30)

    def test_failed_or_missing_grid_cells_are_rejected(self):
        summary, run_manifest, source_reference = _synthetic_inputs()
        with self.assertRaisesRegex(audit.ReleaseAuditError, "60 rows"):
            audit.audit_aggregate(summary.iloc[:-1].copy(), run_manifest, source_reference)
        failed = summary.copy()
        failed.loc[0, "status"] = "nested_fit_unavailable"
        with self.assertRaisesRegex(audit.ReleaseAuditError, "failed or non-success"):
            audit.audit_aggregate(failed, run_manifest, source_reference)

    def test_mismatched_payload_hash_is_rejected(self):
        trusted = "a" * 64
        changed = "b" * 64
        with self.assertRaisesRegex(audit.ReleaseAuditError, "SHA256 values differ"):
            audit.assert_hash_match("synthetic cache", trusted, trusted, changed)

    def test_inconsistent_target_identity_across_conditions_is_rejected(self):
        summary, run_manifest, source_reference = _synthetic_inputs()
        summary.loc[0, "population_identity_sha256"] = "f" * 64
        with self.assertRaisesRegex(audit.ReleaseAuditError, "target identity/count field"):
            audit.audit_aggregate(summary, run_manifest, source_reference)


if __name__ == "__main__":
    unittest.main()
