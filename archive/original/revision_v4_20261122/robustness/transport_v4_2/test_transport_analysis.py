from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from transport_analysis import (
    _cluster_bootstrap_loss_delta, _load_original_eicu_frame, _paired_target_predictions,
    _sampling_plan_for_frame,
)
from paired_auc_ap_ci import _cluster_bootstrap_auc_ap, _weighted_rank_metric_draws
from sklearn.metrics import average_precision_score, roc_auc_score


class TransportAnalysisTests(unittest.TestCase):
    def test_paired_target_adapter_normalizes_stay_id_dtype_and_checks_identity(self):
        fm = pd.DataFrame({
            "db": ["sicdb_zero_offset_v4_2", "sicdb_zero_offset_v4_2"],
            "stay_id": [101, 102], "subject": ["p1", "p2"],
            "rows": [1, 2], "split": [2, 2], "y": [0, 1], "p": [0.1, 0.8],
        })
        baseline = pd.DataFrame({
            "db": ["sicdb_zero_offset_v4_2", "sicdb_zero_offset_v4_2"],
            "stay_id": ["101", "102"], "subject": ["p1", "p2"],
            "rows": [1, 2], "split": [2, 2], "y": [0, 1], "p": [0.2, 0.7],
        })
        paired = _paired_target_predictions(fm, baseline, db="sicdb_zero_offset_v4_2")
        self.assertEqual(len(paired), 2)
        self.assertListEqual(paired["stay_id"].tolist(), ["101", "102"])

        baseline.loc[0, "subject"] = "different"
        with self.assertRaisesRegex(ValueError, "identity/label mismatch in subject"):
            _paired_target_predictions(fm, baseline, db="sicdb_zero_offset_v4_2")

    def test_shared_loss_bootstrap_returns_paired_patient_interval(self):
        frame = pd.DataFrame({
            "subject": ["p1", "p2", "p3", "p4"],
            "y": [0, 1, 1, 0],
            "p_fm": [0.1, 0.7, 0.6, 0.3],
            "p_baseline": [0.2, 0.8, 0.4, 0.2],
        })
        result = _cluster_bootstrap_loss_delta({"baseline": frame}, seed=17, n_boot=1000)
        self.assertEqual(result["baseline"]["brier"]["bootstrap_draws"], 1000)
        self.assertEqual(result["baseline"]["logloss"]["valid_draws"], 1000)
        self.assertLessEqual(result["baseline"]["brier"]["ci_95_lo"],
                             result["baseline"]["brier"]["ci_95_hi"])

    def test_sampling_plan_is_independent_of_prediction_probabilities(self):
        count = 160
        frame = pd.DataFrame({
            "db": ["synthetic"] * count,
            "stay_id": [f"s{index}" for index in range(count)],
            "subject": [f"p{index}" for index in range(count)],
            "rows": np.arange(count), "split": np.full(count, 2),
            "y": np.tile([0, 1], count // 2),
            "p": np.linspace(0.01, 0.99, count),
            "eligible_24h": np.ones(count, dtype=bool),
            "temporal_verified": np.ones(count, dtype=bool),
        })
        first, counts = _sampling_plan_for_frame(frame, db="synthetic")
        changed = frame.copy()
        changed["p"] = np.linspace(0.99, 0.01, count)
        second, _ = _sampling_plan_for_frame(changed, db="synthetic")
        self.assertEqual(first["sampling_plan_sha256"], second["sampling_plan_sha256"])
        self.assertEqual(counts["patient_overlap"], 0)
        self.assertIn(50, counts["supported_budgets"])

    def test_plan_hash_changes_with_cohort_key_and_recovers_after_alias_normalization(self):
        count = 120
        frame = pd.DataFrame({
            "db": ["eicu_semantic_v4_2"] * count,
            "stay_id": [f"s{index}" for index in range(count)],
            "subject": [f"p{index}" for index in range(count)],
            "rows": np.arange(count), "split": np.full(count, 2),
            "y": np.tile([0, 1], count // 2), "p": np.full(count, 0.5),
            "eligible_24h": np.ones(count, dtype=bool),
            "temporal_verified": np.ones(count, dtype=bool),
        })
        repaired_plan, _ = _sampling_plan_for_frame(frame, db="eicu_semantic_v4_2")
        canonical = frame.copy()
        canonical["db"] = "eicu"
        original_key_plan, _ = _sampling_plan_for_frame(canonical, db="eicu")
        self.assertNotEqual(repaired_plan["sampling_plan_sha256"],
                            original_key_plan["sampling_plan_sha256"])
        self.assertEqual(repaired_plan["adaptation_patient_set_sha256"],
                         original_key_plan["adaptation_patient_set_sha256"])

    def test_original_eicu_loader_accepts_ready_engine_manifest_contract(self):
        manifest = {
            "run_id": "status_consistent_primary_core_fix2",
            "readiness": [{
                "db": "eicu", "status": "ready",
                "retrospective_landmark_prediction_ready": True,
            }],
            "predictions": [{
                "analysis_id": "main20k_real", "db": "eicu", "mode": "predict",
                "model_tag": "main20k", "input_condition": "real",
                "retrospective_landmark_prediction_allowed": True,
                "retrospective_landmark_prediction_ready": True,
                "prediction_file": "/tmp/original_eicu_predictions.parquet",
            }],
        }
        expected = pd.DataFrame({"stay_id": ["s1"], "y": [1], "p": [0.8]})
        with patch("transport_analysis._json", return_value=manifest), patch(
            "transport_analysis.load_checked", return_value=({"eicu": expected}, {})
        ) as checked:
            observed = _load_original_eicu_frame()
        pd.testing.assert_frame_equal(observed, expected)
        self.assertEqual(checked.call_args.kwargs["analysis_id"], "main20k_real")
        self.assertEqual(checked.call_args.kwargs["input_condition"], "real")

    def test_weighted_rank_metric_draws_match_sklearn_with_ties(self):
        y = np.array([0, 1, 0, 1, 0, 1], dtype=int)
        p = np.array([0.2, 0.8, 0.2, 0.8, 0.5, 0.5], dtype=float)
        weights = np.array([[1, 1, 1, 1, 1, 1], [2, 1, 0, 2, 1, 3]], dtype=float)
        auc, ap = _weighted_rank_metric_draws(y, p, weights)
        for index, row_weights in enumerate(weights):
            self.assertAlmostEqual(auc[index], roc_auc_score(y, p, sample_weight=row_weights))
            self.assertAlmostEqual(ap[index], average_precision_score(y, p, sample_weight=row_weights))

    def test_shared_rank_bootstrap_reports_1000_patient_draws(self):
        first = pd.DataFrame({
            "db": ["synthetic"] * 6,
            "stay_id": ["s1", "s2", "s3", "s4", "s5", "s6"],
            "subject": ["p1", "p2", "p3", "p4", "p5", "p6"],
            "rows": np.arange(1, 7), "split": [2] * 6,
            "y": [0, 1, 0, 1, 0, 1],
            "p_fm": [0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
            "p_baseline": [0.3, 0.7, 0.4, 0.8, 0.2, 0.5],
        })
        second = first.copy()
        second["p_baseline"] = [0.4, 0.6, 0.2, 0.9, 0.1, 0.5]
        result = _cluster_bootstrap_auc_ap(
            {"baseline_a": first, "baseline_b": second}, seed=19, n_boot=1000,
        )
        self.assertEqual(result["draws"], 1000)
        self.assertGreater(result["one_class_draws"], 0)
        for comparison in result["comparisons"].values():
            expected_valid = 1000 - result["one_class_draws"]
            self.assertEqual(comparison["auroc"]["valid_draws"], expected_valid)
            self.assertEqual(comparison["average_precision"]["valid_draws"], expected_valid)


if __name__ == "__main__":
    unittest.main()
