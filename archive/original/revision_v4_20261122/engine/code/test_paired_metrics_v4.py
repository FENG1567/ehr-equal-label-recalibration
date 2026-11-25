"""Synthetic-only checks for paired v4 metrics and identity gates."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import v4_engine as engine
import paired_metrics_v4 as metrics


def predictions(analysis_id, stays, subjects, labels, probabilities):
    return pd.DataFrame({
        "db": ["synthetic"] * len(stays),
        "analysis_id": [analysis_id] * len(stays),
        "stay_id": stays,
        "subject": subjects,
        "rows": np.asarray(stays) - 1,
        "split": [2] * len(stays),
        "y": labels,
        "p": probabilities,
        "model_tag": ["synthetic_model"] * len(stays),
        "input_condition": ["real"] * len(stays),
        "endpoint": ["hospital"] * len(stays),
        "role": ["evaluation"] * len(stays),
        "prediction_scope": ["retrospective_landmark_prediction_measurement_time"] * len(stays),
        "input_scope": ["retrospective_measurement_time"] * len(stays),
        "clinical_input_availability_verified": [False] * len(stays),
        "eligibility_manifest_sha256": ["a" * 64] * len(stays),
        "eligibility_table_sha256": ["b" * 64] * len(stays),
        "temporal_evidence": ["exact"] * len(stays),
    })


class PairedMetricsTests(unittest.TestCase):
    def test_pair_rows_intersects_by_stay_and_checks_subject_label(self):
        a = predictions("a", [1, 2, 3], ["p1", "p2", "p3"], [0, 1, 0], [.1, .8, .3])
        b = predictions("b", [2, 3, 4], ["p2", "p3", "p4"], [1, 0, 1], [.7, .2, .9])
        paired, flow = metrics.pair_rows(a, b, "synthetic", "hospital")
        self.assertEqual(len(paired), 2)
        self.assertEqual(flow["n_excluded_a"], 1)
        self.assertEqual(flow["n_excluded_b"], 1)
        b.loc[b["stay_id"] == 2, "subject"] = "wrong-patient"
        with self.assertRaises(ValueError):
            metrics.pair_rows(a, b, "synthetic", "hospital")

    def test_weighted_metrics_report_discrimination_and_probability_metrics(self):
        y = np.array([0, 0, 1, 1], dtype=np.int8)
        p = np.array([.1, .4, .6, .9])
        result = metrics.weighted_metrics(y, p)
        self.assertAlmostEqual(result["auroc"], 1.0)
        self.assertAlmostEqual(result["brier"], np.mean((y - p) ** 2))
        self.assertGreater(result["logloss"], 0)
        self.assertTrue(np.isfinite(result["offset_intercept"]))

    def test_offset_and_slope_fit_use_patient_weights(self):
        y = np.array([0, 0, 1, 1], dtype=np.int8)
        p = np.array([.1, .3, .7, .9])
        slope, intercept, offset = metrics.calibration_parameters(
            y, p, np.array([1., 2., 1., 2.]))
        self.assertTrue(np.isfinite(slope))
        self.assertTrue(np.isfinite(intercept))
        self.assertTrue(np.isfinite(offset))

    def test_bootstrap_requires_at_least_1000_patient_draws(self):
        paired = pd.DataFrame({
            "subject": ["p1", "p2"], "y": [0, 1],
            "p_a": [.2, .8], "p_b": [.3, .7],
        })
        with self.assertRaises(ValueError):
            metrics.paired_cluster_bootstrap(paired, n_boot=999)

    def test_validate_predictions_rejects_duplicate_stays(self):
        frame = predictions("a", [1, 1], ["p1", "p1"], [0, 0], [.1, .2])
        with self.assertRaises(ValueError):
            metrics.validate_predictions(frame, "a")

    def test_validate_predictions_rejects_real_time_claim(self):
        frame = predictions("a", [1], ["p1"], [0], [.1])
        frame["clinical_input_availability_verified"] = True
        with self.assertRaisesRegex(ValueError, "real-time input availability"):
            metrics.validate_predictions(frame, "a")

    def test_ordered_digest_matches_engine_and_detects_reordering(self):
        frame = predictions("a", [1, 2], ["p1", "p2"], [0, 1], [.1, .8])
        self.assertEqual(metrics.sha256_ordered_prediction_rows(frame),
                         engine.sha256_ordered_prediction_rows(frame))
        reversed_frame = frame.iloc[::-1].reset_index(drop=True)
        self.assertNotEqual(metrics.sha256_ordered_prediction_rows(frame),
                            metrics.sha256_ordered_prediction_rows(reversed_frame))

    def test_manifest_entry_checks_file_and_current_eligibility_hashes(self):
        frame = predictions("a", [1], ["p1"], [0], [.1])
        hashes = {
            "eligibility_manifest_sha256": "a" * 64,
            "eligibility_table_sha256": "b" * 64,
            "source_npz_sha256": "c" * 64,
            "retrospective_landmark_prediction_allowed": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.parquet"
            path.write_bytes(b"attested file")
            entry = {
                "analysis_id": "a", "db": "synthetic",
                "mode": "predict",
                "prediction_file": str(path),
                "prediction_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "ordered_row_digest_sha256": metrics.sha256_ordered_prediction_rows(frame),
                "n_predictions": 1, "n_prediction_rows": 1, "role": "evaluation",
                "input_scope": "retrospective_measurement_time",
                "clinical_input_availability_verified": False,
                "prediction_scope": "retrospective_landmark_prediction_measurement_time",
                "retrospective_landmark_prediction_allowed": True,
                "retrospective_landmark_prediction_ready": True,
                **hashes,
            }
            metrics.validate_manifest_entry(entry, "a", "synthetic", path, hashes)
            diagnostic = dict(entry, mode="diagnostic",
                              retrospective_landmark_prediction_allowed=False)
            with self.assertRaisesRegex(ValueError, "diagnostic associations"):
                metrics.validate_manifest_entry(diagnostic, "a", "synthetic", path, hashes)
            stale = dict(hashes, eligibility_table_sha256="d" * 64)
            with self.assertRaisesRegex(ValueError, "stale current eligibility/source hash"):
                metrics.validate_manifest_entry(entry, "a", "synthetic", path, stale)


if __name__ == "__main__":
    unittest.main()
