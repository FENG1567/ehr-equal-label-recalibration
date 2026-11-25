"""Focused unit checks for direct calibrated FM/baseline inference helpers."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_calibrated_pairs as pairs  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


class CalibratedPairHelperTests(unittest.TestCase):
    def test_patient_bootstrap_keeps_stay_weighting_with_unequal_cluster_sizes(self) -> None:
        assessment = pd.DataFrame({"subject": ["p1", "p1", "p2", "p3", "p3", "p3"]})
        contrasts = np.array([[0.0], [0.0], [1.0], [3.0], [3.0], [3.0]])
        point, low, high = pairs._cluster_bootstrap_contrasts(assessment, contrasts, seed=91)
        self.assertAlmostEqual(point[0], 10.0 / 6.0)
        self.assertNotAlmostEqual(point[0], (0.0 + 1.0 + 3.0) / 3.0)
        self.assertTrue(np.isfinite(low[0]) and np.isfinite(high[0]))

    def test_loss_is_averaged_per_repeat_before_averaging_probabilities(self) -> None:
        repeat_probabilities = np.array([0.9, 0.1])
        brier, _ = pairs._losses(np.ones(2), repeat_probabilities)
        mean_repeat_loss = float(brier.mean())
        loss_of_mean_probability, _ = pairs._losses(np.array([1.0]),
                                                     np.array([repeat_probabilities.mean()]))
        self.assertAlmostEqual(mean_repeat_loss, 0.41)
        self.assertAlmostEqual(float(loss_of_mean_probability[0]), 0.25)
        self.assertNotAlmostEqual(mean_repeat_loss, float(loss_of_mean_probability[0]))

    def test_failed_fit_uses_jeffreys_probability_and_records_failure(self) -> None:
        fit = pairs._fit_operational("offset", np.zeros(4, dtype=int),
                                     np.array([-1.0, 0.0, 0.5, 1.0]),
                                     np.array([-0.25, 0.25]), 4)
        self.assertFalse(fit["fit_ok"])
        self.assertIn("single_class", fit["fit_status"])
        self.assertAlmostEqual(fit["fallback_probability"], 0.1)
        np.testing.assert_allclose(fit["p"], np.full(2, 0.1))

    def test_conflicted_repeat_draw_hash_is_rejected(self) -> None:
        plan = {"draws": [{"budget": 50, "repeat": 0,
                           "draw_index_sha256": "a", "patient_index_stay_sha256": "b"}]}
        rows = pd.DataFrame([
            {"method": "raw", "budget": 50, "repeat": 0,
             "adaptation_sample_index_sha256": "a",
             "adaptation_sample_patient_index_stay_sha256": "b"},
            {"method": "raw", "budget": 50, "repeat": 0,
             "adaptation_sample_index_sha256": "changed",
             "adaptation_sample_patient_index_stay_sha256": "b"},
        ])
        with self.assertRaisesRegex(ValueError, "conflicted existing repeat hashes"):
            pairs._verify_plan_draw_rows(rows, plan, "fixture", "fixture_model")

    def test_replay_fixture_matches_retained_metrics_and_rejects_drift(self) -> None:
        y = np.array([0, 1, 1, 0])
        fit = pairs._fit_operational("offset", np.array([0, 1, 1, 0]),
                                     np.array([-1.0, 1.0, 0.5, -0.5]),
                                     np.array([-0.7, 0.7, 0.2, -0.2]), 4)
        brier, logloss = pairs._losses(y, fit["p"])
        retained = {"fit_ok": "True", "fit_status": "ok",
                    "is_operational_fallback": "False",
                    "brier": float(brier.mean()), "logloss": float(logloss.mean())}
        self.assertEqual(pairs._as_bool(retained["fit_ok"]), fit["fit_ok"])
        self.assertEqual(retained["fit_status"], fit["fit_status"])
        self.assertEqual(pairs._as_bool(retained["is_operational_fallback"]), False)
        pairs._check_metric(retained["brier"], float(brier.mean()), label="fixture/brier")
        pairs._check_metric(retained["logloss"], float(logloss.mean()), label="fixture/logloss")
        with self.assertRaisesRegex(ValueError, "differs from retained"):
            pairs._check_metric(retained["brier"] + 0.01, float(brier.mean()), label="fixture/drift")


if __name__ == "__main__":
    unittest.main(verbosity=2)
