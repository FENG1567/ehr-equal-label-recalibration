import tempfile
import unittest
import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from no_mg_sensitivity import (atomic_json, paired_bootstrap, prepare_no_mg,
                               sklearn_metric_crosscheck, weighted_metrics)


class NoMgSensitivityTests(unittest.TestCase):
    def test_drops_magnesium_and_sets_constant_only_on_remaining_tokens(self):
        V = np.array([[1, 2, 2, 0], [1, 0, 0, 0]], dtype=np.int64)
        B = np.array([[4, 3, 8, 0], [2, 0, 0, 0]], dtype=np.int64)
        H = np.array([[1, 2, 3, 0], [1, 0, 0, 0]], dtype=np.int64)
        L = np.array([3, 1], dtype=np.int64)
        v, b, h, b1, diag = prepare_no_mg(V, B, H, L, mg_id=2)
        np.testing.assert_array_equal(v, [[1, 0, 0, 0], [1, 0, 0, 0]])
        np.testing.assert_array_equal(b, [[4, 0, 0, 0], [2, 0, 0, 0]])
        np.testing.assert_array_equal(h, [[1, 0, 0, 0], [1, 0, 0, 0]])
        np.testing.assert_array_equal(b1, [[1, 0, 0, 0], [1, 0, 0, 0]])
        self.assertEqual(diag["n_mg_tokens_omitted"], 2)
        self.assertEqual(diag["n_rows_without_non_mg_tokens"], 0)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            atomic_json(path, {"diagnostics": diag})
            self.assertEqual(json.loads(path.read_text())["diagnostics"], diag)

    def test_empty_non_mg_row_is_reported(self):
        V = np.array([[2, 2, 0]], dtype=np.int64)
        B = np.array([[4, 5, 0]], dtype=np.int64)
        H = np.array([[1, 2, 0]], dtype=np.int64)
        _, _, _, b1, diag = prepare_no_mg(V, B, H, np.array([2]), mg_id=2)
        self.assertEqual(diag["n_rows_without_non_mg_tokens"], 1)
        np.testing.assert_array_equal(b1, np.zeros_like(B))

    def test_bootstrap_uses_shared_draws_and_exact_unit_label(self):
        y = np.array([0, 1, 0, 1], dtype=np.int8)
        p0 = np.array([0.1, 0.9, 0.2, 0.8])
        p1 = np.array([0.2, 0.8, 0.1, 0.9])
        out = paired_bootstrap(y, p0, p1, np.array(["R1", "R2", "R3", "R4"]),
                               seed=5, reps=20, unit="record_id_proxy")
        self.assertEqual(out["resampling_unit"], "record_id_proxy")
        self.assertEqual(out["n_resampling_clusters"], 4)
        self.assertTrue(out["draws_shared_between_conditions"])
        self.assertEqual(out["bootstrap_replicates"], 20)

    def test_weighted_metrics_match_sklearn_for_rank_edge_cases(self):
        checked = sklearn_metric_crosscheck()
        self.assertEqual(checked["status"], "passed")
        self.assertEqual(checked["n_cases"], 5)
        self.assertLessEqual(checked["max_absolute_difference_vs_sklearn"], 1e-12)
        y = np.array([0, 1, 0, 1], dtype=np.int8)
        perfect = weighted_metrics(y, np.array([0.1, 0.9, 0.2, 0.8]), np.ones(4))
        inverse = weighted_metrics(y, np.array([0.9, 0.1, 0.8, 0.2]), np.ones(4))
        null = weighted_metrics(y, np.full(4, 0.5), np.ones(4))
        self.assertEqual(perfect["auroc"], 1.0)
        self.assertEqual(inverse["auroc"], 0.0)
        self.assertEqual(null["auroc"], 0.5)

    def test_unicode_ids_survive_non_pickle_serialization(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ids.npz"
            np.savez(path, ids=np.array(["Rα", "Rβ"], dtype=str))
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(archive["ids"].dtype.kind, "U")
                self.assertEqual(archive["ids"].tolist(), ["Rα", "Rβ"])


if __name__ == "__main__":
    unittest.main()
