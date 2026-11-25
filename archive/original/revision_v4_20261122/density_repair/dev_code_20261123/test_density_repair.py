import unittest
import tempfile
import hashlib
import json
from pathlib import Path

import numpy as np

from density_repair import (BASE_SEED, derived_seed, equal_per_stay_view_weights,
                            fit_training_scaler, partition_masks,
                            select_min_validation, thin_triplets,
                            validate_run_dir,
                            sha256_file, sha256_ids, write_selection_npz)
from encode_views import load_selection_for_encoding


class DensityRepairTests(unittest.TestCase):
    def test_thinning_is_deterministic_outcome_blind_and_triplet_aligned(self):
        v = np.array([11, 12, 13, 14, 0, 0], dtype=np.int64)
        b = np.array([21, 22, 23, 24, 0, 0], dtype=np.int64)
        h = np.array([31, 32, 33, 34, 0, 0], dtype=np.int64)
        seed = derived_seed("train_aug_view_0", "stay-a", BASE_SEED)
        first = thin_triplets(v, b, h, 6, 0.5, seed)
        repeat = thin_triplets(v, b, h, 6, 0.5, seed)
        self.assertEqual(first[3], 2)
        self.assertEqual(first[3], repeat[3])
        for left, right in zip(first[:3], repeat[:3]):
            np.testing.assert_array_equal(left, right)
        keep = first[0] != 0
        self.assertEqual(int(keep.sum()), first[3])
        np.testing.assert_array_equal(first[1][keep], b[keep])
        np.testing.assert_array_equal(first[2][keep], h[keep])
        self.assertTrue(np.all(first[1][~keep] == 0))
        self.assertTrue(np.all(first[2][~keep] == 0))
        self.assertTrue(np.all(first[0][4:] == 0))

    def test_minimum_one_token_and_namespace_changes_seed(self):
        v = np.array([5, 0, 0], dtype=np.int32)
        b = np.array([7, 0, 0], dtype=np.int32)
        h = np.array([9, 0, 0], dtype=np.int32)
        thinned = thin_triplets(v, b, h, 1, 0.25,
                                derived_seed("report_retention_0.25", "one"))
        self.assertEqual(thinned[3], 1)
        self.assertEqual(int(np.count_nonzero(thinned[0])), 1)
        self.assertNotEqual(derived_seed("train", "one"),
                            derived_seed("validation", "one"))

    def test_view_weights_sum_to_one_per_stay(self):
        weights = equal_per_stay_view_weights(4, 4)
        np.testing.assert_allclose(weights.reshape(4, 4).sum(axis=0), 1.0)

    def test_partition_masks_are_exact_and_train_scaler_ignores_other_splits(self):
        split = np.array([0, 0, 1, 1, 2, 2])
        train, tune, report = partition_masks(split)
        np.testing.assert_array_equal(train, [1, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(tune, [0, 0, 1, 1, 0, 0])
        np.testing.assert_array_equal(report, [0, 0, 0, 0, 1, 1])
        X = np.array([[1.0], [3.0], [100.0], [-100.0], [1e6], [-1e6]])
        scaler = fit_training_scaler(X, train)
        self.assertEqual(float(scaler.mean_[0]), 2.0)
        X_changed = X.copy()
        X_changed[~train] *= 1000
        scaler_changed = fit_training_scaler(X_changed, train)
        np.testing.assert_array_equal(scaler.mean_, scaler_changed.mean_)
        np.testing.assert_array_equal(scaler.scale_, scaler_changed.scale_)

    def test_validation_selection_uses_smaller_c_on_tie(self):
        chosen = select_min_validation((0.01, 0.1, 1.0, 10.0),
                                       {0.01: 0.2, 0.1: 0.2, 1.0: 0.3, 10.0: 0.4})
        self.assertEqual(chosen, 0.01)

    def test_prepare_to_encoder_selection_load_uses_same_run_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory).resolve()
            path = run_dir / "selection.npz"
            rows = np.array([1, 7], dtype=np.int64)
            stays = np.array(["stay-a", "stay-b"], dtype=object)
            subjects = np.array(["patient-a", "patient-b"], dtype=object)
            write_selection_npz(
                path, rows=rows, stay_id=stays, subject=subjects,
                y=np.array([0, 1], dtype=np.int8), split=np.array([0, 2], dtype=np.int8),
                token_count=np.array([3, 1], dtype=np.int32),
                real=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32))
            identity = {
                "run_directory": str(run_dir),
                "selection_npz_sha256": sha256_file(path),
                "selected_rows_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
                "selected_identity_sha256": sha256_ids("mimic", rows, stays, subjects),
                "external_outcomes_or_INSPIRE_scores_read": False,
            }
            (run_dir / "input_identity.json").write_text(json.dumps(identity))
            loaded_identity, selection = load_selection_for_encoding(run_dir)
            self.assertEqual(loaded_identity["run_directory"], str(run_dir))
            self.assertEqual(selection["stay_id"].dtype.kind, "U")
            self.assertEqual(selection["subject"].dtype.kind, "U")
            np.testing.assert_array_equal(selection["stay_id"], ["stay-a", "stay-b"])
            np.testing.assert_array_equal(selection["subject"], ["patient-a", "patient-b"])
            np.testing.assert_array_equal(selection["rows"], rows)

    def test_run_directory_must_be_a_dedicated_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "density_repair"
            run = root / "source_run"
            run.mkdir(parents=True)
            self.assertEqual(validate_run_dir(run, root), run.resolve())
            with self.assertRaises(ValueError):
                validate_run_dir(root, root)
            with self.assertRaises(ValueError):
                validate_run_dir(Path(directory) / "outside", root)


if __name__ == "__main__":
    unittest.main()
