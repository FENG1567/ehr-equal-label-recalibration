import unittest
import runpy
import json
import hashlib
import itertools
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from run_recovered_metrics import (METRICS_CODE, extract_sequence_npz_hash,
                                   install_checkpointed_bootstrap, compute_shared_weighted_metrics,
                                   exact_population_digest)


class RecoveredGateSchemaTests(unittest.TestCase):
    def test_reads_exact_recovered_gate_sourcehash_schema(self):
        value = "a" * 64
        self.assertEqual(extract_sequence_npz_hash({"sourcehashes": {"sequence_npz": value}}), value)

    def test_does_not_guess_alternate_sourcehash_field_names(self):
        with self.assertRaisesRegex(ValueError, "sourcehashes.sequence_npz"):
            extract_sequence_npz_hash({"sourcehashes": {"sequence_npz_sha256": "a" * 64}})

    def test_rejects_non_object_sourcehash_container(self):
        with self.assertRaisesRegex(ValueError, "sourcehashes object"):
            extract_sequence_npz_hash({"sourcehashes": ["a" * 64]})

    def test_runpy_returns_namespace_separate_from_function_globals(self):
        scope = runpy.run_path(str(METRICS_CODE), run_name="paired_metrics_namespace_test")
        actual = scope["run"].__globals__["paired_cluster_bootstrap"]
        marker = object()
        scope["paired_cluster_bootstrap"] = marker
        self.assertIs(scope["run"].__globals__["paired_cluster_bootstrap"], actual)
        self.assertIsNot(scope["run"].__globals__["paired_cluster_bootstrap"], marker)

    def test_actual_paired_metrics_run_dispatches_checkpoint_shim(self):
        scope = runpy.run_path(str(METRICS_CODE), run_name="paired_metrics_checkpoint_test")
        with TemporaryDirectory() as temp:
            root = Path(temp)
            pred_dir = root / "predictions"
            output_dir = root / "output"
            pred_dir.mkdir()
            output_dir.mkdir()
            manifest_rows = []
            for analysis, probabilities in (("synthetic_a", [.1, .8, .3, .7]),
                                            ("synthetic_b", [.2, .7, .4, .6])):
                frame = pd.DataFrame({
                    "db": ["synthetic"] * 4, "analysis_id": [analysis] * 4,
                    "stay_id": [1, 2, 3, 4], "subject": ["p1", "p2", "p3", "p4"],
                    "rows": [0, 1, 2, 3], "split": [2] * 4,
                    "y": [0, 1, 0, 1], "p": probabilities,
                    "model_tag": ["synthetic"] * 4,
                    "input_condition": ["real"] * 4,
                    "endpoint": ["hospital"] * 4, "role": ["evaluation"] * 4,
                    "prediction_scope": ["retrospective_landmark_prediction_measurement_time"] * 4,
                    "input_scope": ["retrospective_measurement_time"] * 4,
                    "clinical_input_availability_verified": [False] * 4,
                    "eligibility_manifest_sha256": ["a" * 64] * 4,
                    "eligibility_table_sha256": ["b" * 64] * 4,
                    "temporal_evidence": ["exact"] * 4,
                })
                path = pred_dir / f"{analysis}.parquet"
                frame.to_parquet(path, index=False)
                manifest_rows.append({
                    "analysis_id": analysis, "db": "synthetic", "mode": "predict",
                    "prediction_file": str(path),
                    "prediction_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "ordered_row_digest_sha256": scope["sha256_ordered_prediction_rows"](frame),
                    "n_predictions": 4, "n_prediction_rows": 4, "role": "evaluation",
                    "input_scope": "retrospective_measurement_time",
                    "clinical_input_availability_verified": False,
                    "prediction_scope": "retrospective_landmark_prediction_measurement_time",
                    "retrospective_landmark_prediction_allowed": True,
                    "retrospective_landmark_prediction_ready": True,
                    "eligibility_manifest_sha256": "a" * 64,
                    "eligibility_table_sha256": "b" * 64,
                    "source_npz_sha256": "c" * 64,
                    "temporal_evidence": "exact",
                })
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"predictions": manifest_rows}))
            def synthetic_current_hashes(db):
                self.assertEqual(db, "synthetic")
                return {"eligibility_manifest_sha256": "a" * 64,
                        "eligibility_table_sha256": "b" * 64,
                        "source_npz_sha256": "c" * 64,
                        "retrospective_landmark_prediction_allowed": True}
            scope["current_eligibility_hashes"] = synthetic_current_hashes
            scope["run"].__globals__["current_eligibility_hashes"] = synthetic_current_hashes
            state = install_checkpointed_bootstrap(scope, output_dir)
            args = scope["parse_args"]([
                "--prediction-dir", str(pred_dir), "--manifest-path", str(manifest_path),
                "--output-dir", str(output_dir / "metrics"), "--analysis-ids",
                "synthetic_a", "synthetic_b", "--bootstrap", "1000", "--seed", "123",
            ])
            self.assertEqual(scope["run"](args), 0)
            self.assertEqual(state["completed"], 1)
            progress = json.loads((output_dir / "progress.json").read_text())
            self.assertEqual(progress["completed_draws_for_pair"], 1000)
            with np.load(output_dir / "bootstrap_checkpoint.npz") as checkpoint:
                self.assertEqual(int(checkpoint["completed_draws"]), 1000)
            metrics = pd.read_csv(output_dir / "metrics" / "paired_core_metrics.csv")
            self.assertEqual(len(metrics), 7)

    def test_grouped_draw_metrics_match_independent_metrics_for_two_rosters(self):
        scope = runpy.run_path(str(METRICS_CODE), run_name="paired_metrics_grouped_test")
        rosters = [
            {
                "ids": [1, 2, 3, 4, 5, 6],
                "subjects": ["p1", "p1", "p2", "p3", "p3", "p3"],
                "y": np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int8),
                "weights": np.asarray([2., 2., 0., 1., 1., 1.]),
                "probabilities": {
                    "a": np.asarray([.12, .44, .32, .82, .71, .62]),
                    "b": np.asarray([.22, .52, .27, .73, .66, .58]),
                    "c": np.asarray([.18, .37, .41, .77, .69, .54]),
                },
            },
            {
                "ids": [11, 12, 13, 14, 15],
                "subjects": ["q1", "q2", "q2", "q3", "q4"],
                "y": np.asarray([0, 1, 0, 1, 0], dtype=np.int8),
                "weights": np.asarray([0., 2., 2., 1., 3.]),
                "probabilities": {
                    "a": np.asarray([.21, .49, .29, .78, .64]),
                    "b": np.asarray([.16, .57, .38, .71, .59]),
                    "c": np.asarray([.25, .43, .34, .83, .68]),
                },
            },
        ]
        digests = []
        for roster in rosters:
            paired = pd.DataFrame({"stay_id": roster["ids"], "subject": roster["subjects"],
                                   "rows": np.arange(len(roster["ids"])), "split": [2] * len(roster["ids"]),
                                   "y": roster["y"], "p_a": roster["probabilities"]["a"],
                                   "p_b": roster["probabilities"]["b"]})
            digests.append(exact_population_digest(paired))
            grouped = compute_shared_weighted_metrics(roster["y"], roster["probabilities"],
                                                       roster["weights"], scope["weighted_metrics"])
            independent = {analysis: scope["weighted_metrics"](roster["y"], p, roster["weights"])
                           for analysis, p in roster["probabilities"].items()}
            for analysis in roster["probabilities"]:
                for metric in scope["METRICS"]:
                    if np.isnan(independent[analysis][metric]):
                        self.assertTrue(np.isnan(grouped[analysis][metric]))
                    else:
                        self.assertAlmostEqual(grouped[analysis][metric], independent[analysis][metric], places=14)
            for analysis_a, analysis_b in itertools.combinations(sorted(roster["probabilities"]), 2):
                for metric in scope["METRICS"]:
                    grouped_delta = grouped[analysis_b][metric] - grouped[analysis_a][metric]
                    independent_delta = independent[analysis_b][metric] - independent[analysis_a][metric]
                    if np.isnan(independent_delta):
                        self.assertTrue(np.isnan(grouped_delta))
                    else:
                        self.assertAlmostEqual(grouped_delta, independent_delta, places=14)
        self.assertNotEqual(digests[0], digests[1])


if __name__ == "__main__":
    unittest.main()
