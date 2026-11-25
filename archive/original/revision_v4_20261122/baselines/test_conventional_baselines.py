"""Synthetic checks for same-information conventional baseline features/grids."""
from __future__ import annotations

import unittest
import json
import tempfile
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_features import summarize_inputs
from conventional_baselines import (
    _candidate_specs, _complexity_key, _condition_specs_from_manifest,
    _hash_json, _summarize_conditioned_inputs, _unpaired_real_condition_specs,
    _effective_prediction_scope, final_source_fit_mask, main, run,
)
from paired_baselines import (
    INPUT_SCOPE, PRIMARY_SCOPE, compare_artifacts, ordered_row_digest,
    sha256_file, validate_complete_fm_prediction_manifest,
)


def _engine_prediction_manifest(records: list[dict]) -> dict:
    protocol = {
        "run_id": "synthetic-current-run", "mode": "predict",
        "models": ["main20k"], "cohorts": ["mimic"],
        "input_scope": INPUT_SCOPE,
        "clinical_input_availability_verified": False,
    }
    protocol_sha = _hash_json(protocol)
    protocol["run_protocol_sha256"] = protocol_sha
    return {
        "run_id": "synthetic-current-run", "mode": "predict",
        "run_protocol": protocol, "run_protocol_sha256": protocol_sha,
        "input_scope": INPUT_SCOPE,
        "clinical_input_availability_verified": False,
        "readiness": [{
            "db": "mimic", "status": "ready", "temporal_evidence": "exact",
            "prediction_scope": PRIMARY_SCOPE, "input_scope": INPUT_SCOPE,
            "clinical_input_availability_verified": False,
            "retrospective_landmark_prediction_ready": True,
        }],
        "retrospective_landmark_prediction_by_cohort": {"mimic": True},
        "real_time_available_landmark_prediction_by_cohort": {"mimic": False},
        "predictions": records,
    }


class BaselineFeatureTests(unittest.TestCase):
    def test_summary_bins_hours_counts_and_last_token_tie_break(self):
        ds = {
            "L": np.array([3, 2]),
            "V": np.array([[1, 1, 2], [2, 0, 0]]),
            "B": np.array([[1, 3, 10], [5, 0, 0]]),
            "H": np.array([[5, 5, 23], [4, 0, 0]]),
        }
        value, pattern = summarize_inputs(ds, np.array([0, 1]), 2)
        self.assertEqual(value.shape, (2, 18))
        self.assertEqual(pattern.shape, (2, 10))
        self.assertAlmostEqual(float(value[0, 5]), .3)
        self.assertAlmostEqual(float(value[0, 1]), 2 / 128)
        self.assertAlmostEqual(float(value[0, 2]), .2)
        self.assertAlmostEqual(float(value[0, 6]), 5 / 23)
        self.assertTrue(np.all(value[1, 0:9] == 0))
        self.assertTrue(np.all(pattern[1, 0:5] == 0))

    def test_grids_and_deterministic_simpler_ties(self):
        specs = _candidate_specs()
        self.assertEqual([x["C"] for x in specs["conventional_lr"]], [.01, .1, 1., 10.])
        self.assertEqual(len(specs["conventional_hgb"]), 4)
        self.assertEqual(specs["conventional_hgb"][0]["min_samples_leaf"], 20)
        self.assertLess(_complexity_key("conventional_lr", {"C": .01}),
                        _complexity_key("conventional_lr", {"C": .1}))
        self.assertLess(_complexity_key("conventional_hgb", {
            "max_leaf_nodes": 15, "l2_regularization": 10.
        }), _complexity_key("conventional_hgb", {
            "max_leaf_nodes": 31, "l2_regularization": 1.
        }))

    def test_final_source_fit_excludes_validation_and_test(self):
        train = np.array([True, True, False, False, False])
        validation = np.array([False, False, True, True, False])
        test = np.array([False, False, False, False, True])
        final = final_source_fit_mask(train, validation, test)
        np.testing.assert_array_equal(final, train)
        self.assertFalse(np.any(final & validation))
        self.assertFalse(np.any(final & test))

    def test_conditioned_features_use_engine_masked_arrays(self):
        ds = {
            "L": np.array([4]),
            "V": np.array([[1, 2, 2, 3]]),
            "B": np.array([[2, 6, 10, 8]]),
            "H": np.array([[0, 0, 4, 7]]),
        }

        class EngineMask:
            def prepare_batch(self, source, rows, condition, variable_ids):
                self.arguments = (condition, tuple(variable_ids))
                V, B, H = (np.asarray(source[key])[rows].copy()
                           for key in ("V", "B", "H"))
                remove = np.isin(V, np.asarray(variable_ids)) & (H == 0)
                V[remove] = 0
                B[remove] = 0
                H[remove] = 0
                return V, B, H

        engine = EngineMask()
        value, pattern = _summarize_conditioned_inputs(
            ds, np.array([0]), 3, condition="no_h0_therapy",
            variable_ids=(2,), engine=engine,
        )
        self.assertEqual(engine.arguments, ("no_h0_therapy", (2,)))
        # The zero-hour var-2 token is masked while its later token remains.
        self.assertAlmostEqual(float(value[0, 1 * 9 + 1]), 1 / 128)
        self.assertAlmostEqual(float(value[0, 1 * 9 + 2]), 1.0)
        self.assertAlmostEqual(float(value[0, 1 * 9 + 6]), 4 / 23)
        self.assertAlmostEqual(float(pattern[0, 1 * 5 + 1]), 1 / 128)

    def test_condition_specs_require_matching_per_cohort_no_h0_mapping(self):
        condition_hash = "c" * 64
        names = ["vaso", "rrt", "sedation"]
        ids_by_cohort = {
            "mimic": {"vaso": 46, "rrt": 47, "sedation": 48},
            "eicu": {"vaso": 146, "rrt": 147, "sedation": 148},
        }

        def record(db, condition):
            row = {
                "analysis_id": f"main20k_{condition}", "db": db,
                "model_tag": "main20k", "input_condition": condition,
                "mode": "predict", "input_mask_sha256": "d" * 64,
                "n_prediction_rows": 12, "role": "evaluation",
                "prediction_scope": "retrospective_landmark_prediction_measurement_time",
                "input_scope": "retrospective_measurement_time",
                "clinical_input_availability_verified": False,
                "retrospective_landmark_prediction_allowed": True,
            }
            if condition == "no_h0_therapy":
                spec = {
                    "condition": "no_h0_therapy", "canonical_variable_names": names,
                    "variable_ids": [ids_by_cohort[db][name] for name in names],
                    "hour": 0, "masked_features": ["V", "B", "H"],
                    "mask_action": "zero",
                    "stay_retention_rule": "retain iff any nonzero variable token remains",
                    "condition_definition_sha256": condition_hash, "db": db,
                }
                row["input_mask_spec"] = spec
                row["input_mask_sha256"] = _hash_json(spec)
            return row

        document = {
            "status": "complete", "mode": "predict",
            "no_h0_therapy_mapping": {
                "canonical_variable_names": names,
                "variable_ids_by_cohort": ids_by_cohort,
                "condition_definition_sha256": condition_hash,
            },
            "predictions": [record(db, condition)
                            for db in ("mimic", "eicu")
                            for condition in ("real", "no_h0_therapy")],
        }
        conditions, specs = _condition_specs_from_manifest(document, ("mimic", "eicu"))
        self.assertEqual(conditions, ["real", "no_h0_therapy"])
        self.assertEqual(specs["no_h0_therapy"]["eicu"]["variable_ids"], (146, 147, 148))
        self.assertEqual(specs["real"]["mimic"]["variable_ids"], ())

        document["predictions"] = [row for row in document["predictions"]
                                   if not (row["db"] == "eicu"
                                           and row["input_condition"] == "no_h0_therapy")]
        with self.assertRaisesRegex(ValueError, "no_h0_therapy is present"):
            _condition_specs_from_manifest(document, ("mimic", "eicu"))

    def test_unpaired_condition_specs_are_real_only(self):
        class Engine:
            @staticmethod
            def input_mask_hash(condition, variable_ids):
                self.assertEqual(condition, "real")
                self.assertEqual(variable_ids, ())
                return "a" * 64

        conditions, specs = _unpaired_real_condition_specs(Engine, ("mimic", "eicu"))
        self.assertEqual(conditions, ["real"])
        self.assertEqual(specs["real"]["eicu"]["analysis_id"], "main20k_real")
        self.assertEqual(specs["real"]["mimic"]["input_mask_sha256"], "a" * 64)
        self.assertEqual(specs["real"]["eicu"]["variable_ids"], ())

    def test_missing_prediction_scope_uses_registered_evidence_mapping(self):
        class Engine:
            @staticmethod
            def prediction_scope(evidence):
                if evidence not in {"exact", "operational_proxy"}:
                    raise ValueError("unsupported evidence")
                return PRIMARY_SCOPE

        entry = {"evidence": "operational_proxy"}
        self.assertEqual(_effective_prediction_scope(entry, Engine), PRIMARY_SCOPE)
        entry["prediction_scope"] = "retrospective_association_unverified_landmark"
        # An explicit non-primary scope is retained and will still fail the run gate.
        self.assertEqual(_effective_prediction_scope(entry, Engine),
                         "retrospective_association_unverified_landmark")

    def test_current_engine_manifest_requires_valid_hashed_readiness(self):
        record = {
            "analysis_id": "main20k_real", "db": "mimic",
            "model_tag": "main20k", "mode": "predict", "input_condition": "real",
            "input_mask_sha256": "d" * 64,
            "n_prediction_rows": 12, "role": "evaluation",
            "prediction_scope": PRIMARY_SCOPE, "input_scope": INPUT_SCOPE,
            "clinical_input_availability_verified": False,
            "retrospective_landmark_prediction_allowed": True,
        }
        document = _engine_prediction_manifest([record])
        validate_complete_fm_prediction_manifest(document)
        conditions, specs = _condition_specs_from_manifest(document, ("mimic",))
        self.assertEqual(conditions, ["real"])
        self.assertEqual(specs["real"]["mimic"]["analysis_id"], "main20k_real")

        changed = json.loads(json.dumps(document))
        changed["run_protocol"]["run_protocol_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "protocol hash is invalid"):
            validate_complete_fm_prediction_manifest(changed)

        unready = json.loads(json.dumps(document))
        unready["readiness"][0]["status"] = "blocked"
        with self.assertRaisesRegex(ValueError, "does not attest the registered scope"):
            validate_complete_fm_prediction_manifest(unready)

    def test_unpaired_run_requires_explicit_cli_flag(self):
        with self.assertRaisesRegex(ValueError, "explicit unpaired source-fit mode"):
            run(("mimic",), Path("eligibility"), Path("out"))

        with patch("conventional_baselines.run") as mocked_run:
            with self.assertRaises(SystemExit):
                main(["--cohorts", "mimic", "--out-dir", "out"])
            mocked_run.assert_not_called()

            main(["--cohorts", "mimic", "--out-dir", "out", "--allow-unpaired"])
            mocked_run.assert_called_once()
            self.assertTrue(mocked_run.call_args.kwargs["allow_unpaired"])


class PairedBaselineTests(unittest.TestCase):
    def _mask_spec(self, condition: str) -> dict | None:
        if condition == "real":
            return None
        return {
            "condition": "no_h0_therapy",
            "canonical_variable_names": ["vaso", "rrt", "sedation"],
            "variable_ids": [2, 3, 4], "hour": 0,
            "masked_features": ["V", "B", "H"], "mask_action": "zero",
            "stay_retention_rule": "retain iff any nonzero variable token remains",
            "condition_definition_sha256": "f" * 64, "db": "mimic",
        }

    def _frame(self, analysis_id: str, probabilities: list[float], *, rows=None,
               condition="real") -> pd.DataFrame:
        if rows is None:
            rows = [0, 1, 2, 3]
        mask_spec = self._mask_spec(condition)
        mask_hash = _hash_json(mask_spec) if mask_spec is not None else "d" * 64
        return pd.DataFrame({
            "db": ["mimic"] * 4, "analysis_id": [analysis_id] * 4,
            "stay_id": [f"s{i}" for i in range(4)],
            "subject": [f"p{i}" for i in range(4)], "rows": rows,
            "split": [2] * 4, "y": [0, 1, 0, 1], "p": probabilities,
            "endpoint": ["hospital"] * 4, "role": ["evaluation"] * 4,
            "prediction_scope": ["retrospective_landmark_prediction_measurement_time"] * 4,
            "input_scope": ["retrospective_measurement_time"] * 4,
            "input_condition": [condition] * 4,
            "input_mask_sha256": [mask_hash] * 4,
            "clinical_input_availability_verified": [False] * 4,
            "eligibility_manifest_sha256": ["a" * 64] * 4,
            "eligibility_table_sha256": ["b" * 64] * 4,
        })

    def _record(self, frame: pd.DataFrame, path: Path) -> dict:
        return {
            "analysis_id": str(frame["analysis_id"].iloc[0]), "db": "mimic",
            "model_tag": ("main20k" if str(frame["analysis_id"].iloc[0]).startswith("main20k")
                          else str(frame["analysis_id"].iloc[0]).split("_")[1]),
            "mode": "predict",
            "input_condition": str(frame["input_condition"].iloc[0]),
            "input_mask_sha256": str(frame["input_mask_sha256"].iloc[0]),
            "input_mask_spec": self._mask_spec(str(frame["input_condition"].iloc[0])),
            "fm_analysis_id": f"main20k_{frame['input_condition'].iloc[0]}",
            "prediction_file": str(path.resolve()), "prediction_file_sha256": sha256_file(path),
            "ordered_row_digest_sha256": ordered_row_digest(frame),
            "n_prediction_rows": len(frame), "role": "evaluation",
            "prediction_scope": "retrospective_landmark_prediction_measurement_time",
            "input_scope": "retrospective_measurement_time",
            "clinical_input_availability_verified": False,
            "eligibility_manifest_sha256": "a" * 64,
            "eligibility_table_sha256": "b" * 64,
            "source_npz_sha256": "c" * 64,
            "retrospective_landmark_prediction_allowed": True,
        }

    def test_paired_metrics_use_exact_population_and_shared_patient_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fm = self._frame("main20k_real", [.1, .8, .3, .7])
            fm_path = root / "main20k_real.parquet"
            fm.to_parquet(fm_path, index=False)
            fm_record = self._record(fm, fm_path)
            fm_manifest = root / "prediction_manifest.json"
            fm_manifest.write_text(json.dumps(_engine_prediction_manifest([fm_record])))

            records = []
            for analysis_id, probabilities in (
                ("conventional_lr_pattern_only", [.2, .6, .4, .55]),
                ("conventional_hgb_pattern_only", [.15, .75, .35, .65]),
            ):
                frame = self._frame(analysis_id, probabilities)
                path = root / f"{analysis_id}.parquet"
                frame.to_parquet(path, index=False)
                records.append(self._record(frame, path))
            baseline_manifest = root / "baseline_prediction_manifest.json"
            baseline_manifest.write_text(json.dumps({
                "status": "complete", "input_conditions": ["real", "no_h0_therapy"],
                "cohort_readiness": {
                    "mimic": {"retrospective_landmark_prediction_allowed": True}
                }, "predictions": records,
            }))

            no_h0_fm = self._frame("main20k_no_h0_therapy", [.12, .82, .34, .72],
                                   condition="no_h0_therapy")
            no_h0_fm_path = root / "main20k_no_h0_therapy.parquet"
            no_h0_fm.to_parquet(no_h0_fm_path, index=False)
            no_h0_fm_record = self._record(no_h0_fm, no_h0_fm_path)
            fm_manifest.write_text(json.dumps(_engine_prediction_manifest([
                fm_record, no_h0_fm_record,
            ])))

            for analysis_id, probabilities in (
                ("conventional_lr_pattern_only_no_h0_therapy", [.22, .62, .42, .57]),
                ("conventional_hgb_pattern_only_no_h0_therapy", [.18, .78, .38, .68]),
            ):
                frame = self._frame(analysis_id, probabilities, condition="no_h0_therapy")
                path = root / f"{analysis_id}.parquet"
                frame.to_parquet(path, index=False)
                record = self._record(frame, path)
                records.append(record)
            baseline_manifest.write_text(json.dumps({
                "status": "complete", "input_conditions": ["real", "no_h0_therapy"],
                "cohort_readiness": {
                    "mimic": {"retrospective_landmark_prediction_allowed": True}
                }, "predictions": records,
            }))

            audit, metrics, summary = compare_artifacts(
                baseline_manifest, fm_manifest, n_boot=1000, seed=17)
            self.assertEqual(len(audit), 4)
            self.assertEqual(len(metrics), 12)
            self.assertTrue(all(row["valid_paired_bootstrap_draws"] >= 500 for row in metrics))
            self.assertEqual(len({row["shared_bootstrap_weights_sha256"] for row in metrics}), 1)
            self.assertEqual(summary["delta_direction"], "FM minus conventional baseline")
            self.assertTrue(all(row["paired_subject_rows_split_labels_verified"] for row in audit))
            self.assertEqual({row["input_condition"] for row in metrics},
                             {"real", "no_h0_therapy"})
            self.assertEqual({row["fm_analysis_id"] for row in metrics},
                             {"main20k_real", "main20k_no_h0_therapy"})
            incomplete_manifest = root / "incomplete_baseline_manifest.json"
            incomplete_manifest.write_text(json.dumps({
                "status": "complete", "input_conditions": ["real", "no_h0_therapy"],
                "cohort_readiness": {
                    "mimic": {"retrospective_landmark_prediction_allowed": True}
                }, "predictions": [row for row in records
                                   if row["input_condition"] == "real"],
            }))
            with self.assertRaisesRegex(ValueError, "prediction groups differ"):
                compare_artifacts(incomplete_manifest, fm_manifest, n_boot=1000, seed=17)

    def test_pairing_fails_on_source_row_mismatch_even_with_current_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fm = self._frame("main20k_real", [.1, .8, .3, .7])
            fm_path = root / "main20k_real.parquet"
            fm.to_parquet(fm_path, index=False)
            fm_record = self._record(fm, fm_path)
            fm_manifest = root / "prediction_manifest.json"
            fm_manifest.write_text(json.dumps({"status": "complete", "predictions": [fm_record]}))

            mismatched = self._frame("conventional_lr_pattern_only", [.2, .6, .4, .55],
                                     rows=[0, 1, 99, 3])
            baseline_path = root / "baseline.parquet"
            mismatched.to_parquet(baseline_path, index=False)
            baseline_manifest = root / "baseline_prediction_manifest.json"
            baseline_manifest.write_text(json.dumps({
                "status": "complete", "cohort_readiness": {
                    "mimic": {"retrospective_landmark_prediction_allowed": True}
                }, "predictions": [self._record(mismatched, baseline_path)]
            }))
            with self.assertRaisesRegex(ValueError, "source rows mismatch"):
                compare_artifacts(baseline_manifest, fm_manifest, fm_path, n_boot=1000, seed=17)


if __name__ == "__main__":
    unittest.main()

