"""Synthetic-only checks for v4 identity and eligibility guards."""
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import v4_engine as engine


class V4EngineTests(unittest.TestCase):
    def setUp(self):
        self.ds = {
            "L": np.array([3, 2]),
            "V": np.array([[1, 9, 0], [2, 0, 0]]),
            "B": np.array([[4, 5, 0], [6, 0, 0]]),
            "H": np.array([[1, 2, 0], [3, 0, 0]]),
        }

    def test_constant_preserves_variable_time_and_padding(self):
        V, B, H = engine.prepare_batch(self.ds, np.array([0, 1]), "constant")
        np.testing.assert_array_equal(V, self.ds["V"])
        np.testing.assert_array_equal(H, self.ds["H"])
        np.testing.assert_array_equal(B, np.array([[1, 1, 0], [1, 0, 0]]))

    def test_common8_mask_zeroes_excluded_tokens_and_metadata(self):
        ids = tuple(range(1, 9))
        V, B, H = engine.prepare_batch(self.ds, np.array([0, 1]),
                                       "common8_mask", ids)
        np.testing.assert_array_equal(V, np.array([[1, 0, 0], [2, 0, 0]]))
        np.testing.assert_array_equal(B, np.array([[4, 0, 0], [6, 0, 0]]))
        np.testing.assert_array_equal(H, np.array([[1, 0, 0], [3, 0, 0]]))
        with self.assertRaises(ValueError):
            engine.prepare_batch(self.ds, np.array([0]), "common8_mask", (1, 2))

    def test_common8_row_mask_excludes_empty_inputs_before_encoding(self):
        ds = dict(self.ds)
        ds["L"] = np.array([3, 2, 1])
        ds["V"] = np.vstack([self.ds["V"], np.array([99, 0, 0])])
        rows = engine.condition_rows(ds, "common8_mask", tuple(range(1, 9)))
        np.testing.assert_array_equal(rows, np.array([0, 1]))

    def test_no_h0_therapy_masks_only_canonical_tokens_at_h0(self):
        ids = (46, 47, 48)
        ds = {
            "L": np.array([3, 2, 2, 1]),
            "V": np.array([
                [46, 48, 1],
                [46, 47, 0],
                [1, 48, 0],
                [47, 0, 0],
            ]),
            "B": np.array([
                [4, 5, 6],
                [7, 8, 0],
                [9, 10, 0],
                [11, 0, 0],
            ]),
            "H": np.array([
                [0, 1, 0],
                [0, 0, 0],
                [1, 0, 0],
                [0, 0, 0],
            ]),
            "stay_id": np.array([101, 102, 103, 104]),
            "subject": np.array(["p1", "p2", "p3", "p4"]),
            "y_mort": np.array([0, 1, 0, 1]),
        }
        rows = engine.condition_rows(ds, "no_h0_therapy", ids)
        np.testing.assert_array_equal(rows, np.array([0, 2]))
        V, B, H = engine.prepare_inputs(ds, rows, "no_h0_therapy", ids)
        np.testing.assert_array_equal(V, np.array([[0, 48, 1], [1, 0, 0]]))
        np.testing.assert_array_equal(B, np.array([[0, 5, 6], [9, 0, 0]]))
        np.testing.assert_array_equal(H, np.array([[0, 1, 0], [1, 0, 0]]))
        eligible = pd.DataFrame({
            "stay_id": [101, 102, 103, 104],
            "subject": ["p1", "p2", "p3", "p4"],
            "y_mort": [0, 1, 0, 1],
        })
        aligned = engine.align_eligible_rows(
            "mimic", ds, eligible, rows, "no_h0_therapy", ids)
        self.assertListEqual(aligned["stay_id"].tolist(), [101, 103])
        with self.assertRaises(ValueError):
            engine.prepare_inputs(ds, np.array([1]), "no_h0_therapy", ids)

    def test_no_h0_therapy_mapping_reads_each_cohort_vocabulary(self):
        with tempfile.TemporaryDirectory() as tmp:
            processed = Path(tmp) / "data" / "processed"
            processed.mkdir(parents=True)
            variables = ["age", "vaso", "rrt", "sedation"]
            for db in ("mimic", "eicu"):
                (processed / f"meta_{db}.json").write_text(
                    json.dumps({"vars": variables}))
            mapping = engine.load_no_h0_therapy_ids(
                Path(tmp), cohorts=("mimic", "eicu"))
        expected = {"vaso": 2, "rrt": 3, "sedation": 4}
        self.assertEqual(mapping, {"mimic": expected, "eicu": expected})
        spec = engine.no_h0_therapy_input_mask_spec("mimic", (2, 3, 4))
        self.assertEqual(
            spec["condition_definition_sha256"],
            engine.no_h0_therapy_definition_sha256())
        self.assertEqual(engine.canonical_json_sha256(spec),
                         engine.canonical_json_sha256(dict(spec)))

    def test_cache_round_trip_checks_rows_ids_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "embedding.npz"
            rows = np.array([1, 4], dtype=np.int64)
            stays = np.array([101, 104])
            subjects = np.array(["p1", "p2"])
            source_y = np.array([0, 1], dtype=np.float32)
            embedding = np.arange(6, dtype=np.float32).reshape(2, 3)
            runtime = {
                "python_version": "3.11.16", "torch_version": "2.11.0+cu128",
                "encoding_device": "cuda", "encoding_dtype": "float32",
                "model_parameter_dtypes": ["torch.float32"], "torch_num_threads": 8,
            }
            runtime_hash = engine.canonical_json_sha256(runtime)
            metadata = {
                "checkpoint_sha256": "abc", "input_condition": "real",
                "encoder_code_sha256": "encoder", "source_npz_sha256": "source",
                "engine_code_sha256": "full-script-v1", "torch_version": runtime["torch_version"],
                "encoding_device": "cuda", "encoding_dtype": "float32",
                "python_version": runtime["python_version"],
                "model_parameter_dtypes": ["torch.float32"], "torch_num_threads": 8,
                "runtime_identity": runtime, "runtime_identity_sha256": runtime_hash,
            }
            engine.save_embedding_cache(path, rows, stays, subjects, source_y,
                                        embedding, metadata)
            loaded = engine.load_embedding_cache(path, rows, stays, subjects,
                                                 source_y, metadata)
            np.testing.assert_array_equal(loaded, embedding)
            edited_prediction_gate = dict(metadata, engine_code_sha256="full-script-v2")
            engine.load_embedding_cache(path, rows, stays, subjects, source_y,
                                        edited_prediction_gate)
            cpu_runtime = dict(runtime, encoding_device="cpu")
            changed_runtime = dict(
                metadata, encoding_device="cpu", runtime_identity=cpu_runtime,
                runtime_identity_sha256=engine.canonical_json_sha256(cpu_runtime))
            with self.assertRaises(ValueError):
                engine.load_embedding_cache(path, rows, stays, subjects, source_y,
                                            changed_runtime)
            with self.assertRaises(ValueError):
                engine.load_embedding_cache(path, rows, np.array([101, 999]),
                                            subjects, source_y, metadata)
            with self.assertRaises(ValueError):
                engine.load_embedding_cache(path, rows, stays, subjects, source_y,
                                            {"checkpoint_sha256": "wrong"})
            path_cpu = engine.cache_path(Path(tmp), "mimic", "main20k", "real",
                                         "c" * 64, "d" * 64, "e" * 64,
                                         "f" * 64, "1" * 64, "cpu" * 22)
            path_gpu = engine.cache_path(Path(tmp), "mimic", "main20k", "real",
                                         "c" * 64, "d" * 64, "e" * 64,
                                         "f" * 64, "1" * 64, "gpu" * 22)
            self.assertNotEqual(path_cpu, path_gpu)

    def test_eligibility_requires_exact_verified_binary_rows(self):
        frame = pd.DataFrame({
            "db": ["mimic"] * 4,
            "stay_id": [10, 11, 12, 13],
            "subject": ["a", "b", "c", "d"],
            "eligible_24h": [True, False, False, False],
            "temporal_verified": [True, False, True, True],
            "y_mort": [0, 1, np.nan, 1],
            "endpoint": ["icu"] * 4,
            "reason": ["exact"] * 4,
        })
        selected = engine.validate_eligibility_table("mimic", frame)
        self.assertListEqual(selected["stay_id"].tolist(), [10])
        unverified = frame.copy()
        unverified.loc[1, "eligible_24h"] = True
        with self.assertRaises(engine.ClinicalReadinessError):
            engine.validate_eligibility_table("mimic", unverified)
        nonbinary = frame.copy()
        nonbinary.loc[2, "eligible_24h"] = True
        with self.assertRaises(engine.ClinicalReadinessError):
            engine.validate_eligibility_table("mimic", nonbinary)
        duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with self.assertRaises(engine.ClinicalReadinessError):
            engine.validate_eligibility_table("mimic", duplicate)

    def alignment_fixture(self):
        ds = {
            "L": np.array([2, 2, 2]),
            "V": np.array([[1, 9], [9, 9], [2, 9]]),
            "B": np.ones((3, 2), dtype=np.int64),
            "H": np.array([[0, 1], [0, 1], [0, 1]]),
            "stay_id": np.array([101, 102, 103]),
            "subject": np.array(["p1", "p2", "p3"]),
            "y_mort": np.array([0, 1, 0]),
            "split": np.array([0, 2, 2]),
        }
        eligible = pd.DataFrame({
            "db": ["mimic"] * 3,
            "stay_id": [101, 102, 103],
            "subject": ["p1", "p2", "p3"],
            "eligible_24h": [True] * 3,
            "temporal_verified": [True] * 3,
            "y_mort": [0, 1, 0],
            "endpoint": ["hospital"] * 3,
            "reason": ["exact"] * 3,
        })
        return ds, eligible

    def test_alignment_checks_all_audited_ids_labels_before_common8_drop(self):
        ds, eligible = self.alignment_fixture()
        common_ids = (1, 2, 3, 4, 5, 6, 7, 8)
        rows = engine.condition_rows(ds, "common8_mask", common_ids)
        matched = engine.align_eligible_rows(
            "mimic", ds, eligible, rows, "common8_mask", common_ids)
        self.assertListEqual(matched["stay_id"].tolist(), [101, 103])

        changed = eligible.copy()
        changed.loc[1, "y_mort"] = 0
        with self.assertRaisesRegex(engine.ClinicalReadinessError, "outcome labels differ"):
            engine.align_eligible_rows(
                "mimic", ds, changed, rows, "common8_mask", common_ids)

    def test_alignment_rejects_missing_ids_and_unexpected_condition_loss(self):
        ds, eligible = self.alignment_fixture()
        missing = pd.concat([eligible, eligible.iloc[[0]].assign(stay_id=999)],
                            ignore_index=True)
        with self.assertRaisesRegex(engine.ClinicalReadinessError, "absent from the source"):
            engine.align_eligible_rows("mimic", ds, missing, np.array([0, 1, 2]))
        with self.assertRaisesRegex(engine.ClinicalReadinessError, "unexpectedly excluded"):
            engine.align_eligible_rows("mimic", ds, eligible, np.array([0, 2]), "real")

    def test_source_sequence_hash_attestation_is_required_and_exact(self):
        entry = {"sequence_npz_sha256": "a" * 64}
        engine.assert_sequence_npz_attestation("mimic", entry, "a" * 64)
        with self.assertRaises(engine.ClinicalReadinessError):
            engine.assert_sequence_npz_attestation("mimic", entry, "b" * 64)

    def test_probe_artifact_is_scoped_by_eligibility_hash_and_source_ids(self):
        base = {
            "run_id": "status_consistent_primary_core_fix1",
            "model_tag": "main20k", "input_condition": "real",
            "checkpoint_sha256": "a" * 64, "input_mask_sha256": "b" * 64,
            "eligibility_manifest_sha256": "c" * 64,
            "eligibility_table_sha256": "d" * 64,
            "source_npz_sha256": "e" * 64,
            "runtime_identity_sha256": "0" * 64,
            "source_train_identity_sha256": "f" * 64,
            "source_train_label_sha256": "1" * 64,
            "prediction_scope": "retrospective_landmark_prediction_measurement_time",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path_a = engine.source_probe_artifact_path(base, Path(tmp))
            path_b = engine.source_probe_artifact_path(
                dict(base, eligibility_manifest_sha256="2" * 64), Path(tmp))
            path_c = engine.source_probe_artifact_path(
                dict(base, source_train_identity_sha256="3" * 64), Path(tmp))
            path_d = engine.source_probe_artifact_path(
                dict(base, run_id="status_consistent_primary_core_fix2"), Path(tmp))
            self.assertNotEqual(path_a, path_b)
            self.assertNotEqual(path_a, path_c)
            self.assertNotEqual(path_a, path_d)
            payload = {"scope": base["prediction_scope"]}
            engine.atomic_pickle_dump(payload, path_a)
            with path_a.open("rb") as handle:
                self.assertEqual(pickle.load(handle), payload)
            self.assertFalse(list(Path(tmp).glob("*.tmp")))

    def test_source_probe_mask_compatibility_accepts_cohort_specific_no_h0_ids(self):
        source_spec = engine.no_h0_therapy_input_mask_spec("mimic", (2, 3, 4))
        target_spec = engine.no_h0_therapy_input_mask_spec("eicu", (5, 6, 7))
        source_hash = engine.canonical_json_sha256(source_spec)
        target_hash = engine.canonical_json_sha256(target_spec)
        self.assertNotEqual(source_hash, target_hash)
        fitted = {
            "input_mask_sha256": source_hash,
            "input_mask_spec": source_spec,
        }
        self.assertTrue(engine.source_probe_mask_compatible(
            fitted, "no_h0_therapy", target_hash, target_spec))

    def test_source_probe_mask_compatibility_rejects_bad_no_h0_attestations(self):
        source_spec = engine.no_h0_therapy_input_mask_spec("mimic", (2, 3, 4))
        target_spec = engine.no_h0_therapy_input_mask_spec("eicu", (5, 6, 7))
        source_hash = engine.canonical_json_sha256(source_spec)
        target_hash = engine.canonical_json_sha256(target_spec)
        fitted = {"input_mask_sha256": source_hash, "input_mask_spec": source_spec}
        self.assertFalse(engine.source_probe_mask_compatible(
            {"input_mask_sha256": source_hash},
            "no_h0_therapy", target_hash, target_spec))
        self.assertFalse(engine.source_probe_mask_compatible(
            fitted, "no_h0_therapy", target_hash, None))
        bad_definition = dict(target_spec, condition_definition_sha256="0" * 64)
        self.assertFalse(engine.source_probe_mask_compatible(
            fitted, "no_h0_therapy",
            engine.canonical_json_sha256(bad_definition), bad_definition))
        changed_rule = dict(target_spec, hour=1)
        self.assertFalse(engine.source_probe_mask_compatible(
            fitted, "no_h0_therapy",
            engine.canonical_json_sha256(changed_rule), changed_rule))
        misaligned_names = dict(
            target_spec, canonical_variable_names=["vaso", "sedation", "rrt"])
        self.assertFalse(engine.source_probe_mask_compatible(
            fitted, "no_h0_therapy",
            engine.canonical_json_sha256(misaligned_names), misaligned_names))
        self.assertFalse(engine.source_probe_mask_compatible(
            dict(fitted, input_mask_sha256="0" * 64),
            "no_h0_therapy", target_hash, target_spec))

    def test_source_probe_mask_compatibility_keeps_exact_hash_for_other_conditions(self):
        fitted = {"input_mask_sha256": "a" * 64}
        self.assertTrue(engine.source_probe_mask_compatible(
            fitted, "common8_mask", "a" * 64, {"anything": True}))
        self.assertFalse(engine.source_probe_mask_compatible(
            fitted, "common8_mask", "b" * 64, {"anything": True}))

    def test_prediction_run_directory_is_protocol_scoped(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = engine.prediction_run_directory({
                "run_id": "strict_v4_1", "run_protocol_sha256": "a" * 64,
            }, Path(tmp))
            second = engine.prediction_run_directory({
                "run_id": "strict_v4_1", "run_protocol_sha256": "b" * 64,
            }, Path(tmp))
            self.assertNotEqual(first, second)
            self.assertEqual(first.name, "strict_v4_1-" + "a" * 16)
            with self.assertRaises(ValueError):
                engine.prediction_run_directory({
                    "run_id": "../bad", "run_protocol_sha256": "a" * 64,
                }, Path(tmp))

    def test_manifest_gate_fails_closed_for_unavailable_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "eligibility_manifest.json").write_text(json.dumps({
                "cohorts": {"mimic": {"evidence": "unavailable", "status": "ready"}}
            }))
            # Create a placeholder table path; evidence is checked before parquet reads.
            (path / "mimic.parquet").touch()
            with self.assertRaises(engine.ClinicalReadinessError):
                engine.load_ready_eligibility("mimic", path)

    def test_manifest_requires_positive_retrospective_landmark_attestation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "eligibility_manifest.json").write_text(json.dumps({
                "cohorts": {"mimic": {"evidence": "exact", "status": "ready"}}
            }))
            (path / "mimic.parquet").touch()
            with self.assertRaises(engine.ClinicalReadinessError):
                engine.load_ready_eligibility("mimic", path)

    def test_legacy_clinical_claim_does_not_unlock_retrospective_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "eligibility_manifest.json").write_text(json.dumps({
                "cohorts": {"mimic": {
                    "evidence": "exact", "status": "ready",
                    "ready_for_future24h_clinical_claim": True,
                    "sourcehashes": {"sequence_npz": "a" * 64},
                }}
            }))
            (path / "mimic.parquet").touch()
            with self.assertRaisesRegex(engine.ClinicalReadinessError,
                                        "landmark readiness attestation"):
                engine.load_ready_eligibility("mimic", path)

    def test_manifest_requires_audited_sequence_npz_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "eligibility_manifest.json").write_text(json.dumps({
                "cohorts": {"mimic": {
                    "evidence": "exact", "status": "ready",
                    "retrospective_landmark_prediction_allowed": True,
                }}
            }))
            (path / "mimic.parquet").touch()
            with self.assertRaisesRegex(engine.ClinicalReadinessError,
                                        "sourcehashes.sequence_npz"):
                engine.load_ready_eligibility("mimic", path)

    def test_operational_proxy_is_scoped_and_attested(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            frame = pd.DataFrame({
                "db": ["mimic"], "stay_id": [10], "subject": ["p1"],
                "eligible_24h": [True], "temporal_verified": [True],
                "y_mort": [1], "endpoint": ["hospital"], "reason": ["discharge status"]
            })
            (path / "eligibility_manifest.json").write_text(json.dumps({
                "cohorts": {"mimic": {
                    "evidence": "operational_proxy", "status": "ready",
                    "retrospective_landmark_prediction_allowed": True,
                    "ready_for_future24h_clinical_claim": True,
                    "sourcehashes": {"sequence_npz": "a" * 64},
                    "evidence_basis": "still in ICU at hour 24 from discharge status"
                }}
            }))
            (path / "mimic.parquet").touch()
            with patch.object(engine.pd, "read_parquet", return_value=frame):
                selected, entry = engine.load_ready_eligibility("mimic", path)
            self.assertEqual(len(selected), 1)
            self.assertFalse(entry["ready_for_future24h_clinical_claim"])
            self.assertTrue(entry["retrospective_landmark_prediction_allowed"])
            self.assertEqual(engine.prediction_scope(entry["evidence"]),
                             "retrospective_landmark_prediction_measurement_time")

    def test_source_probe_fits_only_source_mask(self):
        X = np.array([[0., 1.], [1., 0.], [2., 0.], [100., 100.]])
        y = np.array([0, 0, 1, 1], dtype=np.int8)
        train = np.array([True, True, True, False])
        scaler, probe = engine.fit_source_probe(X, y, train)
        self.assertTrue(np.allclose(scaler.mean_, X[train].mean(axis=0)))
        self.assertEqual(int(probe.n_features_in_), 2)
        with self.assertRaises(engine.ClinicalReadinessError):
            engine.fit_source_probe(X, y, np.array([False, False, True, True]))


if __name__ == "__main__":
    unittest.main()
