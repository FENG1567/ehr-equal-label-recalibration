from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from frozen_source_transport import (
    _exact_prediction_match, _exact_target_identity_match, _source_choice_map,
    _validate_subject_identity_attestations,
)


class FrozenTransportTests(unittest.TestCase):
    @staticmethod
    def _write_identity_fixture(root: Path) -> tuple[Path, Path]:
        gate_dir = root / "gate"
        gate_dir.mkdir()
        (gate_dir / "cohort.parquet").write_bytes(b"eligibility-table")
        sequence_path = root / "sequence.npz"
        sequence_path.write_bytes(b"sequence-array")
        review_path = root / "review.md"
        review_path.write_text("independent patient-ID review", encoding="utf-8")
        manifest = {
            "cohorts": {
                "cohort": {
                    "source_patient_identity_verified": True,
                    "patient_disjoint_claim_allowed": True,
                    "scope_record_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
                    "sourcehashes": {
                        "sequence_npz": hashlib.sha256(sequence_path.read_bytes()).hexdigest(),
                    },
                },
            },
        }
        manifest_path = gate_dir / "eligibility_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        record = {
            "eligibility_manifest_path": str(manifest_path.resolve()),
            "eligibility_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "eligibility_table_sha256": hashlib.sha256((gate_dir / "cohort.parquet").read_bytes()).hexdigest(),
            "sequence_path": str(sequence_path.resolve()),
            "sequence_npz_sha256": hashlib.sha256(sequence_path.read_bytes()).hexdigest(),
            "scope_record_path": str(review_path.resolve()),
            "scope_record_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
            "metadata_relation": "explicit identity supplement; original gate and prediction bytes untouched",
            "subject_identity_basis": {
                "basis": "reviewed source patient identifier",
                "patient_disjoint_claim_allowed": True,
                "identifier_scope": "within this original data source",
                "resampling_unit": "patient",
                "cross_source_patient_independence_verified": False,
                "cross_scope_patient_independence_verified": False,
            },
        }
        sidecar = {
            "schema": "ehr_fm_v4_2_explicit_subject_identity_attestations",
            "status": "root_accepted_within_source_patient_keys_only",
            "forbidden_inference": "C2012 RecordID is a record_id_proxy.",
            "cohorts": {"cohort": record},
        }
        sidecar_path = root / "attestations.json"
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
        return sidecar_path, gate_dir

    def test_source_choice_adapter_derives_registered_analysis_ids(self):
        rows = [
            {
                "model_tag": "conventional_lr", "feature_type": "pattern_only",
                "selected_parameters": {"C": 0.01},
                "final_model_fit_scope": "MIMIC source split 0 only; split 1 is selection-only and not refit",
                "selected_parameters_recovered_from_run_log": True,
            },
            {
                "model_tag": "conventional_hgb", "feature_type": "pattern_only",
                "selected_parameters": {
                    "learning_rate": 0.1, "max_iter": 200, "early_stopping": False,
                    "max_leaf_nodes": 15, "l2_regularization": 1.0, "min_samples_leaf": 20,
                },
                "final_model_fit_scope": "MIMIC source split 0 only; split 1 is selection-only and not refit",
                "selected_parameters_recovered_from_run_log": True,
            },
            {
                "model_tag": "conventional_lr", "feature_type": "value_plus_pattern",
                "selected_parameters": {"C": 0.01},
                "final_model_fit_scope": "MIMIC source split 0 only; split 1 is selection-only and not refit",
                "selected_parameters_recovered_from_run_log": True,
            },
            {
                "model_tag": "conventional_hgb", "feature_type": "value_plus_pattern",
                "selected_parameters": {
                    "learning_rate": 0.1, "max_iter": 200, "early_stopping": False,
                    "max_leaf_nodes": 15, "l2_regularization": 10.0, "min_samples_leaf": 20,
                },
                "final_model_fit_scope": "MIMIC source split 0 only; split 1 is selection-only and not refit",
                "selected_parameters_recovered_from_run_log": True,
            },
        ]
        mapped = _source_choice_map({"choices": rows})
        self.assertEqual(set(mapped), {
            "conventional_lr_pattern_only", "conventional_hgb_pattern_only",
            "conventional_lr_value_plus_pattern", "conventional_hgb_value_plus_pattern",
        })

    def test_source_reproduction_requires_exact_identity_and_probability_tolerance(self):
        columns = {
            "db": ["mimic", "eicu"], "stay_id": ["m1", "e1"],
            "subject": ["p1", "p2"], "rows": [0, 1], "split": [2, 0],
            "y": [0, 1], "p": [0.2, 0.7],
        }
        generated = pd.DataFrame(columns)
        reference = pd.DataFrame({**columns, "p": [0.2 + 1e-10, 0.7 - 1e-10]})
        result = _exact_prediction_match(generated, reference, analysis_id="synthetic")
        self.assertEqual(result["status"], "pass")
        self.assertLessEqual(result["max_abs_probability_difference"], 1e-8)

        wrong_label = reference.copy()
        wrong_label.loc[0, "y"] = 1
        with self.assertRaisesRegex(ValueError, "differ in y"):
            _exact_prediction_match(generated, wrong_label, analysis_id="synthetic")

        wrong_probability = reference.copy()
        wrong_probability.loc[0, "p"] = 0.20001
        with self.assertRaisesRegex(ValueError, "target scoring is blocked"):
            _exact_prediction_match(generated, wrong_probability, analysis_id="synthetic")

    def test_target_identity_adapter_normalizes_numeric_stay_ids_before_exact_join(self):
        sequence = pd.DataFrame({
            "db": ["sicdb_zero_offset_v4_2"], "stay_id": [101],
            "subject": ["patient-1"], "rows": [0], "split": [2], "y": [1],
        })
        fm = pd.DataFrame({
            "db": ["sicdb_zero_offset_v4_2"], "stay_id": ["101"],
            "subject": ["patient-1"], "rows": [0], "split": [2], "y": [1],
        })
        _exact_target_identity_match(sequence, fm, db="sicdb_zero_offset_v4_2")
        fm.loc[0, "y"] = 0
        with self.assertRaisesRegex(ValueError, "identity/label mismatch in y"):
            _exact_target_identity_match(sequence, fm, db="sicdb_zero_offset_v4_2")

        collision = pd.DataFrame({
            "db": ["sicdb_zero_offset_v4_2", "sicdb_zero_offset_v4_2"],
            "stay_id": pd.Series([1, "1"], dtype=object),
            "subject": ["patient-1", "patient-2"], "rows": [0, 1],
            "split": [2, 2], "y": [1, 0],
        })
        with self.assertRaisesRegex(ValueError, "after string normalization"):
            _exact_target_identity_match(collision, collision, db="sicdb_zero_offset_v4_2")

    def test_identity_sidecar_requires_all_original_artifact_bindings(self):
        with tempfile.TemporaryDirectory() as temporary:
            sidecar_path, gate_dir = self._write_identity_fixture(Path(temporary))
            records, sidecar_sha = _validate_subject_identity_attestations(
                sidecar_path=sidecar_path, eligibility_dir=gate_dir,
                cohorts=("cohort",), expected_sidecar_sha256=None,
            )
            self.assertIn("cohort", records)
            self.assertEqual(records["cohort"]["subject_identity_basis"]["resampling_unit"], "patient")
            self.assertEqual(len(sidecar_sha), 64)

            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["cohorts"]["cohort"]["sequence_npz_sha256"] = "0" * 64
            sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sequence NPZ hash is stale"):
                _validate_subject_identity_attestations(
                    sidecar_path=sidecar_path, eligibility_dir=gate_dir,
                    cohorts=("cohort",), expected_sidecar_sha256=None,
                )

    def test_identity_sidecar_does_not_allow_cross_scope_or_false_patient_basis(self):
        with tempfile.TemporaryDirectory() as temporary:
            sidecar_path, gate_dir = self._write_identity_fixture(Path(temporary))
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["cohorts"]["cohort"]["subject_identity_basis"][
                "cross_scope_patient_independence_verified"] = True
            sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cross-scope patient independence"):
                _validate_subject_identity_attestations(
                    sidecar_path=sidecar_path, eligibility_dir=gate_dir,
                    cohorts=("cohort",), expected_sidecar_sha256=None,
                )


if __name__ == "__main__":
    unittest.main()
