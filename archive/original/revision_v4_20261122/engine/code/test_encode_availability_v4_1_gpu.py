"""Synthetic authorization and key-set checks for availability encoding."""
from __future__ import annotations

import unittest

import run_encode_availability_v4_1_gpu as runner


def locked_inputs() -> tuple[dict, dict]:
    gate = {
        "gate_version": runner.GATE.name,
        "scope_decision": "TEST_ONLY",
        "score_only_authorization": True,
        "analysis_role": "same_original_heldout_mimic_test_only_score_sensitivity",
        "clinical_input_availability_verified": False,
        "ready_for_future24h_clinical_claim": False,
        "scope_record_sha256": runner.EXPECTED_REVIEW_SHA256,
        "cohorts": {},
    }
    identity = {"cohorts": {}}
    for db in runner.COHORTS:
        table_sha = f"table-{db}"
        sequence_sha = runner.EXPECTED_SOURCE_HASHES[db]
        gate["cohorts"][db] = {
            "status": "reviewed_ready",
            "score_only_authorization": True,
            "same_original_heldout_patients": True,
            "source_fit_test_subjects_disjoint_verified": True,
            "clinical_input_availability_verified": False,
            "ready_for_future24h_clinical_claim": False,
            "independent_external_cohort": False,
            "selection_by24_verified": runner.EXPECTED_SELECTION_BY24[db],
            "n_rows": 94343,
            "n_original_heldout_split2_patients": 10848,
            "table_sha256": table_sha,
            "sourcehashes": {
                "eligibility_table": table_sha,
                "sequence_npz": sequence_sha,
                "split_npy": runner.EXPECTED_SPLIT_SHA256,
            },
        }
        identity["cohorts"][db] = {
            "eligibility_manifest_sha256": runner.EXPECTED_GATE_SHA256,
            "eligibility_table_sha256": table_sha,
            "scope_record_sha256": runner.EXPECTED_REVIEW_SHA256,
            "sequence_npz_sha256": sequence_sha,
            "subject_identity_basis": {
                "resampling_unit": "patient",
                "identifier_scope": "within this original data source",
                "patient_disjoint_claim_allowed": True,
                "cross_source_patient_independence_verified": False,
                "cross_scope_patient_independence_verified": False,
            },
        }
    return gate, identity


class AvailabilityEncoderPreflightTests(unittest.TestCase):
    def test_component_contract_has_eight_rows(self) -> None:
        self.assertEqual(len(runner.expected_component_keys()), 8)

    def test_accepts_test_only_scope_and_within_source_patient_basis(self) -> None:
        runner.validate_gate(*locked_inputs())

    def test_rejects_clinical_availability_claim(self) -> None:
        gate, identity = locked_inputs()
        gate["clinical_input_availability_verified"] = True
        with self.assertRaisesRegex(ValueError, "must remain unverified"):
            runner.validate_gate(gate, identity)

    def test_rejects_cross_scope_independence_claim(self) -> None:
        gate, identity = locked_inputs()
        identity["cohorts"][runner.COHORTS[0]]["subject_identity_basis"][
            "cross_scope_patient_independence_verified"
        ] = True
        with self.assertRaisesRegex(ValueError, "cross-scope patient independence"):
            runner.validate_gate(gate, identity)


if __name__ == "__main__":
    unittest.main()
