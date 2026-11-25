"""Synthetic checks for the frozen INSPIRE encoder preflight."""
from __future__ import annotations

import copy
import unittest

import run_encode_inspire_v4_1_gpu_preflightfix3 as runner


def locked_inputs() -> tuple[dict, dict, dict]:
    gate = {
        "gate_version": runner.GATE.name,
        "analysis_role": "exploratory_external_only",
        "source_development_lock_sha256": runner.EXPECTED_DENSITY_LOCK_SHA256,
        "independent_data_review_sha256": runner.EXPECTED_DATA_REVIEW_SHA256,
        "cross_scope_limit": runner.EXPECTED_CROSS_SCOPE_LIMIT,
        "cohorts": {},
    }
    identity = {"cohorts": {}}
    for db in runner.COHORTS:
        table_sha = f"table-{db}"
        sequence_sha = f"sequence-{db}"
        gate["cohorts"][db] = {
            "confirmation_allowed": False,
            "source_patient_identity_verified": True,
            "patient_disjoint_claim_allowed": True,
            "fixed_patient_role_counts": copy.deepcopy(runner.EXPECTED_COUNT_BY_COHORT[db]),
            "pragmatic_confirmation_count_gate_met": False,
            "table_sha256": table_sha,
            "sourcehashes": {"sequence_npz": sequence_sha},
        }
        identity["cohorts"][db] = {
            "eligibility_manifest_sha256": runner.EXPECTED_GATE_MANIFEST_SHA256,
            "eligibility_table_sha256": table_sha,
            "scope_record_sha256": runner.EXPECTED_DATA_REVIEW_SHA256,
            "sequence_npz_sha256": sequence_sha,
            "subject_identity_basis": {
                "resampling_unit": "patient",
                "identifier_scope": "within this original data source",
                "patient_disjoint_claim_allowed": True,
                "cross_source_patient_independence_verified": False,
                "cross_scope_patient_independence_verified": False,
            }
        }
    role_lock = {
        "no_model_performance_used_for_role_assignment": True,
        "no_redraw_or_threshold_relaxation": True,
        "metric_population": "fixed assessment only for all model/calibration comparisons",
    }
    return gate, role_lock, identity


class InspireEncoderPreflightTests(unittest.TestCase):
    def test_component_contract_contains_twenty_unique_rows(self) -> None:
        keys = runner.expected_component_keys()
        self.assertEqual(len(keys), 20)

    def test_accepts_within_source_patient_resampling_and_exploratory_roles(self) -> None:
        runner.validate_locked_gate(*locked_inputs())

    def test_rejects_cross_scope_patient_independence_claim(self) -> None:
        gate, role_lock, identity = locked_inputs()
        identity["cohorts"][runner.COHORTS[0]]["subject_identity_basis"][
            "cross_scope_patient_independence_verified"
        ] = True
        with self.assertRaisesRegex(ValueError, "cross-scope patient independence"):
            runner.validate_locked_gate(gate, role_lock, identity)

    def test_rejects_gate_that_allows_confirmation(self) -> None:
        gate, role_lock, identity = locked_inputs()
        gate["cohorts"][runner.COHORTS[0]]["confirmation_allowed"] = True
        with self.assertRaisesRegex(ValueError, "incorrectly permits confirmation"):
            runner.validate_locked_gate(gate, role_lock, identity)

    def test_accepts_additive_fragment_hash_metadata_when_paths_match(self) -> None:
        central = {field: f"/source/{field}" for field in runner.REGISTRY_PATH_FIELDS}
        fragment = {**central, "sequence_sha256": "locked-sequence-hash"}
        runner.validate_registry_paths(runner.COHORTS[0], central, fragment)

    def test_rejects_registry_path_mismatch(self) -> None:
        central = {field: f"/source/{field}" for field in runner.REGISTRY_PATH_FIELDS}
        fragment = {**central, "split_path": "/source/other-split.npy"}
        with self.assertRaisesRegex(ValueError, "registry paths disagree"):
            runner.validate_registry_paths(runner.COHORTS[0], central, fragment)


if __name__ == "__main__":
    unittest.main()
