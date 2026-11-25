"""Synthetic checks for registered vocabulary shape normalization."""
from __future__ import annotations

import unittest

import v4_engine_registry_maskfix_recovered_v4_2_resume_vocabadapter_21aae5dd as engine


THERAPY_NAMES = ["age", "vaso", "rrt", "sedation"]
EXPECTED_IDS = {"vaso": 2, "rrt": 3, "sedation": 4}


class RegisteredVocabularyAdapterTests(unittest.TestCase):
    def test_vars_list_and_direct_map_are_equivalent(self) -> None:
        as_vars = {"vars": THERAPY_NAMES}
        as_map = {"var_to_id": {name: i for i, name in enumerate(THERAPY_NAMES)}}
        self.assertEqual(engine.registered_no_h0_therapy_ids(as_vars, "test"), EXPECTED_IDS)
        self.assertEqual(engine.registered_no_h0_therapy_ids(as_map, "test"), EXPECTED_IDS)

    def test_availability_payload_order_and_map_match(self) -> None:
        metadata = {
            "variables_in_original_order": THERAPY_NAMES,
            "var_to_id": {name: i for i, name in enumerate(THERAPY_NAMES)},
            "token_value_ids_are_one_based": True,
        }
        self.assertEqual(engine.registered_no_h0_therapy_ids(metadata, "test"), EXPECTED_IDS)

    def test_rejects_colliding_direct_indices(self) -> None:
        metadata = {"var_to_id": {"age": 0, "vaso": 1, "rrt": 1, "sedation": 3}}
        with self.assertRaisesRegex(ValueError, "colliding"):
            engine.registered_no_h0_therapy_ids(metadata, "test")

    def test_rejects_noncontiguous_direct_indices(self) -> None:
        metadata = {"var_to_id": {"age": 0, "vaso": 2, "rrt": 3, "sedation": 4}}
        with self.assertRaisesRegex(ValueError, "contiguous and zero based"):
            engine.registered_no_h0_therapy_ids(metadata, "test")

    def test_rejects_missing_therapy_names(self) -> None:
        metadata = {"vars": ["age", "vaso", "rrt"]}
        with self.assertRaisesRegex(ValueError, "sedation"):
            engine.registered_no_h0_therapy_ids(metadata, "test")

    def test_rejects_disagreeing_order_and_direct_map(self) -> None:
        metadata = {
            "vars": THERAPY_NAMES,
            "var_to_id": {name: i for i, name in enumerate(reversed(THERAPY_NAMES))},
        }
        with self.assertRaisesRegex(ValueError, "differs from the registered variable order"):
            engine.registered_no_h0_therapy_ids(metadata, "test")


if __name__ == "__main__":
    unittest.main()
