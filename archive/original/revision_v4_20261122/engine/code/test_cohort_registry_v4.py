"""Synthetic tests for the revision-owned cohort source registry."""
import json
import tempfile
import unittest
from pathlib import Path

from cohort_registry_v4 import (
    CohortRegistryError,
    DEFAULT_COHORTS,
    PROC,
    registered_cohorts,
    resolve_cohort_paths,
)


class CohortRegistryTests(unittest.TestCase):
    def test_legacy_paths_remain_the_default(self):
        paths = resolve_cohort_paths("mimic")
        self.assertEqual(paths.sequence_path, PROC / "seq_mimic.npz")
        self.assertEqual(paths.split_path, PROC / "split_mimic.npy")
        self.assertEqual(paths.tabular_path, PROC / "tab_mimic.parquet")
        self.assertEqual(paths.vocab_path, PROC / "meta_mimic.json")
        self.assertEqual(registered_cohorts(), tuple(sorted(DEFAULT_COHORTS)))

    def test_new_cohort_uses_only_explicit_revision_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = {
                "sequence_path": root / "inputs" / "seq_x.npz",
                "split_path": root / "inputs" / "split_x.npy",
                "tabular_path": root / "inputs" / "tab_x.parquet",
                "vocab_path": root / "inputs" / "meta_x.json",
            }
            for path in files.values():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            registry = root / "data_registry.json"
            registry.write_text(json.dumps({
                "cohorts": {"x_new": {key: str(value) for key, value in files.items()}}
            }))
            self.assertIn("x_new", registered_cohorts(registry))
            resolved = resolve_cohort_paths("x_new", registry, verify_exists=True)
            self.assertEqual(resolved.sequence_path, files["sequence_path"])
            self.assertEqual(resolved.split_path, files["split_path"])
            self.assertEqual(resolved.tabular_path, files["tabular_path"])
            self.assertEqual(resolved.vocab_path, files["vocab_path"])

    def test_relative_paths_are_resolved_from_registry_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "seq.npz").touch()
            (root / "split.npy").touch()
            registry = root / "data_registry.json"
            registry.write_text(json.dumps({
                "x_new": {"sequence_path": "seq.npz", "split_path": "split.npy"}
            }))
            resolved = resolve_cohort_paths("x_new", registry, verify_exists=True)
            self.assertEqual(resolved.sequence_path, root / "seq.npz")
            self.assertIsNone(resolved.tabular_path)

    def test_unknown_cohort_cannot_silently_use_legacy_paths(self):
        with self.assertRaisesRegex(CohortRegistryError, "not registered"):
            resolve_cohort_paths("unknown")

    def test_new_cohort_requires_sequence_and_split_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "data_registry.json"
            registry.write_text(json.dumps({"x_new": {"tabular_path": "tab.parquet"}}))
            with self.assertRaisesRegex(CohortRegistryError, "sequence_path"):
                resolve_cohort_paths("x_new", registry)


if __name__ == "__main__":
    unittest.main()
