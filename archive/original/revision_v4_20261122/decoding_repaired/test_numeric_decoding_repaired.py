"""Synthetic-only tests for v4 numeric decoding identity and fold safeguards."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import numeric_decoding_repaired as decoding


class NumericDecodingTests(unittest.TestCase):
    def _cache(self, db: str, analysis_id: str, rows: np.ndarray,
               stay_ids: np.ndarray, subjects: np.ndarray,
               embedding: np.ndarray) -> decoding.EmbeddingCache:
        model_tag, condition = analysis_id.split("_", 1)
        entry = {
            "db": db, "model_tag": model_tag, "input_condition": condition,
            "n_rows": int(len(rows)), "checkpoint_sha256": "a" * 64,
            "input_mask_sha256": "b" * 64,
            "row_identity_sha256": decoding.sha256_ordered_identity(
                db, rows, stay_ids, subjects),
            "source_npz_sha256": "c" * 64, "split_sha256": "d" * 64,
            "encoder_code_sha256": "e" * 64,
            "runtime_identity": {"device": "cpu"},
            "runtime_identity_sha256": "f" * 64,
            "encoding_scope": "synthetic nonempty rows",
        }
        return decoding.EmbeddingCache(
            db=db, analysis_id=analysis_id, manifest_entry=entry,
            rows=np.asarray(rows, dtype=np.int64),
            stay_id=np.asarray(stay_ids).astype(str),
            subject=np.asarray(subjects).astype(str),
            embedding=np.asarray(embedding, dtype=np.float32),
            metadata={}, cache_sha256="0" * 64,
        )

    def test_engine_identity_digest_is_order_sensitive(self):
        rows = np.array([2, 5], dtype=np.int64)
        stays = np.array(["s2", "s5"])
        subjects = np.array(["p2", "p5"])
        expected = decoding.sha256_ordered_identity("mimic", rows, stays, subjects)
        self.assertEqual(len(expected), 64)
        self.assertNotEqual(
            expected,
            decoding.sha256_ordered_identity("mimic", rows[::-1], stays[::-1], subjects[::-1]),
        )

    def test_cache_loader_checks_metadata_and_never_exposes_source_mortality(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.npz"
            rows = np.array([0, 1, 2], dtype=np.int64)
            stays = np.array(["s0", "s1", "s2"])
            subjects = np.array(["p0", "p1", "p2"])
            entry = {
                "db": "mimic", "model_tag": "main20k", "input_condition": "real",
                "n_rows": 3, "checkpoint_sha256": "a" * 64,
                "input_mask_sha256": "b" * 64,
                "row_identity_sha256": decoding.sha256_ordered_identity(
                    "mimic", rows, stays, subjects),
                "source_npz_sha256": "c" * 64, "split_sha256": "d" * 64,
                "encoder_code_sha256": "e" * 64,
                "runtime_identity": {"device": "cpu"},
                "runtime_identity_sha256": "f" * 64,
                "encoding_scope": "synthetic nonempty rows",
                "cache": str(path),
            }
            metadata = decoding._metadata_projection(entry)
            np.savez_compressed(
                path, rows=rows, stay_id=stays, subject=subjects,
                source_y=np.array([999.0, 888.0, 777.0], dtype=np.float32),
                embedding=np.arange(6, dtype=np.float32).reshape(3, 2),
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
            cache = decoding.load_embedding_cache(entry, "mimic", "main20k_real")
            np.testing.assert_array_equal(cache.embedding, np.arange(6).reshape(3, 2))
            self.assertFalse(hasattr(cache, "source_y"))

            bad_entry = dict(entry, row_identity_sha256="9" * 64)
            with self.assertRaisesRegex(decoding.DecoderInputError, "row identity"):
                decoding.load_embedding_cache(bad_entry, "mimic", "main20k_real")

    def test_target_join_is_by_stay_id_and_rejects_duplicate_keys(self):
        frame = pd.DataFrame({
            "stay_id": [1.0, 2.0], "creatinine": [1.2, 2.3],
        })
        aligned = decoding.align_tabular_frame(frame, ["2", "1"], ("creatinine",))
        self.assertEqual(aligned["stay_id"].tolist(), ["2", "1"])
        np.testing.assert_allclose(aligned["creatinine"].to_numpy(), [2.3, 1.2])

        duplicate = pd.DataFrame({"stay_id": ["s1", "s1"], "creatinine": [1, 2]})
        with self.assertRaisesRegex(decoding.DecoderInputError, "duplicate stay_id"):
            decoding.align_tabular_frame(duplicate, ["s1"], ("creatinine",))

    def test_corrected_target_requires_wbc_and_platelet_columns(self):
        frame = pd.DataFrame({"stay_id": ["s1"], "wbc": [7.2]})
        with self.assertRaisesRegex(decoding.DecoderInputError, "lacks columns.*platelet"):
            decoding.align_tabular_frame(frame, ["s1"], ("wbc", "platelet"))

    def test_shared_population_intersects_conditions_before_target_availability(self):
        ids = np.array([f"s{i}" for i in range(6)])
        patients = np.array([f"p{i}" for i in range(6)])
        rows = np.arange(6, dtype=np.int64)
        caches = {
            "main20k_real": self._cache(
                "mimic", "main20k_real", rows, ids, patients,
                np.arange(12, dtype=np.float32).reshape(6, 2)),
            "main20k_constant": self._cache(
                "mimic", "main20k_constant", rows, ids, patients,
                np.arange(12, dtype=np.float32).reshape(6, 2) + 1),
            "common20k_common8_mask": self._cache(
                "mimic", "common20k_common8_mask", rows[2:], ids[2:], patients[2:],
                np.arange(8, dtype=np.float32).reshape(4, 2)),
        }
        target = pd.DataFrame({
            "stay_id": ids,
            "creatinine": [100, 101, 102, np.nan, 104, 105],
            "lactate": [200, 201, 202, 203, np.inf, 205],
        })
        pop = decoding.build_common_population(
            caches, target, np.array([0, 0, 0, 1, 2, 2]),
            variables=("creatinine", "lactate"),
        )
        np.testing.assert_array_equal(pop.stay_id, ["s2", "s3", "s4", "s5"])
        np.testing.assert_array_equal(pop.rows, [2, 3, 4, 5])
        for embedding in pop.embedding_by_analysis.values():
            self.assertEqual(embedding.shape[0], 4)
        self.assertEqual(np.isfinite(pop.targets_by_variable["creatinine"]).sum(), 3)
        self.assertEqual(np.isfinite(pop.targets_by_variable["lactate"]).sum(), 3)

    def test_nested_ridge_returns_patient_disjoint_oof_predictions(self):
        rng = np.random.default_rng(123)
        n_patients, stays_per_patient = 30, 2
        groups = np.repeat([f"p{i}" for i in range(n_patients)], stays_per_patient)
        patient_x = rng.normal(size=(n_patients, 4))
        x = np.repeat(patient_x, stays_per_patient, axis=0)
        y = 1.7 * x[:, 0] - 0.4 * x[:, 1] + rng.normal(0, 0.08, len(x))

        for train, test in decoding.make_group_splits(groups, 5):
            self.assertFalse(set(groups[train]).intersection(groups[test]))
        oof, metrics = decoding.grouped_nested_ridge(x, y, groups)
        self.assertEqual(oof.shape, y.shape)
        self.assertTrue(np.isfinite(oof).all())
        self.assertGreater(metrics["r2"], 0.95)
        self.assertEqual(metrics["n_outer_folds"], 5)
        self.assertEqual(sum(metrics["alpha_counts"].values()), 5)

    def test_source_fit_tunes_only_from_source_groups(self):
        rng = np.random.default_rng(41)
        n_train_patients, n_val_patients = 24, 8
        train_groups = np.array([f"train{i}" for i in range(n_train_patients)])
        val_groups = np.array([f"validation{i}" for i in range(n_val_patients)])
        x_train = rng.normal(size=(n_train_patients, 3))
        x_val = rng.normal(size=(n_val_patients, 3))
        y_train = 2.0 * x_train[:, 0] + 0.2 * x_train[:, 1] + rng.normal(0, 0.05, len(x_train))
        y_val = 2.0 * x_val[:, 0] + 0.2 * x_val[:, 1] + rng.normal(0, 0.05, len(x_val))

        # Selection is scored on split 1; the final scaler and Ridge see split 0 only.
        scale_for_validation = decoding.StandardScaler().fit(x_train)
        expected_mse = {}
        for alpha in decoding.ALPHAS:
            candidate = decoding.Ridge(alpha=alpha).fit(
                scale_for_validation.transform(x_train), y_train)
            prediction_val = candidate.predict(scale_for_validation.transform(x_val))
            expected_mse[float(alpha)] = float(
                decoding.mean_squared_error(y_val, prediction_val))
        expected_alpha = min(decoding.ALPHAS, key=lambda alpha: expected_mse[alpha])

        model, metadata = decoding.fit_source_ridge(
            x_train, y_train, train_groups, x_val, y_val, val_groups)
        prediction = model.predict(x_train)
        self.assertIn(metadata["alpha"], decoding.ALPHAS)
        self.assertEqual(metadata["alpha"], expected_alpha)
        for alpha, expected in expected_mse.items():
            self.assertAlmostEqual(
                metadata["alpha_validation_mse_by_alpha"][alpha], expected, places=6)
        self.assertEqual(metadata["n_source_train_patients"], n_train_patients)
        self.assertEqual(metadata["n_source_validation_patients"], n_val_patients)
        np.testing.assert_allclose(
            model.named_steps["scale"].mean_, x_train.mean(axis=0), atol=1e-6)
        self.assertGreater(decoding._metrics(y_train, prediction)["r2"], 0.98)

    def test_source_fit_rejects_patient_overlap_between_splits(self):
        x_train = np.arange(12, dtype=float).reshape(6, 2)
        x_val = np.arange(6, dtype=float).reshape(3, 2)
        y_train = np.arange(6, dtype=float)
        y_val = np.arange(3, dtype=float)
        with self.assertRaisesRegex(decoding.DecoderInputError, "overlapping patients"):
            decoding.fit_source_ridge(
                x_train, y_train, np.array(["a", "b", "c", "d", "e", "f"]),
                x_val, y_val, np.array(["g", "h", "a"]),
            )

    def test_source_split_partitions_are_patient_disjoint(self):
        population = decoding.AlignedPopulation(
            stay_id=np.array(["s0", "s1", "s2", "s3", "s4", "s5"]),
            subject=np.array(["p0", "p1", "p2", "p3", "p4", "p5"]),
            rows=np.arange(6), embedding_by_analysis={}, targets_by_variable={},
            split=np.array([0, 0, 1, 1, 2, 2]),
            n_eligible_stays=6, cache_rows_by_analysis={},
        )
        decoding.validate_source_patient_partitions(population, "mimic")
        overlapping = decoding.AlignedPopulation(
            stay_id=population.stay_id,
            subject=np.array(["p0", "p1", "p0", "p3", "p4", "p5"]),
            rows=population.rows, embedding_by_analysis={},
            targets_by_variable={}, split=population.split,
            n_eligible_stays=6, cache_rows_by_analysis={},
        )
        with self.assertRaisesRegex(decoding.DecoderInputError, "patient overlap"):
            decoding.validate_source_patient_partitions(overlapping, "mimic")

    def test_checkpoint_hash_and_source_row_identity_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "main20k.pt"
            checkpoint.write_bytes(b"synthetic frozen checkpoint")
            checkpoint_hash = decoding.sha256_file(checkpoint)
            entries = {"mimic/main20k_real": {
                "model_tag": "main20k", "checkpoint_sha256": checkpoint_hash,
            }}
            hashes = decoding.validate_checkpoint_hashes(entries, {"main20k": checkpoint})
            self.assertEqual(hashes["main20k"], checkpoint_hash)
            checkpoint.write_bytes(b"changed synthetic checkpoint")
            with self.assertRaisesRegex(decoding.DecoderInputError, "checkpoint SHA256"):
                decoding.validate_checkpoint_hashes(entries, {"main20k": checkpoint})

        cache = self._cache(
            "mimic", "main20k_real", np.array([1, 3]),
            np.array(["s1", "s3"]), np.array(["p1", "p3"]),
            np.ones((2, 2), dtype=np.float32),
        )
        decoding.verify_cache_sequence_identity(
            cache, np.array(["s0", "s1", "s2", "s3"]),
            np.array(["p0", "p1", "p2", "p3"]))
        with self.assertRaisesRegex(decoding.DecoderInputError, "do not match"):
            decoding.verify_cache_sequence_identity(
                cache, np.array(["s0", "wrong", "s2", "s3"]),
                np.array(["p0", "p1", "p2", "p3"]))

    def test_negative_r2_is_retained(self):
        y = np.arange(8, dtype=float)
        self.assertLess(decoding._metrics(y, y[::-1])["r2"], 0.0)

    def test_patient_bootstrap_is_clustered_seeded_and_requires_1000_draws(self):
        groups = np.array(["p1", "p1", "p2", "p3", "p3", "p4"])
        y = np.array([0, 1, 1, 2, 2, 4], dtype=float)
        prediction = y + np.array([.2, -.1, .3, -.4, .1, .2])
        ci_a = decoding.patient_cluster_bootstrap_metrics(
            y, prediction, groups, n_boot=1000, seed=123)
        ci_b = decoding.patient_cluster_bootstrap_metrics(
            y, prediction, groups, n_boot=1000, seed=123)
        self.assertEqual(ci_a, ci_b)
        self.assertEqual(ci_a["bootstrap_replicates"], 1000)
        self.assertEqual(ci_a["bootstrap_cluster_unit"], "patient")
        self.assertEqual(decoding.paired_bootstrap_seed("eicu", "hr", "source_transfer"),
                         decoding.paired_bootstrap_seed("eicu", "hr", "source_transfer"))
        with self.assertRaisesRegex(decoding.DecoderInputError, "at least 1000"):
            decoding.patient_cluster_bootstrap_metrics(
                y, prediction, groups, n_boot=999, seed=123)

    def test_end_to_end_runner_uses_only_synthetic_files_and_checks_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eligibility_dir = root / "eligibility"
            eligibility_dir.mkdir()
            n = 30
            rng = np.random.default_rng(89)
            embeddings = rng.normal(size=(n, 4)).astype(np.float32)
            subjects = np.array([f"patient_{i:03d}" for i in range(n)])
            stays = np.array([f"stay_{i:03d}" for i in range(n)])
            y_mort = np.array([i % 2 for i in range(n)], dtype=np.int8)
            mimic_split = np.array([0] * 14 + [1] * 8 + [2] * 8, dtype=np.int8)
            eicu_split = np.zeros(n, dtype=np.int8)
            target_frames = {}
            resolved_paths = {}
            eligible_frames = {}
            manifest_rows = []
            resolver_calls = []
            checkpoint_paths = {
                "main20k": root / "main20k.pt",
                "abl_const20k": root / "abl_const20k.pt",
            }
            checkpoint_paths["main20k"].write_bytes(b"synthetic main checkpoint bytes")
            checkpoint_paths["abl_const20k"].write_bytes(b"synthetic ablation checkpoint bytes")
            checkpoint_hashes = {
                model: decoding.sha256_file(path) for model, path in checkpoint_paths.items()
            }
            eligibility_entries = {}

            target_db = "eicu_semantic_v4_2"
            for db, split in (("mimic", mimic_split), (target_db, eicu_split)):
                sequence_path = root / f"seq_{db}.npz"
                np.savez_compressed(
                    sequence_path, stay_id=stays, subject=subjects, y_mort=y_mort)
                sequence_hash = decoding.sha256_file(sequence_path)
                split_path = root / f"split_{db}.npy"
                np.save(split_path, split)
                split_hash = decoding.sha256_file(split_path)
                tabular_path = root / f"tab_{db}.parquet"
                tabular_path.write_bytes(f"synthetic target table {db}".encode())
                eligibility_table_path = eligibility_dir / f"{db}.parquet"
                eligibility_table_path.write_bytes(f"synthetic eligibility table {db}".encode())

                numeric = {"stay_id": stays}
                for var_i, variable in enumerate(decoding.VARS):
                    numeric[variable] = (
                        (var_i + 1) * embeddings[:, 0]
                        + 0.3 * embeddings[:, 1]
                        + rng.normal(0, 0.02, n)
                    )
                target_frames[tabular_path] = pd.DataFrame(numeric)
                resolved_paths[db] = SimpleNamespace(
                    tabular_path=tabular_path, split_path=split_path,
                    sequence_path=sequence_path,
                )
                eligible_frames[db] = pd.DataFrame({
                    "db": db, "stay_id": stays, "subject": subjects,
                    "y_mort": y_mort, "endpoint": "hospital",
                })
                eligibility_entries[db] = {
                    "retrospective_landmark_prediction_allowed": True,
                    "evidence": "operational_proxy", "status": "ready",
                    "sourcehashes": {"sequence_npz": sequence_hash},
                }

                for analysis_id, model_tag, condition, row_indices, matrix in (
                    ("main20k_real", "main20k", "real", np.arange(n), embeddings),
                    ("main20k_constant", "main20k", "constant", np.arange(n), embeddings * .8),
                    ("abl_const20k_constant", "abl_const20k", "constant",
                     np.arange(n - 1), embeddings[:n - 1] * .6),
                ):
                    cache_path = root / f"{db}_{analysis_id}.npz"
                    cache_stays, cache_subjects = stays[row_indices], subjects[row_indices]
                    entry = {
                        "db": db, "model_tag": model_tag, "input_condition": condition,
                        "n_rows": int(len(row_indices)),
                        "checkpoint_sha256": checkpoint_hashes[model_tag],
                        "input_mask_sha256": "b" * 64,
                        "row_identity_sha256": decoding.sha256_ordered_identity(
                            db, row_indices, cache_stays, cache_subjects),
                        "source_npz_sha256": sequence_hash, "split_sha256": split_hash,
                        "encoder_code_sha256": "e" * 64,
                        "runtime_identity": {"device": "synthetic_cpu"},
                        "runtime_identity_sha256": "f" * 64,
                        "encoding_scope": "synthetic rows", "cache": str(cache_path),
                    }
                    metadata = decoding._metadata_projection(entry)
                    np.savez_compressed(
                        cache_path, rows=row_indices.astype(np.int64),
                        stay_id=cache_stays, subject=cache_subjects,
                        source_y=np.full(len(row_indices), 12345, dtype=np.float32),
                        embedding=matrix,
                        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
                    )
                    manifest_rows.append(entry)

            manifest_path = root / "embedding_manifest.json"
            manifest_path.write_text(json.dumps({"mode": "encode", "models": manifest_rows}))
            eligibility_manifest_path = eligibility_dir / "eligibility_manifest.json"
            eligibility_manifest_path.write_text(json.dumps({"cohorts": eligibility_entries}))

            def eligibility_loader(db, directory):
                row = dict(eligibility_entries[db])
                row.update({
                    "eligibility_manifest_sha256": decoding.sha256_file(
                        directory / "eligibility_manifest.json"),
                    "eligibility_table_sha256": decoding.sha256_file(
                        directory / f"{db}.parquet"),
                    "sequence_npz_sha256": row["sourcehashes"]["sequence_npz"],
                })
                return eligible_frames[db], row

            source_caches = {
                analysis_id: decoding.load_embedding_cache(
                    decoding.load_embedding_manifest(manifest_path)[1][f"mimic/{analysis_id}"],
                    "mimic", analysis_id,
                )
                for analysis_id in decoding.CORE_ANALYSIS_IDS
            }
            source_population = decoding.build_common_population(
                source_caches, target_frames[resolved_paths["mimic"].tabular_path],
                mimic_split, eligible_frames["mimic"],
            )
            source_reference_rows = []
            for analysis_id in decoding.CORE_ANALYSIS_IDS:
                x = source_population.embedding_by_analysis[analysis_id]
                for variable in decoding.VARS:
                    y = source_population.targets_by_variable[variable]
                    train = (source_population.split == 0) & np.isfinite(y)
                    validation = (source_population.split == 1) & np.isfinite(y)
                    test = (source_population.split == 2) & np.isfinite(y)
                    model, fit_meta = decoding.fit_source_ridge(
                        x[train], y[train], source_population.subject[train],
                        x[validation], y[validation], source_population.subject[validation],
                    )
                    test_metrics = decoding._metrics(y[test], model.predict(x[test]))
                    source_reference_rows.append({
                        "analysis_id": analysis_id, "mode": "source_transfer",
                        "source_db": "mimic", "target_db": "mimic", "variable": variable,
                        "status": "ok", "alpha": fit_meta["alpha"],
                        "source_alpha_validation_mse": fit_meta["alpha_validation_mse"],
                        "source_alpha_validation_mse_by_alpha_json": json.dumps(
                            fit_meta["alpha_validation_mse_by_alpha"], sort_keys=True),
                        "n_source_train_rows": int(train.sum()),
                        "n_source_train_patients": int(len(np.unique(source_population.subject[train]))),
                        "n_source_validation_rows": int(validation.sum()),
                        "n_source_validation_patients": int(len(np.unique(source_population.subject[validation]))),
                        "n_source_test_rows": int(test.sum()),
                        "source_train_population_identity_sha256": decoding._population_hash(
                            source_population, train, "mimic"),
                        "source_validation_population_identity_sha256": decoding._population_hash(
                            source_population, validation, "mimic"),
                        "population_identity_sha256": decoding._population_hash(
                            source_population, test, "mimic"),
                        **test_metrics,
                    })
            source_reference_path = root / "frozen_mimic_source_summary.csv"
            pd.DataFrame(source_reference_rows).to_csv(source_reference_path, index=False)

            def resolver(db):
                resolver_calls.append(db)
                return resolved_paths[db]

            result, run_manifest = decoding.run_decoding(
                manifest_path=manifest_path,
                analysis_ids=decoding.CORE_ANALYSIS_IDS,
                target_dbs=(target_db,), resolver=resolver,
                target_reader=lambda path: target_frames[path],
                checkpoint_resolver=lambda model_tag: checkpoint_paths[model_tag],
                eligibility_dir=eligibility_dir,
                eligibility_loader=eligibility_loader,
                source_reference_path=source_reference_path,
                expected_embedding_manifest_sha256=None,
                expected_eligibility_manifest_sha256=None,
                expected_source_reference_sha256=None,
                n_bootstrap=1000,
                seed=987,
                output_dir=root / "results",
            )
            self.assertEqual(resolver_calls, ["mimic", target_db])
            self.assertEqual(len(result), len(decoding.VARS) * 3 * 2)
            self.assertEqual(set(result["target_db"]), {target_db})
            self.assertTrue(result["status"].isin(("ok", "ok_r2_undefined")).all())
            self.assertTrue(result["bootstrap_replicates"].eq(1000).all())
            self.assertTrue(result["bootstrap_cluster_unit"].eq("patient").all())
            self.assertTrue(result["bootstrap_interval_type"].eq(
                "marginal_patient_cluster_percentile").all())
            self.assertTrue(result["n_eligible_stays"].eq(n).all())
            self.assertTrue(result["n_common_condition_stays"].eq(n - 1).all())
            self.assertTrue(result["cache_rows_by_condition_json"].map(
                lambda value: json.loads(value)["abl_const20k_constant"] == n - 1).all())
            source_rows = result[result["mode"] == "source_transfer"]
            self.assertTrue(source_rows["target_label_usage"].str.contains(
                "source_split0_refit;target_labels_metric_only").all())
            self.assertTrue(source_rows["alpha"].isin(decoding.ALPHAS).all())
            self.assertTrue(source_rows.groupby(["target_db", "variable"])[
                "bootstrap_seed"].nunique().eq(1).all())
            self.assertTrue(source_rows["source_mimic_split2_metrics_match_reference"].all())
            self.assertEqual(len(run_manifest["source_mimic_split2_metric_validation"]), 30)
            self.assertEqual(run_manifest["checkpoint_sha256_by_model"], checkpoint_hashes)
            self.assertEqual(
                set(run_manifest["source_sequence_npz_sha256_by_db"]), {"mimic", target_db})
            self.assertEqual(run_manifest["preflight_counts"]["eligible_count_by_db"],
                             {"mimic": n, target_db: n})
            self.assertEqual(set(result.loc[result["variable"].isin(("wbc", "platelet")),
                                            "variable"]), {"wbc", "platelet"})
            self.assertFalse({"stay_id", "subject", "y_mort", "prediction"}.intersection(result.columns))
            self.assertTrue((root / "results/numeric_decoding_summary.csv").is_file())
            self.assertTrue((root / "results/numeric_decoding_manifest.json").is_file())

            target_path = resolved_paths[target_db].tabular_path
            missing_column_frames = dict(target_frames)
            missing_column_frames[target_path] = target_frames[target_path].drop(columns=["wbc"])
            with self.assertRaisesRegex(decoding.DecoderInputError, "lacks columns.*wbc"):
                decoding.run_decoding(
                    manifest_path=manifest_path, target_dbs=(target_db,), resolver=resolver,
                    target_reader=lambda path: missing_column_frames[path],
                    checkpoint_resolver=lambda model_tag: checkpoint_paths[model_tag],
                    eligibility_dir=eligibility_dir, eligibility_loader=eligibility_loader,
                    source_reference_path=source_reference_path,
                    expected_embedding_manifest_sha256=None,
                    expected_eligibility_manifest_sha256=None,
                    expected_source_reference_sha256=None, preflight_only=True,
                    output_dir=root / "missing_column_results",
                )

            bad_entry = next(row for row in manifest_rows
                             if row["db"] == target_db and row["model_tag"] == "main20k"
                             and row["input_condition"] == "real")
            with np.load(bad_entry["cache"], allow_pickle=False) as loaded:
                arrays = {key: loaded[key].copy() for key in loaded.files
                          if key != "metadata_json"}
            arrays["stay_id"][0] = "misaligned_stay_id"
            bad_entry["row_identity_sha256"] = decoding.sha256_ordered_identity(
                target_db, arrays["rows"], arrays["stay_id"], arrays["subject"])
            arrays["metadata_json"] = np.asarray(json.dumps(
                decoding._metadata_projection(bad_entry), sort_keys=True))
            np.savez_compressed(bad_entry["cache"], **arrays)
            manifest_path.write_text(json.dumps({"mode": "encode", "models": manifest_rows}))
            with self.assertRaisesRegex(decoding.DecoderInputError, "cache IDs do not match source NPZ"):
                decoding.run_decoding(
                    manifest_path=manifest_path, target_dbs=(target_db,), resolver=resolver,
                    target_reader=lambda path: target_frames[path],
                    checkpoint_resolver=lambda model_tag: checkpoint_paths[model_tag],
                    eligibility_dir=eligibility_dir, eligibility_loader=eligibility_loader,
                    source_reference_path=source_reference_path,
                    expected_embedding_manifest_sha256=None,
                    expected_eligibility_manifest_sha256=None,
                    expected_source_reference_sha256=None, preflight_only=True,
                    output_dir=root / "misaligned_id_results",
                )


if __name__ == "__main__":
    unittest.main()
