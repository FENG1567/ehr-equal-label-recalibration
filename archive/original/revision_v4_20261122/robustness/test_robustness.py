"""Targeted unit tests for v4 robustness math, denominators, and ID joins."""
from __future__ import annotations

import json
import hashlib
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from case_mix_selective import (
    _age_sex_cell_counts, _metrics, _selective_rows, age_group, normalize_sex,
)
from hospital_heterogeneity import (
    _patient_cluster_slope_bootstrap, _selected_site_metric_rows, _site_count_preflight,
    _site_bootstrap_ci, _site_metric_table, _solve_weighted_logistic_slope_batch,
    paired_frame, patient_hospital_crossing,
)
from recalibration import (
    _cluster_bootstrap_ci, adaptation_assessment_pools, one_index_per_patient,
    run_db, split_patient_pool,
    verify_shared_population_frames, verify_shared_recalibration_manifests,
    fit_offset, fit_platt,
)
from robustness_common import (
    hindsight_bss, load_checked, manifest_ready, ordered_prediction_digest,
    sha256_file, validate_prediction_manifest,
)


class RobustnessTests(unittest.TestCase):
    @staticmethod
    def _write_prediction_artifacts(root: Path, prediction: pd.DataFrame,
                                    eligibility_tables: dict[str, pd.DataFrame],
                                    cohort_entries: dict[str, dict], *,
                                    top_level: dict | None = None) -> tuple[Path, Path, Path]:
        eligibility_dir = root / "eligibility"
        eligibility_dir.mkdir(parents=True, exist_ok=True)
        for db, table in eligibility_tables.items():
            table.to_parquet(eligibility_dir / f"{db}.parquet", index=False)
        eligibility_manifest = {"cohorts": cohort_entries, **(top_level or {})}
        eligibility_manifest_path = eligibility_dir / "eligibility_manifest.json"
        eligibility_manifest_path.write_text(json.dumps(eligibility_manifest), encoding="utf-8")
        prediction = prediction.copy()
        records = []
        prediction_path = root / "predictions.csv"
        for db in prediction["db"].astype(str).unique():
            entry = cohort_entries[db]
            table_hash = sha256_file(eligibility_dir / f"{db}.parquet")
            manifest_hash = sha256_file(eligibility_manifest_path)
            mask = prediction["db"].astype(str) == db
            prediction.loc[mask, "eligibility_table_sha256"] = table_hash
            prediction.loc[mask, "eligibility_manifest_sha256"] = manifest_hash
        prediction.to_csv(prediction_path, index=False)
        for (analysis_id, db), frame in prediction.groupby(["analysis_id", "db"], sort=True):
            entry = cohort_entries[str(db)]
            records.append({
                "analysis_id": str(analysis_id), "db": str(db),
                "model_tag": str(frame["model_tag"].iloc[0]),
                "input_condition": str(frame["input_condition"].iloc[0]),
                "prediction_file": str(prediction_path.resolve()),
                "prediction_file_sha256": sha256_file(prediction_path),
                "ordered_row_digest_sha256": ordered_prediction_digest(frame),
                "n_prediction_rows": int(len(frame)), "role": str(frame["role"].iloc[0]),
                "prediction_scope": str(frame["prediction_scope"].iloc[0]),
                "input_scope": str(frame["input_scope"].iloc[0]),
                "clinical_input_availability_verified": bool(
                    frame["clinical_input_availability_verified"].iloc[0]),
                "eligibility_manifest_sha256": str(entry["eligibility_manifest_sha256"])
                if entry.get("eligibility_manifest_sha256") else sha256_file(eligibility_manifest_path),
                "eligibility_table_sha256": str(entry["eligibility_table_sha256"])
                if entry.get("eligibility_table_sha256") else sha256_file(eligibility_dir / f"{db}.parquet"),
                "source_npz_sha256": entry["sourcehashes"]["sequence_npz"],
                "retrospective_landmark_prediction_allowed": entry.get(
                    "retrospective_landmark_prediction_allowed", False),
                "retrospective_diagnostic_allowed": entry.get("retrospective_diagnostic_allowed", False),
            })
        prediction_manifest_path = root / "prediction_manifest.json"
        prediction_manifest_path.write_text(json.dumps({"status": "complete", "predictions": records}),
                                            encoding="utf-8")
        return prediction_path, prediction_manifest_path, eligibility_dir

    @staticmethod
    def _eligible_table(db: str, rows: list[list]) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=[
            "db", "stay_id", "subject", "eligible_24h", "temporal_verified",
            "y_mort", "endpoint", "reason", "age", "sex",
        ])

    @staticmethod
    def _cohort_entry(*, allowed: bool = True, evidence: str = "operational_proxy",
                      identity_allowed: bool = True, diagnostic_allowed: bool = False) -> dict:
        return {
            "retrospective_landmark_prediction_allowed": allowed,
            "ready_for_future24h_clinical_claim": False,
            "retrospective_diagnostic_allowed": diagnostic_allowed,
            "evidence": evidence, "status": "ready" if allowed else "blocked",
            "evidence_basis": "audited synthetic eligibility evidence",
            "subject_identity_basis": {
                "basis": "longitudinal patient_id",
                "patient_disjoint_claim_allowed": identity_allowed,
            },
            "sourcehashes": {"sequence_npz": "c" * 64},
        }

    @staticmethod
    def _prediction_rows(*, scope: str = "retrospective_landmark_prediction_measurement_time",
                         role: str = "evaluation", dbs: tuple[str, ...] = ("mimic", "eicu"),
                         analysis_id: str = "main20k_real", model_tag: str = "main20k") -> pd.DataFrame:
        values = []
        for db, stay, subject, split, label, probability in (
            ("mimic", "m1", "p1", 2, 0, .1),
            ("mimic", "m2", "p2", 2, 1, .7),
            ("eicu", "e1", "e1p", 0, 1, .3),
        ):
            if db not in dbs:
                continue
            values.append({
                "db": db, "analysis_id": analysis_id, "stay_id": stay,
                "subject": subject, "rows": len(values), "split": split, "y": label,
                "p": probability, "model_tag": model_tag, "input_condition": "real",
                "endpoint": "hospital", "role": role, "prediction_scope": scope,
                "input_scope": "retrospective_measurement_time",
                "clinical_input_availability_verified": False,
            })
        return pd.DataFrame(values)

    def test_hindsight_bss_reference_algebra(self):
        y = np.array([0, 0, 0, 1, 1, 1], dtype=int)
        prevalence = float(y.mean())
        self.assertAlmostEqual(hindsight_bss(y, np.full(len(y), prevalence)), 0.0)
        self.assertLess(hindsight_bss(y, np.full(len(y), 0.2)), 0.0)
        self.assertAlmostEqual(hindsight_bss(np.array([0, 1]), np.array([0.0, 1.0])), 1.0)

    def test_current_engine_manifest_uses_hashed_protocol_and_readiness_adapter(self):
        protocol = {
            "run_id": "synthetic_current_engine", "mode": "predict",
            "models": ["main20k"], "cohorts": ["mimic"],
            "encoder_code_sha256": "a" * 64, "engine_code_sha256": "b" * 64,
            "eligibility_manifest_sha256": "c" * 64,
            "embedding_manifest_sha256": "d" * 64,
            "eligibility_table_sha256_by_cohort": {"mimic": "e" * 64},
            "source_npz_sha256_attested_by_cohort": {"mimic": "f" * 64},
            "checkpoint_sha256_by_model": {"main20k": "1" * 64},
            "input_scope": "retrospective_measurement_time",
            "clinical_input_availability_verified": False,
        }
        protocol_sha = hashlib.sha256(json.dumps(
            protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")).hexdigest()
        protocol["run_protocol_sha256"] = protocol_sha
        manifest = {
            "mode": "predict", "run_id": "synthetic_current_engine",
            "input_scope": "retrospective_measurement_time",
            "clinical_input_availability_verified": False,
            "run_protocol": protocol, "run_protocol_sha256": protocol_sha,
            "readiness": [{
                "db": "mimic", "status": "ready", "temporal_evidence": "exact",
                "prediction_scope": "retrospective_landmark_prediction_measurement_time",
                "input_scope": "retrospective_measurement_time",
                "clinical_input_availability_verified": False,
                "retrospective_landmark_prediction_ready": True,
            }],
            "retrospective_landmark_prediction_by_cohort": {"mimic": True},
            "real_time_available_landmark_prediction_by_cohort": {"mimic": False},
            "predictions": [{
                "analysis_id": "main20k_real", "db": "mimic", "model_tag": "main20k",
                "mode": "predict", "role": "evaluation",
                "prediction_scope": "retrospective_landmark_prediction_measurement_time",
                "input_scope": "retrospective_measurement_time",
                "clinical_input_availability_verified": False,
                "retrospective_landmark_prediction_allowed": True,
                "n_prediction_rows": 2,
            }],
        }
        validate_prediction_manifest(manifest)
        broken = json.loads(json.dumps(manifest))
        broken["readiness"][0]["retrospective_landmark_prediction_ready"] = False
        with self.assertRaisesRegex(ValueError, "does not attest"):
            validate_prediction_manifest(broken)

    def test_registered_baseline_analysis_id_can_use_manifest_without_model_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction = self._prediction_rows(
                dbs=("mimic",), analysis_id="conventional_lr_pattern_only",
                model_tag="conventional_lr",
            )
            eligibility = {"mimic": self._eligible_table("mimic", [
                ["mimic", "m1", "p1", True, True, 0, "hospital", "ok", 55, "F"],
                ["mimic", "m2", "p2", True, True, 1, "hospital", "ok", 72, "M"],
            ])}
            prediction_path, prediction_manifest, elig = self._write_prediction_artifacts(
                root, prediction, eligibility, {"mimic": self._cohort_entry()})
            document = json.loads(prediction_manifest.read_text(encoding="utf-8"))
            document["predictions"][0].pop("model_tag")
            prediction_manifest.write_text(json.dumps(document), encoding="utf-8")
            loaded, gates = load_checked(
                prediction_path, elig, model_tag="conventional_lr",
                analysis_id="conventional_lr_pattern_only",
                prediction_manifest_path=prediction_manifest,
            )
            self.assertEqual(len(loaded["mimic"]), 2)
            self.assertEqual(gates["mimic"]["analysis_id"], "conventional_lr_pattern_only")
            with self.assertRaisesRegex(ValueError, "not registered"):
                load_checked(
                    prediction_path, elig, model_tag="conventional_hgb",
                    analysis_id="conventional_lr_pattern_only",
                    prediction_manifest_path=prediction_manifest,
                )

    def test_one_index_stay_per_patient_is_outcome_blind_and_order_stable(self):
        df = pd.DataFrame({
            "subject": ["p1", "p1", "p2", "p2", "p3"],
            "stay_id": ["s1", "s2", "s3", "s4", "s5"],
            "y": [0, 1, 0, 1, 1], "p": [.1, .2, .3, .4, .5],
        })
        a = one_index_per_patient(df, 111)
        changed = df.copy()
        changed["y"] = 1 - changed["y"]
        b = one_index_per_patient(changed.sample(frac=1, random_state=8), 111)
        self.assertEqual(set(a["stay_id"]), set(b["stay_id"]))
        self.assertEqual(a["subject"].nunique(), 3)
        adapt, assess, _ = split_patient_pool(a, 42)
        self.assertFalse(set(adapt["subject"]) & set(assess["subject"]))
        self.assertEqual(adapt["subject"].nunique(), len(adapt))
        self.assertEqual(assess["subject"].nunique(), len(assess))

    def test_recalibration_uses_one_adaptation_index_but_all_disjoint_assessment_stays(self):
        rows = []
        for i in range(40):
            for j in range(1 + (i % 3 == 0)):
                rows.append({"subject": f"p{i}", "stay_id": f"s{i}_{j}",
                             "y": int(i % 4 == 0), "p": .1 + .01 * (i % 50)})
        df = pd.DataFrame(rows)
        adapt, assess, _ = adaptation_assessment_pools(df, 23)
        changed = df.assign(y=1 - df["y"])
        adapt_changed, assess_changed, label_stratified = adaptation_assessment_pools(changed, 23)
        self.assertEqual(adapt["subject"].nunique(), len(adapt))
        self.assertTrue(assess["subject"].duplicated().any())
        self.assertFalse(set(adapt["subject"]) & set(assess["subject"]))
        self.assertFalse(label_stratified)
        self.assertEqual(set(adapt["subject"]), set(adapt_changed["subject"]))
        self.assertEqual(set(assess["subject"]), set(assess_changed["subject"]))

    def test_single_class_calibration_fits_fail_for_predeclared_fallback(self):
        y = np.zeros(5, dtype=int)
        logits = np.linspace(-2, 2, 5)
        with self.assertRaises(ValueError):
            fit_offset(y, logits)
        with self.assertRaises(ValueError):
            fit_platt(y, logits)
        smooth = (int(y.sum()) + 0.5) / (len(y) + 1)
        self.assertGreater(smooth, 0)
        self.assertLess(smooth, 1)

    def test_cluster_bootstrap_ci_fails_closed_for_too_few_draws_or_clusters(self):
        lo, hi = _cluster_bootstrap_ci(np.array([.1, .2]), np.array(["p1", "p1"]),
                                       n_boot=1000, seed=5)
        self.assertTrue(np.isnan(lo) and np.isnan(hi))
        lo, hi = _cluster_bootstrap_ci(np.array([.1, .2]), np.array(["p1", "p2"]),
                                       n_boot=999, seed=5)
        self.assertTrue(np.isnan(lo) and np.isnan(hi))
        with self.assertRaisesRegex(ValueError, "at least 1000"):
            run_db("synthetic", pd.DataFrame(), repeats=1, n_boot=999, seed=5)

    def test_recalibration_emits_paired_scores_with_assessment_cluster_ci(self):
        rows = []
        for i in range(60):
            event = int(i % 5 == 0)
            for j in range(1 + int(i % 3 == 0)):
                rows.append({
                    "db": "synthetic", "stay_id": f"s{i}_{j}", "subject": f"p{i}",
                    "y": event, "p": min(.95, max(.02, .04 + .7 * event + .02 * (i % 7))),
                    "eligible_24h": True, "temporal_verified": True,
                    "cohort_status": "retrospective_landmark_prediction_measurement_time",
                })
        draws, summary, ci_rows = run_db(
            "synthetic", pd.DataFrame(rows), repeats=2, n_boot=1000, seed=311,
            budgets=(1, 20), cluster_unit="patient",
        )
        self.assertTrue(any(row["method"] == "offset_operational" for row in draws))
        self.assertTrue(any(row["method"] == "platt_operational" for row in draws))
        split = summary[0]
        self.assertEqual(split["patient_overlap"], 0)
        self.assertGreater(split["n_assessment_stays"], split["n_assessment_patients"])
        self.assertEqual(len(split["fixed_population_identity_sha256"]), 64)
        self.assertEqual(len(split["adaptation_index_patient_stay_sha256"]), 64)
        self.assertEqual(len(split["assessment_population_identity_sha256"]), 64)
        self.assertEqual(len(split["sampling_plan_sha256"]), 64)
        self.assertTrue(ci_rows)
        self.assertTrue(all(row["assessment_bootstrap_draws"] == 1000 for row in ci_rows))
        self.assertTrue(all(row["bootstrap_cluster_unit"] == "patient" for row in ci_rows))
        offset_fallbacks = [row for row in draws
                            if row["method"] == "offset_operational" and row["is_operational_fallback"]]
        self.assertTrue(offset_fallbacks)
        self.assertTrue(all(abs(row["delta_logloss_vs_smoothed_prevalence"]) < 1e-12
                            for row in offset_fallbacks))
        self.assertTrue(any(row["analysis_scope"] == "fallback_only"
                            and row["method"] == "offset_failure_fallback_smoothed_prevalence"
                            for row in ci_rows))
        self.assertTrue(any(row["method"] == "offset_conditional"
                            and row["comparator"] == "smoothed_adaptation_prevalence"
                            and row["analysis_scope"] == "successful_fit_draws"
                            for row in ci_rows))

    def test_shared_population_requires_actual_matching_patient_stay_rows(self):
        fm = pd.DataFrame({
            "db": ["mimic", "mimic"], "stay_id": ["s1", "s2"],
            "subject": ["p1", "p2"], "rows": [8, 9], "split": [2, 2], "y": [0, 1],
        })
        lr = fm.sample(frac=1, random_state=7).reset_index(drop=True)
        digest = verify_shared_population_frames({"fm": fm, "lr": lr})
        self.assertEqual(len(digest), 64)
        mismatch = lr.copy()
        mismatch.loc[mismatch["stay_id"] == "s2", "rows"] = 91
        with self.assertRaisesRegex(ValueError, "exact matched population"):
            verify_shared_population_frames({"fm": fm, "lr": mismatch})

    def test_shared_plan_manifest_comparison_checks_actual_plan_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = {
                "fixed_population_identity_sha256": "a" * 64,
                "adaptation_index_patient_stay_sha256": "b" * 64,
                "adaptation_patient_set_sha256": "c" * 64,
                "assessment_population_identity_sha256": "d" * 64,
                "assessment_patient_set_sha256": "e" * 64,
                "sampling_plan_sha256": "f" * 64,
            }
            first = root / "fm.json"
            second = root / "lr.json"
            first.write_text(json.dumps({"status": "complete", "cohort_sampling_plans": {"mimic": plan}}))
            second.write_text(json.dumps({"status": "complete", "cohort_sampling_plans": {"mimic": plan}}))
            self.assertEqual(verify_shared_recalibration_manifests([first, second])["status"], "verified")
            changed = dict(plan, sampling_plan_sha256="9" * 64)
            second.write_text(json.dumps({"status": "complete", "cohort_sampling_plans": {"mimic": changed}}))
            with self.assertRaisesRegex(ValueError, "shared recalibration plan differs"):
                verify_shared_recalibration_manifests([first, second])

    def test_weighted_score_denominators_use_weight_sum(self):
        y = np.array([0, 1, 1], dtype=int)
        p = np.array([.1, .6, .9])
        weights = np.array([1., 2., 1.])
        got = _metrics(y, p, weights)
        expected = np.average((y - p) ** 2, weights=weights)
        self.assertAlmostEqual(got["brier"], expected)
        self.assertAlmostEqual(got["prevalence"], np.average(y, weights=weights))

    def test_selective_event_retention_denominator_and_random_baseline(self):
        df = pd.DataFrame({
            "stay_id": [f"s{i}" for i in range(10)],
            "y": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            "p": [.99, .51, .49, .48, .52, .5, .5, .5, .5, .5],
            "cohort_status": ["descriptive"] * 10,
        })
        row = _selective_rows("x", df, 1)[0]
        self.assertEqual(row["n_kept_patients"], 5)
        self.assertEqual(row["n_total_events"], 2)
        self.assertAlmostEqual(row["event_retention"], row["n_kept_events"] / 2)
        self.assertEqual(row["n_total_nonevents"], 8)
        self.assertAlmostEqual(row["nonevent_retention"], row["n_kept_nonevents"] / 8)
        self.assertAlmostEqual(row["random_rejection_event_retention_expected"], .5)
        self.assertAlmostEqual(row["random_rejection_nonevent_retention_expected"], .5)
        self.assertLessEqual(row["random_rejection_nonevent_retention_hypergeom_95_lo"], .5)
        self.assertGreaterEqual(row["random_rejection_nonevent_retention_hypergeom_95_hi"], .5)

    def test_sex_normalization_and_age_sex_cells_handle_float_source_codes(self):
        normalized = normalize_sex(pd.Series(
            [0, 1, 0.0, 1.0, "M", "Female", "unknown", None], dtype=object))
        self.assertEqual(normalized.iloc[:6].tolist(), [
            "female", "male", "female", "male", "male", "female",
        ])
        self.assertTrue(pd.isna(normalized.iloc[6]))
        self.assertTrue(pd.isna(normalized.iloc[7]))

        def indexed_cohort(*, unknown: bool = False) -> pd.DataFrame:
            ages: list[int] = []
            sexes: list[float | int] = []
            for age in (20, 55, 68, 80):
                ages.extend([age] * 20)
                sexes.extend([0.0] * 10 + [1.0] * 10)
            if unknown:
                ages.append(20)
                sexes.append(2)
            raw = pd.DataFrame({"age": ages, "sex": sexes})
            return pd.DataFrame({
                "age_group": age_group(raw["age"]),
                "sex_norm": normalize_sex(raw["sex"]),
            })

        counts = _age_sex_cell_counts({
            "mimic": indexed_cohort(), "eicu": indexed_cohort(unknown=True),
        })
        self.assertEqual(len(counts["eicu"]), 8)
        self.assertTrue(all(value == 10 for value in counts["mimic"].values()))
        self.assertTrue(all(value == 10 for value in counts["eicu"].values()))

    def test_hospital_pairing_requires_exact_ids_and_patient_stay_identity(self):
        left = pd.DataFrame({
            "db": ["eicu", "eicu"], "stay_id": ["s1", "s2"],
            "subject": ["p1", "p2"], "rows": [4, 5], "split": [0, 0],
            "y": [1, 0], "endpoint": ["hospital", "hospital"],
            "hospitalid": ["site1", "site2"], "p": [.7, .2],
        })
        right = left.copy()
        right["p"] = [.6, .1]
        paired = paired_frame(left, right, label="synthetic pair")
        self.assertEqual(len(paired), 2)
        self.assertEqual(paired["p_comparator"].tolist(), [.6, .1])

        changed_identity = right.copy()
        changed_identity.loc[1, "subject"] = "other-patient"
        with self.assertRaisesRegex(ValueError, "paired stay identity differs in subject"):
            paired_frame(left, changed_identity, label="synthetic pair")

        changed_ids = right.iloc[:1].copy()
        with self.assertRaisesRegex(ValueError, "exact eligible ID sets differ"):
            paired_frame(left, changed_ids, label="synthetic pair")

        strict_left = left.drop(columns="endpoint").copy()
        strict_left["endpoint_prediction"] = "hospital"
        strict_left["endpoint_eligibility"] = "hospital"
        strict_right = strict_left.copy()
        strict_right["p"] = [.6, .1]
        strict_right.loc[1, "endpoint_eligibility"] = "icu"
        with self.assertRaisesRegex(ValueError, "paired stay identity differs in endpoint_eligibility"):
            paired_frame(strict_left, strict_right, label="strict-loader pair")

    def test_hospital_metrics_retain_one_class_sites_and_exclude_missing_id_from_site_ci(self):
        frame = pd.DataFrame({
            "db": ["eicu"] * 5,
            "stay_id": ["h1a", "h1b", "h2a", "h2b", "missing"],
            "subject": ["p1", "p2", "p3", "p4", "p5"],
            "y": [0, 1, 0, 0, 1], "p": [.1, .8, .2, .15, .7],
            "hospitalid": ["h1", "h1", "h2", "h2", None],
        })
        overall, all_rows = _site_metric_table({"model": frame})
        selected_rows = _selected_site_metric_rows(all_rows)
        self.assertEqual(overall[0]["n_stays"], 5)
        self.assertEqual({row["hospitalid"] for row in selected_rows}, {"h1", "h2"})
        self.assertTrue(any(row["scope"] == "hospitalid_missing_aggregate" for row in all_rows))

        single_class = next(row for row in selected_rows if row["hospitalid"] == "h2")
        self.assertEqual(single_class["auroc_status"], "unestimable_single_class")
        self.assertEqual(single_class["average_precision_status"], "unestimable_no_events")
        self.assertFalse(single_class["site_stability_flag_ge10_events_and_nonevents"])

        intervals = _site_bootstrap_ci(
            selected_rows, group_field="model_id", metrics=("brier",),
            n_boot=1000, seed=19, crossing=True,
        )
        self.assertEqual(intervals[0]["n_hospitals_estimable"], 2)
        self.assertIn("patients cross hospitals", intervals[0]["ci_scope"])

    def test_hospital_site_count_preflight_reports_threshold_workload_without_site_ids(self):
        frame = pd.DataFrame({
            "db": ["eicu"] * 30,
            "hospitalid": ["h1"] * 20 + ["h2"] * 10,
            "y": [1] * 10 + [0] * 10 + [1] * 2 + [0] * 8,
        })
        result = _site_count_preflight({
            "main20k_real": frame,
            "main20k_constant": frame.copy(),
            "conventional_lr_value_plus_pattern": frame.copy(),
        }, n_boot=1000)
        self.assertEqual(result["selected_hospital_ids_with_current_predictions"], 2)
        for model in result["site_counts_by_model"].values():
            self.assertEqual(model["sites_ge_10_events_and_nonevents"], 1)
            self.assertEqual(model["sites_below_10_events_or_nonevents"], 1)
        self.assertEqual(result["workload"]["site_model_checkpoint_jobs"], 6)
        self.assertEqual(result["workload"]["event_threshold_eligible_site_model_bootstrap_upper_bound"], 3)
        self.assertNotIn("h1", json.dumps(result))
        self.assertNotIn("h2", json.dumps(result))

    def test_hospital_crossing_ignores_missing_id_sentinels(self):
        frame = pd.DataFrame({
            "subject": ["p1", "p1", "p2"],
            "hospitalid": ["site1", "site2", " NULL "],
        })
        crossing = patient_hospital_crossing(frame)
        self.assertEqual(crossing["n_mapped_patients"], 1)
        self.assertEqual(crossing["n_patients_crossing_hospitals"], 1)
        self.assertFalse(crossing["patients_are_nested_within_hospital"])

    def test_batched_patient_weighted_calibration_matches_scipy_minimize(self):
        rng = np.random.default_rng(731)
        n_patients = 60
        subjects = np.repeat(np.arange(n_patients), 2)
        base = np.repeat(np.linspace(-1.8, 1.8, n_patients), 2)
        y = (rng.random(len(base)) < expit(-0.2 + 0.35 * base)).astype(int)
        self.assertEqual(np.unique(y).size, 2)
        cluster_counts = 1 + (np.arange(n_patients) % 4)
        row_weights = cluster_counts[subjects].astype(float)

        for offset, scale in ((0.0, 0.5), (5.0, 2.0), (-7.0, 1.25)):
            logits = offset + scale * base
            estimates, failures = _solve_weighted_logistic_slope_batch(
                y, logits, row_weights.reshape(1, -1),
            )
            self.assertTrue(np.isfinite(estimates[0]), failures)

            def objective(beta):
                eta = beta[0] + beta[1] * logits
                loss = np.sum(row_weights * (np.logaddexp(0.0, eta) - y * eta))
                return float(loss + 0.5e-6 * beta[1] ** 2)

            def gradient(beta):
                eta = beta[0] + beta[1] * logits
                residual = row_weights * (expit(eta) - y)
                return np.array([
                    residual.sum(),
                    np.dot(residual, logits) + 1e-6 * beta[1],
                ])

            reference = minimize(
                objective, np.zeros(2), jac=gradient, method="BFGS",
                options={"gtol": 1e-10, "maxiter": 2000},
            )
            self.assertLess(np.linalg.norm(gradient(reference.x), ord=np.inf), 1e-5,
                            reference.message)
            self.assertAlmostEqual(float(estimates[0]), float(reference.x[1]), places=5)
            sklearn_reference = LogisticRegression(
                C=1e6, max_iter=5000, tol=1e-10, solver="lbfgs",
            ).fit(logits.reshape(-1, 1), y, sample_weight=row_weights)
            self.assertAlmostEqual(float(estimates[0]),
                                   float(sklearn_reference.coef_[0, 0]), places=4)

    def test_cluster_slope_bootstrap_flags_single_class_and_separation_draws(self):
        subjects = np.array([f"p{i}" for i in range(40)])
        p = expit(np.linspace(-3.0, 3.0, len(subjects)))
        for outcome in (np.zeros(40, dtype=int), np.ones(40, dtype=int)):
            result, draws = _patient_cluster_slope_bootstrap(
                outcome, p, subjects, n_boot=1000, seed=81,
            )
            self.assertEqual(result["calibration_slope_ci_status"], "unestimable_single_class")
            self.assertEqual(result["calibration_slope_ci_bootstrap_draws_invalid_single_class"], 1000)
            self.assertEqual(np.isfinite(draws).sum(), 0)

        x = np.linspace(-4.0, 4.0, len(subjects))
        separated_y = (x > 0).astype(int)
        separated, separated_draws = _patient_cluster_slope_bootstrap(
            separated_y, expit(x), subjects, n_boot=1000, seed=82,
        )
        self.assertEqual(separated["calibration_slope_ci_status"], "unstable_insufficient_valid_draws")
        self.assertGreater(separated["calibration_slope_ci_bootstrap_draws_invalid_separation"], 0)
        self.assertEqual(np.isfinite(separated_draws).sum(), 0)

        near_y = separated_y.copy()
        near_y[:3] = 1
        near_y[-3:] = 0
        near, near_draws = _patient_cluster_slope_bootstrap(
            near_y, expit(x), subjects, n_boot=1000, seed=83,
        )
        self.assertEqual(near["calibration_slope_ci_status"], "ok")
        self.assertGreater(near["calibration_slope_ci_bootstrap_draws_valid"], 800)
        self.assertTrue(np.isfinite(near["calibration_slope_ci_95_lo"]))
        self.assertTrue(np.isfinite(near["calibration_slope_ci_95_hi"]))
        self.assertTrue(np.isfinite(near_draws[np.isfinite(near_draws)]).all())

    def test_prediction_eligibility_join_is_subset_and_label_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction = self._prediction_rows()
            eligibility = {
                "mimic": self._eligible_table("mimic", [
                    ["mimic", "m1", "p1", True, True, 0, "hospital", "ok", 55, "F"],
                    ["mimic", "m2", "p2", True, True, 1, "hospital", "ok", 72, "M"],
                    ["mimic", "metadata-only", "p3", False, False, 0, "hospital", "not-predicted", 40, "F"],
                ]),
                "eicu": self._eligible_table("eicu", [
                    ["eicu", "e1", "e1p", True, True, 1, "hospital", "ok", 61, "F"],
                    ["eicu", "metadata-only", "e2p", False, False, 0, "hospital", "not-predicted", 43, "M"],
                ]),
            }
            entries = {db: self._cohort_entry(evidence=("exact" if db == "eicu" else "operational_proxy"))
                       for db in eligibility}
            prediction_path, prediction_manifest, elig_dir = self._write_prediction_artifacts(
                root, prediction, eligibility, entries)
            data, gates = load_checked(prediction_path, elig_dir,
                                       prediction_manifest_path=prediction_manifest)
            self.assertEqual(len(data["mimic"]), 2)
            self.assertEqual(gates["mimic"]["n_metadata_only_rows_not_in_prediction_file"], 1)
            self.assertTrue(gates["mimic"]["retrospective_landmark_prediction_allowed"], gates["mimic"])
            self.assertFalse(gates["mimic"]["ready_for_future24h_clinical_claim"])
            self.assertEqual(len(data["eicu"]), 1)

    def test_prediction_label_mismatch_fails_join(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction = self._prediction_rows(dbs=("eicu",))
            prediction.loc[:, "y"] = 0
            eligibility = {"eicu": self._eligible_table("eicu", [
                ["eicu", "e1", "e1p", True, True, 1, "hospital", "ok", 61, "F"],
            ])}
            entries = {"eicu": self._cohort_entry(evidence="exact")}
            prediction_path, prediction_manifest, elig = self._write_prediction_artifacts(
                root, prediction, eligibility, entries)
            with self.assertRaisesRegex(ValueError, "labels do not exactly match"):
                load_checked(prediction_path, elig, prediction_manifest_path=prediction_manifest)

    def test_global_legacy_and_pass_fields_do_not_open_cohort_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root.mkdir(exist_ok=True)
            (root / "eligibility_manifest.json").write_text(json.dumps({
                "ready_for_future24h_clinical_claim": True,
                "cohorts": {"mimic": {
                    "ready_for_future24h_clinical_claim": True,
                    "future24h_clinical_status": "PASS",
                    "evidence": "exact", "status": "ready",
                }},
            }))
            ready, reason = manifest_ready(root, "mimic")
            self.assertFalse(ready)
            self.assertIn("retrospective_landmark_prediction_not_attested", reason)

    def test_explicit_patient_identity_false_overrides_positive_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction = self._prediction_rows(dbs=("mimic",))
            eligibility = {"mimic": self._eligible_table("mimic", [
                ["mimic", "m1", "p1", True, True, 0, "hospital", "ok", 55, "F"],
                ["mimic", "m2", "p2", True, True, 1, "hospital", "ok", 72, "M"],
            ])}
            entry = self._cohort_entry()
            entry["subject_identity_basis"] = {
                "basis": "longitudinal patient_id, no verified longitudinal linkage",
                "patient_disjoint_claim_allowed": False,
            }
            prediction_path, prediction_manifest, elig = self._write_prediction_artifacts(
                root, prediction, eligibility, {"mimic": entry})
            with self.assertRaisesRegex(ValueError, "patient_identity_not_explicitly_verified"):
                load_checked(prediction_path, elig, prediction_manifest_path=prediction_manifest)

    def test_diagnostic_predictions_require_separate_mode_and_never_promote(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction = self._prediction_rows(
                scope="retrospective_association_unverified_landmark", dbs=("eicu",))
            eligibility = {"eicu": self._eligible_table("eicu", [
                ["eicu", "e1", "e1p", False, False, 1, "hospital", "timing-unknown", 61, "F"],
            ])}
            entry = self._cohort_entry(allowed=False, evidence="exact", identity_allowed=False,
                                       diagnostic_allowed=True)
            prediction_path, prediction_manifest, elig = self._write_prediction_artifacts(
                root, prediction, eligibility, {"eicu": entry})
            with self.assertRaisesRegex(ValueError, "prediction scope must be"):
                load_checked(prediction_path, elig, prediction_manifest_path=prediction_manifest)
            data, gates = load_checked(prediction_path, elig,
                                       prediction_manifest_path=prediction_manifest,
                                       diagnostic_only=True)
            self.assertEqual(len(data["eicu"]), 1)
            self.assertFalse(gates["eicu"]["retrospective_landmark_prediction_allowed"])
            self.assertFalse(gates["eicu"]["ready_for_future24h_clinical_claim"])
            self.assertFalse(gates["eicu"]["clinical_promotion_allowed"])

    def test_stale_prediction_hash_and_non_evaluation_role_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction = self._prediction_rows(dbs=("mimic",))
            eligibility = {"mimic": self._eligible_table("mimic", [
                ["mimic", "m1", "p1", True, True, 0, "hospital", "ok", 55, "F"],
                ["mimic", "m2", "p2", True, True, 1, "hospital", "ok", 72, "M"],
            ])}
            entry = self._cohort_entry()
            prediction_path, prediction_manifest, elig = self._write_prediction_artifacts(
                root, prediction, eligibility, {"mimic": entry})
            prediction_path.write_text(prediction_path.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prediction file hash differs"):
                load_checked(prediction_path, elig, prediction_manifest_path=prediction_manifest)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction = self._prediction_rows(role="train", dbs=("mimic",))
            eligibility = {"mimic": self._eligible_table("mimic", [
                ["mimic", "m1", "p1", True, True, 0, "hospital", "ok", 55, "F"],
                ["mimic", "m2", "p2", True, True, 1, "hospital", "ok", 72, "M"],
            ])}
            prediction_path, prediction_manifest, elig = self._write_prediction_artifacts(
                root, prediction, eligibility, {"mimic": self._cohort_entry()})
            with self.assertRaisesRegex(ValueError, "role=evaluation"):
                load_checked(prediction_path, elig, prediction_manifest_path=prediction_manifest)

    def test_prediction_model_and_condition_match_the_current_manifest(self):
        for field, wrong_value, error in (
            ("model_tag", "abl_const20k", "row-level model tag differs"),
            ("input_condition", "no_h0_therapy", "row-level input condition differs"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                prediction = self._prediction_rows(dbs=("mimic",))
                eligibility = {"mimic": self._eligible_table("mimic", [
                    ["mimic", "m1", "p1", True, True, 0, "hospital", "ok", 55, "F"],
                    ["mimic", "m2", "p2", True, True, 1, "hospital", "ok", 72, "M"],
                ])}
                prediction_path, prediction_manifest, elig = self._write_prediction_artifacts(
                    root, prediction, eligibility, {"mimic": self._cohort_entry()})
                document = json.loads(prediction_manifest.read_text(encoding="utf-8"))
                document["predictions"][0][field] = wrong_value
                prediction_manifest.write_text(json.dumps(document), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, error):
                    load_checked(prediction_path, elig, prediction_manifest_path=prediction_manifest)

    @unittest.skipUnless(os.environ.get("V4_CALIBRATION_PREFLIGHT") == "1",
                         "requires the approved server-side current prediction artifacts")
    def test_current_fm_and_baseline_manifests_load_and_share_exact_cohort_pools(self):
        rev = Path(os.environ.get(
            "EHRFM_REV_DIR",
            "/data/admin/fengshuo/ehr_fm_20261102/revision_v4_20261122",
        ))
        eligibility_dir = rev / "temporal/results/eligibility/status_consistent_v4_1/landmark_v4_1"
        fm_manifest = rev / "engine/results/prediction_runs/status_consistent_primary_core_fix2-7da980ffcbd8f00b/prediction_manifest.json"
        baseline_manifest = rev / "baselines/results/primary_status_consistent_20261123/results/baseline_prediction_manifest.json"
        fm_document = json.loads(fm_manifest.read_text(encoding="utf-8"))
        baseline_document = json.loads(baseline_manifest.read_text(encoding="utf-8"))
        validate_prediction_manifest(fm_document)
        validate_prediction_manifest(baseline_document)

        def record(document: dict, analysis_id: str, db: str) -> dict:
            matches = [item for item in document["predictions"]
                       if item.get("analysis_id") == analysis_id and item.get("db") == db]
            self.assertEqual(len(matches), 1, f"missing unique manifest record for {analysis_id}/{db}")
            return matches[0]

        fm_path = record(fm_document, "main20k_real", "mimic")["prediction_file"]
        fm_data, fm_gates = load_checked(
            fm_path, eligibility_dir, model_tag="main20k", analysis_id="main20k_real",
            prediction_manifest_path=fm_manifest,
        )
        self.assertEqual(set(fm_data), {"mimic", "eicu"})
        self.assertTrue(all(gate["retrospective_landmark_prediction_allowed"]
                            for gate in fm_gates.values()))

        baseline_data = {}
        for analysis_id, model_tag in (
            ("conventional_lr_pattern_only", "conventional_lr"),
            ("conventional_hgb_pattern_only", "conventional_hgb"),
            ("conventional_lr_value_plus_pattern", "conventional_lr"),
            ("conventional_hgb_value_plus_pattern", "conventional_hgb"),
        ):
            path = record(baseline_document, analysis_id, "mimic")["prediction_file"]
            loaded, gates = load_checked(
                path, eligibility_dir, model_tag=model_tag, analysis_id=analysis_id,
                prediction_manifest_path=baseline_manifest,
            )
            self.assertEqual(set(loaded), {"mimic", "eicu"})
            self.assertTrue(all(gate["retrospective_landmark_prediction_allowed"]
                                for gate in gates.values()))
            baseline_data[analysis_id] = loaded

        for db in ("mimic", "eicu"):
            arms = {"fm_main20k_real": fm_data[db]}
            for analysis_id, loaded in baseline_data.items():
                arms[analysis_id] = loaded[db]
            identity_frames = {
                name: frame[["db", "stay_id", "subject", "rows", "split", "y"]].copy()
                for name, frame in arms.items()
            }
            self.assertEqual(len(verify_shared_population_frames(identity_frames)), 64)


if __name__ == "__main__":
    unittest.main()

