#!/usr/bin/env python3
"""Aggregate-only paired BIG9 external metrics from frozen prediction artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
ENGINE = REV / "engine"
OUT = REV / "c_big_metrics"
GATE = REV / "temporal/results/eligibility/recovered_reviewed_v4_2"
BIG_RUN = ENGINE / "results/prediction_runs/big9_source_reuse_merged99_20261123-53f9735f920d7679"
CORE_RUN = ENGINE / "results/prediction_runs/recovered_external_primary_v4_2_resume3-04b21bfc05943b7f"
BIG_COMPLETION = REV / "c_big_source_reuse" / "big9_completion.json"
BIG_MANIFEST_SHA = "bb8df14f2a9f7405473f1b0ff7523325f5815ecc4d1133121cbedd4833bd28c1"
CORE_MANIFEST_SHA = "4a52de662bf1e28c165c0387d8a95fa3208afc46e255c1a57e6526ee61d78340"
RECOVERED_GATE_SHA = "d302ce0a493121f9336f40fd58c5dc63a75a663492293c0153b4ba05efbb8e8f"
MERGED99_SHA = "d467d4698a2bf8ce0cab994e55cf253a80e9f660566e1cf6b61398d4dfb925f2"
BOOTSTRAP_REPS = 1000
BASE_SEED = 20261123
TARGETS = ("nwicu_discharge_v4_1", "sicdb_zero_offset_v4_2", "eicu_semantic_v4_2")
BIG_ANALYSES = {
    "big50m_real": ("big50m", "real"),
    "big50m_full_real": ("big50m_full", "real"),
    "big50m_full_common8_mask": ("big50m_full", "common8_mask"),
}
CORE_ANALYSES = {
    "main20k_real": ("main20k", "real"),
    "main20k_common8_mask": ("main20k", "common8_mask"),
}
ANALYSES = {**BIG_ANALYSES, **CORE_ANALYSES}
CONTRASTS = {
    "big50m_real_minus_main20k_real": ("main20k_real", "big50m_real"),
    "big50m_full_real_minus_big50m_real": ("big50m_real", "big50m_full_real"),
    "big50m_full_common8_mask_minus_big50m_full_real": (
        "big50m_full_real", "big50m_full_common8_mask"),
    "big50m_full_common8_mask_minus_main20k_common8_mask": (
        "main20k_common8_mask", "big50m_full_common8_mask"),
}
METRICS = ("auroc", "average_precision", "brier", "logloss",
           "calibration_slope", "calibration_intercept", "offset_intercept")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, payload: dict, replace: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if not replace and path.exists():
            raise FileExistsError(path)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def read_prediction(manifest: dict, run_dir: Path, analysis_id: str,
                    db: str, expected: tuple[str, str], paired, cache: dict) -> tuple[pd.DataFrame, dict]:
    matches = [row for row in manifest.get("predictions", [])
               if row.get("analysis_id") == analysis_id and row.get("db") == db
               and int(row.get("n_predictions", 0)) > 0]
    if len(matches) != 1:
        raise ValueError(f"{analysis_id}/{db}: expected one positive manifest row, got {len(matches)}")
    entry = matches[0]
    path = Path(str(entry.get("prediction_file", "")))
    if not path.is_absolute() or not path.is_file():
        raise FileNotFoundError(f"{analysis_id}/{db}: missing prediction file {path}")
    if path not in cache:
        cache[path] = pd.read_parquet(path)
    frame = cache[path]
    if "db" not in frame:
        raise ValueError(f"{analysis_id}: prediction file has no db column")
    part = frame.loc[frame["db"].astype(str).eq(db)].copy()
    part = paired.validate_predictions(part, analysis_id)
    gate_hashes = paired.current_eligibility_hashes(db, GATE)
    paired.validate_manifest_entry(entry, analysis_id, db, path, gate_hashes)
    if (len(part) != int(entry["n_prediction_rows"])
            or paired.sha256_ordered_prediction_rows(part)
            != str(entry["ordered_row_digest_sha256"]).lower()):
        raise ValueError(f"{analysis_id}/{db}: prediction rows/digest differ from manifest")
    if (str(entry.get("model_tag")) != expected[0]
            or str(entry.get("input_condition")) != expected[1]):
        raise ValueError(f"{analysis_id}/{db}: model/condition differs from requested analysis")
    if entry.get("clinical_input_availability_verified") is not False:
        raise ValueError(f"{analysis_id}/{db}: clinical-input availability must remain unverified")
    return part, entry


def load_inputs() -> tuple[dict, dict, dict[str, dict[str, pd.DataFrame]], dict]:
    sys.path.insert(0, str(ENGINE / "code"))
    sys.path.insert(0, str(REV / "semantic_contrast"))
    sys.path.insert(0, str(REV / "big_metrics"))
    import paired_metrics_v4 as paired
    import semantic_contrast as semantic
    import aggregate_big_metrics as reviewed

    big_path = BIG_RUN / "prediction_manifest.json"
    core_path = CORE_RUN / "prediction_manifest.json"
    if sha256(big_path) != BIG_MANIFEST_SHA:
        raise ValueError("BIG9 source-reuse prediction-manifest SHA mismatch")
    if sha256(core_path) != CORE_MANIFEST_SHA:
        raise ValueError("reviewed core external prediction-manifest SHA mismatch")
    if sha256(GATE / "eligibility_manifest.json") != RECOVERED_GATE_SHA:
        raise ValueError("reviewed recovered gate manifest SHA mismatch")
    merged_manifest = REV / "c_big_source_reuse/embedding_manifest_merged99_c21_big9_d467d4698a2b.json"
    if sha256(merged_manifest) != MERGED99_SHA:
        raise ValueError("immutable merged99 source/component manifest SHA mismatch")
    completion_path = BIG_RUN / "target_source_probe_reuse_completion.json"
    completion = read_json(completion_path)
    if (completion.get("status") != "target_source_probe_reuse_complete"
            or completion.get("prediction_manifest_sha256") != BIG_MANIFEST_SHA
            or completion.get("embedding_manifest_sha256") != MERGED99_SHA
            or completion.get("positive_target_row_count") != 9
            or completion.get("expected_positive_target_row_count") != 9
            or completion.get("fit_source_probe_calls") != 0
            or completion.get("source_probe_refit_forbidden") is not True
            or completion.get("frozen_source_probability_reproduction", {}).get("status")
               != "source_probabilities_reproduced_exactly"
            or completion.get("frozen_source_probability_reproduction", {}).get("analysis_count") != 3):
        raise ValueError("BIG9 source-reuse completion record is incomplete")
    big = read_json(big_path)
    core = read_json(core_path)
    gate = read_json(GATE / "eligibility_manifest.json")
    rows_by_analysis: dict[str, dict[str, pd.DataFrame]] = {}
    entries = {}
    cache: dict[Path, pd.DataFrame] = {}
    for manifest, run_dir, requested, kind in (
        (big, BIG_RUN, BIG_ANALYSES, "big"),
        (core, CORE_RUN, CORE_ANALYSES, "core"),
    ):
        for analysis_id, identity in requested.items():
            for db in TARGETS:
                gate_entry = gate.get("cohorts", {}).get(db, {})
                if gate_entry.get("patient_disjoint_claim_allowed") is not True:
                    raise ValueError(f"{db}: reviewed gate does not permit patient-cluster inference")
                frame, entry = read_prediction(manifest, run_dir, analysis_id, db,
                                               identity, paired, cache)
                if entry.get("resampling_unit") == "record_id_proxy":
                    raise ValueError(f"{db}: patient bootstrap is incompatible with record_id_proxy")
                origin = entry.get("source_probe_origin_run_id")
                if kind == "big" and origin != "big_readout_frozen20_v4_2":
                    raise ValueError(f"{analysis_id}/{db}: wrong frozen BIG source origin {origin}")
                if kind == "big" and entry.get("source_probe_reused_from_prior_run") is not True:
                    raise ValueError(f"{analysis_id}/{db}: frozen BIG source probe was not reused")
                if kind == "core" and entry.get("source_probe_reused_from_prior_run") is not True:
                    raise ValueError(f"{analysis_id}/{db}: reviewed core source probe was not reused")
                if frame["endpoint"].astype(str).nunique() != 1:
                    raise ValueError(f"{analysis_id}/{db}: multiple endpoint populations in one target")
                rows_by_analysis.setdefault(analysis_id, {})[db] = frame
                entries[(analysis_id, db)] = entry
    reviewed.verify_weighted_helpers(semantic.weighted_metrics, paired)
    core_origins = sorted({str(entries[(analysis_id, db)].get("source_probe_origin_run_id", ""))
                           for analysis_id in CORE_ANALYSES for db in TARGETS})
    if not core_origins or "" in core_origins:
        raise ValueError("reviewed core external rows lack a bound source-probe origin")
    return big, core, rows_by_analysis, {
        "entries": entries, "completion": completion, "gate": gate,
        "paired": paired, "semantic": semantic, "reviewed": reviewed,
        "big_manifest_sha256": BIG_MANIFEST_SHA,
        "core_manifest_sha256": CORE_MANIFEST_SHA,
        "big_completion_sha256": sha256(completion_path),
        "gate_sha256": RECOVERED_GATE_SHA,
        "merged99_sha256": MERGED99_SHA,
        "core_source_probe_origin_run_ids": core_origins,
        "no_row_data_copied_local": True,
    }


def add_population(groups: dict, db: str, endpoint: str, identity: pd.DataFrame,
                   arm: str, probabilities: np.ndarray, reviewed) -> dict:
    signature = reviewed.identity_signature(identity)
    key = (db, endpoint, signature)
    group = groups.setdefault(key, {
        "db": db, "endpoint": endpoint, "identity": identity.copy(),
        "signature": signature, "arms": {}, "individuals": set(), "contrasts": [],
    })
    reviewed.add_arm(group, arm, identity, probabilities)
    return group


def metric_summary(values: np.ndarray, reviewed) -> tuple[float | None, float | None, int]:
    low, high, valid = reviewed.percentile(values)
    return (low if np.isfinite(low) else None,
            high if np.isfinite(high) else None, valid)


def bootstrap_groups(groups: dict, paired, semantic, reviewed) -> tuple[list[dict], list[dict]]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    metrics_rows: list[dict] = []
    flow_rows: list[dict] = []
    group_items = sorted(groups.items())
    for group_number, ((db, endpoint, signature), group) in enumerate(group_items, start=1):
        identity = group["identity"].sort_values(["db", "stay_id"], kind="mergesort").reset_index(drop=True)
        y = identity["y"].to_numpy(dtype=np.int8)
        subject = identity["subject"].astype(str).to_numpy()
        subjects, row_to_subject = np.unique(subject, return_inverse=True)
        if len(subjects) < 2 or len(np.unique(y)) != 2:
            raise ValueError(f"{db}/{endpoint}/{signature}: patient roster or labels are insufficient")
        arms = {name: group["arms"][name]["p"] for name in sorted(group["arms"])}
        score_groups = {name: semantic._sorted_score_groups(p) for name, p in arms.items()}
        point = {name: reviewed.metric_bundle(y, p, np.ones(len(y)), score_groups[name],
                                              semantic.weighted_metrics, paired)
                 for name, p in arms.items()}
        boot = {name: {metric: np.full(BOOTSTRAP_REPS, np.nan) for metric in METRICS}
                for name in arms}
        delta_boot = {contrast: {metric: np.full(BOOTSTRAP_REPS, np.nan) for metric in METRICS}
                      for contrast, _, _ in group["contrasts"]}
        optimizer_failures = {name: 0 for name in arms}
        one_class_draws = 0
        seed = paired.derive_seed(BASE_SEED, "big9_recovered_external", db, endpoint, signature)
        rng = np.random.default_rng(seed)
        actual_draw_checked = False
        for draw in range(BOOTSTRAP_REPS):
            sampled = rng.integers(0, len(subjects), size=len(subjects))
            counts = np.bincount(sampled, minlength=len(subjects)).astype(np.float64)
            weights = counts[row_to_subject]
            both_classes = len(np.unique(y[weights > 0])) == 2
            if not both_classes:
                one_class_draws += 1
            current = {}
            for name, p in arms.items():
                values = reviewed.metric_bundle(y, p, weights, score_groups[name],
                                                 semantic.weighted_metrics, paired)
                current[name] = values
                for metric in METRICS:
                    boot[name][metric][draw] = values[metric]
                if both_classes and (not np.isfinite(values["calibration_slope"])
                                     or not np.isfinite(values["calibration_intercept"])):
                    optimizer_failures[name] += 1
                if not actual_draw_checked and both_classes:
                    sklearn_values = paired.weighted_metrics(y, p, weights)
                    for metric in ("auroc", "average_precision", "brier", "logloss"):
                        if not np.isclose(values[metric], sklearn_values[metric], rtol=0, atol=1e-12):
                            raise AssertionError(f"{db}/{endpoint}: weighted {metric} disagrees with sklearn")
                    actual_draw_checked = True
            for contrast, left, right in group["contrasts"]:
                for metric in METRICS:
                    lv, rv = current[left][metric], current[right][metric]
                    if np.isfinite(lv) and np.isfinite(rv):
                        delta_boot[contrast][metric][draw] = rv - lv
            if (draw + 1) % 100 == 0:
                atomic_json(OUT / "big9_external_progress.json", {
                    "status": "running", "group_number": group_number,
                    "group_count": len(group_items), "cohort": db, "endpoint": endpoint,
                    "completed_draws": draw + 1, "requested_draws": BOOTSTRAP_REPS,
                    "population_identity_sha256": signature,
                    "one_class_draws": one_class_draws,
                    "invalid_optimizer_draws_by_arm": optimizer_failures,
                    "checkpoint_contents": "aggregate counts only; no row-level data",
                })
                print(f"BOOTSTRAP_PROGRESS group={group_number}/{len(group_items)} "
                      f"{db}/{endpoint} draws={draw+1}/{BOOTSTRAP_REPS}", flush=True)
        for name in sorted(group["individuals"]):
            for metric in METRICS:
                low, high, valid = metric_summary(boot[name][metric], reviewed)
                metrics_rows.append({
                    "row_type": "individual_full_roster", "cohort": db,
                    "endpoint": endpoint, "analysis_id": name,
                    "model_tag": ANALYSES[name][0], "input_condition": ANALYSES[name][1],
                    "contrast": "", "left_analysis_id": "", "right_analysis_id": "",
                    "metric": metric, "estimate": point[name][metric],
                    "difference_right_minus_left": "", "ci_95_low": low, "ci_95_high": high,
                    "valid_bootstrap": valid, "invalid_bootstrap": BOOTSTRAP_REPS - valid,
                    "optimizer_failure_draws": optimizer_failures[name],
                    "one_class_draws": one_class_draws, "n_stays": len(y),
                    "n_patients": len(subjects), "bootstrap_replicates": BOOTSTRAP_REPS,
                    "seed": seed,
                    "inference_scope": "full prediction roster; patient/subject-cluster bootstrap; conditional on frozen encoder and source-fitted probe",
                })
        for contrast, left, right in group["contrasts"]:
            for metric in METRICS:
                lv, rv = point[left][metric], point[right][metric]
                estimate = rv - lv if np.isfinite(lv) and np.isfinite(rv) else np.nan
                low, high, valid = metric_summary(delta_boot[contrast][metric], reviewed)
                metrics_rows.append({
                    "row_type": "paired_contrast", "cohort": db,
                    "endpoint": endpoint, "analysis_id": "", "model_tag": "",
                    "input_condition": "", "contrast": contrast,
                    "left_analysis_id": left, "right_analysis_id": right,
                    "metric": metric, "estimate": "",
                    "difference_right_minus_left": estimate,
                    "ci_95_low": low, "ci_95_high": high,
                    "valid_bootstrap": valid, "invalid_bootstrap": BOOTSTRAP_REPS - valid,
                    "optimizer_failure_draws": "", "one_class_draws": one_class_draws,
                    "n_stays": len(y), "n_patients": len(subjects),
                    "bootstrap_replicates": BOOTSTRAP_REPS, "seed": seed,
                    "inference_scope": "exact-key paired intersection; shared patient/subject-cluster weights; conditional on frozen encoders and source-fitted probes",
                })
        flow_rows.append({
            "cohort": db, "endpoint": endpoint, "row_type": "bootstrap_population",
            "contrast": "", "left_analysis_id": "", "right_analysis_id": "",
            "n_source_rows_left": "", "n_source_rows_right": "", "n_paired_rows": len(y),
            "n_excluded_left": "", "n_excluded_right": "", "n_unique_patients": len(subjects),
            "paired_identity_labels_verified": True, "population_identity_sha256": signature,
        })
        print(f"BOOTSTRAP_GROUP_COMPLETE {group_number}/{len(group_items)} {db}/{endpoint} "
              f"rows={len(y)} patients={len(subjects)} arms={len(arms)}", flush=True)
        if not actual_draw_checked:
            raise AssertionError("no actual weighted draw was cross-checked against sklearn")
    return metrics_rows, flow_rows


def build_groups(frames: dict[str, dict[str, pd.DataFrame]], paired, reviewed):
    groups: dict = {}
    flows: list[dict] = []
    for db in TARGETS:
        endpoint_values = {name: sorted(df["endpoint"].astype(str).unique())
                           for name, by_db in frames.items() for df in [by_db[db]]}
        endpoints = set().union(*map(set, endpoint_values.values()))
        if len(endpoints) != 1:
            raise ValueError(f"{db}: prediction configs disagree on target endpoint: {endpoint_values}")
        endpoint = next(iter(endpoints))
        for analysis_id, by_db in frames.items():
            part = by_db[db]
            identity = reviewed.identity_from_frame(part)
            group = add_population(groups, db, endpoint, identity, analysis_id,
                                   part["p"].to_numpy(dtype=float), reviewed)
            group["individuals"].add(analysis_id)

        for contrast, (left_id, right_id) in CONTRASTS.items():
            left, right = frames[left_id][db], frames[right_id][db]
            paired_rows, flow = paired.pair_rows(left, right, db, endpoint)
            flow.update({
                "row_type": "paired_contrast", "contrast": contrast,
                "left_analysis_id": left_id, "right_analysis_id": right_id,
                "n_source_rows_left": int(len(left)), "n_source_rows_right": int(len(right)),
                "n_paired_rows": int(len(paired_rows)),
                "n_excluded_left": int(len(left) - len(paired_rows)),
                "n_excluded_right": int(len(right) - len(paired_rows)),
                "n_unique_patients": int(paired_rows["subject"].nunique()),
            })
            if paired_rows.empty:
                raise ValueError(f"{db}/{contrast}: exact-key intersection is empty")
            identity = pd.DataFrame({
                "db": db, "stay_id": paired_rows["stay_id"],
                "subject": paired_rows["subject"], "rows": paired_rows["rows"],
                "split": paired_rows["split"], "y": paired_rows["y"],
                "endpoint": endpoint,
            })
            group = add_population(groups, db, endpoint, identity, left_id,
                                   paired_rows["p_a"].to_numpy(dtype=float), reviewed)
            add_population(groups, db, endpoint, identity, right_id,
                           paired_rows["p_b"].to_numpy(dtype=float), reviewed)
            group["contrasts"].append((contrast, left_id, right_id))
            flows.append(flow)
    if len(flows) != len(TARGETS) * len(CONTRASTS):
        raise AssertionError("BIG9 paired-flow grid is incomplete")
    expected_individual = len(TARGETS) * len(ANALYSES) * len(METRICS)
    expected_contrasts = len(TARGETS) * len(CONTRASTS) * len(METRICS)
    return groups, flows, expected_individual, expected_contrasts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    big, core, by_analysis, provenance = load_inputs()
    paired, reviewed = provenance["paired"], provenance["reviewed"]
    groups, pair_flows, expected_individual, expected_contrasts = build_groups(
        by_analysis, paired, reviewed)
    preflight = {
        "status": "preflight_passed", "target_cohorts": list(TARGETS),
        "big_manifest_sha256": provenance["big_manifest_sha256"],
        "core_external_manifest_sha256": provenance["core_manifest_sha256"],
        "recovered_reviewed_gate_sha256": provenance["gate_sha256"],
        "merged99_manifest_sha256": provenance["merged99_sha256"],
        "positive_analysis_cells": len(ANALYSES) * len(TARGETS),
        "pair_flow_rows": len(pair_flows),
        "unique_bootstrap_rosters": len(groups),
        "expected_individual_metric_rows": expected_individual,
        "expected_paired_metric_rows": expected_contrasts,
        "metrics": list(METRICS), "bootstrap_replicates": BOOTSTRAP_REPS,
        "bootstrap_unit": "patient/subject cluster as authorized by recovered reviewed gate",
        "full_and_common8_rosters_kept_separate": True,
        "patient_level_data_copied_local": False,
    }
    atomic_json(OUT / "big9_external_preflight.json", preflight)
    for db in TARGETS:
        counts = {name: len(by_analysis[name][db]) for name in ANALYSES}
        print(f"PREFLIGHT_TARGET {db} rows={json.dumps(counts, sort_keys=True)}", flush=True)
    print(f"PREFLIGHT_COMPLETE positive_cells={preflight['positive_analysis_cells']} "
          f"pair_flows={len(pair_flows)} bootstrap_rosters={len(groups)}", flush=True)
    if args.preflight_only:
        return 0

    metrics_rows, population_flows = bootstrap_groups(
        groups, paired, provenance["semantic"], reviewed)
    if (len([r for r in metrics_rows if r["row_type"] == "individual_full_roster"])
            != expected_individual):
        raise AssertionError("individual metric row grid is incomplete")
    if (len([r for r in metrics_rows if r["row_type"] == "paired_contrast"])
            != expected_contrasts):
        raise AssertionError("paired metric row grid is incomplete")
    flow_rows = pair_flows + population_flows
    result_tag = f"big9_{provenance['big_manifest_sha256'][:10]}_{provenance['core_manifest_sha256'][:10]}"
    metrics_path = OUT / f"{result_tag}_metrics.csv"
    flow_path = OUT / f"{result_tag}_pair_flow.csv"
    protocol_path = OUT / f"{result_tag}_protocol.json"
    for path in (metrics_path, flow_path, protocol_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite completed output: {path}")
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(flow_rows).to_csv(flow_path, index=False, quoting=csv.QUOTE_MINIMAL)
    protocol = {
        **preflight, "status": "complete",
        "big_run_id": big.get("run_id"), "core_external_run_id": core.get("run_id"),
        "big_completion_sha256": provenance["big_completion_sha256"],
        "big_metrics_input_scopes": {
            "targets": "reviewed recovered external primary eligible rows; retrospective measurement-time readout",
            "clinical_input_availability_verified": False,
            "big_model_pretraining_exposure": "historical exposure is not established absent; eICU native split=2 does not prove no prior exposure",
            "inference": "descriptive and conditional on frozen encoders and source-fitted probes; no causal or scaling-law claim",
        },
        "contrasts_right_minus_left": {name: {"left": pair[0], "right": pair[1]}
                                        for name, pair in CONTRASTS.items()},
        "confidence_intervals": "percentile 95% from 1,000 shared patient/subject-cluster weights on each exact prediction roster; invalid optimizer/class draws retained and counted",
        "calibration": "paired_metrics_v4 calibration parameters; slope penalty 1e-6, unpenalized intercept, fixed-slope offset intercept",
        "ranking_helper": "reviewed semantic_contrast weighted score-group helper; tied/zero-weight cross-check against sklearn and paired_metrics_v4 passed",
        "marginal_population_rule": "each individual arm uses its full positive prediction roster; common8 and exact pair intersections have separate bootstrap rosters",
        "source_probe_origins": {"big": "big_readout_frozen20_v4_2",
                                  "core_external": provenance["core_source_probe_origin_run_ids"]},
        "no_source_fitting_or_reencoding": True,
        "metrics_csv_path": str(metrics_path), "metrics_csv_sha256": sha256(metrics_path),
        "pair_flow_csv_path": str(flow_path), "pair_flow_csv_sha256": sha256(flow_path),
    }
    atomic_json(protocol_path, protocol, replace=False)
    print(f"BIG9_EXTERNAL_METRICS_COMPLETE metrics={metrics_path} "
          f"flow={flow_path} protocol={protocol_path} rows={len(metrics_rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
