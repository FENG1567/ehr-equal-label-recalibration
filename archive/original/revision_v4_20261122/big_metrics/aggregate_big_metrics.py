#!/usr/bin/env python3
"""Audited frozen-probe big-model mortality metrics and patient-cluster CIs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
ENGINE = REV / "engine"
BIG_ID = "big_readout_frozen20_v4_2-cd45103e235da78d"
CORE_ID = "status_consistent_primary_core_fix2-7da980ffcbd8f00b"
BIG_SHA = "ee1c581293fd337f96e2cd3ea299e910eabb863db28795c86e2e810e1125d32a"
CORE_SHA = "b2818a3443c8cf5fe34ef500beb79f3874c2729fa59c0397433d40be29d56971"
GATE_SHA = "ea2b55bc3cf071d48b019d4ba97b4a3dc53d5d7fb3f6ac49a9e7c26007064cb5"
BIG_ANALYSES = {
    "big50m_real": ("big50m", "real"),
    "big50m_full_real": ("big50m_full", "real"),
    "big50m_full_common8_mask": ("big50m_full", "common8_mask"),
}
CORE_ANALYSES = {
    "main20k_real": ("main20k", "real"),
    "main20k_common8_mask": ("main20k", "common8_mask"),
}
DBS = ("mimic", "eicu")
BOOTSTRAP_REPS = 1000
BASE_SEED = 20261123
OUT = REV / "big_metrics"
CONTRASTS = {
    "big50m_real_minus_main20k_real": ("main20k_real", "big50m_real"),
    "big50m_full_real_minus_big50m_real": ("big50m_real", "big50m_full_real"),
    "big50m_full_common8_mask_minus_big50m_full_real": (
        "big50m_full_real", "big50m_full_common8_mask"),
    "big50m_full_common8_mask_minus_main20k_common8_mask": (
        "main20k_common8_mask", "big50m_full_common8_mask"),
}
SLICES = {
    "mimic_test": ("mimic", 2),
    "eicu_all_eligible": ("eicu", None),
    "eicu_native_split2_sensitivity": ("eicu", 2),
}
METRICS = ("auroc", "average_precision", "brier", "logloss",
           "calibration_slope", "calibration_intercept", "offset_intercept")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_and_validate_inputs() -> tuple[dict, dict, dict[str, pd.DataFrame], dict[str, dict]]:
    code_dir = ENGINE / "code"
    sys.path.insert(0, str(code_dir))
    import paired_metrics_v4 as paired

    big_dir = ENGINE / "results/prediction_runs" / BIG_ID
    core_dir = ENGINE / "results/prediction_runs" / CORE_ID
    big_path = big_dir / "prediction_manifest.json"
    core_path = core_dir / "prediction_manifest.json"
    if sha256(big_path) != BIG_SHA:
        raise ValueError("pinned big-model manifest digest changed")
    if sha256(core_path) != CORE_SHA:
        raise ValueError("pinned original-core manifest digest changed")
    gate_dir = REV / "temporal/results/eligibility/status_consistent_v4_1/landmark_v4_1"
    if sha256(gate_dir / "eligibility_manifest.json") != GATE_SHA:
        raise ValueError("pinned landmark gate manifest digest changed")
    gate_hashes = {db: paired.current_eligibility_hashes(db, gate_dir) for db in DBS}

    big = json.loads(big_path.read_text())
    core = json.loads(core_path.read_text())
    frames: dict[str, pd.DataFrame] = {}
    entries: dict[str, dict] = {}
    for manifest, run_dir, requested in (
        (big, big_dir, BIG_ANALYSES), (core, core_dir, CORE_ANALYSES)
    ):
        for analysis_id, (model_tag, condition) in requested.items():
            matches = [e for e in manifest.get("predictions", [])
                       if e.get("analysis_id") == analysis_id
                       and int(e.get("n_predictions", 0)) > 0]
            if len(matches) != len(DBS):
                raise ValueError(f"{analysis_id}: expected positive MIMIC/eICU manifest entries, got {len(matches)}")
            p = run_dir / f"{analysis_id}.parquet"
            frame = pd.read_parquet(p)
            frame = paired.validate_predictions(frame, analysis_id)
            for entry in matches:
                db = str(entry["db"])
                if db not in DBS:
                    raise ValueError(f"{analysis_id}: unexpected cohort {db}")
                paired.validate_manifest_entry(entry, analysis_id, db, p, gate_hashes[db])
                part = frame.loc[frame["db"].astype(str).eq(db)].copy()
                if len(part) != int(entry["n_prediction_rows"]):
                    raise ValueError(f"{analysis_id}/{db}: row count differs from manifest")
                if paired.sha256_ordered_prediction_rows(part) != str(entry["ordered_row_digest_sha256"]).lower():
                    raise ValueError(f"{analysis_id}/{db}: ordered row digest differs from manifest")
                if part["prediction_scope"].astype(str).ne(entry["prediction_scope"]).any():
                    raise ValueError(f"{analysis_id}/{db}: prediction scope mismatch")
                if part["temporal_evidence"].astype(str).ne(entry["temporal_evidence"]).any():
                    raise ValueError(f"{analysis_id}/{db}: temporal evidence mismatch")
                for column in ("eligibility_manifest_sha256", "eligibility_table_sha256"):
                    if part[column].astype(str).str.lower().ne(gate_hashes[db][column]).any():
                        raise ValueError(f"{analysis_id}/{db}: stale {column}")
                if str(entry.get("model_tag")) != model_tag or str(entry.get("input_condition")) != condition:
                    raise ValueError(f"{analysis_id}/{db}: model/condition identity mismatch")
                entries[f"{analysis_id}/{db}"] = entry
                frames.setdefault(analysis_id, frame.loc[frame["db"].astype(str).isin(DBS)].copy())
    return big, core, frames, entries


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def percentile(values: np.ndarray) -> tuple[float, float, int]:
    valid = values[np.isfinite(values)]
    if not len(valid):
        return math.nan, math.nan, 0
    lo, hi = np.quantile(valid, [0.025, 0.975])
    return float(lo), float(hi), int(len(valid))


def metric_bundle(y: np.ndarray, p: np.ndarray, weights: np.ndarray,
                  groups: tuple[np.ndarray, np.ndarray], semantic_metrics,
                  paired_metrics) -> dict[str, float]:
    rank = semantic_metrics(y, p, weights, groups)
    slope, intercept, offset = paired_metrics.calibration_parameters(y, p, weights)
    return {
        "auroc": float(rank["auroc"]),
        "average_precision": float(rank["average_precision"]),
        "brier": float(rank["brier"]),
        "logloss": float(rank["logloss"]),
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "offset_intercept": float(offset),
    }


def verify_weighted_helpers(semantic_metrics, paired_metrics) -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray([0, 1, 0, 1, 1], dtype=np.int8)
    p = np.asarray([0.2, 0.7, 0.2, 0.4, 0.9], dtype=np.float64)
    weights = np.asarray([1, 2, 0, 3, 1], dtype=np.float64)
    groups = semantic_metrics.__globals__["_sorted_score_groups"](p)
    result = semantic_metrics(y, p, weights, groups)
    expected_auc = roc_auc_score(y, p, sample_weight=weights)
    expected_ap = average_precision_score(y, p, sample_weight=weights)
    ordinary = paired_metrics.weighted_metrics(y, p, weights)
    for key, expected in (("auroc", expected_auc), ("average_precision", expected_ap)):
        if not np.isclose(result[key], expected, rtol=0, atol=1e-12):
            raise AssertionError(f"semantic helper {key} disagrees with sklearn on tied/zero-weight data")
        if not np.isclose(result[key], ordinary[key], rtol=0, atol=1e-12):
            raise AssertionError(f"semantic helper {key} disagrees with paired_metrics_v4")
    for key in ("brier", "logloss"):
        if not np.isclose(result[key], ordinary[key], rtol=0, atol=1e-12):
            raise AssertionError(f"semantic helper {key} disagrees with paired_metrics_v4")


def identity_signature(identity: pd.DataFrame) -> str:
    h = hashlib.sha256()
    cols = ["db", "stay_id", "subject", "rows", "split", "y", "endpoint"]
    ordered = identity.sort_values(["db", "stay_id"], kind="mergesort")
    for row in ordered.loc[:, cols].itertuples(index=False, name=None):
        h.update(json.dumps([str(row[0]), str(row[1]), str(row[2]), int(row[3]),
                             int(row[4]), int(row[5]), str(row[6])],
                            separators=(",", ":")).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def identity_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, ["db", "stay_id", "subject", "rows", "split", "y", "endpoint"]].copy()


def add_arm(group: dict, name: str, identity: pd.DataFrame,
            probability: np.ndarray) -> None:
    p = np.asarray(probability, dtype=np.float64)
    if len(identity) != len(p):
        raise ValueError(f"{name}: aligned prediction length mismatch")
    aligned = identity.reset_index(drop=True).copy()
    aligned["_probability"] = p
    aligned = aligned.sort_values(["db", "stay_id"], kind="mergesort").reset_index(drop=True)
    ordered = aligned.drop(columns=["_probability"])
    p = aligned["_probability"].to_numpy(dtype=np.float64)
    existing = group["arms"].get(name)
    if existing is not None:
        if not np.array_equal(existing["p"], p):
            raise ValueError(f"{name}: conflicting predictions on an identical paired population")
    else:
        group["arms"][name] = {"identity": ordered, "p": p}


def selected_slice(frame: pd.DataFrame, db: str, split: int | None,
                   endpoint: str) -> pd.DataFrame:
    part = frame.loc[(frame["db"].astype(str) == db)
                     & (frame["endpoint"].astype(str) == endpoint)].copy()
    if split is not None:
        part = part.loc[part["split"].astype(int) == split].copy()
    return part


def compute_aggregates(frames: dict[str, pd.DataFrame], entries: dict[str, dict]) -> tuple[list[dict], list[dict], dict]:
    sys.path.insert(0, str(ENGINE / "code"))
    import paired_metrics_v4 as paired
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "semantic_contrast"))
    from semantic_contrast import _sorted_score_groups, weighted_metrics as semantic_weighted_metrics

    # Cross-check tied/zero-weight synthetic data against sklearn and the original
    # weighted implementation before using the helper on real bootstrap draws.
    verify_weighted_helpers(semantic_weighted_metrics, paired)

    metric_rows: list[dict] = []
    flow_rows: list[dict] = []
    groups: dict[tuple[str, str, str, str], dict] = {}
    expected_individual = set()
    expected_contrasts = set()

    for slice_name, (db, split) in SLICES.items():
        endpoint_set = sorted(set().union(*[
            set(frames[name].loc[(frames[name]["db"].astype(str) == db), "endpoint"].astype(str))
            for name in list(BIG_ANALYSES) + list(CORE_ANALYSES)
        ]))
        for endpoint in endpoint_set:
            selected = {name: selected_slice(frame, db, split, endpoint)
                        for name, frame in frames.items()}
            population_label = f"{slice_name}/{endpoint}"
            for analysis_id in BIG_ANALYSES:
                part = selected[analysis_id]
                if part.empty:
                    raise ValueError(f"missing individual big-model cell: {population_label}/{analysis_id}")
                identity = identity_from_frame(part)
                signature = identity_signature(identity)
                group = groups.setdefault((slice_name, db, endpoint, signature),
                                          {"identity": identity, "arms": {},
                                           "individuals": [], "contrasts": [],
                                           "signature": signature})
                add_arm(group, analysis_id, identity, part["p"].to_numpy(dtype=float))
                group["individuals"].append(analysis_id)
                expected_individual.add((slice_name, endpoint, analysis_id))

            for contrast, (left_id, right_id) in CONTRASTS.items():
                left = selected[left_id]
                right = selected[right_id]
                paired_rows, flow = paired.pair_rows(left, right, db, endpoint)
                flow.update({
                    "cohort_slice": slice_name, "analysis_id_left": left_id,
                    "analysis_id_right": right_id, "contrast": contrast,
                    "n_source_rows_left": int(len(left)),
                    "n_source_rows_right": int(len(right)),
                    "n_excluded_left": int(len(left) - len(paired_rows)),
                    "n_excluded_right": int(len(right) - len(paired_rows)),
                    "n_paired_stays": int(len(paired_rows)),
                    "n_paired_patients": int(paired_rows["subject"].nunique()),
                })
                flow_rows.append(flow)
                if paired_rows.empty:
                    raise ValueError(f"empty paired population: {population_label}/{contrast}")
                identity = pd.DataFrame({
                    "db": db, "stay_id": paired_rows["stay_id"],
                    "subject": paired_rows["subject"], "rows": paired_rows["rows"],
                    "split": paired_rows["split"], "y": paired_rows["y"],
                    "endpoint": endpoint,
                })
                signature = identity_signature(identity)
                group = groups.setdefault((slice_name, db, endpoint, signature),
                                          {"identity": identity, "arms": {},
                                           "individuals": [], "contrasts": [],
                                           "signature": signature})
                add_arm(group, left_id, identity, paired_rows["p_a"].to_numpy(dtype=float))
                add_arm(group, right_id, identity, paired_rows["p_b"].to_numpy(dtype=float))
                group["contrasts"].append((contrast, left_id, right_id))
                expected_contrasts.add((slice_name, endpoint, contrast))

    # Verify the complete endpoint-specific result grid before spending time on bootstraps.
    expected_eps = {(s, ep) for (s, _, ep, _) in groups}
    if len(expected_individual) != len(expected_eps) * len(BIG_ANALYSES):
        raise AssertionError("individual big-model result grid is incomplete")
    if len(expected_contrasts) != len(expected_eps) * len(CONTRASTS):
        raise AssertionError("paired contrast result grid is incomplete")

    actual_draw_checked = False
    total_groups = len(groups)
    for group_index, (group_key, group) in enumerate(sorted(groups.items()), start=1):
        slice_name, db, endpoint, signature = group_key
        identity = group["identity"].sort_values(["db", "stay_id"], kind="mergesort").reset_index(drop=True)
        y = identity["y"].to_numpy(dtype=np.int8)
        subjects, group_index_by_row = np.unique(identity["subject"].astype(str), return_inverse=True)
        if not len(subjects):
            raise ValueError(f"empty patient set in {group_key}")
        arm_names = sorted(group["arms"])
        arrays = {name: group["arms"][name]["p"] for name in arm_names}
        sorted_groups = {name: _sorted_score_groups(p) for name, p in arrays.items()}
        point = {name: metric_bundle(y, arrays[name], np.ones(len(y)),
                                     sorted_groups[name], semantic_weighted_metrics, paired)
                 for name in arm_names}
        boot = {name: {metric: np.full(BOOTSTRAP_REPS, np.nan)
                       for metric in METRICS} for name in arm_names}
        calibration_failures = {name: 0 for name in arm_names}
        degenerate_class_draws = {name: 0 for name in arm_names}
        seed = paired.derive_seed(BASE_SEED, slice_name, db, endpoint, signature)
        rng = np.random.default_rng(seed)
        for draw in range(BOOTSTRAP_REPS):
            chosen = rng.integers(0, len(subjects), size=len(subjects))
            counts = np.bincount(chosen, minlength=len(subjects)).astype(np.float64)
            row_weights = counts[group_index_by_row]
            both_classes = len(np.unique(y[row_weights > 0])) == 2
            for name in arm_names:
                values = metric_bundle(y, arrays[name], row_weights,
                                       sorted_groups[name], semantic_weighted_metrics, paired)
                for metric in METRICS:
                    boot[name][metric][draw] = values[metric]
                if not both_classes:
                    degenerate_class_draws[name] += 1
                elif not np.isfinite(values["calibration_slope"]) or not np.isfinite(values["calibration_intercept"]):
                    calibration_failures[name] += 1
                if not actual_draw_checked and both_classes:
                    sklearn_values = paired.weighted_metrics(y, arrays[name], row_weights)
                    for metric in ("auroc", "average_precision", "brier", "logloss"):
                        if not np.isclose(values[metric], sklearn_values[metric], rtol=0, atol=1e-12):
                            raise AssertionError(f"actual bootstrap draw disagrees with sklearn weighted {metric}")
                    actual_draw_checked = True
            if (draw + 1) % 100 == 0:
                partial = []
                for name in arm_names:
                    for metric in METRICS:
                        lo, hi, valid = percentile(boot[name][metric][:draw + 1])
                        partial.append({"analysis_id": name, "metric": metric,
                                        "ci_low_so_far": lo if np.isfinite(lo) else None,
                                        "ci_high_so_far": hi if np.isfinite(hi) else None,
                                        "valid_draws": valid,
                                        "invalid_draws": int(draw + 1 - valid)})
                atomic_json(OUT / "aggregate_checkpoint.json", {
                    "status": "running", "cohort_slice": slice_name,
                    "db": db, "endpoint": endpoint, "population_sha256": signature,
                    "group_number": group_index, "group_count": total_groups,
                    "completed_draws": draw + 1, "requested_draws": BOOTSTRAP_REPS,
                    "checkpoint_contents": "aggregate metric intervals only; no row-level data",
                    "arm_metric_summaries": partial,
                })
                progress = {
                    "status": "running", "group_number": group_index,
                    "group_count": total_groups, "cohort_slice": slice_name,
                    "db": db, "endpoint": endpoint,
                    "completed_draws": draw + 1, "requested_draws": BOOTSTRAP_REPS,
                    "population_sha256": signature,
                }
                atomic_json(OUT / "progress.json", progress)
                print(f"BOOTSTRAP_PROGRESS group={group_index}/{total_groups} "
                      f"slice={slice_name} endpoint={endpoint} draws={draw+1}/{BOOTSTRAP_REPS}",
                      flush=True)

        for name in group["individuals"]:
            if name not in BIG_ANALYSES:
                continue
            for metric in METRICS:
                lo, hi, valid = percentile(boot[name][metric])
                metric_rows.append({
                    "row_type": "individual", "cohort_slice": slice_name,
                    "db": db, "endpoint": endpoint, "analysis_id": name,
                    "model_tag": BIG_ANALYSES[name][0],
                    "input_condition": BIG_ANALYSES[name][1],
                    "metric": metric, "estimate": point[name][metric],
                    "ci_95_low": lo, "ci_95_high": hi,
                    "valid_bootstrap": valid, "invalid_bootstrap": BOOTSTRAP_REPS - valid,
                    "calibration_fit_failures": calibration_failures[name],
                    "degenerate_class_draws": degenerate_class_draws[name],
                    "n_stays": int(len(y)), "n_patients": int(len(subjects)),
                    "n_bootstrap": BOOTSTRAP_REPS, "seed": seed,
                    "inference_scope": "conditional on frozen encoder and fitted source probes",
                })
        for contrast, left_id, right_id in group["contrasts"]:
            for metric in METRICS:
                delta_point = (point[right_id][metric] - point[left_id][metric]
                               if np.isfinite(point[right_id][metric]) and np.isfinite(point[left_id][metric])
                               else math.nan)
                delta = boot[right_id][metric] - boot[left_id][metric]
                lo, hi, valid = percentile(delta)
                metric_rows.append({
                    "row_type": "paired_contrast", "cohort_slice": slice_name,
                    "db": db, "endpoint": endpoint, "contrast": contrast,
                    "analysis_id_left": left_id, "analysis_id_right": right_id,
                    "metric": metric, "difference_right_minus_left": delta_point,
                    "ci_95_low": lo, "ci_95_high": hi,
                    "valid_bootstrap": valid, "invalid_bootstrap": BOOTSTRAP_REPS - valid,
                    "n_stays": int(len(y)), "n_patients": int(len(subjects)),
                    "n_bootstrap": BOOTSTRAP_REPS, "seed": seed,
                    "inference_scope": "conditional paired patient-cluster bootstrap on exact-key intersection",
                })
        print(f"BOOTSTRAP_GROUP_COMPLETE group={group_index}/{total_groups} "
              f"slice={slice_name} endpoint={endpoint} arms={len(arm_names)}",
              flush=True)

    if not actual_draw_checked:
        raise AssertionError("no actual weighted bootstrap draw was cross-checked against sklearn")
    # Each requested individual cell has all metrics and every comparison has all metrics.
    individual_metrics = {(r["cohort_slice"], r["endpoint"], r["analysis_id"], r["metric"])
                          for r in metric_rows if r["row_type"] == "individual"}
    contrast_metrics = {(r["cohort_slice"], r["endpoint"], r["contrast"], r["metric"])
                        for r in metric_rows if r["row_type"] == "paired_contrast"}
    all_endpoints = {(s, ep) for (s, _, ep, _) in groups}
    if len(individual_metrics) != len(all_endpoints) * len(BIG_ANALYSES) * len(METRICS):
        raise AssertionError("individual metric result grid is incomplete")
    if len(contrast_metrics) != len(all_endpoints) * len(CONTRASTS) * len(METRICS):
        raise AssertionError("paired contrast metric result grid is incomplete")
    protocol = {
        "status": "complete", "big_manifest_sha256": BIG_SHA,
        "core_manifest_sha256": CORE_SHA, "landmark_gate_manifest_sha256": GATE_SHA,
        "big_analyses": {k: {"model_tag": v[0], "input_condition": v[1]}
                         for k, v in BIG_ANALYSES.items()},
        "core_analyses_for_contrasts": list(CORE_ANALYSES),
        "cohort_slices": {
            "mimic_test": "original MIMIC test partition only (split=2)",
            "eicu_all_eligible": "all eligible rows in original eICU prediction flow",
            "eicu_native_split2_sensitivity": (
                "original eICU native split=2 rows only; this partition does not prove absence of historical pretraining exposure"),
        },
        "contrasts": {name: {"left": pair[0], "right": pair[1],
                             "reported_difference": "right minus left"}
                      for name, pair in CONTRASTS.items()},
        "paired_intersection": "exact db/stay_id key intersection verified for subject, rows, split, outcome, and endpoint via paired_metrics_v4.pair_rows",
        "metrics": list(METRICS), "bootstrap_replicates": BOOTSTRAP_REPS,
        "base_seed": BASE_SEED,
        "bootstrap_unit": "patient/subject cluster sampled with replacement; all stays retained",
        "ci": "percentile 95% bootstrap interval; per-cell full prediction population, paired contrasts on exact intersection",
        "calibration": "paired_metrics_v4 original weighted slope/intercept fit, slope penalty 1e-6, intercept unpenalized; offset is fixed-slope intercept recalibration",
        "ranking_metrics": "verified semantic_contrast descending weighted score-group helper; tied-score and zero-weight synthetic and actual weighted draws cross-checked with sklearn and paired_metrics_v4",
        "parameter_counts": {"main20k": 409099, "big50m": 56778251,
                             "big50m_full": 56778251},
        "interpretation_limit": "descriptive comparisons only; no scaling-law, causal, or more-parameters claim; pretraining data/epochs differ and BIG training membership is unknown",
        "pretraining_exposure": "big-model historical pretraining exposure is documented; the original eICU split=2 sensitivity does not establish absence of historical exposure",
        "checkpoint_metadata_audit_sha256": sha256(REV / "admin/checkpoint_metadata_audit.json"),
        "all_invalid_bootstrap_counts_reported_per_metric": True,
        "source_loading": "existing paired_metrics_v4 validation functions used directly; immutable run manifests and prediction artifacts were hash/digest checked",
        "weighted_actual_draw_crosscheck": actual_draw_checked,
    }
    return metric_rows, flow_rows, protocol


def write_outputs(metric_rows: list[dict], flow_rows: list[dict], protocol: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(OUT / "big_metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(flow_rows).to_csv(OUT / "big_pair_flow.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    atomic_json(OUT / "big_metrics_protocol.json", protocol)
    atomic_json(OUT / "progress.json", {
        "status": "complete", "completed_groups": len({(r.get("cohort_slice"), r.get("db"), r.get("endpoint")) for r in flow_rows}),
        "metric_rows": len(metric_rows), "flow_rows": len(flow_rows),
    })
    atomic_json(OUT / "aggregate_checkpoint.json", {
        "status": "complete", "metric_rows": len(metric_rows),
        "flow_rows": len(flow_rows), "checkpoint_contents": "aggregate metrics and intervals only",
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    big, core, frames, entries = load_and_validate_inputs()
    for key, entry in sorted(entries.items()):
        print(f"PREFLIGHT_OK {key} rows={entry['n_prediction_rows']} digest={entry['ordered_row_digest_sha256']}", flush=True)
    print(f"PREFLIGHT_COMPLETE big={big['run_id']} core={core['run_id']} analysis_files={len(frames)} cells={len(entries)}", flush=True)
    if not args.preflight_only:
        metric_rows, flow_rows, protocol = compute_aggregates(frames, entries)
        write_outputs(metric_rows, flow_rows, protocol)
        print(f"BIG_METRICS_COMPLETE metrics={len(metric_rows)} flow={len(flow_rows)} out={OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
