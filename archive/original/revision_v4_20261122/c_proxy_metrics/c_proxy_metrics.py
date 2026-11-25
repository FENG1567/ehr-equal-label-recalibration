"""Paired source-score metrics for C2012 under missing-magnesium proxy input."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
ENGINE_CODE = REV / "engine/code"
SOURCE_RUN = REV / "engine/results/prediction_runs/c21_source_reuse_merged99_20261123-88c3d7554db2ea17"
SOURCE_MANIFEST = SOURCE_RUN / "prediction_manifest.json"
SOURCE_COMPLETION = SOURCE_RUN / "target_source_probe_reuse_completion.json"
NOMG_DIR = REV / "controls/no_mg"
NOMG_PREDICTIONS = NOMG_DIR / "patient_predictions.npz"
NOMG_AGGREGATE = NOMG_DIR / "no_mg_aggregate_v4_1.json"
NOMG_RANK_AUDIT = NOMG_DIR / "no_mg_rank_recovery_audit_v4_1.json"
GATE_DIR = REV / "temporal/results/eligibility/c2012_external_reviewed_v4_2"
OUT = REV / "c_proxy_metrics/results/paired_c_proxy_full_vs_no_mg_v1"
HELPER_SHA256 = "52333539bbe73367e75c0d001497b8872d2e43fbaef71f1e382d49c02c536841"
EXPECTED_PREDICTION_MANIFEST_SHA256 = "74897f72a3be38bf2ef5a71c244d5db1cd1bfe81e89d7e72198d9162867ad0e1"
EXPECTED_NOMG_SHA256 = "20332c11eaf12e2bb00306f25024f93978b60de4a3dd99caea57b7fbfbea4a12"
EXPECTED_NOMG_AGGREGATE_SHA256 = "42b81883d2fc65ef922f50329514e2955ae3b21d10426b9a8aa9c07d1fa73d38"
EXPECTED_NOMG_RANK_AUDIT_SHA256 = "71c25f4404eb543a4d5dc205cf629c7ebe5766014e588a2f9ce45d42fcce1080"
COHORTS = ("c2012_a_v4_1", "c2012_b_v4_1", "c2012_c_v4_1")
EXPECTED_STAYS = {"c2012_a_v4_1": 3964, "c2012_b_v4_1": 3965, "c2012_c_v4_1": 3957}
EXPECTED_EVENTS = {"c2012_a_v4_1": 528, "c2012_b_v4_1": 543, "c2012_c_v4_1": 549}
N_BOOT = 1000
SEED = 20261123
METRICS = ("auroc", "average_precision", "brier", "logloss",
           "calibration_slope", "calibration_intercept", "offset_intercept")

sys.path.insert(0, str(ENGINE_CODE))
import paired_metrics_v4 as pm  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_hash(path: Path, expected: str, name: str) -> str:
    actual = _sha256(path)
    if actual.lower() != expected.lower():
        raise ValueError(f"{name} SHA256 mismatch: {actual}")
    return actual


def _manifest_entry(manifest: dict[str, Any], cohort: str) -> dict[str, Any]:
    rows = [row for row in manifest.get("predictions", [])
            if row.get("analysis_id") == "main20k_real" and row.get("db") == cohort]
    if len(rows) != 1:
        raise ValueError(f"full-source manifest has {len(rows)} main20k_real/{cohort} entries")
    return rows[0]


def _full_frames() -> tuple[dict[str, pd.DataFrame], dict[str, Any], dict[str, Any], dict[str, str]]:
    manifest_hash = _verify_hash(SOURCE_MANIFEST, EXPECTED_PREDICTION_MANIFEST_SHA256,
                                 "C2012 source-reuse prediction manifest")
    manifest = _json(SOURCE_MANIFEST)
    completion = _json(SOURCE_COMPLETION)
    if (completion.get("status") != "target_source_probe_reuse_complete"
            or completion.get("clinical_input_availability_verified") is not False
            or completion.get("clinical_ready") is not False
            or completion.get("fit_source_probe_calls") != 0
            or completion.get("source_probe_refit_forbidden") is not True
            or str(completion.get("prediction_manifest_sha256", "")).lower() != manifest_hash.lower()):
        raise ValueError("source-reuse completion is incomplete, refit-permitting, or clinically promotable")
    source_repros = [row for row in completion.get("frozen_source_probability_reproduction", {}).get("comparisons", [])
                     if row.get("db") == "mimic"]
    if len(source_repros) != 7 or any(
            row.get("identity_exact") is not True or row.get("probabilities_bitwise_exact") is not True
            for row in source_repros):
        raise ValueError("completion does not attest all seven bitwise-exact original MIMIC source probabilities")
    target_entries = [_manifest_entry(manifest, cohort) for cohort in COHORTS]
    prediction_paths = {Path(row["prediction_file"]).resolve() for row in target_entries}
    prediction_hashes = {str(row.get("prediction_file_sha256", "")).lower() for row in target_entries}
    if len(prediction_paths) != 1 or len(prediction_hashes) != 1:
        raise ValueError("C2012 full-source entries do not share one prediction file/hash")
    prediction_path = next(iter(prediction_paths))
    all_frames = pd.read_parquet(prediction_path, filters=[("db", "in", list(COHORTS))])
    all_frames = pm.validate_predictions(all_frames, "main20k_real")
    frames, identity = {}, {}
    for cohort, entry in zip(COHORTS, target_entries):
        hashes = pm.current_eligibility_hashes(cohort, GATE_DIR)
        pm.validate_manifest_entry(entry, "main20k_real", cohort, prediction_path, hashes)
        frame = all_frames.loc[all_frames["db"].astype(str).eq(cohort)].copy()
        if len(frame) != int(entry["n_prediction_rows"]):
            raise ValueError(f"{cohort}: prediction count differs from its manifest")
        digest = pm.sha256_ordered_prediction_rows(frame)
        if digest.lower() != str(entry["ordered_row_digest_sha256"]).lower():
            raise ValueError(f"{cohort}: ordered prediction row digest mismatch")
        if frame["clinical_input_availability_verified"].astype(str).str.lower().ne("false").any():
            raise ValueError(f"{cohort}: clinical input availability is claimed")
        if not frame["subject"].astype(str).equals(frame["stay_id"].astype(str)):
            # C2012 records are not patients; manifest subject must be the declared record proxy.
            if frame["subject"].isna().any() or frame["subject"].astype(str).isin(("", "nan", "None")).any():
                raise ValueError(f"{cohort}: record_id_proxy subject values are absent")
        frames[cohort] = frame
        identity[cohort] = {
            "prediction_file": str(prediction_path),
            "prediction_file_sha256": str(entry["prediction_file_sha256"]),
            "ordered_row_digest_sha256": digest,
            "eligibility_manifest_sha256": hashes["eligibility_manifest_sha256"],
            "eligibility_table_sha256": hashes["eligibility_table_sha256"],
            "source_npz_sha256": hashes["source_npz_sha256"],
            "n_full_source_rows": int(len(frame)), "full_events": int(frame["y"].sum()),
        }
    return frames, manifest, completion, {"prediction_manifest_sha256": manifest_hash,
                                           **{f"full_{k}": v for cohort, row in identity.items()
                                              for k, v in row.items() if k.endswith("sha256")}}


def _inspect_nomg() -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    _verify_hash(NOMG_PREDICTIONS, EXPECTED_NOMG_SHA256, "noMg patient predictions archive")
    _verify_hash(NOMG_AGGREGATE, EXPECTED_NOMG_AGGREGATE_SHA256, "noMg corrected aggregate")
    _verify_hash(NOMG_RANK_AUDIT, EXPECTED_NOMG_RANK_AUDIT_SHA256, "noMg rank recovery audit")
    aggregate = _json(NOMG_AGGREGATE)
    rank_audit = _json(NOMG_RANK_AUDIT)
    if aggregate.get("status") not in {"complete_no_mg", "complete", "complete_corrected_rank_metrics"}:
        raise ValueError(f"noMg aggregate status is not complete: {aggregate.get('status')}")
    if not rank_audit:
        raise ValueError("noMg rank recovery audit is empty")
    with np.load(NOMG_PREDICTIONS, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return arrays, aggregate, rank_audit


def _preflight() -> dict[str, Any]:
    _verify_hash(ENGINE_CODE / "paired_metrics_v4.py", HELPER_SHA256,
                 "paired_metrics_v4 helper code")
    full, manifest, completion, input_hashes = _full_frames()
    arrays, aggregate, rank_audit = _inspect_nomg()
    cohort_summary = aggregate.get("cohorts", {})
    if set(COHORTS).difference(cohort_summary):
        raise ValueError("noMg corrected aggregate lacks one or more C2012 cohorts")
    shapes = {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
              for key, value in arrays.items()}
    manifest_entries = {cohort: {
        "n_full_source_rows": len(full[cohort]), "n_full_events": int(full[cohort]["y"].sum()),
        "n_nomg_scored": cohort_summary[cohort].get("n_scored"),
        "n_record_id_proxy": cohort_summary[cohort].get("n_record_id_proxy"),
        "no_mg_vs_b1": cohort_summary[cohort].get("no_mg_vs_b1"),
        "npz_candidate_keys": [key for key in arrays if cohort in key],
        "expected_full_rows": EXPECTED_STAYS[cohort],
        "expected_full_events": EXPECTED_EVENTS[cohort],
    } for cohort in COHORTS}
    if any(len(full[cohort]) != EXPECTED_STAYS[cohort]
           or int(full[cohort]["y"].sum()) != EXPECTED_EVENTS[cohort]
           for cohort in COHORTS):
        raise ValueError("full source rows/events differ from registered C2012 denominators")
    result = {
        "status": "artifacts_verified_npz_keys_inspected",
        "cohorts": manifest_entries, "npz_keys_and_shapes": shapes,
        "npz_path_server_only": str(NOMG_PREDICTIONS),
        "aggregate_hashes": {
            "prediction_manifest_sha256": input_hashes["prediction_manifest_sha256"],
            "patient_predictions_sha256": _sha256(NOMG_PREDICTIONS),
            "nomg_aggregate_sha256": _sha256(NOMG_AGGREGATE),
            "rank_recovery_audit_sha256": _sha256(NOMG_RANK_AUDIT),
            "paired_metrics_helper_sha256": _sha256(ENGINE_CODE / "paired_metrics_v4.py"),
        },
        "source_reuse_run_id": manifest.get("run_id"),
        "clinical_input_availability_verified": False,
        "patient_identity_unit": "record_id_proxy; not verified as independent patients",
        "patient_level_exported_to_local": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "preflight_manifest.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["preflight_manifest_path"] = str(path)
    result["preflight_manifest_sha256"] = _sha256(path)
    print(json.dumps({"stage": "c_proxy_artifact_preflight_complete",
                      "manifest": str(path), "manifest_sha256": result["preflight_manifest_sha256"],
                      "npz_keys": list(arrays),
                      "cohort_shapes": {db: {"full": len(full[db]),
                                             "nomg": manifest_entries[db]["n_nomg_scored"]}
                                        for db in COHORTS}}, sort_keys=True), flush=True)
    return result


def _nomg_cohort_frame(arrays: dict[str, np.ndarray], cohort: str,
                       aggregate: dict[str, Any]) -> pd.DataFrame:
    """Build a cohort table from the exact, versioned noMg archive fields."""
    fields = {
        "stay_id": f"{cohort}_stay_id",
        "record_id_proxy": f"{cohort}_record_id_proxy",
        "y": f"{cohort}_y",
        "p_nomg": f"{cohort}_p_nomg",
    }
    missing = [key for key in fields.values() if key not in arrays]
    if missing:
        raise ValueError(f"{cohort}: noMg archive is missing required fields: {missing}")
    values = {name: np.asarray(arrays[key]) for name, key in fields.items()}
    if any(value.ndim != 1 for value in values.values()):
        raise ValueError(f"{cohort}: noMg identity, label, and score fields must be one-dimensional")
    lengths = {len(value) for value in values.values()}
    if len(lengths) != 1:
        raise ValueError(f"{cohort}: noMg field lengths differ: {sorted(lengths)}")

    frame = pd.DataFrame({
        "stay_id": values["stay_id"].astype(str),
        "record_id_proxy": values["record_id_proxy"].astype(str),
        "y": pd.to_numeric(pd.Series(values["y"]), errors="coerce"),
        "p_nomg": pd.to_numeric(pd.Series(values["p_nomg"]), errors="coerce"),
    })
    if frame["stay_id"].isin(("", "nan", "None", "-1")).any():
        raise ValueError(f"{cohort}: noMg stay IDs contain missing values")
    if frame["stay_id"].duplicated().any():
        raise ValueError(f"{cohort}: noMg archive contains duplicate stay IDs")
    if frame["record_id_proxy"].isin(("", "nan", "None", "-1")).any():
        raise ValueError(f"{cohort}: noMg record_id_proxy values contain missing values")
    if not frame["y"].isin((0, 1)).all():
        raise ValueError(f"{cohort}: noMg labels are nonbinary or missing")
    frame["y"] = frame["y"].astype(np.int8)
    if not np.isfinite(frame["p_nomg"].to_numpy(float)).all():
        raise ValueError(f"{cohort}: noMg probabilities contain non-finite values")
    if not frame["p_nomg"].between(0, 1, inclusive="both").all():
        raise ValueError(f"{cohort}: noMg probabilities fall outside [0, 1]")

    expected_scored = int(aggregate.get("n_scored", -1))
    expected_proxies = int(aggregate.get("n_record_id_proxy", -1))
    if len(frame) != EXPECTED_STAYS[cohort] or len(frame) != expected_scored:
        raise ValueError(f"{cohort}: noMg row count differs from registered/aggregate count")
    if int(frame["record_id_proxy"].nunique()) != expected_proxies:
        raise ValueError(f"{cohort}: noMg unique record proxy count differs from aggregate")
    if int(frame["y"].sum()) != EXPECTED_EVENTS[cohort]:
        raise ValueError(f"{cohort}: noMg event count differs from registered C2012 labels")
    return frame


def _pair_full_no_mg(full: pd.DataFrame, nomg: pd.DataFrame,
                     cohort: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join exact stays, verify proxy/label identity, and expose aggregate flow only."""
    full = full.loc[full["db"].astype(str).eq(cohort)].copy()
    if not full["endpoint"].astype(str).eq("hospital").all():
        raise ValueError(f"{cohort}: full-source predictions are not exclusively hospital endpoint rows")
    full["stay_id"] = full["stay_id"].astype(str)
    if full["stay_id"].isin(("", "nan", "None", "-1")).any() or full["stay_id"].duplicated().any():
        raise ValueError(f"{cohort}: full-source stay IDs are missing or duplicated")
    if not full["y"].isin((0, 1)).all():
        raise ValueError(f"{cohort}: full-source labels are not binary")
    if not np.isfinite(full["p"].to_numpy(float)).all() or not full["p"].between(0, 1).all():
        raise ValueError(f"{cohort}: full-source probabilities are invalid")

    full_rows = full.loc[:, ["stay_id", "subject", "y", "p"]].rename(
        columns={"subject": "full_record_id_proxy", "y": "full_y", "p": "p_full"})
    merged = full_rows.merge(nomg, on="stay_id", how="outer", validate="one_to_one",
                             indicator=True)
    shared = merged.loc[merged["_merge"].eq("both")].copy()
    proxy_conflicts = int(shared["full_record_id_proxy"].astype(str).ne(
        shared["record_id_proxy"].astype(str)).sum())
    label_conflicts = int(shared["full_y"].ne(shared["y"]).sum())
    flow = {
        "cohort": cohort,
        "n_full_rows": int(len(full_rows)),
        "n_nomg_rows": int(len(nomg)),
        "n_exact_stay_id_intersection": int(len(shared)),
        "n_full_only": int(merged["_merge"].eq("left_only").sum()),
        "n_nomg_only": int(merged["_merge"].eq("right_only").sum()),
        "n_proxy_identity_conflicts": proxy_conflicts,
        "n_label_conflicts": label_conflicts,
        "n_full_events": int(full_rows["full_y"].sum()),
        "n_nomg_events": int(nomg["y"].sum()),
        "n_paired_events": int(shared["full_y"].sum()),
        "n_paired_non_events": int(len(shared) - shared["full_y"].sum()),
        "n_paired_record_id_proxy_clusters": int(shared["record_id_proxy"].nunique()),
        "identity_unit": "record_id_proxy; not verified as independent patients",
        "exact_stay_id_intersection_used": True,
    }
    if proxy_conflicts or label_conflicts:
        raise ValueError(f"{cohort}: stay intersection contains identity conflicts: "
                         f"{json.dumps(flow, sort_keys=True)}")
    if flow["n_full_only"] or flow["n_nomg_only"]:
        raise ValueError(f"{cohort}: full/noMg stay rosters differ: {json.dumps(flow, sort_keys=True)}")
    if len(shared) != EXPECTED_STAYS[cohort]:
        raise ValueError(f"{cohort}: exact full/noMg intersection differs from registered denominator")

    paired = pd.DataFrame({
        "stay_id": shared["stay_id"].astype(str),
        "subject": shared["record_id_proxy"].astype(str),
        "y": shared["y"].astype(np.int8),
        # paired_cluster_bootstrap reports B minus A: B=full gives full minus noMg.
        "p_a": shared["p_nomg"].astype(float),
        "p_b": shared["p_full"].astype(float),
    })
    return paired, flow


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    preflight = _preflight()
    if args.preflight_only:
        return 0
    # Full pairing is appended after the exact NPZ schema is observed in preflight.
    if preflight.get("status") != "artifacts_verified_npz_keys_inspected":
        raise ValueError("C-proxy preflight did not pass")
    full_frames, source_manifest, completion, input_hashes = _full_frames()
    arrays, aggregate, _rank_audit = _inspect_nomg()
    cohort_summary = aggregate.get("cohorts", {})
    all_rows: list[dict[str, Any]] = []
    flow_rows: list[dict[str, Any]] = []
    for cohort in COHORTS:
        nomg = _nomg_cohort_frame(arrays, cohort, cohort_summary[cohort])
        paired, flow = _pair_full_no_mg(full_frames[cohort], nomg, cohort)
        seed = pm.derive_seed(SEED, "c_proxy_full_minus_no_mg", cohort)
        metric_rows, protocol = pm.paired_cluster_bootstrap(
            paired, n_boot=N_BOOT, seed=seed)
        for row in metric_rows:
            row.update({
                "cohort": cohort,
                "comparison_direction": "full_minus_no_mg",
                "analysis_label_a": "noMg",
                "analysis_label_b": "full",
                "analysis_id_a": "symmetric_mg_omitted_p_nomg",
                "analysis_id_b": "main20k_real",
                "n_paired_stays": int(len(paired)),
                "n_paired_record_id_proxy_clusters": flow["n_paired_record_id_proxy_clusters"],
                "bootstrap_replicates": N_BOOT,
                "seed": seed,
                "bootstrap_unit": "record_id_proxy; not verified as independent patients",
                "evaluation_weighting": protocol["evaluation_weighting"],
                "clinical_input_availability_verified": False,
                "contrast_type": "pipeline contrast; not a fixed-head input intervention",
            })
            all_rows.append(row)
        flow.update({
            "comparison_direction": "full_minus_no_mg",
            "full_condition": "main20k_real",
            "no_mg_condition": "symmetric_mg_omitted_p_nomg",
            "proxy_and_label_identity_verified": True,
        })
        flow_rows.append(flow)
        print(json.dumps({
            "stage": "c_proxy_pair_complete", "cohort": cohort,
            "n_paired_stays": int(len(paired)),
            "n_record_id_proxy_clusters": flow["n_paired_record_id_proxy_clusters"],
            "n_paired_events": flow["n_paired_events"],
            "n_full_only": flow["n_full_only"], "n_nomg_only": flow["n_nomg_only"],
        }, sort_keys=True), flush=True)

    if len(all_rows) != len(COHORTS) * len(pm.METRICS) or len(flow_rows) != len(COHORTS):
        raise AssertionError("C-proxy paired result grid is incomplete")
    metrics_path = OUT / "paired_c_proxy_metrics.csv"
    flow_path = OUT / "paired_c_proxy_sample_flow.csv"
    pd.DataFrame(all_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(flow_rows).to_csv(flow_path, index=False)
    no_mg_hashes = {
        "patient_predictions_sha256": _sha256(NOMG_PREDICTIONS),
        "nomg_aggregate_sha256": _sha256(NOMG_AGGREGATE),
        "rank_recovery_audit_sha256": _sha256(NOMG_RANK_AUDIT),
    }
    analysis_manifest = {
        "status": "complete",
        "schema": "ehr_fm_c2012_full_vs_no_mg_paired_metrics_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_reuse_run_id": source_manifest.get("run_id"),
        "source_prediction_manifest_path": str(SOURCE_MANIFEST),
        "source_prediction_manifest_sha256": _sha256(SOURCE_MANIFEST),
        "source_reuse_completion_path": str(SOURCE_COMPLETION),
        "source_reuse_completion_sha256": _sha256(SOURCE_COMPLETION),
        "input_hashes": {**input_hashes, **no_mg_hashes,
                         "paired_metrics_helper_sha256": _sha256(ENGINE_CODE / "paired_metrics_v4.py"),
                         "preflight_manifest_sha256": _sha256(OUT / "preflight_manifest.json")},
        "no_mg_aggregate_status": aggregate.get("status"),
        "no_mg_rank_audit_present": bool(_rank_audit),
        "cohorts": list(COHORTS),
        "metrics": list(pm.METRICS),
        "bootstrap_replicates": N_BOOT,
        "base_seed": SEED,
        "estimand": ("Paired metric difference full minus noMg on exact stay_id intersections; "
                     "record_id_proxy clusters are resampled with all associated stays retained."),
        "resampling_unit": "record_id_proxy; not verified as independent patients",
        "contrast_type": "pipeline contrast; not a fixed-head input intervention",
        "clinical_input_availability_verified": False,
        "clinical_promotion_allowed": False,
        "refit_or_reencode_performed_in_this_analysis": False,
        "scope_notes": [
            "Scores are retrospective measurement-time predictions; clinical-time input availability is unverified.",
            "Full and noMg conditions are compared as prediction pipelines, not as a fixed-head input intervention.",
            "record_id_proxy is the resampling unit and is not verified as an independent-patient identifier.",
            "Only aggregate metrics and sample-flow tables are exported; patient-level IDs and arrays remain server-only.",
        ],
        "paired_metrics_path": str(metrics_path),
        "paired_metrics_sha256": _sha256(metrics_path),
        "paired_metric_rows": len(all_rows),
        "paired_flow_path": str(flow_path),
        "paired_flow_sha256": _sha256(flow_path),
        "paired_flow_rows": len(flow_rows),
        "patient_level_exported_to_local": False,
    }
    manifest_path = OUT / "analysis_manifest.json"
    manifest_path.write_text(json.dumps(analysis_manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(json.dumps({
        "stage": "c_proxy_metrics_complete", "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path), "metric_rows": len(all_rows),
        "flow_rows": len(flow_rows),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
