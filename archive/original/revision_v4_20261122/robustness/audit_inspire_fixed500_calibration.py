"""No-refit identity and denominator audit for the completed INSPIRE calibration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
GATE = REV / "temporal/results/eligibility/inspire_exploratory_locked_v4_1"
ROLE_LOCK = GATE / "assessment_lock.json"
OUT_DEFAULT = REV / "robustness/results/inspire_fixed500_calibration_20261123_r1"
RUNNER = REV / "robustness/inspire_fixed500_calibration.py"
RUNNER_SHA = "b1ab8d375ccd74b34ebab28f3a74f070854404ea14bb4c4e4c173c67343b7494"
EXPECTED_GATE_SHA = "97c7aeacacb2af9cfce63086d6dc5b9ce9838292a4197e43d3ec98d7f533c019"
EXPECTED_LOCK_SHA = "3ac72b0d60dc47c81d5fb894f827f573735f32e220f430852281947c10152ffd"
COHORTS = ("inspire_postop_v4_1", "inspire_broad_surgery_associated_v4_1")
ARMS = (
    "original_ordinary_C1", "ordinary_tuned", "density_augmented",
    "conventional_lr_pattern_only", "conventional_hgb_pattern_only",
    "conventional_lr_value_plus_pattern", "conventional_hgb_value_plus_pattern",
)
EXPECTED = {
    "inspire_postop_v4_1": {"adaptation_patients": 500, "adaptation_events": 21,
                             "assessment_patients": 3358, "assessment_events": 106,
                             "assessment_nonevents": 3252},
    "inspire_broad_surgery_associated_v4_1": {"adaptation_patients": 500,
                             "adaptation_events": 24, "assessment_patients": 5213,
                             "assessment_events": 186, "assessment_nonevents": 5027},
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest_rows(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(["subject", "stay_id"], kind="stable")
    h = hashlib.sha256()
    for row in ordered[["db", "subject", "stay_id", "y", "split_role"]].itertuples(index=False, name=None):
        record = [str(row[0]), str(row[1]), str(row[2]), int(row[3]), str(row[4])]
        h.update(json.dumps(record, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def atomic_json(path: Path, doc: dict[str, Any]) -> None:
    temp = path.with_name("." + path.name + ".tmp")
    temp.write_text(json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def run(out_dir: Path = OUT_DEFAULT) -> dict[str, Any]:
    if sha(RUNNER) != RUNNER_SHA:
        raise ValueError("calibration runner hash differs from the code used for the completed job")
    gate_path = GATE / "eligibility_manifest.json"
    if sha(gate_path) != EXPECTED_GATE_SHA or sha(ROLE_LOCK) != EXPECTED_LOCK_SHA:
        raise ValueError("fixed INSPIRE gate/assessment lock hash differs")
    manifest_path = out_dir / "inspire_fixed500_calibration_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("status") != "complete" or manifest.get("roles_redrawn") is not False
            or manifest.get("fixed_adaptation_budget") != 500
            or manifest.get("source_model_fit_calls_in_target_scoring_job") != 0
            or manifest.get("source_probe_refit_calls_for_density_heads") != 0
            or manifest.get("target_fit_calls") != 14):
        raise ValueError("completed calibration manifest violates frozen-role/no-source-fit contract")

    predictions = pd.read_parquet(
        out_dir / "predictions/inspire_fixed500_seven_arm_predictions.parquet")
    fit = pd.read_csv(out_dir / "inspire_fixed500_platt_fit_audit.csv")
    metrics = pd.read_csv(out_dir / "inspire_fixed500_metrics.csv")
    paired = pd.read_csv(out_dir / "inspire_fixed500_paired_loss_ci.csv")
    expected_files = manifest.get("output_files", {})
    for rel, record in expected_files.items():
        path = out_dir / rel
        if not path.is_file() or sha(path) != record.get("sha256"):
            raise ValueError(f"output changed after calibration: {rel}")
    if len(fit) != 14 or len(metrics) != 42 or len(paired) != 52:
        raise ValueError("calibration aggregate row counts differ from the completed-run log")
    if (set(fit["analysis_id"].astype(str)) != set(ARMS)
            or any(int(row["adaptation_n"]) != 500 for _, row in fit.iterrows())
            or fit["fallback_used"].astype(str).str.lower().ne("false").any()
            or fit["platt_status"].astype(str).ne("fitted_on_fixed_500_adaptation_only").any()):
        raise ValueError("the 14 fixed-role Platt fits contain a mismatch or fallback")

    lock = json.loads(ROLE_LOCK.read_text(encoding="utf-8"))
    cohort_audits = {}
    for db in COHORTS:
        role_path = GATE / f"{db}.parquet"
        role = pd.read_parquet(role_path)
        role = role[["db", "stay_id", "subject", "y_mort", "split_role"]].copy()
        for col in ("db", "stay_id", "subject", "split_role"):
            role[col] = role[col].astype(str).str.strip()
        role["y"] = pd.to_numeric(role.pop("y_mort"), errors="raise").astype(int)
        target = EXPECTED[db]
        actual_counts = {
            "adaptation_patients": int(role.split_role.eq("adaptation").sum()),
            "adaptation_events": int(role.loc[role.split_role.eq("adaptation"), "y"].sum()),
            "assessment_patients": int(role.split_role.eq("assessment").sum()),
            "assessment_events": int(role.loc[role.split_role.eq("assessment"), "y"].sum()),
            "assessment_nonevents": int(role.loc[role.split_role.eq("assessment"), "y"].eq(0).sum()),
        }
        if actual_counts != target or lock.get("roles", {}).get(db) != target:
            raise ValueError(f"{db}: actual assessment/adaptation counts differ from lock")
        expected_identity = role[["db", "stay_id", "subject", "y", "split_role"]].copy()
        role_digests = {}
        for arm in ARMS:
            frame = predictions.loc[(predictions.db.astype(str) == db)
                                    & (predictions.analysis_id.astype(str) == arm)].copy()
            frame["stay_id"] = frame["stay_id"].astype(str)
            frame["subject"] = frame["subject"].astype(str)
            frame["split_role"] = frame["split_role"].astype(str)
            if len(frame) != len(role) or frame.stay_id.duplicated().any():
                raise ValueError(f"{db}/{arm}: wrong prediction row count or duplicate stays")
            compare = expected_identity.merge(
                frame[["db", "stay_id", "subject", "y", "split_role"]],
                on=["db", "stay_id"], how="outer", validate="one_to_one",
                suffixes=("_lock", "_prediction"), indicator=True)
            if compare["_merge"].ne("both").any():
                raise ValueError(f"{db}/{arm}: prediction stay IDs differ from fixed role map")
            for field in ("subject", "y", "split_role"):
                left, right = compare[f"{field}_lock"], compare[f"{field}_prediction"]
                if not left.astype(str).equals(right.astype(str)):
                    raise ValueError(f"{db}/{arm}: prediction {field} differs from fixed role map")
            if frame.subject.duplicated().any():
                raise ValueError(f"{db}/{arm}: one episode per patient was not preserved")
            role_digests[arm] = {
                "all_role_rows_sha256": digest_rows(frame),
                "adaptation_rows_sha256": digest_rows(frame.loc[frame.split_role.eq("adaptation")]),
                "assessment_rows_sha256": digest_rows(frame.loc[frame.split_role.eq("assessment")]),
                "n_adaptation_patients": int(frame.split_role.eq("adaptation").sum()),
                "adaptation_events": int(frame.loc[frame.split_role.eq("adaptation"), "y"].sum()),
                "n_assessment_patients": int(frame.split_role.eq("assessment").sum()),
                "assessment_events": int(frame.loc[frame.split_role.eq("assessment"), "y"].sum()),
            }
        if len({value["adaptation_rows_sha256"] for value in role_digests.values()}) != 1:
            raise ValueError(f"{db}: the seven arms do not share the same 500 adaptation identities")
        if len({value["assessment_rows_sha256"] for value in role_digests.values()}) != 1:
            raise ValueError(f"{db}: the seven arms do not share the same assessment identities")
        fit_db = fit.loc[fit.db.astype(str).eq(db)]
        if (len(fit_db) != 7 or set(fit_db.analysis_id.astype(str)) != set(ARMS)
                or set(fit_db.adaptation_n.astype(int)) != {500}
                or set(fit_db.adaptation_events.astype(int)) != {target["adaptation_events"]}
                or fit_db.fallback_used.astype(str).str.lower().ne("false").any()):
            raise ValueError(f"{db}: fitted calibrators do not all bind to the locked adaptation pool")
        paired_db = paired.loc[paired.db.astype(str).eq(db)]
        if (paired_db.valid_bootstrap_draws.astype(int).ne(1000).any()
                or paired_db.n_assessment_patients.astype(int).ne(target["assessment_patients"]).any()
                or paired_db.assessment_events.astype(int).ne(target["assessment_events"]).any()
                or paired_db.assessment_nonevents.astype(int).ne(target["assessment_nonevents"]).any()):
            raise ValueError(f"{db}: paired CI draw count or assessment denominator differs")
        cohort_audits[db] = {
            "locked_role_table_sha256": sha(role_path),
            "counts": actual_counts,
            "adaptation_role_identity_sha256": next(iter(role_digests.values()))["adaptation_rows_sha256"],
            "assessment_role_identity_sha256": next(iter(role_digests.values()))["assessment_rows_sha256"],
            "all_seven_arms_match_locked_roles": True,
            "all_seven_arms_share_same_adaptation_and_assessment_identity": True,
            "fit_audit_rows": int(len(fit_db)),
            "platt_failures": int(fit_db.fallback_used.astype(str).str.lower().eq("true").sum()),
            "valid_paired_bootstrap_draws": 1000,
            "main_brier_contrasts": int((paired_db.primary_contrast.astype(str).str.lower().eq("true")
                                          & paired_db.metric.eq("brier")).sum()),
        }
    primary_brier = paired.loc[paired.primary_contrast.astype(str).str.lower().eq("true")
                               & paired.metric.eq("brier")]
    if len(primary_brier) != 4:
        raise ValueError("primary Fig6 Brier subset must contain exactly four contrasts")

    audit = {
        "schema": "ehr_fm_inspire_fixed500_postrun_role_identity_audit_v1",
        "status": "verified",
        "calibration_manifest_sha256_before_audit": sha(manifest_path),
        "calibration_runner_path": str(RUNNER),
        "calibration_runner_sha256": sha(RUNNER),
        "calibration_runner_sha256_matches_launched_code": sha(RUNNER) == RUNNER_SHA,
        "gate_manifest_sha256": sha(gate_path),
        "assessment_lock_sha256": sha(ROLE_LOCK),
        "patient_ids_emitted": False,
        "source_model_fit_calls_in_target_scoring_job": 0,
        "density_source_probe_refit_calls": 0,
        "roles_redrawn": False,
        "cohorts": cohort_audits,
        "fit_audit": {
            "rows": int(len(fit)),
            "all_14_platt_fits_succeeded_without_fallback": True,
            "analysis_ids": sorted(fit.analysis_id.astype(str).unique().tolist()),
            "platt_status_values": sorted(fit.platt_status.astype(str).unique().tolist()),
        },
        "paired_bootstrap": {
            "rows": int(len(paired)), "primary_brier_rows": int(len(primary_brier)),
            "draws_requested_and_valid": 1000,
        },
    }
    audit_path = out_dir / "inspire_fixed500_role_identity_audit.json"
    if audit_path.exists():
        raise FileExistsError(f"preserving pre-existing role audit: {audit_path}")
    atomic_json(audit_path, audit)
    manifest["post_run_role_identity_audit_path"] = str(audit_path)
    manifest["post_run_role_identity_audit_sha256"] = sha(audit_path)
    manifest["executed_runner_code_path"] = str(RUNNER)
    manifest["executed_runner_code_sha256"] = sha(RUNNER)
    manifest["actual_fixed_role_identity_verified_for_all_arms"] = True
    manifest["output_files"][audit_path.name] = {"sha256": sha(audit_path), "bytes": audit_path.stat().st_size}
    atomic_json(manifest_path, manifest)
    audit["calibration_manifest_sha256_after_audit"] = sha(manifest_path)
    # Keep the sidecar hash bound in the run manifest; the post-audit hash is printed for the handoff.
    print(json.dumps({"status": "verified", "audit_path": str(audit_path),
                      "audit_sha256": sha(audit_path),
                      "updated_manifest_sha256": sha(manifest_path),
                      "platt_fit_rows": int(len(fit)), "platt_fallbacks": 0,
                      "primary_brier_rows": int(len(primary_brier)),
                      "role_hashes": {db: {
                          "adaptation": row["adaptation_role_identity_sha256"],
                          "assessment": row["assessment_role_identity_sha256"],
                      } for db, row in cohort_audits.items()}}, sort_keys=True), flush=True)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    run(args.out_dir)


if __name__ == "__main__":
    main()
