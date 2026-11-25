"""Paired, patient-cluster uncertainty for v4 source-only predictions.

Prediction files contain row-level identifiers and outcomes on the server.
This module writes aggregate summaries only and verifies each requested
analysis/cohort file against its row-level prediction manifest before reading
prediction rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
ENGINE = ROOT / "revision_v4_20261122/engine"
TEMPORAL = ROOT / "revision_v4_20261122/temporal/results/eligibility"
REQUIRED = {
    "db", "analysis_id", "stay_id", "subject", "rows", "split", "y", "p",
    "model_tag", "input_condition", "endpoint", "role", "prediction_scope",
    "input_scope", "clinical_input_availability_verified",
    "eligibility_manifest_sha256", "eligibility_table_sha256", "temporal_evidence",
}
INPUT_SCOPE = "retrospective_measurement_time"
ALLOWED_PREDICTION_SCOPES = {
    "retrospective_landmark_prediction_measurement_time",
}
ORDERED_DIGEST_COLUMNS = (
    "db", "analysis_id", "stay_id", "subject", "rows", "split", "y", "p",
    "endpoint", "role", "prediction_scope", "input_scope",
    "eligibility_manifest_sha256", "eligibility_table_sha256",
)
METRICS = ("auroc", "average_precision", "brier", "logloss",
           "calibration_slope", "calibration_intercept", "offset_intercept")
LOGIT_EPS = 1e-12
SLOPE_PENALTY = 1e-6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ordered_prediction_rows(frame: pd.DataFrame) -> str:
    """Hash ordered prediction identity, labels, values, and analysis scope."""
    missing = set(ORDERED_DIGEST_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"prediction digest missing columns: {sorted(missing)}")
    digest = hashlib.sha256()
    for row in frame.loc[:, ORDERED_DIGEST_COLUMNS].itertuples(index=False, name=None):
        record = [
            str(row[0]), str(row[1]), str(row[2]), str(row[3]),
            int(row[4]), int(row[5]), int(row[6]), format(float(row[7]), ".17g"),
            str(row[8]), str(row[9]), str(row[10]), str(row[11]),
            str(row[12]), str(row[13]),
        ]
        digest.update(json.dumps(record, separators=(",", ":"),
                                ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    text = str(value or "")
    return (len(text) == 64
            and all(char in "0123456789abcdefABCDEF" for char in text))


def current_eligibility_hashes(db: str, eligibility_dir: Path = TEMPORAL) -> dict[str, str]:
    manifest_path = Path(eligibility_dir) / "eligibility_manifest.json"
    table_path = Path(eligibility_dir) / f"{db}.parquet"
    if not manifest_path.exists() or not table_path.exists():
        raise ValueError(f"{db}: current eligibility manifest/table is missing")
    manifest = json.loads(manifest_path.read_text())
    entry = manifest.get("cohorts", {}).get(db)
    source_hashes = entry.get("sourcehashes") if isinstance(entry, dict) else None
    source_npz_hash = (source_hashes.get("sequence_npz")
                       if isinstance(source_hashes, dict) else None)
    if not is_sha256(source_npz_hash):
        raise ValueError(f"{db}: current temporal manifest lacks sourcehashes.sequence_npz")
    return {
        "eligibility_manifest_sha256": sha256_file(manifest_path),
        "eligibility_table_sha256": sha256_file(table_path),
        "source_npz_sha256": str(source_npz_hash).lower(),
        "retrospective_landmark_prediction_allowed": (
            entry.get("retrospective_landmark_prediction_allowed") is True
        ),
    }


def validate_manifest_entry(entry: dict, analysis_id: str, db: str,
                            prediction_path: Path,
                            current_hashes: dict[str, str]) -> None:
    if entry.get("analysis_id") != analysis_id or entry.get("db") != db:
        raise ValueError(f"{analysis_id}/{db}: wrong per-analysis manifest entry")
    if Path(str(entry.get("prediction_file", ""))).resolve() != Path(prediction_path).resolve():
        raise ValueError(f"{analysis_id}/{db}: prediction path differs from manifest")
    if not is_sha256(entry.get("prediction_file_sha256")):
        raise ValueError(f"{analysis_id}/{db}: prediction file hash is missing or invalid")
    if sha256_file(prediction_path) != str(entry["prediction_file_sha256"]).lower():
        raise ValueError(f"{analysis_id}/{db}: prediction file SHA256 is stale")
    if not is_sha256(entry.get("ordered_row_digest_sha256")):
        raise ValueError(f"{analysis_id}/{db}: ordered row digest is missing or invalid")
    if int(entry.get("n_prediction_rows", -1)) <= 0:
        raise ValueError(f"{analysis_id}/{db}: manifest has no attested evaluation rows")
    if int(entry.get("n_predictions", -1)) != int(entry["n_prediction_rows"]):
        raise ValueError(f"{analysis_id}/{db}: prediction counts disagree in manifest")
    if entry.get("role") != "evaluation":
        raise ValueError(f"{analysis_id}/{db}: manifest role is not evaluation")
    if entry.get("input_scope") != INPUT_SCOPE:
        raise ValueError(f"{analysis_id}/{db}: unsupported input scope")
    if entry.get("clinical_input_availability_verified") is not False:
        raise ValueError(f"{analysis_id}/{db}: real-time input availability must not be claimed")
    if entry.get("prediction_scope") not in ALLOWED_PREDICTION_SCOPES:
        raise ValueError(f"{analysis_id}/{db}: unsupported prediction scope")
    if entry.get("mode") != "predict":
        raise ValueError(f"{analysis_id}/{db}: diagnostic associations cannot enter paired prediction metrics")
    if (entry.get("retrospective_landmark_prediction_allowed") is not True
            or current_hashes.get("retrospective_landmark_prediction_allowed") is not True
            or entry.get("retrospective_landmark_prediction_ready") is not True):
        raise ValueError(f"{analysis_id}/{db}: retrospective landmark prediction is not attested")
    for key in ("eligibility_manifest_sha256", "eligibility_table_sha256", "source_npz_sha256"):
        if not is_sha256(entry.get(key)):
            raise ValueError(f"{analysis_id}/{db}: manifest field {key} is missing or invalid")
        if str(entry[key]).lower() != current_hashes[key]:
            raise ValueError(f"{analysis_id}/{db}: stale current eligibility/source hash: {key}")


def validate_predictions(frame: pd.DataFrame, analysis_id: str) -> pd.DataFrame:
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"{analysis_id}: prediction table missing {sorted(missing)}")
    if frame["analysis_id"].astype(str).ne(analysis_id).any():
        raise ValueError(f"{analysis_id}: file contains another analysis ID")
    if frame["role"].astype(str).ne("evaluation").any():
        raise ValueError(f"{analysis_id}: only evaluation rows may be summarized")
    frame = frame.copy()
    frame["subject"] = frame["subject"].astype(str)
    frame["y"] = pd.to_numeric(frame["y"], errors="coerce")
    frame["p"] = pd.to_numeric(frame["p"], errors="coerce")
    if not frame["y"].isin((0, 1)).all() or not np.isfinite(frame["p"]).all():
        raise ValueError(f"{analysis_id}: labels or predictions are invalid")
    if not frame["p"].between(0, 1, inclusive="both").all():
        raise ValueError(f"{analysis_id}: probabilities fall outside [0,1]")
    if frame.duplicated(["db", "stay_id"]).any():
        raise ValueError(f"{analysis_id}: duplicate cohort/stay predictions")
    if frame["subject"].isin(("", "nan", "None", "-1")).any():
        raise ValueError(f"{analysis_id}: invalid subject IDs")
    if not frame["endpoint"].isin(("hospital", "icu")).all():
        raise ValueError(f"{analysis_id}: unsupported endpoint value")
    if frame["input_scope"].astype(str).ne(INPUT_SCOPE).any():
        raise ValueError(f"{analysis_id}: predictions are not scoped to retrospective measurement time")
    if frame["clinical_input_availability_verified"].astype(str).str.lower().ne("false").any():
        raise ValueError(f"{analysis_id}: real-time input availability cannot be claimed")
    if not frame["prediction_scope"].isin(ALLOWED_PREDICTION_SCOPES).all():
        raise ValueError(f"{analysis_id}: unsupported prediction scope")
    if not frame["temporal_evidence"].isin(("exact", "operational_proxy")).all():
        raise ValueError(f"{analysis_id}: unsupported temporal evidence value")
    for column in ("eligibility_manifest_sha256", "eligibility_table_sha256"):
        if not frame[column].map(is_sha256).all():
            raise ValueError(f"{analysis_id}: invalid {column}")
    return frame


def pair_rows(a: pd.DataFrame, b: pd.DataFrame, db: str,
              endpoint: str) -> tuple[pd.DataFrame, dict]:
    a = a.loc[(a["db"].astype(str) == db) & (a["endpoint"].astype(str) == endpoint)].copy()
    b = b.loc[(b["db"].astype(str) == db) & (b["endpoint"].astype(str) == endpoint)].copy()
    keys = ["db", "stay_id"]
    shared = a.merge(b, on=keys, how="inner", suffixes=("_a", "_b"),
                     validate="one_to_one")
    exact = (
        shared["subject_a"].astype(str).eq(shared["subject_b"].astype(str))
        & shared["rows_a"].eq(shared["rows_b"])
        & shared["split_a"].eq(shared["split_b"])
        & shared["y_a"].eq(shared["y_b"])
        & shared["endpoint_a"].eq(shared["endpoint_b"])
    )
    if not exact.all():
        raise ValueError(f"{db}/{endpoint}: paired stay identity, split, or label mismatch")
    paired = pd.DataFrame({
        "stay_id": shared["stay_id"], "subject": shared["subject_a"].astype(str),
        "rows": shared["rows_a"], "split": shared["split_a"],
        "y": shared["y_a"].astype(np.int8), "p_a": shared["p_a"].astype(float),
        "p_b": shared["p_b"].astype(float),
    })
    flow = {
        "db": db, "endpoint": endpoint,
        "n_a": int(len(a)), "n_b": int(len(b)),
        "n_paired": int(len(paired)),
        "n_excluded_a": int(len(a) - len(paired)),
        "n_excluded_b": int(len(b) - len(paired)),
        "n_patients_paired": int(paired["subject"].nunique()),
        "paired_identity_labels_verified": True,
    }
    return paired, flow


def calibration_parameters(y: np.ndarray, p: np.ndarray,
                           weights: np.ndarray) -> tuple[float, float, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = weights > 0
    y, p, weights = y[keep], p[keep], weights[keep]
    if not len(y) or len(np.unique(y)) != 2:
        return np.nan, np.nan, np.nan
    lg = logit(np.clip(p, LOGIT_EPS, 1 - LOGIT_EPS))

    def objective(theta):
        slope, intercept = theta
        eta = np.clip(slope * lg + intercept, -60, 60)
        pred = expit(eta)
        loss = np.sum(weights * (np.logaddexp(0, eta) - y * eta))
        loss += 0.5 * SLOPE_PENALTY * slope * slope
        gradient = np.array([
            np.sum(weights * (pred - y) * lg) + SLOPE_PENALTY * slope,
            np.sum(weights * (pred - y)),
        ])
        return float(loss), gradient

    fit = minimize(objective, x0=np.array([1.0, 0.0]), jac=True, method="L-BFGS-B",
                   options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8})
    if not np.isfinite(fit.x).all() or not fit.success:
        slope = intercept = np.nan
    else:
        slope, intercept = map(float, fit.x)
    try:
        offset = float(brentq(
            lambda value: float(np.sum(weights * (expit(np.clip(lg + value, -60, 60)) - y))),
            -60, 60, xtol=1e-10,
        ))
    except (ValueError, RuntimeError, FloatingPointError):
        offset = np.nan
    return slope, intercept, offset


def weighted_metrics(y: np.ndarray, p: np.ndarray,
                     weights: np.ndarray | None = None) -> dict[str, float]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=float)
    w = np.ones(len(y), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    keep = w > 0
    y, p, w = y[keep], p[keep], w[keep]
    if not len(y) or not np.isfinite(p).all() or (w < 0).any():
        return {name: np.nan for name in METRICS}
    prevalence = float(np.average(y, weights=w))
    pp = np.clip(p, LOGIT_EPS, 1 - LOGIT_EPS)
    brier = float(np.average((y - p) ** 2, weights=w))
    logloss = float(np.average(-(y * np.log(pp) + (1 - y) * np.log1p(-pp)), weights=w))
    if len(np.unique(y)) == 2:
        auc = float(roc_auc_score(y, p, sample_weight=w))
        ap = float(average_precision_score(y, p, sample_weight=w))
    else:
        auc = ap = np.nan
    slope, intercept, offset = calibration_parameters(y, p, w)
    return {
        "auroc": auc, "average_precision": ap, "brier": brier,
        "logloss": logloss, "calibration_slope": slope,
        "calibration_intercept": intercept, "offset_intercept": offset,
        "prevalence": prevalence,
    }


def paired_cluster_bootstrap(paired: pd.DataFrame, n_boot: int = 1000,
                            seed: int = 20261122) -> tuple[list[dict], dict]:
    if n_boot < 1000:
        raise ValueError("patient-cluster bootstrap requires at least 1000 replicates")
    if paired.empty:
        raise ValueError("cannot bootstrap an empty paired cohort")
    y = paired["y"].to_numpy(dtype=np.int8)
    pa = paired["p_a"].to_numpy(dtype=float)
    pb = paired["p_b"].to_numpy(dtype=float)
    groups, group_index = np.unique(paired["subject"].astype(str), return_inverse=True)
    metrics_a = weighted_metrics(y, pa)
    metrics_b = weighted_metrics(y, pb)
    boot_a = {name: np.full(n_boot, np.nan) for name in METRICS}
    boot_b = {name: np.full(n_boot, np.nan) for name in METRICS}
    boot_delta = {name: np.full(n_boot, np.nan) for name in METRICS}
    rng = np.random.default_rng(seed)
    for replicate in range(n_boot):
        sampled = rng.integers(0, len(groups), size=len(groups))
        counts = np.bincount(sampled, minlength=len(groups)).astype(float)
        row_weights = counts[group_index]
        draw_a = weighted_metrics(y, pa, row_weights)
        draw_b = weighted_metrics(y, pb, row_weights)
        for name in METRICS:
            boot_a[name][replicate] = draw_a[name]
            boot_b[name][replicate] = draw_b[name]
            if np.isfinite(draw_a[name]) and np.isfinite(draw_b[name]):
                boot_delta[name][replicate] = draw_b[name] - draw_a[name]
    rows = []
    for name in METRICS:
        def summarize(values):
            valid = values[np.isfinite(values)]
            if not len(valid):
                return np.nan, np.nan, 0
            lo, hi = np.quantile(valid, [0.025, 0.975])
            return float(lo), float(hi), int(len(valid))
        alo, ahi, nav = summarize(boot_a[name])
        blo, bhi, nbv = summarize(boot_b[name])
        dlo, dhi, ndv = summarize(boot_delta[name])
        rows.append({
            "metric": name,
            "estimate_a": metrics_a[name], "ci_a_low": alo, "ci_a_high": ahi,
            "valid_boot_a": nav, "invalid_boot_a": int(n_boot - nav),
            "estimate_b": metrics_b[name], "ci_b_low": blo, "ci_b_high": bhi,
            "valid_boot_b": nbv, "invalid_boot_b": int(n_boot - nbv),
            "difference_b_minus_a": (metrics_b[name] - metrics_a[name]
                                     if np.isfinite(metrics_a[name]) and np.isfinite(metrics_b[name])
                                     else np.nan),
            "difference_ci_low": dlo, "difference_ci_high": dhi,
            "valid_paired_boot": ndv, "invalid_paired_boot": int(n_boot - ndv),
        })
    protocol = {
        "bootstrap_unit": "subject, resample patients with replacement; retain all stays",
        "evaluation_weighting": "stay-weighted within each patient-cluster replicate",
        "bootstrap_replicates": int(n_boot), "seed": int(seed),
        "inference_scope": "conditional on frozen encoder and fitted source probes",
        "calibration_slope": "weighted logistic regression of outcome on logit(p), slope penalty 1e-6, intercept unpenalized",
        "offset_intercept": "weighted intercept-only recalibration with logit coefficient fixed at one",
    }
    return rows, protocol


def derive_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256("|".join((str(base_seed), *parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def run(args: argparse.Namespace) -> int:
    pred_dir = Path(args.prediction_dir)
    manifest_arg = getattr(args, "manifest_path", None)
    manifest_path = Path(manifest_arg) if manifest_arg is not None else pred_dir / "prediction_manifest.json"
    if manifest_arg is None and not manifest_path.exists():
        manifest_path = pred_dir.parent / "prediction_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("prediction_manifest.json is required for provenance filtering")
    manifest = json.loads(manifest_path.read_text())
    analysis_ids = args.analysis_ids
    if len(set(analysis_ids)) != len(analysis_ids):
        raise ValueError("analysis IDs must be unique")
    prediction_entries = manifest.get("predictions", [])
    entries_by_analysis: dict[str, dict[str, dict]] = {}
    files_by_analysis: dict[str, Path] = {}
    current_hash_cache: dict[str, dict[str, str]] = {}
    all_attested_dbs: set[str] = set()
    for analysis_id in analysis_ids:
        path = pred_dir / f"{analysis_id}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"prediction file is missing: {path}")
        files_by_analysis[analysis_id] = path
        by_db: dict[str, dict] = {}
        for entry in prediction_entries:
            if entry.get("analysis_id") != analysis_id or int(entry.get("n_predictions", 0)) <= 0:
                continue
            db = str(entry.get("db", ""))
            if not db:
                raise ValueError(f"{analysis_id}: manifest entry has no cohort")
            if db in by_db:
                raise ValueError(f"{analysis_id}/{db}: duplicate per-cohort prediction manifest entries")
            if db not in current_hash_cache:
                current_hash_cache[db] = current_eligibility_hashes(db)
            validate_manifest_entry(entry, analysis_id, db, path, current_hash_cache[db])
            by_db[db] = entry
        if not by_db:
            raise ValueError(f"{analysis_id}: no positive per-cohort prediction attestation")
        entries_by_analysis[analysis_id] = by_db
        all_attested_dbs.update(by_db)
    if not all_attested_dbs:
        raise ValueError("no attested retrospective prediction cohorts were found")
    for db in sorted(all_attested_dbs):
        missing = [analysis_id for analysis_id in analysis_ids
                   if db not in entries_by_analysis[analysis_id]]
        if missing:
            raise ValueError(
                f"{db}: incomplete per-analysis cohort set; missing {missing}. "
                "Regenerate all requested prediction files in one run.")
        scope_values = {entries_by_analysis[analysis_id][db]["prediction_scope"]
                        for analysis_id in analysis_ids}
        evidence_values = {entries_by_analysis[analysis_id][db]["temporal_evidence"]
                           for analysis_id in analysis_ids}
        if len(scope_values) != 1 or len(evidence_values) != 1:
            raise ValueError(f"{db}: requested analyses have different retrospective scopes")

    tables = {}
    for analysis_id in analysis_ids:
        path = files_by_analysis[analysis_id]
        frame = pd.read_parquet(path, filters=[("db", "in", sorted(all_attested_dbs))])
        frame = validate_predictions(frame, analysis_id)
        for db in sorted(all_attested_dbs):
            entry = entries_by_analysis[analysis_id][db]
            cohort = frame.loc[frame["db"].astype(str) == db]
            if len(cohort) != int(entry["n_prediction_rows"]):
                raise ValueError(f"{analysis_id}/{db}: prediction row count differs from manifest")
            if sha256_ordered_prediction_rows(cohort) != str(entry["ordered_row_digest_sha256"]).lower():
                raise ValueError(f"{analysis_id}/{db}: ordered prediction digest differs from manifest")
            if cohort["prediction_scope"].astype(str).ne(entry["prediction_scope"]).any():
                raise ValueError(f"{analysis_id}/{db}: prediction scope differs from manifest")
            if cohort["temporal_evidence"].astype(str).ne(entry["temporal_evidence"]).any():
                raise ValueError(f"{analysis_id}/{db}: temporal evidence differs from manifest")
            for column in ("eligibility_manifest_sha256", "eligibility_table_sha256"):
                if cohort[column].astype(str).str.lower().ne(current_hash_cache[db][column]).any():
                    raise ValueError(f"{analysis_id}/{db}: prediction rows have stale {column}")
        tables[analysis_id] = frame
    conditions = sorted(tables)
    if len(conditions) < 2:
        raise ValueError("at least two conditions are required for paired metrics")
    metric_rows, flow_rows = [], []
    for db in sorted(all_attested_dbs):
        endpoints = sorted(set().union(*(set(table.loc[table["db"].astype(str) == db,
                                                "endpoint"].astype(str))
                                         for table in tables.values())))
        for endpoint in endpoints:
            for analysis_a, analysis_b in combinations(conditions, 2):
                paired, flow = pair_rows(tables[analysis_a], tables[analysis_b], db, endpoint)
                flow.update(analysis_id_a=analysis_a, analysis_id_b=analysis_b,
                            n_source_rows_a=int((tables[analysis_a]["db"].astype(str) == db).sum()),
                            n_source_rows_b=int((tables[analysis_b]["db"].astype(str) == db).sum()))
                flow_rows.append(flow)
                if paired.empty:
                    continue
                seed = derive_seed(args.seed, db, endpoint, analysis_a, analysis_b)
                estimates, protocol = paired_cluster_bootstrap(paired, args.bootstrap, seed)
                for row in estimates:
                    row.update(db=db, endpoint=endpoint,
                               analysis_id_a=analysis_a, analysis_id_b=analysis_b,
                               n_paired_stays=int(len(paired)),
                               n_paired_patients=int(paired["subject"].nunique()),
                               n_bootstrap=int(args.bootstrap), seed=seed,
                               inference_scope=protocol["inference_scope"])
                    metric_rows.append(row)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(out_dir / "paired_core_metrics.csv", index=False)
    pd.DataFrame(flow_rows).to_csv(out_dir / "paired_core_pair_flow.csv", index=False)
    protocol = {
        "analysis_ids": analysis_ids,
        "retrospective_attested_cohorts": sorted(all_attested_dbs),
        "prediction_scope_by_cohort": {
            db: entries_by_analysis[analysis_ids[0]][db]["prediction_scope"]
            for db in sorted(all_attested_dbs)
        },
        "temporal_evidence_by_cohort": {
            db: entries_by_analysis[analysis_ids[0]][db]["temporal_evidence"]
            for db in sorted(all_attested_dbs)
        },
        "input_scope": INPUT_SCOPE,
        "clinical_input_availability_verified": False,
        "metrics": list(METRICS), "bootstrap_replicates": int(args.bootstrap),
        "base_seed": int(args.seed),
        "inference_scope": "conditional on frozen encoder and fitted source probes",
        "paired_identity": "exact stay_id, subject, row, split, endpoint and outcome equality",
        "interval": "percentile 95% patient-cluster bootstrap interval",
    }
    (out_dir / "paired_core_metrics_protocol.json").write_text(json.dumps(protocol, indent=2))
    print(f"PAIRED_CORE_METRICS_COMPLETE retrospective_cohorts={len(all_attested_dbs)} pairs={len(flow_rows)}",
          flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, default=ENGINE / "results/predictions")
    parser.add_argument("--manifest-path", type=Path, default=None,
                        help="explicit run-scoped prediction manifest")
    parser.add_argument("--output-dir", type=Path, default=ENGINE / "results/tables")
    parser.add_argument("--analysis-ids", nargs="+", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261122)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
