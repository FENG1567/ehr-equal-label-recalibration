"""Strict artifact-backed paired FM versus conventional baseline metrics.

The comparison is conditional on the current evaluation prediction artifacts
and their shared, patient-cluster bootstrap weights. It emits aggregate rows
only; identifiers remain in the server-side prediction files.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PRIMARY_SCOPE = "retrospective_landmark_prediction_measurement_time"
INPUT_SCOPE = "retrospective_measurement_time"
ROLE = "evaluation"
PAIR_KEYS = ["db", "stay_id"]
IDENTITY_FIELDS = [
    "db", "stay_id", "subject", "rows", "split", "y", "endpoint",
    "role", "prediction_scope", "input_scope", "input_condition",
]
DIGEST_FIELDS = [
    "db", "analysis_id", "stay_id", "subject", "rows", "split", "y", "p",
    "endpoint", "role", "prediction_scope", "input_scope",
    "eligibility_manifest_sha256", "eligibility_table_sha256",
]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(document: dict[str, Any]) -> str:
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_complete_fm_prediction_manifest(document: dict[str, Any]) -> None:
    """Validate either legacy complete manifests or current engine run manifests.

    The current engine format carries completion evidence in its hashed run
    protocol and per-cohort readiness records instead of a top-level status.
    It must still describe a hashed prediction run with ready cohorts and
    explicit retrospective measurement-time scope.
    """
    if not isinstance(document, dict):
        raise ValueError("FM prediction manifest must be a JSON object")
    if document.get("status") == "complete":
        if document.get("mode") not in (None, "predict"):
            raise ValueError("complete FM manifest is not a prediction run")
        return

    protocol = document.get("run_protocol")
    protocol_sha = str(document.get("run_protocol_sha256", "")).lower()
    if (document.get("mode") != "predict" or not isinstance(protocol, dict)
            or protocol.get("mode") != "predict"):
        raise ValueError("FM prediction manifest is not a completed prediction run")
    run_id = str(document.get("run_id", ""))
    if not run_id or protocol.get("run_id") != run_id:
        raise ValueError("FM prediction manifest run ID differs from its protocol")
    if len(protocol_sha) != 64 or any(ch not in "0123456789abcdef" for ch in protocol_sha):
        raise ValueError("FM prediction manifest has no valid run protocol SHA256")
    protocol_payload = dict(protocol)
    embedded_sha = str(protocol_payload.pop("run_protocol_sha256", "")).lower()
    if (embedded_sha != protocol_sha
            or _canonical_json_sha256(protocol_payload) != protocol_sha):
        raise ValueError("FM prediction manifest run protocol hash is invalid")
    if (document.get("input_scope") != INPUT_SCOPE
            or document.get("clinical_input_availability_verified") is not False
            or protocol.get("input_scope") != INPUT_SCOPE
            or protocol.get("clinical_input_availability_verified") is not False):
        raise ValueError("FM run protocol lacks the registered retrospective measurement-time scope")

    cohorts = protocol.get("cohorts")
    readiness = document.get("readiness")
    if (not isinstance(cohorts, list) or not cohorts
            or not isinstance(readiness, list) or not readiness):
        raise ValueError("FM prediction manifest lacks per-cohort readiness evidence")
    readiness_by_db = {str(row.get("db")): row for row in readiness if isinstance(row, dict)}
    if (len(readiness_by_db) != len(readiness)
            or set(readiness_by_db) != {str(db) for db in cohorts}):
        raise ValueError("FM readiness cohorts differ from the hashed run protocol")
    for db in cohorts:
        row = readiness_by_db[str(db)]
        if (row.get("status") != "ready"
                or row.get("temporal_evidence") not in {"exact", "operational_proxy"}
                or row.get("prediction_scope") != PRIMARY_SCOPE
                or row.get("input_scope") != INPUT_SCOPE
                or row.get("clinical_input_availability_verified") is not False
                or row.get("retrospective_landmark_prediction_ready") is not True
                or document.get("retrospective_landmark_prediction_by_cohort", {}).get(str(db)) is not True
                or document.get("real_time_available_landmark_prediction_by_cohort", {}).get(str(db)) is not False):
            raise ValueError(f"FM prediction manifest does not attest the registered scope for {db}")

    predictions = document.get("predictions")
    models = protocol.get("models")
    if (not isinstance(predictions, list) or not predictions
            or not isinstance(models, list) or not models):
        raise ValueError("FM prediction manifest has no prediction records or model roster")
    seen: set[tuple[str, str]] = set()
    for row in predictions:
        if not isinstance(row, dict):
            raise ValueError("FM prediction manifest contains a malformed prediction record")
        db, analysis_id = str(row.get("db", "")), str(row.get("analysis_id", ""))
        key = (analysis_id, db)
        if (not analysis_id or db not in {str(value) for value in cohorts}
                or key in seen or str(row.get("model_tag", "")) not in {str(value) for value in models}
                or row.get("mode") != "predict" or row.get("role") != ROLE
                or row.get("prediction_scope") != PRIMARY_SCOPE
                or row.get("input_scope") != INPUT_SCOPE
                or row.get("clinical_input_availability_verified") is not False
                or row.get("retrospective_landmark_prediction_allowed") is not True
                or int(row.get("n_prediction_rows", 0) or 0) <= 0):
            raise ValueError("FM prediction manifest contains an invalid or duplicate prediction record")
        seen.add(key)


def ordered_row_digest(frame: pd.DataFrame) -> str:
    """Match the engine digest: row order is preserved and scalars normalized."""
    missing = set(DIGEST_FIELDS) - set(frame.columns)
    if missing:
        raise ValueError(f"prediction rows missing ordered-digest fields: {sorted(missing)}")
    digest = hashlib.sha256()
    for row in frame[DIGEST_FIELDS].itertuples(index=False, name=None):
        normalized = [
            str(row[0]), str(row[1]), str(row[2]), str(row[3]), int(row[4]),
            int(row[5]), int(row[6]), format(float(row[7]), ".17g"),
            str(row[8]), str(row[9]), str(row[10]), str(row[11]),
            str(row[12]), str(row[13]),
        ]
        digest.update(json.dumps(normalized, separators=(",", ":"),
                                ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _manifest_entry(document: dict[str, Any], analysis_id: str, db: str,
                    label: str) -> dict[str, Any]:
    matches = [row for row in document.get("predictions", [])
               if str(row.get("analysis_id")) == analysis_id
               and str(row.get("db")) == db]
    if len(matches) != 1:
        raise ValueError(
            f"{label}: expected one manifest entry for {analysis_id}/{db}; found {len(matches)}"
        )
    return matches[0]


def _verify_artifact_record(frame: pd.DataFrame, record: dict[str, Any], *,
                            analysis_id: str, db: str, expected_path: Path | None,
                            label: str) -> None:
    required_meta = (
        "prediction_file", "prediction_file_sha256", "ordered_row_digest_sha256",
        "n_prediction_rows", "role", "prediction_scope", "input_scope",
        "clinical_input_availability_verified", "eligibility_manifest_sha256",
        "eligibility_table_sha256", "source_npz_sha256",
        "retrospective_landmark_prediction_allowed", "input_condition",
        "input_mask_sha256",
    )
    missing_meta = [key for key in required_meta if record.get(key) in (None, "")]
    if missing_meta:
        raise ValueError(f"{label} manifest entry missing required provenance: {missing_meta}")
    actual_path = Path(record["prediction_file"]).resolve()
    if expected_path is not None and actual_path != expected_path.resolve():
        raise ValueError(f"{label}: manifest prediction path is not the current artifact path")
    if not actual_path.is_file():
        raise FileNotFoundError(f"{label}: current prediction artifact is absent: {actual_path}")
    if sha256_file(actual_path) != str(record["prediction_file_sha256"]):
        raise ValueError(f"{label}: prediction file hash differs from current manifest")
    if str(record["role"]) != ROLE:
        raise ValueError(f"{label}: only evaluation predictions may be compared")
    if str(record["prediction_scope"]) != PRIMARY_SCOPE:
        raise ValueError(f"{label}: prediction scope is not the approved retrospective landmark scope")
    if str(record["input_scope"]) != INPUT_SCOPE:
        raise ValueError(f"{label}: input scope is not retrospective measurement time")
    if not str(record["input_condition"]).strip():
        raise ValueError(f"{label}: input condition is missing")
    mask_hash = str(record["input_mask_sha256"])
    if len(mask_hash) != 64 or any(char not in "0123456789abcdefABCDEF" for char in mask_hash):
        raise ValueError(f"{label}: input mask SHA256 is missing or malformed")
    mask_spec = record.get("input_mask_spec")
    if str(record["input_condition"]) == "no_h0_therapy":
        if not isinstance(mask_spec, dict):
            raise ValueError(f"{label}: no_h0_therapy mask definition is missing")
        if _canonical_json_sha256(mask_spec) != mask_hash.lower():
            raise ValueError(f"{label}: no_h0_therapy mask hash differs from its definition")
    elif mask_spec is not None:
        raise ValueError(f"{label}: unexpected input mask definition for {record['input_condition']}")
    if record["clinical_input_availability_verified"] is not False:
        raise ValueError(f"{label}: clinical input availability must remain explicitly unverified")
    if record["retrospective_landmark_prediction_allowed"] is not True:
        raise ValueError(f"{label}: primary retrospective landmark permission is not attested")
    if len(frame) != int(record["n_prediction_rows"]):
        raise ValueError(f"{label}: row count differs from current manifest")
    if frame.empty:
        raise ValueError(f"{label}: no prediction rows for {analysis_id}/{db}")
    if frame["analysis_id"].astype(str).ne(analysis_id).any():
        raise ValueError(f"{label}: prediction table contains a stale or different analysis ID")
    if frame["db"].astype(str).ne(db).any():
        raise ValueError(f"{label}: prediction table contains another cohort")
    if frame["role"].astype(str).ne(ROLE).any():
        raise ValueError(f"{label}: prediction table contains non-evaluation rows")
    if frame["prediction_scope"].astype(str).ne(PRIMARY_SCOPE).any():
        raise ValueError(f"{label}: prediction rows contain a non-primary scope")
    if frame["input_scope"].astype(str).ne(INPUT_SCOPE).any():
        raise ValueError(f"{label}: prediction rows contain an unsupported input scope")
    if frame["input_condition"].astype(str).ne(str(record["input_condition"])).any():
        raise ValueError(f"{label}: row-level input condition differs from its manifest")
    if "input_mask_sha256" in frame.columns:
        row_mask_hashes = frame["input_mask_sha256"].astype(str).unique()
        if len(row_mask_hashes) != 1 or row_mask_hashes[0] != str(record["input_mask_sha256"]):
            raise ValueError(f"{label}: row-level input mask hash differs from its manifest")
    if frame["clinical_input_availability_verified"].isna().any() or frame[
        "clinical_input_availability_verified"
    ].astype(bool).any():
        raise ValueError(f"{label}: prediction rows claim verified real-time input availability")
    for key in ("eligibility_manifest_sha256", "eligibility_table_sha256"):
        values = frame[key].astype(str).unique()
        if len(values) != 1 or str(values[0]) != str(record[key]):
            raise ValueError(f"{label}: row-level {key} differs from its manifest")
    if ordered_row_digest(frame) != str(record["ordered_row_digest_sha256"]):
        raise ValueError(f"{label}: ordered prediction-row digest differs from current manifest")


def _validate_pair(baseline: pd.DataFrame, fm: pd.DataFrame,
                   analysis_id: str, db: str) -> pd.DataFrame:
    required = set(IDENTITY_FIELDS + ["analysis_id", "p", "input_scope"])
    for label, frame in (("baseline", baseline), ("FM", fm)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} {analysis_id}/{db} missing pair columns: {sorted(missing)}")
        if frame.duplicated(PAIR_KEYS).any():
            raise ValueError(f"{label} {analysis_id}/{db} contains duplicate source keys")
        if not frame["y"].isin([0, 1]).all():
            raise ValueError(f"{label} {analysis_id}/{db} has nonbinary labels")
        p = pd.to_numeric(frame["p"], errors="coerce")
        if p.isna().any() or not np.isfinite(p.to_numpy()).all() or not p.between(0, 1).all():
            raise ValueError(f"{label} {analysis_id}/{db} has invalid probabilities")
        if frame["subject"].astype(str).isin(("", "nan", "None", "-1")).any():
            raise ValueError(f"{label} {analysis_id}/{db} has invalid patient identifiers")
    joined = baseline.merge(fm, on=PAIR_KEYS, how="outer", suffixes=("_baseline", "_fm"),
                            validate="one_to_one", indicator=True)
    if len(baseline) != len(fm) or joined["_merge"].ne("both").any():
        n_missing_fm = int(joined["_merge"].eq("left_only").sum())
        n_missing_baseline = int(joined["_merge"].eq("right_only").sum())
        raise ValueError(
            f"{analysis_id}/{db}: evaluation population mismatch "
            f"(baseline_only={n_missing_fm}, fm_only={n_missing_baseline})"
        )
    for field in IDENTITY_FIELDS[2:]:
        left, right = joined[f"{field}_baseline"], joined[f"{field}_fm"]
        if not left.astype(str).equals(right.astype(str)):
            raise ValueError(f"{analysis_id}/{db}: paired source {field} mismatch")
    return pd.DataFrame({
        "stay_id": joined["stay_id"].astype(str),
        "subject": joined["subject_baseline"].astype(str),
        "rows": pd.to_numeric(joined["rows_baseline"], errors="raise").astype(np.int64),
        "split": pd.to_numeric(joined["split_baseline"], errors="raise").astype(np.int64),
        "y": pd.to_numeric(joined["y_baseline"], errors="raise").astype(np.int8),
        "p_fm": pd.to_numeric(joined["p_fm"], errors="raise").astype(float),
        "p_baseline": pd.to_numeric(joined["p_baseline"], errors="raise").astype(float),
    }).sort_values(["subject", "stay_id"], kind="stable").reset_index(drop=True)


def _metric_values(y: np.ndarray, p: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    keep = weights > 0
    yy, pp, ww = y[keep], p[keep], weights[keep]
    if not len(yy):
        return {"auroc": np.nan, "brier": np.nan, "logloss": np.nan}
    if np.unique(yy).size == 2:
        auc = float(roc_auc_score(yy, pp, sample_weight=ww))
    else:
        auc = np.nan
    clipped = np.clip(pp, 1e-12, 1 - 1e-12)
    return {
        "auroc": auc,
        "brier": float(np.average((yy - pp) ** 2, weights=ww)),
        "logloss": float(np.average(-(yy * np.log(clipped) + (1 - yy) * np.log1p(-clipped)),
                                     weights=ww)),
    }


def _derive_seed(seed: int, db: str) -> int:
    digest = hashlib.sha256(f"{seed}|{db}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _paired_bootstrap(by_analysis: dict[str, pd.DataFrame], n_boot: int,
                      seed: int, db: str, fm_analysis_id: str,
                      input_condition: str) -> list[dict[str, Any]]:
    if n_boot < 1000:
        raise ValueError("paired patient-cluster bootstrap requires at least 1000 replicates")
    first = next(iter(by_analysis.values()))
    fm_p = first["p_fm"].to_numpy(dtype=float)
    y = first["y"].to_numpy(dtype=np.int8)
    groups, group_index = np.unique(first["subject"].astype(str), return_inverse=True)
    if len(groups) < 2:
        raise ValueError(f"{db}: paired patient-cluster bootstrap requires at least two patients")
    models = {name: frame["p_baseline"].to_numpy(dtype=float)
              for name, frame in by_analysis.items()}
    point_fm = _metric_values(y, fm_p, np.ones(len(y), dtype=float))
    point_base = {name: _metric_values(y, p, np.ones(len(y), dtype=float))
                  for name, p in models.items()}
    delta_draws = {name: {metric: np.full(n_boot, np.nan) for metric in ("auroc", "brier", "logloss")}
                   for name in models}
    rng_seed = _derive_seed(seed, db)
    rng = np.random.default_rng(rng_seed)
    weights_digest = hashlib.sha256()
    for replicate in range(n_boot):
        sampled = rng.integers(0, len(groups), size=len(groups))
        cluster_counts = np.bincount(sampled, minlength=len(groups)).astype(np.int32)
        weights = cluster_counts[group_index].astype(float)
        weights_digest.update(cluster_counts.astype("<i4").tobytes())
        metric_fm = _metric_values(y, fm_p, weights)
        for analysis_id, p_base in models.items():
            metric_base = _metric_values(y, p_base, weights)
            for metric in ("auroc", "brier", "logloss"):
                if np.isfinite(metric_fm[metric]) and np.isfinite(metric_base[metric]):
                    delta_draws[analysis_id][metric][replicate] = metric_fm[metric] - metric_base[metric]
    rows: list[dict[str, Any]] = []
    for analysis_id in models:
        for metric in ("auroc", "brier", "logloss"):
            draws = delta_draws[analysis_id][metric]
            valid = draws[np.isfinite(draws)]
            if len(valid) < n_boot // 2:
                low = high = np.nan
            else:
                low, high = np.percentile(valid, [2.5, 97.5])
            point_delta = (point_fm[metric] - point_base[analysis_id][metric]
                           if np.isfinite(point_fm[metric])
                           and np.isfinite(point_base[analysis_id][metric]) else np.nan)
            rows.append({
                "analysis_id": analysis_id, "db": db, "metric": metric,
                "fm_analysis_id": fm_analysis_id,
                "input_condition": input_condition,
                "estimate_fm": point_fm[metric],
                "estimate_baseline": point_base[analysis_id][metric],
                "delta_fm_minus_baseline": point_delta,
                "paired_ci_95_lo": float(low), "paired_ci_95_hi": float(high),
                "n_paired_stays": int(len(y)), "n_paired_patients": int(len(groups)),
                "bootstrap_draws": int(n_boot), "valid_paired_bootstrap_draws": int(len(valid)),
                "invalid_paired_bootstrap_draws": int(n_boot - len(valid)),
                "bootstrap_cluster_unit": "patient",
                "bootstrap_seed": int(rng_seed),
                "shared_bootstrap_weights_sha256": weights_digest.hexdigest(),
                "ci_scope": "shared patient-cluster bootstrap conditional on frozen source-selected models",
            })
    return rows


def compare_artifacts(baseline_manifest_path: str | Path,
                      fm_manifest_path: str | Path,
                      expected_fm_path: str | Path | None = None, *, n_boot: int = 1000,
                      seed: int = 20261122) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    baseline_manifest_path = Path(baseline_manifest_path).resolve()
    fm_manifest_path = Path(fm_manifest_path).resolve()
    baseline_doc = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    fm_doc = json.loads(fm_manifest_path.read_text(encoding="utf-8"))
    if baseline_doc.get("status") != "complete":
        raise ValueError("paired comparison requires a complete current baseline manifest")
    validate_complete_fm_prediction_manifest(fm_doc)
    baseline_manifest_sha = sha256_file(baseline_manifest_path)
    fm_manifest_sha = sha256_file(fm_manifest_path)
    fm_frames: dict[Path, pd.DataFrame] = {}
    audit_rows: list[dict[str, Any]] = []
    paired_by_group: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
    metadata_by_group: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seen_baseline_records: set[tuple[str, str]] = set()
    for base_record in baseline_doc.get("predictions", []):
        analysis_id, db = str(base_record.get("analysis_id")), str(base_record.get("db"))
        if not analysis_id or not db:
            raise ValueError("baseline manifest contains an incomplete prediction record")
        condition = str(base_record.get("input_condition", ""))
        if not condition:
            raise ValueError(f"{analysis_id}/{db}: baseline manifest lacks input_condition")
        record_key = (analysis_id, db)
        if record_key in seen_baseline_records:
            raise ValueError(f"baseline manifest repeats an analysis/cohort record: {record_key}")
        seen_baseline_records.add(record_key)
        baseline_cohort = baseline_doc.get("cohort_readiness", {}).get(db, {})
        if baseline_cohort.get("retrospective_landmark_prediction_allowed") is not True:
            raise ValueError(f"baseline manifest does not attest current eligibility readiness for {db}")
        baseline_path = Path(str(base_record.get("prediction_file", ""))).resolve()
        if not baseline_path.is_file():
            raise FileNotFoundError(f"baseline prediction artifact is absent: {baseline_path}")
        baseline_full = pd.read_parquet(baseline_path)
        baseline_frame = baseline_full.loc[baseline_full["db"].astype(str) == db].copy()
        _verify_artifact_record(baseline_frame, base_record, analysis_id=analysis_id,
                                db=db, expected_path=baseline_path, label="baseline")
        fm_candidates = [row for row in fm_doc.get("predictions", [])
                         if str(row.get("db")) == db
                         and str(row.get("model_tag")) == "main20k"
                         and str(row.get("input_condition")) == condition
                         and str(row.get("mode")) == "predict"
                         and int(row.get("n_prediction_rows", 0) or 0) > 0]
        if len(fm_candidates) != 1:
            raise ValueError(
                f"{analysis_id}/{db}: expected one current main20k FM entry for input "
                f"condition {condition!r}; found {len(fm_candidates)}")
        fm_record = fm_candidates[0]
        fm_analysis_id = str(fm_record.get("analysis_id", ""))
        if not fm_analysis_id:
            raise ValueError(f"{analysis_id}/{db}: FM manifest entry lacks an analysis ID")
        if base_record.get("fm_analysis_id") != fm_analysis_id:
            raise ValueError(f"{analysis_id}/{db}: paired FM analysis ID differs from its manifest record")
        if fm_record.get("retrospective_landmark_prediction_allowed") is not True:
            raise ValueError(f"FM manifest does not attest current clinical readiness for {db}")
        fm_prediction_path = Path(str(fm_record.get("prediction_file", ""))).resolve()
        if expected_fm_path is not None and fm_prediction_path != Path(expected_fm_path).resolve():
            raise ValueError("FM manifest does not name the expected current run-scoped prediction artifact")
        if not fm_prediction_path.is_file():
            raise FileNotFoundError(f"FM prediction artifact is absent: {fm_prediction_path}")
        if fm_prediction_path not in fm_frames:
            fm_frames[fm_prediction_path] = pd.read_parquet(fm_prediction_path)
        fm_frame = fm_frames[fm_prediction_path]
        fm_db = fm_frame.loc[
            (fm_frame["db"].astype(str) == db)
            & (fm_frame["analysis_id"].astype(str) == fm_analysis_id)
        ].copy()
        _verify_artifact_record(fm_db, fm_record, analysis_id=fm_analysis_id, db=db,
                                expected_path=fm_prediction_path, label="FM")
        for field in (
            "eligibility_manifest_sha256", "eligibility_table_sha256", "source_npz_sha256",
            "prediction_scope", "input_scope", "role", "clinical_input_availability_verified",
            "input_condition", "input_mask_sha256",
        ):
            if base_record.get(field) != fm_record.get(field):
                raise ValueError(f"{analysis_id}/{db}: baseline and FM manifest {field} mismatch")
        if base_record.get("input_mask_spec") != fm_record.get("input_mask_spec"):
            raise ValueError(f"{analysis_id}/{db}: baseline and FM input mask definitions differ")
        paired = _validate_pair(baseline_frame, fm_db, analysis_id, db)
        group_key = (db, condition)
        paired_by_group.setdefault(group_key, {})[analysis_id] = paired
        metadata_by_group.setdefault(group_key, []).append({
            "analysis_id": analysis_id, "input_condition": condition,
            "fm_analysis_id": fm_analysis_id, "n_baseline": int(len(baseline_frame)),
            "n_fm": int(len(fm_db)), "n_paired": int(len(paired)),
            "baseline_prediction_file_sha256": base_record["prediction_file_sha256"],
            "fm_prediction_file_sha256": fm_record["prediction_file_sha256"],
            "baseline_ordered_row_digest_sha256": base_record["ordered_row_digest_sha256"],
            "fm_ordered_row_digest_sha256": fm_record["ordered_row_digest_sha256"],
            "eligibility_manifest_sha256": base_record["eligibility_manifest_sha256"],
            "eligibility_table_sha256": base_record["eligibility_table_sha256"],
            "source_npz_sha256": base_record["source_npz_sha256"],
            "prediction_scope": base_record["prediction_scope"],
            "input_scope": base_record["input_scope"], "role": ROLE,
            "paired_identity_labels_scope_verified": True,
        })
    expected_cohorts = set(baseline_doc.get("cohort_readiness", {}))
    expected_conditions = set(baseline_doc.get("input_conditions", []))
    if not expected_cohorts or not expected_conditions:
        raise ValueError("baseline manifest must list all requested cohorts and input conditions")
    expected_groups = {(db, condition)
                       for db in expected_cohorts for condition in expected_conditions}
    if set(paired_by_group) != expected_groups:
        missing = sorted(expected_groups - set(paired_by_group))
        extra = sorted(set(paired_by_group) - expected_groups)
        raise ValueError(
            f"baseline prediction groups differ from requested cohorts/conditions "
            f"(missing={missing}, extra={extra})")
    metric_rows: list[dict[str, Any]] = []
    for (db, condition), model_frames in paired_by_group.items():
        if len(model_frames) < 2:
            raise ValueError(f"{db}/{condition}: expected LR/HGBT conventional analysis arms")
        group_metadata = metadata_by_group[(db, condition)]
        fm_analysis_ids = {row["fm_analysis_id"] for row in group_metadata}
        if len(fm_analysis_ids) != 1:
            raise ValueError(f"{db}/{condition}: conventional arms pair to different FM analyses")
        fm_analysis_id = next(iter(fm_analysis_ids))
        first = next(iter(model_frames.values()))
        for analysis_id, paired in model_frames.items():
            if not first[["stay_id", "subject", "rows", "split", "y"]].equals(
                paired[["stay_id", "subject", "rows", "split", "y"]]
            ) or not np.array_equal(first["p_fm"].to_numpy(), paired["p_fm"].to_numpy()):
                raise ValueError(
                    f"{db}/{condition}: conventional arms do not share exact paired rows and FM probabilities")
        metric_rows.extend(_paired_bootstrap(
            model_frames, n_boot, seed, db, fm_analysis_id, condition))
        audit_rows.extend({
            **row, "input_condition": condition,
            "n_paired_patients": int(first["subject"].nunique()),
            "n_excluded_baseline": 0, "n_excluded_fm": 0,
            "paired_subject_rows_split_labels_verified": True,
            "fm_prediction_manifest_sha256": fm_manifest_sha,
            "baseline_prediction_manifest_sha256": baseline_manifest_sha,
        } for row in group_metadata)
    fm_paths = sorted(fm_frames)
    summary = {
        "baseline_prediction_manifest_sha256": baseline_manifest_sha,
        "fm_prediction_manifest_sha256": fm_manifest_sha,
        "fm_prediction_files": {str(path): sha256_file(path) for path in fm_paths},
        "bootstrap_draws": int(n_boot), "base_seed": int(seed),
        "bootstrap_unit": "patient; identical sampled patient multiplicities are used for FM and every baseline arm and metric",
        "metrics": ["auroc", "brier", "logloss"],
        "delta_direction": "FM minus conventional baseline",
        "comparison_scope": "retrospective landmark prediction at recorded measurement time; clinical input availability not verified",
        "n_comparisons": int(len(audit_rows)),
    }
    return audit_rows, metric_rows, summary
