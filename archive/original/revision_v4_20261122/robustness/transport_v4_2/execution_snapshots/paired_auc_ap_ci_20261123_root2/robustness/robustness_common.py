"""Identity-safe inputs and shared metrics for v4 robustness analyses.

Input contract (from the v4 prediction engine): a Parquet prediction file with
db, stay_id, subject, split, y, p, model_tag, input_condition, endpoint; and
``<eligibility-dir>/<db>.parquet`` with db, stay_id, subject, eligible_24h,
temporal_verified, y_mort, endpoint, and optional age/sex fields.
"""
from __future__ import annotations

import json
import hashlib
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

ID_COLUMNS = ["db", "stay_id", "subject"]
PREDICTION_COLUMNS = ID_COLUMNS + [
    "analysis_id", "rows", "split", "y", "p", "model_tag", "input_condition",
    "endpoint", "role", "prediction_scope", "input_scope",
    "clinical_input_availability_verified", "eligibility_manifest_sha256",
    "eligibility_table_sha256",
]
ELIGIBILITY_COLUMNS = ID_COLUMNS + [
    "eligible_24h", "temporal_verified", "y_mort", "endpoint"
]
INVALID_IDS = {"", "nan", "none", "null", "-1"}
PRIMARY_PREDICTION_SCOPE = "retrospective_landmark_prediction_measurement_time"
DIAGNOSTIC_PREDICTION_SCOPE = "retrospective_association_unverified_landmark"
INPUT_SCOPE = "retrospective_measurement_time"
ROW_DIGEST_FIELDS = [
    "db", "analysis_id", "stay_id", "subject", "rows", "split", "y", "p",
    "endpoint", "role", "prediction_scope", "input_scope",
    "eligibility_manifest_sha256", "eligibility_table_sha256",
]
REGISTERED_BASELINE_ANALYSIS_TAGS = {
    "conventional_lr_pattern_only": "conventional_lr",
    "conventional_hgb_pattern_only": "conventional_hgb",
    "conventional_lr_value_plus_pattern": "conventional_lr",
    "conventional_hgb_value_plus_pattern": "conventional_hgb",
}
_CURRENT_MANIFEST_VALIDATOR = None


def validate_prediction_manifest(document: dict[str, Any]) -> None:
    """Use the paired-baseline adapter for current and completed run manifests."""
    global _CURRENT_MANIFEST_VALIDATOR
    if _CURRENT_MANIFEST_VALIDATOR is None:
        validator_path = Path(__file__).resolve().parent.parent / "baselines" / "paired_baselines.py"
        if not validator_path.is_file():
            raise FileNotFoundError(f"shared current-manifest validator is absent: {validator_path}")
        spec = importlib.util.spec_from_file_location(
            "_v4_paired_baselines_manifest_adapter", validator_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load shared manifest validator: {validator_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CURRENT_MANIFEST_VALIDATOR = module.validate_complete_fm_prediction_manifest
    _CURRENT_MANIFEST_VALIDATOR(document)


def read_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported tabular input: {path}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_prediction_digest(frame: pd.DataFrame) -> str:
    """Digest an ordered prediction artifact using the engine manifest contract."""
    missing = set(ROW_DIGEST_FIELDS) - set(frame.columns)
    if missing:
        raise ValueError(f"prediction rows lack manifest digest fields: {sorted(missing)}")
    digest = hashlib.sha256()
    for row in frame[ROW_DIGEST_FIELDS].itertuples(index=False, name=None):
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


def infer_prediction_manifest_path(predictions_path: str | Path) -> Path:
    path = Path(predictions_path).resolve()
    if path.parent.name == "predictions":
        return path.parent.parent / "prediction_manifest.json"
    return path.parent / "prediction_manifest.json"


def _manifest_cohort(path: str | Path, db: str) -> dict[str, Any] | None:
    try:
        manifest = json.loads((Path(path) / "eligibility_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    row = manifest.get("cohorts", {}).get(db)
    return row if isinstance(row, dict) else None


def _source_sequence_hash(cohort: dict[str, Any]) -> str | None:
    for field in ("sourcehashes", "source_hashes"):
        hashes = cohort.get(field)
        if isinstance(hashes, dict):
            value = hashes.get("sequence_npz", hashes.get("sequence_npz_sha256"))
            if isinstance(value, str):
                return value.lower()
    value = cohort.get("sequence_npz_sha256")
    return value.lower() if isinstance(value, str) else None


def _manifest_prediction_entry(document: dict[str, Any], analysis_id: str,
                               db: str) -> dict[str, Any]:
    entries = [row for row in document.get("predictions", [])
               if str(row.get("analysis_id")) == analysis_id
               and str(row.get("db")) == db]
    if len(entries) != 1:
        raise ValueError(
            f"prediction manifest needs one current entry for {analysis_id}/{db}; found {len(entries)}")
    return entries[0]


def _verify_prediction_artifact(prediction_frame: pd.DataFrame, record: dict[str, Any],
                                predictions_path: str | Path, eligibility_dir: str | Path,
                                analysis_id: str, db: str, *, diagnostic_only: bool,
                                expected_model_tag: str) -> None:
    required = (
        "prediction_file", "prediction_file_sha256", "ordered_row_digest_sha256",
        "n_prediction_rows", "role", "prediction_scope", "input_scope",
        "clinical_input_availability_verified", "eligibility_manifest_sha256",
        "eligibility_table_sha256", "source_npz_sha256", "input_condition",
    )
    missing = [field for field in required if record.get(field) in (None, "")]
    if missing:
        raise ValueError(f"{analysis_id}/{db}: current prediction manifest lacks {missing}")
    actual_path = Path(str(record["prediction_file"])).resolve()
    requested_path = Path(predictions_path).resolve()
    if actual_path != requested_path:
        raise ValueError(f"{analysis_id}/{db}: prediction manifest names a stale or different file")
    if sha256_file(actual_path) != str(record["prediction_file_sha256"]):
        raise ValueError(f"{analysis_id}/{db}: prediction file hash differs from current manifest")
    expected_scope = DIAGNOSTIC_PREDICTION_SCOPE if diagnostic_only else PRIMARY_PREDICTION_SCOPE
    if record["role"] != "evaluation":
        raise ValueError(f"{analysis_id}/{db}: prediction manifest role must be evaluation")
    if record["prediction_scope"] != expected_scope:
        raise ValueError(
            f"{analysis_id}/{db}: prediction scope {record['prediction_scope']!r} "
            f"does not match requested scope {expected_scope!r}")
    if record["input_scope"] != INPUT_SCOPE:
        raise ValueError(f"{analysis_id}/{db}: input scope must be retrospective measurement time")
    if record["clinical_input_availability_verified"] is not False:
        raise ValueError(f"{analysis_id}/{db}: real-time input availability must remain explicitly unverified")
    if diagnostic_only:
        if record.get("retrospective_diagnostic_allowed") is not True:
            raise ValueError(f"{analysis_id}/{db}: separate diagnostic attestation is missing")
        if record.get("retrospective_landmark_prediction_allowed") is True:
            raise ValueError(f"{analysis_id}/{db}: diagnostic rows cannot claim the primary scope")
    elif record.get("retrospective_landmark_prediction_allowed") is not True:
        raise ValueError(f"{analysis_id}/{db}: current primary scope is not attested")
    if len(prediction_frame) != int(record["n_prediction_rows"]):
        raise ValueError(f"{analysis_id}/{db}: prediction row count differs from current manifest")
    if prediction_frame.empty or prediction_frame["analysis_id"].astype(str).ne(analysis_id).any():
        raise ValueError(f"{analysis_id}/{db}: selected prediction rows are empty or carry another analysis ID")
    record_model_tag = record.get("model_tag")
    if record_model_tag in (None, ""):
        registered_model_tag = REGISTERED_BASELINE_ANALYSIS_TAGS.get(analysis_id)
        if registered_model_tag != expected_model_tag:
            raise ValueError(f"{analysis_id}/{db}: manifest omits model tag outside the registered baseline mapping")
    elif str(record_model_tag) != expected_model_tag:
        raise ValueError(f"{analysis_id}/{db}: row-level model tag differs from the current manifest")
    if prediction_frame["model_tag"].astype(str).ne(expected_model_tag).any():
        raise ValueError(f"{analysis_id}/{db}: row-level model tag differs from the requested registered model")
    if prediction_frame["input_condition"].astype(str).ne(str(record["input_condition"])).any():
        raise ValueError(f"{analysis_id}/{db}: row-level input condition differs from current manifest")
    if ordered_prediction_digest(prediction_frame) != str(record["ordered_row_digest_sha256"]):
        raise ValueError(f"{analysis_id}/{db}: ordered prediction rows differ from current manifest digest")
    eligibility_dir = Path(eligibility_dir)
    table_path = eligibility_dir / f"{db}.parquet"
    manifest_path = eligibility_dir / "eligibility_manifest.json"
    if not table_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"{db}: current eligibility table or manifest is absent")
    current_table_hash = sha256_file(table_path)
    current_manifest_hash = sha256_file(manifest_path)
    if str(record["eligibility_table_sha256"]) != current_table_hash:
        raise ValueError(f"{analysis_id}/{db}: prediction uses a stale eligibility table")
    if str(record["eligibility_manifest_sha256"]) != current_manifest_hash:
        raise ValueError(f"{analysis_id}/{db}: prediction uses a stale eligibility manifest")
    cohort = _manifest_cohort(eligibility_dir, db)
    current_sequence_hash = _source_sequence_hash(cohort or {})
    if not current_sequence_hash or str(record["source_npz_sha256"]).lower() != current_sequence_hash:
        raise ValueError(f"{analysis_id}/{db}: prediction source NPZ hash differs from current eligibility attestation")


def _as_bool(series: pd.Series, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": True, "1": True, "yes": True, "y": True,
         "false": False, "0": False, "no": False, "n": False}
    )
    if mapped.isna().any():
        raise ValueError(f"{label} contains missing or unrecognized boolean values")
    return mapped.astype(bool)


def _validate_ids(df: pd.DataFrame, name: str) -> pd.DataFrame:
    for col in ID_COLUMNS:
        if col not in df:
            raise ValueError(f"{name} is missing required identity column {col!r}")
        df[col] = df[col].astype("string").str.strip()
        if df[col].isna().any() or df[col].str.lower().isin(INVALID_IDS).any():
            raise ValueError(f"{name}.{col} contains missing/placeholder identifiers")
    if df.duplicated(["db", "stay_id"]).any():
        examples = df.loc[df.duplicated(["db", "stay_id"], keep=False), ID_COLUMNS].head(4)
        raise ValueError(f"{name} has duplicate db/stay_id source keys: {examples.to_dict('records')}")
    return df


def manifest_ready(path: str | Path, db: str) -> tuple[bool, str]:
    """Require the locked current per-cohort retrospective landmark attestation."""
    manifest_path = Path(path) / "eligibility_manifest.json"
    if not manifest_path.is_file():
        return False, "eligibility_manifest_missing"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"eligibility_manifest_unreadable:{type(exc).__name__}"
    row = manifest.get("cohorts", {}).get(db)
    if not isinstance(row, dict):
        return False, "cohort_specific_attestation_missing"
    if row.get("retrospective_landmark_prediction_allowed") is not True:
        return False, "cohort_retrospective_landmark_prediction_not_attested"
    if ("ready_for_future24h_landmark_analysis" in row
            and row.get("ready_for_future24h_landmark_analysis") is not True):
        return False, "cohort_landmark_readiness_alias_conflicts_with_primary_attestation"
    evidence = str(row.get("evidence", "")).lower()
    if evidence not in {"exact", "operational_proxy"}:
        return False, f"unsupported_outcome_timing_evidence:{evidence or 'missing'}"
    status = str(row.get("status", "")).lower()
    if status in {"unavailable", "not_ready", "blocked", "failed"}:
        return False, f"cohort_status_blocks_retrospective_landmark_prediction:{status}"
    basis = row.get("outcome_timing_basis", row.get("evidence_basis", evidence))
    return True, f"cohort_retrospective_landmark_attestation;evidence={evidence};basis={basis}"


def manifest_diagnostic_allowed(path: str | Path, db: str) -> tuple[bool, str]:
    """Diagnostic association scope is available only under its separate attestation."""
    manifest_path = Path(path) / "eligibility_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"eligibility_manifest_unreadable:{type(exc).__name__}"
    row = manifest.get("cohorts", {}).get(db)
    if not isinstance(row, dict) or row.get("retrospective_diagnostic_allowed") is not True:
        return False, "cohort_retrospective_diagnostic_not_attested"
    return True, "cohort_retrospective_diagnostic_attestation"


def manifest_timing_basis(path: str | Path, db: str) -> str:
    manifest_path = Path(path) / "eligibility_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "not_recorded"
    row = manifest.get("cohorts", {}).get(db)
    if not isinstance(row, dict):
        return "not_recorded"
    basis = row.get("outcome_timing_basis", row.get("evidence_basis", row.get("evidence")))
    return str(basis) if basis is not None else "not_recorded"


def manifest_subject_identity(path: str | Path, db: str) -> tuple[str, bool, str]:
    manifest_path = Path(path) / "eligibility_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "not_recorded", False, "supplied_subject_identifier"
    row = manifest.get("cohorts", {}).get(db)
    if not isinstance(row, dict):
        return "not_recorded", False, "supplied_subject_identifier_not_verified_as_patient"
    basis = row.get("subject_identity_basis")
    if isinstance(basis, dict):
        explicit = basis.get("patient_disjoint_claim_allowed")
        basis_text = str(basis.get("basis", basis))
    else:
        explicit = None
        basis_text = str(basis) if basis is not None else "not_recorded"
    # An explicit false is final; text heuristics and unrelated legacy fields
    # cannot convert a non-longitudinal key into a patient identifier.
    is_patient = explicit is True
    label = "patient" if is_patient else "supplied_subject_identifier_not_verified_as_patient"
    return basis_text, is_patient, label


def load_checked(
    predictions_path: str | Path,
    eligibility_dir: str | Path,
    *,
    model_tag: str = "main20k",
    analysis_id: str | None = None,
    input_condition: str = "real",
    endpoint: str = "hospital",
    prediction_manifest_path: str | Path | None = None,
    diagnostic_only: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    """Verify one current artifact and strict-join it to per-cohort eligibility.

    Primary analyses accept only retrospective landmark predictions made from
    recorded measurement time. Diagnostic association files require the
    explicit ``diagnostic_only`` mode and never open the primary gate.
    """
    predictions_path = Path(predictions_path).resolve()
    pred_all = read_frame(predictions_path)
    missing = sorted(set(PREDICTION_COLUMNS) - set(pred_all.columns))
    if missing:
        raise ValueError(f"Predictions missing required columns: {missing}")
    pred_all["split"] = pd.to_numeric(pred_all["split"], errors="raise").astype(int)
    pred_all["rows"] = pd.to_numeric(pred_all["rows"], errors="raise").astype(np.int64)
    pred_all["y"] = pd.to_numeric(pred_all["y"], errors="coerce")
    pred_all["p"] = pd.to_numeric(pred_all["p"], errors="coerce")
    pred_all["analysis_id"] = pred_all["analysis_id"].astype(str)
    expected_analysis_id = analysis_id or f"{model_tag}_{input_condition}"
    if (expected_analysis_id != f"{model_tag}_{input_condition}"
            and REGISTERED_BASELINE_ANALYSIS_TAGS.get(expected_analysis_id) != model_tag):
        raise ValueError(
            f"analysis ID {expected_analysis_id!r} is not registered for model tag {model_tag!r}")
    pred = pred_all.loc[
        (pred_all["model_tag"].astype(str) == model_tag)
        & (pred_all["input_condition"].astype(str) == input_condition)
        & (pred_all["endpoint"].astype(str) == endpoint)
        & (pred_all["analysis_id"].astype(str) == expected_analysis_id)
    ].copy()
    if pred.empty:
        raise ValueError("No evaluation predictions match the requested model/input/endpoint")
    analysis_ids = pred["analysis_id"].unique()
    if len(analysis_ids) != 1 or str(analysis_ids[0]) != expected_analysis_id:
        raise ValueError(
            f"selected prediction rows must come from exactly {expected_analysis_id}; found {analysis_ids.tolist()}")
    analysis_id = expected_analysis_id
    expected_scope = DIAGNOSTIC_PREDICTION_SCOPE if diagnostic_only else PRIMARY_PREDICTION_SCOPE
    if pred["role"].astype(str).ne("evaluation").any():
        raise ValueError("only current role=evaluation predictions may be analyzed")
    if pred["prediction_scope"].astype(str).ne(expected_scope).any():
        raise ValueError(f"prediction scope must be {expected_scope!r}")
    if pred["input_scope"].astype(str).ne(INPUT_SCOPE).any():
        raise ValueError("prediction input_scope must be retrospective_measurement_time")
    if pred["clinical_input_availability_verified"].isna().any() or pred[
        "clinical_input_availability_verified"
    ].astype(bool).any():
        raise ValueError("clinical input availability must remain explicitly unverified")
    pred = _validate_ids(pred, "selected predictions")
    if pred["y"].isna().any() or not pred["y"].isin([0, 1]).all():
        raise ValueError("Selected predictions must have finite binary labels")
    if pred["p"].isna().any() or not np.isfinite(pred["p"]).all() or not pred["p"].between(0, 1).all():
        raise ValueError("Selected predictions must have finite probabilities in [0, 1]")

    prediction_manifest_path = (Path(prediction_manifest_path).resolve()
                                if prediction_manifest_path is not None
                                else infer_prediction_manifest_path(predictions_path))
    if not prediction_manifest_path.is_file():
        raise FileNotFoundError(f"current prediction manifest is absent: {prediction_manifest_path}")
    try:
        prediction_manifest = json.loads(prediction_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"current prediction manifest cannot be read: {type(exc).__name__}") from exc
    validate_prediction_manifest(prediction_manifest)

    out: dict[str, pd.DataFrame] = {}
    gates: dict[str, dict[str, Any]] = {}
    eligibility_dir = Path(eligibility_dir)
    for db, p_db in pred.groupby("db", sort=True):
        eligibility_path = eligibility_dir / f"{db}.parquet"
        if not eligibility_path.is_file():
            raise FileNotFoundError(f"Required eligibility file is absent: {eligibility_path}")
        eligible = read_frame(eligibility_path)
        missing = sorted(set(ELIGIBILITY_COLUMNS) - set(eligible.columns))
        if missing:
            raise ValueError(f"{eligibility_path} missing required columns: {missing}")
        eligible = _validate_ids(eligible, f"eligibility[{db}]")
        eligible = eligible.loc[eligible["db"].astype(str) == str(db)].copy()
        eligible["y_mort"] = pd.to_numeric(eligible["y_mort"], errors="coerce")
        eligible["eligible_24h"] = _as_bool(eligible["eligible_24h"], f"{db}.eligible_24h")
        eligible["temporal_verified"] = _as_bool(eligible["temporal_verified"], f"{db}.temporal_verified")
        p_keys = set(map(tuple, p_db[ID_COLUMNS].itertuples(index=False, name=None)))
        e_keys = set(map(tuple, eligible[ID_COLUMNS].itertuples(index=False, name=None)))
        if not p_keys.issubset(e_keys):
            absent = list(p_keys - e_keys)[:4]
            raise ValueError(
                f"{db}: prediction IDs are absent from full eligibility metadata (prediction_only={absent})"
            )
        joined = p_db.merge(
            eligible, on=ID_COLUMNS, how="inner", suffixes=("_prediction", "_eligibility"),
            validate="one_to_one",
        )
        if not np.array_equal(joined["y"].to_numpy(float), joined["y_mort"].to_numpy(float)):
            raise ValueError(f"{db}: prediction labels do not exactly match eligibility labels")
        if not (joined["endpoint_prediction"].astype(str) == endpoint).all():
            raise ValueError(f"{db}: prediction endpoint differs from requested endpoint")
        if not (joined["endpoint_eligibility"].astype(str) == endpoint).all():
            raise ValueError(f"{db}: eligibility endpoint differs from requested endpoint")
        if (not diagnostic_only
                and (not joined["eligible_24h"].all() or not joined["temporal_verified"].all())):
            raise ValueError(f"{db}: prediction rows include noneligible or unverified source outcomes")
        if db == "mimic" and joined["split"].ne(2).any():
            raise ValueError("MIMIC robustness predictions must use the locked evaluation split 2")
        artifact_rows = pred_all.loc[
            (pred_all["analysis_id"].astype(str) == analysis_id)
            & (pred_all["db"].astype(str) == str(db))
        ].copy()
        record = _manifest_prediction_entry(prediction_manifest, analysis_id, str(db))
        _verify_prediction_artifact(
            artifact_rows, record, predictions_path, eligibility_dir,
            analysis_id, str(db), diagnostic_only=diagnostic_only,
            expected_model_tag=model_tag,
        )
        if diagnostic_only:
            if record.get("retrospective_diagnostic_allowed") is not True:
                raise ValueError(f"{db}: prediction manifest lacks separate diagnostic-mode attestation")
            if record.get("retrospective_landmark_prediction_allowed") is True:
                raise ValueError(f"{db}: diagnostic prediction cannot claim primary landmark eligibility")
        else:
            if record.get("retrospective_landmark_prediction_allowed") is not True:
                raise ValueError(f"{db}: prediction manifest lacks current primary landmark attestation")
        cohort = _manifest_cohort(eligibility_dir, str(db))
        if cohort is None:
            raise ValueError(f"{db}: cohort-specific eligibility manifest entry is missing")
        manifest_ok, manifest_reason = manifest_ready(eligibility_dir, str(db))
        diagnostic_ok, diagnostic_reason = manifest_diagnostic_allowed(eligibility_dir, str(db))
        timing_basis = manifest_timing_basis(eligibility_dir, str(db))
        identity_basis, identity_ok, cluster_unit = manifest_subject_identity(eligibility_dir, str(db))
        candidate = joined.loc[joined["eligible_24h"]]
        row_ok = bool(
            len(candidate) > 0
            and candidate["temporal_verified"].all()
            and candidate["y_mort"].isin([0, 1]).all()
        )
        ready = bool(not diagnostic_only and manifest_ok and row_ok and identity_ok)
        if diagnostic_only:
            if not diagnostic_ok:
                raise ValueError(f"{db}: diagnostic-only mode is not attested: {diagnostic_reason}")
            gate_reason = "diagnostic_only_no_clinical_promotion"
            joined["cohort_status"] = DIAGNOSTIC_PREDICTION_SCOPE
        else:
            if not manifest_ok or not row_ok or not identity_ok:
                details = ";".join(
                    x for x in [manifest_reason,
                                "row_level_temporal_eligibility_not_all_verified" if not row_ok else "",
                                "patient_identity_not_explicitly_verified" if not identity_ok else ""] if x
                )
                raise ValueError(f"{db}: primary retrospective landmark gate failed: {details}")
            gate_reason = "current_per_cohort_retrospective_landmark_attestation"
            joined["cohort_status"] = PRIMARY_PREDICTION_SCOPE
        out[str(db)] = joined
        gates[str(db)] = {
            "retrospective_landmark_prediction_allowed": ready,
            "ready_for_future24h_clinical_claim": False,
            "clinical_promotion_allowed": False,
            "reason": gate_reason,
            "outcome_timing_basis": timing_basis,
            "operational_proxy_status": "explicitly_recorded" if "proxy" in timing_basis.lower() else "not_identified",
            "outcome_timing_evidence": str(cohort.get("evidence", "not_recorded")),
            "subject_identity_basis": identity_basis,
            "independent_patient_clustering_verified": identity_ok,
            "bootstrap_cluster_unit": cluster_unit,
            "prediction_scope": expected_scope,
            "input_scope": INPUT_SCOPE,
            "clinical_input_availability_verified": False,
            "prediction_manifest": str(prediction_manifest_path),
            "prediction_manifest_sha256": sha256_file(prediction_manifest_path),
            "prediction_file_sha256": sha256_file(predictions_path),
            "eligibility_manifest_sha256": sha256_file(eligibility_dir / "eligibility_manifest.json"),
            "eligibility_table_sha256": sha256_file(eligibility_path),
            "analysis_id": analysis_id,
            "n_joined": int(len(joined)),
            "n_eligible_24h": int(joined["eligible_24h"].sum()),
            "n_temporal_verified": int(joined["temporal_verified"].sum()),
            "n_metadata_only_rows_not_in_prediction_file": int(len(eligible) - len(joined)),
            "n_patients": int(joined["subject"].nunique()),
            "analysis_claim_scope": expected_scope if ready else "diagnostic_only_descriptive",
        }
    return out, gates


def brier(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((y - p) ** 2))


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def hindsight_bss(y: np.ndarray, p: np.ndarray) -> float:
    """Descriptive BSS; assessment prevalence is a hindsight-only reference."""
    y = np.asarray(y, dtype=float)
    prevalence = float(np.mean(y))
    denom = prevalence * (1 - prevalence)
    if denom <= 0:
        return float("nan")
    return 1 - brier(y, p) / denom


def calibration_slope(y: np.ndarray, p: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    y = np.asarray(y, dtype=int)
    x = logit(np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)).reshape(-1, 1)
    if np.unique(y).size != 2:
        return float("nan")
    model = LogisticRegression(C=1e6, max_iter=2000, tol=1e-8, solver="lbfgs")
    model.fit(x, y, sample_weight=sample_weight)
    return float(model.coef_[0, 0])


def write_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)

