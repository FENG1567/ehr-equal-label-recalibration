#!/usr/bin/env python3
"""Aggregate H24 temporal, clinical-flow, raw-field, and ICU bridge audits.

All patient-level inputs and intermediate arrays remain on the server. Only
aggregate CSV/JSON/README outputs are written, under the R5 temporal, cohorts,
and bridge directories. R4 inputs and all embedding caches are read-only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
R4 = ROOT / "revision_v4_20261122"
R5 = ROOT / "revision_v5_20261123"
ELIG_DIR = R4 / "temporal/results/eligibility/recovered_reviewed_v4_2"
EMBED_DIR = R4 / "engine/cache"
REGISTRY_PATH = R4 / "data_registry.json"
MANIFEST_PATH = R4 / "engine/code_snapshots/embedding_manifest_recovered_v4_2_merged41_b2eec1f7aca1.json"
MERGE_AUDIT_PATH = R4 / "engine/code_snapshots/embedding_manifest_merge_audit_recovered_v4_2_b2eec1f7aca1.json"
EXPECTED_MANIFEST_SHA = "b2eec1f7aca1857bda440ddb5ec48e477b2a20f753d5dea28b7715f7a1a243fe"
MIMIC_RAW = Path("/data/admin/fengshuo/if10_20261002/shared/raw/mimic")
EICU_RAW = Path("/data/admin/fengshuo/if10_20261002/shared/raw/eicu")
NWICU_RAW = R4 / "temporal/input/nwicu"
SICDB_RAW = R4 / "temporal/input/sicdb"

TRANSPORT_DIR = R4 / "robustness/transport_v4_2"
SOURCE_TRANSPORT_DIR = R4 / "baselines/transport_v4_2"
for _path in (TRANSPORT_DIR, SOURCE_TRANSPORT_DIR, R4 / "admin", R4 / "robustness",
              R5, R5 / "scripts",
              Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import transport_analysis as _transport  # noqa: E402
import frozen_source_transport as _source_transport  # noqa: E402
from robustness_common import load_checked  # noqa: E402
from recalibration import fit_offset, fit_platt  # noqa: E402
from prepare_nwicu_discharge_v4 import KNOWN_ALIVE as NWICU_KNOWN_ALIVE, KNOWN_DEAD as NWICU_KNOWN_DEAD  # noqa: E402
from v5_curve_tools import paired_cluster_mean_ci  # noqa: E402

COHORTS = {
    "mimic": "mimic",
    "eicu": "eicu_semantic_v4_2",
    "nwicu": "nwicu_discharge_v4_1",
    "sicdb": "sicdb_zero_offset_v4_2",
}
ELIG_NAMES = {
    "mimic": "mimic",
    "eicu": "eicu_semantic_v4_2",
    "nwicu": "nwicu_discharge_v4_1",
    "sicdb": "sicdb_zero_offset_v4_2",
}
SEQ_PATHS = {
    "mimic": ROOT / "data/processed/seq_mimic.npz",
    "eicu": R4 / "datasets/eicu_semantic_v4_2/seq_eicu_semantic_v4_2.npz",
    "nwicu": R4 / "datasets/nwicu_discharge_v4_1/seq_nwicu.npz",
    "sicdb": R4 / "datasets/sicdb_zero_offset_v4_2/seq_sicdb_zero_offset_v4_2.npz",
}
SPLIT_PATHS = {
    "mimic": ROOT / "data/processed/split_mimic.npy",
    "eicu": R4 / "datasets/eicu_semantic_v4_2/split_eicu_semantic_v4_2.npy",
    "nwicu": R4 / "datasets/nwicu_discharge_v4_1/split_nwicu.npy",
    "sicdb": R4 / "datasets/sicdb_zero_offset_v4_2/split_sicdb_zero_offset_v4_2.npy",
}
VOCAB_PATHS = {
    "mimic": ROOT / "data/processed/meta_mimic.json",
    "eicu": R4 / "datasets/eicu_semantic_v4_2/meta_eicu_semantic_v4_2.json",
    "nwicu": R4 / "datasets/nwicu_discharge_v4_1/meta_nwicu.json",
    "sicdb": R4 / "datasets/sicdb_zero_offset_v4_2/meta_sicdb_zero_offset_v4_2.json",
}
CACHE_PATHS = {
    "mimic": EMBED_DIR / "emb_mimic_main20k_real_ckpt-0719db57b91b2618_id-009c81e89cf7b3ad_mask-3f453e2f4b91e5d9_src-9bd20bf1394a_enc-59c3f4059662_rt-edca09b26174.npz",
    "eicu": EMBED_DIR / "emb_eicu_semantic_v4_2_main20k_real_ckpt-0719db57b91b2618_id-b01139f6cab6e963_mask-3f453e2f4b91e5d9_src-e44507da41b0_enc-59c3f4059662_rt-edca09b26174.npz",
}

RUN_ID = os.environ.get("V5_RUN_ID", "")
RUN_MATCH = re.fullmatch(r"v5_temporal_bridge_(audit|bridge)_(\d{8}T\d{6}Z_\d+)", RUN_ID)
if RUN_MATCH is None:
    raise RuntimeError("V5_RUN_ID must be supplied by the unique systemd runner")
RUN_MODE, RUN_TOKEN = RUN_MATCH.groups()
OUT_TEMP = R5 / f"temporal/{RUN_ID}"
OUT_COHORT = R5 / f"cohorts/{RUN_ID}"
OUT_BRIDGE = R5 / f"bridge/v5_icu_endpoint_bridge_{RUN_TOKEN}"
SEED = 20261123
BOOT_DRAWS = 1000
EPS = 1e-6
CS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
RUNTIME_LIMITS = {
    "required_user_slice_memory_high_bytes": 182536110080,
    "systemd_memory_max_bytes": 3221225472,
    "systemd_memory_max_gib": 3,
    "systemd_memory_swap_max_bytes": 0,
    "systemd_cpu_quota_percent": 200,
    "global_serial_lock": "admin/v5_serial_science.lock",
}
PREDICTION_PROVENANCE: dict[str, Any] = {}
PREDICTION_MANIFEST_PATHS = (
    _transport.ORIGINAL_FM_MANIFEST,
    _source_transport.SOURCE_PREDICTION_MANIFEST,
    _source_transport.SOURCE_MANIFEST,
    _transport.BASELINE_PREDICTION_MANIFEST,
    _transport.RECOVERED_MANIFEST,
    _transport.SOURCE_REPRODUCTION_MANIFEST,
    _transport.RECOVERED_GATE / "eligibility_manifest.json",
    _transport.IDENTITY_ATTESTATION_PATH,
)
ORIGINAL_EICU_TIMING_PATH = _transport.ORIGINAL_ELIGIBILITY_GATE / "eicu.parquet"
ORIGINAL_EICU_TIMING_PROVENANCE: dict[str, Any] = {
    "status": "not_loaded", "path": str(ORIGINAL_EICU_TIMING_PATH),
}


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(float(value)):
            return None
        if float(value).is_integer():
            return str(int(value))
    return str(value).strip()


def parse_bool(series: pd.Series, name: str, default: bool = False) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(default).astype(bool)
    low = series.astype("string").str.strip().str.lower()
    mapping = {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
    unknown = set(low.dropna().unique()).difference(mapping)
    if unknown:
        raise ValueError(f"{name} has unknown boolean strings: {sorted(unknown)}")
    return low.map(mapping).fillna(default).astype(bool)


def keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ("db", "stay_id", "subject"):
        if col not in out:
            raise ValueError(f"required identity column missing: {col}")
        out[col] = out[col].map(canonical)
    if out[["db", "stay_id", "subject"]].isna().any().any():
        raise ValueError("null prediction/cohort identity key")
    if out.duplicated(["db", "stay_id", "subject"]).any():
        raise ValueError("duplicate db/stay_id/subject identity")
    return out


def read_eligibility(db: str) -> pd.DataFrame:
    path = ELIG_DIR / f"{ELIG_NAMES[db]}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    if "subject" not in frame and "subject_id" in frame:
        frame["subject"] = frame["subject_id"]
    if "db" not in frame:
        frame["db"] = COHORTS[db]
    frame = keys(frame)
    frame["y_mort"] = pd.to_numeric(frame["y_mort"], errors="coerce")
    if "eligible_24h" in frame:
        frame["eligible_24h"] = parse_bool(frame["eligible_24h"], f"{db}.eligible_24h")
    if db == "eicu":
        frame = attach_original_eicu_timing_metadata(frame)
    return frame


def attach_original_eicu_timing_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Strictly key-join original timing annotations onto repaired eICU identities."""
    global ORIGINAL_EICU_TIMING_PROVENANCE
    path = ORIGINAL_EICU_TIMING_PATH
    if not path.is_file():
        raise FileNotFoundError(f"original eICU timing metadata is missing: {path}")
    original = pd.read_parquet(path)
    if "subject" not in original and "subject_id" in original:
        original["subject"] = original["subject_id"]
    if "db" not in original:
        original["db"] = COHORTS["eicu"]
    else:
        original["db"] = original["db"].replace({"eicu": COHORTS["eicu"]})
    original = keys(original)
    identity = ["db", "stay_id", "subject"]
    source_fields = [
        "rows", "split", "split_expected", "y_mort", "source_y_mort",
        "dod_day_overlaps_landmark", "source_patient_dod_interval_overlaps_24h",
        "source_postlandmark_ordering_issue", "source_icu_exit_after_hospital_exit_after_24",
        "icu_exit_after_hospital_exit_after24", "icu_exit_after_hospital_exit_after_24",
        "hospital_admission_after_landmark",
        "source_hospital_admit_after_24h", "hospital_death_time_hr",
        "hospital_death_time_kind", "icu_death_time_hr", "icu_death_time_kind",
        "source_landmark_contradiction", "source_landmark_issue",
    ]
    available_fields = [name for name in source_fields if name in original.columns]
    if not available_fields:
        raise ValueError("original eICU timing metadata has none of the expected annotation fields")
    right = original[identity + available_fields].rename(
        columns={name: f"original_eicu_{name}" for name in available_fields}
    )
    joined = frame.merge(right, on=identity, how="outer", validate="one_to_one", indicator="original_timing_merge")
    left_only = int(joined["original_timing_merge"].eq("left_only").sum())
    right_only = int(joined["original_timing_merge"].eq("right_only").sum())
    if left_only or right_only or len(joined) != len(frame) or len(joined) != len(original):
        raise ValueError(
            "original/repaired eICU timing metadata strict identity mismatch: "
            f"repaired_only={left_only}, original_only={right_only}, "
            f"repaired_rows={len(frame)}, original_rows={len(original)}"
        )
    current_y = pd.to_numeric(joined["y_mort"], errors="coerce").to_numpy(dtype=float)
    original_label = next((name for name in ("original_eicu_y_mort", "original_eicu_source_y_mort")
                           if name in joined), None)
    labels_match = None
    if original_label is not None:
        original_y = pd.to_numeric(joined[original_label], errors="coerce").to_numpy(dtype=float)
        labels_match = bool(np.array_equal(current_y, original_y, equal_nan=True))
        if not labels_match:
            raise ValueError("original/repaired eICU timing metadata labels differ; no scoring metadata was attached")
    ORIGINAL_EICU_TIMING_PROVENANCE = {
        "status": "strict_identity_join_passed",
        "path": str(path),
        "original_eligibility_manifest": str(_transport.ORIGINAL_ELIGIBILITY_GATE / "eligibility_manifest.json"),
        "n_identity_rows": int(len(joined)),
        "n_original_only": right_only,
        "n_repaired_only": left_only,
        "label_identity_match": labels_match,
        "original_annotation_columns": available_fields,
        "row_and_split_policy": "mapped by db/stay_id/subject; differences are recorded downstream, not treated as identity failure",
    }
    return joined.drop(columns="original_timing_merge")


def read_sequence(db: str, full: bool = True) -> tuple[dict[str, np.ndarray], np.ndarray]:
    seq_path = SEQ_PATHS[db]
    split_path = SPLIT_PATHS[db]
    if not seq_path.is_file() or not split_path.is_file():
        raise FileNotFoundError(f"sequence/split missing for {db}: {seq_path} / {split_path}")
    required = ["stay_id", "subject", "y_mort", "L", "H"]
    if full:
        required.extend(["V", "B"])
    with np.load(seq_path, allow_pickle=False) as archive:
        missing = set(required).difference(archive.files)
        if missing:
            raise ValueError(f"{db} sequence arrays missing {sorted(missing)}")
        ds: dict[str, np.ndarray] = {}
        for name in required:
            ds[name] = np.asarray(archive[name])
            log(f"{db} sequence_array_loaded name={name} shape={ds[name].shape} dtype={ds[name].dtype}")
    split = np.asarray(np.load(split_path, allow_pickle=False), dtype=np.int8)
    n = len(ds["L"])
    if len(split) != n or any(len(ds[name]) != n for name in ("stay_id", "subject", "y_mort")):
        raise ValueError(f"{db} sequence/split row alignment failed")
    if len(np.unique(np.asarray([canonical(v) for v in ds["stay_id"]], dtype=str))) != n:
        raise ValueError(f"{db} sequence stay_id is not unique")
    return ds, split


def sequence_index(db: str, ds: dict[str, np.ndarray], split: np.ndarray) -> pd.DataFrame:
    ix = pd.DataFrame({
        "db": COHORTS[db],
        "stay_id": [canonical(x) for x in ds["stay_id"]],
        "subject": [canonical(x) for x in ds["subject"]],
        "rows": np.arange(len(ds["L"]), dtype=np.int64),
        "split_expected": split.astype(np.int8),
        "y_seq": pd.to_numeric(pd.Series(ds["y_mort"]), errors="coerce"),
        "L": ds["L"].astype(np.int16),
    })
    if ix[["stay_id", "subject"]].isna().any().any() or ix.duplicated(["stay_id", "subject"]).any():
        raise ValueError(f"{db} sequence identity key is missing or duplicated")
    return ix


def prediction_frames() -> dict[str, pd.DataFrame]:
    global PREDICTION_PROVENANCE
    original_fm, _ = load_checked(
        _transport.ORIGINAL_FM_RUN / "main20k_real.parquet",
        _transport.ORIGINAL_ELIGIBILITY_GATE,
        model_tag="main20k", analysis_id="main20k_real", input_condition="real",
        endpoint="hospital", prediction_manifest_path=_transport.ORIGINAL_FM_MANIFEST,
    )
    if "mimic" not in original_fm:
        raise ValueError("verified original FM source predictions do not contain MIMIC")
    original_fm_mimic = original_fm["mimic"]
    del original_fm

    _, source_baselines, _, _ = _source_transport._load_source_frames()
    source_mimic_frames = {
        "lr": source_baselines["conventional_lr_value_plus_pattern"]["mimic"],
        "hgb": source_baselines["conventional_hgb_value_plus_pattern"]["mimic"],
    }
    del source_baselines
    target_frames, target_gates, endpoint_by_db, target_provenance = _transport._load_all_model_frames()
    model_adapters = {
        "fm": "main20k_real",
        "lr": "conventional_lr_value_plus_pattern",
        "hgb": "conventional_hgb_value_plus_pattern",
    }
    selected: dict[str, pd.DataFrame] = {}
    counts: dict[str, dict[str, int]] = {}
    required = ["db", "stay_id", "subject", "rows", "split", "y", "p"]
    for model, adapter in model_adapters.items():
        pieces = []
        if model == "fm":
            pieces.append(original_fm_mimic)
        else:
            pieces.append(source_mimic_frames[model])
        target_model = adapter
        for db, cohort_name in COHORTS.items():
            if db == "mimic":
                continue
            if cohort_name not in target_frames[target_model]:
                raise ValueError(f"verified transport artifact omits {model}/{cohort_name}")
            pieces.append(target_frames[target_model][cohort_name])
        frame = pd.concat([part[required].copy() for part in pieces], ignore_index=True)
        frame = keys(frame)
        frame["rows"] = pd.to_numeric(frame["rows"], errors="raise").astype(np.int64)
        frame["split"] = pd.to_numeric(frame["split"], errors="raise").astype(np.int8)
        frame["y"] = pd.to_numeric(frame["y"], errors="raise").astype(np.int8)
        frame["p"] = pd.to_numeric(frame["p"], errors="raise").astype(float)
        if (~np.isfinite(frame["p"])).any() or ((frame["p"] < 0) | (frame["p"] > 1)).any():
            raise ValueError(f"{model} predictions outside finite probability range")
        counts[model] = {str(db): int(frame["db"].eq(cohort).sum()) for db, cohort in COHORTS.items()}
        selected[model] = frame
        log(f"validated_prediction_adapter model={model} counts={counts[model]}")

    PREDICTION_PROVENANCE = {
        "status": "transport_validated_source_and_recovered_target_artifacts",
        "source_fm_manifest": str(_transport.ORIGINAL_FM_MANIFEST),
        "source_fm_manifest_sha256": sha256(_transport.ORIGINAL_FM_MANIFEST),
        "source_baseline_manifest": str(_source_transport.SOURCE_PREDICTION_MANIFEST),
        "source_baseline_manifest_sha256": sha256(_source_transport.SOURCE_PREDICTION_MANIFEST),
        "source_reproduction_manifest": str(_source_transport.SOURCE_MANIFEST),
        "source_reproduction_manifest_sha256": sha256(_source_transport.SOURCE_MANIFEST),
        "target_transport": target_provenance,
        "target_endpoint_by_db": endpoint_by_db,
        "target_gate_cohorts": sorted(target_gates),
        "validated_counts_by_model_and_cohort": counts,
        "MIMIC": "original FM and original source baseline predictions; no recovered target repair applied",
        "external": "repaired recovered predictions validated by transport._load_all_model_frames; mismatched artifacts are never intersected",
    }
    return selected


def bind_predictions(db: str, eligibility: pd.DataFrame, seq_index: pd.DataFrame,
                    predictions: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    expected = eligibility.merge(seq_index, on=["db", "stay_id", "subject"], how="left",
                                 validate="one_to_one", indicator="seq_merge")
    if not expected["seq_merge"].eq("both").all():
        raise ValueError(f"{db} eligibility does not cover sequence identity")
    if not np.array_equal(expected["y_mort"].to_numpy(dtype=float), expected["y_seq"].to_numpy(dtype=float), equal_nan=True):
        raise ValueError(f"{db} eligibility/source sequence labels mismatch")
    if db == "eicu":
        original_rows = next((name for name in ("original_eicu_rows",) if name in expected), None)
        original_split = next((name for name in ("original_eicu_split", "original_eicu_split_expected")
                               if name in expected), None)
        row_map = {"available": original_rows is not None, "changed": None, "unavailable": len(expected)}
        split_map = {"available": original_split is not None, "changed": None, "unavailable": len(expected)}
        if original_rows is not None:
            old_rows = pd.to_numeric(expected[original_rows], errors="coerce")
            current_rows = pd.to_numeric(expected["rows"], errors="coerce")
            valid = old_rows.notna() & current_rows.notna()
            row_map = {"available": True, "changed": int((valid & old_rows.ne(current_rows)).sum()),
                       "unavailable": int((~valid).sum())}
        if original_split is not None:
            old_split = pd.to_numeric(expected[original_split], errors="coerce")
            current_split = pd.to_numeric(expected["split_expected"], errors="coerce")
            valid = old_split.notna() & current_split.notna()
            split_map = {"available": True, "changed": int((valid & old_split.ne(current_split)).sum()),
                         "unavailable": int((~valid).sum())}
        ORIGINAL_EICU_TIMING_PROVENANCE.update({
            "original_to_repaired_row_mapping": row_map,
            "original_to_repaired_split_mapping": split_map,
            "mapping_identity": "db/stay_id/subject",
        })
    for model, all_rows in predictions.items():
        pred = all_rows.loc[all_rows["db"].eq(COHORTS[db])].copy()
        joined = pred.merge(seq_index, on=["db", "stay_id", "subject", "rows"], how="left",
                            validate="one_to_one", indicator="seq_merge")
        if not joined["seq_merge"].eq("both").all():
            raise ValueError(f"{db}/{model}: prediction row indices do not bind to sequence keys")
        if not np.array_equal(joined["split"].to_numpy(), joined["split_expected"].to_numpy()):
            raise ValueError(f"{db}/{model}: prediction split differs from frozen split array")
        if not np.array_equal(joined["y"].to_numpy(), joined["y_seq"].to_numpy(dtype=np.int8)):
            raise ValueError(f"{db}/{model}: prediction label differs from sequence label")
        meta = eligibility[["db", "stay_id", "subject", "y_mort", "eligible_24h"]].copy()
        if "eligible_24h" not in meta:
            raise ValueError(f"{db}: reviewed eligibility lacks eligible_24h")
        joined = joined.drop(columns=["seq_merge"]).merge(meta, on=["db", "stay_id", "subject"],
                                                            how="left", validate="one_to_one", indicator="elig_merge")
        if not joined["elig_merge"].eq("both").all():
            raise ValueError(f"{db}/{model}: prediction row has no reviewed eligibility row")
        if not np.array_equal(joined["y"].to_numpy(dtype=float), joined["y_mort"].to_numpy(dtype=float), equal_nan=True):
            raise ValueError(f"{db}/{model}: prediction labels differ from reviewed eligibility")
        out[model] = joined.drop(columns="elig_merge")
    for model in ("fm", "lr", "hgb"):
        if model not in out:
            raise ValueError(f"{db}: no {model} predictions")
    keysets = {model: set(zip(frame.db, frame.stay_id, frame.subject, frame.rows, frame.split, frame.y))
               for model, frame in out.items()}
    common = set.intersection(*keysets.values())
    if any(rows != common for rows in keysets.values()):
        raise ValueError(f"{db}: model prediction artifacts do not have the same bound rows/labels")
    return out


def bool_flag(frame: pd.DataFrame, candidates: tuple[str, ...]) -> tuple[pd.Series, str | None]:
    for name in candidates:
        if name in frame:
            series = frame[name]
            if pd.api.types.is_bool_dtype(series.dtype):
                return series.astype("boolean"), name
            low = series.astype("string").str.strip().str.lower()
            mapping = {"true": True, "1": True, "yes": True,
                       "false": False, "0": False, "no": False}
            unknown = set(low.dropna().unique()).difference(mapping)
            if unknown:
                raise ValueError(f"{name} has unknown boolean strings: {sorted(unknown)}")
            return low.map(mapping).astype("boolean"), name
    return pd.Series(pd.NA, index=frame.index, dtype="boolean"), None


def finite_exact_death(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    exact_times: list[pd.Series] = []
    for prefix in ("", "original_eicu_"):
        for scope in ("hospital", "icu"):
            time_col = f"{prefix}{scope}_death_time_hr"
            kind_col = f"{prefix}{scope}_death_time_kind"
            if time_col not in frame or kind_col not in frame:
                continue
            values = pd.to_numeric(frame[time_col], errors="coerce")
            kind = frame[kind_col].astype("string").str.strip().str.lower()
            is_exact_kind = kind.eq("exact_deathtime")
            finite = pd.Series(np.isfinite(values.to_numpy(dtype=float)), index=frame.index)
            usable = is_exact_kind & finite
            exact_times.append(values.where(usable))
    if not exact_times:
        # This predicate means usable exact evidence exists, not that survival
        # is known. A known date overlap without exact evidence is unresolved.
        return pd.Series(False, index=frame.index, dtype="boolean"), pd.Series(np.nan, index=frame.index, dtype=float)
    stack = pd.concat(exact_times, axis=1)
    exact_available = stack.notna().any(axis=1).astype("boolean")
    exact_hr = stack.min(axis=1, skipna=True)
    # A recorded exact timestamp establishes its relation to H24; otherwise the
    # death-time evidence is unavailable and remains nullable.
    return exact_available, exact_hr


def calibration_summary(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    out: dict[str, Any] = {
        "calibration_fixed_slope_intercept": None,
        "calibration_fixed_slope_status": "unavailable",
        "calibration_fixed_slope_unavailable_reason": None,
        "calibration_joint_intercept": None,
        "calibration_joint_slope": None,
        "calibration_joint_status": "unavailable",
        "calibration_joint_unavailable_reason": None,
    }
    if len(y) == 0:
        out["calibration_fixed_slope_unavailable_reason"] = "empty_subset"
        out["calibration_joint_unavailable_reason"] = "empty_subset"
        return out
    if len(np.unique(y)) < 2:
        out["calibration_fixed_slope_unavailable_reason"] = "single_class_outcome"
        out["calibration_joint_unavailable_reason"] = "single_class_outcome"
        return out

    x = np.log(p / (1 - p))
    prevalence = float(np.mean(y))
    try:
        offset = brentq(lambda b: float(np.mean(expit(x + b)) - prevalence), -40.0, 40.0)
        out["calibration_fixed_slope_intercept"] = float(offset)
        out["calibration_fixed_slope_status"] = "fit"
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        out["calibration_fixed_slope_unavailable_reason"] = f"offset_solver_failed:{type(exc).__name__}"

    if np.ptp(x) <= 1e-12:
        out["calibration_joint_unavailable_reason"] = "no_variation_in_model_logit_probability"
        return out

    initial = np.array([math.log(prevalence / (1 - prevalence)), 1.0], dtype=float)

    def joint_objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = beta[0] + beta[1] * x
        prob = expit(eta)
        loss = float(np.sum(np.logaddexp(0.0, eta) - y * eta))
        gradient = np.array([np.sum(prob - y), np.sum((prob - y) * x)], dtype=float)
        return loss, gradient

    try:
        fit = minimize(joint_objective, initial, jac=True, method="BFGS", options={"maxiter": 500, "gtol": 1e-7})
        if fit.success and np.isfinite(fit.x).all():
            out["calibration_joint_intercept"] = float(fit.x[0])
            out["calibration_joint_slope"] = float(fit.x[1])
            out["calibration_joint_status"] = "fit"
        else:
            out["calibration_joint_unavailable_reason"] = f"optimizer_failed:{str(fit.message)}"
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        out["calibration_joint_unavailable_reason"] = f"optimizer_failed:{type(exc).__name__}"
    return out


def metric_summary(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    out: dict[str, Any] = {
        "n_stays": int(len(y)),
        "n_events": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else None,
        "brier": float(np.mean((p - y) ** 2)) if len(y) else None,
        "logloss": float(log_loss(y, p, labels=[0, 1])) if len(y) else None,
        "auroc": "NA",
        "auprc": "NA",
        "auroc_unavailable_reason": "empty_subset" if len(y) == 0 else "single_class_outcome",
        "auprc_unavailable_reason": "empty_subset" if len(y) == 0 else "single_class_outcome",
    }
    if len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
        out["auroc_unavailable_reason"] = None
        out["auprc_unavailable_reason"] = None
    out.update(calibration_summary(y, p))
    return out


def paired_cluster_loss_rows(subject: np.ndarray, y: np.ndarray, p_fm: np.ndarray,
                             p_lr: np.ndarray, draws: int, seed: int) -> list[dict[str, Any]]:
    y = np.asarray(y, dtype=np.int8)
    p_fm = np.clip(np.asarray(p_fm, dtype=float), EPS, 1 - EPS)
    p_lr = np.clip(np.asarray(p_lr, dtype=float), EPS, 1 - EPS)
    brier_diff = (p_fm - y) ** 2 - (p_lr - y) ** 2
    logloss_fm = -y * np.log(p_fm) - (1 - y) * np.log1p(-p_fm)
    logloss_lr = -y * np.log(p_lr) - (1 - y) * np.log1p(-p_lr)
    loss_differences = np.column_stack((brier_diff, logloss_fm - logloss_lr))
    results: list[dict[str, Any]] = []
    for patient_weighted, estimand in ((False, "stay_weighted_primary"),
                                      (True, "patient_weighted_sensitivity")):
        log(f"paired_cluster_ci_begin estimand={estimand} seed={seed} draws={draws} stays={len(y)}")
        try:
            summary = paired_cluster_mean_ci(subject, loss_differences, draws=draws,
                                             seed=seed, patient_weighted=patient_weighted)
            points = np.asarray(summary["point"], dtype=float)
            lows = np.asarray(summary["low"], dtype=float)
            highs = np.asarray(summary["high"], dtype=float)
            n_patients = int(summary["n_patients"])
            failure_reason = None
        except ValueError as exc:
            points = lows = highs = np.full(2, np.nan)
            n_patients = int(pd.Series(subject).nunique())
            failure_reason = str(exc)
        log(f"paired_cluster_ci_complete estimand={estimand} seed={seed} patients={n_patients}")
        for idx, metric in enumerate(("brier", "logloss")):
            results.append({
                "model": "paired_fm_minus_lr", "paired_metric": metric,
                "estimand": estimand, "fm_minus_lr": float(points[idx]) if np.isfinite(points[idx]) else None,
                "low95": float(lows[idx]) if np.isfinite(lows[idx]) else None,
                "high95": float(highs[idx]) if np.isfinite(highs[idx]) else None,
                "n_patients": n_patients, "n_stays": int(len(y)), "draws": draws,
                "seed": seed, "weighting": "patient" if patient_weighted else "stay",
                "unavailable_reason": failure_reason,
            })
    return results


def h24_flags(frame: pd.DataFrame) -> tuple[dict[str, pd.Series], dict[str, str | None]]:
    overlap, c_overlap = bool_flag(frame, (
        "source_patient_dod_interval_overlaps_24h",
        "original_eicu_source_patient_dod_interval_overlaps_24h",
        "original_eicu_dod_day_overlaps_landmark", "dod_day_overlaps_landmark",
    ))
    exact_available, exact_hr = finite_exact_death(frame)
    date_uncertain = overlap & ~exact_available
    exact_by24 = exact_hr.le(24).where(exact_available, pd.NA).astype("boolean")
    exact_resolves_after24 = (overlap & exact_available & exact_hr.gt(24)).astype("boolean")
    gap, c_gap = bool_flag(frame, (
        "source_postlandmark_ordering_issue", "source_icu_exit_after_hospital_exit_after_24",
        "original_eicu_source_postlandmark_ordering_issue",
        "original_eicu_source_icu_exit_after_hospital_exit_after_24",
        "original_eicu_icu_exit_after_hospital_exit_after24",
        "original_eicu_icu_exit_after_hospital_exit_after_24",
        "icu_exit_after_hospital_exit_after24", "icu_exit_after_hospital_exit_after_24",
    ))
    admin, c_admin = bool_flag(frame, (
        "source_hospital_admit_after_24h", "source_hospital_admission_after_landmark",
        "original_eicu_source_hospital_admit_after_24h",
        "original_eicu_hospital_admission_after_landmark", "hospital_admission_after_landmark",
    ))
    landmark, c_landmark = bool_flag(frame, (
        "source_landmark_issue", "source_landmark_contradiction",
        "original_eicu_source_landmark_issue", "original_eicu_source_landmark_contradiction",
    ))
    flags = {
        "date_only_h24_overlap": overlap,
        "date_only_h24_uncertain_after_exact_resolution": date_uncertain,
        "exact_death_by_h24": exact_by24,
        "exact_death_resolves_date_overlap_after_h24": exact_resolves_after24,
        "later_exit_order_gap": gap,
        "administrative_hospital_admit_after_h24": admin,
        "source_landmark_issue": landmark,
    }
    cols = {"date_only_h24_overlap": c_overlap, "later_exit_order_gap": c_gap,
            "administrative_hospital_admit_after_h24": c_admin, "source_landmark_issue": c_landmark}
    return flags, cols


def analyze_predictions(db: str, eligibility: pd.DataFrame,
                       bound: dict[str, pd.DataFrame], split_index: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    merged = bound["fm"].copy()
    flag_map, flag_cols = h24_flags(eligibility)
    flag_frame = eligibility[["db", "stay_id", "subject"]].copy()
    for name, series in flag_map.items():
        flag_frame[name] = series.array
    merged = merged.merge(flag_frame, on=["db", "stay_id", "subject"], validate="one_to_one")
    flag_def = {
        "primary_reviewed_h24": pd.Series(True, index=merged.index, dtype="boolean"),
        "no_unresolved_date_only_h24_overlap": ~merged["date_only_h24_uncertain_after_exact_resolution"],
        "no_later_exit_order_gap": ~merged["later_exit_order_gap"],
        "both_restrictions": (~merged["date_only_h24_uncertain_after_exact_resolution"] & ~merged["later_exit_order_gap"]),
        # Remove proven early deaths; unavailable exact timing is retained as
        # uncertain, never interpreted as documented survival.
        "no_exact_death_by_h24": ~merged["exact_death_by_h24"].fillna(False),
        "no_admin_after_h24": ~merged["administrative_hospital_admit_after_h24"],
        "flagged_date_overlap": merged["date_only_h24_overlap"],
        "flagged_exit_gap": merged["later_exit_order_gap"],
        "flagged_admin_after_h24": merged["administrative_hospital_admit_after_h24"],
    }
    for name, mask in flag_def.items():
        merged[name] = mask.astype("boolean").array
    flow_rows: list[dict[str, Any]] = []
    flag_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    audit = eligibility[["db", "stay_id", "subject", "eligible_24h", "y_mort"]].merge(
        split_index[["db", "stay_id", "subject", "split_expected"]],
        on=["db", "stay_id", "subject"], validate="one_to_one")
    for name, series in flag_map.items():
        audit[name] = series.array
    base_eligible = audit["eligible_24h"].fillna(False).astype(bool)
    for split in (0, 1, 2):
        audit_masks = {
            "primary_reviewed_h24": pd.Series(True, index=audit.index, dtype="boolean"),
            "no_unresolved_date_only_h24_overlap": ~audit["date_only_h24_uncertain_after_exact_resolution"],
            "no_later_exit_order_gap": ~audit["later_exit_order_gap"],
            "both_restrictions": (~audit["date_only_h24_uncertain_after_exact_resolution"] & ~audit["later_exit_order_gap"]),
            "no_exact_death_by_h24": ~audit["exact_death_by_h24"].fillna(False),
            "no_admin_after_h24": ~audit["administrative_hospital_admit_after_h24"],
            "flagged_date_overlap": audit["date_only_h24_overlap"],
            "flagged_exit_gap": audit["later_exit_order_gap"],
            "flagged_admin_after_h24": audit["administrative_hospital_admit_after_h24"],
        }
        for label, mask in audit_masks.items():
            selected = base_eligible & audit["split_expected"].eq(split) & mask.fillna(False).astype(bool)
            flow_rows.append({"db": db, "split": split, "subset": label, "n_stays": int(selected.sum()),
                              "n_patients": int(audit.loc[selected, "subject"].nunique()),
                              "n_events": int(audit.loc[selected, "y_mort"].eq(1).sum()),
                              "split_interpretation": (
                                  "frozen source partition; chronological ordering is not asserted"
                                  if db == "mimic" else "frozen external partition; not temporal validation"
                              ),
                              "flow_basis": "all reviewed eligibility rows and frozen split array"})
    for flag_name, flag in flag_map.items():
        for split in (0, 1, 2):
            row = base_eligible & audit["split_expected"].eq(split)
            flag_rows.append({"db": db, "split": split, "flag": flag_name, "available_source_column": flag_cols.get(flag_name),
                              "n_eligible": int(row.sum()), "n_flagged": int((row & audit[flag_name].fillna(False)).sum()),
                              "n_unflagged": int((row & audit[flag_name].eq(False).fillna(False)).sum()),
                              "n_unknown": int((row & audit[flag_name].isna()).sum()),
                              "flow_basis": "all reviewed eligibility rows and frozen split array"})
    # Flag intersections are computed before any performance summaries.
    for split in (0, 1, 2):
        row = base_eligible & audit["split_expected"].eq(split)
        for a, b, label in (("date_only_h24_uncertain_after_exact_resolution", "later_exit_order_gap", "date_uncertain_x_exit_gap"),
                            ("date_only_h24_overlap", "later_exit_order_gap", "any_date_overlap_x_exit_gap"),
                            ("administrative_hospital_admit_after_h24", "later_exit_order_gap", "admin_after24_x_exit_gap")):
            both = (audit[a] & audit[b]).astype("boolean")
            flag_rows.append({"db": db, "split": split, "flag": label, "available_source_column": None,
                              "n_eligible": int(row.sum()), "n_flagged": int((row & both.fillna(False)).sum()),
                              "n_unflagged": int((row & both.eq(False).fillna(False)).sum()),
                              "n_unknown": int((row & both.isna()).sum()),
                              "flow_basis": "all reviewed eligibility rows and frozen split array"})
    for model, model_frame in bound.items():
        keyed_p = model_frame[["db", "stay_id", "subject", "rows", "split", "y", "p"]].rename(columns={"p": "p_model"})
        scored = merged[["db", "stay_id", "subject", "rows", "split", "y", "eligible_24h", *flag_def.keys()]].merge(
            keyed_p, on=["db", "stay_id", "subject", "rows", "split", "y"], validate="one_to_one")
        for subset, mask in flag_def.items():
            scopes = [(2, "split2_source")] if db == "mimic" else [(None, "external_full_reviewed_roster")]
            for split, split_scope in scopes:
                selected = scored["eligible_24h"].fillna(False).astype(bool) & mask.fillna(False).astype(bool)
                if split is not None:
                    selected &= scored["split"].eq(split)
                met = metric_summary(scored.loc[selected, "y"].to_numpy(dtype=np.int8), scored.loc[selected, "p_model"].to_numpy(dtype=float))
                metric_rows.append({"db": db, "model": model, "split_scope": split_scope,
                                    "split": split if split is not None else "all", "subset": subset,
                                    "split_interpretation": (
                                        "frozen source test partition; chronological ordering is not asserted"
                                        if db == "mimic" else "full reviewed external roster; no temporal validation claim"
                                    ), **met})
    # Paired patient-cluster bootstrap for FM versus value+pattern LR on the same rows.
    pair_keys = ["db", "stay_id", "subject", "rows", "split", "y"]
    paired = bound["fm"][pair_keys + ["p"]].rename(columns={"p": "p_fm"}).merge(
        bound["lr"][pair_keys + ["p"]].rename(columns={"p": "p_lr"}),
        on=pair_keys, how="inner", validate="one_to_one")
    if len(paired) != len(bound["fm"]) or len(paired) != len(bound["lr"]):
        raise ValueError(f"{db}: FM/LR paired key coverage differs")
    paired = paired.merge(merged[["db", "stay_id", "subject", "rows", "split", "y", "eligible_24h", *flag_def.keys()]],
                          on=["db", "stay_id", "subject", "rows", "split", "y"], validate="one_to_one")
    paired_ci_subsets = {
        "primary_reviewed_h24", "no_unresolved_date_only_h24_overlap", "no_later_exit_order_gap",
        "both_restrictions", "no_exact_death_by_h24", "no_admin_after_h24",
    }
    for subset, mask in flag_def.items():
        if subset not in paired_ci_subsets:
            continue
        selected = paired["eligible_24h"].fillna(False).astype(bool) & mask.fillna(False).astype(bool)
        if db == "mimic":
            selected &= paired["split"].eq(2)
        if int(selected.sum()) == 0:
            continue
        paired_rows = paired_cluster_loss_rows(
            paired.loc[selected, "subject"].to_numpy(),
            paired.loc[selected, "y"].to_numpy(dtype=np.int8),
            paired.loc[selected, "p_fm"].to_numpy(dtype=float),
            paired.loc[selected, "p_lr"].to_numpy(dtype=float),
            BOOT_DRAWS, SEED + len(metric_rows),
        )
        metric_rows.extend({
            "db": db, "split_scope": "split2_source" if db == "mimic" else "external_full_reviewed_roster",
            "split": 2 if db == "mimic" else "all", "subset": subset,
            "split_interpretation": (
                "frozen source test partition; chronological ordering is not asserted"
                if db == "mimic" else "full reviewed external roster; no temporal validation claim"
            ),
            **row,
        } for row in paired_rows)
    return flow_rows, flag_rows, metric_rows


def missing_summary(series: pd.Series) -> dict[str, Any]:
    missing = series.isna() | series.astype("string").str.strip().isin({"", "nan", "None", "<NA>"})
    return {"n": int(len(series)), "n_missing": int(missing.sum()), "missing_fraction": float(missing.mean()) if len(series) else None}


def sequence_support(db: str, eligibility: pd.DataFrame, ds: dict[str, np.ndarray], split: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    log(f"{db} sequence_support_begin")
    n = len(ds["L"])
    L = np.asarray(ds["L"], dtype=np.int64)
    H = np.asarray(ds["H"])
    V = np.asarray(ds["V"])
    if V.shape != H.shape or V.shape[0] != n or V.shape[1] != 128:
        raise ValueError(f"{db} V/H shapes must be n x 128")
    if np.any((L < 0) | (L > 128)):
        raise ValueError(f"{db} sequence L outside [0,128]")
    active = np.arange(V.shape[1])[None, :] < L[:, None]
    valid_hour = np.isfinite(H) & (H >= 0) & (H < 24)
    h24_rows = np.all(valid_hour | ~active, axis=1)
    h24_ok = bool(h24_rows.all())
    h_active = np.asarray(H[active], dtype=float)
    subject = np.asarray([canonical(x) for x in ds["subject"]], dtype=str)
    rows = [
        {"db": db, "support": "sequence_all", "n_stays": n, "n_patients": int(len(np.unique(subject))),
         "n_frozen_label_positive": int(np.nansum(np.asarray(ds["y_mort"]) == 1)), "input_event_count": int(L.sum()),
         "empty_sequences": int((L == 0).sum()), "cap128_reached": int((L == 128).sum()),
         "cap128_fraction": float(np.mean(L == 128)), "mean_tokens": float(L.mean()),
         "median_tokens": float(np.median(L)), "p95_tokens": float(np.quantile(L, .95)),
         "input_hours_all_in_0_24": h24_ok, "active_tokens_negative_hour": int(np.sum(h_active < 0)),
         "active_tokens_hour_ge24": int(np.sum(h_active >= 24))},
    ]
    elig_keys = eligibility[["db", "stay_id", "subject", "eligible_24h", "y_mort"]].copy()
    seq_idx = pd.DataFrame({"db": COHORTS[db], "stay_id": [canonical(x) for x in ds["stay_id"]],
                            "subject": subject, "rows": np.arange(n, dtype=np.int64), "split": split, "L": L})
    eligible = elig_keys.merge(seq_idx, on=["db", "stay_id", "subject"], how="left", validate="one_to_one")
    for split_id in (0, 1, 2):
        block = eligible.loc[eligible["split"].eq(split_id)]
        selected = block["eligible_24h"].fillna(False).astype(bool)
        rows.append({"db": db, "support": f"split{split_id}_all", "n_stays": int(len(block)),
                     "n_patients": int(block["subject"].nunique()), "n_frozen_label_positive": None,
                     "eligible_24h_stays": int(selected.sum()),
                     "eligible_24h_patients": int(block.loc[selected, "subject"].nunique()),
                      "eligible_24h_frozen_label_positive": int(pd.to_numeric(block.loc[selected, "y_mort"], errors="coerce").eq(1).sum()),
                     "input_event_count": int(block["L"].sum()), "empty_sequences": int(block["L"].eq(0).sum()),
                     "cap128_reached": int(block["L"].eq(128).sum()),
                     "cap128_fraction": float(block["L"].eq(128).mean()) if len(block) else None,
                     "mean_tokens": float(block["L"].mean()) if len(block) else None,
                     "median_tokens": float(block["L"].median()) if len(block) else None,
                     "p95_tokens": float(block["L"].quantile(.95)) if len(block) else None,
                     "input_hours_all_in_0_24": h24_ok, "active_tokens_negative_hour": None,
                     "active_tokens_hour_ge24": None})
    vocab = json.loads(VOCAB_PATHS[db].read_text(encoding="utf-8"))
    names = [str(x) for x in vocab.get("vars", [])]
    nvars = len(names)
    if nvars < 1:
        raise ValueError(f"{db} vocabulary has no vars")
    counts = np.zeros(nvars + 1, dtype=np.int64)
    stay_count = np.zeros(nvars + 1, dtype=np.int64)
    for start in range(0, n, 4096):
        end = min(n, start + 4096)
        vv = np.asarray(V[start:end], dtype=np.int64)
        ll = L[start:end]
        pos = np.arange(vv.shape[1])[None, :] < ll[:, None]
        local_row, position = np.nonzero(pos & (vv > 0))
        vals = vv[local_row, position]
        if len(vals):
            if vals.min() < 1 or vals.max() > nvars:
                raise ValueError(f"{db} variable token ID outside vocabulary range")
            counts += np.bincount(vals, minlength=nvars + 1)[: nvars + 1]
            row_var = local_row.astype(np.int64) * (nvars + 1) + vals
            row_var = np.unique(row_var)
            unique_var = row_var % (nvars + 1)
            stay_count += np.bincount(unique_var, minlength=nvars + 1)[: nvars + 1]
        if (start // 4096) % 8 == 7 or end == n:
            log(f"{db} variable_coverage_progress rows={end}/{n}")
    if counts.shape[0] != nvars + 1:
        raise ValueError(f"{db} variable count array has an unexpected length")
    variable_rows = [{"db": db, "variable_id": int(i + 1), "variable": names[i],
                      "stays_with_variable": int(stay_count[i + 1]),
                      "fraction_of_stays": float(stay_count[i + 1] / n) if n else None,
                      "event_tokens": int(counts[i + 1])}
                     for i in range(nvars)]
    input_checks = [{"db": db, "audit": "sequence_token_hour_bounds",
                     "scope": "exported active sequence arrays after their existing truncation step",
                     "late_store_availability_comparison": "not measured by this hour-range check",
                     "h24_window_verified_across_all_tokens": h24_ok,
                     "active_token_count": int(L.sum()), "active_hours_min": float(np.min(h_active)) if len(h_active) else None,
                     "active_hours_max": float(np.max(h_active)) if len(h_active) else None,
                     "negative_hour_tokens": int(np.sum(h_active < 0)),
                     "hour_24plus_tokens": int(np.sum(h_active >= 24))}]
    clock_row: dict[str, Any] = {"db": db, "audit": "reviewed_input_clock_flags", "n_rows": int(len(eligibility))}
    for col in ("input_window_le_24h_verified", "clinical_input_availability_verified"):
        if col in eligibility:
            flag = parse_bool(eligibility[col], f"{db}.{col}")
            clock_row[f"{col}_true"] = int(flag.sum())
            clock_row[f"{col}_false"] = int((~flag).sum())
            clock_row[f"{col}_available"] = True
        else:
            clock_row[f"{col}_available"] = False
    for col in ("input_event_count_24h", "input_max_hour_24h", "input_availability_cutoff_hr", "input_window_end_hr"):
        if col in eligibility:
            v = pd.to_numeric(eligibility[col], errors="coerce")
            clock_row[f"{col}_n_observed"] = int(v.notna().sum())
            clock_row[f"{col}_median"] = float(v.median()) if v.notna().any() else None
            clock_row[f"{col}_p95"] = float(v.quantile(.95)) if v.notna().any() else None
            clock_row[f"{col}_max"] = float(v.max()) if v.notna().any() else None
    input_checks.append(clock_row)
    return rows, variable_rows, input_checks


def clinical_support(db: str, eligibility: pd.DataFrame, split_index: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    frame = eligibility.merge(split_index[["db", "stay_id", "subject", "split_expected"]],
                              on=["db", "stay_id", "subject"], validate="one_to_one")
    data_rows: list[dict[str, Any]] = []
    cat_rows: list[dict[str, Any]] = []
    time_rows: list[dict[str, Any]] = []
    n = len(frame)
    data_rows.append({"db": db, "scope": "cohort_all", "n_stays": n, "n_patients": int(frame["subject"].nunique()),
                      "repeat_patients": int((frame.groupby("subject").size() > 1).sum()),
                      "repeat_stays_after_first": int((frame.groupby("subject").size() - 1).clip(lower=0).sum()),
                      "eligible_24h_stays": int(frame["eligible_24h"].fillna(False).astype(bool).sum()),
                      "frozen_label_positive": int(pd.to_numeric(frame["y_mort"], errors="coerce").eq(1).sum()),
                      "frozen_label_unknown": int(pd.to_numeric(frame["y_mort"], errors="coerce").isna().sum()),
                      "hospital_sites": int(frame["hospitalid"].nunique(dropna=True)) if "hospitalid" in frame else None})
    for col in ("age", "age_precision", "sex", "gender", "hospitalid", "source_y_mort", "icu_exit_time", "hospital_exit_time"):
        if col not in frame:
            continue
        summary = missing_summary(frame[col])
        data_rows.append({"db": db, "scope": f"field:{col}", "n_stays": summary["n"], "n_patients": None,
                          "repeat_patients": None, "repeat_stays_after_first": None, "eligible_24h_stays": None,
                          "frozen_label_positive": None, "frozen_label_unknown": summary["n_missing"],
                          "hospital_sites": None, "field_missing_fraction": summary["missing_fraction"]})
        if col in {"sex", "gender", "age_precision", "source_y_mort"}:
            values = frame[col].astype("string").fillna("<MISSING>").value_counts(dropna=False)
            cat_rows.extend({"db": db, "field": col, "value": str(k), "n": int(v)} for k, v in values.items())
    age_col = "age" if "age" in frame else "anchor_age" if "anchor_age" in frame else None
    if age_col:
        age = pd.to_numeric(frame[age_col], errors="coerce")
        for scope, mask in (("all", pd.Series(True, index=frame.index)),
                            ("eligible_24h", frame["eligible_24h"].fillna(False).astype(bool))):
            vals = age.loc[mask].dropna().to_numpy(dtype=float)
            data_rows.append({"db": db, "scope": f"age_{scope}", "n_stays": int(mask.sum()),
                              "n_patients": int(frame.loc[mask, "subject"].nunique()), "repeat_patients": None,
                              "repeat_stays_after_first": None, "eligible_24h_stays": None, "frozen_label_positive": None,
                              "frozen_label_unknown": int(age.loc[mask].isna().sum()),
                              "hospital_sites": None, "age_n": int(len(vals)), "age_mean": float(vals.mean()) if len(vals) else None,
                              "age_median": float(np.median(vals)) if len(vals) else None,
                              "age_p25": float(np.quantile(vals, .25)) if len(vals) else None,
                              "age_p75": float(np.quantile(vals, .75)) if len(vals) else None,
                              "age_min": float(vals.min()) if len(vals) else None, "age_max": float(vals.max()) if len(vals) else None})
    severity_cols = [c for c in frame.columns if any(term in c.lower() for term in ("sofa", "apache", "saps", "oasis", "severity"))]
    if severity_cols:
        for col in severity_cols:
            vals = pd.to_numeric(frame[col], errors="coerce")
            data_rows.append({"db": db, "scope": f"severity:{col}", "n_stays": n, "n_patients": None,
                              "repeat_patients": None, "repeat_stays_after_first": None, "eligible_24h_stays": None,
                               "frozen_label_positive": None, "frozen_label_unknown": int(vals.isna().sum()),
                              "hospital_sites": None, "field_missing_fraction": float(vals.isna().mean()),
                              "severity_median": float(vals.median()) if vals.notna().any() else None,
                              "severity_interpretation": "available source field only; no unified score inferred"})
    else:
        data_rows.append({"db": db, "scope": "severity_fields", "n_stays": n, "n_patients": None,
                          "repeat_patients": None, "repeat_stays_after_first": None, "eligible_24h_stays": None,
                           "frozen_label_positive": None, "frozen_label_unknown": n,
                          "hospital_sites": None, "severity_available": False})
    for col in ("icu_start_time", "icu_exit_time", "hospital_exit_time", "hospital_death_time", "icu_death_time"):
        if col not in frame:
            continue
        dt = pd.to_datetime(frame[col], errors="coerce")
        years = dt.dt.year.dropna()
        if db == "mimic":
            time_interpretation = "de-identified calendar date values; not comparable as true cross-patient calendar time"
            date_basis_fields = [name for name in frame.columns
                                 if name.lower() in {"anchor_year_group", "source_period", "source_release_period"}]
            if date_basis_fields:
                date_evidence = ("reviewed eligibility fields available for date context: "
                                 + ", ".join(sorted(date_basis_fields))
                                 + "; calendar dates remain de-identified")
            else:
                date_evidence = "reviewed eligibility has no anchor_year_group or official source-period field; only de-identified date values are available"
            year_min = year_max = time_min = time_max = None
        else:
            time_interpretation = "source timestamp summary; no temporal-validation role inferred"
            date_evidence = "summarized from the reviewed eligibility table"
            year_min = int(years.min()) if len(years) else None
            year_max = int(years.max()) if len(years) else None
            time_min = dt.min().isoformat() if dt.notna().any() else None
            time_max = dt.max().isoformat() if dt.notna().any() else None
        time_rows.append({"db": db, "field": col, "n_total": n, "n_observed": int(dt.notna().sum()),
                          "n_missing": int(dt.isna().sum()), "year_min": year_min, "year_max": year_max,
                          "time_min": time_min, "time_max": time_max,
                          "time_interpretation": time_interpretation, "date_evidence": date_evidence})
    for col in ("icu_exit_hr", "hospital_exit_hr", "hospital_death_time_hr", "icu_death_time_hr"):
        if col not in frame:
            continue
        v = pd.to_numeric(frame[col], errors="coerce")
        time_rows.append({"db": db, "field": col, "n_total": n, "n_observed": int(v.notna().sum()),
                          "n_missing": int(v.isna().sum()), "n_le_24": int(v.le(24).sum()),
                          "n_gt_24": int(v.gt(24).sum()), "min": float(v.min()) if v.notna().any() else None,
                          "median": float(v.median()) if v.notna().any() else None,
                          "max": float(v.max()) if v.notna().any() else None,
                          "time_interpretation": "relative hours since source admission/index; not actual calendar time"})
    if "icu_exit_hr" in frame and "hospital_exit_hr" in frame:
        icu_exit = pd.to_numeric(frame["icu_exit_hr"], errors="coerce")
        hosp_exit = pd.to_numeric(frame["hospital_exit_hr"], errors="coerce")
        time_rows.append({"db": db, "field": "icu_hospital_exit_order", "n_total": n,
                          "n_both_observed": int((icu_exit.notna() & hosp_exit.notna()).sum()),
                          "n_icu_after_hospital": int((icu_exit > hosp_exit).sum()),
                          "n_hospital_exit_before_icu_and_after24": int((icu_exit > hosp_exit).mul(hosp_exit > 24).sum())})
    return data_rows, cat_rows, time_rows


def raw_nwicu_audit(eligibility: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    adm_path = NWICU_RAW / "admissions.csv.gz"
    icu_path = NWICU_RAW / "icustays.csv.gz"
    log("nwicu_raw_admissions_read_begin")
    adm = pd.read_csv(adm_path, usecols=lambda c: c in {"hadm_id", "hospital_expire_flag", "discharge_location", "deathtime", "dischtime"}, compression="infer", low_memory=False)
    log(f"nwicu_raw_admissions_read_complete rows={len(adm)}")
    log("nwicu_raw_icustays_read_begin")
    icu = pd.read_csv(icu_path, usecols=lambda c: c in {"stay_id", "subject_id", "hadm_id", "intime", "outtime"}, compression="infer", low_memory=False)
    log(f"nwicu_raw_icustays_read_complete rows={len(icu)}")
    for c in ("stay_id", "subject_id", "hadm_id"):
        if c not in icu:
            raise ValueError(f"NWICU icustays missing {c}")
    if "hadm_id" not in adm or "discharge_location" not in adm or "hospital_expire_flag" not in adm:
        raise ValueError("NWICU admissions lacks expected death adjudication fields")
    icu["stay_id"] = icu["stay_id"].map(canonical); icu["subject_id"] = icu["subject_id"].map(canonical); icu["hadm_id"] = icu["hadm_id"].map(canonical)
    adm["hadm_id"] = adm["hadm_id"].map(canonical)
    adm["hospital_expire_flag"] = pd.to_numeric(adm["hospital_expire_flag"], errors="coerce")
    location = adm["discharge_location"].astype("string").str.strip().str.upper()
    adm["historical_exact_expired_string"] = location.eq("EXPIRED")
    adm["known_location"] = location.isin(NWICU_KNOWN_ALIVE | NWICU_KNOWN_DEAD)
    corrected = pd.Series(pd.NA, index=adm.index, dtype="Int8")
    corrected.loc[location.isin(NWICU_KNOWN_ALIVE)] = 0
    corrected.loc[location.isin(NWICU_KNOWN_DEAD)] = 1
    adm["corrected_known_endpoint_death"] = corrected
    adm["flag_binary"] = adm["hospital_expire_flag"].where(adm["hospital_expire_flag"].isin([0, 1]))
    adm["historical_disagreement"] = (adm["flag_binary"].notna() & adm["discharge_location"].notna()
                                      & adm["flag_binary"].ne(adm["historical_exact_expired_string"].astype(int)))
    adm["corrected_disagreement"] = (adm["flag_binary"].notna() & adm["known_location"]
                                     & adm["flag_binary"].ne(adm["corrected_known_endpoint_death"]))
    historical_denom = int((adm["flag_binary"].notna() & adm["discharge_location"].notna()).sum())
    historical_disagreements = int(adm["historical_disagreement"].sum())
    corrected_denom = int((adm["flag_binary"].notna() & adm["known_location"]).sum())
    corrected_disagreements = int(adm["corrected_disagreement"].sum())
    rows = [{"scope": "all_NWICU_admissions", "n_admissions": len(adm),
             "historical_exact_string_denominator": historical_denom,
             "historical_exact_string_disagreements": historical_disagreements,
             "historical_exact_string_disagreement_fraction": historical_disagreements / historical_denom if historical_denom else None,
             "corrected_known_endpoint_denominator": corrected_denom,
             "corrected_known_endpoint_disagreements": corrected_disagreements,
             "corrected_known_endpoint_disagreement_fraction": corrected_disagreements / corrected_denom if corrected_denom else None,
             "n_unknown_discharge_location": int((~adm["known_location"]).sum()),
             "adjudication": "corrected mortality uses the v4 documented KNOWN_ALIVE and KNOWN_DEAD whitelists; historical exact-string comparison is reported separately"}]
    raw_values = adm["discharge_location"].astype("string").fillna("<MISSING>").value_counts(dropna=False)
    cats = [{"scope": "all_NWICU_admissions", "table": "discharge_location_values", "raw_discharge_location": str(k), "n": int(v)} for k, v in raw_values.items()]
    all_xtab = pd.crosstab(adm["flag_binary"].astype("string").fillna("<MISSING>"),
                           adm["discharge_location"].astype("string").fillna("<MISSING>"), dropna=False)
    for flag_value in all_xtab.index:
        for location_value in all_xtab.columns:
            cats.append({"scope": "all_NWICU_admissions", "table": "hospital_expire_flag_by_raw_discharge_location",
                         "hospital_expire_flag": str(flag_value), "raw_discharge_location": str(location_value),
                         "n": int(all_xtab.loc[flag_value, location_value])})
    cohort = eligibility[["db", "stay_id", "subject", "y_mort"]].copy()
    raw = cohort.merge(icu[["stay_id", "subject_id", "hadm_id"]], left_on=["stay_id", "subject"],
                       right_on=["stay_id", "subject_id"], how="left", validate="one_to_one", indicator=True)
    if not raw["_merge"].eq("both").all():
        raise ValueError("NWICU reviewed stays do not fully bind to raw icustays")
    raw = raw.merge(adm[["hadm_id", "discharge_location", "hospital_expire_flag", "historical_exact_expired_string",
                         "known_location", "corrected_known_endpoint_death", "historical_disagreement", "corrected_disagreement"]],
                    on="hadm_id", how="left", validate="many_to_one", indicator="adm_merge")
    if not raw["adm_merge"].eq("both").all():
        raise ValueError("NWICU reviewed stays do not fully bind to raw admissions")
    historical_denom = int((raw["hospital_expire_flag"].isin([0, 1]) & raw["discharge_location"].notna()).sum())
    historical_disagreements = int(raw["historical_disagreement"].sum())
    corrected_denom = int((raw["hospital_expire_flag"].isin([0, 1]) & raw["known_location"]).sum())
    corrected_disagreements = int(raw["corrected_disagreement"].sum())
    frozen = pd.to_numeric(raw["y_mort"], errors="coerce").astype("Int8")
    corrected = raw["corrected_known_endpoint_death"].astype("Int8")
    comparable = corrected.notna() & frozen.notna()
    mismatch = corrected.ne(frozen).fillna(False)
    rows.append({"scope": "reviewed_sequence_stays", "n_stays": len(raw),
                 "historical_exact_string_denominator": historical_denom,
                 "historical_exact_string_disagreements": historical_disagreements,
                 "historical_exact_string_disagreement_fraction": historical_disagreements / historical_denom if historical_denom else None,
                 "corrected_known_endpoint_denominator": corrected_denom,
                 "corrected_known_endpoint_disagreements": corrected_disagreements,
                 "corrected_known_endpoint_disagreement_fraction": corrected_disagreements / corrected_denom if corrected_denom else None,
                 "corrected_vs_frozen_y_mort_denominator": int(comparable.sum()),
                 "corrected_vs_frozen_y_mort_disagreements": int((comparable & mismatch).sum()),
                 "n_unknown_corrected_endpoint": int(corrected.isna().sum()),
                 "n_frozen_y_mort_unknown": int(frozen.isna().sum()),
                 "adjudication": "corrected mortality uses the v4 documented KNOWN_ALIVE and KNOWN_DEAD whitelists"})
    values = raw["discharge_location"].astype("string").fillna("<MISSING>").value_counts(dropna=False)
    cats.extend({"scope": "reviewed_sequence_stays", "table": "discharge_location_values", "raw_discharge_location": str(k), "n": int(v)} for k, v in values.items())
    reviewed_xtab = pd.crosstab(raw["hospital_expire_flag"].astype("string").fillna("<MISSING>"),
                                raw["discharge_location"].astype("string").fillna("<MISSING>"), dropna=False)
    for flag_value in reviewed_xtab.index:
        for location_value in reviewed_xtab.columns:
            cats.append({"scope": "reviewed_sequence_stays", "table": "hospital_expire_flag_by_raw_discharge_location",
                         "hospital_expire_flag": str(flag_value), "raw_discharge_location": str(location_value),
                         "n": int(reviewed_xtab.loc[flag_value, location_value])})
    return rows, cats


def raw_sicdb_audit(eligibility: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = SICDB_RAW / "cases.csv.gz"
    header = pd.read_csv(path, compression="infer", nrows=0).columns.tolist()
    lookup = {str(c).lower(): c for c in header}
    requested = ["caseid", "patientid", "ageonadmission", "dischargestate", "timeofday", "timeofstay", "offsetofdeath", "icuoffset"]
    use = [lookup[x] for x in requested if x in lookup]
    if not any(x.lower() == "caseid" for x in use):
        raise ValueError("SICdb cases source lacks CaseID")
    log("sicdb_raw_cases_read_begin")
    cases = pd.read_csv(path, compression="infer", usecols=use, low_memory=False)
    log(f"sicdb_raw_cases_read_complete rows={len(cases)}")
    cases.columns = [str(c).lower() for c in cases.columns]
    if "caseid" not in cases:
        raise ValueError("SICdb CaseID field not retained")
    cases["caseid"] = cases["caseid"].map(canonical)
    rows: list[dict[str, Any]] = []
    for field in ("dischargestate", "timeofstay", "offsetofdeath", "icuoffset"):
        if field not in cases:
            rows.append({"scope": "all_cases", "field": field, "available": False})
            continue
        vals = cases[field]
        row: dict[str, Any] = {"scope": "all_cases", "field": field, "available": True,
                               "n_cases": len(cases), "n_nonmissing": int(vals.notna().sum()),
                               "n_missing": int(vals.isna().sum()), "n_unique_raw": int(vals.nunique(dropna=True))}
        numeric = pd.to_numeric(vals, errors="coerce")
        if numeric.notna().any():
            row.update({"numeric_min": float(numeric.min()), "numeric_median": float(numeric.median()),
                        "numeric_max": float(numeric.max()), "n_zero": int(numeric.eq(0).sum())})
        rows.append(row)
    cats: list[dict[str, Any]] = []
    if "dischargestate" in cases:
        counts = cases["dischargestate"].astype("string").fillna("<MISSING>").value_counts(dropna=False)
        cats.extend({"field": "DischargeState", "raw_value": str(k), "n": int(v)} for k, v in counts.items())
    refs_path = SICDB_RAW / "d_references.csv.gz"
    if refs_path.is_file():
        refs = pd.read_csv(refs_path, compression="infer", low_memory=False)
        text_cols = [c for c in refs if refs[c].dtype == object or pd.api.types.is_string_dtype(refs[c])]
        if text_cols:
            text = refs[text_cols].astype("string").fillna("").agg(" | ".join, axis=1)
            keep = text.str.contains("TimeOfStay|OffsetOfDeath|ICUOffset|DischargeState|CaseID", case=False, regex=True)
            for val in text.loc[keep].head(200).tolist():
                cats.append({"field": "reference_definition", "raw_value": str(val), "n": None})
    rows.append({"scope": "endpoint_qualification", "field": "interpretation", "available": True,
                 "case_close_is_index_icu_exit": False, "continuous_icu_at_h24_verified": False,
                 "icu_death_verified": False, "endpoint_name": "SICdb case-period/discharge/death proxy",
                 "candidate_event_fields": "DischargeState and OffsetOfDeath; neither verifies an index-ICU death",
                 "status": "case-period/discharge/death proxy only; exploratory"})
    return rows, cats


def explicit_status(values: pd.Series) -> pd.Series:
    low = values.astype("string").str.strip().str.casefold()
    result = pd.Series(pd.NA, index=values.index, dtype="boolean")
    dead = low.eq("expired")
    alive = low.eq("alive")
    result.loc[dead] = True
    result.loc[alive] = False
    return result


def bridge_admin_flow_counts(frame: pd.DataFrame) -> dict[str, Any]:
    flag = frame["admin_hospital_admit_after_h24"].astype("boolean")
    return {
        "n_stays": int(len(frame)),
        "n_patients": int(frame["subject"].nunique()),
        "n_no_admin_after_h24": int(flag.eq(False).fillna(False).sum()),
        "n_admin_after_h24": int(flag.eq(True).fillna(False).sum()),
        "n_admin_after_h24_unknown": int(flag.isna().sum()),
        "n_unit_deaths": int(frame["icu_y"].eq(1).sum()),
        "n_alive_unit_exits": int(frame["icu_y"].eq(0).sum()),
    }


def valid_input_rows(ds: dict[str, np.ndarray]) -> np.ndarray:
    L = np.asarray(ds["L"], dtype=np.int64)
    H = np.asarray(ds["H"])
    valid = L > 0
    active = np.arange(H.shape[1])[None, :] < L[:, None]
    valid &= np.all((np.isfinite(H) & (H >= 0) & (H < 24)) | ~active, axis=1)
    return valid


def mimic_icu_labels(ds: dict[str, np.ndarray], split: np.ndarray) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    path_icu = MIMIC_RAW / "icu/icustays.csv.gz"
    path_adm = MIMIC_RAW / "hosp/admissions.csv.gz"
    icu = pd.read_csv(path_icu, usecols=["stay_id", "subject_id", "hadm_id", "intime", "outtime"], compression="infer", low_memory=False)
    adm = pd.read_csv(path_adm, usecols=["hadm_id", "deathtime", "hospital_expire_flag"], compression="infer", low_memory=False)
    for col in ("stay_id", "subject_id", "hadm_id"):
        icu[col] = icu[col].map(canonical)
    adm["hadm_id"] = adm["hadm_id"].map(canonical)
    if icu["stay_id"].duplicated().any() or adm["hadm_id"].duplicated().any():
        raise ValueError("MIMIC raw stay/admission ID is duplicated")
    seq = pd.DataFrame({"stay_id": [canonical(x) for x in ds["stay_id"]], "subject": [canonical(x) for x in ds["subject"]],
                        "rows": np.arange(len(ds["L"]), dtype=np.int64), "split": split,
                        "input_valid": valid_input_rows(ds), "L": np.asarray(ds["L"], dtype=np.int16)})
    merged = seq.merge(icu, left_on=["stay_id", "subject"], right_on=["stay_id", "subject_id"], how="left", validate="one_to_one", indicator="icu_merge")
    matched = merged["icu_merge"].eq("both")
    if int(matched.sum()) == 0:
        raise ValueError("MIMIC sequence stays do not bind to raw ICU stays")
    merged = merged.merge(adm, on="hadm_id", how="left", validate="many_to_one", indicator="adm_merge")
    admission_matched = matched & merged["adm_merge"].eq("both")
    intime = pd.to_datetime(merged["intime"], errors="coerce")
    outtime = pd.to_datetime(merged["outtime"], errors="coerce")
    deathtime_raw = merged["deathtime"]
    raw_death_text = deathtime_raw.astype("string").str.strip()
    deathtime_recorded = deathtime_raw.notna() & raw_death_text.ne("").fillna(False)
    deathtime = pd.to_datetime(deathtime_raw, errors="coerce")
    deathtime_missing = ~deathtime_recorded
    deathtime_unparsable = deathtime_recorded & deathtime.isna()
    deathtime_valid = deathtime.notna()
    deathtime_status = pd.Series("missing", index=merged.index, dtype="string")
    deathtime_status.loc[deathtime_unparsable] = "unparsable"
    deathtime_status.loc[deathtime_valid] = "valid"
    merged["deathtime_parse_status"] = deathtime_status
    cutoff = intime + pd.Timedelta(hours=24)
    h24_in_icu = intime.notna() & outtime.notna() & outtime.gt(cutoff)
    death_hr = (deathtime - intime).dt.total_seconds() / 3600.0
    proven_early_death = deathtime_valid & intime.notna() & death_hr.le(24)
    hospital_expire_flag = pd.to_numeric(merged["hospital_expire_flag"], errors="coerce")
    source_conflict = hospital_expire_flag.eq(0) & deathtime_valid
    label = pd.Series(pd.NA, index=merged.index, dtype="Int8")
    exact_in_episode = deathtime_valid & intime.notna() & outtime.notna() & deathtime.ge(intime) & deathtime.le(outtime)
    valid_death_candidate = (exact_in_episode & death_hr.gt(24) & h24_in_icu & admission_matched
                             & merged["input_valid"] & ~source_conflict)
    candidates = merged.loc[valid_death_candidate, ["hadm_id", "stay_id"]].copy()
    candidates["death_time"] = deathtime.loc[valid_death_candidate]
    candidates["intime_parsed"] = intime.loc[valid_death_candidate]
    candidates["outtime_parsed"] = outtime.loc[valid_death_candidate]
    candidates["death_at_icu_exit"] = candidates["death_time"].eq(candidates["outtime_parsed"])
    duplicate_death_keys = candidates.groupby(["hadm_id", "death_time"], dropna=False).size()
    n_duplicate_death_keys = int(duplicate_death_keys.gt(1).sum())
    duplicate_death_key_rows = candidates.groupby(["hadm_id", "death_time"], dropna=False)["stay_id"].transform("size").gt(1)
    duplicate_death_candidate_rows = pd.Series(False, index=merged.index, dtype=bool)
    duplicate_death_candidate_rows.loc[candidates.index] = duplicate_death_key_rows.to_numpy(dtype=bool)
    ordered_candidates = candidates.sort_values(
        ["hadm_id", "death_time", "death_at_icu_exit", "intime_parsed", "outtime_parsed", "stay_id"],
        ascending=[True, True, False, False, True, True], kind="mergesort",
    )
    death_winners = ordered_candidates.drop_duplicates(["hadm_id", "death_time"], keep="first")
    death_assigned = pd.Series(False, index=merged.index, dtype=bool)
    death_assigned.loc[death_winners.index] = True
    duplicate_death_candidate_unassigned = valid_death_candidate & ~death_assigned
    label.loc[death_assigned] = 1
    alive_unit_exit = deathtime_missing & hospital_expire_flag.eq(0)
    death_after_unit = deathtime_valid & outtime.notna() & deathtime.gt(outtime)
    label.loc[alive_unit_exit | death_after_unit] = 0
    label.loc[duplicate_death_candidate_unassigned] = pd.NA
    keep = (admission_matched & merged["input_valid"] & h24_in_icu
            & ~proven_early_death & ~source_conflict & ~deathtime_unparsable & label.notna())
    rows = [{"db": "mimic", "split": int(s), "n_sequence_rows": int(len(merged.loc[merged["split"].eq(s)])),
             "n_raw_icu_matched": int((merged["split"].eq(s) & matched).sum()),
             "n_admission_rows_for_conflict_audit": int((merged["split"].eq(s) & admission_matched).sum()),
             "n_hospital_expire_flag_zero_with_deathtime_conflict": int((merged["split"].eq(s) & admission_matched & source_conflict).sum()),
             "n_hospital_expire_flag_missing": int((merged["split"].eq(s) & admission_matched & hospital_expire_flag.isna()).sum()),
              "n_deathtime_genuinely_missing": int((merged["split"].eq(s) & admission_matched & deathtime_missing).sum()),
              "n_deathtime_unparsable": int((merged["split"].eq(s) & admission_matched & deathtime_unparsable).sum()),
              "n_deathtime_valid": int((merged["split"].eq(s) & admission_matched & deathtime_valid).sum()),
              "n_death_episode_assignments": int((merged["split"].eq(s) & death_assigned).sum()),
              "n_duplicate_death_episode_candidate_rows": int((merged["split"].eq(s) & duplicate_death_candidate_rows).sum()),
              "n_unassigned_duplicate_death_episode_candidates": int((merged["split"].eq(s) & duplicate_death_candidate_unassigned).sum()),
              "n_deathtime_equal_icu_exit": int((merged["split"].eq(s) & deathtime_valid & outtime.notna() & deathtime.eq(outtime)).sum()),
             "n_h24_icu_at_risk": int((merged["split"].eq(s) & matched & h24_in_icu & ~proven_early_death).sum()),
             "n_known_unit_death_label": int((merged["split"].eq(s) & label.notna()).sum()),
             "n_model_eligible": int((merged["split"].eq(s) & keep).sum()),
             "n_icu_deaths": int((merged["split"].eq(s) & keep & label.eq(1)).sum()),
             "n_known_surviving_icu_exit": int((merged["split"].eq(s) & keep & label.eq(0)).sum())} for s in (0, 1, 2)]
    conflict_frame = merged.loc[admission_matched].copy()
    conflict_frame["hospital_expire_flag_raw"] = (
        conflict_frame["hospital_expire_flag"].astype("string").fillna("<MISSING>")
    )
    conflict_frame["deathtime_parse_status"] = deathtime_status.loc[admission_matched].astype("string").to_numpy()
    conflict_frame["source_conflict"] = source_conflict.loc[admission_matched].to_numpy()
    cross = conflict_frame.groupby(["split", "hospital_expire_flag_raw", "deathtime_parse_status"],
                                  dropna=False, sort=True).agg(
        n_stays=("stay_id", "size"), n_source_conflicts=("source_conflict", "sum")
    ).reset_index()
    conflict_rows = [{"db": "mimic", "stage": "raw_hospital_expire_flag_by_deathtime_parse_status",
                      "split": int(row.split), "hospital_expire_flag_raw": str(row.hospital_expire_flag_raw),
                      "deathtime_parse_status": str(row.deathtime_parse_status), "n_stays": int(row.n_stays),
                      "n_source_conflicts": int(row.n_source_conflicts),
                      "conflict_definition": "hospital_expire_flag == 0 with a valid parsed deathtime"}
                     for row in cross.itertuples(index=False)]
    conflict_rows.append({"db": "mimic", "stage": "death_episode_assignment", "split": "all",
                          "n_duplicate_death_episode_groups": n_duplicate_death_keys,
                          "n_duplicate_death_episode_candidate_rows": int(duplicate_death_candidate_rows.sum()),
                          "n_unassigned_duplicate_death_episode_candidates": int(duplicate_death_candidate_unassigned.sum()),
                          "assignment_rule": "at most one eligible index ICU episode per admission/deathtime; exact exit equality is retained and preferred, then latest intime, earliest outtime, and lexical stay_id"})
    merged["icu_y"] = label
    merged["keep_bridge_source"] = keep
    return (merged.loc[keep, ["stay_id", "subject", "rows", "split", "icu_y", "intime", "outtime", "deathtime", "deathtime_parse_status", "hospital_expire_flag"]].copy(),
            rows, conflict_rows)


def eicu_icu_labels(ds: dict[str, np.ndarray], split: np.ndarray) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    path = EICU_RAW / "patient.csv.gz"
    raw = pd.read_csv(path, usecols=lambda c: c in {"patientunitstayid", "uniquepid", "unitdischargestatus", "hospitaldischargestatus", "unitdischargeoffset", "hospitaladmitoffset", "hospitalid"}, compression="infer", low_memory=False)
    required = {"patientunitstayid", "uniquepid", "unitdischargestatus", "hospitaldischargestatus",
                "unitdischargeoffset", "hospitaladmitoffset", "hospitalid"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"eICU patient source missing required bridge fields: {sorted(missing)}")
    for col in ("patientunitstayid", "uniquepid", "hospitalid"):
        raw[col] = raw[col].map(canonical)
    if raw["patientunitstayid"].duplicated().any():
        raise ValueError("eICU patientunitstayid is not unique")
    raw["unit_dead"] = explicit_status(raw["unitdischargestatus"])
    raw["hospital_dead"] = explicit_status(raw["hospitaldischargestatus"])
    raw["unit_expired_hospital_alive_conflict"] = raw["unit_dead"].eq(True).fillna(False) & raw["hospital_dead"].eq(False).fillna(False)
    offset = pd.to_numeric(raw["unitdischargeoffset"], errors="coerce")
    offset_finite = pd.Series(np.isfinite(offset.to_numpy(dtype=float)), index=raw.index)
    raw["unitdischargeoffset_finite"] = offset_finite
    raw["unit_h24_at_risk"] = offset_finite & offset.gt(1440)
    admin_offset = pd.to_numeric(raw["hospitaladmitoffset"], errors="coerce")
    admin_finite = pd.Series(np.isfinite(admin_offset.to_numpy(dtype=float)), index=raw.index)
    raw["hospitaladmitoffset_finite"] = admin_finite
    raw["admin_hospital_admit_after_h24"] = admin_offset.gt(1440).where(admin_finite, pd.NA).astype("boolean")
    seq = pd.DataFrame({"stay_id": [canonical(x) for x in ds["stay_id"]], "subject": [canonical(x) for x in ds["subject"]],
                        "rows": np.arange(len(ds["L"]), dtype=np.int64), "split": split,
                        "input_valid": valid_input_rows(ds), "L": np.asarray(ds["L"], dtype=np.int16)})
    merged = seq.merge(raw, left_on=["stay_id", "subject"], right_on=["patientunitstayid", "uniquepid"], how="left", validate="one_to_one", indicator="raw_merge")
    matched = merged["raw_merge"].eq("both")
    status_known = merged["unit_dead"].notna()
    keep = matched & merged["input_valid"] & merged["unit_h24_at_risk"] & status_known & ~merged["unit_expired_hospital_alive_conflict"]
    merged["icu_y"] = merged["unit_dead"].astype("Int8")
    merged["keep_bridge_target"] = keep
    rows = []
    for s in (0, 1, 2):
        in_split = merged["split"].eq(s)
        admin_flag = merged["admin_hospital_admit_after_h24"].astype("boolean")
        rows.append({"db": "eicu", "split": s, "n_sequence_rows": int(in_split.sum()),
                     "n_raw_unit_matched": int((in_split & matched).sum()),
                     "n_h24_unit_at_risk": int((in_split & matched & merged["unit_h24_at_risk"]).sum()),
                     "n_explicit_unit_status": int((in_split & matched & status_known).sum()),
                     "n_unknown_or_unrecognized_unit_status": int((in_split & matched & ~status_known).sum()),
                     "n_explicit_hospital_status": int((in_split & matched & merged["hospital_dead"].notna()).sum()),
                     "n_unknown_or_unrecognized_hospital_status": int((in_split & matched & merged["hospital_dead"].isna()).sum()),
                     "n_invalid_or_missing_unit_offset": int((in_split & matched & ~merged["unitdischargeoffset_finite"].fillna(False)).sum()),
                     "n_invalid_or_missing_admin_offset": int((in_split & matched & ~merged["hospitaladmitoffset_finite"].fillna(False)).sum()),
                     "n_unknown_admin_offset": int((in_split & matched & admin_flag.isna()).sum()),
                     "n_unit_expired_hospital_alive_conflict": int((in_split & matched & merged["unit_expired_hospital_alive_conflict"]).sum()),
                     "n_admin_hospital_admit_after_h24_retained": int((in_split & keep & admin_flag.fillna(False)).sum()),
                     "n_admin_hospital_admit_after_h24_unknown_retained": int((in_split & keep & admin_flag.isna()).sum()),
                     "n_model_eligible": int((in_split & keep).sum()),
                     "n_unit_deaths": int((in_split & keep & merged["icu_y"].eq(1)).sum()),
                     "n_alive_unit_exits": int((in_split & keep & merged["icu_y"].eq(0)).sum())})
    return merged.loc[keep, ["stay_id", "subject", "rows", "split", "icu_y", "unitdischargestatus", "hospitaldischargestatus",
                             "unitdischargeoffset", "hospitaladmitoffset", "admin_hospital_admit_after_h24", "hospitalid"]].copy(), rows


def load_cache(db: str, ds: dict[str, np.ndarray]) -> pd.DataFrame:
    manifest_path = MANIFEST_PATH
    if sha256(manifest_path) != EXPECTED_MANIFEST_SHA:
        raise ValueError("frozen R4 41-row embedding manifest SHA256 changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [r for r in manifest.get("models", []) if r.get("db") == COHORTS[db] and r.get("model_tag") == "main20k" and r.get("input_condition") == "real"]
    if len(rows) != 1:
        raise ValueError(f"{db}: expected exactly one main20k/real row in frozen manifest")
    expected_cache = CACHE_PATHS[db].resolve(strict=True)
    manifest_cache = Path(rows[0]["cache"]).resolve(strict=True)
    if expected_cache != manifest_cache:
        raise ValueError(f"{db}: cache path differs from frozen manifest row")
    with np.load(expected_cache, allow_pickle=False) as z:
        required = {"rows", "stay_id", "subject", "source_y", "embedding", "metadata_json"}
        if not required.issubset(z.files):
            raise ValueError(f"{db}: embedding cache arrays missing {sorted(required - set(z.files))}")
        rows_arr = np.asarray(z["rows"], dtype=np.int64)
        stays = np.asarray(z["stay_id"]).astype(str)
        subjects = np.asarray(z["subject"]).astype(str)
        source_y = np.asarray(z["source_y"], dtype=np.float32)
        embedding = np.asarray(z["embedding"], dtype=np.float32)
        metadata = json.loads(str(z["metadata_json"].item()))
    if metadata.get("db") != COHORTS[db] or metadata.get("model_tag") != "main20k" or metadata.get("input_condition") != "real":
        raise ValueError(f"{db}: frozen embedding metadata identity mismatch")
    if not (len(rows_arr) == len(stays) == len(subjects) == len(source_y) == len(embedding)) or embedding.ndim != 2 or not np.isfinite(embedding).all():
        raise ValueError(f"{db}: invalid embedding cache dimensions/values")
    if len(np.unique(rows_arr)) != len(rows_arr):
        raise ValueError(f"{db}: duplicate row index in embedding cache")
    if rows_arr.size and (rows_arr.min() < 0 or rows_arr.max() >= len(ds["L"])):
        raise ValueError(f"{db}: embedding row index outside sequence")
    seq_stay = np.asarray([canonical(ds["stay_id"][r]) for r in rows_arr], dtype=str)
    seq_subject = np.asarray([canonical(ds["subject"][r]) for r in rows_arr], dtype=str)
    seq_y = np.asarray(ds["y_mort"], dtype=np.float32)[rows_arr]
    if not np.array_equal(stays, seq_stay) or not np.array_equal(subjects, seq_subject) or not np.array_equal(source_y, seq_y, equal_nan=True):
        raise ValueError(f"{db}: embedding cache rows do not identity-bind to sequence arrays")
    return pd.DataFrame({"rows": rows_arr, "stay_id": stays, "subject": subjects, "source_y": source_y,
                         "embedding": list(embedding)})


def score_model(train_x: np.ndarray, train_y: np.ndarray, valid_x: np.ndarray, valid_y: np.ndarray,
                test_x: np.ndarray, test_y: np.ndarray, target_x: np.ndarray,
                model_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    scaler = StandardScaler().fit(train_x)
    x0 = scaler.transform(train_x); x1 = scaler.transform(valid_x); x2 = scaler.transform(test_x)
    xt = scaler.transform(target_x)
    tuning: list[dict[str, Any]] = []
    for c in CS:
        clf = LogisticRegression(C=c, solver="lbfgs", max_iter=3000, tol=1e-8)
        clf.fit(x0, train_y)
        p1 = clf.predict_proba(x1)[:, 1]
        tuning.append({"model": model_name, "C": c, "split1_n": int(len(valid_y)),
                       "split1_events": int(np.sum(valid_y)), "split1_brier": float(np.mean((p1 - valid_y) ** 2))})
    best = min(tuning, key=lambda r: (r["split1_brier"], r["C"]))
    clf = LogisticRegression(C=float(best["C"]), solver="lbfgs", max_iter=3000, tol=1e-8)
    clf.fit(x0, train_y)
    p2 = clf.predict_proba(x2)[:, 1]
    pt = clf.predict_proba(xt)[:, 1]
    return p2, pt, clf.coef_.ravel(), {"model": model_name, "selected_C": float(best["C"]),
                                      "selection_criterion": "split1 Brier; exact ties select smaller C",
                                      "split0_fit_n": int(len(train_y)), "split0_fit_events": int(np.sum(train_y)),
                                      "split1_select_n": int(len(valid_y)), "split2_test_n": int(len(test_y))}, tuning


def brier_logloss_ci(subject: np.ndarray, y: np.ndarray, p_fm: np.ndarray, p_lr: np.ndarray,
                     draws: int, seed: int) -> list[dict[str, Any]]:
    subjects = np.asarray(subject, dtype=str)
    y = np.asarray(y, dtype=np.int8)
    p_fm = np.clip(np.asarray(p_fm, dtype=float), EPS, 1-EPS)
    p_lr = np.clip(np.asarray(p_lr, dtype=float), EPS, 1-EPS)
    if not (len(subjects) == len(y) == len(p_fm) == len(p_lr)):
        raise ValueError("paired bridge score/label/patient lengths differ")
    fm_brier = (p_fm-y)**2; lr_brier = (p_lr-y)**2
    fm_logloss = -y*np.log(p_fm) - (1-y)*np.log1p(-p_fm)
    lr_logloss = -y*np.log(p_lr) - (1-y)*np.log1p(-p_lr)
    differences = np.column_stack((fm_brier-lr_brier, fm_logloss-lr_logloss))
    results = []
    for patient_weighted, estimand in ((False, "stay_weighted_primary"),
                                       (True, "patient_weighted_sensitivity")):
        try:
            summary = paired_cluster_mean_ci(subjects, differences, draws=draws, seed=seed,
                                             patient_weighted=patient_weighted)
            points = np.asarray(summary["point"], dtype=float)
            lows = np.asarray(summary["low"], dtype=float)
            highs = np.asarray(summary["high"], dtype=float)
            n_patients = int(summary["n_patients"])
            failure = None
        except ValueError as exc:
            points = lows = highs = np.full(2, np.nan)
            n_patients = int(pd.Series(subjects).nunique())
            failure = str(exc)
        for index, metric in enumerate(("brier", "logloss")):
            results.append({"metric": metric,
                            "fm_minus_lr": float(points[index]) if np.isfinite(points[index]) else None,
                            "low95": float(lows[index]) if np.isfinite(lows[index]) else None,
                            "high95": float(highs[index]) if np.isfinite(highs[index]) else None,
                            "n_patients": n_patients, "n_stays": int(len(y)), "draws": draws,
                            "seed": seed, "estimand": estimand,
                            "weighting": "patient" if patient_weighted else "stay",
                            "unavailable_reason": failure})
    return results


def recalibrate(p: np.ndarray, y: np.ndarray, method: str) -> tuple[np.ndarray, dict[str, Any]]:
    p = np.asarray(p, dtype=float)
    probabilities_finite = bool(np.isfinite(p).all())
    y = np.asarray(y, dtype=np.int8)
    p = np.clip(p, EPS, 1-EPS)
    prevalence = float((y.sum() + .5) / (len(y) + 1.0)) if len(y) else .5
    prevalence_offset = float(np.log(prevalence / (1.0 - prevalence)))

    def prevalence_fallback(reason: str) -> tuple[np.ndarray, dict[str, Any]]:
        return np.full(len(p), prevalence), {
            "method": method, "status": "jeffreys_prevalence_fallback", "fallback_reason": reason,
            "fallback_probability": prevalence, "offset": None, "intercept": None, "slope": None,
        }

    if method == "prevalence":
        return np.full(len(p), prevalence), {
            "method": method, "status": "jeffreys_prevalence", "fallback_probability": prevalence,
            "offset": prevalence_offset, "intercept": prevalence_offset, "slope": 0.0,
        }
    if method not in {"intercept", "platt"}:
        raise ValueError(method)
    if len(np.unique(y)) < 2:
        return prevalence_fallback("single_class_adaptation_outcome")
    if len(p) != len(y) or not probabilities_finite or not np.isin(y, [0, 1]).all():
        return prevalence_fallback("invalid_adaptation_inputs")
    z = np.log(p / (1-p))
    try:
        if method == "intercept":
            b1, offset = fit_offset(y, z)
        else:
            b1, offset = fit_platt(y, z)
        b0 = float(offset)
        b1 = float(b1)
        if not np.isfinite(b0) or not np.isfinite(b1):
            return prevalence_fallback("nonfinite_fit_coefficients")
    except Exception as exc:
        return prevalence_fallback(f"fit_failed:{type(exc).__name__}:{exc}")
    q = expit(np.clip(b0 + b1 * z, -40, 40))
    return np.clip(q, EPS, 1-EPS), {"method": method, "status": "fit", "offset": b0,
                                    "intercept": b0, "slope": b1}


def apply_recalibration(p: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    """Apply adaptation-only coefficients while preserving a fitted zero slope."""
    p = np.asarray(p, dtype=float)
    if meta.get("status") == "jeffreys_prevalence_fallback":
        return np.full(len(p), float(meta["fallback_probability"]))
    clipped = np.clip(p, EPS, 1-EPS)
    z = np.log(clipped / (1-clipped))
    offset = float(meta["offset"])
    slope = float(meta["slope"])
    return np.clip(expit(np.clip(offset + slope * z, -40, 40)), EPS, 1-EPS)


def run_bridge(mimic_ds: dict[str, np.ndarray], mimic_split: np.ndarray,
               eicu_ds: dict[str, np.ndarray], eicu_split: np.ndarray,
               prediction_data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    OUT_BRIDGE.mkdir(parents=True, exist_ok=True)
    mimic_roster, source_flow, source_conflicts = mimic_icu_labels(mimic_ds, mimic_split)
    eicu_roster, target_flow = eicu_icu_labels(eicu_ds, eicu_split)
    source_subject = pd.Series([canonical(v) for v in mimic_ds["subject"]], dtype="string")
    source_split_arr = np.asarray(mimic_split, dtype=np.int8)
    if not np.isin(source_split_arr, [0, 1, 2]).all():
        raise ValueError("MIMIC source split values outside 0/1/2")
    if source_subject.isna().any():
        raise ValueError("MIMIC source split audit found a missing patient identity")
    source_patients_by_split = {s: set(source_subject.loc[source_split_arr == s].astype(str)) for s in (0, 1, 2)}
    split_overlaps = {
        "split0_x_split1": source_patients_by_split[0] & source_patients_by_split[1],
        "split0_x_split2": source_patients_by_split[0] & source_patients_by_split[2],
        "split1_x_split2": source_patients_by_split[1] & source_patients_by_split[2],
    }
    overlap_counts = {name: len(value) for name, value in split_overlaps.items()}
    source_flow.append({"db": "mimic", "stage": "source_patient_split_disjointness", "split": "all",
                        "n_unique_patients_split0": len(source_patients_by_split[0]),
                        "n_unique_patients_split1": len(source_patients_by_split[1]),
                        "n_unique_patients_split2": len(source_patients_by_split[2]),
                        **{f"n_patient_overlap_{name}": count for name, count in overlap_counts.items()},
                        "status": "passed" if not any(overlap_counts.values()) else "failed"})
    if any(overlap_counts.values()):
        raise ValueError(f"MIMIC source patients overlap frozen splits: {overlap_counts}")
    mimic_vocab = [str(x) for x in json.loads(VOCAB_PATHS["mimic"].read_text(encoding="utf-8"))["vars"]]
    eicu_vocab = [str(x) for x in json.loads(VOCAB_PATHS["eicu"].read_text(encoding="utf-8"))["vars"]]
    vocab_match = mimic_vocab == eicu_vocab
    target_flow.append({"db": "eicu", "split": "all", "stage": "ordered_variable_vocabulary_check",
                        "source_vocabulary_count": len(mimic_vocab), "target_vocabulary_count": len(eicu_vocab),
                        "ordered_source_target_vocabulary_match": vocab_match,
                        "status": "passed" if vocab_match else "failed"})
    if not vocab_match:
        raise ValueError("MIMIC/eICU ordered variable vocabularies differ; positional V+P summaries are not comparable")
    mimic_nvars = len(mimic_vocab)
    eicu_nvars = len(eicu_vocab)
    log(f"bridge_labels source={len(mimic_roster)} target={len(eicu_roster)}")
    cm = load_cache("mimic", mimic_ds)
    ce = load_cache("eicu", eicu_ds)
    cm_rows = set(cm["rows"].astype(int))
    ce_rows = set(ce["rows"].astype(int))
    source_before_cache = len(mimic_roster)
    target_before_cache = len(eicu_roster)
    target_admin = eicu_roster["admin_hospital_admit_after_h24"].astype("boolean")
    target_flow.append({"db": "eicu", "split": "all", "stage": "endpoint_label_roster_before_cache",
                        "n_stays": int(len(eicu_roster)), "n_patients": int(eicu_roster["subject"].nunique()),
                        "n_no_admin_after_h24": int(target_admin.eq(False).fillna(False).sum()),
                        "n_admin_after_h24": int(target_admin.eq(True).fillna(False).sum()),
                        "n_admin_after_h24_unknown": int(target_admin.isna().sum()),
                        "n_unit_deaths": int(eicu_roster["icu_y"].eq(1).sum()),
                        "n_alive_unit_exits": int(eicu_roster["icu_y"].eq(0).sum())})
    mimic_roster = mimic_roster.loc[mimic_roster["rows"].isin(cm_rows)].copy()
    eicu_roster = eicu_roster.loc[eicu_roster["rows"].isin(ce_rows)].copy()
    source_flow.append({"db": "mimic", "split": "all", "stage": "frozen_cache_intersection",
                        "n_label_eligible_before": source_before_cache, "n_model_eligible": len(mimic_roster),
                        "n_rows_without_cache": source_before_cache - len(mimic_roster)})
    target_flow.append({"db": "eicu", "split": "all", "stage": "frozen_cache_intersection",
                        "n_label_eligible_before": target_before_cache, "n_model_eligible": len(eicu_roster),
                        "n_rows_without_cache": target_before_cache - len(eicu_roster),
                        "n_patients": int(eicu_roster["subject"].nunique()),
                        "n_no_admin_after_h24": int(eicu_roster["admin_hospital_admit_after_h24"].eq(False).fillna(False).sum()),
                        "n_admin_after_h24": int(eicu_roster["admin_hospital_admit_after_h24"].eq(True).fillna(False).sum()),
                        "n_admin_after_h24_unknown": int(eicu_roster["admin_hospital_admit_after_h24"].isna().sum())})
    cmx = cm.set_index("rows")["embedding"].to_dict()
    cex = ce.set_index("rows")["embedding"].to_dict()
    source_labels = mimic_roster.set_index("rows")["icu_y"].astype(np.int8)
    source_split = mimic_roster.set_index("rows")["split"].astype(np.int8)
    source_rows = source_labels.index.to_numpy(dtype=np.int64)
    source_rows = np.asarray([r for r in source_rows if r in cmx], dtype=np.int64)
    if len(source_rows) == 0:
        raise ValueError("MIMIC ICU bridge labels do not overlap frozen FM cache")
    source_labels = source_labels.loc[source_rows].to_numpy(dtype=np.int8)
    source_split = source_split.loc[source_rows].to_numpy(dtype=np.int8)
    src_ix = source_rows
    if not np.isin(source_split, [0,1,2]).all():
        raise ValueError("MIMIC source split values outside 0/1/2")
    train = source_split == 0; valid = source_split == 1; test = source_split == 2
    if not train.any() or not valid.any() or not test.any() or len(np.unique(source_labels[train])) < 2:
        raise ValueError("MIMIC ICU endpoint lacks split0/1/2 or both split0 classes")
    X_fm = np.stack([cmx[int(r)] for r in src_ix])
    target_rows_all = eicu_roster["rows"].to_numpy(dtype=np.int64)
    target_rows_keep = np.asarray([r for r in target_rows_all if r in cex], dtype=np.int64)
    eicu_roster = eicu_roster.set_index("rows").loc[target_rows_keep].reset_index()
    target_rows_all = eicu_roster["rows"].to_numpy(dtype=np.int64)
    target = eicu_roster.copy()
    # Keep only target rows with historical frozen FM and LR scores, forming one paired endpoint roster.
    hist = {}
    for model in ("fm", "lr"):
        pred = prediction_data[model].loc[prediction_data[model]["db"].eq(COHORTS["eicu"])].copy()
        pred = pred.rename(columns={"p": f"p_{model}", "y": f"y_hospital_{model}"})
        hist[model] = pred[["db", "stay_id", "subject", "rows", f"p_{model}"]]
    target["db"] = COHORTS["eicu"]
    target = target.merge(hist["fm"], on=["db", "stay_id", "subject", "rows"], how="left", validate="one_to_one", indicator="hist_fm")
    target = target.merge(hist["lr"], on=["db", "stay_id", "subject", "rows"], how="left", validate="one_to_one", indicator="hist_lr")
    target = target.loc[target["hist_fm"].eq("both") & target["hist_lr"].eq("both")].copy()
    target_rows_all = target["rows"].to_numpy(dtype=np.int64)
    target_y = target["icu_y"].to_numpy(dtype=np.int8)
    if len(target) == 0:
        raise ValueError("no common eICU ICU-death roster with historical FM and LR scores")
    target_flow.append({"db": "eicu", "split": "all", "stage": "common_historical_FM_LR_prediction_roster",
                        "n_model_eligible": len(target), "n_patients": int(target["subject"].nunique()),
                        "n_unit_deaths": int(target["icu_y"].eq(1).sum()), "n_alive_unit_exits": int(target["icu_y"].eq(0).sum()),
                        "n_no_admin_after_h24": int(target["admin_hospital_admit_after_h24"].eq(False).fillna(False).sum()),
                        "n_admin_after_h24": int(target["admin_hospital_admit_after_h24"].eq(True).fillna(False).sum()),
                        "n_admin_after_h24_unknown": int(target["admin_hospital_admit_after_h24"].isna().sum())})
    target_fm = np.stack([cex[int(r)] for r in target_rows_all])
    try:
        sys.path.insert(0, str(R4 / "baselines"))
        from baseline_features import summarize_inputs
    finally:
        if sys.path and sys.path[0] == str(R4 / "baselines"):
            sys.path.pop(0)
    mimic_vp_src, _ = summarize_inputs(mimic_ds, src_ix, mimic_nvars)
    target_vp, _ = summarize_inputs(eicu_ds, target_rows_all, eicu_nvars)
    # V+P split arrays use exactly the same MIMIC endpoint rows and source split decisions.
    src_fm_train, src_fm_valid, src_fm_test = X_fm[train], X_fm[valid], X_fm[test]
    src_vp_train, src_vp_valid, src_vp_test = mimic_vp_src[train], mimic_vp_src[valid], mimic_vp_src[test]
    y0, y1, y2 = source_labels[train], source_labels[valid], source_labels[test]
    source_metric_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    bridge_heads: dict[str, np.ndarray] = {"hospital_fm": target["p_fm"].to_numpy(dtype=float),
                                          "hospital_vp_lr": target["p_lr"].to_numpy(dtype=float)}
    bridge_prob_by_model: dict[str, pd.Series] = {
        "hospital_fm": pd.Series(target["p_fm"].to_numpy(dtype=float), index=target_rows_all),
        "hospital_vp_lr": pd.Series(target["p_lr"].to_numpy(dtype=float), index=target_rows_all),
    }
    for name, x0, x1, x2, xt in (("icu_fm", src_fm_train, src_fm_valid, src_fm_test, target_fm),
                                ("icu_vp_lr", src_vp_train, src_vp_valid, src_vp_test, target_vp)):
        p2, pt, _, fit_meta, tuning = score_model(x0, y0, x1, y1, x2, y2, xt, name)
        selection_rows.extend(tuning)
        source_metric_rows.append({"model": name, "selected_C": fit_meta["selected_C"], "split": 2,
                                   **metric_summary(y2, p2)})
        bridge_heads[name] = pt
        bridge_prob_by_model[name] = pd.Series(pt, index=target_rows_all)
    target_metric_rows = []
    for name, p in bridge_heads.items():
        target_metric_rows.append({"model": name, "endpoint": "eICU explicit unit discharge status among H24-at-risk unit stays",
                                   **metric_summary(target_y, p)})
    target_sensitivity_rows: list[dict[str, Any]] = []
    common_admin_flag = target["admin_hospital_admit_after_h24"].astype("boolean")
    common_no_admin = common_admin_flag.eq(False).fillna(False).to_numpy(dtype=bool)
    common_admin_excluded = common_admin_flag.eq(True).fillna(False)
    common_admin_unknown = common_admin_flag.isna()
    for name, p in bridge_heads.items():
        target_sensitivity_rows.append({
            "analysis_scope": "common_historical_FM_LR_prediction_roster_all_stays",
            "population": "no_admin_after_h24", "model": name, "method": "raw_endpoint_score",
            "n_common_stays": int(len(target)), "n_common_patients": int(target["subject"].nunique()),
            "n_no_admin_after_h24": int(common_no_admin.sum()),
            "n_admin_after_h24_excluded": int(common_admin_excluded.sum()),
            "n_admin_after_h24_unknown_excluded": int(common_admin_unknown.sum()),
            **metric_summary(target_y[common_no_admin], p[common_no_admin]),
        })
    # One index unit stay per patient is selected by minimum numeric stay ID, without outcome use.
    target["stay_sort"] = pd.to_numeric(target["stay_id"], errors="coerce")
    target = target.sort_values(["subject", "stay_sort", "stay_id"], kind="mergesort").drop_duplicates("subject", keep="first").copy()
    target = target.sort_values("subject", kind="mergesort").reset_index(drop=True)
    target_flow.append({"db": "eicu", "split": "all", "stage": "one_index_unit_episode_per_common_patient",
                        **bridge_admin_flow_counts(target),
                        "selection_rule": "minimum numeric stay_id within patient, selected before outcomes and without filtering on administrative flag"})
    subjects = target["subject"].astype(str).to_numpy()
    unique_subjects = np.unique(subjects)
    if len(unique_subjects) <= 500:
        raise ValueError(f"eICU common target roster has only {len(unique_subjects)} unique patients; cannot lock 500 adaptation labels")
    rng = np.random.default_rng(SEED)
    adapt_subjects = np.sort(rng.choice(unique_subjects, size=500, replace=False))
    adapt = np.isin(subjects, adapt_subjects)
    assess = ~adapt
    for role, role_mask in (("adaptation", adapt), ("assessment", assess)):
        role_frame = target.loc[role_mask]
        role_counts = bridge_admin_flow_counts(role_frame)
        target_flow.append({"db": "eicu", "split": "all", "stage": f"one_index_patient_{role}_role",
                            "role": role, **role_counts,
                            "n_no_admin_sensitivity_excluded": int(role_counts["n_admin_after_h24"] + role_counts["n_admin_after_h24_unknown"])})
    calibration_rows: list[dict[str, Any]] = []
    calibrated: dict[str, dict[str, np.ndarray]] = {}
    for model, p in bridge_heads.items():
        p = target["rows"].map(bridge_prob_by_model[model]).to_numpy(dtype=float)
        y_adapt = target.loc[adapt, "icu_y"].to_numpy(dtype=np.int8)
        p_adapt = p[adapt]
        y_assess = target.loc[assess, "icu_y"].to_numpy(dtype=np.int8)
        p_assess = p[assess]
        calibrated[model] = {"raw": p_assess}
        for method in ("prevalence", "intercept", "platt"):
            q, meta = recalibrate(p_adapt, y_adapt, method)
            # Reuse coefficients estimated on adaptation patients only; the
            # helper keeps a valid slope of exactly zero distinct from missing.
            q_assess = apply_recalibration(p_assess, meta)
            calibrated[model][method] = q_assess
            calibration_rows.append({"model": model, "method": method, "adaptation_patients": int(adapt.sum()),
                                     "adaptation_events": int(y_adapt.sum()), "assessment_patients": int(assess.sum()),
                                     "assessment_events": int(y_assess.sum()), "offset": meta.get("offset"),
                                     "intercept": meta.get("intercept"), "slope": meta.get("slope"),
                                     "fit_status": meta.get("status"), "fit_fallback_reason": meta.get("fallback_reason"),
                                     **metric_summary(y_assess, q_assess)})
        calibration_rows.append({"model": model, "method": "no_update", "adaptation_patients": int(adapt.sum()),
                                 "adaptation_events": int(y_adapt.sum()), "assessment_patients": int(assess.sum()),
                                 "assessment_events": int(y_assess.sum()), "intercept": None, "slope": None,
                                 "fit_status": "frozen_source_probability", **metric_summary(y_assess, p_assess)})
    # Matched assessment patient bootstrap for ICU FM versus V+P LR, before and after Platt update.
    paired_rows: list[dict[str, Any]] = []
    y_assess = target.loc[assess, "icu_y"].to_numpy(dtype=np.int8)
    subject_assess = target.loc[assess, "subject"].astype(str).to_numpy()
    assessment_admin_flag = target.loc[assess, "admin_hospital_admit_after_h24"].astype("boolean")
    no_admin_assess = assessment_admin_flag.eq(False).fillna(False).to_numpy(dtype=bool)
    n_admin_assess_excluded = int(assessment_admin_flag.eq(True).fillna(False).sum())
    n_admin_assess_unknown = int(assessment_admin_flag.isna().sum())
    n_no_admin_assess = int(no_admin_assess.sum())
    for model, predictions in calibrated.items():
        for method, probabilities in predictions.items():
            target_sensitivity_rows.append({
                "analysis_scope": "one_index_unit_episode_per_patient_assessment_role",
                "population": "no_admin_after_h24", "model": model, "method": method,
                "adaptation_patients_total": int(adapt.sum()),
                "assessment_patients_total": int(assess.sum()),
                "n_no_admin_assessment_patients": n_no_admin_assess,
                "n_admin_after_h24_assessment_excluded": n_admin_assess_excluded,
                "n_admin_after_h24_unknown_assessment_excluded": n_admin_assess_unknown,
                **metric_summary(y_assess[no_admin_assess], probabilities[no_admin_assess]),
            })
    pair_specs = (("hospital_fm", "hospital_vp_lr", "historical hospital FM minus LR"),
                  ("icu_fm", "icu_vp_lr", "new ICU FM minus V+P LR"),
                  ("hospital_fm", "icu_fm", "historical hospital FM minus new ICU FM"),
                  ("hospital_vp_lr", "icu_vp_lr", "historical hospital LR minus new ICU V+P LR"))
    for method in ("raw", "platt"):
        for left, right, label in pair_specs:
            for metric_row in brier_logloss_ci(subject_assess, y_assess, calibrated[left][method],
                                               calibrated[right][method], BOOT_DRAWS, SEED + 101):
                paired_rows.append({"contrast": label, "method": method,
                                    "analysis_subset": "all_assessment_patients", **metric_row})
    if n_no_admin_assess:
        for method in ("raw", "platt"):
            for left, right, label in pair_specs:
                for metric_row in brier_logloss_ci(subject_assess[no_admin_assess], y_assess[no_admin_assess],
                                                   calibrated[left][method][no_admin_assess],
                                                   calibrated[right][method][no_admin_assess],
                                                   BOOT_DRAWS, SEED + 201):
                    paired_rows.append({"contrast": label, "method": method,
                                        "analysis_subset": "no_admin_after_h24_assessment", **metric_row})
    return {
        "source_flow": source_flow,
        "source_conflicts": source_conflicts,
        "target_flow": target_flow,
        "source_metrics": source_metric_rows,
        "source_tuning": selection_rows,
        "target_metrics": target_metric_rows,
        "target_sensitivity": target_sensitivity_rows,
        "target_calibration": calibration_rows,
        "target_paired_bootstrap": paired_rows,
        "roles": [{"role": "adaptation", "n_patients": 500, "seed": SEED, "sampling": "500 unique target patients, one minimum numeric stay_id per patient before outcomes; outcome-blind and method-matched"},
                  {"role": "assessment", "n_patients": int(assess.sum()), "seed": SEED, "sampling": "all remaining unique target patients; disjoint from adaptation"}],
        "endpoint": {"source": "MIMIC exact admissions.deathtime within one assigned icustays.intime/outtime index episode, after H24; deathtime missing, unparsable, and valid are reported separately; hospital_expire_flag=0 defines survival only with genuinely missing deathtime, or exact death after unit exit defines survival; flag=0 with valid deathtime is excluded; deathtime equal to ICU outtime counts within the exiting unit episode", "target": "eICU exact unitdischargestatus whitelist Alive/Expired among finite unitdischargeoffset > 1440 minute stays; unit-expired/hospital-alive conflicts excluded; unknown/unrecognized status or nonfinite offset excluded", "sicdb": "not a verified ICU-death endpoint and excluded from this bridge", "hospitaladmitoffset": "finite offset > 1440 minutes marks admin-after-H24; missing or unparsable values stay unknown and are excluded from no-admin sensitivity; adaptation and assessment roles are frozen before filtering"},
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if not rows:
        pd.DataFrame().to_csv(temporary, index=False)
    else:
        pd.DataFrame(rows).to_csv(temporary, index=False, quoting=csv.QUOTE_MINIMAL)
    os.replace(temporary, path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def stage_status(path: Path, stage: str, status: str, **details: Any) -> None:
    payload = {"stage": stage, "status": status, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "runtime_limits": RUNTIME_LIMITS, **details}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def input_identity(mode: str) -> dict[str, Any]:
    # Large sequence/cache files carry upstream frozen SHA attestations. Avoid
    # an extra full-file hash pass over the pressured data volume.
    paths = [*SEQ_PATHS.values(), *SPLIT_PATHS.values(),
             *[ELIG_DIR / f"{ELIG_NAMES[d]}.parquet" for d in COHORTS]]
    manifest_paths = list(PREDICTION_MANIFEST_PATHS)
    if mode == "bridge":
        paths.extend([MANIFEST_PATH, MERGE_AUDIT_PATH, *CACHE_PATHS.values()])
    paths.extend(manifest_paths)

    # Cache the one filesystem metadata check for every distinct input path.
    # Missing and non-regular paths keep the historical is_file() result;
    # other stat/read errors propagate so identity collection fails closed.
    import stat

    unique_paths = list(dict.fromkeys([*paths, ELIG_DIR / "eligibility_manifest.json"]))
    path_stats: dict[Path, os.stat_result | None] = {}
    for path in unique_paths:
        try:
            info = path.stat()
        except (FileNotFoundError, NotADirectoryError):
            path_stats[path] = None
        else:
            path_stats[path] = info if stat.S_ISREG(info.st_mode) else None

    result: dict[str, Any] = {}
    hash_paths = set(manifest_paths)
    if mode == "bridge":
        hash_paths.update((MANIFEST_PATH, MERGE_AUDIT_PATH))
    elig_manifest = ELIG_DIR / "eligibility_manifest.json"
    hash_paths.add(elig_manifest)
    json_paths = set(manifest_paths)
    json_paths.add(elig_manifest)
    if mode == "bridge":
        json_paths.add(MERGE_AUDIT_PATH)
    json_by_path: dict[Path, Any] = {}
    sha_by_path: dict[Path, str] = {}
    for path in unique_paths:
        info = path_stats[path]
        if info is None or path not in hash_paths:
            continue
        raw = path.read_bytes()
        if len(raw) != info.st_size:
            raise OSError(f"file changed while collecting input identity: {path}")
        sha_by_path[path] = hashlib.sha256(raw).hexdigest()
        if path in json_paths:
            json_by_path[path] = json.loads(raw.decode("utf-8"))

    for path in paths:
        info = path_stats[path]
        result[str(path)] = {"exists": info is not None,
                             "size_bytes": info.st_size if info is not None else None}
        if info is not None and path in hash_paths:
            result[str(path)]["sha256"] = sha_by_path[path]

    prediction_attestations: dict[Path, set[str]] = {}
    resolved_relative_candidates: dict[Path, Path] = {}
    for manifest_path in manifest_paths:
        if manifest_path not in json_by_path:
            continue
        data = json_by_path[manifest_path]
        for record in data.get("predictions", []):
            if not isinstance(record, dict) or not record.get("prediction_file"):
                continue
            candidate = Path(record["prediction_file"])
            if candidate.is_absolute():
                prediction_path = candidate
            else:
                if candidate not in resolved_relative_candidates:
                    resolved_relative_candidates[candidate] = (R4 / candidate).resolve()
                prediction_path = resolved_relative_candidates[candidate]
            expected = prediction_attestations.setdefault(prediction_path, set())
            if record.get("prediction_file_sha256"):
                expected.add(str(record["prediction_file_sha256"]))

    for path in sorted(prediction_attestations, key=str):
        if path not in path_stats:
            try:
                info = path.stat()
            except (FileNotFoundError, NotADirectoryError):
                path_stats[path] = None
            else:
                path_stats[path] = info if stat.S_ISREG(info.st_mode) else None
        info = path_stats[path]
        result[str(path)] = {"exists": info is not None,
                             "size_bytes": info.st_size if info is not None else None,
                             "sha256_attested_by_prediction_manifest": None}
        expected = prediction_attestations[path]
        if expected:
            result[str(path)]["sha256_attested_by_prediction_manifest"] = sorted(expected)

    if path_stats[elig_manifest] is not None:
        data = json_by_path[elig_manifest]
        result["eligibility_manifest"] = {
            "path": str(elig_manifest),
            "sha256": sha_by_path[elig_manifest],
            "cohorts": {key: value.get("sourcehashes", {})
                        for key, value in data.get("cohorts", {}).items()},
        }
    if mode == "bridge" and path_stats[MERGE_AUDIT_PATH] is not None:
        audit = json_by_path[MERGE_AUDIT_PATH]
        for record in audit.get("cache_files", []):
            cache_path = str(record.get("path", ""))
            if cache_path in result:
                result[cache_path]["sha256_attested_by_frozen_merge_audit"] = record.get("sha256")
    return result


def audit_stage() -> None:
    status_h24 = OUT_TEMP / "stage_status.json"
    status_cohort = OUT_COHORT / "stage_status.json"
    manifest_h24 = OUT_TEMP / "h24_manifest.json"
    manifest_cohort = OUT_COHORT / "cohort_manifest.json"
    current_inputs = input_identity("audit")
    if manifest_h24.is_file() and manifest_cohort.is_file():
        old = json.loads(manifest_h24.read_text(encoding="utf-8"))
        if old.get("status") == "complete":
            if old.get("inputs") != current_inputs:
                raise ValueError("completed audit output input identity differs; refusing silent reuse")
            log("audit_stage_already_complete_outputs_preserved")
            return
    for out in (OUT_TEMP, OUT_COHORT):
        if out.exists():
            status = out / "stage_status.json"
            if status.is_file() and json.loads(status.read_text(encoding="utf-8")).get("status") == "running":
                raise RuntimeError(f"audit stage already marked running: {status}")
            if any(out.iterdir()) and not status.is_file():
                raise FileExistsError(f"unowned output prevents rerun: {out}")
        out.mkdir(parents=True, exist_ok=True)
    stage_status(status_h24, "h24_and_raw_field_audit", "running", started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    stage_status(status_cohort, "clinical_flow_and_sequence_support", "running", started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    inputs = current_inputs
    log("audit_stage_start")
    predictions = prediction_frames()
    log("prediction_artifacts_read_complete")
    all_flows: list[dict[str, Any]] = []
    all_flags: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    clinical_rows: list[dict[str, Any]] = []
    clinical_cat_rows: list[dict[str, Any]] = []
    time_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    variable_rows: list[dict[str, Any]] = []
    input_checks: list[dict[str, Any]] = []
    elig_by_db: dict[str, pd.DataFrame] = {}
    completed_cohorts: list[str] = []
    for db in COHORTS:
        log(f"cohort_stage={db}_read_begin")
        elig = read_eligibility(db)
        ds, split = read_sequence(db, full=True)
        idx = sequence_index(db, ds, split)
        bound = bind_predictions(db, elig, idx, predictions)
        flow, flags, metrics = analyze_predictions(db, elig, bound, idx)
        all_flows.extend(flow); all_flags.extend(flags); all_metrics.extend(metrics)
        clinical, cats, times = clinical_support(db, elig, idx[["db", "stay_id", "subject", "split_expected"]])
        clinical_rows.extend(clinical); clinical_cat_rows.extend(cats); time_rows.extend(times)
        seq_support, vars_support, input_support = sequence_support(db, elig, ds, split)
        sequence_rows.extend(seq_support); variable_rows.extend(vars_support); input_checks.extend(input_support)
        elig_by_db[db] = elig
        completed_cohorts.append(db)
        log(f"cohort_stage={db}_complete sequences={len(ds['L'])} eligible={int(elig['eligible_24h'].fillna(False).sum())}")
        del ds
        # Preserve aggregate progress after each cohort; each file is replaced atomically.
        write_csv(OUT_TEMP / "h24_split_flow.csv", all_flows)
        write_csv(OUT_TEMP / "h24_flag_intersections.csv", all_flags)
        write_csv(OUT_TEMP / "h24_performance_metrics_paired_losses.csv", all_metrics)
        write_csv(OUT_COHORT / "clinical_flow_and_missingness.csv", clinical_rows)
        write_csv(OUT_COHORT / "clinical_category_support.csv", clinical_cat_rows)
        write_csv(OUT_COHORT / "outcome_transfer_time_support.csv", time_rows)
        write_csv(OUT_COHORT / "sequence_support.csv", sequence_rows)
        write_csv(OUT_COHORT / "variable_coverage.csv", variable_rows)
        write_csv(OUT_COHORT / "input_time_validation.csv", input_checks)
        stage_status(status_h24, "h24_and_raw_field_audit", "running", completed_cohorts=list(completed_cohorts))
        stage_status(status_cohort, "clinical_flow_and_sequence_support", "running",
                     completed_cohorts=list(completed_cohorts))
    write_csv(OUT_TEMP / "h24_split_flow.csv", all_flows)
    write_csv(OUT_TEMP / "h24_flag_intersections.csv", all_flags)
    write_csv(OUT_TEMP / "h24_performance_metrics_paired_losses.csv", all_metrics)
    write_csv(OUT_COHORT / "clinical_flow_and_missingness.csv", clinical_rows)
    write_csv(OUT_COHORT / "clinical_category_support.csv", clinical_cat_rows)
    write_csv(OUT_COHORT / "outcome_transfer_time_support.csv", time_rows)
    write_csv(OUT_COHORT / "sequence_support.csv", sequence_rows)
    write_csv(OUT_COHORT / "variable_coverage.csv", variable_rows)
    write_csv(OUT_COHORT / "input_time_validation.csv", input_checks)
    log("h24_and_clinical_support_written")
    nw_rows, nw_categories = raw_nwicu_audit(elig_by_db["nwicu"])
    sic_rows, sic_categories = raw_sicdb_audit(elig_by_db["sicdb"])
    write_csv(OUT_TEMP / "nwicu_mortality_raw_audit.csv", nw_rows)
    write_csv(OUT_TEMP / "nwicu_discharge_location_values.csv", nw_categories)
    write_csv(OUT_TEMP / "sicdb_case_field_qualification.csv", sic_rows)
    write_csv(OUT_TEMP / "sicdb_raw_state_and_reference_values.csv", sic_categories)
    log("raw_outcome_field_audits_written")
    readme = (
        "# V5 temporal and raw endpoint audit\n\n"
        "MIMIC source metrics use the original MIMIC FM and conventional source prediction artifacts. "
        "External predictions are loaded through the recovered transport validator. Every cohort binds by database, stay, subject, sequence row, split, and label; mismatched prediction sets are not intersected. "
        "MIMIC source performance is limited to split 2; external metrics use the full reviewed roster. Frozen external partitions are not temporal validation.\n\n"
        "H24 subsets include the reviewed cohort, key exclusions, and flagged date-overlap, exit-gap, and administrative hospital-admit-after-H24 rows. "
        "The hospital-admit flag remains a descriptive flag; no-admin-after-H24 is a reported sensitivity subset. Per-model metrics include AUROC, Brier, log loss, and calibration point estimates. "
        "Paired FM versus LR Brier and log-loss differences use patient-cluster CIs with stay-weighted primary estimates and patient-weighted sensitivity estimates. "
        "The ICU bridge reports both weightings for paired loss differences; its 500-label calibration assessment uses one index episode per patient.\n\n"
        "MIMIC calendar dates are de-identified and are not interpreted as true cross-patient calendar time. Only severity fields present in the reviewed table are summarized; no unified score is inferred. "
        "NWICU corrected mortality uses the documented known-alive and known-dead discharge-location whitelist; historical exact-string disagreements, corrected known-endpoint disagreements, and corrected labels versus frozen y_mort are reported separately with unknown values retained. "
        "SICdb frozen-label counts use neutral names and are not interpreted as verified hospital or ICU deaths. SICdb DischargeState and OffsetOfDeath remain case-discharge/death-proxy fields; zero ICUOffset and case closing time do not verify continuous ICU occupancy at H24 or an index-ICU exit.\n"
        "Each stage is launched with a 3 GiB systemd memory cap, no swap, a 200% CPU quota, and the shared nonblocking serial lock.\n\n"
    )
    readme_path = OUT_TEMP / "README.md"
    readme_temp = readme_path.with_name(f".{readme_path.name}.tmp")
    readme_temp.write_text(readme, encoding="utf-8")
    os.replace(readme_temp, readme_path)
    files_h24 = [str(p) for p in sorted(OUT_TEMP.glob("*.csv"))]
    files_cohort = [str(p) for p in sorted(OUT_COHORT.glob("*.csv"))]
    common = {"run_id": RUN_ID, "status": "complete", "seed": SEED,
              "runtime_limits": RUNTIME_LIMITS,
              "inputs": inputs, "prediction_provenance": PREDICTION_PROVENANCE,
              "aggregate_only": True, "patient_level_rows_written": False}
    write_json_atomic(manifest_h24, {**common, "scope": "H24 sensitivities and raw endpoint fields",
                                     "performance_scope": "MIMIC source split 2 only; external full reviewed roster",
                                     "prediction_binding": "transport-validated original source MIMIC artifacts plus recovered external artifacts; exact db/stay/subject/row/split/label identity is rechecked and mismatched rows are never intersected",
                                     "split_interpretation": "frozen external partitions are not temporal validation; MIMIC split 2 is the frozen source test partition, with chronological ordering not asserted",
                                     "files": files_h24})
    write_json_atomic(manifest_cohort, {**common, "scope": "clinical flow, outcome/transfer boundaries, and exported sequence-time/coverage support",
                                        "severity_fields": "only available source fields are summarized; no unified score is inferred",
                                        "record_time_availability": "sequence hour bounds do not quantify late-store availability or retained-token exposure",
                                        "MIMIC_date_interpretation": "calendar dates are de-identified and are not interpreted as true cross-patient calendar time",
                                        "files": files_cohort})
    stage_status(status_h24, "h24_and_raw_field_audit", "complete", files=files_h24, input_count=len(inputs))
    stage_status(status_cohort, "clinical_flow_and_sequence_support", "complete", files=files_cohort, input_count=len(inputs))
    log("audit_stage_complete")


def bridge_stage() -> None:
    status_path = OUT_BRIDGE / "stage_status.json"
    manifest_path = OUT_BRIDGE / "bridge_manifest.json"
    inputs = input_identity("bridge")
    if manifest_path.is_file():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("status") == "complete":
            if old.get("inputs") != inputs:
                raise ValueError("completed bridge output input identity differs; refusing silent reuse")
            log("bridge_stage_already_complete_outputs_preserved")
            return
    if OUT_BRIDGE.exists() and any(OUT_BRIDGE.iterdir()):
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        if status.get("status") == "running":
            raise RuntimeError(f"bridge stage already marked running: {status_path}")
    OUT_BRIDGE.mkdir(parents=True, exist_ok=True)
    stage_status(status_path, "MIMIC_to_eICU_ICU_endpoint_bridge", "running", started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    log("bridge_stage_start")
    predictions = prediction_frames()
    mimic_ds, mimic_split = read_sequence("mimic", full=True)
    eicu_ds, eicu_split = read_sequence("eicu", full=True)
    bridge = run_bridge(mimic_ds, mimic_split, eicu_ds, eicu_split, predictions)
    write_csv(OUT_BRIDGE / "mimic_source_icu_head_flow.csv", bridge["source_flow"])
    write_csv(OUT_BRIDGE / "mimic_source_conflict_cross_tab.csv", bridge["source_conflicts"])
    write_csv(OUT_BRIDGE / "eicu_target_unit_flow.csv", bridge["target_flow"])
    write_csv(OUT_BRIDGE / "source_split2_icu_head_metrics.csv", bridge["source_metrics"])
    write_csv(OUT_BRIDGE / "source_split1_C_selection.csv", bridge["source_tuning"])
    write_csv(OUT_BRIDGE / "eicu_common_roster_icu_metrics.csv", bridge["target_metrics"])
    write_csv(OUT_BRIDGE / "eicu_no_admin_after_h24_sensitivity.csv", bridge["target_sensitivity"])
    write_csv(OUT_BRIDGE / "eicu_500label_calibration_assessment.csv", bridge["target_calibration"])
    write_csv(OUT_BRIDGE / "eicu_icu_paired_patient_bootstrap.csv", bridge["target_paired_bootstrap"])
    write_csv(OUT_BRIDGE / "eicu_patient_roles.csv", bridge["roles"])
    files = [str(p) for p in sorted(OUT_BRIDGE.glob("*.csv"))]
    manifest = {"run_id": RUN_ID, "status": "complete", "seed": SEED,
                "runtime_limits": RUNTIME_LIMITS,
                "bootstrap_draws": BOOT_DRAWS, "inputs": inputs, "bridge": bridge["endpoint"],
                "prediction_provenance": PREDICTION_PROVENANCE,
                "source_score_scope": "new MIMIC ICU head source performance on split 2 only",
                "target_score_scope": "common eICU unit ICU-death roster with historical hospital FM/LR and new ICU FM/LR scores",
                "files": files}
    write_json_atomic(manifest_path, manifest)
    stage_status(status_path, "MIMIC_to_eICU_ICU_endpoint_bridge", "complete", files=files, input_count=len(inputs))
    log("bridge_stage_complete")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "audit"
    if mode not in {"audit", "bridge"}:
        raise ValueError("mode must be audit or bridge")
    if mode != RUN_MODE:
        raise ValueError(f"runner mode {RUN_MODE!r} does not match requested stage {mode!r}")
    try:
        if mode == "audit":
            audit_stage()
        else:
            bridge_stage()
    except BaseException as exc:
        out = OUT_BRIDGE if mode == "bridge" else OUT_TEMP
        try:
            stage_status(out / "stage_status.json", mode, "failed", error_type=type(exc).__name__,
                         error=str(exc), traceback=traceback.format_exc())
            if mode == "audit":
                stage_status(OUT_COHORT / "stage_status.json", "clinical_flow_and_sequence_support", "failed",
                             error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        except Exception:
            pass
        log(f"stage_failed mode={mode} error={type(exc).__name__}:{exc}")
        raise


if __name__ == "__main__":
    main()
