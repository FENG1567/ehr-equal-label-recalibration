#!/usr/bin/env python3
"""Build corrected, versioned Challenge 2012 A/B/C artifacts.

This builder is intentionally isolated from the legacy processed files. It
reads the source records and outcomes, applies the canonical v4 variable
definitions and frozen bins, and writes one versioned directory plus a
C2012-owned registry fragment. It does not train or score a model.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
import os
import re
import tarfile
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd


PROJECT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = PROJECT / "revision_v4_20261122"
EXT = REV / "c2012_extension"
INPUT = EXT / "input"
CHALLENGE = PROJECT / "data/external/challenge2012"
EXTRACT_SOURCE = PROJECT / "code/extract.py"
FROZEN_BINS = PROJECT / "data/processed/bins.json"
LEGACY_A_SEQUENCE = PROJECT / "data/processed/seq_c2012.npz"
ENGINE_CODE = REV / "engine/code"
OUTPUT_ROOT = EXT / "results/c2012_v4_1"
REGISTRY_FRAGMENT = EXT / "registry_c2012_v4.json"

MAXH = 24
MAXLEN = 128
NBINS = 10
SEED = 20261102
BIN_VARS = {"vaso", "rrt", "sedation", "sex"}
INCLUSION_RULE = (
    "PhysioNet Challenge 2012 sets A/B/C include adult ICU stays of at least 48 hours"
)
IDENTITY_BASIS = (
    "RecordID is used only as a stay/admission proxy; person identity is unverified"
)

# Challenge 2012 names are mapped only where the canonical v4 feature has the
# same clinical meaning. In particular, HCT is not haemoglobin and HCO3 is not
# base excess; neither is mapped. SaO2 is not assumed to be peripheral SpO2.
# These, MechVent and troponins remain provenance-only unmapped source variables.
PARAMETER_MAP = {
    "age": "age",
    "gender": "sex",
    "height": "height",
    "weight": "weight",
    "hr": "hr",
    "sysabp": "sbp",
    "nisysabp": "sbp",
    "diasabp": "dbp",
    "nidiasabp": "dbp",
    "map": "map",
    "nimap": "map",
    "resprate": "rr",
    "temp": "temp",
    "gcs": "gcs_total",
    "fio2": "fio2",
    "pao2": "pao2",
    "paco2": "paco2",
    "ph": "ph",
    "creatinine": "creatinine",
    "bun": "bun",
    "urine": "urine",
    "lactate": "lactate",
    "wbc": "wbc",
    "platelets": "platelet",
    "sodium": "sodium",
    "na": "sodium",
    "potassium": "potassium",
    "k": "potassium",
    "calcium": "calcium",
    "magnesium": "magnesium",
    "mg": "magnesium",
    "phosphate": "phosphate",
    "glucose": "glucose",
    "albumin": "albumin",
    "bilirubin": "bili",
    "ast": "ast",
    "alt": "alt",
}
HEADER_FALLBACKS = {
    "age": "age",
    "gender": "sex",
    "height": "height",
    "weight": "weight",
}


@dataclass
class RawRecord:
    record_id: int
    file_name: str
    header: dict[str, str]
    events: list[tuple[float, str, str]]
    source_sha256: str
    source_bytes: int


def canonical_parameter(name: str) -> str | None:
    """Return the supported canonical v4 variable for a Challenge name."""
    return PARAMETER_MAP.get(name.strip().casefold())


def parse_time_to_hour(value: str) -> float:
    """Parse elapsed Challenge time; hours may exceed 23 (e.g. 35:19)."""
    parts = value.strip().split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"invalid Challenge elapsed time: {value!r}")
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2]) if len(parts) == 3 else 0
    except ValueError as exc:
        raise ValueError(f"invalid Challenge elapsed time: {value!r}") from exc
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError(f"invalid Challenge elapsed time: {value!r}")
    return hours + minutes / 60.0 + seconds / 3600.0


def parse_record_text(text: str, *, file_name: str, source_sha256: str = "",
                      source_bytes: int = 0) -> RawRecord:
    """Parse one native Challenge record without altering its source text."""
    header: dict[str, str] = {}
    events: list[tuple[float, str, str]] = []
    record_id_from_series: str | None = None
    in_series = False
    for row in csv.reader(io.StringIO(text)):
        fields = [field.strip().lstrip("\ufeff") for field in row]
        if not fields or not any(fields):
            continue
        if fields[0].casefold() == "time" and len(fields) >= 2 and fields[1].casefold() == "parameter":
            in_series = True
            continue
        if not in_series:
            line = fields[0]
            if line.startswith("#") or ":" not in line:
                continue
            name, value = line.split(":", 1)
            header[name.strip().casefold()] = value.strip()
            continue
        if len(fields) < 3:
            raise ValueError(f"malformed Challenge event row in {file_name}: {fields!r}")
        time_hr = parse_time_to_hour(fields[0])
        # A source-C row has an empty parameter. Keep it so normalization
        # records it as unmapped; never infer a clinical variable name.
        parameter = fields[1].strip()
        if parameter.casefold() == "recordid":
            if record_id_from_series is not None:
                raise ValueError(f"duplicate RecordID row in {file_name}")
            record_id_from_series = fields[2].strip()
            continue
        if parameter.casefold() == "icutype":
            # ICUType is admission metadata, not one of the canonical v4
            # features. Preserve it in neither the token stream nor unmapped
            # clinical-variable QC counts.
            continue
        events.append((time_hr, parameter, fields[2]))
    raw_id = header.get("recordid") or record_id_from_series
    if raw_id is None:
        raise ValueError(f"Challenge record has no RecordID header: {file_name}")
    try:
        record_id = int(raw_id)
    except ValueError as exc:
        raise ValueError(f"invalid RecordID {raw_id!r} in {file_name}") from exc
    stem = Path(file_name).stem
    if stem.isdigit() and int(stem) != record_id:
        raise ValueError(f"RecordID {record_id} does not match filename {file_name}")
    return RawRecord(record_id, file_name, header, events, source_sha256, source_bytes)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_record_inventory(records: list[RawRecord]) -> str:
    payload = "\n".join(
        f"{record.file_name}\t{record.source_bytes}\t{record.source_sha256}"
        for record in sorted(records, key=lambda item: item.file_name)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def iter_set_records(code: str) -> tuple[list[RawRecord], dict[str, Any]]:
    """Read a source set from A's directory or B/C's immutable tar archive."""
    code = code.upper()
    records: list[RawRecord] = []
    if code == "A":
        source_dir = CHALLENGE / "set-a"
        paths = sorted(source_dir.glob("*.txt"), key=lambda path: path.name)
        for path in paths:
            content = path.read_bytes()
            record = parse_record_text(
                content.decode("utf-8-sig"), file_name=path.name,
                source_sha256=hashlib.sha256(content).hexdigest(), source_bytes=len(content),
            )
            records.append(record)
        source_file = source_dir
        source_hash = _digest_record_inventory(records)
    elif code in ("B", "C"):
        archive_path = INPUT / f"set-{code.casefold()}.tar.gz"
        prefix = f"set-{code.casefold()}/"
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive.getmembers():
                if not member.isfile() or not member.name.startswith(prefix) or not member.name.lower().endswith(".txt"):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"cannot read {member.name} from {archive_path}")
                content = extracted.read()
                file_name = Path(member.name).name
                record = parse_record_text(
                    content.decode("utf-8-sig"), file_name=file_name,
                    source_sha256=hashlib.sha256(content).hexdigest(), source_bytes=len(content),
                )
                records.append(record)
        source_file = archive_path
        source_hash = _file_digest(archive_path)
    else:
        raise ValueError(f"unsupported Challenge set: {code}")

    ids = [record.record_id for record in records]
    if not records or len(ids) != len(set(ids)):
        raise ValueError(f"set {code}: no records or duplicate RecordIDs")
    info = {
        "source_path": str(source_file),
        "source_sha256": source_hash,
        "raw_file_count": len(records),
        "raw_total_bytes": int(sum(record.source_bytes for record in records)),
        "raw_file_inventory_sha256": _digest_record_inventory(records),
        "record_ids_sha256": _id_hash(ids),
        "record_ids": sorted(ids),
    }
    return records, info


def _id_hash(ids: Iterable[int | str]) -> str:
    return hashlib.sha256("\n".join(sorted(str(item) for item in ids)).encode("utf-8")).hexdigest()


def _numeric_value(raw_value: str | None, *, parameter: str) -> tuple[float | None, str | None]:
    if raw_value is None or not raw_value.strip():
        return None, "blank"
    value = raw_value.strip()
    if parameter.casefold() == "age" and value.casefold().startswith(">"):
        match = re.search(r"\d+(?:\.\d+)?", value)
        if match:
            return float(match.group(0)) + 1.0, None
    try:
        number = float(value)
    except ValueError:
        return None, "non_numeric"
    if not math.isfinite(number):
        return None, "non_finite"
    if number == -1.0:
        return None, "missing_sentinel_minus_1"
    return number, None


def _get_extractor_definitions(path: Path = EXTRACT_SOURCE) -> tuple[list[str], dict[str, tuple[float, float]]]:
    """Read only literal VARS/RANGE assignments from the canonical extractor."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {"VARS", "RANGE"}:
                found[target.id] = ast.literal_eval(node.value)
    if set(found) != {"VARS", "RANGE"}:
        raise ValueError(f"could not read canonical VARS and RANGE from {path}")
    return found["VARS"], found["RANGE"]


def _raw_header(record: RawRecord, name: str) -> str | None:
    return record.header.get(name.casefold())


def _accept_value(value: float, variable: str, ranges: dict[str, tuple[float, float]]) -> bool:
    lo, hi = ranges.get(variable, (-1e9, 1e9))
    return lo <= value <= hi


def normalize_records(
    records: list[RawRecord], vars_: list[str], ranges: dict[str, tuple[float, float]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Map values, remove missing sentinels, clip to 0-24h, then clean ranges."""
    if not set(PARAMETER_MAP.values()).issubset(set(vars_)):
        missing = sorted(set(PARAMETER_MAP.values()).difference(vars_))
        raise ValueError(f"canonical variables are absent from extract.py: {missing}")
    events: list[tuple[int, float, str, float]] = []
    qc: dict[str, Any] = {
        "raw_event_count": 0,
        "raw_events_outside_0_24h": 0,
        "accepted_event_count": 0,
        "missing_sentinel_removed_by_parameter": Counter(),
        "blank_or_invalid_removed_by_parameter": Counter(),
        "unmapped_parameter_count_within_24h": Counter(),
        "range_drop_count_by_variable": Counter(),
        "accepted_event_count_by_variable": Counter(),
        "unit_conversion_count_by_variable": Counter(),
        "header_descriptor_fallback_count_by_variable": Counter(),
        "conversion_factors": {"fio2": 100.0, "magnesium": 2.430},
    }
    for record in records:
        valid_at_zero: set[str] = set()
        for time_hr, parameter, raw_value in record.events:
            qc["raw_event_count"] += 1
            if not 0.0 <= time_hr < MAXH:
                qc["raw_events_outside_0_24h"] += 1
                continue
            var = canonical_parameter(parameter)
            if var is None:
                qc["unmapped_parameter_count_within_24h"][parameter] += 1
                continue
            value, missing_reason = _numeric_value(raw_value, parameter=parameter)
            if value is None:
                if missing_reason == "missing_sentinel_minus_1":
                    qc["missing_sentinel_removed_by_parameter"][parameter] += 1
                else:
                    qc["blank_or_invalid_removed_by_parameter"][parameter] += 1
                continue
            if var == "fio2":
                value *= 100.0
                qc["unit_conversion_count_by_variable"][var] += 1
            elif var == "magnesium":
                value *= 2.430
                qc["unit_conversion_count_by_variable"][var] += 1
            if not _accept_value(value, var, ranges):
                qc["range_drop_count_by_variable"][var] += 1
                continue
            events.append((record.record_id, time_hr, var, value))
            qc["accepted_event_count"] += 1
            qc["accepted_event_count_by_variable"][var] += 1
            if time_hr == 0.0:
                valid_at_zero.add(var)

        # Some conforming copies place admission descriptors in the header
        # rather than repeating the corresponding 00:00 CSV row. The official
        # source defines these descriptors as admission-time values, so use a
        # header value only when no valid 00:00 event was already accepted.
        for header_name, var in HEADER_FALLBACKS.items():
            if var in valid_at_zero:
                continue
            raw_value = _raw_header(record, header_name)
            value, missing_reason = _numeric_value(raw_value, parameter=header_name)
            if value is None:
                if missing_reason == "missing_sentinel_minus_1":
                    qc["missing_sentinel_removed_by_parameter"][header_name] += 1
                continue
            if not _accept_value(value, var, ranges):
                qc["range_drop_count_by_variable"][var] += 1
                continue
            events.append((record.record_id, 0.0, var, value))
            qc["accepted_event_count"] += 1
            qc["accepted_event_count_by_variable"][var] += 1
            qc["header_descriptor_fallback_count_by_variable"][var] += 1

    frame = pd.DataFrame(events, columns=["stay_id", "hour", "var", "val"])
    if len(frame):
        frame = frame.sort_values(["stay_id", "hour", "var"], kind="mergesort").reset_index(drop=True)
    for field in (
        "missing_sentinel_removed_by_parameter",
        "blank_or_invalid_removed_by_parameter",
        "unmapped_parameter_count_within_24h",
        "range_drop_count_by_variable",
        "accepted_event_count_by_variable",
        "unit_conversion_count_by_variable",
        "header_descriptor_fallback_count_by_variable",
    ):
        qc[field] = dict(sorted(qc[field].items()))
    qc["normalized_accepted_event_count"] = int(len(frame))
    return frame, qc


def outcome_label_from_survival(survival: float, los: float) -> int | None:
    """Apply the official LOS/Survival consistency rule for hospital death."""
    if not math.isfinite(survival) or not math.isfinite(los):
        return None
    if survival == -1 or survival > los:
        return 0
    if 2 <= survival <= los:
        return 1
    return None


def _read_outcomes(set_code: str) -> tuple[pd.DataFrame, Path]:
    path = CHALLENGE / "outcomes-a.txt" if set_code == "A" else INPUT / f"Outcomes-{set_code.casefold()}.txt"
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    required = {"RecordID", "Length_of_stay", "Survival", "In-hospital_death"}
    if not required.issubset(frame.columns):
        raise ValueError(f"set {set_code} outcomes lack columns: {sorted(required - set(frame.columns))}")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[list(required)].isna().any().any():
        raise ValueError(f"set {set_code} outcomes contain null/non-numeric required values")
    frame["RecordID"] = frame["RecordID"].astype("int64")
    if frame["RecordID"].duplicated().any():
        raise ValueError(f"set {set_code} outcomes have duplicate RecordIDs")
    return frame.sort_values("RecordID").reset_index(drop=True), path


def _aggregate_hourly(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["stay_id", "hb", "var", "val"])
    frame = events.loc[(events["hour"] >= 0) & (events["hour"] < MAXH)].copy()
    frame["hb"] = np.floor(frame["hour"]).astype(int).clip(0, MAXH - 1)
    aggregated = frame.groupby(["stay_id", "hb", "var"], as_index=False)["val"].mean()
    return aggregated.sort_values(["stay_id", "hb", "var"], kind="mergesort").reset_index(drop=True)


def _digitize(values: np.ndarray, variable: str, edges: dict[str, list[float]]) -> np.ndarray:
    bins = np.digitize(values, edges.get(variable, []), right=False) + 1
    return np.clip(bins, 1, NBINS)


def _build_arrays(
    aggregated: pd.DataFrame, cohort: pd.DataFrame, vars_: list[str], edges: dict[str, list[float]],
) -> tuple[dict[str, np.ndarray], pd.DataFrame, np.ndarray, dict[str, Any]]:
    var2id = {name: index for index, name in enumerate(vars_)}
    if len(aggregated):
        aggregated = aggregated[aggregated["var"].isin(var2id)].copy()
        aggregated["vid"] = aggregated["var"].map(var2id).astype(np.int16)
        aggregated = aggregated.sort_values(["stay_id", "hb", "vid"], kind="mergesort").reset_index(drop=True)
    stays = cohort["stay_id"].to_numpy(dtype=np.int64)
    sid2i = {int(stay): index for index, stay in enumerate(stays)}
    n_records = len(stays)
    V = np.zeros((n_records, MAXLEN), dtype=np.int16)
    B = np.zeros((n_records, MAXLEN), dtype=np.int16)
    H = np.zeros((n_records, MAXLEN), dtype=np.int16)
    L = np.zeros(n_records, dtype=np.int16)
    rng = np.random.default_rng(SEED)
    if len(aggregated):
        for stay, group in aggregated.groupby("stay_id", sort=False):
            index = sid2i.get(int(stay))
            if index is None:
                continue
            n = len(group)
            if n > MAXLEN:
                selected = rng.choice(n, MAXLEN, replace=False)
                group = group.iloc[np.sort(selected)]
                n = MAXLEN
            V[index, :n] = group["vid"].to_numpy(dtype=np.int16) + 1
            values = group["val"].to_numpy(dtype=float)
            names = group["var"].to_numpy()
            token_bins = np.ones(n, dtype=np.int16)
            for var in set(names):
                mask = names == var
                token_bins[mask] = _digitize(values[mask], var, edges)
            B[index, :n] = token_bins
            H[index, :n] = group["hb"].to_numpy(dtype=np.int16)
            L[index] = n

    wide = aggregated.pivot_table(index="stay_id", columns="var", values="val", aggfunc="mean") if len(aggregated) else pd.DataFrame()
    tabular = pd.DataFrame({"stay_id": stays})
    for var in vars_:
        tabular[var] = wide[var].reindex(stays).to_numpy() if var in wide.columns else np.nan

    subjects = pd.Series(cohort["subject_id"].astype(str).to_numpy()).drop_duplicates()
    subjects = subjects.sample(frac=1.0, random_state=SEED)
    n_train, n_val_end = int(0.7 * len(subjects)), int(0.8 * len(subjects))
    split_sets = {
        "train": set(subjects.iloc[:n_train].tolist()),
        "val": set(subjects.iloc[n_train:n_val_end].tolist()),
        "test": set(subjects.iloc[n_val_end:].tolist()),
    }
    split = np.array([
        0 if subject in split_sets["train"] else (1 if subject in split_sets["val"] else 2)
        for subject in cohort["subject_id"].astype(str).to_numpy()
    ], dtype=np.int8)
    y_mort = cohort["mort_hosp"].to_numpy(dtype=np.float32)
    arrays = {
        "V": V, "B": B, "H": H, "L": L,
        "stay_id": stays,
        "y_mort": y_mort,
        "y_aki": np.full(n_records, np.nan, dtype=np.float32),
        "y_long": np.full(n_records, np.nan, dtype=np.float32),
        "subject": cohort["subject_id"].astype(str).to_numpy(dtype="U32"),
    }
    meta = {
        "n": int(n_records),
        "mean_len": float(L.mean()) if n_records else 0.0,
        "max_len": int(L.max()) if n_records else 0,
        "vars": vars_,
        "nbins": NBINS,
        "maxh": MAXH,
        "maxlen": MAXLEN,
        "seed": SEED,
        "prev": {
            "mort": float(np.mean(y_mort)) if n_records else None,
            "aki": None,
            "long": None,
        },
        "unavailable_targets": {"aki": "not supplied by Challenge 2012", "long_icu": "not supplied by Challenge 2012"},
        "identity_basis": IDENTITY_BASIS,
        "patient_disjoint_claim_allowed": False,
    }
    return arrays, tabular, split, meta


def _selected_hash(eligibility: pd.DataFrame) -> str:
    selected = eligibility.loc[eligibility["eligible_24h"]]
    rows = sorted(
        f"{row.db}|{row.stay_id}|{row.subject}|{int(row.y_mort)}"
        for row in selected.itertuples()
    )
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def count_outcome_mismatches(expected_labels, actual_labels) -> int:
    """Pandas may coerce unavailable rule labels from None to float NaN."""
    return sum(pd.isna(expected) or int(expected) != int(actual)
               for expected, actual in zip(expected_labels, actual_labels))


def _build_eligibility(
    set_code: str, outcomes: pd.DataFrame, events: pd.DataFrame, arrays: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    row_by_id = {int(stay): index for index, stay in enumerate(arrays["stay_id"])}
    age_by_id: dict[int, float] = {}
    sex_by_id: dict[int, float] = {}
    height_by_id: dict[int, float] = {}
    for variable, destination in (("age", age_by_id), ("sex", sex_by_id), ("height", height_by_id)):
        subset = events.loc[(events["var"] == variable) & (events["hour"] == 0.0)]
        if len(subset):
            for stay, value in subset.groupby("stay_id", sort=False)["val"].mean().items():
                destination[int(stay)] = float(value)
    # ``events`` is the transformed input stream. The array length is the
    # count after hourly (stay, hour, variable) aggregation, matching build_data.
    n_by_id = {int(stay): int(length) for stay, length in zip(arrays["stay_id"], arrays["L"])}
    source_label = outcomes["In-hospital_death"].to_numpy(dtype=np.int64)
    survival = outcomes["Survival"].to_numpy(dtype=float)
    los = outcomes["Length_of_stay"].to_numpy(dtype=float)
    ids = outcomes["RecordID"].to_numpy(dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for i, stay in enumerate(ids):
        age = age_by_id.get(int(stay), np.nan)
        sex = sex_by_id.get(int(stay), np.nan)
        height = height_by_id.get(int(stay), np.nan)
        official_label = outcome_label_from_survival(float(survival[i]), float(los[i]))
        consistent = official_label is not None and official_label == int(source_label[i])
        adult = math.isfinite(float(age)) and float(age) >= 18.0
        input_count = n_by_id.get(int(stay), 0)
        input_ok = input_count > 0
        eligible = adult and consistent and input_ok
        if not adult:
            reason = "adult_age_unavailable_or_under_18"
        elif not consistent:
            reason = "outcome_inconsistent_with_official_los_survival_rule"
        elif not input_ok:
            reason = "no_valid_mapped_event_in_first_24h"
        else:
            reason = "eligible_24h"
        rows.append({
            "db": f"c2012_{set_code.casefold()}_v4_1",
            "set_code": set_code,
            "record_id": int(stay),
            "stay_id": int(stay),
            "subject_id": str(stay),
            "subject": str(stay),
            "identity_basis": IDENTITY_BASIS,
            "person_identity_verified": False,
            "age": age,
            "sex": sex,
            "height": height,
            "y_mort": int(source_label[i]),
            "source_y_mort": int(source_label[i]),
            "endpoint": "hospital",
            "source_endpoint_known": True,
            "survival_days": float(survival[i]),
            "length_of_stay_days": float(los[i]),
            "source_outcome_consistency_label": official_label,
            "source_outcome_consistent": bool(consistent),
            "adult_eligible": bool(adult),
            "input_event_count_24h": int(input_count),
            "input_available_24h": bool(input_ok),
            "input_window_start_hr": 0.0,
            "input_window_end_hr": 24.0,
            "input_availability_cutoff_hr": 24.0,
            "input_window_le_24h_verified": True,
            "input_scope": "retrospective_measurement_time",
            "inputs_temporal_scope": "retrospective_measurement_time",
            "clinical_input_availability_verified": False,
            "eligible_24h": bool(eligible),
            "temporal_verified": bool(eligible),
            "reason": reason,
            "retrospective_landmark_prediction_allowed": True,
            "ready_for_future24h_landmark_analysis": bool(eligible),
            "ready_for_retrospective_24h_measurement_analysis": bool(eligible),
            "ready_for_future24h_clinical_claim": False,
            "official_study_minimum_48h_proof": True,
            "landmark_lower_bound_hr": 48.0,
            "landmark_lower_bound_kind": "official_study_minimum_48h",
            "icu_exit_time": None,
            "icu_exit_hr": np.nan,
            "icu_exit_time_kind": "unavailable_exact_time_official_48h_lower_bound_only",
            "hospital_exit_time": None,
            "hospital_exit_hr": np.nan,
            "hospital_death": bool(source_label[i] == 1),
            "hospital_death_time": None,
            "hospital_death_time_hr": np.nan,
            "hospital_death_time_kind": "unavailable_outcomes_file_has_no_exact_timestamp",
            "alive_at_24h": True,
            "in_icu_at_24h": True,
            "source_landmark_issue": False,
            "source_postlandmark_ordering_issue": False,
            "source_time_issue": False,
            "source_time_status_conflict": False,
            "patient_disjoint_claim_allowed": False,
        })
    frame = pd.DataFrame(rows)
    summary = {
        "n_rows": int(len(frame)),
        "n_eligible_24h": int(frame["eligible_24h"].sum()),
        "n_eligible_positive": int((frame.loc[frame["eligible_24h"], "y_mort"] == 1).sum()),
        "n_eligible_negative": int((frame.loc[frame["eligible_24h"], "y_mort"] == 0).sum()),
        "n_adult_eligible": int(frame["adult_eligible"].sum()),
        "n_source_outcome_inconsistent": int((~frame["source_outcome_consistent"]).sum()),
        "n_no_valid_mapped_event_in_first_24h": int((~frame["input_available_24h"]).sum()),
        "source_join_coverage": 1.0,
        "selected_keys_labels_sha256": _selected_hash(frame),
        "exact_icu_exit_timestamp_available": False,
        "icu_exit_lower_bound_hr": 48.0,
        "lower_bound_evidence": INCLUSION_RULE,
        "patient_disjoint_claim_allowed": False,
    }
    return frame, summary


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _outcome_and_raw_id_check(code: str, records: list[RawRecord], outcomes: pd.DataFrame) -> None:
    source_ids = {record.record_id for record in records}
    outcome_ids = set(outcomes["RecordID"].astype(int))
    if source_ids != outcome_ids:
        missing_outcome = sorted(source_ids - outcome_ids)
        missing_raw = sorted(outcome_ids - source_ids)
        raise ValueError(
            f"set {code} source/outcome ID mismatch: raw_without_outcome={missing_outcome[:20]}, "
            f"outcome_without_raw={missing_raw[:20]}"
        )


def _write_set(
    stage: Path, code: str, records: list[RawRecord], source_info: dict[str, Any],
    outcomes: pd.DataFrame, outcome_path: Path, outcome_hash: str,
    vars_: list[str], ranges: dict[str, tuple[float, float]], bins: dict[str, list[float]],
    bins_hash: str,
) -> dict[str, Any]:
    event_frame, unit_qc = normalize_records(records, vars_, ranges)
    record_ids = sorted(record.record_id for record in records)
    outcome_frame = outcomes.sort_values("RecordID").reset_index(drop=True)
    cohort = pd.DataFrame({
        "stay_id": outcome_frame["RecordID"].to_numpy(dtype=np.int64),
        "subject_id": outcome_frame["RecordID"].astype(str).to_numpy(),
        "mort_hosp": outcome_frame["In-hospital_death"].to_numpy(dtype=np.float32),
    })
    hourly = _aggregate_hourly(event_frame)
    arrays, tabular, split, meta = _build_arrays(hourly, cohort, vars_, bins)
    eligibility, summary = _build_eligibility(code, outcome_frame, event_frame, arrays)
    set_dir = stage / code
    set_dir.mkdir(parents=True)
    cohort_id = f"c2012_{code.casefold()}_v4_1"
    sequence_path = set_dir / f"seq_{cohort_id}.npz"
    split_path = set_dir / f"split_{cohort_id}.npy"
    tabular_path = set_dir / f"tab_{cohort_id}.parquet"
    vocab_path = set_dir / f"meta_{cohort_id}.json"
    eligibility_path = set_dir / f"eligibility_{cohort_id}.parquet"

    np.savez_compressed(sequence_path, **arrays)
    np.save(split_path, split)
    tabular.to_parquet(tabular_path, index=False)
    eligibility.to_parquet(eligibility_path, index=False)
    _atomic_json(vocab_path, meta)

    output_hashes = {
        "sequence_path": _file_digest(sequence_path),
        "split_path": _file_digest(split_path),
        "tabular_path": _file_digest(tabular_path),
        "vocab_path": _file_digest(vocab_path),
        "eligibility_path": _file_digest(eligibility_path),
    }
    n_consistency_labels = outcome_frame.apply(
        lambda row: outcome_label_from_survival(float(row["Survival"]), float(row["Length_of_stay"])), axis=1
    )
    observed_label_mismatches = count_outcome_mismatches(
        n_consistency_labels, outcome_frame["In-hospital_death"])
    set_manifest = {
        "schema_version": "4.1",
        "cohort_id": cohort_id,
        "set_code": code,
        "derivation": "Challenge 2012 source records; canonical v4 mapping and frozen bins; first 24 hours only",
        "cohort_identity": {
            "basis": IDENTITY_BASIS,
            "patient_disjoint_claim_allowed": False,
            "bootstrap_unit": "record_id_proxy",
        },
        "temporal_evidence": {
            "kind": "official_study_minimum_48h_lower_bound",
            "lower_bound_hours": 48.0,
            "exact_icu_exit_time_available": False,
            "exact_hospital_exit_time_available": False,
            "exact_death_time_available": False,
            "input_time_window": "0 <= elapsed_hour < 24",
            "inclusion_rule": INCLUSION_RULE,
        },
        "source": {
            **source_info,
            "outcomes_path": str(outcome_path),
            "outcomes_sha256": outcome_hash,
            "source_record_count": len(records),
            "source_outcome_join_coverage": 1.0,
        },
        "builder": {
            "extract_source": str(EXTRACT_SOURCE),
            "extract_source_sha256": _file_digest(EXTRACT_SOURCE),
            "frozen_bins_path": str(FROZEN_BINS),
            "frozen_bins_sha256": bins_hash,
            "variable_count": len(vars_),
            "variables": vars_,
            "max_hours": MAXH,
            "max_tokens": MAXLEN,
            "n_bins": NBINS,
            "seed": SEED,
            "hourly_aggregation": "mean by (RecordID, floor(elapsed_hour), canonical variable)",
            "maxlen_sampling": "uniform without replacement, sorted selected indices, one seeded RNG across sorted RecordIDs",
            "unit_qc": unit_qc,
            "label_mismatch_count_by_official_rule": observed_label_mismatches,
        },
        "counts": {**summary, "n_records": len(records), "n_hourly_events": int(len(hourly))},
        "targets": {
            "mort_hosp": "Outcomes set file In-hospital_death",
            "aki": None,
            "long_icu": None,
            "unavailable_targets_encoded_as_nan": ["y_aki", "y_long"],
        },
        "outputs": {
            "sequence_path": str(OUTPUT_ROOT / code / sequence_path.name),
            "split_path": str(OUTPUT_ROOT / code / split_path.name),
            "tabular_path": str(OUTPUT_ROOT / code / tabular_path.name),
            "vocab_path": str(OUTPUT_ROOT / code / vocab_path.name),
            "eligibility_path": str(OUTPUT_ROOT / code / eligibility_path.name),
            "hashes": output_hashes,
        },
    }
    _atomic_json(set_dir / "source_manifest.json", set_manifest)
    return {
        "code": code,
        "cohort_id": cohort_id,
        "record_ids": record_ids,
        "summary": summary,
        "unit_qc": unit_qc,
        "source_manifest": set_manifest,
        "paths": {
            "sequence_path": str(OUTPUT_ROOT / code / sequence_path.name),
            "split_path": str(OUTPUT_ROOT / code / split_path.name),
            "tabular_path": str(OUTPUT_ROOT / code / tabular_path.name),
            "vocab_path": str(OUTPUT_ROOT / code / vocab_path.name),
            "eligibility_path": str(OUTPUT_ROOT / code / eligibility_path.name),
            "manifest_path": str(OUTPUT_ROOT / code / "source_manifest.json"),
        },
    }


def _read_legacy_a_ids() -> set[int]:
    if not LEGACY_A_SEQUENCE.is_file():
        return set()
    with np.load(LEGACY_A_SEQUENCE, allow_pickle=False) as legacy:
        if "stay_id" not in legacy.files:
            raise ValueError(f"legacy C2012 sequence lacks stay_id: {LEGACY_A_SEQUENCE}")
        return {int(value) for value in legacy["stay_id"]}


def build() -> dict[str, Any]:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite versioned output: {OUTPUT_ROOT}")
    if REGISTRY_FRAGMENT.exists():
        raise FileExistsError(f"refusing to overwrite C2012 registry fragment: {REGISTRY_FRAGMENT}")
    EXT.mkdir(parents=True, exist_ok=True)
    result_parent = OUTPUT_ROOT.parent
    result_parent.mkdir(parents=True, exist_ok=True)

    bins_hash_before = _file_digest(FROZEN_BINS)
    bins = json.loads(FROZEN_BINS.read_text(encoding="utf-8"))
    vars_, ranges = _get_extractor_definitions()
    if len(vars_) == 0 or len(set(vars_)) != len(vars_):
        raise ValueError("extract.py VARS is empty or contains duplicates")
    if set(bins).difference(vars_):
        raise ValueError(f"frozen bins contain unknown variables: {sorted(set(bins).difference(vars_))}")

    outcomes_by_set: dict[str, tuple[pd.DataFrame, Path, str]] = {}
    for code in ("A", "B", "C"):
        outcomes, outcome_path = _read_outcomes(code)
        outcome_hash = _file_digest(outcome_path)
        outcomes_by_set[code] = (outcomes, outcome_path, outcome_hash)

    legacy_ids = _read_legacy_a_ids()
    # Outcomes are small and supply the complete record-ID lists needed for
    # cross-set provenance. Parse one raw set at a time below to keep peak RAM
    # bounded while still checking each raw ID set against its outcomes.
    source_ids = {
        code: set(payload[0]["RecordID"].astype(int))
        for code, payload in outcomes_by_set.items()
    }
    intersections: dict[str, dict[str, Any]] = {}
    for first, second in (("A", "B"), ("A", "C"), ("B", "C")):
        ids = sorted(source_ids[first].intersection(source_ids[second]))
        intersections[f"set_{first}_and_set_{second}"] = {"count": len(ids), "record_ids": ids}
    for code in ("A", "B", "C"):
        ids = sorted(source_ids[code].intersection(legacy_ids))
        intersections[f"set_{code}_and_legacy_seq_c2012"] = {"count": len(ids), "record_ids": ids}

    record_provenance = {
        "record_identity_scope": "RecordID only; no person-level deduplication or independence claim",
        "set_record_id_hashes": {code: _id_hash(source_ids[code]) for code in ("A", "B", "C")},
        "intersections": intersections,
        "legacy_artifact": {
            "path": str(LEGACY_A_SEQUENCE),
            "n_record_ids": len(legacy_ids),
            "record_ids_sha256": _id_hash(legacy_ids),
            "historical_pretraining": {
                "reported_source_sets": ["A"],
                "actual_training_membership_known": False,
                "verified_training_record_ids": False,
                "verified_training_record_count": None,
                "legacy_A_artifact_overlap_is_exposure_possible_proxy_only": True,
                "note": "The overlap with the legacy A artifact is not a verified historical training membership count.",
            },
        },
    }

    stage = Path(tempfile.mkdtemp(prefix=".c2012_v4_1.tmp-", dir=result_parent))
    set_results: dict[str, dict[str, Any]] = {}
    for code in ("A", "B", "C"):
        records, source_info = iter_set_records(code)
        outcomes, outcome_path, outcome_hash = outcomes_by_set[code]
        _outcome_and_raw_id_check(code, records, outcomes)
        print(json.dumps({"phase": "source_loaded", "set": code, "n_records": len(records),
                          "source_sha256": source_info["source_sha256"],
                          "outcomes_sha256": outcome_hash}, sort_keys=True), flush=True)
        set_results[code] = _write_set(
            stage, code, records, source_info, outcomes, outcome_path, outcome_hash,
            vars_, ranges, bins, bins_hash_before,
        )
        print(json.dumps({"phase": "set_built", "set": code,
                          "counts": set_results[code]["summary"],
                          "unit_qc": set_results[code]["unit_qc"]}, sort_keys=True), flush=True)

    overall_manifest = {
        "schema_version": "4.1",
        "artifact_version": "c2012_v4_1",
        "scope": "corrected Challenge 2012 source sets A/B/C; no model fitting or performance evaluation",
        "sets": {code: result["source_manifest"] for code, result in set_results.items()},
        "record_id_provenance": record_provenance,
        "source_reference": "https://physionet.org/content/challenge-2012/1.0.0/",
        "source_semantics": {
            "all_records_adults": True,
            "study_minimum_icu_stay_hours": 48,
            "timestamp_basis": "elapsed time since ICU admission",
            "missing_sentinel": -1,
            "survival_consistency_rule": {
                "survival_gt_length_of_stay": "survivor",
                "survival_eq_minus_1": "survivor",
                "2_le_survival_le_length_of_stay": "in_hospital_death",
            },
            "fiO2_unit": "fraction converted to percent",
            "magnesium_unit": "mmol/L converted to mg/dL using factor 2.430",
            "weight": "time series observations retained only for 0 <= hour < 24",
        },
    }
    _atomic_json(stage / "record_id_provenance.json", record_provenance)
    _atomic_json(stage / "build_manifest.json", overall_manifest)
    os.replace(stage, OUTPUT_ROOT)

    fragment_cohorts: dict[str, Any] = {}
    for result in set_results.values():
        entry = dict(result["paths"])
        # Keep the resolver's four path fields plus C2012-specific provenance
        # links. The current resolver ignores additional metadata fields.
        fragment_cohorts[result["cohort_id"]] = entry
    fragment = {
        "schema_version": "4.1",
        "owner": "c2012_extension",
        "artifact_version": "c2012_v4_1",
        "cohorts": fragment_cohorts,
    }
    _atomic_json(REGISTRY_FRAGMENT, fragment)

    if bins_hash_before != _file_digest(FROZEN_BINS):
        raise RuntimeError("frozen bins.json changed during build")
    if not ENGINE_CODE.is_dir():
        raise FileNotFoundError(f"registry resolver code is missing: {ENGINE_CODE}")
    import sys
    sys.path.insert(0, str(ENGINE_CODE))
    from cohort_registry_v4 import resolve_cohort_paths
    registry_checks = {}
    for cohort_id, entry in fragment_cohorts.items():
        paths = resolve_cohort_paths(cohort_id, registry_path=REGISTRY_FRAGMENT, verify_exists=True)
        expected = {key: Path(entry[key]) for key in ("sequence_path", "split_path", "tabular_path", "vocab_path")}
        observed = {
            "sequence_path": paths.sequence_path,
            "split_path": paths.split_path,
            "tabular_path": paths.tabular_path,
            "vocab_path": paths.vocab_path,
        }
        if observed != expected:
            raise RuntimeError(f"registry resolution mismatch for {cohort_id}: {observed} != {expected}")
        registry_checks[cohort_id] = paths.as_metadata()

    final_result = {
        "output_root": str(OUTPUT_ROOT),
        "registry_fragment": str(REGISTRY_FRAGMENT),
        "sets": {code: result["summary"] for code, result in set_results.items()},
        "record_id_intersections": {key: value["count"] for key, value in intersections.items()},
        "registry_resolution": registry_checks,
        "bins_unchanged": True,
    }
    _atomic_json(OUTPUT_ROOT / "build_audit.json", final_result)
    print(json.dumps({"phase": "build_complete", **final_result}, sort_keys=True), flush=True)
    return final_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", required=True,
                        help="write new versioned outputs and a C2012-owned registry fragment")
    args = parser.parse_args()
    if args.write:
        build()


if __name__ == "__main__":
    main()
