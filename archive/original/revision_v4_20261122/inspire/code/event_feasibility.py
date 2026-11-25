#!/usr/bin/env python3
"""Create server-only first-24h event extracts and variable coverage aggregates."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

BASE = Path("/data/admin/fengshuo/ehr_fm_20261102/revision_v4_20261122/inspire")
PROJECT = Path("/data/admin/fengshuo/ehr_fm_20261102")
COHORT = BASE / "normalized" / "feasibility" / "cohort_first_icuin.csv"
PARAMETERS = BASE / "input" / "metadata" / "parameters.csv"
MAPPING = PROJECT / "mapping_table.csv"
OUTDIR = BASE / "normalized" / "feasibility"
AUDIT = BASE / "audit"
EVENT_PATHS = {
    "labs": BASE / "input" / "clean" / "labs.csv.gz",
    "ward_vitals": BASE / "input" / "clean" / "ward_vitals.csv.gz",
}

SOURCE_TO_CANONICAL = {
    "ward_vitals": {
        "hr": "hr",
        "art_sbp": "sbp", "nibp_sbp": "sbp",
        "art_dbp": "dbp", "nibp_dbp": "dbp",
        "art_mbp": "map", "nibp_mbp": "map",
        "rr": "rr", "bt": "temp", "spo2": "spo2",
        "gcs_m": "gcs_motor", "gcs_e": "gcs_eye",
        "fio2": "fio2", "uo": "urine", "crrt": "rrt",
    },
    "labs": {
        "pao2": "pao2", "paco2": "paco2", "ph": "ph", "be": "be",
        "creatinine": "creatinine", "bun": "bun", "lacate": "lactate",
        "wbc": "wbc", "hb": "hgb", "platelet": "platelet", "ptinr": "inr",
        "aptt": "aptt", "sodium": "sodium", "potassium": "potassium",
        "chloride": "chloride", "calcium": "calcium", "phosphorus": "phosphate",
        "glucose": "glucose", "albumin": "albumin", "total_bilirubin": "bili",
        "ast": "ast", "alt": "alt", "crp": "crp",
    },
}
GCS_COMPONENTS = {"gcs_e", "gcs_m", "gcs_v"}
ID_CANDIDATES = ["subject_id", "hadm_id", "op_id"]


def resolve_columns(path: Path) -> tuple[list[str], str, str, str, list[str]]:
    cols = list(pd.read_csv(path, compression="gzip", nrows=0).columns)
    lowered = {column.lower(): column for column in cols}
    time_col = next((lowered[name] for name in ("chart_time", "charttime") if name in lowered), None)
    item_col = next((lowered[name] for name in ("item_name", "label", "item") if name in lowered), None)
    value_col = next((lowered[name] for name in ("value", "valuenum", "value_num") if name in lowered), None)
    if not time_col or not item_col or not value_col:
        raise RuntimeError(f"Cannot resolve event columns from {path.name}: {cols}")
    ids = [lowered[name] for name in ID_CANDIDATES if name in lowered]
    return cols, time_col, item_col, value_col, ids


def write_chunk(frame: pd.DataFrame, target: Path, first: bool) -> None:
    mode = "wb" if first else "ab"
    with gzip.open(target, mode) as handle:
        frame.to_csv(handle, index=False, header=first, quoting=csv.QUOTE_MINIMAL)


def process_table(
    table: str,
    cohort: pd.DataFrame,
    units: dict[tuple[str, str], str],
    canonical_units: dict[str, str],
) -> dict[str, object]:
    source = EVENT_PATHS[table]
    target = OUTDIR / f"{table}_first24h.csv.gz"
    temp_target = OUTDIR / f"{table}_first24h.csv.gz.partial"
    component_file = OUTDIR / "ward_gcs_components_first24h.csv.gz.partial"
    if target.exists() or temp_target.exists():
        raise FileExistsError(f"Refusing to overwrite normalized event output: {target}")
    if table == "ward_vitals" and component_file.exists():
        raise FileExistsError(f"Refusing to overwrite temporary GCS component output: {component_file}")
    columns, time_col, item_col, value_col, event_ids = resolve_columns(source)
    join_keys = [key for key in ["hadm_id", "subject_id"] if key in event_ids and key in cohort.columns]
    if not join_keys:
        join_keys = [key for key in ["op_id", "subject_id"] if key in event_ids and key in cohort.columns]
    if not join_keys:
        raise RuntimeError(f"No safe cohort join key found in {table} fields {event_ids}")
    cohort_fields = list(dict.fromkeys(join_keys + [
        "subject_id", "hadm_id", "op_id", "icuin_time", "icuout_time",
        "landmark24_time", "patient_group", "confirmed_alive_in_icu_at_24h",
    ]))
    cohort_fields = [column for column in cohort_fields if column in cohort.columns]
    lookup = cohort[cohort_fields].copy()
    lookup["icuin_time"] = pd.to_numeric(lookup["icuin_time"], errors="coerce")
    if "icuout_time" in lookup:
        lookup["icuout_time"] = pd.to_numeric(lookup["icuout_time"], errors="coerce")
    if "landmark24_time" not in lookup:
        lookup["landmark24_time"] = lookup["icuin_time"] + 1440.0
    lookup = lookup.drop_duplicates(join_keys)

    item_map = SOURCE_TO_CANONICAL[table]
    usecols = list(dict.fromkeys(event_ids + [time_col, item_col, value_col]))
    dtype = {column: "string" for column in event_ids}
    counts: dict[str, dict[str, object]] = {}
    seen_subjects: dict[str, set[str]] = {}
    risk_subjects: dict[str, set[str]] = {}
    unique_values: dict[str, set[float]] = {}
    min_values: dict[str, float] = {}
    max_values: dict[str, float] = {}
    source_rows = kept_rows = 0
    first_write = True
    first_components_write = True

    for chunk in pd.read_csv(source, compression="gzip", usecols=usecols, dtype=dtype, chunksize=400_000, low_memory=False):
        source_rows += len(chunk)
        chunk[time_col] = pd.to_numeric(chunk[time_col], errors="coerce")
        chunk[item_col] = chunk[item_col].astype("string").str.lower()
        chunk[value_col] = pd.to_numeric(chunk[value_col], errors="coerce")
        if table == "ward_vitals":
            gcs_rows = chunk[chunk[item_col].isin(GCS_COMPONENTS) & chunk[value_col].notna()].copy()
            if not gcs_rows.empty:
                gcs_rows = gcs_rows.merge(lookup, on=join_keys, how="inner", validate="many_to_one")
                gcs_rows = gcs_rows[gcs_rows[time_col].ge(gcs_rows["icuin_time"]) & gcs_rows[time_col].lt(gcs_rows["landmark24_time"])]
                if "icuout_time" in gcs_rows:
                    gcs_rows = gcs_rows[gcs_rows["icuout_time"].isna() | gcs_rows[time_col].lt(gcs_rows["icuout_time"])]
                if not gcs_rows.empty:
                    component_cols = [c for c in ["subject_id", "hadm_id", "op_id", "icuin_time", time_col, item_col, value_col] if c in gcs_rows.columns]
                    component_out = gcs_rows[component_cols].rename(columns={
                        time_col: "measurement_time", item_col: "source_item", value_col: "value",
                    })
                    with gzip.open(component_file, "wb" if first_components_write else "ab") as handle:
                        component_out.to_csv(handle, index=False, header=first_components_write)
                    first_components_write = False

        mapped = chunk[chunk[item_col].isin(item_map) & chunk[value_col].notna()].copy()
        if mapped.empty:
            continue
        mapped = mapped.merge(lookup, on=join_keys, how="inner", validate="many_to_one")
        mapped = mapped[mapped[time_col].ge(mapped["icuin_time"]) & mapped[time_col].lt(mapped["landmark24_time"])]
        if "icuout_time" in mapped:
            mapped = mapped[mapped["icuout_time"].isna() | mapped[time_col].lt(mapped["icuout_time"])]
        if mapped.empty:
            continue
        mapped["canonical_variable"] = mapped[item_col].map(item_map)
        mapped["canonical_unit"] = mapped["canonical_variable"].map(canonical_units).fillna("")
        mapped["time_from_icuin_min"] = mapped[time_col] - mapped["icuin_time"]
        mapped["source_table"] = table
        mapped["source_item"] = mapped[item_col]
        mapped["source_unit"] = [units.get((table, str(item)), "") for item in mapped[item_col]]
        mapped["unit_conversion"] = "identity_or_equivalent_unit"
        crp_rows = mapped["canonical_variable"].eq("crp") & mapped["source_unit"].str.lower().eq("mg/dl")
        if crp_rows.any():
            mapped.loc[crp_rows, value_col] = mapped.loc[crp_rows, value_col] * 10.0
            mapped.loc[crp_rows, "unit_conversion"] = "mg/dL_to_mg/L_multiplier_10"
        platelet_rows = mapped["canonical_variable"].eq("platelet") & mapped["source_unit"].str.lower().eq("/nl")
        mapped.loc[platelet_rows, "unit_conversion"] = "numeric_identity_1_per_nL_equals_1e9_per_L"
        mapped["measurement_availability_time"] = "not_supplied; measurement time used"
        out_cols = [c for c in [
            "subject_id", "hadm_id", "op_id", "patient_group", "icuin_time",
            time_col, "time_from_icuin_min", "canonical_variable", value_col,
            "source_table", "source_item", "source_unit", "canonical_unit", "unit_conversion",
            "measurement_availability_time", "confirmed_alive_in_icu_at_24h",
        ] if c in mapped.columns]
        output = mapped[out_cols].rename(columns={time_col: "measurement_time", value_col: "value"})
        write_chunk(output, temp_target, first_write)
        first_write = False
        kept_rows += len(output)

        for variable, group in mapped.groupby("canonical_variable", sort=False):
            key = str(variable)
            values = group[value_col].astype(float)
            counts.setdefault(key, {"events": 0})
            counts[key]["events"] = int(counts[key]["events"]) + int(len(group))
            patient_values = set(group["subject_id"].dropna().astype(str))
            seen_subjects.setdefault(key, set()).update(patient_values)
            risk_group = group[group["confirmed_alive_in_icu_at_24h"].fillna(False)] if "confirmed_alive_in_icu_at_24h" in group else group.iloc[0:0]
            risk_subjects.setdefault(key, set()).update(risk_group["subject_id"].dropna().astype(str))
            unique_values.setdefault(key, set()).update(values.round(6).unique().tolist())
            min_values[key] = min(min_values.get(key, float("inf")), float(values.min()))
            max_values[key] = max(max_values.get(key, float("-inf")), float(values.max()))

    coverage = {}
    for variable in sorted(item_map.values()):
        coverage[variable] = {
            "events_in_first24h": int(counts.get(variable, {}).get("events", 0)),
            "subjects_with_any": len(seen_subjects.get(variable, set())),
            "strict_confirmed_at_risk_subjects_with_any": len(risk_subjects.get(variable, set())),
            "distinct_rounded_values_6dp": len(unique_values.get(variable, set())),
            "min": min_values.get(variable),
            "max": max_values.get(variable),
        }

    if table == "ward_vitals" and component_file.exists():
        components = pd.read_csv(component_file, compression="gzip", dtype={"subject_id": "string", "hadm_id": "string", "op_id": "string"})
        group_keys = [c for c in ["subject_id", "hadm_id", "op_id", "icuin_time", "measurement_time"] if c in components.columns]
        wide = components.pivot_table(index=group_keys, columns="source_item", values="value", aggfunc="first")
        if GCS_COMPONENTS.issubset(set(wide.columns)):
            complete = wide[list(GCS_COMPONENTS)].dropna()
            if not complete.empty:
                derived = complete.sum(axis=1).rename("value").reset_index()
                derived["time_from_icuin_min"] = derived["measurement_time"] - derived["icuin_time"]
                derived["canonical_variable"] = "gcs_total"
                derived["source_table"] = "ward_vitals"
                derived["source_item"] = "gcs_e+gcs_m+gcs_v"
                derived["source_unit"] = "3-15"
                derived["canonical_unit"] = canonical_units.get("gcs_total", "3-15")
                derived["unit_conversion"] = "sum_components_at_same_recorded_time"
                derived["measurement_availability_time"] = "not_supplied; measurement time used"
                derived = derived.merge(
                    cohort[["subject_id", "confirmed_alive_in_icu_at_24h"]].drop_duplicates("subject_id"),
                    on="subject_id", how="left", validate="many_to_one",
                )
                derived["patient_group"] = derived["subject_id"]
                derived_columns = [
                    "subject_id", "hadm_id", "op_id", "patient_group", "icuin_time",
                    "measurement_time", "time_from_icuin_min", "canonical_variable", "value",
                    "source_table", "source_item", "source_unit", "canonical_unit", "unit_conversion",
                    "measurement_availability_time", "confirmed_alive_in_icu_at_24h",
                ]
                derived = derived[[column for column in derived_columns if column in derived.columns]]
                with gzip.open(temp_target, "wb" if first_write else "ab") as handle:
                    derived.to_csv(handle, index=False, header=first_write)
                first_write = False
                coverage["gcs_total"] = {
                    "events_in_first24h": int(len(derived)),
                    "subjects_with_any": int(derived["subject_id"].nunique()),
                    "strict_confirmed_at_risk_subjects_with_any": int(derived.loc[derived["confirmed_alive_in_icu_at_24h"].fillna(False), "subject_id"].nunique()),
                    "distinct_rounded_values_6dp": int(derived["value"].round(6).nunique()),
                    "min": float(derived["value"].min()),
                    "max": float(derived["value"].max()),
                }
        component_file.unlink()

    if first_write:
        empty_columns = [
            "subject_id", "hadm_id", "op_id", "patient_group", "icuin_time",
            "measurement_time", "time_from_icuin_min", "canonical_variable", "value",
            "source_table", "source_item", "source_unit", "canonical_unit", "unit_conversion",
            "measurement_availability_time", "confirmed_alive_in_icu_at_24h",
        ]
        with gzip.open(temp_target, "wb") as handle:
            pd.DataFrame(columns=empty_columns).to_csv(handle, index=False)
    temp_target.replace(target)

    return {
        "table": table,
        "source_file": source.name,
        "source_header_fields": columns,
        "source_rows_read": source_rows,
        "first24h_mapped_event_rows_written": kept_rows,
        "normalized_event_path": str(target),
        "join_keys": join_keys,
        "event_time_window": "[icuin_time, icuin_time + 1440 minutes)",
        "time_unit": "minutes relative to initial operation admission, per official dictionary",
        "mapped_variable_coverage": coverage,
        "raw_rows_emitted_to_logs": False,
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(COHORT, dtype={"subject_id": "string", "hadm_id": "string", "op_id": "string"})
    for column in ["icuin_time", "icuout_time", "landmark24_time"]:
        if column in cohort:
            cohort[column] = pd.to_numeric(cohort[column], errors="coerce")
    parameters = pd.read_csv(PARAMETERS, dtype={"Table": "string", "Label": "string", "Unit": "string"})
    units = {
        (str(row["Table"]).lower(), str(row["Label"]).lower()): str(row["Unit"])
        for _, row in parameters.iterrows()
    }
    mapping = pd.read_csv(MAPPING, usecols=["library", "canonical_variable", "unit"], dtype="string")
    canonical_mapping = mapping.loc[mapping["library"].str.lower().eq("mimic")].drop_duplicates("canonical_variable")
    canonical = list(canonical_mapping["canonical_variable"])
    canonical_units = dict(zip(canonical_mapping["canonical_variable"], canonical_mapping["unit"]))
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda table: process_table(table, cohort, units, canonical_units), ["labs", "ward_vitals"]))
    operations_summary = json.loads((AUDIT / "operations_feasibility.json").read_text(encoding="utf-8"))
    static_coverage = operations_summary.get("static_variable_coverage", {})
    static = ["age", "sex", "weight", "height", "bmi"]
    event_coverage = {}
    for result in results:
        for variable, values in result["mapped_variable_coverage"].items():
            event_coverage.setdefault(variable, {"events_in_first24h": 0, "subjects_with_any": 0, "strict_confirmed_at_risk_subjects_with_any": 0, "distinct_rounded_values_6dp": 0, "min": None, "max": None})
            current = event_coverage[variable]
            current["events_in_first24h"] += values.get("events_in_first24h", 0)
            current["subjects_with_any"] = max(current["subjects_with_any"], values.get("subjects_with_any", 0))
            current["strict_confirmed_at_risk_subjects_with_any"] = max(current["strict_confirmed_at_risk_subjects_with_any"], values.get("strict_confirmed_at_risk_subjects_with_any") or 0)
            current["distinct_rounded_values_6dp"] = max(current["distinct_rounded_values_6dp"], values.get("distinct_rounded_values_6dp", 0))
            if values.get("min") is not None:
                current["min"] = values["min"] if current["min"] is None else min(current["min"], values["min"])
            if values.get("max") is not None:
                current["max"] = values["max"] if current["max"] is None else max(current["max"], values["max"])
    event_intersection = [
        variable for variable in canonical
        if variable in event_coverage and event_coverage[variable]["events_in_first24h"] > 0
    ]
    static_intersection = [
        variable for variable in static
        if static_coverage.get(variable, {}).get("selected_first_per_subject_with_value", 0) > 0
    ]
    candidate_features = [variable for variable in canonical if variable in set(static_intersection + event_intersection)]
    variable_rows = []
    for variable in canonical:
        expected_unit = canonical_units.get(variable, "")
        if variable in static:
            values = static_coverage.get(variable, {})
            variable_rows.append({
                "canonical_variable": variable,
                "source_table": "operations",
                "source_fields": variable,
                "canonical_unit": expected_unit,
                "availability_type": "static_baseline",
                "status": "observed" if values.get("selected_first_per_subject_with_value", 0) else "mapped_zero_coverage",
                **values,
            })
            continue
        values = event_coverage.get(variable, {})
        source_fields = [f"{table}:{item}" for table, item_map in SOURCE_TO_CANONICAL.items() for item, target in item_map.items() if target == variable]
        if variable == "gcs_total" and values:
            source_fields = ["ward_vitals:gcs_e+gcs_m+gcs_v (same recorded time)"]
        event_count = values.get("events_in_first24h", 0)
        status = "observed" if event_count > 0 else ("mapped_zero_coverage" if source_fields else "not_available_without_excluded_source_table")
        variable_rows.append({
            "canonical_variable": variable,
            "source_table": ";".join(sorted({field.split(":", 1)[0] for field in source_fields})),
            "source_fields": ";".join(source_fields),
            "canonical_unit": expected_unit,
            "availability_type": "time_varying_first24h",
            "status": status,
            **values,
        })
    payload = {
        "dataset": "INSPIRE 1.4.2",
        "protocol_path": "/data/admin/fengshuo/ehr_fm_20261102/revision_v4_20261122/admin/execution_protocol_v4.md",
        "protocol_sha256": "605f9440026bc0074c3ede3ee450adfd94d394219b0f8fcd5e92014776f9d9ff",
        "cohort_rows": int(len(cohort)),
        "canonical_variable_count": len(canonical),
        "canonical_variables": canonical,
        "static_variables_from_operations": static,
        "static_variables_with_values": static_intersection,
        "event_variable_intersection_observed": event_intersection,
        "availability_only_candidate_feature_intersection": candidate_features,
        "availability_only_candidate_intersection_sha256": hashlib.sha256("\n".join(candidate_features).encode("utf-8")).hexdigest(),
        "availability_freeze_status": "pre-performance availability snapshot; final model feature lock remains with root",
        "static_variable_coverage": static_coverage,
        "outcome_semantics": "Missing inhosp_death_time remains unknown; no null-to-survivor conversion without documented source semantics and complete follow-up.",
        "event_tables": results,
        "measurement_availability_time": "No acquisition/result availability timestamp was supplied; measurement time is used and this is disclosed.",
        "intraoperative_vitals_policy": "vitals.csv.gz was not used; only ward_vitals and labs events in the first-24h window were processed.",
        "quantization_note": "Official INSPIRE dictionary states measurements are percentile-binned; per-variable distinct counts and ranges are descriptive, not raw-resolution claims.",
        "outcome_and_exposure_policy": "Feasibility only; no model performance or model selection is included.",
    }
    (AUDIT / "event_feasibility.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(variable_rows).to_csv(AUDIT / "variable_coverage_24h.csv", index=False)
    print(json.dumps({"cohort_rows": len(cohort), "tables": [{"table": item["table"], "rows": item["first24h_mapped_event_rows_written"]} for item in results], "audit": str(AUDIT / "event_feasibility.json")}))


if __name__ == "__main__":
    main()
