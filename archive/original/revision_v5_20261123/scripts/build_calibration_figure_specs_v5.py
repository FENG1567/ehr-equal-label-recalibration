#!/usr/bin/env python3
"""Build verified Figure 1, 5, and 6 specs from completed aggregate outputs.

This script reads aggregate CSVs only. It does not fit models, calculate
statistics, or draw figures. Its JSON output is a verified subset intended to
be merged into the six-figure bundle consumed by plot_v5_figures.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "ehr_fm_v5_calibration_sites_1"
COHORTS = (
    ("eicu_semantic_v4_2", "eICU"),
    ("nwicu_discharge_v4_1", "NWICU"),
)
MODELS = (
    ("main20k_real", "FM"),
    ("conventional_lr_value_plus_pattern", "V+P LR"),
)
CALIBRATION_METHODS = (
    ("raw", "Raw"),
    ("platt_500_repeat0", "Fixed 500-label Platt"),
)
BUDGETS = (50, 200, 500, 2000)
CONTRASTS = (
    ("raw_gap", "Raw FM−V+P LR gap"),
    ("platt_gap", "Post-Platt FM−V+P LR gap"),
    ("gap_change", "Gap change (post-Platt−raw)"),
)
REQUIRED_TABLES = {
    "calibration": (
        "calibration_curve_points.csv",
        "calibration_risk_bins.csv",
        "calibration_risk_summary.csv",
        "paired_loss_gap_bootstrap.csv",
    ),
    "sites": ("across_site_adaptation_summary.csv",),
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(document, dict), f"expected JSON object: {path.name}")
    return document


def verify_section(folder_arg: Path, expected_section: str) -> dict[str, Any]:
    folder = folder_arg.resolve(strict=True)
    require(folder.is_dir(), f"{expected_section} input is not a directory")
    manifest_path = folder / "jobmanifest.json"
    checksum_path = folder / "jobmanifest.sha256"
    schema_path = folder / "schema.json"
    require(manifest_path.is_file() and checksum_path.is_file() and schema_path.is_file(),
            f"{expected_section} input lacks its completion manifest or schema")

    checksum_text = checksum_path.read_text(encoding="utf-8").strip().split()
    require(checksum_text and SHA256_RE.fullmatch(checksum_text[0]) is not None,
            f"{expected_section} manifest checksum is malformed")
    manifest_hash = sha256_file(manifest_path)
    require(manifest_hash == checksum_text[0],
            f"{expected_section} completion manifest checksum does not match")
    manifest = read_json(manifest_path)
    require(manifest.get("status") == "complete",
            f"{expected_section} outputs are not marked complete")
    require(manifest.get("schema_version") == SCHEMA_VERSION,
            f"{expected_section} schema version is unsupported")
    require(manifest.get("section") == expected_section,
            f"expected {expected_section} section, found {manifest.get('section')!r}")
    require(manifest.get("patient_level_exported_to_local") is False,
            f"{expected_section} provenance does not exclude patient-level exports")

    documentation = manifest.get("documentation")
    require(isinstance(documentation, dict), f"{expected_section} documentation metadata is missing")
    schema_hash = sha256_file(schema_path)
    require(documentation.get("schema_sha256") == schema_hash,
            f"{expected_section} schema hash does not match its completion manifest")
    schema = read_json(schema_path)
    tables = manifest.get("tables")
    require(isinstance(tables, dict) and tables == schema,
            f"{expected_section} manifest and schema table maps differ")

    if expected_section == "calibration":
        protocol = manifest.get("calibration_protocol", {})
        require(protocol.get("cohorts") == [cohort for cohort, _ in COHORTS],
                "calibration outputs do not cover both locked cohorts")
        require(protocol.get("models") == [model for model, _ in MODELS],
                "calibration outputs do not cover both locked model families")
        require(protocol.get("fixed_fit") == "server-only Platt fit, budget 500, repeat 0",
                "calibration outputs do not attest the fixed 500-label repeat-0 fit")
        require(protocol.get("curve_cluster_bootstrap_draws") == 1000,
                "calibration curve bootstrap protocol is not the locked 1000-draw design")
    else:
        protocol = manifest.get("site_protocol", {})
        require(protocol.get("selected_hospitals") == 24,
                "site outputs do not attest the preselected 24-site analysis")
        require(protocol.get("repeats") == 200 and 500 in protocol.get("budgets", []),
                "site outputs lack the locked 200-repeat, 500-patient design")
        require(protocol.get("primary_assessment_weighting") ==
                "all eligible stays from globally held-out patients",
                "site assessment denominator differs from the locked protocol")
        require(isinstance(protocol.get("across_site_estimands"), dict),
                "site outputs lack across-site estimand metadata")

    return {
        "folder": folder,
        "manifest": manifest,
        "tables": tables,
        "verified": {
            "section": expected_section,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_hash,
            "schema_path": str(schema_path),
            "schema_sha256": schema_hash,
            "schema_version": SCHEMA_VERSION,
        },
        "loaded_tables": {},
    }


def load_verified_table(section: dict[str, Any], filename: str) -> list[dict[str, str]]:
    cache = section["loaded_tables"]
    if filename in cache:
        return cache[filename]
    entry = section["tables"].get(filename)
    require(isinstance(entry, dict), f"required table is not registered: {filename}")
    path = section["folder"] / filename
    require(path.is_file(), f"required aggregate table is missing: {filename}")
    expected_hash = entry.get("sha256")
    require(isinstance(expected_hash, str) and SHA256_RE.fullmatch(expected_hash) is not None,
            f"aggregate table has no valid SHA256: {filename}")
    require(sha256_file(path) == expected_hash, f"aggregate table hash mismatch: {filename}")
    column_schema = entry.get("columns")
    require(isinstance(column_schema, list), f"aggregate table schema is missing: {filename}")
    expected_columns = [str(column.get("name")) for column in column_schema
                        if isinstance(column, dict)]
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == expected_columns,
                f"aggregate table columns differ from schema: {filename}")
        require(not ({"subject", "stay_id"} & set(reader.fieldnames or [])),
                f"patient-level identifiers found in aggregate table: {filename}")
        rows.extend(dict(row) for row in reader)
    require(len(rows) == entry.get("rows"), f"aggregate table row count differs from schema: {filename}")
    require(sha256_file(path) == expected_hash, f"aggregate table changed while being read: {filename}")
    cache[filename] = rows
    return rows


def source_file(section: dict[str, Any], filename: str) -> dict[str, str]:
    path = section["folder"] / filename
    if filename == "jobmanifest.json":
        digest = section["verified"]["manifest_sha256"]
    elif filename == "schema.json":
        digest = section["verified"]["schema_sha256"]
    else:
        digest = section["tables"][filename]["sha256"]
    return {"path": str(path.resolve()), "sha256": str(digest)}


def source_files(*items: tuple[dict[str, Any], str]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for section, filename in items:
        item = source_file(section, filename)
        if item["path"] not in seen:
            output.append(item)
            seen.add(item["path"])
    return output


def select(rows: list[dict[str, str]], filters: dict[str, Any], label: str) -> list[dict[str, str]]:
    for key in filters:
        require(rows and key in rows[0], f"{label} lacks required selector field {key}")
    chosen = [row for row in rows if all(row.get(key) == str(value)
                                        for key, value in filters.items())]
    return chosen


def one(rows: list[dict[str, str]], filters: dict[str, Any], label: str) -> dict[str, str]:
    found = select(rows, filters, label)
    require(len(found) == 1, f"{label} selector must match exactly one row; found {len(found)}")
    return found[0]


def number(row: dict[str, str], key: str, label: str) -> float:
    require(key in row, f"{label} lacks numeric field {key}")
    try:
        value = float(row[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} has invalid numeric field {key}: {row.get(key)!r}") from exc
    require(math.isfinite(value), f"{label} has nonfinite required field {key}")
    return value


def integer(row: dict[str, str], key: str, label: str) -> int:
    value = number(row, key, label)
    require(value.is_integer(), f"{label} has noninteger field {key}")
    return int(value)


def boolean(row: dict[str, str], key: str, label: str) -> bool:
    raw = row.get(key, "").strip().lower()
    require(raw in {"true", "false", "1", "0", "yes", "no"},
            f"{label} has invalid boolean field {key}: {row.get(key)!r}")
    return raw in {"true", "1", "yes"}


def support_bins(rows: list[dict[str, str]], filters: dict[str, Any], label: str) -> list[dict[str, Any]]:
    selected = select(rows, filters, label)
    result = []
    for row in sorted(selected, key=lambda item: integer(item, "bin", label)):
        result.append({
            "bin": integer(row, "bin", label),
            "n_stays": integer(row, "n_stays", label),
            "events": integer(row, "events", label),
            "mean_predicted": number(row, "mean_predicted", label),
            "observed": number(row, "observed", label),
        })
    return result


def round_axis_max(values: list[float]) -> float:
    require(values and all(math.isfinite(value) and 0 <= value <= 1 for value in values),
            "calibration axis support is empty or outside probability bounds")
    maximum = max(values)
    return min(1.0, max(0.05, math.ceil(maximum * 20 - 1e-10) / 20.0))


def build_figure_1(calibration: dict[str, Any]) -> dict[str, Any]:
    curve_name = "calibration_curve_points.csv"
    bins_name = "calibration_risk_bins.csv"
    summary_name = "calibration_risk_summary.csv"
    curves = load_verified_table(calibration, curve_name)
    bins = load_verified_table(calibration, bins_name)
    summaries = load_verified_table(calibration, summary_name)
    direct_artifacts = calibration["manifest"].get("inputs_internal", {}).get(
        "direct_server_only_artifacts", {})
    axis_support: list[float] = []
    panel_data: list[tuple[dict[str, Any], dict[str, Any], list[float]]] = []

    for cohort, cohort_label in COHORTS:
        cohort_risk_records: list[dict[str, str]] = []
        for model_id, model_label in MODELS:
            for method, method_label in CALIBRATION_METHODS:
                selector = {"cohort": cohort, "model_id": model_id, "method": method}
                risk = one(summaries, selector, "calibration risk summary")
                cohort_risk_records.append(risk)
                require(integer(risk, "n_stays", "calibration risk summary") > 0,
                        "calibration summary has no assessment stays")
                require(integer(risk, "events", "calibration risk summary") >= 0,
                        "calibration summary has invalid event support")
                require(integer(risk, "nonevents", "calibration risk summary") >= 0,
                        "calibration summary has invalid nonevent support")
                require(integer(risk, "events", "calibration risk summary")
                        + integer(risk, "nonevents", "calibration risk summary")
                        == integer(risk, "n_stays", "calibration risk summary"),
                        "calibration event and nonevent counts do not sum to assessment stays")
                observed = number(risk, "observed_event_rate", "calibration risk summary")
                require(abs(observed - integer(risk, "events", "calibration risk summary") /
                            integer(risk, "n_stays", "calibration risk summary")) <= 1e-9,
                        "calibration event rate differs from its event/stay denominator")
                if method == "raw":
                    require(integer(risk, "fixed_calibrator_budget", "calibration risk summary") == 0
                            and integer(risk, "fixed_calibrator_repeat", "calibration risk summary") == -1,
                            "raw calibration selector has unexpected fixed-fit metadata")
                else:
                    require(integer(risk, "fixed_calibrator_budget", "calibration risk summary") == 500
                            and integer(risk, "fixed_calibrator_repeat", "calibration risk summary") == 0,
                            "post-Platt curve is not the fixed 500-label repeat-0 fit")
                    fit_ok = boolean(risk, "platt_fit_ok", "calibration risk summary")
                    fallback = boolean(risk, "fixed_calibrator_fallback", "calibration risk summary")
                    require(fit_ok or fallback,
                            "failed fixed Platt fit has no attested operational fallback")
                expected_plan = direct_artifacts.get(cohort, {}).get("sampling_plan_sha256")
                require(isinstance(expected_plan, str) and SHA256_RE.fullmatch(expected_plan) is not None,
                        f"{cohort}: completion manifest lacks the verified calibration sampling plan")
                require(risk.get("global_sampling_plan_sha256") == expected_plan,
                        f"{cohort}: curve summary sampling plan differs from completion manifest")
                point_rows = select(curves, selector, "calibration curve points")
                require(point_rows, f"missing calibration curve points for {selector}")
                point_rows.sort(key=lambda row: integer(row, "point_index", "calibration curve point"))
                point_indices = [integer(row, "point_index", "calibration curve point") for row in point_rows]
                require(len(point_indices) == len(set(point_indices)),
                        f"duplicate calibration point index for {selector}")
                curve_status = str(risk.get("curve_status", ""))
                curve_type = str(point_rows[0].get("curve_type", ""))
                require(all(row.get("curve_type") == curve_type for row in point_rows),
                        f"mixed curve types for {selector}")
                point_statuses = {str(row.get("curve_fit_status", "")) for row in point_rows}
                require(len(point_statuses) == 1,
                        f"mixed curve point statuses for {selector}")
                require(curve_type != "unavailable" and curve_status != "insufficient_valid_bootstrap",
                        f"calibration curve or its patient-cluster interval is unavailable for {selector}")
                xs: list[float] = []
                ys: list[float] = []
                lows: list[float] = []
                highs: list[float] = []
                valid_counts = set()
                requested_counts = set()
                for point in point_rows:
                    requested_counts.add(integer(point, "bootstrap_requested", "calibration curve point"))
                    valid_counts.add(integer(point, "bootstrap_valid", "calibration curve point"))
                    xs.append(number(point, "predicted_probability", "calibration curve point"))
                    ys.append(number(point, "smoothed_observed_risk", "calibration curve point"))
                    lows.append(number(point, "ci95_lower", "calibration curve point"))
                    highs.append(number(point, "ci95_upper", "calibration curve point"))
                require(requested_counts == {1000} and len(valid_counts) == 1,
                        f"calibration bootstrap metadata is incomplete for {selector}")
                valid_draws = next(iter(valid_counts))
                require(valid_draws >= 900,
                        f"calibration curve has fewer than 900 valid patient-cluster draws for {selector}")
                require(all(0 <= value <= 1 for value in xs + ys + lows + highs),
                        f"calibration probabilities leave [0,1] for {selector}")
                require(all(xs[i] <= xs[i + 1] for i in range(len(xs) - 1)),
                        f"calibration curve risk grid is not ordered for {selector}")
                is_smooth = curve_type == "restricted_cubic_logistic"
                if is_smooth:
                    require(len(point_rows) == 61 and point_indices == list(range(61))
                            and curve_status == "ok" and point_statuses == {curve_status},
                            f"smooth calibration curve is not a complete 61-point fit for {selector}")
                else:
                    require(len(point_rows) == 1 and point_statuses == {curve_type},
                            f"constant/fallback calibration reference must be one actual point for {selector}")

                bins_selector = selector
                support = support_bins(bins, bins_selector, "calibration risk bins")
                require(bool(support) or not is_smooth,
                        f"smooth calibration curve has no empirical support bins for {selector}")
                if support:
                    require(sum(item["n_stays"] for item in support)
                            == integer(risk, "n_stays", "calibration risk summary")
                            and sum(item["events"] for item in support)
                            == integer(risk, "events", "calibration risk summary"),
                            f"support bins do not cover the calibration assessment denominator for {selector}")
                for item in support:
                    axis_support.append(item["mean_predicted"])
                    require(item["n_stays"] > 0 and 0 <= item["observed"] <= 1,
                            f"invalid support bin for {selector}")
                    require(abs(item["observed"] - item["events"] / item["n_stays"]) <= 1e-9,
                            f"support-bin event rate differs from its denominator for {selector}")
                if xs:
                    axis_support.extend(xs + ys + highs)
                line_x = xs if is_smooth else []
                line_y = ys if is_smooth else []
                line_low = lows if is_smooth else []
                line_high = highs if is_smooth else []
                if not is_smooth:
                    axis_support.append(number(risk, "mean_predicted_probability", "calibration risk summary"))
                label = model_label
                if method == "platt_500_repeat0" and not boolean(
                        risk, "platt_fit_ok", "calibration risk summary"):
                    label += " (Jeffreys fallback)"
                series = {
                    "label": label,
                    "x": line_x,
                    "y": line_y,
                    "low": line_low,
                    "high": line_high,
                    "mean_predicted": number(risk, "mean_predicted_probability", "calibration risk summary"),
                    "observed": observed,
                    "decile_mean_risks": [item["mean_predicted"] for item in support],
                    "support_bins": support,
                    "curve_fit_status": curve_status,
                    "curve_type": curve_type,
                    "bootstrap_requested": 1000,
                    "bootstrap_valid": valid_draws,
                    "source_selector": {
                        "curve_file": curve_name,
                        "curve_where": selector,
                        "curve_rows": len(point_rows),
                        "curve_point_indices": point_indices,
                        "risk_summary_file": summary_name,
                        "risk_summary_where": selector,
                        "support_bins_file": bins_name,
                        "support_bins_where": selector,
                    },
                }
                panel_data.append(({
                    "cohort": cohort,
                    "cohort_label": cohort_label,
                    "method": method,
                    "method_label": method_label,
                    "series": series,
                    "risk": risk,
                }, risk, axis_support[-max(1, len(support) + len(xs) + 1):]))

        common_counts = {(integer(row, "n_stays", "calibration risk summary"),
                          integer(row, "events", "calibration risk summary"),
                          number(row, "observed_event_rate", "calibration risk summary"))
                         for row in cohort_risk_records}
        require(len(common_counts) == 1,
                f"{cohort}: calibration curves do not share one assessment denominator")

    shared_axis_max = round_axis_max(axis_support)
    panels: list[dict[str, Any]] = []
    for cohort, cohort_label in COHORTS:
        for method, method_label in CALIBRATION_METHODS:
            group = [(item, risk) for item, risk, _local_support in panel_data
                     if item["cohort"] == cohort and item["method"] == method]
            require(len(group) == 2,
                    f"{cohort}/{method}: Figure 1 requires both model families")
            reference_risk = group[0][1]
            for _item, risk in group[1:]:
                require((integer(risk, "n_stays", "calibration risk summary"),
                         integer(risk, "events", "calibration risk summary"),
                         number(risk, "observed_event_rate", "calibration risk summary")) ==
                        (integer(reference_risk, "n_stays", "calibration risk summary"),
                         integer(reference_risk, "events", "calibration risk summary"),
                         number(reference_risk, "observed_event_rate", "calibration risk summary")),
                        f"{cohort}/{method}: paired model curves do not share an assessment denominator")
            panels.append({
                "title": f"{cohort_label} · {method_label}",
                "axis_max": shared_axis_max,
                "n_stays": integer(reference_risk, "n_stays", "calibration risk summary"),
                "events": integer(reference_risk, "events", "calibration risk summary"),
                "observed": number(reference_risk, "observed_event_rate", "calibration risk summary"),
                "n_patients": integer(reference_risk, "n_patients", "calibration risk summary"),
                "denominator": "all eligible stays from the fixed assessment patients",
                "series": [item["series"] for item, _risk in group],
            })
    require(len(panels) == 4, "Figure 1 requires exactly four cohort-by-calibration panels")
    return {
        "number": 1,
        "kind": "calibration_grid",
        "claim": "Raw and fixed 500-label Platt predictions retain model-specific probability calibration error across both assessment cohorts.",
        "panels": panels,
        "source_files": source_files(
            (calibration, curve_name), (calibration, bins_name), (calibration, summary_name),
            (calibration, "jobmanifest.json")),
        "uncertainty_scope": (
            "Smooth curves use pointwise 95% patient-cluster bootstrap intervals over the fixed assessment patients "
            "(1000 requested draws; at least 900 valid per curve). Curves are restricted to observed p01–p99 support. "
            "Constant-score or operational Jeffreys-fallback predictions are shown as their actual mean-prediction/event-rate point with the available patient-cluster interval; no curve is imputed."
        ),
        "assessment_scope": "All eligible assessment stays; the same patient-level assessment roles are used for FM and V+P LR within each cohort.",
        "axis_max": shared_axis_max,
    }


def build_figure_5(sites: dict[str, Any]) -> dict[str, Any]:
    filename = "across_site_adaptation_summary.csv"
    rows = load_verified_table(sites, filename)
    protocol = sites["manifest"]["site_protocol"]
    selected_sites = int(protocol.get("selected_hospitals", 0))
    feasibility_where = {"budget": 500, "model_id": MODELS[0][0], "method": "no_update",
                         "metric": "brier", "aggregation": "macro", "weighting": "stay_weighted"}
    feasibility_row = one(rows, feasibility_where, "500-label across-site feasibility")
    feasible_sites = integer(feasibility_row, "feasible_site_count", "site feasibility")
    infeasible_sites = integer(feasibility_row, "infeasible_site_count", "site feasibility")
    require(selected_sites == 24
            and integer(feasibility_row, "selected_site_count", "site feasibility") == selected_sites
            and feasible_sites > 0 and feasible_sites + infeasible_sites == selected_sites,
            "500-label site feasibility counts are incomplete or inconsistent")
    estimands = protocol["across_site_estimands"]
    require("macro_patient_weighted" in estimands and "bootstrap" in estimands,
            "site manifest lacks across-site estimand or uncertainty definitions")

    all_selected: list[dict[str, str]] = []
    for model_id, _ in MODELS:
        for method in ("no_update", "platt"):
            for metric in ("brier", "logloss"):
                where = {"budget": 500, "model_id": model_id, "method": method,
                         "metric": metric, "aggregation": "macro", "weighting": "stay_weighted"}
                row = one(rows, where, "across-site adaptation summary")
                require(row.get("status") == "complete_1000_global_patient_cluster_draws",
                        f"500-label site summary has incomplete uncertainty for {where}")
                require(integer(row, "selected_site_count", "site summary") == selected_sites
                        and integer(row, "feasible_site_count", "site summary") == feasible_sites
                        and integer(row, "infeasible_site_count", "site summary") == infeasible_sites,
                        f"site summary feasibility differs from manifest for {where}")
                require(integer(row, "adaptation_repeats_averaged", "site summary") == 200,
                        f"site summary does not average all 200 adaptation repeats for {where}")
                require(integer(row, "bootstrap_requested", "site summary") == 1000
                        and integer(row, "bootstrap_valid", "site summary") == 1000,
                        f"site summary lacks 1000 valid patient-cluster draws for {where}")
                denominator = {
                    key: integer(row, key, "site summary")
                    for key in ("feasible_assessment_stays", "feasible_unique_assessment_patients",
                                "feasible_site_patient_memberships", "global_cross_site_assessment_patients")
                }
                all_selected.append({"row": row, "where": where, "denominator": denominator})

    rows_by_metric: list[dict[str, Any]] = []
    for metric, metric_label in (("brier", "Brier score"), ("logloss", "Log loss")):
        plot_rows: list[dict[str, Any]] = []
        interval_max = 0.0
        for model_id, model_label in MODELS:
            series = []
            for method, label in (("no_update", "Raw"),
                                  ("platt", "Platt (500 labels)")):
                item = next(entry for entry in all_selected
                            if entry["where"]["model_id"] == model_id
                            and entry["where"]["method"] == method
                            and entry["where"]["metric"] == metric)
                row = item["row"]
                estimate = number(row, "estimate", "site summary")
                low = number(row, "bootstrap_ci_95_lower", "site summary")
                high = number(row, "bootstrap_ci_95_upper", "site summary")
                require(low <= high and estimate >= 0 and low >= 0,
                        f"invalid nonnegative loss interval for {model_id}/{method}/{metric}")
                fit_info: dict[str, Any] = {}
                if method == "platt":
                    fit_info = {
                        "fit_attempts": integer(row, "fit_attempts", "site summary"),
                        "fit_successes": integer(row, "fit_successes", "site summary"),
                        "fit_failures": integer(row, "fit_failures", "site summary"),
                        "fit_failure_rate": number(row, "fit_failure_rate", "site summary"),
                    }
                    require(fit_info["fit_attempts"] == fit_info["fit_successes"] + fit_info["fit_failures"],
                            "site Platt fit counts do not reconcile")
                series.append({
                    "label": label,
                    "estimate": estimate,
                    "low": low,
                    "high": high,
                    "selected_site_count": selected_sites,
                    "feasible_site_count": feasible_sites,
                    "infeasible_site_count": infeasible_sites,
                    **fit_info,
                    "source_selector": {
                        "file": filename,
                        "where": item["where"],
                        "rows": 1,
                        "uncertainty_fields": ["bootstrap_ci_95_lower", "bootstrap_ci_95_upper"],
                    },
                })
                interval_max = max(interval_max, high)
            plot_rows.append({"label": model_label, "series": series})
        denominator_set = {
            tuple(sorted(item["denominator"].items()))
            for item in all_selected if item["where"]["metric"] == metric
        }
        require(len(denominator_set) == 1,
                f"{metric}: across-site methods/models do not share a denominator")
        denominator = dict(next(iter(denominator_set)))
        rows_by_metric.append({
            "title": (f"{metric_label}\n500 labels; {feasible_sites}/{selected_sites} sites feasible"),
            "xlabel": f"Macro mean {metric_label.lower()}",
            "reference": 0,
            "xlim": [0, max(interval_max * 1.08, 1e-6)],
            "show_labels": True,
            "denominators": denominator,
            "selected_site_count": selected_sites,
            "feasible_site_count": feasible_sites,
            "infeasible_site_count": infeasible_sites,
            "rows": plot_rows,
        })
    return {
        "number": 5,
        "kind": "forest_grid",
        "left_margin": 0.17,
        "wspace": 0.50,
        "claim": "Across the preselected eICU sites feasible for a 500-patient update, Platt adaptation changes macro site-level Brier and log-loss performance.",
        "panels": rows_by_metric,
        "source_files": source_files((sites, filename), (sites, "jobmanifest.json")),
        "uncertainty_scope": (
            "Macro stay-weighted loss averages each feasible site equally after averaging 200 adaptation repeats. "
            "Intervals are 95% bootstrap intervals from 1000 shared global eICU patient-cluster resamples; "
            "cross-site patients retain one shared multiplicity. The panel title reports selected, feasible, and infeasible site counts."
        ),
        "adaptation_scope": "Random within-site adaptation with globally held-out patient assessment; no temporal validation claim.",
        "site_estimand": estimands["macro_stay_weighted"],
        "fit_failure_note": "Failed Platt fits operationally use the prespecified Jeffreys prevalence fallback; per-model counts are attached to the plotted series metadata.",
    }


def build_figure_6(calibration: dict[str, Any]) -> dict[str, Any]:
    filename = "paired_loss_gap_bootstrap.csv"
    rows = load_verified_table(calibration, filename)
    panels = []
    for cohort, cohort_label in COHORTS:
        for metric, metric_label in (("brier", "Brier"), ("logloss", "Log loss")):
            series = []
            for contrast, label in CONTRASTS:
                estimates: list[float] = []
                lows: list[float] = []
                highs: list[float] = []
                row_selectors = []
                denominator_values = set()
                adaptation_scopes = set()
                for budget in BUDGETS:
                    where = {"cohort": cohort, "budget": budget, "metric": metric,
                             "contrast": contrast, "weighting": "stay_weighted"}
                    row = one(rows, where, "paired loss gap summary")
                    require(integer(row, "bootstrap_requested", "paired loss gap") == 1000
                            and integer(row, "bootstrap_valid", "paired loss gap") == 1000,
                            f"paired gap lacks 1000 valid patient-cluster draws for {where}")
                    require(integer(row, "adaptation_draws", "paired loss gap") == 200,
                            f"paired gap does not include all 200 adaptation repeats for {where}")
                    estimate = number(row, "estimate", "paired loss gap")
                    low = number(row, "bootstrap_ci_95_lower", "paired loss gap")
                    high = number(row, "bootstrap_ci_95_upper", "paired loss gap")
                    require(low <= high, f"paired gap interval endpoints are reversed for {where}")
                    estimates.append(estimate)
                    lows.append(low)
                    highs.append(high)
                    denominator_values.add((integer(row, "assessment_stays", "paired loss gap"),
                                            integer(row, "assessment_patients", "paired loss gap")))
                    adaptation_scopes.add(str(row.get("interval_scope", "")))
                    row_selectors.append({"where": where, "rows": 1})
                require(len(denominator_values) == 1 and all(adaptation_scopes),
                        f"{cohort}/{metric}/{contrast}: inconsistent denominator or uncertainty scope")
                series.append({
                    "label": label,
                    "budgets": list(BUDGETS),
                    "estimate": estimates,
                    "low": lows,
                    "high": highs,
                    "source_selector": {
                        "file": filename,
                        "weighting": "stay_weighted",
                        "row_count": len(row_selectors),
                        "rows": row_selectors,
                    },
                    "assessment_denominator": {
                        "stays": next(iter(denominator_values))[0],
                        "patients": next(iter(denominator_values))[1],
                    },
                    "interval_scope": sorted(adaptation_scopes),
                })
            panels.append({
                "title": f"{cohort_label} · {metric_label}",
                "ylabel": "Loss gap (FM − V+P LR)",
                "series": series,
                "positive_gap_interpretation": "FM has greater loss than V+P LR",
                "contrast_definition": "gap_change = post-Platt gap − raw gap",
            })
    require(len(panels) == 4 and all(len(panel["series"]) == 3 for panel in panels),
            "Figure 6 requires four cohort-by-score panels and all three paired contrasts")
    return {
        "number": 6,
        "kind": "budget_grid",
        "claim": "Across all four adaptation budgets, Platt updating changes the paired FM-versus-V+P LR Brier and log-loss gaps in cohort-specific ways.",
        "panels": panels,
        "source_files": source_files((calibration, filename), (calibration, "jobmanifest.json")),
        "uncertainty_scope": (
            "Point estimates compare the FM and V+P LR per-stay loss gaps using the means over 200 adaptation fits. "
            "Intervals are paired 95% patient-cluster bootstrap intervals conditional on frozen assessment scores "
            "(1000 draws); stay-weighted estimates are shown. Positive gaps mean greater FM loss; gap change is post-Platt minus raw."
        ),
        "adaptation_budgets": list(BUDGETS),
    }


def build_specs(calibration_dir: Path, sites_dir: Path) -> dict[str, Any]:
    calibration = verify_section(calibration_dir, "calibration")
    sites = verify_section(sites_dir, "sites")
    for filename in REQUIRED_TABLES["calibration"]:
        require(filename in calibration["tables"], f"calibration completion manifest lacks {filename}")
    for filename in REQUIRED_TABLES["sites"]:
        require(filename in sites["tables"], f"site completion manifest lacks {filename}")
    figures = [build_figure_1(calibration), build_figure_5(sites), build_figure_6(calibration)]
    return {
        "status": "analysis_inputs_verified",
        "bundle_scope": "calibration_figure_specs_subset; merge with verified Figures 2–4 before final plotting",
        "input_verification": {
            "calibration": calibration["verified"],
            "sites": sites["verified"],
        },
        "figures": figures,
    }


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration_dir", type=Path,
                        help="completed aggregate calibration output directory")
    parser.add_argument("sites_dir", type=Path,
                        help="completed aggregate site output directory")
    parser.add_argument("output_json", type=Path,
                        help="destination for verified Figure 1, 5, and 6 specs")
    args = parser.parse_args()
    document = build_specs(args.calibration_dir, args.sites_dir)
    write_json_atomic(args.output_json, document)
    print(json.dumps({"status": document["status"],
                      "figures": [figure["number"] for figure in document["figures"]],
                      "output": str(args.output_json.resolve())}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
