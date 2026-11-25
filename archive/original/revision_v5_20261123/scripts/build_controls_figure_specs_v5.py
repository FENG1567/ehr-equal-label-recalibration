#!/usr/bin/env python3
"""Build verified Figure 2–4 specs from completed v5 control aggregates.

The builder consumes only aggregate CSVs and their completion manifests. It
does not fit models, calculate statistics, or draw figures. Output is a subset
for merging into the six-figure bundle consumed by plot_v5_figures.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ARMS = (
    ("constant_main20k_cache", "Constant embedding"),
    ("within_variable_shuffle", "Within-variable shuffle"),
    ("distinct_patient_donor", "Distinct-patient donor"),
)
FEATURE_FM = "main20k_real"
FEATURE_DIRECT = "direct_token_summary_90"
SOURCE = "mimic"
TARGET_COHORTS = (
    ("eicu_semantic_v4_2", "eICU"),
    ("nwicu_discharge_v4_1", "NWICU"),
)
REFERENCE_MODELS = (
    ("main20k_C1_historical", "historical C1"),
    ("conventional_lr_value_plus_pattern", "V+P LR"),
    ("conventional_hgb_value_plus_pattern", "V+P HGBT"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path.name}")
    return value


def resolve_local_file(folder: Path, recorded_path: Any, label: str) -> Path:
    require(isinstance(recorded_path, str) and recorded_path.strip(),
            f"{label} has no recorded path")
    basename = Path(recorded_path.replace("\\", "/")).name
    require(basename not in {"", ".", ".."}, f"{label} has an invalid filename")
    path = folder / basename
    require(path.is_file(), f"{label} is missing from the supplied result directory: {basename}")
    return path.resolve(strict=True)


def verify_json_pointer(folder: Path, document: dict[str, Any], path_key: str,
                        hash_key: str, label: str) -> tuple[Path, dict[str, Any]]:
    path = resolve_local_file(folder, document.get(path_key), label)
    expected = document.get(hash_key)
    require(isinstance(expected, str) and SHA256_RE.fullmatch(expected) is not None,
            f"{label} has no valid SHA256")
    require(sha256_file(path) == expected, f"{label} SHA256 mismatch")
    return path, read_json(path)


def declared_csv_artifacts(folder: Path, manifest: dict[str, Any],
                           manifest_name: str) -> dict[str, dict[str, Any]]:
    """Resolve every manifest-declared CSV/path hash pair against this folder."""
    found: dict[str, dict[str, Any]] = {}

    def visit(node: Any, prefix: str = "manifest") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key.endswith("_csv_sha256"):
                    path_key = key[:-7]
                    recorded = node.get(path_key)
                    path = resolve_local_file(folder, recorded, f"{prefix}.{key}")
                    require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
                            f"{prefix}.{key} is malformed")
                    require(sha256_file(path) == value,
                            f"{prefix}.{key} does not match {path.name}")
                    previous = found.get(path.name)
                    require(previous is None or previous["sha256"] == value,
                            f"conflicting hashes recorded for {path.name}")
                    found[path.name] = {"path": path, "sha256": value,
                                        "manifest_field": f"{prefix}.{key}"}
                else:
                    visit(value, f"{prefix}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                visit(value, f"{prefix}[{index}]")

    visit(manifest, manifest_name)

    # The source-choice lock records the source validation-grid CSV separately.
    if manifest.get("source_choice_lock"):
        _lock_path, lock = verify_json_pointer(
            folder, manifest, "source_choice_lock", "source_choice_lock_sha256",
            "source-choice lock",
        )
        grid = resolve_local_file(folder, lock.get("grid_csv"), "source C grid CSV")
        grid_hash = lock.get("grid_csv_sha256")
        require(isinstance(grid_hash, str) and SHA256_RE.fullmatch(grid_hash) is not None,
                "source C grid CSV has no valid SHA256")
        require(sha256_file(grid) == grid_hash, "source C grid CSV SHA256 mismatch")
        previous = found.get(grid.name)
        require(previous is None or previous["sha256"] == grid_hash,
                f"conflicting hashes recorded for {grid.name}")
        found[grid.name] = {"path": grid, "sha256": grid_hash,
                            "manifest_field": "source_choice_lock.grid_csv_sha256"}

    # Every aggregate CSV staged in a run directory must be covered by the
    # completion manifest (or its source-choice lock).
    csv_files = {path.name for path in folder.glob("*.csv")}
    unverified = sorted(csv_files - set(found))
    require(not unverified,
            f"{manifest_name} has CSVs without manifest checksums: {unverified}")
    return found


def verify_run(folder_arg: Path, manifest_name: str) -> dict[str, Any]:
    folder = folder_arg.resolve(strict=True)
    require(folder.is_dir(), f"result input is not a directory: {folder}")
    manifest_path = folder / manifest_name
    require(manifest_path.is_file(), f"missing completion manifest: {manifest_name}")
    manifest = read_json(manifest_path)
    require(manifest.get("status") == "complete",
            f"{manifest_name} is not marked complete")
    require(manifest.get("patient_level_predictions_exported") is False,
            f"{manifest_name} does not attest aggregate-only exports")
    artifacts = declared_csv_artifacts(folder, manifest, manifest_name)
    return {"folder": folder, "manifest": manifest, "manifest_path": manifest_path,
            "manifest_sha256": sha256_file(manifest_path), "artifacts": artifacts,
            "loaded": {}}


def load_csv(run: dict[str, Any], filename: str) -> list[dict[str, str]]:
    cache = run["loaded"]
    if filename in cache:
        return cache[filename]
    entry = run["artifacts"].get(filename)
    require(entry is not None, f"CSV is not checksum-registered: {filename}")
    path = entry["path"]
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == entry["sha256"],
            f"aggregate CSV changed before reading: {filename}")
    text = payload.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    require(reader.fieldnames is not None and len(reader.fieldnames) > 0,
            f"aggregate CSV has no header: {filename}")
    require(not ({"subject", "stay_id"} & set(reader.fieldnames)),
            f"patient-level identifiers found in aggregate CSV: {filename}")
    rows = [dict(row) for row in reader]
    require(hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"],
            f"aggregate CSV changed while being read: {filename}")
    cache[filename] = rows
    return rows


def source_file(run: dict[str, Any], filename: str) -> dict[str, str]:
    entry = run["artifacts"].get(filename)
    require(entry is not None, f"source CSV is not checksum-registered: {filename}")
    return {"path": str(entry["path"]), "sha256": entry["sha256"]}


def source_files(*pairs: tuple[dict[str, Any], str]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for run, filename in pairs:
        item = source_file(run, filename)
        if item["path"] not in seen:
            output.append(item)
            seen.add(item["path"])
    return output


def select(rows: list[dict[str, str]], filters: dict[str, Any], label: str) -> list[dict[str, str]]:
    require(bool(rows), f"{label} table is empty")
    for key in filters:
        require(key in rows[0], f"{label} lacks selector field {key}")
    return [row for row in rows if all(row.get(key) == str(value)
                                      for key, value in filters.items())]


def one(rows: list[dict[str, str]], filters: dict[str, Any], label: str) -> dict[str, str]:
    matched = select(rows, filters, label)
    require(len(matched) == 1,
            f"{label} selector must match exactly one row; found {len(matched)}: {filters}")
    return matched[0]


def number(row: dict[str, str], key: str, label: str) -> float:
    require(key in row, f"{label} lacks numeric field {key}")
    try:
        value = float(row[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} has invalid numeric field {key}: {row.get(key)!r}") from exc
    require(math.isfinite(value), f"{label} has nonfinite numeric field {key}")
    return value


def maybe_number(row: dict[str, str], key: str, label: str) -> float | None:
    raw = row.get(key, "")
    if raw is None or raw.strip().lower() in {"", "nan", "na", "null"}:
        return None
    return number(row, key, label)


def selector(filename: str, filters: dict[str, Any],
             uncertainty_fields: list[str] | None = None) -> dict[str, Any]:
    result = {"file": filename, "where": filters, "rows": 1}
    if uncertainty_fields:
        result["uncertainty_fields"] = uncertainty_fields
    return result


def verify_controls(score: dict[str, Any], decoder: dict[str, Any],
                    readout: dict[str, Any]) -> None:
    sm = score["manifest"]
    dm = decoder["manifest"]
    rm = readout["manifest"]
    require(sm.get("cohort") == "eicu_semantic_v4_2",
            "fixed-head controls are not the repaired eICU analysis")
    require(sm.get("scope") == "repaired eICU reviewed main20k_real roster",
            "fixed-head controls have an unsupported roster scope")
    require(isinstance(sm.get("observed_runtime_probability_parity"), dict)
            and sm["observed_runtime_probability_parity"].get("within_tolerance") is True,
            "fixed-head controls do not attest observed-score parity")
    require(dm.get("schema") == "ehr_fm_v5_direct_token_numeric_decoder_1",
            "numeric decoder manifest schema is unsupported")
    require(dm.get("source_db") == "mimic" and dm.get("target_db") == "eicu_semantic_v4_2",
            "numeric decoder source/target cohorts are unsupported")
    require(dm.get("analysis_id") == "main20k_real",
            "numeric decoder is not based on the locked main20k_real analysis")
    require(rm.get("selected_c_calibration_500", {}).get("status") == "complete",
            "readout selected-C calibration evidence is incomplete")
    require(rm.get("model_ids", {}).get("selected_candidate") == "main20k_C_selected",
            "readout manifest does not identify the source-selected C model")
    require(rm.get("model_ids", {}).get("historical_reference") == "main20k_C1_historical",
            "readout manifest does not identify historical C1 explicitly")
    require(rm.get("same_roster_identity_checks", "").startswith("all source and target model arms passed"),
            "readout manifest does not attest matched model rosters")
    require(set(rm.get("target_evaluated", [])) == {name for name, _ in TARGET_COHORTS},
            "readout does not cover both locked target cohorts")

    _lock_path, lock = verify_json_pointer(
        readout["folder"], rm, "source_choice_lock", "source_choice_lock_sha256",
        "source-choice lock",
    )
    require(lock.get("status") == "source_selection_frozen"
            and lock.get("target_outcomes_opened_before_lock") is False,
            "source C selection was not frozen before target outcome scoring")
    require(rm.get("source_c1_refit_matches_historical_probe_within_2e-6") is True,
            "historical C1 reproduction check failed")
    _guard_path, guard = verify_json_pointer(
        readout["folder"], rm, "source_c1_reproduction_guard",
        "source_c1_reproduction_guard_sha256", "historical C1 reproduction guard",
    )
    require(guard.get("status") == "passed" and float(guard.get("historical_head_C", -1)) == 1.0,
            "historical C1 reproduction guard does not attest a C=1 head")
    maxdiff = number({"value": str(guard.get("max_abs_probability_difference"))},
                    "value", "historical C1 reproduction guard")
    tolerance = number({"value": str(guard.get("tolerance"))}, "value",
                       "historical C1 reproduction guard")
    require(maxdiff <= tolerance, "historical C1 reproduction exceeds its attested tolerance")


def build_figure_2(score: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = score["manifest"]
    metrics_file = "fixed_head_control_metrics.csv"
    contrasts_file = "fixed_head_control_paired_contrasts.csv"
    metrics = load_csv(score, metrics_file)
    contrasts = load_csv(score, contrasts_file)
    expected_head = manifest.get("fixed_head")
    require(isinstance(expected_head, str) and expected_head,
            "fixed-head manifest does not identify the source head")
    csv_head = 'historical main20k C=1 source head'
    require(expected_head == 'historical frozen MIMIC main20k C=1 source scaler and logistic probe',
            'fixed-head manifest source identity changed')
    require({row.get('fixed_head') for row in metrics} == {csv_head},
            'fixed-head metric source identity changed')
    score_filters = {
        "cohort": "eicu_semantic_v4_2",
        "scope": "reviewed_external_roster",
        "fixed_head": csv_head,
    }
    denominator = number({"value": str(manifest.get("n_stays"))}, "value",
                         "fixed-head manifest")
    panels = []
    for metric, title, xlabel, direction_field in (
        ("auroc", "Ranking", "Paired AUROC difference (perturbation − observed)",
         "positive_difference_favors_candidate_arm"),
        ("brier", "Brier score", "Paired Brier difference (perturbation − observed)",
         "negative_loss_difference_favors_candidate_arm"),
        ("logloss", "Log loss", "Paired log-loss difference (perturbation − observed)",
         "negative_loss_difference_favors_candidate_arm"),
    ):
        series = []
        for arm, label in ARMS:
            filters = {"cohort": "eicu_semantic_v4_2", "candidate_arm": arm,
                       "reference_arm": "observed_real_cache", "metric": metric}
            if metric != "auroc":
                # Producer enum is `stay`; the plotted estimand is stay-weighted.
                filters["weighting"] = "stay"
            row = one(contrasts, filters, f"Figure 2 {metric} contrast")
            require(row.get(direction_field, "").lower() == "true",
                    f"Figure 2 {metric} contrast has unexpected sign convention")
            require(number(row, "n_stays", "Figure 2 contrast") == denominator,
                    f"Figure 2 {metric}/{arm} denominator differs from the locked roster")
            series.append({
                "label": label,
                "estimate": number(row, "difference_candidate_minus_reference",
                                    f"Figure 2 {metric} contrast"),
                "low": number(row, "ci_95_low", f"Figure 2 {metric} contrast"),
                "high": number(row, "ci_95_high", f"Figure 2 {metric} contrast"),
                "source_selector": selector(contrasts_file, filters,
                                            ["ci_95_low", "ci_95_high"]),
            })
        panels.append({"title": title, "xlabel": xlabel, "reference": 0,
                       "show_labels": False,
                       "rows": [{"label": "Perturbation versus observed tokens",
                                 "series": series}]})

    # The changed-token fractions are descriptive and have no interval in the
    # producer output. Retain them as a separate supplement point panel.
    fraction_rows = []
    fraction_series = []
    arms_to_show = (("observed_real_cache", "Observed cache"),
                    ("constant_main20k_cache", "Constant embedding"),
                    ("within_variable_shuffle", "Within-variable shuffle"),
                    ("distinct_patient_donor", "Distinct-patient donor"))
    for arm, label in arms_to_show:
        filters = {**score_filters, "arm_id": arm}
        row = one(metrics, filters, "Figure 2 token-change metrics")
        require(row.get("input_condition") == ("real" if arm == "observed_real_cache" else
                "constant" if arm == "constant_main20k_cache" else arm),
                f"Figure 2 token-change condition identity differs for {arm}")
        require(number(row, "n_stays", "Figure 2 token-change metrics") == denominator,
                f"Figure 2 token-change denominator differs for {arm}")
        fraction = number(row, "changed_token_fraction", "Figure 2 token-change metrics")
        require(0 <= fraction <= 1, f"Figure 2 token-change fraction is out of range for {arm}")
        fraction_series.append({"label": label, "estimate": fraction, "low": None, "high": None,
                                "source_selector": selector(metrics_file, filters)})
    fraction_rows.append({"label": "Repaired eICU reviewed roster", "series": fraction_series})
    figure2 = {
        "number": 2, "kind": "forest_grid", "left_margin": 0.17, "wspace": 0.5,
        "claim": ("With the historical frozen MIMIC C=1 head fixed, perturbing the repaired eICU "
                  "embedding changes ranking and proper-score performance relative to observed tokens."),
        "panels": panels,
        "source_files": source_files((score, metrics_file), (score, contrasts_file)),
        "uncertainty_scope": (
            "AUROC intervals are paired 95% patient-cluster bootstrap intervals on the full reviewed eICU roster. "
            "Brier/log-loss intervals are paired 95% patient-cluster intervals, stay-weighted, conditional on fixed predictions. "
            "Negative loss differences favor the perturbation; positive AUROC differences favor it. The historical source head is fixed."
        ),
        "fixed_head": expected_head,
        "token_changed_fraction_supplement_number": "S2",
    }
    supplement = {
        "number": "S2", "kind": "forest_grid", "left_margin": 0.24,
        "claim": "The observed, constant, within-variable shuffle, and donor conditions alter different fractions of retained eICU tokens.",
        "panels": [{"title": "Changed-token fraction · eICU", "xlabel": "Fraction of active retained tokens changed",
                    "reference": 0, "rows": fraction_rows}],
        "source_files": source_files((score, metrics_file),),
        "uncertainty_scope": "Descriptive fractions from the verified aggregate; the producer reports no sampling interval for this quantity.",
        "denominator_n_stays": int(denominator),
    }
    return figure2, supplement


def build_figure_3(decoder: dict[str, Any]) -> dict[str, Any]:
    manifest = decoder["manifest"]
    variables = manifest.get("target_variables")
    require(isinstance(variables, list) and len(variables) == 10
            and len(set(variables)) == 10 and all(isinstance(v, str) and v for v in variables),
            "numeric decoder must declare all ten unique target variables")
    filename = "numeric_token_decoder_metrics.csv"
    metrics = load_csv(decoder, filename)
    mode_specs = (
        ("mimic_to_eicu_frozen_source_transfer", "MIMIC-fitted source transfer → eICU"),
        ("target_local_nested_cv", "Target-local nested patient-group CV → eICU"),
    )
    arms = ((FEATURE_FM, "Frozen FM embedding"),
            (FEATURE_DIRECT, "Direct V/B/H token summary"))
    panels = []
    for mode, title in mode_specs:
        rows = []
        for variable in variables:
            series = []
            denominator: int | None = None
            for feature_id, label in arms:
                filters = {"cohort": "eicu_semantic_v4_2", "mode": mode,
                           "variable": variable, "feature_id": feature_id,
                           "visibility": "all"}
                row = one(metrics, filters, f"Figure 3 {mode}/{variable}/{feature_id}")
                require(row.get("status") in {"complete", "no_rows_in_visibility_stratum"},
                        f"Figure 3 row has unsupported status for {variable}/{feature_id}")
                n = (int(number(row, "n_stays", f"Figure 3 {variable}/{feature_id}"))
                     if row.get("n_stays", "").strip() else 0)
                if denominator is None:
                    denominator = n
                else:
                    require(n == denominator,
                            f"Figure 3 model arms have different denominators for {mode}/{variable}")
                estimate = maybe_number(row, "r2", f"Figure 3 {variable}/{feature_id}")
                low = maybe_number(row, "r2_lo", f"Figure 3 {variable}/{feature_id}")
                high = maybe_number(row, "r2_hi", f"Figure 3 {variable}/{feature_id}")
                if estimate is None:
                    require(row.get("r2_status") == "unavailable_constant_target"
                            or row.get("status") == "no_rows_in_visibility_stratum",
                            f"Figure 3 missing R² is not explained for {mode}/{variable}/{feature_id}")
                    low = high = None
                elif low is None or high is None:
                    # Plotter supports honest point-only estimates.
                    low = high = None
                else:
                    require(low <= high, f"Figure 3 R² interval is reversed for {mode}/{variable}/{feature_id}")
                series.append({"label": label, "estimate": estimate, "low": low, "high": high,
                               "missing_label": row.get("r2_status", row.get("status", "Unavailable")),
                               "n_stays": n,
                               "source_selector": selector(filename, filters,
                                                           ["r2_lo", "r2_hi"] if low is not None else None)})
            rows.append({"label": f"{variable} (n={denominator})", "series": series,
                         "variable": variable, "denominator_n_stays": denominator})
        panels.append({"title": title, "xlabel": "R² (same-window numerical recoverability)",
                       "reference": 0, "rows": rows, "show_labels": True})
    contrasts_file = "numeric_token_decoder_paired_contrasts.csv"
    contrasts = load_csv(decoder, contrasts_file)
    contrasts_by_mode = {}
    for mode, _title in mode_specs:
        mode_rows = select(contrasts, {"cohort": "eicu_semantic_v4_2", "mode": mode,
                                       "candidate_feature": FEATURE_DIRECT,
                                       "reference_feature": FEATURE_FM,
                                       "visibility": "all", "metric": "r2"},
                           f"Figure 3 paired contrasts {mode}")
        require(len(mode_rows) == len(variables)
                and {row.get("variable") for row in mode_rows} == set(variables),
                f"Figure 3 paired direct-minus-FM R² contrasts must contain exactly one row per variable in {mode}")
        contrasts_by_mode[mode] = len(mode_rows)
    return {
        "number": 3, "kind": "forest_grid", "left_margin": 0.31, "wspace": 0.62,
        "claim": ("Same-window numerical recoverability differs between the frozen FM embedding and direct "
                  "V/B/H token summaries, with source transfer and target-local nested fits shown separately."),
        "panels": panels,
        "source_files": source_files((decoder, filename), (decoder, contrasts_file)),
        "uncertainty_scope": (
            "Points are the actual R² values for each feature arm on the shared eICU finite-label roster; "
            "available intervals are the producer's patient-cluster bootstrap intervals. Where the producer did not "
            "provide a finite interval, the point is shown without an interval. Negative R² values are retained. "
            "Row labels carry variable-specific stay denominators; no visibility-stratum selection is applied."
        ),
        "modes": [mode for mode, _title in mode_specs],
        "variables": variables,
        "paired_contrast_rows_by_mode": contrasts_by_mode,
    }


def build_figure_4(readout: dict[str, Any]) -> dict[str, Any]:
    manifest = readout["manifest"]
    lock_path, lock = verify_json_pointer(readout["folder"], manifest, "source_choice_lock",
                                          "source_choice_lock_sha256", "source-choice lock")
    selected_c = number({"value": str(lock.get("selected_C"))}, "value", "source-choice lock")
    source_file_name = "source_c_mimic_split2_loss_contrasts.csv"
    target_file_name = "source_c_paired_loss_contrasts.csv"
    source_rows = load_csv(readout, source_file_name)
    target_rows = load_csv(readout, target_file_name)
    expected_ids = manifest.get("model_ids", {})
    require(expected_ids.get("selected_candidate") == "main20k_C_selected"
            and expected_ids.get("historical_reference") == "main20k_C1_historical",
            "Figure 4 model identities are not explicitly attested")
    target_endpoints = manifest.get("target_endpoints", {})
    row_specs: dict[str, list[tuple[str, dict[str, str], str, dict[str, Any]]]] = {
        "brier": [], "logloss": []}
    for metric in ("brier", "logloss"):
        # MIMIC source test carries an explicit historical C1 comparator.
        for reference_id, reference_label in REFERENCE_MODELS:
            filters = {"cohort": "mimic", "scope": "split2_source_test",
                       "candidate_model": "main20k_C_selected",
                       "reference_model": reference_id, "metric": metric,
                       "weighting": "stay_weighted"}
            row = one(source_rows, filters, f"Figure 4 MIMIC {metric}/{reference_id}")
            series_label = f"Selected C={selected_c:g} − {reference_label}"
            filters_for_panel = {**filters}
            row_specs[metric].append(("MIMIC hospital-death split 2", row,
                                      series_label,
                                      {"file": source_file_name, "where": filters_for_panel,
                                       "rows": 1,
                                       "uncertainty_fields": ["ci_95_low", "ci_95_high"]}))

        # Only the named conventional references are available on the same
        # repaired target rosters; the C1 source-probe replay is source-only.
        for cohort, cohort_label in TARGET_COHORTS:
            for reference_id, reference_label in REFERENCE_MODELS[1:]:
                filters = {"cohort": cohort, "scope": "reviewed_external_roster",
                           "candidate_model": "main20k_C_selected",
                           "reference_model": reference_id, "metric": metric,
                           "weighting": "stay_weighted"}
                row = one(target_rows, filters, f"Figure 4 {cohort}/{metric}/{reference_id}")
                endpoint = target_endpoints.get(cohort)
                label = cohort_label if not isinstance(endpoint, str) or not endpoint else f"{cohort_label} · {endpoint}"
                series_label = f"Selected C={selected_c:g} − {reference_label}"
                row_specs[metric].append((label, row, series_label,
                                          {"file": target_file_name, "where": filters,
                                           "rows": 1,
                                           "uncertainty_fields": ["ci_95_low", "ci_95_high"]}))

    panels = []
    for metric, title, xlabel in (
        ("brier", "Brier loss", "Paired stay-weighted difference (selected C − reference)"),
        ("logloss", "Log loss", "Paired stay-weighted difference (selected C − reference)"),
    ):
        grouped: dict[str, dict[str, Any]] = {}
        for cohort_label, row, series_label, source_selector in row_specs[metric]:
            delta = number(row, "difference_new_minus_reference", f"Figure 4 {metric} contrast")
            low = number(row, "ci_95_low", f"Figure 4 {metric} contrast")
            high = number(row, "ci_95_high", f"Figure 4 {metric} contrast")
            n_stays = number(row, "n_stays", f"Figure 4 {metric} contrast")
            require(n_stays > 0 and n_stays.is_integer(),
                    f"Figure 4 {metric} contrast has no assessment stays")
            require(low <= high, f"Figure 4 {metric} interval is reversed")
            group_label = f"{cohort_label} (n={int(n_stays)})"
            grouped.setdefault(group_label, {"n_stays": int(n_stays), "series": []})["series"].append({
                "label": series_label, "estimate": delta, "low": low, "high": high,
                "source_selector": source_selector,
            })
        rows = [{"label": cohort_label, "series": item["series"]}
                for cohort_label, item in grouped.items()]
        panels.append({"title": title, "xlabel": xlabel, "reference": 0,
                       "rows": rows, "show_labels": True})
    return {
        "number": 4, "kind": "forest_grid", "left_margin": 0.30, "wspace": 0.7,
        "claim": ("The MIMIC source-selected FM regularization sensitivity changes paired Brier and log-loss "
                  "relative to conventional value-plus-pattern models on the source split and the same repaired target rosters."),
        "panels": panels,
        "source_files": source_files((readout, source_file_name), (readout, target_file_name)),
        "uncertainty_scope": (
            "Differences are selected-C minus reference losses, stay-weighted on matched assessment rosters. "
            "Intervals are the producer's paired 95% patient-cluster bootstrap intervals conditional on frozen scores. "
            "Negative differences favor the source-selected candidate. The source panel explicitly contrasts against "
            "historical main20k C1; target panels compare the same selected candidate with V+P LR and V+P HGBT."
        ),
        "source_selected_C": selected_c,
        "selected_candidate_model_id": expected_ids["selected_candidate"],
        "historical_C1_model_id": expected_ids["historical_reference"],
        "historical_C1_reproduction_guard": {
            "path": str(resolve_local_file(readout["folder"], manifest.get("source_c1_reproduction_guard"),
                                            "historical C1 guard")),
            "sha256": manifest.get("source_c1_reproduction_guard_sha256"),
        },
        "source_choice_lock": {"path": str(lock_path), "sha256": manifest["source_choice_lock_sha256"]},
        "target_endpoint_metadata": target_endpoints,
    }


def build_specs(score_dir: Path, decoder_dir: Path, readout_dir: Path) -> dict[str, Any]:
    score = verify_run(score_dir, "fixed_head_control_score_manifest.json")
    decoder = verify_run(decoder_dir, "numeric_token_decoder_manifest.json")
    readout = verify_run(readout_dir, "readout_manifest.json")
    verify_controls(score, decoder, readout)
    figure2, supplement2 = build_figure_2(score)
    figure3 = build_figure_3(decoder)
    figure4 = build_figure_4(readout)
    return {
        "status": "analysis_inputs_verified",
        "bundle_scope": "controls_figure_specs_subset; merge with verified Figures 1, 5, and 6 before final plotting",
        "input_verification": {
            "fixed_head_controls": {"manifest_path": str(score["manifest_path"]),
                                     "manifest_sha256": score["manifest_sha256"]},
            "numeric_decoder": {"manifest_path": str(decoder["manifest_path"]),
                                "manifest_sha256": decoder["manifest_sha256"]},
            "source_selected_readout": {"manifest_path": str(readout["manifest_path"]),
                                         "manifest_sha256": readout["manifest_sha256"]},
        },
        "figures": [figure2, figure3, figure4, supplement2],
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
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls-score-dir", required=True, type=Path,
                        help="completed fixed_head_control score output directory")
    parser.add_argument("--decoder-dir", required=True, type=Path,
                        help="completed direct-token numeric decoder output directory")
    parser.add_argument("--readout-dir", required=True, type=Path,
                        help="completed source-selected-C readout output directory")
    parser.add_argument("--output", required=True, type=Path,
                        help="JSON path for the verified Figure 2–4 spec subset")
    args = parser.parse_args()
    bundle = build_specs(args.controls_score_dir, args.decoder_dir, args.readout_dir)
    write_json_atomic(args.output, bundle)
    print(f"CONTROLS_FIGURE_SPECS_VERIFIED output={args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
