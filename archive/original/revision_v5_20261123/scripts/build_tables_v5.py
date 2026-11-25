#!/usr/bin/env python3
"""Build publication table JSON from completed aggregate-only v5 CSVs.

This script is an assembly step. It does not read patient-level inputs or run
statistical analyses; inputs are the three reviewed aggregate CSV artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


BASELINE_COHORTS = (
    "mimic",
    "eicu_semantic_v4_2",
    "nwicu_discharge_v4_1",
    "sicdb_zero_offset_v4_2",
)
MAIN_COHORTS = BASELINE_COHORTS[:3]
CLINICAL_COHORTS = ("mimic", "eicu", "nwicu")
CALIBRATION_COHORTS = ("eicu_semantic_v4_2", "nwicu_discharge_v4_1")
MODELS = ("main20k_real", "conventional_lr_value_plus_pattern")
METHODS = ("intercept", "platt")
BUDGETS = (50, 500)
CLINICAL_COUNTS = {
    "mimic": (73936, 53964, 8143),
    "eicu": (130544, 102957, 11943),
    "nwicu": (19972, 17385, 2226),
}
TABLE1_COUNTS = {
    "mimic": (14890, 10848, 1675),
    "eicu_semantic_v4_2": (130544, 102957, 11943),
    "nwicu_discharge_v4_1": (19972, 17385, 2226),
}


def fail(message: str) -> None:
    raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path, label: str) -> tuple[list[dict[str, str]], set[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} input does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            fail(f"{label} input has no CSV header: {path}")
        rows = list(reader)
        fields = set(reader.fieldnames)
    return rows, fields


def require_columns(fields: set[str], expected: set[str], label: str) -> None:
    missing = sorted(expected - fields)
    if missing:
        fail(f"{label} input is missing required columns: {missing}")


def get_text(row: dict[str, str], key: str, label: str) -> str:
    value = row.get(key)
    if value is None or not value.strip():
        fail(f"{label}: missing {key}")
    return value.strip()


def get_number(row: dict[str, str], key: str, label: str) -> float:
    raw = get_text(row, key, label)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{label}: {key} is not numeric: {raw!r}") from exc
    if not math.isfinite(value):
        fail(f"{label}: {key} is not finite: {raw!r}")
    return value


def get_integer(row: dict[str, str], key: str, label: str) -> int:
    value = get_number(row, key, label)
    if value < 0 or not value.is_integer():
        fail(f"{label}: {key} must be a nonnegative integer, got {value}")
    return int(value)


def fmt(value: float, digits: int = 3) -> str:
    """Render finite numbers without locale-dependent grouping separators."""
    if not math.isfinite(value):
        fail(f"Refusing to render nonfinite table value {value}")
    return f"{value:.{digits}f}"


def fmt_interval(row: dict[str, str], point: str, low: str, high: str,
                 label: str, digits: int = 3) -> str:
    estimate = get_number(row, point, label)
    lower = get_number(row, low, label)
    upper = get_number(row, high, label)
    if lower > estimate or estimate > upper:
        fail(f"{label}: invalid interval for {point}: {lower} <= {estimate} <= {upper}")
    return f"{fmt(estimate, digits)} ({fmt(lower, digits)}–{fmt(upper, digits)})"


def one_per_key(rows: list[dict[str, str]], key: str, label: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        value = get_text(row, key, label)
        if value in result:
            fail(f"{label}: duplicate {key}={value!r}")
        result[value] = row
    return result


def baseline_tables(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    rows, fields = read_csv(path, "baseline")
    required = {
        "cohort_id", "cohort", "endpoint", "n_stays", "n_patients", "events",
        "auroc", "auroc_ci_95_low", "auroc_ci_95_high", "brier",
        "brier_ci_95_low", "brier_ci_95_high", "logloss_point_only",
        "calibration_slope", "calibration_slope_ci_95_low",
        "calibration_slope_ci_95_high", "offset_intercept_fixed_slope_1", "offset_ci_95_low",
        "offset_ci_95_high",
    }
    require_columns(fields, required, "baseline")
    indexed = one_per_key(rows, "cohort_id", "baseline")
    if set(indexed) != set(BASELINE_COHORTS):
        fail(f"baseline cohorts differ from contract: expected {BASELINE_COHORTS}, found {tuple(indexed)}")

    main_rows: list[list[str]] = []
    sic_rows: list[list[str]] = []
    logloss_rows: list[list[str]] = []
    selectors: list[dict[str, str]] = []
    labels = {
        "mimic": "MIMIC test — hospital death",
        "eicu_semantic_v4_2": "Repaired eICU — hospital death",
        "nwicu_discharge_v4_1": "NWICU — hospital death",
        "sicdb_zero_offset_v4_2": "SICdb — exploratory case-period mortality proxy",
    }
    for cohort_id in BASELINE_COHORTS:
        row = indexed[cohort_id]
        label = f"baseline[{cohort_id}]"
        stays, patients, deaths = (get_integer(row, key, label)
                                   for key in ("n_stays", "n_patients", "events"))
        if stays <= 0 or patients <= 0 or patients > stays or deaths > stays:
            fail(f"{label}: inconsistent stay/patient/death denominators")
        if cohort_id in TABLE1_COUNTS and (stays, patients, deaths) != TABLE1_COUNTS[cohort_id]:
            fail(f"{label}: denominators {(stays, patients, deaths)} differ from frozen contract {TABLE1_COUNTS[cohort_id]}")
        display = [
            labels[cohort_id], f"{stays}/{patients}/{deaths}",
            fmt_interval(row, "auroc", "auroc_ci_95_low", "auroc_ci_95_high", label),
            fmt_interval(row, "brier", "brier_ci_95_low", "brier_ci_95_high", label, 4),
            fmt_interval(row, "calibration_slope", "calibration_slope_ci_95_low",
                         "calibration_slope_ci_95_high", label),
            fmt_interval(row, "offset_intercept_fixed_slope_1", "offset_ci_95_low", "offset_ci_95_high", label),
        ]
        logloss_rows.append([
            labels[cohort_id], f"{stays}/{patients}/{deaths}",
            fmt(get_number(row, "logloss_point_only", label), 4),
        ])
        if cohort_id == "sicdb_zero_offset_v4_2":
            sic_rows.append(display)
        selectors.append({"source": "baseline", "cohort_id": cohort_id,
                          "source_cohort": get_text(row, "cohort", label),
                          "source_endpoint": get_text(row, "endpoint", label)})
        if cohort_id in MAIN_COHORTS:
            main_rows.append(display)

    primary = {
        "title": "Primary cohort performance and calibration",
        "columns": ["Cohort and endpoint", "Stays/patients/deaths",
                    "AUROC (95% CI)", "Brier score (95% CI)",
                    "Calibration slope (95% CI)", "Offset intercept (95% CI)"],
        "rows": main_rows,
        "widths": [1.55, 0.9, 1.1, 1.15, 1.3, 1.3],
        "legend": (
            "AUROC, Brier score, calibration slope, and fixed-slope offset intercept are reported with "
            "marginal 95% patient-cluster percentile intervals. Counts are stays/patients/deaths. "
            "MIMIC performance is the source-readout test on eligible frozen split 2 (14890 stays; "
            "10848 patients); the historical encoder used 20000 MIMIC stay rows over two pretraining "
            "epochs; exact historical membership is not certified in this table. This is not an "
            "independent external validation. eICU and NWICU are external "
            "hospital-death cohorts. Inputs are retrospective measurement-time observations; bedside "
            "availability by hour 24 was not verified."
        ),
    }
    sic = {
        "title": "Exploratory SICdb case-period mortality proxy",
        "columns": ["Cohort and endpoint", "Stays/patients/events",
                    "AUROC (95% CI)", "Brier score (95% CI)",
                    "Calibration slope (95% CI)", "Offset intercept (95% CI)"],
        "rows": sic_rows,
        "widths": [1.7, 0.9, 1.1, 1.1, 1.25, 1.25],
        "legend": (
            "SICdb is shown as an exploratory case-period mortality proxy; its available fields do not "
            "verify ICU death or continuous ICU occupancy through hour 24. Estimates and denominators "
            "are preserved from the reviewed aggregate source."
        ),
    }
    logloss = {
        "title": "Point-only log loss across reviewed cohorts",
        "columns": ["Cohort and endpoint", "Stays/patients/events", "Log loss (point only)"],
        "rows": logloss_rows,
        "widths": [1.9, 1.0, 1.1],
        "legend": "Log loss is shown as a point estimate; no interval was available in the reviewed Table 1 aggregate.",
    }
    provenance = {"path": str(path), "sha256": sha256(path),
                  "selectors": selectors,
                  "selected_fields": sorted(required)}
    return primary, sic, logloss, provenance


def calibration_table(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    rows, fields = read_csv(path, "calibration")
    required = {
        "cohort", "model_id", "budget", "method", "attempts", "fit_successes",
        "fallback_count", "events_p05", "events_p95", "brier_mean", "brier_p05",
        "brier_p95", "logloss_mean", "logloss_p05", "logloss_p95",
        "fit_attempts", "fit_failures",
    }
    require_columns(fields, required, "calibration")
    selected: dict[tuple[str, str, int, str], dict[str, str]] = {}
    for row in rows:
        cohort = get_text(row, "cohort", "calibration")
        model = get_text(row, "model_id", "calibration")
        method = get_text(row, "method", "calibration")
        budget = get_integer(row, "budget", "calibration")
        if cohort not in CALIBRATION_COHORTS or model not in MODELS or budget not in BUDGETS or method not in METHODS:
            continue
        key = (cohort, model, budget, method)
        if key in selected:
            fail(f"calibration: duplicate selector {key}")
        selected[key] = row
    expected = {
        (cohort, model, budget, method)
        for cohort in CALIBRATION_COHORTS for model in MODELS
        for budget in BUDGETS for method in METHODS
    }
    if set(selected) != expected:
        fail(f"calibration fitted-update cells incomplete: missing={sorted(expected - set(selected))}; extra={sorted(set(selected) - expected)}")

    table_rows: list[list[str]] = []
    selectors: list[dict[str, Any]] = []
    for key in sorted(expected, key=lambda x: (CALIBRATION_COHORTS.index(x[0]), MODELS.index(x[1]), x[2], METHODS.index(x[3]))):
        cohort, model, budget, method = key
        row = selected[key]
        label = f"calibration[{cohort}/{model}/{budget}/{method}]"
        attempts = get_integer(row, "attempts", label)
        fit_attempts = get_integer(row, "fit_attempts", label)
        successes = get_integer(row, "fit_successes", label)
        failures = get_integer(row, "fit_failures", label)
        fallbacks = get_integer(row, "fallback_count", label)
        if attempts != 200 or fit_attempts != 200 or successes + failures != 200 or failures != 200 - successes:
            fail(f"{label}: expected complete accounting for all 200 attempts")
        if fallbacks != failures:
            fail(f"{label}: fallback count {fallbacks} does not match failed-fit count {failures}")
        ev_lo = get_number(row, "events_p05", label)
        ev_hi = get_number(row, "events_p95", label)
        if ev_lo < 0 or ev_hi > budget or ev_lo > ev_hi:
            fail(f"{label}: invalid empirical event P05/P95")
        metric_values: list[str] = []
        for metric in ("brier", "logloss"):
            mean = get_number(row, f"{metric}_mean", label)
            lo = get_number(row, f"{metric}_p05", label)
            hi = get_number(row, f"{metric}_p95", label)
            if min(mean, lo, hi) < 0 or lo > hi:
                fail(f"{label}: invalid {metric} mean or empirical P05/P95")
            metric_values.append(f"{fmt(mean, 5)} ({fmt(lo, 5)}–{fmt(hi, 5)})")
        table_rows.append([
            "eICU" if cohort == CALIBRATION_COHORTS[0] else "NWICU",
            "FM" if model == MODELS[0] else "V+P LR",
            "Intercept-only" if method == "intercept" else "Platt",
            str(budget), f"{attempts}/{failures}/{fallbacks}",
            f"{fmt(ev_lo, 1)}–{fmt(ev_hi, 1)}", metric_values[0], metric_values[1],
        ])
        selectors.append({"cohort": cohort, "model_id": model, "budget": budget, "method": method})

    table = {
        "title": "Global calibration updates at 50 and 500 labels",
        "columns": ["Cohort", "Model", "Update", "Budget", "Attempts/failed/fallback",
                    "Events P05–P95", "Brier mean (P05–P95)", "Log loss mean (P05–P95)"],
        "rows": table_rows,
        "widths": [0.8, 0.8, 0.95, 0.55, 1.15, 0.95, 1.45, 1.5],
        "legend": (
            "Each cell summarizes all 200 completed adaptation attempts; failed fits remain included and "
            "use the prespecified Jeffreys prevalence fallback. Attempts/failed/fallback gives all three "
            "counts. Event and loss P05–P95 values are empirical 90% ranges across the 200 draws, not "
            "confidence intervals. Losses are evaluated on globally held-out assessment patients and "
            "include all assessment stays. These results concern only the completed 50- and 500-label budgets."
        ),
    }
    return table, {"path": str(path), "sha256": sha256(path),
                   "selectors": selectors, "selected_fields": sorted(required)}


def clinical_table(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    rows, fields = read_csv(path, "clinical")
    required = {
        "database", "scope", "n_stays", "n_patients", "n_events_y1",
        "repeat_patients_n", "repeat_stays_after_first_n", "age_n_missing_stays",
        "age_median_source_units", "age_q25_source_units", "age_q75_source_units",
        "age_topcode_value", "age_at_topcode_n", "sex_male_1_n", "sex_female_0_n",
        "sex_unknown_n", "sex_denominator_stays", "token_total", "token_mean",
        "token_median", "token_q25", "token_q75", "token_cap128_n",
        "token_cap128_definition", "source_split2_rows", "source_split2_patients",
    }
    require_columns(fields, required, "clinical")
    indexed = one_per_key(rows, "database", "clinical")
    if set(indexed) != set(CLINICAL_COHORTS):
        fail(f"clinical cohorts differ from contract: expected {CLINICAL_COHORTS}, found {tuple(indexed)}")
    parsed: dict[str, dict[str, Any]] = {}
    selectors: list[dict[str, str]] = []
    for cohort in CLINICAL_COHORTS:
        row = indexed[cohort]
        label = f"clinical[{cohort}]"
        if get_text(row, "scope", label) != "current_eligible_24h":
            fail(f"{label}: unexpected scope")
        stays = get_integer(row, "n_stays", label)
        patients = get_integer(row, "n_patients", label)
        if (stays, patients) != CLINICAL_COUNTS[cohort][:2]:
            fail(f"{label}: denominators {(stays, patients)} differ from frozen contract {CLINICAL_COUNTS[cohort][:2]}")
        events = get_integer(row, "n_events_y1", label)
        if events != CLINICAL_COUNTS[cohort][2]:
            fail(f"{label}: event count {events} differs from frozen contract {CLINICAL_COUNTS[cohort][2]}")
        repeated_patients = get_integer(row, "repeat_patients_n", label)
        repeated_stays = get_integer(row, "repeat_stays_after_first_n", label)
        age_missing = get_integer(row, "age_n_missing_stays", label)
        age_topcode = get_integer(row, "age_topcode_value", label)
        age_topcoded_n = get_integer(row, "age_at_topcode_n", label)
        female = get_integer(row, "sex_female_0_n", label)
        male = get_integer(row, "sex_male_1_n", label)
        unknown = get_integer(row, "sex_unknown_n", label)
        sex_denom = get_integer(row, "sex_denominator_stays", label)
        token_total = get_integer(row, "token_total", label)
        token_capped = get_integer(row, "token_cap128_n", label)
        if events > stays or repeated_patients > patients or repeated_stays > stays - patients:
            fail(f"{label}: inconsistent event or repeat-patient counts")
        if age_missing > stays or age_topcoded_n > stays - age_missing:
            fail(f"{label}: inconsistent age counts")
        if female + male + unknown != stays or sex_denom != stays:
            fail(f"{label}: sex counts do not reconcile to eligible stay denominator")
        if token_capped > stays or token_total < 0:
            fail(f"{label}: invalid sequence-token counts")
        if get_text(row, "token_cap128_definition", label) != "L >= 128 among currently eligible stays":
            fail(f"{label}: unexpected token cap definition")
        median = get_number(row, "age_median_source_units", label)
        age_q25 = get_number(row, "age_q25_source_units", label)
        age_q75 = get_number(row, "age_q75_source_units", label)
        tok_median = get_number(row, "token_median", label)
        tok_q25 = get_number(row, "token_q25", label)
        tok_q75 = get_number(row, "token_q75", label)
        token_mean = get_number(row, "token_mean", label)
        if age_missing == stays or age_q25 > median or median > age_q75 or tok_q25 > tok_median or tok_median > tok_q75:
            fail(f"{label}: invalid age or token quantiles")
        parsed[cohort] = {
            "stays": stays, "patients": patients, "events": events,
            "repeat_patients": repeated_patients, "repeat_stays": repeated_stays,
            "age_missing": age_missing, "age_median": median, "age_q25": age_q25,
            "age_q75": age_q75, "age_topcode": age_topcode,
            "age_topcoded_n": age_topcoded_n, "female": female, "male": male,
            "unknown": unknown, "token_total": token_total, "token_mean": token_mean,
            "token_median": tok_median, "token_q25": tok_q25, "token_q75": tok_q75,
            "token_capped": token_capped,
        }
        selectors.append({"source": "clinical", "database": cohort,
                          "scope": "current_eligible_24h"})
    mimic = indexed["mimic"]
    if (get_integer(mimic, "source_split2_rows", "clinical[mimic]") != 14890
            or get_integer(mimic, "source_split2_patients", "clinical[mimic]") != 10848):
        fail("clinical[mimic]: split-2 stay/patient counts differ from frozen test roster")

    heads = ["Measure"] + [
        f"{name} (eligible stays n={parsed[cohort]['stays']})"
        for cohort, name in (("mimic", "MIMIC"), ("eicu", "eICU"), ("nwicu", "NWICU"))
    ]
    def cells(fn):
        return [fn(parsed[cohort]) for cohort in CLINICAL_COHORTS]
    table_rows = [
        ["Eligible patients"] + cells(lambda x: str(x["patients"])),
        ["Deaths/events (y=1; all eligible stays)"] + cells(lambda x: str(x["events"])),
        ["Age median (IQR), source units"] + cells(lambda x: f"{fmt(x['age_median'], 1)} ({fmt(x['age_q25'], 1)}–{fmt(x['age_q75'], 1)})"),
        ["Age missing; top-coded at source threshold"] + cells(lambda x: f"{x['age_missing']}; {x['age_topcode']} years: {x['age_topcoded_n']}"),
        ["Sex female/male/unknown (stay counts)"] + cells(lambda x: f"{x['female']}/{x['male']}/{x['unknown']}"),
        ["Repeat patients; extra stays after first"] + cells(lambda x: f"{x['repeat_patients']}; {x['repeat_stays']}"),
        ["Sequence length L median (IQR)"] + cells(lambda x: f"{fmt(x['token_median'], 1)} ({fmt(x['token_q25'], 1)}–{fmt(x['token_q75'], 1)})"),
        ["Total sequence tokens; mean per stay"] + cells(lambda x: f"{x['token_total']}; {fmt(x['token_mean'], 2)}"),
        ["Stays with L ≥ 128"] + cells(lambda x: f"{x['token_capped']}/{x['stays']}"),
        ["MIMIC frozen source split 2 (stays/patients)"] + [
            f"{get_integer(mimic, 'source_split2_rows', 'clinical[mimic]')}/{get_integer(mimic, 'source_split2_patients', 'clinical[mimic]')}",
            "—", "—",
        ],
    ]
    table = {
        "title": "Supplementary clinical and sequence characteristics of all eligible stays",
        "columns": heads,
        "rows": table_rows,
        "widths": [1.8, 1.0, 1.0, 1.0],
        "legend": (
            "All summaries use the exact eligible-stay denominator shown in each cohort header. Deaths are "
            "the reviewed binary outcome count. Age is in source units and remains top-coded (MIMIC/NWICU 91; "
            "eICU 90); top-coded values are not exact ages. Sex counts are female/male/unknown with missing "
            "retained as unknown. Repeat patients have more than one eligible stay; extra stays are counted "
            "after each patient's first. Sequence length L is the frozen token count; L ≥ 128 is the cap count. "
            "MIMIC's full eligible population has 73936 stays and 53964 patients; its distinct frozen split-2 "
            "test has 14890 stays and 10848 patients and is not the all-eligible clinical denominator."
        ),
    }
    return table, {"path": str(path), "sha256": sha256(path),
                   "selectors": selectors, "selected_fields": sorted(required)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="Reviewed v4 table1_source_data.csv aggregate")
    parser.add_argument("--calibration", type=Path, required=True,
                        help="Completed global_50_500_summary.csv aggregate")
    parser.add_argument("--clinical", type=Path, required=True,
                        help="Completed clinical_summary.csv aggregate")
    parser.add_argument("--output", type=Path, required=True,
                        help="Destination table_data.json")
    args = parser.parse_args()

    table1, sic_supplement, logloss_supplement, baseline_provenance = baseline_tables(args.baseline)
    table2, calibration_provenance = calibration_table(args.calibration)
    table3, clinical_provenance = clinical_table(args.clinical)
    result = {
        "schema_version": "ehr_fm_v5_table_data_1",
        "main_tables": [table1, table2],
        "supplementary_tables": [sic_supplement, logloss_supplement, table3],
        "provenance": {
            "status": "complete_aggregate_inputs_validated",
            "sources": {
                "baseline": baseline_provenance,
                "calibration": calibration_provenance,
                "clinical": clinical_provenance,
            },
            "global_selectors": {
                "baseline_primary_cohorts": list(MAIN_COHORTS),
                "baseline_supplementary_cohort": "sicdb_zero_offset_v4_2",
                "calibration_cohorts": list(CALIBRATION_COHORTS),
                "calibration_models": list(MODELS),
                "calibration_methods": list(METHODS),
                "calibration_budgets": list(BUDGETS),
                "clinical_cohorts": list(CLINICAL_COHORTS),
                "clinical_scope": "current_eligible_24h",
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
