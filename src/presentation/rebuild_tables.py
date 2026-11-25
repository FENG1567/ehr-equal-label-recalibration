"""Rebuild the revised 11 public tables from aggregates and verify each cell."""
import csv
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def records(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def one(rows, **filters):
    found = [row for row in rows if all(str(row[key]) == str(value) for key, value in filters.items())]
    if len(found) != 1:
        raise ValueError(f"Expected one aggregate row for {filters}; found {len(found)}")
    return found[0]


def fmt(value):
    return f"{float(value):.5f}"


def interval(row, metric):
    return f"{fmt(row[metric + '_mean'])} ({fmt(row[metric + '_p05'])}–{fmt(row[metric + '_p95'])})"


def integer(value):
    return int(float(value or 0))


def canonical(rows):
    return [[re.sub(r"\s+", "", str(value)).replace("–", "-").replace("—", "-") for value in row] for row in rows]


def check_cells(name, expected, actual):
    if canonical(expected) != canonical(actual):
        raise ValueError(f"Rebuilt cells differ from data/presentation/tables.json: {name}")


def rebuild():
    legacy_path = Path(__file__).with_name("rebuild_tables_r2.py")
    spec = importlib.util.spec_from_file_location("legacy_table_rebuilder", legacy_path)
    legacy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy_module)
    legacy = legacy_module.rebuild()  # checks the former seven-table canonical snapshot

    snapshot = json.loads((ROOT / "data/presentation/tables.json").read_text(encoding="utf-8"))
    plan_main2_rows = [list(row) for row in snapshot["main_tables"][1]["rows"]]
    plan_s2_rows = {
        item["number"]: [list(row) for row in item["rows"]]
        for item in snapshot["supplementary_tables"] if item["number"] in {"2A", "2B", "2C", "2D"}
    }
    source = records(ROOT / "data/aggregates/Data15/calibration/global_50_500_summary.csv")
    cohort_labels = [("eicu_semantic_v4_2", "eICU"), ("nwicu_discharge_v4_1", "NWICU")]

    main2_rows = []
    for cohort, label in cohort_labels:
        for budget in (50, 500):
            for model in ("FM", "V+P LR"):
                raw = one(source, cohort=cohort, model=model, budget=budget, method="no_update")
                updated = one(source, cohort=cohort, model=model, budget=budget, method="platt")
                main2_rows.append([
                    label,
                    str(budget),
                    model,
                    f"{float(updated['events_p05']):.1f}–{float(updated['events_p95']):.1f}",
                    f"{integer(updated['fit_failures'])}/{integer(updated['fallback_count'])}",
                    f"{fmt(raw['brier_mean'])} → {fmt(updated['brier_mean'])}",
                    f"{fmt(raw['logloss_mean'])} → {fmt(updated['logloss_mean'])}",
                ])
    if len(main2_rows) != 8 or any(len(row) != 7 for row in main2_rows):
        raise ValueError("Main Table 2 must contain eight rows and seven columns")
    check_cells("Table 2", main2_rows, plan_main2_rows)
    snapshot["main_tables"][1]["rows"] = main2_rows

    blocks = (
        ("2A", "eicu_semantic_v4_2", 50),
        ("2B", "eicu_semantic_v4_2", 500),
        ("2C", "nwicu_discharge_v4_1", 50),
        ("2D", "nwicu_discharge_v4_1", 500),
    )
    methods = (("no_update", "No update"), ("intercept", "Intercept-only"), ("platt", "Platt"))
    for number, cohort, budget in blocks:
        table = next(item for item in snapshot["supplementary_tables"] if item["number"] == number)
        rows = []
        for model in ("FM", "V+P LR"):
            for method, label in methods:
                row = one(source, cohort=cohort, model=model, budget=budget, method=method)
                attempts = 0 if method == "no_update" else integer(row.get("fit_attempts") or row.get("attempts"))
                failures = 0 if method == "no_update" else integer(row.get("fit_failures"))
                fallback = 0 if method == "no_update" else integer(row.get("fallback_count"))
                rows.append([
                    model,
                    label,
                    f"{attempts}/{failures}/{fallback}",
                    f"{float(row['events_p05']):.1f}–{float(row['events_p95']):.1f}",
                    interval(row, "brier"),
                    interval(row, "logloss"),
                ])
        if len(rows) != 6 or any(len(row) != 6 for row in rows):
            raise ValueError(f"Supplementary Table {number} must contain six rows and six columns")
        check_cells(f"Supplementary Table {number}", rows, plan_s2_rows[number])
        table["rows"] = rows

    # Keep all seven former canonical bodies and verify them before output.
    check_cells("Table 1", legacy["main_tables"][0]["rows"], snapshot["main_tables"][0]["rows"])
    if canonical([legacy["main_tables"][0]["columns"]]) != canonical([snapshot["main_tables"][0]["columns"]]):
        raise ValueError("Column headings differ from data/presentation/tables.json: Table 1")
    current_by_previous_number = {
        item["previous_display_number"]: item for item in snapshot["supplementary_tables"]
    }
    if len(legacy["supplementary_tables"]) != 5:
        raise ValueError("Expected five legacy supplementary table bodies")
    for old_index, actual in enumerate(legacy["supplementary_tables"], 1):
        old_number = str(old_index)
        expected = current_by_previous_number[old_number]

        check_cells(f"Canonical former Supplementary Table {old_number}", actual["rows"], expected["rows"])
        if canonical([actual["columns"]]) != canonical([expected["columns"]]):
            raise ValueError(f"Column headings differ from data/presentation/tables.json: {expected['title']}")

    if len(snapshot["main_tables"]) + len(snapshot["supplementary_tables"]) != 11:
        raise ValueError("Expected exactly 11 physical tables")
    return snapshot
