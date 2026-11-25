"""Generate mapping_table.csv: 7 libraries x 48 canonical variables.

Coverage for MIMIC/eICU/NWICU is computed from the extracted event tables
(fraction of cohort stays with >=1 observation in 0-24 h).  For the four
libraries without event-level data on this server the row is registered as
summary_only / not_available with the reason, never silently filled.
"""
import os, json
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract import VARS, MIMIC_CHART, MIMIC_LAB, EICU_LAB, EICU_VITAL, NWICU_CHART

ROOT = "/data/admin/fengshuo/ehr_fm_20261102"
INT = f"{ROOT}/data/interim"
PRIOR = "/data/admin/fengshuo/multidb_benchmark_20261101/05_docs/mapping_table.csv"

UNIT = {"age": "years", "sex": "M/F", "weight": "kg", "height": "cm", "bmi": "kg/m2",
        "hr": "bpm", "sbp": "mmHg", "dbp": "mmHg", "map": "mmHg", "rr": "/min",
        "temp": "C", "spo2": "%", "etco2": "mmHg", "gcs_total": "3-15",
        "gcs_motor": "1-6", "gcs_eye": "1-4", "fio2": "%", "peep": "cmH2O",
        "tv": "mL", "ppeak": "cmH2O", "pao2": "mmHg", "paco2": "mmHg", "ph": "-",
        "be": "mmol/L", "creatinine": "mg/dL", "bun": "mg/dL", "urine": "mL",
        "lactate": "mmol/L", "wbc": "10^9/L", "hgb": "g/dL", "platelet": "10^9/L",
        "inr": "-", "aptt": "s", "sodium": "mmol/L", "potassium": "mmol/L",
        "chloride": "mmol/L", "calcium": "mg/dL", "magnesium": "mg/dL",
        "phosphate": "mg/dL", "glucose": "mg/dL", "albumin": "g/dL",
        "bili": "mg/dL", "ast": "U/L", "alt": "U/L", "crp": "mg/L",
        "vaso": "0/1", "rrt": "0/1", "sedation": "0/1"}

ALIAS = {"temp": "temp_c", "hgb": "hemoglobin", "platelet": "platelet",
         "bili": "bilirubin_total", "be": "base_excess"}


def coverage(db):
    ev = pd.read_parquet(f"{INT}/events_{db}.parquet")
    co = pd.read_parquet(f"{INT}/cohort_{db}.parquet")
    n = co["stay_id"].nunique()
    sub = ev[ev["hour"] < 24]
    c = sub.groupby("var")["stay_id"].nunique() / max(n, 1)
    return c.to_dict(), int(n)


def src_mimic(v):
    if v in MIMIC_CHART:
        return "icu/chartevents", ",".join(str(i) for i, _ in MIMIC_CHART[v])
    if v in MIMIC_LAB:
        return "hosp/labevents", str(MIMIC_LAB[v])
    return {"urine": ("icu/outputevents", "226559,..."),
            "vaso": ("icu/inputevents", "221906,221289,221662,222315,221749"),
            "rrt": ("icu/procedureevents", "225441,225802"),
            "sedation": ("icu/inputevents", "222168,221668,225942"),
            "age": ("hosp/patients", "anchor_age"), "sex": ("hosp/patients", "gender"),
            "weight": ("icu/chartevents", "226512,224639"),
            "height": ("icu/chartevents", "226730"),
            "bmi": ("derived", "weight/(height/100)^2")}.get(v, ("-", "-"))


def src_eicu(v):
    if v in EICU_LAB:
        return "lab", EICU_LAB[v]
    if v in EICU_VITAL:
        return "vitalPeriodic(truncated)", EICU_VITAL[v]
    return {"urine": ("intakeOutput", "celllabel~urine"),
            "vaso": ("infusionDrug", "norepinephrine|dopamine|vasopressin|..."),
            "sedation": ("infusionDrug", "propofol|midazolam|fentanyl|dexmedetomidine"),
            "rrt": ("treatment", "renal replacement"),
            "age": ("patient", "age"), "sex": ("patient", "gender"),
            "weight": ("patient", "admissionweight"),
            "height": ("patient", "admissionheight"),
            "bmi": ("derived", "weight/(height/100)^2")}.get(v, ("-", "-"))


def src_nwicu(v):
    if v in NWICU_CHART:
        return "nw_icu/chartevents", ",".join(map(str, NWICU_CHART[v]))
    return {"age": ("nw_hosp/patients", "anchor_age"), "sex": ("nw_hosp/patients", "gender"),
            "weight": ("nw_icu/chartevents", "326531"),
            "height": ("nw_icu/chartevents", "326707"),
            "bmi": ("nw_icu/chartevents", "300001")}.get(v, ("nw_hosp/labevents", "label~regex"))


def main():
    prior = pd.read_csv(PRIOR)
    prior["lib"] = prior["library"].str.lower()
    cov = {}; nstay = {}
    for db in ["mimic", "eicu", "nwicu", "sicdb", "hirid", "c2012", "zigong"]:
        try:
            cov[db], nstay[db] = coverage(db)
        except Exception as e:
            cov[db], nstay[db] = {}, 0
            print("coverage failed", db, e)

    rows = []
    for lib in ["mimic", "eicu", "nwicu", "sicdb", "hirid", "c2012", "zigong", "amsterdam", "inspire"]:
        for v in VARS:
            note = ""
            if lib == "mimic":
                t, f = src_mimic(v); st = "event_level"; c = min(cov["mimic"].get(v, 0.0), 1.0)
                if v in ("age", "sex"):
                    c = 1.0; note = "cohort table, not an event"
            elif lib == "eicu":
                t, f = src_eicu(v); st = "event_level"; c = min(cov["eicu"].get(v, 0.0), 1.0)
                if v in ("age", "sex"):
                    c = 1.0; note = "cohort table, not an event"
                if v in EICU_VITAL:
                    note = "vitalPeriodic gzip truncated: coverage reflects partial file"
            elif lib == "nwicu":
                t, f = src_nwicu(v); st = "event_level"; c = min(cov["nwicu"].get(v, 0.0), 1.0)
                if v in ("age", "sex"):
                    c = 1.0; note = "cohort table, not an event"
            elif lib == "zigong":
                t, f = "events_zigong (built locally)", "dtNursingChart.csv / dtLab.csv (hand-mapped)"
                st = "event_level"; c = min(cov.get(lib, {}).get(v, 0.0), 1.0)
                if v in ("age", "sex"):
                    c = 1.0; note = "cohort table, not an event"
                if v in ("gcs_total", "gcs_motor", "gcs_eye", "weight", "height", "bmi", "vaso", "rrt", "sedation", "peep", "tv", "ppeak", "be", "ph"):
                    note = "not mapped from Zigong source tables"
            elif lib in ("sicdb", "hirid", "c2012"):
                t, f = f"events_{lib} (built locally)", "see code/*_build.py / c2012_build.py"
                st = "event_level"; c = min(cov.get(lib, {}).get(v, 0.0), 1.0)
                if v in ("age", "sex"):
                    c = 1.0; note = "cohort table, not an event"
                if lib == "c2012" and v in ("vaso", "rrt", "sedation", "gcs_motor", "gcs_eye"):
                    note = "not measured in Challenge 2012"
                if lib == "hirid" and v in ("vaso", "sedation"):
                    note = "drug table not extracted"
                if lib == "sicdb" and v in ("vaso", "sedation", "gcs_total", "inr", "aptt"):
                    note = "not available in SICdb extract"
            else:
                key = ALIAS.get(v, v)
                m = prior[(prior.lib == lib) & (prior.canonical_variable == key)]
                if len(m):
                    t = m.iloc[0]["source_table"]; f = str(m.iloc[0]["source_field"])
                    c = float(m.iloc[0]["coverage"]) if pd.notna(m.iloc[0]["coverage"]) else np.nan
                    st = "summary_only"
                    note = "event-level stream not available on server; prior 27-var summary extract only"
                else:
                    t, f, c = "not_event_level_on_server", "n/a", np.nan
                    st = "not_available"
                    note = "no event-level stream and no prior summary mapping"
            rows.append({"library": lib, "canonical_variable": v, "source_table": t,
                         "source_field": f, "unit": UNIT.get(v, ""), "conversion": "-" if v not in ("temp",) else "F->C",
                         "coverage": round(float(c), 4) if pd.notna(c) else "",
                         "status": st, "note": note})
    df = pd.DataFrame(rows)
    df.to_csv(f"{ROOT}/mapping_table.csv", index=False)
    print(df.groupby(["library", "status"]).size())
    print("rows", len(df), "empty structured cells:",
          int(((df[["source_table", "source_field", "status"]] == "-").sum().sum())))


if __name__ == "__main__":
    main()
