"""Adapt PhysioNet Challenge 2012 (set-a) to the project's event/cohort format.

Format: one txt per record with lines RecordID,Parameter,Time,Value.
Time = HH:MM (may be negative).  Static fields (Age/Gender/Height/Weight/ICUType)
appear at time 00:00.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

ROOT = "/data/admin/fengshuo/ehr_fm_20261102"
SRC = f"{ROOT}/data/external/challenge2012/set-a"
OUTCOMES = f"{ROOT}/data/external/challenge2012/outcomes-a.txt"
OUT = f"{ROOT}/data/interim"

MAP = {
    "HR": "hr", "Temp": "temp", "SysABP": "sbp", "DiasABP": "dbp", "MAP": "map",
    "RespRate": "rr", "SaO2": "spo2", "FiO2": "fio2", "PaO2": "pao2", "PaCO2": "paco2",
    "pH": "ph", "Creatinine": "creatinine", "BUN": "bun", "Lactate": "lactate",
    "WBC": "wbc", "Hemoglobin": "hgb", "Platelets": "platelet", "Sodium": "sodium",
    "Potassium": "potassium", "Chloride": "chloride", "Calcium": "calcium",
    "Magnesium": "magnesium", "Phosphate": "phosphate", "Glucose": "glucose",
    "Albumin": "albumin", "Bilirubin": "bili", "AST": "ast", "ALT": "alt",
    "INR": "inr", "PTT": "aptt", "Urine": "urine", "GCS": "gcs_total",
}
STATIC = {"Age": "age", "Gender": "sex", "Height": "height", "Weight": "weight"}
RANGE = {"hr": (20, 250), "sbp": (30, 300), "dbp": (10, 200), "map": (20, 220), "rr": (2, 80),
         "temp": (28, 43), "spo2": (40, 100), "fio2": (21, 100), "pao2": (20, 700),
         "paco2": (10, 150), "ph": (6.5, 7.9), "creatinine": (0.1, 25), "bun": (1, 300),
         "lactate": (0.1, 30), "wbc": (0.05, 200), "hgb": (1.5, 25), "platelet": (1, 1500),
         "sodium": (90, 190), "potassium": (1, 10), "chloride": (50, 140), "calcium": (3, 20),
         "magnesium": (0.3, 6), "phosphate": (0.2, 20), "glucose": (10, 900),
         "albumin": (0.8, 7), "bili": (0.1, 60), "ast": (1, 20000), "alt": (1, 20000),
         "inr": (0.5, 20), "aptt": (10, 200), "urine": (0, 5000), "gcs_total": (3, 15)}


def parse(f):
    rid = os.path.basename(f).replace(".txt", "")
    rows, stat = [], {}
    with open(f, errors="replace") as fh:
        next(fh, None)
        for line in fh:
            p = line.strip().split(",")
            if len(p) < 3:
                continue
            t, name, v = p[0], p[1], p[2]
            if name in STATIC:
                try:
                    stat[STATIC[name]] = float(v)
                except Exception:
                    pass
                continue
            if name not in MAP:
                continue
            try:
                hh, mm = t.split(":")
                hour = int(hh) + int(mm) / 60.0
                val = float(v)
            except Exception:
                continue
            if hour < 0 or hour > 48:
                continue
            rows.append((rid, hour, MAP[name], val))
    return rid, rows, stat


def main():
    files = sorted(glob.glob(os.path.join(SRC, "*.txt")))
    print("records", len(files))
    ev, meta = [], []
    for i, f in enumerate(files):
        rid, rows, stat = parse(f)
        ev.extend(rows)
        meta.append(dict(stay_id=rid, **stat))
        if i % 1000 == 0:
            print(" ", i, flush=True)
    ev = pd.DataFrame(ev, columns=["stay_id", "hour", "var", "val"])
    ev["hb"] = np.floor(ev["hour"]).astype(int)
    ev = ev.groupby(["stay_id", "hb", "var"], as_index=False)["val"].mean().rename(columns={"hb": "hour"})
    keep = np.ones(len(ev), bool)
    for v, (lo, hi) in RANGE.items():
        m = ev["var"] == v
        keep &= ~(m & ((ev.val < lo) | (ev.val > hi)))
    ev = ev[keep]

    co = pd.DataFrame(meta)
    for c in ["age", "sex", "weight", "height"]:
        if c not in co.columns:
            co[c] = np.nan
    oc = pd.read_csv(OUTCOMES)
    oc.columns = [c.strip() for c in oc.columns]
    co["stay_id"] = co["stay_id"].astype("int64")
    co = co.merge(oc, left_on="stay_id", right_on="RecordID", how="left")
    co["subject_id"] = co["stay_id"].astype(str)
    co["age"] = pd.to_numeric(co.get("age"), errors="coerce")
    co["sex"] = pd.to_numeric(co.get("sex"), errors="coerce")   # 0 female, 1 male per challenge
    co["los_icu_days"] = pd.to_numeric(co["Length_of_stay"], errors="coerce")
    co["long_icu"] = (co["los_icu_days"] > 3).astype(int)
    co["mort_hosp"] = pd.to_numeric(co["In-hospital_death"], errors="coerce")
    cr = ev[ev["var"] == "creatinine"].sort_values(["stay_id", "hour"])
    aki = {}
    for sid, g in cr.groupby("stay_id"):
        h = g["hour"].values; v = g["val"].values
        base = v[h < 24].min() if (h < 24).any() else v.min()
        flag = 0
        for i in range(len(v)):
            if h[i] < 24:
                continue
            if (v[i] - base >= 0.3 and h[i] - 24 <= 48) or v[i] >= 1.5 * base:
                flag = 1; break
        aki[sid] = flag
    co["aki"] = co["stay_id"].map(aki)
    co = co[["stay_id", "subject_id", "age", "sex", "weight", "height",
             "los_icu_days", "mort_hosp", "long_icu", "aki"]]
    co = co[co["age"].fillna(0) >= 18]

    ev.to_parquet(f"{OUT}/events_c2012.parquet", index=False)
    co.to_parquet(f"{OUT}/cohort_c2012.parquet", index=False)
    print("stays", len(co), "events", len(ev), "vars", ev["var"].nunique())
    print("mort", round(co.mort_hosp.mean(), 3), "long", round(co.long_icu.mean(), 3),
          "aki", round(co.aki.mean(), 3))
    print(ev[ev.hour < 24]["var"].value_counts().head(10).to_dict())


if __name__ == "__main__":
    main()
