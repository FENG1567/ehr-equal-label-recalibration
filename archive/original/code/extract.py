"""Extract 0-7 day event streams for the 48-variable canonical model.

Outputs (per database) to data/interim/:
  events_<db>.parquet   : stay_id, hour (float, hours since ICU index), var, val
  cohort_<db>.parquet   : stay_id, subject_id, age, sex, los_icu_days,
                          mort_hosp, aki, long_icu, n_events, has_24h
All timestamps are relative to the ICU index time (MIMIC/NWICU intime,
eICU unitadmittime).  Features use hour < 24; AKI uses the full 0-168h window.
"""
import os, sys, json, re
import numpy as np
import pandas as pd
import polars as pl

ROOT = "/data/admin/fengshuo/ehr_fm_20261102"
MIMIC = "/data/admin/fengshuo/if10_20261002/shared/raw/mimic"
EICU10 = "/data/admin/fengshuo/if10_20261002/shared/raw/eicu"
EICU = "/data/admin/fengshuo/mi_confounding_20261101/data_raw/eicu"
NWICU = ("/data/admin/fengshuo/mi_confounding_20261101/data_raw/nwicu/"
         "northwestern-icu-nwicu-database-0.1.0/data")
INT = os.path.join(ROOT, "data", "interim")
os.makedirs(INT, exist_ok=True)

# ----------------------------------------------------------------------------
# canonical variables (48)
# ----------------------------------------------------------------------------
VARS = [
    # demographics / static
    "age", "sex", "weight", "height", "bmi",
    # vitals
    "hr", "sbp", "dbp", "map", "rr", "temp", "spo2", "etco2",
    # neuro
    "gcs_total", "gcs_motor", "gcs_eye",
    # respiratory
    "fio2", "peep", "tv", "ppeak", "pao2", "paco2", "ph", "be",
    # renal / metabolic
    "creatinine", "bun", "urine", "lactate",
    # haematology
    "wbc", "hgb", "platelet", "inr", "aptt",
    # chemistry
    "sodium", "potassium", "chloride", "calcium", "magnesium",
    "phosphate", "glucose",
    # liver / inflammation
    "albumin", "bili", "ast", "alt", "crp",
    # treatment
    "vaso", "rrt", "sedation",
]
VAR2ID = {v: i for i, v in enumerate(VARS)}

# plausible physiological ranges; values outside are dropped (units documented in mapping_table.csv)
RANGE = {
    "age": (18, 110), "sex": (0, 1), "weight": (25, 300), "height": (120, 230),
    "bmi": (10, 90), "hr": (20, 250), "sbp": (30, 300), "dbp": (10, 200),
    "map": (20, 220), "rr": (2, 80), "temp": (28, 43), "spo2": (40, 100),
    "etco2": (5, 100), "gcs_total": (3, 15), "gcs_motor": (1, 6), "gcs_eye": (1, 4),
    "fio2": (21, 100), "peep": (0, 30), "tv": (100, 1500), "ppeak": (5, 70),
    "pao2": (20, 700), "paco2": (10, 150), "ph": (6.5, 7.9), "be": (-40, 40),
    "creatinine": (0.1, 25), "bun": (1, 300), "urine": (0, 5000), "lactate": (0.1, 30),
    "wbc": (0.05, 200), "hgb": (1.5, 25), "platelet": (1, 1500), "inr": (0.5, 20),
    "aptt": (10, 200), "sodium": (90, 190), "potassium": (1, 10), "chloride": (50, 140),
    "calcium": (3, 20), "magnesium": (0.3, 6), "phosphate": (0.2, 20),
    "glucose": (10, 900), "albumin": (0.8, 7), "bili": (0.1, 60), "ast": (1, 20000),
    "alt": (1, 20000), "crp": (0.1, 500), "vaso": (0, 1), "rrt": (0, 1),
    "sedation": (0, 1),
}

MIMIC_CHART = {
    "hr": [(220045, 1)], "sbp": [(220050, 1), (220179, 1)],
    "dbp": [(220051, 1), (220180, 1)],
    "map": [(220052, 1), (220181, 1), (225312, 1)],
    "rr": [(220210, 1), (224690, 1)],
    "temp": [(223762, 1), (223761, "F2C")],
    "spo2": [(220277, 1)], "etco2": [(228640, 1)],
    "gcs_eye": [(220739, 1)], "gcs_motor": [(223901, 1)], "gcs_verbal": [(223900, 1)],
    "fio2": [(223835, 1)], "peep": [(220339, 1)], "tv": [(224685, 1)],
    "ppeak": [(224695, 1)], "weight": [(226512, 1), (224639, 1)],
    "height": [(226730, 1)],
}
MIMIC_LAB = {
    "creatinine": 50912, "bun": 51006, "lactate": 50813, "wbc": 51301,
    "hgb": 51222, "platelet": 51265, "sodium": 50983, "potassium": 50971,
    "chloride": 50902, "calcium": 50893, "magnesium": 50960, "phosphate": 50970,
    "glucose": 50931, "albumin": 50862, "bili": 50885, "ast": 50878,
    "alt": 50861, "inr": 51237, "aptt": 51275, "ph": 50820, "paco2": 50818,
    "pao2": 50821, "be": 50802, "crp": 50889,
}
MIMIC_VASO = [221906, 221289, 221662, 222315, 221749]
MIMIC_SED = [222168, 221668, 225942]
MIMIC_RRT = [225441, 225802, 225803]

EICU_LAB = {
    "creatinine": "creatinine", "bun": "bun", "lactate": "lactate",
    "wbc": "wbc x 1000", "hgb": "hgb", "platelet": "platelets x 1000",
    "sodium": "sodium", "potassium": "potassium", "chloride": "chloride",
    "calcium": "calcium", "magnesium": "magnesium", "phosphate": "phosphate",
    "glucose": "glucose", "albumin": "albumin", "bili": "total bilirubin",
    "ast": "ast (sgot)", "alt": "alt (sgpt)", "inr": "pt - inr", "aptt": "ptt",
    "ph": "ph", "paco2": "paco2", "pao2": "pao2", "be": "base excess",
    "crp": "crp", "temp": "temperature", "fio2": "fio2", "peep": "peep",
    "tv": "tv", "rr": "respiratory rate", "spo2": "o2 sat (%)",
}
EICU_VITAL = {"hr": "heartrate", "sbp": "systemicsystolic", "dbp": "systemicdiastolic",
              "map": "systemicmean", "rr": "respiration", "temp": "temperature",
              "spo2": "sao2", "etco2": "etco2"}
EICU_VASO = r"norepinephrine|dopamine|vasopressin|phenylephrine|epinephrine|dobutamine"
EICU_SED = r"propofol|midazolam|fentanyl|dexmedetomidine"

NWICU_CHART = {"hr": [320045], "sbp": [320050, 320179], "dbp": [320051, 320180],
               "rr": [320210], "spo2": [320277], "temp": [323761]}
NWICU_LABNAME = {
    "creatinine": r"creatinine", "bun": r"urea nitrogen|bun", "lactate": r"lactate",
    "wbc": r"white blood", "hgb": r"hemoglobin", "platelet": r"platelet",
    "sodium": r"sodium", "potassium": r"potassium", "chloride": r"chloride",
    "calcium": r"calcium", "magnesium": r"magnesium", "phosphate": r"phosphate",
    "glucose": r"glucose", "albumin": r"albumin", "bili": r"bilirubin, total",
    "ast": r"aspartate", "alt": r"alanine", "inr": r"inr", "aptt": r"ptt",
    "ph": r"^ph$", "paco2": r"pco2", "pao2": r"po2", "crp": r"c-reactive",
}

LOG = []


def log(msg):
    print(msg, flush=True)
    LOG.append(msg)


def clean(df, var):
    lo, hi = RANGE.get(var, (-1e9, 1e9))
    n0 = len(df)
    df = df[(df["val"] >= lo) & (df["val"] <= hi)]
    return df, n0 - len(df)


def cached_scan(path, ids, cols, schema, out):
    """Filter a large csv.gz once, cache the filtered frame as parquet."""
    if os.path.exists(out):
        print(f"  cache hit {out}", flush=True)
        return pl.read_parquet(out)
    df = (pl.scan_csv(path, schema_overrides=schema)
          .filter(pl.col("itemid").is_in(ids))
          .select(cols).collect())
    df.write_parquet(out)
    print(f"  cached {out} rows={df.height:,}", flush=True)
    return df

def to_events(pairs, stay_ids_ok=None):
    """pairs: list of (stay_id, hour, var, val)."""
    df = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame(columns=["stay_id", "hour", "var", "val"])
    df = df[df["hour"] >= 0]
    df = df[df["stay_id"].isin(stay_ids_ok)] if stay_ids_ok is not None else df
    out = []
    for v, g in df.groupby("var", sort=False):
        g, drop = clean(g, v)
        if drop:
            log(f"  range-drop {v}: {drop}")
        out.append(g)
    return pd.concat(out) if out else df.iloc[0:0]


# ----------------------------------------------------------------------------
# MIMIC-IV
# ----------------------------------------------------------------------------
def extract_mimic():
    log("== MIMIC ==")
    icu = pd.read_csv(f"{MIMIC}/icu/icustays.csv.gz",
                      usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"])
    pat = pd.read_csv(f"{MIMIC}/hosp/patients.csv.gz",
                      usecols=["subject_id", "gender", "anchor_age"])
    adm = pd.read_csv(f"{MIMIC}/hosp/admissions.csv.gz",
                      usecols=["hadm_id", "hospital_expire_flag", "admittime", "dischtime"])
    icu["intime"] = pd.to_datetime(icu["intime"])
    icu = icu.merge(pat, on="subject_id", how="left").merge(adm, on="hadm_id", how="left")
    icu = icu[icu["anchor_age"] >= 18].copy()
    icu["age"] = icu["anchor_age"].clip(upper=91)
    icu["sex"] = (icu["gender"] == "M").astype(int)
    icu["mort_hosp"] = (icu["discharge_location"].astype(str).str.upper() == "EXPIRED").astype(int)
    icu["los_icu_days"] = icu["los"].astype(float)
    icu["long_icu"] = (icu["los_icu_days"] > 3).astype(int)
    stay_ok = set(icu["stay_id"].astype("int64"))
    intime = dict(zip(icu["stay_id"].astype("int64"), icu["intime"]))
    hadm2stay = dict(zip(icu["hadm_id"].astype("int64"), icu["stay_id"].astype("int64")))

    item2var = {}
    for v, lst in MIMIC_CHART.items():
        for iid, tr in lst:
            item2var[iid] = (v, tr)
    ids = list(item2var)
    log(f"  chartevents filter {len(ids)} itemids")
    ce = cached_scan(f"{MIMIC}/icu/chartevents.csv.gz", ids,
                     ["stay_id", "charttime", "itemid", "valuenum"],
                     {"stay_id": pl.Int64, "itemid": pl.Int64, "valuenum": pl.Float64},
                     f"{INT}/_ce_mimic.parquet")
    log(f"  chartevents rows kept {ce.height:,}")
    ce = ce.to_pandas()
    ce["charttime"] = pd.to_datetime(ce["charttime"])
    ce["hour"] = (ce["charttime"] - ce["stay_id"].map(intime)).dt.total_seconds() / 3600.0
    ce = ce[(ce["hour"] >= 0) & (ce["hour"] <= 168)]
    pairs = []
    for iid, (v, tr) in item2var.items():
        g = ce[ce["itemid"] == iid]
        if not len(g):
            continue
        val = g["valuenum"].astype(float)
        if tr == "F2C":
            val = (val - 32) * 5 / 9
        pairs.append(pd.DataFrame({"stay_id": g["stay_id"].values, "hour": g["hour"].values,
                                   "var": v, "val": val.values}))
    # GCS total from components charted at the same time
    gcs = ce[ce["itemid"].isin([220739, 223900, 223901])]
    if len(gcs):
        piv = gcs.pivot_table(index=["stay_id", "charttime"], columns="itemid",
                              values="valuenum", aggfunc="min")
        need = [c for c in [220739, 223900, 223901] if c in piv.columns]
        if len(need) == 3:
            s = piv[need].sum(axis=1)
            tot = s.reset_index()
            tot.columns = ["stay_id", "charttime", "gcs"]
            tot["hour"] = (tot["charttime"] - tot["stay_id"].map(intime)).dt.total_seconds() / 3600.0
            tot = tot[(tot["hour"] >= 0) & (tot["hour"] <= 168)]
            pairs.append(pd.DataFrame({"stay_id": tot["stay_id"].values, "hour": tot["hour"].values,
                                       "var": "gcs_total", "val": tot["gcs"].values}))

    # labs (linked through hadm_id)
    lab_ids = list(MIMIC_LAB.values())
    log(f"  labevents filter {len(lab_ids)} itemids")
    lb = cached_scan(f"{MIMIC}/hosp/labevents.csv.gz", lab_ids,
                     ["hadm_id", "charttime", "itemid", "valuenum"],
                     {"hadm_id": pl.Int64, "itemid": pl.Int64, "valuenum": pl.Float64},
                     f"{INT}/_lb_mimic.parquet").to_pandas()
    log(f"  labevents rows kept {len(lb):,}")
    lb["charttime"] = pd.to_datetime(lb["charttime"])
    lb["stay_id"] = lb["hadm_id"].map(hadm2stay)
    lb = lb[lb["stay_id"].notna()]
    lb["hour"] = (lb["charttime"] - lb["stay_id"].map(intime)).dt.total_seconds() / 3600.0
    lb = lb[(lb["hour"] >= 0) & (lb["hour"] <= 168)]
    lb["stay_id"] = lb["stay_id"].astype("int64")
    for v, iid in MIMIC_LAB.items():
        g = lb[lb["itemid"] == iid]
        if len(g):
            pairs.append(pd.DataFrame({"stay_id": g["stay_id"].values, "hour": g["hour"].values,
                                       "var": v, "val": g["valuenum"].astype(float).values}))
    # urine output
    uo = (pl.scan_csv(f"{MIMIC}/icu/outputevents.csv.gz",
                      schema_overrides={"stay_id": pl.Int64, "itemid": pl.Int64,
                                        "value": pl.Float64})
          .filter(pl.col("itemid").is_in([226559, 226560, 226561, 226584, 226563,
                                          226564, 226565, 226567, 226557, 226558,
                                          227488, 227489]))
          .select(["stay_id", "charttime", "value"]).collect(streaming=True)).to_pandas()
    uo["charttime"] = pd.to_datetime(uo["charttime"])
    uo["hour"] = (uo["charttime"] - uo["stay_id"].map(intime)).dt.total_seconds() / 3600.0
    uo = uo[(uo["hour"] >= 0) & (uo["hour"] <= 168)]
    pairs.append(pd.DataFrame({"stay_id": uo["stay_id"].values, "hour": uo["hour"].values,
                               "var": "urine", "val": uo["value"].astype(float).values}))

    # vasopressors / sedation / RRT (binary per 4h block to keep sequence compact)
    for path, ids_, var, tbl in [
            (f"{MIMIC}/icu/inputevents.csv.gz", MIMIC_VASO, "vaso", "inputevents"),
            (f"{MIMIC}/icu/inputevents.csv.gz", MIMIC_SED, "sedation", "inputevents"),
            (f"{MIMIC}/icu/procedureevents.csv.gz", MIMIC_RRT, "rrt", "procedureevents")]:
        tcol = "starttime"
        try:
            d = (pl.scan_csv(path, schema_overrides={"stay_id": pl.Int64, "itemid": pl.Int64})
                 .filter(pl.col("itemid").is_in(ids_))
                 .select(["stay_id", tcol]).collect(streaming=True)).to_pandas()
        except Exception as e:
            log(f"  !! {tbl} {var} failed: {e}")
            continue
        if not len(d):
            continue
        d[tcol] = pd.to_datetime(d[tcol])
        d["hour"] = ((d[tcol] - d["stay_id"].map(intime)).dt.total_seconds() / 3600.0).clip(lower=0)
        d = d[d["hour"] <= 168]
        d["hour4"] = (d["hour"] // 4) * 4.0
        agg = d.drop_duplicates(["stay_id", "hour4"])
        pairs.append(pd.DataFrame({"stay_id": agg["stay_id"].values, "hour": agg["hour4"].values,
                                   "var": var, "val": 1.0}))

    ev = to_events(pairs, stay_ok)
    cohort = icu[["stay_id", "subject_id", "age", "sex", "los_icu_days", "mort_hosp", "long_icu"]].copy()
    cohort["stay_id"] = cohort["stay_id"].astype("int64")
    keep = event_keep(ev)
    cohort = cohort[cohort["stay_id"].isin(keep)].copy()
    # AKI (KDIGO creatinine, 7-day window)
    cohort["aki"] = aki_label(ev, cohort["stay_id"])
    finish(ev, cohort, "mimic")


def event_keep(ev, min_events=8):
    n = ev[ev["hour"] < 24].groupby("stay_id").size()
    return set(n[n >= min_events].index)


def aki_label(ev, stays):
    """KDIGO creatinine: >=0.3 mg/dL rise within 48h or >=1.5x baseline within 7d."""
    cr = ev[ev["var"] == "creatinine"].sort_values(["stay_id", "hour"])
    out = {}
    for sid, g in cr.groupby("stay_id"):
        h = g["hour"].values
        v = g["val"].values
        base = v[h < 24].min() if (h < 24).any() else v.min()
        if not np.isfinite(base) or base <= 0:
            out[sid] = np.nan
            continue
        lab = 0
        for i in range(len(v)):
            if h[i] < 24:
                continue
            delta = v[i] - base
            if delta >= 0.3 and (h[i] - 24) <= 48:
                lab = 1
                break
            if v[i] >= 1.5 * base:
                lab = 1
                break
        out[sid] = lab
    return pd.Series(stays.map(out))


# ----------------------------------------------------------------------------
# eICU
# ----------------------------------------------------------------------------
def extract_eicu():
    log("== eICU ==")
    pat = pd.read_csv(f"{EICU10}/patient.csv.gz")
    pat["age"] = pd.to_numeric(pat["age"].astype(str).str.replace("> 89", "90", regex=False), errors="coerce")
    pat = pat[pat["age"] >= 18].copy()
    pat["mort_hosp"] = (pat["hospitaldischargestatus"].astype(str).str.lower() == "expired").astype(int)
    pat["los_icu_days"] = pat["unitdischargeoffset"].astype(float) / 1440.0
    pat["long_icu"] = (pat["los_icu_days"] > 3).astype(int)
    pat["sex"] = (pat["gender"] == "Male").astype(int)
    pat["stay_id"] = pat["patientunitstayid"].astype("int64")
    log(f"  patients {len(pat):,}")
    stay_ok = set(pat["stay_id"])
    pairs = []

    # labs
    labmap = {k.lower(): v for k, v in EICU_LAB.items()}
    lb = (pl.scan_csv(f"{EICU10}/lab.csv.gz",
                      schema_overrides={"patientunitstayid": pl.Int64, "labresult": pl.Float64})
          .select(["patientunitstayid", "labresultoffset", "labname", "labresult"])
          .collect(streaming=True)).to_pandas()
    lb["labname"] = lb["labname"].astype(str).str.lower().str.strip()
    lb = lb[lb["labname"].isin(labmap)]
    lb["var"] = lb["labname"].map(labmap)
    lb["stay_id"] = lb["patientunitstayid"].astype("int64")
    lb["hour"] = lb["labresultoffset"].astype(float) / 60.0
    lb = lb[(lb["hour"] >= 0) & (lb["hour"] <= 168)]
    for v, g in lb.groupby("var"):
        pairs.append(pd.DataFrame({"stay_id": g["stay_id"].values, "hour": g["hour"].values,
                                   "var": v, "val": g["labresult"].astype(float).values}))
    log(f"  lab rows kept {len(lb):,}")

    # vitalPeriodic (truncated gzip: read via tolerant decompression)
    slim = f"{ROOT}/data/raw/eicu_vitalperiodic_hourly.csv.gz"
    vp_csv = os.path.join(INT, "eicu_vitalperiodic.partial.csv")
    if os.path.exists(slim):
        vp = pd.read_csv(slim)
        log(f"  using hourly slim vitals rows={len(vp):,}")
        vp["stay_id"] = vp["patientunitstayid"].astype("int64")
        vp["hour"] = vp["hour"].astype(float)
    else:
        if not os.path.exists(vp_csv):
            os.system(f"zcat {EICU}/vitalPeriodic.csv.gz 2>/dev/null > {vp_csv}")
        try:
            vp = pd.read_csv(vp_csv)
        except Exception as e:
            log(f"  !! vitalPeriodic read failed {e}")
            vp = pd.DataFrame()
        if len(vp):
            vp["stay_id"] = vp["patientunitstayid"].astype("int64")
            vp["hour"] = vp["observationoffset"].astype(float) / 60.0
    if len(vp):
        vp = vp[(vp["hour"] >= 0) & (vp["hour"] <= 168) & vp["stay_id"].isin(stay_ok)]
        for v, col in EICU_VITAL.items():
            if col not in vp.columns:
                continue
            g = vp[["stay_id", "hour", col]].dropna()
            if len(g):
                pairs.append(pd.DataFrame({"stay_id": g["stay_id"].values, "hour": g["hour"].values,
                                           "var": v, "val": g[col].astype(float).values}))
        log(f"  vitalPeriodic rows {len(vp):,}")

    # vasopressors / sedation from infusionDrug
    inf = (pl.scan_csv(f"{EICU10}/infusionDrug.csv.gz",
                       schema_overrides={"patientunitstayid": pl.Int64})
           .select(["patientunitstayid", "infusionoffset", "drugname"])
           .collect(streaming=True)).to_pandas()
    inf["drug"] = inf["drugname"].astype(str).str.lower()
    inf["stay_id"] = inf["patientunitstayid"].astype("int64")
    inf["hour"] = inf["infusionoffset"].astype(float) / 60.0
    inf = inf[(inf["hour"] >= 0) & (inf["hour"] <= 168)]
    for var, pat_ in [("vaso", EICU_VASO), ("sedation", EICU_SED)]:
        g = inf[inf["drug"].str.contains(pat_, regex=True, na=False)]
        if len(g):
            g = g.drop_duplicates(["stay_id", "hour"])
            pairs.append(pd.DataFrame({"stay_id": g["stay_id"].values, "hour": g["hour"].values,
                                       "var": var, "val": 1.0}))

    # urine from intakeOutput
    io = (pl.scan_csv(f"{EICU10}/intakeOutput.csv.gz",
                      schema_overrides={"patientunitstayid": pl.Int64, "cellvaluenumeric": pl.Float64})
          .filter(pl.col("celllabel").str.contains("(?i)urine"))
          .select(["patientunitstayid", "intakeoutputoffset", "cellvaluenumeric"])
          .collect(streaming=True)).to_pandas()
    if len(io):
        io["stay_id"] = io["patientunitstayid"].astype("int64")
        io["hour"] = io["intakeoutputoffset"].astype(float) / 60.0
        io = io[(io["hour"] >= 0) & (io["hour"] <= 168)].dropna(subset=["cellvaluenumeric"])
        pairs.append(pd.DataFrame({"stay_id": io["stay_id"].values, "hour": io["hour"].values,
                                   "var": "urine", "val": io["cellvaluenumeric"].astype(float).values}))
    log(f"  io urine rows {len(io):,}")

    ev = to_events(pairs, stay_ok)
    cohort = pat[["stay_id", "uniquepid", "age", "sex", "los_icu_days", "mort_hosp", "long_icu"]].copy()
    cohort = cohort.rename(columns={"uniquepid": "subject_id"})
    keep = event_keep(ev)
    cohort = cohort[cohort["stay_id"].isin(keep)].copy()
    cohort["aki"] = aki_label(ev, cohort["stay_id"])
    finish(ev, cohort, "eicu")


# ----------------------------------------------------------------------------
# NWICU
# ----------------------------------------------------------------------------
def extract_nwicu():
    log("== NWICU ==")
    icu = pd.read_csv(f"{NWICU}/nw_icu/icustays.csv.gz")
    pat = pd.read_csv(f"{NWICU}/nw_hosp/patients.csv.gz")
    adm = pd.read_csv(f"{NWICU}/nw_hosp/admissions.csv.gz",
                      usecols=["hadm_id", "hospital_expire_flag", "discharge_location"])
    icu["intime"] = pd.to_datetime(icu["intime"])
    icu = icu.merge(pat, on="subject_id", how="left").merge(adm, on="hadm_id", how="left")
    icu = icu[icu["anchor_age"] >= 18].copy()
    icu["age"] = icu["anchor_age"].clip(upper=91)
    icu["sex"] = (icu["gender"] == "M").astype(int)
    icu["mort_hosp"] = (icu["discharge_location"].astype(str).str.upper() == "EXPIRED").astype(int)
    icu["los_icu_days"] = icu["los"].astype(float)
    icu["long_icu"] = (icu["los_icu_days"] > 3).astype(int)
    stay_ok = set(icu["stay_id"].astype("int64"))
    intime = dict(zip(icu["stay_id"].astype("int64"), icu["intime"]))
    log(f"  icustays {len(icu):,}")

    item2var = {}
    for v, ids in NWICU_CHART.items():
        for i in ids:
            item2var[i] = v
    ce = (pl.scan_csv(f"{NWICU}/nw_icu/chartevents.csv.gz",
                      schema_overrides={"stay_id": pl.Int64, "itemid": pl.Int64,
                                        "valuenum": pl.Float64})
          .filter(pl.col("itemid").is_in(list(item2var)))
          .select(["stay_id", "charttime", "itemid", "valuenum"])
          .collect(streaming=True)).to_pandas()
    ce["charttime"] = pd.to_datetime(ce["charttime"])
    ce["hour"] = (ce["charttime"] - ce["stay_id"].map(intime)).dt.total_seconds() / 3600.0
    ce = ce[(ce["hour"] >= 0) & (ce["hour"] <= 168)]
    ce["var"] = ce["itemid"].map(item2var)
    ce["val"] = ce["valuenum"].astype(float)
    ce.loc[ce["var"] == "temp", "val"] = (ce.loc[ce["var"] == "temp", "val"] - 32) * 5 / 9
    pairs = [ce[["stay_id", "hour", "var", "val"]]]

    labdict = pd.read_csv(f"{NWICU}/nw_hosp/d_labitems.csv.gz")
    labdict["lbl"] = labdict["label"].astype(str).str.lower()
    labmap = {}
    for v, pat_ in NWICU_LABNAME.items():
        m = labdict[labdict["lbl"].str.contains(pat_, regex=True, na=False)]
        if len(m):
            labmap.update({int(i): v for i in m["itemid"]})
    log(f"  lab itemids mapped {len(labmap)} -> {sorted(set(labmap.values()))}")
    lb = (pl.scan_csv(f"{NWICU}/nw_hosp/labevents.csv.gz",
                      schema_overrides={"hadm_id": pl.Int64, "itemid": pl.Int64,
                                        "valuenum": pl.Float64})
          .filter(pl.col("itemid").is_in(list(labmap)))
          .select(["hadm_id", "charttime", "itemid", "valuenum"])
          .collect(streaming=True)).to_pandas()
    if len(lb):
        hadm2stay = dict(zip(icu["hadm_id"].astype("int64"), icu["stay_id"].astype("int64")))
        lb["charttime"] = pd.to_datetime(lb["charttime"])
        lb["stay_id"] = lb["hadm_id"].map(hadm2stay)
        lb = lb[lb["stay_id"].notna()]
        lb["stay_id"] = lb["stay_id"].astype("int64")
        lb["hour"] = (lb["charttime"] - lb["stay_id"].map(intime)).dt.total_seconds() / 3600.0
        lb = lb[(lb["hour"] >= 0) & (lb["hour"] <= 168)]
        lb["var"] = lb["itemid"].map(labmap)
        lb["val"] = lb["valuenum"].astype(float)
        pairs.append(lb[["stay_id", "hour", "var", "val"]])

    ev = to_events(pairs, stay_ok)
    cohort = icu[["stay_id", "subject_id", "age", "sex", "los_icu_days", "mort_hosp", "long_icu"]].copy()
    cohort["stay_id"] = cohort["stay_id"].astype("int64")
    keep = event_keep(ev)
    cohort = cohort[cohort["stay_id"].isin(keep)].copy()
    cohort["aki"] = aki_label(ev, cohort["stay_id"])
    finish(ev, cohort, "nwicu")


def finish(ev, cohort, db):
    ev = ev.sort_values(["stay_id", "hour"])
    ev["var"] = ev["var"].astype(str)
    cohort["db"] = db
    ev[["stay_id", "hour", "var", "val"]].to_parquet(f"{INT}/events_{db}.parquet", index=False)
    cohort.to_parquet(f"{INT}/cohort_{db}.parquet", index=False)
    stats = ev[ev["hour"] < 24].groupby("var").size().sort_values(ascending=False)
    log(f"  stays={len(cohort):,} events_24h={int((ev['hour'] < 24).sum()):,} vars={len(stats)}")
    log(f"  mortality={cohort['mort_hosp'].mean():.3f} long_icu={cohort['long_icu'].mean():.3f} "
        f"aki={cohort['aki'].mean():.3f} (n_aki={int(cohort['aki'].notna().sum())})")
    json.dump({k: int(v) for k, v in stats.items()},
              open(f"{INT}/varcount_{db}.json", "w"), indent=1)
    with open(f"{ROOT}/logs/extract_{db}.log", "w") as fh:
        fh.write("\n".join(LOG))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "mimic"):
        extract_mimic()
    if which in ("all", "eicu"):
        extract_eicu()
    if which in ("all", "nwicu"):
        extract_nwicu()
