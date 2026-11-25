"""Probe source dictionaries / label vocabularies (read-only, fast).

Usage: python probe_schemas.py [mimic|eicu|nwicu|all]
Writes logs/schema_probe.txt
"""
import sys, os, gzip, json
from collections import Counter
import pandas as pd

ROOT = "/data/admin/fengshuo/ehr_fm_20261102"
MIMIC = "/data/admin/fengshuo/if10_20261002/shared/raw/mimic"
MIMIC2 = "/data/admin/fengshuo/mi_confounding_20261101/data_raw/mimic"
EICU = "/data/admin/fengshuo/mi_confounding_20261101/data_raw/eicu"
EICU10 = "/data/admin/fengshuo/if10_20261002/shared/raw/eicu"
NWICU = ("/data/admin/fengshuo/mi_confounding_20261101/data_raw/nwicu/"
         "northwestern-icu-nwicu-database-0.1.0/data")
OUT = os.path.join(ROOT, "logs", "schema_probe.txt")


def top_labnames():
    c = Counter()
    for chunk in pd.read_csv(f"{EICU10}/lab.csv.gz", usecols=["labname"],
                             chunksize=4_000_000, compression="gzip"):
        c.update(chunk["labname"].astype(str).str.lower().str.strip())
    return c


def mimic_dicts():
    di = pd.read_csv(f"{MIMIC}/icu/d_items.csv.gz")
    dl = pd.read_csv(f"{MIMIC}/hosp/d_labitems.csv.gz")
    di["lbl"] = di["label"].astype(str).str.lower()
    dl["lbl"] = dl["label"].astype(str).str.lower()
    return di, dl


def eicu_dicts():
    rc = pd.read_csv(f"{EICU10}/respiratoryCharting.csv.gz",
                     usecols=["respchartvaluelabel"], nrows=2_000_000)
    rc2 = pd.read_csv(f"{EICU10}/respiratoryCare.csv.gz", nrows=200_000)
    inf = pd.read_csv(f"{EICU10}/infusionDrug.csv.gz", usecols=["drugname"],
                      nrows=3_000_000)
    return rc, rc2, inf


def nwicu_dicts():
    di = pd.read_csv(f"{NWICU}/nw_icu/d_items.csv.gz")
    return di


def main(which="all"):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        if which in ("all", "eicu"):
            c = top_labnames()
            fh.write("=== eICU labname top 120 ===\n")
            for k, v in c.most_common(120):
                fh.write(f"{v:>10d}  {k}\n")
            rc, rc2, inf = eicu_dicts()
            fh.write("\n=== eICU respchartvaluelabel top 80 ===\n")
            for k, v in Counter(rc["respchartvaluelabel"].astype(str).str.lower()).most_common(80):
                fh.write(f"{v:>8d}  {k}\n")
            fh.write("\n=== eICU respiratoryCare cols ===\n")
            fh.write(",".join(rc2.columns) + "\n")
            fh.write("\n=== eICU drugname top 60 ===\n")
            for k, v in Counter(inf["drugname"].astype(str).str.lower()).most_common(60):
                fh.write(f"{v:>8d}  {k}\n")
        if which in ("all", "nwicu"):
            di = nwicu_dicts()
            fh.write("\n=== NWICU d_items (chartevents, numeric) top 120 ===\n")
            sub = di[di["linksto"] == "chartevents"]
            for _, r in sub.head(120).iterrows():
                fh.write(f"{r['itemid']}\t{r['label']}\t{r['abbreviation']}\t{r.get('unitname')}\n")
        if which in ("all", "mimic"):
            di, dl = mimic_dicts()
            fh.write("\n=== MIMIC d_items selected ===\n")
            pat = "heart rate|blood pressure|respir|temperature|o2 satur|gcs|glasgow|fio2|peep|tidal|pressure"
            for _, r in di[di["lbl"].str.contains(pat, na=False)].head(120).iterrows():
                fh.write(f"{r['itemid']}\t{r['label']}\t{r['linksto']}\t{r.get('unitname')}\n")
            fh.write("\n=== MIMIC d_labitems selected ===\n")
            pat2 = ("creatinine|urea nitrogen|lactate|white blood|hemoglobin|platelet|sodium|potassium|"
                    "chloride|calcium|magnesium|phosphate|glucose|albumin|bilirubin|aspartate|alanine|"
                    "inr|ptt|ph|pco2|po2|base excess|c reactive")
            for _, r in dl[dl["lbl"].str.contains(pat2, na=False)].head(120).iterrows():
                fh.write(f"{r['itemid']}\t{r['label']}\t{r.get('fluid')}\t{r.get('category')}\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
