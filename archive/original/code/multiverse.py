"""Upgrade 3: multiverse sensitivity of the main transfer result (mortality).

Each configuration rebuilds the tokenised sequences (when the tokenisation
changes), pretrains a 0.41 M encoder with the same budget, and probes the same
six databases.  The point is to show the calibration/transfer conclusion is not
an artefact of one arbitrary tokenisation choice.
"""
import glob
import os

import pandas as pd

ROOT = "/data/admin/fengshuo/ehr_fm_20261102"
RES = f"{ROOT}/results/tables"

CFG = {
    "base": "NBINS=10, MAXLEN=128, mask=0.20 (main20k)",
    "b5": "NBINS=5", "b20": "NBINS=20",
    "l32": "MAXLEN=32", "l64": "MAXLEN=64", "l256": "MAXLEN=256",
    "m015": "mask=0.15", "m030": "mask=0.30",
}

rows = []
def add(tag, label):
    f = f"{RES}/probe_{tag}_mort.csv"
    if not os.path.exists(f):
        return
    d = pd.read_csv(f)
    for _, r in d.iterrows():
        if r["db"] not in ("mimic_test", "eicu_all", "nwicu_all"):
            continue
        rows.append(dict(config=tag, label=label, db=r["db"], auroc=r["auroc"],
                         slope=r["slope"], bss=r["bss"], rrr=r.get("rrr_80")))

add("main20k", CFG["base"])
for k in ["b5", "b20", "l32", "l64", "l256", "m015", "m030"]:
    add(f"mv_{k}", CFG[k])

if rows:
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/multiverse.csv", index=False)
    print(df.pivot_table(index=["config", "label"], columns="db", values=["auroc", "slope"]).round(3).to_string())
else:
    print("no multiverse results yet")
