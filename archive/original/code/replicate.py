"""Seed replication of the database-count slice (K=1,2,3 at fixed 20k budget).

The subset of stays is held fixed across seeds (sample_seed), only model init
and batch order vary, so the spread measures optimisation noise, not sampling.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fm
from analysis import pretrain_multi

ROOT = fm.ROOT
RES = fm.RES
SEED = 20261102
DBS = {1: ["mimic"], 2: ["mimic", "eicu"], 3: ["mimic", "eicu", "nwicu"]}

rows = []
for k, dbs in DBS.items():
    for s in [1, 2, 3]:
        tag = f"rep_k{k}_s{s}"
        if not os.path.exists(f"{fm.MODELS}/{tag}.pt"):
            pretrain_multi(dbs, 20000, tag, epochs=2, seed=SEED + s, sample_seed=SEED)
        df, _ = fm.run_probe(tag, "mort", out=f"{RES}/probe_{tag}_mort.csv")
        d = df[df.db.isin(["mimic_test", "eicu_all", "nwicu_all"])].copy()
        d["k"] = k
        d["seed"] = SEED + s
        d["tag"] = tag
        rows.append(d)
        print(d[["tag", "db", "auroc", "slope", "bss"]].round(3).to_string(index=False), flush=True)

out = pd.concat(rows)
out.to_csv(f"{RES}/replication_k.csv", index=False)
print("\n=== replicate summary (mean +/- sd over seeds) ===")
agg = out.groupby(["k", "db"])["auroc"].agg(["mean", "std", "min", "max", "count"]).round(3)
print(agg.to_string())
agg.to_csv(f"{RES}/replication_summary.csv")
