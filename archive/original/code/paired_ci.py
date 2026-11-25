"""Paired bootstrap CIs for the key contrasts (replaces the fixed 0.026 gate).

Compares, on identical patients and identical probe protocol:
  main20k (full values)  vs  abl_const20k (constant values)
  main20k (all vars)     vs  common20k   (8 common vars)
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fm
from fm import load_model, SeqData, embed_all

ROOT = fm.ROOT
RES = f"{ROOT}/results/tables"
SEED = 20261102
rng = np.random.default_rng(SEED)


def preds(tag, db):
    model, dev, _ = load_model(tag, "auto")
    Em, dm = embed_all(model, "mimic", dev, tag=tag)
    ym = dm.y["y_mort"].astype(float)
    okE = np.isfinite(Em).all(1)
    tr = (dm.split == 0) & np.isfinite(ym) & okE
    sc = StandardScaler().fit(Em[tr])
    lr = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(Em[tr]), ym[tr])
    if db == "mimic":
        E, ds = Em, dm
    else:
        E, ds = embed_all(model, db, dev, tag=f"{tag}_{db}")
    y = ds.y["y_mort"].astype(float)
    ok = np.isfinite(y) & np.isfinite(E).all(1)
    if db == "mimic":
        ok &= (ds.split == 2)
    p = lr.predict_proba(sc.transform(E[ok]))[:, 1]
    return y[ok], p


def paired(a, b, db, B=1000):
    ya, pa = preds(a, db)
    yb, pb = preds(b, db)
    n = min(len(ya), len(yb))
    ya, pa, yb, pb = ya[:n], pa[:n], yb[:n], pb[:n]
    assert np.allclose(ya, yb), "label mismatch"
    d = roc_auc_score(ya, pa) - roc_auc_score(yb, pb)
    boot = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if len(np.unique(ya[i])) < 2:
            continue
        boot.append(roc_auc_score(ya[i], pa[i]) - roc_auc_score(yb[i], pb[i]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(db=db, n=n, auroc_a=roc_auc_score(ya, pa), auroc_b=roc_auc_score(yb, pb),
                delta=d, lo=float(lo), hi=float(hi))


rows = []
for a, b, label in [("main20k", "abl_const20k", "full vs constant-value"),
                    ("main20k", "common20k", "all vars vs 8 common vars")]:
    for db in ["eicu", "nwicu"]:
        r = paired(a, b, db)
        r["contrast"] = label
        rows.append(r)
        print(r, flush=True)

df = pd.DataFrame(rows)
df.to_csv(f"{RES}/paired_bootstrap.csv", index=False)
print(df.round(3).to_string(index=False))
