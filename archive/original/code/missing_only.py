"""How much signal is in the OBSERVATION PATTERN alone?

Logistic regression on 48 binary indicators (variable measured in 0-24h or not),
no values at all.  This is the floor that any value-based model must beat.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = "/data/admin/fengshuo/ehr_fm_20261102"
PROC = f"{ROOT}/data/processed"
RES = f"{ROOT}/results/tables"

rows = []
tab = {d: pd.read_parquet(f"{PROC}/tab_{d}.parquet") for d in ["mimic", "eicu", "nwicu"]}
feats = [c for c in tab["mimic"].columns if c != "stay_id"]
Xm = np.isnan(tab["mimic"][feats].values).astype(float)
z = np.load(f"{PROC}/seq_mimic.npz", allow_pickle=True)
split = np.load(f"{PROC}/split_mimic.npy")
y = z["y_mort"].astype(float)
tr = (split == 0) & np.isfinite(y)
sc = StandardScaler().fit(Xm[tr])
lr = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(Xm[tr]), y[tr])

for db in ["mimic", "eicu", "nwicu"]:
    if db == "mimic":
        X = Xm; yy = z["y_mort"].astype(float); mask = np.isfinite(yy)
        for nm, m in [("mimic_test", mask & (split == 2)), ("mimic_val", mask & (split == 1))]:
            p = lr.predict_proba(sc.transform(X[m]))[:, 1]
            rows.append(dict(db=nm, n=int(m.sum()), prev=float(yy[m].mean()),
                             auroc=roc_auc_score(yy[m], p), brier=brier_score_loss(yy[m], p)))
    else:
        Xt = np.isnan(tab[db][feats].values).astype(float)
        zz = np.load(f"{PROC}/seq_{db}.npz", allow_pickle=True)
        yt = zz["y_mort"].astype(float); ok = np.isfinite(yt)
        p = lr.predict_proba(sc.transform(Xt[ok]))[:, 1]
        rows.append(dict(db=db, n=int(ok.sum()), prev=float(yt[ok].mean()),
                         auroc=roc_auc_score(yt[ok], p), brier=brier_score_loss(yt[ok], p)))

df = pd.DataFrame(rows)
df.to_csv(f"{RES}/missingness_only.csv", index=False)
print(df.round(3).to_string(index=False))
