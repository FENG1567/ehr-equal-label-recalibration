"""Tabular LR predictions + a value-shuffled tabular control.

The shuffled control permutes each variable's values across patients
independently (marginal preserved, patient linkage destroyed); if the real
model does not beat it, the model is not using patient-specific values.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = "/data/admin/fengshuo/ehr_fm_20261102"
PROC = f"{ROOT}/data/processed"
RES = f"{ROOT}/results/tables"
SEED = 20261102
rng = np.random.default_rng(SEED)


def fit_predict(shuffle=False):
    tab = {d: pd.read_parquet(f"{PROC}/tab_{d}.parquet") for d in ["mimic", "eicu", "nwicu"]}
    feats = [c for c in tab["mimic"].columns if c != "stay_id"]
    zm = np.load(f"{PROC}/seq_mimic.npz", allow_pickle=True)
    split = np.load(f"{PROC}/split_mimic.npy")
    y = zm["y_mort"].astype(float)
    Xm = tab["mimic"][feats].values.astype(float)
    keep = ~np.all(np.isnan(Xm[split == 0]), axis=0)
    feats = [f for f, k in zip(feats, keep) if k]
    Xm = Xm[:, keep]
    med = np.nanmedian(Xm[split == 0], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    Xm = np.where(np.isnan(Xm), med, Xm)
    if shuffle:
        for j in range(Xm.shape[1]):
            Xm[:, j] = rng.permutation(Xm[:, j])
    tr = (split == 0) & np.isfinite(y)
    sc = StandardScaler().fit(Xm[tr])
    lr = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(Xm[tr]), y[tr])
    out = {}
    sel = np.isfinite(y) & (split == 2)
    out["mimic_test_y"] = y[sel]
    out["mimic_test_p"] = lr.predict_proba(sc.transform(Xm[sel]))[:, 1]
    for db in ["eicu", "nwicu"]:
        t = tab[db]
        Xt = t[feats].values.astype(float)
        Xt = np.where(np.isnan(Xt), med, Xt)
        if shuffle:
            for j in range(Xt.shape[1]):
                Xt[:, j] = rng.permutation(Xt[:, j])
        zz = np.load(f"{PROC}/seq_{db}.npz", allow_pickle=True)
        yt = zz["y_mort"].astype(float); ok = np.isfinite(yt)
        out[f"{db}_all_y"] = yt[ok]
        out[f"{db}_all_p"] = lr.predict_proba(sc.transform(Xt[ok]))[:, 1]
    return out


for tag, sh in [("tab", False), ("tab_shuf", True)]:
    o = fit_predict(sh)
    np.savez(f"{RES}/preds_{tag}_mort.npz", **o)
    print(tag, {k: v.shape for k, v in o.items()})
