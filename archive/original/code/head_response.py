"""Paired within-encoder intervention: does the frozen encoder keep values, and
does the mortality head respond to them?

For each database we build two views of the SAME patients with the SAME
encoder:
  real   : original value bins
  perm   : value bins permuted across patients within each variable
           (variable identity, timestamps, counts and missingness unchanged)

Then we measure
  (a) value decodability from the representation (ridge R^2),
  (b) response of the fixed mortality probe (delta logit, AUROC, Brier, log loss).
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fm
from fm import SeqData, load_model

ROOT = fm.ROOT
RES = f"{ROOT}/results/tables"
DBS = ["mimic", "eicu", "nwicu"]
DEC_VARS = ["sbp", "creatinine", "lactate"]


def permute_values(ds, rng):
    V = ds.V.copy(); B = ds.B.copy()
    flat_v = V.reshape(-1); flat_b = B.reshape(-1)
    for vid in np.unique(flat_v):
        if vid == 0:
            continue
        pos = np.where(flat_v == vid)[0]
        flat_b[pos] = rng.permutation(flat_b[pos])
    return V, flat_b.reshape(B.shape)


@torch.no_grad()
def embed_arrays(model, ds, V, B, dev, bs=512):
    out = np.zeros((len(ds), model.d), np.float32)
    for s in range(0, len(ds), bs):
        idx = np.arange(s, min(s + bs, len(ds)))
        m = max(int(ds.L[idx].max()), 1)
        Vb = np.zeros((len(idx), m), np.int64); Bb = np.zeros_like(Vb); Hb = np.zeros_like(Vb)
        for j, i in enumerate(idx):
            k = int(ds.L[i])
            Vb[j, :k] = V[i, :k]; Bb[j, :k] = B[i, :k]; Hb[j, :k] = ds.H[i, :k]
        Vt, Bt, Ht = [torch.from_numpy(t).to(dev) for t in (Vb, Bb, Hb)]
        out[idx] = model.embed(Vt, Bt, Ht, Vt == 0).cpu().numpy()
    return out


def decode(E, y, mask):
    X = E[mask]; yy = y[mask]
    ok = np.isfinite(yy) & np.isfinite(X).all(1)
    X, yy = X[ok], yy[ok]
    if len(yy) < 300 or np.std(yy) < 1e-6:
        return np.nan
    kf = KFold(5, shuffle=True, random_state=0)
    pred = cross_val_predict(RidgeCV(alphas=[1.0, 10.0, 100.0, 1000.0]), X, yy, cv=kf)
    return float(r2_score(yy, pred))


def main():
    dev = fm.pick_device("auto")
    model, _, _ = load_model("main20k", "auto")
    rng = np.random.default_rng(20261117)
    tabm = pd.read_parquet(f"{ROOT}/data/processed/tab_mimic.parquet")

    # probe fitted on real MIMIC train
    dsm = SeqData("mimic")
    Em, _ = (None, None)
    Em = embed_arrays(model, dsm, dsm.V, dsm.B, dev)
    ym = dsm.y["y_mort"].astype(float)
    okm = np.isfinite(ym) & np.isfinite(Em).all(1)
    tr = okm & (dsm.split == 0)
    sc = StandardScaler().fit(Em[tr])
    lr = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(Em[tr]), ym[tr])

    rows = []
    for db in DBS:
        ds = SeqData(db)
        Vp, Bp = permute_values(ds, rng)
        E_real = Em if db == "mimic" else embed_arrays(model, ds, ds.V, ds.B, dev)
        E_perm = embed_arrays(model, ds, Vp, Bp, dev)
        y = ds.y["y_mort"].astype(float)
        ev = np.isfinite(y) & np.isfinite(E_real).all(1) & np.isfinite(E_perm).all(1)
        if db == "mimic":
            ev &= dsm.split == 2
        zr = lr.decision_function(sc.transform(E_real[ev]))
        zp = lr.decision_function(sc.transform(E_perm[ev]))
        pr, pp = 1 / (1 + np.exp(-zr)), 1 / (1 + np.exp(-zp))
        yy = y[ev]
        row = dict(db=db, n=int(ev.sum()),
                   auroc_real=float(roc_auc_score(yy, pr)),
                   auroc_perm=float(roc_auc_score(yy, pp)),
                   brier_real=float(brier_score_loss(yy, pr)),
                   brier_perm=float(brier_score_loss(yy, pp)),
                   logloss_real=float(log_loss(yy, pr)),
                   logloss_perm=float(log_loss(yy, pp)),
                   mean_abs_dlogit=float(np.mean(np.abs(zr - zp))),
                   corr_logit=float(np.corrcoef(zr, zp)[0, 1]))
        tab = tabm if db == "mimic" else pd.read_parquet(f"{ROOT}/data/processed/tab_{db}.parquet")
        for v in DEC_VARS:
            if v in tab.columns:
                yv = tab[v].values.astype(float)
                row[f"decode_{v}_real"] = decode(E_real, yv, ev)
                row[f"decode_{v}_perm"] = decode(E_perm, yv, ev)
        rows.append(row)
        print(row, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/head_response_paired.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
