"""Control for head_response: within-patient temporal shuffle of values.

Keeps each patient's own value distribution but destroys value-time alignment,
so the sequence remains physiologically plausible.  If AUROC drops here, values
carry timing-specific information; if it does not, the cross-patient drop was
mostly implausibility.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fm
from fm import SeqData, load_model
from head_response import embed_arrays

ROOT = fm.ROOT
RES = f"{ROOT}/results/tables"
DBS = ["mimic", "eicu", "nwicu"]


def within_shuffle(ds, rng):
    V = ds.V.copy(); B = ds.B.copy()
    for i in range(len(ds)):
        k = int(ds.L[i])
        if k < 2:
            continue
        v = V[i, :k]; b = B[i, :k]
        for vid in np.unique(v):
            pos = np.where(v == vid)[0]
            if len(pos) > 1:
                b[pos] = rng.permutation(b[pos])
        B[i, :k] = b
    return V, B


def main():
    dev = fm.pick_device("auto")
    model, _, _ = load_model("main20k", "auto")
    rng = np.random.default_rng(20261118)
    dsm = SeqData("mimic")
    Em = embed_arrays(model, dsm, dsm.V, dsm.B, dev)
    ym = dsm.y["y_mort"].astype(float)
    okm = np.isfinite(ym) & np.isfinite(Em).all(1)
    tr = okm & (dsm.split == 0)
    sc = StandardScaler().fit(Em[tr])
    lr = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(Em[tr]), ym[tr])

    rows = []
    for db in DBS:
        ds = SeqData(db)
        Vw, Bw = within_shuffle(ds, rng)
        Ew = embed_arrays(model, ds, Vw, Bw, dev)
        Ereal = Em if db == "mimic" else embed_arrays(model, ds, ds.V, ds.B, dev)
        y = ds.y["y_mort"].astype(float)
        ev = np.isfinite(y) & np.isfinite(Ew).all(1) & np.isfinite(Ereal).all(1)
        if db == "mimic":
            ev &= dsm.split == 2
        zr = lr.decision_function(sc.transform(Ereal[ev]))
        zw = lr.decision_function(sc.transform(Ew[ev]))
        pr, pw = 1 / (1 + np.exp(-zr)), 1 / (1 + np.exp(-zw))
        yy = y[ev]
        rows.append(dict(db=db, n=int(ev.sum()),
                         auroc_real=float(roc_auc_score(yy, pr)),
                         auroc_within=float(roc_auc_score(yy, pw)),
                         brier_real=float(brier_score_loss(yy, pr)),
                         brier_within=float(brier_score_loss(yy, pw)),
                         mean_abs_dlogit=float(np.mean(np.abs(zr - zw)))))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/head_response_within.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
