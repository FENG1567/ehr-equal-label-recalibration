"""Controlled observation-density perturbation.

Drop a fixed fraction of measured tokens at inference, keeping the clinical
window and variable set identical, and measure how discrimination and
calibration of the frozen encoder + source probe change.
"""
import json
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

ROOT = fm.ROOT
RES = f"{ROOT}/results/tables"
DBS = ["mimic", "eicu", "nwicu", "sicdb", "hirid", "c2012"]
KEEPS = [1.0, 0.75, 0.5, 0.25]
SEED = 20261117


@torch.no_grad()
def embed_keep(model, db, dev, keep, rng, bs=512):
    ds = SeqData(db)
    out = np.zeros((len(ds), model.d), np.float32)
    Lout = np.zeros(len(ds), np.int64)
    for s in range(0, len(ds), bs):
        idx = np.arange(s, min(s + bs, len(ds)))
        m = max(int(ds.L[idx].max()), 1)
        V = np.zeros((len(idx), m), np.int64); B = np.zeros_like(V); H = np.zeros_like(V)
        for j, i in enumerate(idx):
            k = int(ds.L[i])
            if k == 0:
                continue
            nk = max(1, int(round(k * keep)))
            sel = np.sort(rng.choice(k, nk, replace=False))
            V[j, :nk] = ds.V[i, sel]; B[j, :nk] = ds.B[i, sel]; H[j, :nk] = ds.H[i, sel]
        Lout[idx] = (V > 0).sum(1)
        Vt, Bt, Ht = [torch.from_numpy(t).to(dev) for t in (V, B, H)]
        out[idx] = model.embed(Vt, Bt, Ht, Vt == 0).cpu().numpy()
    return out, ds, Lout


def calib(y, p):
    z = np.clip(p, 1e-6, 1 - 1e-6); lg = np.log(z / (1 - z)).reshape(-1, 1)
    slope = LogisticRegression(C=1e6, max_iter=500).fit(lg, y).coef_[0, 0]
    prev = y.mean(); b = brier_score_loss(y, p)
    # binned ICI
    q = np.quantile(p, np.linspace(0, 1, 11)); q[0], q[-1] = -np.inf, np.inf
    idx = np.clip(np.digitize(p, q[1:-1]), 0, 9)
    ici = np.mean([abs(y[idx == j].mean() - p[idx == j].mean()) for j in range(10) if (idx == j).sum() > 0])
    return slope, b, 1 - b / (prev * (1 - prev)), ici


def main():
    dev = fm.pick_device("auto")
    model, _, _ = load_model("main20k", "auto")
    rows = []
    for keep in KEEPS:
        rng = np.random.default_rng(SEED)
        Em, dm, _ = embed_keep(model, "mimic", dev, keep, rng)
        ym = dm.y["y_mort"].astype(float)
        ok = np.isfinite(ym) & np.isfinite(Em).all(1)
        tr = ok & (dm.split == 0)
        sc = StandardScaler().fit(Em[tr])
        lr = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(Em[tr]), ym[tr])
        for db in DBS:
            if db == "mimic":
                E, ds = Em, dm
                ev = ok & (dm.split == 2)
            else:
                E, ds, _ = embed_keep(model, db, dev, keep, rng)
                ev = np.isfinite(ds.y["y_mort"].astype(float)) & np.isfinite(E).all(1)
            y = ds.y["y_mort"].astype(float)[ev]
            p = lr.predict_proba(sc.transform(E[ev]))[:, 1]
            s, b, bss, ici = calib(y, p)
            rows.append(dict(keep=keep, db=db, n=int(ev.sum()), prev=float(y.mean()),
                             auroc=float(roc_auc_score(y, p)), slope=float(s),
                             brier=float(b), bss=float(bss), ici=float(ici)))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/density_perturbation.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
