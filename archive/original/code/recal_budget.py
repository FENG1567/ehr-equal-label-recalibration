"""Recalibration budget curves.

For each target database, how much local labelled data is needed before
intercept-only or Platt recalibration restores calibration?  Budgets are
reported in cases AND events.
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

ROOT = fm.ROOT
RES = f"{ROOT}/results/tables"
CACHE = f"{ROOT}/data/processed/preds_main20k.npz"
DBS = ["mimic", "eicu", "nwicu", "sicdb", "hirid", "c2012"]
BUDGETS = [50, 200, 500, 2000]
REPEATS = 20
SEED = 20261117


@torch.no_grad()
def embeddings(model, db, dev, bs=512):
    ds = SeqData(db)
    out = np.zeros((len(ds), model.d), np.float32)
    for s in range(0, len(ds), bs):
        idx = np.arange(s, min(s + bs, len(ds)))
        m = max(int(ds.L[idx].max()), 1)
        V = np.zeros((len(idx), m), np.int64); B = np.zeros_like(V); H = np.zeros_like(V)
        for j, i in enumerate(idx):
            k = int(ds.L[i])
            V[j, :k] = ds.V[i, :k]; B[j, :k] = ds.B[i, :k]; H[j, :k] = ds.H[i, :k]
        Vt, Bt, Ht = [torch.from_numpy(t).to(dev) for t in (V, B, H)]
        out[idx] = model.embed(Vt, Bt, Ht, Vt == 0).cpu().numpy()
    y = ds.y["y_mort"].astype(float)
    return out, y, ds.split if db == "mimic" else None


def ici(y, p):
    q = np.quantile(p, np.linspace(0, 1, 11)); q[0], q[-1] = -np.inf, np.inf
    idx = np.clip(np.digitize(p, q[1:-1]), 0, 9)
    return np.mean([abs(y[idx == j].mean() - p[idx == j].mean()) for j in range(10) if (idx == j).sum() > 0])


def prepare_preds():
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        return {k: d[k] for k in d.files}
    dev = fm.pick_device("auto")
    model, _, _ = load_model("main20k", "auto")
    Em, ym, sp = embeddings(model, "mimic", dev)
    ok = np.isfinite(ym) & np.isfinite(Em).all(1)
    tr = ok & (sp == 0)
    sc = StandardScaler().fit(Em[tr])
    lr = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(Em[tr]), ym[tr])
    store = {}
    for db in DBS:
        if db == "mimic":
            E, y, sp2 = Em, ym, sp
            ev = ok & (sp2 == 2)
        else:
            E, y, _ = embeddings(model, db, dev)
            ev = np.isfinite(y) & np.isfinite(E).all(1)
        p = lr.predict_proba(sc.transform(E[ev]))[:, 1]
        store[f"{db}_y"] = y[ev]
        store[f"{db}_p"] = p
    np.savez_compressed(CACHE, **store)
    return store


def fit_arm(y, lg, arm):
    if arm == "none":
        return lambda z: z
    if arm == "intercept":
        lr = LogisticRegression(C=1e6, max_iter=500).fit(lg.reshape(-1, 1), y)
        b = lr.intercept_[0]
        return lambda z, b=b: 1 / (1 + np.exp(-(np.log(np.clip(z, 1e-6, 1 - 1e-6) / (1 - np.clip(z, 1e-6, 1 - 1e-6))) + b)))
    lr = LogisticRegression(C=1e6, max_iter=500).fit(lg.reshape(-1, 1), y)
    a, b = lr.coef_[0, 0], lr.intercept_[0]
    return lambda z, a=a, b=b: 1 / (1 + np.exp(-(a * np.log(np.clip(z, 1e-6, 1 - 1e-6) / (1 - np.clip(z, 1e-6, 1 - 1e-6))) + b)))


def main():
    store = prepare_preds()
    rng = np.random.default_rng(SEED)
    rows = []
    for db in DBS:
        y = store[f"{db}_y"]; p = store[f"{db}_p"]
        ok = np.isfinite(y) & np.isfinite(p)
        y, p = y[ok], p[ok]
        pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
        rng.shuffle(pos); rng.shuffle(neg)
        n_pos = len(pos) // 2
        ad = np.concatenate([pos[:n_pos], neg[: len(neg) // 2]])
        te = np.concatenate([pos[n_pos:], neg[len(neg) // 2:]])
        lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        for n in BUDGETS:
            if n > len(ad):
                continue
            for arm in ["none", "intercept", "platt"]:
                bss, icis = [], []
                y_ad = y[ad]; lg_ad = lg[ad]
                pos_ad = np.where(y_ad == 1)[0]; neg_ad = np.where(y_ad == 0)[0]
                n_pos = max(2, int(round(n * len(pos_ad) / max(len(ad), 1))))
                n_pos = min(n_pos, len(pos_ad))
                if n_pos < 2 or n - n_pos < 2:
                    continue
                for _ in range(REPEATS):
                    idx = np.concatenate([
                        rng.choice(pos_ad, n_pos, replace=False),
                        rng.choice(neg_ad, n - n_pos, replace=False),
                    ])
                    f = fit_arm(y_ad[idx], lg_ad[idx], arm)
                    q = f(p[te])
                    prev = y[te].mean(); b = brier_score_loss(y[te], np.clip(q, 1e-6, 1 - 1e-6))
                    bss.append(1 - b / (prev * (1 - prev)))
                    icis.append(ici(y[te], q))
                rows.append(dict(db=db, n_adapt=n, events=int(n_pos),
                                 arm=arm, bss_mean=float(np.mean(bss)), bss_sd=float(np.std(bss)),
                                 ici_mean=float(np.mean(icis))))
                print(rows[-1], flush=True)
        # event counts for the smallest budget
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/recal_budget.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
