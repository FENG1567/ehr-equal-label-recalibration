"""Common-variable rerun: is the transfer gap an artefact of missing variables?

Restrict the vocabulary to variables measured in ALL six databases (coverage
>= 5% everywhere), retrain the same 0.41 M encoder, and re-probe.  If the gap
and the calibration collapse persist, they are not caused by differing variable
coverage.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fm
from fm import EHRTransformer, SeqData, collate, pick_device, set_threads, load_model

ROOT = fm.ROOT
INT = f"{ROOT}/data/interim"
RES = f"{ROOT}/results/tables"
SEED = 20261102
DBS = ["mimic", "eicu", "nwicu", "sicdb", "hirid", "c2012"]


def common_ids(min_cov=0.05):
    import json
    VARS = json.load(open(f"{fm.PROC}/meta_mimic.json"))["vars"]
    covs = []
    for db in DBS:
        ev = pd.read_parquet(f"{INT}/events_{db}.parquet")
        co = pd.read_parquet(f"{INT}/cohort_{db}.parquet")
        ev = ev[ev["hour"] < 24]
        n = co["stay_id"].nunique()
        c = ev.groupby("var")["stay_id"].nunique() / max(n, 1)
        covs.append(c.reindex(VARS).fillna(0.0))
    C = pd.concat(covs, axis=1).min(axis=1)
    keep = [v for v in VARS if C.get(v, 0) >= min_cov]
    ids = [VARS.index(v) + 1 for v in keep]
    print(f"common vars ({len(keep)}): {keep}", flush=True)
    return ids, keep


def mask_batch(V, T, W, ids):
    m = torch.isin(V, torch.tensor(ids))
    V = V * m
    T = torch.where(m, T, torch.full_like(T, -100))
    W = W * m.float()
    return V, T, W


def train(tag="common20k", n=20000, epochs=2, bs=256, seed=SEED):
    set_threads(); torch.manual_seed(seed); np.random.seed(seed)
    dev = pick_device("auto")
    ids, keep = common_ids()
    np.save(f"{RES}/common_vars.npy", np.array(keep, dtype=object), allow_pickle=True)
    ds = SeqData("mimic")
    y = ds.y["y_mort"]
    idx = np.where((ds.split == 0) & (ds.L > 4))[0]
    rng = np.random.default_rng(seed)
    if n and n < len(idx):
        idx = rng.choice(idx, n, replace=False)
    model = EHRTransformer(fm.nvar()).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    for ep in range(epochs):
        model.train(); tot, nb = 0.0, 0
        for s in range(0, len(idx), bs):
            bi = idx[s:s + bs]
            V, B, H, T, W = collate(bi, ds, 0.2, 0.1, rng)
            V, T, W = mask_batch(V, T, W, ids)
            V, B, H, T, W = [t.to(dev) for t in (V, B, H, T, W)]
            out = model.forward(V, B, H, V == 0)
            logits = model.mlm_logits(out)
            loss = F.cross_entropy(logits.reshape(-1, fm.NBINS + 1), T.reshape(-1),
                                   reduction="none", ignore_index=-100).reshape(T.shape)
            loss = (loss * W).sum() / W.sum().clamp(min=1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss); nb += 1
        print(f"  {tag} epoch {ep+1}/{epochs} loss={tot/max(nb,1):.4f}", flush=True)
    torch.save({"state": model.state_dict(),
                "cfg": dict(d=model.d, layers=2, heads=4, ff=512, nvar=fm.nvar()),
                "tag": tag, "n": int(len(idx)), "db": "mimic", "seed": seed}, f"{fm.MODELS}/{tag}.pt")
    print("saved", tag, flush=True)


@torch.no_grad()
def embed_masked(model, db, dev, ids, bs=512):
    ds = SeqData(db)
    out = np.zeros((len(ds), model.d), np.float32)
    allowed = np.array(ids, dtype=np.int64)
    for s in range(0, len(ds), bs):
        idx = np.arange(s, min(s + bs, len(ds)))
        m = max(int(ds.L[idx].max()), 1)
        V = np.zeros((len(idx), m), np.int64); B = np.zeros_like(V); H = np.zeros_like(V)
        for j, i in enumerate(idx):
            k = int(ds.L[i])
            V[j, :k] = ds.V[i, :k]; B[j, :k] = ds.B[i, :k]; H[j, :k] = ds.H[i, :k]
        keep = np.isin(V, allowed)
        V = V * keep
        Vt, Bt, Ht = [torch.from_numpy(t).to(dev) for t in (V, B, H)]
        out[idx] = model.embed(Vt, Bt, Ht, Vt == 0).cpu().numpy()
    return out, ds


def probe(tag="common20k"):
    model, dev, _ = load_model(tag, "auto")
    ids, keep = common_ids()
    Em, dm = embed_masked(model, "mimic", dev, ids)
    ym = dm.y["y_mort"].astype(float)
    okE = np.isfinite(Em).all(1)
    tr = (dm.split == 0) & np.isfinite(ym) & okE
    sc = StandardScaler().fit(Em[tr])
    lr = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(Em[tr]), ym[tr])
    rows = []
    for db in DBS:
        E, ds = (Em, dm) if db == "mimic" else embed_masked(model, db, dev, ids)
        y = (ym if db == "mimic" else ds.y["y_mort"].astype(float))
        okE2 = np.isfinite(E).all(1)
        mask = np.isfinite(y) & okE2
        if db == "mimic":
            mask &= (ds.split == 2)
        p = lr.predict_proba(sc.transform(E[mask]))[:, 1]
        yy = y[mask]; prev = float(yy.mean()); b = brier_score_loss(yy, p)
        z = np.clip(p, 1e-6, 1 - 1e-6); lg = np.log(z/(1-z)).reshape(-1, 1)
        slope = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500).fit(lg, yy).coef_[0, 0]
        rows.append(dict(db=db, n=int(mask.sum()), prev=prev, auroc=float(roc_auc_score(yy, p)),
                         slope=float(slope), brier=float(b), bss=float(1 - b/(prev*(1-prev)))))
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/common_variable_rerun.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("all", "train"):
        train()
    if cmd in ("all", "probe"):
        probe()
