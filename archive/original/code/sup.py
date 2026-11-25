"""Supervised-from-scratch encoder (no MLM) and its value-shuffled twin.

Tests whether END-TO-END supervised learning on the same tokenisation uses the
values, or also collapses to the observation pattern.
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fm
from fm import EHRTransformer, NBINS, SeqData, TASKS, pick_device, set_threads

ROOT = fm.ROOT
RES = fm.RES
MODELS = fm.MODELS
SEED = 20261102


def shuffled_copy(ds, seed=SEED):
    """Permute value bins within each (sequence, variable) group: keeps the
    observation pattern (var, hour) and the value multiset, destroys alignment."""
    rng = np.random.default_rng(seed)
    B = ds.B.copy()
    for i in range(len(ds.L)):
        n = int(ds.L[i])
        v = ds.V[i, :n]
        b = B[i, :n].copy()
        for var in np.unique(v):
            m = v == var
            if m.sum() > 1:
                b[m] = rng.permutation(b[m])
        B[i, :n] = b
    return B


def batches(idx, ds, B, bs, shuffle=True, rng=None):
    order = rng.permutation(idx) if shuffle else idx
    for s in range(0, len(order), bs):
        bi = order[s:s + bs]
        m = int(ds.L[bi].max())
        V = np.zeros((len(bi), m), np.int64)
        Bb = np.zeros_like(V)
        H = np.zeros_like(V)
        for j, i in enumerate(bi):
            n = int(ds.L[i])
            V[j, :n] = ds.V[i, :n]
            Bb[j, :n] = B[i, :n]
            H[j, :n] = ds.H[i, :n]
        yield bi, V, Bb, H


def train_eval(tag, shuffle=False, cross=False, n=20000, epochs=3, lr=1e-3, bs=256, seed=SEED):
    set_threads()
    torch.manual_seed(seed); np.random.seed(seed)
    dev = pick_device("auto")
    ds = SeqData("mimic")
    B = fm.cross_shuffle_B(ds.V, ds.B, ds.L, seed) if cross else (shuffled_copy(ds, seed) if shuffle else ds.B)
    y = ds.y["y_mort"].astype(float)
    idx = np.where((ds.split == 0) & np.isfinite(y) & (ds.L > 4))[0]
    rng = np.random.default_rng(seed)
    if n and n < len(idx):
        idx = rng.choice(idx, n, replace=False)
    model = EHRTransformer(fm.nvar()).to(dev)
    head = nn.Linear(model.d, 1).to(dev)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=0.01)
    pos_w = torch.tensor([(1 - y[idx].mean()) / max(y[idx].mean(), 1e-3)], dtype=torch.float32, device=dev)
    for ep in range(epochs):
        model.train(); head.train(); tot, nb = 0.0, 0
        for bi, V, Bb, H in batches(idx, ds, B, bs, True, rng):
            V, Bb, H = [torch.from_numpy(t).to(dev) for t in (V, Bb, H)]
            e = model.embed(V, Bb, H, V == 0)
            logit = head(e).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(
                logit, torch.tensor(y[bi], dtype=torch.float32, device=dev), pos_weight=pos_w)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
            opt.step(); tot += float(loss); nb += 1
        print(f"  {tag} epoch {ep+1}/{epochs} loss={tot/max(nb,1):.4f}", flush=True)

    model.eval(); head.eval()
    out = {}
    with torch.no_grad():
        for db in ["mimic", "eicu", "nwicu"]:
            d2 = SeqData(db)
            Bd = shuffcopy_cached(d2, B if db == "mimic" else None, seed) if db == "mimic" else (
                shuffled_copy(d2, seed) if shuffle else d2.B)
            ps = np.zeros(len(d2), np.float32)
            for bi, V, Bb, H in batches(np.arange(len(d2)), d2, Bd, 512, False, rng):
                V, Bb, H = [torch.from_numpy(t).to(dev) for t in (V, Bb, H)]
                e = model.embed(V, Bb, H, V == 0)
                ps[bi] = torch.sigmoid(head(e).squeeze(-1)).cpu().numpy()
            yd = d2.y["y_mort"].astype(float)
            key = "mimic_test" if db == "mimic" else f"{db}_all"
            sel = (d2.split == 2) if db == "mimic" else np.isfinite(yd)
            out[f"{key}_y"] = yd[sel]
            out[f"{key}_p"] = ps[sel]
    np.savez(f"{RES}/preds_{tag}_mort.npz", **out)
    torch.save({"state": model.state_dict(), "cfg": dict(d=model.d, layers=2, heads=4, ff=512,
                                                         nvar=fm.nvar()), "tag": tag},
               f"{MODELS}/{tag}.pt")
    print("saved", tag, {k: v.shape for k, v in out.items()}, flush=True)


def shuffcopy_cached(d2, B, seed):
    return B if B is not None else d2.B


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--cross", action="store_true")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=3)
    a = ap.parse_args()
    train_eval(a.tag, shuffle=a.shuffle, cross=a.cross, n=a.n, epochs=a.epochs)
