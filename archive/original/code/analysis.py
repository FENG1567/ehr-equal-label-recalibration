"""Baselines, fine-tuning, and the scaling grid.

  python analysis.py tab --task mort
  python analysis.py randprobe --ckpt m20k --task mort
  python analysis.py finetune --ckpt m20k --target eicu --task mort --frac 0.1
  python analysis.py mult --dbs mimic,eicu --n 20000 --tag k2_20k --epochs 3
  python analysis.py grid
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fm
from fm import (EHRTransformer, collate, NBINS, MASK_ID, SeqData, TASKS,
                calib_metrics, rejection_curve, fit_probe, apply_probe,
                embed_all, pick_device, set_threads, MODELS, RES, SEED)

ROOT = fm.ROOT
PROC = fm.PROC


# ------------------------------------------------------------------ combined pretraining
class ConcatData:
    def __init__(self, dbs, shuffle=False, cross=False):
        zs = [SeqData(d) for d in dbs]
        self.V = np.concatenate([z.V for z in zs])
        self.B = np.concatenate([z.B for z in zs])
        self.H = np.concatenate([z.H for z in zs])
        self.L = np.concatenate([z.L for z in zs])
        # only training-fold patients of each source db
        sp = np.concatenate([z.split for z in zs])
        self.train_mask = sp == 0
        if cross:
            self.B = fm.cross_shuffle_B(self.V, self.B, self.L)
        if shuffle:
            rng = np.random.default_rng(20261102)
            self.B = self.B.copy()
            for i in range(len(self.L)):
                n = int(self.L[i])
                v = self.V[i, :n]
                b = self.B[i, :n].copy()
                for var in np.unique(v):
                    m = v == var
                    if m.sum() > 1:
                        b[m] = rng.permutation(b[m])
                self.B[i, :n] = b

    def __len__(self):
        return len(self.L)


def pretrain_multi(dbs, n_total, tag, epochs=3, d=128, layers=2, heads=4, ff=512,
                   seed=SEED, lr=1e-3, bs=256, device="auto", drop_frac=0.1, mask_frac=0.2,
                   shuffle=False, sample_seed=SEED, cross=False):
    set_threads()
    torch.manual_seed(seed); np.random.seed(seed)
    dev = pick_device(device)
    ds = ConcatData(dbs, shuffle=shuffle, cross=cross)
    idx = np.where(ds.train_mask & (ds.L > 4))[0]
    if n_total and n_total < len(idx):
        idx = np.random.default_rng(sample_seed).choice(idx, n_total, replace=False)
    model = EHRTransformer(fm.nvar(), d=d, layers=layers, heads=heads, ff=ff).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    nparam = sum(p.numel() for p in model.parameters())
    t0 = time.time()
    print(f"[pretrain_multi] dbs={dbs} n={len(idx)} params={nparam:,} dev={dev}", flush=True)
    for ep in range(epochs):
        perm = rng.permutation(idx)
        tot, nb = 0.0, 0
        for s in range(0, len(perm), bs):
            bi = perm[s:s + bs]
            V, B, H, T, W = collate(bi, ds, mask_frac, drop_frac, rng)
            V, B, H, T, W = [t.to(dev) for t in (V, B, H, T, W)]
            y = model.forward(V, B, H, V == 0)
            logits = model.mlm_logits(y)
            loss = F.cross_entropy(logits.reshape(-1, NBINS + 1), T.reshape(-1),
                                   reduction="none", ignore_index=-100).reshape(T.shape)
            loss = (loss * W).sum() / W.sum().clamp(min=1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss); nb += 1
        print(f"  epoch {ep+1}/{epochs} loss={tot/max(nb,1):.4f} t={time.time()-t0:.0f}s", flush=True)
    ck = {"state": model.state_dict(), "cfg": dict(d=d, layers=layers, heads=heads, ff=ff, nvar=fm.nvar()),
          "tag": tag, "n": int(len(idx)), "db": "+".join(dbs), "seed": seed}
    torch.save(ck, f"{MODELS}/{tag}.pt")
    return {"tag": tag, "n": int(len(idx)), "dbs": dbs, "params": nparam,
            "loss": tot / max(nb, 1), "sec": time.time() - t0}


# ------------------------------------------------------------------ tabular baseline
def tab_baseline(task, seed=SEED):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from fm import rejection_curve
    tr = pd.read_parquet(f"{PROC}/tab_mimic.parquet")
    ytr = pd.read_parquet(f"{PROC}/cohort_mimic.parquet")[["stay_id", "mort_hosp", "long_icu", "aki"]] if False else None
    zm = np.load(f"{PROC}/seq_mimic.npz", allow_pickle=True)
    split = np.load(f"{PROC}/split_mimic.npy")
    ycol = TASKS[task]
    y = zm[ycol].astype(float)
    feats = [c for c in tr.columns if c != "stay_id"]
    X = tr[feats].values.astype(float)
    keep = ~np.all(np.isnan(X[split == 0]), axis=0)
    feats = [f for f, k in zip(feats, keep) if k]
    X = X[:, keep]
    med = np.nanmedian(X[split == 0], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X = np.where(np.isnan(X), med, X)
    ok_tr = np.isfinite(y) & (split == 0)
    sc = StandardScaler().fit(X[ok_tr])
    lr = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(X[ok_tr]), y[ok_tr])
    rows = []
    for name, mask in [("mimic_val", (split == 1) & np.isfinite(y)), ("mimic_test", (split == 2) & np.isfinite(y))]:
        p = lr.predict_proba(sc.transform(X[mask]))[:, 1]
        m = calib_metrics(y[mask], p); m.update({"db": name, "task": task, "model": "tab_lr"})
        rows.append(m)
    for db in ["eicu", "nwicu"]:
        t = pd.read_parquet(f"{PROC}/tab_{db}.parquet")
        z = np.load(f"{PROC}/seq_{db}.npz", allow_pickle=True)
        Xt = np.where(np.isnan(t[feats].values.astype(float)), med, t[feats].values.astype(float))
        p = lr.predict_proba(sc.transform(Xt))[:, 1]
        yt = z[ycol].astype(float); ok = np.isfinite(yt)
        m = calib_metrics(yt[ok], p[ok]); m.update({"db": db, "task": task, "model": "tab_lr"})
        rc = rejection_curve(yt[ok], p[ok])
        m["err_full"], m["err_80"] = rc[-1]["error"], rc[3]["error"]
        m["rrr_80"] = 1 - rc[3]["error"] / rc[-1]["error"] if rc[-1]["error"] > 0 else np.nan
        rows.append(m)
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/baseline_tab_{task}.csv", index=False)
    return df


# ------------------------------------------------------------------ random-init paired control
def random_probe(tag, task):
    ck = torch.load(f"{MODELS}/{tag}.pt", map_location="cpu", weights_only=False)
    c = ck["cfg"]
    torch.manual_seed(SEED + 7)
    m = EHRTransformer(c["nvar"], d=c["d"], layers=c["layers"], heads=c["heads"], ff=c["ff"])
    dev = pick_device("auto"); m.to(dev).eval()
    rows = []
    Em, dm = embed_all(m, "mimic", dev)
    y = dm.y[TASKS[task]]; ok = np.isfinite(y)
    sc, lr = fit_probe(Em[(dm.split == 0) & ok], y[(dm.split == 0) & ok])
    for name, mask in [("mimic_val", dm.split == 1), ("mimic_test", dm.split == 2)]:
        p = apply_probe(sc, lr, Em[mask]); mm = calib_metrics(y[mask], p)
        mm.update({"db": name, "task": task, "model": f"rand_{tag}"}); rows.append(mm)
    for db in ["eicu", "nwicu"]:
        Et, dt = embed_all(m, db, dev); yt = dt.y[TASKS[task]]; ok2 = np.isfinite(yt)
        p = apply_probe(sc, lr, Et[ok2]); mm = calib_metrics(yt[ok2], p)
        rc = rejection_curve(yt[ok2], p)
        mm.update({"db": db, "task": task, "model": f"rand_{tag}",
                   "err_full": rc[-1]["error"], "err_80": rc[3]["error"],
                   "rrr_80": 1 - rc[3]["error"] / rc[-1]["error"] if rc[-1]["error"] > 0 else np.nan})
        rows.append(mm)
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/randprobe_{tag}_{task}.csv", index=False)
    return df


# ------------------------------------------------------------------ fine-tune on target
def finetune(tag, target, task, frac=0.1, epochs=3, lr=1e-4, bs=128, seed=SEED):
    set_threads()
    dev = pick_device("auto")
    model, _, ck = fm.load_model(tag, dev)
    head = nn.Linear(model.d, 1).to(dev)
    opt = torch.optim.AdamW([{"params": model.parameters(), "lr": lr},
                             {"params": head.parameters(), "lr": 1e-3}], weight_decay=0.01)
    ds = SeqData(target)
    y = ds.y[TASKS[task]]
    tr_idx = np.where((ds.split == 0) & np.isfinite(y))[0]
    rng = np.random.default_rng(seed)
    if frac < 1:
        tr_idx = rng.choice(tr_idx, max(int(frac * len(tr_idx)), 200), replace=False)
    te_idx = np.where((ds.split == 2) & np.isfinite(y))[0]
    model.train(); head.train()
    pos_w = torch.tensor([(1 - y[tr_idx].mean()) / max(y[tr_idx].mean(), 1e-3)], dtype=torch.float32, device=dev)
    for ep in range(epochs):
        perm = rng.permutation(tr_idx); tot, nb = 0, 0
        for s in range(0, len(perm), bs):
            bi = perm[s:s + bs]
            V, B, H, _, _ = collate(bi, ds, mask_frac=0.0, drop_frac=0.0, rng=rng)
            V, B, H = V.to(dev), B.to(dev), H.to(dev)
            e = model.embed(V, B, H, V == 0)
            logit = head(e).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logit, torch.tensor(y[bi], dtype=torch.float32, device=dev),
                                                      pos_weight=pos_w)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        print(f"  ft epoch {ep+1} loss={tot/max(nb,1):.4f}", flush=True)
    model.eval(); head.eval()
    with torch.no_grad():
        V = np.zeros((len(te_idx), int(ds.L[te_idx].max())), np.int64); B = np.zeros_like(V); H = np.zeros_like(V)
        for j, i in enumerate(te_idx):
            n = ds.L[i]; V[j, :n] = ds.V[i, :n]; B[j, :n] = ds.B[i, :n]; H[j, :n] = ds.H[i, :n]
        e = model.embed(torch.from_numpy(V).to(dev), torch.from_numpy(B).to(dev), torch.from_numpy(H).to(dev),
                        torch.from_numpy(V).to(dev) == 0)
        p = torch.sigmoid(head(e).squeeze(-1)).cpu().numpy()
    m = calib_metrics(y[te_idx], p); m.update({"db": f"{target}_test", "task": task,
                                               "model": f"ft_{tag}_frac{frac}"})
    rc = rejection_curve(y[te_idx], p)
    m.update({"err_full": rc[-1]["error"], "err_80": rc[3]["error"],
              "rrr_80": 1 - rc[3]["error"] / rc[-1]["error"] if rc[-1]["error"] > 0 else np.nan})
    df = pd.DataFrame([m])
    df.to_csv(f"{RES}/finetune_{tag}_{target}_{task}.csv", index=False)
    return df


# ------------------------------------------------------------------ scaling grid
CELLS = [("s05k", ["mimic"], 5000, SEED), ("s20k", ["mimic"], 20000, SEED),
         ("s50k", ["mimic"], 50000, SEED), ("smax", ["mimic"], 70000, SEED),
         ("k2_20k", ["mimic", "eicu"], 20000, SEED),
         ("k3_20k", ["mimic", "eicu", "nwicu"], 20000, SEED),
         ("s20k_r2", ["mimic"], 20000, SEED + 1)]


def grid(epochs=3, cells=None):
    cells = cells or CELLS
    out = []
    for tag, dbs, n, sd in cells:
        if not os.path.exists(f"{MODELS}/{tag}.pt"):
            info = pretrain_multi(dbs, n, tag, epochs=epochs, seed=sd)
        df, _ = fm.run_probe(tag, "mort", out=f"{RES}/probe_{tag}_mort.csv")
        row = df[df.db.isin(["mimic_test", "eicu_all", "nwicu_all"])].copy()
        row["cell"] = tag; row["n"] = n; row["k"] = len(dbs); row["seed"] = sd
        out.append(row)
        print(row[["cell", "db", "auroc", "slope", "ici", "brier", "rrr_80"]].to_string(index=False), flush=True)
    allr = pd.concat(out)
    allr.to_csv(f"{RES}/scaling_grid.csv", index=False)
    return allr


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("tab"); a1.add_argument("--task", default="mort")
    a2 = sub.add_parser("randprobe"); a2.add_argument("--ckpt", required=True); a2.add_argument("--task", default="mort")
    a3 = sub.add_parser("finetune"); a3.add_argument("--ckpt", required=True); a3.add_argument("--target", default="eicu")
    a3.add_argument("--task", default="mort"); a3.add_argument("--frac", type=float, default=0.1)
    a4 = sub.add_parser("mult"); a4.add_argument("--dbs", default="mimic"); a4.add_argument("--n", type=int, default=20000)
    a4.add_argument("--tag", required=True); a4.add_argument("--epochs", type=int, default=3)
    a4.add_argument("--shuffle", action="store_true")
    a5 = sub.add_parser("grid"); a5.add_argument("--epochs", type=int, default=3)
    a5.add_argument("--cells", default="")
    a = ap.parse_args()
    if a.cmd == "tab":
        print(tab_baseline(a.task).to_string(index=False))
    elif a.cmd == "randprobe":
        print(random_probe(a.ckpt, a.task).to_string(index=False))
    elif a.cmd == "finetune":
        print(finetune(a.ckpt, a.target, a.task, a.frac).to_string(index=False))
    elif a.cmd == "mult":
        print(pretrain_multi(a.dbs.split(","), a.n, a.tag, epochs=a.epochs, shuffle=a.shuffle))
    elif a.cmd == "grid":
        cells = None
        if a.cells:
            cells = [(c.split(":")[0], c.split(":")[1].split("+"), int(c.split(":")[2]), SEED) for c in a.cells.split(",")]
        grid(a.epochs, cells)


if __name__ == "__main__":
    main()
