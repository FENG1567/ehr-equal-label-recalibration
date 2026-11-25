"""Small masked-EHR transformer: pretraining, probing, transfer, calibration, rejection.

Examples
  python fm.py pretrain --db mimic --n 20000 --epochs 4 --tag m20k
  python fm.py probe --ckpt models/m20k.pt --task mort --tag m20k
  python fm.py finetune --ckpt models/m20k.pt --target eicu --task mort --frac 0.1
  python fm.py baseline_tab --task mort
"""
import os, sys, json, time, math, argparse, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = "/data/admin/fengshuo/ehr_fm_20261102"
PROC = f"{ROOT}/data/processed"
MODELS = f"{ROOT}/models"
RES = f"{ROOT}/results/tables"
os.makedirs(MODELS, exist_ok=True); os.makedirs(RES, exist_ok=True)
SEED = 20261102
NBINS = int(os.environ.get("FM_NBINS", 10))
MASK_ID = NBINS + 1
TASKS = {"mort": "y_mort", "aki": "y_aki", "long": "y_long"}


_THREADS_SET = False


def set_threads():
    global _THREADS_SET
    if _THREADS_SET:
        return
    try:
        torch.set_num_threads(16)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    _THREADS_SET = True


set_threads()


def cross_shuffle_B(V, B, L, seed=20261102):
    """Reassign each variable's value bins to OTHER patients (vectorised roll).

    For every variable, the flat list of its value bins (ordered by patient) is
    rolled by a random offset, so patient i receives value bins that belonged to
    other patients, while the marginal distribution per variable is preserved
    and every patient's observation pattern (V, H, L) stays intact.
    This is the strong null: it destroys patient-specific VALUES, not just time.
    """
    rng = np.random.default_rng(seed)
    B2 = B.copy()
    for v in range(1, int(V.max()) + 1):
        m = V == v
        if not m.any():
            continue
        pos = np.argwhere(m)
        vals = B[m].copy()
        k = int(rng.integers(1, max(len(vals), 2)))
        B2[pos[:, 0], pos[:, 1]] = np.roll(vals, k)
    return B2


def pick_device(arg="auto"):
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free > 6e9:
            return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------- data
SFX = os.environ.get("SEQ_SFX", "")


class SeqData:
    def __init__(self, db):
        z = np.load(f"{PROC}/seq_{db}{SFX}.npz", allow_pickle=True)
        self.V, self.B, self.H, self.L = z["V"], z["B"], z["H"], z["L"]
        self.stay = z["stay_id"]
        self.y = {k: z[k] for k in ["y_mort", "y_aki", "y_long"]}
        self.split = np.load(f"{PROC}/split_{db}.npy")
        self.subject = z["subject"]

    def __len__(self):
        return len(self.L)


def collate(idx, ds, mask_frac=0.2, drop_frac=0.0, rng=None):
    rng = rng or np.random.default_rng()
    L = ds.L[idx]
    m = int(L.max())
    Bv = np.zeros((len(idx), m), dtype=np.int64)
    Bb = np.zeros((len(idx), m), dtype=np.int64)
    Bh = np.zeros((len(idx), m), dtype=np.int64)
    tgt = np.full((len(idx), m), -100, dtype=np.int64)
    w = np.zeros((len(idx), m), dtype=np.float32)
    for j, i in enumerate(idx):
        n = int(ds.L[i])
        v = ds.V[i, :n].astype(np.int64); b = ds.B[i, :n].astype(np.int64); h = ds.H[i, :n].astype(np.int64)
        keep = np.ones(n, dtype=bool)
        if drop_frac > 0:
            keep = rng.random(n) >= drop_frac
            if keep.sum() < 8:
                keep = np.ones(n, dtype=bool)
        v, b, h = v[keep], b[keep], h[keep]
        n = len(v)
        sel = rng.random(n) < mask_frac
        if sel.sum() == 0 and n:
            sel[rng.integers(n)] = True
        tgt[j, :n] = np.where(sel, b, -100)
        r = rng.random(n)
        b = b.copy()
        b[sel & (r < 0.8)] = MASK_ID
        b[sel & (r >= 0.8) & (r < 0.9)] = rng.integers(1, NBINS + 1, size=int((sel & (r >= 0.8) & (r < 0.9)).sum()))
        Bv[j, :n] = v; Bb[j, :n] = b; Bh[j, :n] = h
        # variable-balanced weights
        w[j, :n] = np.where(sel, 1.0, 0.0)
    return (torch.from_numpy(Bv), torch.from_numpy(Bb), torch.from_numpy(Bh),
            torch.from_numpy(tgt), torch.from_numpy(w))


# ---------------------------------------------------------------- model
class EHRTransformer(nn.Module):
    def __init__(self, nvar, nbin=NBINS, nhour=24, d=128, layers=2, heads=4, ff=512, dropout=0.1):
        super().__init__()
        self.var = nn.Embedding(nvar + 1, d, padding_idx=0)
        self.bin = nn.Embedding(nbin + 2, d, padding_idx=0)
        self.hour = nn.Embedding(nhour, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, nbin + 1)
        self.d = d

    def forward(self, v, b, h, pad_mask=None):
        x = self.var(v) + self.bin(b) + self.hour(h)
        y = self.enc(x, src_key_padding_mask=pad_mask)
        return self.norm(y)

    def mlm_logits(self, y):
        return self.out(y)

    def embed(self, v, b, h, pad_mask, ret_hidden=False):
        y = self.forward(v, b, h, pad_mask)
        m = (~pad_mask).float().unsqueeze(-1)
        e = (y * m).sum(1) / m.sum(1).clamp(min=1)
        return (e, y) if ret_hidden else e


def nvar():
    return 48


# ---------------------------------------------------------------- pretrain
def pretrain(db="mimic", n=None, epochs=4, d=128, layers=2, heads=4, ff=512,
             tag="pilot", seed=SEED, lr=1e-3, bs=256, max_len=128, mask_frac=0.2,
             drop_frac=0.1, device="auto"):
    set_threads()
    torch.manual_seed(seed); np.random.seed(seed)
    dev = pick_device(device)
    ds = SeqData(db)
    idx = np.where((ds.split == 0) & (ds.L > 4))[0]
    if n is not None and n < len(idx):
        idx = np.random.default_rng(seed).choice(idx, n, replace=False)
    model = EHRTransformer(nvar(), d=d, layers=layers, heads=heads, ff=ff).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[pretrain] db={db} n={len(idx)} epoch={epochs} params={nparam:,} dev={dev}", flush=True)
    t0 = time.time()
    for ep in range(epochs):
        perm = rng.permutation(idx)
        tot, nb = 0.0, 0
        for s in range(0, len(perm), bs):
            bi = perm[s:s + bs]
            V, B, H, T, W = collate(bi, ds, mask_frac, drop_frac, rng)
            V, B, H, T, W = [t.to(dev) for t in (V, B, H, T, W)]
            pad = V == 0
            y = model.forward(V, B, H, pad)
            logits = model.mlm_logits(y)
            loss = F.cross_entropy(logits.reshape(-1, NBINS + 1), T.reshape(-1),
                                   reduction="none", ignore_index=-100).reshape(T.shape)
            loss = (loss * W).sum() / W.sum().clamp(min=1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss); nb += 1
        print(f"  epoch {ep+1}/{epochs} loss={tot/max(nb,1):.4f} time={time.time()-t0:.0f}s", flush=True)
    ck = {"state": model.state_dict(), "cfg": dict(d=d, layers=layers, heads=heads, ff=ff, nvar=nvar()),
          "tag": tag, "n": int(len(idx)), "db": db, "seed": seed}
    torch.save(ck, f"{MODELS}/{tag}.pt")
    json.dump({"tag": tag, "n": int(len(idx)), "epochs": epochs, "params": nparam,
               "final_loss": tot / max(nb, 1), "sec": time.time() - t0, "db": db},
              open(f"{MODELS}/{tag}.json", "w"), indent=1)
    print("saved", f"{MODELS}/{tag}.pt", flush=True)


def load_model(tag, device="auto"):
    dev = pick_device(device)
    ck = torch.load(f"{MODELS}/{tag}.pt", map_location="cpu", weights_only=False)
    c = ck["cfg"]
    m = EHRTransformer(c["nvar"], d=c["d"], layers=c["layers"], heads=c["heads"], ff=c["ff"])
    m.load_state_dict(ck["state"]); m.to(dev).eval()
    return m, dev, ck


def random_model_like(tag, seed=SEED):
    ck = torch.load(f"{MODELS}/{tag}.pt", map_location="cpu", weights_only=False)
    c = ck["cfg"]; torch.manual_seed(seed)
    return EHRTransformer(c["nvar"], d=c["d"], layers=c["layers"], heads=c["heads"], ff=c["ff"])


@torch.no_grad()
def embed_all(model, db, dev, bs=512, max_len=None, tag=None):
    cache = f"{PROC}/emb_{tag}_{db}.npy" if tag else None
    if cache and os.path.exists(cache):
        return np.load(cache), SeqData(db)
    ds = SeqData(db)
    out = np.zeros((len(ds), model.d), dtype=np.float32)
    for s in range(0, len(ds), bs):
        idx = np.arange(s, min(s + bs, len(ds)))
        m = int(ds.L[idx].max())
        V = np.zeros((len(idx), m), np.int64); B = np.zeros_like(V); H = np.zeros_like(V)
        for j, i in enumerate(idx):
            n = min(ds.L[i], m)
            V[j, :n] = ds.V[i, :n]; B[j, :n] = ds.B[i, :n]; H[j, :n] = ds.H[i, :n]
        Vt, Bt, Ht = [torch.from_numpy(t).to(dev) for t in (V, B, H)]
        e = model.embed(Vt, Bt, Ht, Vt == 0)
        out[idx] = e.cpu().numpy()
    if cache:
        np.save(cache, out)
    return out, ds


# ---------------------------------------------------------------- metrics
def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def calib_metrics(y, p):
    """slope (joint fit), intercept with slope fixed to 1, ICI, Brier, BSS."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, brier_score_loss
    y = np.asarray(y, float); p = np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    out = {"n": int(len(y)), "prev": float(y.mean()) if len(y) else np.nan}
    if len(np.unique(y)) < 2:
        return {**out, "auroc": np.nan, "slope": np.nan, "intercept": np.nan,
                "ici": np.nan, "brier": np.nan, "bss": np.nan}
    out["auroc"] = float(roc_auc_score(y, p))
    z = _logit(p).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(z, y)
    out["slope"] = float(lr.coef_[0, 0]); out["intercept"] = float(lr.intercept_[0])
    # intercept-only, slope fixed 1
    from scipy.optimize import minimize_scalar

    def nll(a):
        q = np.clip(p, 1e-6, 1 - 1e-6)
        zz = _logit(q) + a
        return float(np.mean(np.logaddexp(0, zz) - y * zz))
    r = minimize_scalar(nll, bounds=(-5, 5), method="bounded")
    out["intercept_fixed_slope"] = float(r.x)
    out["brier"] = float(brier_score_loss(y, p))
    p0 = y.mean()
    out["bss"] = float(1 - out["brier"] / (p0 * (1 - p0)))
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        if len(p) > 20000:            # LOWESS is O(n*h); subsample deterministically
            ii = np.random.default_rng(20261102).choice(len(p), 20000, replace=False)
            ps, ys = p[ii], y[ii]
        else:
            ps, ys = p, y
        sm = lowess(ys, ps, frac=2.0 / 3.0, return_sorted=False)
        out["ici"] = float(np.mean(np.abs(np.clip(sm, 0, 1) - ps)))
    except Exception:
        bins = np.quantile(p, np.linspace(0, 1, 11))
        bins[-1] += 1e-9
        idx = np.digitize(p, bins[1:-1])
        err = [abs(y[idx == k].mean() - p[idx == k].mean()) for k in range(10) if (idx == k).sum() > 5]
        out["ici"] = float(np.mean(err)) if err else np.nan
    return out


def bootstrap_ci(y, p, fn, B=400, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    for _ in range(B):
        i = rng.integers(0, n, n)
        try:
            vals.append(fn(y[i], p[i]))
        except Exception:
            pass
    vals = np.array([v for v in vals if np.isfinite(v)])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) > 20 else [np.nan, np.nan]


def rejection_curve(y, p, coverages=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0), thr=0.5):
    """entropy-based selective prediction; error = misclassification at fixed threshold."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    ent = -(np.clip(p, 1e-9, 1 - 1e-9) * np.log(np.clip(p, 1e-9, 1 - 1e-9)) +
            (1 - np.clip(p, 1e-9, 1 - 1e-9)) * np.log(np.clip(1 - p, 1e-9, 1 - 1e-9)))
    order = np.argsort(ent)  # most confident first
    err = ((p >= thr).astype(float) != y).astype(float)
    out = []
    for c in coverages:
        k = max(int(round(c * len(y))), 1)
        sel = order[:k]
        out.append({"coverage": c, "error": float(err[sel].mean()),
                    "brier": float(np.mean((p[sel] - y[sel]) ** 2)),
                    "auroc": float(roc_auc_score(y[sel], p[sel])) if len(np.unique(y[sel])) > 1 else np.nan})
    return out


def roc_auc_score(y, p):
    from sklearn.metrics import roc_auc_score as f
    return f(y, p)


# ---------------------------------------------------------------- probe / transfer
def fit_probe(X, y, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=C, max_iter=3000, solver="lbfgs")
    lr.fit(sc.transform(X), y)
    return sc, lr


def apply_probe(sc, lr, X):
    return lr.predict_proba(sc.transform(X))[:, 1]


def run_probe(tag, task, target_dbs=("eicu", "nwicu"), device="auto", out=None):
    model, dev, ck = load_model(tag, device)
    Em, dm = embed_all(model, "mimic", dev, tag=tag)
    ym = dm.y[TASKS[task]]
    tr = dm.split == 0; va = dm.split == 1; te = dm.split == 2
    ok = np.isfinite(ym)
    sc, lr = fit_probe(Em[tr & ok], ym[tr & ok])
    rows, preds = [], {}

    def add(name, yy, X, store=None):
        p = apply_probe(sc, lr, X)
        m = calib_metrics(yy, p)
        m.update({"db": name, "task": task, "model": tag})
        rows.append(m)
        if store:
            preds[store] = (yy, p)
        return p

    add("mimic_val", ym[va], Em[va])
    add("mimic_test", ym[te], Em[te], store="mimic_test")
    for db in target_dbs:
        Et, dt = embed_all(model, db, dev, tag=tag)
        yt = dt.y[TASKS[task]]
        ok2 = np.isfinite(yt)
        p = add(f"{db}_all", yt[ok2], Et[ok2], store=f"{db}_all")
        rc = rejection_curve(yt[ok2], p, thr=0.5)
        rows[-1]["err_full"] = rc[-1]["error"]
        rows[-1]["err_80"] = rc[3]["error"]
        rows[-1]["rrr_80"] = 1 - rc[3]["error"] / rc[-1]["error"] if rc[-1]["error"] > 0 else np.nan
    df = pd.DataFrame(rows)
    path = out or f"{RES}/probe_{tag}_{task}.csv"
    df.to_csv(path, index=False)
    try:
        np.savez(f"{RES}/preds_{tag}_{task}.npz",
                 **{f"{k}_y": np.asarray(v[0], dtype=float) for k, v in preds.items()},
                 **{f"{k}_p": np.asarray(v[1], dtype=float) for k, v in preds.items()})
    except Exception as e:
        print("preds save failed", e)
    return df, preds


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("pretrain")
    p1.add_argument("--db", default="mimic"); p1.add_argument("--n", type=int, default=None)
    p1.add_argument("--epochs", type=int, default=4); p1.add_argument("--tag", default="pilot")
    p1.add_argument("--d", type=int, default=128); p1.add_argument("--layers", type=int, default=2)
    p1.add_argument("--heads", type=int, default=4); p1.add_argument("--ff", type=int, default=512)
    p1.add_argument("--seed", type=int, default=SEED); p1.add_argument("--device", default="auto")
    p1.add_argument("--mask", type=float, default=0.2)
    p2 = sub.add_parser("probe")
    p2.add_argument("--ckpt", required=True); p2.add_argument("--task", default="mort")
    p2.add_argument("--device", default="auto")
    a = ap.parse_args()
    if a.cmd == "pretrain":
        pretrain(db=a.db, n=a.n, epochs=a.epochs, tag=a.tag, d=a.d, layers=a.layers,
                 heads=a.heads, ff=a.ff, seed=a.seed, device=a.device, mask_frac=a.mask)
    elif a.cmd == "probe":
        df, _ = run_probe(a.ckpt, a.task, device=a.device)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
