"""Bootstrap 95% CIs for AUROC, calibration slope, ICI, selective-error RRR.

Fast implementations: direct AUC, 1-feature logistic slope, quantile-binned ICI,
entropy-sorted selective error. 300 patient-level bootstrap replicates.
"""
import sys, glob, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

ROOT = "/data/admin/fengshuo/ehr_fm_20261102"
RES = f"{ROOT}/results/tables"
B = 300
rng = np.random.default_rng(20261102)


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fast_stats(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    out = {}
    out["auroc"] = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
    z = _logit(p).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200).fit(z, y)
    out["slope"] = float(lr.coef_[0, 0])
    # binned ICI (10 equal-count bins) - cheap and stable
    q = np.quantile(p, np.linspace(0, 1, 11)); q[-1] += 1e-9
    idx = np.clip(np.digitize(p, q[1:-1]), 0, 9)
    errs = []
    for k in range(10):
        m = idx == k
        if m.sum() >= 20:
            errs.append(abs(y[m].mean() - p[m].mean()))
    out["ici"] = float(np.mean(errs)) if errs else np.nan
    # selective error at 80% coverage, threshold 0.5
    ent = -(np.clip(p, 1e-9, 1 - 1e-9) * np.log(np.clip(p, 1e-9, 1 - 1e-9)) +
            (1 - np.clip(p, 1e-9, 1 - 1e-9)) * np.log(np.clip(1 - p, 1e-9, 1 - 1e-9)))
    order = np.argsort(ent)
    e = ((p >= 0.5).astype(float) != y).astype(float)
    k80 = int(0.8 * len(y))
    out["err_full"] = float(e.mean())
    out["err_80"] = float(e[order[:k80]].mean())
    out["rrr80"] = 1 - out["err_80"] / out["err_full"] if out["err_full"] > 0 else np.nan
    return out


rows = []
for f in sorted(glob.glob(f"{RES}/preds_*_*.npz")):
    z = np.load(f); tag = os.path.basename(f)[6:-4]
    for k in [x[:-2] for x in z.files if x.endswith("_y")]:
        y = z[f"{k}_y"]; p = z[f"{k}_p"]
        ok = np.isfinite(y) & np.isfinite(p); y, p = y[ok], p[ok]
        if len(y) < 100 or len(np.unique(y)) < 2:
            continue
        base = fast_stats(y, p)
        # subsample large databases inside bootstrap for speed
        n = len(y)
        boot = np.empty((B, 4)); 
        for b in range(B):
            i = rng.integers(0, n, n)
            try:
                s = fast_stats(y[i], p[i])
                boot[b] = [s["auroc"], s["slope"], s["ici"], s["rrr80"]]
            except Exception:
                boot[b] = np.nan
        lo = np.nanpercentile(boot, 2.5, axis=0); hi = np.nanpercentile(boot, 97.5, axis=0)
        rows.append({"tag": tag, "db": k, "n": n, **{f"m_{a}": b for a, b in base.items()},
                     "auroc_lo": lo[0], "auroc_hi": hi[0], "slope_lo": lo[1], "slope_hi": hi[1],
                     "ici_lo": lo[2], "ici_hi": hi[2], "rrr_lo": lo[3], "rrr_hi": hi[3]})
        print(k, {a: round(b, 3) for a, b in base.items()}, flush=True)

df = pd.DataFrame(rows)
df.to_csv(f"{RES}/ci_all.csv", index=False)
print(df.round(3).to_string(index=False))
