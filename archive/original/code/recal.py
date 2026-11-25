"""Target intercept recalibration (slope fixed to 1) on a held-out adaptation set,
then re-evaluate selective prediction.

Protocol
  adaptation : target patients in split 0 (never used for training the probe)
  evaluation : target patients in split 2
  fit        : a single intercept a on logit(p); slope stays 1
  sensitivity: full Platt (a,b) two-parameter recalibration
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = "/data/admin/fengshuo/ehr_fm_20261102"
RES = f"{ROOT}/results/tables"
PROC = f"{ROOT}/data/processed"
SEED = 20261102


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_intercept(y, p):
    z = logit(p)

    def nll(a):
        return float(np.mean(np.logaddexp(0, z + a) - y * (z + a)))

    r = minimize_scalar(nll, bounds=(-15, 15), method="bounded")
    return float(r.x)


def fit_platt(y, p):
    z = logit(p).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(z, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def ici_bin(y, p, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    q[-1] += 1e-9
    idx = np.clip(np.digitize(p, q[1:-1]), 0, bins - 1)
    e = [abs(y[idx == k].mean() - p[idx == k].mean()) for k in range(bins) if (idx == k).sum() >= 20]
    return float(np.mean(e)) if e else np.nan


def slope_of(y, p):
    z = logit(p).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(z, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def selective(y, p, cov=0.8, thr=0.5):
    ent = -(np.clip(p, 1e-9, 1 - 1e-9) * np.log(np.clip(p, 1e-9, 1 - 1e-9)) +
            (1 - np.clip(p, 1e-9, 1 - 1e-9)) * np.log(np.clip(1 - p, 1e-9, 1 - 1e-9)))
    order = np.argsort(ent)
    err = ((p >= thr).astype(float) != y).astype(float)
    k = max(int(round(cov * len(y))), 1)
    return float(err.mean()), float(err[order[:k]].mean())


def row(tag, db, arm, y, p):
    a = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
    s, b = slope_of(y, p)
    ef, e80 = selective(y, p)
    return dict(tag=tag, db=db, arm=arm, n=len(y), prev=float(y.mean()), auroc=a,
                slope=s, intercept=b, ici=ici_bin(y, p), brier=brier_score_loss(y, p),
                bss=1 - brier_score_loss(y, p) / (y.mean() * (1 - y.mean())),
                err_full=ef, err_80=e80, rrr80=1 - e80 / ef if ef > 0 else np.nan)


rows = []
for f in sorted(glob.glob(f"{RES}/preds_*_mort.npz")):
    tag = os.path.basename(f)[6:-4]
    z = np.load(f)
    for db in ["eicu", "nwicu"]:
        ky, kp = f"{db}_all_y", f"{db}_all_p"
        if ky not in z.files:
            continue
        y, p = z[ky], z[kp]
        ok = np.isfinite(y) & np.isfinite(p)
        y, p = y[ok], p[ok]
        split = np.load(f"{PROC}/split_{db}.npy")
        assert len(split) == len(y), (len(split), len(y))
        adapt = np.where(split == 0)[0]
        rng = np.random.default_rng(SEED)
        adapt = rng.choice(adapt, min(2000, len(adapt)), replace=False)
        ev = np.where(split == 2)[0]
        # intercept-only fit on adaptation set
        a = fit_intercept(y[adapt], p[adapt])
        p_int = 1 / (1 + np.exp(-(logit(p) + a)))
        # Platt sensitivity
        s2, b2 = fit_platt(y[adapt], p[adapt])
        p_pl = 1 / (1 + np.exp(-(s2 * logit(p) + b2)))
        for arm, pp in [("raw", p), ("intercept", p_int), ("platt", p_pl)]:
            rows.append(row(tag, db, arm, y[ev], pp[ev]))
        print(f"{tag} {db}: a={a:+.3f} n_adapt={len(adapt)} n_eval={len(ev)}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(f"{RES}/recalibration.csv", index=False)
print(df.round(3).to_string(index=False))
