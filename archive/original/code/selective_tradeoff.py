"""Selective-prediction trade-off: what does rejecting 20% actually cost?

For each external database and the flagship encoder (big50m):
  - raw probabilities vs Platt-recalibrated on a 2,000-patient adaptation set
  - at 80% coverage (entropy rejection): sensitivity / specificity / PPV / NPV,
    retained prevalence, and the relative error reduction (threshold 0.5)
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = "/data/admin/fengshuo/ehr_fm_20261102"
RES = f"{ROOT}/results/tables"
PROC = f"{ROOT}/data/processed"
DBS = ["mimic_test", "eicu", "nwicu", "sicdb", "hirid", "c2012"]
SEED = 20261102


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def platt(y, p_adapt, p_all):
    z = logit(p_adapt).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(z, y)
    return 1 / (1 + np.exp(-(lr.coef_[0, 0] * logit(p_all) + lr.intercept_[0])))


def selective(y, p, cov=0.8, thr=0.5):
    ent = -(np.clip(p, 1e-9, 1 - 1e-9) * np.log(np.clip(p, 1e-9, 1 - 1e-9)) +
            (1 - np.clip(p, 1e-9, 1 - 1e-9)) * np.log(np.clip(1 - p, 1e-9, 1 - 1e-9)))
    order = np.argsort(ent)
    k = max(int(round(cov * len(y))), 1)
    sel = order[:k]
    ys, ps = y[sel], p[sel]
    pred = (ps >= thr).astype(float)
    tp = ((pred == 1) & (ys == 1)).sum(); fp = ((pred == 1) & (ys == 0)).sum()
    fn = ((pred == 0) & (ys == 1)).sum(); tn = ((pred == 0) & (ys == 0)).sum()
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1); npv = tn / max(tn + fn, 1)
    err = ((pred != ys)).mean()
    return dict(cov=cov, n_ret=len(sel), prev_ret=float(ys.mean()), sens=float(sens),
                spec=float(spec), ppv=float(ppv), npv=float(npv), err=float(err))


def load(db, tag="big50m"):
    f = f"{RES}/preds_{tag}_mort.npz"
    z = np.load(f)
    key = "mimic_test" if db == "mimic_test" else f"{db}_all"
    if f"{key}_y" not in z.files:
        f2 = f"{RES}/preds_{tag}_{db}_mort.npz"
        if not os.path.exists(f2):
            return None
        z = np.load(f2); key = f"{db}_all"
    return z[f"{key}_y"], z[f"{key}_p"]


def main():
    rows = []
    for db in DBS:
        r = load(db)
        if r is None:
            continue
        y, p = r
        ok = np.isfinite(y) & np.isfinite(p)
        y, p = y[ok], p[ok]
        split = np.load(f"{PROC}/split_{db.replace('mimic_test','mimic')}.npy")
        if db == "mimic_test":
            ev = np.where(split == 2)[0]
        else:
            ev = np.where(split == 2)[0]
        ev = ev[ev < len(y)]
        adapt = np.where(split == 0)[0]
        adapt = adapt[adapt < len(y)]
        rng = np.random.default_rng(SEED)
        adapt = rng.choice(adapt, min(2000, len(adapt)), replace=False)
        p_pl = platt(y[adapt], p[adapt], p)
        full = dict(n=len(ev), prev=float(y[ev].mean()), auroc=float(roc_auc_score(y[ev], p[ev])))
        for arm, pp in [("raw", p), ("platt", p_pl)]:
            s8 = selective(y[ev], pp[ev], 0.8)
            s10 = selective(y[ev], pp[ev], 1.0)
            rows.append(dict(db=db, arm=arm, **{f"full_{k}": v for k, v in full.items()},
                             **{f"c80_{k}": v for k, v in s8.items()},
                             err_full=s10["err"],
                             rrr=1 - s8["err"] / s10["err"] if s10["err"] > 0 else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/selective_tradeoff.csv", index=False)
    cols = ["db", "arm", "n", "prev", "auroc", "c80_prev_ret", "c80_sens", "c80_spec", "c80_ppv", "rrr"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
