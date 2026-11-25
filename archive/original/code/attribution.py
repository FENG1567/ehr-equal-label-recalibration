"""Descriptive attribution: what domain differences track the calibration-slope drop?

For each external database compute:
  prev        in-hospital mortality
  cov_mean    mean coverage (fraction of stays with >=1 observation in 0-24h) over 48 vars
  n_vars      number of variables with coverage >= 1%
  density     mean events per stay in 0-24h (observation intensity)
  jaccard     Jaccard overlap of the observed-variable set with MIMIC
Then regress slope / BSS on these (n is small: descriptive only, with bootstrap CI).
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

ROOT = "/data/admin/fengshuo/ehr_fm_20261102"
INT = f"{ROOT}/data/interim"
PROC = f"{ROOT}/data/processed"
RES = f"{ROOT}/results/tables"

DBS = ["mimic", "eicu", "nwicu", "sicdb", "hirid", "c2012"]
LABEL = {"mimic": "MIMIC-IV(test)", "eicu": "eICU-CRD", "nwicu": "NWICU",
         "sicdb": "SICdb", "hirid": "HiRID", "c2012": "Challenge2012"}


def describe(db):
    co = pd.read_parquet(f"{INT}/cohort_{db}.parquet")
    ev = pd.read_parquet(f"{INT}/events_{db}.parquet")
    ev = ev[ev["hour"] < 24]
    n = co["stay_id"].nunique()
    cov = ev.groupby("var")["stay_id"].nunique() / max(n, 1)
    cov = cov.reindex([c for c in cov.index]).fillna(0)
    n_ev = ev.groupby("stay_id").size()
    return dict(db=db, n=n, prev=float(co["mort_hosp"].mean()),
                cov_mean=float(cov.clip(upper=1).mean()), n_vars=int((cov >= 0.01).sum()),
                density=float(n_ev.mean()), covset=set(cov[cov >= 0.01].index))


def load_probe():
    rows = []
    for tag in ["main20k", "big20k"]:
        main = pd.read_csv(f"{RES}/probe_{tag}_mort.csv")
        for _, r in main.iterrows():
            db = {"mimic_test": "mimic", "eicu_all": "eicu", "nwicu_all": "nwicu"}.get(r["db"])
            if db:
                rows.append(dict(tag=tag, db=db, auroc=r["auroc"], slope=r["slope"], bss=r["bss"]))
        for db in ["sicdb", "hirid", "c2012"]:
            f = f"{RES}/probe_{tag}_{db}_mort.csv"
            if os.path.exists(f):
                r = pd.read_csv(f).iloc[0]
                rows.append(dict(tag=tag, db=db, auroc=r["auroc"], slope=r["slope"], bss=r["bss"]))
    return pd.DataFrame(rows)


def boot_ci(y, X, B=1000, seed=20261103):
    rng = np.random.default_rng(seed)
    n = len(y)
    coefs = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        m = LinearRegression().fit(X[i], y[i])
        coefs.append(np.r_[m.intercept_, m.coef_])
    coefs = np.array(coefs)
    return np.percentile(coefs, 2.5, axis=0), np.percentile(coefs, 97.5, axis=0)


def main():
    desc = pd.DataFrame([describe(d) for d in DBS])
    covsets = {r.db: r.covset for _, r in desc.iterrows()}
    src = covsets["mimic"]
    desc["jaccard"] = [len(covsets[d] & src) / max(len(covsets[d] | src), 1) for d in desc.db]
    desc["log_prev_ratio"] = np.log(desc.prev / desc.loc[desc.db == "mimic", "prev"].values[0])
    desc = desc.drop(columns=["covset"])
    desc.to_csv(f"{RES}/domain_descriptors.csv", index=False)
    print(desc.round(3).to_string(index=False))

    pr = load_probe()
    df = pr.merge(desc, on="db", how="left")
    df["is_big"] = (df.tag == "big20k").astype(int)
    df = df[df.db != "mimic"]          # external only for the drop model
    df.to_csv(f"{RES}/domain_attribution_points.csv", index=False)
    print("\n", df[["tag", "db", "auroc", "slope", "bss", "prev", "cov_mean", "density", "jaccard"]].round(3).to_string(index=False))

    feats = ["log_prev_ratio", "cov_mean", "density"]
    X = df[feats].values
    y = df["slope"].values
    m = LinearRegression().fit(X, y)
    lo, hi = boot_ci(y, X)
    print("\n=== OLS  slope ~ log(prevalence ratio) + coverage + density  (n=%d) ===" % len(y))
    print(f"R2 = {m.score(X, y):.3f}")
    for name, c, l, h in zip(["intercept"] + feats, np.r_[m.intercept_, m.coef_], lo, hi):
        print(f"  {name:18s} {c:+.3f}  [{l:+.3f}, {h:+.3f}]")

    print("\n=== Spearman correlations with slope ===")
    for f in feats + ["jaccard"]:
        r, p = spearmanr(df[f], df["slope"])
        print(f"  {f:18s} rho={r:+.2f}  p={p:.3f}")

    print("\n=== same for BSS ===")
    yb = df["bss"].values
    mb = LinearRegression().fit(X, yb)
    print(f"R2 = {mb.score(X, yb):.3f}")
    for name, c in zip(["intercept"] + feats, np.r_[mb.intercept_, mb.coef_]):
        print(f"  {name:18s} {c:+.3f}")


if __name__ == "__main__":
    main()
