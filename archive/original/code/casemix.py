"""Is the cross-database AUROC drop explained by case-mix?

Direct standardisation: AUROC inside age-band x sex strata, re-weighted to the
SOURCE (MIMIC test) composition.  If standardised AUROC recovers most of the
drop, the gap is case-mix, not model failure.
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = "/data/admin/fengshuo/ehr_fm_20261102"
RES = f"{ROOT}/results/tables"
INT = f"{ROOT}/data/interim"
PROC = f"{ROOT}/data/processed"


def strata(age, sex):
    a = pd.cut(pd.Series(age), [17, 44, 64, 79, 200], labels=["18-44", "45-64", "65-79", "80+"])
    s = np.where(np.asarray(sex) == 1, "M", "F")
    return pd.Series(a.astype(str) + "_" + s)


def std_auroc(y, p, st, w):
    recs = []
    for g in np.unique(st):
        m = st == g
        if m.sum() < 100 or len(np.unique(y[m])) < 2:
            continue
        recs.append((g, roc_auc_score(y[m], p[m]), int(m.sum()), float(w.get(g, 0.0))))
    if not recs:
        return np.nan, 0, pd.DataFrame()
    d = pd.DataFrame(recs, columns=["stratum", "auroc", "n", "w"])
    ws = d.w.sum()
    return (float((d.auroc * d.w).sum() / ws) if ws > 0 else np.nan), len(d), d


def main():
    co = {d: pd.read_parquet(f"{INT}/cohort_{d}.parquet") for d in ["mimic", "eicu", "nwicu"]}
    st = {d: strata(co[d]["age"].values, co[d]["sex"].values) for d in co}
    split = np.load(f"{PROC}/split_mimic.npy")
    src = (split == 2)
    # align: strands for mimic_test rows are st['mimic'][src]
    w = st["mimic"][src].value_counts(normalize=True).to_dict()

    rows = []
    for tag in ["main20k", "big20k", "shuf20k"]:
        f = f"{RES}/preds_{tag}_mort.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f)
        ysrc = z["mimic_test_y"]; psrc = z["mimic_test_p"]
        src_auc = roc_auc_score(ysrc, psrc)
        for db, key in [("mimic_test", "mimic_test"), ("eicu", "eicu_all"), ("nwicu", "nwicu_all")]:
            y = z[f"{key}_y"]; p = z[f"{key}_p"]
            if db == "mimic_test":
                s = st["mimic"][src].values
            else:
                yy = np.load(f"{PROC}/seq_{db}.npz", allow_pickle=True)["y_mort"]
                ok = np.isfinite(yy)
                s = st[db].values[ok]
            assert len(s) == len(y), (db, len(s), len(y))
            raw = roc_auc_score(y, p)
            sadj, nst, tab = std_auroc(y, p, s, w)
            rows.append(dict(tag=tag, db=db, n=len(y), raw_auroc=raw,
                             standardized_auroc=sadj, n_strata=nst,
                             gap_raw=raw - src_auc,
                             gap_after_standardisation=sadj - src_auc if np.isfinite(sadj) else np.nan))
            if db != "mimic_test" and nst:
                print(f"\n{tag} {db}: raw={raw:.3f} std={sadj:.3f} (source={src_auc:.3f})")
                print(tab.sort_values("n", ascending=False).round(3).to_string(index=False))
    df = pd.DataFrame(rows)
    df.to_csv(f"{RES}/casemix.csv", index=False)
    print("\n", df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
