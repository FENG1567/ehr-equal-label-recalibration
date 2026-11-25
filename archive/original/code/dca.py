"""Decision-curve analysis (net benefit) for the transfer + recalibration arms.

NB(pt) = TP/N - FP/N * pt/(1-pt);  treat-all = prev - (1-prev)*pt/(1-pt).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = "/data/admin/fengshuo/ehr_fm_20261102"
RES = f"{ROOT}/results/tables"
PROC = f"{ROOT}/data/processed"
THR = np.round(np.arange(0.01, 0.501, 0.01), 3)


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def net_benefit(y, p, thr):
    n = len(y)
    pred = p >= thr
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    return tp / n - fp / n * (thr / (1 - thr))


def curve(y, p):
    return pd.DataFrame({"threshold": THR,
                         "nb": [net_benefit(y, p, t) for t in THR],
                         "prev_nb": [y.mean() - (1 - y.mean()) * t / (1 - t) for t in THR]})


def main():
    rows = []
    # zero-label arms
    for f in sorted(glob.glob(f"{RES}/preds_*_mort.npz")):
        tag = os.path.basename(f).replace("preds_", "").replace("_mort.npz", "")
        if tag not in ("main20k", "big20k", "sup20k"):
            continue
        z = np.load(f)
        for db in ["eicu_all", "nwicu_all"]:
            if f"{db}_y" not in z.files:
                continue
            y, p = z[f"{db}_y"], z[f"{db}_p"]
            d = curve(y, p); d["arm"] = f"{tag}:raw"; d["db"] = db; rows.append(d)
    # SICdb
    f = f"{RES}/preds_main20k_sicdb_mort.npz"
    if os.path.exists(f):
        z = np.load(f)
        d = curve(z["sicdb_all_y"], z["sicdb_all_p"]); d["arm"] = "main20k:raw"; d["db"] = "sicdb_all"; rows.append(d)
    f = f"{RES}/preds_big20k_sicdb_mort.npz"
    if os.path.exists(f):
        z = np.load(f)
        d = curve(z["sicdb_all_y"], z["sicdb_all_p"]); d["arm"] = "big20k:raw"; d["db"] = "sicdb_all"; rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(f"{RES}/dca_curves.csv", index=False)

    # summary: net benefit at clinical thresholds, and best threshold
    summ = []
    for (arm, db), g in df.groupby(["arm", "db"]):
        prev = g.prev_nb.iloc[0] * 0 + np.nan  # placeholder
        base = g[g.threshold == 0.10]
        summ.append(dict(arm=arm, db=db,
                         nb_at_05=float(g.loc[(g.threshold - 0.05).abs().idxmin(), "nb"]),
                         nb_at_10=float(g.loc[(g.threshold - 0.10).abs().idxmin(), "nb"]),
                         nb_at_20=float(g.loc[(g.threshold - 0.20).abs().idxmin(), "nb"]),
                         max_nb=float(g.nb.max()),
                         argmax_thr=float(g.loc[g.nb.idxmax(), "threshold"])))
    s = pd.DataFrame(summ)
    s.to_csv(f"{RES}/dca_summary.csv", index=False)
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
