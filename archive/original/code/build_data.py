"""Build token sequences + tabular features + labels from interim event tables.

Usage: python build_data.py [mimic|eicu|nwicu|all]
Writes data/processed/{seq_<db>.npz, tab_<db>.parquet, bins.json, meta.json}
"""
import os, sys, json
import numpy as np
import pandas as pd

ROOT = "/data/admin/fengshuo/ehr_fm_20261102"
INT = f"{ROOT}/data/interim"
PROC = f"{ROOT}/data/processed"
os.makedirs(PROC, exist_ok=True)
from extract import VARS, VAR2ID, RANGE  # noqa

NBINS = int(__import__("os").environ.get("NBINS", 10))
MAXLEN = int(__import__("os").environ.get("MAXLEN", 128))
MAXH = int(__import__("os").environ.get("MAXH", 24))
SFX = __import__("os").environ.get("SFX", "")
SEED = 20261102
BIN_VARS = ["vaso", "rrt", "sedation", "sex"]


def fit_edges():
    """Quantile edges fitted on MIMIC training subjects only (no target leakage)."""
    ev = pd.read_parquet(f"{INT}/events_mimic.parquet")
    co = pd.read_parquet(f"{INT}/cohort_mimic.parquet")
    rng = np.random.default_rng(SEED)
    subs = co["subject_id"].drop_duplicates().sample(frac=1.0, random_state=SEED)
    n_tr = int(0.7 * len(subs))
    tr = set(subs.iloc[:n_tr])
    ev_tr = ev[(ev["hour"] < 24) & (ev["stay_id"].isin(set(co.loc[co["subject_id"].isin(tr), "stay_id"])))]
    edges = {}
    for v in VARS:
        x = ev_tr.loc[ev_tr["var"] == v, "val"].values.astype(float)
        if v in BIN_VARS:
            edges[v] = [0.5]
            continue
        if len(x) < 200:
            lo, hi = RANGE.get(v, (0, 1))
            edges[v] = list(np.linspace(lo, hi, NBINS + 1)[1:-1])
            continue
        q = np.quantile(x, np.linspace(0, 1, NBINS + 1)[1:-1])
        q = np.unique(np.round(q, 4))
        edges[v] = [float(t) for t in q]
    json.dump(edges, open(f"{PROC}/bins{SFX}.json", "w"), indent=1)
    return edges


def digitize(x, v, edges):
    e = edges.get(v, [])
    b = np.digitize(x, e, right=False) + 1  # 1..len(e)+1
    return np.clip(b, 1, NBINS)


def build(db, edges):
    ev = pd.read_parquet(f"{INT}/events_{db}.parquet")
    co = pd.read_parquet(f"{INT}/cohort_{db}.parquet")
    ev = ev[ev["hour"] < MAXH].copy()
    ev["hb"] = np.floor(ev["hour"]).astype(int).clip(0, MAXH - 1)
    # one value per (stay, hour-bin, var): mean (binary vars: max)
    agg = ev.groupby(["stay_id", "hb", "var"], as_index=False)["val"].mean()
    agg = agg[agg["var"].isin(VAR2ID)].copy()
    agg["vid"] = agg["var"].map(VAR2ID).astype(np.int16)
    agg = agg.sort_values(["stay_id", "hb", "vid"]).reset_index(drop=True)

    stays = co["stay_id"].values
    sid2i = {int(s): i for i, s in enumerate(stays)}
    N = len(stays)
    V = np.zeros((N, MAXLEN), dtype=np.int16)     # 0 = pad
    B = np.zeros((N, MAXLEN), dtype=np.int16)
    H = np.zeros((N, MAXLEN), dtype=np.int16)
    L = np.zeros(N, dtype=np.int16)

    edges_arr = edges
    rng = np.random.default_rng(SEED)
    g = agg.groupby("stay_id", sort=False)
    for sid, d in g:
        i = sid2i.get(int(sid))
        if i is None:
            continue
        n = len(d)
        if n > MAXLEN:
            sel = rng.choice(n, MAXLEN, replace=False)   # uniform subsample preserves time distribution
            d = d.iloc[np.sort(sel)]
            n = MAXLEN
        V[i, :n] = d["vid"].values + 1                    # +1 so 0 stays pad
        vals = d["val"].values.astype(float)
        vs = d["var"].values
        bins = np.ones(n, dtype=np.int16)
        for v in set(vs):
            m = vs == v
            bins[m] = digitize(vals[m], v, edges_arr)
        B[i, :n] = bins
        H[i, :n] = d["hb"].values
        L[i] = n

    # tabular features: mean of each var in 0-24h (NaN when unobserved)
    tab = pd.DataFrame(index=stays)
    wide = agg.pivot_table(index="stay_id", columns="var", values="val", aggfunc="mean")
    for v in VARS:
        tab[v] = wide[v].reindex(stays).values if v in wide.columns else np.nan
    tab = tab.reset_index().rename(columns={"index": "stay_id"})
    tab.to_parquet(f"{PROC}/tab_{db}.parquet", index=False)

    np.savez_compressed(f"{PROC}/seq_{db}{SFX}.npz", V=V, B=B, H=H, L=L,
                        stay_id=stays.astype(np.int64),
                        y_mort=co["mort_hosp"].values.astype(np.float32),
                        y_aki=co["aki"].values.astype(np.float32),
                        y_long=co["long_icu"].values.astype(np.float32),
                        subject=co["subject_id"].astype(str).values.astype('U32'))
    # splits by subject
    subs = pd.Series(co["subject_id"].astype(str).values).drop_duplicates()
    subs = subs.sample(frac=1.0, random_state=SEED)
    n1, n2 = int(0.7 * len(subs)), int(0.8 * len(subs))
    split = {}
    for name, ss in [("train", subs.iloc[:n1]), ("val", subs.iloc[n1:n2]), ("test", subs.iloc[n2:])]:
        split[name] = set(ss.tolist())
    part = np.array([0 if s in split["train"] else (1 if s in split["val"] else 2)
                     for s in co["subject_id"].astype(str).values], dtype=np.int8)
    np.save(f"{PROC}/split_{db}.npy", part)
    meta = {"n": int(N), "mean_len": float(L.mean()), "max_len": int(L.max()),
            "vars": VARS, "nbins": NBINS,
            "prev": {t: float(np.nanmean(co[c])) for t, c in
                     [("mort", "mort_hosp"), ("aki", "aki"), ("long", "long_icu")]}}
    json.dump(meta, open(f"{PROC}/meta_{db}{SFX}.json", "w"), indent=1)
    print(db, meta, flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    edges = fit_edges() if not os.path.exists(f"{PROC}/bins{SFX}.json") else json.load(open(f"{PROC}/bins{SFX}.json"))
    for db in (["mimic", "eicu", "nwicu"] if which == "all" else [which]):
        build(db, edges)
