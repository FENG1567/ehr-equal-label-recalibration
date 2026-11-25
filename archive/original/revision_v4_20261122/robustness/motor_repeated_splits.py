"""Ten predeclared patient-group MOTOR splits with split-conditional CIs.

Uses only frozen representations at the exact bounded source paths below. It
does not encode/retrain an EHR foundation model and makes no noninferiority
claim. Every available condition for a cohort is evaluated on the same
stay-ID intersection and split per seed.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
MOTOR = ROOT / "data/external/motor"
INTERIM = ROOT / "data/interim"
OUT = ROOT / "revision_v4_20261122/robustness"
SPLIT_SEEDS = tuple(range(20261122, 20261132))
MOTOR_SPECS = {
    "mimic2k": ("mimic", {
        "real": MOTOR / "exp/real/reprs.pkl",
        "const": MOTOR / "exp/const/reprs.pkl",
        "shuffled": MOTOR / "exp/shuffled/reprs.pkl",
        "temporal_shuffle": MOTOR / "exp/temporal_shuffle/reprs.pkl",
        "patient_median": MOTOR / "exp/patient_median/reprs.pkl",
    }),
    "mimic6k": ("mimic", {
        "real": MOTOR / "exp3/mimic6000/real/reprs.pkl",
        "const": MOTOR / "exp3/mimic6000/const/reprs.pkl",
    }),
    "eicu1k": ("eicu", {
        "real": MOTOR / "exp3/eicu1000/real/reprs.pkl",
        "const": MOTOR / "exp3/eicu1000/const/reprs.pkl",
        "shuffled": MOTOR / "exp3/eicu1000/shuffled/reprs.pkl",
    }),
}


def load_repr(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as fh:
        record = pickle.load(fh)
    stay_ids = np.asarray([int(x) for x in record["patient_ids"]], dtype=np.int64)
    x = np.asarray(record["representations"], dtype=np.float32)
    if x.ndim != 2 or len(x) != len(stay_ids) or x.shape[1] < 768:
        raise ValueError(f"Unexpected MOTOR representation shape at bounded path {path}: {x.shape}")
    if len(np.unique(stay_ids)) != len(stay_ids) or not np.isfinite(x[:, :768]).all():
        raise ValueError(f"Invalid or duplicated stay IDs/representations at {path}")
    return stay_ids, x[:, :768]


def group_split(y: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, bool]:
    unique, inverse = np.unique(groups.astype(str), return_inverse=True)
    patient_y = np.zeros(len(unique), dtype=int)
    np.maximum.at(patient_y, inverse, np.asarray(y, dtype=int))
    stratified = np.unique(patient_y).size == 2 and int(np.bincount(patient_y).min()) >= 2
    if stratified:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        train_i, test_i = next(splitter.split(np.zeros((len(unique), 1)), patient_y))
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(unique))
        n_test = max(1, int(round(.30 * len(unique))))
        test_i, train_i = perm[:n_test], perm[n_test:]
    tr_groups, te_groups = set(unique[train_i]), set(unique[test_i])
    if tr_groups & te_groups:
        raise AssertionError("patient overlap in predetermined MOTOR split")
    train = np.flatnonzero(np.isin(groups.astype(str), list(tr_groups)))
    test = np.flatnonzero(np.isin(groups.astype(str), list(te_groups)))
    return train, test, stratified


def patient_bootstrap_weights(groups: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    unique, inverse = np.unique(groups.astype(str), return_inverse=True)
    rng = np.random.default_rng(seed)
    weights = np.zeros((n_boot, len(groups)), dtype=np.int16)
    for b in range(n_boot):
        sampled = rng.integers(0, len(unique), size=len(unique))
        weights[b] = np.bincount(sampled, minlength=len(unique))[inverse]
    return weights


def metric_bootstrap(y: np.ndarray, predictions: dict[str, np.ndarray],
                     weights: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    auc = {key: np.full(len(weights), np.nan) for key in predictions}
    brier = {key: np.full(len(weights), np.nan) for key in predictions}
    for key, pred in predictions.items():
        for b, weight in enumerate(weights):
            if float(np.sum(weight * y)) == 0 or float(np.sum(weight * (1-y))) == 0:
                continue
            auc[key][b] = roc_auc_score(y, pred, sample_weight=weight)
            brier[key][b] = brier_score_loss(y, pred, sample_weight=weight)
    return auc, brier


def ci(values: np.ndarray) -> tuple[float, float, int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan"), 0
    lo, hi = (float(x) for x in np.percentile(values, [2.5, 97.5]))
    return lo, hi, int(len(values))


def load_cohort(db: str) -> pd.DataFrame:
    path = INTERIM / f"cohort_{db}.parquet"
    cohort = pd.read_parquet(path)
    required = {"stay_id", "subject_id", "mort_hosp"}
    if not required.issubset(cohort.columns):
        raise ValueError(f"{path} lacks columns {sorted(required)}")
    cohort = cohort[["stay_id", "subject_id", "mort_hosp"]].copy()
    cohort["stay_id"] = pd.to_numeric(cohort["stay_id"], errors="raise").astype(np.int64)
    cohort["subject_id"] = cohort["subject_id"].astype("string").str.strip()
    cohort["mort_hosp"] = pd.to_numeric(cohort["mort_hosp"], errors="coerce")
    if cohort["stay_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate stay IDs")
    return cohort


def evaluate_tag(tag: str, db: str, paths: dict[str, Path], *, n_boot: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    available = {condition: path for condition, path in paths.items() if path.is_file()}
    if "real" not in available:
        return [], [{"cohort": tag, "db": db, "status": "real_representation_missing",
                     "known_real_path": str(paths["real"])}]
    loaded = {condition: load_repr(path) for condition, path in available.items()}
    common_ids = set(loaded["real"][0].tolist())
    for ids, _ in loaded.values():
        common_ids.intersection_update(ids.tolist())
    cohort = load_cohort(db).set_index("stay_id", drop=False)
    common_ids.intersection_update(cohort.index.tolist())
    common = np.asarray(sorted(common_ids), dtype=np.int64)
    if len(common) == 0:
        raise ValueError(f"{tag}: no exact stay-ID intersection across MOTOR reps and cohort labels")
    aligned: dict[str, np.ndarray] = {}
    for condition, (ids, x) in loaded.items():
        position = {int(stay): i for i, stay in enumerate(ids)}
        aligned[condition] = x[[position[int(stay)] for stay in common]]
    labels = cohort.loc[common, "mort_hosp"].to_numpy(dtype=float)
    groups = cohort.loc[common, "subject_id"].astype(str).to_numpy()
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all() or pd.isna(groups).any():
        raise ValueError(f"{tag}: aligned mortality labels or subject IDs are invalid")
    output: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    missing = sorted(set(paths) - set(available))
    for seed_i, seed in enumerate(SPLIT_SEEDS):
        train, test, stratified = group_split(labels, groups, seed)
        if set(groups[train]) & set(groups[test]):
            raise AssertionError(f"{tag}/{seed}: train/test patient overlap")
        if len(np.unique(labels[train])) != 2:
            audits.append({"cohort": tag, "db": db, "seed": seed, "status": "training_single_class",
                           "n_train": len(train), "n_test": len(test), "missing_conditions": ";".join(missing)})
            continue
        predictions: dict[str, np.ndarray] = {}
        for condition, x in aligned.items():
            scaler = StandardScaler().fit(x[train])
            x_train = scaler.transform(x[train])
            x_test = scaler.transform(x[test])
            model = LogisticRegression(C=1.0, max_iter=3000, random_state=seed, solver="lbfgs")
            model.fit(x_train, labels[train].astype(int))
            predictions[condition] = model.predict_proba(x_test)[:, 1]
        y_test = labels[test].astype(int)
        groups_test = groups[test]
        weights = patient_bootstrap_weights(groups_test, n_boot, seed + 1000)
        auc_boot, brier_boot = metric_bootstrap(y_test, predictions, weights)
        audit = {
            "cohort": tag, "db": db, "seed": seed, "seed_order": seed_i + 1,
            "status": "ok", "n_common_stays": len(common), "n_patients": int(len(np.unique(groups))),
            "n_train_stays": len(train), "n_test_stays": len(test),
            "n_train_patients": int(len(np.unique(groups[train]))),
            "n_test_patients": int(len(np.unique(groups_test))),
            "n_train_events": int(labels[train].sum()), "n_test_events": int(y_test.sum()),
            "n_test_nonevents": int((1-y_test).sum()), "patient_overlap": 0,
            "patient_label_stratified": stratified, "bootstrap_draws": n_boot,
            "bootstrap_scope": "fixed patient-group split; conditional on fitted condition-specific LR models",
            "missing_conditions": ";".join(missing), "conditions": ";".join(predictions),
        }
        audits.append(audit)
        for condition, pred in predictions.items():
            alo, ahi, aval = ci(auc_boot[condition])
            blo, bhi, bval = ci(brier_boot[condition])
            output.append({
                "cohort": tag, "db": db, "seed": seed, "condition": condition,
                "n_test_stays": len(test), "n_test_patients": int(len(np.unique(groups_test))),
                "prevalence": float(y_test.mean()), "auroc": float(roc_auc_score(y_test, pred)),
                "auroc_fixed_split_cluster_ci_95_lo": alo, "auroc_fixed_split_cluster_ci_95_hi": ahi,
                "brier": float(brier_score_loss(y_test, pred)),
                "brier_fixed_split_cluster_ci_95_lo": blo, "brier_fixed_split_cluster_ci_95_hi": bhi,
                "valid_auc_bootstrap_draws": aval, "valid_brier_bootstrap_draws": bval,
                "bootstrap_draws": n_boot, "patient_overlap": 0,
                "fixed_split_ci_scope": "patient-cluster bootstrap conditional on this split and fitted LR",
                "noninferiority_claim": False,
            })
        for condition in predictions:
            if condition == "real":
                continue
            d_auc = auc_boot["real"] - auc_boot[condition]
            d_brier = brier_boot["real"] - brier_boot[condition]
            alo, ahi, aval = ci(d_auc)
            blo, bhi, bval = ci(d_brier)
            output.append({
                "cohort": tag, "db": db, "seed": seed,
                "condition": f"paired_delta_real_minus_{condition}",
                "n_test_stays": len(test), "n_test_patients": int(len(np.unique(groups_test))),
                "prevalence": float(y_test.mean()),
                "auroc_delta": float(roc_auc_score(y_test, predictions["real"]) - roc_auc_score(y_test, predictions[condition])),
                "auroc_delta_fixed_split_cluster_ci_95_lo": alo,
                "auroc_delta_fixed_split_cluster_ci_95_hi": ahi,
                "brier_delta": float(brier_score_loss(y_test, predictions["real"]) - brier_score_loss(y_test, predictions[condition])),
                "brier_delta_fixed_split_cluster_ci_95_lo": blo,
                "brier_delta_fixed_split_cluster_ci_95_hi": bhi,
                "valid_auc_bootstrap_draws": aval, "valid_brier_bootstrap_draws": bval,
                "bootstrap_draws": n_boot, "patient_overlap": 0,
                "fixed_split_ci_scope": "paired patient-cluster bootstrap with shared weights; conditional on this split and fitted LR",
                "noninferiority_claim": False,
            })
    return output, audits


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args(argv)
    if args.bootstrap < 1000:
        ap.error("--bootstrap must be at least 1000 fixed-split patient-cluster draws")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows, all_audits = [], []
    for tag, (db, paths) in MOTOR_SPECS.items():
        print(json.dumps({"stage": "motor_start", "cohort": tag,
                          "available": [key for key, value in paths.items() if value.is_file()],
                          "seeds": SPLIT_SEEDS}), flush=True)
        rows, audits = evaluate_tag(tag, db, paths, n_boot=args.bootstrap)
        all_rows.extend(rows)
        all_audits.extend(audits)
    pd.DataFrame(all_rows).to_csv(out_dir / "motor_repeated_split_metrics.csv", index=False)
    pd.DataFrame(all_audits).to_csv(out_dir / "motor_repeated_split_audit.csv", index=False)
    (out_dir / "motor_repeated_split_manifest.json").write_text(json.dumps({
        "status": "complete", "seed_list": SPLIT_SEEDS, "n_predeclared_splits": len(SPLIT_SEEDS),
        "selection": "all 10 fixed seeds reported; no best-split selection",
        "confidence_intervals": "1000+ patient-cluster bootstraps per split; conditional on split and fitted LR",
        "model": "condition-specific StandardScaler + LogisticRegression(C=1); frozen representations only",
        "noninferiority_claim": False,
        "known_representation_paths": {tag: {cond: str(path) for cond, path in paths.items()}
                                        for tag, (_, paths) in MOTOR_SPECS.items()},
    }, indent=2), encoding="utf-8")
    print(json.dumps({"metric_rows": len(all_rows), "split_audit_rows": len(all_audits)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

