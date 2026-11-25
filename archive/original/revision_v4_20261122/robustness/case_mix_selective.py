"""Age/sex case-mix and threshold-free selective-prediction robustness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from sklearn.metrics import roc_auc_score

from recalibration import one_index_per_patient
from robustness_common import (
    brier, calibration_slope, load_checked, logloss, write_rows,
)

AGE_LABELS = ["18-49", "50-64", "65-74", "75+"]
COVERAGES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def age_group(age: pd.Series) -> pd.Series:
    x = pd.to_numeric(age, errors="coerce")
    return pd.cut(x, bins=[18, 50, 65, 75, np.inf], right=False,
                  labels=AGE_LABELS, include_lowest=True).astype("string")


def normalize_sex(sex: pd.Series) -> pd.Series:
    x = sex.astype("string").str.strip().str.lower()
    x = x.replace({"m": "male", "1": "male", "1.0": "male",
                   "f": "female", "0": "female", "0.0": "female"})
    return x.where(x.isin(["female", "male"]))


def _slope_cluster_ci(y: np.ndarray, p: np.ndarray, base_weight: np.ndarray | None,
                      n_boot: int, seed: int, cluster_unit: str,
                      context: str) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    point = calibration_slope(y, p, base_weight)
    rng = np.random.default_rng(seed)
    estimates = []
    progress_interval = max(1, n_boot // 10)
    for draw in range(n_boot):
        chosen = rng.integers(0, n, size=n)
        counts = np.bincount(chosen, minlength=n).astype(float)
        if base_weight is not None:
            weights = counts * base_weight
        else:
            weights = counts
        estimate = calibration_slope(y, p, weights)
        if np.isfinite(estimate):
            estimates.append(estimate)
        if (draw + 1) % progress_interval == 0 or draw + 1 == n_boot:
            print(json.dumps({
                "stage": "case_mix_slope_bootstrap_progress", "context": context,
                "draw_completed": draw + 1, "draws_requested": n_boot,
                "valid_estimates": len(estimates),
            }, sort_keys=True), flush=True)
    if len(estimates) < max(100, n_boot // 2):
        lo = hi = float("nan")
        status = "too_few_valid_cluster_bootstrap_draws"
    else:
        lo, hi = (float(v) for v in np.percentile(estimates, [2.5, 97.5]))
        status = "ok"
    return {
        "calibration_slope": point,
        "slope_cluster_ci_95_lo": lo,
        "slope_cluster_ci_95_hi": hi,
        "bootstrap_cluster_unit": cluster_unit,
        "independent_patient_clustering_verified": cluster_unit == "patient",
        "slope_bootstrap_draws_requested": n_boot,
        "slope_bootstrap_draws_valid": len(estimates),
        "slope_ci_status": status,
        "slope_ci_scope": "patient-cluster bootstrap conditional on fixed prediction model and case-mix target weights",
    }


def _metrics(y: np.ndarray, p: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if weights is None:
        return {
            "prevalence": float(y.mean()), "brier": brier(y, p),
            "logloss": logloss(y, p),
            "auroc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan"),
        }
    return {
        "prevalence": float(np.average(y, weights=weights)),
        "brier": float(np.average((y - p) ** 2, weights=weights)),
        "logloss": float(np.average(-y * np.log(np.clip(p, 1e-12, 1))
                                    -(1-y) * np.log(np.clip(1-p, 1e-12, 1)), weights=weights)),
        "auroc": float(roc_auc_score(y, p, sample_weight=weights)) if np.unique(y).size == 2 else float("nan"),
    }


def _selective_rows(db: str, df: pd.DataFrame, seed: int) -> list[dict[str, Any]]:
    y = df["y"].to_numpy(dtype=int)
    p = np.clip(df["p"].to_numpy(dtype=float), 1e-12, 1 - 1e-12)
    n = len(df)
    n_events = int(y.sum())
    n_nonevents = int(n - n_events)
    confidence = np.abs(p - 0.5)
    order = np.lexsort((df["stay_id"].astype(str).to_numpy(), -confidence))
    rows = []
    for coverage in COVERAGES:
        n_keep = min(n, max(1, int(round(coverage * n))))
        kept = order[:n_keep]
        yk, pk = y[kept], p[kept]
        retained_events = int(yk.sum())
        retained_nonevents = int(n_keep - retained_events)
        if n_events:
            random_lo = int(hypergeom.ppf(0.025, n, n_events, n_keep)) / n_events
            random_hi = int(hypergeom.ppf(0.975, n, n_events, n_keep)) / n_events
            event_retention = retained_events / n_events
        else:
            random_lo = random_hi = event_retention = float("nan")
        if n_nonevents:
            random_nonevent_lo = int(hypergeom.ppf(0.025, n, n_nonevents, n_keep)) / n_nonevents
            random_nonevent_hi = int(hypergeom.ppf(0.975, n, n_nonevents, n_keep)) / n_nonevents
            nonevent_retention = retained_nonevents / n_nonevents
        else:
            random_nonevent_lo = random_nonevent_hi = nonevent_retention = float("nan")
        rows.append({
            "db": db,
            "cohort_status": str(df["cohort_status"].iloc[0]),
            "coverage_target": coverage,
            "coverage_observed": n_keep / n,
            "n_total_patients": n,
            "n_kept_patients": n_keep,
            "n_total_events": n_events,
            "n_kept_events": retained_events,
            "event_retention": event_retention,
            "n_total_nonevents": n_nonevents,
            "n_kept_nonevents": retained_nonevents,
            "nonevent_retention": nonevent_retention,
            "random_rejection_event_retention_expected": n_keep / n,
            "random_rejection_event_retention_hypergeom_95_lo": random_lo,
            "random_rejection_event_retention_hypergeom_95_hi": random_hi,
            "random_rejection_nonevent_retention_expected": n_keep / n,
            "random_rejection_nonevent_retention_hypergeom_95_lo": random_nonevent_lo,
            "random_rejection_nonevent_retention_hypergeom_95_hi": random_nonevent_hi,
            "selected_prevalence": float(yk.mean()),
            "selected_auroc": float(roc_auc_score(yk, pk)) if np.unique(yk).size == 2 else float("nan"),
            "selected_brier": brier(yk, pk),
            "selected_logloss": logloss(yk, pk),
            "selection_rule": "highest |p-0.5|; deterministic stay_id tie break; one outcome-blind random stay per patient",
            "random_baseline": "matched patient count under uniform random rejection without replacement; event and nonevent retention reference is hypergeometric",
            "fixed_threshold_risk_reduction_claim": False,
        })
    return rows


def _age_sex_cell_counts(indexed_data: dict[str, pd.DataFrame]) -> dict[str, dict[str, int]]:
    """Count normalized age-sex cells, including explicit zero cells."""
    cells = [f"{age}|{sex}" for age in AGE_LABELS for sex in ("female", "male")]
    result: dict[str, dict[str, int]] = {}
    for db, frame in indexed_data.items():
        complete = frame.dropna(subset=["age_group", "sex_norm"])
        names = complete["age_group"].astype(str) + "|" + complete["sex_norm"].astype(str)
        observed = names.value_counts().to_dict()
        result[db] = {cell: int(observed.get(cell, 0)) for cell in cells}
    return result


def _age_sex_standardized_rows(indexed_data: dict[str, pd.DataFrame],
                               gates: dict[str, dict[str, Any]], *,
                               n_boot: int, seed: int) -> list[dict[str, Any]]:
    rows_standardized: list[dict[str, Any]] = []
    eligible_mix = {
        db: frame.dropna(subset=["age_group", "sex_norm"]).copy()
        for db, frame in indexed_data.items() if "sex_norm" in frame
    }
    if len(eligible_mix) >= 2:
        support_counts = _age_sex_cell_counts(eligible_mix)
        common_cells = sorted(
            cell for cell in next(iter(support_counts.values()))
            if all(support_counts[db][cell] >= 10 for db in eligible_mix)
        )
        if common_cells:
            distributions = {}
            for db, df in eligible_mix.items():
                df["age_sex_cell"] = df["age_group"].astype(str) + "|" + df["sex_norm"].astype(str)
                sub = df.loc[df["age_sex_cell"].isin(common_cells)]
                distributions[db] = sub["age_sex_cell"].value_counts(normalize=True)
            target = pd.concat(distributions.values(), axis=1).fillna(0).mean(axis=1)
            target = target / target.sum()
            for di, (db, df) in enumerate(eligible_mix.items()):
                sub = df.loc[df["age_sex_cell"].isin(common_cells)].copy()
                local = sub["age_sex_cell"].value_counts(normalize=True)
                weights = sub["age_sex_cell"].map(lambda cell: float(target[cell] / local[cell])).to_numpy()
                sy, sp = sub["y"].to_numpy(dtype=int), sub["p"].to_numpy(dtype=float)
                raw = _metrics(sy, sp)
                standardized = _metrics(sy, sp, weights)
                row = {
                    "db": db, "status": "ok", "n_common_support_patients": int(len(sub)),
                    "n_patients_before_common_support": int(len(df)),
                    "common_age_sex_cells": ";".join(common_cells),
                    "target_distribution": json.dumps({k: float(v) for k, v in target.items()}, sort_keys=True),
                    "target_rule": "equal-weight mean of each cohort's age-sex distribution over common cells with >=10 patients per cohort",
                    "cohort_status": str(df["cohort_status"].iloc[0]),
                    "prediction_source": "current v4 core prediction condition; never legacy56m predictions",
                }
                row.update({f"raw_{k}": v for k, v in raw.items()})
                row.update({f"age_sex_standardized_{k}": v for k, v in standardized.items()})
                if np.unique(sy).size == 2:
                    row.update(_slope_cluster_ci(sy, sp, weights, n_boot,
                                                seed + 10000 + di,
                                                gates[db]["bootstrap_cluster_unit"],
                                                context=f"{db}/age_sex_common_support"))
                else:
                    row["calibration_slope"] = float("nan")
                    row["slope_cluster_ci_95_lo"] = float("nan")
                    row["slope_cluster_ci_95_hi"] = float("nan")
                    row["slope_ci_status"] = "single_class_common_support"
                rows_standardized.append(row)
        else:
            rows_standardized.append({
                "status": "insufficient_common_age_sex_support",
                "cohorts": ";".join(sorted(eligible_mix)),
                "note": "No age-sex cell had at least 10 patients in every cohort; no cross-cohort standardized estimate was produced.",
                "cell_counts_by_cohort": json.dumps(support_counts, sort_keys=True),
            })
    else:
        rows_standardized.append({
            "status": "fewer_than_two_cohorts_with_complete_age_sex",
            "cohorts_with_age_sex": ";".join(sorted(eligible_mix)),
        })
    return rows_standardized


def run(data: dict[str, pd.DataFrame], gates: dict[str, dict[str, Any]], *,
        n_boot: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows_age: list[dict[str, Any]] = []
    rows_selective: list[dict[str, Any]] = []
    rows_raw_mix: list[dict[str, Any]] = []
    rows_standardized: list[dict[str, Any]] = []
    indexed_data: dict[str, pd.DataFrame] = {}
    for di, (db, joined) in enumerate(data.items()):
        usable = joined.loc[joined["eligible_24h"] & joined["temporal_verified"]].copy()
        if usable.empty:
            rows_raw_mix.append({"db": db, "status": "no_temporally_eligible_rows"})
            continue
        indexed = one_index_per_patient(usable, seed + di)
        indexed_data[db] = indexed
        cluster_unit = gates[db]["bootstrap_cluster_unit"]
        rows_selective.extend(_selective_rows(db, indexed, seed + di))
        if "age" not in indexed:
            rows_raw_mix.append({"db": db, "status": "age_unavailable_from_current_predictions_eligibility_join"})
            continue
        indexed["age_group"] = age_group(indexed["age"])
        y = indexed["y"].to_numpy(dtype=int)
        p = indexed["p"].to_numpy(dtype=float)
        for group in AGE_LABELS:
            sub = indexed.loc[indexed["age_group"] == group]
            if sub.empty:
                rows_age.append({"db": db, "age_group": group, "status": "no_patients"})
                continue
            sy, sp = sub["y"].to_numpy(dtype=int), sub["p"].to_numpy(dtype=float)
            row = {
                "db": db, "age_group": group, "n_patients": int(len(sub)),
                "events": int(sy.sum()), "non_events": int((1-sy).sum()),
                "cohort_status": str(sub["cohort_status"].iloc[0]),
                "prediction_source": "current v4 core prediction condition; never legacy56m predictions",
                "bootstrap_cluster_unit": cluster_unit,
            }
            row.update(_metrics(sy, sp))
            if np.unique(sy).size == 2:
                row.update(_slope_cluster_ci(sy, sp, None, n_boot,
                                            seed + di * 100 + len(rows_age), cluster_unit,
                                            context=f"{db}/age:{group}"))
                row["status"] = "ok"
            else:
                row.update({"calibration_slope": float("nan"),
                            "slope_cluster_ci_95_lo": float("nan"),
                            "slope_cluster_ci_95_hi": float("nan"),
                            "slope_bootstrap_draws_requested": n_boot,
                            "slope_bootstrap_draws_valid": 0,
                            "slope_ci_status": "single_class_stratum"})
                row["status"] = "single_class_stratum"
            rows_age.append(row)

        if "sex" in indexed:
            indexed["sex_norm"] = normalize_sex(indexed["sex"])
        complete = indexed.dropna(subset=["age_group", "sex_norm"] if "sex_norm" in indexed else ["age_group"]).copy()
        raw_row = {
            "db": db, "status": "ok", "n_index_patients": int(len(indexed)),
            "n_age_available": int(indexed["age_group"].notna().sum()),
            "n_age_missing": int(indexed["age_group"].isna().sum()),
            "n_sex_missing": int(indexed["sex_norm"].isna().sum()) if "sex_norm" in indexed else None,
            "n_age_or_sex_missing": int(len(indexed) - len(complete)) if "sex_norm" in indexed else int(indexed["age_group"].isna().sum()),
            "n_age_sex_missing": int((indexed["age_group"].isna() | indexed["sex_norm"].isna()).sum()) if "sex_norm" in indexed else None,
            "n_complete_age_sex": int(len(complete)) if "sex_norm" in indexed else 0,
            "n_excluded_age_or_sex_missing": int(len(indexed) - len(complete)) if "sex_norm" in indexed else int(indexed["age_group"].isna().sum()),
            "one_patient_record_selection": "one seeded outcome-blind random eligible stay per patient; not chronological",
            "cohort_status": str(indexed["cohort_status"].iloc[0]),
            "prediction_source": "current v4 core prediction condition; never legacy56m predictions",
                "bootstrap_cluster_unit": cluster_unit,
        }
        raw_row.update(_metrics(y, p))
        rows_raw_mix.append(raw_row)

    rows_standardized = _age_sex_standardized_rows(
        indexed_data, gates, n_boot=n_boot, seed=seed,
    )
    return rows_age, rows_selective, rows_raw_mix, rows_standardized


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--eligibility-dir", required=True)
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--model-tag", default="main20k")
    parser.add_argument("--input-condition", default="real")
    parser.add_argument("--endpoint", default="hospital")
    parser.add_argument("--prediction-manifest")
    parser.add_argument("--diagnostic-only", action="store_true",
                         help="allow separately attested retrospective association predictions; outputs remain descriptive")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261122)
    args = parser.parse_args(argv)
    if args.bootstrap < 1000:
        parser.error("--bootstrap must be at least 1000 patient-cluster draws")
    data, gates = load_checked(
        args.predictions, args.eligibility_dir, model_tag=args.model_tag,
        input_condition=args.input_condition, endpoint=args.endpoint,
        prediction_manifest_path=args.prediction_manifest,
        diagnostic_only=args.diagnostic_only,
    )
    age, selective, mix, standardized = run(data, gates, n_boot=args.bootstrap, seed=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "age_stratum_performance_slopes.csv", age)
    write_rows(out_dir / "selective_prediction.csv", selective)
    write_rows(out_dir / "age_sex_case_mix_raw.csv", mix)
    write_rows(out_dir / "age_sex_case_mix_standardized.csv", standardized)
    (out_dir / "case_mix_selection_manifest.json").write_text(json.dumps({
        "status": "complete", "seed": args.seed, "bootstrap_draws": args.bootstrap,
        "cohort_gates": gates, "coverages": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "selective_rule": "retain highest |p-0.5|; fixed patient count; event and nonevent retention compared with matched-coverage hypergeometric random-rejection reference",
        "one_patient_record_selection": "one seeded outcome-blind random eligible stay per patient; this is not described as chronological",
        "fixed_0.5_threshold_risk_reduction_claim": False,
        "age_stratum_slope_ci": "patient-cluster bootstrap; at least 1000 draws; fixed current predictions",
        "age_sex_standardization": "common-support direct standardization; all source cohorts weighted to equal-weight mean age-sex distribution; cells require >=10 patients per cohort",
        "prediction_source": "current v4 core predictions; legacy56m predictions are not accepted by this entry point",
        "prediction_scope": "retrospective association diagnostic only" if args.diagnostic_only else "retrospective landmark prediction at recorded measurement time",
        "clinical_input_availability_verified": False,
        "clinical_promotion_allowed": False,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"age_rows": len(age), "selective_rows": len(selective),
                      "standardized_rows": len(standardized), "cohorts": gates}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

