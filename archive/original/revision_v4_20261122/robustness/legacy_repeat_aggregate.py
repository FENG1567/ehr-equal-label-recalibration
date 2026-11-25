"""Aggregate-only descriptive audit of the historical v2 repeat CSV.

This script deliberately cannot produce patient-level joins or future-risk
claims. The legacy file contains BSS, fit status, and repeat hashes, but no
ID-bound assessment probabilities needed to calculate paired Brier/log loss.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def aggregate(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    required = {"db", "n_adapt", "repeat", "arm", "events", "adaptation_sample_sha256", "fit_ok", "bss"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"legacy repeat CSV lacks columns: {missing}")
    key = ["db", "n_adapt", "repeat"]
    for _, cell in df.groupby(key, sort=False):
        if cell["arm"].duplicated().any():
            raise ValueError("duplicate legacy calibration arm within one repeated patient draw")
        if cell["adaptation_sample_sha256"].nunique() != 1 or cell["events"].nunique() != 1:
            raise ValueError("legacy arms are not paired on the same natural adaptation draw")
    df["fit_ok"] = df["fit_ok"].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if df["fit_ok"].isna().any():
        raise ValueError("legacy fit_ok values are invalid")
    summary = []
    for (db, budget, arm), cell in df.groupby(["db", "n_adapt", "arm"], sort=True):
        vals = cell.loc[cell["fit_ok"], "bss"].dropna().to_numpy(float)
        n_attempt = len(cell)
        summary.append({
            "db": db, "n_adapt": int(budget), "arm": arm,
            "cohort_label": "historical_nonlandmark_v2_descriptive_only",
            "n_attempted": n_attempt, "n_fit_success": int(cell["fit_ok"].sum()),
            "n_fit_fail": int((~cell["fit_ok"]).sum()),
            "fit_failure_rate": float((~cell["fit_ok"]).mean()),
            "bss_mean": float(np.mean(vals)) if len(vals) else np.nan,
            "bss_median": float(np.median(vals)) if len(vals) else np.nan,
            "bss_p05": float(np.percentile(vals, 5)) if len(vals) else np.nan,
            "bss_p95": float(np.percentile(vals, 95)) if len(vals) else np.nan,
            "negative_bss_n": int(np.sum(vals < 0)),
            "negative_bss_rate_among_fit_success": float(np.mean(vals < 0)) if len(vals) else np.nan,
            "bss_assessment_prevalence_reference": "hindsight descriptive only; not a deployable comparator",
        })
    wide = df.pivot(index=key, columns="arm", values="bss").reset_index()
    hash_check = df.groupby(key)["adaptation_sample_sha256"].nunique()
    if not (hash_check == 1).all():
        raise AssertionError("paired adaptation sample hash check failed")
    paired = []
    for _, row in wide.iterrows():
        for arm in sorted(set(df["arm"]) - {"prevalence", "none"}):
            if arm not in wide or "prevalence" not in wide or "none" not in wide:
                continue
            for comparator in ("prevalence", "none"):
                delta = row.get(arm, np.nan) - row.get(comparator, np.nan)
                if pd.notna(delta):
                    paired.append({
                        "db": row["db"], "n_adapt": int(row["n_adapt"]),
                        "repeat": int(row["repeat"]), "arm": arm,
                        "comparator": "same_budget_adaptation_prevalence" if comparator == "prevalence" else "no_recalibration",
                        "paired_delta_bss": float(delta),
                        "pairing": "same repeat and identical adaptation_sample_sha256",
                        "cohort_label": "historical_nonlandmark_v2_descriptive_only",
                    })
    paired_df = pd.DataFrame(paired)
    return pd.DataFrame(summary), paired_df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-repeats", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary, paired = aggregate(args.legacy_repeats)
    summary.to_csv(out / "legacy_repeat_aggregate_summary.csv", index=False)
    paired.to_csv(out / "legacy_repeat_aggregate_paired_delta.csv", index=False)
    (out / "legacy_repeat_aggregate_manifest.json").write_text(json.dumps({
        "status": "complete", "cohort_label": "historical_nonlandmark_v2_descriptive_only",
        "source": str(args.legacy_repeats), "patient_level_data_exported": False,
        "metrics_available": ["BSS", "fit_failure_rate", "negative_BSS_rate", "paired_delta_BSS"],
        "not_available_from_source": ["paired_Brier", "paired_logloss", "ID-bound patient bootstrap CI", "future24h_clinical_claim"],
        "BSS_reference": "hindsight assessment-prevalence reference; descriptive only, not an oracle deployment comparator",
        "interpretation": "Old-cohort aggregate audit only. Do not use as evidence on the new 24-hour landmark cohort.",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"summary_rows": len(summary), "paired_rows": len(paired),
                      "cohort_label": "historical_nonlandmark_v2_descriptive_only"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

