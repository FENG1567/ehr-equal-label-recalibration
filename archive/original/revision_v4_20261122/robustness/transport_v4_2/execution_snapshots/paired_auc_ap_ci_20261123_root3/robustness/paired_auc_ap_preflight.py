"""Read-only artifact preflight for the paired AUROC/AP bootstrap."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paired_baselines import sha256_file
from transport_analysis import (
    ANALYSIS_IDS, TARGET_COHORTS, _load_all_model_frames,
    _paired_target_predictions,
)


def preflight(out_path: Path) -> dict[str, Any]:
    frames, gates, endpoints, provenance = _load_all_model_frames()
    rows: list[dict[str, Any]] = []
    for db in TARGET_COHORTS:
        fm = frames["main20k_real"][db]
        for analysis_id in ANALYSIS_IDS:
            paired = _paired_target_predictions(fm, frames[analysis_id][db], db=db)
            rows.append({
                "db": db,
                "analysis_id": analysis_id,
                "endpoint": endpoints[db],
                "n_stays": int(len(paired)),
                "n_patients": int(paired["subject"].astype(str).nunique()),
                "events": int(paired["y"].sum()),
                "nonevents": int(len(paired) - paired["y"].sum()),
                "paired_identity_validated": True,
            })
    expected_rows = len(TARGET_COHORTS) * len(ANALYSIS_IDS)
    if len(rows) != expected_rows or not all(row["paired_identity_validated"] for row in rows):
        raise ValueError("paired prediction preflight did not validate every cohort/comparator")
    if out_path.exists():
        raise FileExistsError(f"refusing to replace existing preflight output: {out_path}")
    document = {
        "status": "preflight_complete",
        "schema": "recovered_transport_paired_auc_ap_preflight_v1",
        "analysis_scope": "saved predictions only; no fits, bootstrap, or patient-level export",
        "cohorts": list(TARGET_COHORTS),
        "comparators": list(ANALYSIS_IDS),
        "validated_combinations": len(rows),
        "input_provenance": provenance,
        "cohort_gate_provenance": {
            db: {
                "shared_population_identity_sha256": gates[db]["shared_population_identity_sha256"],
                "eligibility_manifest_sha256": gates[db]["eligibility_manifest_sha256"],
                "eligibility_table_sha256": gates[db]["eligibility_table_sha256"],
            }
            for db in TARGET_COHORTS
        },
        "aggregate_counts": rows,
        "patient_level_exported_to_local": False,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_name(f".{out_path.name}.tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(out_path)
    document["output_path"] = str(out_path)
    document["output_sha256"] = sha256_file(out_path)
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    document = preflight(args.out)
    print(json.dumps({
        "status": document["status"],
        "validated_combinations": document["validated_combinations"],
        "output_path": document["output_path"],
        "output_sha256": document["output_sha256"],
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
