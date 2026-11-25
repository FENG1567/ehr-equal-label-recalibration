#!/usr/bin/env python3
"""Run audited paired metrics for the completed corrected core prediction run."""
from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
ENGINE = REV / "engine"
CODE = ENGINE / "code"
ELIGIBILITY = REV / "temporal/results/eligibility/status_consistent_v4_1/landmark_v4_1"
PREDICTION_DIR = ENGINE / "results/prediction_runs/status_consistent_primary_core_fix2-7da980ffcbd8f00b"
PREDICTION_MANIFEST = PREDICTION_DIR / "prediction_manifest.json"
OUTPUT_DIR = ENGINE / "results/paired_metrics/status_consistent_primary_core_fix2_1000boot"
ANALYSIS_IDS = [
    "main20k_real",
    "main20k_constant",
    "main20k_common8_mask",
    "main20k_no_h0_therapy",
    "abl_const20k_real",
    "abl_const20k_constant",
    "common20k_common8_mask",
]
BOOTSTRAP = 1000
SEED = 20261122


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if OUTPUT_DIR.exists():
    raise FileExistsError(f"refusing to overwrite paired-metrics output: {OUTPUT_DIR}")
if not PREDICTION_MANIFEST.is_file():
    raise FileNotFoundError(f"prediction manifest is missing: {PREDICTION_MANIFEST}")
if not (CODE / "paired_metrics_v4.py").is_file():
    raise FileNotFoundError("paired_metrics_v4.py is missing")

metrics = runpy.run_path(str(CODE / "paired_metrics_v4.py"), run_name="paired_metrics_v4")
metrics["TEMPORAL"] = ELIGIBILITY
# The function's default captured the module's original TEMPORAL at definition time.
metrics["current_eligibility_hashes"].__defaults__ = (ELIGIBILITY,)
args = metrics["parse_args"]([
    "--prediction-dir", str(PREDICTION_DIR),
    "--manifest-path", str(PREDICTION_MANIFEST),
    "--output-dir", str(OUTPUT_DIR),
    "--analysis-ids", *ANALYSIS_IDS,
    "--bootstrap", str(BOOTSTRAP),
    "--seed", str(SEED),
])
status = metrics["run"](args)
if status != 0:
    raise SystemExit(status)

context = {
    "run_id": "status_consistent_primary_core_fix2",
    "prediction_manifest": str(PREDICTION_MANIFEST),
    "prediction_manifest_sha256": sha256_file(PREDICTION_MANIFEST),
    "paired_metrics_code": str(CODE / "paired_metrics_v4.py"),
    "paired_metrics_code_sha256": sha256_file(CODE / "paired_metrics_v4.py"),
    "eligibility_dir": str(ELIGIBILITY),
    "eligibility_manifest_sha256": sha256_file(ELIGIBILITY / "eligibility_manifest.json"),
    "analysis_ids": ANALYSIS_IDS,
    "bootstrap_replicates": BOOTSTRAP,
    "base_seed": SEED,
}
(OUTPUT_DIR / "paired_core_metrics_run_context.json").write_text(
    json.dumps(context, indent=2, sort_keys=True) + "\n")
