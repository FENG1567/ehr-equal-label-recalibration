#!/usr/bin/env python3
"""Run the reviewed recovered cohorts from frozen cached embeddings on CPU."""
from __future__ import annotations

import hashlib
import json
import os
import re
import runpy
import sys
from pathlib import Path

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
ENGINE = REV / "engine"
ELIGIBILITY = REV / "temporal/results/eligibility/recovered_reviewed_v4_2"
RUN_ID = "recovered_external_primary_v4_2_resume3"
SOURCE_PROBE_REUSE_RUN_ID = "recovered_external_primary_v4_2_vocabfix1"
MODELS = ("main20k", "abl_const20k", "common20k")
COHORTS = ("mimic", "nwicu_discharge_v4_1", "sicdb_zero_offset_v4_2",
           "eicu_semantic_v4_2")
ENGINE_SCRIPT = ENGINE / "code/v4_engine_registry_maskfix_recovered_v4_2_resume_reusefix_a725994d9d5d.py"
EMBEDDING_MANIFEST = ENGINE / "results/embedding_manifest.json"
FROZEN_EMBEDDING_MANIFEST = (
    ENGINE / "code_snapshots/embedding_manifest_recovered_v4_2_merged41_"
    "b2eec1f7aca1.json"
)
MERGE_AUDIT = (
    ENGINE / "code_snapshots/embedding_manifest_merge_audit_recovered_v4_2_"
    "b2eec1f7aca1.json"
)
ELIGIBILITY_MANIFEST = ELIGIBILITY / "eligibility_manifest.json"
RUNS_DIR = ENGINE / "results/prediction_runs"
CONSUMER_SIDECAR = RUNS_DIR / f"{RUN_ID}_cpu_consumer_runtime.json"

EXPECTED_ENGINE_SHA256 = "a725994d9d5dd33d8a2b23ba42afb029c98c5f530658ddda20e76f259a4b9528"
EXPECTED_EMBEDDING_SHA256 = "b2eec1f7aca1857bda440ddb5ec48e477b2a20f753d5dea28b7715f7a1a243fe"
EXPECTED_ELIGIBILITY_SHA256 = "d302ce0a493121f9336f40fd58c5dc63a75a663492293c0153b4ba05efbb8e8f"
EXPECTED_ENCODER_SHA256 = "59c3f4059662fb9e1d81baf20120c8668a761a926f5cab5ff4e654f7c16c1abc"

for hash_name, digest in {
    "EXPECTED_ENGINE_SHA256": EXPECTED_ENGINE_SHA256,
    "EXPECTED_EMBEDDING_SHA256": EXPECTED_EMBEDDING_SHA256,
    "EXPECTED_ELIGIBILITY_SHA256": EXPECTED_ELIGIBILITY_SHA256,
    "EXPECTED_ENCODER_SHA256": EXPECTED_ENCODER_SHA256,
}.items():
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{hash_name} must be exactly 64 lowercase hexadecimal characters")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_record(module) -> dict[str, str]:
    return {"version": str(module.__version__), "file": str(Path(module.__file__).resolve())}


required_files = (ENGINE_SCRIPT, EMBEDDING_MANIFEST, FROZEN_EMBEDDING_MANIFEST,
                  MERGE_AUDIT, ELIGIBILITY_MANIFEST)
missing = [str(path) for path in required_files if not path.is_file()]
if missing:
    raise FileNotFoundError(f"prediction source or provenance files are missing: {missing}")

expected_hashes = (
    ("engine source", ENGINE_SCRIPT, EXPECTED_ENGINE_SHA256),
    ("live embedding manifest", EMBEDDING_MANIFEST, EXPECTED_EMBEDDING_SHA256),
    ("eligibility manifest", ELIGIBILITY_MANIFEST, EXPECTED_ELIGIBILITY_SHA256),
)
for label, path, expected in expected_hashes:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA256 mismatch: {actual} != {expected}")
if EMBEDDING_MANIFEST.read_bytes() != FROZEN_EMBEDDING_MANIFEST.read_bytes():
    raise RuntimeError("live embedding manifest differs from its frozen 41-row snapshot")

merge_audit = json.loads(MERGE_AUDIT.read_text())
if (merge_audit.get("merged_manifest_sha256") != EXPECTED_EMBEDDING_SHA256
        or merge_audit.get("merged_row_count") != 41
        or merge_audit.get("cache_file_sha256_checks") != 41):
    raise RuntimeError("merged embedding manifest audit is incomplete or mismatched")

embedding = json.loads(EMBEDDING_MANIFEST.read_text())
embedding_rows = embedding.get("models", [])
if len(embedding_rows) != 41:
    raise RuntimeError(f"expected 41 frozen embedding rows, found {len(embedding_rows)}")
if embedding.get("encoder_code_sha256") != EXPECTED_ENCODER_SHA256:
    raise RuntimeError("merged manifest does not use the verified recovered-cohort encoder")

eligibility = json.loads(ELIGIBILITY_MANIFEST.read_text())
eligibility_cohorts = eligibility.get("cohorts", {})
if not set(COHORTS).issubset(eligibility_cohorts):
    raise RuntimeError("corrected eligibility manifest is missing a requested cohort")

existing = [path for path in RUNS_DIR.glob(f"{RUN_ID}-*") if path.is_dir()]
if existing:
    raise FileExistsError(f"prediction run ID already exists; refusing overwrite: {existing}")
if CONSUMER_SIDECAR.exists():
    raise FileExistsError(f"consumer runtime sidecar already exists; refusing overwrite: {CONSUMER_SIDECAR}")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
import pyarrow  # noqa: E402

controlled_sys_path = [Path(path).resolve() for path in sys.path if path]
if any(".local" in path.parts for path in controlled_sys_path):
    raise RuntimeError("root-controlled sys.path unexpectedly contains a user-local path")
for name, module in (("numpy", np), ("pandas", pd), ("scipy", scipy),
                     ("scikit_learn", sklearn), ("pyarrow", pyarrow)):
    module_file = Path(module.__file__).resolve()
    if ".local" in module_file.parts:
        raise RuntimeError(f"{name} resolved from a user-local package path: {module_file}")
    if not any(module_file.is_relative_to(root) for root in controlled_sys_path):
        raise RuntimeError(f"{name} is outside the root-controlled sys.path: {module_file}")
numpy_ram_dir = os.environ.get("V4_NUMPY_RAM_DIR", "")
if not numpy_ram_dir:
    raise RuntimeError("root-controlled V4_NUMPY_RAM_DIR is missing")
numpy_ram_root = Path(numpy_ram_dir).resolve()
if not Path(np.__file__).resolve().is_relative_to(numpy_ram_root):
    raise RuntimeError(f"NumPy did not load from verified CPU RAM copy: {np.__file__}")

versions = {
    "python": sys.version.split()[0],
    "python_executable": sys.executable,
    "numpy": package_record(np),
    "pandas": package_record(pd),
    "scipy": package_record(scipy),
    "scikit_learn": package_record(sklearn),
    "pyarrow": package_record(pyarrow),
}
expected_versions = {"python": "3.12.3", "numpy": "1.26.4", "pandas": "2.1.4",
                     "scipy": "1.11.4", "scikit_learn": "1.8.0", "pyarrow": "18.1.0"}
for name, expected in expected_versions.items():
    actual = versions[name] if name == "python" else versions[name]["version"]
    if actual != expected:
        raise RuntimeError(f"verified CPU runtime mismatch for {name}: {actual} != {expected}")

consumer = {
    "run_id": RUN_ID,
    "source_probe_reuse_run_id": SOURCE_PROBE_REUSE_RUN_ID,
    "stage": "cache_only_source_head_fit_and_prediction",
    "consumer_device": "cpu",
    "consumer_runtime": versions,
    "consumer_did_not_import_torch": True,
    "numpy_source": "verified CPU RAM copy; copied from system Python 3.12, independently of the GPU producer",
    "producer_embedding_manifest": str(EMBEDDING_MANIFEST),
    "producer_embedding_manifest_sha256": sha256_file(EMBEDDING_MANIFEST),
    "frozen_embedding_manifest": str(FROZEN_EMBEDDING_MANIFEST),
    "merge_audit": str(MERGE_AUDIT),
    "eligibility_manifest": str(ELIGIBILITY_MANIFEST),
    "eligibility_manifest_sha256": sha256_file(ELIGIBILITY_MANIFEST),
    "engine_source": str(ENGINE_SCRIPT),
    "engine_source_sha256": sha256_file(ENGINE_SCRIPT),
    "entry_script": str(Path(__file__).resolve()),
    "entry_script_sha256": sha256_file(Path(__file__)),
    "models": list(MODELS),
    "cohorts": list(COHORTS),
    "cache_embeddings_recomputed": False,
}
RUNS_DIR.mkdir(parents=True, exist_ok=True)
CONSUMER_SIDECAR.write_text(json.dumps(consumer, indent=2, sort_keys=True) + "\n")
print(f"CPU_CONSUMER_RUNTIME_SIDECAR={CONSUMER_SIDECAR}", flush=True)
print(f"CPU_CONSUMER_RUNTIME={json.dumps(versions, sort_keys=True)}", flush=True)
print(f"PREDICTION_EMBEDDING_MANIFEST_SHA256={EXPECTED_EMBEDDING_SHA256}", flush=True)

sys.argv = [str(ENGINE_SCRIPT), "--mode", "predict",
            "--models", *MODELS,
            "--cohorts", *COHORTS,
            "--eligibility-dir", str(ELIGIBILITY),
            "--run-id", RUN_ID,
            "--reuse-source-probes-from-run-id", SOURCE_PROBE_REUSE_RUN_ID,
            "--device", "cpu", "--batch-size", "128"]
runpy.run_path(str(ENGINE_SCRIPT), run_name="__main__")
