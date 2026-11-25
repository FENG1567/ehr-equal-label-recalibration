#!/usr/bin/env python3
"""Run cache-only core prediction inside the root-verified isolated CPU entry."""
from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
ENGINE = REV / "engine"
ELIGIBILITY = REV / "temporal/results/eligibility/status_consistent_v4_1/landmark_v4_1"
RUN_ID = "status_consistent_primary_core_fix2"
ENGINE_SCRIPT = ENGINE / "code_snapshots/v4_engine_pre_registry_20261123_maskfix_status_consistent_primary_core_fix1.py"
EMBEDDING_MANIFEST = ENGINE / "results/embedding_manifest.json"
ELIGIBILITY_MANIFEST = ELIGIBILITY / "eligibility_manifest.json"
RUNS_DIR = ENGINE / "results/prediction_runs"
CONSUMER_SIDECAR = RUNS_DIR / f"{RUN_ID}_cpu_consumer_runtime.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_record(module) -> dict[str, str]:
    return {"version": str(module.__version__), "file": str(Path(module.__file__).resolve())}


if not ENGINE_SCRIPT.is_file() or not EMBEDDING_MANIFEST.is_file() or not ELIGIBILITY_MANIFEST.is_file():
    raise FileNotFoundError("prediction source or required provenance manifest is missing")
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
expected = {"python": "3.12.3", "numpy": "1.26.4", "pandas": "2.1.4",
            "scipy": "1.11.4", "scikit_learn": "1.8.0", "pyarrow": "18.1.0"}
for name, want in expected.items():
    actual = versions[name] if name == "python" else versions[name]["version"]
    if actual != want:
        raise RuntimeError(f"verified CPU runtime mismatch for {name}: {actual} != {want}")

embedding = json.loads(EMBEDDING_MANIFEST.read_text())
producer_identities = {}
for row in embedding.get("models", []):
    identity = row.get("runtime_identity")
    identity_hash = row.get("runtime_identity_sha256")
    if isinstance(identity, dict) and isinstance(identity_hash, str):
        producer_identities[identity_hash] = identity
consumer = {
    "run_id": RUN_ID,
    "stage": "cache_only_source_head_fit_and_prediction",
    "consumer_device": "cpu",
    "consumer_runtime": versions,
    "consumer_did_not_import_torch": True,
    "numpy_source": "verified CPU RAM copy; copied from system Python 3.12, independently of the GPU producer",
    "producer_embedding_manifest": str(EMBEDDING_MANIFEST),
    "producer_embedding_manifest_sha256": sha256_file(EMBEDDING_MANIFEST),
    "producer_runtime_identities_by_sha256": producer_identities,
    "eligibility_manifest": str(ELIGIBILITY_MANIFEST),
    "eligibility_manifest_sha256": sha256_file(ELIGIBILITY_MANIFEST),
    "engine_source": str(ENGINE_SCRIPT),
    "engine_source_sha256": sha256_file(ENGINE_SCRIPT),
    "entry_script": str(Path(__file__).resolve()),
    "entry_script_sha256": sha256_file(Path(__file__)),
    "cache_embeddings_recomputed": False,
}
RUNS_DIR.mkdir(parents=True, exist_ok=True)
CONSUMER_SIDECAR.write_text(json.dumps(consumer, indent=2, sort_keys=True) + "\n")
print(f"CPU_CONSUMER_RUNTIME_SIDECAR={CONSUMER_SIDECAR}", flush=True)
print(f"CPU_CONSUMER_RUNTIME={json.dumps(versions, sort_keys=True)}", flush=True)

sys.argv = [str(ENGINE_SCRIPT), "--mode", "predict",
            "--models", "main20k", "abl_const20k", "common20k",
            "--cohorts", "mimic", "eicu",
            "--eligibility-dir", str(ELIGIBILITY),
            "--run-id", RUN_ID,
            "--device", "cpu", "--batch-size", "128"]
runpy.run_path(str(ENGINE_SCRIPT), run_name="__main__")


