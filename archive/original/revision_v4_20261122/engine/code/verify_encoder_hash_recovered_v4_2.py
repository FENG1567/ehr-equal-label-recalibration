#!/usr/bin/env python3
"""Verify registry integration preserves the frozen encoder implementation hash."""
from __future__ import annotations

import json
import runpy
from pathlib import Path

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
ENGINE = REV / "engine"
ARCHIVE = ENGINE / "code_snapshots/v4_engine_pre_registry_20261123_ad400410a0fe2493.py"
INTEGRATED = ENGINE / "code/v4_engine.py"
FROZEN_MANIFEST = ENGINE / "code_snapshots/embedding_manifest_mimic_eicu_core_plus_big20_20261123.json"

manifest = json.loads(FROZEN_MANIFEST.read_text())
expected_hash = manifest.get("encoder_code_sha256")
if not isinstance(expected_hash, str) or len(manifest.get("models", [])) != 20:
    raise ValueError("frozen 20-row manifest lacks the expected encoder identity")

archive_namespace = runpy.run_path(str(ARCHIVE))
integrated_namespace = runpy.run_path(str(INTEGRATED))
archive_hash = archive_namespace["encoder_code_hash"]()
integrated_hash = integrated_namespace["encoder_code_hash"]()
print(f"FROZEN_ENCODER_HASH={expected_hash}", flush=True)
print(f"ARCHIVED_ENCODER_HASH={archive_hash}", flush=True)
print(f"INTEGRATED_ENCODER_HASH={integrated_hash}", flush=True)
if archive_hash != expected_hash or integrated_hash != expected_hash:
    raise RuntimeError("integrated encoder hash does not match the frozen cache identity")
print("ENCODER_HASH_VERIFIED", flush=True)
