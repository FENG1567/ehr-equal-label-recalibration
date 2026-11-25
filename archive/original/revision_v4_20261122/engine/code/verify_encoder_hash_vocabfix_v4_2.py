#!/usr/bin/env python3
"""Verify the registry/vocabulary fix preserves the frozen encoder identity."""
from __future__ import annotations

import json
import runpy
from pathlib import Path

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
ENGINE = REV / "engine"
ENGINE_SCRIPT = (
    ENGINE / "code/v4_engine_registry_maskfix_recovered_v4_2_resume_"
    "18da7bee9da7.py"
)
FROZEN_MANIFEST = (
    ENGINE / "code_snapshots/embedding_manifest_recovered_v4_2_merged41_"
    "b2eec1f7aca1.json"
)
EXPECTED_ENCODER_SHA256 = (
    "59c3f4059662fb9e1d81baf20120c8668a761a926f5cab5ff4e654f7c16c1abc"
)

manifest = json.loads(FROZEN_MANIFEST.read_text())
if manifest.get("encoder_code_sha256") != EXPECTED_ENCODER_SHA256:
    raise RuntimeError("frozen 41-row manifest encoder identity changed")
namespace = runpy.run_path(str(ENGINE_SCRIPT))
actual = namespace["encoder_code_hash"]()
print(f"EXPECTED_ENCODER_SHA256={EXPECTED_ENCODER_SHA256}", flush=True)
print(f"VOCABFIX_ENCODER_SHA256={actual}", flush=True)
if actual != EXPECTED_ENCODER_SHA256:
    raise RuntimeError("vocabulary-loader fix changed the encoder hash")
print("ENCODER_HASH_VERIFIED", flush=True)
