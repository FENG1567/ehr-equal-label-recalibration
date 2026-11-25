#!/usr/bin/env python3
"""Run cached variant derivation and selected-row verification with phase logs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time


CODE = Path(__file__).resolve().parent
FILES = (CODE / "run_variants.py", CODE / "rebuild_metadata_cache.py",
         CODE / "derive_variants.py", CODE / "verify_variants.py")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


print(json.dumps({
    "phase": "runner_start",
    "mode": sys.argv[1] if len(sys.argv) > 1 else "both",
    "time_epoch": time.time(),
    "code_sha256": {path.name: digest(path) for path in FILES},
}, sort_keys=True), flush=True)
mode = sys.argv[1] if len(sys.argv) > 1 else "both"
if mode not in {"metadata", "derive", "verify", "both"}:
    raise ValueError("mode must be metadata, derive, verify, or both")
if mode == "metadata":
    print("phase=metadata_import_begin", flush=True)
    import rebuild_metadata_cache
    print("phase=metadata_import_complete", flush=True)
    print("phase=metadata_begin", flush=True)
    rebuild_metadata_cache.main()
    print("phase=metadata_complete", flush=True)
if mode in {"derive", "both"}:
    print("phase=derive_import_begin", flush=True)
    import derive_variants
    print("phase=derive_import_complete", flush=True)
    print("phase=derive_begin", flush=True)
    derive_variants.main()
    print("phase=derive_complete", flush=True)
if mode in {"verify", "both"}:
    print("phase=verify_import_begin", flush=True)
    import verify_variants
    print("phase=verify_import_complete", flush=True)
    print("phase=verify_begin", flush=True)
    verify_variants.main()
    print("phase=verify_complete", flush=True)
