#!/usr/bin/env python3
"""Run only the v5 numeric decoder under a new, collision-checked run ID."""
from __future__ import annotations

import os
import re
import runpy
import sys
from pathlib import Path

R5 = Path("/data/admin/fengshuo/ehr_fm_20261102/revision_v5_20261123")
DECODER = R5 / "scripts/v5_numeric_token_decoder.py"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: v5_numeric_token_decoder_recovery.py RUN_ID")
    run_id = sys.argv[1]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", run_id):
        raise ValueError("RUN_ID contains unsafe path characters")
    output = R5 / "decoding" / f"direct_token_summary_{run_id}"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite numeric decoder output: {output}")
    if not DECODER.is_file():
        raise FileNotFoundError(f"numeric decoder script is missing: {DECODER}")

    os.environ["V5_CONTROL_RUN_ID"] = run_id
    sys.argv = [str(DECODER)]
    runpy.run_path(str(DECODER), run_name="__main__")


if __name__ == "__main__":
    main()
