"""Compatibility entrypoint: evaluate existing MOTOR caches with the corrected v2 split."""
import argparse
from pathlib import Path
import sys

V2_CODE = Path(__file__).resolve().parents[1] / "revision_v2_20261117" / "code"
sys.path.insert(0, str(V2_CODE))

import decoding_motor_v2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_args", nargs="*", help="accepted for compatibility and ignored")
    args = parser.parse_args()
    if args.legacy_args:
        print("Legacy cohort/condition arguments are ignored; evaluating configured existing v2 MOTOR caches.", flush=True)
    else:
        print("Legacy MOTOR build/evaluation replaced; existing-cache v2 evaluation only.", flush=True)
    raise SystemExit(decoding_motor_v2.main(["--mode", "motor", "--device", "cpu"]))
