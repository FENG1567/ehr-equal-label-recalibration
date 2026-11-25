"""Compatibility entrypoint: use the paired patient-cluster v2 MOTOR evaluator."""
import argparse
from pathlib import Path
import sys

V2_CODE = Path(__file__).resolve().parents[1] / "revision_v2_20261117" / "code"
sys.path.insert(0, str(V2_CODE))

import decoding_motor_v2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--motor-bootstrap", type=int, default=1000)
    args = parser.parse_args()
    argv = ["--mode", "motor", "--batch-size", str(args.batch_size),
            "--motor-bootstrap", str(args.motor_bootstrap)]
    argv.extend(["--device", args.device])
    raise SystemExit(decoding_motor_v2.main(argv))
