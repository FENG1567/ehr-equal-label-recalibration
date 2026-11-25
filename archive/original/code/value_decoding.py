"""Compatibility entrypoint for the leakage-aware v2 value decoder."""
import argparse
from pathlib import Path
import sys

V2_CODE = Path(__file__).resolve().parents[1] / "revision_v2_20261117" / "code"
sys.path.insert(0, str(V2_CODE))

import decoding_motor_v2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--decoder-bootstrap", type=int, default=200)
    args = parser.parse_args()
    argv = ["--mode", "decoder", "--batch-size", str(args.batch_size),
            "--decoder-bootstrap", str(args.decoder_bootstrap)]
    if args.device is not None:
        argv.extend(["--device", args.device])
    raise SystemExit(decoding_motor_v2.main(argv))
