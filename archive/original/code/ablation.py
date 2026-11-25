"""Compatibility entrypoint for the corrected constant-input probe.

The legacy default trained and overwrote ``abl_const20k.pt``. Training from
this entrypoint is disabled; use the existing checkpoint with the v2 probe.
"""
import argparse
from pathlib import Path
import sys

V2_CODE = Path(__file__).resolve().parents[1] / "revision_v2_20261117" / "code"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("probe", "train", "rich", "all"), default="probe")
    args = parser.parse_args(argv)
    if args.mode != "probe":
        parser.error(
            f"legacy mode {args.mode!r} is disabled; encoder training and legacy ablation outputs "
            "are not produced. Use mode 'probe' for the corrected v2 constant-input evaluation."
        )
    sys.path.insert(0, str(V2_CODE))
    import decoding_motor_v2
    return decoding_motor_v2.main(["--mode", "probe"])


if __name__ == "__main__":
    raise SystemExit(main())
