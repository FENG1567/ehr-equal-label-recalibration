"""Run corrected v2 evaluation into revision_v2_20261117 without rebuilding MOTOR caches.

This replaces the legacy S0-S9 copier, which recomputed the old baseline and
reused the affected v1 decoder, calibration, and MOTOR tables. All outputs
from this entrypoint stay in the v2 revision directory; original v1 results
are left intact.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
V2 = ROOT / "revision_v2_20261117"
V2_CODE = V2 / "code"
sys.path.insert(0, str(V2_CODE))

import calibration_v2
import decoding_motor_v2
import fresh_baseline_v2


def preflight():
    required = [
        ROOT / "models/main20k.pt",
        ROOT / "models/abl_const20k.pt",
        ROOT / "results/tables/final_table1.csv",
        ROOT / "data/external/eicu_hospital.csv",
    ]
    for db in decoding_motor_v2.DBS:
        required.extend([
            ROOT / f"data/processed/seq_{db}.npz",
            ROOT / f"data/processed/split_{db}.npy",
        ])
    missing = [str(path) for path in required if not path.is_file()]
    missing_motor = []
    for tag, (_, paths) in decoding_motor_v2.MOTOR_SPECS.items():
        if not paths["real"].is_file():
            missing_motor.append(f"{tag}: {paths['real']}")
    if missing or missing_motor:
        details = []
        if missing:
            details.append("missing v2 evaluation inputs: " + ", ".join(missing))
        if missing_motor:
            details.append("missing existing MOTOR real representations (not rebuilt here): " + "; ".join(missing_motor))
        raise FileNotFoundError("\n".join(details))
    for tag, (_, paths) in decoding_motor_v2.MOTOR_SPECS.items():
        present = [name for name, path in paths.items() if path.is_file()]
        absent = [name for name, path in paths.items() if not path.is_file()]
        print(f"MOTOR {tag}: existing conditions={','.join(present)}; missing={','.join(absent) or 'none'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate inputs and show the v2 route without running analysis")
    parser.add_argument("--repeats", type=int, default=200, help="patient-disjoint calibration repetitions")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    preflight()
    if args.check:
        print("Route: fresh_baseline_v2 -> decoding_motor_v2 --mode all -> calibration_v2")
        print(f"Corrected outputs: {V2 / 'results/tables'}")
        print("MOTOR representations are read from existing caches; no cache builder is called.")
        return 0

    print("=== corrected v2 baseline ===", flush=True)
    fresh_baseline_v2.main()
    print("=== corrected decoder, constant probe, and existing-cache MOTOR evaluation ===", flush=True)
    decoding_motor_v2.main(["--mode", "all"])
    print("=== corrected patient-disjoint calibration and hospital evaluation ===", flush=True)
    calibration_v2.main(["--repeats", str(args.repeats)])
    print(f"V2_FINAL_PIPELINE_COMPLETE: {V2 / 'results/tables'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
