"""Compatibility entrypoint for patient-disjoint v2 calibration analysis."""
from pathlib import Path
import sys

V2_CODE = Path(__file__).resolve().parents[1] / "revision_v2_20261117" / "code"
sys.path.insert(0, str(V2_CODE))

import calibration_v2


if __name__ == "__main__":
    raise SystemExit(calibration_v2.main())
