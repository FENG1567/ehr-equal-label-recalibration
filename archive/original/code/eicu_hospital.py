"""Compatibility entrypoint for corrected patient-disjoint eICU hospital analysis."""
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
V2_CODE = ROOT / "revision_v2_20261117" / "code"
sys.path.insert(0, str(V2_CODE))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    import calibration_v2
    cache_path = ROOT / "revision_v2_20261117/cache/preds_main20k_v2.npz"
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"{cache_path} is missing; run fresh_baseline_v2.py before hospital evaluation."
        )
    cache = np.load(cache_path, allow_pickle=False)
    data = calibration_v2.load_aligned("eicu", cache)
    calibration_v2.hospitals(data)
    print("eICU hospital results written under revision_v2_20261117/results/tables", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
