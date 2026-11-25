"""Run the paired AUROC/AP supplement from a checked source snapshot.

The isolated CPU entry points deliberately ignore CWD and PYTHONPATH. This
wrapper loads the project modules from the adjacent versioned snapshot and
verifies every Python file against the pre-launch SHA256 inventory first.
"""
from __future__ import annotations

import hashlib
import os
import runpy
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    snapshot = Path(__file__).resolve().parent
    revision = Path("/data/admin/fengshuo/ehr_fm_20261102/revision_v4_20261122")
    inventory = snapshot / "source_sha256sums.txt"
    if not inventory.is_file():
        raise FileNotFoundError(f"snapshot hash inventory is absent: {inventory}")

    expected: dict[Path, str] = {}
    for line in inventory.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(None, 1)
        path = (snapshot / relative.strip().lstrip("*")).resolve()
        expected[path] = digest
    actual_paths = {path.resolve() for path in snapshot.rglob("*.py")}
    if actual_paths != set(expected):
        raise RuntimeError("snapshot Python file set differs from its hash inventory")
    for path, digest in expected.items():
        if _sha256(path) != digest:
            raise RuntimeError(f"snapshot source SHA256 mismatch: {path}")

    if not sys.argv[1:]:
        raise SystemExit("usage: entry.py {paired_ci|preflight|attest_eicu} [arguments]")
    task, *arguments = sys.argv[1:]
    scripts = {
        "paired_ci": snapshot / "robustness" / "paired_auc_ap_ci.py",
        "preflight": snapshot / "robustness" / "paired_auc_ap_preflight.py",
        "attest_eicu": snapshot / "robustness" / "attest_eicu_draw_hashes_r2.py",
    }
    if task not in scripts:
        raise SystemExit(f"unknown task {task!r}; choose {', '.join(scripts)}")
    script = scripts[task]
    if not script.is_file():
        raise FileNotFoundError(f"snapshot task script is absent: {script}")
    import_paths = [
        snapshot / "robustness",
        snapshot / "baselines" / "transport_v4_2",
        snapshot / "baselines",
    ]
    sys.path[:0] = [str(path) for path in import_paths]
    os.chdir(revision)
    print(
        "PAIR_CI_SOURCE_SNAPSHOT_SHA256=" + _sha256(inventory),
        flush=True,
    )
    sys.argv = [str(script), *arguments]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
