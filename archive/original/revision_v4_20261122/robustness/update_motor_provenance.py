"""Add source hashes and explicit provenance limits to completed MOTOR outputs.

This only hashes frozen representation/cohort inputs and updates the owned
MOTOR manifest; it does not rerun a split, fit, or metric calculation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
DEFAULT_MANIFEST = REV / "robustness/motor_repeated_split_manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _canonical(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, (bool,)):
        return "1" if value else "0"
    try:
        number = float(value)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError, OverflowError):
        pass
    return str(value).strip()


def source_label_digest(path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(path, columns=["stay_id", "subject_id", "mort_hosp"])
    for key in ("stay_id", "subject_id"):
        frame[key] = frame[key].astype("string").str.strip()
    frame["mort_hosp"] = pd.to_numeric(frame["mort_hosp"], errors="coerce")
    if frame["stay_id"].duplicated().any():
        raise ValueError(f"duplicate stay_id in source cohort {path}")
    h = hashlib.sha256()
    for row in frame.sort_values("stay_id", kind="stable").itertuples(index=False, name=None):
        for value in row:
            token = _canonical(value).encode("utf-8")
            h.update(len(token).to_bytes(8, "little"))
            h.update(token)
    return {
        "path": str(path), "file_sha256": sha256_file(path),
        "stay_subject_label_sha256": h.hexdigest(),
        "n_stays": int(len(frame)), "n_subject_identifiers": int(frame["subject_id"].nunique()),
    }


def update(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repr_files: dict[str, Any] = {}
    db_for_tag = {"mimic2k": "mimic", "mimic6k": "mimic", "eicu1k": "eicu"}
    source_cohorts = {}
    for tag, conditions in manifest.get("known_representation_paths", {}).items():
        repr_files[tag] = {}
        for condition, raw_path in conditions.items():
            path = Path(raw_path)
            repr_files[tag][condition] = {
                "path": str(path), "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        db = db_for_tag.get(tag)
        if db and db not in source_cohorts:
            cohort_path = ROOT / f"data/interim/cohort_{db}.parquet"
            source_cohorts[db] = source_label_digest(cohort_path)
    manifest["representation_provenance"] = {
        "repr_files": repr_files,
        "source_cohort_labels": source_cohorts,
        "source_encoder_checkpoint_sha256": "unknown: checkpoint/config identity is not embedded in these frozen reprs.pkl files",
        "source_encoder_version_and_training_membership": "unknown from available representation artifacts",
        "known_generation_context": [
            "The compatibility entrypoint evaluates existing cached representations; it does not regenerate the frozen representations.",
            "The legacy configuration was reviewed as using a shared model path and LOINC mapping; exact checkpoint hashes are unavailable from reprs.pkl.",
            "Constant representations use per-cohort medians, exposing the target cohort's unlabeled feature distribution in the const condition.",
        ],
        "temporal_scope": "MOTOR source/evaluation cohorts and their original row definitions; ICU 24-hour index alignment is unverified, so this is exploratory and not future-24-hour clinical validation.",
        "patient_independence": "split grouping uses cohort subject_id; source hashes cover stay_id, subject_id, and mort_hosp without exporting row data",
        "noninferiority_claim": False,
    }
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, manifest_path)
    return manifest["representation_provenance"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = ap.parse_args(argv)
    provenance = update(Path(args.manifest))
    print(json.dumps({
        "repr_files_hashed": sum(len(x) for x in provenance["repr_files"].values()),
        "source_cohorts_hashed": sorted(provenance["source_cohort_labels"]),
        "checkpoint_identity": "unknown",
        "motor_metrics_rerun": False,
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

