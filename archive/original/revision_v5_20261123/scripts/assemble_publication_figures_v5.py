#!/usr/bin/env python3
"""Assemble the verified six-figure v5 publication bundle without editing results.

The inputs are already-bound figure-spec JSON files. This script only checks
their identities and source-file hashes, then copies the figure objects into a
single plotter-ready document. It performs no analysis or numerical transforms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


EXPECTED_SECTIONS = {
    "calibration": {"1", "5", "6"},
    "controls": {"2", "3", "4", "S2"},
    "sensitivity": {"S1", "S3", "S4"},
}
FIGURE_ORDER = ("1", "2", "3", "4", "5", "6", "S1", "S2", "S3", "S4")
SHA256_LENGTH = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_spec(path_arg: Path, section: str,
              source_hashes: dict[Path, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path_arg.resolve(strict=True)
    require(path.is_file(), f"{section} spec is not a file: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {section} spec JSON: {path}") from exc
    require(isinstance(document, dict), f"{section} spec must be a JSON object")
    require(document.get("status") == "analysis_inputs_verified",
            f"{section} specs are not marked analysis_inputs_verified")
    figures = document.get("figures")
    require(isinstance(figures, list) and figures,
            f"{section} spec has no figures list")

    found: set[str] = set()
    for index, figure in enumerate(figures):
        label = f"{section} figure index {index}"
        require(isinstance(figure, dict), f"{label} is not an object")
        number = figure.get("number")
        require(isinstance(number, (str, int)) and not isinstance(number, bool),
                f"{label} has no valid figure number")
        figure_id = str(number)
        require(figure_id not in found, f"duplicate figure id {figure_id} in {section} specs")
        found.add(figure_id)
        require(isinstance(figure.get("claim"), str) and figure["claim"].strip(),
                f"{label} has no claim")
        require(isinstance(figure.get("uncertainty_scope"), str)
                and figure["uncertainty_scope"].strip(),
                f"{label} has no uncertainty scope")
        sources = figure.get("source_files")
        require(isinstance(sources, list) and sources,
                f"{label} has no source file bindings")
        for source_index, source in enumerate(sources):
            source_label = f"{label} source {source_index}"
            require(isinstance(source, dict), f"{source_label} is not an object")
            source_path_raw, expected_hash = source.get("path"), source.get("sha256")
            require(isinstance(source_path_raw, str) and source_path_raw.strip(),
                    f"{source_label} has no path")
            require(isinstance(expected_hash, str) and len(expected_hash) == SHA256_LENGTH
                    and all(char in "0123456789abcdef" for char in expected_hash),
                    f"{source_label} has no valid SHA256")
            source_path = Path(source_path_raw).resolve(strict=True)
            require(source_path.is_file(), f"{source_label} path is not a file: {source_path}")
            actual_hash = source_hashes.get(source_path)
            if actual_hash is None:
                actual_hash = sha256_file(source_path)
                source_hashes[source_path] = actual_hash
            require(actual_hash == expected_hash,
                    f"{source_label} SHA256 mismatch: {source_path}")

    require(found == EXPECTED_SECTIONS[section],
            f"{section} figure ids must be {sorted(EXPECTED_SECTIONS[section])}; found {sorted(found)}")
    return document, {"path": str(path), "sha256": sha256_file(path),
                      "figure_ids": sorted(found)}


def assemble(calibration_path: Path, controls_path: Path,
             sensitivity_path: Path) -> dict[str, Any]:
    source_hashes: dict[Path, str] = {}
    calibration, calibration_info = read_spec(calibration_path, "calibration", source_hashes)
    controls, controls_info = read_spec(controls_path, "controls", source_hashes)
    sensitivity, sensitivity_info = read_spec(sensitivity_path, "sensitivity", source_hashes)

    # Keep each figure-spec object unchanged; only put them in manuscript order.
    figures_by_id: dict[str, dict[str, Any]] = {}
    for document in (calibration, controls, sensitivity):
        for figure in document["figures"]:
            figure_id = str(figure["number"])
            require(figure_id not in figures_by_id,
                    f"duplicate figure id across sections: {figure_id}")
            figures_by_id[figure_id] = figure
    require(set(figures_by_id) == set(FIGURE_ORDER),
            f"final figure ids must be {list(FIGURE_ORDER)}")

    verified_sources = [
        {"path": str(path), "sha256": digest}
        for path, digest in sorted(source_hashes.items(), key=lambda item: str(item[0]))
    ]
    return {
        "status": "analysis_inputs_verified",
        "bundle_scope": "six main publication figures plus supplements S1-S4; source-bound specs merged without numerical changes",
        "input_verification": {
            "calibration_specs": calibration_info,
            "controls_specs": controls_info,
            "sensitivity_specs": sensitivity_info,
            "verified_source_file_count": len(verified_sources),
            "verified_source_files": verified_sources,
        },
        "figures": [figures_by_id[figure_id] for figure_id in FIGURE_ORDER],
    }


def write_json_atomic(path_arg: Path, document: dict[str, Any]) -> Path:
    path = path_arg.resolve()
    require(not path.exists(), f"refusing to overwrite existing bundle: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard-link install is atomic and refuses an output created concurrently.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-spec", required=True, type=Path,
                        help="existing verified Figures 1/5/6 specs (use the r2 revision)")
    parser.add_argument("--controls-spec", required=True, type=Path,
                        help="verified controls builder output with Figures 2/3/4 and S2")
    parser.add_argument("--sensitivity-spec", required=True, type=Path,
                        help="verified sensitivity builder output with S1/S3/S4")
    parser.add_argument("--output", required=True, type=Path,
                        help="new plotter-ready publication bundle JSON")
    args = parser.parse_args()
    bundle = assemble(args.calibration_spec, args.controls_spec, args.sensitivity_spec)
    output = write_json_atomic(args.output, bundle)
    print(json.dumps({"status": bundle["status"],
                      "figures": [str(figure["number"]) for figure in bundle["figures"]],
                      "verified_source_file_count": bundle["input_verification"]["verified_source_file_count"],
                      "output": str(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
