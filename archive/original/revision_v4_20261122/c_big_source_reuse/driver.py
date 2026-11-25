#!/usr/bin/env python3
"""Immutable component merges and frozen-source target prediction reuse.

This driver imports the pinned, read-only source-reuse runner and extends only
its manifest input and cohort-group configuration. Source fitting remains
forbidden by the imported runner.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

REV = Path("/data/admin/fengshuo/ehr_fm_20261102/revision_v4_20261122")
OWN = REV / "c_big_source_reuse"
ENGINE = REV / "engine"
RUNNER = ENGINE / "code/run_target_source_probe_reuse_merged69.py"
RUNNER_SHA = "0b9cf4b844519cd363066dc63ff2a8f59d7ddc64cae029fcb2dd3b2293baa013"
BASELINE = ENGINE / "code_snapshots/embedding_manifest_recovered_v4_2_merged69_10fa455935a8.json"
BASELINE_SHA = "10fa455935a87d296562cbc1b00c9b61563b514ba3dc72667647973a65db02fa"
ENCODER_SHA = "59c3f4059662fb9e1d81baf20120c8668a761a926f5cab5ff4e654f7c16c1abc"
ENGINE_SHA = "21aae5dd6b013a6fc9a4943c50799607e30690c611e322cb5799bfa21a8e95da"
SOURCE_ORIGIN_CORE = "status_consistent_primary_core_fix2"
SOURCE_ORIGIN_BIG = "big_readout_frozen20_v4_2"

COMPONENTS = {
    "c21": {
        "cohorts": ("c2012_a_v4_1", "c2012_b_v4_1", "c2012_c_v4_1"),
        "gate": REV / "temporal/results/eligibility/c2012_external_reviewed_v4_2",
        "gate_sha": "165f8467a511fd6c6f4228d4c57a93aa3ec3fa6876776c482eea68cba9b2649b",
        "models": ("main20k", "abl_const20k", "common20k"),
        "origin": SOURCE_ORIGIN_CORE,
        "run_id": "c21_source_reuse_merged99_20261123",
        "row_count": 21,
    },
    "big9": {
        "cohorts": ("nwicu_discharge_v4_1", "sicdb_zero_offset_v4_2", "eicu_semantic_v4_2"),
        "gate": REV / "temporal/results/eligibility/recovered_reviewed_v4_2",
        "gate_sha": "d302ce0a493121f9336f40fd58c5dc63a75a663492293c0153b4ba05efbb8e8f",
        "models": ("big50m", "big50m_full"),
        "origin": SOURCE_ORIGIN_BIG,
        "run_id": "big9_source_reuse_merged99_20261123",
        "row_count": 9,
    },
}
_MODEL_CONDITIONS: dict[str, tuple[str, ...]] | None = None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def keyset(rows: list[dict]) -> set[tuple[str, str, str]]:
    keys = {(str(row.get("db")), str(row.get("model_tag")),
             str(row.get("input_condition"))) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("duplicate (db, model_tag, input_condition) manifest row")
    return keys


def expected_component_keys(component: str) -> set[tuple[str, str, str]]:
    global _MODEL_CONDITIONS
    config = COMPONENTS[component]
    if _MODEL_CONDITIONS is None:
        engine_script = ENGINE / "code/v4_engine_registry_maskfix_recovered_v4_2_resume_vocabadapter_21aae5dd.py"
        if sha256_file(engine_script) != ENGINE_SHA:
            raise ValueError("frozen engine source SHA256 mismatch while reading model conditions")
        engine_code = str(ENGINE / "code")
        if engine_code not in sys.path:
            sys.path.insert(0, engine_code)
        engine_namespace = runpy.run_path(str(engine_script), run_name="c_big_model_condition_schema")
        _MODEL_CONDITIONS = {
            tag: tuple(engine_namespace["model_conditions"](tag))
            for item in COMPONENTS.values() for tag in item["models"]
        }
    return {(db, model, condition)
            for db in config["cohorts"]
            for model in config["models"]
            for condition in _MODEL_CONDITIONS[model]}


def atomic_immutable(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o444)


def merge_component(component: str, component_manifest: Path,
                    expected_component_sha: str) -> Path:
    config = COMPONENTS[component]
    if sha256_file(BASELINE) != BASELINE_SHA:
        raise ValueError("frozen merged69 baseline SHA256 mismatch")
    if sha256_file(RUNNER) != RUNNER_SHA:
        raise ValueError("read-only source-reuse runner SHA256 mismatch")
    if sha256_file(component_manifest) != expected_component_sha:
        raise ValueError(f"{component} component manifest SHA256 mismatch")
    baseline = read_json(BASELINE)
    frozen = read_json(component_manifest)
    baseline_rows = baseline.get("models", [])
    component_rows = frozen.get("models", [])
    if len(baseline_rows) != 69 or len(component_rows) != config["row_count"]:
        raise ValueError("baseline/component manifest row count mismatch")
    if baseline.get("encoder_code_sha256") != ENCODER_SHA:
        raise ValueError("baseline functional encoder identity mismatch")
    if (frozen.get("mode") != "encode" or frozen.get("clinical_ready") is not False
            or frozen.get("encoder_code_sha256") != ENCODER_SHA
            or frozen.get("engine_code_sha256") != ENGINE_SHA
            or frozen.get("encoding_device") != "cuda"):
        raise ValueError("component is not the expected frozen encoder-only manifest")

    baseline_keys = keyset(baseline_rows)
    component_keys = keyset(component_rows)
    expected = expected_component_keys(component)
    if component_keys != expected:
        missing = sorted(expected - component_keys)
        extra = sorted(component_keys - expected)
        raise ValueError(f"{component} component key grid mismatch; missing={missing}; extra={extra}")
    if baseline_keys & component_keys:
        raise ValueError(f"{component} component overlaps immutable merged69 baseline")

    merged = dict(baseline)
    merged["models"] = baseline_rows + component_rows
    merged["source_probe_reuse_manifest_merge"] = {
        "status": "immutable_external_component_union",
        "purpose": "frozen_source_probe_reuse_only",
        "base_manifest_path": str(BASELINE),
        "base_manifest_sha256": BASELINE_SHA,
        "component": component,
        "component_manifest_path": str(component_manifest),
        "component_manifest_sha256": expected_component_sha,
        "component_row_count": len(component_rows),
        "merged_row_count": len(merged["models"]),
        "overlapping_keys": 0,
        "clinical_input_availability_verified": False,
        "prediction_or_source_head_fit_run": False,
    }
    if keyset(merged["models"]) != baseline_keys | component_keys:
        raise ValueError("merged manifest key integrity check failed")
    raw = (json.dumps(merged, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    row_total = len(merged["models"])
    out = OWN / f"embedding_manifest_merged{row_total}_{component}_{digest[:12]}.json"
    if out.exists():
        if sha256_file(out) != digest or out.read_bytes() != raw:
            raise FileExistsError(f"refusing overwrite or path collision: {out}")
    else:
        atomic_immutable(out, raw)
    audit = {
        "status": "immutable_external_component_union",
        "component": component,
        "merged_manifest_path": str(out),
        "merged_manifest_sha256": digest,
        "base_manifest_path": str(BASELINE),
        "base_manifest_sha256": BASELINE_SHA,
        "component_manifest_path": str(component_manifest),
        "component_manifest_sha256": expected_component_sha,
        "base_row_count": len(baseline_rows),
        "component_row_count": len(component_rows),
        "merged_row_count": row_total,
        "base_key_count": len(baseline_keys),
        "component_key_count": len(component_keys),
        "overlapping_keys": 0,
        "expected_component_grid_exact": True,
        "clinical_input_availability_verified": False,
        "prediction_or_source_head_fit_run": False,
    }
    audit_raw = (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode()
    audit_path = OWN / f"merge_{component}_{digest[:12]}_audit.json"
    if audit_path.exists():
        if audit_path.read_bytes() != audit_raw:
            raise FileExistsError(f"refusing overwrite or path collision: {audit_path}")
    else:
        atomic_immutable(audit_path, audit_raw)
    print(f"MERGED_MANIFEST={out}", flush=True)
    print(f"MERGED_MANIFEST_SHA256={digest}", flush=True)
    print(f"MERGED_ROWS={row_total} BASE_ROWS=69 COMPONENT_ROWS={len(component_rows)}", flush=True)
    print(f"MERGE_AUDIT={audit_path}", flush=True)
    return out


def load_frozen_component(component: str, component_manifest: Path,
                          expected_component_sha: str) -> tuple[dict, list[dict], set]:
    config = COMPONENTS[component]
    if sha256_file(component_manifest) != expected_component_sha:
        raise ValueError(f"{component} component manifest SHA256 mismatch")
    frozen = read_json(component_manifest)
    rows = frozen.get("models", [])
    if len(rows) != config["row_count"]:
        raise ValueError(f"{component} component row count mismatch")
    if (frozen.get("mode") != "encode" or frozen.get("clinical_ready") is not False
            or frozen.get("encoder_code_sha256") != ENCODER_SHA
            or frozen.get("engine_code_sha256") != ENGINE_SHA
            or frozen.get("encoding_device") != "cuda"):
        raise ValueError(f"{component} is not a frozen encoder-only manifest")
    keys = keyset(rows)
    expected = expected_component_keys(component)
    if keys != expected:
        raise ValueError(f"{component} component grid does not match its frozen target grid")
    return frozen, rows, keys


def merge_union(c21_manifest: Path, c21_sha: str,
                big9_manifest: Path, big9_sha: str) -> Path:
    if sha256_file(BASELINE) != BASELINE_SHA:
        raise ValueError("frozen merged69 baseline SHA256 mismatch")
    if sha256_file(RUNNER) != RUNNER_SHA:
        raise ValueError("read-only source-reuse runner SHA256 mismatch")
    baseline = read_json(BASELINE)
    if (len(baseline.get("models", [])) != 69
            or baseline.get("encoder_code_sha256") != ENCODER_SHA):
        raise ValueError("frozen merged69 baseline identity or row count mismatch")
    _, c_rows, c_keys = load_frozen_component("c21", c21_manifest, c21_sha)
    _, b_rows, b_keys = load_frozen_component("big9", big9_manifest, big9_sha)
    base_rows = baseline["models"]
    base_keys = keyset(base_rows)
    if base_keys & (c_keys | b_keys) or c_keys & b_keys:
        raise ValueError("component union overlaps the frozen baseline or another component")
    expected = base_keys | expected_component_keys("c21") | expected_component_keys("big9")
    rows = base_rows + c_rows + b_rows
    if keyset(rows) != expected or len(rows) != 99:
        raise ValueError("merged99 manifest does not contain the exact 69+21+9 key union")
    merged = dict(baseline)
    merged["models"] = rows
    merged["source_probe_reuse_manifest_merge"] = {
        "status": "immutable_external_component_union",
        "purpose": "frozen_source_probe_reuse_only",
        "base_manifest_path": str(BASELINE),
        "base_manifest_sha256": BASELINE_SHA,
        "components": {
            "c21": {"manifest_path": str(c21_manifest), "manifest_sha256": c21_sha,
                    "row_count": len(c_rows)},
            "big9": {"manifest_path": str(big9_manifest), "manifest_sha256": big9_sha,
                     "row_count": len(b_rows)},
        },
        "merged_row_count": len(rows),
        "overlapping_keys": 0,
        "clinical_input_availability_verified": False,
        "prediction_or_source_head_fit_run": False,
    }
    raw = (json.dumps(merged, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    out = OWN / f"embedding_manifest_merged99_c21_big9_{digest[:12]}.json"
    if out.exists():
        if sha256_file(out) != digest or out.read_bytes() != raw:
            raise FileExistsError(f"refusing overwrite or path collision: {out}")
    else:
        atomic_immutable(out, raw)
    audit = {
        "status": "immutable_external_component_union",
        "merged_manifest_path": str(out),
        "merged_manifest_sha256": digest,
        "base_manifest_path": str(BASELINE),
        "base_manifest_sha256": BASELINE_SHA,
        "components": merged["source_probe_reuse_manifest_merge"]["components"],
        "base_row_count": len(base_rows),
        "component_row_counts": {"c21": len(c_rows), "big9": len(b_rows)},
        "merged_row_count": len(rows),
        "base_key_count": len(base_keys),
        "component_key_counts": {"c21": len(c_keys), "big9": len(b_keys)},
        "overlapping_keys": 0,
        "expected_union_grid_exact": True,
        "clinical_input_availability_verified": False,
        "prediction_or_source_head_fit_run": False,
    }
    audit_raw = (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode()
    audit_path = OWN / f"merge_merged99_c21_big9_{digest[:12]}_audit.json"
    if audit_path.exists():
        if audit_path.read_bytes() != audit_raw:
            raise FileExistsError(f"refusing overwrite or path collision: {audit_path}")
    else:
        atomic_immutable(audit_path, audit_raw)
    print(f"MERGED_MANIFEST={out}", flush=True)
    print(f"MERGED_MANIFEST_SHA256={digest}", flush=True)
    print("MERGED_ROWS=99 BASE_ROWS=69 C21_ROWS=21 BIG9_ROWS=9 OVERLAPPING_KEYS=0", flush=True)
    print(f"MERGE_AUDIT={audit_path}", flush=True)
    return out


def import_readonly_runner():
    if sha256_file(RUNNER) != RUNNER_SHA:
        raise ValueError("read-only source-reuse runner SHA256 mismatch")
    spec = importlib.util.spec_from_file_location("readonly_source_reuse_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import pinned source-reuse runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_group(component: str, manifest_path: Path, preflight_only: bool) -> None:
    config = COMPONENTS[component]
    if sha256_file(BASELINE) != BASELINE_SHA:
        raise ValueError("frozen merged69 baseline SHA256 mismatch")
    if sha256_file(RUNNER) != RUNNER_SHA:
        raise ValueError("read-only source-reuse runner SHA256 mismatch")
    manifest = read_json(manifest_path)
    merge_meta = manifest.get("source_probe_reuse_manifest_merge", {})
    expected_union = (keyset(read_json(BASELINE)["models"])
                      | expected_component_keys("c21") | expected_component_keys("big9"))
    component_meta = merge_meta.get("components", {})
    if (len(manifest.get("models", [])) != 99
            or keyset(manifest.get("models", [])) != expected_union
            or merge_meta.get("base_manifest_sha256") != BASELINE_SHA
            or merge_meta.get("merged_row_count") != 99
            or set(component_meta) != {"c21", "big9"}
            or component_meta[component]["row_count"] != config["row_count"]):
        raise ValueError("selected immutable merged99 manifest has wrong component provenance")
    manifest_sha = sha256_file(manifest_path)
    gate_manifest = config["gate"] / "eligibility_manifest.json"
    if sha256_file(gate_manifest) != config["gate_sha"]:
        raise ValueError(f"{component} reviewed target gate SHA256 mismatch")

    runner = import_readonly_runner()
    runner.MERGED = manifest_path
    runner.MERGED_SHA = manifest_sha
    runner.SNAPSHOTS = OWN / "pinned_engine_snapshots"
    runner.GROUPS = {component: {
        "gate": config["gate"], "gate_sha": config["gate_sha"],
        "targets": config["cohorts"], "models": config["models"],
        "origin": config["origin"], "run_id": config["run_id"],
    }}
    pinned_engine = runner.pinned_engine()
    baseline_rows = read_json(BASELINE)["models"]
    original_run_path = runner.runpy.run_path
    engine_namespace_seen = {"value": False}

    def patched_run_path(path, *args, **kwargs):
        namespace = original_run_path(path, *args, **kwargs)
        try:
            is_pinned = Path(path).resolve() == pinned_engine.resolve()
        except (TypeError, OSError):
            is_pinned = False
        if is_pinned and not engine_namespace_seen["value"]:
            engine_namespace_seen["value"] = True
            original_loader = namespace["load_encoding_manifest"]
            first_load = {"value": True}

            def exact_component_manifest_loader(*loader_args, **loader_kwargs):
                loaded_manifest, entries = original_loader(*loader_args, **loader_kwargs)
                if first_load["value"]:
                    first_load["value"] = False
                    rows = loaded_manifest.get("models", [])
                    if len(rows) != 99:
                        raise ValueError("runtime merged manifest row count changed")
                    runtime_expected = (keyset(baseline_rows)
                                        | expected_component_keys("c21")
                                        | expected_component_keys("big9"))
                    if keyset(rows) != runtime_expected:
                        raise ValueError("runtime merged manifest does not contain exact merged99 key union")
                    legacy_guard_view = dict(loaded_manifest)
                    legacy_guard_view["models"] = baseline_rows
                    return legacy_guard_view, entries
                return loaded_manifest, entries

            namespace["load_encoding_manifest"] = exact_component_manifest_loader
        return namespace

    runner.runpy.run_path = patched_run_path
    try:
        runner.run(component, preflight_only)
    finally:
        runner.runpy.run_path = original_run_path
    if not engine_namespace_seen["value"]:
        raise RuntimeError("pinned engine was not loaded by the read-only runner")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-component", choices=tuple(COMPONENTS))
    parser.add_argument("--component-manifest", type=Path)
    parser.add_argument("--component-sha256")
    parser.add_argument("--prepare-union", action="store_true")
    parser.add_argument("--c21-manifest", type=Path)
    parser.add_argument("--c21-sha256")
    parser.add_argument("--big9-manifest", type=Path)
    parser.add_argument("--big9-sha256")
    parser.add_argument("--group", choices=tuple(COMPONENTS))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_union:
        if args.group or args.manifest or args.prepare_component or args.preflight_only:
            parser.error("--prepare-union cannot be combined with component/group prediction options")
        if not all((args.c21_manifest, args.c21_sha256, args.big9_manifest, args.big9_sha256)):
            parser.error("--prepare-union requires both frozen component manifests and SHA256 values")
        merge_union(args.c21_manifest, args.c21_sha256,
                    args.big9_manifest, args.big9_sha256)
        return
    if args.prepare_component:
        if args.group or args.manifest or args.preflight_only:
            parser.error("--prepare-component cannot be combined with --group/--manifest")
        if args.component_manifest is None or args.component_sha256 is None:
            parser.error("--prepare-component requires --component-manifest and --component-sha256")
        merge_component(args.prepare_component, args.component_manifest, args.component_sha256)
        return
    if not args.group or not args.manifest or args.component_manifest or args.component_sha256:
        parser.error("prediction invocation requires only --group and --manifest")
    run_group(args.group, args.manifest, args.preflight_only)


if __name__ == "__main__":
    main()
