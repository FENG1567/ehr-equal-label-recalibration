#!/usr/bin/env python3
"""Encode the locked INSPIRE v4.1 component without changing frozen inputs."""
from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
import tempfile
from pathlib import Path


ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
ENGINE = REV / "engine"
CODE = ENGINE / "code"
SNAPSHOTS = ENGINE / "code_snapshots"
LIVE_MANIFEST = ENGINE / "results/embedding_manifest.json"
FROZEN_41 = SNAPSHOTS / "embedding_manifest_recovered_v4_2_merged41_b2eec1f7aca1.json"
ENGINE_SCRIPT = CODE / "v4_engine_registry_maskfix_recovered_v4_2_resume_reusefix_a725994d9d5d.py"
REGISTRY_PATH = REV / "data_registry.json"
INSPIRE_FRAGMENT = REV / "inspire/normalized/v4/registry_inspire_v4.json"
GATE = REV / "temporal/results/eligibility/inspire_exploratory_locked_v4_1"
GATE_MANIFEST = GATE / "eligibility_manifest.json"
ASSESSMENT_LOCK = GATE / "assessment_lock.json"
DENSITY_LOCK = REV / "density_repair/run_source_20261123_03/freeze_lock.json"
DATA_REVIEW = REV / "admin/inspire_scope_record_20261123.md"
COHORTS = (
    "inspire_postop_v4_1",
    "inspire_broad_surgery_associated_v4_1",
)
MODELS = ("main20k", "abl_const20k", "common20k", "big50m", "big50m_full")
EXPECTED_ENCODER_SHA256 = "59c3f4059662fb9e1d81baf20120c8668a761a926f5cab5ff4e654f7c16c1abc"
EXPECTED_ENGINE_SHA256 = "a725994d9d5dd33d8a2b23ba42afb029c98c5f530658ddda20e76f259a4b9528"
EXPECTED_41_SHA256 = "b2eec1f7aca1857bda440ddb5ec48e477b2a20f753d5dea28b7715f7a1a243fe"
EXPECTED_DENSITY_LOCK_SHA256 = "fc69bcbe2e6433fe5d6e055e4e288eba9634857eec642822aaf9d5d2b1da5f1d"
EXPECTED_DATA_REVIEW_SHA256 = "adcd041bb10b9f8683e9eae574b4eabf68ab3d3f8e468dc8b958fe3902a329ed"
EXPECTED_COUNT_BY_COHORT = {
    "inspire_postop_v4_1": {"adaptation_patients": 500, "adaptation_events": 21,
                            "assessment_patients": 3358, "assessment_events": 106,
                            "assessment_nonevents": 3252},
    "inspire_broad_surgery_associated_v4_1": {
        "adaptation_patients": 500, "adaptation_events": 24,
        "assessment_patients": 5213, "assessment_events": 186,
        "assessment_nonevents": 5027,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest snapshot: {destination}")
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp",
                                     dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(source.read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    required = (LIVE_MANIFEST, FROZEN_41, ENGINE_SCRIPT, REGISTRY_PATH,
                INSPIRE_FRAGMENT, GATE_MANIFEST, ASSESSMENT_LOCK,
                DENSITY_LOCK, DATA_REVIEW)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing locked INSPIRE/encoder input: {missing}")

    if sha256_file(FROZEN_41) != EXPECTED_41_SHA256:
        raise ValueError("frozen 41-row embedding snapshot SHA256 mismatch")
    if sha256_file(LIVE_MANIFEST) != EXPECTED_41_SHA256:
        raise ValueError("live embedding manifest is not the frozen 41-row component")
    if LIVE_MANIFEST.read_bytes() != FROZEN_41.read_bytes():
        raise ValueError("live embedding manifest differs from the frozen 41-row snapshot")
    if sha256_file(ENGINE_SCRIPT) != EXPECTED_ENGINE_SHA256:
        raise ValueError("immutable engine snapshot SHA256 mismatch")
    if sha256_file(DENSITY_LOCK) != EXPECTED_DENSITY_LOCK_SHA256:
        raise ValueError("source-only method lock SHA256 mismatch")
    if sha256_file(DATA_REVIEW) != EXPECTED_DATA_REVIEW_SHA256:
        raise ValueError("independent INSPIRE data review SHA256 mismatch")

    frozen_manifest = json.loads(FROZEN_41.read_text(encoding="utf-8"))
    frozen_rows = frozen_manifest.get("models", [])
    if len(frozen_rows) != 41:
        raise ValueError(f"frozen embedding manifest row count is {len(frozen_rows)}, expected 41")
    gate = json.loads(GATE_MANIFEST.read_text(encoding="utf-8"))
    role_lock = json.loads(ASSESSMENT_LOCK.read_text(encoding="utf-8"))
    if (gate.get("gate_version") != GATE.name
            or gate.get("analysis_role") != "exploratory_external_only"
            or gate.get("source_development_lock_sha256") != EXPECTED_DENSITY_LOCK_SHA256
            or gate.get("independent_data_review_sha256") != EXPECTED_DATA_REVIEW_SHA256):
        raise ValueError("locked INSPIRE gate provenance is incomplete or mismatched")
    if role_lock.get("no_model_performance_used_for_role_assignment") is not True:
        raise ValueError("INSPIRE assessment roles are not performance-blind")
    if role_lock.get("no_redraw_or_threshold_relaxation") is not True:
        raise ValueError("INSPIRE assessment roles permit redraw or threshold relaxation")
    if role_lock.get("metric_population") != "fixed assessment only for all model/calibration comparisons":
        raise ValueError("INSPIRE assessment-only metric population is not locked")
    if not set(COHORTS).issubset(gate.get("cohorts", {})):
        raise ValueError("locked INSPIRE gate is missing one of the versioned cohorts")
    for db in COHORTS:
        entry = gate["cohorts"][db]
        if entry.get("confirmation_allowed") is not False:
            raise ValueError(f"{db}: exploratory gate incorrectly permits confirmation")
        if entry.get("patient_disjoint_claim_allowed") is not False:
            raise ValueError(f"{db}: patient-disjoint claim is not blocked")
        if entry.get("fixed_patient_role_counts") != EXPECTED_COUNT_BY_COHORT[db]:
            raise ValueError(f"{db}: frozen adaptation/assessment counts changed")
        if entry.get("pragmatic_confirmation_count_gate_met") is not False:
            raise ValueError(f"{db}: confirmation count gate is unexpectedly marked met")

    central_registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    central = central_registry.get("cohorts", central_registry)
    fragment_payload = json.loads(INSPIRE_FRAGMENT.read_text(encoding="utf-8"))
    fragment = fragment_payload.get("cohorts", fragment_payload)
    if not set(COHORTS).issubset(central) or not set(COHORTS).issubset(fragment):
        raise ValueError("INSPIRE cohorts are not registered centrally and in the owned fragment")
    for db in COHORTS:
        if central[db] != fragment[db]:
            raise ValueError(f"{db}: central and INSPIRE-owned registry paths disagree")
        table_path = GATE / f"{db}.parquet"
        if sha256_file(table_path) != gate["cohorts"][db].get("table_sha256"):
            raise ValueError(f"{db}: locked eligibility table SHA256 mismatch")
        if gate["cohorts"][db].get("sourcehashes", {}).get("sequence_npz") != sha256_file(Path(central[db]["sequence_path"])):
            raise ValueError(f"{db}: locked eligibility source NPZ does not match registry data")
        cache_matches = list((ENGINE / "cache").glob(f"emb_{db}_*.npz"))
        if cache_matches:
            raise FileExistsError(
                f"found prior INSPIRE embedding caches for {db}; inspect before running: "
                f"{[str(path) for path in cache_matches[:5]]}")

    sys.path.insert(0, str(CODE))
    engine = runpy.run_path(str(ENGINE_SCRIPT), run_name="inspire_encode_preflight")
    if engine["encoder_code_hash"]() != EXPECTED_ENCODER_SHA256:
        raise ValueError("current INSPIRE encoder identity differs from frozen caches")

    sys.argv = [str(ENGINE_SCRIPT), "--mode", "encode",
                "--models", *MODELS, "--cohorts", *COHORTS,
                "--device", "cuda", "--batch-size", "128"]
    runpy.run_path(str(ENGINE_SCRIPT), run_name="__main__")

    manifest = json.loads(LIVE_MANIFEST.read_text(encoding="utf-8"))
    rows = manifest.get("models", [])
    expected_keys = {
        (db, model_tag, condition)
        for db in COHORTS
        for model_tag, condition in (
            ("main20k", "real"), ("main20k", "constant"),
            ("main20k", "common8_mask"), ("main20k", "no_h0_therapy"),
            ("abl_const20k", "real"), ("abl_const20k", "constant"),
            ("common20k", "common8_mask"), ("big50m", "real"),
            ("big50m_full", "real"), ("big50m_full", "common8_mask"),
        )
    }
    observed_keys = {(row.get("db"), row.get("model_tag"), row.get("input_condition"))
                     for row in rows}
    if observed_keys != expected_keys or len(rows) != 20:
        raise ValueError(f"INSPIRE encoding manifest has an unexpected key set: "
                         f"missing={sorted(expected_keys-observed_keys)}, "
                         f"unexpected={sorted(observed_keys-expected_keys)}")
    if manifest.get("encoder_code_sha256") != EXPECTED_ENCODER_SHA256:
        raise ValueError("INSPIRE component encoder SHA256 does not match the frozen encoder")
    if manifest.get("encoding_device") != "cuda":
        raise ValueError("INSPIRE component was not encoded on the approved isolated GPU")
    incomplete = [row for row in rows if not row.get("cache") or row.get("status") not in (None, "complete")]
    if incomplete:
        raise ValueError(f"INSPIRE component contains incomplete cache records: {incomplete}")
    for row in rows:
        if not Path(row["cache"]).is_file():
            raise FileNotFoundError(f"encoded cache is missing: {row['cache']}")

    output_sha = sha256_file(LIVE_MANIFEST)
    snapshot = SNAPSHOTS / f"embedding_manifest_inspire_v4_1_20row_{output_sha[:12]}.json"
    atomic_copy(LIVE_MANIFEST, snapshot)
    print(json.dumps({
        "status": "inspire_embedding_component_complete",
        "manifest": str(snapshot),
        "manifest_sha256": output_sha,
        "rows": len(rows),
        "cohorts": list(COHORTS),
        "model_tags": list(MODELS),
        "encoder_code_sha256": manifest["encoder_code_sha256"],
        "gate_manifest_sha256": sha256_file(GATE_MANIFEST),
        "assessment_lock_sha256": sha256_file(ASSESSMENT_LOCK),
        "frozen_41_sha256": EXPECTED_41_SHA256,
        "clinical_input_availability_verified": False,
        "predictive_performance_emitted": False,
    }, sort_keys=True), flush=True)
    print("INSPIRE_20ROW_ENCODING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
