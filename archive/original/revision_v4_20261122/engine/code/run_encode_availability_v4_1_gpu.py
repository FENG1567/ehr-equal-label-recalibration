#!/usr/bin/env python3
"""Encode the two reviewed MIMIC H24 sensitivity components, without scoring."""
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
INS20_SNAPSHOT = SNAPSHOTS / "embedding_manifest_inspire_v4_1_20row_1b100d82660d.json"
ENGINE_SCRIPT = CODE / "v4_engine_registry_maskfix_recovered_v4_2_resume_vocabadapter_21aae5dd.py"
REGISTRY_PATH = REV / "data_registry.json"
GATE = REV / "temporal/results/eligibility/availability_testonly_reviewed_v4_1"
GATE_MANIFEST = GATE / "eligibility_manifest.json"
REVIEW = REV / "admin/availability_controls_scope_record_20261123.md"
IDENTITY_ATTESTATIONS = REV / "admin/subject_identity_attestations_v4_2.json"
COHORTS = (
    "mimic_available_by24_v4_1",
    "mimic_available_by24_fixedselection_v4_1",
)
MODELS = ("main20k",)
CONDITIONS = ("real", "constant", "common8_mask", "no_h0_therapy")
REGISTRY_PATH_FIELDS = ("sequence_path", "split_path", "tabular_path", "vocab_path")
EXPECTED_ENCODER_SHA256 = "59c3f4059662fb9e1d81baf20120c8668a761a926f5cab5ff4e654f7c16c1abc"
EXPECTED_ENGINE_SHA256 = "21aae5dd6b013a6fc9a4943c50799607e30690c611e322cb5799bfa21a8e95da"
EXPECTED_INS20_SHA256 = "1b100d82660dc56bdcdfc0512387d174605368305e5a79e5ba3d4c314000ca2d"
EXPECTED_GATE_SHA256 = "4a8c73898cf770a98ece8a6bc6d79185f1b8dfb82c70fd90c751fbb17001e9ea"
EXPECTED_REVIEW_SHA256 = "503bcede649006f21c13097b178bf18c721c1657b6ec53764bd8da0d4cd1a13c"
EXPECTED_IDENTITY_SHA256 = "199cdf9dfeb277ef49c30f4a4648a6b06bbf1a65958cd1cf84b36f986bb9cd55"
EXPECTED_SPLIT_SHA256 = "ed5bd94933d22daaa4cd5fdff82ce48b36b72ea50d2db7e4642906036c043c1a"
EXPECTED_SOURCE_HASHES = {
    "mimic_available_by24_v4_1": "078446262d0bc5f424f203c7fd2f195e0e4108917840c122ec88c7e51ebcc67b",
    "mimic_available_by24_fixedselection_v4_1": "d25d245fbaadc2c8ce0068afab87b72e95e151293308c2945791ef09c4c9ccd1",
}
EXPECTED_SELECTION_BY24 = {
    "mimic_available_by24_v4_1": True,
    "mimic_available_by24_fixedselection_v4_1": False,
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


def expected_component_keys() -> set[tuple[str, str, str]]:
    return {(db, "main20k", condition) for db in COHORTS for condition in CONDITIONS}


def validate_registry_paths(db: str, central: dict, fragment: dict) -> None:
    if any(central.get(field) != fragment.get(field) for field in REGISTRY_PATH_FIELDS):
        raise ValueError(f"{db}: central and owned registry paths disagree")


def validate_gate(gate: dict, identity: dict) -> None:
    if gate.get("gate_version") != GATE.name:
        raise ValueError("availability gate version changed")
    if gate.get("scope_decision") != "TEST_ONLY":
        raise ValueError("availability gate no longer records the test-only decision")
    if gate.get("score_only_authorization") is not True:
        raise ValueError("availability gate no longer authorizes score-only held-out use")
    if gate.get("analysis_role") != "same_original_heldout_mimic_test_only_score_sensitivity":
        raise ValueError("availability analysis role changed")
    if gate.get("clinical_input_availability_verified") is not False:
        raise ValueError("clinical input availability must remain unverified")
    if gate.get("ready_for_future24h_clinical_claim") is not False:
        raise ValueError("availability gate must keep future clinical claims closed")
    if gate.get("scope_record_sha256") != EXPECTED_REVIEW_SHA256:
        raise ValueError("availability gate review binding changed")
    gate_cohorts = gate.get("cohorts", {})
    identity_cohorts = identity.get("cohorts", {})
    if not set(COHORTS).issubset(gate_cohorts) or not set(COHORTS).issubset(identity_cohorts):
        raise ValueError("availability gate or identity sidecar is missing a cohort")
    for db in COHORTS:
        entry = gate_cohorts[db]
        if entry.get("status") != "reviewed_ready":
            raise ValueError(f"{db}: availability table is not reviewed_ready")
        if entry.get("score_only_authorization") is not True:
            raise ValueError(f"{db}: score-only authorization is not explicit")
        if entry.get("same_original_heldout_patients") is not True:
            raise ValueError(f"{db}: original held-out patients are not preserved")
        if entry.get("source_fit_test_subjects_disjoint_verified") is not True:
            raise ValueError(f"{db}: source-fit/test separation is not verified")
        if entry.get("clinical_input_availability_verified") is not False:
            raise ValueError(f"{db}: clinical availability must remain unverified")
        if entry.get("ready_for_future24h_clinical_claim") is not False:
            raise ValueError(f"{db}: future clinical claim must remain closed")
        if entry.get("independent_external_cohort") is not False:
            raise ValueError(f"{db}: derivative MIMIC data cannot be labeled external")
        if entry.get("selection_by24_verified") is not EXPECTED_SELECTION_BY24[db]:
            raise ValueError(f"{db}: selection-by-H24 status changed")
        if entry.get("n_rows") != 94343 or entry.get("n_original_heldout_split2_patients") != 10848:
            raise ValueError(f"{db}: original held-out roster counts changed")
        if entry.get("sourcehashes", {}).get("split_npy") != EXPECTED_SPLIT_SHA256:
            raise ValueError(f"{db}: split hash differs from the reviewed original patient split")
        if entry.get("table_sha256") != entry.get("sourcehashes", {}).get("eligibility_table"):
            raise ValueError(f"{db}: eligibility table provenance is inconsistent")

        attestation = identity_cohorts[db]
        if attestation.get("eligibility_manifest_sha256") != EXPECTED_GATE_SHA256:
            raise ValueError(f"{db}: patient identity attestation uses a different gate")
        if attestation.get("eligibility_table_sha256") != entry.get("table_sha256"):
            raise ValueError(f"{db}: patient identity attestation uses a different table")
        if attestation.get("scope_record_sha256") != EXPECTED_REVIEW_SHA256:
            raise ValueError(f"{db}: patient identity attestation uses a different review")
        if attestation.get("sequence_npz_sha256") != entry.get("sourcehashes", {}).get("sequence_npz"):
            raise ValueError(f"{db}: patient identity attestation uses a different sequence")
        basis = attestation.get("subject_identity_basis", {})
        if (basis.get("resampling_unit") != "patient"
                or basis.get("identifier_scope") != "within this original data source"
                or basis.get("patient_disjoint_claim_allowed") is not True):
            raise ValueError(f"{db}: within-source patient resampling attestation is incomplete")
        if basis.get("cross_source_patient_independence_verified") is not False:
            raise ValueError(f"{db}: cross-source patient independence must remain unverified")
        if basis.get("cross_scope_patient_independence_verified") is not False:
            raise ValueError(f"{db}: cross-scope patient independence must remain unverified")


def validate_component_manifest(manifest: dict, gate: dict) -> list[dict]:
    rows = manifest.get("models", [])
    observed = {(row.get("db"), row.get("model_tag"), row.get("input_condition"))
                for row in rows}
    expected = expected_component_keys()
    if observed != expected or len(rows) != len(expected):
        raise ValueError(f"availability component key mismatch: missing={sorted(expected-observed)}, "
                         f"unexpected={sorted(observed-expected)}")
    if manifest.get("mode") != "encode":
        raise ValueError("availability component is not an encoder manifest")
    if manifest.get("encoder_code_sha256") != EXPECTED_ENCODER_SHA256:
        raise ValueError("availability component encoder hash changed")
    if manifest.get("engine_code_sha256") != EXPECTED_ENGINE_SHA256:
        raise ValueError("availability component engine hash changed")
    if manifest.get("encoding_device") != "cuda":
        raise ValueError("availability component was not encoded on CUDA")
    if manifest.get("clinical_input_availability_verified") is not False:
        raise ValueError("availability component cannot verify real-time availability")

    expected_sources = {db: gate["cohorts"][db]["sourcehashes"]["sequence_npz"] for db in COHORTS}
    expected_splits = {db: gate["cohorts"][db]["sourcehashes"]["split_npy"] for db in COHORTS}
    if manifest.get("source_npz_sha256_by_cohort") != expected_sources:
        raise ValueError("availability manifest source NPZ hashes differ from the reviewed gate")
    if manifest.get("split_sha256_by_cohort") != expected_splits:
        raise ValueError("availability manifest split hashes differ from the reviewed gate")
    runtimes = manifest.get("runtime_identity_by_model_tag", {})
    if set(runtimes) != set(MODELS) or any(
            item.get("runtime_identity", {}).get("encoding_device") != "cuda"
            for item in runtimes.values()):
        raise ValueError("availability component runtime identity is incomplete or not CUDA")

    cache_root = (ENGINE / "cache").resolve()
    for row in rows:
        if row.get("status") not in (None, "complete") or int(row.get("n_rows", 0)) <= 0:
            raise ValueError(f"availability cache row is incomplete: {row}")
        if row.get("source_npz_sha256") != expected_sources[row["db"]]:
            raise ValueError(f"{row['db']}: cache manifest source hash differs from gate")
        if row.get("split_sha256") != expected_splits[row["db"]]:
            raise ValueError(f"{row['db']}: cache manifest split hash differs from gate")
        cache = Path(row.get("cache", "")).resolve(strict=True)
        if not cache.is_relative_to(cache_root):
            raise ValueError(f"availability cache path escapes engine/cache: {cache}")
    return rows


def validate_condition_preflight(engine: dict) -> dict:
    if tuple(engine["model_conditions"]("main20k")) != CONDITIONS:
        raise ValueError("integrated engine main20k conditions differ from the authorized 4-SPEC set")
    common_ids, _ = engine["load_common_ids"]()
    if len(common_ids) != 8 or len(set(common_ids)) != 8:
        raise ValueError("common8 vocabulary did not resolve to eight distinct token IDs")
    no_h0_by_cohort = engine["load_no_h0_therapy_ids_registered"](COHORTS)
    expected_no_h0 = {"vaso": 46, "rrt": 47, "sedation": 48}
    if any(no_h0_by_cohort.get(db) != expected_no_h0 for db in COHORTS):
        raise ValueError(f"registered no-H0 token mapping differs from frozen MIMIC mapping: {no_h0_by_cohort}")

    selected_by_cohort: dict[str, dict[str, int]] = {}
    for db in COHORTS:
        paths = engine["resolve_cohort_paths"](db, verify_exists=True)
        ds = engine["load_sequence_registered"](db, paths)
        engine["validate_sequence"](ds, db)
        selected_by_cohort[db] = {}
        for condition in CONDITIONS:
            mask_ids = (
                tuple(common_ids) if condition == "common8_mask" else
                tuple(no_h0_by_cohort[db][name] for name in ("vaso", "rrt", "sedation"))
                if condition == "no_h0_therapy" else ()
            )
            rows = engine["condition_rows"](ds, condition, mask_ids)
            if not len(rows):
                raise ValueError(f"{db}/{condition}: no rows survive the frozen input mask")
            sample_rows = rows[:128]
            prepared = engine["prepare_batch"](ds, sample_rows, condition, mask_ids)
            if (len(prepared) != 3 or any(array.ndim != 2 for array in prepared)
                    or any(array.shape != prepared[0].shape for array in prepared[1:])
                    or len(prepared[0]) != len(sample_rows)):
                raise ValueError(f"{db}/{condition}: input mask produced invalid V/B/H batch shapes")
            selected_by_cohort[db][condition] = int(len(rows))
        del ds
    return {
        "common8_variable_ids": list(common_ids),
        "no_h0_variable_ids_by_cohort": no_h0_by_cohort,
        "selected_rows_by_cohort_condition": selected_by_cohort,
    }


def main() -> None:
    preflight_only = sys.argv[1:] == ["--preflight-only"]
    finalize_existing = sys.argv[1:] == ["--finalize-existing"]
    if sys.argv[1:] and not (preflight_only or finalize_existing):
        raise ValueError("supported standalone options are --preflight-only and --finalize-existing")

    fragments = {
        db: REV / f"availability_repair/dataset/{db}/registry_fragment.json"
        for db in COHORTS
    }
    required = (LIVE_MANIFEST, INS20_SNAPSHOT, ENGINE_SCRIPT, REGISTRY_PATH,
                GATE_MANIFEST, REVIEW, IDENTITY_ATTESTATIONS, *fragments.values())
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing locked availability/encoder input: {missing}")
    if sha256_file(INS20_SNAPSHOT) != EXPECTED_INS20_SHA256:
        raise ValueError("frozen INSPIRE 20-row manifest SHA256 mismatch")
    if not finalize_existing and (sha256_file(LIVE_MANIFEST) != EXPECTED_INS20_SHA256
                                  or LIVE_MANIFEST.read_bytes() != INS20_SNAPSHOT.read_bytes()):
        raise ValueError("live manifest is not the frozen INSPIRE 20-row component")
    if sha256_file(ENGINE_SCRIPT) != EXPECTED_ENGINE_SHA256:
        raise ValueError("immutable engine snapshot SHA256 mismatch")
    if sha256_file(GATE_MANIFEST) != EXPECTED_GATE_SHA256:
        raise ValueError("availability eligibility manifest SHA256 mismatch")
    if sha256_file(REVIEW) != EXPECTED_REVIEW_SHA256:
        raise ValueError("independent availability review SHA256 mismatch")
    if sha256_file(IDENTITY_ATTESTATIONS) != EXPECTED_IDENTITY_SHA256:
        raise ValueError("root-accepted subject identity sidecar SHA256 mismatch")

    gate = json.loads(GATE_MANIFEST.read_text(encoding="utf-8"))
    identity = json.loads(IDENTITY_ATTESTATIONS.read_text(encoding="utf-8"))
    validate_gate(gate, identity)
    registry_payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    central = registry_payload.get("cohorts", registry_payload)
    for db, fragment_path in fragments.items():
        if db not in central:
            raise ValueError(f"{db}: central registry entry is missing")
        fragment = json.loads(fragment_path.read_text(encoding="utf-8"))
        validate_registry_paths(db, central[db], fragment)
        gate_entry = gate["cohorts"][db]
        dataset_manifest_path = Path(fragment["manifest_path"])
        if sha256_file(dataset_manifest_path) != gate_entry.get("sourcehashes", {}).get("dataset_manifest"):
            raise ValueError(f"{db}: dataset manifest hash differs from the reviewed gate")
        dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
        output_hashes = dataset_manifest.get("output_hashes", {})
        for key, path_key in (("sequence_npz", "sequence_path"), ("split_npy", "split_path")):
            expected = gate_entry.get("sourcehashes", {}).get(key)
            if not expected or sha256_file(Path(central[db][path_key])) != expected:
                raise ValueError(f"{db}: registered {path_key} hash differs from reviewed gate")
        gate_hashes = gate_entry.get("sourcehashes", {})
        registry_file_hashes = {
            "sequence_path": gate_hashes.get("sequence_npz"),
            "split_path": gate_hashes.get("split_npy"),
            "tabular_path": gate_hashes.get("tabular_parquet"),
            "vocab_path": gate_hashes.get("vocab_json"),
        }
        for path_key in REGISTRY_PATH_FIELDS:
            artifact_path = Path(central[db][path_key])
            actual_hash = sha256_file(artifact_path)
            gate_hash = registry_file_hashes[path_key]
            if not gate_hash or actual_hash != gate_hash:
                raise ValueError(f"{db}: reviewed gate does not bind {path_key} to its current file")
            dataset_output_hash = output_hashes.get(artifact_path.name, {}).get("sha256")
            if path_key != "vocab_path" and (
                    not dataset_output_hash or actual_hash != dataset_output_hash):
                raise ValueError(f"{db}: dataset manifest does not bind {path_key} to its current file")
        if (fragment.get("clinical_input_availability_verified") is not False
                or fragment.get("retrospective_landmark_prediction_allowed") is not False):
            raise ValueError(f"{db}: owned registry fragment widened availability claims")
        if gate_entry.get("sourcehashes", {}).get("source_registry_fragment") != sha256_file(fragment_path):
            raise ValueError(f"{db}: availability gate registry-fragment hash mismatch")
        table_path = Path(gate_entry["eligibility_table_path"])
        if sha256_file(table_path) != gate_entry.get("table_sha256"):
            raise ValueError(f"{db}: reviewed eligibility table hash mismatch")
        if gate_entry.get("sourcehashes", {}).get("sequence_npz") != EXPECTED_SOURCE_HASHES[db]:
            raise ValueError(f"{db}: source NPZ differs from the frozen component expectation")

        cache_matches = list((ENGINE / "cache").glob(f"emb_{db}_*.npz"))
        if cache_matches and not finalize_existing:
            raise FileExistsError(f"found prior {db} cache files; inspect before encoding")

    sys.path.insert(0, str(CODE))
    engine = runpy.run_path(str(ENGINE_SCRIPT), run_name="availability_encoder_preflight")
    if engine["encoder_code_hash"]() != EXPECTED_ENCODER_SHA256:
        raise ValueError("current encoder identity differs from frozen caches")
    condition_preflight = validate_condition_preflight(engine)
    if preflight_only:
        print(json.dumps({
            "status": "availability_encoder_preflight_passed",
            "gate_manifest_sha256": EXPECTED_GATE_SHA256,
            "review_sha256": EXPECTED_REVIEW_SHA256,
            "identity_attestations_sha256": EXPECTED_IDENTITY_SHA256,
            "frozen_ins20_sha256": EXPECTED_INS20_SHA256,
            "encoder_code_sha256": EXPECTED_ENCODER_SHA256,
            "cohorts": list(COHORTS),
            "cache_absence_verified": True,
            "condition_preflight": condition_preflight,
            "prediction_or_performance_emitted": False,
        }, sort_keys=True), flush=True)
        print("AVAILABILITY_ENCODER_PREFLIGHT_COMPLETE", flush=True)
        return

    if not finalize_existing:
        sys.argv = [str(ENGINE_SCRIPT), "--mode", "encode",
                    "--models", *MODELS, "--cohorts", *COHORTS,
                    "--device", "cuda", "--batch-size", "128"]
        engine_runner = runpy.run_path(str(ENGINE_SCRIPT), run_name="availability_encode_run")
        if engine_runner["main"]() != 0:
            raise RuntimeError("integrated engine returned a nonzero availability encode status")

    manifest = json.loads(LIVE_MANIFEST.read_text(encoding="utf-8"))
    rows = validate_component_manifest(manifest, gate)
    manifest_sha = sha256_file(LIVE_MANIFEST)
    snapshot = SNAPSHOTS / f"embedding_manifest_availability_v4_1_8row_{manifest_sha[:12]}.json"
    atomic_copy(LIVE_MANIFEST, snapshot)
    print(json.dumps({
        "status": "availability_embedding_component_complete",
        "manifest": str(snapshot),
        "manifest_sha256": manifest_sha,
        "rows": len(rows),
        "cohorts": list(COHORTS),
        "model_tags": list(MODELS),
        "encoder_code_sha256": manifest["encoder_code_sha256"],
        "gate_manifest_sha256": EXPECTED_GATE_SHA256,
        "frozen_ins20_sha256": EXPECTED_INS20_SHA256,
        "clinical_input_availability_verified": False,
        "predictive_performance_emitted": False,
    }, sort_keys=True), flush=True)
    print("AVAILABILITY_8ROW_ENCODING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
