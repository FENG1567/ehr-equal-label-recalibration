"""Strict v4.2 embedding-cache attestation for v5 representation analyses.

The v4 merged manifest does not carry per-cache byte hashes. Those hashes are
bound through its frozen merge audit. This validator authenticates the audit
against the pinned v4 code/release evidence, then checks the exact cache bytes.
It attests the cache archive; it does not authenticate historical pretraining
membership.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV4 = ROOT / "revision_v4_20261122"
EXPECTED_MERGED_MANIFEST_SHA256 = (
    "b2eec1f7aca1857bda440ddb5ec48e477b2a20f753d5dea28b7715f7a1a243fe"
)
EXPECTED_MERGE_AUDIT_SHA256 = (
    "ba792d7c600c134bbaa79deebba123183217b4bf474a5102c7100ac65f09740d"
)
EXPECTED_BIG_READOUT_SOURCE_SHA256 = (
    "3ed5d90ae466fb55347877f56db3644803628df2f9498a6cb8c74f9028e9ece0"
)
EXPECTED_NUMERIC_RELEASE_AUDIT_SCRIPT_SHA256 = (
    "73bae834897be9a2bd726abf734d6b4e4f6f475f7931544eeafa123d18c856ad"
)
EXPECTED_NUMERIC_RELEASE_AUDIT_JSON_SHA256 = (
    "28c849b82cf538985e483b51378efb0e51a1212415e0427cd719f6513c1a0b5c"
)
MERGE_AUDIT_PATH = (
    REV4 / "engine/code_snapshots/embedding_manifest_merge_audit_recovered_v4_2_b2eec1f7aca1.json"
)
BIG_READOUT_SOURCE_PATH = REV4 / "big_readout/run_big_readout_frozen20.py"
NUMERIC_RELEASE_AUDIT_SCRIPT_PATH = (
    REV4 / "decoding_repaired/release_audit/audit_numeric_decoding_release.py"
)
NUMERIC_RELEASE_AUDIT_JSON_PATH = (
    REV4 / "decoding_repaired/release_audit/numeric_decoding_release_audit.json"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_mask_hash(condition: str, variable_ids: tuple[int, ...] = ()) -> str:
    payload = json.dumps(
        {"condition": str(condition), "variable_ids": [int(x) for x in variable_ids]},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_path(value: Any) -> str:
    return str(Path(str(value)).resolve())


def _validate_anchor_sources() -> dict[str, str]:
    paths_and_hashes = (
        (BIG_READOUT_SOURCE_PATH, EXPECTED_BIG_READOUT_SOURCE_SHA256),
        (NUMERIC_RELEASE_AUDIT_SCRIPT_PATH, EXPECTED_NUMERIC_RELEASE_AUDIT_SCRIPT_SHA256),
        (NUMERIC_RELEASE_AUDIT_JSON_PATH, EXPECTED_NUMERIC_RELEASE_AUDIT_JSON_SHA256),
    )
    observed = {}
    for path, expected in paths_and_hashes:
        actual = sha256_file(path)
        if actual.lower() != expected.lower():
            raise ValueError(f"v4 cache-audit anchor source changed: {path}")
        observed[str(path)] = actual
    release_audit = json.loads(NUMERIC_RELEASE_AUDIT_JSON_PATH.read_text(encoding="utf-8"))
    if release_audit.get("status") != "PASS":
        raise ValueError("v4 numeric release audit is not a passing attestation")
    if release_audit.get("audit_script_sha256") != observed[str(NUMERIC_RELEASE_AUDIT_SCRIPT_PATH)]:
        raise ValueError("v4 numeric release-audit JSON does not bind its audit script bytes")
    trusted = release_audit.get("trusted_artifacts", {})
    if (trusted.get("embedding_manifest_sha256") != EXPECTED_MERGED_MANIFEST_SHA256
            or trusted.get("cache_audit_manifest_sha256") != EXPECTED_MERGE_AUDIT_SHA256):
        raise ValueError("v4 numeric release audit does not bind the expected manifest and cache audit")
    big_source = BIG_READOUT_SOURCE_PATH.read_text(encoding="utf-8")
    if EXPECTED_MERGE_AUDIT_SHA256 not in big_source:
        raise ValueError("frozen v4 source readout does not pin the expected merge-audit digest")
    return observed


def validate_cache_entry(merged_manifest_path: str | Path,
                         entry: Mapping[str, Any], db: str) -> dict[str, Any]:
    """Validate one merged-manifest entry against the frozen v4 audit and bytes."""
    merged_manifest_path = Path(merged_manifest_path)
    manifest_sha = sha256_file(merged_manifest_path)
    if manifest_sha.lower() != EXPECTED_MERGED_MANIFEST_SHA256:
        raise ValueError("frozen merged embedding-manifest SHA256 differs")
    manifest = json.loads(merged_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("mode") != "encode" or len(manifest.get("models", [])) != 41:
        raise ValueError("expected the frozen 41-row merged encoding manifest")

    model_tag = str(entry.get("model_tag", ""))
    condition = str(entry.get("input_condition", ""))
    if not model_tag or not condition or str(entry.get("db", "")) != str(db):
        raise ValueError(f"{db}: incomplete or mismatched merged-manifest cache entry")
    rows = [row for row in manifest["models"]
            if str(row.get("db")) == str(db)
            and str(row.get("model_tag")) == model_tag
            and str(row.get("input_condition")) == condition]
    if len(rows) != 1:
        raise ValueError(f"{db}/{model_tag}/{condition}: expected one unique merged-manifest row")
    frozen_entry = rows[0]
    for field in ("cache", "checkpoint_sha256", "source_npz_sha256", "split_sha256",
                  "encoder_code_sha256", "row_identity_sha256", "n_rows",
                  "input_mask_sha256"):
        if str(entry.get(field)) != str(frozen_entry.get(field)):
            raise ValueError(f"{db}/{model_tag}/{condition}: loader entry differs in {field}")

    mask_spec = frozen_entry.get("input_mask_spec")
    variable_ids: tuple[int, ...] = ()
    if isinstance(mask_spec, dict):
        variable_ids = tuple(int(x) for x in mask_spec.get("variable_ids", ()))
    expected_mask_hash = input_mask_hash(condition, variable_ids)
    if str(frozen_entry.get("input_mask_sha256", "")).lower() != expected_mask_hash:
        raise ValueError(f"{db}/{model_tag}/{condition}: condition-specific input-mask hash differs")

    anchor_hashes = _validate_anchor_sources()
    audit_sha = sha256_file(MERGE_AUDIT_PATH)
    if audit_sha.lower() != EXPECTED_MERGE_AUDIT_SHA256:
        raise ValueError("frozen v4 cache merge-audit SHA256 differs from its pinned release/code anchors")
    audit = json.loads(MERGE_AUDIT_PATH.read_text(encoding="utf-8"))
    if (audit.get("merged_manifest_sha256") != manifest_sha
            or int(audit.get("merged_row_count", -1)) != 41
            or int(audit.get("cache_file_sha256_checks", -1)) != 41):
        raise ValueError("frozen v4 cache audit is not bound to the expected 41-row merged manifest")

    key = [str(db), model_tag, condition]
    matches = [item for item in audit.get("cache_files", [])
               if item.get("key") == key]
    if len(matches) != 1:
        raise ValueError(f"{db}/{model_tag}/{condition}: cache-audit key is not unique")
    audited = matches[0]
    cache_path = Path(str(frozen_entry["cache"]))
    if _canonical_path(audited.get("path")) != _canonical_path(cache_path):
        raise ValueError(f"{db}/{model_tag}/{condition}: cache path differs from merge audit")
    expected_cache_sha = str(audited.get("sha256", "")).lower()
    if len(expected_cache_sha) != 64:
        raise ValueError(f"{db}/{model_tag}/{condition}: cache audit lacks a valid SHA256")
    actual_cache_sha = sha256_file(cache_path)
    if actual_cache_sha.lower() != expected_cache_sha:
        raise ValueError(f"{db}/{model_tag}/{condition}: cache bytes differ from frozen merge audit")

    return {
        "status": "verified_frozen_v4_cache_archive",
        "attestation_scope": "frozen merged-manifest/cache-audit bytes; not historical pretraining membership",
        "db": str(db), "model_tag": model_tag, "input_condition": condition,
        "key": key, "merged_manifest_sha256": manifest_sha,
        "merge_audit_path": str(MERGE_AUDIT_PATH), "merge_audit_sha256": audit_sha,
        "audit_anchor_sources": anchor_hashes,
        "cache_path": str(cache_path), "cache_sha256": actual_cache_sha,
        "row_identity_sha256": str(frozen_entry["row_identity_sha256"]),
        "input_mask_sha256": expected_mask_hash,
    }


def validate_real_constant_pair(merged_manifest_path: str | Path,
                                real_entry: Mapping[str, Any],
                                constant_entry: Mapping[str, Any],
                                db: str) -> dict[str, Any]:
    """Verify separate condition hashes and shared model/roster provenance."""
    real = validate_cache_entry(merged_manifest_path, real_entry, db)
    constant = validate_cache_entry(merged_manifest_path, constant_entry, db)
    if real["input_condition"] != "real" or constant["input_condition"] != "constant":
        raise ValueError(f"{db}: expected distinct real and constant input conditions")
    for field in ("model_tag", "merged_manifest_sha256", "row_identity_sha256"):
        if real[field] != constant[field]:
            raise ValueError(f"{db}: real and constant cache provenance differs in {field}")
    if real["input_mask_sha256"] == constant["input_mask_sha256"]:
        raise ValueError(f"{db}: real and constant condition-specific input-mask hashes unexpectedly match")
    for field in ("source_npz_sha256", "split_sha256", "checkpoint_sha256",
                  "encoder_code_sha256", "n_rows"):
        if str(real_entry.get(field)) != str(constant_entry.get(field)):
            raise ValueError(f"{db}: real and constant entries differ in shared provenance {field}")
    return {
        "status": "verified_real_constant_cache_pair",
        "same_sequence_checkpoint_encoder_ordered_identity": True,
        "condition_specific_input_mask_hashes_differ": True,
        "real": real, "constant": constant,
        "attestation_scope": real["attestation_scope"],
    }
