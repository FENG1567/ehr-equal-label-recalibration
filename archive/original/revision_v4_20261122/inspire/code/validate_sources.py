#!/usr/bin/env python3
"""Quarantine-only integrity audit for the three required INSPIRE gzip tables."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = Path("/data/admin/fengshuo/ehr_fm_20261102/revision_v4_20261122/inspire")
INPUT = BASE / "input" / "quarantine"
OUTPUT = BASE / "audit" / "source_integrity.json"
EXPECTED_SIZES = {
    "operations.csv.gz": 6164195,
    "labs.csv.gz": 89868892,
    "ward_vitals.csv.gz": 166249451,
}
FALLBACK_SHA256 = {
    "operations.csv.gz": "0c366d4945f258ecf317261f54a36b116bb170c583b2c6c59c558711b15cebf7",
    "labs.csv.gz": "297858717e286273a14e502d5ef17bce7e633893e376625a368757c7162d66eb",
    "ward_vitals.csv.gz": "1c70aee2ae25327b54f2ddee4bb29523bae57d6c74847f949e9e8d0fc4045e14",
}


def read_supplied_sums() -> dict[str, str]:
    manifest = INPUT / "SHA256SUMS.txt"
    sums = FALLBACK_SHA256.copy()
    if not manifest.is_file():
        return sums
    for line in manifest.read_text(encoding="utf-8").splitlines():
        pieces = line.strip().split(maxsplit=1)
        if len(pieces) == 2:
            digest, name = pieces
            if name in EXPECTED_SIZES:
                sums[name] = digest.lower()
    return sums


def audit_one(name: str, expected_hash: str) -> dict[str, object]:
    path = INPUT / name
    if not path.is_file():
        return {"file": name, "state": "missing", "valid_for_ingestion": False}
    size = path.stat().st_size
    result: dict[str, object] = {
        "file": name,
        "received_bytes": size,
        "expected_source_bytes": EXPECTED_SIZES[name],
        "transfer_complete_by_size": size == EXPECTED_SIZES[name],
        "supplied_sha256": expected_hash,
        "official_checksum_reference": "local supplied SHA256SUMS.txt; official file listing requires credentialed access",
    }
    with path.open("rb") as raw:
        result["magic_hex"] = raw.read(8).hex()
        raw.seek(0)
        digest = hashlib.file_digest(raw, "sha256").hexdigest()
    result["actual_sha256"] = digest
    result["supplied_sha_match"] = digest == expected_hash
    if size != EXPECTED_SIZES[name]:
        result.update({"gzip_crc": "not_checked_incomplete_transfer", "csv_header_fields": [], "valid_for_ingestion": False, "state": "incomplete_transfer"})
        return result
    try:
        uncompressed_bytes = 0
        with gzip.open(path, "rb") as stream:
            first_line = stream.readline()
            if not first_line:
                raise ValueError("empty decompressed table")
            header = next(csv.reader([first_line.decode("utf-8-sig").rstrip("\r\n")]))
            uncompressed_bytes += len(first_line)
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                uncompressed_bytes += len(block)
        result["gzip_crc"] = "pass"
        result["uncompressed_bytes"] = uncompressed_bytes
        result["csv_header_fields"] = header
        result["valid_for_ingestion"] = bool(result["supplied_sha_match"])
        result["state"] = "checksum_and_gzip_verified" if result["valid_for_ingestion"] else "checksum_mismatch_gzip_valid_quarantined"
    except Exception as exc:  # noqa: BLE001 - provenance audit must record malformed archives.
        result["gzip_crc"] = "fail"
        result["gzip_error_type"] = type(exc).__name__
        result["gzip_error"] = str(exc)
        result["csv_header_fields"] = []
        result["valid_for_ingestion"] = False
        result["state"] = "gzip_invalid_quarantined"
    return result


def main() -> None:
    sums = read_supplied_sums()
    with ThreadPoolExecutor(max_workers=8) as pool:
        audited = list(pool.map(lambda name: audit_one(name, sums[name]), EXPECTED_SIZES))
    payload = {
        "dataset_version_claim": "INSPIRE 1.4.2 from adjacent source metadata; official file list could not be retrieved without credentialed access",
        "official_sha256_fetch": "HTTP 403 without an authorized PhysioNet session; see logs/official_SHA_fetch.log",
        "supplied_manifest_present": (INPUT / "SHA256SUMS.txt").is_file(),
        "files": audited,
        "ingestion_gate": "closed_until_all_required_tables_have_expected_size_and_supplied_checksum_and_gzip_crc_pass",
        "raw_records_emitted": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "files": [{"file": item["file"], "state": item["state"], "valid_for_ingestion": item["valid_for_ingestion"]} for item in audited]}))


if __name__ == "__main__":
    main()
