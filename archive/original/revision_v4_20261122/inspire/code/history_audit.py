#!/usr/bin/env python3
"""Search project history for INSPIRE handling without exposing result values."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
OUT = Path("/data/admin/fengshuo/ehr_fm_20261102/revision_v4_20261122/inspire/audit/history_audit.json")
SEARCH_ROOTS = [
    ROOT / "protocol.md",
    ROOT / "code",
    ROOT / "docs",
    ROOT / "logs",
    ROOT / "results",
    ROOT / "revision_v2_20261117",
    ROOT / "revision_v3_20261121",
]
TEXT_SUFFIXES = {".py", ".r", ".sh", ".sql", ".md", ".txt", ".json", ".log", ".yaml", ".yml"}
TERMS = {
    "inspire": b"inspire",
    "orstart_time": b"orstart_time",
    "icuin_time": b"icuin_time",
    "ward_vitals": b"ward_vitals",
    "operations_csv": b"operations.csv",
    "preprocess": b"preprocess",
    "split": b"split",
    "checkpoint": b"checkpoint",
    "training": b"training",
}
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_FILES = 4000


def collect_files() -> list[Path]:
    found: list[Path] = []
    for root in SEARCH_ROOTS:
        if root.is_file():
            found.append(root)
            continue
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if len(found) >= MAX_FILES:
                return found
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            if any(part.startswith(".") for part in path.relative_to(ROOT).parts):
                continue
            try:
                if path.stat().st_size <= MAX_FILE_BYTES:
                    found.append(path)
            except OSError:
                continue
    return found


def scan(path: Path) -> dict[str, object] | None:
    counts = {key: 0 for key in TERMS}
    try:
        with path.open("rb") as handle:
            for line in handle:
                lowered = line.lower()
                for key, term in TERMS.items():
                    counts[key] += lowered.count(term)
        stat = path.stat()
    except (OSError, ValueError):
        return None
    if not any(counts.values()):
        return None
    return {
        "path_relative_to_project": str(path.relative_to(ROOT)),
        "size_bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
        "keyword_occurrences": counts,
    }


def main() -> None:
    files = collect_files()
    with ThreadPoolExecutor(max_workers=8) as pool:
        hits = [hit for hit in pool.map(scan, files) if hit is not None]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "search_roots": [str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path) for path in SEARCH_ROOTS],
        "text_file_count_scanned": len(files),
        "hit_file_count": len(hits),
        "max_files_reached": len(files) >= MAX_FILES,
        "max_file_size_bytes": MAX_FILE_BYTES,
        "keyword_counts_by_file": hits,
        "data_policy": "No patient rows, metric values, or matched line contents are emitted.",
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"text_file_count_scanned": len(files), "hit_file_count": len(hits), "output": str(OUT)}))


if __name__ == "__main__":
    main()
