#!/usr/bin/env python3
"""Build a hashed, source-split0-only clean pretraining dataset in R5."""
import argparse
import ast
import hashlib
import json
import os
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
R4 = ROOT / "revision_v4_20261122"
R5 = ROOT / "revision_v5_20261123"
ELIG = R4 / "temporal/results/eligibility/recovered_reviewed_v4_2"
SOURCE_EVENTS = ROOT / "data/interim/events_mimic.parquet"
SOURCE_EVENTS_SCAN = Path("/dev/shm/ehr_fm_v5_inputs_20261123/events_mimic.parquet")
SOURCE_COHORT = ROOT / "data/interim/cohort_mimic.parquet"
SOURCE_SPLIT = ROOT / "data/processed/split_mimic.npy"
SOURCE_SEQ = ROOT / "data/processed/seq_mimic.npz"
SOURCE_META = ROOT / "data/processed/meta_mimic.json"
SOURCE_ELIG = ELIG / "mimic.parquet"
EICU_BASE = R4 / "datasets/eicu_semantic_v4_2"
EICU_EVENTS = EICU_BASE / "interim/events_eicu_semantic_v4_2.parquet"
EICU_COHORT = EICU_BASE / "interim/cohort_eicu_semantic_v4_2.parquet"
EICU_META = EICU_BASE / "meta_eicu_semantic_v4_2.json"
EICU_ELIG = ELIG / "eicu_semantic_v4_2.parquet"
NWICU_EVENTS = ROOT / "data/interim/events_nwicu.parquet"
NWICU_COHORT = ROOT / "data/interim/cohort_nwicu.parquet"
NWICU_META = R4 / "datasets/nwicu_discharge_v4_1/meta_nwicu.json"
NWICU_ELIG = ELIG / "nwicu_discharge_v4_1.parquet"
CODE_EXTRACT = ROOT / "code/extract.py"
CODE_BUILD = ROOT / "code/build_data.py"
CODE_FM = ROOT / "code/fm.py"
CPU_ENTRY = R5 / "scripts/v5_cpu_entry_ram.py"
LOCK_FILE = R5 / "admin/clean_pretraining_lock.md"
PROTOCOL_COPY = R5 / "admin/protocol_v5.md"
NBINS = 10
MAXLEN = 128
SAMPLE_N = 20000
SAMPLE_SEED = 20261123
TOKEN_SEED = 20261123
PARTITIONS = 64
BATCH_ROWS = 250000
RSS_ABORT_GIB = float(os.environ.get("V5_RSS_ABORT_GIB", "7.5"))
PART_SCHEMA = pa.schema([
    ("stay_id", pa.int64()), ("hour", pa.float64()),
    ("var", pa.string()), ("val", pa.float64())
])
os.umask(0o077)


def log(message):
    print(datetime.now(timezone.utc).isoformat(timespec="seconds"), message, flush=True)


def rss_gib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(value / (1024 * 1024), 3)


VERIFIED_HASHES = {
    str(SOURCE_EVENTS): {
        "bytes": 243670624,
        "mtime_ns": 1793593229925494972,
        "sha256": "8f4ec20e642eb38020f76b82c3a57a9b9d285bf21f9ad639d05c7e03c0fdb6ec",
        "verified_log": str(R5 / "logs/v5_clean_pretraining_prepare_20261123_145851.log"),
    },
    str(SOURCE_COHORT): {
        "bytes": 2059865,
        "mtime_ns": 1793593230128490791,
        "sha256": "dd1d06997e128a18475553b8e18cfd1c81b2cdd79f95e67ba1746afaea5cec2a",
        "verified_log": str(R5 / "logs/v5_clean_pretraining_prepare_20261123_145851.log"),
    },
    str(SOURCE_EVENTS_SCAN): {
        "bytes": 243670624,
        "mtime_ns": 1795417775965174083,
        "sha256": "8f4ec20e642eb38020f76b82c3a57a9b9d285bf21f9ad639d05c7e03c0fdb6ec",
        "verified_log": str(R5 / "logs/stage_direct_input_probe.log"),
    },
}


def sha256_file(path):
    path = Path(path)
    initial = path.stat()
    known = VERIFIED_HASHES.get(str(path))
    if known is not None:
        if initial.st_size == known["bytes"] and initial.st_mtime_ns == known["mtime_ns"]:
            result = {
                "path": str(path), "bytes": int(initial.st_size), "mtime_ns": int(initial.st_mtime_ns),
                "sha256": known["sha256"], "method": "reused_after_exact_size_mtime_check",
                "verified_against_log": known["verified_log"],
            }
            log("HASH_REUSE_VERIFIED path=" + str(path) + " bytes=" + str(initial.st_size) +
                " mtime_ns=" + str(initial.st_mtime_ns) + " sha256=" + known["sha256"])
            return result
        log("HASH_REUSE_MISS path=" + str(path) + " observed_bytes=" + str(initial.st_size) +
            " observed_mtime_ns=" + str(initial.st_mtime_ns))
    h = hashlib.sha256()
    n = 0
    next_report = 64 * 1024 * 1024
    last_report = time.monotonic()
    with path.open("rb") as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            h.update(block)
            n += len(block)
            now = time.monotonic()
            if n >= next_report or now - last_report >= 30:
                position = stream.tell()
                log("HASH_PROGRESS path=" + str(path) + " logical_bytes=" + str(n) +
                    " file_pos=" + str(position) + " total_bytes=" + str(initial.st_size) +
                    " rss_gib=" + str(rss_gib()))
                atomic_json(RUN_DIR / "progress.json", {
                    "stage": "hashing_input_file", "path": str(path),
                    "logical_bytes": int(n), "file_pos": int(position),
                    "total_bytes": int(initial.st_size), "mtime_ns": int(initial.st_mtime_ns),
                })
                if rss_gib() >= RSS_ABORT_GIB:
                    raise MemoryError("RSS reached the abort threshold while hashing " + str(path))
                while next_report <= n:
                    next_report += 64 * 1024 * 1024
                last_report = now
    final = path.stat()
    if n != initial.st_size or final.st_size != initial.st_size or final.st_mtime_ns != initial.st_mtime_ns:
        raise RuntimeError("file changed or byte count mismatched during hash: " + str(path))
    return {"path": str(path), "bytes": n, "mtime_ns": int(final.st_mtime_ns),
            "sha256": h.hexdigest(), "method": "stream_sha256"}

def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def assign_value(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return ast.literal_eval(node.value)
    raise RuntimeError("frozen code has no literal assignment for " + name)


def read_frozen_vocab():
    meta = json.loads(SOURCE_META.read_text())
    meta_vars = meta.get("vars")
    extract_tree = ast.parse(CODE_EXTRACT.read_text())
    build_tree = ast.parse(CODE_BUILD.read_text())
    source_vars = assign_value(extract_tree, "VARS")
    range_map = assign_value(extract_tree, "RANGE")
    binary_vars = assign_value(build_tree, "BIN_VARS")
    if not isinstance(meta_vars, list) or len(meta_vars) != 48:
        raise RuntimeError("frozen MIMIC meta must define exactly 48 variable IDs")
    if meta_vars != source_vars:
        raise RuntimeError("meta_mimic variable order differs from read-only extract.py")
    if len(set(meta_vars)) != 48 or set(meta_vars) - set(range_map):
        raise RuntimeError("frozen variable vocabulary/range map is incomplete")
    if set(binary_vars) != {"sex", "vaso", "rrt", "sedation"}:
        raise RuntimeError("read-only binary-variable policy changed")
    for path in [EICU_META, NWICU_META]:
        other_vars = json.loads(path.read_text()).get("vars")
        if other_vars != meta_vars:
            raise RuntimeError("target dataset variable order differs from frozen MIMIC meta: " + str(path))
    return meta_vars, range_map, binary_vars


def int_ids(series, label):
    values = pd.to_numeric(series, errors="raise").to_numpy(dtype=np.int64)
    if len(np.unique(values)) != len(values):
        raise RuntimeError(label + " stay_id values are not unique")
    return values


def read_role_table(path, id_col, subject_col):
    frame = pq.read_table(path, columns=[id_col, subject_col, "eligible_24h"]).to_pandas()
    frame["stay_id"] = pd.to_numeric(frame[id_col], errors="raise").astype(np.int64)
    frame["subject"] = frame[subject_col].astype(str)
    frame["eligible_24h"] = frame["eligible_24h"].fillna(False).astype(bool)
    if frame["stay_id"].duplicated().any():
        raise RuntimeError("duplicate stay IDs in reviewed eligibility table: " + str(path))
    if frame["subject"].isna().any() or (frame["subject"].str.len() == 0).any():
        raise RuntimeError("missing reviewed patient IDs in eligibility table: " + str(path))
    return frame[["stay_id", "subject", "eligible_24h"]]


def load_source_roles(resume_members_path=None, resume_members_sha256=None):
    cohort = pq.read_table(SOURCE_COHORT, columns=["stay_id", "subject_id"]).to_pandas()
    if cohort["stay_id"].duplicated().any():
        raise RuntimeError("source cohort stay IDs are not unique")
    split = np.load(SOURCE_SPLIT, allow_pickle=False)
    if len(split) != len(cohort) or not np.isin(split, [0, 1, 2]).all():
        raise RuntimeError("source split array is not aligned with source cohort")
    with np.load(SOURCE_SEQ, allow_pickle=False) as legacy:
        legacy_stays = legacy["stay_id"].astype(np.int64)
        legacy_subjects = legacy["subject"].astype(str)
        if not np.array_equal(legacy_stays, cohort["stay_id"].to_numpy(dtype=np.int64)):
            raise RuntimeError("legacy sequence/source cohort stay order mismatch")
        if not np.array_equal(legacy_subjects, cohort["subject_id"].astype(str).to_numpy()):
            raise RuntimeError("legacy sequence/source cohort patient order mismatch")
        legacy_v = legacy["V"]
        if legacy_v.shape[0] != len(cohort) or int(legacy_v.max(initial=0)) > 48:
            raise RuntimeError("legacy sequence variable IDs exceed frozen 48-variable mapping")
        legacy_var_id_max = int(legacy_v.max(initial=0))
    cohort["subject"] = cohort["subject_id"].astype(str)
    cohort["split"] = split.astype(np.int8)
    patient_split_n = cohort.groupby("subject", sort=False)["split"].nunique()
    if (patient_split_n > 1).any():
        raise RuntimeError("a source patient appears in multiple frozen splits")
    roles = read_role_table(SOURCE_ELIG, "stay_id", "subject_id")
    check = roles.merge(cohort[["stay_id", "subject", "split"]], on="stay_id", how="left",
                        validate="one_to_one", suffixes=("_elig", "_cohort"), indicator=True)
    if not (check["_merge"] == "both").all():
        raise RuntimeError("reviewed source H24 roster contains unknown source stays")
    if not (check["subject_elig"] == check["subject_cohort"]).all():
        raise RuntimeError("reviewed source subject/stay identity mismatch")
    roles = check[["stay_id", "subject_elig", "eligible_24h", "split"]].rename(
        columns={"subject_elig": "subject"})
    eligible = roles.loc[roles["eligible_24h"]].copy()
    split_counts = {str(k): int(v) for k, v in roles.loc[roles["eligible_24h"], "split"].value_counts().sort_index().items()}
    candidates = eligible.loc[eligible["split"] == 0, ["stay_id", "subject"]].sort_values("stay_id")
    if len(candidates) < SAMPLE_N:
        raise RuntimeError("only " + str(len(candidates)) + " eligible split0 source stays; required exactly 20000")
    rng = np.random.default_rng(SAMPLE_SEED)
    selected_pos = rng.choice(len(candidates), SAMPLE_N, replace=False)
    expected_members = candidates.iloc[selected_pos].sort_values("stay_id").reset_index(drop=True)
    if resume_members_path is not None:
        resume_members_path = Path(resume_members_path)
        observed_hash = sha256_file(resume_members_path)["sha256"]
        if resume_members_sha256 is not None and observed_hash != resume_members_sha256:
            raise RuntimeError("resume member checkpoint SHA-256 mismatch")
        members = pd.read_csv(resume_members_path, dtype={"subject": str})
        if list(members.columns) != ["stay_id", "subject"]:
            raise RuntimeError("resume member checkpoint has unexpected columns")
        members["stay_id"] = pd.to_numeric(members["stay_id"], errors="raise").astype(np.int64)
        members["subject"] = members["subject"].astype(str)
        members = members.sort_values("stay_id").reset_index(drop=True)
        if members["stay_id"].duplicated().any() or len(members) != SAMPLE_N:
            raise RuntimeError("resume member checkpoint has duplicate or wrong-count stays")
        if not np.array_equal(members["stay_id"].to_numpy(), expected_members["stay_id"].to_numpy()) or not np.array_equal(
                members["subject"].to_numpy(), expected_members["subject"].astype(str).to_numpy()):
            raise RuntimeError("resume member checkpoint does not match frozen eligible split0 sample")
        log("MEMBERS_CHECKPOINT_REUSED path=" + str(resume_members_path) +
            " sha256=" + observed_hash)
    else:
        members = expected_members
    members_path = RUN_PRIVATE / "source_pretrain_members_seed20261123.csv"
    members.to_csv(members_path, index=False)
    os.chmod(members_path, 0o600)
    selected = set(members["stay_id"].to_numpy(dtype=np.int64).tolist())
    eligible["pretrain_member"] = eligible["stay_id"].isin(selected).astype(np.int8)
    eligible = eligible.sort_values("stay_id").reset_index(drop=True)
    return eligible, members, split_counts, {
        "source_cohort_rows": int(len(cohort)),
        "source_split_all_stays": {str(k): int(v) for k, v in pd.Series(split).value_counts().sort_index().items()},
        "source_eligible_h24_by_split": split_counts,
        "source_split0_h24_candidate_n": int(len(candidates)),
        "pretraining_member_n": int(len(members)),
        "pretraining_member_path": str(members_path),
        "pretraining_member_sha256": sha256_file(members_path)["sha256"],
        "legacy_sequence_variable_id_max": legacy_var_id_max,
        "patient_split_unique": True,
    }


def fit_edges(range_map, binary_vars, selected_ids):
    selected_ids = np.asarray(selected_ids, dtype=np.int64)
    if len(np.unique(selected_ids)) != len(selected_ids):
        raise RuntimeError("bin-fit stay IDs are not unique")
    log("BIN_FIT_START scope=all_eligible_source_split0_h24 fit_stays=" + str(len(selected_ids)))
    value_lists = {v: [] for v in VARS}
    counts = {v: 0 for v in VARS}
    events_seen = 0
    events_candidate = 0
    events_retained = 0
    pf = pq.ParquetFile(SOURCE_EVENTS_SCAN)
    need = {"stay_id", "hour", "var", "val"}
    if not need.issubset(set(pf.schema_arrow.names)):
        raise RuntimeError("source floating event schema does not have stay_id/hour/var/val")
    for batch_no, batch in enumerate(pf.iter_batches(
            columns=["stay_id", "hour", "var", "val"], batch_size=BATCH_ROWS, use_threads=False), start=1):
        frame = batch.to_pandas()
        events_seen += len(frame)
        sid = frame["stay_id"].to_numpy(dtype=np.int64, copy=False)
        hour = frame["hour"].to_numpy(dtype=np.float64, copy=False)
        val = frame["val"].to_numpy(dtype=np.float64, copy=False)
        var = frame["var"].astype(str)
        keep = np.isin(sid, selected_ids) & np.isfinite(hour) & (hour >= 0) & (hour < 24)
        keep &= var.isin(VARS).to_numpy()
        frame = frame.loc[keep, ["var", "val"]]
        events_candidate += len(frame)
        if len(frame):
            for variable, group in frame.groupby("var", sort=False):
                lo, hi = range_map[variable]
                values = group["val"].to_numpy(dtype=np.float64, copy=False)
                good = np.isfinite(values) & (values >= lo) & (values <= hi)
                values = values[good]
                if len(values):
                    value_lists[variable].append(values.copy())
                    counts[variable] += int(len(values))
                    events_retained += int(len(values))
        if batch_no % 20 == 0:
            log("BIN_SCAN fit_stays=" + str(len(selected_ids)) + " batch=" + str(batch_no) +
                " rows_seen=" + str(events_seen) + " candidates=" + str(events_candidate) +
                " retained=" + str(events_retained) + " rss_gib=" + str(rss_gib()))
        if rss_gib() >= RSS_ABORT_GIB:
            raise MemoryError("RSS reached the abort threshold during source bin fit")
    edges = {}
    modes = {}
    for variable in VARS:
        if variable in binary_vars:
            edges[variable] = [0.5]
            modes[variable] = "binary_threshold_0.5"
        elif counts[variable] >= 200:
            vals = np.concatenate(value_lists[variable])
            quantiles = np.quantile(vals, np.linspace(0, 1, NBINS + 1)[1:-1])
            edges[variable] = np.unique(np.round(quantiles, 4)).astype(float).tolist()
            modes[variable] = "source_split0_h24_quantile_10bin"
        else:
            lo, hi = range_map[variable]
            edges[variable] = np.linspace(lo, hi, NBINS + 1)[1:-1].astype(float).tolist()
            modes[variable] = "source_count_lt_200_range_fallback"
        value_lists[variable] = []
    return edges, {
        "fit_scope": "all_eligible_source_split0_h24",
        "fit_stay_n": int(len(selected_ids)),
        "fit_rows_seen": int(events_seen),
        "fit_rows_candidate": int(events_candidate),
        "fit_rows_retained": int(events_retained),
        "per_variable_source_event_count": counts,
        "per_variable_edge_mode": modes,
    }


def digitize_values(values, variable_ids, edges):
    result = np.empty(len(values), dtype=np.int16)
    for vid in np.unique(variable_ids):
        mask = variable_ids == vid
        variable = VARS[int(vid)]
        bins = np.digitize(values[mask], edges[variable], right=False) + 1
        result[mask] = np.clip(bins, 1, NBINS)
    return result


def build_sequences(name, event_path, roster, edges, out_path):
    ids = roster["stay_id"].to_numpy(dtype=np.int64)
    subjects = roster["subject"].astype(str).to_numpy()
    if len(np.unique(ids)) != len(ids):
        raise RuntimeError(name + " roster contains duplicate stay IDs")
    id_to_pos = {int(sid): i for i, sid in enumerate(ids)}
    vtok = np.zeros((len(ids), MAXLEN), dtype=np.int16)
    btok = np.zeros((len(ids), MAXLEN), dtype=np.int16)
    htok = np.zeros((len(ids), MAXLEN), dtype=np.int16)
    lengths = np.zeros(len(ids), dtype=np.int16)
    summaries = np.full((len(ids), len(VARS)), np.nan, dtype=np.float32)
    member_flag = roster.get("pretrain_member", pd.Series(np.zeros(len(roster), dtype=np.int8))).to_numpy(dtype=np.int8)
    split_flag = roster.get("split", pd.Series(np.full(len(roster), -1, dtype=np.int8))).to_numpy(dtype=np.int8)
    rng = np.random.default_rng(TOKEN_SEED)
    partition_root = RUN_PRIVATE / "staging" / name
    partition_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(partition_root, 0o700)
    writers = {}
    input_rows = 0
    candidate_rows = 0
    kept_rows = 0
    out_of_range_rows = 0
    nonfinite_rows = 0
    unsupported_rows = 0
    pf = pq.ParquetFile(event_path)
    required = {"stay_id", "hour", "var", "val"}
    if not required.issubset(set(pf.schema_arrow.names)):
        raise RuntimeError(name + " event schema does not have stay_id/hour/var/val")
    # np.isin treats a Python set as a scalar object, which rejects every ID.
    target_ids = np.asarray(ids, dtype=np.int64)
    for batch_no, batch in enumerate(pf.iter_batches(
            columns=["stay_id", "hour", "var", "val"], batch_size=BATCH_ROWS, use_threads=False), start=1):
        frame = batch.to_pandas()
        input_rows += len(frame)
        sid = frame["stay_id"].to_numpy(dtype=np.int64, copy=False)
        hour = frame["hour"].to_numpy(dtype=np.float64, copy=False)
        value = frame["val"].to_numpy(dtype=np.float64, copy=False)
        var = frame["var"].astype(str)
        mask = np.isin(sid, target_ids) & np.isfinite(hour) & (hour >= 0) & (hour < 24)
        mask &= var.isin(VARS).to_numpy()
        unsupported_rows += int((np.isin(sid, target_ids) & np.isfinite(hour) & (hour >= 0) &
                                 (hour < 24) & ~var.isin(VARS).to_numpy()).sum())
        frame = frame.loc[mask, ["stay_id", "hour", "var", "val"]].copy()
        candidate_rows += len(frame)
        if not len(frame):
            if batch_no % 20 == 0:
                log(name + " PARTITION batch=" + str(batch_no) + " rows=" + str(input_rows) +
                    " kept=" + str(kept_rows) + " rss_gib=" + str(rss_gib()))
            continue
        vals = frame["val"].to_numpy(dtype=np.float64, copy=False)
        finite = np.isfinite(vals)
        nonfinite_rows += int((~finite).sum())
        ranges = frame["var"].map(lambda v: RANGE_MAP[v])
        lower = ranges.map(lambda x: x[0]).to_numpy(dtype=np.float64)
        upper = ranges.map(lambda x: x[1]).to_numpy(dtype=np.float64)
        in_range = finite & (vals >= lower) & (vals <= upper)
        out_of_range_rows += int((finite & ~in_range).sum())
        frame = frame.loc[in_range, ["stay_id", "hour", "var", "val"]]
        kept_rows += len(frame)
        if not len(frame):
            continue
        part_ids = np.remainder(frame["stay_id"].to_numpy(dtype=np.int64), PARTITIONS)
        for part_no in np.unique(part_ids):
            sub = frame.loc[part_ids == part_no, ["stay_id", "hour", "var", "val"]]
            table = pa.Table.from_pandas(sub, schema=PART_SCHEMA, preserve_index=False, safe=True)
            writer = writers.get(int(part_no))
            if writer is None:
                part_path = partition_root / ("part_" + str(int(part_no)).zfill(3) + ".parquet")
                writer = pq.ParquetWriter(part_path, PART_SCHEMA, compression="zstd", use_dictionary=["var"])
                writers[int(part_no)] = writer
            writer.write_table(table)
        if batch_no % 20 == 0:
            log(name + " PARTITION batch=" + str(batch_no) + " rows=" + str(input_rows) +
                " candidates=" + str(candidate_rows) + " kept=" + str(kept_rows) +
                " rss_gib=" + str(rss_gib()))
    for writer in writers.values():
        writer.close()
    if not writers:
        raise RuntimeError(name + " produced no H24 canonical event rows")
    part_summaries = []
    for part_no in sorted(writers):
        part_path = partition_root / ("part_" + str(part_no).zfill(3) + ".parquet")
        table = pq.read_table(part_path, use_threads=False)
        part_stays = np.asarray(table["stay_id"].to_numpy(), dtype=np.int64)
        part_hours = np.floor(table["hour"].to_numpy()).astype(np.int16)
        part_values = np.asarray(table["val"].to_numpy(), dtype=np.float64)
        part_vars = np.asarray([VARIABLE_TO_ID[str(v)] for v in table["var"].to_pylist()], dtype=np.int16)
        unique_stays, stay_inverse = np.unique(part_stays, return_inverse=True)
        key = (stay_inverse * 24 + part_hours) * len(VARS) + part_vars
        order = np.argsort(key, kind="stable")
        sorted_key = key[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_key)) + 1]
        counts = np.diff(np.r_[starts, len(sorted_key)])
        unique_key = sorted_key[starts]
        gval = np.add.reduceat(part_values[order], starts) / counts
        gvid = (unique_key % len(VARS)).astype(np.int16)
        ghour = ((unique_key // len(VARS)) % 24).astype(np.int16)
        gstay = unique_stays[unique_key // (24 * len(VARS))]
        gbin = digitize_values(gval, gvid, edges)
        stay_starts = np.r_[0, np.flatnonzero(np.diff(gstay)) + 1]
        stay_ends = np.r_[stay_starts[1:], len(gstay)]
        for left, right in zip(stay_starts, stay_ends):
            sid = gstay[left]
            pos = id_to_pos.get(int(sid))
            if pos is None:
                raise RuntimeError(name + " partition contains a stay outside frozen roster")
            n = int(right - left)
            vids = gvid[left:right]
            values = gval[left:right]
            counts_by_var = np.bincount(vids, minlength=len(VARS))
            sums_by_var = np.bincount(vids, weights=values, minlength=len(VARS))
            present = counts_by_var > 0
            summaries[pos, present] = (sums_by_var[present] / counts_by_var[present]).astype(np.float32)
            if n > MAXLEN:
                selected = np.sort(rng.choice(n, MAXLEN, replace=False))
                n = MAXLEN
            else:
                selected = np.arange(n)
            lengths[pos] = n
            vtok[pos, :n] = gvid[left:right][selected] + 1
            btok[pos, :n] = gbin[left:right][selected]
            htok[pos, :n] = ghour[left:right][selected]
        part_summaries.append({"partition": int(part_no), "input_rows": int(len(part_stays)),
                               "aggregated_stay_hour_var_rows": int(len(gstay)),
                               "bytes": int(part_path.stat().st_size)})
        del table, part_stays, part_hours, part_values, part_vars, order, sorted_key
        if part_no % 8 == 0:
            log(name + " AGGREGATE partition=" + str(part_no) +
                " completed=" + str(len(part_summaries)) + " rss_gib=" + str(rss_gib()))
        if rss_gib() >= RSS_ABORT_GIB:
            raise MemoryError(name + " RSS reached the 8 GiB ceiling")
    if int(lengths.max(initial=0)) > MAXLEN:
        raise RuntimeError(name + " token length exceeds fixed 128")
    active_tokens = np.arange(MAXLEN)[None, :] < lengths[:, None]
    if (btok[active_tokens] < 1).any() or (btok[active_tokens] > NBINS).any():
        raise RuntimeError(name + " binned values are outside 1..10")
    if (htok[active_tokens] < 0).any() or (htok[active_tokens] > 23).any():
        raise RuntimeError(name + " measurement hour bucket outside 0..23")
    arrays = {
        "V": vtok, "B": btok, "H": htok, "L": lengths,
        "stay_id": ids.astype(np.int64), "subject": subjects,
        "summary": summaries, "split": split_flag, "pretrain_member": member_flag,
    }
    np.savez_compressed(out_path, **arrays)
    os.chmod(out_path, 0o600)
    return {
        "input_rows": int(input_rows), "h24_canonical_candidate_rows": int(candidate_rows),
        "kept_after_range_check_rows": int(kept_rows), "nonfinite_values_dropped": int(nonfinite_rows),
        "finite_out_of_range_values_dropped": int(out_of_range_rows),
        "unsupported_vocabulary_rows": int(unsupported_rows),
        "eligible_stays": int(len(ids)), "stays_with_tokens": int((lengths > 0).sum()),
        "stays_with_zero_tokens": int((lengths == 0).sum()),
        "token_count_min": int(lengths.min()) if len(lengths) else 0,
        "token_count_max": int(lengths.max(initial=0)),
        "token_count_mean": float(lengths.mean()) if len(lengths) else 0.0,
        "tokenization_seed": TOKEN_SEED, "partition_count": PARTITIONS,
        "partition_summaries": part_summaries,
        "sequence_path": str(out_path), "sequence_sha256": sha256_file(out_path)["sha256"],
    }


def main():
    global VARS, VARIABLE_TO_ID, RANGE_MAP, BINARY_VARS, RUN_DIR, RUN_PRIVATE
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-members", type=Path)
    parser.add_argument("--resume-members-sha256")
    parser.add_argument("--resume-bins-checkpoint", type=Path)
    parser.add_argument("--resume-hash-checkpoint", type=Path)
    args = parser.parse_args()
    if args.resume_members_sha256 and not args.resume_members:
        parser.error("--resume-members-sha256 requires --resume-members")
    VARS, RANGE_MAP, BINARY_VARS = read_frozen_vocab()
    VARIABLE_TO_ID = {v: i for i, v in enumerate(VARS)}
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    RUN_DIR = R5 / "results/clean_pretraining" / ("run_" + run_id)
    RUN_PRIVATE = RUN_DIR / "private"
    RUN_PRIVATE.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(RUN_DIR, 0o700)
    os.chmod(RUN_PRIVATE, 0o700)
    log("RUN_START " + str(RUN_DIR))
    atomic_json(RUN_DIR / "progress.json", {"stage": "started", "run_id": run_id})
    input_paths = {
        "source_events": SOURCE_EVENTS, "source_events_ram_snapshot": SOURCE_EVENTS_SCAN,
        "source_cohort": SOURCE_COHORT,
        "source_split": SOURCE_SPLIT, "source_legacy_sequence_key": SOURCE_SEQ,
        "source_meta": SOURCE_META, "source_eligibility": SOURCE_ELIG,
        "eicu_repaired_events": EICU_EVENTS, "eicu_repaired_cohort": EICU_COHORT,
        "eicu_meta": EICU_META, "eicu_eligibility": EICU_ELIG,
        "nwicu_events": NWICU_EVENTS, "nwicu_cohort": NWICU_COHORT,
        "nwicu_meta": NWICU_META, "nwicu_eligibility": NWICU_ELIG,
        "reviewed_eligibility_manifest": ELIG / "eligibility_manifest.json",
    }
    code_paths = {
        "extract_py": CODE_EXTRACT, "build_data_py": CODE_BUILD, "fm_py": CODE_FM,
        "cpu_entry": CPU_ENTRY, "prepare_script": Path(__file__),
        "clean_pretraining_lock": LOCK_FILE, "protocol_copy": PROTOCOL_COPY,
    }
    file_hashes = {}
    hash_checkpoint = {}
    if args.resume_hash_checkpoint:
        hash_checkpoint_doc = json.loads(args.resume_hash_checkpoint.read_text())
        if hash_checkpoint_doc.get("stage") not in {
                "hashes_checkpoint", "hashed_inputs_and_code", "members_locked", "source_bins_locked"}:
            raise RuntimeError("resume hash checkpoint is not a recognized preparation progress file")
        hash_checkpoint = hash_checkpoint_doc.get("hashes")
        if not isinstance(hash_checkpoint, dict):
            raise RuntimeError("resume hash checkpoint has no hash mapping")
        log("HASH_CHECKPOINT_LOADED path=" + str(args.resume_hash_checkpoint) +
            " entries=" + str(len(hash_checkpoint)))
    hash_items = list({**input_paths, **code_paths}.items())
    for hash_index, (name, path) in enumerate(hash_items, start=1):
        path = Path(path)
        saved_hash = hash_checkpoint.get(name)
        reused = False
        if isinstance(saved_hash, dict) and saved_hash.get("path") == str(path):
            try:
                current_stat = path.stat()
                digest = str(saved_hash.get("sha256", ""))
                metadata_match = (
                    current_stat.st_size == int(saved_hash.get("bytes", -1))
                    and current_stat.st_mtime_ns == int(saved_hash.get("mtime_ns", -1))
                    and len(digest) == 64
                    and all(char in "0123456789abcdef" for char in digest.lower())
                )
            except (OSError, TypeError, ValueError):
                metadata_match = False
            if metadata_match:
                file_hashes[name] = {
                    **saved_hash, "method": "reused_from_progress_checkpoint",
                    "verified_against_progress": str(args.resume_hash_checkpoint),
                }
                log("HASH_REUSE_CHECKPOINT name=" + name + " bytes=" + str(current_stat.st_size) +
                    " mtime_ns=" + str(current_stat.st_mtime_ns) + " sha256=" + digest)
                reused = True
            else:
                log("HASH_CHECKPOINT_MISS name=" + name + " path=" + str(path) +
                    " reason=size_mtime_or_digest_mismatch")
        if not reused:
            log("HASH_START " + name + " " + str(path))
            file_hashes[name] = sha256_file(path)
        log("HASH_OK " + name + " bytes=" + str(file_hashes[name]["bytes"]) +
            " mtime_ns=" + str(file_hashes[name]["mtime_ns"]) +
            " method=" + file_hashes[name]["method"] +
            " sha256=" + file_hashes[name]["sha256"])
        atomic_json(RUN_DIR / "progress.json", {
            "stage": "hashes_checkpoint", "run_id": run_id,
            "completed_hash_count": hash_index, "total_hash_count": len(hash_items),
            "last_completed_hash": name, "hashes": file_hashes,
        })
    atomic_json(RUN_DIR / "progress.json", {"stage": "hashed_inputs_and_code",
                                           "run_id": run_id, "hashes": file_hashes})
    required_hashes = {"source_events", "source_events_ram_snapshot"}
    missing_hashes = sorted(required_hashes.difference(file_hashes))
    if missing_hashes:
        raise RuntimeError("required source event hashes are missing: " + ", ".join(missing_hashes))
    source_event_hash = file_hashes["source_events"]["sha256"]
    scan_event_hash = file_hashes["source_events_ram_snapshot"]["sha256"]
    if source_event_hash != scan_event_hash or file_hashes["source_events"]["bytes"] != file_hashes["source_events_ram_snapshot"]["bytes"]:
        raise RuntimeError("RAM source-event snapshot differs from the frozen original source file")
    source_roles, members, split_counts, source_summary = load_source_roles(
        args.resume_members, args.resume_members_sha256)
    if args.resume_members:
        source_summary["pretraining_member_checkpoint_source"] = str(args.resume_members)
    source_ids = set(source_roles["stay_id"].to_numpy(dtype=np.int64).tolist())
    if not set(members["stay_id"].to_numpy(dtype=np.int64).tolist()).issubset(source_ids):
        raise RuntimeError("pretraining member list is not a subset of eligible source roles")
    member_path = Path(source_summary["pretraining_member_path"])
    source_summary["pretraining_member_sha256"] = sha256_file(member_path)["sha256"]
    source_selected = members["stay_id"].to_numpy(dtype=np.int64)
    log("MEMBERS_LOCKED " + json.dumps({k: v for k, v in source_summary.items()
                                        if k != "pretraining_member_path"}) +
        " rss_gib=" + str(rss_gib()))
    atomic_json(RUN_DIR / "progress.json", {
        "stage": "members_locked", "run_id": run_id, "source": source_summary,
        "member_file": str(member_path), "hashes": file_hashes,
    })
    bin_fit_ids = source_roles.loc[source_roles["split"] == 0, "stay_id"].to_numpy(dtype=np.int64)
    expected_bin_fit_n = int(source_summary["source_eligible_h24_by_split"].get("0", 0))
    if len(bin_fit_ids) != expected_bin_fit_n or len(np.unique(bin_fit_ids)) != len(bin_fit_ids):
        raise RuntimeError("eligible split0 H24 bin-fit roster count/uniqueness mismatch")
    bin_fit_ids_sha256 = hashlib.sha256(
        np.sort(bin_fit_ids).astype("<i8", copy=False).tobytes()).hexdigest()
    checkpoint_path = RUN_DIR / "clean_source_bins_edges_checkpoint.json"
    if args.resume_bins_checkpoint:
        checkpoint = json.loads(args.resume_bins_checkpoint.read_text())
        if checkpoint.get("schema") != "v5_clean_source_only_bins_checkpoint_v1":
            raise RuntimeError("unsupported source-bin checkpoint schema")
        if checkpoint.get("fit_scope") != "all_eligible_source_split0_h24" or int(
                checkpoint.get("fit_stay_n", -1)) != len(bin_fit_ids):
            raise RuntimeError("source-bin checkpoint fit population mismatch")
        if checkpoint.get("fit_stay_ids_sha256") != bin_fit_ids_sha256:
            raise RuntimeError("source-bin checkpoint roster hash mismatch")
        if checkpoint.get("prepare_script_sha256") != file_hashes["prepare_script"]["sha256"] or checkpoint.get(
                "range_map_code_sha256") != file_hashes["extract_py"]["sha256"]:
            raise RuntimeError("source-bin checkpoint code hash mismatch")
        if checkpoint.get("source_event_sha256") != source_event_hash or checkpoint.get(
                "source_event_scan_sha256") != scan_event_hash:
            raise RuntimeError("source-bin checkpoint event hash mismatch")
        edges = checkpoint.get("edges")
        bin_summary = checkpoint.get("fit_summary")
        if not isinstance(edges, dict) or set(edges) != set(VARS) or not isinstance(bin_summary, dict):
            raise RuntimeError("source-bin checkpoint is incomplete")
        for variable in VARS:
            edge_array = np.asarray(edges[variable], dtype=np.float64)
            if edge_array.ndim != 1 or not np.isfinite(edge_array).all() or (
                    len(edge_array) > 1 and not (np.diff(edge_array) > 0).all()):
                raise RuntimeError("invalid source-bin checkpoint edges for " + variable)
            if variable in BINARY_VARS and edge_array.tolist() != [0.5]:
                raise RuntimeError("binary source-bin checkpoint edge changed for " + variable)
        if int(bin_summary.get("fit_rows_seen", -1)) <= 0 or int(
                bin_summary.get("fit_rows_candidate", -1)) < 0 or int(
                bin_summary.get("fit_rows_retained", -1)) < 0:
            raise RuntimeError("source-bin checkpoint fit summary is incomplete")
        log("BIN_CHECKPOINT_REUSED path=" + str(args.resume_bins_checkpoint))
    else:
        edges, bin_summary = fit_edges(RANGE_MAP, BINARY_VARS, bin_fit_ids)
        bin_summary["fit_stay_ids_sha256"] = bin_fit_ids_sha256
        checkpoint = {
            "schema": "v5_clean_source_only_bins_checkpoint_v1",
            "fit_scope": "all_eligible_source_split0_h24",
            "fit_stay_n": int(len(bin_fit_ids)),
            "fit_stay_ids_sha256": bin_fit_ids_sha256,
            "prepare_script_sha256": file_hashes["prepare_script"]["sha256"],
            "range_map_code_sha256": file_hashes["extract_py"]["sha256"],
            "source_event_sha256": source_event_hash,
            "source_event_scan_sha256": scan_event_hash,
            "edges": edges, "fit_summary": bin_summary,
        }
        atomic_json(checkpoint_path, checkpoint)
        log("BINS_EDGE_CHECKPOINT_WRITTEN path=" + str(checkpoint_path) +
            " sha256=" + sha256_file(checkpoint_path)["sha256"])
    bin_summary["fit_scope"] = "all_eligible_source_split0_h24"
    bin_summary["fit_stay_n"] = int(len(bin_fit_ids))
    bin_summary["fit_stay_ids_sha256"] = bin_fit_ids_sha256
    bins_payload = {
        "schema": "v5_clean_source_only_bins_v1", "source": "MIMIC-IV",
        "n_bins": NBINS, "variable_order": VARS, "binary_variables": sorted(BINARY_VARS),
        "source_fit_scope": "all eligible split0 H24 stays",
        "source_fit_stay_n": int(len(bin_fit_ids)),
        "source_fit_stay_ids_sha256": bin_fit_ids_sha256,
        "pretraining_member_n": SAMPLE_N,
        "pretraining_member_seed": SAMPLE_SEED,
        "pretraining_member_sha256": source_summary["pretraining_member_sha256"],
        "fit_window": "[0,24) measurement-hour offset from ICU index",
        "quantile_rule": "q=0.1..0.9; round to 4 decimals; unique ties; digitize right=False; clip 1..10",
        "range_fallback_rule": "for non-binary variables with fewer than 200 finite in-range events among all eligible split0 H24 source stays, use read-only extract.py RANGE linear edges",
        "edges_checkpoint_path": str(args.resume_bins_checkpoint or checkpoint_path),
        "edges_checkpoint_sha256": sha256_file(args.resume_bins_checkpoint or checkpoint_path)["sha256"],
        "edges": edges, "fit_summary": bin_summary,
        "source_event_sha256": source_event_hash,
        "source_event_scan_path": str(SOURCE_EVENTS_SCAN),
        "source_event_scan_sha256": scan_event_hash,
    }
    bins_path = RUN_DIR / "clean_source_bins.json"
    atomic_json(bins_path, bins_payload)
    bins_hash = sha256_file(bins_path)["sha256"]
    log("BINS_LOCKED sha256=" + bins_hash + " rss_gib=" + str(rss_gib()))
    atomic_json(RUN_DIR / "progress.json", {
        "stage": "source_bins_locked", "run_id": run_id, "source": source_summary,
        "hashes": file_hashes,
        "bins_path": str(bins_path), "bins_sha256": bins_hash,
        "edges_checkpoint_path": str(args.resume_bins_checkpoint or checkpoint_path),
        "edges_checkpoint_sha256": sha256_file(args.resume_bins_checkpoint or checkpoint_path)["sha256"],
        "bin_summary": bin_summary,
    })
    source_roles["pretrain_member"] = source_roles["stay_id"].isin(
        set(source_selected.tolist())).astype(np.int8)
    target_defs = [
        {"name": "eicu", "events": EICU_EVENTS, "cohort": EICU_COHORT, "meta": EICU_META,
         "eligibility": EICU_ELIG},
        {"name": "nwicu", "events": NWICU_EVENTS, "cohort": NWICU_COHORT, "meta": NWICU_META,
         "eligibility": NWICU_ELIG},
    ]
    cohort_keys = pq.read_table(SOURCE_COHORT, columns=["stay_id", "subject_id"]).to_pandas()
    source_identity = cohort_keys.assign(stay_id=pd.to_numeric(cohort_keys["stay_id"], errors="raise").astype(np.int64),
                                         subject=cohort_keys["subject_id"].astype(str))
    source_identity = source_identity[["stay_id", "subject"]]
    source_roster = source_roles[["stay_id", "subject", "eligible_24h", "split", "pretrain_member"]]
    source_roster_path = RUN_PRIVATE / "source_eligible_roster_tokens_only.csv"
    source_roster.to_csv(source_roster_path, index=False)
    os.chmod(source_roster_path, 0o600)
    datasets = {
        "mimic": build_sequences("mimic", SOURCE_EVENTS_SCAN, source_roster, edges,
                                 RUN_PRIVATE / "mimic_clean_tokens.npz")
    }
    log("DATASET_DONE mimic " + json.dumps(datasets["mimic"]) + " rss_gib=" + str(rss_gib()))
    atomic_json(RUN_DIR / "progress.json", {"stage": "mimic_tokens_built", "run_id": run_id,
                                           "source": source_summary, "bins_sha256": bins_hash,
                                           "datasets": datasets})
    for spec in target_defs:
        name = spec["name"]
        role = read_role_table(spec["eligibility"], "stay_id", "subject")
        cohort = pq.read_table(spec["cohort"], columns=["stay_id", "subject_id"]).to_pandas()
        cohort["stay_id"] = pd.to_numeric(cohort["stay_id"], errors="raise").astype(np.int64)
        cohort["subject"] = cohort["subject_id"].astype(str)
        if cohort["stay_id"].duplicated().any():
            raise RuntimeError(name + " cohort has duplicate stay IDs")
        merged = role.loc[role["eligible_24h"]].merge(
            cohort[["stay_id", "subject"]], on="stay_id", how="left",
            validate="one_to_one", suffixes=("_elig", "_cohort"), indicator=True)
        if not (merged["_merge"] == "both").all():
            raise RuntimeError(name + " eligible H24 roster has stay IDs absent from cohort")
        if not (merged["subject_elig"] == merged["subject_cohort"]).all():
            raise RuntimeError(name + " eligible patient identity mismatch across reviewed roster and event cohort")
        target_roster = merged.rename(columns={"subject_elig": "subject"})[
            ["stay_id", "subject", "eligible_24h"]].sort_values("stay_id").reset_index(drop=True)
        path = RUN_PRIVATE / (name + "_clean_tokens.npz")
        datasets[name] = build_sequences(name, spec["events"], target_roster, edges, path)
        log("DATASET_DONE " + name + " " + json.dumps(datasets[name]) + " rss_gib=" + str(rss_gib()))
        atomic_json(RUN_DIR / "progress.json", {
            "stage": name + "_tokens_built", "run_id": run_id, "source": source_summary,
            "bins_sha256": bins_hash, "datasets": datasets,
        })
    final_manifest = {
        "schema": "v5_clean_pretraining_preparation_v1", "run_id": run_id,
        "status": "pretraining_inputs_built_not_yet_root_gate_approved",
        "hash_checkpoint_reused_from": str(args.resume_hash_checkpoint) if args.resume_hash_checkpoint else None,
        "write_scope": str(RUN_DIR), "source": source_summary,
        "source_roles_and_labels_used": "Only stay_id, subject_id, eligible_24h and frozen split were read for source membership; no label column entered membership, bin fitting, or token construction.",
        "source_vocab": {"variable_order": VARS, "variable_id_definition": "legacy frozen meta_mimic vars order, ID stored as index+1; padding=0",
                         "legacy_sequence_ids_checked": True},
        "binning": {"path": str(bins_path), "sha256": bins_hash, "summary": bin_summary},
        "tokenization": {"max_tokens": MAXLEN, "window": "[0,24)", "hour_bucket": "floor(measurement hour)",
                         "aggregation": "mean per stay_id × floor(hour) × variable",
                         "sampling": "uniform without replacement to 128; fixed seed 20261123; after aggregation",
                         "partitioning": "64 lossless stay_id modulo partitions; no sorted-input assumption",
                         "treatment": "vaso/rrt/sedation are retained as binary start indicators"},
        "datasets": datasets, "input_hashes": {k: v for k, v in file_hashes.items() if k in input_paths},
        "code_hashes": {k: v for k, v in file_hashes.items() if k in code_paths},
        "label_isolation": "Token files contain no outcomes. Readout/evaluation will reread reviewed labels only after root accepts these frozen artifacts.",
        "historical_scope": "New clean source-only sensitivity; it does not authenticate old checkpoint membership or old bins.",
        "runtime": {"cpu_entry": str(CPU_ENTRY), "threads": 2, "cpu_affinity": "0-3 maximum",
                    "rss_gib_peak": rss_gib(), "max_allowed_rss_gib": 8},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_json(RUN_DIR / "preparation_manifest.json", final_manifest)
    log("PREPARATION_COMPLETE manifest=" + str(RUN_DIR / "preparation_manifest.json") +
        " rss_gib=" + str(rss_gib()))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("PREPARATION_FAILED")
        raise
