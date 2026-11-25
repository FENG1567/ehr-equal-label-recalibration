#!/usr/bin/env python3
"""Exploratory temporal H24 proxy validation for MIMIC-IV.

This script is deliberately staged: prepare builds private tokens and a frozen
roster; train fits the numeric masked-bin encoder and same-seed random encoder;
evaluate fits source-only readouts, calibrates on a fixed later-patient sample,
and writes aggregate-only reports. It does not estimate prospective clinical
benefit or prove when a measurement was accessible to a clinician.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
R4 = ROOT / "revision_v4_20261122"
R5 = ROOT / "revision_v5_20261123"
R7 = ROOT / "revision_v7_20261124"
ELIG = R4 / "temporal/results/eligibility/recovered_reviewed_v4_2/mimic.parquet"
DATASET = R4 / "availability_repair/dataset/mimic_available_by24_v4_1"
SOURCE_PARTS = DATASET / "source_parts"
PATIENTS = R7 / "recovered_inputs/patients.csv.gz"
EXPECTED_PATIENTS_SHA256 = "4c7507de7ecad8c5ba7647ea4d26018e88ff6672c5fc22ab59c2a5697d687cdd"
COHORT = ROOT / "data/interim/cohort_mimic.parquet"
SPLIT = ROOT / "data/processed/split_mimic.npy"
EXTRACT = ROOT / "code/extract.py"
BUILD_DATA = ROOT / "code/build_data.py"
META = ROOT / "data/processed/meta_mimic.json"
MAPPING = ROOT / "mapping_table.csv"
V5_MODEL = R5 / "scripts/v5_clean_pretraining_gpu.py"
RUN_DEFAULT = R7 / "temporal_availability_v7"
PROTOCOL = R7 / "protocol_v7_implementation_gate_before_results.md"
SEEDS = (20261123, 20261124, 20261125)
PRETRAIN_SEED = 20261124
LATER_SAMPLE_SEED = 20261124
BOOT_SEED = 20261124
PRETRAIN_MAX = 20000
LATE_ADAPT_N = 500
MAXLEN, NBINS, MASK_ID = 128, 10, 11
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
BOOTSTRAPS = 500
DCA_THRESHOLDS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
EVENT_TABLES = ("chartevents", "derived_gcs", "outputevents")
EVENT_SCHEMA = {"stay_id", "hour", "var", "val"}
BIN_VARS_EXPECTED = {"vaso", "rrt", "sedation", "sex"}


def log(message: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), message, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_record(path: Path) -> dict:
    st = Path(path).stat()
    return {"path": str(path), "bytes": int(st.st_size), "mtime_ns": int(st.st_mtime_ns),
            "sha256": sha256_file(path)}


def atomic_json(path: Path, payload: dict, replace: bool = False) -> None:
    path = Path(path)
    tmp = path.with_name("." + path.name + ".tmp")
    if tmp.exists() or (path.exists() and not replace):
        raise FileExistsError(f"refusing to overwrite {path} or stale temporary")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path = Path(path)
    tmp = path.with_name("." + path.name + ".tmp")
    if tmp.exists() or path.exists():
        raise FileExistsError(f"refusing to overwrite {path} or stale temporary")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays) -> None:
    path = Path(path)
    tmp = path.with_name("." + path.name + ".tmp")
    if tmp.exists() or path.exists():
        raise FileExistsError(f"refusing to overwrite {path} or stale temporary")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def literal_assignment(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return ast.literal_eval(node.value)
    raise RuntimeError(f"no literal AST constant {name} found")


def frozen_vocabulary() -> tuple[list[str], dict[str, tuple[float, float]], set[str], dict]:
    xtree, btree = ast.parse(EXTRACT.read_text(encoding="utf-8")), ast.parse(BUILD_DATA.read_text(encoding="utf-8"))
    variables = literal_assignment(xtree, "VARS")
    ranges = literal_assignment(xtree, "RANGE")
    binary = set(literal_assignment(btree, "BIN_VARS"))
    if len(variables) != 48 or len(set(variables)) != 48 or set(variables) != set(ranges):
        raise RuntimeError("frozen extract.py must define 48 unique variables and one range each")
    if binary != BIN_VARS_EXPECTED:
        raise RuntimeError(f"unexpected binary-variable policy {sorted(binary)}")
    mapping = pd.read_csv(MAPPING, dtype=str).fillna("")
    mimic = mapping.loc[(mapping["library"] == "mimic") & mapping["canonical_variable"].isin(variables)]
    units = {}
    for var in variables:
        rows = mimic.loc[mimic["canonical_variable"] == var]
        if rows.empty:
            raise RuntimeError(f"mapping_table.csv has no MIMIC unit/source row for {var}")
        units[var] = [{k: str(row[k]) for k in ("source_table", "source_field", "unit", "conversion")}
                      for _, row in rows.iterrows()]
    if json.loads(META.read_text(encoding="utf-8")).get("vars") != variables:
        raise RuntimeError("meta_mimic.json order differs from AST-frozen extract.py variables")
    return variables, {k: tuple(map(float, ranges[k])) for k in variables}, binary, units


def require_source_provenance() -> tuple[list[Path], dict]:
    manifest_path = DATASET / "dataset_manifest.json"
    counts_path = DATASET / "source_stage_counts.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete_data_only_availability_sensitivity_gate_closed":
        raise RuntimeError("v4.1 dataset manifest is not the completed data-only availability artifact")
    if manifest.get("source_chart_and_store_cutoff_by24_verified") is not True:
        raise RuntimeError("upstream manifest does not verify charttime/storetime H24 cutoff")
    rule = str(manifest.get("source_availability_rule", ""))
    for phrase in ("[ICU intime, intime+24h)", "storetime known and <= intime+24h"):
        if phrase not in rule:
            raise RuntimeError(f"unexpected source availability rule: missing {phrase!r}")
    counts = json.loads(counts_path.read_text(encoding="utf-8"))
    if int(counts.get("assembled_first24_event_rows", 0)) <= 0:
        raise RuntimeError("source_stage_counts.json has no assembled first-24h events")
    parts = []
    table_status = {}
    for table in EVENT_TABLES:
        folder = SOURCE_PARTS / table / "events"
        files = sorted(folder.glob("part-*.parquet"))
        if not files:
            raise RuntimeError(f"no final event parquet files in {folder}")
        # Never include chartevents/gcs_components or raw source table directories.
        if any("gcs_components" in str(p) for p in files):
            raise RuntimeError("GCS component files must not be read alongside derived totals")
        parts.extend(files)
        status_path = SOURCE_PARTS / table / "table_status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") != "complete":
                raise RuntimeError(f"table source stage is not complete: {status_path}")
            table_status[table] = {"path": str(status_path), "sha256": sha256_file(status_path),
                                   "source_file_sha256": status.get("source_file_sha256"),
                                   "event_rows_written": status.get("event_rows_written"),
                                   "chunks": status.get("chunks")}
        elif table == "derived_gcs":
            # Derived GCS has no separate status file in the current layout;
            # the completed stage is certified in the top-level stage counts.
            table_status[table] = {"path": None, "status": "certified_by_source_stage_counts",
                                   "derived_event_rows": counts.get("derived_gcs", {}).get("derived_event_rows"),
                                   "component_rows": counts.get("derived_gcs", {}).get("component_rows")}
        else:
            raise FileNotFoundError(f"missing completion status {status_path}")
        row_count = 0
        for part in files:
            row_count += int(pq.ParquetFile(part).metadata.num_rows)
            schema_names = set(pq.read_schema(part).names)
            if not EVENT_SCHEMA.issubset(schema_names):
                raise RuntimeError(f"unexpected canonical event schema in {part}: {sorted(schema_names)}")
        if table == "derived_gcs":
            expected_rows = int(counts.get("derived_gcs", {}).get("derived_event_rows", -1))
        else:
            expected_rows = int(table_status[table].get("event_rows_written", -1))
        if row_count != expected_rows:
            raise RuntimeError(f"{table}: selected completed parquet row sum {row_count} != status count {expected_rows}")
        table_status[table]["selected_events_dir_parquet_n"] = len(files)
        table_status[table]["selected_events_dir_row_n"] = row_count
    return parts, {"manifest_path": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
                   "manifest": manifest, "counts_path": str(counts_path), "counts_sha256": sha256_file(counts_path),
                   "table_status": table_status}


def _year_bounds(group: str) -> tuple[int, int]:
    years = [int(x) for x in re.findall(r"\d{4}", str(group))]
    if len(years) != 2 or years[1] < years[0]:
        raise RuntimeError(f"cannot parse anchor_year_group {group!r}")
    return years[0], years[1]


def calendar_time_bounds(timestamp, anchor_year: int, group_lo: int, group_hi: int):
    """Conservative date envelope under MIMIC's patient-specific date shift."""
    timestamp = pd.Timestamp(timestamp)
    shifted_lo = pd.Timestamp(year=int(anchor_year), month=1, day=1)
    shifted_hi = pd.Timestamp(year=int(anchor_year) + 1, month=1, day=1)
    real_lo = pd.Timestamp(year=int(group_lo), month=1, day=1)
    real_hi = pd.Timestamp(year=int(group_hi) + 1, month=1, day=1)
    return timestamp - (shifted_hi - real_lo), timestamp - (shifted_lo - real_hi)


def classify_calendar_interval(lower, upper) -> str:
    boundary = pd.Timestamp("2017-01-01")
    if pd.Timestamp(upper) < boundary:
        return "early"
    if pd.Timestamp(lower) >= boundary:
        return "late"
    return "uncertain"


def build_cohort_rosters() -> tuple[dict[str, pd.DataFrame], dict]:
    # Import only the columns needed; frozen sequence/split roles are aligned by cohort row order.
    gate_columns = set(pq.read_schema(ELIG).names)
    exit_col = "hospital_exit_time" if "hospital_exit_time" in gate_columns else "dischtime"
    if exit_col not in gate_columns:
        raise RuntimeError("reviewed H24 gate lacks hospital_exit_time/dischtime needed for temporal label cutoff")
    gate = pq.read_table(ELIG, columns=["eligible_24h", "y_mort", "subject_id", "stay_id", "intime", exit_col]).to_pandas()
    gate = gate.loc[gate["eligible_24h"].fillna(False).astype(bool)].copy()
    gate["subject_id"] = pd.to_numeric(gate["subject_id"], errors="raise").astype(np.int64)
    gate["stay_id"] = pd.to_numeric(gate["stay_id"], errors="raise").astype(np.int64)
    gate["intime"] = pd.to_datetime(gate["intime"], errors="raise")
    gate["dischtime"] = pd.to_datetime(gate[exit_col], errors="coerce")
    gate["y"] = pd.to_numeric(gate["y_mort"], errors="raise").astype(np.int8)
    if gate["stay_id"].duplicated().any() or not gate["y"].isin([0, 1]).all():
        raise RuntimeError("reviewed eligible gate has duplicate stays or non-binary y_mort")
    cohort = pq.read_table(COHORT, columns=["stay_id", "subject_id"]).to_pandas()
    split = np.load(SPLIT, allow_pickle=False)
    if len(cohort) != len(split) or not np.isin(split, [0, 1, 2]).all():
        raise RuntimeError("split_mimic.npy is not aligned with cohort_mimic.parquet")
    cohort["stay_id"] = pd.to_numeric(cohort["stay_id"], errors="raise").astype(np.int64)
    cohort["subject_id"] = pd.to_numeric(cohort["subject_id"], errors="raise").astype(np.int64)
    cohort["split"] = split.astype(np.int8)
    if cohort["stay_id"].duplicated().any():
        raise RuntimeError("source cohort stay IDs are not unique")
    patient_roles = cohort.groupby("subject_id", sort=False)["split"].nunique()
    if (patient_roles > 1).any():
        raise RuntimeError("legacy source split contains a patient in multiple partitions")
    gate = gate.merge(cohort, on=["stay_id", "subject_id"], how="left", validate="one_to_one", indicator=True)
    if not gate["_merge"].eq("both").all():
        raise RuntimeError("reviewed H24 eligibility gate has stays absent from source split cohort")
    gate = gate.drop(columns="_merge")
    if sha256_file(PATIENTS) != EXPECTED_PATIENTS_SHA256:
        raise RuntimeError("restored MIMIC-IV 3.1 patient metadata hash mismatch")
    patients = pd.read_csv(PATIENTS, usecols=["subject_id", "anchor_year", "anchor_year_group"])
    patients["subject_id"] = pd.to_numeric(patients["subject_id"], errors="raise").astype(np.int64)
    patients["anchor_year"] = pd.to_numeric(patients["anchor_year"], errors="raise").astype(np.int64)
    if patients["subject_id"].duplicated().any():
        raise RuntimeError("patients.csv.gz has duplicate subject IDs")
    patients[["anchor_lo", "anchor_hi"]] = pd.DataFrame(
        [_year_bounds(x) for x in patients["anchor_year_group"]], index=patients.index)
    gate = gate.merge(patients[["subject_id", "anchor_year", "anchor_year_group", "anchor_lo", "anchor_hi"]],
                      on="subject_id", how="left", validate="many_to_one")
    if gate["anchor_year"].isna().any():
        raise RuntimeError("patients.csv.gz lacks an eligible gate subject")
    # Apply the full possible patient-specific calendar-date shift envelope;
    # do not approximate it as an integer year difference.
    bounds = [calendar_time_bounds(row.intime, row.anchor_year, row.anchor_lo, row.anchor_hi)
              for row in gate.itertuples(index=False)]
    gate["true_time_lo"] = [x[0] for x in bounds]
    gate["true_time_hi"] = [x[1] for x in bounds]
    gate["time_class"] = [classify_calendar_interval(lo, hi) for lo, hi in bounds]
    if gate["dischtime"].isna().any():
        raise RuntimeError("eligible MIMIC hospital stays lack gate hospital_exit_time/dischtime")
    if (gate["dischtime"] < gate["intime"] + pd.Timedelta(hours=24)).any():
        raise RuntimeError("eligible H24 stay has hospital dischtime earlier than ICU intime+24h")
    discharge_bounds = [calendar_time_bounds(row.dischtime, row.anchor_year, row.anchor_lo, row.anchor_hi)
                        for row in gate.itertuples(index=False)]
    gate["true_dischtime_lo"] = [x[0] for x in discharge_bounds]
    gate["true_dischtime_hi"] = [x[1] for x in discharge_bounds]
    gate["discharge_before_cutoff"] = gate["true_dischtime_hi"] < pd.Timestamp("2017-01-01")
    late_subjects = set(gate.loc[gate["time_class"] == "late", "subject_id"].astype(int))
    early = gate.loc[(gate["time_class"] == "early") & gate["discharge_before_cutoff"]
                     & ~gate["subject_id"].isin(late_subjects)].copy()
    late = gate.loc[gate["time_class"] == "late"].copy()
    early = early.sort_values(["subject_id", "intime", "stay_id"]).drop_duplicates("subject_id", keep="first")
    late = late.sort_values(["subject_id", "intime", "stay_id"]).drop_duplicates("subject_id", keep="first")
    early = early.sort_values(["split", "subject_id", "intime", "stay_id"]).reset_index(drop=True)
    late = late.sort_values(["subject_id", "intime", "stay_id"]).reset_index(drop=True)
    if early["subject_id"].duplicated().any() or late["subject_id"].duplicated().any():
        raise RuntimeError("patient-level roster has repeated subjects")
    if set(early["subject_id"]) & set(late["subject_id"]):
        raise RuntimeError("early development and later validation patients overlap")
    if not set(early["split"].unique()).issubset({0, 1, 2}) or any(early["time_class"] != "early"):
        raise RuntimeError("early cohort split/time eligibility failed")
    if any(late["time_class"] != "late"):
        raise RuntimeError("late cohort time eligibility failed")
    if len(late) <= LATE_ADAPT_N:
        raise RuntimeError(f"later eligible patient count {len(late)} cannot support fixed 500 adaptation + nonempty test")
    rng = np.random.default_rng(LATER_SAMPLE_SEED)
    adapted_positions = np.sort(rng.choice(len(late), LATE_ADAPT_N, replace=False))
    adapt = late.iloc[adapted_positions].copy().sort_values(["subject_id", "intime", "stay_id"])
    test = late.drop(late.index[adapted_positions]).copy()
    if len(adapt) != LATE_ADAPT_N or set(adapt.subject_id) & set(test.subject_id):
        raise RuntimeError("later adaptation/test partition is invalid")
    adapt["partition"] = "late_adapt"
    test["partition"] = "late_test"
    early["partition"] = "early_split" + early["split"].astype(str)
    if set(early.loc[early["split"] == 0, "subject_id"]) & set(early.loc[early["split"] != 0, "subject_id"]):
        raise RuntimeError("early split0 patients overlap early validation/test")
    rosters = {"early": early, "late_adapt": adapt.reset_index(drop=True), "late_test": test.reset_index(drop=True)}
    proof = {"gate_eligible_rows": int(len(gate)), "date_bounds_formula":
             "true_lo=t-(Jan1(anchor_year+1)-Jan1(group_lo)); true_hi=t-(Jan1(anchor_year)-Jan1(group_hi+1)); half-open calendar bounds",
             "early_rule": "intime upper < 2017-01-01 and dischtime upper < 2017-01-01",
             "late_rule": "intime lower >= 2017-01-01", "uncertain_cross_boundary_excluded": True,
             "early_discharge_upper_before_cutoff_required": True,
             "discharge_time_source": f"{exit_col} in SHA-bound reviewed eligibility gate; constructed from original admissions.dischtime",
             "any_late_eligible_stay_excludes_patient_from_early": True,
             "one_earliest_eligible_stay_per_patient": True,
             "tie_break": ["intime ascending", "stay_id ascending"],
             "early_patients_by_split": {str(k): int(v) for k, v in early["split"].value_counts().sort_index().items()},
             "early_patients": int(len(early)), "later_patients": int(len(late)),
             "late_adaptation_n": int(len(adapt)), "late_test_n": int(len(test)),
             "late_sample_seed": LATER_SAMPLE_SEED, "patient_partitions_disjoint": True}
    return rosters, proof


def prepare(run_dir: Path) -> dict:
    run_dir = Path(run_dir).resolve()
    private = run_dir / "private"
    aggregate = run_dir / "aggregate"
    if (run_dir / "preparation_manifest.json").exists():
        raise FileExistsError("preparation manifest exists; inspect this run instead of overwriting")
    run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    private.mkdir(mode=0o700, exist_ok=True)
    os.chmod(run_dir, 0o700); os.chmod(private, 0o700)
    aggregate.mkdir(mode=0o755, exist_ok=True)
    variables, ranges, binary, units = frozen_vocabulary()
    event_files, upstream = require_source_provenance()
    rosters, time_proof = build_cohort_rosters()
    all_roster = pd.concat(list(rosters.values()), ignore_index=True)
    all_roster = all_roster.sort_values(["partition", "subject_id", "intime", "stay_id"]).reset_index(drop=True)
    all_roster["split_for_encoder"] = np.where(all_roster["partition"].str.startswith("early_split"),
                                                 all_roster["split"].astype(np.int8), -1).astype(np.int8)
    ids = all_roster["stay_id"].to_numpy(dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise RuntimeError("combined early/later roster has duplicate stay IDs")
    id_to_row = {int(s): i for i, s in enumerate(ids)}
    split0_ids = set(all_roster.loc[(all_roster["partition"] == "early_split0"), "stay_id"].astype(int))
    if not split0_ids:
        raise RuntimeError("early source split0 has no eligible patients")

    event_hashes = []
    for p in event_files:
        event_hashes.append(file_record(p))
    provenance_files = [ELIG, COHORT, SPLIT, PATIENTS, EXTRACT, BUILD_DATA, META, MAPPING, PROTOCOL,
                        V5_MODEL, DATASET / "dataset_manifest.json", DATASET / "source_stage_counts.json"]
    input_hashes = [file_record(p) for p in provenance_files]
    input_hashes.extend(event_hashes)
    table_status = upstream["table_status"]
    for table, record in table_status.items():
        if record.get("path"):
            record["sha256"] = sha256_file(Path(record["path"]))

    # Stream only the three certified final event datasets. Events are staged in
    # 64 private stay-ID partitions to keep peak RAM bounded on the 3-GiB host.
    staging = private / "event_staging"
    staging.mkdir(mode=0o700, exist_ok=False)
    writers = {}
    early_train_values = {v: [] for v in variables}
    counts = {"source_rows": 0, "roster_candidates": 0, "kept_rows": 0,
              "nonfinite_dropped": 0, "range_dropped": 0}
    roster_ids = np.sort(ids)
    # Map stay IDs to row IDs using a sorted lookup instead of a huge pandas join per batch.
    sort_order = np.argsort(ids)
    sorted_ids = ids[sort_order]
    variables_set = set(variables)
    pa_schema = pa.schema([("stay_id", pa.int64()), ("hour", pa.float64()),
                           ("var", pa.string()), ("val", pa.float64())])
    try:
        for file_no, event_path in enumerate(event_files, start=1):
            pf = pq.ParquetFile(event_path)
            for batch in pf.iter_batches(columns=["stay_id", "hour", "var", "val"], batch_size=100000, use_threads=False):
                frame = batch.to_pandas()
                counts["source_rows"] += len(frame)
                sid = frame["stay_id"].to_numpy(dtype=np.int64, copy=False)
                hour = frame["hour"].to_numpy(dtype=np.float64, copy=False)
                val = frame["val"].to_numpy(dtype=np.float64, copy=False)
                var = frame["var"].astype(str).to_numpy()
                pos = np.searchsorted(sorted_ids, sid)
                matched = (pos < len(sorted_ids))
                matched[matched] &= sorted_ids[pos[matched]] == sid[matched]
                candidate = matched & np.isfinite(hour) & (hour >= 0) & (hour < 24)
                candidate &= np.isfinite(val)
                candidate &= np.fromiter((x in variables_set for x in var), dtype=bool, count=len(var))
                counts["roster_candidates"] += int(candidate.sum())
                if not candidate.any():
                    continue
                rows = np.flatnonzero(candidate)
                lower = np.fromiter((ranges[var[i]][0] for i in rows), dtype=np.float64, count=len(rows))
                upper = np.fromiter((ranges[var[i]][1] for i in rows), dtype=np.float64, count=len(rows))
                in_range = (val[rows] >= lower) & (val[rows] <= upper)
                counts["range_dropped"] += int((~in_range).sum())
                rows = rows[in_range]
                if not len(rows):
                    continue
                counts["kept_rows"] += len(rows)
                kept_sid, kept_hour, kept_var, kept_val = sid[rows], hour[rows], var[rows], val[rows]
                mapped_rows = sort_order[np.searchsorted(sorted_ids, kept_sid)]
                train_mask = np.fromiter((int(x) in split0_ids for x in kept_sid), dtype=bool, count=len(kept_sid))
                if train_mask.any():
                    train_df = pd.DataFrame({"var": kept_var[train_mask], "val": kept_val[train_mask]})
                    for v, g in train_df.groupby("var", sort=False):
                        early_train_values[v].append(g["val"].to_numpy(dtype=np.float64, copy=True))
                part_ids = np.remainder(kept_sid, 64)
                for part_no in np.unique(part_ids):
                    mask = part_ids == part_no
                    table = pa.Table.from_pydict({"stay_id": kept_sid[mask], "hour": kept_hour[mask],
                                                  "var": kept_var[mask], "val": kept_val[mask]}, schema=pa_schema)
                    writer = writers.get(int(part_no))
                    if writer is None:
                        p = staging / f"part_{int(part_no):03d}.parquet"
                        writer = pq.ParquetWriter(p, pa_schema, compression="zstd", use_dictionary=["var"])
                        writers[int(part_no)] = writer
                    writer.write_table(table)
            log(f"PREPARE event_file={file_no}/{len(event_files)} rows={counts['source_rows']} kept={counts['kept_rows']}")
    finally:
        for writer in writers.values():
            writer.close()
    if not writers:
        raise RuntimeError("no in-roster H24 events passed range and finiteness filters")
    edges, edge_modes, fit_counts = {}, {}, {}
    for v in variables:
        values = np.concatenate(early_train_values[v]) if early_train_values[v] else np.empty(0, dtype=float)
        fit_counts[v] = int(len(values))
        if v in binary:
            edges[v], edge_modes[v] = [0.5], "binary_threshold_0.5"
        elif len(values) >= 200:
            edges[v] = np.unique(np.round(np.quantile(values, np.linspace(0, 1, NBINS + 1)[1:-1]), 4)).astype(float).tolist()
            edge_modes[v] = "early_split0_h24_quantile_10bin"
        else:
            lo, hi = ranges[v]
            edges[v] = np.linspace(lo, hi, NBINS + 1)[1:-1].astype(float).tolist()
            edge_modes[v] = "early_split0_count_lt_200_range_fallback"
    del early_train_values

    n = len(all_roster)
    V = np.zeros((n, MAXLEN), dtype=np.int16)
    B = np.zeros((n, MAXLEN), dtype=np.int16)
    H = np.zeros((n, MAXLEN), dtype=np.int16)
    L = np.zeros(n, dtype=np.int16)
    features = np.full((n, len(variables), 6), np.nan, dtype=np.float32)
    token_seed_base = int(PRETRAIN_SEED)
    variable_to_id = {v: i for i, v in enumerate(variables)}
    for part_no, part_path in enumerate(sorted(staging.glob("part_*.parquet")), start=1):
        frame = pq.read_table(part_path, use_threads=False).to_pandas()
        if not len(frame):
            continue
        frame["hour_bin"] = np.floor(frame["hour"].to_numpy(dtype=float)).astype(np.int16)
        grouped = frame.groupby(["stay_id", "hour_bin", "var"], sort=True, as_index=False)["val"].mean()
        grouped["vid"] = grouped["var"].map(variable_to_id).astype(np.int16)
        for stay, d in grouped.groupby("stay_id", sort=False):
            row = id_to_row.get(int(stay))
            if row is None:
                raise RuntimeError("event staging yielded stay outside frozen roster")
            d = d.sort_values(["hour_bin", "vid"], kind="stable")
            vals = d["val"].to_numpy(dtype=np.float64)
            vids = d["vid"].to_numpy(dtype=np.int16)
            hours = d["hour_bin"].to_numpy(dtype=np.int16)
            length = min(len(d), MAXLEN)
            if len(d) > MAXLEN:
                # Per-stay RNG makes a patient's token subset invariant to
                # file ordering or addition/removal of other patients.
                stay_rng = np.random.default_rng(np.random.SeedSequence([token_seed_base, int(stay)]))
                selected_pos = np.sort(stay_rng.choice(len(d), MAXLEN, replace=False))
            else:
                selected_pos = np.arange(len(d))
            raw_bins = np.empty(len(d), dtype=np.int16)
            for vid in np.unique(vids):
                mask = vids == vid
                raw_bins[mask] = np.clip(np.digitize(vals[mask], edges[variables[int(vid)]], right=False) + 1,
                                         1, NBINS).astype(np.int16)
            V[row, :length] = vids[selected_pos] + 1
            B[row, :length] = raw_bins[selected_pos]
            H[row, :length] = hours[selected_pos]
            L[row] = length
            # Value+pattern baseline sees exactly the same selected hourly
            # records as the encoder; overflow/padding never becomes a value.
            selected_vals, selected_vids = vals[selected_pos], vids[selected_pos]
            for vid in np.unique(selected_vids):
                selected = selected_vals[selected_vids == vid]
                features[row, vid] = [float(np.mean(selected)), float(np.min(selected)), float(np.max(selected)),
                                      float(np.std(selected, ddof=0)), float(len(selected)), 1.0]
        if part_no % 8 == 0 or part_no == len(list(staging.glob("part_*.parquet"))):
            log(f"PREPARE aggregate_partition={part_no} rows={len(grouped)}")
        del frame, grouped

    # Preserve all eligible roster members, including empty-event patients.
    member_candidates = np.flatnonzero((all_roster["partition"].to_numpy() == "early_split0") & (L > 0))
    if len(member_candidates) == 0:
        raise RuntimeError("early split0 has no patient with retained H24 event tokens")
    member_rng = np.random.default_rng(PRETRAIN_SEED)
    chosen = np.sort(member_rng.choice(member_candidates, min(PRETRAIN_MAX, len(member_candidates)), replace=False))
    pretrain_flag = np.zeros(n, dtype=np.int8)
    pretrain_flag[chosen] = 1
    member_path = private / "pretraining_members_seed20261124.csv"
    with member_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["stay_id", "subject_id"])
        for row in chosen:
            writer.writerow([int(ids[row]), int(all_roster.iloc[row]["subject_id"])])
        f.flush(); os.fsync(f.fileno())
    os.chmod(member_path, 0o600)
    token_path = private / "tokens_v7.npz"
    atomic_npz(token_path, V=V, B=B, H=H, L=L, stay_id=ids.astype(np.int64),
               subject_id=all_roster["subject_id"].to_numpy(dtype=np.int64),
               y=all_roster["y"].to_numpy(dtype=np.int8),
               partition=all_roster["partition"].to_numpy(dtype="U16"),
               split=all_roster["split_for_encoder"].to_numpy(dtype=np.int8),
               summary=features, pretrain_member=pretrain_flag)
    roster_path = private / "roster_v7.csv"
    all_roster[["stay_id", "subject_id", "y", "intime", "anchor_year", "anchor_year_group",
                "true_time_lo", "true_time_hi", "dischtime", "true_dischtime_lo", "true_dischtime_hi",
                "time_class", "partition", "split_for_encoder"]].to_csv(roster_path, index=False)
    os.chmod(roster_path, 0o600)
    bins_path = private / "bins_v7.json"
    atomic_json(bins_path, {"schema": "v7_early_split0_hourly_10quantile_bins_v1", "variables": variables,
                            "ranges": {k: list(v) for k, v in ranges.items()}, "edges": edges,
                            "fit_count": fit_counts, "edge_mode": edge_modes,
                            "fit_scope": "early split0 patients only; H24 eligible raw events after frozen range checks",
                            "n_bins": NBINS, "binary_variables": sorted(binary)})
    schema = {"token_archive": {"path": str(token_path), "sha256": sha256_file(token_path),
                                "fields": {"V": {"dtype": str(V.dtype), "shape": list(V.shape)},
                                           "B": {"dtype": str(B.dtype), "shape": list(B.shape)},
                                           "H": {"dtype": str(H.dtype), "shape": list(H.shape)},
                                           "L": {"dtype": str(L.dtype), "shape": list(L.shape)},
                                           "stay_id": {"dtype": "int64", "shape": [n]},
                                           "subject_id": {"dtype": "int64", "shape": [n]},
                                           "y": {"dtype": "int8", "shape": [n]},
                                           "partition": {"dtype": "U16", "shape": [n]},
                                           "split": {"dtype": "int8", "shape": [n]},
                                           "summary": {"dtype": str(features.dtype), "shape": list(features.shape)},
                                           "pretrain_member": {"dtype": "int8", "shape": [n]}}},
              "event_input_schema": sorted(EVENT_SCHEMA | {"source_table", "itemid"}),
              "variables": [{"index": i + 1, "name": v, "unit_mapping": units[v], "range": list(ranges[v])}
                            for i, v in enumerate(variables)],
              "padding": {"V": 0, "B": 0, "H": 0, "empty_patient_embedding": "zero vector"}}
    prep_doc = {"schema": "temporal_availability_v7_preparation_v1", "status": "prepared",
                "interpretation": "exploratory retrospective H24 record-availability proxy; no prospective clinical benefit claim",
                "resource_plan": {"cpu_threads": 2, "ram_limit_gib": 3, "gpu_limit_gib": 6},
                "time_partition_evidence": time_proof,
                "upstream_h24_proxy_evidence": {"manifest_path": upstream["manifest_path"],
                    "manifest_sha256": upstream["manifest_sha256"],
                    "source_availability_rule": upstream["manifest"].get("source_availability_rule"),
                    "source_chart_and_store_cutoff_by24_verified": True,
                    "clinical_input_availability_verified": upstream["manifest"].get("clinical_input_availability_verified"),
                    "limitation": "recorded storetime is only a documentation proxy; it does not establish clinician access",
                    "excluded_sources": ["inputevents", "procedureevents", "chartevents/gcs_components"]},
                "source_layout": {"tables": list(EVENT_TABLES), "final_event_part_n": len(event_files),
                                  "table_status": table_status, "source_stage_counts_path": upstream["counts_path"],
                                  "source_stage_counts_sha256": upstream["counts_sha256"],
                                  "source_rows_scanned": counts["source_rows"], **counts},
                "event_filter": {"source_filter_attested_upstream": "charttime in [0,24h), storetime known and <= H24",
                                 "reapplied": ["stay in fixed early/later patient roster", "hour in [0,24)",
                                               "finite value", "frozen physiologic range"],
                                 "hourly_aggregation": "mean per stay-hour-variable", "max_tokens": MAXLEN,
                                 "tokenization_seed": PRETRAIN_SEED},
                "pretraining_members": {"n": int(len(chosen)), "requested_max_n": PRETRAIN_MAX,
                                        "sample_seed": PRETRAIN_SEED, "selection": "early split0 patients with >=1 retained token; no labels",
                                        "path": str(member_path), "sha256": sha256_file(member_path)},
                "bins": {"path": str(bins_path), "sha256": sha256_file(bins_path),
                         "fit_scope": "early split0 H24 event rows only"},
                "tensor_schema": schema, "protocol_hash": file_record(PROTOCOL),
                "feature_units_ranges": {"variables": variables, "units": units,
                                                              "ranges": {k: list(v) for k, v in ranges.items()}},
                "input_hashes": input_hashes,
                "code_hashes": {str(p): sha256_file(p) for p in (Path(__file__).resolve(), V5_MODEL, EXTRACT, BUILD_DATA)},
                "private_outputs": {"tokens": str(token_path), "roster": str(roster_path), "members": str(member_path),
                                    "bins": str(bins_path)},
                "aggregate_dir": str(aggregate), "event_counts": counts,
                "token_counts": {"patients": int(n), "with_tokens": int((L > 0).sum()), "zero_token_patients": int((L == 0).sum()),
                                 "tokens_total": int(L.sum()), "pretrain_members": int(pretrain_flag.sum())}}
    atomic_json(run_dir / "preparation_manifest.json", prep_doc)
    log(f"PREPARE_COMPLETE patients={n} pretrain={len(chosen)} token_sha={schema['token_archive']['sha256']}")
    return prep_doc


def import_v5_model():
    if os.environ.get("V5_GPU_LOCK_HELD") != "1":
        raise RuntimeError("train/evaluate requires the root-held R5 GPU lock (V5_GPU_LOCK_HELD=1)")
    if not V5_MODEL.is_file():
        raise FileNotFoundError(V5_MODEL)
    spec = importlib.util.spec_from_file_location("v7_v5_model", V5_MODEL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_prepared(run_dir: Path) -> tuple[dict, dict]:
    manifest_path = run_dir / "preparation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("run prepare first; preparation_manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "temporal_availability_v7_preparation_v1" or manifest.get("status") != "prepared":
        raise RuntimeError("preparation manifest is incomplete")
    if not PROTOCOL.is_file() or sha256_file(PROTOCOL) != manifest.get("protocol_hash", {}).get("sha256"):
        raise RuntimeError("protocol hash changed or protocol file is absent")
    for source in (Path(__file__).resolve(), V5_MODEL, EXTRACT, BUILD_DATA):
        expected = manifest.get("code_hashes", {}).get(str(source))
        if not expected or sha256_file(source) != expected:
            raise RuntimeError("reviewed implementation hash changed: " + str(source))
    for entry in (manifest["bins"], manifest["pretraining_members"]):
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise RuntimeError("frozen bin/member artifact hash changed")
    token_path = Path(manifest["private_outputs"]["tokens"])
    if sha256_file(token_path) != manifest["tensor_schema"]["token_archive"]["sha256"]:
        raise RuntimeError("token archive hash mismatch")
    with np.load(token_path, allow_pickle=False) as z:
        data = {k: np.asarray(z[k]) for k in z.files}
    if set(data) != {"V", "B", "H", "L", "stay_id", "subject_id", "y", "partition", "split", "summary", "pretrain_member"}:
        raise RuntimeError("token archive has unexpected fields")
    if data["V"].ndim != 2 or data["V"].shape[1] != MAXLEN or data["B"].shape != data["V"].shape or data["H"].shape != data["V"].shape:
        raise RuntimeError("fixed token dimensions invalid")
    members = np.flatnonzero(data["pretrain_member"] == 1)
    member_table = pd.read_csv(manifest["pretraining_members"]["path"], dtype=np.int64)
    if (len(members) != manifest["pretraining_members"]["n"]
            or list(member_table.columns) != ["stay_id", "subject_id"]
            or not np.array_equal(member_table["stay_id"], data["stay_id"][members])
            or not np.array_equal(member_table["subject_id"], data["subject_id"][members])
            or not np.all(data["partition"][members] == "early_split0")
            or not np.all(data["L"][members] > 0)):
        raise RuntimeError("actual training tensors do not match frozen early-source membership")
    return manifest, data


def train(run_dir: Path, root_approved: bool = False) -> dict:
    if not root_approved:
        raise RuntimeError("training is gated until root reviews the frozen private token/role/member artifacts")
    manifest, data = load_prepared(run_dir)
    module = import_v5_model()
    import torch
    training_dir = run_dir / "private/training"
    training_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(training_dir, 0o700)
    manifest_path = training_dir / "training_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("training manifest already exists; refusing to overwrite")
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device unavailable")
    torch.cuda.set_device(0)
    prop = torch.cuda.get_device_properties(0)
    if prop.total_memory < 6 * 1024**3:
        raise RuntimeError("GPU has less than the required 6-GiB allocation envelope")
    torch.cuda.set_per_process_memory_fraction(min(1.0, (6 * 1024**3) / prop.total_memory), 0)
    device = torch.device("cuda:0")
    members = np.flatnonzero(data["pretrain_member"] == 1)
    if len(members) != int(manifest["pretraining_members"]["n"]) or not len(members):
        raise RuntimeError("pretraining member vector differs from frozen manifest")
    V, B, H = (data[k][members].astype(np.int64, copy=False) for k in ("V", "B", "H"))
    if int(sum(p.numel() for p in module.EHRTransformer().parameters())) != 409099:
        raise RuntimeError("reused V5 EHRTransformer parameter count differs from 409,099")
    results = []
    for seed in SEEDS:
        torch.manual_seed(int(seed)); torch.cuda.manual_seed_all(int(seed))
        initial = module.EHRTransformer()
        init_state = {k: v.detach().cpu().clone() for k, v in initial.state_dict().items()}
        init_hash = module.state_hash(init_state)
        init_path = training_dir / f"seed_{seed}_initial.pt"
        torch.save({"schema": "v7_random_encoder_v1", "seed": seed, "state_dict": init_state,
                    "state_dict_sha256": init_hash}, init_path)
        os.chmod(init_path, 0o600)
        model = module.EHRTransformer().to(device)
        model.load_state_dict(init_state, strict=True)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        rng = np.random.default_rng(int(seed) + 0x51A7)
        losses = []
        for epoch in range(2):
            order = rng.permutation(len(members))
            loss_sum = 0.0; masked_total = 0
            for start in range(0, len(order), 256):
                ix = order[start:start + 256]
                v = V[ix].copy(); b = B[ix].copy(); h = H[ix].copy()
                target = np.full(b.shape, -100, dtype=np.int64)
                active = v > 0
                selected = active & (rng.random(active.shape) < 0.20)
                if not selected.any():
                    continue
                target[selected] = b[selected]
                policy = rng.random(selected.shape)
                replace_mask = selected & (policy < 0.80)
                random_mask = selected & (policy >= 0.80) & (policy < 0.90)
                b[replace_mask] = MASK_ID
                if random_mask.any():
                    b[random_mask] = rng.integers(1, NBINS + 1, size=int(random_mask.sum()), dtype=np.int64)
                vt, bt, ht, tt = (torch.from_numpy(x).to(device) for x in (v, b, h, target))
                logits = model.mlm_logits(model(vt, bt, ht, vt == 0))
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, NBINS + 1), tt.reshape(-1),
                                                           ignore_index=-100, reduction="sum") / int(selected.sum())
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                loss_sum += float(loss.detach().cpu()) * int(selected.sum())
                masked_total += int(selected.sum())
            if not masked_total:
                raise RuntimeError(f"no masked-token updates for seed {seed} epoch {epoch + 1}")
            losses.append(loss_sum / masked_total)
            if torch.cuda.max_memory_allocated(0) > 6 * 1024**3:
                raise MemoryError("GPU allocation exceeded 6 GiB")
            log(f"TRAIN seed={seed} epoch={epoch + 1} loss={losses[-1]:.6f} masked={masked_total}")
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        cp = training_dir / f"seed_{seed}_numeric_pretrained.pt"
        torch.save({"schema": "v7_numeric_pretrained_v1", "seed": seed, "epochs": 2,
                    "initial_state_sha256": init_hash, "epoch_losses": losses,
                    "config": {"batch_size": 256, "learning_rate": 0.001, "weight_decay": 0.01,
                               "gradient_norm_cap": 1.0, "mask_fraction": 0.20, "threads": 2},
                    "state_dict": state, "state_dict_sha256": module.state_hash(state)}, cp)
        os.chmod(cp, 0o600)
        results.append({"seed": int(seed), "initial_checkpoint": str(init_path), "initial_sha256": sha256_file(init_path),
                        "initial_state_sha256": init_hash, "numeric_checkpoint": str(cp),
                        "numeric_sha256": sha256_file(cp), "numeric_state_sha256": module.state_hash(state),
                        "epoch_losses": losses})
        del model, initial, optimizer
        torch.cuda.empty_cache()
    doc = {"schema": "temporal_availability_v7_training_v1", "status": "complete",
           "preparation_manifest_sha256": sha256_file(run_dir / "preparation_manifest.json"),
           "model_source_sha256": sha256_file(V5_MODEL), "parameter_n": 409099,
           "seeds": list(SEEDS), "arms": ["numeric_pretrained", "same_seed_random_encoder"],
           "epochs": 2, "optimizer": "AdamW", "loss": "masked value-bin prediction",
           "checkpoints": results}
    atomic_json(manifest_path, doc)
    return doc


def make_value_pattern(summary: np.ndarray, train_rows: np.ndarray) -> tuple[np.ndarray, dict]:
    # summary: n x 48 x (mean,min,max,std,count,observed); only value moments need imputation.
    n, nvar, _ = summary.shape
    raw = summary.copy()
    medians = np.zeros((nvar, 4), dtype=np.float32)
    out = np.zeros((n, nvar, 6), dtype=np.float32)
    for v in range(nvar):
        for k in range(4):
            observed_train = raw[train_rows, v, k]
            observed_train = observed_train[np.isfinite(observed_train)]
            med = float(np.median(observed_train)) if len(observed_train) else 0.0
            medians[v, k] = med
            x = raw[:, v, k]
            missing = ~np.isfinite(x)
            x[missing] = med
            out[:, v, k] = x
        # Missing variables have a true zero count and an observed=0 indicator;
        # no padded token or missing aggregate is represented as a zero value.
        out[:, v, 4] = np.nan_to_num(raw[:, v, 4], nan=0.0)
        out[:, v, 5] = np.nan_to_num(raw[:, v, 5], nan=0.0)
    return out.reshape(n, nvar * 6), {"train_medians": medians.tolist(), "features_per_variable":
                                      ["mean", "min", "max", "std", "hour_count", "observed"]}


def _safe_prevalence(y: np.ndarray) -> float:
    return float((int(np.sum(y)) + 0.5) / (len(y) + 1.0))


def fit_readout(X: np.ndarray, y: np.ndarray, train: np.ndarray, valid: np.ndarray, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if not train.any() or not valid.any() or len(np.unique(y[train])) != 2:
        raise RuntimeError("source readout requires nonempty early train/validation and both train classes")
    scaler = StandardScaler().fit(X[train])
    xt, xv = scaler.transform(X[train]), scaler.transform(X[valid])
    scores, models = {}, {}
    for c in C_GRID:
        clf = LogisticRegression(C=float(c), solver="lbfgs", max_iter=3000, random_state=int(seed))
        clf.fit(xt, y[train])
        p = clf.predict_proba(xv)[:, 1]
        scores[str(c)] = float(np.mean((p - y[valid]) ** 2))
        models[float(c)] = clf
    chosen = min(C_GRID, key=lambda c: (scores[str(c)], c))
    return scaler, models[chosen], float(chosen), scores


def _load_training(run_dir: Path) -> tuple[dict, dict]:
    path = run_dir / "private/training/training_manifest.json"
    if not path.is_file():
        raise FileNotFoundError("train stage must complete before evaluate")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != "temporal_availability_v7_training_v1" or doc.get("status") != "complete":
        raise RuntimeError("training manifest incomplete")
    if doc.get("seeds") != list(SEEDS) or doc.get("parameter_n") != 409099:
        raise RuntimeError("training manifest seeds/model do not match frozen protocol")
    checkpoints = {}
    for row in doc.get("checkpoints", []):
        for arm, path_key, hash_key in (("random", "initial_checkpoint", "initial_sha256"),
                                        ("numeric", "numeric_checkpoint", "numeric_sha256")):
            path = Path(row[path_key])
            if not path.is_file() or sha256_file(path) != row[hash_key]:
                raise RuntimeError(f"checkpoint missing/hash mismatch: {path}")
            checkpoints[(int(row["seed"]), arm)] = path
    if set(checkpoints) != {(s, a) for s in SEEDS for a in ("random", "numeric")}:
        raise RuntimeError("training manifest lacks exact 3x2 seed/model checkpoint roster")
    return doc, checkpoints


def _embed(model, data: dict, rows: np.ndarray, torch, device) -> np.ndarray:
    out = np.zeros((len(rows), model.d), dtype=np.float32)
    valid = np.flatnonzero(data["L"][rows] > 0)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(valid), 256):
            positions = valid[start:start + 256]
            src = rows[positions]
            width = int(data["L"][src].max())
            v, b, h = (data[k][src, :width].astype(np.int64, copy=True) for k in ("V", "B", "H"))
            vt, bt, ht = (torch.from_numpy(x).to(device) for x in (v, b, h))
            out[positions] = model.embed(vt, bt, ht, vt == 0).detach().cpu().numpy()
    if not np.isfinite(out).all():
        raise FloatingPointError("embedding contains nonfinite values")
    return out


def _fit_platt(cal_y: np.ndarray, cal_p: np.ndarray, target_p: np.ndarray) -> tuple[np.ndarray, dict]:
    from scipy.special import logit
    from sklearn.linear_model import LogisticRegression
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    cal_y = np.asarray(cal_y, dtype=np.int8)
    cal_p = np.asarray(cal_p, dtype=float)
    target_p = np.asarray(target_p, dtype=float)
    if len(cal_y) != len(cal_p) or not len(cal_y):
        raise RuntimeError("Platt adaptation labels and probabilities are empty or misaligned")
    fallback = _safe_prevalence(cal_y)
    if len(np.unique(cal_y)) < 2:
        return np.full(len(target_p), fallback, dtype=float), {"status": "single_class_jeffreys_prevalence",
                                                          "fallback_prevalence": fallback}
    cal_logits = logit(np.clip(cal_p, 1e-6, 1 - 1e-6))
    target_logits = logit(np.clip(target_p, 1e-6, 1 - 1e-6))
    if not np.isfinite(cal_logits).all() or not np.isfinite(target_logits).all():
        return np.full(len(target_p), fallback, dtype=float), {"status": "nonfinite_score_jeffreys_prevalence",
                                                          "fallback_prevalence": fallback}
    if np.ptp(cal_logits) <= 1e-12:
        return np.full(len(target_p), fallback, dtype=float), {"status": "constant_score_unidentifiable_jeffreys_prevalence",
                                                          "fallback_prevalence": fallback}
    try:
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=3000, tol=1e-10)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(cal_logits.reshape(-1, 1), cal_y)
        adapted = model.predict_proba(target_logits.reshape(-1, 1))[:, 1]
        if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all() or not np.isfinite(adapted).all():
            raise FloatingPointError("nonfinite Platt coefficients or predictions")
        return adapted, {"status": "fit", "intercept": float(model.intercept_[0]), "slope": float(model.coef_[0, 0])}
    except Exception as exc:
        return np.full(len(target_p), fallback, dtype=float), {"status": "fit_failed_jeffreys_prevalence:" + type(exc).__name__,
                                                          "fallback_prevalence": fallback}


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    from scipy.special import logit
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    y = np.asarray(y, dtype=np.int8); p = np.asarray(p, dtype=float)
    pclip = np.clip(p, 1e-6, 1 - 1e-6)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    ap = float(average_precision_score(y, p)) if np.any(y) else None
    ci = cs = None; status = "single_class_not_estimable"
    if len(np.unique(y)) == 2:
        try:
            m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=3000, tol=1e-10)
            lp = logit(pclip).reshape(-1, 1); m.fit(lp, y)
            ci, cs, status = float(m.intercept_[0]), float(m.coef_[0, 0]), "fit"
        except Exception as exc:
            status = "fit_failed:" + type(exc).__name__
    return {"n_patients": int(len(y)), "events": int(y.sum()), "prevalence": float(y.mean()),
            "auroc": auc, "average_precision": ap, "brier": float(brier_score_loss(y, p)),
            "logloss": float(log_loss(y, pclip, labels=[0, 1])), "calibration_intercept": ci,
            "calibration_slope": cs, "calibration_status": status}


def _bootstrap_ci(y: np.ndarray, p: np.ndarray, seed: int) -> dict:
    # One stay per patient by construction, so patient bootstrap resamples rows.
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    y = np.asarray(y, dtype=np.int8); p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    keys = {"brier": [], "logloss": [], "auroc": [], "average_precision": []}
    for _ in range(BOOTSTRAPS):
        ix = rng.integers(0, len(y), size=len(y))
        yy, pp = y[ix], p[ix]
        keys["brier"].append(float(brier_score_loss(yy, pp)))
        keys["logloss"].append(float(log_loss(yy, np.clip(pp, 1e-6, 1 - 1e-6), labels=[0, 1])))
        keys["auroc"].append(float(roc_auc_score(yy, pp)) if len(np.unique(yy)) == 2 else float("nan"))
        keys["average_precision"].append(float(average_precision_score(yy, pp)) if np.any(yy) else float("nan"))
    return {m: {"low_2_5": float(np.nanquantile(v, .025)), "high_97_5": float(np.nanquantile(v, .975)),
                "valid_replicates": int(np.isfinite(v).sum()), "replicates": BOOTSTRAPS} for m, v in keys.items()}


def _dca_rows(dataset: str, seed: int, arm: str, calibration: str, y: np.ndarray, p: np.ndarray) -> list[dict]:
    y = np.asarray(y, dtype=np.int8); p = np.asarray(p, dtype=float); n = len(y)
    rng = np.random.default_rng(BOOT_SEED + seed + n + sum(ord(c) for c in dataset + arm + calibration))
    boot = {t: {"model": [], "treat_all": []} for t in DCA_THRESHOLDS}
    for _ in range(BOOTSTRAPS):
        ix = rng.integers(0, n, size=n)
        yy, pp = y[ix], p[ix]
        prevalence_b = float(yy.mean())
        for threshold in DCA_THRESHOLDS:
            odds = threshold / (1 - threshold)
            tp_b = float(np.mean((pp >= threshold) & (yy == 1)))
            fp_b = float(np.mean((pp >= threshold) & (yy == 0)))
            boot[threshold]["model"].append(100 * (tp_b - fp_b * odds))
            boot[threshold]["treat_all"].append(100 * (prevalence_b - (1 - prevalence_b) * odds))
    rows = []
    for threshold in DCA_THRESHOLDS:
        odds = threshold / (1 - threshold)
        tp = float(np.mean((p >= threshold) & (y == 1)))
        fp = float(np.mean((p >= threshold) & (y == 0)))
        model_nb = 100 * (tp - fp * odds)
        all_nb = 100 * (float(y.mean()) - (1 - float(y.mean())) * odds)
        model_ci = np.quantile(boot[threshold]["model"], [.025, .975])
        all_ci = np.quantile(boot[threshold]["treat_all"], [.025, .975])
        rows.append({"dataset": dataset, "seed": seed, "arm": arm, "calibration": calibration,
                     "strategy": "model", "threshold": threshold, "net_benefit_per_100": model_nb,
                     "ci_low_per_100": float(model_ci[0]), "ci_high_per_100": float(model_ci[1]),
                     "bootstrap_replicates": BOOTSTRAPS, "n": n})
        rows.append({"dataset": dataset, "seed": seed, "arm": arm, "calibration": calibration,
                     "strategy": "treat_all", "threshold": threshold, "net_benefit_per_100": all_nb,
                     "ci_low_per_100": float(all_ci[0]), "ci_high_per_100": float(all_ci[1]),
                     "bootstrap_replicates": BOOTSTRAPS, "n": n})
        rows.append({"dataset": dataset, "seed": seed, "arm": arm, "calibration": calibration,
                     "strategy": "treat_none", "threshold": threshold, "net_benefit_per_100": 0.0,
                     "ci_low_per_100": 0.0, "ci_high_per_100": 0.0,
                     "bootstrap_replicates": BOOTSTRAPS, "n": n})
    return rows


def _paired_bootstrap_contrasts(y: np.ndarray, predictions: dict[str, np.ndarray], seed: int) -> list[dict]:
    y = np.asarray(y, dtype=np.int8)
    pairs = (("numeric_pretrained", "value_pattern"),
             ("numeric_pretrained", "same_seed_random_encoder"))
    present = set(predictions)
    rows = []
    for left, right in pairs:
        if left not in present or right not in present:
            continue
        rng = np.random.default_rng(seed)
        brier_delta, logloss_delta = [], []
        p_left, p_right = predictions[left], predictions[right]
        point_brier = float(np.mean((p_left - y) ** 2 - (p_right - y) ** 2))
        pcl, pcr = np.clip(p_left, 1e-6, 1 - 1e-6), np.clip(p_right, 1e-6, 1 - 1e-6)
        point_logloss = float(np.mean(-(y * np.log(pcl) + (1 - y) * np.log(1 - pcl))
                                      + y * np.log(pcr) + (1 - y) * np.log(1 - pcr)))
        for _ in range(BOOTSTRAPS):
            ix = rng.integers(0, len(y), size=len(y))
            yy, pl, pr = y[ix], p_left[ix], p_right[ix]
            brier_delta.append(float(np.mean((pl - yy) ** 2 - (pr - yy) ** 2)))
            plc, prc = np.clip(pl, 1e-6, 1 - 1e-6), np.clip(pr, 1e-6, 1 - 1e-6)
            ll = -(yy * np.log(plc) + (1 - yy) * np.log(1 - plc))
            lr = -(yy * np.log(prc) + (1 - yy) * np.log(1 - prc))
            logloss_delta.append(float(np.mean(ll - lr)))
        for metric, values in (("brier", brier_delta), ("logloss", logloss_delta)):
            point = point_brier if metric == "brier" else point_logloss
            rows.append({"contrast": f"{left}_minus_{right}", "metric": metric,
                         "difference": point, "ci_low_2_5": float(np.quantile(values, .025)),
                         "ci_high_97_5": float(np.quantile(values, .975)), "replicates": BOOTSTRAPS,
                         "unit": "paired patient bootstrap; negative favors left"})
    return rows


def evaluate(run_dir: Path) -> dict:
    prep, data = load_prepared(run_dir)
    module = import_v5_model()
    import torch
    from sklearn.preprocessing import StandardScaler
    training, checkpoints = _load_training(run_dir)
    if training.get("model_source_sha256") != sha256_file(V5_MODEL):
        raise RuntimeError("training and evaluation model implementation hash mismatch")
    if training.get("preparation_manifest_sha256") != sha256_file(run_dir / "preparation_manifest.json"):
        raise RuntimeError("training/preparation manifest hash mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device unavailable")
    torch.set_num_threads(2); torch.cuda.set_device(0); device = torch.device("cuda:0")
    gpu_total = int(torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.set_per_process_memory_fraction(min(1.0, (6 * 1024**3) / gpu_total), 0)
    aggregate_dir = run_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
    private_eval = run_dir / "private/evaluation"
    private_eval.mkdir(parents=True, exist_ok=True, mode=0o700); os.chmod(private_eval, 0o700)
    # Roster has one stay per patient by design; save per-seed predictions privately.
    partitions = data["partition"].astype(str)
    train_rows = np.flatnonzero(partitions == "early_split0")
    valid_rows = np.flatnonzero(partitions == "early_split1")
    early_test_rows = np.flatnonzero(partitions == "early_split2")
    adapt_rows = np.flatnonzero(partitions == "late_adapt")
    late_test_rows = np.flatnonzero(partitions == "late_test")
    if any(len(x) == 0 for x in (train_rows, valid_rows, early_test_rows, adapt_rows, late_test_rows)):
        raise RuntimeError("one of early train/validation/test or late adaptation/test partitions is empty")
    y = data["y"].astype(np.int8)
    if len(np.unique(y[train_rows])) != 2:
        raise RuntimeError("early split0 training labels are single-class")
    # Baseline summary imputation medians are fit only on early split0.
    vp, vp_meta = make_value_pattern(data["summary"].astype(np.float32), train_rows)
    eval_rows = {"early_split2": early_test_rows, "late_adapt": adapt_rows, "late_test": late_test_rows}
    metric_rows, bootstrap_rows, dca_rows, contrast_rows = [], [], [], []
    prediction_manifest = []
    for seed in SEEDS:
        arm_features = {}
        paired_predictions = {}
        for arm in ("numeric", "random"):
            cp = torch.load(checkpoints[(seed, arm)], map_location="cpu", weights_only=False)
            state = cp["state_dict"]
            if module.state_hash(state) != cp.get("state_dict_sha256"):
                raise RuntimeError(f"state hash mismatch for seed={seed} arm={arm}")
            model = module.EHRTransformer(); model.load_state_dict(state, strict=True); model.to(device)
            emb = _embed(model, data, np.arange(len(y)), torch, device)
            arm_features["numeric_pretrained" if arm == "numeric" else "same_seed_random_encoder"] = emb
            del model; torch.cuda.empty_cache()
        arm_features["value_pattern"] = vp
        for arm, X in arm_features.items():
            scaler, clf, chosen_c, c_scores = fit_readout(X, y, train_rows, valid_rows, seed)
            Xscaled = scaler.transform(X)
            p = clf.predict_proba(Xscaled)[:, 1]
            # Fit only on the fixed adaptation rows, then predict the full roster
            # so later-test indexing remains aligned to the original token rows.
            p_adapt, platt_meta = _fit_platt(y[adapt_rows], p[adapt_rows], p)
            # The same 500 labels are shown descriptively as adaptation-sample performance.
            for dataset, rows, pred, calibration in (
                ("early_split2", early_test_rows, p[early_test_rows], "raw"),
                ("late_test", late_test_rows, p[late_test_rows], "raw"),
                ("late_test", late_test_rows, p_adapt[late_test_rows], "late500_platt_updated"),
                ("late_adapt", adapt_rows, p[adapt_rows], "raw"),
                ("late_adapt", adapt_rows, p_adapt[adapt_rows], "same_labels_platt_descriptive")):
                result = _metrics(y[rows], pred)
                metric_rows.append({"seed": seed, "arm": arm, "dataset": dataset, "calibration": calibration,
                                    "selected_C": chosen_c, "validation_brier_by_C": json.dumps(c_scores, sort_keys=True),
                                    "platt_status": platt_meta.get("status"), **result})
                if dataset in ("early_split2", "late_test") and calibration in ("raw", "late500_platt_updated"):
                    paired_predictions.setdefault((dataset, calibration), {})[arm] = np.asarray(pred, dtype=float)
                if calibration in ("raw", "late500_platt_updated", "same_labels_platt_descriptive"):
                    dca_rows.extend(_dca_rows(dataset, seed, arm, calibration, y[rows], pred))
                if calibration in ("raw", "late500_platt_updated", "same_labels_platt_descriptive"):
                    ci = _bootstrap_ci(y[rows], pred, BOOT_SEED + seed + int(np.sum(rows)))
                    for metric, bound in ci.items():
                        bootstrap_rows.append({"seed": seed, "arm": arm, "dataset": dataset,
                                               "calibration": calibration, "metric": metric,
                                               **bound})
            pred_path = private_eval / f"predictions_seed_{seed}_{arm}.npz"
            atomic_npz(pred_path, stay_id=data["stay_id"].astype(np.int64), subject_id=data["subject_id"].astype(np.int64),
                       y=y, partition=partitions.astype("U16"), p=p.astype(np.float32),
                       p_late500_platt=p_adapt.astype(np.float32))
            prediction_manifest.append({"seed": seed, "arm": arm, "path": str(pred_path), "sha256": sha256_file(pred_path)})
            del Xscaled, scaler, clf, p, p_adapt
        del arm_features
        for (dataset, calibration), predictions in paired_predictions.items():
            paired_y = y[early_test_rows if dataset == "early_split2" else late_test_rows]
            for row in _paired_bootstrap_contrasts(paired_y, predictions,
                                                   BOOT_SEED + seed + len(paired_y)):
                contrast_rows.append({"seed": seed, "dataset": dataset, "calibration": calibration, **row})
        log(f"EVALUATE seed={seed} completed")
    metrics_path = aggregate_dir / "metrics_v7.csv"
    boot_path = aggregate_dir / "patient_bootstrap_v7.csv"
    dca_path = aggregate_dir / "decision_curve_v7.csv"
    contrast_path = aggregate_dir / "paired_model_bootstrap_v7.csv"
    atomic_csv(metrics_path, metric_rows, list(metric_rows[0].keys()))
    atomic_csv(boot_path, bootstrap_rows, list(bootstrap_rows[0].keys()))
    atomic_csv(dca_path, dca_rows, list(dca_rows[0].keys()))
    atomic_csv(contrast_path, contrast_rows, list(contrast_rows[0].keys()))
    summary_path = aggregate_dir / "evaluation_manifest_v7.json"
    summary = {"schema": "temporal_availability_v7_evaluation_v1", "status": "complete",
               "interpretation": "exploratory retrospective H24 record proxy; not prospective clinical utility",
               "preparation_manifest_sha256": sha256_file(run_dir / "preparation_manifest.json"),
               "training_manifest_sha256": sha256_file(run_dir / "private/training/training_manifest.json"),
               "seeds": list(SEEDS), "readout": {"features": "StandardScaler + LogisticRegression",
                   "C_grid": list(C_GRID), "selection": "early split1 Brier; tie smaller C",
                   "baseline": vp_meta},
               "late_adaptation": {"n": LATE_ADAPT_N, "calibration": "Platt on source-readout probabilities and the same fixed 500 later-patient labels",
                                   "fallback": "Jeffreys-smoothed prevalence for single-class or failed fit"},
               "bootstrap": {"replicates": BOOTSTRAPS, "unit": "patient (one stay per patient)", "seed": BOOT_SEED},
               "decision_curve": {"thresholds": list(DCA_THRESHOLDS), "scope": "theoretical threshold-conditioned net benefit only"},
               "predictions_private": prediction_manifest,
               "aggregate_files": {str(metrics_path): sha256_file(metrics_path), str(boot_path): sha256_file(boot_path),
                                   str(dca_path): sha256_file(dca_path), str(contrast_path): sha256_file(contrast_path)}}
    atomic_json(summary_path, summary)
    log("EVALUATE_COMPLETE metrics=" + str(metrics_path))
    return summary


def synthetic_self_test() -> None:
    # Tiny helper-level checks; no private source data, GPU, or old main() calls.
    x = np.array([[[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                   [2, 1, 3, 0.5, 2, 1]],
                  [[4, 3, 5, 1, 1, 1], [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]], dtype=np.float32)
    z, _ = make_value_pattern(x, np.array([0]))
    assert z.shape == (2, 12) and z[0, 0] == 0 and z[0, 6] == 2
    assert z[0, 4] == 0 and z[0, 5] == 0
    # Held-out values do not supply the all-missing training-variable median.
    assert z[1, 0] == 4 and z[1, 6] == 2 and z[1, 10] == 0 and z[1, 11] == 0
    assert abs(_safe_prevalence(np.array([0, 0], dtype=np.int8)) - 1.0 / 6.0) < 1e-12
    lo, hi = calendar_time_bounds(pd.Timestamp("2000-02-29 12:00"), 2000, 2014, 2015)
    assert lo < hi and classify_calendar_interval(lo, hi) == "early"
    lo, hi = calendar_time_bounds(pd.Timestamp("2000-06-01"), 2000, 2019, 2021)
    assert classify_calendar_interval(lo, hi) == "late"
    lo, hi = calendar_time_bounds(pd.Timestamp("2000-06-01"), 2000, 2016, 2018)
    assert classify_calendar_interval(lo, hi) == "uncertain"
    assert classify_calendar_interval(pd.Timestamp("2016-12-31"), pd.Timestamp("2017-01-01")) == "uncertain"
    assert classify_calendar_interval(pd.Timestamp("2017-01-01"), pd.Timestamp("2017-02-01")) == "late"
    single, meta = _fit_platt(np.array([1, 1], dtype=np.int8), np.array([.2, .7]), np.linspace(.01, .99, 19))
    assert meta["status"] == "single_class_jeffreys_prevalence" and len(single) == 19 and np.allclose(single, 5.0 / 6.0)
    global_p = np.linspace(.02, .98, 1200)
    adapt_rows = np.arange(1, 1000, 2)[:500]
    adapt_y = (adapt_rows % 4 >= 2).astype(np.int8)
    adapted, adapt_meta = _fit_platt(adapt_y, global_p[adapt_rows], global_p)
    assert len(adapted) == len(global_p) and np.isfinite(adapted).all()
    assert adapt_meta["status"] == "fit", adapt_meta
    rows = _dca_rows("x", 1, "a", "raw", np.array([0, 1]), np.array([.1, .9]))
    assert len(rows) == len(DCA_THRESHOLDS) * 3
    pairs = {"numeric_pretrained": np.array([.1, .9]), "value_pattern": np.array([.2, .8]),
             "same_seed_random_encoder": np.array([.3, .7])}
    contrasts = _paired_bootstrap_contrasts(np.array([0, 1]), pairs, 123)
    assert len(contrasts) == 4 and all("difference" in row for row in contrasts)
    print("synthetic helper checks passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "train", "evaluate", "all"), default="prepare")
    parser.add_argument("--run-dir", type=Path, default=RUN_DEFAULT)
    parser.add_argument("--root-approved-training", action="store_true",
                        help="required only after root reviews the prepared private artifacts")
    parser.add_argument("--self-test", action="store_true", help="run small pure-helper checks and exit")
    args = parser.parse_args()
    if args.self_test:
        synthetic_self_test(); return
    run_dir = args.run_dir
    try:
        if args.stage in ("prepare", "all"):
            prepare(run_dir)
        if args.stage in ("train", "all"):
            train(run_dir, root_approved=args.root_approved_training)
        if args.stage in ("evaluate", "all"):
            evaluate(run_dir)
    except Exception as exc:
        failure = {"schema": "temporal_availability_v7_failure_v1", "status": "failed",
                   "stage": args.stage, "error_type": type(exc).__name__, "error": str(exc),
                   "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            atomic_json(run_dir / "failure_manifest.json", failure)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
