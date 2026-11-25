#!/usr/bin/env python3
"""Read only Parquet metadata for the preserved temporal snapshot."""
import json
from pathlib import Path

print("schema_probe=pyarrow_import_begin", flush=True)
import pyarrow.parquet as pq
print("schema_probe=pyarrow_import_complete", flush=True)

root = Path("/data/admin/fengshuo/ehr_fm_20261102/revision_v4_20261122/temporal/results/eligibility/v4_pre_4_1_20261122_224500")
for db in ("mimic", "eicu"):
    path = root / f"{db}.parquet"
    print(f"schema_probe={db}_metadata_begin", flush=True)
    parquet = pq.ParquetFile(path)
    print(json.dumps({"db": db, "rows": parquet.metadata.num_rows, "columns": parquet.schema.names}, sort_keys=True), flush=True)
print("schema_probe=complete", flush=True)
