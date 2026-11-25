"""Diagnose the 34% event drop in MOTOR code mapping.

Builds one tiny database per variable (plus an all-variable one) and runs
clmbr_create_batches, capturing the "dropped X out of Y" line for each.
"""
import os
import re
import shutil
import subprocess
import sys

import pandas as pd

ROOT = "/data/admin/fengshuo/ehr_fm_20261102"
INT = f"{ROOT}/data/interim"
BASE = f"{ROOT}/data/external/motor/drop_diag"
VENV = f"{ROOT}/venv_motor/bin/python"
FEMR = f"{ROOT}/third_party/femr0116"
MODEL = f"{ROOT}/data/external/motor"

LOINC = {
    "hr": "8867-4", "sbp": "8480-6", "dbp": "8462-4", "map": "8478-0", "rr": "9279-1",
    "temp": "8310-5", "spo2": "2708-6", "creatinine": "2160-0", "bun": "3094-0",
    "lactate": "2524-7", "wbc": "6690-2", "hgb": "718-7", "platelet": "777-3",
    "sodium": "2951-2", "potassium": "2823-3", "chloride": "2075-0", "calcium": "17861-6",
    "magnesium": "19123-9", "phosphate": "2777-1", "glucose": "2345-7", "albumin": "1751-7",
    "bili": "1975-2", "ast": "1920-8", "alt": "1742-6", "crp": "1988-5", "inr": "6301-6",
    "aptt": "3173-2", "ph": "2744-1", "paco2": "2019-8", "pao2": "2703-7", "fio2": "3150-0",
}


def build(name, vars_, n_pat=80):
    d = os.path.join(BASE, name)
    if os.path.exists(d):
        shutil.rmtree(d)
    src = os.path.join(d, "csv")
    os.makedirs(src)
    ev = pd.read_parquet(f"{INT}/events_mimic.parquet")
    co = pd.read_parquet(f"{INT}/cohort_mimic.parquet")
    stays = co["stay_id"].values[:n_pat]
    ev = ev[ev["stay_id"].isin(set(stays)) & ev["var"].isin(vars_)].copy()
    base = pd.Timestamp("2015-01-01")
    ev["start"] = (base + pd.to_timedelta(ev["hour"], unit="h")).dt.strftime("%Y-%m-%dT%H:%M:%S")
    ev["code"] = ev["var"].map(lambda v: f"LOINC/{LOINC[v]}")
    out = ev.rename(columns={"stay_id": "patient_id"})[["patient_id", "code", "start", "val"]].rename(columns={"val": "value"})
    out.to_csv(os.path.join(src, "events.csv"), index=False)

    env = dict(os.environ, PYTHONPATH=FEMR, PATH=f"{FEMR}/bin:" + os.environ.get("PATH", ""))
    runner = (
        "import sys; sys.argv=['etl', %r, %r, %r, '--num_threads', '4'];"
        "from femr.etl_pipelines.simple import etl_simple_femr_program; etl_simple_femr_program()"
        % (src, os.path.join(d, "db"), os.path.join(d, "work"))
    )
    r = subprocess.run([VENV, "-c", runner], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(name, "ETL FAIL", r.stderr[-500:])
        return None

    # prediction times
    times = os.path.join(d, "times.csv")
    code = (
        "import sys; sys.path.insert(0,%r); import femr.datasets, csv, datetime;"
        "db=femr.datasets.PatientDatabase(%r); w=csv.writer(open(%r,'w',newline=''));"
        "w.writerow(['patient_id','prediction_time']);"
        "[w.writerow([p,(db[p].events[0].start+datetime.timedelta(hours=24)).isoformat()]) for p in db if db[p].events]"
        % (FEMR, os.path.join(d, "db"), times)
    )
    subprocess.run([VENV, "-c", code], env=env, capture_output=True, text=True)

    cmd = [
        f"{FEMR}/bin/clmbr_create_batches", os.path.join(d, "batches"),
        "--data_path", os.path.join(d, "db"), "--task", "labeled_patients",
        "--labeled_patients_path", times, "--val_start", "70",
        "--dictionary_path", f"{MODEL}/dictionary", "--is_hierarchical",
        "--transformer_vocab_size", "65536", "--batch_size", "64",
    ]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    blob = (r.stdout or "") + (r.stderr or "")
    m = re.search(r"dropped (\d+) out of (\d+)", blob)
    hit = m.groups() if m else ("?", "?")
    print(f"{name:12s} nvars={len(vars_):2d} events={len(out):6d} dropped={hit[0]}/{hit[1]}", flush=True)
    return dict(name=name, nvars=len(vars_), events=len(out), dropped=hit[0], total=hit[1])


if __name__ == "__main__":
    os.makedirs(BASE, exist_ok=True)
    res = []
    for v in ["hr", "creatinine", "lactate", "platelet", "wbc", "hgb"]:
        res.append(build(v, [v]))
    res.append(build("all31", list(LOINC)))
    pd.DataFrame([r for r in res if r]).to_csv(f"{ROOT}/results/tables/motor_drop_diag.csv", index=False)
