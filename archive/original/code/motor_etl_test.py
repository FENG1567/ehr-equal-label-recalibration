"""Build a tiny femr PatientDatabase from MIMIC events (MOTOR smoke test).

Steps: events CSV (patient_id, code, start, value) -> etl_simple_femr ->
PatientDatabase -> print one patient.  Codes use LOINC so they match the MOTOR
dictionary.
"""
import os
import subprocess
import sys

import pandas as pd

ROOT = "/data/admin/fengshuo/ehr_fm_20261102"
INT = f"{ROOT}/data/interim"
TMP = f"{ROOT}/data/external/motor/etl_test"
VENV = f"{ROOT}/venv_motor/bin/python"

# canonical var -> LOINC code (same mapping validated against MOTOR dictionary)
LOINC = {
    "hr": "8867-4", "sbp": "8480-6", "dbp": "8462-4", "map": "8478-0", "rr": "9279-1",
    "temp": "8310-5", "spo2": "2708-6", "creatinine": "2160-0", "bun": "3094-0",
    "lactate": "2524-7", "wbc": "6690-2", "hgb": "718-7", "platelet": "777-3",
    "sodium": "2951-2", "potassium": "2823-3", "chloride": "2075-0", "calcium": "17861-6",
    "magnesium": "19123-9", "phosphate": "2777-1", "glucose": "2345-7", "albumin": "1751-7",
    "bili": "1975-2", "ast": "1920-8", "alt": "1742-6", "crp": "1988-5", "inr": "6301-6",
    "aptt": "3173-2", "ph": "2744-1", "paco2": "2019-8", "pao2": "2703-7", "fio2": "3150-0",
}


def main(n_patients=200):
    os.makedirs(TMP, exist_ok=True)
    src = os.path.join(TMP, "simple_csv")
    os.makedirs(src, exist_ok=True)
    ev = pd.read_parquet(f"{INT}/events_mimic.parquet")
    co = pd.read_parquet(f"{INT}/cohort_mimic.parquet")
    stays = co["stay_id"].values[:n_patients]
    ev = ev[ev["stay_id"].isin(set(stays)) & ev["var"].isin(LOINC)].copy()
    # synthetic, monotone timestamps per stay (no need for real dates)
    base = pd.Timestamp("2015-01-01")
    ev["start"] = (base + pd.to_timedelta(ev["hour"], unit="h")).dt.strftime("%Y-%m-%dT%H:%M:%S")
    ev["code"] = ev["var"].map(lambda v: f"LOINC/{LOINC[v]}")
    out = ev.rename(columns={"stay_id": "patient_id"})[["patient_id", "code", "start", "val"]]
    out = out.rename(columns={"val": "value"})
    out.to_csv(os.path.join(src, "events.csv"), index=False)
    print("events", len(out), "patients", out.patient_id.nunique(), flush=True)

    target = os.path.join(TMP, "db")
    temp = os.path.join(TMP, "work")
    for p in (target, temp):
        os.makedirs(p, exist_ok=True)
    runner = (
        "import sys; sys.argv=['etl', %r, %r, %r, '--num_threads', '4'];"
        "from femr.etl_pipelines.simple import etl_simple_femr_program;"
        "etl_simple_femr_program()" % (src, target, temp)
    )
    cmd = [VENV, "-c", runner]
    env = dict(os.environ, PYTHONPATH=f"{ROOT}/third_party/femr0116")
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print("ETL rc", r.returncode)
    print(r.stdout[-2000:])
    print(r.stderr[-3000:])

    if r.returncode == 0:
        check = (
            "import sys; sys.path.insert(0,'%s/third_party/femr0116');"
            "import femr.datasets;"
            "db=femr.datasets.PatientDatabase('%s');"
            "pids=list(db); print('n patients',len(pids));"
            "p=db[pids[0]]; print('events',len(p.events));"
            "print(p.events[:5])" % (ROOT, target)
        )
        r2 = subprocess.run([VENV, "-c", check], env=env, capture_output=True, text=True)
        print("LOAD rc", r2.returncode)
        print(r2.stdout[-2000:])
        print(r2.stderr[-2000:])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
