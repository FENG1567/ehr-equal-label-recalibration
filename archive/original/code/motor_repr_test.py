"""Run femr_compute_representations with the MOTOR checkpoint on the test DB."""
import os
import subprocess
import sys

ROOT = "/data/admin/fengshuo/ehr_fm_20261102"
TMP = f"{ROOT}/data/external/motor/etl_test"
DB = f"{TMP}/db"
MODEL = f"{ROOT}/data/external/motor"
VENV = f"{ROOT}/venv_motor/bin/python"
FEMR = f"{ROOT}/third_party/femr0116"
OUT = f"{TMP}/reprs.pkl"


def main():
    env = dict(os.environ)
    env["PYTHONPATH"] = FEMR
    env["PATH"] = f"{FEMR}/bin:" + env.get("PATH", "")
    # prediction times: 24 h after the first event of each patient
    code = (
        "import sys; sys.path.insert(0,'%s');"
        "import femr.datasets, csv, datetime;"
        "db=femr.datasets.PatientDatabase('%s');"
        "w=csv.writer(open('%s/prediction_times.csv','w',newline=''));"
        "w.writerow(['patient_id','prediction_time']);"
        "[w.writerow([p, (db[p].events[0].start + datetime.timedelta(hours=24)).isoformat()])"
        " for p in db if len(db[p].events)>0];"
        "print('wrote', sum(1 for _ in open('%s/prediction_times.csv'))-1)"
        % (FEMR, DB, TMP, TMP)
    )
    r = subprocess.run([VENV, "-c", code], env=env, capture_output=True, text=True)
    print("pred times rc", r.returncode, r.stdout.strip(), r.stderr[-500:])

    runner = (
        "import sys; sys.argv=['x', %r, '--data_path', %r, '--model_path', %r,"
        " '--prediction_times_path', %r, '--batch_size', '64'];"
        "from femr.models.scripts import new_compute_representations;"
        "new_compute_representations()" % (OUT, DB, MODEL, f"{TMP}/prediction_times.csv")
    )
    r = subprocess.run([VENV, "-c", runner], env=env, capture_output=True, text=True)
    print("repr rc", r.returncode)
    print("STDOUT tail:", r.stdout[-3000:])
    print("STDERR tail:", r.stderr[-4000:])
    print("exists", os.path.exists(OUT), os.path.getsize(OUT) if os.path.exists(OUT) else 0)


if __name__ == "__main__":
    main()
