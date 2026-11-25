"""Describe the prespecified stable historical hospital subset, without refitting."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
R5=Path('/data/admin/fengshuo/ehr_fm_20261102/revision_v5_20261123')
reference=R5/'sites/v5_20261123/original_v4_site_reference.csv'
with reference.open(newline='') as handle:
    ref=next(csv.DictReader(handle))
source=R5.parent/'revision_v4_20261122/robustness/results/hospital_primary_v4_20261123_0515/eicu_hospital_metrics.csv'
expected=ref['source_metrics_sha256']
if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
    raise ValueError('Historical site metrics hash differs')
with source.open(newline='') as handle:
    rows=[r for r in csv.DictReader(handle) if r['model_id']=='main20k_real']
stable=[r for r in rows if r['site_stability_flag_ge10_events_and_nonevents'].lower()=='true']
finite=[r for r in rows if r['calibration_slope'] and np.isfinite(float(r['calibration_slope']))]
if (len(rows),len(finite),len(stable))!=(206,193,147):
    raise ValueError('Historical site denominators differ')
out=R5/'publication/build_20261124/stable_site_slope_summary.csv'
if out.exists():raise FileExistsError(out)
data=[]
for name,subset in [('all_finite_points',finite),('stable_ge10_events_and_nonevents',stable)]:
    slopes=np.asarray([float(r['calibration_slope']) for r in subset])
    if not np.isfinite(slopes).all():raise ValueError('Nonfinite stable slope')
    data.append(dict(scope=name,n_sites=len(subset),slope_min=float(slopes.min()),slope_q25=float(np.quantile(slopes,.25)),slope_median=float(np.median(slopes)),slope_q75=float(np.quantile(slopes,.75)),slope_max=float(slopes.max()),source_sha256=expected,interpretation='descriptive hospital point-estimate distribution; not an across-hospital confidence interval; original inputs'))
with out.open('x',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
print(json.dumps(dict(status='complete',output=str(out),summaries=data)),flush=True)
