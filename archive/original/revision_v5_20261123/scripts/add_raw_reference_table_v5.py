"""Add all raw 50/500-budget references to the completed calibration table."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

R5=Path('/data/admin/fengshuo/ehr_fm_20261102/revision_v5_20261123')
B=R5/'publication/build_20261124'
source=R5/'calibration/v5_20261123/global_recalibration_repeat_scores.csv'
original=B/'table_data.json'
output=B/'table_data_r2.json'
if output.exists():
    raise FileExistsError(output)
doc=json.loads(original.read_text())
with source.open(newline='') as handle:
    rows=list(csv.DictReader(handle))
table=doc['main_tables'][1]
if len(table['rows'])!=16:
    raise ValueError('Expected all original 16 fitted-update rows')
for cohort,label in [('eicu_semantic_v4_2','eICU'),('nwicu_discharge_v4_1','NWICU')]:
    for model,model_label in [('main20k_real','FM'),('conventional_lr_value_plus_pattern','V+P LR')]:
        for budget in (50,500):
            selected=[r for r in rows if r['cohort']==cohort and r['model_id']==model and r['method']=='no_update' and int(r['budget'])==budget]
            if len(selected)!=200 or set(int(r['repeat']) for r in selected)!=set(range(200)):
                raise ValueError('Raw repeat coverage differs')
            vals=[]
            for metric in ('brier','logloss'):
                x=np.array([float(r[metric]) for r in selected])
                if not np.isfinite(x).all(): raise ValueError('Nonfinite raw scores')
                lo,hi=np.quantile(x,[.05,.95])
                vals.append(f'{x.mean():.5f} ({lo:.5f}–{hi:.5f})')
            events=np.array([int(r['events']) for r in selected])
            lo,hi=np.quantile(events,[.05,.95])
            table['rows'].append([label,model_label,'No update',str(budget),'0/0/0',f'{lo:.1f}–{hi:.1f}',*vals])
order={'No update':0,'Intercept-only':1,'Platt':2}
table['rows'].sort(key=lambda r:(['eICU','NWICU'].index(r[0]),['FM','V+P LR'].index(r[1]),int(r[3]),order[r[2]]))
table['legend']+=' No-update rows are raw references on the same assessment roster, repeated across the same 200 sampling plans; no calibration fit is attempted (0/0/0). Their event ranges describe the associated adaptation samples. A narrowing model gap may reflect deterioration of a comparator; compare absolute raw and updated losses for both models. The four label budgets are retained in the full source data.'
doc.setdefault('additional_provenance',{})['raw_table_references']={
 'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'original_table_json_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
 'raw_cells':8,'repeats_per_cell':200,'fitted_cells_preserved':16}
output.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
print('TABLE_RAW_REFERENCES_COMPLETE output='+str(output),flush=True)
