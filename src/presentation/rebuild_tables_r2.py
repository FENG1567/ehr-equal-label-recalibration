"""Rebuild all seven scientific table bodies from aggregate CSVs."""
import csv,importlib.util,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def records(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def one(rows,**filters):
    selected=[r for r in rows if all(str(r[k])==str(v) for k,v in filters.items())]
    if len(selected)!=1:raise ValueError('Nonunique aggregate table selector: '+str(filters))
    return selected[0]
def rebuild():
    snapshot=json.loads((ROOT/'data/presentation/tables_source_r2.json').read_text(encoding='utf-8'))
    p=ROOT/'archive/original/revision_v5_20261123/scripts/build_tables_v5.py'
    spec=importlib.util.spec_from_file_location('original_table_builder',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    t1,s1,s2,_=m.baseline_tables(ROOT/'data/presentation/table_source_baseline.csv')
    t2,_=m.calibration_table(ROOT/'data/aggregates/Data15/calibration/global_50_500_summary.csv')
    s3,_=m.clinical_table(ROOT/'data/aggregates/Data18/clinical/clinical_summary.csv')
    data=records(ROOT/'data/aggregates/Data15/calibration/global_50_500_summary.csv')
    for cohort,label in [('eicu_semantic_v4_2','eICU'),('nwicu_discharge_v4_1','NWICU')]:
        for model in ['FM','V+P LR']:
            for budget in [50,500]:
                r=one(data,cohort=cohort,model=model,budget=budget,method='no_update')
                cells=[f"{float(r[k+'_mean']):.5f} ({float(r[k+'_p05']):.5f}–{float(r[k+'_p95']):.5f})" for k in ['brier','logloss']]
                t2['rows'].append([label,model,'No update',str(budget),'0/0/0',f"{float(r['events_p05']):.1f}–{float(r['events_p95']):.1f}",*cells])
    order={'No update':0,'Intercept-only':1,'Platt':2}
    t2['rows'].sort(key=lambda r:(['eICU','NWICU'].index(r[0]),['FM','V+P LR'].index(r[1]),int(r[3]),order[r[2]]))
    metrics=records(ROOT/'data/aggregates/Data20/metrics_v7.csv');s4=[]
    for seed in [20261123,20261124,20261125]:
        for arm,label in [('numeric_pretrained','Numeric pretrained'),('same_seed_random_encoder','Random encoder'),('value_pattern','Continuous summary LR')]:
            a=one(metrics,seed=seed,arm=arm,dataset='late_test',calibration='raw')
            b=one(metrics,seed=seed,arm=arm,dataset='late_test',calibration='late500_platt_updated')
            s4.append([str(seed),label,f"{float(a['auroc']):.4f}",f"{float(a['brier']):.6f}",f"{float(b['brier']):.6f}",f"{float(a['logloss']):.6f}",f"{float(b['logloss']):.6f}"])
    metrics=records(ROOT/'data/aggregates/Data22/metrics.csv');gaps=records(ROOT/'data/aggregates/Data22/paired_gaps.csv');s5=[]
    for db in ['eicu','nwicu']:
        for budget in [500,2000]:
            for seed in [20261123,20261124,20261125]:
                a=one(metrics,dataset=db,model=f'fm_seed_{seed}',budget=budget);b=one(metrics,dataset=db,model='token_summary_lr',budget=budget)
                bg=one(gaps,dataset=db,seed=seed,budget=budget,metric='brier');lg=one(gaps,dataset=db,seed=seed,budget=budget,metric='logloss')
                def interval(r):return f"{float(r['updated_gap']):+.7f} ({float(r['updated_ci_low']):+.7f}, {float(r['updated_ci_high']):+.7f})"
                s5.append([db.upper(),str(seed),str(budget),f"{float(a['brier']):.6f}",f"{float(b['brier']):.6f}",interval(bg),interval(lg)])
    all_rows=[t1['rows'],t2['rows'],s1['rows'],s2['rows'],s3['rows'],s4,s5]
    tables=snapshot['main_tables']+snapshot['supplementary_tables']
    def canonical(rows):return [[re.sub(r'\s+','',str(v)).replace('–','-').replace('—','-') for v in row] for row in rows]
    for body,table in zip(all_rows,tables):
        if canonical(body)!=canonical(table['rows']):raise ValueError('Rebuilt table differs from final candidate: '+table['title'])
        table['rows']=body
    return snapshot
