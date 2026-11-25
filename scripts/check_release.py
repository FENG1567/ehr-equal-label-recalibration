#!/usr/bin/env python3
"""Fail on restricted file types, strong credential signatures, and individual rows."""
from __future__ import annotations
import csv, json, re, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BLOCKED={'.npz','.npy','.parquet','.pkl','.pickle','.pt','.pth','.ckpt','.db','.sqlite','.log','.pem','.key','.xlsx','.zip','.gz','.bak','.pyc'}
ID_COLUMNS={'subject_id','patient_id','patientunitstayid','uniquepid','hadm_id','stay_id','record_id','recordid'}
SIGNATURES=[re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
            re.compile(r'(?i)(?:password|passwd|secret|api[_-]?key|access[_-]?token)\s*[:=]\s*[\"\'][^\"\'\s]{8,}[\"\']'),
            re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}')]

def main():
    problems=[];counts={'files':0,'aggregate_csv':0,'aggregate_json':0};maximum=0
    for path in ROOT.rglob('*'):
        if not path.is_file() or any(p in {'.git','outputs','.venv','__pycache__'} for p in path.relative_to(ROOT).parts):continue
        counts['files']+=1;rel=str(path.relative_to(ROOT));maximum=max(maximum,path.stat().st_size)
        if path.suffix.lower() in BLOCKED:problems.append({'path':rel,'reason':'blocked binary/raw/cache extension'})
        if path.stat().st_size>100*1024**2:problems.append({'path':rel,'reason':'exceeds regular Git file limit'})
        if path.suffix in {'.py','.sh','.json','.md','.txt','.yml','.yaml','.toml','.csv','.cff'}:
            text=path.read_text(encoding='utf-8-sig',errors='strict')
            if any(pattern.search(text) for pattern in SIGNATURES):problems.append({'path':rel,'reason':'possible credential; value not printed'})
        if 'data/aggregates' not in rel.replace('\\','/'):continue
        if path.suffix=='.csv':
            counts['aggregate_csv']+=1
            with path.open(encoding='utf-8-sig',newline='') as f:
                reader=csv.DictReader(f);headers={h.lower() for h in (reader.fieldnames or [])}
                if headers & {'hospitalid','hospital_id','site_id'}:
                    for row in reader:
                        for key in ['n_patients','eligible_unique_patients']:
                            if row.get(key) and float(row[key])<10:
                                problems.append({'path':rel,'reason':'site-linked cohort below public support threshold'});break
            if headers & ID_COLUMNS:problems.append({'path':rel,'reason':'individual identifier column in aggregate CSV'})
        if path.suffix=='.json':
            counts['aggregate_json']+=1
            def scan(node):
                if isinstance(node,dict):
                    for k,v in node.items():
                        if k.lower() in ID_COLUMNS and isinstance(v,list) and len(v)>1:
                            problems.append({'path':rel,'reason':'identifier array in aggregate JSON'})
                        if isinstance(v,list) and len(v)>1 and all(isinstance(x,(str,int,float)) for x in v) and re.search(r'(?i)(patient|subject|stay|member|record).*ids?$|^ids?.*(patient|subject|stay|member|record)',k):
                            problems.append({'path':rel,'reason':'possible individual identifier vector in JSON'})
                        scan(v)
                elif isinstance(node,list):
                    for v in node:scan(v)
            scan(json.loads(path.read_text(encoding='utf-8-sig')))
    print(json.dumps({'status':'passed' if not problems else 'failed','counts':counts,
                      'maximum_file_bytes':maximum,'problems':problems},indent=2))
    return 1 if problems else 0

if __name__=='__main__':sys.exit(main())
