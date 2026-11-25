#!/usr/bin/env python3
"""Reproduce published displays from shareable aggregate results, without patients."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, os, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(key,'1')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def verify():
    records=read(ROOT/'provenance/file_manifest.json')['files']
    for entry in records:
        path=(ROOT/entry['path']).resolve()
        if ROOT not in path.parents:raise ValueError('Manifest path outside repository')
        if not path.is_file() or sha(path)!=entry['sha256']:
            raise ValueError('Missing or changed release input: '+entry['path'])
    for n in range(1,23):
        if not (ROOT/f'data/aggregates/Data{n}').is_dir():raise ValueError(f'Missing Data{n}')
    return len(records)

def tables(output):
    path=ROOT/'src/presentation/rebuild_tables.py'
    spec=importlib.util.spec_from_file_location('public_table_rebuilder',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    document=module.rebuild()
    folder=output/'tables';folder.mkdir(parents=True,exist_ok=True)
    written=[]
    for group in ['main_tables','supplementary_tables']:
        prefix='Table' if group=='main_tables' else 'TableS'
        for table in document[group]:
            base=folder/f"{prefix}{table['number']}"
            with base.with_suffix('.csv').open('w',encoding='utf-8',newline='') as stream:
                w=csv.writer(stream);w.writerow(table['columns']);w.writerows(table['rows'])
            def clean(x):return str(x).replace('|','\\|').replace('\n',' ')
            lines=[f"# {prefix}{table['number']}. {table['title']}",'',
                   '| '+' | '.join(map(clean,table['columns']))+' |',
                   '| '+' | '.join(['---']*len(table['columns']))+' |']
            lines += ['| '+' | '.join(map(clean,row))+' |' for row in table['rows']]
            lines+=['',table['legend'],'']
            base.with_suffix('.md').write_text('\n'.join(lines),encoding='utf-8')
            written.extend([base.with_suffix('.csv'),base.with_suffix('.md')])
    return written

def figures(output):
    folder=output/'figures';folder.mkdir(parents=True,exist_ok=True)
    before={p.relative_to(folder):p.stat().st_mtime_ns for p in folder.rglob('*') if p.is_file()}
    started=time.time_ns()
    for name in ('rebuild_figures_v12.py','rebuild_figures_v12_additions.py'):
        script=ROOT/'src/presentation'/name
        subprocess.run([sys.executable,str(script),'--repo',str(ROOT),'--output',str(folder)],check=True)
    written=[]
    for path in folder.rglob('*'):
        if not path.is_file():continue
        previous=before.get(path.relative_to(folder))
        if previous is None or path.stat().st_mtime_ns!=previous or path.stat().st_mtime_ns>=started:
            written.append(path)
    return written

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs')
    parser.add_argument('--tables-only',action='store_true')
    args=parser.parse_args()
    count=verify();print(f'VERIFIED {count} release inputs',flush=True)
    if args.verify_only:return
    output=args.output.resolve()
    protected=[ROOT/'data',ROOT/'src',ROOT/'reference',ROOT/'provenance',ROOT/'archive']
    if output==ROOT or any(output==p or p in output.parents for p in protected):raise ValueError('Output cannot overwrite repository inputs')
    output.mkdir(parents=True,exist_ok=True)
    written=tables(output)
    if not args.tables_only:written+=figures(output)
    result={'scope':'Aggregate display reproduction only; no clinical refitting or bootstrap resampling',
            'release_inputs_verified':count,'tables':11,'figures':0 if args.tables_only else 12,
            'outputs':[{'path':p.relative_to(output).as_posix(),'sha256':sha(p)} for p in sorted(written)]}
    (output/'reproduction_manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('AGGREGATE_REPRODUCTION_COMPLETE',flush=True)

if __name__=='__main__':main()
