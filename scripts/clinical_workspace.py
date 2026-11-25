#!/usr/bin/env python3
"""Inspect prerequisites or stage original code in a new private workspace."""
from __future__ import annotations
import argparse,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--inventory',action='store_true')
    g.add_argument('--prepare',type=Path)
    g.add_argument('--check',type=Path)
    a=p.parse_args()
    spec=json.loads((ROOT/'config/clinical_stages.json').read_text(encoding='utf-8'))
    if a.inventory:
        print(json.dumps(spec,indent=2));return 0
    target=(a.prepare or a.check).resolve()
    original=Path(spec['original_root']).resolve()
    if a.prepare:
        if target==original or original in target.parents or target in original.parents or target==ROOT or ROOT in target.parents or target in ROOT.parents:
            raise ValueError('Use a new private directory outside the repository and production root')
        if target.exists() and any(target.iterdir()):raise FileExistsError('Private staging directory must be empty')
        target.mkdir(parents=True,exist_ok=True)
        for src in (ROOT/'archive/original').rglob('*.py'):
            dest=target/src.relative_to(ROOT/'archive/original')
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
        shutil.copy2(ROOT/'config/variable_mapping.csv',target/'mapping_table.csv')
        (target/'RECONSTRUCTION_NOT_RUN.txt').write_text('Original code staged only. Supply authorized private data and follow docs/CLINICAL_REPRODUCTION.md. Original absolute paths and hash gates remain. No clinical analysis has run.\n')
        print(json.dumps({'status':'code_staged_only','workspace':str(target)},indent=2));return 0
    rows=[]
    for stage in spec['stages']:
        rows.append({'stage':stage['id'],'script_present':(target/stage['script']).is_file(),
                     'missing_inputs':[x for x in stage['needs'] if not (target/x).exists()]})
    print(json.dumps({'status':'prerequisites_missing' if any(x['missing_inputs'] or not x['script_present'] for x in rows) else 'path_presence_only',
                      'warning':'Existence is not scientific validity or hash verification. No models or patient files were read.',
                      'workspace':str(target),'stages':rows},indent=2))
    return 2 if any(x['missing_inputs'] or not x['script_present'] for x in rows) else 0

if __name__=='__main__':sys.exit(main())
