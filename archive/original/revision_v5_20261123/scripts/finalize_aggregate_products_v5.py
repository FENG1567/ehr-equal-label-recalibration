"""Bind verified revision aggregates, then render the six main and four supplement figures."""
import argparse
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

R5=Path('/data/admin/fengshuo/ehr_fm_20261102/revision_v5_20261123')
S=R5/'scripts'
B=R5/'publication/build_20261124'
os.environ['MPLCONFIGDIR']=str(R5/'admin/runtime/matplotlib_cache')
os.environ['MPLBACKEND']='Agg'
p=argparse.ArgumentParser()
p.add_argument('--decoder-dir',required=True,type=Path)
p.add_argument('--bound-table-sha256')
p.add_argument('--stable-summary-sha256')
a=p.parse_args()
decoder=a.decoder_dir.resolve(strict=True)
if decoder.parent != (R5/'decoding').resolve() or not decoder.name.startswith('direct_token_summary_'):
    raise ValueError('Unexpected decoder result directory')
if json.loads((decoder/'numeric_token_decoder_manifest.json').read_text())['status']!='complete':
    raise ValueError('Decoder is incomplete')

def call(name,args):
    print('PUBLICATION_STAGE_START '+name,flush=True)
    sys.argv=[str(S/name),*map(str,args)]
    try:
        runpy.run_path(str(S/name),run_name='__main__')
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    print('PUBLICATION_STAGE_COMPLETE '+name,flush=True)

for filename,digest,script in [
 ('table_data_r2.json',a.bound_table_sha256,'add_raw_reference_table_v5.py'),
 ('stable_site_slope_summary.csv',a.stable_summary_sha256,'stable_site_summary_v5.py')]:
    if digest:
        if hashlib.sha256((B/filename).read_bytes()).hexdigest()!=digest:
            raise ValueError('Previously bound publication product changed: '+filename)
        print('PUBLICATION_PRODUCT_PRESERVED '+filename+' sha256='+digest,flush=True)
    else:
        call(script,[])
call('build_controls_figure_specs_v5.py',[
 '--controls-score-dir',R5/'controls/representation_controls_v5rc_20261123T175424Z_4058026/scores',
 '--decoder-dir',decoder,'--readout-dir',R5/'readout/source_c_selection_v5rc_20261123T175424Z_4058026',
 '--output',B/'controls_figure_specs.json'])
call('build_sensitivity_figure_specs_v5.py',[])
call('assemble_publication_figures_v5.py',[
 '--calibration-spec',B/'calibration_figure_specs_r2.json',
 '--controls-spec',B/'controls_figure_specs.json',
 '--sensitivity-spec',B/'sensitivity_figure_specs.json',
 '--output',B/'publication_figures_v5_bundle.json'])
call('validate_figure.py',[S/'plot_v5_figures.py','--json'])
call('plot_v5_figures.py',[B/'publication_figures_v5_bundle.json',B/'publication_figures_v5'])
call('stage_new_supplement_v5.py',['--decoder-dir',decoder,'--output',B/'new_supplement_aggregate'])
print('PUBLICATION_AGGREGATES_COMPLETE',flush=True)
