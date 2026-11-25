"""Bind completed aggregate calibration and clinical results for publication."""
from pathlib import Path
import json
import runpy
import sys

R5 = Path('/data/admin/fengshuo/ehr_fm_20261102/revision_v5_20261123')
OUTPUT = R5 / 'publication/build_20261124'
OUTPUT.mkdir(parents=True, exist_ok=True)
for name in ('calibration_figure_specs.json', 'table_data.json'):
    if (OUTPUT / name).exists():
        raise RuntimeError('Refusing to overwrite an existing bound publication input: ' + name)

def execute(name, arguments):
    path = R5 / 'scripts' / name
    sys.argv = [str(path), *map(str, arguments)]
    runpy.run_path(str(path), run_name='__main__')

execute('build_calibration_figure_specs_v5.py', [
    R5 / 'calibration/v5_20261123', R5 / 'sites/v5_20261123',
    OUTPUT / 'calibration_figure_specs.json',
])
execute('build_tables_v5.py', [
    '--baseline', R5 / 'scripts/table1_source_data.csv',
    '--calibration', R5 / 'calibration/v5_20261123/global_50_500_summary.csv',
    '--clinical', R5 / 'cohorts/eligible_clinical_20261123T172226Z_3852910/clinical_summary.csv',
    '--output', OUTPUT / 'table_data.json',
])
print(json.dumps({'stage': 'calibration_publication_inputs_bound', 'output': str(OUTPUT)}), flush=True)
