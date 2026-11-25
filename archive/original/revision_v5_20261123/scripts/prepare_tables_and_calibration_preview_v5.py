"""Complete aggregate table binding and the three already verified figure renders."""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import numpy
import scipy
import sklearn
print(json.dumps({'stage': 'publication_cpu_runtime_versions',
                  'numpy': numpy.__version__, 'scipy': scipy.__version__,
                  'sklearn': sklearn.__version__}), flush=True)

R5 = Path('/data/admin/fengshuo/ehr_fm_20261102/revision_v5_20261123')
root = R5 / 'publication/build_20261124'
cache = R5 / 'admin/runtime/matplotlib_cache'
cache.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(cache)
os.environ['MPLBACKEND'] = 'Agg'

def execute(name, arguments):
    path = R5 / 'scripts' / name
    sys.argv = [str(path), *map(str, arguments)]
    try:
        runpy.run_path(str(path), run_name='__main__')
    except SystemExit as result:
        if result.code not in (None, 0):
            raise

execute('build_tables_v5.py', [
    '--baseline', R5 / 'scripts/table1_source_data.csv',
    '--calibration', R5 / 'calibration/v5_20261123/global_50_500_summary.csv',
    '--clinical', R5 / 'cohorts/eligible_clinical_20261123T172226Z_3852910/clinical_summary.csv',
    '--output', root / 'table_data.json'])
audit = io.StringIO()
with contextlib.redirect_stdout(audit):
    execute('validate_figure.py', [R5 / 'scripts/plot_v5_figures.py', '--json'])
(root / 'plot_source_preflight.json').write_text(audit.getvalue())
execute('render_calibration_preview_v5.py', [])
print(json.dumps({'stage': 'tables_and_calibration_preview_complete',
                  'output': str(root), 'six_figure_release': False}), flush=True)
