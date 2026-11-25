"""Aggregate-only point previews from the pinned completed big readout run."""
import csv
import hashlib
import json
from pathlib import Path
import runpy

REV = Path('/data/admin/fengshuo/ehr_fm_20261102/revision_v4_20261122')
RUN = REV / 'engine/results/prediction_runs/big_readout_frozen20_v4_2-cd45103e235da78d'
GATE = REV / 'temporal/results/eligibility/status_consistent_v4_1/landmark_v4_1'
OUT = REV / 'big_readout/point_preview'
EXPECTED = 'ee1c581293fd337f96e2cd3ea299e910eabb863db28795c86e2e810e1125d32a'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import pandas as pd
    import numpy as np
    manifest_path = RUN / 'prediction_manifest.json'
    if sha(manifest_path) != EXPECTED or OUT.exists():
        raise ValueError('Changed manifest or existing output')
    manifest = json.loads(manifest_path.read_text())
    ns = runpy.run_path(str(REV / 'engine/code/paired_metrics_v4.py'), run_name='big_preview_metrics')
    frames, rows = {}, []
    for entry in manifest['predictions']:
        if entry.get('n_predictions', 0) <= 0:
            continue
        db, analysis = entry['db'], entry['analysis_id']
        path = Path(entry['prediction_file'])
        hashes = ns['current_eligibility_hashes'](db, GATE)
        ns['validate_manifest_entry'](entry, analysis, db, path, hashes)
        if path not in frames:
            frames[path] = ns['validate_predictions'](pd.read_parquet(path), analysis)
        subset = frames[path].loc[frames[path].db.astype(str) == db]
        if len(subset) != entry['n_prediction_rows'] or ns['sha256_ordered_prediction_rows'](subset) != entry['ordered_row_digest_sha256']:
            raise ValueError('Prediction identity digest mismatch')
        scopes = [('all_audited_evaluation', subset)]
        if db == 'eicu':
            scopes.append(('native_split2_sensitivity', subset.loc[subset.split.astype(int) == 2]))
        for scope, selected in scopes:
            y = selected.y.to_numpy(dtype=np.int8)
            metrics = ns['weighted_metrics'](y, selected.p.to_numpy(dtype=float))
            rows.append({'db': db, 'analysis_id': analysis, 'scope': scope,
                         'stays': len(selected), 'patients': selected.subject.astype(str).nunique(),
                         'events': int(y.sum()), **metrics,
                         'ci_status': 'point_preview_only_intervals_pending',
                         'pretraining_limit': 'target features exposed; historical members not fully bound; native split2 does not prove clean holdout'})
        print('BIG_POINT_PREVIEW_COMPLETE ' + db + '/' + analysis, flush=True)
    if len(rows) != 9:
        raise ValueError('Incomplete point preview grid')
    OUT.mkdir(parents=True)
    with (OUT / 'big_point_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / 'preview_audit.json').write_text(json.dumps({'status': 'points_only_intervals_pending',
        'rows': len(rows), 'manifest_sha256': EXPECTED, 'manifest': str(manifest_path),
        'scope_note': 'Original input versions, feature-exposed configurations; no scaling or clean zero-exposure inference'}, indent=2))


if __name__ == '__main__':
    main()
