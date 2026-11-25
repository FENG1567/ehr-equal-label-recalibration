"""Bind completed clean-pretraining, H24 and ICU-bridge aggregate figures."""
import csv
import hashlib
import json
from pathlib import Path

R5 = Path('/data/admin/fengshuo/ehr_fm_20261102/revision_v5_20261123')
clean = R5 / 'results/clean_pretraining/run_20261123_192508_879635_unicode_20261123T132809Z/evaluation_results.json'
h24 = R5 / 'temporal/v5_temporal_bridge_audit_20261123T151936Z_2833855/h24_performance_metrics_paired_losses.csv'
bridge = R5 / 'bridge/v5_icu_endpoint_bridge_20261123T165428Z_3598145/eicu_icu_paired_patient_bootstrap.csv'
expected = {
    clean: 'a35b65fefac6af2a47c3e5e7086e9402f8b2012f610a60d46706ff40eca25c4a',
    h24: '717a47b876db13cbd8dd2d6dbffe69f28e095511c950a65af594d743987f196d',
    bridge: '0e3990ef522e1083081079cf08f470836bfb6b0ec389fae7f49f9d58c8db35aa',
}
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
for path, digest in expected.items():
    if sha(path) != digest:
        raise ValueError('Completed aggregate identity changed: ' + path.name)
def source(path):
    return [{'path': str(path), 'sha256': expected[path]}]
def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))
def one(rows, filters):
    selected = [row for row in rows if all(row.get(key) == str(value) for key, value in filters.items())]
    if len(selected) != 1:
        raise ValueError('Expected exactly one aggregate row: ' + str(filters))
    return selected[0]
def series(label, value, low, high, selector):
    return dict(label=label, estimate=float(value), low=float(low), high=float(high), source_selector=selector)
def spec(number, panels, path, uncertainty, claim, left=.29):
    return dict(number=number, kind='forest_grid', panels=panels, source_files=source(path),
                uncertainty_scope=uncertainty, claim=claim, left_margin=left, wspace=.75)

document = json.loads(clean.read_text())
if document['status'] != 'complete' or document.get('seed_averaged_predictions_used') is not False:
    raise ValueError('Clean-pretraining result is not complete or used seed-averaged predictions')
panels = []
contrasts = [('numeric_real_vs_constant_trained_constant', 'vs constant-trained'),
             ('numeric_real_vs_numeric_constant_input', 'vs constant input'),
             ('numeric_real_vs_random_initial_real', 'vs random')]
for metric, title in [('brier', 'Brier score'), ('logloss', 'Log loss')]:
    rows = []
    for cohort, label in [('mimic', 'MIMIC'), ('eicu', 'eICU'), ('nwicu', 'NWICU')]:
        for seed in ('20261123', '20261124', '20261125'):
            groups = []
            for contrast, contrast_label in contrasts:
                item = document['paired_patient_cluster_bootstrap']['results'][cohort][seed][contrast][metric]
                low, high = item['patient_cluster_bootstrap_95ci']
                groups.append(series(contrast_label, item['left_minus_right'], low, high,
                                     dict(cohort=cohort, seed=seed, contrast=contrast, metric=metric)))
            rows.append(dict(label=label + ' / ' + seed, series=groups))
    panels.append(dict(title=title, xlabel='Numeric-real minus comparator', rows=rows))
figures = [spec('S1', panels, clean,
    '95% patient-cluster percentile intervals conditional on each fitted seed/model; all three seeds shown; seed ranges are not confidence intervals',
    'Numeric inputs outperform the constant controls, while advantages over random initialization vary across seeds and target databases.')]

h24_rows = read_csv(h24)
panels = []
for metric, title in [('brier', 'Brier score'), ('logloss', 'Log loss')]:
    rows = []
    for subset, label in [('primary_reviewed_h24', 'Primary H24 set'),
                          ('no_later_exit_order_gap', 'No later exit-order gap')]:
        filters = dict(db='mimic', model='paired_fm_minus_lr', subset=subset,
                       paired_metric=metric, weighting='stay', estimand='stay_weighted_primary')
        item = one(h24_rows, filters)
        rows.append(dict(label=label, series=[series('FM minus V+P LR', item['fm_minus_lr'],
                         item['low95'], item['high95'], filters)]))
    panels.append(dict(title='MIMIC / ' + title, xlabel='FM minus V+P LR', rows=rows))
figures.append(spec('S3', panels, h24,
    '1000 patient-cluster draws of fixed paired scores; source split 2, not chronological validation; populations differ across restrictions',
    'Removing later administrative exit-order gaps changes the evaluated population and preserves the direction of the paired probability-score differences.'))

bridge_rows = read_csv(bridge)
panels = []
for metric, title in [('brier', 'Brier score'), ('logloss', 'Log loss')]:
    rows = []
    for contrast, label in [('historical hospital FM minus LR', 'Hospital-trained heads'),
                             ('new ICU FM minus V+P LR', 'ICU-trained heads')]:
        groups = []
        for method, method_label in [('raw', 'Raw'), ('platt', 'Platt / 500 labels')]:
            filters = dict(contrast=contrast, method=method, metric=metric, weighting='stay',
                           analysis_subset='all_assessment_patients')
            item = one(bridge_rows, filters)
            groups.append(series(method_label, item['fm_minus_lr'], item['low95'], item['high95'], filters))
        rows.append(dict(label=label, series=groups))
    panels.append(dict(title='eICU ICU death / ' + title, xlabel='FM minus V+P LR', rows=rows))
figures.append(spec('S4', panels, bridge,
    '1000 patient-cluster draws conditional on frozen source heads and one realized 500-patient adaptation set; one index episode per assessment patient',
    'An endpoint-specific secondary bridge in the same databases narrows some raw differences but leaves probability-score differences after one matched update.'))
output = R5 / 'publication/build_20261124/sensitivity_figure_specs.json'
if output.exists():
    raise RuntimeError('Refusing to overwrite previously bound sensitivity specifications')
output.write_text(json.dumps({'status': 'analysis_inputs_verified', 'figures': figures}, indent=2)+'\n')
print(json.dumps({'stage': 'sensitivity_figures_bound', 'output': str(output)}), flush=True)
