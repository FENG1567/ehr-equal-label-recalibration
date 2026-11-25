"""Hospital figure from complete aggregate tables, with a frozen display rule.

Quantitative grid: 24 largest sites by evaluation-stay count, plus all-site
paired probability error. No selection by performance or significance.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

MODELS = ('main20k_real', 'main20k_constant', 'conventional_lr_value_plus_pattern')
COLORS = dict(zip(MODELS, ('#3E7893', '#D58A4B', '#4A8F7B')))
LABELS = dict(zip(MODELS, ('FM: real input', 'FM: constant input', 'LR: values + pattern')))
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7,
    'pdf.fonttype': 42, 'svg.fonttype': 'none',
})


def number(row, key):
    value = row.get(key, '')
    return float(value) if value not in ('', None) else float('nan')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete':
        raise ValueError('Hospital analysis is incomplete')
    with args.data.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    sites = {}
    for row in rows:
        if row['scope'] != 'selected_hospital_all_stays':
            raise ValueError('Missing-ID aggregates must be explicitly separated before plotting')
        key = (row['model_id'], row['hospitalid'])
        if key in sites:
            raise ValueError('Duplicate model/site row')
        sites[key] = row
    real = {site: row for (model, site), row in sites.items() if model == MODELS[0]}
    n_sites = len(real)
    if n_sites != manifest['selected_hospital_ids_with_current_predictions'] or n_sites != 206:
        raise ValueError('The current figure contract requires all 206 audited sites')
    for model in MODELS:
        if {site for m, site in sites if m == model} != set(real):
            raise ValueError('Model hospital rosters differ')
        for site, reference in real.items():
            row = sites[(model, site)]
            if any(row[k] != reference[k] for k in ('n_stays', 'n_patients', 'events', 'nonevents')):
                raise ValueError('Within-site model populations differ')
            if not math.isfinite(number(row, 'brier')):
                raise ValueError('A selected site lacks its Brier score')
    selected = sorted(real, key=lambda site: (-int(real[site]['n_stays']), int(site)))[:24]
    if len(selected) != 24:
        raise ValueError('Not enough sites for the requested 24-site display')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.out_dir / 'fig5_source_data.csv'
    with source_path.open('w', newline='', encoding='utf-8') as stream:
        fields = list(rows[0]) + ['displayed_in_forest']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{**r, 'displayed_in_forest': r['hospitalid'] in selected} for r in rows])
    fig = plt.figure(figsize=(7.2047244, 6.1023622))  # 183 x 155 mm
    grid = fig.add_gridspec(1, 2, width_ratios=(1.45, 1), left=.21, right=.98,
                           bottom=.12, top=.88, wspace=.38)
    forest = fig.add_subplot(grid[0, 0])
    scatter = fig.add_subplot(grid[0, 1])
    nonestimable = []
    lower, upper = [0.0], [1.1]
    markers = ('o', 's', '^')
    for mi, model in enumerate(MODELS):
        for yi, site in enumerate(selected):
            row = sites[(model, site)]
            value = number(row, 'calibration_slope')
            lo, hi = number(row, 'calibration_slope_ci_95_lo'), number(row, 'calibration_slope_ci_95_hi')
            y = yi + (mi - 1) * .23
            if row['calibration_slope_ci_status'] != 'ok' or not np.isfinite([value, lo, hi]).all():
                nonestimable.append({'model': model, 'hospitalid': site,
                                     'status': row['calibration_slope_ci_status']})
                continue
            if number(row, 'calibration_slope_ci_bootstrap_draws_requested') < 1000:
                raise ValueError('A plotted calibration slope lacks the required bootstrap')
            forest.plot([lo, hi], [y, y], color=COLORS[model], lw=.7, alpha=.9)
            forest.plot(value, y, marker=markers[mi], ms=3, color=COLORS[model],
                        markeredgecolor='white', markeredgewidth=.25)
            lower.append(lo)
            upper.append(hi)
    forest.axvline(1, ls='--', lw=.8, color='#7F8A90')
    forest.set_yticks(range(24), [f"{site} ({int(real[site]['n_stays']):,})" for site in selected])
    forest.set_ylim(23.8, -.8)
    forest.set_xlim(min(lower) - .04, max(upper) + .06)
    forest.set_xlabel('Calibration slope (95% patient-cluster CI)')
    forest.set_ylabel('Hospital ID (eligible stays)')
    forest.set_title('Largest 24 hospitals by sample count', loc='left', pad=9)
    forest.text(-.50, 1.035, 'a', transform=forest.transAxes, fontsize=9, fontweight='bold')
    forest.grid(axis='x', color='#E1E6E8', lw=.5)
    forest.tick_params(axis='y', length=0)
    if nonestimable:
        forest.text(.98, .01, 'Unestimable entries retained; see source data',
                    transform=forest.transAxes, ha='right', fontsize=7)

    x = np.array([number(real[s], 'brier') for s in real])
    y = np.array([number(sites[(MODELS[2], s)], 'brier') for s in real])
    is_selected = np.array([s in selected for s in real])
    limit = max(float(x.max()), float(y.max())) * 1.08
    scatter.plot([0, limit], [0, limit], color='#8A949A', lw=.8, ls='--')
    scatter.scatter(x[~is_selected], y[~is_selected], s=12, color='#83959E', alpha=.7,
                    edgecolor='white', linewidth=.3)
    scatter.scatter(x[is_selected], y[is_selected], s=19, color='#3E7893', alpha=.9,
                    edgecolor='#202B30', linewidth=.4)
    scatter.set_xlim(0, limit)
    scatter.set_ylim(0, limit)
    scatter.set_aspect('equal', adjustable='box')
    scatter.set_xlabel('FM real-input Brier score')
    scatter.set_ylabel('Values + pattern LR Brier score')
    scatter.set_title(f'All {n_sites} hospitals', loc='left', pad=9)
    scatter.text(-.28, 1.08, 'b', transform=scatter.transAxes, fontsize=9, fontweight='bold')
    scatter.text(.03, .97, 'Below diagonal: lower error with LR', ha='left', va='top',
                 fontsize=7, transform=scatter.transAxes)
    scatter.grid(color='#E1E6E8', lw=.4)
    handles = [Line2D([0], [0], color=COLORS[m], marker=markers[i], ms=3.5,
                      lw=.8, label=LABELS[m]) for i, m in enumerate(MODELS)]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.52, .97),
               ncol=3, frameon=False, columnspacing=1.4, handlelength=1.6)
    fig.text(.21, .035, 'Original eICU input version; paired within-site comparisons. '
             'Outlined points identify the 24 displayed sites.', fontsize=7)
    fig.savefig(args.out_dir / 'fig5.pdf', facecolor='white')
    fig.savefig(args.out_dir / 'fig5.svg', facecolor='white')
    fig.savefig(args.out_dir / 'fig5.png', dpi=300, facecolor='white')
    plt.close(fig)
    qa = {
        'status': 'rendered_pending_visual_qa', 'backend': 'python', 'figure': 'fig5',
        'input_csv_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'analysis_manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'source_data_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'full_source_rows': len(rows), 'n_sites_all_models': n_sites,
        'forest_site_count': len(selected), 'forest_site_ids': selected,
        'forest_selection': 'largest eligible stay counts, ascending numeric ID tie break; no effect-based selection',
        'forest_unestimable_entries_retained': nonestimable,
        'brier_scatter_sites': len(x), 'brier_scatter_excluded_sites': 0,
        'within_site_ci': '1000 patient-cluster percentile draws; conditional on fixed predictors',
        'cross_hospital_patients': manifest['patient_hospital_crossing']['n_patients_crossing_hospitals'],
        'hospital_summary_ci_scope': manifest['bootstrap_scope'],
        'multiplicity': 'descriptive unadjusted intervals; no significance stars',
        'size_mm': [183, 155], 'minimum_font_pt': 7, 'png_dpi': 300,
        'clinical_availability_verified': False,
        'input_version': 'original eICU core, not repaired laboratory-input sensitivity',
    }
    (args.out_dir / 'fig5_qa.json').write_text(json.dumps(qa, indent=2), encoding='utf-8')
    print('FIG5_RENDER_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
