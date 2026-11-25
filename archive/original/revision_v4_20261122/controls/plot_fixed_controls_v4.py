"""Plot aggregate fixed-head and random-initialization controls.

Build anew under the v4.2 figure contract; prior figures supply style only.
All uncertainty is read from the completed paired patient bootstrap report.
No raw/patient-level inputs, fitting, filtering by performance, or simulation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.linewidth': 0.7,
    'pdf.fonttype': 42, 'svg.fonttype': 'none',
})

COLORS = {'within': '#558A82', 'donor': '#BF7853', 'random': '#666E9B'}
LABELS = {'within': 'Within-stay shuffle', 'donor': 'Different-patient values',
          'random': 'Random initialization'}
METRICS = ('auroc', 'average_precision', 'brier', 'logloss')


def read_rows(path: Path):
    report = json.loads(path.read_text(encoding='utf-8'))
    if report.get('status') != 'analysis_complete':
        raise ValueError('The aggregate control report is not complete')
    rows = []
    for db in ('mimic', 'eicu'):
        fixed = report['controls']['control_a_fixed_main20k_head'][db]
        comparisons = [(name, value['aggregate'])
                       for name, value in fixed['conditions'].items()]
        random = report['controls']['control_b_random_initialized_same_architecture'][db]
        comparisons.append(('random', random['random_minus_trained_paired_patient_bootstrap']))
        for condition, result in comparisons:
            if result['bootstrap_replicates'] < 1000:
                raise ValueError('Insufficient patient bootstrap draws')
            for metric in METRICS:
                lo, hi = result['paired_patient_bootstrap_95pct_ci'][metric]
                point = result['change_alternative_minus_base'][metric]
                values = [lo, hi, point, result['base_metrics'][metric],
                          result['alternative_metrics'][metric]]
                if not np.isfinite(values).all() or lo > hi:
                    raise ValueError(f'Invalid metric/CI: {db}/{condition}/{metric}')
                rows.append({
                    'cohort': db, 'condition': condition, 'metric': metric,
                    'base': result['base_metrics'][metric],
                    'alternative': result['alternative_metrics'][metric],
                    'delta_alternative_minus_base': point, 'ci95_lo': lo, 'ci95_hi': hi,
                    'n_stays': result['n_rows'], 'n_patients': result['n_patients'],
                    'bootstrap_draws': result['bootstrap_replicates'],
                    'resampling_unit': 'patient',
                    'source_path_in_json': ('controls/control_a_fixed_main20k_head/' + db
                        + '/conditions/' + condition + '/aggregate' if condition != 'random'
                        else 'controls/control_b_random_initialized_same_architecture/'
                        + db + '/random_minus_trained_paired_patient_bootstrap'),
                })
    if len(rows) != 24:
        raise ValueError(f'Expected all 24 metric contrasts, got {len(rows)}')
    return rows


def panel(ax, rows, *, metric, conditions, letter, title, xlim):
    lookup = {(r['cohort'], r['condition'], r['metric']): r for r in rows}
    ylabels, y = [], 0
    for db in ('mimic', 'eicu'):
        for condition in conditions:
            r = lookup[(db, condition, metric)]
            marker = 'o' if db == 'mimic' else 's'
            point = r['delta_alternative_minus_base']
            lo, hi = r['ci95_lo'], r['ci95_hi']
            ax.plot([lo, hi], [y, y], color=COLORS[condition], lw=1.4,
                    solid_capstyle='round', zorder=2)
            ax.scatter([point], [y], marker=marker, s=23, color=COLORS[condition],
                       edgecolors='white', linewidths=.5, zorder=3)
            ylabels.append(('MIMIC-IV' if db == 'mimic' else 'eICU')
                           + (' / within' if condition == 'within' else
                              ' / donor' if condition == 'donor' else ''))
            y += 1
        y += .4
    positions = []
    pos = 0
    for db in ('mimic', 'eicu'):
        for _ in conditions:
            positions.append(pos)
            pos += 1
        pos += .4
    ax.set_yticks(positions, labels=ylabels)
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.axvline(0, color='#8A9399', lw=.8, ls='--', zorder=1)
    ax.grid(axis='x', color='#E1E6E8', lw=.5)
    ax.set_xlabel('Change in AUROC' if metric == 'auroc' else 'Change in Brier score')
    ax.set_title(title, loc='left', pad=9)
    ax.text(-.35, 1.08, letter, transform=ax.transAxes, fontweight='bold', fontsize=9)
    ax.tick_params(axis='y', length=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    args = parser.parse_args()
    rows = read_rows(args.input)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source = args.out_dir / 'fig2_source_data.csv'
    with source.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(2, 2, figsize=(7.2047244, 5.0))  # 183 x 127 mm
    fig.subplots_adjust(left=.21, right=.98, top=.88, bottom=.12, wspace=.77, hspace=.85)
    panel(axes[0, 0], rows, metric='auroc', conditions=('within', 'donor'),
          letter='a', title='Fixed head: discrimination', xlim=(-.28, .025))
    panel(axes[0, 1], rows, metric='brier', conditions=('within', 'donor'),
          letter='b', title='Fixed head: probability error', xlim=(-.002, .026))
    panel(axes[1, 0], rows, metric='auroc', conditions=('random',),
          letter='c', title='Random vs pretrained encoder', xlim=(-.016, .009))
    panel(axes[1, 1], rows, metric='brier', conditions=('random',),
          letter='d', title='Random vs pretrained encoder', xlim=(-.002, .021))
    handles = [Line2D([0], [0], color=COLORS[c], marker='o', lw=1.2, markersize=4,
                      label=LABELS[c]) for c in ('within', 'donor', 'random')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .985),
               ncol=3, frameon=False, columnspacing=1.5, handlelength=1.7)
    fig.savefig(args.out_dir / 'fig2.pdf', facecolor='white')
    fig.savefig(args.out_dir / 'fig2.svg', facecolor='white')
    fig.savefig(args.out_dir / 'fig2.png', dpi=300, facecolor='white')
    plt.close(fig)
    note = {
        'status': 'rendered_pending_visual_qa', 'figure': 'fig2', 'backend': 'python',
        'source_report_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'source_data_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_data_rows': len(rows), 'excluded_rows': 0,
        'plotted_metrics': ['auroc', 'brier'],
        'unplotted_metrics_retained_in_source': ['average_precision', 'logloss'],
        'size_mm': [183, 127], 'png_dpi': 300, 'resampling_unit': 'patient',
        'ci': '1000 paired patient-cluster percentile bootstrap draws; fixed fitted heads/encoder and one random initialization',
        'head_protocol': 'a,b fixed pretrained source head; c,d separately source-fitted C1 readouts for pretrained and random encoders',
        'cohorts': 'MIMIC-IV split2 and original eICU input version; not corrected eICU_semantic_v4_2',
        'interpretation': 'Negative AUROC change is worse; positive Brier change is worse. Interventions are representation-response controls, not clinical causal effects.',
        'multiplicity': 'Descriptive paired intervals; no multiple-comparison correction or p-value stars',
        'image_integrity': 'No experimental images, cropping, intensity adjustment, or synthetic observations',
    }
    (args.out_dir / 'fig2_qa.json').write_text(json.dumps(note, indent=2), encoding='utf-8')
    print('FIG2_RENDER_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
