"""Draw revision figures from a verified aggregate-only result bundle.

This module performs no model fitting or statistical selection. The bundle
must state the source rows and uncertainty scope for every displayed panel.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = ['#335C81', '#B96D40', '#4F817C', '#806B9C', '#686868', '#B2913D']
plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.spines.right': False, 'axes.spines.top': False,
    'axes.linewidth': .65, 'lines.linewidth': 1.2,
    'legend.frameon': False, 'legend.fontsize': 7,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'savefig.transparent': False,
})


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def finite(value):
    return value is not None and np.isfinite(float(value))


def panel_label(ax, index):
    ax.text(-.16, 1.08, chr(97 + index), transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='left')


def clean(ax):
    ax.tick_params(direction='out', length=3, width=.65, pad=3)
    ax.grid(False)


def checked_series(series, required):
    for entry in series:
        for key in required:
            if key not in entry:
                raise ValueError(f'missing plotted field {key}')
        if not entry.get('source_selector'):
            raise ValueError('every plotted series requires a source-row selector')


def calibration_grid(spec):
    if len(spec['panels']) != 4:
        raise ValueError('calibration grid requires the four locked panels')
    fig, axes = plt.subplots(2, 2, figsize=(170 / 25.4, 170 / 25.4))
    handles, labels = {}, []
    for index, (ax, panel) in enumerate(zip(axes.flat, spec['panels'])):
        checked_series(panel['series'], ['label', 'x', 'y', 'low', 'high'])
        max_risk = float(panel['axis_max'])
        ax.plot([0, max_risk], [0, max_risk], color='#969696', lw=.8, ls='--')
        note = []
        for j, curve in enumerate(panel['series']):
            color = curve.get('color', COLORS[j])
            x, y = np.asarray(curve['x'], float), np.asarray(curve['y'], float)
            if len(x) != len(y):
                raise ValueError('calibration grid and curve lengths differ')
            if len(x):
                if not (np.isfinite(x).all() and np.isfinite(y).all()):
                    raise ValueError('nonfinite published calibration curve')
                line, = ax.plot(x, y, color=color, label=curve['label'])
                lo = np.asarray(curve['low'], float)
                hi = np.asarray(curve['high'], float)
                if len(lo) == len(x) and len(hi) == len(x):
                    valid = np.isfinite(lo) & np.isfinite(hi)
                    ax.fill_between(x, lo, hi, where=valid, color=color, alpha=.15,
                                    linewidth=0, interpolate=False)
                for risk in curve.get('decile_mean_risks', []):
                    ax.plot([risk, risk], [.005 + j * .012, .014 + j * .012],
                            color=color, lw=.8, transform=ax.get_xaxis_transform())
            else:
                if not (finite(curve.get('mean_predicted')) and finite(curve.get('observed'))):
                    raise ValueError('constant score requires its actual mean and event rate')
                line, = ax.plot(curve['mean_predicted'], curve['observed'], 'o',
                                ms=4, color=color, label=curve['label'])
            if curve['label'] not in handles:
                handles[curve['label']] = line
                labels.append(curve['label'])
            note.append(f"{curve['label']}: mean risk {curve['mean_predicted']:.3f}")
        ax.set(xlim=(0, max_risk), ylim=(0, max_risk),
               xlabel='Predicted risk', ylabel='Observed event probability',
               title=panel['title'])
        ax.text(.03, .97,
                f"n = {panel['n_stays']}; deaths = {panel['events']}\n"
                f"Observed = {panel['observed']:.3f}\n" + '\n'.join(note),
                transform=ax.transAxes, fontsize=6.7, ha='left', va='top')
        clean(ax)
        panel_label(ax, index)
    fig.legend([handles[key] for key in labels], labels, loc='upper center',
               ncol=len(labels), bbox_to_anchor=(.54, .997))
    fig.subplots_adjust(left=.12, right=.97, bottom=.09, top=.91,
                        hspace=.48, wspace=.38)
    return fig


def forest_grid(spec):
    panels = spec['panels']
    rows = max(len(panel['rows']) for panel in panels)
    height = min(215, max(88, 6.2 * rows + 38))
    fig, axes = plt.subplots(1, len(panels), squeeze=False,
                             figsize=(170 / 25.4, height / 25.4))
    handles, labels = {}, []
    for index, (ax, panel) in enumerate(zip(axes.flat, panels)):
        ax.axvline(float(panel.get('reference', 0)), color='#929292', ls='--', lw=.8)
        ylabels = []
        for i, row in enumerate(panel['rows']):
            ylabels.append(row['label'])
            groups = row['series']
            checked_series(groups, ['label', 'estimate', 'low', 'high'])
            offset = np.linspace(-.22, .22, len(groups)) if len(groups) > 1 else [0]
            for j, (entry, dy) in enumerate(zip(groups, offset)):
                color = entry.get('color', COLORS[j % len(COLORS)])
                yy = i + dy
                if not finite(entry['estimate']):
                    ax.text(.98, yy, entry.get('missing_label', 'Unavailable'),
                            transform=ax.get_yaxis_transform(), ha='right', va='center',
                            color='#777777', fontsize=6.5)
                    continue
                estimate = float(entry['estimate'])
                if finite(entry['low']) and finite(entry['high']):
                    low, high = float(entry['low']), float(entry['high'])
                    if low > estimate or high < estimate:
                        # Percentile intervals may not contain the point; draw
                        # actual endpoints rather than invent symmetric errors.
                        ax.plot([low, high], [yy, yy], color=color, lw=1)
                    else:
                        ax.errorbar(estimate, yy,
                                    xerr=[[estimate - low], [high - estimate]],
                                    color=color, lw=1, capsize=1.5, fmt='none')
                point, = ax.plot(estimate, yy, marker=entry.get('marker', 'o'),
                                 color=color, ms=3.7, lw=0)
                if entry['label'] not in handles:
                    handles[entry['label']] = point
                    labels.append(entry['label'])
        ax.set_yticks(np.arange(len(ylabels)))
        ax.set_yticklabels(ylabels)
        ax.set(ylim=(len(ylabels) - .45, -.6), xlabel=panel['xlabel'], title=panel['title'])
        if panel.get('xlim'):
            ax.set_xlim(panel['xlim'])
        if not panel.get('show_labels', True):
            ax.set_yticklabels([])
        clean(ax)
        panel_label(ax, index)
    if labels:
        fig.legend([handles[key] for key in labels], labels, loc='upper center',
                   ncol=min(4, len(labels)), bbox_to_anchor=(.56, .995))
    fig.subplots_adjust(left=float(spec.get('left_margin', .27)), right=.98,
                        bottom=.14 if height < 120 else .09, top=.89,
                        wspace=float(spec.get('wspace', .65)))
    return fig


def budget_grid(spec):
    if len(spec['panels']) != 4:
        raise ValueError('paired gap budget display requires four panels')
    fig, axes = plt.subplots(2, 2, figsize=(170 / 25.4, 155 / 25.4))
    handles, labels = {}, []
    for index, (ax, panel) in enumerate(zip(axes.flat, spec['panels'])):
        ax.axhline(0, color='#929292', ls='--', lw=.8)
        checked_series(panel['series'], ['label', 'budgets', 'estimate', 'low', 'high'])
        for j, entry in enumerate(panel['series']):
            xx = np.asarray(entry['budgets'], int)
            if not (np.isfinite(xx).all() and (xx > 0).all()):
                raise ValueError('log-budget coordinates must be finite and positive')
            yy = np.asarray(entry['estimate'], float)
            lo, hi = np.asarray(entry['low'], float), np.asarray(entry['high'], float)
            if not (len(xx) == len(yy) == len(lo) == len(hi)):
                raise ValueError('budget-series lengths differ')
            color = entry.get('color', COLORS[j])
            line, = ax.plot(xx, yy, 'o-', color=color, ms=3.5, lw=1)
            ax.fill_between(xx, lo, hi, color=color, alpha=.13, linewidth=0)
            if entry['label'] not in handles:
                handles[entry['label']] = line
                labels.append(entry['label'])
        ax.set_xscale('log')
        ax.set_xticks([50, 200, 500, 2000], ['50', '200', '500', '2000'])
        ax.minorticks_off()
        ax.set(xlabel='Unique adaptation patients', ylabel=panel['ylabel'], title=panel['title'])
        clean(ax)
        panel_label(ax, index)
    fig.legend([handles[key] for key in labels], labels, loc='upper center',
               ncol=3, bbox_to_anchor=(.55, .998))
    fig.subplots_adjust(left=.14, right=.97, bottom=.10, top=.90, wspace=.48, hspace=.48)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    document = json.loads(args.bundle.read_text(encoding='utf-8'))
    if document.get('status') != 'analysis_inputs_verified':
        raise ValueError('final plotting requires completed, verified aggregate inputs')
    figures = document['figures']
    if sorted(str(f['number']) for f in figures if not str(f['number']).startswith('S')) != ['1','2','3','4','5','6']:
        raise ValueError('six main figure identities are required')
    args.output.mkdir(parents=True, exist_ok=True)
    builders = {'calibration_grid': calibration_grid, 'forest_grid': forest_grid,
                'budget_grid': budget_grid}
    manifest = []
    for spec in figures:
        if not spec.get('source_files') or not spec.get('uncertainty_scope'):
            raise ValueError('figure lacks source data or uncertainty definition')
        for source in spec['source_files']:
            source_path = Path(source['path'])
            if sha256(source_path) != source['sha256']:
                raise ValueError('aggregate source file differs from locked bundle')
        fig = builders[spec['kind']](spec)
        base = args.output / f"Figure{spec['number']}"
        fig.savefig(base.with_suffix('.pdf'), bbox_inches=None)
        fig.savefig(base.with_suffix('.svg'), bbox_inches=None)
        fig.savefig(base.with_suffix('.png'), dpi=300, bbox_inches=None)
        fig.savefig(base.with_suffix('.tiff'), dpi=600, bbox_inches=None,
                    pil_kwargs={'compression': 'tiff_lzw'})
        bounds = fig.get_size_inches().tolist()
        if bounds[1] * 25.4 > 225.01:
            raise ValueError('figure exceeds the locked maximum height')
        pdf = base.with_suffix('.pdf')
        if pdf.stat().st_size >= 10 * 1024**2:
            raise ValueError('figure PDF exceeds 10 MiB')
        manifest.append({'number': str(spec['number']), 'kind': spec['kind'],
                         'width_mm': bounds[0] * 25.4, 'height_mm': bounds[1] * 25.4,
                         'pdf_sha256': sha256(pdf), 'source_files': spec['source_files'],
                         'uncertainty_scope': spec['uncertainty_scope'],
                         'status': 'rendered_pending_visual_and_scientific_QA'})
        plt.close(fig)
        print(f"RENDERED Figure{spec['number']}", flush=True)
    (args.output / 'figure_render_manifest.json').write_text(
        json.dumps({'bundle_sha256': sha256(args.bundle), 'figures': manifest}, indent=2),
        encoding='utf-8')


if __name__ == '__main__':
    main()
