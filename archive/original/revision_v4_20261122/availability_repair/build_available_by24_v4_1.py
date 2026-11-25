"""Build the frozen-cohort MIMIC input-availability sensitivity dataset.

This is a data-only extraction. It never fits bins, opens a checkpoint, scores
a model, or changes the central registry. Raw CSV.GZ files are streamed once;
completed source tables are checkpointed so a restart can continue at the next
table. If interrupted mid-table, only that table's owned parts are rebuilt.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time

print('AVAILABILITY_MODULE_IMPORTS_START', flush=True)
import numpy as np
print('AVAILABILITY_NUMPY_READY', flush=True)
import pandas as pd
print('AVAILABILITY_PANDAS_READY', flush=True)

ROOT = Path('/data/admin/fengshuo/ehr_fm_20261102')
REV = ROOT / 'revision_v4_20261122'
RAW = Path('/data/admin/fengshuo/if10_20261002/shared/raw/mimic')
NAME = 'mimic_available_by24_v4_1'
WORK = REV / 'availability_repair'
OUT = WORK / 'dataset' / NAME
PARTS = OUT / 'source_parts'
GATE = REV / 'temporal/results/eligibility/status_consistent_v4_1/landmark_v4_1'
OLD_PROC = ROOT / 'data/processed'
OLD_INT = ROOT / 'data/interim'
CHUNK_SIZE = 250_000
HORIZON_HOURS = 24
MAXLEN = 128
NBINS = 10
SEED = 20261102


class HashingReader:
    """Non-seekable binary reader that hashes compressed bytes as consumed."""
    def __init__(self, path: Path):
        self.path = path
        self.handle = path.open('rb')
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size=-1):
        data = self.handle.read(size)
        self.digest.update(data)
        self.bytes_read += len(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def readable(self):
        return True

    def seekable(self):
        return False

    def tell(self):
        return self.handle.tell()

    def close(self):
        self.handle.close()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')
    os.replace(tmp, path)


def load_constants(path: Path):
    wanted = {'VARS', 'RANGE', 'MIMIC_CHART', 'MIMIC_LAB',
              'MIMIC_VASO', 'MIMIC_SED', 'MIMIC_RRT'}
    found = {}
    for node in ast.parse(path.read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    found[target.id] = ast.literal_eval(node.value)
    if set(found) != wanted:
        raise ValueError(f'Original MIMIC mapping declarations missing: {sorted(wanted-set(found))}')
    found['VAR2ID'] = {v: i for i, v in enumerate(found['VARS'])}
    if len(found['VARS']) != 48:
        raise ValueError('Frozen vocabulary is not the expected 48-variable contract')
    return found


def load_frozen_digitize(path: Path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'digitize']
    if len(nodes) != 1:
        raise ValueError('The original digitize function contract changed')
    namespace = {'np': np, 'NBINS': NBINS}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['digitize']


def measurement_availability(measured, stored, intime, horizon_hours=24):
    """Classify a measurement window and the strict H24 storetime rule."""
    measured = pd.to_datetime(measured, errors='coerce')
    stored = pd.to_datetime(stored, errors='coerce')
    intime = pd.to_datetime(intime, errors='coerce')
    cutoff = intime + pd.Timedelta(hours=horizon_hours)
    hour = (measured - intime).dt.total_seconds() / 3600.0
    window = measured.notna() & intime.notna() & hour.ge(0) & hour.lt(horizon_hours)
    late = window & stored.notna() & stored.gt(cutoff)
    unknown = window & stored.isna()
    allowed = window & stored.notna() & stored.le(cutoff)
    return hour, window, late, unknown, allowed


def therapy_availability(started, ended, stored, intime, horizon_hours=24):
    """Apply the recorded-treatment initiation and pre-ICU-overlap contract."""
    started = pd.to_datetime(started, errors='coerce')
    ended = pd.to_datetime(ended, errors='coerce')
    stored = pd.to_datetime(stored, errors='coerce')
    intime = pd.to_datetime(intime, errors='coerce')
    cutoff = intime + pd.Timedelta(hours=horizon_hours)
    hour = (started - intime).dt.total_seconds() / 3600.0
    valid_start = started.notna() & intime.notna()
    pre = valid_start & hour.lt(0)
    window = valid_start & hour.ge(0) & hour.lt(horizon_hours)
    late = (pre | window) & stored.notna() & stored.gt(cutoff)
    unknown_store = (pre | window) & stored.isna()
    stored_in_time = stored.notna() & stored.le(cutoff)
    ongoing = pre & ended.notna() & ended.gt(intime)
    ended_before = pre & ended.notna() & ended.le(intime)
    unknown_end = pre & ended.isna()
    allowed_pre = ongoing & stored_in_time
    allowed_in = window & stored_in_time
    allowed = allowed_pre | allowed_in
    hour = hour.where(~allowed_pre, 0.0)
    return {
        'hour': hour, 'window': window, 'pre_icu': pre, 'ongoing_overlap': ongoing,
        'ended_before_icu': ended_before, 'unknown_end': unknown_end,
        'late_storetime': late, 'unknown_storetime': unknown_store,
        'allowed_pre_icu': allowed_pre, 'allowed_in_icu': allowed_in,
        'allowed': allowed,
    }


def finite_range_mask(values, bounds):
    vals = pd.to_numeric(values, errors='coerce').astype('float64')
    finite = pd.Series(np.isfinite(vals.to_numpy()), index=vals.index)
    lo, hi = bounds
    in_range = finite & vals.ge(lo) & vals.le(hi)
    return vals, finite, in_range


def same_frozen_labels(current, frozen):
    """Exact elementwise label check while treating paired missing values alike."""
    return np.array_equal(np.asarray(current), np.asarray(frozen), equal_nan=True)


def gcs_total_events(components: pd.DataFrame, intime_by_stay, value_range):
    """Use the original same-stay/charttime min-per-item GCS contract.

    Components must already have passed their own H24 storetime checks. Require
    all three source items at a timestamp before deriving a total.
    """
    need = [220739, 223900, 223901]
    if components.empty:
        return pd.DataFrame(columns=['stay_id', 'hour', 'var', 'val', 'source_table', 'itemid']), {
            'component_rows': 0, 'complete_component_timestamps': 0,
            'incomplete_component_timestamps': 0, 'range_valid_totals': 0,
        }
    c = components.copy()
    c['itemid'] = pd.to_numeric(c['itemid'], errors='raise').astype('int64')
    c['val'] = pd.to_numeric(c['val'], errors='coerce').astype('float64')
    c = c.loc[np.isfinite(c['val'].to_numpy()) & c['itemid'].isin(need)].copy()
    per_item = (c.groupby(['stay_id', 'charttime', 'itemid'], as_index=False,
                          sort=False, observed=True)['val'].min())
    wide = per_item.pivot(index=['stay_id', 'charttime'], columns='itemid', values='val')
    complete = wide.reindex(columns=need).notna().all(axis=1)
    full = wide.loc[complete, need]
    totals = full.sum(axis=1)
    finite = np.isfinite(totals.to_numpy())
    lo, hi = value_range
    in_range = finite & totals.ge(lo).to_numpy() & totals.le(hi).to_numpy()
    selected = totals.loc[in_range].rename('val').reset_index()
    if len(selected):
        selected['hour'] = ((selected['charttime'] - selected['stay_id'].map(intime_by_stay))
                            .dt.total_seconds() / 3600.0)
        selected = selected.loc[selected['hour'].ge(0) & selected['hour'].lt(HORIZON_HOURS)].copy()
    selected['var'] = 'gcs_total'
    selected['source_table'] = 'derived_gcs_components'
    selected['itemid'] = 0
    selected = selected[['stay_id', 'hour', 'var', 'val', 'source_table', 'itemid']]
    summary = {
        'component_rows': int(len(c)),
        'complete_component_timestamps': int(complete.sum()),
        'incomplete_component_timestamps': int((~complete).sum()),
        'range_valid_totals': int(len(selected)),
    }
    return selected, summary


def _add_byvar(stage, vars_, mask, counts):
    if not len(vars_):
        return
    selected = vars_.loc[mask]
    if len(selected):
        for var, count in selected.value_counts(dropna=False, sort=False).items():
            counts[str(var)][stage] += int(count)


def _write_parquet_atomic(frame, target):
    tmp = target.with_suffix('.parquet.tmp')
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, target)


def hourly_aggregate(events, var2id):
    """Original hourly mean with concrete string groups and observed groups only."""
    ev = events.loc[events['hour'].lt(HORIZON_HOURS)].copy()
    ev['var'] = ev['var'].astype(str)
    ev['hb'] = np.floor(pd.to_numeric(ev['hour'], errors='raise')).astype('int16').clip(0, HORIZON_HOURS - 1)
    ev['val'] = pd.to_numeric(ev['val'], errors='coerce').astype('float64')
    if not np.isfinite(ev['val'].to_numpy()).all():
        raise ValueError('A nonfinite source cell reached hourly aggregation')
    agg = (ev.groupby(['stay_id', 'hb', 'var'], as_index=False, sort=False,
                      observed=True)['val'].mean())
    agg = agg.loc[agg['var'].isin(var2id)].copy()
    agg['vid'] = agg['var'].map(var2id).astype(np.int16)
    return agg.sort_values(['stay_id', 'hb', 'vid']).reset_index(drop=True)


def _source_csv_chunks(path, usecols, dtypes):
    raw = HashingReader(path)
    gz = gzip.GzipFile(fileobj=raw, mode='rb')
    text = io.TextIOWrapper(gz, encoding='utf-8', newline='')
    reader = pd.read_csv(text, usecols=usecols, dtype=dtypes, chunksize=CHUNK_SIZE,
                         low_memory=False, on_bad_lines='error')
    return raw, gz, text, reader


def _ensure_int_ids(series):
    return pd.to_numeric(series, errors='coerce').astype('Int64')


def table_specifications(constants):
    chart = {int(iid): (str(var), str(transform))
             for var, items in constants['MIMIC_CHART'].items() for iid, transform in items}
    labs = {int(iid): (str(var), '1') for var, iid in constants['MIMIC_LAB'].items()}
    urine_ids = [226559, 226560, 226561, 226584, 226563, 226564,
                 226565, 226567, 226557, 226558, 227488, 227489]
    urine = {iid: ('urine', '1') for iid in urine_ids}
    therapies = {int(iid): (var, '1') for items, var in
                 [(constants['MIMIC_VASO'], 'vaso'), (constants['MIMIC_SED'], 'sedation')]
                 for iid in items}
    rrt = {int(iid): ('rrt', '1') for iid in constants['MIMIC_RRT']}
    return [
        {'table': 'chartevents', 'relpath': 'icu/chartevents.csv.gz', 'key': 'stay_id',
         'time': 'charttime', 'value': 'valuenum', 'mapping': chart, 'therapy': False},
        {'table': 'labevents', 'relpath': 'hosp/labevents.csv.gz', 'key': 'hadm_id',
         'time': 'charttime', 'value': 'valuenum', 'mapping': labs, 'therapy': False},
        {'table': 'outputevents', 'relpath': 'icu/outputevents.csv.gz', 'key': 'stay_id',
         'time': 'charttime', 'value': 'value', 'mapping': urine, 'therapy': False},
        {'table': 'inputevents', 'relpath': 'icu/inputevents.csv.gz', 'key': 'stay_id',
         'time': 'starttime', 'endtime': 'endtime', 'value': None,
         'mapping': therapies, 'therapy': True},
        {'table': 'procedureevents', 'relpath': 'icu/procedureevents.csv.gz', 'key': 'stay_id',
         'time': 'starttime', 'endtime': 'endtime', 'value': None,
         'mapping': rrt, 'therapy': True},
    ]


def _new_counter():
    return defaultdict(Counter)


def _process_measurement_chunk(chunk, spec, starts, hadm_to_stay, cohort_ids,
                               ranges, counters, gcs_out=None):
    itemid = _ensure_int_ids(chunk['itemid'])
    mapped_mask = itemid.isin(spec['mapping'])
    mapped = chunk.loc[mapped_mask].copy()
    if mapped.empty:
        return pd.DataFrame(columns=['stay_id', 'hour', 'var', 'val', 'source_table', 'itemid'])
    mapped['itemid'] = itemid.loc[mapped_mask].astype('int64')
    mapped['var'] = mapped['itemid'].map({i: v for i, (v, _) in spec['mapping'].items()}).astype(str)
    mapped['transform'] = mapped['itemid'].map({i: t for i, (_, t) in spec['mapping'].items()}).astype(str)
    _add_byvar('mapped_item_records', mapped['var'], pd.Series(True, index=mapped.index), counters)
    if spec['key'] == 'hadm_id':
        mapped['stay_id'] = _ensure_int_ids(mapped['hadm_id']).map(hadm_to_stay)
    else:
        mapped['stay_id'] = _ensure_int_ids(mapped['stay_id'])
    linked = mapped['stay_id'].notna() & mapped['stay_id'].isin(cohort_ids)
    _add_byvar('cohort_linked_records', mapped['var'], linked, counters)
    mapped = mapped.loc[linked].copy()
    if mapped.empty:
        return pd.DataFrame(columns=['stay_id', 'hour', 'var', 'val', 'source_table', 'itemid'])
    mapped['stay_id'] = mapped['stay_id'].astype('int64')
    intime = mapped['stay_id'].map(starts)
    measured = pd.to_datetime(mapped[spec['time']], errors='coerce')
    if spec['therapy']:
        decision = therapy_availability(measured, mapped[spec['endtime']], mapped['storetime'], intime)
        _add_byvar('pre_icu_starts', mapped['var'], decision['pre_icu'], counters)
        _add_byvar('pre_icu_ongoing_overlap', mapped['var'], decision['ongoing_overlap'], counters)
        _add_byvar('pre_icu_ended_before_or_at_icu_excluded', mapped['var'], decision['ended_before_icu'], counters)
        _add_byvar('pre_icu_unknown_end_excluded', mapped['var'], decision['unknown_end'], counters)
        _add_byvar('pre_icu_late_storetime_excluded', mapped['var'], decision['pre_icu'] & decision['late_storetime'], counters)
        _add_byvar('pre_icu_unknown_storetime_excluded', mapped['var'], decision['pre_icu'] & decision['unknown_storetime'], counters)
        _add_byvar('pre_icu_ongoing_storetime_eligible', mapped['var'], decision['allowed_pre_icu'], counters)
        _add_byvar('in_icu_start_window_records', mapped['var'], decision['window'], counters)
        _add_byvar('in_icu_late_storetime_excluded', mapped['var'], decision['window'] & decision['late_storetime'], counters)
        _add_byvar('in_icu_unknown_storetime_excluded', mapped['var'], decision['window'] & decision['unknown_storetime'], counters)
        _add_byvar('in_icu_storetime_eligible', mapped['var'], decision['allowed_in_icu'], counters)
        available = decision['allowed']
        hour = decision['hour']
        selected = mapped.loc[available].copy()
        selected['hour'] = hour.loc[available]
        if selected.empty:
            return pd.DataFrame(columns=['stay_id', 'hour', 'var', 'val', 'source_table', 'itemid'])
        selected['hour'] = (np.floor(selected['hour'].astype(float) / 4.0) * 4.0).astype(float)
        selected = selected.drop_duplicates(['stay_id', 'hour', 'var'])
        selected['val'] = 1.0
    else:
        hour, window, late, unknown, allowed = measurement_availability(
            measured, mapped['storetime'], intime)
        _add_byvar('first24_measurement_time_records', mapped['var'], window, counters)
        _add_byvar('late_storetime_excluded', mapped['var'], late, counters)
        _add_byvar('missing_storetime_excluded', mapped['var'], unknown, counters)
        _add_byvar('storetime_eligible_records', mapped['var'], allowed, counters)
        vals = pd.to_numeric(mapped[spec['value']], errors='coerce').astype('float64')
        f2c = mapped['transform'].eq('F2C')
        vals.loc[f2c] = (vals.loc[f2c] - 32.0) * 5.0 / 9.0
        finite = pd.Series(np.isfinite(vals.to_numpy()), index=mapped.index)
        _add_byvar('nonfinite_value_excluded', mapped['var'], allowed & ~finite, counters)
        finite_allowed = allowed & finite
        _add_byvar('finite_storetime_eligible_records', mapped['var'], finite_allowed, counters)
        gcs_itemids = {220739, 223900, 223901}
        gcs_rows = finite_allowed & mapped['itemid'].isin(gcs_itemids)
        if gcs_out is not None and gcs_rows.any():
            gc = mapped.loc[gcs_rows, ['stay_id', spec['time'], 'itemid']].copy()
            gc['charttime'] = pd.to_datetime(gc[spec['time']], errors='coerce')
            gc['val'] = vals.loc[gcs_rows].to_numpy(dtype='float64')
            gcs_out.append(gc[['stay_id', 'charttime', 'itemid', 'val']])
        range_valid = finite_allowed.copy()
        for var in mapped.loc[finite_allowed, 'var'].unique():
            if var == 'gcs_verbal':
                continue
            lo, hi = ranges.get(var, (-np.inf, np.inf))
            vmask = mapped['var'].eq(var)
            valid = vals.ge(lo) & vals.le(hi)
            invalid = finite_allowed & vmask & ~valid
            _add_byvar('range_invalid_excluded', mapped['var'], invalid, counters)
            range_valid &= ~vmask | valid
        # The verbal component is retained only as an input to same-time GCS total.
        direct = range_valid & mapped['var'].ne('gcs_verbal')
        _add_byvar('accepted_event_rows', mapped['var'], direct, counters)
        selected = mapped.loc[direct, ['stay_id', 'itemid', 'var']].copy()
        selected['hour'] = hour.loc[direct].to_numpy(dtype='float64')
        selected['val'] = vals.loc[direct].to_numpy(dtype='float64')
        selected['source_table'] = spec['table']
        selected = selected[['stay_id', 'hour', 'var', 'val', 'source_table', 'itemid']]
        return selected
    selected['source_table'] = spec['table']
    return selected[['stay_id', 'hour', 'var', 'val', 'source_table', 'itemid']]


def process_table(spec, starts, hadm_to_stay, cohort_ids, ranges, code_hash):
    table = spec['table']
    path = RAW / spec['relpath']
    if not path.is_file():
        raise FileNotFoundError(path)
    table_dir = PARTS / table
    event_dir = table_dir / 'events'
    gcs_dir = table_dir / 'gcs_components'
    status_path = table_dir / 'table_status.json'
    source_stat = path.stat()
    if status_path.exists():
        previous = json.loads(status_path.read_text(encoding='utf-8'))
        if previous.get('status') == 'complete':
            if previous.get('source_size') != source_stat.st_size or previous.get('source_mtime_ns') != source_stat.st_mtime_ns:
                raise RuntimeError(f'{table}: completed source changed; refusing reuse')
            if previous.get('script_sha256') != code_hash:
                raise RuntimeError(f'{table}: completed table was produced by different code')
            print(json.dumps({'stage': 'table_reused_complete_checkpoint', 'table': table}), flush=True)
            return previous
    if table_dir.exists():
        shutil.rmtree(table_dir)
    event_dir.mkdir(parents=True, exist_ok=True)
    if table == 'chartevents':
        gcs_dir.mkdir(parents=True, exist_ok=True)
    cols = [spec['key'], 'itemid', spec['time'], 'storetime']
    dtypes = {'itemid': 'Int64', spec['key']: 'Int64'}
    if spec['value']:
        cols.append(spec['value'])
    else:
        cols.append(spec['endtime'])
    reader_parts = _source_csv_chunks(path, cols, dtypes)
    raw, gz, text, reader = reader_parts
    counters = _new_counter()
    raw_rows = chunks = emitted = gcs_component_rows = 0
    gcs_parts = []
    atomic_json(status_path, {'status': 'running', 'table': table,
        'source_path': str(path), 'source_size': source_stat.st_size,
        'source_mtime_ns': source_stat.st_mtime_ns, 'script_sha256': code_hash,
        'started_unix': time.time(), 'chunk_size': CHUNK_SIZE})
    try:
        for chunks, chunk in enumerate(reader, start=1):
            raw_rows += int(len(chunk))
            if spec['therapy']:
                events = _process_measurement_chunk(chunk, spec, starts, hadm_to_stay,
                                                     cohort_ids, ranges, counters)
            else:
                events = _process_measurement_chunk(chunk, spec, starts, hadm_to_stay,
                                                     cohort_ids, ranges, counters,
                                                     gcs_parts if table == 'chartevents' else None)
            if len(events):
                events['var'] = events['var'].astype(str)
                _write_parquet_atomic(events, event_dir / f'part-{chunks:06d}.parquet')
                emitted += int(len(events))
            if table == 'chartevents' and gcs_parts:
                gc = gcs_parts.pop()
                gcs_component_rows += len(gc)
                _write_parquet_atomic(gc, gcs_dir / f'part-{chunks:06d}.parquet')
            if chunks == 1 or chunks % 10 == 0:
                state = {'status': 'running', 'table': table, 'chunks': chunks,
                         'raw_records_scanned': raw_rows, 'event_rows_written': emitted,
                         'gcs_component_rows_written': gcs_component_rows,
                         'source_bytes_consumed': raw.bytes_read,
                         'partial_source_sha256': raw.digest.copy().hexdigest(),
                         'stage_counts_by_variable': {v: dict(c) for v, c in counters.items()}}
                atomic_json(status_path, state)
                print(json.dumps({'stage': 'source_chunk_progress', 'table': table,
                                  'chunks': chunks, 'raw_records_scanned': raw_rows,
                                  'event_rows_written': emitted,
                                  'source_bytes_consumed': raw.bytes_read}), flush=True)
    finally:
        text.close()
        gz.close()
        raw.close()
    if raw.bytes_read != source_stat.st_size:
        raise RuntimeError(f'{table}: stream consumed {raw.bytes_read} of {source_stat.st_size} bytes')
    end_stat = path.stat()
    if (end_stat.st_size, end_stat.st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
        raise RuntimeError(f'{table}: source changed during one-pass ETL')
    if table == 'chartevents' and gcs_parts:
        raise RuntimeError('Unexpected unflushed GCS component batch')
    summary = {'status': 'complete', 'table': table, 'source_path': str(path),
        'source_size': source_stat.st_size, 'source_mtime_ns': source_stat.st_mtime_ns,
        'source_file_sha256': raw.digest.hexdigest(), 'script_sha256': code_hash,
        'raw_records_scanned': raw_rows, 'chunks': chunks,
        'event_rows_written': emitted, 'gcs_component_rows_written': gcs_component_rows,
        'chunk_size': CHUNK_SIZE, 'stage_counts_by_variable': {v: dict(c) for v, c in counters.items()},
        'completed_unix': time.time()}
    atomic_json(status_path, summary)
    print(json.dumps({'stage': 'table_complete', 'table': table,
                      'raw_records_scanned': raw_rows, 'chunks': chunks,
                      'event_rows_written': emitted,
                      'source_file_sha256': summary['source_file_sha256']}), flush=True)
    return summary


def read_parts(directory, columns=None):
    frames = []
    if directory.exists():
        for path in sorted(directory.glob('part-*.parquet')):
            frames.append(pd.read_parquet(path, columns=columns))
    if not frames:
        return pd.DataFrame(columns=columns or [])
    return pd.concat(frames, ignore_index=True)


def build_products(events, cohort, bins, digitize, constants, frozen_split):
    variables = constants['VARS']
    var2id = constants['VAR2ID']
    ranges = constants['RANGE']
    if len(bins) != len(variables) or set(bins) != set(variables):
        raise ValueError('Frozen bins do not match original 48-variable vocabulary')
    agg = hourly_aggregate(events, var2id)
    stays = cohort['stay_id'].to_numpy(dtype=np.int64)
    sid2i = {int(s): i for i, s in enumerate(stays)}
    n_rows = len(stays)
    V = np.zeros((n_rows, MAXLEN), dtype=np.int16)
    B = np.zeros((n_rows, MAXLEN), dtype=np.int16)
    H = np.zeros((n_rows, MAXLEN), dtype=np.int16)
    L = np.zeros(n_rows, dtype=np.int16)
    rng = np.random.default_rng(SEED)
    for sid, frame in agg.groupby('stay_id', sort=False, observed=True):
        row = sid2i.get(int(sid))
        if row is None:
            continue
        size = len(frame)
        if size > MAXLEN:
            idx = rng.choice(size, MAXLEN, replace=False)
            frame = frame.iloc[np.sort(idx)]
            size = MAXLEN
        V[row, :size] = frame['vid'].to_numpy(dtype=np.int16) + 1
        vals = frame['val'].to_numpy(dtype=np.float64)
        vars_ = frame['var'].to_numpy(dtype=object)
        token_bins = np.ones(size, dtype=np.int16)
        for variable in set(vars_):
            select = vars_ == variable
            token_bins[select] = digitize(vals[select], variable, bins)
        B[row, :size] = token_bins
        H[row, :size] = frame['hb'].to_numpy(dtype=np.int16)
        L[row] = size
    tab = pd.DataFrame(index=stays)
    wide = agg.pivot_table(index='stay_id', columns='var', values='val', aggfunc='mean', observed=True)
    for variable in variables:
        tab[variable] = wide[variable].reindex(stays).to_numpy() if variable in wide.columns else np.nan
    tab = tab.reset_index().rename(columns={'index': 'stay_id'})
    if not np.array_equal(stays, cohort['stay_id'].to_numpy(dtype=np.int64)):
        raise AssertionError('Cohort order changed during product build')
    if not np.array_equal(frozen_split, np.load(OLD_PROC / 'split_mimic.npy')):
        raise ValueError('Frozen split differs from the original MIMIC split')
    active = np.arange(MAXLEN)[None, :] < L[:, None]
    if np.any(V[active] < 1) or np.any(V[active] > len(variables)):
        raise AssertionError('Active sequence has invalid vocabulary ids')
    if np.any(B[active] < 1) or np.any(B[active] > NBINS):
        raise AssertionError('Active sequence has invalid frozen bins')
    if np.any(H[active] < 0) or np.any(H[active] >= HORIZON_HOURS):
        raise AssertionError('Active sequence has invalid first-24-hour bins')
    if not np.all(V[~active] == 0) or not np.all(B[~active] == 0) or not np.all(H[~active] == 0):
        raise AssertionError('Sequence padding convention changed')
    return agg, tab, V, B, H, L


def preflight():
    print('PREFLIGHT_START', flush=True)
    constants = load_constants(ROOT / 'code/extract.py')
    print('PREFLIGHT_MAPPINGS_READY', flush=True)
    bins = json.loads((OLD_PROC / 'bins.json').read_text(encoding='utf-8'))
    print('PREFLIGHT_BINS_READY', flush=True)
    cohort = pd.read_parquet(OLD_INT / 'cohort_mimic.parquet', columns=['stay_id'])
    print('PREFLIGHT_COHORT_READY', flush=True)
    with np.load(OLD_PROC / 'seq_mimic.npz', allow_pickle=False) as old:
        stay_ids = old['stay_id']
    print('PREFLIGHT_SEQUENCE_READY', flush=True)
    gate = pd.read_parquet(GATE / 'mimic.parquet')
    print('PREFLIGHT_GATE_READY', flush=True)
    if len(cohort) != 94_343 or not np.array_equal(cohort.stay_id.to_numpy(dtype=np.int64), stay_ids):
        raise ValueError('Original cohort/sequence row-order preflight failed')
    if len(gate) != len(cohort) or not np.array_equal(gate.stay_id.to_numpy(dtype=np.int64), stay_ids):
        raise ValueError(f'Cleared gate shape/order differs: gate={len(gate)} cohort={len(cohort)}')
    sources = []
    for spec in table_specifications(constants):
        path = RAW / spec['relpath']
        if not path.is_file():
            raise FileNotFoundError(path)
        sources.append({'table': spec['table'], 'path': str(path),
                        'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns})
    print(json.dumps({'status': 'preflight_passed', 'dataset': NAME,
        'cohort_stays': len(cohort), 'gate_rows': len(gate),
        'candidate_eligible_stays': int(gate.eligible_24h.fillna(False).astype(bool).sum()),
        'source_event_tables': sources, 'variables': len(constants['VARS']),
        'frozen_bins_variables': len(bins), 'output_exists': OUT.exists()}, indent=2), flush=True)


def main():
    print('AVAILABILITY_ETL_START', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    if (OUT / 'RUN_COMPLETE.json').exists():
        raise FileExistsError(f'Completed output exists; preserving it: {OUT}')
    source_code = ROOT / 'code/extract.py'
    builder_code = ROOT / 'code/build_data.py'
    bins_path = OLD_PROC / 'bins.json'
    cohort_path = OLD_INT / 'cohort_mimic.parquet'
    old_seq_path = OLD_PROC / 'seq_mimic.npz'
    split_path = OLD_PROC / 'split_mimic.npy'
    event_path = OLD_INT / 'events_mimic.parquet'
    eligibility_path = GATE / 'mimic.parquet'
    eligibility_manifest_path = GATE / 'eligibility_manifest.json'
    constants = load_constants(source_code)
    digitize = load_frozen_digitize(builder_code)
    print('MAPPINGS_READY', flush=True)
    cohort = pd.read_parquet(cohort_path)
    print('COHORT_READY', flush=True)
    cohort['stay_id'] = pd.to_numeric(cohort['stay_id'], errors='raise').astype('int64')
    if cohort['stay_id'].duplicated().any():
        raise ValueError('Frozen original cohort has duplicate stay identifiers')
    cohort_ids = set(cohort['stay_id'].tolist())
    with np.load(old_seq_path, allow_pickle=False) as old:
        old_stay = old['stay_id'].copy()
        old_subject = old['subject'].astype(str)
        old_y = old['y_mort'].copy()
        old_aki = old['y_aki'].copy()
        old_long = old['y_long'].copy()
    print('ORIGINAL_SEQUENCE_READY', flush=True)
    if len(cohort) != 94_343 or not np.array_equal(cohort['stay_id'].to_numpy(), old_stay):
        raise ValueError('Frozen cohort must retain the original 94,343 stay order')
    if not np.array_equal(cohort['subject_id'].astype(str).to_numpy(), old_subject):
        raise ValueError('Frozen original subject order differs from sequence')
    if not same_frozen_labels(cohort['mort_hosp'].to_numpy(dtype=np.float32), old_y):
        raise ValueError('Frozen original mortality labels differ')
    if not same_frozen_labels(cohort['aki'].to_numpy(dtype=np.float32), old_aki):
        raise ValueError('Frozen original AKI labels differ')
    if not same_frozen_labels(cohort['long_icu'].to_numpy(dtype=np.float32), old_long):
        raise ValueError('Frozen original long-ICU labels differ')

    # Reconstruct the original adult ICU and HADM attribution order exactly:
    # labs map a hadm_id to the last adult ICU stay in source metadata order.
    icu_path = RAW / 'icu/icustays.csv.gz'
    patients_path = RAW / 'hosp/patients.csv.gz'
    icu = pd.read_csv(icu_path, usecols=['subject_id', 'hadm_id', 'stay_id', 'intime'],
                      dtype={'subject_id': 'Int64', 'hadm_id': 'Int64', 'stay_id': 'Int64'})
    patients = pd.read_csv(patients_path, usecols=['subject_id', 'anchor_age'],
                           dtype={'subject_id': 'Int64', 'anchor_age': 'float64'})
    icu['intime'] = pd.to_datetime(icu['intime'], errors='coerce')
    adult = icu.merge(patients, on='subject_id', how='left', sort=False)
    adult = adult.loc[adult['anchor_age'].ge(18)].copy()
    if not cohort_ids.issubset(set(adult['stay_id'].dropna().astype('int64'))):
        raise ValueError('Original cohort has a stay absent from adult ICU metadata')
    starts = dict(zip(adult['stay_id'].astype('int64'), adult['intime']))
    hadm_to_stay = dict(zip(adult['hadm_id'].astype('int64'), adult['stay_id'].astype('int64')))
    if not cohort_ids.issubset(starts):
        raise ValueError('Adult ICU metadata does not cover frozen cohort')
    print('ICU_METADATA_READY', flush=True)

    print('HASH_INPUTS_START', flush=True)
    source_hashes = {
        str(p): sha256_file(p) for p in [source_code, builder_code, bins_path, cohort_path,
            old_seq_path, split_path, event_path, eligibility_path,
            eligibility_manifest_path, icu_path, patients_path]
    }
    print('HASH_INPUTS_READY', flush=True)
    eligibility_manifest = json.loads(eligibility_manifest_path.read_text(encoding='utf-8'))
    if eligibility_manifest.get('cohorts', {}).get('mimic', {}).get('retrospective_landmark_prediction_allowed') is not True:
        raise ValueError('The source cleared gate is not open for candidate metadata copy')
    gate = pd.read_parquet(eligibility_path)
    print('CLEARED_GATE_READY', flush=True)
    if gate['stay_id'].duplicated().any() or len(gate) != len(cohort):
        raise ValueError('Cleared eligibility gate does not preserve all frozen cohort rows')
    if not np.array_equal(gate['stay_id'].to_numpy(dtype=np.int64), old_stay):
        raise ValueError('Cleared eligibility stay order differs from frozen sequence')
    bins = json.loads(bins_path.read_text(encoding='utf-8'))
    npbins = [len(bins[v]) for v in constants['VARS']]
    if any(b > NBINS - 1 for b in npbins):
        raise ValueError('Frozen digitization bins exceed the original NBINS=10 contract')
    frozen_split = np.load(split_path, allow_pickle=False)
    if len(frozen_split) != len(cohort):
        raise ValueError('Frozen split length differs from original cohort')
    code_hash = sha256_file(Path(__file__))
    print(json.dumps({'stage': 'metadata_ready', 'dataset': NAME, 'cohort_stays': len(cohort),
                      'candidate_eligible_stays': int(gate['eligible_24h'].fillna(False).astype(bool).sum()),
                      'script_sha256': code_hash}), flush=True)

    table_summaries = []
    for spec in table_specifications(constants):
        summary = process_table(spec, starts, hadm_to_stay, cohort_ids,
                                constants['RANGE'], code_hash)
        table_summaries.append(summary)

    # Derive GCS only after all chart chunks are joined, so same-time components
    # on adjacent CSV chunks cannot be split into false missing-component cases.
    gcs_components = read_parts(PARTS / 'chartevents/gcs_components')
    gcs_events, gcs_summary = gcs_total_events(gcs_components, starts,
                                                constants['RANGE']['gcs_total'])
    gcs_event_dir = PARTS / 'derived_gcs' / 'events'
    gcs_event_dir.mkdir(parents=True, exist_ok=True)
    if len(gcs_events):
        _write_parquet_atomic(gcs_events, gcs_event_dir / 'part-000001.parquet')
    gcs_summary['derived_event_rows'] = int(len(gcs_events))

    event_frames = []
    for spec in table_specifications(constants):
        event_frames.append(read_parts(PARTS / spec['table'] / 'events'))
    event_frames.append(gcs_events)
    events = pd.concat([f for f in event_frames if len(f)], ignore_index=True)
    if events.empty:
        events = pd.DataFrame(columns=['stay_id', 'hour', 'var', 'val', 'source_table', 'itemid'])
    events['stay_id'] = pd.to_numeric(events['stay_id'], errors='raise').astype('int64')
    events['hour'] = pd.to_numeric(events['hour'], errors='raise').astype('float64')
    events['var'] = events['var'].astype(str)
    events['val'] = pd.to_numeric(events['val'], errors='coerce').astype('float64')
    finite_events = np.isfinite(events['val'].to_numpy())
    if not finite_events.all():
        raise ValueError('Nonfinite source value remains in the assembled event table')
    events = events.loc[events['hour'].ge(0) & events['hour'].lt(HORIZON_HOURS)].copy()
    events = events.sort_values(['stay_id', 'hour', 'var'], kind='mergesort').reset_index(drop=True)
    final_events = OUT / f'events_{NAME}.parquet'
    _write_parquet_atomic(events[['stay_id', 'hour', 'var', 'val']], final_events)
    agg, tab, V, B, H, L = build_products(events, cohort, bins, digitize, constants, frozen_split)
    tab_path = OUT / f'tab_{NAME}.parquet'
    tab.to_parquet(tab_path, index=False)
    seq_path = OUT / f'seq_{NAME}.npz'
    np.savez_compressed(seq_path, V=V, B=B, H=H, L=L,
        stay_id=cohort['stay_id'].to_numpy(dtype=np.int64),
        y_mort=cohort['mort_hosp'].to_numpy(dtype=np.float32),
        y_aki=cohort['aki'].to_numpy(dtype=np.float32),
        y_long=cohort['long_icu'].to_numpy(dtype=np.float32),
        subject=cohort['subject_id'].astype(str).to_numpy(dtype='U32'))
    out_split = OUT / f'split_{NAME}.npy'
    shutil.copy2(split_path, out_split)
    if not np.array_equal(np.load(out_split, allow_pickle=False), frozen_split):
        raise ValueError('Output split differs from frozen original')
    cohort_out = OUT / f'cohort_{NAME}.parquet'
    shutil.copy2(cohort_path, cohort_out)
    vocab_path = OUT / f'vocab_{NAME}.json'
    vocab = {'dataset': NAME, 'variables_in_original_order': constants['VARS'],
             'var_to_id': constants['VAR2ID'], 'padding_value_id': 0,
             'token_value_ids_are_one_based': True, 'bin_count': NBINS,
             'frozen_bin_source': str(bins_path), 'frozen_bin_sha256': source_hashes[str(bins_path)]}
    atomic_json(vocab_path, vocab)
    meta_path = OUT / f'meta_{NAME}.json'
    meta = {'n': int(len(cohort)), 'mean_len': float(L.mean()), 'max_len': int(L.max()),
            'vars': constants['VARS'], 'nbins': NBINS,
            'prev': {t: float(np.nanmean(cohort[c])) for t, c in
                     [('mort', 'mort_hosp'), ('aki', 'aki'), ('long', 'long_icu')]}}
    atomic_json(meta_path, meta)

    candidate_eligibility = gate.copy()
    candidate_eligibility['db'] = NAME
    candidate_eligibility['candidate_eligible_24h'] = candidate_eligibility['eligible_24h'].fillna(False).astype(bool)
    candidate_eligibility['eligible_24h'] = False
    candidate_eligibility['retrospective_landmark_prediction_allowed'] = False
    candidate_eligibility['reason'] = 'available_by24_input_sensitivity_pending_scope_record'
    eligibility_out = OUT / 'candidate_eligibility.parquet'
    candidate_eligibility.to_parquet(eligibility_out, index=False)

    stage_counts = {'tables': table_summaries, 'derived_gcs': gcs_summary,
        'assembled_first24_event_rows': int(len(events)),
        'hourly_observed_group_rows': int(len(agg)),
        'cohort_stays': int(len(cohort)),
        'empty_inputs_by_variable': {v: int(tab[v].isna().sum()) for v in constants['VARS']},
        'tabular_finite_cells_by_variable': {v: int(np.isfinite(tab[v].to_numpy(dtype=np.float64)).sum())
                                             for v in constants['VARS']}}
    atomic_json(OUT / 'source_stage_counts.json', stage_counts)
    verification = {
        'status': 'passed', 'active_tokens': int(L.sum()), 'empty_stay_inputs': int((L == 0).sum()),
        'max_active_tokens': int(L.max()), 'finite_assembled_source_cells': int(len(events)),
        'finite_hourly_aggregate_cells': int(np.isfinite(agg['val'].to_numpy()).sum()),
        'active_token_bins_min': int(B[V != 0].min()) if np.any(V != 0) else None,
        'active_token_bins_max': int(B[V != 0].max()) if np.any(V != 0) else None,
        'all_active_bins_within_1_to_10': bool(np.all((B[V != 0] >= 1) & (B[V != 0] <= NBINS))),
        'all_active_hours_within_0_to_23': bool(np.all((H[V != 0] >= 0) & (H[V != 0] < 24))),
        'tabular_finite_cells_by_variable': stage_counts['tabular_finite_cells_by_variable'],
        'stay_subject_mortality_aki_long_order_preserved': True,
        'frozen_split_identical': True,
    }
    atomic_json(OUT / 'finite_cell_bin_verification.json', verification)

    attribution = defaultdict(lambda: {'source_tables': [], 'raw_itemids': [], 'source_transform': 'identity',
                                       'plausible_range': None, 'canonical_value_unit': 'as defined by original MIMIC mapping'})
    for spec in table_specifications(constants):
        for iid, (var, transform) in spec['mapping'].items():
            row = attribution[var]
            if spec['table'] not in row['source_tables']:
                row['source_tables'].append(spec['table'])
            row['raw_itemids'].append(int(iid))
            if transform == 'F2C':
                row['source_transform'] = 'Fahrenheit to Celsius (preserved extract.py F2C mapping)'
            lo, hi = constants['RANGE'].get(var, (-np.inf, np.inf))
            row['plausible_range'] = [float(lo), float(hi)]
    attribution['gcs_total'] = {'source_tables': ['chartevents'],
        'raw_itemids': [220739, 223900, 223901],
        'source_transform': 'same-stay/same-charttime min per component, then sum all 3; all component rows pass H24',
        'plausible_range': list(constants['RANGE']['gcs_total']),
        'canonical_value_unit': 'original extractor derived GCS points'}
    atomic_json(OUT / 'input_field_attribution.json', dict(attribution))

    endpoint_metadata = {
        'status': 'candidate_metadata_copied_gate_closed_pending_review',
        'dataset': NAME, 'source_eligibility_manifest': str(eligibility_manifest_path),
        'source_eligibility_manifest_sha256': source_hashes[str(eligibility_manifest_path)],
        'source_eligibility_table_sha256': source_hashes[str(eligibility_path)],
        'source_gate_retrospective_landmark_prediction_allowed': True,
        'candidate_eligible_24h_stays': int(candidate_eligibility['candidate_eligible_24h'].sum()),
        'candidate_eligible_24h_events': int(candidate_eligibility.loc[candidate_eligibility.candidate_eligible_24h, 'event_24h'].sum()) if 'event_24h' in candidate_eligibility else None,
        'eligible_24h': False, 'retrospective_landmark_prediction_allowed': False,
        'source_chart_and_store_cutoff_by24_verified': True,
        'clinical_input_availability_verified': False,
        'scope': 'Retrospective input sensitivity using selected source records measured and stored by H24. This does not establish that clinicians or nurses could view the data by H24 or at the event hour. Pre-ICU treatment carryover is placed at H0 only when endtime overlaps ICU admission and storetime meets the H24 cutoff; its recorded availability is by the landmark, not necessarily by H0.'}
    atomic_json(OUT / 'endpoint_input_metadata.json', endpoint_metadata)

    registry_fragment = {'dataset': NAME, 'sequence_path': str(seq_path), 'split_path': str(out_split),
        'tabular_path': str(tab_path), 'vocab_path': str(vocab_path),
        'cohort_path': str(cohort_out), 'candidate_eligibility_path': str(eligibility_out),
        'manifest_path': str(OUT / 'dataset_manifest.json'),
        'retrospective_landmark_prediction_allowed': False,
        'clinical_input_availability_verified': False}
    atomic_json(OUT / 'registry_fragment.json', registry_fragment)
    schema = {'dataset': NAME, 'event_columns': ['stay_id', 'hour', 'var', 'val'],
        'event_var_type': 'string', 'event_value_type': 'finite float64',
        'token_arrays': {'V': 'int16; 0 pad, 1..48 variable id',
                         'B': 'int16; 0 pad, 1..10 frozen bin id',
                         'H': 'int16; 0 pad, 0..23 first-24-hour bin', 'L': 'int16'},
        'cohort_rows': len(cohort), 'feature_vocabulary': constants['VARS'],
        'aggregation': 'Preserved build_data.py semantics: floor hour to [0,23], mean by stay/hour/variable, sort, random MAXLEN=128 subsampling with SEED=20261102.'}
    atomic_json(OUT / 'schema.json', schema)

    final_paths = [final_events, tab_path, seq_path, out_split, cohort_out, eligibility_out,
                   vocab_path, meta_path, OUT / 'source_stage_counts.json', OUT / 'finite_cell_bin_verification.json',
                   OUT / 'input_field_attribution.json', OUT / 'endpoint_input_metadata.json',
                   OUT / 'registry_fragment.json', OUT / 'schema.json']
    output_hashes = {p.name: {'sha256': sha256_file(p), 'size': p.stat().st_size} for p in final_paths}
    manifest = {'status': 'complete_data_only_availability_sensitivity_gate_closed',
        'dataset': NAME, 'cohort_rows': len(cohort), 'source_hashes': source_hashes,
        'source_table_file_hashes': {t['table']: t['source_file_sha256'] for t in table_summaries},
        'output_hashes': output_hashes, 'source_stage_counts_path': str(OUT / 'source_stage_counts.json'),
        'finite_verification_path': str(OUT / 'finite_cell_bin_verification.json'),
        'source_availability_rule': 'measurement charttime/starttime in [ICU intime, intime+24h); storetime known and <= intime+24h; missing or late storetime excluded. Pre-ICU therapy initiation allowed only when documented endtime > ICU intime and storetime meets the same cutoff.',
        'source_chart_and_store_cutoff_by24_verified': True,
        'lab_attribution': 'Preserved original convention: hadm_id maps to last adult ICU stay in raw icustays metadata order.',
        'range_and_units': 'Original item maps, F2C conversion, and RANGE applied before hourly aggregation; nonfinite source values removed.',
        'original_vocabulary_and_bins_frozen': True, 'original_maxlen': MAXLEN, 'original_nbins': NBINS,
        'original_seed': SEED, 'source_cohort_stays': len(cohort),
        'candidate_eligible_stays': endpoint_metadata['candidate_eligible_24h_stays'],
        'eligible_24h': False, 'retrospective_landmark_prediction_allowed': False,
        'clinical_input_availability_verified': False,
        'limitations': ['storetime is a recorded-system proxy, not proof of clinician or nurse access time',
            'the sensitivity combines exclusion of late/missing-storetime records with a restricted pre-ICU therapy-overlap rule; losses must not be attributed to one mechanism alone',
            'ongoing pre-ICU treatment may be recorded by H24 but is not necessarily available at H0',
            'static demographic fields without source storetime were not turned into input events',
            'the new eligibility gate is held for a later scope decision; no model scoring or tuning was performed']}
    atomic_json(OUT / 'dataset_manifest.json', manifest)
    atomic_json(OUT / 'RUN_COMPLETE.json', {'status': manifest['status'],
        'dataset': NAME, 'manifest_sha256': sha256_file(OUT / 'dataset_manifest.json'),
        'completed_unix': time.time()})
    print(json.dumps({'stage': 'ETL_COMPLETE', 'dataset': NAME, 'output_dir': str(OUT),
                      'cohort_stays': len(cohort), 'candidate_eligible_stays': endpoint_metadata['candidate_eligible_24h_stays'],
                      'event_rows': len(events), 'active_tokens': int(L.sum()),
                      'manifest_sha256': sha256_file(OUT / 'dataset_manifest.json')}), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--preflight']:
        preflight()
    else:
        main()
