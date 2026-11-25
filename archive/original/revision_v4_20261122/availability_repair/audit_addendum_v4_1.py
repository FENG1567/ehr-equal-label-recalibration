"""Audit cached hourly values and sampled token-key drift without rereading events."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd


ROOT = Path('/data/admin/fengshuo/ehr_fm_20261102')
REV = ROOT / 'revision_v4_20261122'
FULL_NAME = 'mimic_available_by24_v4_1'
FIXED_NAME = 'mimic_available_by24_fixedselection_v4_1'
OLD_SEQ = ROOT / 'data/processed/seq_mimic.npz'
FULL_OUT = REV / 'availability_repair/dataset' / FULL_NAME
FIXED_OUT = REV / 'availability_repair/dataset' / FIXED_NAME
AUDIT = FIXED_OUT / 'audits'
RTOL = 1e-12
ATOL = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')
    os.replace(tmp, path)


def load_seq(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in ('V', 'B', 'H', 'L', 'stay_id')}


def check_cohort(old: dict, full: dict, fixed: dict) -> None:
    if not np.array_equal(old['stay_id'], full['stay_id']) or not np.array_equal(old['stay_id'], fixed['stay_id']):
        raise ValueError('Old, full-rebuild, and fixed-selection stay order differs')
    if len(old['stay_id']) != 94_343:
        raise ValueError(f'Unexpected frozen cohort size: {len(old["stay_id"])}')
    for name, seq in [('old', old), ('full', full), ('fixed', fixed)]:
        if seq['V'].shape != seq['B'].shape or seq['V'].shape != seq['H'].shape:
            raise ValueError(f'{name} token arrays have incompatible shapes')
        if seq['V'].shape[0] != len(seq['stay_id']) or seq['V'].shape[1] != 128:
            raise ValueError(f'{name} token array shape violates frozen contract')


def key_overlap_by_stay(old: dict, full: dict) -> pd.DataFrame:
    n = len(old['stay_id'])
    rows = np.arange(n, dtype=np.int32)
    result = {
        'row': rows,
        'stay_id': old['stay_id'].astype(np.int64, copy=False),
        'old_sequence_tokens': old['L'].astype(np.int32, copy=False),
        'full_rebuild_sequence_tokens': full['L'].astype(np.int32, copy=False),
    }
    overlap = np.zeros(n, dtype=np.int32)
    old_only = np.zeros(n, dtype=np.int32)
    full_only = np.zeros(n, dtype=np.int32)
    bin_equal = np.zeros(n, dtype=np.int32)
    bin_changed = np.zeros(n, dtype=np.int32)
    order_changed = np.zeros(n, dtype=bool)
    for row in range(n):
        old_len = int(old['L'][row])
        full_len = int(full['L'][row])
        old_values = old['V'][row, :old_len]
        old_hours = old['H'][row, :old_len]
        full_values = full['V'][row, :full_len]
        full_hours = full['H'][row, :full_len]
        if np.any(old_values < 1) or np.any(old_values > 48) or np.any((old_hours < 0) | (old_hours >= 24)):
            raise ValueError(f'Invalid old active token at row {row}')
        if np.any(full_values < 1) or np.any(full_values > 48) or np.any((full_hours < 0) | (full_hours >= 24)):
            raise ValueError(f'Invalid full-rebuild active token at row {row}')
        old_keys = old_hours.astype(np.int32) * 48 + old_values.astype(np.int32) - 1
        full_keys = full_hours.astype(np.int32) * 48 + full_values.astype(np.int32) - 1
        if len(np.unique(old_keys)) != old_len or len(np.unique(full_keys)) != full_len:
            raise ValueError(f'Duplicate active (V,H) key for row {row}')
        common_keys, old_idx, full_idx = np.intersect1d(
            old_keys, full_keys, assume_unique=True, return_indices=True)
        n_common = len(common_keys)
        overlap[row] = n_common
        old_only[row] = old_len - n_common
        full_only[row] = full_len - n_common
        old_bins = old['B'][row, :old_len][old_idx]
        full_bins = full['B'][row, :full_len][full_idx]
        same_bins = old_bins == full_bins
        bin_equal[row] = int(same_bins.sum())
        bin_changed[row] = int((~same_bins).sum())
        order_changed[row] = not np.array_equal(old_keys, full_keys)
    result.update({
        'old_full_vh_overlap': overlap,
        'old_vh_keys_only': old_only,
        'full_vh_keys_only': full_only,
        'sampled_key_set_changed': (old_only + full_only) > 0,
        'sampled_key_order_changed': order_changed,
        'sampled_keys_changed': ((old_only + full_only) > 0) | order_changed,
        'old_full_overlap_bins_equal': bin_equal,
        'old_full_overlap_bins_changed': bin_changed,
    })
    return pd.DataFrame(result)


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    manifest_path = FIXED_OUT / 'dataset_manifest.json'
    complete_path = FIXED_OUT / 'RUN_COMPLETE.json'
    fixed_seq_path = FIXED_OUT / f'seq_{FIXED_NAME}.npz'
    full_seq_path = FULL_OUT / f'seq_{FULL_NAME}.npz'
    original_hourly_path = AUDIT / 'original_hourly_24h.parquet'
    available_hourly_path = AUDIT / 'available_hourly_24h.parquet'
    position_map_path = AUDIT / 'oldtoken_position_map.parquet'
    hourly_per_stay_path = AUDIT / 'hourly_input_comparison_per_stay.parquet'
    expected = [manifest_path, complete_path, OLD_SEQ, fixed_seq_path, full_seq_path,
                original_hourly_path, available_hourly_path, position_map_path, hourly_per_stay_path]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing persisted audit inputs: ' + ', '.join(missing))

    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    run_complete = json.loads(complete_path.read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete_fixed_old_selection_availability_sensitivity_gate_closed':
        raise ValueError('Fixed-selection manifest has unexpected status')
    original_manifest_hash = sha256_file(manifest_path)
    if run_complete.get('manifest_sha256') != original_manifest_hash:
        raise ValueError('RUN_COMPLETE manifest hash does not match current manifest')
    seq_hash_before = sha256_file(fixed_seq_path)
    seq_entry = manifest.get('output_hashes', {}).get(fixed_seq_path.name, {})
    if seq_entry.get('sha256') != seq_hash_before:
        raise ValueError('Fixed-selection sequence differs from its frozen manifest hash')

    input_hashes = {str(path): sha256_file(path) for path in expected if path not in (manifest_path, complete_path)}
    old = load_seq(OLD_SEQ)
    full = load_seq(full_seq_path)
    fixed = load_seq(fixed_seq_path)
    check_cohort(old, full, fixed)

    old_hourly = pd.read_parquet(original_hourly_path, columns=['stay_id', 'hb', 'vid', 'val'])
    new_hourly = pd.read_parquet(available_hourly_path, columns=['stay_id', 'hb', 'vid', 'val'])
    keys = ['stay_id', 'hb', 'vid']
    for label, frame in [('original', old_hourly), ('available', new_hourly)]:
        if frame[keys].isna().any().any() or frame['val'].isna().any() or not np.isfinite(frame['val'].to_numpy(dtype=np.float64)).all():
            raise ValueError(f'{label} hourly table contains null/nonfinite cells')
        if frame.duplicated(keys).any():
            raise ValueError(f'{label} hourly table has duplicate keys')
    old_count = old_hourly.groupby('stay_id', sort=False, observed=True).size().rename('original_hourly_groups')
    new_count = new_hourly.groupby('stay_id', sort=False, observed=True).size().rename('available_hourly_groups')
    common = old_hourly.merge(new_hourly, on=keys, how='inner', sort=False, validate='one_to_one',
                              suffixes=('_original', '_available'))
    original_values = common['val_original'].to_numpy(dtype=np.float64)
    available_values = common['val_available'].to_numpy(dtype=np.float64)
    exact = original_values == available_values
    close = np.isclose(original_values, available_values, rtol=RTOL, atol=ATOL, equal_nan=False)
    absolute_error = np.abs(original_values - available_values)
    relative_denominator = np.maximum(np.abs(original_values), np.abs(available_values))
    relative_error = np.divide(absolute_error, relative_denominator,
                               out=np.zeros_like(absolute_error), where=relative_denominator != 0)
    common_by_stay = common.groupby('stay_id', sort=False, observed=True).size().rename('common_hourly_groups')
    exact_by_stay = common.assign(_exact=exact).groupby('stay_id', sort=False, observed=True)['_exact'].sum().rename('hourly_exact_values')
    close_by_stay = common.assign(_close=close).groupby('stay_id', sort=False, observed=True)['_close'].sum().rename('hourly_allclose_values')
    hourly = pd.DataFrame({'stay_id': old['stay_id'].astype(np.int64)}).set_index('stay_id')
    for series in (old_count, new_count, common_by_stay, exact_by_stay, close_by_stay):
        hourly = hourly.join(series, how='left')
    hourly = hourly.fillna(0).astype(np.int64).reset_index()
    hourly['hourly_key_sets_identical'] = (
        hourly['original_hourly_groups'].eq(hourly['available_hourly_groups']) &
        hourly['common_hourly_groups'].eq(hourly['original_hourly_groups']))
    hourly['hourly_values_exactly_equal'] = hourly['hourly_exact_values'].eq(hourly['common_hourly_groups'])
    hourly['hourly_values_allclose'] = hourly['hourly_allclose_values'].eq(hourly['common_hourly_groups'])
    hourly['hourly_input_unchanged_exactly'] = hourly['hourly_key_sets_identical'] & hourly['hourly_values_exactly_equal']
    hourly['hourly_input_unchanged_allclose'] = hourly['hourly_key_sets_identical'] & hourly['hourly_values_allclose']

    sequence = key_overlap_by_stay(old, full)
    position_map = pd.read_parquet(position_map_path,
                                   columns=['row', 'stay_id', 'status', 'old_bin', 'new_bin'])
    if position_map['row'].isna().any() or position_map['stay_id'].isna().any():
        raise ValueError('Token-position map has null row/stay ids')
    retained_map = position_map.loc[position_map['status'].eq('retained')].copy()
    if not position_map['status'].isin(['retained', 'removed_unavailable_key']).all():
        raise ValueError('Token-position map has unexpected status')
    retained_map['fixed_bin_equal'] = retained_map['old_bin'].astype(np.int16).eq(retained_map['new_bin'].astype(np.int16))
    fixed_bin_by_stay = (retained_map.groupby('stay_id', sort=False, observed=True)
                         .agg(fixed_selection_retained_keys=('status', 'size'),
                              fixed_selection_bins_equal=('fixed_bin_equal', 'sum')))
    fixed_bin_by_stay['fixed_selection_bins_changed'] = (
        fixed_bin_by_stay['fixed_selection_retained_keys'] - fixed_bin_by_stay['fixed_selection_bins_equal'])
    fixed_total = position_map.groupby('stay_id', sort=False, observed=True).agg(
        fixed_selection_old_keys=('status', 'size'),
        fixed_selection_removed_keys=('status', lambda s: int(s.eq('removed_unavailable_key').sum())))
    fixed_bin_by_stay = fixed_total.join(fixed_bin_by_stay, how='left').fillna(0).astype(np.int64).reset_index()
    per_stay = sequence.merge(hourly, on='stay_id', how='left', validate='one_to_one')
    per_stay = per_stay.merge(fixed_bin_by_stay, on='stay_id', how='left', validate='one_to_one')
    int_cols = [c for c in per_stay.columns if c.endswith('_groups') or c.endswith('_tokens') or
                c.endswith('_overlap') or c.endswith('_only') or c.endswith('_equal') or c.endswith('_changed') or
                c.startswith('fixed_selection_') or c in ('hourly_exact_values', 'hourly_allclose_values')]
    for column in int_cols:
        if per_stay[column].dtype.kind in 'fc':
            per_stay[column] = per_stay[column].fillna(0).astype(np.int64)
    per_stay['hourly_input_unchanged_exactly'] = per_stay['hourly_input_unchanged_exactly'].fillna(False).astype(bool)
    per_stay['hourly_input_unchanged_allclose'] = per_stay['hourly_input_unchanged_allclose'].fillna(False).astype(bool)
    unchanged_key_changes = per_stay.loc[
        per_stay['hourly_input_unchanged_exactly'] & per_stay['sampled_keys_changed']].copy()

    per_stay_path = AUDIT / 'audit_addendum_per_stay.parquet'
    changed_path = AUDIT / 'unchanged_hourly_inputs_sampled_keys_changed.parquet'
    per_stay.to_parquet(per_stay_path, index=False)
    unchanged_key_changes.to_parquet(changed_path, index=False)

    overlap_total = int(per_stay['old_full_vh_overlap'].sum())
    old_total = int(per_stay['old_sequence_tokens'].sum())
    full_total = int(per_stay['full_rebuild_sequence_tokens'].sum())
    exact_total = int(exact.sum())
    close_total = int(close.sum())
    retained_bin_equal = int(retained_map['fixed_bin_equal'].sum())
    retained_bin_changed = int((~retained_map['fixed_bin_equal']).sum())
    full_overlap_bin_equal = int(per_stay['old_full_overlap_bins_equal'].sum())
    full_overlap_bin_changed = int(per_stay['old_full_overlap_bins_changed'].sum())
    report = {
        'status': 'complete_cached_artifact_audit_no_event_or_raw_reread',
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Compared persisted hourly aggregates, sequence NPZ files, and fixed-selection token-position map only; no event parquet or raw table was opened.',
        'inputs': input_hashes,
        'hourly_value_comparison': {
            'original_hourly_groups': int(len(old_hourly)),
            'available_hourly_groups': int(len(new_hourly)),
            'common_hourly_keys': int(len(common)),
            'original_only_hourly_keys': int(len(old_hourly) - len(common)),
            'available_only_hourly_keys': int(len(new_hourly) - len(common)),
            'numeric_exact_equal_values_on_common_keys': exact_total,
            'common_values_allclose': close_total,
            'common_values_allclose_fraction': float(close_total / len(common)) if len(common) else None,
            'all_common_values_exactly_equal': bool(exact.all()),
            'all_common_values_allclose': bool(close.all()),
            'allclose': {'rtol': RTOL, 'atol': ATOL, 'equal_nan': False},
            'maximum_absolute_difference': float(absolute_error.max()) if len(absolute_error) else None,
            'maximum_relative_difference': float(relative_error.max()) if len(relative_error) else None,
            'stays_with_identical_hourly_key_sets': int(per_stay['hourly_key_sets_identical'].sum()),
            'stays_with_identical_hourly_inputs_exactly': int(per_stay['hourly_input_unchanged_exactly'].sum()),
            'stays_with_identical_hourly_inputs_allclose': int(per_stay['hourly_input_unchanged_allclose'].sum()),
            'exact_equality_definition': 'float64 numeric == on values joined by identical (stay_id,hb,vid) keys',
        },
        'fixed_selection_frozen_bin_comparison': {
            'retained_original_token_keys': int(len(retained_map)),
            'retained_keys_with_equal_old_and_recomputed_frozen_bin': retained_bin_equal,
            'retained_keys_with_changed_old_and_recomputed_frozen_bin': retained_bin_changed,
            'bin_equality_fraction': float(retained_bin_equal / len(retained_map)) if len(retained_map) else None,
            'comparison_source': str(position_map_path),
        },
        'old_vs_full_rebuild_sampled_vh_key_overlap': {
            'old_active_keys': old_total,
            'full_rebuild_active_keys': full_total,
            'common_stay_vh_keys': overlap_total,
            'old_only_stay_vh_keys': int(per_stay['old_vh_keys_only'].sum()),
            'full_only_stay_vh_keys': int(per_stay['full_vh_keys_only'].sum()),
            'old_full_overlap_fraction_of_old_keys': float(overlap_total / old_total) if old_total else None,
            'old_full_overlap_fraction_of_full_keys': float(overlap_total / full_total) if full_total else None,
            'overlapping_keys_with_equal_old_and_full_bins': full_overlap_bin_equal,
            'overlapping_keys_with_changed_old_and_full_bins': full_overlap_bin_changed,
            'stays_with_changed_sampled_key_sets': int(per_stay['sampled_key_set_changed'].sum()),
            'stays_with_changed_sampled_keys_including_order': int(per_stay['sampled_keys_changed'].sum()),
            'stays_with_exactly_unchanged_hourly_inputs_but_changed_sampled_keys': int(len(unchanged_key_changes)),
            'identified_stays_path': str(changed_path),
            'per_stay_detail_path': str(per_stay_path),
        },
        'per_stay_table': str(per_stay_path),
        'unchanged_hourly_inputs_sampled_keys_changed_stays': str(changed_path),
        'limits': ['Hourly value comparisons cover common keys; key-set losses and additions are reported separately.',
                   'Sampled-key overlap is defined per stay on (V,H), with the variable id converted from one-based sequence ids to zero-based hourly vid.',
                   'Frozen-bin comparisons use the existing fixed-selection position map and the old/full sequence NPZs.'],
    }
    atomic_json(AUDIT / 'audit_addendum.json', report)

    if sha256_file(fixed_seq_path) != seq_hash_before:
        raise AssertionError('Fixed-selection sequence bytes changed during audit')
    report_path = AUDIT / 'audit_addendum.json'
    manifest['audit_addendum'] = {
        'path': str(report_path),
        'sha256': sha256_file(report_path),
        'per_stay_path': str(per_stay_path),
        'per_stay_sha256': sha256_file(per_stay_path),
        'unchanged_hourly_inputs_sampled_keys_changed_path': str(changed_path),
        'unchanged_hourly_inputs_sampled_keys_changed_sha256': sha256_file(changed_path),
        'fixed_selection_sequence_sha256_unchanged': seq_hash_before,
    }
    atomic_json(manifest_path, manifest)
    run_complete['manifest_sha256'] = sha256_file(manifest_path)
    run_complete['audit_addendum_sha256'] = sha256_file(report_path)
    run_complete['completed_unix'] = time.time()
    atomic_json(complete_path, run_complete)
    print(json.dumps({
        'status': report['status'],
        'audit_addendum': str(report_path),
        'audit_addendum_sha256': sha256_file(report_path),
        'stays_with_unchanged_hourly_inputs_but_changed_sampled_keys': len(unchanged_key_changes),
        'fixed_selection_sequence_sha256': seq_hash_before,
        'manifest_sha256': sha256_file(manifest_path),
        'run_complete_manifest_sha256': json.loads(complete_path.read_text(encoding='utf-8'))['manifest_sha256'],
    }, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
