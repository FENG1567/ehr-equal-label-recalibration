"""Postprocess a fixed-old-token availability sensitivity without raw rereads.

Inputs are the completed H24-filtered event parquet, the frozen original NPZ,
and the original cohort/split/bin files. This writes a separate sequence whose
keys are exactly the old retained (stay, variable, hour) keys that remain
available in the H24 hourly aggregate. It never opens raw CSVs or scores a model.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from build_available_by24_v4_1 import (
    ROOT, REV, OLD_INT, OLD_PROC,
    NAME as FULL_NAME, MAXLEN, NBINS,
    atomic_json, hourly_aggregate, load_constants, load_frozen_digitize,
    same_frozen_labels, sha256_file,
)

NAME = 'mimic_available_by24_fixedselection_v4_1'
OUT = REV / 'availability_repair' / 'dataset' / NAME
FULL_OUT = REV / 'availability_repair' / 'dataset' / FULL_NAME
AUDIT_OUT = OUT / 'audits'


def build_fixed_sequence(old, new_hourly, variables, bins, digitize):
    """Keep old active token keys in old order; remove unavailable keys only."""
    old_v = np.asarray(old['V'])
    old_b = np.asarray(old['B'])
    old_h = np.asarray(old['H'])
    old_l = np.asarray(old['L'])
    stay_ids = np.asarray(old['stay_id'], dtype=np.int64)
    n_rows = len(stay_ids)
    if old_v.shape != old_b.shape or old_v.shape != old_h.shape or old_v.shape[0] != n_rows:
        raise ValueError('Frozen original token arrays have incompatible shapes')
    if old_v.shape[1] != MAXLEN:
        raise ValueError('Frozen original MAXLEN differs from the 128-token contract')
    positions = np.arange(old_v.shape[1], dtype=np.int16)
    active = positions[None, :] < old_l[:, None]
    row_idx, old_pos = np.nonzero(active)
    if np.any(old_v[active] == 0):
        raise ValueError('Original active token has padding variable id')
    tokens = pd.DataFrame({
        'row': row_idx.astype(np.int32),
        'old_position': old_pos.astype(np.int16),
        'stay_id': stay_ids[row_idx],
        'hb': old_h[active].astype(np.int16),
        'vid': (old_v[active].astype(np.int16) - 1),
        'old_bin': old_b[active].astype(np.int16),
    })
    if tokens.duplicated(['stay_id', 'hb', 'vid']).any():
        raise ValueError('Original sampled token keys are not unique')
    available = new_hourly[['stay_id', 'hb', 'vid', 'val']].copy()
    if available.duplicated(['stay_id', 'hb', 'vid']).any():
        raise ValueError('New hourly aggregate contains duplicate stay/hour/variable keys')
    merged = tokens.merge(available, on=['stay_id', 'hb', 'vid'], how='left',
                          sort=False, validate='one_to_one', indicator=True)
    merged = merged.sort_values(['row', 'old_position'], kind='mergesort').reset_index(drop=True)
    kept_mask = merged['_merge'].eq('both')
    kept = merged.loc[kept_mask].copy()
    if len(kept):
        if not np.isfinite(kept['val'].to_numpy(dtype=np.float64)).all():
            raise ValueError('Nonfinite hourly value reached fixed-token binning')
        kept['new_position'] = kept.groupby('row', sort=False, observed=True).cumcount().astype(np.int16)
    else:
        kept['new_position'] = pd.Series(dtype=np.int16)
    v = np.zeros((n_rows, MAXLEN), dtype=np.int16)
    b = np.zeros((n_rows, MAXLEN), dtype=np.int16)
    h = np.zeros((n_rows, MAXLEN), dtype=np.int16)
    length = np.zeros(n_rows, dtype=np.int16)
    if len(kept):
        rows = kept['row'].to_numpy(dtype=np.int64)
        pos = kept['new_position'].to_numpy(dtype=np.int64)
        vids = kept['vid'].to_numpy(dtype=np.int16)
        vals = kept['val'].to_numpy(dtype=np.float64)
        v[rows, pos] = vids + 1
        h[rows, pos] = kept['hb'].to_numpy(dtype=np.int16)
        for vid in np.unique(vids):
            mask = vids == vid
            variable = variables[int(vid)]
            b[rows[mask], pos[mask]] = digitize(vals[mask], variable, bins)
        length = np.bincount(rows, minlength=n_rows).astype(np.int16)
    position_map = merged[['row', 'stay_id', 'old_position', 'hb', 'vid', 'old_bin', '_merge']].copy()
    position_map['status'] = np.where(kept_mask, 'retained', 'removed_unavailable_key')
    new_position = np.full(len(position_map), -1, dtype=np.int32)
    new_bin = np.zeros(len(position_map), dtype=np.int16)
    if len(kept):
        kept_indices = np.flatnonzero(kept_mask.to_numpy())
        new_position[kept_indices] = kept['new_position'].to_numpy(dtype=np.int32)
        new_bin[kept_indices] = b[
            kept['row'].to_numpy(dtype=np.int64),
            kept['new_position'].to_numpy(dtype=np.int64)]
    position_map['new_position'] = pd.array(np.where(new_position < 0, pd.NA, new_position), dtype='Int32')
    position_map['new_bin'] = new_bin
    position_map['var'] = position_map['vid'].map({i: v for i, v in enumerate(variables)}).astype(str)
    position_map['available_hour_bin'] = position_map['hb'].astype(np.int16)
    position_map = position_map[['row', 'stay_id', 'old_position', 'new_position', 'status',
                                 'var', 'vid', 'available_hour_bin', 'old_bin', 'new_bin']]
    active_new = np.arange(MAXLEN)[None, :] < length[:, None]
    if np.any(v[active_new] < 1) or np.any(v[active_new] > len(variables)):
        raise AssertionError('Fixed sequence has invalid active variable ids')
    if np.any(b[active_new] < 1) or np.any(b[active_new] > NBINS):
        raise AssertionError('Fixed sequence has invalid active frozen bins')
    if np.any(h[active_new] < 0) or np.any(h[active_new] >= 24):
        raise AssertionError('Fixed sequence has invalid hour bins')
    if np.any(v[~active_new]) or np.any(b[~active_new]) or np.any(h[~active_new]):
        raise AssertionError('Fixed sequence padding is not zero')
    return v, b, h, length, position_map


def compare_hourly(old_hourly, new_hourly, stay_ids, subject_ids):
    keys = ['stay_id', 'hb', 'vid']
    old = old_hourly[keys + ['val']].rename(columns={'val': 'old_val'})
    new = new_hourly[keys + ['val']].rename(columns={'val': 'new_val'})
    merged = old.merge(new, on=keys, how='outer', sort=False,
                       validate='one_to_one', indicator=True)
    both = merged['_merge'].eq('both')
    same_value = pd.Series(False, index=merged.index)
    if both.any():
        same_value.loc[both] = np.isclose(
            merged.loc[both, 'old_val'].to_numpy(dtype=np.float64),
            merged.loc[both, 'new_val'].to_numpy(dtype=np.float64), rtol=0.0, atol=1e-12)
    merged['same_value'] = same_value
    old_counts = old.groupby('stay_id', sort=False, observed=True).size().rename('old_groups')
    new_counts = new.groupby('stay_id', sort=False, observed=True).size().rename('new_groups')
    old_only = merged.loc[merged['_merge'].eq('left_only')].groupby('stay_id', sort=False, observed=True).size().rename('old_only')
    new_only = merged.loc[merged['_merge'].eq('right_only')].groupby('stay_id', sort=False, observed=True).size().rename('new_only')
    common = merged.loc[both].groupby('stay_id', sort=False, observed=True).size().rename('common')
    same = merged.loc[same_value].groupby('stay_id', sort=False, observed=True).size().rename('same_value')
    per_stay = pd.DataFrame({'stay_id': stay_ids}).set_index('stay_id')
    for series in [old_counts, new_counts, old_only, new_only, common, same]:
        per_stay = per_stay.join(series, how='left')
    per_stay = per_stay.fillna(0).astype(np.int64).reset_index()
    per_stay['identical_available_hourly_inputs'] = (
        per_stay['old_groups'].eq(per_stay['new_groups']) &
        per_stay['common'].eq(per_stay['old_groups']) &
        per_stay['same_value'].eq(per_stay['old_groups']))
    subject_map = pd.DataFrame({'stay_id': stay_ids, 'subject_id': subject_ids})
    per_stay = per_stay.merge(subject_map, on='stay_id', how='left', validate='one_to_one')
    subject_summary = (per_stay.groupby('subject_id', sort=False, observed=True)
        .agg(cohort_stays=('stay_id', 'size'), unchanged_stays=('identical_available_hourly_inputs', 'sum'))
        .reset_index())
    summary = {
        'old_hourly_groups': int(len(old_hourly)),
        'new_available_hourly_groups': int(len(new_hourly)),
        'common_hourly_groups': int(both.sum()),
        'old_only_hourly_groups_removed': int(merged['_merge'].eq('left_only').sum()),
        'new_only_hourly_groups': int(merged['_merge'].eq('right_only').sum()),
        'common_groups_with_value_unchanged_atol_1e-12': int(same_value.sum()),
        'stays_with_identical_hourly_keys_and_values': int(per_stay['identical_available_hourly_inputs'].sum()),
        'subjects_with_all_cohort_stays_identical': int(subject_summary.cohort_stays.eq(subject_summary.unchanged_stays).sum()),
        'subject_count': int(subject_summary.subject_id.nunique()),
        'value_equality': 'np.isclose(rtol=0, atol=1e-12)',
    }
    return summary, per_stay


def token_key_audit(old, new, fixed_v, fixed_h, fixed_l):
    old_v, old_h, old_l = old['V'], old['H'], old['L']
    new_v, new_h, new_l = new['V'], new['H'], new['L']
    old_active = np.arange(old_v.shape[1])[None, :] < old_l[:, None]
    new_active = np.arange(new_v.shape[1])[None, :] < new_l[:, None]
    stay_ids = old['stay_id']
    if not np.array_equal(stay_ids, new['stay_id']):
        raise ValueError('Original and full-rebuild sequence stay order differs')
    old_rows = np.nonzero(old_active)[0]
    old_keys = (stay_ids[old_rows].astype(np.uint64) * np.uint64(24 * 48) +
                old_h[old_active].astype(np.uint64) * np.uint64(48) +
                (old_v[old_active].astype(np.uint64) - np.uint64(1)))
    new_rows = np.nonzero(new_active)[0]
    new_keys = (stay_ids[new_rows].astype(np.uint64) * np.uint64(24 * 48) +
                new_h[new_active].astype(np.uint64) * np.uint64(48) +
                (new_v[new_active].astype(np.uint64) - np.uint64(1)))
    fixed_active = np.arange(fixed_v.shape[1])[None, :] < fixed_l[:, None]
    fixed_rows = np.nonzero(fixed_active)[0]
    fixed_keys = (stay_ids[fixed_rows].astype(np.uint64) * np.uint64(24 * 48) +
                  fixed_h[fixed_active].astype(np.uint64) * np.uint64(48) +
                  (fixed_v[fixed_active].astype(np.uint64) - np.uint64(1)))
    old_keys = np.unique(old_keys)
    new_keys = np.unique(new_keys)
    fixed_keys = np.unique(fixed_keys)
    if len(np.setdiff1d(fixed_keys, old_keys, assume_unique=True)):
        raise AssertionError('Fixed-selection output added a key absent from original tokens')
    overlap = int(np.intersect1d(old_keys, new_keys, assume_unique=True).size)
    fixed_overlap = int(np.intersect1d(old_keys, fixed_keys, assume_unique=True).size)
    return {
        'old_active_token_keys': int(len(old_keys)), 'full_rebuild_active_token_keys': int(len(new_keys)),
        'old_vs_full_rebuild_key_overlap': overlap,
        'old_keys_still_available_in_new_hourly_aggregate': int(len(fixed_keys)),
        'old_keys_removed_as_unavailable': int(len(old_keys) - len(fixed_keys)),
        'fixed_variant_keys_added': int(len(fixed_keys) - fixed_overlap),
        'fixed_variant_is_subset_of_old_active_keys': True,
    }


def main():
    if (OUT / 'RUN_COMPLETE.json').exists():
        raise FileExistsError(f'Completed fixedselection output exists: {OUT}')
    OUT.mkdir(parents=True, exist_ok=True)
    AUDIT_OUT.mkdir(parents=True, exist_ok=True)
    constants = load_constants(ROOT / 'code/extract.py')
    variables = constants['VARS']
    digitize = load_frozen_digitize(ROOT / 'code/build_data.py')
    bins_path = OLD_PROC / 'bins.json'
    bins = json.loads(bins_path.read_text(encoding='utf-8'))
    old_seq_path = OLD_PROC / 'seq_mimic.npz'
    split_path = OLD_PROC / 'split_mimic.npy'
    old_event_path = OLD_INT / 'events_mimic.parquet'
    full_event_path = FULL_OUT / f'events_{FULL_NAME}.parquet'
    full_seq_path = FULL_OUT / f'seq_{FULL_NAME}.npz'
    full_manifest_path = FULL_OUT / 'dataset_manifest.json'
    full_manifest = json.loads(full_manifest_path.read_text(encoding='utf-8'))
    if full_manifest.get('status') != 'complete_data_only_availability_sensitivity_gate_closed':
        raise ValueError('Full H24 rebuild manifest is incomplete or has unexpected status')
    if full_manifest.get('clinical_input_availability_verified') is not False:
        raise ValueError('Clinical availability claim must remain false')
    if not (FULL_OUT / 'RUN_COMPLETE.json').exists():
        raise FileNotFoundError('Full H24 event dataset is not complete')
    src_hashes = {str(p): sha256_file(p) for p in
        [old_seq_path, split_path, old_event_path, full_event_path, full_seq_path,
         bins_path, ROOT / 'code/extract.py', ROOT / 'code/build_data.py', full_manifest_path]}
    with np.load(old_seq_path, allow_pickle=False) as old_npz:
        old = {k: old_npz[k].copy() for k in ['V', 'B', 'H', 'L', 'stay_id', 'y_mort', 'y_aki', 'y_long', 'subject']}
    with np.load(full_seq_path, allow_pickle=False) as full_npz:
        full = {k: full_npz[k].copy() for k in ['V', 'B', 'H', 'L', 'stay_id', 'y_mort', 'y_aki', 'y_long', 'subject']}
    if len(old['stay_id']) != 94_343 or not np.array_equal(old['stay_id'], full['stay_id']):
        raise ValueError('Original and new full-rebuild stay identity/order mismatch')
    for label in ['y_mort', 'y_aki', 'y_long']:
        if not same_frozen_labels(old[label], full[label]):
            raise ValueError('Full rebuild label drift: ' + label)
    if not np.array_equal(np.load(split_path, allow_pickle=False), np.load(FULL_OUT / f'split_{FULL_NAME}.npy', allow_pickle=False)):
        raise ValueError('Full rebuild split drift')

    print('READ_ORIGINAL_HOURLY_EVENTS_START', flush=True)
    old_events = pd.read_parquet(old_event_path, columns=['stay_id', 'hour', 'var', 'val'],
                                 filters=[('hour', '>=', 0), ('hour', '<', 24)])
    old_events['var'] = old_events['var'].astype(str)
    old_hourly = hourly_aggregate(old_events, constants['VAR2ID'])
    old_hourly = old_hourly[['stay_id', 'hb', 'var', 'val', 'vid']]
    del old_events
    old_hourly_path = AUDIT_OUT / 'original_hourly_24h.parquet'
    old_hourly.to_parquet(old_hourly_path, index=False)
    print(json.dumps({'stage': 'original_hourly_aggregate_ready', 'groups': len(old_hourly)}), flush=True)

    print('READ_FILTERED_H24_EVENTS_START', flush=True)
    new_events = pd.read_parquet(full_event_path, columns=['stay_id', 'hour', 'var', 'val'])
    new_events['var'] = new_events['var'].astype(str)
    new_hourly = hourly_aggregate(new_events, constants['VAR2ID'])
    new_hourly = new_hourly[['stay_id', 'hb', 'var', 'val', 'vid']]
    del new_events
    new_hourly_path = AUDIT_OUT / 'available_hourly_24h.parquet'
    new_hourly.to_parquet(new_hourly_path, index=False)
    print(json.dumps({'stage': 'available_hourly_aggregate_ready', 'groups': len(new_hourly)}), flush=True)

    subject = old['subject'].astype(str)
    hourly_summary, per_stay = compare_hourly(old_hourly, new_hourly,
                                               old['stay_id'], subject)
    per_stay_path = AUDIT_OUT / 'hourly_input_comparison_per_stay.parquet'
    per_stay.to_parquet(per_stay_path, index=False)
    hourly_audit = {'status': 'complete_cached_event_comparison_no_raw_reread',
        'original_event_path': str(old_event_path), 'available_event_path': str(full_event_path),
        'source_hashes': src_hashes, **hourly_summary}
    atomic_json(AUDIT_OUT / 'hourly_input_comparison.json', hourly_audit)

    fixed_v, fixed_b, fixed_h, fixed_l, position_map = build_fixed_sequence(
        old, new_hourly, variables, bins, digitize)
    position_map_path = AUDIT_OUT / 'oldtoken_position_map.parquet'
    position_map.to_parquet(position_map_path, index=False)
    token_audit = token_key_audit(old, full, fixed_v, fixed_h, fixed_l)
    token_audit.update({
        'old_active_tokens': int(old['L'].sum()),
        'full_rebuild_active_tokens': int(full['L'].sum()),
        'fixedselection_active_tokens': int(fixed_l.sum()),
        'stays_with_zero_retained_old_tokens': int((fixed_l == 0).sum()),
        'stays_with_any_removed_old_tokens': int((fixed_l < old['L']).sum()),
        'stays_with_unchanged_full_hourly_inputs': int(hourly_summary['stays_with_identical_hourly_keys_and_values']),
        'global_rng_resampling_can_shift_later_stay_keys_in_full_rebuild': True,
        'fixedselection_policy': 'Original active (stay,V,H) keys only; absent keys removed; survivors compacted stably by original position; B recalculated from new available hourly means with frozen bins; no RNG and no added keys.'})
    atomic_json(AUDIT_OUT / 'token_key_overlap.json', token_audit)

    seq_path = OUT / f'seq_{NAME}.npz'
    np.savez_compressed(seq_path, V=fixed_v, B=fixed_b, H=fixed_h, L=fixed_l,
        stay_id=old['stay_id'], y_mort=old['y_mort'], y_aki=old['y_aki'],
        y_long=old['y_long'], subject=old['subject'])
    out_split = OUT / f'split_{NAME}.npy'
    shutil.copy2(split_path, out_split)
    if not np.array_equal(np.load(out_split, allow_pickle=False), np.load(split_path, allow_pickle=False)):
        raise ValueError('Fixedselection split differs from original frozen split')
    tab_source = FULL_OUT / f'tab_{FULL_NAME}.parquet'
    tab_path = OUT / f'tab_{NAME}.parquet'
    shutil.copy2(tab_source, tab_path)
    cohort_source = OLD_INT / 'cohort_mimic.parquet'
    cohort_path = OUT / f'cohort_{NAME}.parquet'
    shutil.copy2(cohort_source, cohort_path)
    eligibility = pd.read_parquet(FULL_OUT / 'candidate_eligibility.parquet')
    eligibility['db'] = NAME
    eligibility['eligible_24h'] = False
    eligibility['retrospective_landmark_prediction_allowed'] = False
    eligibility['clinical_input_availability_verified'] = False
    eligibility['reason'] = 'fixed_old_token_selection_pending_scope_record'
    eligibility_path = OUT / 'candidate_eligibility.parquet'
    eligibility.to_parquet(eligibility_path, index=False)
    vocab_path = OUT / f'vocab_{NAME}.json'
    vocab = {'dataset': NAME, 'variables_in_original_order': variables,
        'var_to_id': constants['VAR2ID'], 'padding_value_id': 0,
        'token_value_ids_are_one_based': True, 'bin_count': NBINS,
        'frozen_bin_source': str(bins_path), 'frozen_bin_sha256': src_hashes[str(bins_path)]}
    atomic_json(vocab_path, vocab)
    meta_path = OUT / f'meta_{NAME}.json'
    meta = {'n': int(len(old['stay_id'])), 'mean_len': float(fixed_l.mean()),
        'max_len': int(fixed_l.max()), 'vars': variables, 'nbins': NBINS,
        'prev': {'mort': float(np.nanmean(old['y_mort'])),
                 'aki': float(np.nanmean(old['y_aki'])),
                 'long': float(np.nanmean(old['y_long']) )}}
    atomic_json(meta_path, meta)
    endpoint = json.loads((FULL_OUT / 'endpoint_input_metadata.json').read_text(encoding='utf-8'))
    endpoint['dataset'] = NAME
    endpoint['status'] = 'candidate_metadata_copied_gate_closed_pending_review'
    endpoint['eligible_24h'] = False
    endpoint['retrospective_landmark_prediction_allowed'] = False
    endpoint['clinical_input_availability_verified'] = False
    endpoint['source_chart_and_store_cutoff_by24_verified'] = True
    atomic_json(OUT / 'endpoint_input_metadata.json', endpoint)
    seq_hashes = {
        seq_path.name: {'sha256': sha256_file(seq_path), 'size': seq_path.stat().st_size},
        out_split.name: {'sha256': sha256_file(out_split), 'size': out_split.stat().st_size},
        tab_path.name: {'sha256': sha256_file(tab_path), 'size': tab_path.stat().st_size},
        eligibility_path.name: {'sha256': sha256_file(eligibility_path), 'size': eligibility_path.stat().st_size},
        position_map_path.name: {'sha256': sha256_file(position_map_path), 'size': position_map_path.stat().st_size},
    }
    manifest = {'status': 'complete_fixed_old_selection_availability_sensitivity_gate_closed',
        'dataset': NAME, 'source_full_rebuild_dataset': FULL_NAME,
        'source_full_rebuild_manifest_sha256': src_hashes[str(full_manifest_path)],
        'source_hashes': src_hashes, 'output_hashes': seq_hashes,
        'cohort_rows': int(len(old['stay_id'])),
        'candidate_eligible_stays': int(eligibility['candidate_eligible_24h'].sum()),
        'eligible_24h': False, 'retrospective_landmark_prediction_allowed': False,
        'clinical_input_availability_verified': False,
        'source_chart_and_store_cutoff_by24_verified': True,
        'sequence_key_policy': token_audit['fixedselection_policy'],
        'tabular_provenance': f'Reuses {FULL_NAME} tabular means, aggregated from only H24-filtered finite source events; baseline sequence input is this fixedselection sequence.',
        'hourly_input_comparison': str(AUDIT_OUT / 'hourly_input_comparison.json'),
        'token_key_overlap_audit': str(AUDIT_OUT / 'token_key_overlap.json'),
        'oldtoken_position_map': str(position_map_path),
        'limits': ['Recorded storetime is a system timestamp proxy, not proof of clinician view/access.',
            'Fixedselection removes unavailable originally sampled keys and recomputes their bins; it does not add newly available unsampled keys.',
            'Full rebuild resampling can change later patients token keys because the original global RNG stream advances differently when earlier patients cross the MAXLEN threshold.']}
    atomic_json(OUT / 'dataset_manifest.json', manifest)
    registry = {'dataset': NAME, 'sequence_path': str(seq_path), 'split_path': str(out_split),
        'tabular_path': str(tab_path), 'vocab_path': str(vocab_path),
        'cohort_path': str(cohort_path), 'candidate_eligibility_path': str(eligibility_path),
        'manifest_path': str(OUT / 'dataset_manifest.json'),
        'retrospective_landmark_prediction_allowed': False,
        'clinical_input_availability_verified': False}
    atomic_json(OUT / 'registry_fragment.json', registry)
    atomic_json(OUT / 'RUN_COMPLETE.json', {'status': manifest['status'],
        'dataset': NAME, 'manifest_sha256': sha256_file(OUT / 'dataset_manifest.json'),
        'completed_unix': time.time()})
    print(json.dumps({'stage': 'FIXEDSELECTION_COMPLETE', 'dataset': NAME,
        'stays': int(len(old['stay_id'])), 'old_tokens': int(old['L'].sum()),
        'fixed_tokens': int(fixed_l.sum()), 'removed_keys': int(old['L'].sum() - fixed_l.sum()),
        'full_rebuild_tokens': int(full['L'].sum()),
        'unchanged_stays': int(hourly_summary['stays_with_identical_hourly_keys_and_values']),
        'manifest_sha256': sha256_file(OUT / 'dataset_manifest.json')}), flush=True)


if __name__ == '__main__':
    main()
