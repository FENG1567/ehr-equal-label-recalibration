"""Aggregate cached MIMIC availability-token audit within the reviewed v4.2 gate.

This script consumes only the already persisted per-stay audit, its JSON/hash
bindings, the frozen sequence identity arrays, the split vector, and the reviewed
eligibility gate. It never reads raw events or recomputes availability bins.
Run by the root agent in the dedicated R5 CPU environment after review.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path('/data/admin/fengshuo/ehr_fm_20261102')
R4 = ROOT / 'revision_v4_20261122'
R5 = ROOT / 'revision_v5_20261123'
SEQ_PATH = ROOT / 'data/processed/seq_mimic.npz'
SPLIT_PATH = ROOT / 'data/processed/split_mimic.npy'
FIXED_DIR = R4 / 'availability_repair/dataset/mimic_available_by24_fixedselection_v4_1'
AUDIT_DIR = FIXED_DIR / 'audits'
AUDIT_PATH = AUDIT_DIR / 'audit_addendum.json'
PER_STAY_PATH = AUDIT_DIR / 'audit_addendum_per_stay.parquet'
DATASET_MANIFEST_PATH = FIXED_DIR / 'dataset_manifest.json'
RUN_COMPLETE_PATH = FIXED_DIR / 'RUN_COMPLETE.json'
GATE_DIR = R4 / 'temporal/results/eligibility/recovered_reviewed_v4_2'
GATE_PATH = GATE_DIR / 'mimic.parquet'
GATE_MANIFEST_PATH = GATE_DIR / 'eligibility_manifest.json'

EXPECTED = {
    'sequence_sha256': '9bd20bf1394a34ba112a4de9a8d7e8971821bef90558a9987be4b9da5ae06824',
    'gate_manifest_sha256': 'd302ce0a493121f9336f40fd58c5dc63a75a663492293c0153b4ba05efbb8e8f',
    'gate_table_sha256': 'c2774cd59e4df8706960e4fa58680ea6ec72b4af22adddf82b2389dc5b74749f',
    'n_sequence_rows': 94_343,
    'n_eligible': 73_936,
    'old_active_keys': 11_699_353,
    'retained_original_token_keys': 11_424_635,
    'fixed_selection_bins_changed': 8_098,
    'old_full_common_keys': 7_465_152,
    'stays_sampled_keys_changed': 86_304,
}
FROZEN_MIMIC_SPLIT_SHA256 = 'ed5bd94933d22daaa4cd5fdff82ce48b36b72ea50d2db7e4642906036c043c1a'

# These names are taken from audit_addendum_v4_1.py's per_stay construction.
PER_STAY_COLUMNS = [
    'row', 'stay_id', 'old_sequence_tokens', 'full_rebuild_sequence_tokens',
    'old_full_vh_overlap', 'old_vh_keys_only', 'full_vh_keys_only',
    'sampled_key_set_changed', 'sampled_key_order_changed', 'sampled_keys_changed',
    'old_full_overlap_bins_equal', 'old_full_overlap_bins_changed',
    'fixed_selection_old_keys', 'fixed_selection_removed_keys',
    'fixed_selection_retained_keys', 'fixed_selection_bins_equal',
    'fixed_selection_bins_changed',
]
TOKEN_COLUMNS = [
    'old_sequence_tokens', 'full_rebuild_sequence_tokens', 'old_full_vh_overlap',
    'old_vh_keys_only', 'full_vh_keys_only', 'old_full_overlap_bins_equal',
    'old_full_overlap_bins_changed', 'fixed_selection_old_keys',
    'fixed_selection_removed_keys', 'fixed_selection_retained_keys',
    'fixed_selection_bins_equal', 'fixed_selection_bins_changed',
]
BOOL_COLUMNS = ['sampled_key_set_changed', 'sampled_key_order_changed', 'sampled_keys_changed']


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return value


def normalized_stays(values, source: str) -> np.ndarray:
    numeric = pd.to_numeric(pd.Series(values), errors='raise').to_numpy(dtype=np.int64)
    if (numeric <= 0).any() or len(np.unique(numeric)) != len(numeric):
        raise ValueError(f'{source}: stay_id must be positive and unique')
    return numeric


def cached_full_report_checks(report: dict, per: pd.DataFrame) -> dict:
    """Reconcile all-sequence totals before any eligible subset is computed."""
    section = report.get('old_vs_full_rebuild_sampled_vh_key_overlap')
    fixed = report.get('fixed_selection_frozen_bin_comparison')
    if not isinstance(section, dict) or not isinstance(fixed, dict):
        raise ValueError('Cached audit report is missing its full-cohort summary sections')
    sums = {name: int(per[name].sum()) for name in TOKEN_COLUMNS}
    stays = {
        'stays_with_changed_sampled_key_sets': int(per['sampled_key_set_changed'].sum()),
        'stays_with_changed_sampled_keys_including_order': int(per['sampled_keys_changed'].sum()),
        'stays_with_sampled_key_order_changed': int(per['sampled_key_order_changed'].sum()),
    }
    report_pairs = {
        'old_active_keys': (section.get('old_active_keys'), sums['old_sequence_tokens']),
        'full_rebuild_active_keys': (section.get('full_rebuild_active_keys'), sums['full_rebuild_sequence_tokens']),
        'common_stay_vh_keys': (section.get('common_stay_vh_keys'), sums['old_full_vh_overlap']),
        'old_only_stay_vh_keys': (section.get('old_only_stay_vh_keys'), sums['old_vh_keys_only']),
        'full_only_stay_vh_keys': (section.get('full_only_stay_vh_keys'), sums['full_vh_keys_only']),
        'overlapping_keys_with_equal_old_and_full_bins': (
            section.get('overlapping_keys_with_equal_old_and_full_bins'), sums['old_full_overlap_bins_equal']),
        'overlapping_keys_with_changed_old_and_full_bins': (
            section.get('overlapping_keys_with_changed_old_and_full_bins'), sums['old_full_overlap_bins_changed']),
        'stays_with_changed_sampled_key_sets': (
            section.get('stays_with_changed_sampled_key_sets'), stays['stays_with_changed_sampled_key_sets']),
        'stays_with_changed_sampled_keys_including_order': (
            section.get('stays_with_changed_sampled_keys_including_order'),
            stays['stays_with_changed_sampled_keys_including_order']),
        'retained_original_token_keys': (
            fixed.get('retained_original_token_keys'), sums['fixed_selection_retained_keys']),
        'retained_keys_with_changed_old_and_recomputed_frozen_bin': (
            fixed.get('retained_keys_with_changed_old_and_recomputed_frozen_bin'),
            sums['fixed_selection_bins_changed']),
        'retained_keys_with_equal_old_and_recomputed_frozen_bin': (
            fixed.get('retained_keys_with_equal_old_and_recomputed_frozen_bin'),
            sums['fixed_selection_bins_equal']),
    }
    mismatches = {name: list(pair) for name, pair in report_pairs.items()
                  if pair[0] is None or int(pair[0]) != int(pair[1])}
    for name, expected_key in [
        ('old_active_keys', 'old_active_keys'),
        ('retained_original_token_keys', 'retained_original_token_keys'),
        ('retained_keys_with_changed_old_and_recomputed_frozen_bin', 'fixed_selection_bins_changed'),
        ('common_stay_vh_keys', 'old_full_common_keys'),
        ('stays_with_changed_sampled_key_sets', 'stays_sampled_keys_changed'),
    ]:
        expected = EXPECTED[expected_key]
        observed = int(report_pairs[name][1])
        if observed != expected:
            mismatches[f'expected_{name}'] = [expected, observed]
    if mismatches:
        raise ValueError('Full-cache aggregate reconciliation failed: ' + json.dumps(mismatches, sort_keys=True))
    return {'per_stay_sums': sums, 'per_stay_changed_stays': stays,
            'cached_summary_reconciliation': 'passed_exactly'}


def resource_evidence() -> dict:
    evidence = {'proc_self_cgroup': None, 'cgroup_v2_directory': None,
                'memory_max': None, 'cpu_max': None, 'memory_swap_max': None}
    try:
        cgroup_line = next(line for line in Path('/proc/self/cgroup').read_text().splitlines()
                           if line.startswith('0::'))
        relative = cgroup_line.split('::', 1)[1].lstrip('/')
        cg = Path('/sys/fs/cgroup') / relative
        evidence['proc_self_cgroup'] = cgroup_line
        evidence['cgroup_v2_directory'] = str(cg)
        for key, filename in [('memory_max', 'memory.max'), ('cpu_max', 'cpu.max'),
                              ('memory_swap_max', 'memory.swap.max')]:
            path = cg / filename
            evidence[key] = path.read_text().strip() if path.is_file() else None
    except (OSError, StopIteration):
        pass
    return evidence


def aggregate_row(label: str, mask: np.ndarray, subjects: np.ndarray,
                  split: np.ndarray, per: pd.DataFrame) -> dict:
    if mask.dtype != np.bool_ or len(mask) != len(per):
        raise ValueError(f'{label}: invalid aggregation mask')
    frame = per.loc[mask]
    result = {
        'scope': label,
        'n_patients': int(pd.Series(subjects[mask]).nunique()),
        'n_stays': int(mask.sum()),
        'original_sequence_tokens': int(frame['old_sequence_tokens'].sum()),
        'fixed_selection_retained_tokens': int(frame['fixed_selection_retained_keys'].sum()),
        'fixed_selection_removed_tokens': int(frame['fixed_selection_removed_keys'].sum()),
        'fixed_selection_retained_bin_equal_tokens': int(frame['fixed_selection_bins_equal'].sum()),
        'fixed_selection_retained_bin_changed_tokens': int(frame['fixed_selection_bins_changed'].sum()),
        'full_rebuild_sequence_tokens': int(frame['full_rebuild_sequence_tokens'].sum()),
        'old_full_common_sampled_keys': int(frame['old_full_vh_overlap'].sum()),
        'old_full_old_only_sampled_keys': int(frame['old_vh_keys_only'].sum()),
        'old_full_full_only_sampled_keys': int(frame['full_vh_keys_only'].sum()),
        'old_full_common_keys_bin_equal': int(frame['old_full_overlap_bins_equal'].sum()),
        'old_full_common_keys_bin_changed': int(frame['old_full_overlap_bins_changed'].sum()),
        'stays_with_changed_sampled_key_set': int(frame['sampled_key_set_changed'].sum()),
        'stays_with_changed_sampled_key_order': int(frame['sampled_key_order_changed'].sum()),
        'stays_with_changed_sampled_keys_including_order': int(frame['sampled_keys_changed'].sum()),
    }
    result['retained_fraction_of_original_tokens'] = (
        result['fixed_selection_retained_tokens'] / result['original_sequence_tokens']
        if result['original_sequence_tokens'] else None)
    result['removed_fraction_of_original_tokens'] = (
        result['fixed_selection_removed_tokens'] / result['original_sequence_tokens']
        if result['original_sequence_tokens'] else None)
    result['fixed_bin_change_fraction_among_retained_tokens'] = (
        result['fixed_selection_retained_bin_changed_tokens'] / result['fixed_selection_retained_tokens']
        if result['fixed_selection_retained_tokens'] else None)
    result['full_rebuild_old_common_fraction'] = (
        result['old_full_common_sampled_keys'] / result['original_sequence_tokens']
        if result['original_sequence_tokens'] else None)
    return result


def main() -> None:
    input_paths = [AUDIT_PATH, PER_STAY_PATH, DATASET_MANIFEST_PATH, RUN_COMPLETE_PATH,
                   GATE_PATH, GATE_MANIFEST_PATH, SEQ_PATH, SPLIT_PATH]
    missing = [str(p) for p in input_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('Required persisted inputs missing: ' + ', '.join(missing))

    # Hash every consumed file before parsing or loading it.
    input_hashes = {str(path): sha256_file(path) for path in input_paths}
    if input_hashes[str(SEQ_PATH)] != EXPECTED['sequence_sha256']:
        raise ValueError('Frozen MIMIC sequence hash differs from the attested source')
    if input_hashes[str(SPLIT_PATH)] != FROZEN_MIMIC_SPLIT_SHA256:
        raise ValueError('Frozen MIMIC split hash differs from the attested source')
    if input_hashes[str(GATE_MANIFEST_PATH)] != EXPECTED['gate_manifest_sha256']:
        raise ValueError('Reviewed eligibility manifest hash differs from the root-cleared v4.2 manifest')
    if input_hashes[str(GATE_PATH)] != EXPECTED['gate_table_sha256']:
        raise ValueError('MIMIC eligibility table hash differs from the root-cleared v4.2 table')

    report = read_json(AUDIT_PATH)
    dataset_manifest = read_json(DATASET_MANIFEST_PATH)
    run_complete = read_json(RUN_COMPLETE_PATH)
    gate_manifest = read_json(GATE_MANIFEST_PATH)
    if report.get('status') != 'complete_cached_artifact_audit_no_event_or_raw_reread':
        raise ValueError('Unexpected cached audit report status')
    if report.get('per_stay_table') != str(PER_STAY_PATH):
        raise ValueError('Cached report per-stay path binding differs')
    report_sha = input_hashes[str(AUDIT_PATH)]
    per_stay_sha = input_hashes[str(PER_STAY_PATH)]
    audit_binding = dataset_manifest.get('audit_addendum', {})
    if not isinstance(audit_binding, dict):
        raise ValueError('Dataset manifest has no audit_addendum hash binding')
    if audit_binding.get('path') != str(AUDIT_PATH) or audit_binding.get('sha256') != report_sha:
        raise ValueError('Dataset manifest audit_addendum binding is stale')
    if run_complete.get('manifest_sha256') != input_hashes[str(DATASET_MANIFEST_PATH)]:
        raise ValueError('RUN_COMPLETE does not bind the current dataset manifest')
    if run_complete.get('audit_addendum_sha256') != report_sha:
        raise ValueError('RUN_COMPLETE does not bind the current audit_addendum')
    prior_per_stay_sha = audit_binding.get('per_stay_sha256')
    if prior_per_stay_sha is not None and prior_per_stay_sha != per_stay_sha:
        raise ValueError('Dataset manifest per-stay hash binding is stale')
    per_stay_binding = ('verified_existing_manifest_hash' if prior_per_stay_sha is not None
                        else 'current_hash_recorded_no_legacy_per_stay_hash')

    audit_inputs = report.get('inputs', {})
    if audit_inputs.get(str(SEQ_PATH)) != input_hashes[str(SEQ_PATH)]:
        raise ValueError('Cached audit report does not bind the current original sequence NPZ')
    gate_cohort = gate_manifest.get('cohorts', {}).get('mimic', {})
    if gate_manifest.get('gate_version') != 'recovered_reviewed_v4_2':
        raise ValueError('Unexpected eligibility gate version')
    if gate_cohort.get('retrospective_landmark_prediction_allowed') is not True:
        raise ValueError('MIMIC gate does not authorize its reviewed retrospective scope')
    if gate_cohort.get('clinical_input_availability_verified') is not False:
        raise ValueError('Gate clinical availability scope is not explicitly false')
    source_hashes = gate_cohort.get('sourcehashes', {})
    if source_hashes.get('sequence_npz') != input_hashes[str(SEQ_PATH)]:
        raise ValueError('Eligibility manifest sequence hash does not bind the frozen NPZ')
    manifest_split_sha = source_hashes.get('split_npy')
    if manifest_split_sha is not None and manifest_split_sha != input_hashes[str(SPLIT_PATH)]:
        raise ValueError('Eligibility manifest split hash conflicts with the frozen split')
    split_hash_binding = ('reviewed_manifest_sourcehashes' if manifest_split_sha is not None
                          else 'explicit_frozen_split_sha256')

    per = pd.read_parquet(PER_STAY_PATH, columns=PER_STAY_COLUMNS)
    if len(per) != EXPECTED['n_sequence_rows']:
        raise ValueError(f'Per-stay table row count {len(per)} != {EXPECTED["n_sequence_rows"]}')
    if per[PER_STAY_COLUMNS].isna().any().any():
        raise ValueError('Per-stay audit has null required fields')
    per['stay_id'] = normalized_stays(per['stay_id'], 'per-stay audit')
    per_rows = pd.to_numeric(per['row'], errors='raise').to_numpy(dtype=np.int64)
    for name in TOKEN_COLUMNS:
        vals = pd.to_numeric(per[name], errors='raise').to_numpy(dtype=np.int64)
        if (vals < 0).any():
            raise ValueError(f'Per-stay token count is negative: {name}')
        per[name] = vals
    for name in BOOL_COLUMNS:
        if not per[name].isin([True, False]).all():
            raise ValueError(f'Per-stay status is not boolean: {name}')
        per[name] = per[name].astype(bool)

    # Only identity columns are read from the frozen NPZ; never load token arrays.
    with np.load(SEQ_PATH, allow_pickle=False) as archive:
        needed = {'stay_id', 'subject'}
        if not needed.issubset(archive.files):
            raise ValueError(f'Frozen sequence NPZ lacks identity fields: {sorted(needed - set(archive.files))}')
        seq_stay = normalized_stays(archive['stay_id'], 'sequence NPZ')
        seq_subject = archive['subject'].astype(str)
    if len(seq_stay) != EXPECTED['n_sequence_rows'] or len(seq_subject) != len(seq_stay):
        raise ValueError('Frozen sequence identity row count differs from contract')
    if pd.Series(seq_subject).isin(['', 'nan', 'None', 'null']).any():
        raise ValueError('Frozen NPZ has an empty/invalid subject identity')
    position_by_stay = {int(stay): i for i, stay in enumerate(seq_stay)}
    if set(per['stay_id'].tolist()) != set(seq_stay.tolist()):
        raise ValueError('Per-stay audit stay keys differ from the frozen sequence')
    expected_per_rows = per['stay_id'].map(position_by_stay).to_numpy(dtype=np.int64)
    if not np.array_equal(per_rows, expected_per_rows):
        raise ValueError('Per-stay source row mapping differs from the frozen sequence')
    per = per.set_index('stay_id').loc[seq_stay].reset_index()

    split = np.load(SPLIT_PATH, allow_pickle=False)
    if split.shape != (len(seq_stay),) or not np.isin(split, [0, 1, 2]).all():
        raise ValueError('Split vector shape or codes differ from the frozen 0/1/2 contract')
    split = split.astype(np.int8, copy=False)

    gate = pd.read_parquet(GATE_PATH, columns=['stay_id', 'subject', 'eligible_24h', 'y_mort'])
    if len(gate) != EXPECTED['n_sequence_rows']:
        raise ValueError(f'Gate row count {len(gate)} != {EXPECTED["n_sequence_rows"]}')
    gate['stay_id'] = normalized_stays(gate['stay_id'], 'eligibility gate')
    if gate['subject'].isna().any() or gate['subject'].astype(str).isin(['', 'nan', 'None', 'null']).any():
        raise ValueError('Eligibility gate has an empty/invalid subject identity')
    if gate['eligible_24h'].isna().any() or not gate['eligible_24h'].isin([True, False]).all():
        raise ValueError('Eligibility gate eligible_24h is not a complete boolean mask')
    gate['eligible_24h'] = gate['eligible_24h'].astype(bool)
    if set(gate['stay_id'].tolist()) != set(seq_stay.tolist()):
        raise ValueError('Gate stay keys differ from the frozen sequence')
    gate = gate.set_index('stay_id').loc[seq_stay].reset_index()
    if not np.array_equal(gate['subject'].astype(str).to_numpy(), seq_subject):
        raise ValueError('Gate subject identity differs from the frozen NPZ one-to-one')
    selected = gate['eligible_24h'].to_numpy(dtype=bool)
    if int(selected.sum()) != EXPECTED['n_eligible']:
        raise ValueError(f'Gate eligible_24h count {int(selected.sum())} != {EXPECTED["n_eligible"]}')
    selected_labels = pd.to_numeric(gate.loc[selected, 'y_mort'], errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(selected_labels).all() or not np.isin(selected_labels, [0.0, 1.0]).all():
        raise ValueError('Selected gate mortality labels are not complete binary values')

    # Fail closed on all joins; no intersection or silent row truncation is used.
    if per['stay_id'].nunique() != len(seq_stay) or gate['stay_id'].nunique() != len(seq_stay):
        raise ValueError('Stay identity is not one-to-one across the source tables')
    patient_split = pd.DataFrame({'subject': seq_subject, 'split': split}).groupby('subject')['split'].nunique()
    if (patient_split > 1).any():
        raise ValueError('A subject appears in more than one frozen split')

    full_cache_reconciliation = cached_full_report_checks(report, per)
    per_identity = per['stay_id'].to_numpy(dtype=np.int64)
    if not np.array_equal(per_identity, seq_stay):
        raise ValueError('Identity join changed row order before eligibility aggregation')

    masks = [('all_sequence', np.ones(len(seq_stay), dtype=bool)),
             ('eligible_24h', selected)]
    for code in (0, 1, 2):
        in_split = split == code
        masks.extend([(f'all_sequence_split_{code}', in_split),
                      (f'eligible_split_{code}', selected & in_split)])
    rows = [aggregate_row(name, mask, seq_subject, split, per) for name, mask in masks]
    aggregate = pd.DataFrame(rows)

    completed = datetime.now(timezone.utc)
    run_id = f"v5_availability_eligible_{completed.strftime('%Y%m%dT%H%M%SZ')}_{os.getpid()}"
    out_root = R5 / 'availability'
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = out_root / run_id
    if out_dir.exists():
        raise FileExistsError(f'Refusing to overwrite output directory: {out_dir}')
    out_dir.mkdir(parents=False, exist_ok=False)
    csv_path = out_dir / 'availability_eligible_aggregate.csv'
    aggregate.to_csv(csv_path, index=False)
    output_csv_sha = sha256_file(csv_path)

    manifest = {
        'status': 'complete_cached_availability_eligible_aggregate_only',
        'run_id': run_id,
        'completed_utc': completed.isoformat(),
        'scope': 'MIMIC; all-sequence rows and reviewed recovered_reviewed_v4_2 eligible_24h rows; counts stratified by frozen split 0/1/2.',
        'estimand': 'Composite retained-token/fixed-bin/full-rebuild sampled-key audit proxy derived from the persisted v4 audit; not a direct per-variable late-store attribution.',
        'input_hashes_sha256': input_hashes,
        'per_stay_hash_binding': per_stay_binding,
        'per_stay_current_sha256': per_stay_sha,
        'cached_full_sequence_reconciliation': full_cache_reconciliation,
        'expected_full_cache_checks': EXPECTED,
        'eligibility': {
            'gate_version': gate_manifest.get('gate_version'),
            'eligible_column': 'eligible_24h',
            'selected_rows': int(selected.sum()),
            'selected_labels_column': 'y_mort',
            'selected_labels_complete_binary': True,
            'patient_identity_join': 'sequence NPZ stay_id+subject exactly equals gate order; all 94,343 rows required',
            'clinical_input_availability_verified': False,
            'mask_limit': 'Eligibility mask is applied only after all-sequence cached totals reconcile exactly to the historical report.',
        },
        'split': {'values': [0, 1, 2], 'sha256': input_hashes[str(SPLIT_PATH)],
                  'expected_sha256': FROZEN_MIMIC_SPLIT_SHA256,
                  'binding': split_hash_binding,
                  'gate_manifest_split_npy_sha256': manifest_split_sha,
                  'counts_all_sequence': {str(code): int((split == code).sum()) for code in (0, 1, 2)},
                  'patient_disjoint': True},
        'aggregation_definitions': {
            'fixed_selection_retained_tokens': 'sum(fixed_selection_retained_keys); source: cached token-position audit.',
            'fixed_selection_removed_tokens': 'sum(fixed_selection_removed_keys); source: cached token-position audit.',
            'fixed_selection_retained_bin_changed_tokens': 'sum(fixed_selection_bins_changed), denominator is retained tokens.',
            'full_rebuild_sampled_key_changes': 'old/full overlap and old-only/full-only (V,H) keys; order and stay flags from cached per-stay audit.',
            'patient_denominator': 'unique NPZ subject within each row scope.',
            'stay_denominator': 'unique stay_id rows within each row scope.',
            'token_denominator': 'sum of cached old_sequence_tokens within each row scope.',
        },
        'interpretation_limits': [
            'Historical Data6 raw_record_time counts are candidate records and are not counted or described here as retained tokens.',
            'The availability measure is a composite proxy based on persisted fixed-selection retained-token and frozen-bin audit fields; it does not directly attribute every late-store event category.',
            'This audit does not establish bedside availability or prospective clinical input availability.',
            'Old-versus-full-rebuild sampled-key drift is retained as a separate sampling/rebuild change and must not be described as a pure clinical-availability effect.',
            'No raw event table was read, no bins were recomputed, and no patient-level rows or identifiers are written to R5 output.',
        ],
        'resource_evidence': resource_evidence(),
        'output': {'aggregate_csv': str(csv_path), 'aggregate_csv_sha256': output_csv_sha},
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
    readme = f'''# MIMIC eligible availability audit addendum\n\nRun: `{run_id}`\n\nThis report aggregates the persisted v4 per-stay availability audit for all 94,343 frozen MIMIC sequence stays, the reviewed 73,936-row `eligible_24h` subset, and split-specific all-sequence and eligible subsets. The CSV contains aggregate counts and denominators only.\n\nThe audit is a composite proxy built from cached fixed-selection retained-token, frozen-bin, and old-versus-full sampled-key fields. It does not directly attribute every late-store event category and does not establish bedside or prospective clinical input availability. Old-versus-full rebuild key drift is a sampling/rebuild change and must not be presented as a pure clinical-availability effect. Historical Data6 `raw_record_time` counts are candidate records, not retained-token counts.\n\nNo raw event table was opened; bins were not recomputed; patient-level identifiers are not included in the output. See `manifest.json` for source hashes, exact denominator definitions, gate/split checks, and full-cache reconciliation.\n'''
    (out_dir / 'README.md').write_text(readme, encoding='utf-8')
    print(json.dumps({'status': manifest['status'], 'output_dir': str(out_dir),
                      'aggregate_csv_sha256': output_csv_sha,
                      'n_eligible': int(selected.sum()),
                      'per_stay_hash_binding': per_stay_binding}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
