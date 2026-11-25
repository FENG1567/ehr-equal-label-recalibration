"""Create a separate, test-only scoring authorization for both H24 variants.

This copies the already reviewed original MIMIC source gate unchanged and
constructs clean, whitelist eligibility tables for the original held-out split
only. It never changes candidate datasets, model files, or the central registry.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path('/data/admin/fengshuo/ehr_fm_20261102')
REV = ROOT / 'revision_v4_20261122'
BASE = REV / 'temporal/results/eligibility/status_consistent_v4_1/landmark_v4_1'
OUT = REV / 'temporal/results/eligibility/availability_testonly_reviewed_v4_1'
STAGE = OUT.with_name(OUT.name + '_stage')
SCOPE_RECORD = REV / 'admin/availability_controls_scope_record_20261123.md'
SOURCE_SEQ = ROOT / 'data/processed/seq_mimic.npz'
SOURCE_SPLIT = ROOT / 'data/processed/split_mimic.npy'
SOURCE_BINS = ROOT / 'data/processed/bins.json'
ENGINE_CODE = REV / 'engine/code'
AUDIT_ROOT = REV / 'availability_repair/dataset/mimic_available_by24_fixedselection_v4_1/audits'
EXPECTED_ROSTER = {'stays': 14_890, 'patients': 10_848, 'deaths': 1_675}
EXPECTED_COHORT_ROWS = 94_343
EXPECTED_SOURCE_ELIGIBLE = 73_936
EXPECTED_FULL_MANIFEST = '3a0c7d58acef6339a6f87d1fc81d8803c8aaeaf870eae918539fb9891990aa84'
EXPECTED_FIXED_MANIFEST = '48e5f65f78a69a0ed4e105381ab55ab431b257844f1e425dd5e18fff0b641aed'
EXPECTED_FULL_SEQUENCE = '078446262d0bc5f424f203c7fd2f195e0e4108917840c122ec88c7e51ebcc67b'
EXPECTED_FIXED_SEQUENCE = 'd25d245fbaadc2c8ce0068afab87b72e95e151293308c2945791ef09c4c9ccd1'
EXPECTED_FIXED_PREADDENDUM_SEQUENCE = EXPECTED_FIXED_SEQUENCE

VARIANTS = (
    {
        'db': 'mimic_available_by24_v4_1',
        'directory': REV / 'availability_repair/dataset/mimic_available_by24_v4_1',
        'selection_by24_verified': True,
        'analysis_role': 'same_original_heldout_test_only_score_sensitivity',
        'input_scope': 'documented_timestamp_H24_proxy_with_new_random_sampling',
        'interpretation': (
            'Sequence keys were sampled from the H24-filtered documented-source aggregate. '
            'This is a timestamp-based input proxy, not verified real-time clinical availability; '
            'the comparison includes new random token resampling.'),
    },
    {
        'db': 'mimic_available_by24_fixedselection_v4_1',
        'directory': REV / 'availability_repair/dataset/mimic_available_by24_fixedselection_v4_1',
        'selection_by24_verified': False,
        'analysis_role': 'same_original_heldout_test_only_score_sensitivity',
        'input_scope': 'controlled_retrospective_fixed_original_key_perturbation',
        'interpretation': (
            'Controlled retrospective perturbation of the original measurement-time-sampled '
            'keyset: unavailable keys are removed and survivors retain their old relative order. '
            'The old key selection can depend on records unavailable by H24; selection_by24_verified=false.'),
    },
)

TABLE_COLUMNS = [
    'row', 'db', 'stay_id', 'subject', 'split', 'split_role', 'y_mort',
    'source_y_mort', 'source_endpoint_known', 'source_y_mort_match', 'endpoint',
    'eligible_24h', 'temporal_verified', 'retrospective_landmark_prediction_allowed',
    'clinical_input_availability_verified', 'ready_for_future24h_clinical_claim',
    'independent_external_cohort', 'derivative_of', 'resampling_unit',
    'same_original_heldout_patients', 'selection_by24_verified', 'input_scope',
    'analysis_role', 'reason',
]
FORBIDDEN_READINESS_ALIASES = {
    'ready_for_future24h_landmark_analysis',
    'ready_for_retrospective_24h_measurement_analysis',
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')
    os.replace(tmp, path)


def verify_scope_record(text: str) -> None:
    required = [
        'SCOPE: TEST_ONLY',
        '14,890 stays, 10,848 subjects, 1,675 hospital deaths',
        'mimic_available_by24_v4_1',
        'mimic_available_by24_fixedselection_v4_1',
        'all 94,343 ordered stay and subject IDs',
        'Every active token in both variants matched',
        'Source/test patients are disjoint',
        'selection_by24_verified=false',
    ]
    missing = [token for token in required if token not in text]
    if missing:
        raise ValueError('Scope record is absent or does not cover the exact test-only scope: ' + repr(missing))


def as_bool(series: pd.Series, name: str) -> np.ndarray:
    if series.isna().any():
        raise ValueError(f'{name} contains null boolean values')
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.to_numpy(dtype=bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {'true': True, '1': True, 'yes': True, 'false': False, '0': False, 'no': False}
    unknown = set(normalized.unique()) - set(mapping)
    if unknown:
        raise ValueError(f'{name} contains invalid boolean values: {sorted(unknown)}')
    return normalized.map(mapping).to_numpy(dtype=bool)


def select_test_rows(original_eligible: np.ndarray, split: np.ndarray,
                     lengths: np.ndarray) -> np.ndarray:
    original_eligible = np.asarray(original_eligible, dtype=bool)
    split = np.asarray(split)
    lengths = np.asarray(lengths)
    if original_eligible.ndim != 1 or split.ndim != 1 or lengths.ndim != 1:
        raise ValueError('Eligibility, split, and length must be one-dimensional')
    if not (len(original_eligible) == len(split) == len(lengths)):
        raise ValueError('Eligibility, split, and length row counts differ')
    if not np.isin(split, [0, 1, 2]).all():
        raise ValueError('Frozen split contains a code outside train/validation/test')
    if np.any(lengths < 0):
        raise ValueError('Sequence contains a negative active length')
    return original_eligible & (split == 2) & (lengths > 0)


def assert_identity_and_labels(frame: pd.DataFrame, stay_id: np.ndarray,
                               subject: np.ndarray, label: np.ndarray,
                               context: str) -> None:
    if not np.array_equal(frame['stay_id'].astype(str).to_numpy(), np.asarray(stay_id).astype(str)):
        raise ValueError(f'{context}: ordered stay IDs differ from sequence')
    if not np.array_equal(frame['subject'].astype(str).to_numpy(), np.asarray(subject).astype(str)):
        raise ValueError(f'{context}: ordered subject IDs differ from sequence')
    observed = pd.to_numeric(frame['y_mort'], errors='coerce').to_numpy(dtype=np.float64)
    expected = np.asarray(label, dtype=np.float64)
    if not np.array_equal(observed, expected, equal_nan=True):
        raise ValueError(f'{context}: source y_mort differs from sequence labels')


def assert_split_alignment(split: np.ndarray, expected: np.ndarray, context: str) -> None:
    if not np.array_equal(np.asarray(split), np.asarray(expected)):
        raise ValueError(f'{context}: split codes differ from frozen original split')


def assert_sequence_contract(ds: dict[str, np.ndarray], context: str) -> None:
    n = len(ds['L'])
    if ds['V'].shape != ds['B'].shape or ds['V'].shape != ds['H'].shape:
        raise ValueError(f'{context}: V/B/H shapes differ')
    if ds['V'].shape != (n, 128) or np.any(ds['L'] < 0) or np.any(ds['L'] > 128):
        raise ValueError(f'{context}: sequence shape or lengths violate the 128-token contract')
    active = np.arange(128)[None, :] < ds['L'][:, None]
    if np.any((ds['V'][active] < 1) | (ds['V'][active] > 48)):
        raise ValueError(f'{context}: invalid active variable ID')
    if np.any((ds['B'][active] < 1) | (ds['B'][active] > 10)):
        raise ValueError(f'{context}: invalid active frozen bin')
    if np.any((ds['H'][active] < 0) | (ds['H'][active] >= 24)):
        raise ValueError(f'{context}: invalid active H24 hour')
    if np.any(ds['V'][~active]) or np.any(ds['B'][~active]) or np.any(ds['H'][~active]):
        raise ValueError(f'{context}: nonzero sequence padding')


def build_whitelist_table(source: pd.DataFrame, ds: dict[str, np.ndarray], split: np.ndarray,
                          selected: np.ndarray, variant: dict[str, Any]) -> pd.DataFrame:
    n = len(source)
    if len(ds['stay_id']) != n or len(split) != n or len(selected) != n:
        raise ValueError('Source, sequence, split, and authorization rows do not align')
    endpoint = source['endpoint'].astype(str).to_numpy()
    if not np.isin(endpoint, ['hospital']).all():
        raise ValueError('MIMIC source endpoint must be hospital discharge mortality')
    source_y = pd.to_numeric(source['y_mort'], errors='coerce').to_numpy(dtype=np.float64)
    seq_y = np.asarray(ds['y_mort'], dtype=np.float64)
    label_known = np.isin(seq_y, [0, 1])
    label_match = label_known & (source_y == seq_y)
    if not np.array_equal(source_y, seq_y, equal_nan=True):
        raise ValueError('Source-gate y_mort and sequence y_mort are not identical')
    if (selected & ~label_known).any() or (selected & ~label_match).any():
        raise ValueError('Selected test rows must have known, matching source hospital mortality')
    if not np.array_equal(selected, select_test_rows(
            as_bool(source['eligible_24h'], 'source eligible_24h') &
            as_bool(source['temporal_verified'], 'source temporal_verified'),
            split, ds['L'])):
        raise ValueError('Authorization rows differ from original eligible split-2 nonempty-input rule')

    reasons = np.full(n, 'excluded_unknown_source_endpoint', dtype=object)
    reasons[split == 0] = 'excluded_train_split'
    reasons[split == 1] = 'excluded_validation_split'
    base_eligible = (as_bool(source['eligible_24h'], 'source eligible_24h') &
                     as_bool(source['temporal_verified'], 'source temporal_verified'))
    reasons[(split == 2) & ~base_eligible] = 'excluded_original_source_gate'
    reasons[(split == 2) & base_eligible & (ds['L'] == 0)] = 'excluded_empty_variant_sequence'
    reasons[selected] = 'authorized_original_test_split_score_only'
    result = pd.DataFrame({
        'row': np.arange(n, dtype=np.int32),
        'db': variant['db'],
        'stay_id': source['stay_id'].to_numpy(copy=True),
        'subject': np.asarray(ds['subject']).astype(str),
        'split': np.asarray(split, dtype=np.int8),
        'split_role': np.choose(np.asarray(split, dtype=np.int8), ['train', 'validation', 'test']),
        'y_mort': seq_y,
        'source_y_mort': source_y,
        'source_endpoint_known': label_known,
        'source_y_mort_match': label_match,
        'endpoint': endpoint,
        'eligible_24h': selected,
        'temporal_verified': selected,
        'retrospective_landmark_prediction_allowed': selected,
        'clinical_input_availability_verified': np.zeros(n, dtype=bool),
        'ready_for_future24h_clinical_claim': np.zeros(n, dtype=bool),
        'independent_external_cohort': np.zeros(n, dtype=bool),
        'derivative_of': np.full(n, 'mimic', dtype=object),
        'resampling_unit': np.full(n, 'patient', dtype=object),
        'same_original_heldout_patients': selected,
        'selection_by24_verified': np.full(n, bool(variant['selection_by24_verified']), dtype=bool),
        'input_scope': np.full(n, variant['input_scope'], dtype=object),
        'analysis_role': np.full(n, variant['analysis_role'], dtype=object),
        'reason': reasons,
    })
    if set(result.columns) != set(TABLE_COLUMNS):
        raise AssertionError('Fresh whitelist output schema changed unexpectedly')
    if FORBIDDEN_READINESS_ALIASES & set(result.columns):
        raise AssertionError('Generic readiness aliases must be omitted')
    return result[TABLE_COLUMNS]


def load_sequence(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in
                ('V', 'B', 'H', 'L', 'stay_id', 'subject', 'y_mort', 'y_aki', 'y_long')}


def read_and_verify_inputs() -> dict[str, Any]:
    if OUT.exists() or STAGE.exists():
        raise FileExistsError('Preserve existing authorization or partial stage; never overwrite')
    if not SCOPE_RECORD.is_file():
        raise FileNotFoundError('A documented test-only scope record is required')
    scope_text = SCOPE_RECORD.read_text(encoding='utf-8')
    verify_scope_record(scope_text)
    scope_sha = sha256_file(SCOPE_RECORD)
    scope_copy = Path('/data/admin/fengshuo/ehr_fm_20261102/revision_v4_20261122/admin/availability_controls_scope_record_20261123.md')
    if not scope_copy.is_file() or sha256_file(scope_copy) != scope_sha:
        raise ValueError('Server scope record is missing or differs from the local record')
    base_manifest_path = BASE / 'eligibility_manifest.json'
    base_table_path = BASE / 'mimic.parquet'
    required = [base_manifest_path, base_table_path, SOURCE_SEQ, SOURCE_SPLIT, SOURCE_BINS]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError('Original source gate or frozen cohort files are incomplete')
    base_manifest = json.loads(base_manifest_path.read_text(encoding='utf-8'))
    base_entry = base_manifest.get('cohorts', {}).get('mimic')
    if not isinstance(base_entry, dict) or base_entry.get('retrospective_landmark_prediction_allowed') is not True:
        raise ValueError('Original MIMIC source eligibility profile is not positively authorized')
    base_table = pd.read_parquet(base_table_path)
    with np.load(SOURCE_SEQ, allow_pickle=False) as archive:
        original = {key: archive[key].copy() for key in
                    ('V', 'B', 'H', 'L', 'stay_id', 'subject', 'y_mort', 'y_aki', 'y_long')}
    original_split = np.load(SOURCE_SPLIT, allow_pickle=False)
    if len(base_table) != EXPECTED_COHORT_ROWS or len(original['stay_id']) != EXPECTED_COHORT_ROWS:
        raise ValueError('Original source gate/sequence do not contain the frozen 94,343 rows')
    assert_sequence_contract(original, 'original source MIMIC')
    assert_identity_and_labels(base_table, original['stay_id'], original['subject'], original['y_mort'], 'original source gate')
    if len(original_split) != EXPECTED_COHORT_ROWS or not np.isin(original_split, [0, 1, 2]).all():
        raise ValueError('Original frozen split has invalid size or split code')
    original_eligible = as_bool(base_table['eligible_24h'], 'original source eligible_24h') & \
                        as_bool(base_table['temporal_verified'], 'original source temporal_verified')
    if int(original_eligible.sum()) != EXPECTED_SOURCE_ELIGIBLE:
        raise ValueError('Original source gate eligible MIMIC count changed')
    if int(base_entry.get('counts', {}).get('n_eligible_24h', -1)) != EXPECTED_SOURCE_ELIGIBLE:
        raise ValueError('Original source manifest MIMIC eligible count changed')
    if base_entry.get('sourcehashes', {}).get('sequence_npz') != sha256_file(SOURCE_SEQ):
        raise ValueError('Original source gate sequence hash does not match frozen source NPZ')

    subject = np.asarray(original['subject']).astype(str)
    split_subjects = [set(subject[original_split == code]) for code in (0, 1, 2)]
    if any(split_subjects[a] & split_subjects[b]
           for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('Frozen source split has cross-split patient overlap')
    source_fit_eligible = original_eligible & (original_split == 0)
    test_eligible = original_eligible & (original_split == 2)
    if int(source_fit_eligible.sum()) != 51_714 or int(test_eligible.sum()) != EXPECTED_ROSTER['stays']:
        raise ValueError('Original eligible split 0/test roster differs from the documented scope record')
    fit_subjects = set(subject[source_fit_eligible])
    selected_subjects = set(subject[test_eligible])
    if fit_subjects & selected_subjects:
        raise ValueError('Original source fit and held-out test patients overlap')
    if int(np.sum(original['y_mort'][test_eligible] == 1)) != EXPECTED_ROSTER['deaths']:
        raise ValueError('Original test-split hospital-death count differs from the documented scope record')
    if len(selected_subjects) != EXPECTED_ROSTER['patients']:
        raise ValueError('Original test-split patient count differs from the documented scope record')

    semantic_path = AUDIT_ROOT / 'semantic_review_scope.json'
    revision_path = AUDIT_ROOT / 'metadata_revision_audit.json'
    addendum_path = AUDIT_ROOT / 'audit_addendum.json'
    semantic = json.loads(semantic_path.read_text(encoding='utf-8'))
    revision = json.loads(revision_path.read_text(encoding='utf-8'))
    addendum = json.loads(addendum_path.read_text(encoding='utf-8'))
    semantic_full = semantic.get('datasets', {}).get('mimic_available_by24_v4_1', {})
    semantic_fixed = semantic.get('datasets', {}).get('mimic_available_by24_fixedselection_v4_1', {})
    if semantic_full.get('dataset_manifest_sha256') != EXPECTED_FULL_MANIFEST:
        raise ValueError('Full-rebuild semantic sidecar does not bind the reviewed manifest')
    if semantic_fixed.get('dataset_manifest_sha256') != EXPECTED_FIXED_MANIFEST:
        raise ValueError('Fixed-selection semantic sidecar does not bind the current reviewed manifest')
    if semantic_fixed.get('sequence_sha256') != EXPECTED_FIXED_SEQUENCE:
        raise ValueError('Fixed-selection semantic sidecar does not bind the reviewed sequence')
    if semantic_fixed.get('selection_by24_verified') is not False:
        raise ValueError('Fixed-selection semantic review scope must keep selection_by24_verified=false')
    if semantic.get('shared_claim_limits', {}).get('clinical_input_availability_verified') is not False:
        raise ValueError('Semantic review sidecar must keep clinical input availability false')
    if revision.get('sequence_integrity', {}).get('fixed_selection_npz_sha256_before_and_after') != EXPECTED_FIXED_SEQUENCE:
        raise ValueError('Metadata revision sidecar does not bind the current fixed sequence')
    if revision.get('revision', {}).get('fixed_selection_manifest', {}).get('current_sha256') != EXPECTED_FIXED_MANIFEST:
        raise ValueError('Metadata revision sidecar does not bind the current fixed manifest')
    if addendum.get('status') != 'complete_cached_artifact_audit_no_event_or_raw_reread':
        raise ValueError('Cached-artifact audit addendum is incomplete')

    source_hashes = {
        'source_gate_manifest': sha256_file(base_manifest_path),
        'source_gate_mimic_table': sha256_file(base_table_path),
        'source_sequence_npz': sha256_file(SOURCE_SEQ),
        'source_split_npy': sha256_file(SOURCE_SPLIT),
        'source_frozen_bins_json': sha256_file(SOURCE_BINS),
        'scope_record_sha': scope_sha,
        'semantic_review_scope': sha256_file(semantic_path),
        'metadata_revision_audit': sha256_file(revision_path),
        'audit_addendum': sha256_file(addendum_path),
    }
    variants = []
    reviewed_roster_ids: np.ndarray | None = None
    reviewed_roster_subjects: np.ndarray | None = None
    for spec in VARIANTS:
        dataset_manifest_path = spec['directory'] / 'dataset_manifest.json'
        registry_path = spec['directory'] / 'registry_fragment.json'
        if not dataset_manifest_path.is_file() or not registry_path.is_file():
            raise FileNotFoundError(f"{spec['db']}: completed dataset manifest/registry fragment missing")
        ds_manifest_sha = sha256_file(dataset_manifest_path)
        expected_manifest = EXPECTED_FULL_MANIFEST if spec['db'] == 'mimic_available_by24_v4_1' else EXPECTED_FIXED_MANIFEST
        if ds_manifest_sha != expected_manifest:
            raise ValueError(f"{spec['db']}: current reviewed manifest hash changed")
        ds_manifest = json.loads(dataset_manifest_path.read_text(encoding='utf-8'))
        registry = json.loads(registry_path.read_text(encoding='utf-8'))
        if registry.get('dataset') != spec['db']:
            raise ValueError(f"{spec['db']}: source registry fragment names another dataset")
        seq_path = spec['directory'] / f"seq_{spec['db']}.npz"
        split_path = spec['directory'] / f"split_{spec['db']}.npy"
        tab_path = spec['directory'] / f"tab_{spec['db']}.parquet"
        vocab_path = spec['directory'] / f"vocab_{spec['db']}.json"
        cohort_path = spec['directory'] / f"cohort_{spec['db']}.parquet"
        cohort_manifest = ds_manifest.get('output_hashes', {})
        for path in (seq_path, split_path, tab_path, vocab_path, cohort_path):
            if not path.is_file():
                raise FileNotFoundError(f"{spec['db']}: required source artifact missing: {path}")
        seq_sha = sha256_file(seq_path)
        expected_seq = EXPECTED_FULL_SEQUENCE if spec['db'] == 'mimic_available_by24_v4_1' else EXPECTED_FIXED_SEQUENCE
        if seq_sha != expected_seq:
            raise ValueError(f"{spec['db']}: reviewed sequence SHA changed")
        split_sha = sha256_file(split_path)
        if split_sha != source_hashes['source_split_npy']:
            raise ValueError(f"{spec['db']}: split bytes differ from frozen original split")
        for path in (seq_path, split_path, tab_path):
            entry = cohort_manifest.get(path.name, {})
            if entry.get('sha256') != sha256_file(path):
                raise ValueError(f"{spec['db']}: output hash mismatch for {path.name}")
        ds = load_sequence(seq_path)
        assert_sequence_contract(ds, spec['db'])
        assert_identity_and_labels(base_table, ds['stay_id'], ds['subject'], ds['y_mort'], spec['db'] + ' source gate')
        for key in ('y_aki', 'y_long'):
            if not np.array_equal(ds[key], original[key], equal_nan=True):
                raise ValueError(f"{spec['db']}: frozen label drift for {key}")
        assert_split_alignment(np.load(split_path, allow_pickle=False), original_split, spec['db'])
        cohort = pd.read_parquet(cohort_path)
        cohort_for_check = cohort.rename(columns={
            'subject_id': 'subject', 'mort_hosp': 'y_mort',
        })
        assert_identity_and_labels(cohort_for_check, ds['stay_id'], ds['subject'], ds['y_mort'], spec['db'] + ' cohort')
        selected = select_test_rows(original_eligible, original_split, ds['L'])
        table = build_whitelist_table(base_table, ds, original_split, selected, spec)
        if int(selected.sum()) != EXPECTED_ROSTER['stays']:
            raise ValueError(f"{spec['db']}: selected stay roster differs from independent GO")
        selected_subjects = set(table.loc[selected, 'subject'])
        if len(selected_subjects) != EXPECTED_ROSTER['patients']:
            raise ValueError(f"{spec['db']}: selected patient count differs from independent GO")
        if int(table.loc[selected, 'y_mort'].sum()) != EXPECTED_ROSTER['deaths']:
            raise ValueError(f"{spec['db']}: selected hospital-death count differs from independent GO")
        roster_ids = table.loc[selected, 'stay_id'].astype(str).to_numpy()
        roster_subjects = table.loc[selected, 'subject'].astype(str).to_numpy()
        if reviewed_roster_ids is None:
            reviewed_roster_ids = roster_ids
            reviewed_roster_subjects = roster_subjects
        elif (not np.array_equal(roster_ids, reviewed_roster_ids)
              or not np.array_equal(roster_subjects, reviewed_roster_subjects)):
            raise ValueError('The two variants do not use the same ordered original held-out patient roster')
        variants.append({
            **spec, 'dataset_manifest_path': dataset_manifest_path,
            'dataset_manifest': ds_manifest, 'dataset_manifest_sha256': ds_manifest_sha,
            'source_registry_path': registry_path, 'source_registry_sha256': sha256_file(registry_path),
            'sequence_path': seq_path, 'sequence_sha256': seq_sha,
            'split_path': split_path, 'split_sha256': split_sha,
            'tabular_path': tab_path, 'tabular_sha256': sha256_file(tab_path),
            'vocab_path': vocab_path, 'vocab_sha256': sha256_file(vocab_path),
            'cohort_path': cohort_path, 'cohort_sha256': sha256_file(cohort_path),
            'table': table, 'selected': selected,
        })
    return {
        'scope_sha256': scope_sha, 'scope_text': scope_text,
        'base_manifest': base_manifest, 'base_entry': base_entry,
        'base_manifest_path': base_manifest_path, 'base_manifest_sha256': source_hashes['source_gate_manifest'],
        'base_table': base_table, 'base_table_path': base_table_path,
        'base_table_sha256': source_hashes['source_gate_mimic_table'],
        'original': original, 'original_split': original_split,
        'source_hashes': source_hashes, 'fit_subjects': fit_subjects,
        'selected_subjects': selected_subjects, 'source_fit_stays': int(source_fit_eligible.sum()),
        'variants': variants,
    }


def build_entry(v: dict[str, Any], context: dict[str, Any], table_path: Path,
                table_sha256: str) -> dict[str, Any]:
    selected = v['table'].loc[v['selected']]
    label = selected['y_mort'].to_numpy(dtype=np.float64)
    return {
        'status': 'reviewed_ready',
        'evidence': 'operational_proxy',
        'evidence_basis': 'Documented test-only scope; documented timestamp H24 input sensitivity, score-only original held-out MIMIC rows.',
        'retrospective_landmark_prediction_allowed': True,
        'retrospective_diagnostic_allowed': False,
        'prediction_scope': 'test_only_retrospective_score_only_sensitivity',
        'input_scope': 'retrospective_measurement_time',
        'variant_input_scope': v['input_scope'],
        'analysis_role': v['analysis_role'],
        'clinical_input_availability_verified': False,
        'ready_for_future24h_clinical_claim': False,
        'independent_external_cohort': False,
        'derivative_of': 'mimic',
        'resampling_unit': 'patient',
        'same_original_heldout_patients': True,
        'selection_by24_verified': bool(v['selection_by24_verified']),
        'n_rows': EXPECTED_COHORT_ROWS,
        'n_eligible_24h': len(selected),
        'n_eligible_positive': int(np.sum(label == 1)),
        'n_eligible_negative': int(np.sum(label == 0)),
        'n_eligible_patients': int(selected['subject'].nunique()),
        'n_original_source_eligible': EXPECTED_SOURCE_ELIGIBLE,
        'n_original_source_fit_split0_eligible': context['source_fit_stays'],
        'n_original_source_fit_split0_patients': len(context['fit_subjects']),
        'n_original_heldout_split2_patients': len(context['selected_subjects']),
        'source_fit_test_subjects_disjoint_verified': True,
        'n_source_fit_test_subject_overlap': 0,
        'sourcehashes': {
            'sequence_npz': v['sequence_sha256'],
            'split_npy': v['split_sha256'],
            'tabular_parquet': v['tabular_sha256'],
            'vocab_json': v['vocab_sha256'],
            'cohort_table': v['cohort_sha256'],
            'dataset_manifest': v['dataset_manifest_sha256'],
            'source_registry_fragment': v['source_registry_sha256'],
            'eligibility_table': table_sha256,
            'source_gate_manifest': context['base_manifest_sha256'],
            'source_gate_mimic_table': context['base_table_sha256'],
            'source_sequence_npz': context['source_hashes']['source_sequence_npz'],
            'source_split_npy': context['source_hashes']['source_split_npy'],
            'frozen_bins_json': context['source_hashes']['source_frozen_bins_json'],
            'scope_record': context['scope_sha256'],
            'semantic_review_scope': context['source_hashes']['semantic_review_scope'],
            'metadata_revision_audit': context['source_hashes']['metadata_revision_audit'],
            'audit_addendum': context['source_hashes']['audit_addendum'],
        },
        'scope_sha256': context['scope_sha256'],
        'score_only_authorization': True,
        'authorization_limit': (
            'Allows scoring only on the same original MIMIC split-2 test stays using the frozen original source head/checkpoint; '
            'no source refit, external-cohort claim, clinical real-time availability claim, or gate-wide promotion.'),
        'interpretation': v['interpretation'],
        'table_sha256': table_sha256,
        'eligibility_table_path': str(table_path),
    }


def validate_flags(table: pd.DataFrame, expected_selected: np.ndarray, context: str) -> None:
    for name in ('eligible_24h', 'temporal_verified', 'retrospective_landmark_prediction_allowed'):
        observed = table[name].to_numpy(dtype=bool)
        if not np.array_equal(observed, expected_selected):
            raise ValueError(f'{context}: {name} includes a train/validation or otherwise unauthorized row')
    if table['clinical_input_availability_verified'].astype(bool).any():
        raise ValueError(f'{context}: clinical availability must remain false for every row')
    if table['ready_for_future24h_clinical_claim'].astype(bool).any():
        raise ValueError(f'{context}: future clinical claims must remain false for every row')
    if table.loc[expected_selected, 'split'].ne(2).any():
        raise ValueError(f'{context}: selected authorization includes a non-test split row')
    if table.loc[expected_selected, 'source_endpoint_known'].ne(True).any():
        raise ValueError(f'{context}: selected authorization includes unknown source endpoints')
    if table.loc[expected_selected, 'source_y_mort_match'].ne(True).any():
        raise ValueError(f'{context}: selected source labels do not match actual sequence labels')
    actual_y = pd.to_numeric(table['y_mort'], errors='coerce').to_numpy(dtype=np.float64)
    source_y = pd.to_numeric(table['source_y_mort'], errors='coerce').to_numpy(dtype=np.float64)
    known = np.isin(source_y, [0, 1])
    exact_match = known & (actual_y == source_y)
    if not np.array_equal(table['source_endpoint_known'].to_numpy(dtype=bool), known):
        raise ValueError(f'{context}: source_endpoint_known does not match the actual source label')
    if not np.array_equal(table['source_y_mort_match'].to_numpy(dtype=bool), exact_match):
        raise ValueError(f'{context}: source_y_mort_match does not match actual labels')
    if (expected_selected & (~known | ~exact_match)).any():
        raise ValueError(f'{context}: selected outcome is unknown or differs from source y_mort')
    split_role = np.choose(table['split'].to_numpy(dtype=np.int8), ['train', 'validation', 'test'])
    if not np.array_equal(table['split_role'].astype(str).to_numpy(), split_role):
        raise ValueError(f'{context}: split role does not match the frozen split code')
    if FORBIDDEN_READINESS_ALIASES & set(table.columns):
        raise ValueError(f'{context}: forbidden generic readiness alias present')


def load_engine_module():
    sys.path.insert(0, str(ENGINE_CODE))
    import v4_engine as engine  # noqa: PLC0415
    return engine


def verify_with_engine(folder: Path, context: dict[str, Any], expected_tables: dict[str, pd.DataFrame],
                       expected_sequences: dict[str, Path]) -> dict[str, dict[str, Any]]:
    engine = load_engine_module()
    evidence_results: dict[str, dict[str, Any]] = {}
    source_selected, source_evidence = engine.load_ready_eligibility('mimic', folder)
    if len(source_selected) != EXPECTED_SOURCE_ELIGIBLE:
        raise ValueError('Actual engine loader did not preserve the original 73,936-row source MIMIC profile')
    engine.assert_sequence_npz_attestation('mimic', source_evidence, sha256_file(SOURCE_SEQ))
    evidence_results['mimic'] = {
        'selected_rows': len(source_selected),
        'eligibility_manifest_sha256': source_evidence['eligibility_manifest_sha256'],
        'eligibility_table_sha256': source_evidence['eligibility_table_sha256'],
        'sequence_npz_sha256': source_evidence['sequence_npz_sha256'],
    }
    for db, expected_table in expected_tables.items():
        selected, entry = engine.load_ready_eligibility(db, folder)
        seq_path = expected_sequences[db]
        engine.assert_sequence_npz_attestation(db, entry, sha256_file(seq_path))
        expected_ids = expected_table.loc[expected_table['eligible_24h'], 'stay_id'].astype(str).to_numpy()
        if not np.array_equal(selected['stay_id'].astype(str).to_numpy(), expected_ids):
            raise ValueError(f'{db}: engine loader changed ordered test-only stay IDs')
        if not np.array_equal(selected['subject'].astype(str).to_numpy(),
                              expected_table.loc[expected_table['eligible_24h'], 'subject'].astype(str).to_numpy()):
            raise ValueError(f'{db}: engine loader changed ordered test-only subject IDs')
        if not np.array_equal(selected['y_mort'].to_numpy(dtype=np.float64),
                              expected_table.loc[expected_table['eligible_24h'], 'y_mort'].to_numpy(dtype=np.float64)):
            raise ValueError(f'{db}: engine loader changed selected mortality labels')
        if entry.get('retrospective_landmark_prediction_allowed') is not True:
            raise ValueError(f'{db}: score-only row-level authorization not recognized by engine')
        if entry.get('clinical_input_availability_verified') is not False:
            raise ValueError(f'{db}: clinical availability claim is not closed')
        evidence_results[db] = {
            'selected_rows': len(selected),
            'selected_patients': int(selected['subject'].nunique()),
            'selected_deaths': int(selected['y_mort'].sum()),
            'eligibility_manifest_sha256': entry['eligibility_manifest_sha256'],
            'eligibility_table_sha256': entry['eligibility_table_sha256'],
            'sequence_npz_sha256': entry['sequence_npz_sha256'],
            'retrospective_landmark_prediction_allowed': entry['retrospective_landmark_prediction_allowed'],
            'clinical_input_availability_verified': entry.get('clinical_input_availability_verified'),
        }
    return evidence_results


def main() -> None:
    context = read_and_verify_inputs()
    if sha256_file(context['base_table_path']) != context['base_table_sha256']:
        raise AssertionError('Source MIMIC gate table changed during finalization')
    STAGE.mkdir(parents=True, exist_ok=False)
    source_table_stage = STAGE / 'mimic.parquet'
    shutil.copy2(context['base_table_path'], source_table_stage)
    if sha256_file(source_table_stage) != context['base_table_sha256']:
        raise ValueError('Source MIMIC compatibility table copy is not byte-identical')

    source_entry = dict(context['base_entry'])
    cohort_entries: dict[str, dict[str, Any]] = {'mimic': source_entry}
    proposed_registry: dict[str, Any] = {
        'registry_version': OUT.name,
        'purpose': 'Proposed new-ID-only registry fragment; central data_registry.json is not modified.',
        'scope_sha256': context['scope_sha256'],
        'cohorts': {},
    }
    frames: dict[str, pd.DataFrame] = {}
    sequences: dict[str, Path] = {}
    for v in context['variants']:
        db = v['db']
        table = v['table']
        validate_flags(table, v['selected'], db)
        table_path = STAGE / f'{db}.parquet'
        table.to_parquet(table_path, index=False)
        table_sha = sha256_file(table_path)
        entry = build_entry(v, context, OUT / table_path.name, table_sha)
        cohort_entries[db] = entry
        proposed_registry['cohorts'][db] = {
            'sequence_path': str(v['sequence_path']),
            'split_path': str(v['split_path']),
            'tabular_path': str(v['tabular_path']),
            'vocab_path': str(v['vocab_path']),
            'cohort_path': str(v['cohort_path']),
            'eligibility_table_path': str(OUT / table_path.name),
            'eligibility_manifest_path': str(OUT / 'eligibility_manifest.json'),
            'source_dataset_manifest_sha256': v['dataset_manifest_sha256'],
            'source_registry_fragment_sha256': v['source_registry_sha256'],
            'eligibility_table_sha256': table_sha,
            'scope_sha256': context['scope_sha256'],
            'selection_by24_verified': bool(v['selection_by24_verified']),
            'analysis_role': v['analysis_role'],
            'independent_external_cohort': False,
            'derivative_of': 'mimic',
            'resampling_unit': 'patient',
        }
        frames[db] = table
        sequences[db] = v['sequence_path']
        print(json.dumps({'stage': 'TESTONLY_TABLE_WRITTEN', 'db': db,
                          'rows': len(table), 'eligible_test_rows': int(v['selected'].sum()),
                          'positive_test_rows': int(table.loc[v['selected'], 'y_mort'].sum()),
                          'patients': int(table.loc[v['selected'], 'subject'].nunique()),
                          'table_sha256': table_sha}, sort_keys=True), flush=True)

    manifest = {
        'status': 'complete_testonly_scope_record',
        'gate_version': OUT.name,
        'scope_decision': 'TEST_ONLY',
        'analysis_role': 'same_original_heldout_mimic_test_only_score_sensitivity',
        'clinical_input_availability_verified': False,
        'ready_for_future24h_clinical_claim': False,
        'score_only_authorization': True,
        'source_eligibility_compatibility': {
            'profile': 'original reviewed MIMIC source-head eligibility; copied unchanged for source-head reuse',
            'original_source_gate_manifest_sha256': context['base_manifest_sha256'],
            'original_source_mimic_table_sha256': context['base_table_sha256'],
            'n_original_eligible': EXPECTED_SOURCE_ELIGIBLE,
        },
        'scope_record_path': str(SCOPE_RECORD),
        'scope_record_sha256': context['scope_sha256'],
        'source_hashes': context['source_hashes'],
        'semantic_review_scope_path': str(AUDIT_ROOT / 'semantic_review_scope.json'),
        'metadata_revision_audit_path': str(AUDIT_ROOT / 'metadata_revision_audit.json'),
        'audit_addendum_path': str(AUDIT_ROOT / 'audit_addendum.json'),
        'eligibility_table_columns': TABLE_COLUMNS,
        'forbidden_generic_readiness_aliases_omitted': sorted(FORBIDDEN_READINESS_ALIASES),
        'cohorts': cohort_entries,
        'limits': [
            'Per-cohort retrospective_landmark_prediction_allowed=true authorizes score-only comparison on original split-2 test rows only.',
            'No source refit, train/validation evaluation as a target, external-cohort claim, or central registry change is authorized.',
            'Clinical/real-time input availability remains unverified for both variants.',
            'Full rebuild is a documented-timestamp H24 proxy with new random token sampling; fixed-selection is a controlled retrospective perturbation with selection_by24_verified=false.',
            'Pre-ICU therapy endtime>ICU entry and storetime<=H24 does not prove the endtime itself was known by H24.',
        ],
    }
    manifest_path = STAGE / 'eligibility_manifest.json'
    atomic_json(manifest_path, manifest)
    registry_path = STAGE / 'registry_fragment.json'
    atomic_json(registry_path, proposed_registry)

    expected_stage_tables = dict(frames)
    stage_engine_validation = verify_with_engine(STAGE, context, expected_stage_tables, sequences)
    stage_registry_hash = sha256_file(registry_path)
    completion = {
        'status': 'complete_testonly_scope_record',
        'gate_version': OUT.name,
        'scope_sha256': context['scope_sha256'],
        'manifest_sha256': sha256_file(manifest_path),
        'registry_fragment_sha256': stage_registry_hash,
        'original_source_table_sha256': context['base_table_sha256'],
        'engine_validation_stage': stage_engine_validation,
        'completed_unix': time.time(),
    }
    atomic_json(STAGE / 'RUN_COMPLETE.json', completion)

    # Atomic same-filesystem publication. The original source gate and all
    # candidate dataset artifacts remain byte-for-byte untouched.
    os.replace(STAGE, OUT)
    published_validation = verify_with_engine(OUT, context, expected_stage_tables, sequences)
    completion['engine_validation_published'] = published_validation
    completion['published_manifest_sha256'] = sha256_file(OUT / 'eligibility_manifest.json')
    completion['published_registry_fragment_sha256'] = sha256_file(OUT / 'registry_fragment.json')
    completion['published_run_unix'] = time.time()
    atomic_json(OUT / 'RUN_COMPLETE.json', completion)
    print(json.dumps({
        'stage': 'AVAILABILITY_TESTONLY_AUTHORIZATION_COMPLETE',
        'output': str(OUT),
        'scope_sha256': context['scope_sha256'],
        'manifest_sha256': sha256_file(OUT / 'eligibility_manifest.json'),
        'registry_fragment_sha256': sha256_file(OUT / 'registry_fragment.json'),
        'run_complete_sha256': sha256_file(OUT / 'RUN_COMPLETE.json'),
        'source_table_sha256': context['base_table_sha256'],
        'engine_validation': published_validation,
    }, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
