"""Aggregate-only clinical summary of currently eligible v4.2 stays.

Reads only the reviewed eligibility gates, frozen sequence identity/length NPZs,
and the MIMIC split vector. No patient-level rows are written.
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
GATE_DIR = R4 / 'temporal/results/eligibility/recovered_reviewed_v4_2'

COHORTS = {
    'mimic': {
        'gate': GATE_DIR / 'mimic.parquet',
        'sequence': ROOT / 'data/processed/seq_mimic.npz',
        'gate_sha256': 'c2774cd59e4df8706960e4fa58680ea6ec72b4af22adddf82b2389dc5b74749f',
        'sequence_sha256': '9bd20bf1394a34ba112a4de9a8d7e8971821bef90558a9987be4b9da5ae06824',
        'eligible_stays': 73936, 'eligible_patients': 53964,
        'db_aliases': {'mimic', 'mimic_iv', 'mimic-iv'}, 'age_topcode': 91,
    },
    'eicu': {
        'gate': GATE_DIR / 'eicu_semantic_v4_2.parquet',
        'sequence': R4 / 'datasets/eicu_semantic_v4_2/seq_eicu_semantic_v4_2.npz',
        'gate_sha256': 'fb14b4a31c924b123923df0ed76d1c3c87c8495b230fa5c8584b67a974b180a8',
        'sequence_sha256': 'e44507da41b045b23b2242dfb2de760b9956a961bf12522c3d55c001633da4e9',
        'eligible_stays': 130544, 'eligible_patients': 102957,
        'db_aliases': {'eicu', 'eicu_crd', 'eicu_semantic_v4_2'}, 'age_topcode': 90,
    },
    'nwicu': {
        'gate': GATE_DIR / 'nwicu_discharge_v4_1.parquet',
        'sequence': R4 / 'datasets/nwicu_discharge_v4_1/seq_nwicu.npz',
        'gate_sha256': '6d3176b44c203fba6f2d14595668e66bd2cedd44558b59590fa55277f8b74eaf',
        'sequence_sha256': 'c654c70a17019bb12cfd57645257786ac303d1acd7fa33ba7db342eb4b336560',
        'eligible_stays': 19972, 'eligible_patients': 17385,
        'db_aliases': {'nwicu', 'nwicu_discharge_v4_1'}, 'age_topcode': 91,
    },
}
SPLIT_PATH = ROOT / 'data/processed/split_mimic.npy'
FROZEN_MIMIC_SPLIT_SHA256 = 'ed5bd94933d22daaa4cd5fdff82ce48b36b72ea50d2db7e4642906036c043c1a'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def key_value(value, label: str) -> str:
    if pd.isna(value):
        raise ValueError(f'{label}: null identity')
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode('utf-8', errors='strict')
    if isinstance(value, (int, np.integer)):
        result = str(int(value))
    elif isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not float(value).is_integer():
            raise ValueError(f'{label}: non-integral numeric identity')
        result = str(int(value))
    else:
        result = str(value).strip()
        if not result or result.lower() in {'nan', 'none', 'null'}:
            raise ValueError(f'{label}: empty identity')
        if re.fullmatch(r'-?\d+\.0+', result):
            result = result.split('.', 1)[0]
    return result


def cgroup_evidence() -> dict:
    result = {'proc_self_cgroup': None, 'cgroup_v2_directory': None,
              'memory_max': None, 'cpu_max': None, 'memory_swap_max': None}
    try:
        line = next(x for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
        cgroup = Path('/sys/fs/cgroup') / line.split('::', 1)[1].lstrip('/')
        result['proc_self_cgroup'] = line
        result['cgroup_v2_directory'] = str(cgroup)
        for key, filename in [('memory_max', 'memory.max'), ('cpu_max', 'cpu.max'),
                              ('memory_swap_max', 'memory.swap.max')]:
            path = cgroup / filename
            result[key] = path.read_text().strip() if path.is_file() else None
    except (OSError, StopIteration):
        pass
    return result


def read_sequence(path: Path, database: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {'stay_id', 'L'}
        if not required.issubset(archive.files):
            raise ValueError(f'{database}: NPZ missing {sorted(required - set(archive.files))}')
        subject_key = 'subject' if 'subject' in archive.files else 'subject_id'
        if subject_key not in archive.files:
            raise ValueError(f'{database}: NPZ missing subject/subject_id')
        stays = np.asarray(archive['stay_id'])
        subjects = np.asarray(archive[subject_key])
        lengths = np.asarray(archive['L'])
    if stays.ndim != 1 or subjects.ndim != 1 or lengths.ndim != 1:
        raise ValueError(f'{database}: stay_id, subject, and L must be vectors')
    if not (len(stays) == len(subjects) == len(lengths)):
        raise ValueError(f'{database}: NPZ identity and L lengths differ')
    lens_numeric = pd.to_numeric(pd.Series(lengths), errors='raise').to_numpy(dtype=float)
    if not np.isfinite(lens_numeric).all() or (lens_numeric < 0).any() or not np.equal(lens_numeric, np.floor(lens_numeric)).all():
        raise ValueError(f'{database}: L must contain nonnegative integer lengths')
    stay_keys = [key_value(x, f'{database} NPZ stay_id') for x in stays]
    subject_keys = [key_value(x, f'{database} NPZ subject') for x in subjects]
    return np.asarray(stay_keys, dtype=object), np.asarray(subject_keys, dtype=object), lens_numeric.astype(np.int64)


def selected_series(frame: pd.DataFrame, names: tuple[str, ...], label: str) -> pd.Series:
    found = [name for name in names if name in frame.columns]
    if not found:
        raise ValueError(f'{label}: missing any supported field {names}')
    if len(found) > 1:
        # Multiple aliases may coexist; require them to represent the same values.
        first = frame[found[0]]
        for other in found[1:]:
            if not first.equals(frame[other]):
                raise ValueError(f'{label}: conflicting aliases {found}')
    return frame[found[0]]


def summarize(database: str, spec: dict, input_hashes: dict[str, str]) -> tuple[dict, dict]:
    gate_path, seq_path = spec['gate'], spec['sequence']
    if input_hashes[str(gate_path)] != spec['gate_sha256']:
        raise ValueError(f'{database}: gate SHA-256 mismatch')
    if input_hashes[str(seq_path)] != spec['sequence_sha256']:
        raise ValueError(f'{database}: sequence NPZ SHA-256 mismatch')
    gate = pd.read_parquet(gate_path)
    db = selected_series(gate, ('db',), f'{database} gate database key').astype(str).str.strip().str.lower()
    if set(db.unique()) - spec['db_aliases']:
        raise ValueError(f'{database}: gate contains unexpected db values {sorted(set(db.unique()) - spec["db_aliases"])}')
    gate_stay = selected_series(gate, ('stay_id',), f'{database} gate stay_id')
    gate_subject = selected_series(gate, ('subject', 'subject_id'), f'{database} gate subject')
    eligible = selected_series(gate, ('eligible_24h',), f'{database} gate eligible_24h')
    if eligible.isna().any() or not eligible.isin([True, False]).all():
        raise ValueError(f'{database}: eligible_24h must be a complete boolean field')
    eligible = eligible.to_numpy(dtype=bool)
    y = selected_series(gate, ('y_mort', 'y'), f'{database} gate outcome')
    ages = selected_series(gate, ('age',), f'{database} gate age')
    sexes = selected_series(gate, ('sex',), f'{database} gate sex')

    seq_stays, seq_subjects, lengths = read_sequence(seq_path, database)
    gate_keys = [(database,
                  key_value(gate_stay.iloc[i], f'{database} gate stay_id'),
                  key_value(gate_subject.iloc[i], f'{database} gate subject'))
                 for i in range(len(gate))]
    seq_keys = [(database, str(stay), str(subject)) for stay, subject in zip(seq_stays, seq_subjects)]
    if len(set(gate_keys)) != len(gate_keys) or len(set(seq_keys)) != len(seq_keys):
        raise ValueError(f'{database}: db/stay/subject keys are not unique')
    seq_position = {key: i for i, key in enumerate(seq_keys)}
    if set(gate_keys) != set(seq_keys):
        missing = len(set(seq_keys) - set(gate_keys))
        extra = len(set(gate_keys) - set(seq_keys))
        raise ValueError(f'{database}: key coverage mismatch (missing={missing}, extra={extra})')
    aligned_positions = np.fromiter((seq_position[key] for key in gate_keys), dtype=np.int64, count=len(gate_keys))
    eligible_n = int(eligible.sum())
    patients = gate_subject.iloc[np.flatnonzero(eligible)].map(lambda x: key_value(x, f'{database} eligible subject'))
    patient_counts = patients.value_counts()
    n_patients = int(patient_counts.size)
    if eligible_n != spec['eligible_stays'] or n_patients != spec['eligible_patients']:
        raise ValueError(f'{database}: eligible stays/patients {(eligible_n, n_patients)} != '
                         f'contract {(spec["eligible_stays"], spec["eligible_patients"])}')

    selected_y = pd.to_numeric(y.iloc[np.flatnonzero(eligible)], errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(selected_y).all() or not np.isin(selected_y, [0.0, 1.0]).all():
        raise ValueError(f'{database}: selected y values must be complete binary 0/1')
    events = int(np.count_nonzero(selected_y == 1.0))

    age_raw = ages.iloc[np.flatnonzero(eligible)]
    age_num = pd.to_numeric(age_raw, errors='coerce')
    if (age_raw.notna() & age_num.isna()).any():
        raise ValueError(f'{database}: nonnumeric age value')
    age_values = age_num.to_numpy(dtype=float)
    age_present = np.isfinite(age_values)
    if np.isinf(age_values).any():
        raise ValueError(f'{database}: nonfinite age value')
    observed_ages = age_values[age_present]
    if (observed_ages < 0).any():
        raise ValueError(f'{database}: negative age value')
    age_q25, age_median, age_q75 = (np.percentile(observed_ages, [25, 50, 75]).tolist()
                                    if len(observed_ages) else (None, None, None))

    sex_raw = sexes.iloc[np.flatnonzero(eligible)]
    sex_num = pd.to_numeric(sex_raw, errors='coerce')
    if (sex_raw.notna() & sex_num.isna()).any():
        raise ValueError(f'{database}: nonnumeric sex code')
    sex_values = sex_num.to_numpy(dtype=float)
    if (sex_raw.notna().to_numpy() & ~np.isfinite(sex_values)).any():
        raise ValueError(f'{database}: nonfinite sex code')
    sex_present = np.isfinite(sex_values)
    if not np.isin(sex_values[sex_present], [0.0, 1.0]).all():
        raise ValueError(f'{database}: nonmissing sex values must be 0 female or 1 male')
    sex_male = int(np.count_nonzero(sex_values == 1.0))
    sex_female = int(np.count_nonzero(sex_values == 0.0))
    sex_unknown = int((~sex_present).sum())
    if sex_male + sex_female + sex_unknown != eligible_n:
        raise ValueError(f'{database}: sex counts do not reconcile to eligible stay denominator')

    selected_lengths = lengths[aligned_positions[eligible]]
    token_q25, token_median, token_q75 = np.percentile(selected_lengths, [25, 50, 75]).tolist()
    cap128_n = int(np.count_nonzero(selected_lengths >= 128))
    result = {
        'database': database, 'scope': 'current_eligible_24h',
        'n_stays': eligible_n, 'n_patients': n_patients, 'n_events_y1': events,
        'repeat_patients_n': int((patient_counts > 1).sum()),
        'repeat_stays_after_first_n': int((patient_counts - 1).clip(lower=0).sum()),
        'age_n_missing_stays': int((~age_present).sum()),
        'age_mean_source_units': float(np.mean(observed_ages)) if len(observed_ages) else None,
        'age_median_source_units': age_median, 'age_q25_source_units': age_q25,
        'age_q75_source_units': age_q75, 'age_iqr_source_units': float(age_q75 - age_q25) if len(observed_ages) else None,
        'age_topcode_value': spec['age_topcode'],
        'age_at_topcode_n': int(np.count_nonzero(observed_ages == spec['age_topcode'])),
        'age_interpretation': f'{spec["age_topcode"]} is source-top-coded; do not interpret it as exact age.',
        'sex_male_1_n': sex_male, 'sex_female_0_n': sex_female, 'sex_unknown_n': sex_unknown,
        'sex_denominator_stays': eligible_n,
        'token_total': int(selected_lengths.sum()), 'token_mean': float(np.mean(selected_lengths)),
        'token_median': float(token_median), 'token_q25': float(token_q25), 'token_q75': float(token_q75),
        'token_iqr': float(token_q75 - token_q25), 'token_cap128_n': cap128_n,
        'token_cap128_fraction': float(cap128_n / eligible_n),
        'token_cap128_definition': 'L >= 128 among currently eligible stays',
        'severity_composite': 'not_provided_no_verified_derived_table',
        'source_split2_rows': None, 'source_split2_patients': None,
        'source_split2_description': None,
    }
    evidence = {'n_gate_rows': int(len(gate)), 'n_sequence_rows': int(len(seq_stays)),
                'key_binding': 'strict_db_stay_subject_one_to_one_full_coverage',
                'eligible_stays': eligible_n, 'eligible_patients': n_patients,
                'eligible_event_y1': events, 'selected_y_column': next(c for c in ('y_mort', 'y') if c in gate.columns)}
    return result, {'evidence': evidence, 'aligned_positions': aligned_positions,
                    'eligible_sequence_positions': aligned_positions[eligible]}


def main() -> None:
    input_paths = [p for spec in COHORTS.values() for p in (spec['gate'], spec['sequence'])] + [SPLIT_PATH]
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError('Required inputs missing: ' + ', '.join(missing))
    input_hashes = {str(path): sha256_file(path) for path in input_paths}
    if input_hashes[str(SPLIT_PATH)] != FROZEN_MIMIC_SPLIT_SHA256:
        raise ValueError('Frozen MIMIC split hash differs from the attested source')

    rows, cohort_evidence = [], {}
    mimic_eligible_sequence_positions = None
    for database, spec in COHORTS.items():
        row, evidence = summarize(database, spec, input_hashes)
        rows.append(row)
        cohort_evidence[database] = evidence['evidence']
        if database == 'mimic':
            mimic_eligible_sequence_positions = evidence['eligible_sequence_positions']

    mimic_stays, mimic_subjects, _ = read_sequence(COHORTS['mimic']['sequence'], 'mimic')
    split = np.load(SPLIT_PATH, allow_pickle=False)
    if split.shape != (len(mimic_stays),) or not np.isin(split, [0, 1, 2]).all():
        raise ValueError('MIMIC split vector must bind one 0/1/2 code per NPZ row')
    if mimic_eligible_sequence_positions is None:
        raise ValueError('MIMIC eligible sequence positions were not produced by the validated gate join')
    if len(np.unique(mimic_eligible_sequence_positions)) != len(mimic_eligible_sequence_positions):
        raise ValueError('MIMIC eligible gate keys map to duplicate NPZ rows')
    eligible_in_sequence_order = np.zeros(len(mimic_stays), dtype=bool)
    eligible_in_sequence_order[mimic_eligible_sequence_positions] = True
    split2 = (split == 2) & eligible_in_sequence_order
    split2_rows = int(split2.sum())
    split2_patients = int(pd.Series(mimic_subjects[split2]).nunique())
    if (split2_rows, split2_patients) != (14890, 10848):
        raise ValueError(f'MIMIC source_split2 contract mismatch: {(split2_rows, split2_patients)}')
    if pd.DataFrame({'subject': mimic_subjects, 'split': split}).groupby('subject')['split'].nunique().gt(1).any():
        raise ValueError('MIMIC subject spans multiple split codes')
    rows[0]['source_split2_rows'] = split2_rows
    rows[0]['source_split2_patients'] = split2_patients
    rows[0]['source_split2_description'] = (
        'Eligible MIMIC stays assigned to source split 2 by frozen NPZ row order; not a performance denominator.')

    summary = pd.DataFrame(rows)
    completed = datetime.now(timezone.utc)
    run_id = f'eligible_clinical_{completed.strftime("%Y%m%dT%H%M%SZ")}_{os.getpid()}'
    output_dir = R5 / 'cohorts' / run_id
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / 'clinical_summary.csv'
    summary.to_csv(csv_path, index=False)
    readme_path = output_dir / 'README.md'
    readme_path.write_text(
        '# Current eligible clinical summary\n\n'
        'Aggregate-only summaries for stays with `eligible_24h=true` in the reviewed v4.2 gates. '
        'Gate rows are matched to frozen sequence NPZ rows by a strict one-to-one `(db, stay_id, subject)` key; '
        'gate input order is not assumed. No patient-level rows or raw identifiers are exported.\n\n'
        'Age values retain source top-coding (MIMIC/NWICU 91; eICU 90); top-coded values are not exact ages. '
        'Sex uses source mapping 1=male and 0=female; missing values remain unknown. '
        'Token statistics use NPZ `L` only among current eligible stays; cap-128 counts `L >= 128`. '
        'No verified derived severity composite was provided, so no score is reported; APACHE/SOFA were not read or inferred.\n\n'
        'MIMIC split-2 counts include only currently eligible stays whose frozen NPZ row is assigned to split 2 '
        '(split SHA in manifest). They are not presented as model performance counts.\n',
        encoding='utf-8')
    output_hashes = {csv_path.name: sha256_file(csv_path), readme_path.name: sha256_file(readme_path)}
    manifest = {
        'status': 'complete_aggregate_only_eligible_clinical_summary',
        'run_id': run_id, 'completed_utc': completed.isoformat(),
        'runtime': {'python': os.sys.version, 'numpy': np.__version__, 'pandas': pd.__version__,
                    'actual_cgroup': cgroup_evidence()},
        'input_hashes_sha256': input_hashes,
        'expected_input_hashes_sha256': {str(spec['gate']): spec['gate_sha256'] for spec in COHORTS.values()} | {
            str(spec['sequence']): spec['sequence_sha256'] for spec in COHORTS.values()},
        'cohorts': cohort_evidence,
        'mimic_source_split2': {'rows': split2_rows, 'patients': split2_patients,
                                'split_sha256': input_hashes[str(SPLIT_PATH)],
                                'expected_split_sha256': FROZEN_MIMIC_SPLIT_SHA256,
                                'binding': 'eligible gate keys projected through strict db/stay/subject NPZ alignment, then intersected with split_mimic.npy indexed by seq_mimic.npz row order',
                                'performance_denominator': False},
        'denominator_definitions': {
            'eligible': 'rows where reviewed gate eligible_24h is true',
            'source_split2': 'eligible MIMIC rows where the frozen split vector equals 2; patients are distinct eligible subject IDs',
            'events': 'selected eligible y value equals 1; selected y must be complete binary 0/1',
            'sex': 'stay counts over all eligible stays, with missing retained as unknown',
            'repeat_patients': 'eligible patients with more than one eligible stay',
            'token_cap128': 'eligible rows with NPZ L >= 128 divided by eligible stay count',
            'severity': 'not provided; no validated derived table was available',
        },
        'output_sha256': output_hashes,
        'privacy': 'Aggregate-only; no patient-level rows or raw identifiers in outputs.',
    }
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': manifest['status'], 'output_dir': str(output_dir),
                      'output_sha256': output_hashes}, ensure_ascii=False))


if __name__ == '__main__':
    main()
