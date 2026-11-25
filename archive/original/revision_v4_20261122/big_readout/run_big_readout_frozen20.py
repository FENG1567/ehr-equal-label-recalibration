"""Source-only big-model heads from frozen caches; no encoder computation.

The owned engine snapshot changes only three manifest path expressions to the
immutable 20-row manifest. Cache and source-probe directories stay canonical.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

REV = Path('/data/admin/fengshuo/ehr_fm_20261102/revision_v4_20261122')
ENGINE = REV / 'engine'
CODE = ENGINE / 'code'
OWN = REV / 'big_readout'
SCRIPT = OWN / 'v4_engine_big_frozen20.py'
BASE = CODE / 'v4_engine_registry_maskfix_recovered_v4_2_resume_reusefix_a725994d9d5d.py'
SNAPSHOT = ENGINE / 'code_snapshots/embedding_manifest_mimic_eicu_core_plus_big20_20261123.json'
MERGE_AUDIT = ENGINE / 'code_snapshots/embedding_manifest_merge_audit_recovered_v4_2_b2eec1f7aca1.json'
GATE = REV / 'temporal/results/eligibility/status_consistent_v4_1/landmark_v4_1'
RUN_ID = 'big_readout_frozen20_v4_2'
HASHES = {
    SCRIPT: '293ea8607351970699e908259fe630f116ffdbd8b2787145b55d9e58106a90db',
    BASE: 'a725994d9d5dd33d8a2b23ba42afb029c98c5f530658ddda20e76f259a4b9528',
    SNAPSHOT: '3ccc3927e7bf7e6a9d31be1136c2870ad8c9fd51bcbb2240424b2a4ff005bb58',
    MERGE_AUDIT: 'ba792d7c600c134bbaa79deebba123183217b4bf474a5102c7100ac65f09740d',
    GATE / 'eligibility_manifest.json': 'ea2b55bc3cf071d48b019d4ba97b4a3dc53d5d7fb3f6ac49a9e7c26007064cb5',
}
CONDITIONS = (('big50m', 'real'), ('big50m_full', 'real'), ('big50m_full', 'common8_mask'))
EXPECTED_ENCODER = '59c3f4059662fb9e1d81baf20120c8668a761a926f5cab5ff4e654f7c16c1abc'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def preflight():
    records = []
    for path, expected in HASHES.items():
        if sha(path) != expected:
            raise ValueError('Pinned input changed: ' + str(path))
    needle = 'ENGINE / "results/embedding_manifest.json"'
    replacement = 'Path("' + str(SNAPSHOT) + '")'
    base = BASE.read_text()
    if base.count(needle) != 3 or base.replace(needle, replacement) != SCRIPT.read_text():
        raise ValueError('Owned snapshot differs by more than frozen manifest binding')
    sys.path.insert(0, str(CODE))
    ns = runpy.run_path(str(SCRIPT), run_name='big_frozen_definitions')
    if ns['encoder_code_hash']() != EXPECTED_ENCODER:
        raise ValueError('Functional encoder hash changed')
    manifest, entries = ns['load_encoding_manifest']()
    if len(manifest['models']) != 20:
        raise ValueError('Wrong frozen cache inventory')
    audit = json.loads(MERGE_AUDIT.read_text())
    audited_cache = {tuple(row['key']): row for row in audit['cache_files']}
    for db, count, events in (('mimic', 73936, 8143), ('eicu', 130544, 11943)):
        eligible, attestation = ns['load_ready_eligibility'](db, GATE)
        if len(eligible) != count or int(eligible['y_mort'].sum()) != events:
            raise ValueError('Wrong audited cohort counts: ' + db)
        for model, condition in CONDITIONS:
            key = (db, model, condition)
            entry, trusted = entries[key], audited_cache[key]
            if str(entry['cache']) != trusted['path'] or sha(Path(entry['cache'])) != trusted['sha256']:
                raise ValueError('Cache payload differs from frozen merge audit: ' + repr(key))
            if sha(Path('/data/admin/fengshuo/ehr_fm_20261102/models') / (model + '.pt')) != entry['checkpoint_sha256']:
                raise ValueError('Checkpoint differs: ' + model)
            records.append({'key': list(key), 'cache_sha256': trusted['sha256'],
                            'checkpoint_sha256': entry['checkpoint_sha256']})
    result = {'status': 'preflight_pass', 'run_id': RUN_ID, 'cache_checks': records,
              'hash_bindings': {str(p): d for p, d in HASHES.items()},
              'functional_encoder_sha256': EXPECTED_ENCODER,
              'scope': 'source-only supervised heads; target features exposed during historical pretraining; membership not fully verified'}
    return ns, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    ns, proof = preflight()
    if args.preflight_only:
        print(json.dumps(proof, indent=2), flush=True)
        print('BIG_READOUT_PREFLIGHT_PASS', flush=True)
        return
    proof_path = OWN / 'big_readout_input_attestation.json'
    if proof_path.exists():
        raise FileExistsError('Refusing to overwrite attestation: ' + str(proof_path))
    proof_path.write_text(json.dumps(proof, indent=2))
    prediction_args = ns['parse_args'](['--mode', 'predict', '--models', 'big50m', 'big50m_full',
                                        '--cohorts', 'mimic', 'eicu', '--eligibility-dir', str(GATE),
                                        '--run-id', RUN_ID, '--device', 'cpu'])
    status = ns['run_predict'](prediction_args)
    if status != 0:
        raise SystemExit(status)
    matches = list((ENGINE / 'results/prediction_runs').glob(RUN_ID + '-*/prediction_manifest.json'))
    if len(matches) != 1:
        raise ValueError('Missing unique final prediction manifest')
    manifest = json.loads(matches[0].read_text())
    positive = [row for row in manifest['predictions'] if row.get('n_predictions', 0) > 0]
    expected = {(db, tag + '_' + condition) for db in ('mimic', 'eicu') for tag, condition in CONDITIONS}
    if {(row['db'], row['analysis_id']) for row in positive} != expected or len(positive) != 6:
        raise ValueError('Missing big-model evaluation cells')
    proof['status'] = 'predictions_complete'
    proof['prediction_manifest'] = str(matches[0])
    proof['prediction_manifest_sha256'] = sha(matches[0])
    (OWN / 'big_readout_completion.json').write_text(json.dumps(proof, indent=2))
    print('BIG_READOUT_FROZEN20_COMPLETE ' + str(matches[0]), flush=True)


if __name__ == '__main__':
    main()
