"""Conditional reconstruction from current frozen files, never historical certification."""
import hashlib, json, os
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT=Path('/data/admin/fengshuo/ehr_fm_20261102')
R7=ROOT/'revision_v7_20261124'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    out=R7/'historical_reconstruction';out.mkdir(mode=0o700)
    private=out/'private';private.mkdir(mode=0o700)
    auditpath=ROOT/'revision_v4_20261122/admin/checkpoint_metadata_audit.json'
    audit=json.loads(auditpath.read_text())
    meta=next(x for x in audit['records'] if x['tag']=='main20k')
    weights=ROOT/'models/main20k.pt'
    assert sha(weights)==meta['file_sha256'],'checkpoint changed since metadata audit'
    code=ROOT/'code/fm.py';text=code.read_text()
    for snippet in ['idx = np.where((ds.split == 0) & (ds.L > 4))[0]',
                    'idx = np.random.default_rng(seed).choice(idx, n, replace=False)']:
        assert snippet in text,'conditional sampling implementation changed'
    seq=ROOT/'data/processed/seq_mimic.npz';splitpath=ROOT/'data/processed/split_mimic.npy'
    cohortpath=ROOT/'data/interim/cohort_mimic.parquet'
    hashes={str(p.relative_to(ROOT)):sha(p) for p in [seq,splitpath,cohortpath,code,weights,auditpath]}
    # No outcome columns are opened.
    cohort=pq.read_table(cohortpath,columns=['stay_id','subject_id']).to_pandas()
    split=np.load(splitpath,allow_pickle=False)
    with np.load(seq,allow_pickle=False) as z:
        stays=z['stay_id'].astype(np.int64);length=z['L'].astype(np.int64)
    assert len(split)==len(cohort)==len(stays)
    assert np.array_equal(stays,cohort.stay_id.to_numpy(dtype=np.int64))
    assert np.isin(split,[0,1,2]).all()
    role_patients=[set(cohort.loc[split==i,'subject_id'].astype(str)) for i in [0,1,2]]
    disjoint=all(not role_patients[i]&role_patients[j] for i in range(3) for j in range(i+1,3))
    assert disjoint,'current source roles share patients'
    seed=int(meta['stored_seed']);n=int(meta['stored_n'])
    eligible=np.flatnonzero((split==0)&(length>4))
    assert n<=len(eligible)
    members=np.random.default_rng(seed).choice(eligible,n,replace=False) if n<len(eligible) else eligible
    m=cohort.iloc[members].copy();m.insert(0,'sequence_row',members)
    memberpath=private/'candidate_members.csv';m.to_csv(memberpath,index=False)
    os.chmod(memberpath,0o600)
    pts=set(m.subject_id.astype(str))
    report={
      'status':'conditional_reconstruction_complete_historical_membership_unverified',
      'model':'main20k','stored_seed':seed,'stored_member_n':n,
      'current_split0_length_gt4_candidates':int(len(eligible)),
      'reconstructed_candidate_stays':int(len(m)), 'reconstructed_candidate_patients':len(pts),
      'candidate_patient_overlap_source_split1':len(pts&role_patients[1]),
      'candidate_patient_overlap_source_split2':len(pts&role_patients[2]),
      'current_patient_roles_disjoint':disjoint,
      'current_input_hashes':hashes,
      'private_candidate_members_sha256':sha(memberpath),
      'labels_read':False,
      'historical_input_hash_chain_available':False,
      'historical_membership_verified':False,
      'interpretation':'The candidate roster reproduces the stored seed and sample count under the preserved current sampling implementation and current frozen sequence/split files. No training-time member roster or input hashes were stored. Therefore neither this roster nor its zero readout-holdout overlap authenticates actual historical encoder membership.',
      'not_extended_to_multi_database_checkpoints':'Their multiple data sources and exposure remain separate historical comparisons.'}
    (out/'aggregate.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
