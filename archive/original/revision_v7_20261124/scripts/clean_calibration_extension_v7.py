"""Independent equal-label calibration using the auditable V5 clean encoders.

The conventional comparator is a lossy summary of exactly the same quantized
tokens. This is a retrospective sensitivity, not a new clinical deployment.
"""
import gc, hashlib, importlib.util, json, os, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore',message=".*penalty.*deprecated.*",category=FutureWarning)

ROOT=Path('/data/admin/fengshuo/ehr_fm_20261102')
R7=ROOT/'revision_v7_20261124'
RUN=ROOT/'revision_v5_20261123/results/clean_pretraining/run_20261123_192508_879635_unicode_20261123T132809Z'
SEEDS=[20261123,20261124,20261125];BUDGETS=[500,2000];REPEATS=200;NBOOT=500
OUT=R7/'clean_calibration_extension_v7';PRIVATE=OUT/'private'
PROTOCOL=R7/'clean_calibration_protocol_v7.md'

def log(s):print(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),s,flush=True)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write_json(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def losses(y,p):
 p=np.asarray(p,dtype=float);assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
 q=np.clip(p,1e-6,1-1e-6)
 return np.stack(((p-y)**2,-(y*np.log(q)+(1-y)*np.log1p(-q))),axis=1)
def features(tokens):
 V,B,H=(tokens[k] for k in ['V','B','H']);n=len(V);X=np.zeros((n,48,6),np.float32)
 for vid in range(1,49):
  mask=V==vid;c=mask.sum(1);nz=c>0;den=np.maximum(c,1)
  total=(B*mask).sum(1);mean=total/den
  X[:,vid-1,0]=mean
  X[nz,vid-1,1]=np.where(mask,B,100).min(1)[nz]
  X[:,vid-1,2]=np.where(mask,B,0).max(1)
  X[:,vid-1,3]=np.sqrt(np.maximum((B.astype(np.float32)**2*mask).sum(1)/den-mean**2,0))
  X[:,vid-1,4]=c;X[:,vid-1,5]=(H*mask).sum(1)/den
 return X.reshape(n,-1)
def tokens_for(db,prep):
 p=Path(prep['datasets'][db]['sequence_path']);assert sha(p)==prep['datasets'][db]['sequence_sha256']
 with np.load(p,allow_pickle=False) as z:d={k:np.asarray(z[k]) for k in z.files}
 assert len(np.unique(d['stay_id']))==len(d['stay_id'])
 return d
def read_predictions(db,seed,evaluation,tok):
 name=f'{db}_seed_{seed}_numeric_real.npz';p=RUN/'private/evaluation/predictions'/name
 assert sha(p)==evaluation['private_predictions']['files']['predictions/'+name]
 with np.load(p,allow_pickle=False) as z:d={k:np.asarray(z[k]) for k in z.files}
 assert str(d['dataset'].item())==db and int(d['seed'].item())==seed and str(d['arm'].item())=='numeric_real'
 assert np.array_equal(d['stay_id'],tok['stay_id'])
 assert np.array_equal(d['subject'].astype(str),tok['subject'].astype(str))
 assert np.array_equal(d['split'],tok['split'])
 assert set(np.unique(d['y'])).issubset({0,1})
 return d
def patient_mean_ci(differences,inverse,counts,plans):
 sums=np.bincount(inverse,weights=differences,minlength=len(counts))
 vals=np.array([sums[ix].sum()/counts[ix].sum() for ix in plans])
 return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]

def main():
 assert sha(PROTOCOL)=='c54010cd5fe6c5c5c07272842a36f4abe42263407e91ea11e6ec0106a2404b4c'
 assert sha(RUN/'evaluation_results.json')=='a35b65fefac6af2a47c3e5e7086e9402f8b2012f610a60d46706ff40eca25c4a'
 assert sha(R7/'scripts/temporal_availability_v7.py')=='966629123d4d5269f1e28061deb4e98826d2a806bbee9b69c149141e9b3cf662'
 OUT.mkdir(mode=0o700,exist_ok=False);PRIVATE.mkdir(mode=0o700)
 evaluation=json.loads((RUN/'evaluation_results.json').read_text())
 prep=json.loads((RUN/'preparation_manifest.json').read_text())
 assert evaluation['status']=='complete'
 assert sha(RUN/'preparation_manifest.json')==evaluation['preparation_manifest_sha256']
 assert sha(RUN/'private/training/training_manifest.json')==evaluation['training_manifest_sha256']
 assert sha(RUN/'private/evaluation/source_readouts.json')==evaluation['source_readout_manifest_sha256']
 members=Path(prep['source']['pretraining_member_path'])
 assert sha(members)==prep['source']['pretraining_member_sha256']
 assert int(prep['source']['pretraining_member_n'])==20000
 bins=Path(prep['binning']['path']);assert sha(bins)==prep['binning']['sha256']
 spec=importlib.util.spec_from_file_location('locked_temporal_helpers',R7/'scripts/temporal_availability_v7.py')
 helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
 assert helper.sha256_file(Path(helper.__file__))=='966629123d4d5269f1e28061deb4e98826d2a806bbee9b69c149141e9b3cf662'
 source=tokens_for('mimic',prep)
 gatepath=ROOT/'revision_v4_20261122/temporal/results/eligibility/recovered_reviewed_v4_2/mimic.parquet'
 assert sha(gatepath)=='c2774cd59e4df8706960e4fa58680ea6ec72b4af22adddf82b2389dc5b74749f'
 gate=pq.read_table(gatepath,columns=['stay_id','subject_id','eligible_24h','y_mort']).to_pandas()
 gate=gate[gate.eligible_24h.fillna(False).astype(bool)].copy();gate['stay_id']=gate.stay_id.astype('int64')
 gate=gate.set_index('stay_id').loc[source['stay_id']]
 assert np.array_equal(gate.subject_id.astype(str),source['subject'].astype(str))
 y=gate.y_mort.to_numpy(dtype=np.int8);assert np.isin(y,[0,1]).all()
 split=source['split'];tr=np.flatnonzero(split==0);va=np.flatnonzero(split==1);te=np.flatnonzero(split==2)
 X=features(source);scaler=StandardScaler().fit(X[tr].astype(float))
 xt=scaler.transform(X[tr].astype(float));xv=scaler.transform(X[va].astype(float))
 grid={};best=None;chosen=None
 for C in [.001,.01,.1,1.,10.,100.]:
  model=LogisticRegression(C=C,solver='newton-cholesky',max_iter=200,tol=1e-8)
  with warnings.catch_warnings():
   warnings.simplefilter('error',ConvergenceWarning);model.fit(xt,y[tr])
  score=float(np.mean((model.predict_proba(xv)[:,1]-y[va])**2));grid[str(C)]=score
  if best is None or (score,C)<(best,chosen):best,chosen,baseline=score,C,model
  log(f'SOURCE_BASELINE C={C} validation_brier={score:.8f}')
 np.savez_compressed(PRIVATE/'source_token_lr.npz',mean=scaler.mean_,scale=scaler.scale_,coef=baseline.coef_,intercept=baseline.intercept_,C=np.array([chosen]))
 source_test_p=baseline.predict_proba(scaler.transform(X[te].astype(float)))[:,1]
 metrics=[dict(dataset='mimic_split2',model='token_summary_lr',budget=0,calibration='raw',n_stays=len(te),
               n_patients=len(np.unique(source['subject'][te])),events=int(y[te].sum()),
               brier=float(losses(y[te],source_test_p)[:,0].mean()),logloss=float(losses(y[te],source_test_p)[:,1].mean()))]
 del X,xt,xv,source,gate;gc.collect()
 paired=[];support=[];flows=[];private_hashes={}
 for db in ['eicu','nwicu']:
  tok=tokens_for(db,prep);X=features(tok)
  # Prediction is a sigmoid of standardized source-fitted linear features.
  base_p=expit((X.astype(float)-scaler.mean_)@ (baseline.coef_[0]/scaler.scale_)+baseline.intercept_[0])
  del X;gc.collect()
  preds={'token_summary_lr':base_p};truth=None
  for seed in SEEDS:
   fp=read_predictions(db,seed,evaluation,tok)
   if truth is None:truth=fp['y'].astype(np.int8)
   assert np.array_equal(truth,fp['y'])
   preds[f'fm_seed_{seed}']=fp['p'].astype(float)
  subject=tok['subject'].astype(str);stay=tok['stay_id'].astype(np.int64)
  unique=np.unique(subject);rng=np.random.default_rng(20261124)
  order=rng.permutation(len(unique));adapt_subjects=unique[order[:len(unique)//2]]
  adapt_mask=np.isin(subject,adapt_subjects);assessment=np.flatnonzero(~adapt_mask)
  stay_order=np.argsort(stay,kind='stable');_,first=np.unique(subject[stay_order],return_index=True)
  patient_index=stay_order[first];pool=patient_index[adapt_mask[patient_index]]
  assert not (set(subject[pool])&set(subject[assessment]))
  ytest=truth[assessment];u,inverse=np.unique(subject[assessment],return_inverse=True);counts=np.bincount(inverse)
  boot_rng=np.random.default_rng(20261124)
  bootplans=boot_rng.integers(0,len(u),size=(NBOOT,len(u)),dtype=np.int32)
  raw={k:losses(ytest,p[assessment]) for k,p in preds.items()}
  flow=dict(dataset=db,n_stays=len(stay),n_patients=len(unique),adapt_pool_patients=len(pool),
            assessment_stays=len(assessment),assessment_patients=len(u),assessment_events=int(ytest.sum()))
  flows.append(flow);log('ROLES '+json.dumps(flow))
  for k,l in raw.items():metrics.append(dict(dataset=db,model=k,budget=0,calibration='raw',n_stays=len(assessment),n_patients=len(u),events=int(ytest.sum()),brier=float(l[:,0].mean()),logloss=float(l[:,1].mean())))
  for budget in BUDGETS:
   if len(pool)<budget:
    support.append(dict(dataset=db,model='all',budget=budget,status='infeasible',attempts=0));continue
   rng=np.random.default_rng(20261124+budget)
   plans=np.stack([rng.choice(pool,budget,replace=False) for _ in range(REPEATS)])
   planpath=PRIVATE/f'{db}_plans_{budget}.npz'
   np.savez_compressed(planpath,adaptation_rows=plans,assessment_rows=assessment)
   os.chmod(planpath,0o600);private_hashes[planpath.name]=sha(planpath)
   meanloss={k:np.zeros_like(l) for k,l in raw.items()};fail={k:0 for k in preds};states={k:{} for k in preds}
   events=[]
   for rep,calrows in enumerate(plans):
    yc=truth[calrows];events.append(int(yc.sum()))
    for k,p in preds.items():
     post,info=helper._fit_platt(yc,p[calrows],p[assessment]);state=info['status']
     states[k][state]=states[k].get(state,0)+1;fail[k]+=int(state!='fit')
     meanloss[k]+=losses(ytest,post)/REPEATS
    if (rep+1)%50==0:log(f'UPDATED {db} budget={budget} repeat={rep+1}')
   for k,l in meanloss.items():
    metrics.append(dict(dataset=db,model=k,budget=budget,calibration='platt_mean_loss_across_fits',n_stays=len(assessment),n_patients=len(u),events=int(ytest.sum()),brier=float(l[:,0].mean()),logloss=float(l[:,1].mean())))
    support.append(dict(dataset=db,model=k,budget=budget,status='complete',attempts=REPEATS,failures=fail[k],fallbacks=fail[k],
                        adaptation_events_min=min(events),adaptation_events_p05=float(np.quantile(events,.05)),adaptation_events_p95=float(np.quantile(events,.95)),adaptation_events_max=max(events),fit_status_counts=json.dumps(states[k],sort_keys=True)))
   for seed in SEEDS:
    key=f'fm_seed_{seed}'
    for j,metric in enumerate(['brier','logloss']):
     a=raw[key][:,j]-raw['token_summary_lr'][:,j]
     b=meanloss[key][:,j]-meanloss['token_summary_lr'][:,j]
     aci=patient_mean_ci(a,inverse,counts,bootplans);bci=patient_mean_ci(b,inverse,counts,bootplans)
     cci=patient_mean_ci(b-a,inverse,counts,bootplans)
     paired.append(dict(dataset=db,seed=seed,budget=budget,metric=metric,raw_gap=float(a.mean()),raw_ci_low=aci[0],raw_ci_high=aci[1],updated_gap=float(b.mean()),updated_ci_low=bci[0],updated_ci_high=bci[1],gap_change=float((b-a).mean()),change_ci_low=cci[0],change_ci_high=cci[1],bootstrap_draws=NBOOT))
  del tok,preds,raw,bootplans;gc.collect()
 pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
 pd.DataFrame(paired).to_csv(OUT/'paired_gaps.csv',index=False)
 pd.DataFrame(support).to_csv(OUT/'fit_support.csv',index=False)
 pd.DataFrame(flows).to_csv(OUT/'sample_flow.csv',index=False)
 manifest=dict(status='complete',analysis='auditable_clean_pretraining_equal_label_calibration',seeds=SEEDS,budgets=BUDGETS,repetitions=REPEATS,
  baseline_C=chosen,baseline_validation_brier_grid=grid,baseline_parameter_sha256=sha(PRIVATE/'source_token_lr.npz'),baseline_features='same retained quantized tokens; mean/min/max/std bin,count,mean hour; absent variable zero',
  original_preparation_sha256=sha(RUN/'preparation_manifest.json'),original_evaluation_sha256=sha(RUN/'evaluation_results.json'),
  original_training_sha256=evaluation['training_manifest_sha256'],original_readouts_sha256=evaluation['source_readout_manifest_sha256'],
  pretraining_member_n=20000,pretraining_member_sha256=sha(members),source_only_bins_sha256=sha(bins),
  protocol_sha256=sha(PROTOCOL),script_sha256=sha(Path(__file__)),helper_sha256=sha(Path(helper.__file__)),
  patient_roles_disjoint=True,patient_predictions_exported=False,bootstrap='500 shared assessment-patient cluster draws; stay weighted; conditional on frozen models and 200 realized fits',
  loss_estimand='mean per-stay loss across fits, not loss from averaged probabilities',external_availability_verified=False,prospective_benefit_established=False,
  private_plan_hashes=private_hashes,source_data_hashes={p.name:sha(p) for p in OUT.glob('*.csv')})
 write_json(OUT/'manifest.json',manifest)
 (OUT/'README.md').write_text('Source-only clean pretraining and equal-label calibration\n\nAll three encoder seeds are retained. The conventional predictor summarizes the exact same quantized tokens, so its inputs differ from the raw-value summaries in the historical main comparison. Both use source-only preprocessing and source-validation model selection. Every update shares adaptation patients across predictors; assessment patients are disjoint. All attempted fits, fallback predictions and absolute proper scores are retained. Paired intervals condition on the frozen models and realized updates. This retrospective measurement-time sensitivity does not establish bedside availability or prospective clinical benefit.\n')
 log('CLEAN_CALIBRATION_COMPLETE')

def self_test():
 toy={'V':np.array([[1,1,2,0]],dtype=np.int16),
      'B':np.array([[2,4,10,0]],dtype=np.int16),
      'H':np.array([[0,2,3,0]],dtype=np.int16)}
 x=features(toy).reshape(1,48,6)
 assert np.allclose(x[0,0],[3,2,4,1,2,1])
 assert np.allclose(x[0,1],[10,10,10,0,1,3])
 assert np.all(x[0,2:]==0)
 l=losses(np.array([0,1]),np.array([.1,.9]))
 assert np.allclose(l[:,0],.01) and np.allclose(l[:,1],-np.log(.9))
 ci=patient_mean_ci(np.array([.2,.2]),np.array([0,0]),np.array([2]),np.array([[0],[0]]))
 assert np.allclose(ci,[.2,.2])
 print('CLEAN_CALIBRATION_HELPERS_PASSED',flush=True)

if __name__=='__main__':
 if '--self-test' in sys.argv:self_test()
 else:main()
