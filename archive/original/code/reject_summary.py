"""Coverage-error curves and two selective-error definitions from saved predictions."""
import glob, os
import numpy as np, pandas as pd
ROOT="/data/admin/fengshuo/ehr_fm_20261102"; RES=f"{ROOT}/results/tables"
def curve(y,p,thr=0.5,coverages=(0.5,0.6,0.7,0.8,0.9,0.95,1.0)):
    ent=-(np.clip(p,1e-9,1-1e-9)*np.log(np.clip(p,1e-9,1-1e-9))+(1-np.clip(p,1e-9,1-1e-9))*np.log(np.clip(1-p,1e-9,1-1e-9)))
    order=np.argsort(ent); e=((p>=thr).astype(float)!=y).astype(float)
    out=[]
    for c in coverages:
        k=max(int(round(c*len(y))),1); s=order[:k]
        out.append(dict(coverage=c,err=float(e[s].mean()),brier=float(np.mean((p[s]-y[s])**2)),
                        prev=float(y[s].mean())))
    return pd.DataFrame(out)
rows=[]
for f in sorted(glob.glob(f"{RES}/preds_*_*.npz")):
    z=np.load(f); tag=os.path.basename(f)[6:-4]
    for k in [x[:-2] for x in z.files if x.endswith("_y")]:
        y=z[f"{k}_y"]; p=z[f"{k}_p"]; ok=np.isfinite(y)&np.isfinite(p); y,p=y[ok],p[ok]
        if len(y)<100 or len(np.unique(y))<2: continue
        d=curve(y,p); d["tag"]=tag; d["db"]=k; rows.append(d)
        full=d[d.coverage==1.0].iloc[0]; c80=d[d.coverage==0.8].iloc[0]
        print(f"{tag:14s} {k:12s} err {full.err:.3f}->{c80.err:.3f} (RRR {1-c80.err/full.err:+.1%})  "
              f"brier {full.brier:.4f}->{c80.brier:.4f} (BRR {1-c80.brier/full.brier:+.1%})  prev {full.prev:.3f}->{c80.prev:.3f}")
pd.concat(rows).to_csv(f"{RES}/reject_curves.csv",index=False)
