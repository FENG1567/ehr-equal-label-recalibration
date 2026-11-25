"""Recompute ICI as 10-bin equal-count mean|obs-pred| (LOWESS is unstable with a spike at p~0)."""
import glob, os
import numpy as np, pandas as pd
RES="/data/admin/fengshuo/ehr_fm_20261102/results/tables"
def bici(y,p,bins=10):
    q=np.quantile(p,np.linspace(0,1,bins+1)); q[-1]+=1e-9
    idx=np.clip(np.digitize(p,q[1:-1]),0,bins-1); e=[]
    for k in range(bins):
        m=idx==k
        if m.sum()>=20: e.append(abs(y[m].mean()-p[m].mean()))
    return float(np.mean(e)) if e else np.nan
for f in sorted(glob.glob(f"{RES}/preds_*.npz")):
    tag=os.path.basename(f)[6:-4]; z=np.load(f); pth=f"{RES}/probe_{tag}.csv"
    if not os.path.exists(pth): continue
    df=pd.read_csv(pth); df["ici_lowess"]=df["ici"]
    for k in [x[:-2] for x in z.files if x.endswith("_y")]:
        y=z[f"{k}_y"]; p=z[f"{k}_p"]; ok=np.isfinite(y)&np.isfinite(p)
        if ok.sum()<100: continue
        m=df.db==k
        if m.any(): df.loc[m,"ici"]=bici(y[ok],p[ok])
    df.to_csv(pth,index=False); print(tag, df[["db","ici","ici_lowess"]].round(3).to_dict("records"))
