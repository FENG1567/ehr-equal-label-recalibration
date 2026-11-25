import pandas as pd
RES="/data/admin/fengshuo/ehr_fm_20261102/results/tables"
cells=[("s05k",5000,1),("main20k",20000,1),("s50k",50000,1),("k2_20k",20000,2),("k3_20k",20000,3)]
rows=[]
for tag,n,k in cells:
    d=pd.read_csv(f"{RES}/probe_{tag}_mort.csv")
    d=d[d.db.isin(["mimic_test","eicu_all","nwicu_all"])].copy()
    d["cell"]=tag; d["n"]=n; d["k"]=k; d["seed"]=20261102
    rows.append(d)
g=pd.concat(rows); g.to_csv(f"{RES}/scaling_grid.csv",index=False)
print(g.pivot_table(index=["cell","n","k"],columns="db",values="auroc").round(3).to_string())
