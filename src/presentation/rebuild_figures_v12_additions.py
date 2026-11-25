#!/usr/bin/env python3
"""Render r4-numbered site, Data22, and temporal figures from aggregate CSVs only."""
import argparse, hashlib, json, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
import pandas as pd

MM=1/25.4
FM="#3E7392"; LR="#C27642"; GRAY="#777777"
SEED={20261123:"#3E7392",20261124:"#C27642",20261125:"#5E8F89"}
FONT_DIR=os.environ.get("EHR_FIGURE_FONT_DIR")
if FONT_DIR:
 for font_file in (Path(FONT_DIR)/"arial.ttf",Path(FONT_DIR)/"arialbd.ttf"):
  if font_file.is_file():font_manager.fontManager.addfont(str(font_file))
plt.rcParams.update({"font.family":"Arial","font.sans-serif":["Arial","Liberation Sans","DejaVu Sans"],
 "font.weight":"bold","axes.labelweight":"bold","axes.titleweight":"bold",
 "font.size":9,"axes.labelsize":9.2,"axes.titlesize":10.2,"xtick.labelsize":8.4,
 "ytick.labelsize":8.4,"legend.fontsize":8.4,"svg.fonttype":"none","pdf.fonttype":42,
 "ps.fonttype":42,"axes.linewidth":.7,"axes.spines.right":False,"axes.spines.top":False,
 "legend.frameon":False,"figure.facecolor":"white","axes.facecolor":"white"})
def digest(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def clean(ax,grid="x"):
 ax.tick_params(direction="out",length=3.2,width=.75,pad=4)
 if grid:ax.grid(axis=grid,color="#ddd",lw=.45);ax.set_axisbelow(True)
def panel(ax,s):
 t=ax.text(-.09,1.055,s.upper(),transform=ax.transAxes,fontweight="bold",fontsize=18,
           fontfamily="Arial",va="bottom",ha="left",clip_on=False)
 t.set_gid("panel-letter")
def format_text(fig):
 """Apply the requested readable Arial-bold type hierarchy to every text item."""
 for ax in fig.axes:
  ax.title.set_fontfamily("Arial");ax.title.set_fontweight("bold");ax.title.set_fontsize(10.2)
  ax.xaxis.label.set_fontfamily("Arial");ax.xaxis.label.set_fontweight("bold");ax.xaxis.label.set_fontsize(9.2)
  ax.yaxis.label.set_fontfamily("Arial");ax.yaxis.label.set_fontweight("bold");ax.yaxis.label.set_fontsize(9.2)
  for t in (*ax.get_xticklabels(),*ax.get_yticklabels()):
   t.set_fontfamily("Arial");t.set_fontweight("bold");t.set_fontsize(8.4)
  for t in ax.texts:
   t.set_fontfamily("Arial");t.set_fontweight("bold")
   if t.get_gid()!="panel-letter":t.set_fontsize(max(t.get_fontsize(),8.5))
  leg=ax.get_legend()
  if leg:
   for t in leg.get_texts():t.set_fontfamily("Arial");t.set_fontweight("bold");t.set_fontsize(8.4)
   if leg.get_title().get_text():leg.get_title().set_fontfamily("Arial");leg.get_title().set_fontweight("bold");leg.get_title().set_fontsize(8.8)
 for t in fig.texts:
  t.set_fontfamily("Arial");t.set_fontweight("bold");t.set_fontsize(max(t.get_fontsize(),8.5))
def save(fig,out,stem,h):
 w,h=(float(x)/MM for x in fig.get_size_inches())
 format_text(fig)
 out.mkdir(parents=True,exist_ok=True);(out/"main").mkdir(exist_ok=True);(out/"preview150").mkdir(exist_ok=True)
 base=out/"main"/stem
 fig.savefig(base.with_suffix(".png"),dpi=300);fig.savefig(out/"preview150"/(stem+".png"),dpi=300)
 fig.savefig(base.with_suffix(".svg"));fig.savefig(base.with_suffix(".pdf"))
 fig.savefig(base.with_suffix(".tiff"),dpi=600,pil_kwargs={"compression":"tiff_lzw"})
 plt.close(fig)
 return {"figure":stem,"width_mm":w,"height_mm":h,"path":base.relative_to(out).as_posix()}
def in_repo(repo,rel):
 p=repo/rel
 if not p.is_file():raise FileNotFoundError(rel)
 return p
def site_figure(repo,out):
 d=repo/"data/aggregates/Data15/sites"
 s=pd.read_csv(d/"site_adaptation_summary.csv")
 r=pd.read_csv(d/"site_roster.csv")
 a=pd.read_csv(d/"across_site_adaptation_summary.csv")
 r=r[r.selected_top24.astype(str).str.lower().isin(["true","1"])].sort_values("eligible_patient_rank_largest_first").reset_index(drop=True)
 if len(r)!=24:raise ValueError("expected 24 selected sites")
 r["site_label"]=[f"Site{i:02d}" for i in range(1,25)]
 s=s[(s.budget==500)&s.model.isin(["FM","V+P LR"])&s.method.isin(["no_update","platt"])&
     (s.weighting=="stay_weighted")&(s.feasible_site.astype(str).str.lower()=="true")&
     (s.status=="complete_200_repeats")].merge(r[["hospitalid","site_label"]],on="hospitalid",validate="many_to_one")
 if len(s)!=96:raise ValueError(f"expected 96 hospital summaries, got {len(s)}")
 fig=plt.figure(figsize=(183*MM,184*MM));gs=fig.add_gridspec(2,2,height_ratios=[3.25,1.15],hspace=.42,wspace=.30)
 top=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1])];bot=[fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])]
 for j,(ax,metric,title) in enumerate(zip(top,["brier_mean","logloss_mean"],["Brier score","Log loss"])):
  offsets={("FM","no_update"):-.17,("FM","platt"):-.17,("V+P LR","no_update"):.17,("V+P LR","platt"):.17}
  for model,color,marker in [("FM",FM,"o"),("V+P LR",LR,"s")]:
   raw=s[(s.model==model)&(s.method=='no_update')].sort_values('rank_by_eligible_patients')
   updated=s[(s.model==model)&(s.method=='platt')].sort_values('rank_by_eligible_patients')
   if not np.array_equal(raw.hospitalid.to_numpy(),updated.hospitalid.to_numpy()):raise ValueError('Site pairing differs')
   yy=np.arange(24)+offsets[(model,'no_update')]
   for k in range(24):ax.plot([raw.iloc[k][metric],updated.iloc[k][metric]],[yy[k],yy[k]],color=color,lw=.65,alpha=.8,zorder=2)
   for method in ["no_update","platt"]:
    q=s[(s.model==model)&(s.method==method)].sort_values("rank_by_eligible_patients")
    y=np.arange(24)+offsets[(model,method)]
    ax.scatter(q[metric].to_numpy(float),y,s=15,marker=marker,
      facecolors=("white" if method=="no_update" else color),edgecolors=color,lw=.9,zorder=3)
  ax.set_yticks(range(24));ax.set_yticklabels([f"Site{i:02d}" for i in range(1,25)] if j==0 else [])
  ax.set_ylim(23.6,-.6);ax.set_xlabel(title);ax.set_title("Mean across 200 repeats",pad=7)
  ax.set_ylabel("Preselected hospital" if j==0 else "");clean(ax);panel(ax,"ab"[j])
 handles=[Line2D([],[],marker=m,color=c,ls="none",markerfacecolor=f,ms=4,label=l)
  for m,c,f,l in [("o",FM,"white","FM · raw"),("o",FM,FM,"FM · 500-label Platt"),
   ("s",LR,"white","V+P LR · raw"),("s",LR,LR,"V+P LR · 500-label Platt")]]
 fig.legend(handles=handles,loc="upper center",ncol=2,bbox_to_anchor=(.56,.99),
            columnspacing=1.5,handletextpad=.55,labelspacing=.55)
 for j,(ax,metric,title) in enumerate(zip(bot,["brier","logloss"],["Brier macro estimate","Log-loss macro estimate"])):
  for yi,model,color,marker in [(0,"FM",FM,"o"),(1,"V+P LR",LR,"s")]:
   for method in ["no_update","platt"]:
    q=a[(a.budget==500)&(a.model==model)&(a.method==method)&(a.metric==metric)&
        (a.aggregation=="macro")&(a.weighting=="stay_weighted")]
    if len(q)!=1:raise ValueError(f"missing unique macro row {model}/{method}/{metric}")
    z=q.iloc[0];x=float(z.estimate);lo=float(z.bootstrap_ci_95_lower);hi=float(z.bootstrap_ci_95_upper)
    yy=yi+(-.12 if method=="no_update" else .12)
    ax.errorbar(x,yy,xerr=[[x-lo],[hi-x]],fmt=marker,color=color,
      markerfacecolor=("white" if method=="no_update" else color),ms=4,capsize=2,lw=.9)
  ax.set_yticks([0,1],["FM","V+P LR"]);ax.set_xlabel(title);ax.set_title("Equal-site mean (95% CI)",pad=6)
  ax.set_ylim(1.35,-.35);ax.xaxis.set_major_locator(MaxNLocator(nbins=4));ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
  clean(ax);panel(ax,"cd"[j])
 fig.text(.53,.055,"Site means: 200 adaptation repeats.\nMacro intervals: shared patient-cluster resampling.",
          ha="center",fontsize=9,color="#333")
 fig.subplots_adjust(left=.19,right=.98,bottom=.17,top=.88)
 rec=save(fig,out,"Figure3",192);rec.update({"previous_display_figure":"Figure4","release_version":"2026.11.25-r7"})
 rec.update({"claim":"Scores vary across the 24 preselected hospitals; equal-site macro estimates and their source intervals are also shown.",
  "source_fields":{"site":"brier_mean/logloss_mean; no_update=raw; platt=500-label Platt","macro":"estimate and bootstrap_ci_95_lower/upper"},
  "site_points":"Per-site repeat mean, no interval bars","macro_ci":"95% global patient-cluster bootstrap; shared across sites",
  "site_labels":"Roster rank by eligible patients mapped to Site01–Site24; hospital IDs not displayed",
  "source_hashes":{f:digest(d/f) for f in ["site_adaptation_summary.csv","site_roster.csv","across_site_adaptation_summary.csv","site_adaptation_repeat_scores.csv"]}})
 (out/"manifests").mkdir(exist_ok=True)
 (out/"manifests"/"Figure3_contract.json").write_text(json.dumps(rec,indent=2)+"\n")
 return rec

def updated_gap_figure(repo,out):
 d=pd.read_csv(repo/"data/aggregates/Data22/paired_gaps.csv")
 fig,axs=plt.subplots(2,2,figsize=(183*MM,151*MM))
 cells=[("eicu","eICU","brier","Brier score"),("eicu","eICU","logloss","Log loss"),
        ("nwicu","NWICU","brier","Brier score"),("nwicu","NWICU","logloss","Log loss")]
 for i,(ax,(ds,dl,metric,ml)) in enumerate(zip(axs.flat,cells)):
  q=d[(d.dataset.str.lower()==ds)&(d.metric==metric)&d.seed.isin(SEED)&d.budget.isin([500,2000])]
  if len(q)!=6:raise ValueError(f"need six paired gap rows for {ds}/{metric}")
  ax.axhline(0,color="#888",ls="--",lw=.7)
  for seed_index,(seed,color) in enumerate(SEED.items()):
   z=q[q.seed==seed].sort_values("budget");x=np.asarray([0.,1.])+(seed_index-1)*.08;y=z.updated_gap.to_numpy(float)
   lo=z.updated_ci_low.to_numpy(float);hi=z.updated_ci_high.to_numpy(float)
   ax.plot(x,y,"o-",color=color,ms=3.7,lw=1,label=str(seed))
   ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt="none",ecolor=color,capsize=2,lw=.9)
  ax.set_xlim(-.24,1.24);ax.set_xticks([0,1],["500","2000"]);ax.minorticks_off()
  ax.set(xlabel="Recalibration patients",ylabel="FM − token-summary LR\nloss",title=f"{dl} · {ml}")
  clean(ax,"y");panel(ax,"abcd"[i])
 h,l=axs.flat[0].get_legend_handles_labels();fig.legend(h,l,ncol=3,loc="upper center",bbox_to_anchor=(.53,.985),
  title="Training seed",title_fontsize=8.8,columnspacing=1.6,handletextpad=.55,labelspacing=.55)
 fig.subplots_adjust(left=.21,right=.98,bottom=.13,top=.86,hspace=.50,wspace=.46)
 rec=save(fig,out,"Figure4",168);rec.update({"previous_display_figure":"Figure5","release_version":"2026.11.25-r7"});rel="data/aggregates/Data22/paired_gaps.csv";src=in_repo(repo,rel)
 rec.update({"claim":"Updated FM-minus-quantized-token-summary-LR loss gaps vary by dataset, budget, and seed.",
  "source":rel,"source_sha256":digest(src),
  "field_selector":"dataset in {eicu,nwicu}; 3 training seeds; budget in {500,2000}; metric in {brier,logloss}",
  "fields":["updated_gap","updated_ci_low","updated_ci_high","bootstrap_draws"],
  "display_transform":"Budgets 500 and 2000 are categorical positions; seed markers offset by -0.08,0,+0.08 for readability, without changing any source estimate or budget.",
  "uncertainty":"Exact paired intervals from aggregate input; no new bootstrap; historical original-input LR not shown."})
 (out/"manifests").mkdir(exist_ok=True);(out/"manifests"/"Figure4_contract.json").write_text(json.dumps(rec,indent=2)+"\n")
 return rec
def temporal_figure(repo,out):
 s=pd.read_csv(repo/"data/aggregates/Data20/paired_model_bootstrap_v7.csv")
 d=pd.read_csv(repo/"data/aggregates/Data20/decision_curve_v7.csv")
 seeds=[20261123,20261124,20261125]
 q=s[(s.dataset=="late_test")&(s.contrast=="numeric_pretrained_minus_value_pattern")&
     s.seed.isin(seeds)&s.calibration.isin(["raw","late500_platt_updated"])&
     s.metric.isin(["brier","logloss"])]
 if len(q)!=12:raise ValueError(f"expected 12 temporal score rows, got {len(q)}")
 fig=plt.figure(figsize=(183*MM,190*MM));gs=fig.add_gridspec(2,2,height_ratios=[.72,1.20],hspace=.45,wspace=.42)
 top=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1])]
 for j,(ax,metric,title) in enumerate(zip(top,["brier","logloss"],["Brier score","Log loss"])):
  for i,seed in enumerate(seeds):
   for cal,offset,marker in [("raw",-.13,"o"),("late500_platt_updated",.13,"s")]:
    z=q[(q.seed==seed)&(q.metric==metric)&(q.calibration==cal)]
    if len(z)!=1:raise ValueError(f"missing temporal score {seed}/{metric}/{cal}")
    v=z.iloc[0];x=float(v.difference);lo=float(v.ci_low_2_5);hi=float(v.ci_high_97_5);color=GRAY if cal=='raw' else FM
    ax.errorbar(x,i+offset,xerr=[[x-lo],[hi-x]],fmt=marker,color=color,
       markerfacecolor=("white" if cal=="raw" else color),ms=4,capsize=2,lw=.9)
  ax.axvline(0,color="#888",ls="--",lw=.7);ax.set_yticks(range(3),[str(x) for x in seeds]);ax.invert_yaxis()
  ax.set(xlabel=f"FM − summary LR · {title}",ylabel="Training seed" if j==0 else "",title=f"Later-period · {title}")
  clean(ax);panel(ax,"ab"[j])
 h=[Line2D([],[],marker="o",color=GRAY,markerfacecolor="white",ls="none",ms=4,label="Raw"),
    Line2D([],[],marker="s",color=FM,markerfacecolor=FM,ls="none",ms=4,label="500-label Platt")]
 fig.legend(handles=h,ncol=2,loc="upper center",bbox_to_anchor=(.53,.985),title="Calibration",
            title_fontsize=8.8,columnspacing=1.6,handletextpad=.55,labelspacing=.55)
 late=d[(d.dataset=="late_test")&(d.seed==20261123)].copy()
 models=late[(late.strategy=="model")&late.arm.isin(["numeric_pretrained","value_pattern"])&
             late.calibration.isin(["raw","late500_platt_updated"])]
 if models.empty:raise ValueError("Data20 late-test DCA rows missing")
 ax=fig.add_subplot(gs[1,0]);df=fig.add_subplot(gs[1,1])
 colors={"numeric_pretrained":FM,"value_pattern":LR}; names={"numeric_pretrained":"FM","value_pattern":"Summary LR"}
 styles={"raw":"--","late500_platt_updated":"-"}
 for arm in ["numeric_pretrained","value_pattern"]:
  for cal in ["raw","late500_platt_updated"]:
   z=models[(models.arm==arm)&(models.calibration==cal)].sort_values("threshold")
   if len(z)!=7:raise ValueError(f"expected full 7-point DCA {arm}/{cal}")
   x=z.threshold.to_numpy(float);y=z.net_benefit_per_100.to_numpy(float)
   ax.plot(x,y,color=colors[arm],ls=styles[cal],lw=1.05,label=f"{names[arm]} · {'raw' if cal=='raw' else '500-label Platt'}")
   if cal=="late500_platt_updated":ax.fill_between(x,z.ci_low_per_100,z.ci_high_per_100,color=colors[arm],alpha=.1,lw=0)
 for strategy,name,style in [("treat_all","Treat all",":"),("treat_none","Treat none","-.")]:
  z=late[(late.arm=="numeric_pretrained")&(late.calibration=="raw")&(late.strategy==strategy)].sort_values("threshold")
  if len(z)!=7:raise ValueError(f"incomplete {strategy} DCA range")
  ax.plot(z.threshold,z.net_benefit_per_100,color=GRAY if strategy=="treat_all" else "#222",ls=style,lw=.9,label=name)
 ax.set_xlim(float(late.threshold.min()),float(late.threshold.max()))
 all_y=np.r_[models.net_benefit_per_100.to_numpy(float),
    late[(late.arm=="numeric_pretrained")&(late.calibration=="raw")&late.strategy.isin(["treat_all","treat_none"])].net_benefit_per_100.to_numpy(float)]
 lo=float(np.min(all_y));hi=float(np.max(all_y));pad=max((hi-lo)*.08,.1);ax.set_ylim(lo-pad,hi+pad)
 ax.set(xlabel="Threshold probability",ylabel="Net benefit per 100 patients",title="Absolute curves · seed 20261123")
 abs_handles,abs_labels=ax.get_legend_handles_labels()
 clean(ax);panel(ax,"c")
 for cal,color,label0 in [("raw",GRAY,"Raw"),("late500_platt_updated",FM,"500-label Platt")]:
  a=models[(models.arm=="numeric_pretrained")&(models.calibration==cal)].sort_values("threshold")
  b=models[(models.arm=="value_pattern")&(models.calibration==cal)].sort_values("threshold")
  if not np.array_equal(a.threshold.to_numpy(),b.threshold.to_numpy()):raise ValueError("DCA thresholds differ")
  df.plot(a.threshold,a.net_benefit_per_100.to_numpy()-b.net_benefit_per_100.to_numpy(),
          "o-",color=color,markerfacecolor=("white" if cal=="raw" else color),ms=3.3,lw=1,label=label0)
 df.axhline(0,color="#888",ls="--",lw=.7);df.set_xlim(float(late.threshold.min()),float(late.threshold.max()))
 df.set(xlabel="Threshold probability",ylabel="Net-benefit difference per 100 patients\n(FM − summary LR)",title="Between-model net benefit")
 diff_handles,diff_labels=df.get_legend_handles_labels()
 clean(df);panel(df,"d")
 ax.legend(abs_handles,abs_labels,loc="upper center",bbox_to_anchor=(.5,-.24),ncol=2,
           handlelength=1.7,columnspacing=1.3,handletextpad=.45,labelspacing=.48,borderaxespad=0)
 df.legend(diff_handles,diff_labels,loc="upper center",bbox_to_anchor=(.5,-.24),ncol=2,
           handlelength=1.6,columnspacing=1.0,handletextpad=.45,labelspacing=.5,borderaxespad=0)
 fig.subplots_adjust(left=.20,right=.98,bottom=.20,top=.87,hspace=.45,wspace=.45)
 rec=save(fig,out,"Figure5",220);rec.update({"previous_display_figure":"Figure6","release_version":"2026.11.25-r7"})
 files=["paired_model_bootstrap_v7.csv","decision_curve_v7.csv"]
 rec.update({"claim":"Temporal input sensitivity is shown as retrospective score and decision-curve comparisons.",
  "source_hashes":{f:digest(repo/"data/aggregates/Data20"/f) for f in files},
  "score_selector":"late_test; contrast numeric_pretrained_minus_value_pattern; 3 seeds; raw/late500_platt_updated",
  "DCA_selector":"late_test; seed 20261123; all 7 thresholds; both model arms plus treat-all/treat-none",
  "difference":"FM net-benefit point estimate minus summary-LR point estimate at matched threshold and calibration; no CI",
  "uncertainty":"Source score and absolute Platt DCA intervals retained; no interval added to difference panel.",
  "caveat":"Retrospective system-time input; no prospective benefit inference."})
 (out/"manifests").mkdir(exist_ok=True);(out/"manifests"/"Figure5_contract.json").write_text(json.dumps(rec,indent=2)+"\n")
 return rec

def main():
 p=argparse.ArgumentParser();p.add_argument("--repo",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
 p.add_argument('--only',nargs='*',type=int,choices=[4,5,6])
 a=p.parse_args();repo=a.repo.resolve();out=a.output.resolve()
 targets=a.only or [4,5,6]
 rec=[]
 if a.only and (out/'additions_manifest.json').exists():
  selected_stems={4:'Figure3',5:'Figure4',6:'Figure5'}
  rec=[x for x in json.loads((out/'additions_manifest.json').read_text())['figures'] if x['figure'] not in {selected_stems[n] for n in targets}]
 new=[{4:site_figure,5:updated_gap_figure,6:temporal_figure}[n](repo,out) for n in targets];rec+=new
 (out/"additions_manifest.json").write_text(json.dumps({"status":"rendered_pending_visual_QA","figures":rec},indent=2)+"\n")
 for x in new:print("RENDERED "+x["figure"],flush=True)
 print("ADDITIONS_COMPLETE",flush=True)
if __name__=="__main__":main()
