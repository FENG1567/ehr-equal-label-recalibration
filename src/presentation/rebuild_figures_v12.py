#!/usr/bin/env python3
"""Aggregate-only v12 figure renderer; no patient-level reads or analyses."""
import argparse, hashlib, json, math, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

MM=1/25.4
C=["#3E7392","#C27642","#5E8F89","#8173A6","#B18B3B","#777777"]
FONT_DIR=os.environ.get("EHR_FIGURE_FONT_DIR")
if FONT_DIR:
 for font_path in sorted(Path(FONT_DIR).glob("*.ttf")):
  font_manager.fontManager.addfont(str(font_path))
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
 "font.size":9.5,"font.weight":"bold","axes.labelsize":10,"axes.labelweight":"bold",
 "axes.titlesize":11,"axes.titleweight":"bold","xtick.labelsize":9,"ytick.labelsize":9,
 "xtick.major.width":.8,"ytick.major.width":.8,"legend.fontsize":9.2,
 "svg.fonttype":"none","pdf.fonttype":42,"ps.fonttype":42,
 "axes.linewidth":.8,"axes.spines.right":False,"axes.spines.top":False,
 "legend.frameon":False,"figure.facecolor":"white","axes.facecolor":"white"})
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def finite(x):
 try:return x is not None and math.isfinite(float(x))
 except:return False

def reader_label(text):
 if text=='Raw':return 'Before recalibration'
 return (text.replace('Selected C minus reference loss','Selected readout − comparator')
  .replace('Platt / 500 labels','Platt (500 patients)')
  .replace('Hospital-trained heads','Hospital-death heads')
  .replace('ICU-trained heads','ICU-death heads')
  .replace('vs historical C1','vs readout (C=1)')
  .replace('vs constant-trained','vs constant-bin training')
  .replace('vs constant input','vs constant-bin scoring')
  .replace('vs random','vs random encoder')
  .replace('Direct V/B/H token summary','Direct token summary')
  .replace('No later exit-order gap','Discharge-order\nrestriction')
  .replace('Primary H24 set','Primary 24-hour cohort')
  .replace('Within-variable shuffle','Within-stay\nvariable shuffle')
  .replace('Changed-token fraction · eICU','Numeric-token changes · eICU')
  .replace('Fraction of retained tokens changed','Fraction of retained numeric tokens changed'))
def label(fig,ax,s):
 # Put the panel letter in the reserved gutter outside the axes title and data.
 box=ax.get_position()
 fig.text(max(.015,box.x0-.055),min(.985,box.y1+.008),s.upper(),
          ha="right",va="bottom",fontfamily="Arial",fontweight="bold",fontsize=18)
def save(fig,out,stem,w=183,h=140):
 w,h=(float(x)/MM for x in fig.get_size_inches())
 out.mkdir(parents=True,exist_ok=True); (out/"preview150").mkdir(exist_ok=True)
 sub="supplement" if stem.startswith("FigureS") else "main"
 d=out/sub;d.mkdir(exist_ok=True);base=d/stem
 fig.savefig(base.with_suffix(".png"),dpi=300)
 fig.savefig(out/"preview150"/f"{stem}.png",dpi=300)
 fig.savefig(base.with_suffix(".svg"));fig.savefig(base.with_suffix(".pdf"))
 fig.savefig(base.with_suffix(".tiff"),dpi=600,pil_kwargs={"compression":"tiff_lzw"})
 plt.close(fig)
 return {"figure":stem,"path":base.relative_to(out).as_posix(),"width_mm":w,"height_mm":h}
def forest(ax,p,rows=None,show=True):
 rows=rows if rows is not None else p["rows"]
 names=[]
 for i,r in enumerate(rows):
  es=r.get("series",[]); offsets=np.linspace(-.18,.18,len(es)) if len(es)>1 else [0]
  for j,(e,off) in enumerate(zip(es,offsets)):
   y=i+off;color=e.get("color",C[j%len(C)]);x=e.get("estimate")
   if not finite(x):
    ax.text(.98,y,e.get("missing_label","Unavailable"),transform=ax.get_yaxis_transform(),ha="right",fontsize=9.2,fontweight="bold",color="#666")
    continue
   lo,hi=e.get("low"),e.get("high")
   if finite(lo) and finite(hi):
    lo=float(lo);hi=float(hi)
    emph=p.get('short_ci_emphasis',False)
    if lo<=float(x)<=hi: ax.errorbar(x,y,xerr=[[x-lo],[hi-x]],fmt="none",ecolor=color,capsize=2.6 if emph else 1.4,lw=1.0 if emph else .8,zorder=3 if emph else 2)
    else: ax.plot([lo,hi],[y,y],color=color,lw=.8)
   # White edges separate nearby series without moving any estimate.
   pt,=ax.plot(x,y,"o",color=color,ms=2.8 if p.get('short_ci_emphasis') else (3.0 if len(es)>2 else 3.6),
               markeredgecolor='white' if len(es)>2 else color,
               markeredgewidth=.35 if len(es)>2 else .5,label=e.get("label",""))
   if e.get("label") and e["label"] not in names:names.append(e["label"])
 ax.axvline(float(p.get("reference",0)),color="#888",ls="--",lw=.7)
 ax.set_yticks(range(len(rows)));ax.set_yticklabels([reader_label(r.get("label","")) for r in rows] if show else [])
 ax.set_ylim(len(rows)-.45,-.55);ax.set_xlabel(reader_label(p.get("xlabel","")),labelpad=5);ax.set_title(reader_label(p.get("title","")),pad=8)
 if p.get("xlim"):ax.set_xlim(p["xlim"])
 ax.tick_params(direction="out",length=3,width=.8,pad=4);ax.grid(axis="x",color="#ddd",lw=.5);ax.set_axisbelow(True)
 return names
def direct_rows(p):
 rows=[]
 for row in p.get("rows",[]):
  for j,entry in enumerate(row.get("series",[])):
   entry=dict(entry);entry.setdefault("color",C[j%len(C)])
   rows.append({"label":entry.get("label","Class"),"series":[entry]})
 return rows
def calibration(spec):
 fig,axs=plt.subplots(2,2,figsize=(183*MM,172*MM));handles={}
 for i,(ax,p) in enumerate(zip(axs.flat,spec["panels"])):
  m=float(p["axis_max"]);ax.plot([0,m],[0,m],"--",color="#999",lw=.7)
  for j,e in enumerate(p["series"]):
   color=C[j];x=np.asarray(e["x"],float);y=np.asarray(e["y"],float)
   if len(x):
    line,=ax.plot(x,y,color=color,lw=1.25,label=e["label"])
   else:
    line,=ax.plot(float(e["mean_predicted"]),float(e["observed"]),"o",color=color,ms=4,label=e["label"])
   lo=np.asarray(e.get("low",[]),float);hi=np.asarray(e.get("high",[]),float)
   if len(lo)==len(x)==len(hi):ax.fill_between(x,lo,hi,color=color,alpha=.14,lw=0)
   for risk in e.get("decile_mean_risks",[]):
    ax.plot([risk,risk],[.008+j*.013,.017+j*.013],color=color,lw=.75,transform=ax.get_xaxis_transform())
   handles.setdefault(e["label"],line)
  title=p['title'].replace('Fixed 500-label Platt','Platt (500 patients)').replace('Raw','Before recalibration')
  ax.set(xlim=(0,m),ylim=(0,m),xlabel="Predicted risk",ylabel="Observed event probability",title=title)
  note=f"n = {p['n_stays']:,}; deaths = {p['events']:,}\nObserved = {p['observed']:.3f}"
  note+=''.join(f"\n{e['label']}: mean risk {e['mean_predicted']:.3f}" for e in p['series'])
  ax.text(.035,.965,note,transform=ax.transAxes,va="top",fontsize=9.0,fontweight="bold",
          bbox={"facecolor":"white","edgecolor":"none","alpha":.94,"pad":2.2})
  ax.tick_params(labelsize=9,pad=4)
 fig.legend(handles.values(),handles.keys(),loc="upper center",ncol=2,bbox_to_anchor=(.52,.995),
            prop={"family":"Arial","weight":"bold","size":9.2})
 fig.subplots_adjust(left=.16,right=.98,bottom=.09,top=.89,hspace=.49,wspace=.27)
 for i,ax in enumerate(axs.flat):label(fig,ax,"ABCD"[i])
 return fig
def classes(spec,height):
 fig,axs=plt.subplots(3,1,figsize=(183*MM,height*MM))
 for i,(ax,p) in enumerate(zip(axs,spec["panels"])):
  q=dict(p);q["title"]=["Ranking / AUROC change","Brier-score change","Log-loss change"][i]
  forest(ax,q,direct_rows(p))
  lo,hi=ax.get_xlim();ax.set_xlim(lo,hi+(hi-lo)*.30)
  for yi,row in enumerate(direct_rows(p)):
   entry=row['series'][0]
   if finite(entry.get('estimate')):
    anchor=float(entry['high']) if finite(entry.get('high')) else float(entry['estimate'])
    ax.annotate(f"{float(entry['estimate']):+.4g}",(anchor,yi),xytext=(7,0),textcoords='offset points',va='center',ha='left',fontsize=9.2,fontweight="bold",color=entry.get('color',C[0]),clip_on=False)
 fig.subplots_adjust(left=.32,right=.955,bottom=.09,top=.91,hspace=.72)
 for i,ax in enumerate(axs):label(fig,ax,"ABC"[i])
 return fig
def budget(spec):
 fig,axs=plt.subplots(2,2,figsize=(183*MM,151*MM));handles={}
 for i,(ax,p) in enumerate(zip(axs.flat,spec["panels"])):
  ax.axhline(0,color="#888",ls="--",lw=.7)
  for j,e in enumerate(p["series"]):
   x=np.asarray(e["budgets"],float);y=np.asarray(e["estimate"],float);lo=np.asarray(e["low"],float);hi=np.asarray(e["high"],float)
   line,=ax.plot(x,y,"o-",color=C[j],ms=3.5,lw=1,label=e["label"]);ax.fill_between(x,lo,hi,color=C[j],alpha=.13,lw=0)
   handles.setdefault(e["label"],line)
  ax.set_xscale("log");ax.set_xticks([50,200,500,2000],["50","200","500","2000"]);ax.minorticks_off()
  ax.set(xlabel="Recalibration patients",ylabel=p["ylabel"],title=p["title"])
 fig.legend(handles.values(),handles.keys(),loc="upper center",ncol=3,bbox_to_anchor=(.52,.995),
            prop={"family":"Arial","weight":"bold","size":9.2})
 fig.subplots_adjust(left=.17,right=.98,bottom=.11,top=.89,hspace=.48,wspace=.30)
 for i,ax in enumerate(axs.flat):label(fig,ax,"ABCD"[i])
 return fig
def forest_grid(spec,height):
 panels=spec["panels"]
 max_rows=max((len(p.get("rows",[])) for p in panels),default=0)
 stacked=len(panels)>1 and max_rows>=8
 nrows=len(panels) if stacked else 1
 ncols=1 if stacked else len(panels)
 fig_height=max(height,156) if stacked else height
 fig,axs=plt.subplots(nrows,ncols,figsize=(183*MM,fig_height*MM),squeeze=False);handles={}
 # A comparator keeps its color when another comparator is unavailable in a row.
 palette={}
 for pp in spec['panels']:
  for row in pp.get('rows',[]):
   for entry in row.get('series',[]):
    key=entry.get('label','')
    if key not in palette:palette[key]=entry.get('color',C[len(palette)%len(C)])
 for pp in spec['panels']:
  for row in pp.get('rows',[]):
   for entry in row.get('series',[]):entry['color']=palette[entry.get('label','')]
 direct=str(spec.get("number"))=="S2"
 axes=axs[:,0] if stacked else axs[0]
 for i,(ax,p) in enumerate(zip(axes,panels)):
  rows=direct_rows(p) if direct else p.get("rows",[])
  if str(spec.get('number'))=='S1':p=dict(p,xlabel='Numeric-input model minus comparator')
  if direct:
   rows[0]['label']='Observed inputs'
  shared_rows=(not stacked and i>0 and
               [r.get('label','') for r in rows]==[r.get('label','') for r in panels[0].get('rows',[])])
  display_panel=dict(p,short_ci_emphasis=str(spec.get('number')) in {'3','4'})
  forest(ax,display_panel,rows,show=stacked or (p.get('show_labels',True) and not shared_rows))
  ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
  if direct:ax.set_xlim(-.03,1.02)
  if not direct:
   for row in rows:
    for j,entry in enumerate(row.get('series',[])):
     key=entry.get('label','')
     if key:handles.setdefault(key,Line2D([],[],marker=entry.get('marker','o'),color=entry.get('color',C[j%len(C)]),ls='none',ms=3.7,label=key))
 if len(handles)==1:handles={}
 if handles:
  fig.legend(handles.values(),[reader_label(x) for x in handles.keys()],loc="upper center",ncol=min(4,len(handles)),bbox_to_anchor=(.52,.995),
             prop={"family":"Arial","weight":"bold","size":9.2})
 if stacked:
  fig.subplots_adjust(left=.28,right=.98,bottom=.075,top=.88 if handles else .93,hspace=.38)
 elif len(panels)>1:
  fig.subplots_adjust(left=.25,right=.985,bottom=.18 if height<=85 else .14,top=.74 if handles else .80,wspace=.34)
 else:
  fig.subplots_adjust(left=.25,right=.98,bottom=.20 if height<=75 else .15,top=.74 if handles else .80)
 if not direct:
  for i,ax in enumerate(axes):label(fig,ax,chr(65+i))
 return fig
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--repo",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
 ap.add_argument('--only',nargs='*',help='Optional current figure filenames to render')
 a=ap.parse_args();repo=a.repo.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 specpath=repo/"data/presentation/figures.json";bundle=json.loads(specpath.read_text(encoding="utf-8"))
 specs={str(x["number"]):x for x in bundle["figures"]}
 jobs=[
  ("1","Figure2",172,calibration),("2","Figure6",157,classes),
  ("6","Figure1",151,budget),("S1","FigureS3",130,forest_grid),
  ("S2","FigureS5",60,forest_grid),("S3","FigureS4",66,forest_grid),
  ("S4","FigureS2",72,forest_grid),("3","FigureS6",135,forest_grid),
  ("4","FigureS1",80,forest_grid)]
 if a.only:jobs=[j for j in jobs if j[1] in a.only]
 manifest=[]
 if a.only and (out/'render_manifest.json').exists():manifest=[x for x in json.loads((out/'render_manifest.json').read_text())['figures'] if x['figure'] not in a.only]
 conclusions={
 'Figure2':'Calibration curves vary by database and by the fixed 500-label probability update.',
 'Figure6':'Numerical perturbations change ranking and probability losses differently.',
 'Figure1':'Equal target-label budgets leave context-specific residual loss gaps.',
 'FigureS3':'Numeric-input contrasts with constant controls and random initialization differ by seed and cohort.',
 'FigureS5':'The numerical controls change distinct fractions of retained numerical tokens.',
 'FigureS4':'Alternative H24 eligibility rules change the assessment population and paired score gaps.',
 'FigureS2':'Endpoint-specific source heads and updating yield distinct paired ICU-death score gaps.',
 'FigureS6':'Same-window target recovery differs between frozen embeddings and direct token summaries.',
 'FigureS1':'Source-selected readout regularization changes probability-loss contrasts.'}
 for old,stem,height,build in jobs:
  s=specs[old]
  for src in s.get("source_files",[]):
   p=repo/src["path"]
   if not p.is_file() or sha(p)!=src["sha256"]:raise ValueError("source hash mismatch: "+src["path"])
  fig=build(s,height) if build in (classes,forest_grid) else build(s)
  rec=save(fig,out,stem,183,height)
  previous_labels={"Figure2":"Figure1","Figure6":"Figure2","Figure1":"Figure3",
   "Figure3":"Figure4","Figure4":"Figure5","Figure5":"Figure6",
   "FigureS3":"FigureS1","FigureS5":"FigureS2","FigureS4":"FigureS3",
   "FigureS2":"FigureS4","FigureS6":"FigureS5","FigureS1":"FigureS6"}
  rec.update({"old_source_figure":old,"previous_display_figure":previous_labels[stem],
   "release_version":"2026.11.25-r7","uncertainty_scope":s.get("uncertainty_scope","")})
  (out/"manifests").mkdir(exist_ok=True)
  (out/"manifests"/(stem+"_contract.json")).write_text(json.dumps({
   "figure":stem,"previous_display_figure":previous_labels[stem],"release_version":"2026.11.25-r7","backend":"Python/matplotlib","archetype":"quantitative grid","claim":conclusions[stem],
   "source_bundle_sha256":sha(specpath),"source_figure_number":old,
   "source_selectors":[e.get('source_selector') for p in s['panels'] for e in (p.get('series',[]) + [e for row in p.get('rows',[]) for e in row.get('series',[])])],
   "source_files":s.get("source_files",[]),"uncertainty_scope":s.get("uncertainty_scope",""),
   "source_map":"See ../source_map.json","width_mm":rec['width_mm'],"height_mm":rec['height_mm'],
   "exports":["PDF","SVG","600-dpi TIFF","300-dpi PNG","300-dpi preview"]
  },indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
  manifest.append(rec);print("RENDERED "+stem,flush=True)
 (out/"source_map.json").write_text(json.dumps({
    "release_version":"2026.11.25-r7",
  "previous_display_labels_to_current":{
   "Main Figure 1":"Main Figure 2","Main Figure 2":"Main Figure 6",
   "Main Figure 3":"Main Figure 1","Main Figure 4":"Main Figure 3",
   "Main Figure 5":"Main Figure 4","Main Figure 6":"Main Figure 5",
   "Supplementary Figure S1":"Supplementary Figure S3",
   "Supplementary Figure S2":"Supplementary Figure S5",
   "Supplementary Figure S3":"Supplementary Figure S4",
   "Supplementary Figure S4":"Supplementary Figure S2",
   "Supplementary Figure S5":"Supplementary Figure S6",
   "Supplementary Figure S6":"Supplementary Figure S1"},
  "historical_source_figure_identities":"Preserved in source_figure_number and data/presentation/figures.json.",
  
  
  
  
  
  "removed_rows":[],"new_analysis":"None"
 },indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
 (out/"render_manifest.json").write_text(json.dumps({"figures":manifest,"status":"partial_old_bundle_rendered"},indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
 print("PARTIAL_RENDER_COMPLETE",flush=True)
if __name__=="__main__":main()
