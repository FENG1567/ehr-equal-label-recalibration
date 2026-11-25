"""Supplementary temporal sensitivity figure from aggregate source data."""
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R=Path('/data/admin/fengshuo/ehr_fm_20261102/revision_v7_20261124')
S=R/'temporal_availability_v7_r2/aggregate'
O=R/'figure_v7_r2';O.mkdir(exist_ok=False)
g=pd.read_csv(S/'paired_model_bootstrap_v7.csv')
d=pd.read_csv(S/'decision_curve_v7.csv')
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'pdf.fonttype':42,'svg.fonttype':'none',
 'axes.spines.right':False,'axes.spines.top':False,'legend.frameon':False,'axes.linewidth':.7})
fig=plt.figure(figsize=(7.2,6.6));gs=fig.add_gridspec(2,2,height_ratios=[1,1.25],hspace=.62,wspace=.35)
axes=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,:])]
styles={'raw':('#7D858C','o',-.12),'late500_platt_updated':('#2E6D9F','s',.12)}
selectors=[]
for j,metric in enumerate(['brier','logloss']):
 ax=axes[j]
 for yi,seed in enumerate([20261123,20261124,20261125]):
  for calibration,(color,marker,offset) in styles.items():
   q=g[(g.seed==seed)&(g.dataset=='late_test')&(g.calibration==calibration)&
       (g.contrast=='numeric_pretrained_minus_value_pattern')&(g.metric==metric)]
   assert len(q)==1
   row=q.iloc[0];point=float(row.difference);lo=float(row.ci_low_2_5);hi=float(row.ci_high_97_5)
   ax.errorbar(point,yi+offset,xerr=[[point-lo],[hi-point]],fmt=marker,color=color,ms=4,
               capsize=2,elinewidth=1.1)
   selectors.append({'panel':chr(65+j),'seed':seed,'metric':metric,'calibration':calibration,
                     'point':point,'low':lo,'high':hi})
 ax.axvline(0,color='#7D858C',ls=':',lw=.9)
 ax.set_yticks(range(3),[str(s) for s in [20261123,20261124,20261125]])
 ax.invert_yaxis();ax.set_ylabel('Training seed')
 ax.set_xlabel('FM minus summary LR\n'+('Brier score' if metric=='brier' else 'Log loss'))
 ax.text(-.15,1.12,chr(65+j),transform=ax.transAxes,fontweight='bold',fontsize=12)
 ax.set_title('Later-period proper-score difference',pad=11)
 ax.grid(axis='x',alpha=.15)
fig.legend(handles=[Line2D([],[],color=c,marker=m,ls='',label=l) for l,(c,m,_) in
 [('Raw',styles['raw']),('500-label Platt',styles['late500_platt_updated'])]],
 loc='upper left',bbox_to_anchor=(.12,.925),ncol=2,fontsize=7)
from matplotlib.ticker import MaxNLocator
for score_ax in axes[:2]:score_ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
ax=axes[2];part=d[(d.seed==20261123)&(d.dataset=='late_test')]
for arm,color,label in [('numeric_pretrained','#2E6D9F','Numeric pretraining'),
                         ('value_pattern','#B87542','Summary LR')]:
 for calibration,linestyle,suffix in [('raw','--','raw'),('late500_platt_updated','-','500-label Platt')]:
  q=part[(part.arm==arm)&(part.calibration==calibration)&(part.strategy=='model')].sort_values('threshold')
  assert len(q)==7
  ax.plot(q.threshold,q.net_benefit_per_100,linestyle=linestyle,color=color,lw=1.4,label=label+', '+suffix)
  if calibration!='raw':
   ax.fill_between(q.threshold.to_numpy(),q.ci_low_per_100.to_numpy(),q.ci_high_per_100.to_numpy(),color=color,alpha=.12)
ref=part[(part.arm=='numeric_pretrained')&(part.calibration=='raw')&(part.strategy=='treat_all')].sort_values('threshold')
ax.plot(ref.threshold,ref.net_benefit_per_100,color='#8A8A8A',ls=':',lw=1,label='Treat all')
ax.axhline(0,color='#555555',lw=.8,label='Treat none')
ax.set_xticks([.01,.05,.10,.15,.20,.30]);ax.set_xlabel('Threshold probability')
ax.set_ylabel('Theoretical net benefit per 100 patients')
ax.set_title('Decision curves in later-period assessment  |  seed 20261123',pad=11)
ax.text(-.065,1.12,'C',transform=ax.transAxes,fontweight='bold',fontsize=12)
ax.grid(alpha=.15);ax.legend(loc='upper right',fontsize=6.5,ncol=2)
fig.subplots_adjust(left=.12,right=.98,top=.81,bottom=.12)
fig.text(.12,.985,'Conservative calendar separation and H24 recorded-time inputs',fontsize=10,fontweight='bold')
fig.text(.12,.955,'Later assessment: 12,872 patients, 1,588 deaths; shared adaptation: 500 patients, 43 deaths',fontsize=8)
fig.text(.12,.035,'Intervals: 500 paired patient bootstrap draws, conditional on fixed models and one adaptation fit.\nRecorded system time and theoretical net benefit do not establish prospective clinical benefit.',fontsize=7,color='#454545')
for extension in ['pdf','svg','png','tiff']:
 fig.savefig(O/('FigureS5.'+extension),dpi=600 if extension=='tiff' else 180,bbox_inches='tight',facecolor='white')
plt.close(fig)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(O/'source_data.json').write_text(json.dumps({'figure':'Supplementary Figure5','source_hashes':{p.name:sha(p) for p in [S/'paired_model_bootstrap_v7.csv',S/'decision_curve_v7.csv']},'proper_score_selectors':selectors,'decision_curve_seed':20261123,'assessment_n':12872,'events':1588,'interpretation':'restricted-input retrospective sensitivity; no prospective benefit claim','files':{p.name:sha(p) for p in O.glob('FigureS5.*')}},indent=2)+'\n')
print('FIGURE_S5_COMPLETE',flush=True)
