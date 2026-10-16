"""Primary whole-step time and warm-profile gap, kept in separate panels."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
steps=json.loads((P/'STEP_COMPARISONS.json').read_text())['comparisons']
profiles={r['tag']:r for r in json.loads((P/'PROFILE_COMPONENTS.json').read_text())['runs']}
fig,axes=plt.subplots(2,2,figsize=(12,7.8))
for row,model in enumerate(('7b','13b')):
 s=next(x for x in steps if x['model']==model and x['stage']=='final' and x['cohort'].startswith('2x'))
 a={str(x['state']):x for x in profiles[f'before_{model}_profile_warm']['summary']}
 b={str(x['state']):x for x in profiles[f'restore_{model}_profile_warm']['summary']}
 keys=['1','2','0','all'];x=np.arange(4);width=.35
 primary_a=[s['source'][k]['before_ms'] for k in keys[:3]]+[s['before_ms']]
 primary_b=[s['source'][k]['after_ms'] for k in keys[:3]]+[s['after_ms']]
 panels=[(primary_a,primary_b,'Whole decode step / profile OFF'),
         ([a[k]['whole_gap_ms'] for k in keys],[b[k]['whole_gap_ms'] for k in keys],'Forward-to-forward gap / warm profile ON')]
 for col,(av,bv,title) in enumerate(panels):
  ax=axes[row,col];ax.bar(x-width/2,av,width,label='Current ver1.2',color='#64748b');ax.bar(x+width/2,bv,width,label='Respond before KV restore',color='#0f766e')
  ax.set_xticks(x,['P1 hit','P2 hit','Miss','All']);ax.set_ylabel('Latency (ms)');ax.set_title(f'Llama2-{model.upper()} + 68M | {title}',fontsize=10)
  ax.grid(axis='y',alpha=.22);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
  for i,(v,z) in enumerate(zip(av,bv)):
   ax.text(i,max(v,z)+.01*max(av),f'{100*(1-z/v):+.1f}%',ha='center',fontsize=8)
  ax.set_ylim(0,max(av+bv)*1.15)
axes[0,0].legend(fontsize=8,loc='lower left')
fig.suptitle('B1 tree / K1=K2=3 / C=3 / identical outputs and acceptance events',fontsize=12)
fig.text(.5,.015,'Percent labels: latency reduction. Fixed 12-input latency diagnosis; RTX3090. Profile ON/OFF are different measurement cohorts.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(P/'LATENCY_COMPARISON.png',dpi=180);fig.savefig(P/'LATENCY_COMPARISON.pdf');plt.close(fig)
