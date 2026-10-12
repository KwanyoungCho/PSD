"""Standalone figures for the verified common-fix report."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=Path(__file__).resolve().parent;out=R/'figs';out.mkdir(exist_ok=True)
data=json.loads((R/'ANALYSIS.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
colors=['#7f8c8d','#2980b9','#e67e22'];names=['Paper a82f7d2','Before common fixes 3d51664','After common fixes']
fig,ax=plt.subplots(figsize=(10,4.4));x=np.arange(3);width=.24
for i,arm in enumerate(['paper','before','after']):
 vals=[data['runs'][f'{c}/{arm}.json']['summary']['tps'] for c in ['tree_r0','tree_r1','chain']]
 bars=ax.bar(x+(i-1)*width,vals,width,color=colors[i],label=names[i]);ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=8)
ax.set_xticks(x,['Tree repeat 0','Tree repeat 1','Chain']);ax.set_ylim(0,110);ax.set_ylabel('Decode tokens / second');ax.set_title('480 stored inputs per run | B=1, T=1 | Same step population')
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3,frameon=False,fontsize=8);fig.tight_layout();fig.savefig(out/'TPS.png');fig.savefig(out/'TPS.pdf');plt.close(fig)
comp=json.loads((R/'COMPONENTS.json').read_text())['arms']
keys=['target_pre_ms','target_post_ms','draft_p1_replay_ms','draft_p2_replay_ms','target_tree_walk_ms','draft_p1_total_ms','draft_p2_total_ms']
labels=['Target pre-exit','Target post-exit','Draft P1 replay','Draft P2 replay','Tree accept walk','Draft P1 envelope','Draft P2 envelope']
fig,axs=plt.subplots(1,2,figsize=(12,4.5),gridspec_kw={'width_ratios':[4,3]})
for ax,indices in [(axs[0],range(4)),(axs[1],range(4,7))]:
 x=np.arange(len(indices))
 for i,arm in enumerate(['paper','before','after']):
  vals=[comp['profile_'+arm]['summary'][keys[j]]['median'] for j in indices]
  bars=ax.bar(x+(i-1)*width,vals,width,color=colors[i],label=names[i]);ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=7)
 ax.set_xticks(x,[labels[j] for j in indices],rotation=15,ha='right');ax.set_ylabel('Median milliseconds');ax.margins(y=.18)
fig.suptitle('Separate 24-input profiles | Event spans include launch gaps; envelopes overlap')
handles,lbl=axs[0].get_legend_handles_labels();fig.legend(handles,lbl,loc='lower center',ncol=3,frameon=False,fontsize=8);fig.tight_layout(rect=[0,.08,1,.94]);fig.savefig(out/'COMPONENTS.png');fig.savefig(out/'COMPONENTS.pdf');plt.close(fig)
