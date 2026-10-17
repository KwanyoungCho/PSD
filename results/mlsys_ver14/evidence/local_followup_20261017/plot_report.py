"""Local whole-step and component evidence; keep measurement scopes separate."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
read=lambda n:json.loads((HERE/n).read_text())
d=read('COMPARISONS.json');h=read('HELDOUT.json');p=read('PROFILES.json');m=read('MIXED_ANALYSIS.json')
labels={'buckets':'Smaller model graphs','overlap':'q receive overlap','prep':'Remove redundant preparation'}
fig,ax=plt.subplots(figsize=(9,4))
for i,v in enumerate(('buckets','overlap','prep')):
    r=d[v];x=r['reduction_pct'];lo,hi=r['ci95_pct']
    ax.errorbar(x,i-.10,xerr=[[x-lo],[hi-x]],fmt='o',color='#386fa6',capsize=4)
    if v in h:
        x=h[v]['reduction_pct'];lo,hi=h[v]['ci95_pct']
        ax.errorbar(x,i+.10,xerr=[[x-lo],[hi-x]],fmt='s',color='#d27536',capsize=4)
ax.axvline(0,color='black',lw=.8);ax.set(yticks=range(3),yticklabels=[labels[v] for v in ('buckets','overlap','prep')],xlabel='Whole-step reduction (%) — positive is faster',title='Local RTX4090 | fixed parameters | question-cluster 95% CI')
ax.plot([],[],'o',color='#386fa6',label='12-input process repeats');ax.plot([],[],'s',color='#d27536',label='Disjoint 48 questions / 56 turns')
ax.legend(fontsize=8);ax.grid(axis='x',alpha=.2);fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(HERE/f'WHOLE_STEP.{ext}',dpi=180)
plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(13,4))
tags=[('baseline_pair0_profile','prep_pair0_profile'),('baseline_pair1_profile','overlap_pair1_profile'),('baseline_pair0_profile','buckets_pair0_profile')]
for ax,(before,after),v in zip(axes,tags,('prep','overlap','buckets')):
    keys=['response_ms','response_to_forward_ms','tree_verify_meta_cpu_ms','tree_proxy_graph_replay_ms']
    aa=p[before]['summary']['all'];bb=p[after]['summary']['all'];xs=np.arange(len(keys))
    ax.bar(xs-.18,[aa.get(k,0) for k in keys],.36,label='Before',color='#8c9caa')
    ax.bar(xs+.18,[bb.get(k,0) for k in keys],.36,label='Candidate',color='#3e876e')
    ax.set(xticks=xs,xticklabels=['Response\nspan*','Response→\nforward','CPU tree\nmetadata','Proxy graph\ninput+replay'],ylabel='ms',title=labels[v]);ax.grid(axis='y',alpha=.2)
axes[0].legend(fontsize=8)
fig.suptitle('Profile ON diagnostics, not whole-step gain estimates\n*q-overlap response span ends at enqueue; payload readiness is enforced at consumers.',fontsize=10)
fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(HERE/f'BREAKDOWN.{ext}',dpi=180)
plt.close(fig)
fig,ax=plt.subplots(figsize=(7,3.8));x=m['all']['current_ms'];lo,hi=m['all']['current_ci95_ms']
ax.errorbar([0],[x],yerr=[[x-lo],[hi-x]],fmt='o',capsize=6,color='#3e876e')
ax.axhline(0,color='gray');ax.set(xticks=[0],xticklabels=['Prep change (current step)'],ylabel='Step latency change (ms)',title='Randomized modes inside one engine, 12 question clusters\nSource, valid nodes, context, pass and previous source controlled; one-step lag included')
ax.grid(axis='y',alpha=.2);fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(HERE/f'WITHIN_ENGINE.{ext}',dpi=180)
