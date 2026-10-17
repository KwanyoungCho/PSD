from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
c=json.loads((HERE/'COMPARISON.json').read_text())
p=json.loads((HERE/'PROFILES.json').read_text())
assert p['baseline_fused_profile'] and p['fused_bulk_profile']
fig,axes=plt.subplots(1,3,figsize=(18,5.1))
colors=['#667b93','#e79739'];width=.36
for ax,title,keys in [(axes[0],'Draft work (profile ON)',[
    ('Hit response P1','1','draft:hit_cache_respond_hit_k1'),
    ('Hit response P2','2','draft:hit_cache_respond_hit_k2'),
    ('Glue P1','1','draft:glue'),('Glue P2','2','draft:glue')]),
    (axes[1],'Target wait and model (profile ON)',[
    ('Response P1','1','target:target_recv_response_wait'),
    ('Response P2','2','target:target_recv_response_wait'),
    ('Model pre+post','all',None)])]:
    for j,tag in enumerate(('baseline_fused_profile','fused_bulk_profile')):
        values=[p[tag][source].get(key,0) if key else p[tag][source]['target:graph_pre']+p[tag][source]['target:graph_post'] for _,source,key in keys]
        bars=ax.bar(np.arange(len(keys))+(j-.5)*width,values,width,color=colors[j],label='Existing' if j==0 else 'Combined')
        ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=3)
    ax.set_xticks(range(len(keys)),[k[0] for k in keys],rotation=20)
    ax.set_title(title);ax.set_ylabel('Milliseconds');ax.set_ylim(0,ax.get_ylim()[1]*1.12);ax.grid(axis='y',alpha=.2)
r=c.get('heldout_final') or c['fused']
for j in range(2):
    values=[r['sources'][source]['step_ms'][j] for source in ('1','2','0','all')]
    bars=axes[2].bar(np.arange(4)+(j-.5)*width,values,width,color=colors[j],label='Existing' if j==0 else 'Combined')
    axes[2].bar_label(bars,fmt='%.2f',fontsize=8,padding=3,rotation=30)
axes[2].set_xticks(range(4),['P1 hit','P2 hit','Miss','All'])
axes[2].set_title('Whole step (profile OFF, '+('heldout final' if c.get('heldout_final') else '3 process pairs')+')')
axes[2].set_ylabel('Milliseconds');axes[2].set_ylim(0,24);axes[2].grid(axis='y',alpha=.2)
axes[0].legend(frameon=False)
fig.suptitle('Llama2-7B + llama-68m, B1 / K1=8 K2=5 / exit 22 / M=15\nComponents overlap; component savings must not be added to whole-step savings.',fontsize=12)
fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(HERE/('BREAKDOWN.'+ext),dpi=180,bbox_inches='tight')
