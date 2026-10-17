from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent
r=json.loads((H/'DIRECT_COMPARISON.json').read_text())['tree_T07']
f,ax=plt.subplots(figsize=(9,4.5));w=.35;x=np.arange(4)
for j in range(2):
 v=[r['sources'][s]['step_ms'][j] for s in ('1','2','0','all')]
 bars=ax.bar(x+(j-.5)*w,v,w,label='ver1.3' if j==0 else 'ver1.4',color=['#667b93','#e79739'][j]);ax.bar_label(bars,fmt='%.3f',padding=3,rotation=25,fontsize=8)
ax.set_xticks(x,['P1 hit','P2 hit','Miss','All']);ax.set_ylabel('Whole step latency (ms)');ax.set_ylim(0,24);ax.grid(axis='y',alpha=.2);ax.legend(frameon=False)
ax.set_title('Direct ver1.3 vs ver1.4: Llama2-7B + llama-68m\nB1, T0.7, K1/K2=8/5, exit22, M15; two GPU pairs, reversed run order')
f.tight_layout()
for ext in ['pdf','png']:f.savefig(H/('DIRECT_LATENCY.'+ext),dpi=180,bbox_inches='tight')
