"""Standalone figures for the measured scope; never plot prototype speed as TPS."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=Path(__file__).resolve().parent
x=json.loads((p/'NUMBERS.json').read_text())
out=p/'figs';out.mkdir(exist_ok=True)
colors=['#a3a3a3','#3c78a8','#198b77','#db903c']
fig,axes=plt.subplots(2,2,figsize=(10,6.5),layout='constrained')
for col,(model,title) in enumerate([('llama2','LayerSkip-Llama2-7B + AMD135M'),('llama3','LayerSkip-Llama3-8B + Qwama0.5B')]):
    groups=[x[model][s] for s in ['base','packed','candidate','ssd']]
    if any(g['n']!=3 for g in groups):raise ValueError('Three complete runs required')
    for row,key in enumerate(['tps','al']):
        ax=axes[row,col]
        means=[g[key]['mean'] for g in groups];errs=[g[key]['sd'] for g in groups]
        ax.bar(np.arange(4),means,yerr=errs,capsize=4,color=colors,width=.65)
        ax.set_xticks(np.arange(4),['Base','Packed','Final','SSD'])
        ax.set_ylabel('Decode tokens / second' if key=='tps' else 'AL (including recovery)')
        ax.set_ylim(0,max(means)*1.2)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        for i,y in enumerate(means):ax.text(i,y+max(means)*.04,f'{y:.1f}' if key=='tps' else f'{y:.3f}',ha='center',fontsize=9)
        if row==0:ax.set_title(title,fontsize=11)
fig.suptitle('B=8, T=0.7: 480 first-turn inputs × 3 seeds\nFinal: Llama2 packed + fan-out 4; Llama3 packed + mixed-miss AR',fontsize=11)
fig.savefig(out/'batch8_tps_al.png',dpi=180)
fig.savefig(out/'batch8_tps_al.pdf')
plt.close(fig)
