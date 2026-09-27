"""Standalone figures from saved measurements; no model runs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
COLORS={40:'#cc6677',56:'#228833',72:'#4477aa'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.dpi':180,'pdf.fonttype':42})


def save(fig,name):
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(HERE/'figures'/f'{name}.{ext}',bbox_inches='tight')
    plt.close(fig)


def main():
    (HERE/'figures').mkdir(exist_ok=True)
    validation=json.loads((HERE/'validation_summary.json').read_text())
    rows=validation['rows']
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    variables=[('actual_cycle_ms','cycle_ms','Verification cycle (ms)',validation['cycle_mape_pct']),
               ('u','tokens_per_step','Emitted tokens per step',validation['reward_mape_pct']),
               ('tps','tps','Output tokens per second',validation['tps_mape_pct'])]
    for ax,(a,p,label,mape) in zip(axes,variables):
        xx=[];yy=[]
        for layer in COLORS:
            subset=[r for r in rows if r['config']['exit']==layer]
            x=[r[a] if a in r else r['actual'][a] for r in subset]
            y=[r['predicted'][p] for r in subset]
            xx+=x;yy+=y;ax.scatter(x,y,color=COLORS[layer],label=f'Exit {layer}',s=35)
        lo=min(xx+yy)*.95;hi=max(xx+yy)*1.05
        ax.plot([lo,hi],[lo,hi],color='.5',ls='--',lw=1)
        ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Measured',ylabel='Predicted',title=f'{label}\nMAPE {mape:.2f}%')
    axes[0].legend(frameon=False)
    fig.suptitle('Frozen 5-anchor model: 19 configurations on held-out prompts',y=1.03)
    save(fig,'primary_prediction')
    fig,axes=plt.subplots(1,3,figsize=(10,3.3),sharey=True)
    mats=[]
    for layer in COLORS:
        mats.append(np.array([[next(r['actual']['tps'] for r in rows if
            (r['config']['exit'],r['config']['k1'],r['config']['k2'])==(layer,k1,k2))
            for k2 in [2,4]] for k1 in [4,6,10]]))
    for ax,layer,mat in zip(axes,COLORS,mats):
        im=ax.imshow(mat,vmin=min(m.min() for m in mats),vmax=max(m.max() for m in mats),cmap='viridis')
        for i in range(3):
            for j in range(2):ax.text(j,i,f'{mat[i,j]:.1f}',ha='center',va='center',color='white' if mat[i,j]<70 else 'black')
        ax.set(xticks=[0,1],xticklabels=[2,4],yticks=[0,1,2],yticklabels=[4,6,10],xlabel='P2 forwards',title=f'Exit {layer}')
    axes[0].set_ylabel('P1 forwards')
    fig.suptitle('Measured output TPS (B1, T=0.7; 16 prompts, 128 output tokens)',y=1.04)
    save(fig,'validation_grid')
    if (HERE/'confirmation_summary.json').exists():
        data=json.loads((HERE/'confirmation_summary.json').read_text());rr=data['rows']
        fig,axes=plt.subplots(1,2,figsize=(10,3.8))
        labels=[f"Exit {r['config']['exit']}, K={r['config']['k1']}/{r['config']['k2']}" for r in rr]
        y=np.arange(len(rr))
        axes[0].barh(y,[r['tps'] for r in rr],color=[COLORS[r['config']['exit']] for r in rr])
        axes[0].set(yticks=y,yticklabels=labels,xlabel='Output tokens per second');axes[0].invert_yaxis()
        for i,r in enumerate(rr):axes[0].text(r['tps']+.4,i,f"{r['tps']:.1f}",va='center')
        axes[0].set_xlim(0,max(r['tps'] for r in rr)*1.16)
        for i,r in enumerate(rr):
            mean=r['gain_vs_default_pct'];low,high=r['gain_vs_default_ci95_pct']
            axes[1].plot([low,high],[i,i],color=COLORS[r['config']['exit']],lw=2)
            axes[1].scatter([mean],[i],color=COLORS[r['config']['exit']])
        axes[1].axvline(0,color='.6',ls='--');axes[1].set(yticks=y,yticklabels=[],xlabel='Gain vs exit 56, K=8/4 (%), paired 95% CI')
        axes[1].invert_yaxis();fig.suptitle('Independent confirmation: 16 prompts x 2 seeds, 256 output tokens',y=1.03)
        save(fig,'confirmation')
    if (HERE/'candidate_summary.json').exists():
        data=json.loads((HERE/'candidate_summary.json').read_text());rr=data['results']
        fig,ax=plt.subplots(figsize=(6.5,3.8))
        x=[r['layer'] for r in rr]
        ax.plot(x,[100*r['proxy_baseline'] for r in rr],'o-',label='Proxy / top-M / original h')
        ax.plot(x,[100*r['calibrated_proxy'] for r in rr],'^--',color='.35',label='Proxy with calibrated allocation')
        ax.plot(x,[100*r['validation'] for r in rr],'s-',label='Policy chosen on calibration prompts')
        ax.set(xlabel='Probe layer index',ylabel='Pre-dedup root coverage (%)',title='Held-out snapshot replay, root budget 15')
        ax.legend(frameon=False);save(fig,'candidate_replay')


if __name__=='__main__':main()
