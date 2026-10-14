"""Standalone AL/structure figures from final, validated full-corpus results."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

P=Path(__file__).resolve().parent
R=json.loads((P/'RESULTS.json').read_text())
F=json.loads((P/'FINAL_ANALYSIS.json').read_text())
assert all(r['status']=='complete' for r in R.values()) and F['state']=='complete'
OUT=P/'figs';OUT.mkdir(exist_ok=True)
NAMES=['chain_m32','last_c3_m32','frontier_c3_m32','last_c8_m32','frontier_c8_m32','last_c8_m64','frontier_c8_m64']
LABELS=['Chain\nC1 / M32','Last\nC3 / M32','Frontier\nC3 / M32','Last\nC8 / M32','Frontier\nC8 / M32','Last\nC8 / M64','Frontier\nC8 / M64']
COLORS=['#667788','#1f77b4','#e6842a','#649fce','#efb477','#15608f','#b45400']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.axisbelow':True,'pdf.fonttype':42,'savefig.dpi':180})

def metric(h,n,k):return R[h]['arms'][n]['question_quality']['all'][k]['mean']
def avg(n,k):return np.mean([metric(h,n,k) for h in R])
def save(fig,name,pdf):
    fig.savefig(OUT/(name+'.png'),bbox_inches='tight');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
    pdf.savefig(fig,bbox_inches='tight');plt.close(fig)

rows=[]
for h,r in R.items():
    for n in NAMES:
        a=r['arms'][n];row=dict(server=h,method=n)
        row.update({k:v['mean'] for k,v in a['question_quality']['all'].items()})
        for ph in (1,2):
            s=a['shape'][str(ph)]
            row.update({f'p{ph}_hit_{k}':s[k]['mean'] for k in ('valid_k','tree_max_depth','reached_depth_ceiling')})
        rows.append(row)
with (P/'FINAL_TABLE.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

with PdfPages(OUT/'ALL_FIGURES.pdf') as pdf:
    fig,axes=plt.subplots(1,2,figsize=(14,4.3),sharey=True)
    for ax,(h,r) in zip(axes,R.items()):
        vals=[metric(h,n,'al') for n in NAMES]
        ax.bar(range(7),vals,color=COLORS,width=.75)
        ax.set_xticks(range(7),LABELS,fontsize=8);ax.set_ylim(0,4.4)
        ax.axhline(vals[0],color='#667788',ls='--',lw=1)
        ax.set_title('Local / seed 16101' if h=='local' else 'eslab18 / seed 26101')
        ax.grid(axis='y',alpha=.2)
        for i,v in enumerate(vals):ax.text(i,v+.05,f'{v:.3f}',ha='center',fontsize=9)
    axes[0].set_ylabel('AL (question mean; confirmed root included)')
    fig.suptitle('Same draft forward budget: 8 roots x 8 rounds per phase',fontsize=13)
    fig.tight_layout();save(fig,'01_overall_AL',pdf)

    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,k,title in zip(axes,['p1_al','p2_al','miss_al'],['P1-hit AL','P2-hit AL','Miss AL (unchanged chain K=8)']):
        vals=[avg(n,k) for n in NAMES];ax.bar(range(7),vals,color=COLORS)
        for h,off,marker in [('local',-.10,'o'),('remote',.10,'^')]:
            ax.scatter(np.arange(7)+off,[metric(h,n,k) for n in NAMES],c='black',marker=marker,s=18,
                       label='Local' if h=='local' else 'eslab18',zorder=3)
        ax.set_xticks(range(7),LABELS,fontsize=7);ax.set_title(title);ax.set_ylim(0,5.3);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Conditional AL; each arm has its own hit population');axes[2].legend(fontsize=8)
    fig.suptitle('Bars: mean of two replica-level means; markers: individual replicas',fontsize=11)
    fig.tight_layout();save(fig,'02_phase_AL',pdf)

    fig,axes=plt.subplots(1,3,figsize=(14,4.8))
    subset=['frontier_c3_m32','frontier_c8_m32','frontier_c8_m64'];labs=['C3 / M32','C8 / M32','C8 / M64']
    for ph,col in [(1,'#1f77b4'),(2,'#e6842a')]:
        for ax,k,mul,title in [(axes[0],'tree_max_depth',1,'Depth of actually hit trees'),
                               (axes[1],'valid_k',1,'Nodes of actually hit trees'),
                               (axes[2],'reached_depth_ceiling',100,'Accepted path reaches maximum tree depth')]:
            y=[np.mean([R[h]['arms'][n]['shape'][str(ph)][k]['mean'] for h in R])*mul for n in subset]
            ax.plot(range(3),y,'o-',color=col,label=f'P{ph}')
            for i,v in enumerate(y):ax.annotate(f'{v:.1f}',(i,v),xytext=(0,8 if ph==1 else -14),textcoords='offset points',ha='center')
            ax.set_xticks(range(3),labs);ax.set_title(title,fontsize=10);ax.grid(alpha=.2);ax.margins(x=.15,y=.2)
    axes[0].axhline(4,color='gray',ls='--',lw=1,label='C8/M32 depth ceiling')
    axes[0].set_ylabel('Depth');axes[1].set_ylabel('Nodes');axes[2].set_ylabel('Verification events (%)');axes[0].legend(fontsize=8)
    fig.suptitle('Prefix verification budget limits depth even when generation has 8 rounds',fontsize=12)
    fig.tight_layout();save(fig,'03_depth_and_cap',pdf)

    pairs=[('chain_m32','last_c3_m32'),('chain_m32','frontier_c3_m32'),('last_c3_m32','frontier_c3_m32'),
           ('last_c3_m32','last_c8_m32'),('frontier_c3_m32','frontier_c8_m32'),
           ('last_c8_m32','last_c8_m64'),('frontier_c8_m32','frontier_c8_m64'),
           ('frontier_c3_m32','frontier_c8_m64')]
    short=dict(zip(NAMES,['Chain','Last C3/M32','Frontier C3/M32','Last C8/M32','Frontier C8/M32','Last C8/M64','Frontier C8/M64']))
    fig,ax=plt.subplots(figsize=(10,5.6))
    for i,(a,b) in enumerate(pairs):
        v=F['comparisons'][a+' -> '+b]['paired_question_average_over_replicas'];d=v['delta'];lo,hi=v['ci95']
        ax.errorbar(d,i,xerr=[[d-lo],[hi-d]],fmt='o',capsize=4,color='#1f77b4')
        ax.text(1.1,i,f'{d:+.3f} [{lo:+.3f}, {hi:+.3f}]',va='center',fontsize=9)
    ax.set_yticks(range(len(pairs)),[short[a]+' -> '+short[b] for a,b in pairs]);ax.invert_yaxis()
    ax.axvline(0,color='gray',ls='--');ax.grid(axis='x',alpha=.2);ax.set_xlim(-1.02,1.72)
    ax.set_xlabel('AL difference (after - before)');ax.set_title('Paired question bootstrap, unadjusted 95% interval\nTwo replicas averaged within each of 480 questions',fontsize=11)
    fig.tight_layout();save(fig,'04_paired_effects',pdf)

    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    subset=['chain_m32','last_c3_m32','frontier_c3_m32','frontier_c8_m32']
    for ph,ax in enumerate(axes,1):
        for n in subset:
            y=[np.mean([R[h]['arms'][n]['forest']['by_priority_rank'][f'p{ph}_r{r}']['processed_mean'] for h in R]) for r in range(8)]
            assert abs(sum(y)-64)<1e-6
            ax.plot(range(1,9),y,'o-',label=short[n],color=COLORS[NAMES.index(n)])
        ax.set_xticks(range(1,9));ax.set_xlabel('Candidate priority rank (1 = highest)');ax.set_title(f'P{ph}: 64 active draft inputs shared across 8 roots');ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean forward input rows allocated per root');axes[1].legend(fontsize=8)
    fig.suptitle('All exported forests; not only hit roots. Weighting effectiveness needs a separate ablation.',fontsize=10)
    fig.tight_layout();save(fig,'05_root_budget_allocation',pdf)

    groups=sorted(k for k in R['local']['arms']['chain_m32']['question_quality'] if k!='all')
    subset=['last_c3_m32','frontier_c3_m32','frontier_c8_m32','frontier_c8_m64']
    z=np.array([[np.mean([R[h]['arms'][n]['question_quality'][g]['al']['mean']-R[h]['arms']['chain_m32']['question_quality'][g]['al']['mean'] for h in R]) for n in subset] for g in groups])
    fig,ax=plt.subplots(figsize=(9,5));v=max(abs(z.min()),abs(z.max()))
    im=ax.imshow(z,cmap='RdBu',vmin=-v,vmax=v,aspect='auto')
    ax.set_xticks(range(len(subset)),[short[n] for n in subset],fontsize=9);ax.set_yticks(range(len(groups)),groups)
    for i in range(len(groups)):
        for j in range(len(subset)):ax.text(j,i,f'{z[i,j]:+.3f}',ha='center',va='center',color='white' if abs(z[i,j])>.55*v else 'black')
    ax.set_title('AL difference from chain by task (80 questions each)\nMean of two replica-level differences',fontsize=11)
    fig.colorbar(im,ax=ax,label='AL difference');fig.tight_layout();save(fig,'06_task_AL',pdf)
print('Saved six PNG/PDF figures, combined PDF, and FINAL_TABLE.csv')
