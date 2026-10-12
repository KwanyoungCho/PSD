"""Review figures with explicit timing scopes; rebuild after analyze.py."""
import json
from pathlib import Path
import statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
HERE=Path(__file__).resolve().parent
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
OUT=HERE/'figs';OUT.mkdir(exist_ok=True)
COLORS={'legacy':'#777777','graph0':'#e69f00','graph1':'#009e73','trim1':'#0072b2'}
def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight');plt.close(fig)

micro=json.loads((HERE/'MICRO.json').read_text())
fig,axs=plt.subplots(2,2,figsize=(11,7))
for row,kind in enumerate(('chain','tree')):
 for col,t in enumerate((.7,0)):
    ax=axs[row,col];cs=[c for c in micro[kind]['cases'] if c['temperature']==t]
    for k in cs[0]['median_ms']:
        ys=[st.mean(c['median_ms'][k] for c in cs if c['batch']==b) for b in (1,2,4,8)]
        ax.plot([1,2,4,8],ys,marker='o',label=k)
    ax.set(title=f'{kind}, T={t}',xlabel='Batch',ylabel='Acceptance wall time (ms)',xticks=[1,2,4,8]);ax.legend(fontsize=8)
fig.suptitle('Fixed logits: mean of case medians (two vocabularies / sizes)\nExcludes target forward and communication; includes host results',fontsize=12)
fig.tight_layout();save(fig,'01_acceptance_micro')

rows=json.loads((HERE/'MODEL_RESULTS.json').read_text())
fig,axs=plt.subplots(2,2,figsize=(11,7))
for row,model in enumerate(('llama2','llama3')):
 for col,label in enumerate(('verify','accept')):
    ax=axs[row,col]
    for arm in ('legacy','graph0','graph1','trim1'):
        vals=[r for r in rows if r['kind']=='profile' and r['name'].startswith(model) and f'_{arm}_' in r['name']]
        vals.sort(key=lambda r:r['batch']);xs=[];ys=[]
        for r in vals:
            z=r['steady_profile'];val=z['verify_ms'] if label=='verify' else z['gpu_span_ms'].get('batch_tree_accept')
            if val:xs.append(r['batch']);ys.append(val['mean'])
        if xs:ax.plot(xs,ys,marker='o',label=arm,color=COLORS[arm])
    ax.set(title=f'{model} — '+('whole verification wall' if label=='verify' else 'acceptance GPU span'),xlabel='Batch',ylabel='Mean ms',xticks=[1,2,4,8]);ax.legend(fontsize=8)
fig.suptitle('Full target/draft models; 48-prompt diagnosis, measured pass\nFull batch; cap/clip, first19 steps, captures and next steps excluded',fontsize=12)
fig.tight_layout();save(fig,'02_model_verification')

fig,axs=plt.subplots(1,2,figsize=(11,4))
for ax,model in zip(axs,('llama2','llama3')):
    cs=[r for r in rows if r['kind']=='profile' and r['name'].startswith(model) and '_trim1_' in r['name']];cs.sort(key=lambda r:r['batch'])
    x=np.arange(len(cs));base=np.zeros(len(cs))
    for label in ('batch_kv_restore','batch_kv_stage','batch_glue_forward'):
        y=[r['steady_profile']['gpu_span_ms'].get(label,{}).get('mean',0) for r in cs]
        ax.bar(x,y,bottom=base,label=label);base+=np.array(y)
    ax.set(title=model,xlabel='Batch',ylabel='Mean GPU span per observed step (ms)',xticks=x,xticklabels=[r['batch'] for r in cs]);ax.legend(fontsize=8)
fig.suptitle('Current draft service: restoration, staging and glue compute\nComponent cost summary, not an end-to-end timeline',fontsize=12)
fig.tight_layout();save(fig,'03_existing_kv_cost')

full=[r for r in rows if r['kind']=='full']
if full:
 fig,axs=plt.subplots(2,2,figsize=(11,7))
 for rr,model in enumerate(('llama2','llama3')):
  for cc,metric in enumerate(('tps','verify')):
    ax=axs[rr,cc]
    for j,arm in enumerate(('graph0','graph1','trim1')):
        for i,b in enumerate((1,8)):
            cs=[r for r in full if r['name'].startswith(f'{model}_b{b}_{arm}_')]
            if not cs:continue
            y=[r['performance']['tps'] if metric=='tps' else r['verify_ms']['mean'] for r in cs]
            x=i+(j-1)*.24
            ax.bar(x,st.mean(y),width=.23,label=arm if i==0 else None,color=COLORS[arm])
            ax.scatter([x]*len(y),y,c='black',s=12,zorder=3)
    ax.set(title=model,ylabel='Decode TPS*' if metric=='tps' else 'Whole verification mean (ms)',xticks=[0,1],xticklabels=['B1','B8']);ax.legend(fontsize=8)
 fig.suptitle('Full 480 prompts; profiling OFF; dots = independent process repeats\n*Nominal-B steps without cap/clip only; all outputs still completed',fontsize=12)
 fig.tight_layout();save(fig,'04_full_dataset')

chains=[r for r in rows if r.get('family')=='chain']
if chains:
 fig,axs=plt.subplots(1,2,figsize=(11,4))
 for ax,model in zip(axs,('llama2','llama3')):
    for arm in ('general','fast','packed'):
        cs=[r for r in chains if r['name'].startswith(model) and f'_{arm}_' in r['name']];cs.sort(key=lambda r:r['batch'])
        ax.plot([r['batch'] for r in cs],[r['steady_profile']['verify_ms']['mean'] for r in cs],marker='o',label=arm)
    ax.set(title=model,xlabel='Batch',ylabel='Whole verification wall mean (ms)',xticks=[1,2,4,8]);ax.legend()
 fig.suptitle('Chain full-model diagnosis: same proxy graph; general / fast / fast + packed',fontsize=12)
 fig.tight_layout();save(fig,'05_chain_verification')

greedy=[r for r in rows if r.get('family')=='greedy' and r['kind']=='full']
if greedy:
 fig,axs=plt.subplots(2,2,figsize=(11,7))
 for rr,model in enumerate(('llama2','llama3')):
  for cc,metric in enumerate(('verify','tps')):
   ax=axs[rr,cc]
   for host in (0,1):
    for i,b in enumerate((1,8)):
     cs=[r for r in greedy if r['name']==f'{model}_greedy_b{b}_host{host}_full']
     if not cs:continue
     r=cs[0];y=r['verify_ms']['mean'] if metric=='verify' else r['performance']['tps']
     ax.bar(i+(host-.5)*.32,y,width=.31,label=('Current graph' if host==0 else 'Host prototype') if i==0 else None,
            color=('#009e73' if host==0 else '#cc79a7'))
   ax.set(title=model,ylabel='Whole verification mean (ms)' if metric=='verify' else 'Decode TPS*',
          xticks=[0,1],xticklabels=['B1','B8']);ax.legend(fontsize=8)
 fig.suptitle('Greedy-only benchmark adapter; full480, output64, one measured pass\nSame original graph capture; no production default change; nominal-B / non-boundary steps',fontsize=12)
 fig.tight_layout();save(fig,'06_greedy_prototype')
