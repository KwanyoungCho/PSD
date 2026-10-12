"""Original-paper contrasts: explicit scopes, repeated runs, standalone exports."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

R=Path(__file__).resolve().parent;OUT=R/'figs';OUT.mkdir(exist_ok=True)
bundle=PdfPages(OUT/'ALL_FIGURES.pdf')
d=json.loads((R/'ANALYSIS.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'figure.dpi':130})
names={'chain_b1':'Llama2 chain B1','llama3_chain_b1':'Llama3 chain B1','chain_b8':'Llama2 chain B8',
       'tree_b1':'Llama2 minimal tree B1','kv_b1':'Llama2 KV repairs B1','unified_b1':'Llama2 unified tree B1',
       'tree_chain_b1':'Llama2 tree + chain fast B1'}
order=list(names)
full=sorted([a for a in d['aggregates'] if a['scope']=='full'],key=lambda a:order.index(a['contrast']))

def save(fig,name):
 fig.savefig(OUT/(name+'.png'),dpi=220,bbox_inches='tight')
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');bundle.savefig(fig,bbox_inches='tight');plt.close(fig)

if full:
 fig,axs=plt.subplots(1,2,figsize=(14,4.7));x=np.arange(len(full));w=.36
 for ax,metric,label in [(axs[0],'verify','Verification wall time (ms)'),(axs[1],'tps','Eligible-step TPS (tokens/s)')]:
  for arm,offset,color in [('base',-w/2,'#7b8b9a'),('candidate',w/2,'#287d8e')]:
   key=f'{metric}_{arm}_ms' if metric=='verify' else f'{metric}_{arm}'
   vals=[a[key] for a in full];bars=ax.bar(x+offset,vals,w,color=color,label='Paper reference' if arm=='base' else 'Candidate')
   ax.bar_label(bars,fmt='%.2f' if metric=='verify' else '%.1f',fontsize=8,padding=3 if arm=='base' else 14)
   for i,a in enumerate(full):
    raw=[p[key] for p in d['pairs'] if p['contrast']==a['contrast'] and p['scope']=='full']
    ax.scatter([i+offset]*len(raw),raw,s=13,color='black',zorder=4)
  ax.set_ylim(0,max(max(a[f'{metric}_{z}_ms' if metric=='verify' else f'{metric}_{z}'] for z in ('base','candidate')) for a in full)*1.15)
  ax.set_ylabel(label);ax.set_xticks(x,[names[a['contrast']] for a in full],rotation=28,ha='right');ax.legend(fontsize=8)
 fig.suptitle('Actual paper-code controls | dense models, T=1, all 480 prompts\nDots: process runs; unified and tree+chain-fast each combine one PIX and one NODE contrast',fontsize=12)
 save(fig,'01_paper_latency_tps')

 fig,ax=plt.subplots(figsize=(10,4.5))
 for i,a in enumerate(full):
  for arm,off in [('base',-.18),('candidate',.18)]:
   selected={p[arm] for p in d['pairs'] if p['contrast']==a['contrast'] and p['scope']=='full'}
   rows=[r for r in d['rows'] if r['name'] in selected]
   verify=np.mean([r['summary']['verify_ms'] for r in rows]);step=np.mean([r['summary']['step_ms'] for r in rows])
   if step<verify:raise AssertionError('Timing scope is not nested')
   ax.bar(i+off,verify,.34,color='#3687a0',label='Verification' if i==0 and arm=='base' else None)
   ax.bar(i+off,step-verify,.34,bottom=verify,color='#e5ae62',label='Draft wait + other host work' if i==0 and arm=='base' else None)
   ax.text(i+off,step+.3,'P' if arm=='base' else 'C',ha='center',fontsize=9)
 ax.set_xticks(range(len(full)),[names[a['contrast']] for a in full],rotation=25,ha='right')
 ax.set_ylabel('Mean eligible batch-step wall time (ms)');ax.legend()
 ax.set_title('Whole-step breakdown: P = paper reference, C = candidate\nRemainder is a wall-time difference, not GPU kernel time')
 save(fig,'02_step_breakdown')

 fig,axs=plt.subplots(1,2,figsize=(11,4.4));x=np.arange(len(full))
 for ax,metric,label in [(axs[0],'al','Accepted length (including recovery)'),(axs[1],'hit','Request cache hit (%)')]:
  factor=100 if metric=='hit' else 1
  for arm,off,col in [('base',-.18,'#7b8b9a'),('candidate',.18,'#287d8e')]:
   bars=ax.bar(x+off,[a[f'{metric}_{arm}']*factor for a in full],.36,color=col,label='Paper' if arm=='base' else 'Candidate')
   ax.bar_label(bars,fmt='%.2f',padding=3 if arm=='base' else 14,fontsize=8)
  ax.set_ylim(0,max(a[f'{metric}_{z}']*factor for a in full for z in ('base','candidate'))*1.18)
  ax.set_ylabel(label);ax.set_xticks(x,[names[a['contrast']] for a in full],rotation=28,ha='right');ax.legend(fontsize=8)
 axs[0].set_title('Check whether the accepted workload changed')
 axs[1].set_title('Unified executor consumes RNG differently')
 save(fig,'03_al_hit')

prefix=[]
for model in ('llama2','llama3'):
 p=R/(model+'_paper_prefix.json')
 if p.exists():
  data=json.loads(p.read_text())
  if data['status']=='complete':prefix.append((model,data))
if prefix:
 fig,axs=plt.subplots(1,len(prefix),figsize=(6*len(prefix),4),squeeze=False)
 for ax,(model,data) in zip(axs[0],prefix):
  keys=sorted({(c['batch'],c['length']) for c in data['cases']})
  for arm,off,col in [('cold',-.18,'#7b8b9a'),('cached',.18,'#287d8e')]:
   values=[np.mean([c['arms'][arm]['prefill_ms'] for c in data['cases'] if (c['batch'],c['length'])==k]) for k in keys]
   bars=ax.bar(np.arange(len(keys))+off,values,.36,color=col,label=arm);ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=2)
  ax.set_xticks(range(len(keys)),[f'B{b}\nL{n}' for b,n in keys]);ax.set_ylabel('Prefill wall time (ms)');ax.set_title(model+' | narrow paper KV repairs');ax.legend()
 fig.suptitle('Correct cold vs cached prefix: 3 repeats, same greedy outputs\nThis is prefix reuse, not replacement of tree glue',y=1.05)
 save(fig,'04_paper_prefix')
profile=R/'PROFILE_BREAKDOWN.json'
if profile.exists():
 p=json.loads(profile.read_text())['arms']
 groups=[('Target pre-exit','graph_pre','batch_target_pre'),
         ('Target post-exit','graph_post','batch_target_post'),
         ('Final logits','final_logits','batch_target_final_logits'),
         ('Acceptance*','verify_sample_accept','batch_tree_accept')]
 fig,axs=plt.subplots(1,2,figsize=(11,4.2),gridspec_kw={'width_ratios':[1.3,1]})
 for ax,cols in zip(axs,[groups[:2],groups[2:]]):
  x=np.arange(len(cols))
  for arm,idx,off,color in [('base',1,-.18,'#7b8b9a'),('unified',2,.18,'#287d8e')]:
   values=[p[arm]['cuda_span_ms'][g[idx]]['mean'] for g in cols]
   bars=ax.bar(x+off,values,.36,color=color,label='Paper reference' if arm=='base' else 'Unified')
   ax.bar_label(bars,fmt='%.3f',padding=3 if arm=='base' else 14,fontsize=9)
  ax.set_xticks(x,[g[0] for g in cols]);ax.set_ylabel('Conditional CUDA event span (ms)')
  ax.set_ylim(0,max(p[a]['cuda_span_ms'][g[i]]['mean'] for g in cols for a,i in [('base',1),('unified',2)])*1.2)
  ax.legend(fontsize=9)
 fig.suptitle('Target spans | 16 measured prompts, excludes capture steps\n*Paper acceptance includes KV dispatch; unified acceptance excludes its separate commit span',fontsize=11)
 save(fig,'05_target_spans')
bundle.close()
print('Figures:',len(list(OUT.glob('*.pdf')))-1)
