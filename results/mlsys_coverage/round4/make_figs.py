"""Standalone figures from saved aggregates; no inference or refitting."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
OUT=HERE/'figs'; OUT.mkdir(exist_ok=True)
rows=json.loads((HERE/'RESULTS.json').read_text())
lookup={(r['group'],r['name']):r for r in rows}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})

def save(fig,name):
 fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight')
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

names=['reach','q_gain','reach_gain','reach_frontier','reach_gain_frontier','dense_reach','dense_reach_gain_frontier']
labels=['Reach (old table)','q + gain','Reach + gain','Reach + frontier','Reach + gain + frontier','Reach (dense calib.)','Reach + gain + frontier (dense calib.)']
fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
for ax,model in zip(axes,['llama2','llama3']):
 for i,name in enumerate(names):
  r=lookup.get((model+'_full',name))
  if not r or 'delta_clean_al_ci95' not in r:continue
  b=lookup[(model+'_full',r['reference'])]['boundary_excluded_al']
  v=r['delta_clean_al']/b*100;ci=np.array(r['delta_clean_al_ci95'])/b*100
  ax.errorbar(v,i,xerr=[[max(0,v-ci[0])],[max(0,ci[1]-v)]],fmt='o',capsize=3,color='#176d8b' if 'dense' not in name else '#d37430')
 ax.axvline(0,color='0.6',lw=1);ax.set_title(model.replace('llama','Llama')+' / B=8');ax.set_xlabel('Boundary-excluded AL change vs q-path (%)');ax.grid(axis='x',alpha=.2)
axes[0].set_yticks(range(len(names)),labels);axes[0].invert_yaxis()
fig.suptitle('480 questions; 3 target-seed passes; paired question bootstrap 95% CI',fontsize=11)
fig.tight_layout();save(fig,'01_tree_policy_al')

fig,axes=plt.subplots(2,2,figsize=(10,7))
for i,model in enumerate(['llama2','llama3']):
 for j,batch in enumerate([1,8]):
  ax=axes[i,j];group=model+('_b1' if batch==1 else '_full')
  for name,label in [('miss_chain1','chain1'),('q_path','chain2 (default)'),('miss_chain4','chain4'),('miss_star3','star3'),('miss_tree2x2','tree2x2')]:
   r=lookup.get((group,name))
   if not r:continue
   tps=r['later_pass_tps_mean'] or r['tps_mean']
   ax.scatter(tps,r['boundary_excluded_al'],s=40)
   ax.annotate(label,(tps,r['boundary_excluded_al']),xytext=(4,4),textcoords='offset points',fontsize=8)
  ax.set_title(model.replace('llama','Llama')+f' / B={batch}');ax.set_xlabel('Returned-token TPS (B8: later passes)');ax.set_ylabel('Boundary-excluded AL incl. recovery');ax.grid(alpha=.2)
fig.suptitle('Miss fallback: depth/width tradeoff; same model and GPU pair within each panel',fontsize=11)
fig.tight_layout();save(fig,'02_miss_tradeoff')

profiles=json.loads((HERE/'timeline.json').read_text())
prof={r['name']:r for r in profiles}
profile_names=['llama2_b1_k4_2','llama3_b1_k4_2','llama2_b8_k4_2_bulk0','llama3_b8_k4_2_bulk0','llama2_b8_k4_2_fused','llama3_b8_k4_2_fused','llama2_b8_k2_1_bulk1','llama3_b8_k2_1_bulk1']
# Explicitly resolve recorded experiment names, never silently pick a tail batch.
print('Available timing names:',list(prof))
chosen=[n for n in profile_names if n in prof]
if chosen:
 fig,ax=plt.subplots(figsize=(10,4.5))
 for i,name in enumerate(chosen):
  g=prof[name]['groups']['full_batch']
  if not g['n']:continue
  for off,key,col,lab in [(-.15,'p1_before_proxy_ms','#176d8b','P1 before proxy'),(.15,'p2_before_ready_ms','#d37430','P2 before target ready')]:
   ax.scatter(g['metrics'][key]['p50'],i+off,color=col,label=lab if i==0 else None)
 ax.axvline(0,c='0.5');ax.set_yticks(range(len(chosen)),chosen);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
 ax.set_xlabel('Median deadline margin (ms): positive = finishes in time');ax.legend();fig.tight_layout();save(fig,'03_phase_deadlines')

# One actual step per panel nearest its group's median P2 deadline margin.
# Do not add medians from different steps into a synthetic timeline.
choices=['llama2_b1_k4_2','llama3_b1_k4_2','llama2_b8_k4_2_bulk0','llama3_b8_k4_2_bulk0','llama2_b8_k4_2_fused','llama3_b8_k4_2_fused']
choices=[n for n in choices if n in prof]
if choices:
 fig,axes=plt.subplots(len(choices),1,figsize=(11,2*len(choices)),squeeze=False)
 legend=[]
 for ax,name in zip(axes[:,0],choices):
  r=prof[name];ss=[s for s in r['steps'] if s['batch']==r['args']['batches'][0]]
  if not ss:continue
  median=np.median([s['p2_before_ready_ms'] for s in ss]);s=min(ss,key=lambda s:abs(s['p2_before_ready_ms']-median));t=s['timeline']
  for row,label,key,col in [(1,'Target pre','batch_target_pre','#176d8b'),(1,'Target post','batch_target_post','#6bb0c7'),(0,'Glue','batch_glue','#a8a8a8'),(0,'P1','batch_p1_total','#d37430'),(0,'P2','batch_p2_total','#e8b987')]:
   v=t[key];bar=ax.barh(row,v['end_ms']-v['start_ms'],left=v['start_ms'],height=.42,color=col,label=label)
   if len(legend)<5:legend.append(bar)
  ax.axvline(t['batch_proxy_receive']['end_ms'],c='#666',ls=':',label='Proxy received')
  ax.axvline(t['target_postprocess']['end_ms'],c='#333',ls='--',label='Target ready')
  ax.set_yticks([0,1],['Draft','Target']);ax.set_title(f'{name}: actual step {s["step"]}, active B={s["batch"]}, hit={s["hit_rate"]:.2f}',fontsize=10)
  ax.set_xlabel('ms from target pre-forward start');ax.grid(axis='x',alpha=.2)
 axes[0,0].legend(loc='upper left',bbox_to_anchor=(0,1.65),ncol=7,fontsize=8)
 fig.tight_layout();save(fig,'04_actual_step_timeline')

fig,axes=plt.subplots(1,2,figsize=(9,3.8))
for ax,model in zip(axes,['llama2','llama3']):
 names2=['baseline','fused','fused_bulk','fused_bulk_parallel'];xs=np.arange(len(names2))
 for i,name in enumerate(names2):
  r=lookup.get((model+'_optimized',name))
  if not r:continue
  for j,tps in enumerate(r['tps']):
   ax.bar(i+(j-.5)*.32,tps,width=.3,color=['#a1c7d5','#176d8b'][j],label=f'Pass {j+1}' if i==0 else None)
 ax.set_xticks(xs,['Base','Fused','+ bulk','+ parallel'],rotation=15);ax.set_title(model.replace('llama','Llama')+' / B=8');ax.set_ylabel('Returned-token TPS');ax.grid(axis='y',alpha=.2)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.96),ncol=2,fontsize=9)
fig.suptitle('Same tree algorithm; two target-seed passes in one engine',fontsize=11,y=1.02)
fig.tight_layout(rect=(0,0,1,.87));save(fig,'05_implementation_tps')
