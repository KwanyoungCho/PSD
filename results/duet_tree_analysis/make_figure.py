"""Export three distinct estimands; no speedup figure from instrumented runs."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import HERE

a=json.loads((HERE/'analysis.json').read_text())
b=json.loads((HERE/'rollout_analysis.json').read_text())
s=a['summary']['all']['prompt_balanced']
fig,axes=plt.subplots(1,3,figsize=(12,4.2),layout='constrained')
panels=[
    (['q-path','Calibrated\nreach'],[s['q_path_reach_mse'],s['phase_sibling_q_bin_reach_mse']],
     '(a) Held-out node prediction','Reach MSE (lower is better)'),
    (['q greedy','Cal. DP','Oracle DP'],[a['budget_diagnostics']['4']['prompt_balanced'][k] for k in ['q_greedy','phase_sibling_q_bin_dp','oracle_dp']],
     '(b) Fixed-pool subset, cap = 4','Conditional accepted tokens'),
    (['q-path','Calibrated\nreach'],[b[k]['prompt_balanced']['hit_al'] for k in ['baseline','calibrated']],
     '(c) Instrumented construction','Accepted descendants / hit'),
]
for ax,(labels,values,title,ylabel) in zip(axes,panels):
    bars=ax.bar(labels,values,color=['#496D89','#BD683D','#668767'][:len(values)],width=.58)
    ax.bar_label(bars,fmt='%.4f',padding=4,fontsize=10)
    ax.set_title(title,fontsize=11);ax.set_ylabel(ylabel,fontsize=10)
    ax.set_ylim(0,max(values)*1.2);ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('DUET tree audit: better prediction did not establish better construction',fontsize=13)
fig.supxlabel('16 prompts, one seed. Middle: diagnostic pruning only. Right: instrumented rollout, not TPS.',fontsize=10)
fig.savefig(HERE/'comparison.png',dpi=180)
fig.savefig(HERE/'comparison.pdf')

c=json.loads((HERE/'benchmark_analysis.json').read_text())['paired']['output_tps_relative']
point=c['point']*100;lo,hi=[x*100 for x in c['ci95']]
fig2,ax=plt.subplots(figsize=(7,2.5),layout='constrained')
ax.errorbar([point],[0],xerr=[[point-lo],[hi-point]],fmt='o',capsize=7,color='#496D89')
ax.axvline(0,color='gray',linestyle='--',linewidth=1)
ax.set_yticks([]);ax.set_ylim(-.6,.6)
ax.set_xlabel('Change in output TPS (%) — prefill included')
ax.set_title(f'Untraced confirmation: {point:+.2f}%  (95% CI {lo:+.2f}% to {hi:+.2f}%)',fontsize=11)
ax.spines[['top','right','left']].set_visible(False)
fig2.supxlabel('16 reused prompts × 2 seeds; prompt-cluster bootstrap; fixed policy',fontsize=9)
fig2.savefig(HERE/'benchmark_comparison.png',dpi=180)
fig2.savefig(HERE/'benchmark_comparison.pdf')
