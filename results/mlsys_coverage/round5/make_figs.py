"""Standalone diagnostic/confirmation figures; no profiled TPS as final speed."""
import json,argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from make_plans import HERE


def finish(fig,name):
    out=HERE/'figs';out.mkdir(exist_ok=True)
    fig.savefig(out/(name+'.png'),dpi=200,bbox_inches='tight')
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tuning-only',action='store_true');args=parser.parse_args()
    screen=json.loads((HERE/'SCREEN_RESULTS.json').read_text())
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for ax,(model,b) in zip(axes.flat,[('llama2',1),('llama2',8),('llama3',1),('llama3',8)]):
        rows=[r for r in screen if r['name'].startswith(f'{model}_b{b}_') and r.get('profile_steady_steps',0)>10
              and not r['timing_excluded'] and r['args']['limit']==0 and r['args']['max_new_tokens']==96]
        for mode,color,label in [('ssd','#4c78a8','SSD'),('duet-tree','#e45756','DUET')]:
            rs=[r for r in rows if r['args']['mode']==mode]
            ax.scatter([r['profile_steady_step_ms'] for r in rs],[r['profile_steady_al'] for r in rs],
                       s=25,alpha=.65,label=label,color=color)
        ax.set(title=f'{model.upper()}, B={b}',xlabel='Steady profiled step time (ms)',ylabel='AL on the same steady steps')
        ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Tuning48: observed AL / step-time tradeoff (diagnostic, instrumented)')
    finish(fig,'01_tuning_pareto')
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,(model,b) in zip(axes.flat,[('llama2',1),('llama2',8),('llama3',1),('llama3',8)]):
        rows=sorted([r for r in screen if r['name'].startswith(f'{model}_b{b}_duet_e')],key=lambda r:r['args']['exit_layer'])
        for key,label,color in [('p1_slack_ms','P1 before proxy','#4c78a8'),('p2_model_slack_ms','P2 before final logits','#f58518'),('p2_slack_ms','P2 before ready','#54a24b')]:
            ax.plot([r['args']['exit_layer'] for r in rows],[r['profile'][key]['p50'] for r in rows],marker='o',label=label,color=color)
        ax.axhline(0,color='black',lw=.8,ls='--');ax.grid(alpha=.2)
        ax.set(title=f'{model.upper()}, B={b}, K1/K2=4/2',xlabel='Exit layer (0-based)',ylabel='Median slack (ms)')
        ax.legend(fontsize=8)
    fig.suptitle('Anchor timelines: negative slack means the phase missed the boundary')
    finish(fig,'02_phase_deadlines')
    if args.tuning_only:return
    final=json.loads((HERE/'FINAL_RESULTS.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    rows=[r for r in final['comparisons'] if r['comparison']=='fast']
    for i,r in enumerate(rows):
        f=r['full480'];ci=f['delta_al_ci95'];d=f['delta_al'];label=f"{r['model'].upper()} B{r['batch']}"
        axes[0].errorbar(d,i,xerr=[[d-ci[0]],[ci[1]-d]],fmt='o',capsize=4,color='#e45756')
        for j,p in enumerate(r['throughput']):axes[1].scatter(p['ratio'],i+(j-.5)*.14,color=['#4c78a8','#f58518'][j],label=f'Paired repeat {j}' if i==0 else None)
    labels=[f"{r['model'].upper()} B{r['batch']}" for r in rows]
    for ax in axes:ax.set_yticks(range(len(rows)),labels);ax.grid(axis='x',alpha=.2)
    axes[0].axvline(0,color='black',lw=.8);axes[0].set(xlabel='DUET - SSD AL*',title='Full480, question bootstrap 95% CI')
    axes[1].axvline(1,color='black',lw=.8);axes[1].set(xlabel='DUET / SSD TPS*',title='Warm uninstrumented paired GPU repeats');axes[1].legend()
    fig.suptitle('Initial frozen TPS-priority settings, before implementation follow-ups')
    finish(fig,'03_frozen_full_comparison')
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for ax,r in zip(axes.flat,rows):
        groups=list(r['groups']);values=[r['groups'][g]['delta_al'] for g in groups]
        ci=np.array([r['groups'][g]['delta_al_ci95'] for g in groups])
        ax.errorbar(range(len(groups)),values,yerr=[np.array(values)-ci[:,0],ci[:,1]-np.array(values)],fmt='o',capsize=3)
        ax.axhline(0,color='black',lw=.8);ax.grid(axis='y',alpha=.2)
        ax.set_xticks(range(len(groups)),groups,rotation=25,ha='right')
        ax.set(title=f"{r['model'].upper()}, B={r['batch']}",ylabel='DUET - SSD AL*')
    fig.suptitle('Initial frozen settings: Full480 categories, 80 questions each; nominal 95% intervals')
    finish(fig,'04_category_al')
    follow=json.loads((HERE/'FOLLOWUP_RESULTS.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for ax,r in zip(axes.flat,rows):
        for rep,p in enumerate(r['throughput']):
            t=next(x for x in follow['ladder_trim'] if (x['model'],x['batch'],x['role'],x['replicate'])==
                   (r['model'],r['batch'],'duet_fast',rep))
            s=next(x for x in follow['proxy_stream'] if (x['model'],x['batch'],x['replicate'])==
                   (r['model'],r['batch'],rep))
            values=[p['ratio'],t['optimized']['boundary_excluded_step_tps']/p['ssd_tps'],
                    s['optimized']['boundary_excluded_step_tps']/p['ssd_tps']]
            ax.plot(range(3),values,'o-',label=f"Repeat {rep}, GPUs {p['gpus']}")
        ax.axhline(1,color='black',ls='--',lw=.8);ax.grid(alpha=.2)
        ax.set_xticks(range(3),['Base','Ladder trim','Trim + proxy stream'])
        ax.set(title=f"{r['model'].upper()}, B={r['batch']}",ylabel='DUET / SSD TPS*');ax.legend(fontsize=8)
    fig.suptitle('Original frozen settings, identical outputs: full480 implementation effects')
    finish(fig,'05_implementation_vs_ssd')
    breakdown=json.loads((HERE/'BREAKDOWN.json').read_text())
    byname={r['name']:r for r in breakdown}
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    colors={'batch_target_pre':'#4c78a8','batch_target_post':'#203b65','batch_target_proxy':'#f2a541',
            'batch_proxy_side':'#f2a541','batch_tree_accept':'#9d69aa','batch_tree_commit':'#ba8bbf',
            'batch_glue':'#8bc6bf','batch_p1_total':'#54a24b','batch_p2_total':'#e45756'}
    for ax,(model,b) in zip(axes.flat,[('llama2',1),('llama2',8),('llama3',1),('llama3',8)]):
        pair=[byname[f'{model}_b{b}_stream_l0_profile_s{s}']['representative_step'] for s in (0,1)]
        assert pair[0]['step']==pair[1]['step']
        for arm,step in enumerate(pair):
            timeline=step['timeline']
            for label,color in colors.items():
                if label not in timeline or (arm==1 and label=='batch_target_proxy'):continue
                e=timeline[label];draft=label in ('batch_glue','batch_p1_total','batch_p2_total')
                y=3-arm*2-int(draft)
                # Proxy work is a separate CUDA stream, drawn as a thin bar
                # above its target lane so actual overlap remains visible.
                height=.25 if label=='batch_proxy_side' else .45
                yy=y+.28 if label=='batch_proxy_side' else y-.2
                ax.broken_barh([(e['start_ms'],e['end_ms']-e['start_ms'])],(yy,height),facecolors=color)
            end=timeline['target_postprocess']['end_ms']
            ax.plot([end,end],[2-arm*2-.3,3-arm*2+.6],color='black',lw=.8,ls=':')
        ax.set_yticks(range(4),['Draft: overlap','Target: overlap','Draft: serial proxy','Target: serial proxy'])
        ax.set(title=f'{model.upper()} B{b}, matched step {pair[0]["step"]}',xlabel='Time from target pre start (ms)')
        ax.grid(axis='x',alpha=.2)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig.legend(handles=[Patch(color=c,label=n) for n,c in [('Target pre','#4c78a8'),('Target post','#203b65'),
        ('Proxy head/score','#f2a541'),('Draft glue','#8bc6bf'),('P1','#54a24b'),('P2','#e45756'),('Accept','#9d69aa'),('Commit','#ba8bbf')]]+
        [Line2D([0],[0],color='black',ls=':',label='Target ready')],loc='outside lower center',ncol=5)
    fig.suptitle('Matched real steps: proxy overlap after ladder trim\nOriginal frozen fast settings (instrumented)')
    finish(fig,'06_proxy_overlap_timeline')
    selected=json.loads((HERE/'POSTOPT_RESULTS.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for i,r in enumerate(selected):
        for j,(key,label,color) in enumerate([('full480','Full480','#e45756'),
                ('heldout432','Selection-excluded432','#4c78a8')]):
            f=r[key];d=f['delta_al'];ci=f['delta_al_ci95']
            axes[0].errorbar(d,i+(j-.5)*.18,xerr=[[d-ci[0]],[ci[1]-d]],fmt='o',capsize=4,
                            color=color,label=label if i==0 else None)
        for j,p in enumerate(r['throughput']):
            axes[1].scatter(p['ratio'],i+(j-.5)*.18,color=['#4c78a8','#f58518'][j],
                            label=f'Paired repeat {j}' if i==0 else None)
    labels=[f"{r['model'].upper()} B{r['batch']}" for r in selected]
    for ax in axes:
        ax.set_yticks(range(len(selected)),labels);ax.grid(axis='x',alpha=.2);ax.legend()
    axes[0].axvline(0,color='black',lw=.8)
    axes[0].set(xlabel='DUET - SSD AL*',title='Question bootstrap nominal 95% intervals')
    axes[1].axvline(1,color='black',lw=.8)
    axes[1].set(xlabel='DUET / SSD TPS*',title='Measured full480, same-GPU comparisons')
    fig.suptitle('Final TPS-priority points after implementation and deadline revalidation')
    finish(fig,'07_final_selected_comparison')

if __name__=='__main__':main()
