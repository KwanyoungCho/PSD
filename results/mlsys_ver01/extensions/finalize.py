"""Reproduce the extension review tables and figures from preserved raw runs.

No profile/diagnostic TPS is evidence. Pure systems use matching-step latency;
only the mixed-miss policy uses online TPS, with group bootstrap uncertainty.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import analyze as a


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path(__file__).parent)
    args = p.parse_args()
    r, out = args.raw_root, args.output
    nova = r/'nova_latest'
    out.mkdir(parents=True, exist_ok=True)
    result = dict(scope='full dense TP1 7B/8B; fixed, not globally tuned parameters',
                  primary='direct latency; mixed-miss only uses TPS/AL',
                  profiles={}, ablations={}, sensitivity={}, prewarm={}, fixed_target={},
                  campaigns={}, checks=a.load(nova/'mlsys_ver01/extensions/CHECKS.json'))
    for model in ('llama2', 'llama3'):
        result['profiles'][model] = a.profile_compare(nova/'profiles_v2', model+'_base', model+'_optimized')
        result['ablations'][model] = {
            x+'->'+y:a.profile_compare(nova/'latency_ablation',model+'_'+x,model+'_'+y)
            for x,y in [('reference','fused'),('fused','bulk'),('bulk','trim')]}
        folder = r/'llama2_sensitivity' if model=='llama2' else nova/'llama3_sensitivity'
        result['sensitivity'][model]=a.sensitivity(folder)
        assert set(result['sensitivity'][model])=={'1','2','3'}, 'Wait for every sensitivity cell'
        folder=nova/('packed_prewarm' if model=='llama2' else 'packed_prewarm_llama3')
        lazy,warm=(a.load(folder/(n+'.json')) for n in ('lazy','prewarm'))
        assert lazy['status']==warm['status']=='complete'
        result['prewarm'][model]=dict(lazy=a.metrics(lazy),prewarm=a.metrics(warm),
            comparison=a.compare(lazy,warm),startup_seconds=warm['review_prewarm_seconds'],
            graphs=warm['review_prewarm_graphs'])
        path=r/('target_components/llama2_target.json' if model=='llama2' else 'component_llama3_4090_v2.json')
        target=a.load(path)
        # v2 did not yet emit a status field or account for uniform fallback.
        assert len(target['cells'])==30
        for c in target['cells']:
            if c['implementation']=='fixed':
                c['physical_queries_corrected']=min(c['batch']*9,8*((sum(c['lengths'])+c['batch']+7)//8))
            else:c['physical_queries_corrected']=c['physical_queries']
        result['fixed_target'][model]=target
    for name,folder in [('orion_llama2',r/'llama2_full_v2'),('nova_llama3',nova/'llama3_full_v2')]:
        result['campaigns'][name]=a.campaign(folder)
    result['expected']=a.mixed_expected(nova)
    result['fixed_accept']=a.load(r/'accept_components/fixed_accept.json')
    assert all(c['specialized_same_seed_equal'] for c in result['fixed_accept']['cells'])
    for c in result['fixed_accept']['cells']:
        c['means']={k:{metric:float(np.mean([v[metric] for v in values]))
                          for metric in ('cuda_ms','wall_ms')} for k,values in c['samples'].items()}
    before,after=(a.load(r/'b1_regression'/(n+'.json')) for n in ('before','after_native'))
    assert before['status']==after['status']=='complete'
    assert before['prompt_ids']==after['prompt_ids']
    old_events=[s['events'] for s in before['steps']]
    new_events=[s['events'] for s in after['steps']]
    keys=set(old_events[0][0])
    result['b1_regression']=dict(outputs_identical=before['outputs']==after['outputs'],
        events_identical=old_events==[[{k:e[k] for k in keys} for e in es] for es in new_events],
        added_event_fields=sorted(set(new_events[0][0])-keys),
        before_commit=before['paper_commit'],after_commit=after['paper_commit'],
        before=a.metrics(before),after=a.metrics(after),prompts=len(after['outputs']),
        scope='before uses historical dtype/resource adapter; after is native. Same prompts, settings, seeds and cold-prefix; functional parity only, not a performance claim')
    assert result['b1_regression']['outputs_identical'] and result['b1_regression']['events_identical']
    (out/'FINAL_ANALYSIS.json').write_text(json.dumps(result,indent=2))
    figures(result,out/'figs')
    report_tables(result,out/'TABLES.txt')
    print('Saved FINAL_ANALYSIS.json, TABLES.txt and figures')


def report_tables(d,path):
    lines=[]
    for model,c in d['profiles'].items():
        lines.append(model+' component latency (ms)')
        for name in ('draft/batch_p1_replay','draft/batch_p1_total','draft/batch_p2_replay','draft/batch_p2_total',
                     'target/batch_target_pre','target/batch_target_post','target/batch_target_proxy','target/batch_tree_accept'):
            x=c[name];lines.append(f"{name}: {x['before_ms']:.4f} -> {x['after_ms']:.4f} ({x['percent']:+.2f}%), n={x['steps']}")
    lines.append('\nMixed-miss: same parameters per pair; wait -> AR')
    for model,c in d['sensitivity'].items():
        for k,x in c.items():
            w,v,co=x['wait'],x['ar'],x['comparison'];ci=co['tps_percent_ci']
            lines.append(f"{model} K2={k}: TPS {w['tps']:.2f}->{v['tps']:.2f} ({co['tps_percent']:+.2f}%, CI[{ci[0]:+.2f},{ci[1]:+.2f}]); "
                f"AL {w['al']:.4f}->{v['al']:.4f}; hit {100*w['hit']:.2f}->{100*v['hit']:.2f}%; "
                f"raw TPS {w['raw_decode_tps']:.2f}->{v['raw_decode_tps']:.2f}; "
                f"retained {100*w['retained_step_fraction']:.1f}->{100*v['retained_step_fraction']:.1f}%")
    path.write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


def figures(d,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'white','savefig.facecolor':'white'})
    dest.mkdir(exist_ok=True)
    pdf=PdfPages(dest/'REVIEW.pdf')
    def save(fig,name):
        fig.savefig(dest/(name+'.png'),dpi=180,bbox_inches='tight')
        fig.savefig(dest/(name+'.pdf'),bbox_inches='tight')
        pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for i,model in enumerate(('llama2','llama3')):
        c=d['profiles'][model]
        for j,labels in enumerate([['draft/batch_p1_replay','draft/batch_p2_replay'],
                                   ['target/batch_target_proxy','target/batch_tree_accept']]):
            ax=axes[i,j];x=np.arange(2)
            for shift,field,color,label in [(-.18,'before_ms','#9ca3af','Before optimization'),(.18,'after_ms','#007c91','After optimization')]:
                bars=ax.bar(x+shift,[c[k][field] for k in labels],.35,color=color,label=label)
                ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
            ax.set_xticks(x,['Draft P1 replay','Draft P2 replay'] if j==0 else ['Target proxy','Target acceptance'])
            ax.set_ylabel('CUDA interval (ms)');ax.set_title(model.replace('llama','Llama ')+' / RTX 3090')
            ax.set_ylim(0,max(c[k]['before_ms'] for k in labels)*1.23)
            if i==0 and j==0:ax.legend(fontsize=8)
    fig.suptitle('Direct component latency: identical output and verification events\nB=8, T=0.7; 48 prompts; matched steps after capture exclusion',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.92));save(fig,'01_component_latency')
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    shapes=['uniform','half_short','one_hit']
    for ax,model in zip(axes,('llama2','llama3')):
        cells=d['fixed_target'][model]['cells'];x=np.arange(3)
        for off,name,color in [(-.24,'padded','#9ca3af'),(0,'replan','#e4a752'),(.24,'fixed','#007c91')]:
            vals=[next(c['wall_ms'] for c in cells if c['batch']==8 and c['shape']==s and c['implementation']==name and not c['stream']) for s in shapes]
            bars=ax.bar(x+off,vals,.23,color=color,label=name);ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=2)
        ax.set_xticks(x,['8 nodes each','4x8 + 4x2 nodes','1x8 + 7x0 nodes'])
        ax.set_title(model.replace('llama','Llama ')+' / RTX 4090');ax.set_ylabel('Target call wall latency (ms)')
        ax.set_ylim(0,max(c['wall_ms'] for c in cells if c['batch']==8)*1.2)
        ax.legend(fontsize=8,loc='lower left')
    fig.suptitle('Packed tree: same synthetic shape, real model + prefilled KV\nSteady replay, 20 repeats; includes proxy, excludes draft. Numerical outputs may differ.',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.87));save(fig,'02_packed_shapes')
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    for ax,model in zip(axes,('llama2','llama3')):
        cells=d['fixed_target'][model]['cells'];x=np.arange(3)
        for off,stream,color in [(-.18,False,'#9ca3af'),(.18,True,'#007c91')]:
            vals=[next(c['wall_ms'] for c in cells if c['batch']==8 and c['shape']==s and c['implementation']=='padded' and c['stream']==stream) for s in shapes]
            bars=ax.bar(x+off,vals,.35,color=color,label='Side stream' if stream else 'Main stream');ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=2)
        ax.set_xticks(x,['Uniform','Half short','One hit']);ax.set_ylabel('Target call wall latency (ms)')
        ax.set_title(model.replace('llama','Llama ')+' / RTX 4090');ax.legend(fontsize=8)
        ax.set_ylim(0,max(c['wall_ms'] for c in cells if c['batch']==8)*1.15)
    fig.suptitle('Batched proxy side stream: overlap does not ensure a shorter target call\nSame inputs; concurrent proxy / later layers contend for GPU resources',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.86));save(fig,'03_proxy_stream')
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for i,model in enumerate(('llama2','llama3')):
        cells=d['sensitivity'][model];ks=[1,2,3];co=[cells[str(k)]['comparison'] for k in ks]
        val=np.array([c['tps_percent'] for c in co]);ci=np.array([c['tps_percent_ci'] for c in co])
        axes[i,0].errorbar(ks,val,yerr=np.maximum(0,np.stack([val-ci[:,0],ci[:,1]-val])),fmt='o-',color='#007c91',capsize=5)
        axes[i,0].axhline(0,color='#666',ls='--',lw=1);axes[i,0].set_ylabel('AR vs wait TPS change (%)')
        for name,label,color in [('wait','Wait for miss draft','#777'),('ar','Mixed-miss AR','#007c91')]:
            axes[i,1].plot(ks,[cells[str(k)][name]['al'] for k in ks],'o-',label=label,color=color)
        axes[i,1].set_ylabel('Accepted length (AL)');axes[i,1].legend(fontsize=8)
        for ax in axes[i]:
            ax.set_xticks(ks);ax.set_xlabel('K2 (K1=4, exit index=21 fixed)');ax.set_title(model.replace('llama','Llama ')+(' / RTX 4090' if i==0 else ' / RTX 3090'))
    fig.suptitle('Mixed-miss policy sensitivity: TPS and AL together\n480 prompts per cell; paired 95% bootstrap CI over 60 prompt groups; not optimal tuning',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.91));save(fig,'04_mixed_miss')
    pdf.close()


if __name__=='__main__':
    main()
