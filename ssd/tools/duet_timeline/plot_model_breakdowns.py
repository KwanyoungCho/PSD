#!/usr/bin/env python3
"""Offline diagnostics for the selected eslab18 model pairs.

Uses request-anchor-v2, never sums nested spans or visually shifts GPU clocks.
The source handoff is read-only. Figures describe its measured v1.0 settings.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/duet-diagnostics-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

SSD = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SSD/'bench'))
from duet_profile_analysis import VERSION, index_events, parse_events, require
from plot_duet_aligned_timeline import COLORS

NAMES = {'l2_7b':'LayerSkip Llama2-7B + llama-68M',
         'l2_13b':'LayerSkip Llama2-13B + AMD-Llama-135M',
         'l3_8b':'LayerSkip Llama3-8B + Axcel Llama3.1 drafter-340M',
         'code34b':'LayerSkip CodeLlama-34B + AMD-Llama-135M-code',
         'code34b_throughput':'CodeLlama-34B + AMD-135M-code / throughput alternative'}
STATES = {0:'Miss',1:'P1 hit',2:'P2 hit'}
STATE_COLOR = {0:'#c75435',1:'#277da8',2:'#2b945b'}
TARGET = [('target_wait','Request / cache wait'),('setup','Verify setup'),
          ('target_pre','Transformer before exit'),('target_post','Transformer after exit'),
          ('final_head','Final LM head'),('accept','Sampling / acceptance'),
          ('postprocess','Postprocess'),('proxy_side','Proxy side stream (overlaps)'),
          ('target_forward','Full target forward envelope')]
DRAFT = [('response','Cache response / miss draft'),('glue','Glue'),
         ('D1','DS total'),('ds_graph','DS graph replay'),
         ('ds_other','DS outside graph (elapsed)'),('proxy_wait_ms','Proxy receive wait'),
         ('D2','PS total'),('ps_graph','PS graph replay'),
         ('ps_other','PS outside graph (elapsed)')]
MARGIN_KEYS = [('gap1','DS vs estimated proxy arrival','#287ca4'),
               ('gap2_verify','PS vs final target head','#be5636'),
               ('gap2','PS vs next request','#469664')]


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def union_ms(events, start='wall_start_ns', end='wall_end_ns'):
    """Union rather than sum: enclosing/repeated helper labels may overlap."""
    intervals=sorted((int(e[start]),int(e[end])) for e in events)
    total=0; stop=None
    for a,b in intervals:
        require(b>=a, 'Negative event span')
        total += max(0,b-max(a,stop if stop is not None else a))
        stop=max(stop if stop is not None else b,b)
    return total/1e6


def group_events(events, anchor):
    """Assign all spans with complete request anchors, including outer parents.

    Parent scopes can start immediately before their request marker. Permit
    the next anchor only when it is enclosed by that CPU dispatch scope and
    the explicit step ID agrees. Never use a sparse label to count epochs.
    """
    _,epochs=index_events(events,anchor,{anchor})
    markers=sorted((e for e in events if e.get('label')==anchor),
                   key=lambda e:e['cpu_dispatch_start_ns'])
    starts=[e['cpu_dispatch_start_ns'] for e in markers]
    keys=[(ep,sid) for ep,sids in enumerate(epochs) for sid in sids]
    out=defaultdict(list)
    for e in events:
        if e.get('step_id') is None or e.get('label') in ('_anchor','draft_recv_cmd'):
            continue
        i=bisect_right(starts,e['cpu_dispatch_start_ns'])-1
        candidates=[]
        if i+1<len(keys) and keys[i+1][1]==int(e['step_id']) \
                and e['cpu_dispatch_start_ns']<=starts[i+1]<=e['cpu_dispatch_end_ns'] \
                and markers[i+1].get('parent_label')==e['label']:
            candidates.append(i+1)
        elif i>=0 and keys[i][1]==int(e['step_id']):candidates.append(i)
        require(len(candidates)==1,f"Cannot assign {e['label']} step {e['step_id']} to an anchor")
        require(e['wall_end_ns']>=e['wall_start_ns'],'Negative GPU interval')
        out[keys[candidates[0]]].append(e)
    return out


def by_label(events):
    out=defaultdict(list)
    for e in events:out[e['label']].append(e)
    return out


def one(events,label):
    found=[e for e in events if e['label']==label]
    require(len(found)==1,f'Expected one {label}, got {len(found)}')
    return found[0]


def percentile(values):
    a=np.asarray(values,dtype=float)
    return dict(n=len(a),mean=float(a.mean()),p05=float(np.quantile(a,.05)),
                p25=float(np.quantile(a,.25)),p50=float(np.median(a)),
                p75=float(np.quantile(a,.75)),p95=float(np.quantile(a,.95)),
                minimum=float(a.min()),maximum=float(a.max()))


def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def read_arm(source,key,selection):
    folder=source/'runs'/selection['profile']
    inputs=[folder/x for x in ('result.json','status.json','timing.json',
                              'physical_tree_shapes.json','process.log')]
    inputs+=sorted((folder/'profile').glob('*.json'))
    before={str(p):sha(p) for p in inputs}
    result=load(folder/'result.json');status=load(folder/'status.json')
    require(status['status']=='complete' and status['returncode']==0,'Incomplete source run')
    require(result['status']=='complete' and result['source_unchanged'],'Source changed during measurement')
    profiles={}
    for role in ('target_rank0','draft'):
        paths=list((folder/'profile').glob('duet_profile_'+role+'_*.json'))
        require(len(paths)==1,'Ambiguous profile file selection')
        profiles[role]=load(paths[0])
    rows,audit=parse_events(profiles['draft'],profiles['target_rank0'],result['records'],result['config'])
    original=load(folder/'timing.json')
    require(rows==original['steps'],'Recomputed timing differs from frozen source')
    require(audit==original['audit'],'Recomputed audit differs from frozen source')
    grouped={'target':group_events(profiles['target_rank0'],'target_send_request'),
             'draft':group_events(profiles['draft'],'draft_recv_request')}
    active=[r for r in result['records'] if r['metrics']['phase_events']]
    outcomes={(epoch,s['step_id']):s for epoch,r in enumerate(active,1)
              for s in r['metrics']['phase_events']}
    for row in rows:
        pair=(row['epoch'],row['step_id']); t=grouped['target'][pair]; d=grouped['draft'][pair]
        td=by_label(t); dd=by_label(d)
        row.update({k:union_ms(td[label]) for k,label in (
            ('target_wait','target_spec_wait'),('setup','verify_setup'),
            ('target_pre','graph_pre'),('target_post','graph_post'),
            ('final_head','final_logits'),('accept','verify_sample_accept'),
            ('postprocess','target_postprocess'),('proxy_side','exit_proxy_side'))})
        row['target_forward']=(one(t,'final_logits')['wall_end_ns']-one(t,'graph_pre')['wall_start_ns'])/1e6
        row['target_step_span']=(one(t,'target_postprocess')['wall_end_ns']-one(t,'target_send_request')['wall_start_ns'])/1e6
        row['response']=union_ms([e for e in d if e['label'].startswith('hit_cache_respond_')])
        row['glue']=union_ms(dd['glue'])
        row['ds_graph']=union_ms(dd['p1_graph_replay']);row['ps_graph']=union_ms(dd['p2_graph_replay'])
        row['ds_other']=row['D1']-row['ds_graph'];row['ps_other']=row['D2']-row['ps_graph']
        require(min(row['ds_other'],row['ps_other'])>=-1e-6,'Graph exceeds phase envelope')
        row['valid_k']=outcomes[pair]['valid_k'];row['accepted_len']=outcomes[pair]['accepted_len']
    runtime=[json.loads(line.split(' ',1)[1]) for line in (folder/'process.log').read_text().splitlines()
             if line.startswith('[runtime-manifest] ')]
    model_meta=next(x for x in runtime if x['role']=='target' and x['rank']==0)
    return dict(key=key,name=NAMES[key],selection=selection,folder=folder,result=result,
                rows=rows,audit=audit,grouped=grouped,input_hashes=before,
                physical_shapes=load(folder/'physical_tree_shapes.json'),runtime=model_meta,
                frozen_timing_equal=True)


def subtitle(arm):
    c=arm['result']['config'];r=arm['result']['resolved_config']
    return (f"exit {c['exit']} (0-based) | K1/K2 {c['k1']}/{c['k2']} | "
            f"G=M DS/PS {r['duet_p1_tree_verify_nodes']}/{r['duet_p2_tree_verify_nodes']} | "
            f"C={r['duet_tree_c_tensor']} | {r['duet_tree_expansion']} | "
            f"target GPUs={c['tp']} + draft GPU=1 | B=1, BF16, T=0.7")


def decorate(fig,arm,title,footer=None):
    fig.suptitle(arm['name']+'\n'+title,fontsize=16,fontweight='bold',y=.985)
    fig.text(.5,.927,subtitle(arm),ha='center',fontsize=9)
    fig.text(.5,.904,f"Measured v1.0 / {arm['result']['commit'][:8]} / eslab18 RTX3090 / "
             f"{arm['audit']['requests']} profiled turns, {len(arm['rows'])} eligible steps / output cap 256",
             ha='center',fontsize=9,color='#555555')
    fig.text(.02,.014,footer or 'CUDA-event elapsed spans include queue/host stalls; not GPU busy time. Nested/overlapping components are NOT additive.',
             fontsize=8,color='#555555')


def savefig(fig,out,name,pdf,allpdf):
    fig.savefig(out/(name+'.png'),dpi=145,facecolor='white')
    pdf.savefig(fig);allpdf.savefig(fig);plt.close(fig)


def bar_distribution(ax,rows,fields,title):
    y=np.arange(len(fields)); stats=[percentile([r[k] for r in rows]) for k,_ in fields]
    p50=[s['p50'] for s in stats];p95=[s['p95'] for s in stats]
    ax.barh(y,p50,color='#527da2',alpha=.85,label='p50')
    ax.scatter(p95,y,marker='|',s=130,c='#b63f38',label='p95',zorder=3)
    limit=max(p95)*1.28
    for iy,s in enumerate(stats):
        ax.text(limit*.99,iy,f"{s['p50']:.2f} / {s['p95']:.2f}",va='center',ha='right',fontsize=8)
    ax.set_yticks(y,[v for _,v in fields],fontsize=9);ax.invert_yaxis();ax.set_xlim(0,limit)
    ax.set_title(title,loc='left',fontsize=12)
    ax.set_xlabel('ms — bars: p50; red ticks: p95; right: p50 / p95');ax.grid(axis='x',alpha=.18)


def dashboard(arm,out,pdf,allpdf):
    rows=arm['rows'];fig,axes=plt.subplots(2,2,figsize=(18,12))
    decorate(fig,arm,'Component latency, phase deadlines, and every measured step')
    bar_distribution(axes[0,0],rows,TARGET,'Target / rank 0 — independent elapsed spans')
    bar_distribution(axes[0,1],rows,DRAFT,'Draft — phase envelope and graph replay')
    ax=axes[1,0]
    for key,label,color in MARGIN_KEYS:
        a=np.sort([r[key] for r in rows if r[key] is not None])
        if len(a):ax.plot(a,np.arange(1,len(a)+1)/len(a),label=label,color=color,lw=2)
    ax.axvline(0,color='black',ls='--',lw=1);ax.axvspan(-1,1,color='gray',alpha=.1)
    ax.set(title='Deadline margins — negative means late',xlabel='margin (ms); ±1 ms shaded for clock uncertainty',ylabel='Empirical cumulative fraction')
    ax.legend(fontsize=9);ax.grid(alpha=.2)
    ax=axes[1,1]
    for k,label,color in [('cycle','Target pre → next target pre','#515765'),('D1','DS elapsed','#277da8'),('D2','PS elapsed','#2b945b')]:
        ax.plot([r[k] for r in rows],label=label,color=color,alpha=.8,lw=.9)
    epochs=[r['epoch'] for r in rows]
    for i in range(1,len(rows)):
        if epochs[i]!=epochs[i-1]:ax.axvline(i,color='gray',lw=.7,ls=':')
    ax.set(title='All eligible steps (request boundaries dotted)',xlabel='Eligible step index',ylabel='ms');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.subplots_adjust(left=.18,right=.98,top=.855,bottom=.08,wspace=.62,hspace=.36)
    savefig(fig,out,'01_components_and_deadlines',pdf,allpdf)


def source_breakdown(arm,out,pdf,allpdf):
    fig,axes=plt.subplots(2,2,figsize=(18,12));rows=arm['rows']
    decorate(fig,arm,'P1 hit / P2 hit / miss — component distributions and phase composition')
    for ax,fields,title in ((axes[0,0],TARGET[:-1],'Target p50 by current cache outcome'),
                           (axes[0,1],DRAFT,'Draft p50 by current cache outcome')):
        for state in (1,2,0):
            group=[r for r in rows if r['state']==state]
            if not group:continue
            ax.plot([np.median([r[k] for r in group]) for k,_ in fields],np.arange(len(fields)),
                    marker='o',color=STATE_COLOR[state],label=f'{STATES[state]} n={len(group)}')
        ax.set_yticks(np.arange(len(fields)),[label for _,label in fields],fontsize=9);ax.invert_yaxis()
        ax.set(title=title,xlabel='ms; components overlap, do not sum');ax.grid(alpha=.18);ax.legend(fontsize=8)
    ax=axes[1,0]; y=0
    for state in (1,2,0):
        group=[r for r in rows if r['state']==state]
        if not group:continue
        for phase in ('ds','ps'):
            graph=np.mean([r[phase+'_graph'] for r in group]);other=np.mean([r[phase+'_other'] for r in group])
            ax.barh(y,graph,color='#356b96',label='Graph replay' if y==0 else None)
            ax.barh(y,other,left=graph,color='#d6a64c',label='Other elapsed within phase' if y==0 else None)
            ax.text(graph+other+.25,y,f'{graph+other:.2f}',va='center',fontsize=9)
            ax.text(-.02,y,f'{STATES[state]} / {phase.upper()}',ha='right',va='center',transform=ax.get_yaxis_transform(),fontsize=9)
            y+=1
    ax.set_yticks([]);ax.invert_yaxis();ax.margins(x=.2);ax.set(title='Draft phase mean: exclusive graph + remainder',xlabel='ms; remainder is NOT pure CPU time')
    ax.legend(fontsize=8);ax.grid(axis='x',alpha=.18)
    ax=axes[1,1]
    for state in (1,2,0):
        group=[r for r in rows if r['state']==state]
        if group:ax.scatter([r['context'] for r in group],[r['cycle'] for r in group],s=12,alpha=.55,color=STATE_COLOR[state],label=STATES[state])
    ax.set(title='Context length vs next-target-start interval',xlabel='Context tokens before verification',ylabel='ms');ax.grid(alpha=.18);ax.legend(fontsize=9)
    fig.subplots_adjust(left=.18,right=.98,top=.855,bottom=.08,wspace=.62,hspace=.36)
    savefig(fig,out,'02_cache_source_breakdown',pdf,allpdf)


def representatives(rows):
    result=[]
    for state in (1,2,0):
        group=sorted((r for r in rows if r['state']==state),key=lambda r:(r['target_step_span'],r['epoch'],r['step_id']))
        used=set()
        for q in (.25,.5,.75):
            if not group:continue
            row=group[round(q*(len(group)-1))];key=(row['epoch'],row['step_id'])
            if key in used:continue
            used.add(key);result.append((state,q,row))
    return result


def draw_events(ax,events,previous,origin,xlim,title,markers):
    labels=list(dict.fromkeys(e['label'] for e in sorted(events,key=lambda e:e['cpu_dispatch_start_ns'])))
    if previous:labels=['Previous step: '+e['label'] for e in previous]+labels
    labels=list(dict.fromkeys(labels));positions={label:i for i,label in enumerate(labels)}
    for e,is_previous in [(e,True) for e in previous]+[(e,False) for e in events]:
        label=('Previous step: ' if is_previous else '')+e['label'];y=positions[label]
        start=(e['wall_start_ns']-origin)/1e6;end=(e['wall_end_ns']-origin)/1e6
        color=COLORS.get(e['label'],('#6a8d92',''))[0]
        ax.broken_barh([(start,end-start)],(y-.32,.48),facecolors=color,alpha=.45 if is_previous else .85)
        if end-start>.5:ax.text((start+end)/2,y-.08,f'{end-start:.2f}',ha='center',va='center',fontsize=7)
        a=(e['cpu_dispatch_start_ns']-origin)/1e6;b=(e['cpu_dispatch_end_ns']-origin)/1e6
        ax.hlines(y+.25,a,b,color='black',lw=1.5,alpha=.6)
    for value,label,color in markers:
        ax.axvline(value,color=color,ls='--',lw=.85,alpha=.7,label=label)
    ax.set_yticks(range(len(labels)),labels,fontsize=7.5);ax.set_ylim(len(labels)-.4,-.8)
    ax.set_xlim(*xlim);ax.set_xlabel('ms from current target pre-exit start');ax.set_title(title,loc='left',fontsize=11)
    ax.grid(axis='x',alpha=.17)


def timeline(arm,row,state,q,out,pdf,allpdf):
    pair=(row['epoch'],row['step_id']); t=arm['grouped']['target'][pair];d=arm['grouped']['draft'][pair]
    nt=arm['grouped']['target'][(row['epoch'],row['step_id']+1)]
    origin=one(t,'graph_pre')['wall_start_ns']
    lo=(one(t,'target_send_request')['wall_start_ns']-origin)/1e6
    hi=max(one(d,'merge_cache')['wall_end_ns'],one(nt,'graph_pre')['wall_start_ns'])
    hi=(hi-origin)/1e6
    previous=[e for e in arm['grouped']['draft'].get((row['epoch'],row['step_id']-1),[])
              if e['label'] in ('p2_graph_replay','p2_cache_merge','merge_cache') and
              e['wall_end_ns']>origin+lo*1e6]
    marks=[(0,'Target pre start','#555555'),
           ((one(t,'graph_pre')['wall_end_ns']-origin)/1e6,'Exit layer complete','#b58227'),
           ((one(t,'final_logits')['wall_end_ns']-origin)/1e6,'Final head complete','#d04b39'),
           ((one(nt,'target_send_request')['wall_start_ns']-origin)/1e6,'Next request','#8951a5')]
    fig,axes=plt.subplots(1,2,figsize=(24,14))
    decorate(fig,arm,f"{STATES[state]} / p{round(q*100)} target-step duration / request {row['epoch']}, UID {row['uid']}, step {row['step_id']}",
             'Colored bars: raw CUDA-event elapsed spans. Black lines below: CPU dispatch. No clock shift / no minimum bar-width inflation. Nested rows are NOT additive.')
    margins=f"DS margin {row['gap1']:+.3f} ms (estimated); PS margin {row['gap2_verify']:+.3f} ms; next outcome {STATES[row['next_state']]}"
    fig.text(.5,.88,f"valid proposal nodes={row['valid_k']}, AL incl. recovery={row['accepted_len']}; {margins}",ha='center',fontsize=10)
    draw_events(axes[0],t,[],origin,(lo-.5,hi+.8),'Target rank 0 — includes proxy side stream',marks)
    draw_events(axes[1],d,previous,origin,(lo-.5,hi+.8),'Draft — response, glue, DS, proxy wait, PS',marks)
    axes[0].legend(loc='lower right',fontsize=8)
    fig.subplots_adjust(left=.14,right=.985,top=.84,bottom=.07,wspace=.55)
    name=f"timeline_{state}_p{round(q*100):02d}_req{row['epoch']}_step{row['step_id']}"
    savefig(fig,out,name,pdf,allpdf)
    return dict(state=state,status=STATES[state],quantile=q,uid=row['uid'],epoch=row['epoch'],step_id=row['step_id'],
                target_step_span_ms=row['target_step_span'],next_state=row['next_state'],image=name+'.png')


def export_tables(arm,out):
    rows=arm['rows'];simple=[{k:v for k,v in r.items() if k!='config'} for r in rows]
    write_csv(out/'steps.csv',simple)
    fields=TARGET+DRAFT+[(k,l) for k,l,_ in MARGIN_KEYS]+[('cycle','Target start interval')]
    stats=[]
    for state in ('all',1,2,0):
        group=rows if state=='all' else [r for r in rows if r['state']==state]
        if not group:continue
        for key,label in fields:
            values=[r[key] for r in group if r[key] is not None]
            if values:stats.append(dict(state=state,component=key,label=label,**percentile(values)))
    write_csv(out/'component_statistics.csv',stats)
    intervals=[];detailed=[]
    for role,groups in arm['grouped'].items():
        labels=set(e['label'] for r in rows for e in groups[(r['epoch'],r['step_id'])])
        for state in (1,2,0):
            selected=[r for r in rows if r['state']==state]
            for label in sorted(labels):
                values=[];cpu=[]
                for row in selected:
                    events=[e for e in groups[(row['epoch'],row['step_id'])] if e['label']==label]
                    if events:
                        values.append(union_ms(events));cpu.append(union_ms(events,'cpu_dispatch_start_ns','cpu_dispatch_end_ns'))
                if values:detailed.append(dict(role=role,state=state,label=label,eligible=len(selected),present=len(values),
                    gpu_mean=float(np.mean(values)),gpu_p50=float(np.median(values)),gpu_p95=float(np.quantile(values,.95)),
                    cpu_mean=float(np.mean(cpu)),cpu_p50=float(np.median(cpu)),cpu_p95=float(np.quantile(cpu,.95))))
        for row in rows:
            pair=(row['epoch'],row['step_id']);origin=one(arm['grouped']['target'][pair],'graph_pre')['wall_start_ns']
            for e in groups[pair]:
                intervals.append(dict(role=role,epoch=pair[0],step_id=pair[1],uid=row['uid'],state=row['state'],
                    label=e['label'],parent=e.get('parent_label'),gpu_start_ms=(e['wall_start_ns']-origin)/1e6,
                    gpu_end_ms=(e['wall_end_ns']-origin)/1e6,cpu_start_ms=(e['cpu_dispatch_start_ns']-origin)/1e6,
                    cpu_end_ms=(e['cpu_dispatch_end_ns']-origin)/1e6))
    write_csv(out/'all_label_statistics.csv',detailed);write_csv(out/'event_intervals.csv',intervals)
    parameters=dict(config=arm['result']['config'],kwargs=arm['result']['kwargs'],
                    resolved_config=arm['result']['resolved_config'],physical_tree_shapes=arm['physical_shapes'])
    (out/'PARAMETERS.json').write_text(json.dumps(parameters,indent=2)+'\n')
    return stats


def overview(arms,out,allpdf):
    fig,axes=plt.subplots(1,3,figsize=(21,7)); y=np.arange(len(arms))
    labels=[a['name'].replace('LayerSkip ','') for a in arms]
    fig.suptitle('Model-pair latency diagnostics — selected measured v1.0 settings',fontsize=17,fontweight='bold')
    for key,label,delta in [('target_pre','Target pre',-.27),('target_post','Target post',-.09),('D1','DS',.09),('D2','PS',.27)]:
        axes[0].barh(y+delta,[np.median([r[key] for r in a['rows']]) for a in arms],height=.16,label=label)
    axes[0].set_yticks(y,labels,fontsize=9);axes[0].set(title='Component p50 (independent)',xlabel='ms');axes[0].legend(fontsize=9)
    for ax,key,title in [(axes[1],'gap1','DS vs estimated proxy arrival'),(axes[2],'gap2_verify','PS vs final target head')]:
        vals=[percentile([r[key] for r in a['rows']]) for a in arms]
        med=np.array([v['p50'] for v in vals]);low=np.array([v['p05'] for v in vals]);high=np.array([v['p95'] for v in vals])
        ax.errorbar(med,y,xerr=[med-low,high-med],fmt='o',capsize=4,color='#356b96')
        ax.axvline(0,color='red',ls='--',lw=1);ax.set_yticks(y,[]);ax.set(title=title+'\np50 and p05–p95',xlabel='margin (ms); negative = late')
    for ax in axes:ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.text(.02,.025,'eslab18 RTX3090; B1 BF16 T0.7. Four profiled turns per setting. No model inference rerun; not ver1.1 latency. Target GPU counts differ.',fontsize=9)
    fig.subplots_adjust(left=.25,right=.98,bottom=.12,top=.80,wspace=.20)
    fig.savefig(out/'OVERVIEW.png',dpi=150);allpdf.savefig(fig);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();source=args.source.resolve();out=args.output.resolve()
    require(out!=source and source not in out.parents,'Keep output outside the read-only source handoff')
    out.mkdir(parents=True,exist_ok=True)
    selected=load(source/'selected.json'); selections=list(selected.items())
    selections.append(('code34b_throughput',selected['code34b']['throughput_alternative']))
    arms=[read_arm(source,k,s) for k,s in selections]
    allmeta=[]
    with PdfPages(out/'ALL_MODELS_DETAILED.pdf') as allpdf:
        overview(arms,out,allpdf)
        for arm in arms:
            dest=out/arm['key'];dest.mkdir(exist_ok=True)
            with PdfPages(dest/'DETAILED.pdf') as pdf:
                dashboard(arm,dest,pdf,allpdf);source_breakdown(arm,dest,pdf,allpdf)
                reps=[timeline(arm,row,state,q,dest,pdf,allpdf) for state,q,row in representatives(arm['rows'])]
            write_csv(dest/'representatives.csv',reps);stats=export_tables(arm,dest)
            audit={**arm['audit'],'frozen_timing_equal':True,'source_hashes':arm['input_hashes'],
                   'source_counts':dict(Counter(r['state'] for r in arm['rows'])),
                   'representative_count':len(reps),'clock_shift_ms':0,
                   'selection_criterion':'p25/p50/p75 of current target request-to-postprocess elapsed, within cache source',
                   'unprofiled_target_ranks':'Only target rank0 was recorded; other target ranks not synthesized',
                   'within_phase_remainder':'DS/PS envelope minus replay; elapsed stalls/host work, not pure CPU time'}
            (dest/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
            allmeta.append(dict(key=arm['key'],name=arm['name'],profile=arm['folder'].name,
                runtime_commit=arm['result']['commit'],config=arm['result']['config'],parameters_path=arm['key']+'/PARAMETERS.json',
                pdf=arm['key']+'/DETAILED.pdf',audit=audit,statistics=stats,representatives=reps))
            print(arm['key'],len(arm['rows']),'eligible steps;',len(reps),'timelines',flush=True)
    for arm in arms:
        for path,checksum in arm['input_hashes'].items():require(sha(Path(path))==checksum,'Input changed while plotting')
    summary=dict(status='complete',parser=VERSION,source=str(source),source_selected_sha256=sha(source/'selected.json'),
                 plotting_tool_sha256=sha(Path(__file__)),parser_sha256=sha(SSD/'bench/duet_profile_analysis.py'),
                 measured_version='v1.0 with documented model/resource support commits',
                 analysis_branch='MLSys-ver1.1',new_inference_run=False,source_unchanged=True,models=allmeta)
    (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    cards=[]
    for a in allmeta:
        links=' '.join(f'<a href="{a["key"]}/{r["image"]}">{r["status"]} p{round(r["quantile"]*100)} (req{r["epoch"]}/step{r["step_id"]})</a>' for r in a['representatives'])
        cards.append(f'<section><h2>{html.escape(a["name"])}</h2><p>{html.escape(a["profile"])} | n={a["audit"]["rows"]} | '+
            f'<a href="{a["pdf"]}">Detailed PDF</a> · <a href="{a["key"]}/PARAMETERS.json">Exact parameters</a> · '+
            f'<a href="{a["key"]}/component_statistics.csv">Component CSV</a> · <a href="{a["key"]}/all_label_statistics.csv">All labels / CPU &amp; GPU CSV</a> · '+
            f'<a href="{a["key"]}/event_intervals.csv">Every measured event CSV</a></p>'+
            f'<a href="{a["key"]}/01_components_and_deadlines.png"><img src="{a["key"]}/01_components_and_deadlines.png"></a>'+
            f'<a href="{a["key"]}/02_cache_source_breakdown.png"><img src="{a["key"]}/02_cache_source_breakdown.png"></a><p class="links">{links}</p></section>')
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>DUET model breakdown diagnostics</title>
<style>body{font:16px system-ui;max-width:1600px;margin:36px auto;padding:0 24px;color:#20303a;background:#f5f7f9}section{background:white;padding:24px;margin:30px 0;border:1px solid #d7e0e4;border-radius:8px}img{width:100%;height:auto}a{color:#175f9c}.links a{display:inline-block;margin:8px 12px 8px 0}p{line-height:1.7}</style>
<h1>모델 조합별 상세 latency breakdown</h1><p>18번 서버에서 최종 선택한 <b>ver1.0 실측 설정</b>을 ver1.1 작업 트리에서 재분석했습니다. 새 모델 실행이나 ver1.1 성능 측정이 아닙니다. 각 설정은 profile용 4턴·출력 상한256이며, full dataset/전역 최적값으로 해석하지 않습니다.</p>
<p><b>읽는 순서:</b> 모델별 PDF → component 분포 → cache 상태별 비교 → 대표 step의 원시 시간축. 파란색·초록색 DS/PS와 target을 같은 시간축에서 확인합니다. Timeline의 색 막대는 CUDA-event elapsed, 검은 선은 CPU dispatch이며, 부모·자식 구간은 더하면 안 됩니다. CPU/GPU event 구간이 GPU busy time 자체를 뜻하지 않습니다.</p>
<p>Warmup·요청별 첫2단계·마지막 단계 제외. 긴 step 임의 제거 없음. 서로 다른 요청의 step ID를 섞지 않았고 draft 시계를 인위적으로 이동하지 않았습니다. DS arrival은 추정이며 ±1ms 차이는 clock 불확실성에 유의합니다. Code34B 12/6은 유효 miss 표본0개, 16/6은2개뿐입니다.</p>
<p><a href="ALL_MODELS_DETAILED.pdf">전체 상세 PDF</a> · <a href="SUMMARY.json">수치/근거 JSON</a> · <a href="REPORT.md">설명 문서</a></p><img src="OVERVIEW.png">'''+''.join(cards)+'</html>'
    (out/'index.html').write_text(page)


if __name__=='__main__':main()
