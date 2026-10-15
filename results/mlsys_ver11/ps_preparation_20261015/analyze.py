"""Native event comparisons: no duration-based censoring, no summed overlaps."""
import argparse
from bisect import bisect_right
from collections import defaultdict,Counter
import hashlib,json
from pathlib import Path
import numpy as np

COMPONENTS=['build','dispatch','prepare1','prepare2','graph','other','ps_total','target_forward','p1_graph']
NAMES={'build':'PS root/layout build','dispatch':'Slot derivation + validity dispatch',
       'prepare1':'PS root metadata preparation','prepare2':'Executor input buffer preparation',
       'graph':'PS CUDA graph (all forwards + selection)','other':'Other PS gaps + merge',
       'ps_total':'PS total (proxy received to cache ready)','target_forward':'Target pre + post + final logits',
       'p1_graph':'DS CUDA graph'}


def load_profile(folder,proc):
    rows=[]
    files=sorted(folder.glob('duet_profile_'+proc+'_*.json'))
    assert files,(folder,proc)
    for f in files:
        rows.extend(e for e in json.loads(f.read_text()) if e['label']!='_anchor')
    return sorted(rows,key=lambda e:e['cpu_dispatch_start_ns'])


def group_events(events,anchor):
    anchors=[e for e in events if e['label']==anchor]
    starts=[e['cpu_dispatch_start_ns'] for e in anchors]
    assert len(set(starts))==len(starts)
    keys=[];epoch=-1;last=0;epochs=[]
    for e in anchors:
        sid=e['step_id']
        if sid<=last:
            assert sid==1,(anchor,sid)
            epoch+=1;epochs.append([])
        elif epoch<0:
            assert sid==1;epoch=0;epochs.append([])
        assert sid==len(epochs[-1])+1,(anchor,epoch,sid)
        epochs[-1].append(sid);keys.append((epoch,sid));last=sid
    groups=defaultdict(lambda:defaultdict(list))
    for e in events:
        i=bisect_right(starts,e['cpu_dispatch_start_ns'])-1
        if i>=0:groups[keys[i]][e['label']].append(e)
    return groups,epochs


def read_arm(folder):
    run=json.loads((folder/'run.json').read_text()); assert run['status']=='complete'
    draft,de=group_events(load_profile(folder,'draft'),'draft_recv_request')
    target,te=group_events(load_profile(folder,'target_rank0'),'target_send_request')
    groups=run['warm_groups']+run['groups']
    expected=[list(range(1,len(g['steps'])+1)) for g in groups]
    assert de==te==expected,('Incomplete native coverage',len(de),len(te),len(expected))
    def start(e):return e['wall_start_ns']/1e6
    def end(e):return e['wall_end_ns']/1e6
    def dur(e):return (e['wall_end_ns']-e['wall_start_ns'])/1e6
    def one(events,label):
        es=events[label];assert len(es)==1,(folder,label,len(es));return es[0]
    rows=[];skip=Counter()
    for epoch,g in enumerate(groups):
        records=g['request_records'];assert len(records)==1
        meta=records[0]
        for i,step in enumerate(g['steps']):
            sid=i+1
            if g['phase']=='warm':skip['warm']+=1;continue
            if i<2:skip['first_two']+=1;continue
            if i==len(g['steps'])-1:skip['terminal']+=1;continue
            if step['cap'] or step['clipped']:skip['cap_clip']+=1;continue
            d=draft[epoch,sid];t=target[epoch,sid]
            build=one(d,'phase2_build');wait=one(d,'proxy_wait');merge=one(d,'merge_cache')
            preps=sorted(d['p2_prepare'],key=start);assert len(preps)==2,(epoch,sid,len(preps))
            graph=one(d,'p2_graph_replay')
            assert end(build)<=start(preps[0])+.005
            assert end(preps[0])<=start(preps[1])+.005
            assert end(preps[1])<=start(graph)+.005
            assert end(graph)<=end(merge)+.005
            ps=end(merge)-end(wait)
            row=dict(epoch=epoch,step_id=sid,uid=meta['paper_uid'],question_id=meta['question_id'],
                     source=step['events'][0]['source'],build=dur(build),
                     dispatch=start(preps[0])-end(build),prepare1=dur(preps[0]),prepare2=dur(preps[1]),
                     graph=dur(graph),ps_total=ps,
                     p1_graph=dur(one(d,'p1_graph_replay')),
                     target_forward=sum(dur(one(t,label)) for label in ('graph_pre','graph_post','final_logits')))
            row['other']=ps-sum(row[k] for k in ('build','dispatch','prepare1','prepare2','graph'))
            assert row['other']>=-.01
            # Host dispatch is a second observation, not added to GPU intervals.
            row['dispatch_cpu']=(preps[0]['cpu_dispatch_start_ns']-build['cpu_dispatch_end_ns'])/1e6
            rows.append(row)
    return run,rows,dict(skipped=dict(skip),total_steps=sum(map(len,de)),measured_steps=len(rows),
                         covered_requests=len(de),full_event_coverage=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reps',type=int,nargs='+',default=[1,2,3]);a=p.parse_args()
    result=dict(pairs=[],components={},per_source={});pooled=[];means={}
    for rep in a.reps:
        b,br,ba=read_arm(a.raw/f'before_graph{rep}');c,cr,ca=read_arm(a.raw/f'after_graph{rep}')
        assert b['outputs']==c['outputs'],'Output mismatch'
        assert b['kwargs']==c['kwargs']
        assert len(br)==len(cr)
        differences=[]
        for x,y in zip(br,cr):
            for k in ('epoch','step_id','uid','source'):assert x[k]==y[k],k
            differences.append(dict(question_id=x['question_id'],uid=x['uid'],source=x['source'],rep=rep,
                before={k:x[k] for k in COMPONENTS},after={k:y[k] for k in COMPONENTS}))
        pooled.extend(differences)
        pair=dict(rep=rep,output_exact=len(b['outputs']),args=b['args'],kwargs=b['kwargs'],before=ba,after=ca,
            breakdown_ms={k:dict(before=float(np.mean([x[k] for x in br])),after=float(np.mean([x[k] for x in cr]))) for k in COMPONENTS},
            algorithms_same=dict(al_before=b['summary']['al'],al_after=c['summary']['al'],hit_before=b['summary']['hit'],hit_after=c['summary']['hit']))
        result['pairs'].append(pair)
    # Cluster paired mean differences by input question, retaining all steps/repeats.
    rng=np.random.default_rng(721)
    questions=sorted({r['question_id'] for r in pooled})
    boot=rng.integers(0,len(questions),size=(10000,len(questions)))
    for k in COMPONENTS:
        old=np.array([r['before'][k] for r in pooled]);new=np.array([r['after'][k] for r in pooled])
        sums=[];counts=[]
        for q in questions:
            values=[r['after'][k]-r['before'][k] for r in pooled if r['question_id']==q]
            sums.append(sum(values));counts.append(len(values))
        samples=np.array(sums)[boot].sum(axis=1)/np.array(counts)[boot].sum(axis=1)
        result['components'][k]=dict(before_ms=float(old.mean()),after_ms=float(new.mean()),
            delta_ms=float((new-old).mean()),change_pct=float((new.mean()/old.mean()-1)*100),
            paired_question_bootstrap_delta_ci95=np.quantile(samples,[.025,.975]).tolist(),
            before_p50_ms=float(np.median(old)),after_p50_ms=float(np.median(new)))
    for source in (0,1,2):
        rows=[r for r in pooled if r['source']==source]
        result['per_source'][str(source)]=dict(n=len(rows),breakdown={k:{arm:float(np.mean([r[arm][k] for r in rows])) for arm in ('before','after')} for k in COMPONENTS})
    result.update(matched_step_pairs=len(pooled),questions=len(questions),
        notes='CUDA event wall intervals; all measured requests covered. Exclude warm/first2/final/cap/clip only; no latency outlier censoring. PS graph includes selection. Bootstrap clusters by question over repeats.',
        run_sha256={p.parent.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.raw.glob('*_graph*/run.json')})
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'BREAKDOWN.json').write_text(json.dumps(result,indent=2))
    (a.output/'MATCHED_STEPS.json').write_text(json.dumps(pooled))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#acd996','#aaaaaa','#f5c97e','#f28f43','#26785a','#b4a8cd']
    keys=['build','dispatch','prepare1','prepare2','graph','other']
    fig,axes=plt.subplots(2,1,figsize=(13,7),height_ratios=[1,1.5],layout='constrained')
    for i,arm in enumerate(('before','after')):
        left=0
        for k,col in zip(keys,colors):
            width=result['components'][k][arm+'_ms'];axes[0].barh(i,width,left=left,color=col,label=NAMES[k] if i==0 else None)
            if width>.28:axes[0].text(left+width/2,i,f'{width:.2f}',ha='center',va='center',fontsize=9)
            left+=width
        axes[0].text(left+.04,i,f'{left:.3f} ms',va='center')
    axes[0].set_yticks([0,1],['Before: 0916871','After: PS preparation']);axes[0].invert_yaxis();axes[0].set_xlabel('PS time (ms); mean of identical matched steps')
    axes[0].set_xlim(0,result['components']['ps_total']['before_ms']*1.15)
    x=np.arange(len(keys))
    for i,arm in enumerate(('before','after')):axes[1].bar(x+(i-.5)*.34,[result['components'][k][arm+'_ms'] for k in keys],width=.34,label=arm)
    axes[1].set_xticks(x,['Root/layout\nbuild','Dispatch\n(slot + guards)','Root metadata\npreparation','Graph input\npreparation','Graph\nforwards + tree','Other gaps\n+ merge']);axes[1].set_ylabel('ms');axes[1].legend()
    fig.suptitle('B1 tree PS preparation — Llama2 7B + AMD 135M, BF16, T=0.7\nK1/K2=6/3, exit20, C3, M1/M2=12/8; RTX4090, paired AB/BA + isolated confirmation')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=8)
    fig.savefig(a.output/'BREAKDOWN_COMPARISON.pdf');fig.savefig(a.output/'BREAKDOWN_COMPARISON.png',dpi=180)
    plt.close(fig)
    print(json.dumps(result['components'],indent=2))

if __name__=='__main__':main()
