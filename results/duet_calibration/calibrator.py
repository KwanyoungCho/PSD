"""New DUET reward/time calibrator; legacy gap selector is only a control."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def load_run(path):
    result=json.loads((path/'result.json').read_text())
    meta=json.loads((path/'complete.json').read_text())
    return dict(path=str(path), config=meta['config'], meta=meta, records=result['records'])


def metrics(run):
    rec=run['records']; ev=[e for r in rec for e in r['metrics']['phase_events']]
    pi=np.array([sum(e['source']==c for e in ev)/len(ev) for c in range(3)])
    al=np.array([np.mean([e['accepted_spec_len'] for e in ev if e['source']==c]) for c in range(3)])
    u=np.mean([e['accepted_len'] for e in ev])
    if not np.isclose(u,1+pi@al): raise ValueError('Reward decomposition failed')
    return dict(tps=sum(r['output_tokens'] for r in rec)/sum(r['wall_s'] for r in rec),
                steps=len(ev),pi=pi.tolist(),al=al.tolist(),u=float(u),
                wall_s=sum(r['wall_s'] for r in rec),
                elapsed_s=run['meta']['elapsed_s'])


def label_events(events,label):
    out=defaultdict(list);epoch=0;previous=None
    for ev in events:
        if ev['label'] != label or ev.get('step_id') is None: continue
        sid=int(ev['step_id'])
        if previous is not None and sid<previous: epoch+=1
        previous=sid
        out[(epoch,sid)].append(ev)
    return out


def profile_rows(run):
    path=Path(run['path']); d=json.loads(next(path.glob('duet_profile_draft_*.json')).read_text())
    t=json.loads(next(path.glob('duet_profile_target_rank0_*.json')).read_text())
    labels={name:label_events(t,name) for name in ['target_send_request','graph_pre','proxy_send_enqueue']}
    labels.update({name:label_events(d,name) for name in ['phase1_build','proxy_wait','merge_cache']})
    def ev(label,key):
        x=labels[label].get(key,[])
        return x[0] if len(x)==1 else None
    def begin(e): return e['wall_start_ns']/1e6
    def end(e): return e['wall_end_ns']/1e6
    lag=[]
    for key,values in labels['proxy_wait'].items():
        pw=values[0];ps=ev('proxy_send_enqueue',key)
        if ps and end(pw)-begin(pw)>.2 and 0<=end(pw)-end(ps)<5: lag.append(end(pw)-end(ps))
    transport=float(np.median(lag)) if lag else .1
    rows=[];errors=[];dropped=defaultdict(int)
    cfg=run['config']
    for i,record in enumerate(run['records']):
        # Epoch zero is the excluded warmup. Step IDs reset at each generate().
        epoch=i+1;events=record['metrics']['phase_events'];prefix=record['prompt_tokens']
        for idx,event in enumerate(events):
            sid=event['step_id'];key=(epoch,sid);nkey=(epoch,sid+1)
            cur={label:ev(label,key) for label in labels}
            nxt={label:ev(label,nkey) for label in ['target_send_request','graph_pre']}
            context=prefix;prefix+=event['accepted_len']
            if idx<2 or idx==len(events)-1: continue
            if not all(cur.values()) or not all(nxt.values()): dropped['missing']+=1;continue
            origin=begin(cur['graph_pre']);F=begin(nxt['target_send_request'])-origin
            cache=end(cur['merge_cache'])-origin
            S=begin(nxt['graph_pre'])-max(begin(nxt['target_send_request']),end(cur['merge_cache']))
            x=begin(cur['phase1_build'])-origin
            D1=end(cur['phase1_build'])-begin(cur['phase1_build'])
            D2=end(cur['merge_cache'])-end(cur['proxy_wait'])
            P=end(cur['proxy_send_enqueue'])+transport-origin
            # Separate actual measured arrival from the inferred uncensored arrival.
            wait=end(cur['proxy_wait'])-origin
            reconstructed=max(F,max(x+D1,P)+D2)+S
            actual=begin(nxt['graph_pre'])-origin
            errors.append(reconstructed-actual)
            if min(F,D1,D2)<0 or S<-.5 or actual>250: dropped['invalid']+=1;continue
            c=event['source'];nc=events[idx+1]['source']
            n=cfg['k1'] if c==1 else cfg['k2']
            nn=cfg['k1'] if nc==1 else cfg['k2']
            rows.append(dict(config=cfg,prompt=i,state=c,next_state=nc,n=n,next_n=nn,context=context,
                             F=F,x=x,D1=D1,D2=D2,P=P,S=max(0,S),cycle=actual,
                             gap1=P-(x+D1),gap2=F-cache))
    return rows,dict(transport_ms=transport,transport_samples=len(lag),rows=len(rows),dropped=dict(dropped),
                     reconstruction_mae_ms=float(np.mean(np.abs(errors))) if errors else None)


def features(kind,cfg,state,context):
    n=cfg['k1'] if state==1 else cfg['k2'];l=cfg['exit'];k1=cfg['k1'];k2=cfg['k2'];ctx=context/512
    if kind=='F': return [1,n,ctx]
    if kind=='P': return [1,l/80,l*n/80,ctx*l/80]
    if kind=='x': return [1,n,ctx]
    if kind=='D1': return [1,k1,k1*n,k1*ctx]
    if kind=='D2': return [1,k2,k2*ctx]
    if kind=='S': return [1,n,float(state==0),k2*float(state==0)]
    raise KeyError(kind)


class Calibrator:
    def __init__(self,model): self.model=model

    @classmethod
    def fit(cls,runs):
        allrows=[];diagnostics={};anchor={}
        for run in runs:
            rows,diag=profile_rows(run);allrows+=rows;diagnostics[run['config']['tag']]=diag
            anchor[run['config']['tag']]=dict(config=run['config'],metrics=metrics(run))
        if len(allrows)<100: raise ValueError('Too few profile rows')
        components={}
        # Fit group means so long/slow prompts do not get extra regression weight.
        for kind in ['F','P','x','D1','D2','S']:
            groups=defaultdict(list)
            for row in allrows:
                state=row['next_state'] if kind=='S' else row['state']
                groups[(row['config']['tag'],row['prompt'],state)].append(row)
            X=[];Y=[]
            for group in groups.values():
                r=group[0];state=r['next_state'] if kind=='S' else r['state']
                X.append(np.mean([features(kind,z['config'],state,z['context']) for z in group],axis=0))
                Y.append(np.median([z[kind] for z in group]))
            X=np.array(X);Y=np.array(Y);coef=np.linalg.lstsq(X,Y,rcond=None)[0]
            components[kind]=dict(coef=coef.tolist(),group_count=len(Y),mae_ms=float(np.mean(abs(X@coef-Y))))
        b=next(r for r in runs if r['config']['exit']==56 and r['config']['k1']==10 and r['config']['k2']==4)
        cfg=b['config'];base=metrics(b)
        reward_samples={str(c):[e['accepted_spec_len'] for rec in b['records'] for e in rec['metrics']['phase_events'] if e['source']==c] for c in range(3)}
        pref=[(r['prompt_tokens'],r['metrics']['prefill_total_time']*1000) for run in runs for r in run['records']]
        prefcoef=np.linalg.lstsq(np.array([[1,n/512] for n,t in pref]),np.array([t for n,t in pref]),rcond=None)[0]
        model=dict(components=components,anchors=anchor,base=base,base_config=cfg,
                   reward_samples=reward_samples,prefill_coef=prefcoef.tolist(),diagnostics=diagnostics,
                   bounds=dict(exit=[40,72],k1=[6,10],k2=[2,4],budget=[15,15]),
                   assumptions='Additive phase-frequency effects, truncated empirical conditional AL, grouped affine component costs inside max overlap. Not a global optimality guarantee.')
        return cls(model)

    def probabilities(self,cfg):
        base=np.array(self.model['base']['pi']);p=base.copy()
        def anchor(e,k1,k2):
            return np.array(next(a['metrics']['pi'] for a in self.model['anchors'].values() if
                                 (a['config']['exit'],a['config']['k1'],a['config']['k2'])==(e,k1,k2)))
        p+=(cfg['k1']-10)/(-4)*(anchor(56,6,4)-base)
        p+=(cfg['k2']-4)/(-2)*(anchor(56,10,2)-base)
        endpoint=40 if cfg['exit']<56 else 72
        p+=(cfg['exit']-56)/(endpoint-56)*(anchor(endpoint,10,4)-base)
        p=np.clip(p,0,1)
        return p/p.sum()

    def predict(self,cfg,contexts,output_len=128):
        if cfg['mode']!='chain' or cfg['budget']!=15 or cfg['candidate']!='legacy':
            return dict(config=cfg,status='requires_adapter_calibration')
        pi=self.probabilities(cfg)
        al=np.array([np.mean(np.minimum(self.model['reward_samples'][str(c)], cfg['k1'] if c==1 else cfg['k2'])) for c in range(3)])
        # Correct the truncated-chain approximation with the observed anchor
        # selection effects. Earlier exits can change which contexts hit P1/P2.
        base=np.array(self.model['base']['al'])
        def anchor_al(e,k1,k2):
            return np.array(next(a['metrics']['al'] for a in self.model['anchors'].values() if
                                 (a['config']['exit'],a['config']['k1'],a['config']['k2'])==(e,k1,k2)))
        endpoint=40 if cfg['exit']<56 else 72
        al+=(cfg['exit']-56)/(endpoint-56)*(anchor_al(endpoint,10,4)-base)
        truncated6=base.copy();truncated6[1]=np.mean(np.minimum(self.model['reward_samples']['1'],6))
        al+=(cfg['k1']-10)/(-4)*(anchor_al(56,6,4)-truncated6)
        truncated2=np.array([np.mean(np.minimum(self.model['reward_samples'][str(c)],10 if c==1 else 2)) for c in range(3)])
        al+=(cfg['k2']-4)/(-2)*(anchor_al(56,10,2)-truncated2)
        al=np.clip(al,0,[cfg['k2'],cfg['k1'],cfg['k2']])
        u=1+pi@al
        times=[];cycles=[];g1=[];g2=[]
        for length in contexts:
            ctx=length+output_len/2
            vals={}
            for c in range(3):
                v={kind:float(np.dot(self.model['components'][kind]['coef'],features(kind,cfg,c,ctx))) for kind in self.model['components']}
                for kind in ['F','D1','D2','S','P']:v[kind]=max(.01,v[kind])
                cache=max(v['x']+v['D1'],v['P'])+v['D2']
                vals[c]=dict(cycle=max(v['F'],cache)+v['S'],gap1=v['P']-v['x']-v['D1'],gap2=v['F']-cache)
            cycle=sum(pi[c]*vals[c]['cycle'] for c in range(3));cycles.append(cycle)
            g1.append(sum(pi[c]*abs(vals[c]['gap1']) for c in range(3)))
            g2.append(sum(pi[c]*abs(vals[c]['gap2']) for c in range(3)))
            prefill=max(0,float(np.dot(self.model['prefill_coef'],[1,length/512])))
            times.append((prefill+output_len/u*cycle)/1000)
        return dict(config=cfg,status='extrapolation' if cfg['k1']<6 else 'interpolation',pi=pi.tolist(),al=al.tolist(),
                    tokens_per_step=float(u),cycle_ms=float(np.mean(cycles)),
                    tps=len(contexts)*output_len/sum(times),absolute_gap_ms=float(np.mean(g1)+np.mean(g2)))


def freeze():
    out=HERE/'frozen.json'
    if out.exists():raise RuntimeError('Frozen model already exists')
    plan=json.loads((HERE/'plan.json').read_text());runs=[]
    for cfg in plan['calibration']:
        path=HERE/'runs/calibration'/f'{cfg["tag"]}_s913'
        runs.append(load_run(path))
    model=Calibrator.fit(runs)
    contexts=[x['input_tokens'] for x in plan['datasets']['calibration']['mapping']]
    predictions=[model.predict(cfg,contexts) for cfg in plan['validation']]
    ranked=sorted(predictions,key=lambda p:p['tps'],reverse=True)
    gap=sorted(predictions,key=lambda p:p['absolute_gap_ms'])
    measured=sorted(runs,key=lambda r:metrics(r)['tps'],reverse=True)
    payload=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),model=model.model,predictions=predictions,
                 selection=dict(top1=ranked[0]['config']['tag'],top3=[p['config']['tag'] for p in ranked[:3]],
                                gap_only=gap[0]['config']['tag'],best_measured_anchor=measured[0]['config']['tag']),
                 calibration_cost_s=sum(r['meta']['elapsed_s'] for r in runs),
                 calibration_generate_s=sum(metrics(r)['wall_s'] for r in runs),
                 hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                         [HERE/'plan.json',Path(__file__),HERE/'run_bench.py']+
                         [Path(r['path'])/'result.json' for r in runs]})
    out.write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload['selection'],indent=2))
    print('Predicted top five:',[(p['config']['tag'],round(p['tps'],2)) for p in ranked[:5]])


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze']);args=ap.parse_args()
    freeze()
