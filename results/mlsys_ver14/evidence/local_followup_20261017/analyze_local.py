"""Exact-parity gate plus matched whole-step and native profile analysis."""
import json
from pathlib import Path
from collections import defaultdict
from importlib.util import spec_from_file_location,module_from_spec
import numpy as np

HERE=Path(__file__).resolve().parent
RAW=HERE/'raw/runs'


def load(tag):
    p=RAW/tag
    if not (p/'status.json').exists():return None
    s=json.loads((p/'status.json').read_text())
    if s['status']!='complete':return None
    return json.loads((p/'run.json').read_text())


def parity(a,b):
    aa=a['warm_groups']+a['groups'];bb=b['warm_groups']+b['groups']
    assert a['kwargs']==b['kwargs'] and a['prompt_ids']==b['prompt_ids']
    assert a['args']['seed']==b['args']['seed']
    events=lambda gg:[[(e['source'],e['cache_hit'],e['accepted_len'],e['valid_k'])
        for s in g['steps'] for e in s['events']] for g in gg]
    return dict(outputs_equal=[g['outputs'] for g in aa]==[g['outputs'] for g in bb],
        events_equal=events(aa)==events(bb),requests=len(aa))


def rows(d):
    walls=json.loads((RAW/d['arm']/'WALL_STEPS.json').read_text())
    allsteps=[s for g in d['warm_groups']+d['groups'] for s in g['steps']]
    assert len(walls)==len(allsteps)
    wi={id(s):w for s,w in zip(allsteps,walls)}
    out={}
    for g in d['groups']:
        for s in g['steps']:
            e=s['events'][0]
            if s['cap'] or s['clipped'] or not s.get('performance_included',True):continue
            w=wi[id(s)];assert w['native_seconds']==s['seconds']
            out[(g['phase'],g['index'],e['step_id'])]=dict(
                question=str(d['input_metadata'][g['index']]['question_id']),source=e['source'] if e['cache_hit'] else 0,
                sec=s['seconds'],wall=w['wall_seconds'],verify=sum(s['verify']))
    return out


def compare(pairs):
    comparisons=[];collected=[];parities=[]
    for before,after in pairs:
        a,b=load(before),load(after)
        if a is None or b is None:return None
        pp=parity(a,b);pp.update(before=before,after=after);parities.append(pp)
        if not(pp['outputs_equal'] and pp['events_equal']):
            return dict(status='rejected_parity',parity=parities)
        aa,bb=rows(a),rows(b);keys=sorted(aa.keys()&bb.keys())
        rr=[dict(question=aa[k]['question'],source=aa[k]['source'],a=aa[k],b=bb[k]) for k in keys]
        asec=sum(x['a']['sec'] for x in rr);bsec=sum(x['b']['sec'] for x in rr)
        comparisons.append(dict(before=before,after=after,n=len(rr),
            reduction_pct=100*(1-bsec/asec)))
        collected.extend(rr)
    grouped=defaultdict(lambda:[0.,0.])
    for x in collected:
        grouped[x['question']][0]+=x['a']['sec'];grouped[x['question']][1]+=x['b']['sec']
    mat=np.asarray([v for k,v in sorted(grouped.items())]);rng=np.random.default_rng(20261017)
    boot=mat[rng.integers(0,len(mat),(10000,len(mat)))].sum(axis=1)
    ci=np.quantile(100*(1-boot[:,1]/boot[:,0]),[.025,.975]).tolist()
    sources={}
    for source in (1,2,0,'all'):
        rr=[x for x in collected if source=='all' or x['source']==source]
        if not rr:continue
        sources[str(source)]={'n':len(rr)}
        for key,label in [('sec','step_ms'),('wall','independent_wall_ms'),('verify','target_call_accept_ms')]:
            v=[sum(x[side][key] for x in rr)*1000/len(rr) for side in ('a','b')]
            sources[str(source)][label]=v
    v=sources['all']['step_ms'];w=sources['all']['independent_wall_ms']
    return dict(status='complete',parity=parities,process_pairs=comparisons,
        reduction_pct=100*(1-v[1]/v[0]),ci95_pct=ci,sources=sources,
        adopted_candidate=bool(len(pairs)>=2 and ci[0]>0 and all(x['reduction_pct']>0 for x in comparisons) and w[1]<w[0]))


def main():
    result={}
    for slot,variant in [(0,'buckets'),(1,'overlap')]:
        first=(f'baseline_pair{slot}_a',f'{variant}_pair{slot}_b')
        second=(f'baseline_pair{slot}_d',f'{variant}_pair{slot}_c')
        pairs=[first,second] if load(second[0]) and load(second[1]) else [first]
        result[variant]=compare(pairs)
    pairs=[('baseline_pair0_d','prep_pair0_b')]
    if load('baseline_pair0_prep_d') and load('prep_pair0_c'):
        pairs.append(('baseline_pair0_prep_d','prep_pair0_c'))
    result['prep']=compare(pairs)
    a,b=load('baseline_pair0_a'),load('baseline_pair1_a')
    if a and b:result['baseline_cross_pair_parity']=parity(a,b)
    (HERE/'COMPARISONS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
