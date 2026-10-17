"""Within-engine randomized prep effect, controlling work and one-step lag."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE/'raw/runs'


def estimate(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[(r['uid'],r['phase'],r['source'],r['valid'],r['context']//128,r['previous_source'])].append(r)
    by_uid=defaultdict(lambda:[np.zeros((2,2)),np.zeros(2),0])
    for key,rr in groups.items():
        if min(sum(r['mode']==x for r in rr) for x in (0,1))<2:continue
        x=np.asarray([[r['mode'],r['previous_mode']] for r in rr],float)
        y=np.asarray([r['ms'] for r in rr]);x-=x.mean(axis=0);y-=y.mean()
        out=by_uid[key[0]];out[0]+=x.T@x;out[1]+=x.T@y;out[2]+=len(rr)
    if not by_uid:return None
    mats=np.stack([v[0] for k,v in sorted(by_uid.items())]);vecs=np.stack([v[1] for k,v in sorted(by_uid.items())])
    beta=np.linalg.solve(mats.sum(axis=0),vecs.sum(axis=0))
    rng=np.random.default_rng(297401);ix=rng.integers(0,len(mats),(10000,len(mats)))
    bb=np.linalg.solve(mats[ix].sum(axis=1),vecs[ix].sum(axis=1)[...,None])[...,0]
    return dict(samples=sum(v[2] for v in by_uid.values()),uid_clusters=len(by_uid),
        current_ms=float(beta[0]),previous_ms=float(beta[1]),combined_ms=float(beta.sum()),
        current_ci95_ms=np.quantile(bb[:,0],[.025,.975]).tolist(),
        combined_ci95_ms=np.quantile(bb.sum(axis=1),[.025,.975]).tolist())


def main():
    folder=ROOT/'mixed_prep'
    assert json.loads((folder/'status.json').read_text())['status']=='complete'
    d=json.loads((folder/'run.json').read_text());wall=json.loads((folder/'WALL_STEPS.json').read_text())
    base=json.loads((ROOT/'baseline_pair0_d/run.json').read_text())
    assert d['outputs'][:len(base['outputs'])]==base['outputs']
    original_events=lambda gg:[[(e['source'],e['cache_hit'],e['valid_k'],e['accepted_len']) for s in g['steps'] for e in s['events']] for g in gg]
    assert original_events(d['groups'][:len(base['groups'])])==original_events(base['groups'])
    i=0;rows=[]
    for g in d['warm_groups']+d['groups']:
        context=len(d['prompt_ids'][g['index']]);previous_mode=None;previous_source=None
        for s in g['steps']:
            w=wall[i];i+=1;e=s['events'][0];source=e['source'] if e['cache_hit'] else 0
            assert w['native_seconds']==s['seconds']
            if g['phase']!='warm' and e['step_id']>10 and not s['cap'] and not s['clipped'] and s.get('performance_included',True):
                rows.append(dict(uid=d['input_metadata'][g['index']]['uid'],phase=g['phase'],source=source,
                    previous_source=previous_source,context=context,valid=e['valid_k'],mode=w['prep_mode'],
                    previous_mode=previous_mode,ms=1000*s['seconds']))
            context+=s['emitted'];previous_mode=w['prep_mode'];previous_source=source
    result=dict(status='complete',reference_first24_outputs_and_events_equal=True,measured_requests=len(d['groups']),
        all=estimate(rows),by_source={str(src):estimate([r for r in rows if r['source']==src]) for src in (1,2,0)},
        meaning='Negative ms means faster. Randomized current/previous mode regression with within uid/pass/source/valid/context128/previous-source strata; UID-cluster CI. Diagnostic adapter only, not deployed runtime.')
    (HERE/'MIXED_ANALYSIS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
