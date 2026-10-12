"""Identical population and paired prompt-level comparisons; no raw TPS mixing."""
import json,hashlib,statistics,csv
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent

def included(d):
    return [s for s in d['steps'] if s['batch']==d['args']['batch'] and not(s['cap'] or s['clipped'])]
def group_values(d):
    out=[]
    for g in d['groups']:
        s=[s for s in g['steps'] if s['batch']==d['args']['batch'] and not(s['cap'] or s['clipped'])]
        out.append((sum(x['emitted'] for x in s),sum(x['seconds'] for x in s)))
    return np.array(out)
def pair(a,b):
    assert a['prompt_ids']==b['prompt_ids']
    va,vb=group_values(a),group_values(b)
    rng=np.random.default_rng(417);diff=[]
    for _ in range(2000):
        idx=rng.integers(0,len(va),len(va));xa=va[idx].sum(0);xb=vb[idx].sum(0)
        diff.append((xb[0]/xb[1])/(xa[0]/xa[1])-1)
    def pct(x,y):return 100*(y/x-1)
    return dict(tps_pct=pct(a['summary']['tps'],b['summary']['tps']),
        paired_prompt_bootstrap_tps_pct_ci95=(100*np.quantile(diff,[.025,.975])).tolist(),
        interpretation='Prompt resampling interval only; does not quantify run-to-run or hardware uncertainty.',
        outputs_equal=a['outputs']==b['outputs'],different_outputs=sum(x!=y for x,y in zip(a['outputs'],b['outputs'])),
        al_delta=b['summary']['al']-a['summary']['al'] if a['summary']['al'] is not None else None,
        hit_delta=b['summary']['hit']-a['summary']['hit'] if a['summary']['hit'] is not None else None,
        verify_ms_delta=b['summary']['verify_ms']-a['summary']['verify_ms'] if a['summary']['verify_ms'] else None)

def main():
    runs={};native=[];telemetry=[]
    for path in sorted(R.glob('*/*.json')):
        if path.name in ('plan.json','campaign.json'):continue
        try:d=json.loads(path.read_text())
        except (ValueError,UnicodeDecodeError):continue
        if not isinstance(d,dict) or 'groups' not in d or d.get('status')!='complete':continue
        key=str(path.relative_to(R));runs[key]=d
        for i,g in enumerate(d['groups']):
            p=g.get('native_performance')
            if not p:continue
            keep=[s for s in g['steps'] if s['batch']==d['args']['batch'] and not(s['cap'] or s['clipped'])]
            tokens=sum(s['emitted'] for s in keep);seconds=sum(s['seconds'] for s in keep)
            ev=[e for s in keep for e in s['events']]
            assert p['emitted_tokens']==tokens and p['included_steps']==len(keep),(key,i)
            assert abs(p['seconds']-seconds)<1e-9,(key,i,p['seconds'],seconds)
            assert p['excluded_steps']==len(g['steps'])-len(keep)
            if ev:
                assert abs(p['al']-statistics.mean(e['accepted_len'] for e in ev))<1e-12
                hits=[e['cache_hit'] for e in ev if e.get('cache_hit') is not None]
                if hits: assert abs(p['cache_hit']-statistics.mean(hits))<1e-12
            native.append(dict(run=key,group=i,all_fields_match=True))
        tf=path.with_suffix('.telemetry.jsonl')
        if tf.exists():
            ts=[json.loads(x) for x in tf.read_text().splitlines()];foreign=sorted({p for x in ts for p in x['foreign']})
            assert not foreign,(key,foreign)
            stats={}
            for x in ts:
                for line in x['gpu'].splitlines():
                    gpu,temp,power,clock,util,mem=[float(y.strip().split()[0]) for y in line.split(',')]
                    stats.setdefault(int(gpu),[]).append((temp,clock,util,mem))
            telemetry.append(dict(run=key,samples=len(ts),external_pids=foreign,gpus={k:dict(max_temp=max(y[0] for y in v),mean_clock=statistics.mean(y[1] for y in v),mean_util=statistics.mean(y[2] for y in v),active_clock=statistics.mean(y[1] for y in v if y[2]>=50) if any(y[2]>=50 for y in v) else None) for k,v in stats.items()}))
    comparisons={}
    for campaign in ('tree_r0','tree_r1','chain','temperature'):
        for a,b in [('paper','before'),('before','after'),('paper','after')]:
            ak,bk=f'{campaign}/{a}.json',f'{campaign}/{b}.json'
            if ak in runs and bk in runs:comparisons[f'{campaign}:{a}->{b}']=pair(runs[ak],runs[bk])
    for a,b in [('0_before','1_after'),('3_before','2_after')]:
        ak,bk=f'abba/{a}.json',f'abba/{b}.json'
        if ak in runs and bk in runs:comparisons[f'abba:{a}->{b}']=pair(runs[ak],runs[bk])
    accounting={}
    for key,d in runs.items():
        ev=[e for st in d['steps'] for e in st['events']]
        if not ev:continue
        accepted=sum(e['accepted_len'] for e in ev)
        emitted=sum(st['emitted'] for st in d['steps'])
        accounting[key]=dict(raw_accepted_tokens=accepted,emitted_decode_tokens=emitted,
            unreturned_accepted_tokens=accepted-emitted,
            raw_counter_inflation_pct=100*(accepted/emitted-1) if emitted else None,
            cap_steps=sum(st['cap'] for st in d['steps']),
            clipped_steps=sum(st['clipped'] for st in d['steps']),
            partial_batch_steps=sum(st['batch']!=d['args']['batch'] for st in d['steps']),
            note='Reconstructed raw accepted-token counter before the user-approved whole-step filter; not mixed with filtered TPS.')
    out=dict(accounting=accounting,runs={k:dict(summary=v['summary'],kwargs=v['kwargs'],commit=v['paper_commit'],dirty=v['paper_dirty'],prompts=len(v['groups']),source_hashes=v['source_hashes']) for k,v in runs.items()},comparisons=comparisons,native_population_groups_checked=len(native),native_checks=native,telemetry=telemetry)
    (R/'ANALYSIS.json').write_text(json.dumps(out,indent=2))
    with (R/'PERFORMANCE.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['run','B','T','prompts','AL','cache_hit','TPS','verify_ms','steps','all_steps','tokens','seconds'])
        for k,v in runs.items():
            s=v['summary'];w.writerow([k,v['args']['batch'],v['args']['temperature'],len(v['outputs']),s['al'],s['hit'],s['tps'],s['verify_ms'],s['steps'],s['all_steps'],s['tokens'],s['seconds']])
    print('COMPLETE RUNS',len(runs),'NATIVE GROUPS CHECKED',len(native));print(json.dumps(comparisons,indent=2))
if __name__=='__main__':main()
