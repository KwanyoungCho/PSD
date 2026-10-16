"""Matched profile-OFF whole-step comparisons; raw and selected cohorts."""
from collections import defaultdict
import argparse
import json
from pathlib import Path
import numpy as np


def load(root,tag):
    path=root/'runs'/tag/'run.json'
    j=json.loads(path.read_text())
    if j['status']!='complete':raise ValueError(f'Incomplete: {tag}')
    return j


def index_steps(j,filtered=True):
    out={}
    for g in j['groups']:
        for s in g['steps']:
            if s['cap'] or s['clipped'] or s['batch']!=1:continue
            if filtered and not s.get('performance_included',True):continue
            e=s['events'][0]
            out[(g['phase'],g['index'],e['step_id'])]=dict(s,uid=g['index'],state=e['source'] if e['cache_hit'] else 0)
    return out


def assert_parity(a,b):
    if a['outputs']!=b['outputs']:raise AssertionError('Output token ids differ')
    def events(j):
        return [(e.get('source'),e.get('cache_hit'),e.get('valid_k'),e.get('accepted_len')) for s in j['steps'] for e in s['events']]
    if events(a)!=events(b):raise AssertionError('AL/hit/proposal events differ')


def compare(root,model,variant,filtered,reference='before',stage='perf'):
    clusters=defaultdict(lambda:np.zeros(3));repeats=[];total_a=total_b=0
    by_source=defaultdict(lambda:np.zeros(3))
    verify_total=np.zeros(2)
    excluded=[]
    for rep in ('a','b'):
        before=load(root,f'{reference}_{model}_{stage}_{rep}')
        after=load(root,f'{variant}_{model}_{stage}_{rep}')
        assert_parity(before,after)
        aa=index_steps(before,filtered);bb=index_steps(after,filtered)
        keys=sorted(aa.keys()&bb.keys())
        vals=np.zeros(3)
        verify_pair=np.zeros(2)
        for k in keys:
            a,b=aa[k],bb[k]
            row=np.asarray([a['seconds'],b['seconds'],1.])
            vals+=row;clusters[a['uid']]+=row;by_source[a['state']]+=row
            verify_pair+=np.asarray([sum(a['verify']),sum(b['verify'])])
        if not vals[2]:raise ValueError('Empty matched population')
        repeats.append(dict(repeat=rep,matched_steps=int(vals[2]),before_ms=1000*vals[0]/vals[2],
            after_ms=1000*vals[1]/vals[2],reduction_pct=100*(1-vals[1]/vals[0]),
            before_verify_ms=1000*verify_pair[0]/vals[2],after_verify_ms=1000*verify_pair[1]/vals[2]))
        verify_total+=verify_pair
        total_a+=len(aa);total_b+=len(bb)
        excluded.append(dict(repeat=rep,before=before['summary'].get('excluded_long_steps'),
                             after=after['summary'].get('excluded_long_steps')))
    matrix=np.stack(list(clusters.values()));total=matrix.sum(0)
    rng=np.random.default_rng(20261017)
    resampled=matrix[rng.integers(0,len(matrix),size=(10000,len(matrix)))].sum(1)
    delta=1000*(resampled[:,0]-resampled[:,1])/resampled[:,2]
    pct=100*(1-resampled[:,1]/resampled[:,0])
    return dict(model=model,stage=stage,reference=reference,variant=variant,cohort='2x-filtered matched intersection' if filtered else 'raw cap/clip-only matched steps',
        output_and_event_parity=True,uid_clusters=len(clusters),matched_steps=int(total[2]),
        before_available_steps=total_a,after_available_steps=total_b,
        before_ms=1000*total[0]/total[2],after_ms=1000*total[1]/total[2],
        before_verify_ms=1000*verify_total[0]/total[2],after_verify_ms=1000*verify_total[1]/total[2],
        before_outside_verify_ms=1000*(total[0]-verify_total[0])/total[2],after_outside_verify_ms=1000*(total[1]-verify_total[1])/total[2],
        reduction_ms=1000*(total[0]-total[1])/total[2],reduction_pct=100*(1-total[1]/total[0]),
        reduction_ms_ci95=np.quantile(delta,[.025,.975]).tolist(),reduction_pct_ci95=np.quantile(pct,[.025,.975]).tolist(),
        repeats=repeats,excluded_long_steps=excluded,
        source={str(s):dict(n=int(v[2]),before_ms=1000*v[0]/v[2],after_ms=1000*v[1]/v[2],reduction_pct=100*(1-v[1]/v[0])) for s,v in by_source.items()},
        uncertainty='UID cluster bootstrap across the fixed 12-input diagnostic set; process-repeat effects reported separately; not full-dataset or multi-hardware evidence')


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();rows=[]
    for model in ('7b','13b'):
        for reference,variant in (('before','snapshot'),('before','inputs'),('snapshot','inputs')):
            if not all((args.raw/'runs'/f'{arm}_{model}_perf_{rep}'/'status.json').exists() for arm in (reference,variant) for rep in ('a','b')):continue
            for filtered in (False,True):rows.append(compare(args.raw,model,variant,filtered,reference))
        if all((args.raw/'runs'/f'{arm}_{model}_final_{rep}'/'status.json').exists() for arm in ('before','restore') for rep in ('a','b')):
            for filtered in (False,True):rows.append(compare(args.raw,model,'restore',filtered,stage='final'))
    if all((args.raw/'runs'/f'{arm}_7b_chain_perf_{rep}'/'status.json').exists() for arm in ('before','inputs') for rep in ('a','b')):
        for filtered in (False,True):rows.append(compare(args.raw,'7b_chain','inputs',filtered))
    result=dict(status='complete' if len(rows)==18 else 'partial',comparisons=rows,
        primary='Profile OFF complete decode-step time. No inference preparation seconds are subtracted from retained steps.')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for row in rows:
        if row['cohort'].startswith('2x'):
            print(row['model'],row['reference'],row['variant'],f"{row['before_ms']:.4f} -> {row['after_ms']:.4f} ms",row['reduction_pct_ci95'],flush=True)


if __name__=='__main__':main()
