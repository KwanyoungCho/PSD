"""Full-paper B1 AL comparisons, clustered by question within server/seed.

Online methods visit different contexts. Source-conditioned AL uses different
hit populations; the paired conditional CI uses only questions observed in both.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sys
import warnings
import numpy as np
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'ssd/bench'))
from mlsys_request_reporting import classify, common_question_summary
NAMES=('al','hit_percent','miss_al','p1_al','p2_al','miss_share','p1_share','p2_share')
PRIMARY=('chain_m32','last_c3_m32','frontier_c3_m32','last_c8_m32','frontier_c8_m32')
EXPECTED=(*PRIMARY,'last_c8_m64','frontier_c8_m64')

def mean(x):
    x=np.asarray(x,dtype=float); x=x[np.isfinite(x)]
    return float(x.mean()) if len(x) else None

def question_stats(r):
    """Exclude clipped/capped steps; include graph preparation for AL only."""
    turns=defaultdict(list)
    for group in r['groups']:
        if not group['phase'].startswith('measured'): continue
        requests={v['seq_id']:v for v in group['request_records']}
        counts={sid:np.zeros(9) for sid in requests}
        for step in group['steps']:
            if step['cap'] or step['clipped']: continue
            if step['batch']!=1: raise ValueError('Primary protocol requires B1')
            for sid,e in zip(step['seq_ids'],step['events']):
                row=counts[sid]; source=e['source']
                row[:3]+=[e['accepted_len'],e['cache_hit'],1]
                row[3+2*source:5+2*source]+=[e['accepted_len'],1]
        for sid,v in counts.items():
            def div(n,d): return float(n/d) if d else float('nan')
            values=[div(v[0],v[2]),100*div(v[1],v[2]),
                *[div(v[3+2*i],v[4+2*i]) for i in range(3)],
                *[100*div(v[4+2*i],v[2]) for i in range(3)]]
            turns[str(requests[sid]['question_id'])].append(values)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        return {q:np.nanmean(v,axis=0) for q,v in turns.items()}

def paired(a,b):
    if a['prompt_ids']!=b['prompt_ids'] or a['input_metadata']!=b['input_metadata']:
        raise ValueError('Unmatched input identities/token IDs')
    common=common_question_summary({'a':a['request_records'],'b':b['request_records']})['question_ids']
    ga,gb=question_stats(a),question_stats(b); keys=sorted(set(common)&ga.keys()&gb.keys())
    rng=np.random.default_rng(143); out={}
    for i,k in enumerate(NAMES):
        rows=[(ga[q][i],gb[q][i]) for q in keys if np.isfinite(ga[q][i]) and np.isfinite(gb[q][i])]
        if not rows:
            out[k]=dict(questions=0,before=None,after=None,delta=None,ci95=None);continue
        aa,bb=np.asarray(rows).T; diff=bb-aa
        draws=np.mean(diff[rng.integers(len(diff),size=(5000,len(diff)))],axis=1)
        out[k]=dict(questions=len(diff),before=float(aa.mean()),after=float(bb.mean()),delta=float(diff.mean()),
                    ci95=np.quantile(draws,[.025,.975]).tolist())
    return dict(all_turns_decoded_common_questions=len(keys),metrics=out,
       conditional_note='P1/P2/miss paired CI requires finite conditional AL in both arms; headline conditional means use each arm\'s own population.')

def quality(r,selected=None):
    rows=question_stats(r)
    if selected is not None: rows={q:v for q,v in rows.items() if q in selected}
    groups={str(m['question_id']):m['group'] for m in r['input_metadata']}
    result={}
    for group in ['all',*sorted(set(groups.values()))]:
        x=np.array([v for q,v in rows.items() if group=='all' or groups[q]==group]).reshape(-1,len(NAMES))
        result[group]={k:dict(mean=mean(x[:,i]),questions=int(np.isfinite(x[:,i]).sum())) for i,k in enumerate(NAMES)}
    return result

def eligible_events(r):
    return [e for s in r['steps'] if not s['cap'] and not s['clipped'] for e in s['events']]

def shape(r):
    es=eligible_events(r); result={}
    for source in (0,1,2):
        rows=[e for e in es if e['source']==source]; out={'verification_events':len(rows)}
        for field in ('accepted_len','valid_k','tree_max_depth','tree_sibling_width','reached_depth_ceiling','node_utilization'):
            x=[float(e[field]) for e in rows if e.get(field) is not None]
            out[field]=dict(mean=mean(x),p50=float(np.median(x)),p90=float(np.quantile(x,.9))) if x else None
        out['depth_histogram']=dict(sorted(Counter(e.get('tree_max_depth') for e in rows).items()))
        out['by_depth']={str(d):dict(events=len(z),al=mean([e['accepted_len'] for e in z])) for d in out['depth_histogram']
                          if (z:=[e for e in rows if e.get('tree_max_depth')==d])}
        result[str(source)]=out
    return result

def forest(path):
    compressed=path.with_suffix(path.suffix+'.gz')
    if compressed.exists():path=compressed
    if not path.exists():return None
    stats={};by_call={}
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as src:
      for line in src:
        x=json.loads(line);key=(x['phase'],x['root'])
        row=stats.setdefault(key,dict(n=0,generated=0,processed=0,served=0,generated_depth=0,clipped=0,empty=0))
        row['n']+=1
        for field in ('generated','processed','served','generated_depth'):row[field]+=x[field]
        row['clipped']+=x['generated']>x['served'];row['empty']+=x['processed']==0
        if x['physical_rows']!=64 or x['roots']!=8:raise AssertionError('Unequal physical forward budget')
        ck=(x['seq'],x['step'],x['phase']);v=by_call.setdefault(ck,[0,0,0,x['C']])
        v[0]+=x['processed'];v[1]+=x['generated'];v[2]+=1
    calls=list(by_call.values())
    if not calls: raise AssertionError('Empty forest trace')
    return dict(scope='all exported forests including warmup and final-step work',
       calls=len(calls),missing_root_records=sum(x[2]!=8 for x in calls),
       active_forward_rows=dict(mean=mean([x[0] for x in calls]),minimum=min(x[0] for x in calls)),
       physical_forward_rows_per_request_phase=64,full_active_calls=sum(x[0]==64 for x in calls),
       inconsistent_generated=sum(x[1]!=x[0]*x[3] for x in calls),
       by_priority_rank={f'p{p}_r{r}':{**x,**{k+'_mean':x[k]/x['n'] for k in ('generated','processed','served','generated_depth')}} for (p,r),x in sorted(stats.items())})

def validate(r):
    a=r['args']; expected=json.loads((HERE/'SPEC_BENCH_FULL.json').read_text())
    for k,v in dict(batch=1,limit=560,output_tokens=1024,input_cap=0,temperature=.7,k1=8,k2=8,exit_layer=21,passes=1).items():
        if a[k]!=v:raise AssertionError((k,a[k],v))
    if r['prompt_ids']!=[m['input_ids'] for m in expected]: raise AssertionError('Paper token IDs differ')
    if len(r['outputs'])!=560 or len(r['request_records'])!=560:raise AssertionError('Incomplete 560-turn run')
    if [m['uid'] for m in r['input_metadata']]!=[m['uid'] for m in expected]:raise AssertionError('Paper UID order differs')
    for m,q in zip(expected,r['request_records']):
        if q['paper_uid']!=m['uid'] or q['input_tokens']!=len(m['input_ids']):raise AssertionError('Request identity mismatch')
        classify(q)
        if q['completion_tokens']>1024:raise AssertionError('Output cap exceeded')
    return True

def campaign(folders):
    reports={};locations={};excluded=[];pending=[]
    for folder in folders:
        p=folder/'campaign.json'
        if not p.exists():pending.append(str(folder));continue
        for run in json.loads(p.read_text()):
            if run.get('status')!='complete' or run.get('external_pids') or run.get('returncode'):
                excluded.append(dict(directory=str(folder),run=run));continue
            f=folder/(run['name']+'.json')
            if not f.exists():pending.append(str(f));continue
            r=json.loads(f.read_text())
            if r['status']!='complete':excluded.append(run);continue
            validate(r)
            if run['name'] in reports:raise ValueError('Duplicate arm')
            reports[run['name']]=r;locations[run['name']]=folder
    if len({r['args']['seed'] for r in reports.values()})>1:raise ValueError('Mixing independent seeds')
    ref=next(iter(reports.values()),None)
    for r in reports.values():
        if r['source_hashes']!=ref['source_hashes'] or r['adapter_hashes']!=ref['adapter_hashes']:
            raise ValueError('Runtime differs across arms')
    prim={k:r for k,r in reports.items() if k in PRIMARY}
    population=common_question_summary({k:r['request_records'] for k,r in prim.items()}) if prim else None
    arms={}
    for k,r in reports.items():
        f=forest(locations[k]/(k+'_profile')/'equal_forest.jsonl')
        if not f or f['missing_root_records'] or f['inconsistent_generated']:raise AssertionError('Invalid forest accounting')
        ss=r['steps']; es=eligible_events(r)
        arms[k]=dict(question_quality=quality(r),
            common_primary_question_quality=quality(r,set(population['question_ids'])) if k in PRIMARY else None,
            shape=shape(r),forest=f,parameters={**r['args'],'actual_equal_budget':r['equal_budget']},
            auxiliary_timing=r['summary'],
            population=dict(questions=480,turns=560,eligible_verification_events=len(es),
                cap_steps=sum(bool(s['cap']) for s in ss),clipped_steps=sum(bool(s['clipped']) for s in ss),
                graph_preparation_steps=sum(bool(s.get('graph_preparation')) for s in ss),
                terminal_reasons=dict(Counter(x['finish_reason'] for x in r['request_records'])),
                zero_decode_uids=[x['paper_uid'] for x in r['request_records'] if classify(x)=='context_guard_no_decode']),
            raw_path=str(locations[k]/(k+'.json')))
    pairs=[('chain_m32',k) for k in PRIMARY[1:]]
    pairs += [('last_c3_m32','frontier_c3_m32'),('last_c8_m32','frontier_c8_m32'),('last_c3_m32','last_c8_m32'),('frontier_c3_m32','frontier_c8_m32'),('last_c8_m32','last_c8_m64'),('frontier_c8_m32','frontier_c8_m64')]
    comparisons={a+' -> '+b:paired(reports[a],reports[b]) for a,b in pairs if a in reports and b in reports}
    return dict(status='complete' if set(EXPECTED)<=reports.keys() else 'partial',directories=list(map(str,folders)),
        arms=arms,comparisons=comparisons,excluded=excluded,pending=pending,
        common_primary_questions=population['question_ids'] if population else [],
        scope='Within server/seed, paired questions; AL includes one confirmed root/target token; source-conditional populations differ.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--local',type=Path,nargs='*',default=[]);ap.add_argument('--remote',type=Path,nargs='*',default=[])
    ap.add_argument('--output',type=Path,default=HERE/'RESULTS.json');a=ap.parse_args()
    result={k:campaign(v) for k,v in [('local',a.local),('remote',a.remote)] if v}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for k,c in result.items():
        print(k,c['status'])
        for name,v in c['arms'].items():print(name,v['question_quality']['all'])
if __name__=='__main__':main()
