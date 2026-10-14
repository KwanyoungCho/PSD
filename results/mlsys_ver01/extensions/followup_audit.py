"""Audit graph/setup tails without dropping them from reported throughput.

Uses two full passes in the same engine, not independent repetitions. Only
capture CPU intervals are instrumented; no CUDA event/synchronization is added.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'ssd/bench'))
from mlsys_measurement import capture_intervals
import analyze as a


def stats(data):
    steps=[s for s in data['steps'] if a.eligible(s,data['args']['batch'])]
    events=[e for s in steps for e in s['events']]
    seconds=sum(s['seconds'] for s in steps)
    lengths=np.array([s['seconds'] for s in steps])
    med=float(np.median(lengths)); slow=[s for s in steps if s['seconds']>5*med]
    return dict(steps=len(steps), all_steps=len(data['steps']),
        tokens=sum(s['emitted'] for s in steps),seconds=seconds,
        tps=sum(s['emitted'] for s in steps)/seconds,
        al=float(np.mean([e['accepted_len'] for e in events])),
        hit=float(np.mean([e['cache_hit'] for e in events])),
        actual_output_tokens=sum(map(len,data['outputs'])),
        raw_decode_tps=sum(map(len,data['outputs']))/sum(s['seconds'] for s in data['steps']),
        mean_step_ms=1000*seconds/len(steps),median_step_ms=1000*med,max_step_ms=1000*float(max(lengths)),
        slow_definition='step > 5 * within-cell eligible median; diagnostic only; none removed',
        slow_steps=len(slow),slow_time_fraction=sum(s['seconds'] for s in slow)/seconds)


def phase(data,name):
    groups=[g for g in data['groups'] if g['phase']==name]
    d=dict(data,groups=groups,steps=[s for g in groups for s in g['steps']],
           outputs=[o for g in groups for o in g['outputs']])
    d['summary']=stats(d)
    return d


def union_length(intervals):
    end=-1;total=0
    for left,right in sorted(intervals):
        total+=max(0,right-max(end,left));end=max(end,right)
    return total


def capture_stats(data,captures):
    rows=[];med=data['summary']['median_step_ms']/1000
    for gi,g in enumerate(data['groups']):
        for s in g['steps']:
            if not a.eligible(s,data['args']['batch']):continue
            left,right=s['audit_start_ns'],s['audit_end_ns']
            cs=[c for c in captures if c['start_ns']<right and c['end_ns']>left]
            ns=union_length([(max(left,c['start_ns']),min(right,c['end_ns'])) for c in cs])
            rows.append(dict(group=gi,prompt_index=g['index'],step=s['events'][0]['step_id'],
                seconds=s['seconds'],verify_ms=1000*sum(s['verify']),
                slow=s['seconds']>5*med,capture_overlap_ms=ns/1e6,
                capture_overlap_fraction=ns/(right-left),
                captures=[dict(label=c['label'],pid=c['pid'],
                    duration_ms=(c['end_ns']-c['start_ns'])/1e6,
                    starts_before_step=c['start_ns']<left) for c in cs]))
    all_steps=data['steps'];start=min(s['audit_start_ns'] for s in all_steps);end=max(s['audit_end_ns'] for s in all_steps)
    inphase=[c for c in captures if start<=c['start_ns']<end]
    return dict(capture_counts_in_phase=dict(Counter(c['label'] for c in inphase)),
        eligible_steps_overlapping_capture=sum(bool(r['captures']) for r in rows),
        slow_steps_overlapping_capture=sum(r['slow'] and bool(r['captures']) for r in rows),
        slow_steps_with_majority_capture_overlap=sum(r['slow'] and r['capture_overlap_fraction']>=.5 for r in rows),
        longest_steps=sorted(rows,key=lambda r:r['seconds'],reverse=True)[:12],
        rows=rows)


def timeline(folder,name):
    data=a.load(folder/(name+'.json'));warm=len(data['warm_groups'])
    allowed={(warm+i,s['events'][0]['step_id']) for i,g in enumerate(data['groups'])
             for s in g['steps'] if a.eligible(s,data['args']['batch'])}
    rows={};captured=set()
    for f in (folder/name).glob('duet_profile_*.json'):
        proc='draft' if 'draft_' in f.name else 'target'
        first='batch_kv_restore' if proc=='draft' else 'target_send_request';group=-1
        for r in sorted(a.load(f)[1:],key=lambda x:x['idx']):
            if r['label']==first and r['step_id']==1:group+=1
            key=(group,r['step_id']);rows.setdefault(key,{})[proc+'/'+r['label']]=r
            if 'capture' in r['label']:captured.add(key)
    late1=[];late2=[]
    for key in sorted(allowed):
        prev=(key[0],key[1]-1);nxt=(key[0],key[1]+1)
        if key in captured or prev in captured:continue
        r=rows[key]
        late1.append((r['draft/batch_p1_total']['wall_end_ns']-r['target/batch_target_proxy']['wall_end_ns'])/1e6)
        if nxt in rows and 'target/target_send_request' in rows[nxt]:
            late2.append((r['draft/batch_p2_total']['wall_end_ns']-rows[nxt]['target/target_send_request']['wall_start_ns'])/1e6)
    return dict(scope='GPU event timestamps mapped to host clock; approximate cross-device alignment. Current and prior capture-labelled steps excluded; small48-input subset.',
        p1_end_minus_proxy_end=dict(n=len(late1),median_ms=float(np.median(late1)),positive_fraction=float(np.mean(np.array(late1)>0))),
        p2_end_minus_next_request=dict(n=len(late2),median_ms=float(np.median(late2)),positive_fraction=float(np.mean(np.array(late2)>0))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw-root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path(__file__).with_name('FOLLOWUP_AUDIT.json'))
    args=p.parse_args();root=args.raw_root;nova=root/'nova_latest';folder=nova/'review_followup'
    result=dict(scope='Llama3 dense TP1 B8 T=.7 K1/K2=4/2 exit-index21; two full480 passes per engine; one process per policy',
        inference_limits=['Second pass is not an independent seed/process repetition; draft RNG continues.',
            'Capture CPU overlap shows where preparation occurs; do not add overlapping intervals as overhead.',
            'No slow step is removed from TPS. No global parameter optimality is claimed.'],
        old_tail_audit={},repeat={},comparisons={},provenance={})
    for model,oldfolder in [('llama2',root/'llama2_sensitivity'),('llama3',nova/'llama3_sensitivity')]:
        result['old_tail_audit'][model]={arm:stats(a.load(oldfolder/f'k2_2_{arm}.json')) for arm in ('wait','ar')}
    data={arm:a.load(folder/f'{arm}.json') for arm in ('wait','ar')}
    assert all(d['status']=='complete' for d in data.values())
    assert data['wait']['source_hashes']==data['ar']['source_hashes']
    olddata={arm:a.load(nova/'llama3_sensitivity'/f'k2_2_{arm}.json') for arm in data}
    phases={}
    for arm,d in data.items():
        captures=capture_intervals(folder/f'{arm}_captures')
        phases[arm]={name:phase(d,name) for name in ('measured_first','measured_second')}
        result['repeat'][arm]={name:dict(metrics=v['summary'],capture=capture_stats(v,captures)) for name,v in phases[arm].items()}
        first=phases[arm]['measured_first'];old=olddata[arm]
        result['provenance'][arm]=dict(runtime_hashes_match_prior=d['source_hashes']==old['source_hashes'],
            kwargs_match_prior=d['kwargs']==old['kwargs'],first_outputs_match_prior=first['outputs']==old['outputs'],
            first_events_match_prior=[s['events'] for s in first['steps']]==[s['events'] for s in old['steps']],
            sha256=hashlib.sha256((folder/f'{arm}.json').read_bytes()).hexdigest())
    for name in ('measured_first','measured_second'):
        result['comparisons'][name]=a.compare(phases['wait'][name],phases['ar'][name])
    result['campaign']=a.load(folder/'campaign.json')
    assert all(c['status']=='complete' and c['exit_code']==0 and not c['external_pids'] for c in result['campaign'])
    result['telemetry']={}
    for arm in ('wait','ar'):
        trs=[json.loads(line) for line in (folder/f'{arm}.telemetry.jsonl').read_text().splitlines()]
        by_gpu={}
        for t in trs:
            assert not t['foreign'], t
            for line in t['gpu'].splitlines():
                cols=[s.strip() for s in line.split(',')]
                by_gpu.setdefault(cols[0],[]).append([float(cols[1]),float(cols[3].split()[0])])
        result['telemetry'][arm]={gpu:dict(samples=len(rows),max_temperature_c=float(np.max(rows,axis=0)[0]),
            mean_sm_clock_mhz=float(np.mean(rows,axis=0)[1])) for gpu,rows in by_gpu.items()}
    result['timeline']={model:timeline(nova/'profiles_v2',model+'_optimized') for model in ('llama2','llama3')}
    result['padding']={}
    for model,fd in [('llama2',root/'llama2_full_v2'),('llama3',nova/'llama3_full_v2')]:
        result['padding'][model]={}
        for arm in ('optimized','packed','mixed'):
            d=a.load(fd/f'{arm}.json');samples=[]
            for s in d['steps']:
                if a.eligible(s,d['args']['batch']):
                    es=s['events'];samples.append([sum(e['valid_k']+1 for e in es),
                        es[0]['physical_batch_queries'],len(es)*(es[0]['physical_verify_width']+1)])
            result['padding'][model][arm]=dict(steps=len(samples),
                actual_queries=float(np.mean(samples,axis=0)[0]),
                physical_queries=float(np.mean(samples,axis=0)[1]),
                dense_proxy_queries=float(np.mean(samples,axis=0)[2]),
                caution='Different trajectories; descriptive shape counts, not paired causal speed evidence')
    args.output.write_text(json.dumps(result,indent=2))
    for arm,cells in result['repeat'].items():
        for name,c in cells.items():
            print(arm,name,json.dumps(c['metrics']))
            print('captures',c['capture']['capture_counts_in_phase'],'slow covered',c['capture']['slow_steps_overlapping_capture'])
    print('COMPARISONS',json.dumps(result['comparisons'],indent=2))
    print('PROVENANCE',json.dumps(result['provenance'],indent=2))


if __name__=='__main__':main()
