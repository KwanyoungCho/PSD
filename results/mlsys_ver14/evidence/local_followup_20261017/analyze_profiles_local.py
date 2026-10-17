"""Native-event timing; no profile-ON value is used as adoption evidence."""
import json
from pathlib import Path
import sys
from collections import defaultdict
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'ssd/tools/duet_timeline'))
from plot_consecutive_steps import read_profiles

RAW=HERE/'raw/runs'
result={}
for folder in sorted(RAW.glob('*_profile')):
    if not (folder/'status.json').exists():continue
    if json.loads((folder/'status.json').read_text())['status']!='complete':continue
    d=json.loads((folder/'run.json').read_text());groups,_,hashes=read_profiles(folder)
    samples=[]
    for i,g in enumerate(d['groups']):
        ep=len(d['warm_groups'])+i
        for s in g['steps']:
            e=s['events'][0];sid=e['step_id']
            if sid<=10 or s['cap'] or s['clipped']:continue
            evs=groups['target'][(ep,sid)]
            by=defaultdict(list)
            for ev in evs:by[ev['label']].append(ev)
            one=lambda label:by[label][0]
            prev=[ev for ev in groups['target'][(ep,sid-1)] if ev['label']=='final_logits']
            row=dict(source=e['source'] if e['cache_hit'] else 0,valid=e['valid_k'],
                step_ms=1000*s['seconds'],target_call_accept_ms=1000*sum(s['verify']),
                response_ms=(one('target_recv_response_wait')['wall_end_ns']-one('target_send_request')['wall_end_ns'])/1e6,
                gap_ms=(one('graph_pre')['wall_start_ns']-prev[0]['wall_end_ns'])/1e6,
                response_to_forward_ms=(one('graph_pre')['wall_start_ns']-one('target_recv_response_wait')['wall_end_ns'])/1e6)
            for label,es in by.items():
                row[label+'_ms']=sum(ev['wall_end_ns']-ev['wall_start_ns'] for ev in es)/1e6
            samples.append(row)
    result[folder.name]=dict(source_hashes=hashes,summary={})
    for source in (1,2,0,'all'):
        ss=[x for x in samples if source=='all' or x['source']==source]
        keys=sorted({k for x in ss for k in x if k.endswith('_ms')})
        result[folder.name]['summary'][str(source)]=dict(n=len(ss),
            **{key:float(np.mean([x.get(key,0) for x in ss])) for key in keys})
    result[folder.name]['by_valid']={str(v):dict(n=len(ss),
        **{key:float(np.mean([x[key] for x in ss])) for key in ('graph_pre_ms','graph_post_ms','final_logits_ms','target_call_accept_ms')})
        for v in sorted({x['valid'] for x in samples if x['source'] in (1,2)})
        if (ss:=[x for x in samples if x['valid']==v and x['source'] in (1,2)])}
(HERE/'PROFILES.json').write_text(json.dumps(result,indent=2)+'\n')
for tag,d in result.items():
    ss=d['summary']['all']
    print(tag,{key:round(ss.get(key,0),4) for key in ('response_ms','gap_ms','graph_pre_ms','graph_post_ms','final_logits_ms','target_wait_parent_q_ready_ms')})
