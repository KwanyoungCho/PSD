"""Associate current response wait with the preceding PS, not current PS."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'ssd/tools/duet_timeline'))
from plot_consecutive_steps import read_profiles,transition
from plot_model_wait_timelines import command_index
from plot_model_breakdowns import one


def elapsed(event):return (event['wall_end_ns']-event['wall_start_ns'])/1e6


def main():
    configs=[('chain',HERE,Path('/data/chokwans99/duet-chain40-k10-20261016'),{'7b','13b'}),
             ('tree',HERE.parent/'local_phase_latency_20261015',Path('/data/chokwans99/duet-local-phase-20261015'),{'7b_base','13b_base'})]
    result=[]
    for mode,folder,raw,names in configs:
        with gzip.open(folder/'SAMPLES.json.gz','rt') as f:runs=json.load(f)
        for run in runs:
            name=run.get('name',run.get('model'))
            if name not in names:continue
            groups,commands,hashes=read_profiles(raw/name);index=command_index(commands)
            rows=[]
            for s in run['rows']:
                epoch,sid=s['epoch'],s['step_id']
                tr=transition(groups,index,epoch,sid-1)
                t=groups['target'][(epoch,sid)];d=groups['draft'][(epoch,sid)]
                wait=one(t,'target_recv_response_wait');send=one(t,'target_send_request')
                ready=tr['draft_receive_ready_ns']
                overlap=max(0,min(ready,wait['wall_end_ns'])-wait['wall_start_ns'])/1e6
                row=dict(epoch=epoch,step_id=sid,uid=s['uid'],state=s['state'],
                    previous_ps_signed_late_ms=-tr['ready_margin_ms'],
                    previous_ps_positive_late_ms=max(0,-tr['ready_margin_ms']),
                    previous_ps_overlap_response_ms=overlap,
                    response_wait_ms=elapsed(wait),send_ms=elapsed(send),speculate_total_ms=s['target_wait_ms'],
                    response_after_ps_ready_ms=elapsed(wait)-overlap,
                    command_recv_ms=elapsed(tr['command']),request_receive_ms=elapsed(one(d,'draft_recv_request')),
                    draft_response_pack_ms=elapsed(one(d,'draft_response_pack')),
                    draft_response_send_ms=elapsed(one(d,'draft_send_response')),
                    chain_or_tree_lookup_ms=sum(elapsed(e) for e in d if e['label'].startswith('hit_cache_respond')),
                    target_q_receive_ms=sum(elapsed(e) for e in t if e['label'].startswith('target_recv_parent_q') or e['label']=='target_recv_chain_q'))
                assert row['response_after_ps_ready_ms']>=0
                rows.append(row)
            fields=[k for k in rows[0] if k.endswith('_ms')]
            summaries=[]
            for state in (1,2,0,'all'):
                for timing in ('all','previous_ps_ready','previous_ps_late'):
                    sub=[r for r in rows if (state=='all' or r['state']==state) and (
                        timing=='all' or (r['previous_ps_signed_late_ms']>0)==(timing=='previous_ps_late'))]
                    if not sub:continue
                    summaries.append(dict(state=state,timing=timing,n=len(sub),
                        **{k:float(np.mean([r[k] for r in sub])) for k in fields}))
            result.append(dict(mode=mode,run=name,rows=rows,summary=summaries,source_hashes=hashes))
            print(mode,name,json.dumps([r for r in summaries if r['timing']=='all']),flush=True)
    out=dict(note='Current wait is matched to previous PS. Overlap partitions are observational elapsed intervals, not isolated causal kernel costs. Tree and chain differ in K, roots, topology, payload and outputs.',runs=result)
    (HERE/'RESPONSE_WAIT_DIAGNOSIS.json.gz').write_bytes(gzip.compress(json.dumps(out).encode(),mtime=0))
    (HERE/'RESPONSE_WAIT_SUMMARY.json').write_text(json.dumps(dict(note=out['note'],runs=[{k:v for k,v in r.items() if k!='rows'} for r in result]),indent=2)+'\n')


if __name__=='__main__':main()
