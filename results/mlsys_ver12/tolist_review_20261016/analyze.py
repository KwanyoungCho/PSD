"""Matched before/after component latency; no latency-based exclusions."""
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
RAW=Path('/data/chokwans99/duet-tolist-review-20261016')
sys.path.insert(0,str(ROOT/'ssd/tools/duet_timeline'))
from plot_consecutive_steps import read_profiles
from plot_model_breakdowns import one
from plot_model_wait_timelines import command_index
from export_profile_timelines import normalize_outcomes,preparation_intervals,preparation_affected
from phase_averages import sample,metrics

FIELDS=['fanout_cpu_ms','state_cpu_ms','phase2_build_ms','dispatch_gap_ms','prepare_a_ms','prepare_b_ms','pre_forward_ms','ps_graph_ms','ps_total_ms','ps_margin_ms','step_ms']


def read(name):
    folder=RAW/'runs'/name;run=json.loads((folder/'run.json').read_text());assert run['status']=='complete'
    assert len(run['groups'])==len(run['outputs'])==24
    outcomes=normalize_outcomes(run);groups,commands,hashes=read_profiles(folder)
    index=command_index(commands);preps=preparation_intervals(folder)
    audits={}
    for path in folder.glob('fanout_audit_request_*.json'):
        data=json.loads(path.read_text());by=defaultdict(dict)
        for r in data['rows']:
            assert r['kind'] not in by[r['step_id']]
            by[r['step_id']][r['kind']]=r
        for sid,r in by.items():audits[data['epoch'],sid]=r
    excluded=Counter();rows=[]
    for q in outcomes['requests']:
        for i,s in enumerate(q['steps']):
            key=q['epoch'],s['step_id'];a=audits[key]
            assert set(a)=={'fanout_layout','state_readback'}
            assert a['fanout_layout']['host_list']==name.startswith('before')
            assert a['fanout_layout']['width']==9
            reason=('warmup' if q['warmup'] else 'terminal' if i==len(q['steps'])-1
                    else 'cap_clip' if s['cap'] or s['clipped'] else 'first10' if i<10
                    else 'preparation' if preparation_affected(groups,*key,preps) else None)
            if reason:excluded[reason]+=1;continue
            row=dict(epoch=key[0],step_id=key[1],uid=q['uid'],state=s['source'],next_state=q['steps'][i+1]['source'],
                accepted_len=s['accepted_len'],emitted_len=s['emitted_len'],verify_width=s['verify_width'],context_tokens=s['context_tokens'])
            row=sample(groups,index,row,3,3,name);row.update(metrics(row))
            d=groups['draft'][key];b=one(d,'phase2_build');pw=one(d,'proxy_wait');graph=one(d,'p2_graph_replay')
            prep=sorted([e for e in d if e['label']=='p2_prepare'],key=lambda e:e['wall_start_ns']);assert len(prep)==2
            ms=lambda x,y:(y-x)/1e6
            row.update(fanout_cpu_ms=a['fanout_layout']['cpu_ms'],state_cpu_ms=a['state_readback']['cpu_ms'],
                phase2_build_ms=ms(b['wall_start_ns'],b['wall_end_ns']),dispatch_gap_ms=ms(b['wall_end_ns'],prep[0]['wall_start_ns']),
                prepare_a_ms=ms(prep[0]['wall_start_ns'],prep[0]['wall_end_ns']),prepare_b_ms=ms(prep[1]['wall_start_ns'],prep[1]['wall_end_ns']),
                pre_forward_ms=ms(pw['wall_end_ns'],graph['wall_start_ns']),ps_graph_ms=ms(graph['wall_start_ns'],graph['wall_end_ns']))
            rows.append(row)
    for path in [folder/'run.json',folder/'COMMAND.json',folder/'SOURCE_MANIFEST.json',*folder.glob('fanout_audit_request_*.json')]:
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(name=name,rows=rows,excluded=dict(excluded),outputs=run['outputs'],
        source_hashes=hashes,summary=run['summary'],source_audited_steps=len(audits))


def summarize(pairs):
    result=[]
    for state in (1,2,0,'all'):
        sub=[(a,b) for a,b in pairs if state=='all' or a['state']==state]
        uids=sorted({a['uid'] for a,b in sub});weights=np.random.default_rng(9101).multinomial(len(uids),np.ones(len(uids))/len(uids),size=10000)
        counts=np.asarray([sum(a['uid']==u for a,b in sub) for u in uids])
        fields={}
        for key in FIELDS:
            before=float(np.mean([a[key] for a,b in sub]));after=float(np.mean([b[key] for a,b in sub]))
            totals=np.asarray([sum(b[key]-a[key] for a,b in sub if a['uid']==u) for u in uids])
            den=weights@counts;boot=(weights@totals)/den
            fields[key]=dict(before=before,after=after,delta=after-before,
                change_pct=100*(after/before-1) if before else None,delta95=np.quantile(boot,[.025,.975]).tolist())
        result.append(dict(state=state,n=len(sub),independent_uids=len(uids),fields=fields))
    return result


def main():
    runs=[read(n) for n in ('before1','after1','after2','before2')]
    by={r['name']:r for r in runs};pairs=[];comparisons=[]
    for a,b in [('before1','after1'),('before2','after2')]:
        left,right=by[a],by[b];assert left['outputs']==right['outputs'],f'Output mismatch {a}/{b}'
        x={(r['epoch'],r['step_id']):r for r in left['rows']};y={(r['epoch'],r['step_id']):r for r in right['rows']}
        common=sorted(x.keys()&y.keys());pp=[(x[k],y[k]) for k in common]
        assert all((a['state'],a['accepted_len'],a['verify_width'],a['context_tokens'])==(b['state'],b['accepted_len'],b['verify_width'],b['context_tokens']) for a,b in pp)
        pairs.extend(pp);comparisons.append(dict(before=a,after=b,outputs_equal=True,matched_steps=len(pp),summary=summarize(pp)))
    out=dict(status='complete',model='LayerSkip-Llama2-13B + JackFram/llama-68m',
        schedule=['before1','after1','after2','before2'],paired_outputs='48/48 output arrays identical',
        matched_steps=len(pairs),independent_input_turns=12,
        bootstrap='10000 UID-clustered paired draws, keeping passes and A/B repetitions together; observational run variation remains',
        runs=[{k:v for k,v in r.items() if k not in ('outputs','rows')} for r in runs],
        comparisons=comparisons,combined=summarize(pairs))
    (HERE/'SUMMARY.json').write_text(json.dumps(out,indent=2)+'\n')
    (HERE/'PAIRED_SAMPLES.json.gz').write_bytes(gzip.compress(json.dumps(dict(runs=runs,pairs=pairs)).encode(),mtime=0))
    for c in comparisons:
        print(c['before'],c['after'],c['summary'][-1]['fields'])
    print('COMBINED',out['combined'][-1]['fields'])


if __name__=='__main__':main()
