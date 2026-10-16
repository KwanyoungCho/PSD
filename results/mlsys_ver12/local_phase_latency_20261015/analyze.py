"""Recompute phase timing and node statistics from preserved native profiles."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys
import numpy as np
import hashlib

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'ssd/tools/duet_timeline'))
from plot_consecutive_steps import read_profiles
from plot_model_wait_timelines import command_index
from phase_averages import sample, metrics, summarize
from export_profile_timelines import normalize_outcomes, preparation_intervals, preparation_affected


def stats(values):
    x = np.asarray(values, dtype=float)
    if not len(x): return dict(n=0, mean=None, min=None, p05=None, p50=None, p90=None, p95=None, max=None)
    return dict(n=len(x), mean=float(x.mean()), min=float(x.min()),p05=float(np.quantile(x,.05)),p50=float(np.median(x)),
                p90=float(np.quantile(x,.90)), p95=float(np.quantile(x,.95)), max=float(x.max()))


def analyze(folder):
    run=json.loads((folder/'run.json').read_text())
    outcomes=normalize_outcomes(run)
    groups, commands, hashes=read_profiles(folder)
    preparations=preparation_intervals(folder)
    index=command_index(commands)
    probe={}
    for p in sorted(folder.glob('phase_probe_request_*.json')):
        raw=json.loads(p.read_text())
        for rec in raw['records']:
            key=(raw['epoch'],rec['step_id'],rec['phase'])
            if key in probe: raise AssertionError(('Duplicate probe',key))
            probe[key]=rec
    for p in [folder/'run.json',folder/'COMMAND.json',*folder.glob('phase_probe_request_*.json'),*folder.glob('preparation_*.jsonl')]:
        hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    expected={(r['epoch'],s['step_id']) for r in outcomes['requests'] for s in r['steps']}
    assert expected == set(groups['target']) == set(groups['draft'])
    skipped=Counter(); rows=[]
    for request in outcomes['requests']:
        ep=request['epoch'];steps=request['steps']
        for i, step in enumerate(steps):
            reason=('warmup' if request['warmup'] else 'initial10' if i<10 else
                    'terminal' if i==len(steps)-1 else 'cap_clip' if step['cap'] or step['clipped'] else None)
            if reason: skipped[reason]+=1;continue
            sid=step['step_id']
            if preparation_affected(groups,ep,sid,preparations):
                skipped['preparation']+=1;continue
            evs=groups['target'][(ep,sid)]+groups['draft'][(ep,sid)]
            if any('capture' in e['label'] for e in evs):
                skipped['capture']+=1;continue
            row=dict(epoch=ep,step_id=sid,uid=request['uid'],state=step['source'],next_state=steps[i+1]['source'])
            row.update({k:step[k] for k in ('accepted_len','emitted_len','context_tokens','verify_width') if k in step})
            value=sample(groups,index,row,outcomes['k1'],outcomes['k2'],folder.name)
            value.update(metrics(value))
            value['hit_nodes']=step['verify_width'] if step['source'] in (1,2) else None
            for phase,prefix in (('p1','ds'),('p2','ps')):
                if not probe:continue
                rec=probe[(ep,sid,phase)]
                value[prefix+'_probe']=rec
                forward=sum(f['forward_ms'] for f in rec['forward'])
                graph=(value[prefix+'_graph_end_ns']-value[prefix+'_graph_start_ns'])/1e6
                assert forward <= graph+0.02, (folder,ep,sid,phase,forward,graph)
                value[prefix+'_forward_ms']=forward
                value[prefix+'_forward_once_ms']=forward/len(rec['forward'])
                value[prefix+'_graph_other_ms']=graph-forward
                value[prefix+'_nonforward_ms']=value[prefix+'_total_ms']-forward
                if prefix=='ps':
                    value[prefix+'_nonforward_ms']+=(value['draft_ready_ns']-value['ps_end_ns'])/1e6
                value[prefix+'_compute_window_ms']=forward+value[prefix+'_margin_ms']
            rows.append(value)
    summary=[]
    for state in (1,2,0,'all'):
        sub=[r for r in rows if state=='all' or r['state']==state]
        keys=['step_ms','ds_margin_ms','ps_margin_ms','context_tokens','verify_width']
        if probe:
            keys += [p+'_'+f for p in ('ds','ps') for f in
                     ('forward_ms','forward_once_ms','graph_other_ms','nonforward_ms','compute_window_ms')]
        item=dict(state=state,steps=len(sub),metrics={k:stats([r[k] for r in sub]) for k in keys})
        item['negative_margin_steps']={phase:sum(r[phase+'_margin_ms']<0 for r in sub) for phase in ('ds','ps')}
        if probe:
            for prefix in ('ds','ps'):
                recs=[r[prefix+'_probe'] for r in sub]
                item[prefix]=dict(all_root_nodes=stats([n for r in recs for n in r['nodes']]),
                    active_roots=stats([r['active_roots'] for r in recs]),
                    root_node_histogram=dict(Counter(n for r in recs for n in r['nodes'])),
                    physical_widths=dict(Counter(str(r['round_widths']) for r in recs)),
                    rounds=[dict(round=f+1,**{k:stats([r['forward'][f][k] for r in recs])
                        for k in ('width','model_ms','head_ms','forward_ms')}) for f in range(3)])
        item['hit_tree_nodes']=stats([r['hit_nodes'] for r in sub if r['hit_nodes'] is not None])
        summary.append(item)
    return dict(name=folder.name,args=run['args'],source_hashes=hashes,
                eligibility=dict(skipped=dict(skipped),included=len(rows)),
                performance=summarize(rows),summary=summary,rows=rows,
                outputs=run['outputs'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,default=Path('/data/chokwans99/duet-local-phase-20261015'))
    a=p.parse_args();results=[]
    for folder in sorted(a.raw.iterdir()):
        if not folder.name.startswith(('7b_','13b_')) or not (folder/'run.json').exists():continue
        if json.loads((folder/'run.json').read_text())['status']!='complete':continue
        result=analyze(folder);results.append(result)
        print(folder.name,result['eligibility'],flush=True)
    if not results: raise RuntimeError('No completed campaign runs')
    (HERE/'SAMPLES.json.gz').write_bytes(gzip.compress(json.dumps(results).encode(),mtime=0))
    (HERE/'SUMMARY.json').write_text(json.dumps([{k:v for k,v in r.items() if k not in ('rows','outputs')} for r in results],indent=2)+'\n')
    parity={}
    for model in ('7b','13b'):
        arms={r['name']:r for r in results}
        if model+'_control' in arms and model+'_base' in arms:
            left=arms[model+'_control']['outputs'];right=arms[model+'_base']['outputs']
            parity[model]=dict(requests=len(left),identical=sum(x==y for x,y in zip(left,right)),same_count=len(left)==len(right))
    (HERE/'PROBE_PARITY.json').write_text(json.dumps(parity,indent=2)+'\n')


if __name__=='__main__':main()
