"""Whole-step matched A/B and source-specific native-event diagnostics."""
import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
from statistics import mean
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/'ssd/tools/duet_timeline'))
from plot_consecutive_steps import read_profiles


def module(name,filename):
    path=REPO/'results/mlsys_ver12/target_gap_optimization_20261017'/filename
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m);return m


def profile(folder):
    data=json.loads((folder/'run.json').read_text());groups,_,hashes=read_profiles(folder)
    rows=[]
    for i,group in enumerate(data['groups']):
        ep=len(data['warm_groups'])+i
        for s in group['steps']:
            event=s['events'][0];sid=event['step_id']
            if sid<=10 or s['cap'] or s['clipped'] or s['batch']!=1:continue
            target=groups['target'][(ep,sid)];draft=groups['draft'][(ep,sid)]
            def target_event(label,previous=False):
                evs=groups['target'][(ep,sid-1)] if previous else target
                es=[e for e in evs if e['label']==label]
                if len(es)!=1:raise ValueError((ep,sid,label,len(es)))
                return es[0]
            points=[target_event('final_logits',True)['wall_end_ns'],
                    target_event('target_send_request')['wall_start_ns'],
                    target_event('target_send_request')['wall_end_ns'],
                    target_event('target_recv_response_wait')['wall_end_ns'],
                    target_event('graph_pre')['wall_start_ns']]
            assert all(b>=a for a,b in zip(points,points[1:])),(folder,ep,sid,points)
            row=dict(uid=group['index'],step_id=sid,state=event['source'] if event['cache_hit'] else 0,
                whole_gap_ms=(points[-1]-points[0])/1e6,response_ms=(points[3]-points[2])/1e6,
                response_to_forward_ms=(points[4]-points[3])/1e6,step_ms=s['seconds']*1000)
            for side,evs in [('target',target),('draft',draft)]:
                for e in evs:
                    if side=='draft' and not any(k in e['label'] for k in ('hit','response','send','restore','glue','phase1','phase2','p1_graph','p2_graph')):continue
                    row[side+'_'+e['label']+'_ms']=(e['wall_end_ns']-e['wall_start_ns'])/1e6
            rows.append(row)
    summaries=[]
    keys=set(k for r in rows for k in r if k.endswith('_ms'))
    for state in (1,2,0,'all'):
        selected=[r for r in rows if state=='all' or r['state']==state]
        if not selected:continue
        result=dict(state=state,n=len(selected),component_samples={})
        for key in sorted(keys):
            vals=[r[key] for r in selected if key in r]
            if vals:result[key]=mean(vals);result['component_samples'][key]=len(vals)
        summaries.append(result)
    parity_path=folder/'PARITY.json'
    parity=json.loads(parity_path.read_text()) if parity_path.exists() else None
    return dict(tag=folder.name,profile_on=True,summary=summaries,source_sha256=hashes,
                reference_parity=parity,
                warning='Nested markers are not additive. Sender enqueue spans are not communication-complete spans.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,default=HERE/'raw');a=p.parse_args()
    step=module('prior_matched_steps','analyze_steps.py');profiles=[];comparisons=[];wall=[];excluded=[]
    for folder in sorted((a.raw/'runs').glob('*_profile')):
        if (folder/'status.json').exists() and json.loads((folder/'status.json').read_text())['status']=='complete':
            profiles.append(profile(folder))
    for model in ('7b','13b'):
        for variant in ('send','early','prepack'):
            if all((a.raw/'runs'/f'{v}_{model}_final_{rep}'/'status.json').exists()
                   for v in ('before',variant) for rep in ('a','b')):
                for filtered in (False,True):
                    comparisons.append(step.compare(a.raw,model,variant,filtered,stage='final'))
                    wall.append(compare_wall(a.raw,model,variant,'final',filtered,step))
        if all((a.raw/'runs'/tag/'status.json').exists() for tag in (
            f'before_{model}_final_b',f'hostlist_{model}_host_a',
            f'hostlist_{model}_host_b',f'before_{model}_host_b')):
            parity=[json.loads((a.raw/'runs'/f'hostlist_{model}_host_{rep}'/'PARITY.json').read_text())
                    for rep in ('a','b')]
            if not all(x['outputs_equal'] and x['events_equal'] for x in parity):
                excluded.append(dict(model=model,variant='hostlist',reason='Output/event parity failed; entire model/variant comparison excluded, no matched-prefix cherry-picking',parity=parity))
                continue
            original_load=step.load
            def alias_load(root,tag):
                if tag==f'before_{model}_host_a':tag=f'before_{model}_final_b'
                return original_load(root,tag)
            step.load=alias_load
            try:
                for filtered in (False,True):
                    row=step.compare(a.raw,model,'hostlist',filtered,stage='host')
                    row['first_before_tag']=f'before_{model}_final_b'
                    comparisons.append(row)
                    wall.append(compare_wall(a.raw,model,'hostlist','host',filtered,step))
            finally:step.load=original_load
    (HERE/'PROFILES.json').write_text(json.dumps(dict(runs=profiles),indent=2)+'\n')
    (HERE/'STEP_COMPARISONS.json').write_text(json.dumps(dict(comparisons=comparisons,excluded_comparisons=excluded,
        primary='Profile OFF full decode-step, exact outputs and source/AL parity, raw and 2x-filtered matched intersections'),indent=2)+'\n')
    (HERE/'WALL_COMPARISONS.json').write_text(json.dumps(dict(comparisons=wall),indent=2)+'\n')
    for item in profiles:
        allrow=next(r for r in item['summary'] if r['state']=='all')
        print(item['tag'],{k:round(allrow[k],4) for k in ('response_ms','whole_gap_ms','step_ms')})
    for r in comparisons:
        if r['cohort'].startswith('2x'):print(r['model'],r['variant'],r['reduction_pct'],r['reduction_pct_ci95'])


def wall_index(folder,data):
    wall=json.loads((folder/'WALL_STEPS.json').read_text())
    groups=data['warm_groups']+data['groups'];out={};i=0
    assert len(wall)==sum(len(g['steps']) for g in groups),(folder,len(wall))
    for g in groups:
        for s in g['steps']:
            w=wall[i];i+=1
            assert w['native_seconds']==s['seconds']
            assert w['source']==[e.get('source') for e in s['events']]
            assert w['accepted_len']==[e.get('accepted_len') for e in s['events']]
            assert w['wall_seconds']>=w['native_seconds']
            key=(g['phase'],g['index'],s['events'][0]['step_id'])
            out[key]=w['wall_seconds']
    return out


def compare_wall(root,model,variant,stage,filtered,step):
    totals=[0.,0.,0];repeats=[]
    for rep in ('a','b'):
        atag=f'before_{model}_{stage}_{rep}'
        if stage=='host' and rep=='a':atag=f'before_{model}_final_b'
        btag=f'{variant}_{model}_{stage}_{rep}'
        aa=step.load(root,atag);bb=step.load(root,btag)
        ai=step.index_steps(aa,filtered);bi=step.index_steps(bb,filtered)
        aw=wall_index(root/'runs'/atag,aa);bw=wall_index(root/'runs'/btag,bb)
        keys=ai.keys()&bi.keys();a=sum(aw[k] for k in keys);b=sum(bw[k] for k in keys);n=len(keys)
        totals=[x+y for x,y in zip(totals,(a,b,n))]
        repeats.append(dict(repeat=rep,n=n,before_ms=1000*a/n,after_ms=1000*b/n,
                            reduction_pct=100*(1-b/a)))
    a,b,n=totals
    return dict(model=model,variant=variant,stage=stage,filtered=filtered,n=n,
        before_ms=1000*a/n,after_ms=1000*b/n,reduction_pct=100*(1-b/a),repeats=repeats,
        observer='Independent perf_counter around the complete LLMEngine.step; no GPU synchronization')


if __name__=='__main__':main()
