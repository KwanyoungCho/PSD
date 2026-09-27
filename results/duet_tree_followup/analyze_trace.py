"""Recompute every traced frontier choice and allocation on CPU."""
import ast
from collections import Counter
import json
import math
from pathlib import Path
from types import SimpleNamespace
import torch
from allocation import allocate,make_constants
from policy_hook import POLICIES

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def main():
    torch.set_num_threads(1)
    source=ROOT/'ssd/ssd/engine/helpers/p2_tree.py';tree=ast.parse(source.read_text())
    names=('_arena_select_global','_arena_fanout_global');scope=dict(torch=torch,math=math)
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            for arg in node.args.args:arg.annotation=None
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==names[0])
    modified=ast.unparse(node).replace('ar.depth == f','ar.depth <= f');ns=dict(scope)
    exec(compile(modified,str(source)+':frontier','exec'),ns)
    calibration=json.loads((HERE/'gain_calibration.json').read_text());summary={}
    for policy in ('reach','reach_gain_frontier'):
        folder=HERE/'runs'/('trace_'+policy)
        cp=folder/'completion.json'
        if not cp.exists() or json.loads(cp.read_text())['exit_code']!=0:raise ValueError('Incomplete full frontier trace '+policy)
        valid=json.loads((folder/'validated.json').read_text())
        if valid['questions']!=480 or valid['turns']!=560:raise ValueError('Traced corpus is incomplete')
        stats=Counter();phases={1:Counter(),2:Counter()};traces=0
        paths=list(folder.glob('frontier.*.jsonl'))
        if not paths:raise ValueError('No frontier trace')
        for path in paths:
            for line in path.open():
                data=json.loads(line);phase=int(data['phase']);counts=phases[phase];traces+=1
                r=len(data['root_prior']);b=data['max_nodes'];F=data['depth_cap'];cap=len(data['valid'][0])
                const=make_constants(b,calibration['curves'][str(phase)])
                for f,w in enumerate(data['widths']):
                    ar=SimpleNamespace(capacity=cap,device='cpu',n=torch.tensor(data['n'][f]),
                        **{k:torch.tensor(data[k][f],dtype=torch.bool if k=='valid' else torch.double if k in ('raw_q','logpri') else torch.long)
                           for k in ('valid','state','depth','root','raw_q','logpri')})
                    remaining=torch.tensor(data['remaining'][f]);thresholds=(0.,0.) if phase==1 else (.01,.03)
                    fn=ns[names[0]] if POLICIES[policy]['frontier'] else scope[names[0]]
                    sel,valid=fn(ar,w,f,F,remaining,F-f-1,r,*thresholds)
                    actual=torch.tensor(data['sel'][f][:w]);av=torch.tensor(data['selected'][f][:w])
                    if not torch.equal(sel,actual) or not torch.equal(valid,av):raise ValueError('Traced frontier choice mismatch')
                    if POLICIES[policy]['gain']:
                        fan=allocate(ar.root[sel],ar.logpri[sel],valid,remaining,F-f-1,const)
                    else:fan=scope[names[1]](ar,sel,valid,remaining,3,r,F-f-1)
                    if not torch.equal(fan,torch.tensor(data['fanout'][f][:w])):raise ValueError('Traced fanout mismatch')
                    counts.update(rounds=1,physical_forward_lanes=w,selected_parents=int(valid.sum()),
                        zero_child_selected=int((valid&(fan==0)).sum()),children=int(fan.sum()),
                        delayed_expansions=int((valid&(ar.depth[sel]<f)).sum()))
                    live=(torch.arange(cap)<ar.n)&ar.valid&(ar.state==0)&(ar.depth<F)
                    counts['unexpanded_below_depth_cap']+=int(live.sum())
                    counts['older_frontier_nodes']+=int((live&(ar.depth<f)).sum())
                    conf=live&(ar.depth<=f)
                    if f and thresholds[1]>0:
                        counts['raw_q_gate_exclusions']+=int((conf&(ar.raw_q<thresholds[1])).sum())
                    used=torch.zeros(r,dtype=torch.long).scatter_add_(0,ar.root[sel],fan)
                    if (used>remaining).any():raise ValueError('Root node capacity exceeded')
                stats['forests']+=1
        for counts in phases.values():stats.update(counts)
        # Existing actual runner audit independently checks samples/q references.
        audits=[json.loads(line) for path in folder.glob('node_audit.*.jsonl') for line in path.open()]
        if not audits or any(not a.get('passed',True) for a in audits):raise ValueError('Node audit missing/failed')
        summary[policy]=dict(totals=dict(stats),phases={str(k):dict(v) for k,v in phases.items()},node_audit_rows=len(audits))
    result=dict(passed=True,complete=True,summary=summary,
        scope='All traced forests, including unhit roots; choice/shape/proposal audit. Missing-branch target p/q is not supplied by this trace.')
    (HERE/'frontier_analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
