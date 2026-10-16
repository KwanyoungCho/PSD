"""Refine large target-gap candidates from stored profiles; no model imports."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'ssd/tools/duet_timeline'))
from plot_consecutive_steps import read_profiles


def one(events,label):
    found=[e for e in events if e['label']==label]
    assert len(found)==1,(label,len(found))
    return found[0]


def main():
    with gzip.open(HERE/'DETAILS.json.gz','rt') as f:
        diagnosis=json.load(f)
    runs=[]
    for run in diagnosis['runs']:
        base=('/data/chokwans99/duet-local-phase-20261015' if run['mode']=='tree'
              else '/data/chokwans99/duet-chain40-k10-20261016')
        groups,_,hashes=read_profiles(Path(base)/run['run'])
        rows=[]
        for r in run['rows']:
            events=groups['target'][(r['epoch'],r['step_id'])]
            wait=one(events,'target_recv_response_wait')
            spec=one(events,'target_spec_wait')
            setup=one(events,'verify_setup')
            forward=one(events,'graph_pre')
            parts=dict(response_decode_ms=(spec['wall_end_ns']-wait['wall_end_ns'])/1e6,
                verifier_and_runner_input_ms=(setup['wall_start_ns']-spec['wall_end_ns'])/1e6,
                runner_setup_ms=(forward['wall_start_ns']-setup['wall_start_ns'])/1e6)
            assert min(parts.values())>=0,(r['epoch'],r['step_id'],parts)
            assert abs(sum(parts.values())-r['next_response_decode_and_verify_setup_ms'])<1e-8
            qsel=[e for e in events if e['label']=='tree_parent_q_select']
            # This interval includes callback setup + IPC + prepare_decode +
            # run dispatch, not just one CPU helper's isolated cost.
            if qsel:
                parts['after_q_selection_to_runner_setup_ms']=(setup['wall_start_ns']-qsel[0]['wall_end_ns'])/1e6
            major=(r['previous_accept_and_kv_ms']+r['next_response_block_ms']
                   +r['next_response_decode_and_verify_setup_ms'])
            rows.append(dict(state=r['state'],**parts,
                major_path_ms=major,target_gap_ms=r['target_forward_to_forward_gap_ms']))
        summaries=[]
        for state in (1,2,0,'all'):
            selected=[r for r in rows if state=='all' or r['state']==state]
            fields=sorted({k for r in selected for k in r if k.endswith('_ms')})
            s=dict(state=state,n=len(selected))
            for k in fields:
                vals=[r[k] for r in selected if k in r]
                s[k]=mean(vals)
                s[k+'_samples']=len(vals)
            s['major_path_fraction']=s['major_path_ms']/s['target_gap_ms']
            summaries.append(s)
        runs.append(dict(mode=run['mode'],run=run['run'],summary=summaries,trace_sha256=hashes))

    # Audit concrete source operations rather than interpreting all time as
    # a removable copy. Static call sites are not dynamic sync counts.
    sources=['ssd/ssd/engine/speculator_async.py','ssd/ssd/engine/verifier.py',
        'ssd/ssd/engine/draft_runner.py','ssd/ssd/engine/model_runner.py',
        'ssd/ssd/engine/helpers/runner_helpers.py',
        'ssd/ssd/utils/verify_fast.py','ssd/ssd/engine/helpers/p2_tree.py']
    operations=[]
    for rel in sources:
        p=ROOT/rel
        tree=ast.parse(p.read_text())
        selected_names={'extend_seqs_for_verify','_speculation_request','verify',
            '_service_spec_request','hit_cache_and_respond','prepare_decode',
            'prepare_decode_tensors_from_seqs','prepare_block_tables_from_seqs',
            'tree_verify_walk_tensor','verify_stochastic'}
        for node in ast.walk(tree):
            if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) or node.name not in selected_names:
                continue
            calls=[]
            for call in ast.walk(node):
                if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr in ('tolist','item','cpu','cuda'):
                    calls.append(dict(line=call.lineno,operation=ast.unparse(call.func)))
            operations.append(dict(file=rel,function=node.name,sites=sorted(calls,key=lambda x:x['line'])))
    result=dict(status='complete',gpu_used=False,runtime_changed=False,
        method='Existing trace partition checks and AST source audit',
        measurement_limit='Profile-ON elapsed spans; no verified optimization gain. Static sites are not per-step synchronization counts.',
        runs=runs,source_operations=operations,
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        prior_gpu_tree_accept_result='results/mlsys_ver01/REPORT.md: B1 GPU accept and precomputed-probability walk not consistently faster; not adopted')
    (HERE/'PRIORITY_REVIEW.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    for run in runs:
        s=next(s for s in run['summary'] if s['state']==1)
        print(run['mode'],run['run'],{k:round(v,4) for k,v in s.items()
            if isinstance(v,float) and not k.endswith('_samples')},flush=True)


if __name__=='__main__':main()
