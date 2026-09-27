"""Audit every observed tree/event, corpus/history, and deterministic raw snapshots."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from features import extract, groups
from calibrate import from_edges

HERE=Path(__file__).resolve().parent
FULL=HERE.parent/'duet_tree_al_full'
ROOT=HERE.parents[1]


def float32_walk_reach(z):
    """Mirror actual walk updates, including its 1e-12 guards, before normalization."""
    par=z['par'].tolist(); sib=z['sib'].tolist(); tokens=z['tok']; n=len(par)
    alpha=np.zeros(n)
    for parent,children in groups(par,sib).items():
        residual=z['p'][parent+1].copy(); draft=z['q'][children[0]].copy()
        for j in children:
            token=tokens[j]
            # An unreachable degenerate proposal has no effect on joint reach.
            a=min(1.,float(residual[token])/float(draft[token])) if draft[token]>0 else 0.
            alpha[j]=a
            residual=np.maximum(residual-draft,0); mass=float(residual.sum())
            residual=residual/mass if mass>1e-12 else np.zeros_like(residual)
            draft[token]=0;mass=float(draft.sum())
            draft=draft/mass if mass>1e-12 else np.zeros_like(draft)
    return from_edges(dict(n=n,par=par,sib=sib),alpha)[0]


def main():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('/home/chokwans99/awq_calibrated/layerskip_llama2_70b',local_files_only=True)
    plan=json.loads((FULL/'plan.json').read_text())
    questions=json.loads((FULL/'questions.json').read_text())
    assert hashlib.sha256((FULL/'questions.json').read_bytes()).hexdigest()==plan['questions_sha256']
    assert hashlib.sha256(Path(plan['calibration_path']).read_bytes()).hexdigest()==plan['calibration_sha256']
    expected=[f'{q["question_id"]}_t{t}' for q in questions for t in range(len(q['turns']))]
    lookup={q['question_id']:q for q in questions}; reports=[]
    for policy in ['q_path','phase_sibling_q_bin']:
        folder=HERE/'runs'/policy
        completion=json.loads((folder/'completion.json').read_text());assert completion['exit_code']==0
        command=json.loads((folder/'command.json').read_text())
        for rel,digest in command['sources'].items():
            if rel.endswith(('features.py','observe.py','core.py','score_hook.py','run_full.py','runtime.py')):
                assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
        log=(folder/'run.log').read_text()
        assert ('[tree score hook] phase_sibling_q_bin' in log)==(policy=='phase_sibling_q_bin')
        records=[json.loads(s) for s in (folder/'records.jsonl').read_text().splitlines()]
        assert [r['uid'] for r in records]==expected
        traces=[json.loads(s) for s in (folder/'trees.jsonl').read_text().splitlines()]
        assert [t['serial'] for t in traces]==list(range(len(traces)))
        indexed={(t['uid'],t['event_index']):t for t in traces};assert len(indexed)==len(traces)
        histories={};hit_count=0;raw_count=0;maxerror=0.;float32_error=0.
        for r in records:
            q=lookup[r['question_id']];turn=r['turn']
            current=tok.encode(q['turns'][turn],add_special_tokens=False)
            prompt=current if turn==0 else histories[q['question_id']]+tok.encode('\n\n',add_special_tokens=False)+current
            assert hashlib.sha256(json.dumps(prompt).encode()).hexdigest()==r['input_sha256']
            assert len(prompt)==r['input_tokens']
            ids=r['output_ids'];histories[q['question_id']]=prompt+ids
            assert 0<len(ids)<=128 and tok.eos_token_id not in ids[:-1]
            assert len(ids)==128 or ids[-1]==tok.eos_token_id
            events=r['metrics']['phase_events']
            if events:assert sum(e['accepted_len'] for e in events[:-1])<len(ids)<=sum(e['accepted_len'] for e in events)
            hits=0
            for ei,e in enumerate(events):
                if not e['cache_hit']:continue
                t=indexed[(r['uid'],ei)];hit_count+=1
                assert t['policy']==policy and t['hit_index']==hits;hits+=1
                assert t['question_id']==r['question_id'] and t['group']==r['group'] and t['turn']==turn
                assert t['phase']==e['source'] and t['n']==e['valid_k'] and t['observed_al']==e['accepted_spec_len']
                assert t['is_final_event']==(ei==len(events)-1)
                assert max(t['depth'])<={1:4,2:2}[t['phase']] and t['n']<={1:8,2:6}[t['phase']]
                rho,_=from_edges(t,np.asarray(t['alpha']))
                assert np.allclose(rho,t['reach'],atol=1e-10)
                assert abs(sum(t['terminal'])-1)<1e-8
                assert abs(sum(t['reach'])-t['true_al'])<1e-8
                assert ('raw_file' in t)==(t['serial']%32==0)
                if 'raw_file' in t:
                    f=folder/t['raw_file'];assert hashlib.sha256(f.read_bytes()).hexdigest()==t['raw_sha256']
                    with np.load(f) as z:
                        check=extract(**dict(z))
                        float32_error=max(float32_error,float(np.max(abs(float32_walk_reach(z)-t['reach']))))
                    for key,value in check.items():
                        error=float(np.max(np.abs(np.asarray(value)-np.asarray(t[key]))))
                        maxerror=max(maxerror,error)
                        assert error<1e-9,(key,error)
                    raw_count+=1
        assert hit_count==len(traces)
        assert float32_error<1e-4,('Material float32/double discrepancy',float32_error)
        reports.append(dict(policy=policy,questions=480,turns=560,trees=len(traces),nodes=sum(t['n'] for t in traces),
            phase_counts=dict(Counter(t['phase'] for t in traces)),raw_audits=raw_count,max_recompute_error=maxerror,
            maximum_float32_vs_normalized_double_reach_error=float32_error,
            records_sha256=hashlib.sha256((folder/'records.jsonl').read_bytes()).hexdigest(),
            trees_sha256=hashlib.sha256((folder/'trees.jsonl').read_bytes()).hexdigest()))
        print('audited',policy,'trees',len(traces),'raw',raw_count,flush=True)
    result=dict(passed=True,jobs=reports,checks=['all questions and actual multi-turn history',
        'natural EOS/cap and emitted suffix accounting','every hit mapped one-to-one to a tree',
        'phase, depth, node budgets and score hook','exact reach and terminal identities',
        'every deterministic raw snapshot re-derived','frozen calibration and collector source checksums'],
        limits=['Observer changes asynchronous timing. No TPS or paired-trajectory AL gain claim.',
            'Served trees only; unexpanded future distributions and unused root allocations are not observed.'])
    (HERE/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
