"""Passive full-vocabulary tree observer; reranking disabled (G=M).

Runs existing generation and verification without replacing their policies.
Diagnostic synchronization/I/O means timings are NOT performance results.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'ssd'))
sys.path.insert(0,str(ROOT/'ssd/bench'))

# Spawned workers import this module too; install before executor construction.
if os.environ.get('DUET_TREE_SCORE_MODE', 'q_path') != 'q_path':
    from score_hook import install
    install()


def main():
    import torch
    import bench
    import ssd.engine.helpers.p2_tree as p2
    from ssd.engine.verifier import Verifier
    dest=Path(os.environ['DUET_TREE_AUDIT_OUT']);dest.mkdir(exist_ok=True)
    plan=json.loads(Path(os.environ.get('DUET_TREE_AUDIT_PLAN',str(HERE/'probe_plan.json'))).read_text())
    state=dict(prompt=-1,phase=0,step=0)
    snapshots=[];records=[]
    original_walk=p2.tree_verify_walk_tensor
    original_method=Verifier._tree_verify_walk
    def method(self,result,*args,**kwargs):
        state['phase']=int(result.phase_source[0]) if result.phase_source is not None else 0
        return original_method(self,result,*args,**kwargs)
    Verifier._tree_verify_walk=method
    def walk(ti,p_logits,q_probs,temp,coin_fn,mult_fn):
        answer=original_walk(ti,p_logits,q_probs,temp,coin_fn,mult_fn)
        step=state['step'];state['step']+=1
        if state['prompt']<0 or step%plan['stride']:return answer
        n=int(ti['valid'])
        qref=ti['parent_q_ref'][:n].long().to(q_probs.device)
        p=torch.softmax(p_logits[:n+1].float()/float(temp),dim=-1).detach().cpu().numpy()
        q=q_probs.index_select(0,qref).detach().float().cpu().numpy()
        arrays=dict(p=p,q=q,par=np.asarray(ti['parent_local'][:n]),
                    sib=np.asarray(ti['sib_order'][:n]),tok=np.asarray(ti['tok'][:n]),
                    path=np.asarray(answer[0],dtype=np.int64))
        filename=f"prompt{state['prompt']:03d}_tree{step:04d}.npz"
        np.savez_compressed(dest/filename,**arrays)
        snapshots.append(dict(file=filename,prompt=state['prompt'],step=step,phase=state['phase'],
                              valid=n,temperature=float(temp),terminal=int(answer[1]),
                              sha256=hashlib.sha256((dest/filename).read_bytes()).hexdigest()))
        return answer
    p2.tree_verify_walk_tensor=walk
    original_bench=bench.run_benchmark
    def run(args,llm,prompts,sampling_params):
        if len(prompts)!=len(plan['mapping']):raise ValueError('Prompt count mismatch')
        if args.duet_p1_tree_verify_nodes!=args.duet_p1_tree_max_nodes or args.duet_p2_tree_verify_nodes!=args.duet_p2_tree_max_nodes:
            raise ValueError('This observer requires unpruned G=M trees')
        outputs=[]
        original_bench(args,llm,prompts[:1],sampling_params[:1])
        for i,(prompt,param) in enumerate(zip(prompts,sampling_params)):
            digest=hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
            if digest!=plan['mapping'][i]['token_sha256']:raise ValueError('Prompt hash mismatch')
            state.update(prompt=i,step=0)
            bench.reset_metrics()
            out,elapsed,metrics=original_bench(args,llm,[prompt],[param])
            emitted=sum(len(x['token_ids']) for x in out)
            if emitted!=args.output_len:raise ValueError('Output length mismatch')
            outputs.extend(out)
            records.append(dict(prompt=i,output_tokens=emitted,tree_hits=state['step'],
                                metrics=json.loads(json.dumps(metrics))))
            payload=dict(schema='tree_full_distribution_observer_v1',args=vars(args),
                         scope='Served trees only, G=M, diagnostic not TPS.',
                         records=records,snapshots=snapshots,plan=plan)
            tmp=dest/'manifest.tmp';tmp.write_text(json.dumps(payload,indent=2));tmp.replace(dest/'manifest.json')
            print('[tree observer]',i,emitted,'snapshots',len(snapshots),flush=True)
        state['prompt']=-1
        return outputs,1.0,records[-1]['metrics']  # Not a timing benchmark.
    bench.run_benchmark=run
    bench.main()


if __name__=='__main__':main()
