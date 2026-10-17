"""Interleaved component measurements; run after GPU4 timing work ends."""
from pathlib import Path
import importlib.util
import json
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch

ROOT=Path('/data/chokwans99/duet-local-followup-20261017')
HERE=Path(__file__).resolve().parent


def reference(name,relative):
    spec=importlib.util.spec_from_file_location(name,ROOT/'code/baseline'/relative)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod


@torch.inference_mode()
def main():
    from ssd.engine.helpers.p2_tree import TreeProxyCUDAGraph,parse_tree_ints
    from ssd.engine.model_runner import ModelRunner
    from ssd.engine.helpers.root_policy import options
    old=reference('old_proxy_component','ssd/ssd/engine/helpers/p2_tree.py')
    old_runner=reference('old_runner_component','ssd/ssd/engine/model_runner.py')
    policy=options(SimpleNamespace(duet_root_source='complement',duet_root_normalization='topm',duet_root_overlap_mix=.25))
    results=[]
    for nv in (3,6,15):
        kwargs=dict(nv=nv,vocab_size=32000,wire_n=81,depth_steps=8,dtype=torch.bfloat16,device='cuda:0',top_k=81,c_max=3,policy=policy)
        a=old.TreeProxyCUDAGraph(**kwargs);b=TreeProxyCUDAGraph(**kwargs)
        par=[-1]+[(j-1)//2 for j in range(1,nv)];sib=[0]+[(j-1)%2 for j in range(1,nv)]
        a.prepare_topology(par,sib);b.prepare_topology(par,sib)
        x=torch.randn(nv+1,32000,device='cuda:0',dtype=torch.bfloat16)
        qref=[p+1 for p in par]
        q=torch.randn(nv,32000,device='cuda:0',dtype=torch.bfloat16)[qref]
        tokens=torch.arange(nv,device='cuda:0')
        for _ in range(5):a.replay(x,q,tokens,.7,.7);b.replay(x,q,tokens,.7,.7)
        torch.cuda.synchronize()
        for left,right in zip(a.replay(x,q,tokens,.7,.7),b.replay(x,q,tokens,.7,.7)):assert torch.equal(left,right)
        samples=[]
        for repeat in range(5):
            for label,g in [('a',a),('b',b),('b',b),('a',a)]:
                torch.cuda.synchronize();t=time.perf_counter()
                for _ in range(20):g.replay(x,q,tokens,.7,.7)
                torch.cuda.synchronize()
                samples.append(dict(repeat=repeat,arm=label,ms=(time.perf_counter()-t)*1000/20))
        old_ms=np.mean([s['ms'] for s in samples if s['arm']=='a']);new_ms=np.mean([s['ms'] for s in samples if s['arm']=='b'])
        serial=[]
        for repeat in range(5):
            for label,g in [('a',a),('b',b),('b',b),('a',a)]:
                t=time.perf_counter()
                for _ in range(10):
                    g.replay(x,q,tokens,.7,.7)
                    torch.cuda.synchronize()
                serial.append(dict(repeat=repeat,arm=label,ms=(time.perf_counter()-t)*1000/10))
        # Measure actual runner code until the same deliberate invalid-KV
        # boundary, so no transformer/GPU work contaminates validation cost.
        wire=tuple([nv,max(qref)+1,1]+list(range(nv))+par+sib+qref)
        parsed=parse_tree_ints(wire,nv)
        cfg=SimpleNamespace(duet_tree_wire_nodes=nv,hf_config=SimpleNamespace(vocab_size=32000),duet_tree_c_tensor=3)
        owner=SimpleNamespace(config=cfg,_duet_tree_meta=wire,_duet_validated_tree=(wire,parsed),_duet_verify_context_len_cpu=0)
        ids=torch.zeros(nv+1,dtype=torch.int64,device='cpu')
        cpu=[]
        for repeat in range(5):
            for label,cls in [('a',old_runner.ModelRunner),('b',ModelRunner),('b',ModelRunner),('a',old_runner.ModelRunner)]:
                t=time.perf_counter()
                for _ in range(1000):
                    try:cls._run_tree_verify(owner,ids,None,False)
                    except RuntimeError as exc:assert 'invalid KV geometry' in str(exc)
                cpu.append(dict(repeat=repeat,arm=label,us=(time.perf_counter()-t)*1e6/1000))
        results.append(dict(nodes=nv,proxy_replay_before_ms=float(old_ms),proxy_replay_after_ms=float(new_ms),
            proxy_reduction_pct=float(100*(1-new_ms/old_ms)),proxy_bitwise_equal=True,proxy_samples=samples,
            serial_proxy_before_ms=float(np.mean([s['ms'] for s in serial if s['arm']=='a'])),
            serial_proxy_after_ms=float(np.mean([s['ms'] for s in serial if s['arm']=='b'])),serial_samples=serial,
            validation_before_us=float(np.mean([s['us'] for s in cpu if s['arm']=='a'])),
            validation_after_us=float(np.mean([s['us'] for s in cpu if s['arm']=='b'])),validation_samples=cpu))
    (HERE/'COMPONENTS.json').write_text(json.dumps(dict(status='pass',results=results,
        note='Interleaved hot fixed-input component timings, not whole-model improvement percentages.'),indent=2)+'\n')


if __name__=='__main__':main()
