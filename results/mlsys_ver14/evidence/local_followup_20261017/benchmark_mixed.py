"""Randomized step-level prep A/B within one engine, diagnostic adapter only.

Swapping equivalent Python entry points happens outside timed engine.step.
One CUDA model, graph pool and allocator are shared, reducing process noise.
"""
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import benchmark_wall as wall
from ssd.engine.model_runner import ModelRunner
from ssd.engine.verifier import Verifier
from ssd.engine.helpers.p2_tree import TreeProxyCUDAGraph
from ssd.engine.llm_engine import LLMEngine

ROOT=Path('/data/chokwans99/duet-local-followup-20261017')


def reference(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/'code/baseline'/path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod


old_runner=reference('mixed_old_runner','ssd/ssd/engine/model_runner.py')
old_verifier=reference('mixed_old_verifier','ssd/ssd/engine/verifier.py')
old_proxy=reference('mixed_old_proxy','ssd/ssd/engine/helpers/p2_tree.py')
pairs=[(ModelRunner,'_run_tree_verify',old_runner.ModelRunner._run_tree_verify,ModelRunner._run_tree_verify),
       (Verifier,'verify',old_verifier.Verifier.verify,Verifier.verify),
       (Verifier,'_compute_and_send_proxy_tree',old_verifier.Verifier._compute_and_send_proxy_tree,Verifier._compute_and_send_proxy_tree),
       (TreeProxyCUDAGraph,'replay',old_proxy.TreeProxyCUDAGraph.replay,TreeProxyCUDAGraph.replay)]
rng=random.Random(187431)  # Private RNG; never changes proposal/verify RNG.
timed_step=LLMEngine.step


def step(self,*args,**kwargs):
    mode=rng.getrandbits(1)
    for cls,name,old,new in pairs:setattr(cls,name,new if mode else old)
    n=len(wall._wall_steps)
    result=timed_step(self,*args,**kwargs)
    if len(wall._wall_steps)==n+1:wall._wall_steps[-1]['prep_mode']=mode
    return result


LLMEngine.step=step
if __name__=='__main__':
    try:wall.main()
    finally:
        (Path(os.environ['SSD_PROFILE_DIR'])/'WALL_STEPS.json').write_text(json.dumps(wall._wall_steps)+'\n')
