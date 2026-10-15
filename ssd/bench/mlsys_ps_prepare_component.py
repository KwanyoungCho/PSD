"""Matched validity-check latency; reference method loaded from frozen source."""
import argparse, ast, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ssd.engine.draft_runner import DraftRunner, _P2StepState
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    source=a.baseline/'ssd/ssd/engine/draft_runner.py'
    tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DraftRunner')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_p2_step_state')
    method.decorator_list=[];ns={'torch':torch,'_P2StepState':_P2StepState}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),ns)
    old=ns['_p2_step_state'];new=DraftRunner._p2_step_state
    r=object.__new__(DraftRunner);r.device=torch.device('cuda');r.config=SimpleNamespace(max_model_len=2048)
    r.block_size=256;r.kv_cache=torch.empty(2,1,16,256,1,1,device='cuda')
    rows=[]
    for F,W,R in ((3,15,10),(6,26,20),(8,32,32)):
        args=dict(rope0=torch.full((W,),510,dtype=torch.int64,device='cuda'),
                  step_slot_maps=torch.arange(F*W,device='cuda').reshape(F,W)+512,
                  step_context_lens=torch.full((F,1),512,dtype=torch.int64,device='cuda'),
                  dbt=torch.arange(8,dtype=torch.int32,device='cuda')[None,:],F=F,W=W,root_count=R)
        for _ in range(25):
            assert old(r,**args)==new(r,**args)
        # Paired AB/BA, one readback per invocation; no model/other process on this GPU.
        samples={'before':[],'after':[]}
        for rep in range(100):
            for name,fn in ([('before',old),('after',new)] if rep%2==0 else [('after',new),('before',old)]):
                torch.cuda.synchronize();start=time.perf_counter_ns()
                for _ in range(8):fn(r,**args)
                torch.cuda.synchronize();samples[name].append((time.perf_counter_ns()-start)/1e6/8)
        rows.append(dict(F=F,W=W,R=R,samples_ms=samples,mean_ms={k:float(np.mean(v)) for k,v in samples.items()},median_ms={k:float(np.median(v)) for k,v in samples.items()}))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(rows=rows,baseline_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(DraftRunner._p2_step_state.__code__.co_filename),Path(__file__).parents[1]/'ssd/engine/helpers/p2_prepare.py']},
        cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),gpu=torch.cuda.get_device_name()),indent=2))
    print([{k:v for k,v in r.items() if k!='samples_ms'} for r in rows])

if __name__=='__main__':main()
