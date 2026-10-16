"""Fixed inputs: original versus device-only layout update in one process."""
import ast
import json
import os
from pathlib import Path
import sys
import time
import types
import numpy as np

HERE=Path(__file__).resolve().parent
RAW=Path('/data/chokwans99/duet-tolist-review-20261016')
sys.path.insert(0,str(RAW/'after/ssd'))
import torch
from torch.profiler import profile,ProfilerActivity
from ssd.engine.draft_runner import DraftRunner


def before_method():
    path=RAW/'before/ssd/ssd/engine/draft_runner.py'
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='DraftRunner')
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_update_phase2_layout_inplace')
    ns=dict(torch=torch,os=os)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    return ns[node.name]


def main():
    torch.set_num_threads(1)
    r=object.__new__(DraftRunner)
    r.config=types.SimpleNamespace(duet_p2_tree_policy='on',use_eagle=False,
        stochastic_only=True,greedy_only=False)
    r.split_k2_layout=types.SimpleNamespace(MQ_LEN=9)
    funcs=dict(before=before_method(),after=DraftRunner._update_phase2_layout_inplace)
    counts=torch.tensor([[2,0,3,4]],device='cuda')
    os.environ.update(SSD_TREE_EXEC='1',SSD_TREE_ARENA='1',SSD_TREE_STAGE1='0',SSD_TREE_LAB='0')
    for fn in funcs.values():
        for _ in range(100):fn(r,counts,3)
    torch.cuda.synchronize()
    out={}
    for name,fn in funcs.items():
        elapsed=[]
        for batch in range(20):
            torch.cuda.synchronize();t=time.perf_counter_ns()
            for _ in range(100):fn(r,counts,3)
            torch.cuda.synchronize();elapsed.append((time.perf_counter_ns()-t)/1e6/100)
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
            for _ in range(5):fn(r,counts,3)
            torch.cuda.synchronize()
        events=[dict(name=e.key,calls=e.count,self_cpu_us=e.self_cpu_time_total) for e in prof.key_averages()]
        out[name]=dict(mean_ms=float(np.mean(elapsed)),median_ms=float(np.median(elapsed)),
            batches=elapsed,host_list=r.split_k2_layout.fan_out_list is not None,profile=events)
        prof.export_chrome_trace(str(HERE/f'micro_{name}.json'))
    (HERE/'MICROBENCH.json').write_text(json.dumps(dict(note='Fixed counts, one process. Synchronize only around each 100-update batch; kernel capture cost excluded. End-to-end PS timings determine adoption.',results=out),indent=2)+'\n')
    print({name:round(r['mean_ms'],6) for name,r in out.items()})


if __name__=='__main__':main()
