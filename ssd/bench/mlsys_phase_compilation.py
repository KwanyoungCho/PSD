"""Record first-use Triton compilation/loading and CUDA capture on CPU only."""
import json
import os
from pathlib import Path
import time


def install():
    import triton.compiler
    import torch
    path=Path(os.environ['SSD_PROFILE_DIR'])/f'preparation_{os.getpid()}.jsonl'
    path.parent.mkdir(parents=True,exist_ok=True)
    def write(kind,start,name):
        end=time.monotonic_ns()
        with path.open('a') as f:
            f.write(json.dumps(dict(kind=kind,start_ns=start,end_ns=end,name=name,pid=os.getpid()))+'\n')
    original=triton.compiler.compile
    def compile(src,*args,**kwargs):
        start=time.monotonic_ns()
        try:return original(src,*args,**kwargs)
        finally:write('triton_first_use',start,getattr(src,'name','unknown'))
    triton.compiler.compile=compile
    enter,leave=torch.cuda.graph.__enter__,torch.cuda.graph.__exit__
    def begin(self):
        self._phase_audit_start=time.monotonic_ns()
        return enter(self)
    def end(self,*args):
        try:return leave(self,*args)
        finally:write('cuda_graph_capture',self._phase_audit_start,'graph')
    torch.cuda.graph.__enter__,torch.cuda.graph.__exit__=begin,end
