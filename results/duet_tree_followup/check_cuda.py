"""Run only after GPUs are available: allocation CPU/GPU/eager/graph parity."""
import json
from pathlib import Path
import numpy as np
import torch
from allocation import allocate, make_constants


def main():
    rng=np.random.default_rng(92227);cases=0
    for b,r,w in [(8,9,27),(8,9,15),(8,3,9),(6,15,15)]:
        curve=[0,.1,.9,.99]
        cc=make_constants(b,curve);cg=make_constants(b,curve,'cuda')
        # Fixed captured inputs change on every replay.
        cpu=[torch.zeros(w,dtype=torch.long),torch.zeros(w,dtype=torch.double),
             torch.zeros(w,dtype=torch.bool),torch.zeros(r,dtype=torch.long)]
        gpu=[x.cuda() for x in cpu]
        for _ in range(3):allocate(*gpu,1,cg)
        torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=allocate(*gpu,1,cg)
        for _ in range(50):
            # Interleaved roots; keep per-root counts <=8 as selector guarantees.
            roots=np.arange(w)%r;valid=rng.random(w)>.2
            cpu=[torch.tensor(roots),torch.tensor(rng.random(w)).log(),torch.tensor(valid),torch.tensor(rng.integers(0,b+1,r))]
            for dst,src in zip(gpu,cpu):dst.copy_(src)
            graph.replay()
            a=out.cpu();expected=allocate(*cpu,1,cc);eager=allocate(*gpu,1,cg).cpu()
            if not torch.equal(a,expected) or not torch.equal(a,eager):raise ValueError('CUDA allocation mismatch')
            cases+=1
    result=dict(passed=True,cases=cases,device=torch.cuda.get_device_name(0))
    path=Path(__file__).resolve().parent/'cuda_checks.json'
    path.write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
