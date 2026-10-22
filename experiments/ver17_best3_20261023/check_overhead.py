"""Matched codec+selector timing; exact baseline functions loaded from AST."""
import ast
import hashlib
import json
from pathlib import Path
import statistics
import time
import torch
from ssd.engine.helpers import p2_tree as pt
from ssd.engine.draft_runner import DraftRunner

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/mlsys_ver17/additional_review'
BASE = Path('/home/chokwans99/PSD-ver17-baseline-20261023')


def load_function(path, name, constants):
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    node.decorator_list=[]
    env=dict(torch=torch,**constants)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),env)
    return env[name]


def main():
    p=BASE/'ssd/ssd/engine/helpers/p2_tree.py'
    const=dict(PIV_SHIFT=32,PIV_VER_BIT=48,_PIV_QMAX=65535,_TOK_MASK=2**32-1,_LOG_MIN=-6.)
    pack_old=load_function(p,'pack_piv',const);unpack_old=load_function(p,'unpack_piv',const)
    oldsel=load_function(BASE/'ssd/ssd/engine/draft_runner.py','_select_proxy_sourced_tokens_unified',{})
    score=torch.tensor([0.,1e-12,1e-9,1e-8,.1,1.],device='cuda')
    tok=torch.arange(6,device='cuda')
    _,legacy=unpack_old(pack_old(tok,score));_,fixed=pt.unpack_piv(pt.pack_piv(tok,score))
    evidence=dict(old_scores=legacy.tolist(),fixed_scores=fixed.tolist(),
                  input_scores=score.tolist(),
                  old_zero_budget=pt.alloc_policy_root_budgets(legacy,'dynamic',total=48,beta=.5,cap=8).tolist(),
                  fixed_zero_budget=pt.alloc_policy_root_budgets(fixed,'dynamic',total=48,beta=.5,cap=8).tolist())
    rows=[]
    for B,N in ((1,120),(1,256),(16,120)):
        pos=torch.arange(N,device='cuda').remainder(24).expand(B,N).contiguous()
        tok=torch.arange(N,device='cuda').add(100).expand(B,N).contiguous()
        scores=torch.linspace(.5,1e-4,N,device='cuda').expand(B,N).contiguous()
        roots=torch.full((B,24,4),-1,dtype=torch.int64,device='cuda')
        def call(arm):
            pack,unpack,select=(pack_old,unpack_old,oldsel) if arm=='before' else (pt.pack_piv,pt.unpack_piv,DraftRunner._select_proxy_sourced_tokens_unified)
            clean,value=unpack(pack(tok,scores))
            return select(dict(chosen_pos=pos,chosen_tok=clean,chosen_piv=value),roots,K_rank=23,total_budget=24)
        for _ in range(30):call('before');call('after')
        torch.cuda.synchronize()
        measures={arm:[] for arm in ('before','after')}
        for i in range(16):
            for arm in (('before','after') if i%2==0 else ('after','before')):
                torch.cuda.synchronize();start=time.perf_counter()
                for _ in range(100):call(arm)
                torch.cuda.synchronize()
                measures[arm].append((time.perf_counter()-start)*1e6/100)
        profiles={}
        for arm in ('before','after'):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                call(arm);torch.cuda.synchronize()
            events=prof.events()
            profiles[arm]=dict(kernels=sum(e.device_type==torch.autograd.DeviceType.CUDA and not e.name.startswith(('Memcpy','Memset')) for e in events),
                DtoH=[e.name for e in events if 'DtoH' in e.name],HtoD=[e.name for e in events if 'HtoD' in e.name])
        row=dict(batch=B,wire_candidates=N,raw_us=measures,profile=profiles,
                 summary_us={a:dict(mean=statistics.mean(v),sd=statistics.stdev(v)) for a,v in measures.items()})
        rows.append(row);print(json.dumps(row),flush=True)
    data=dict(status='complete',timing='eager pack+unpack+selector, alternating 16 blocks x100 calls, no model',rows=rows,reproduction=evidence,
              baseline=str(BASE),baseline_p2_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    (OUT/'OVERHEAD.json').write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':main()
