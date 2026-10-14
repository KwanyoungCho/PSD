"""Fixed-work cost of observed/expected position weights; no model execution.

All capture, allocations and warmup are excluded. Dense synthetic logits
match model vocabulary/dtype and exercise the actual tensor operations.
These results are component latency, not end-to-end inference speedups.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

import torch
import triton
import triton.language as tl

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'ssd'))


@triton.jit
def overlap_parts(E,Q,O,V:tl.constexpr,PARTS:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0); part=tl.program_id(1)
    x=part*BLOCK+tl.arange(0,BLOCK)
    e=tl.load(E+row*V+x,x<V,0).to(tl.float32)
    q=tl.load(Q+row*V+x,x<V,0).to(tl.float32)
    tl.store(O+row*PARTS+part,tl.sum(tl.minimum(e,q),0))


def overlap(e,q):
    b,k,v=q.shape;parts=triton.cdiv(v,4096)
    out=torch.empty(b*k,parts,device=q.device,dtype=torch.float32)
    overlap_parts[(b*k,parts)](e,q,out,v,parts,4096)
    return out.sum(-1).reshape(b,k).clamp(0,1)


def hazard(alpha):
    reach=torch.cat([torch.ones_like(alpha[:,:1]),alpha.cumprod(-1)],-1)
    return torch.cat([reach[:,:-1]*(1-alpha),reach[:,-1:]],-1)


def hat_scalars(ey,qy):
    return hazard((ey/(qy+1e-10)).clamp(max=1))


def observed(e,q,y):
    return hat_scalars(e.gather(-1,y).squeeze(-1),q.gather(-1,y).squeeze(-1))


def mixture(e,q,y,fused=False):
    mean=overlap(e,q) if fused else torch.minimum(e,q).sum(-1).clamp(0,1)
    return .75*observed(e,q,y)+.25*hazard(mean)


def selected_probability(logits,y,t):
    z=logits.float()/t
    return (z.gather(-1,y).squeeze(-1)-z.logsumexp(-1)).exp()


def roots(el,ql,y,t,source,mix=False,fused=False,qy=None):
    b,k=y.shape[:2]
    e=(el.float()/t).softmax(-1)
    q=(ql.float()/t).softmax(-1) if qy is None else None
    ek=e[:,:k].contiguous()
    h=(mixture(ek,q,y,fused) if mix else observed(ek,q,y)) if qy is None else hat_scalars(ek.gather(-1,y).squeeze(-1),qy)
    if source=='proxy':s=ek.clone()
    elif source=='complement':s=ek*(1-q)
    else:s=(ek-q).clamp_min(0)
    s.scatter_(-1,y,0)
    s=torch.cat([s,e[:,k:k+1]],1)
    values,ids=s.topk(17,-1)
    piv=h[:,:,None]*values/values.sum(-1,keepdim=True).clamp_min(1e-10)
    scores,chosen=piv.flatten(1).topk(15,-1)
    return chosen//17,ids.flatten(1).gather(1,chosen),scores


def capture(fn):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):out=fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=fn()
    for _ in range(15):graph.replay()
    torch.cuda.synchronize()
    return graph,out


def benchmark(functions,rounds,repeats,seed):
    graphs={};outputs={};samples={name:[] for name in functions}
    for name,fn in functions.items():graphs[name],outputs[name]=capture(fn)
    order=list(graphs);rng=random.Random(seed)
    for rnd in range(rounds):
        rng.shuffle(order)
        for name in order:
            g=graphs[name]
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):g.replay()
            end.record();end.synchronize()
            samples[name].append(start.elapsed_time(end)*1000/repeats)
    stats={name:dict(median_us=statistics.median(x),round_us=x,min_us=min(x),max_us=max(x)) for name,x in samples.items()}
    return stats,outputs


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--rounds',type=int,default=9);ap.add_argument('--repeats',type=int,default=40)
    ap.add_argument('--replicate',type=int,default=0);args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(1);torch.manual_seed(8321)
    result=dict(scope=__doc__,gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
        args=vars(args)|{'output':str(args.output)},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        gpu_environment=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,clocks.sm,temperature.gpu','--format=csv,noheader'],text=True),
        cells=[],correctness=[])
    configs=[('llama2',32000,torch.float16),('llama3',128256,torch.bfloat16)]
    for name,v,dtype in configs:
        for b,k,t in [(1,4,1.),(1,4,.7),(1,12,.7),(8,4,.7),(8,12,.7)]:
            el=torch.randn(b,k+1,v,device='cuda',dtype=dtype)*2
            ql=torch.randn(b,k,v,device='cuda',dtype=dtype)*2
            e=(el[:,:k].float()/t).softmax(-1).contiguous();q=(ql.float()/t).softmax(-1)
            y=torch.multinomial(q.reshape(b*k,v),1).reshape(b,k,1)
            ey=e.gather(-1,y).squeeze(-1);qy=q.gather(-1,y).squeeze(-1)
            scalar=hat_scalars(ey,qy);full=observed(e,q,y)
            torch.testing.assert_close(scalar,full,rtol=0,atol=0)
            torch.testing.assert_close(mixture(e,q,y,True),mixture(e,q,y),rtol=2e-5,atol=2e-6)
            torch.testing.assert_close(selected_probability(ql,y,t),qy,rtol=5e-5,atol=1e-7)
            common=dict(model=name,batch=b,positions=k,vocab=v,dtype=str(dtype),temperature=t)
            functions={
                'position_scalar_hat':lambda:hat_scalars(ey,qy),
                'position_dense_hat':lambda:observed(e,q,y),
                'position_dense_mix':lambda:mixture(e,q,y),
                'position_fused_mix':lambda:mixture(e,q,y,True),
                'position_logits_full_hat':lambda:observed((el[:,:k].float()/t).softmax(-1),(ql.float()/t).softmax(-1),y),
                'position_logits_lse_hat':lambda:hat_scalars(selected_probability(el[:,:k],y,t),selected_probability(ql,y,t)),
                'sender_qy_from_probs':lambda:q.gather(-1,y),
                'sender_qy_from_logits':lambda:selected_probability(ql,y,t),
            }
            for source in ('proxy','complement','residual'):
                functions[source+'_hat']=lambda source=source:roots(el,ql,y,t,source)
                functions[source+'_mix']=lambda source=source:roots(el,ql,y,t,source,True)
                functions[source+'_mix_fused']=lambda source=source:roots(el,ql,y,t,source,True,True)
            functions['proxy_hat_scalar_q']=lambda:roots(el,ql,y,t,'proxy',qy=qy)
            stats,outputs=benchmark(functions,args.rounds,args.repeats,args.replicate+31*b+k)
            for a,z in zip(outputs['proxy_hat'],outputs['proxy_hat_scalar_q']):torch.testing.assert_close(a,z,rtol=0,atol=0)
            for source in ('proxy','complement','residual'):
                torch.testing.assert_close(outputs[source+'_mix'][2],outputs[source+'_mix_fused'][2],rtol=2e-5,atol=2e-6)
            result['cells'].append(common|{'timing':stats})
            result['correctness'].append(common|{'scalar_hat_equal':True,'proxy_scalar_root_equal':True,'fused_mix_close':True})
            print(name,b,k,t,'position mix-hat us',stats['position_dense_mix']['median_us']-stats['position_dense_hat']['median_us'],flush=True)
            del outputs,functions,e,q,el,ql;gc.collect();torch.cuda.empty_cache()
        # Tree sibling residuals already require full e/q, even with observed weights.
        from ssd.engine.helpers.batch_tree_sampling import ladder,pack_topologies
        for b in (1,8):
            n=12
            parents=[(i//3)-1 for i in range(n)];siblings=[i%3 for i in range(n)]
            topo=pack_topologies([parents]*b,[siblings]*b,n,'cuda',3)
            tokens=torch.arange(1,n+1,device='cuda').expand(b,n)
            p=torch.randn(b,n+1,v,device='cuda').softmax(-1)
            q=torch.randn(b,n,v,device='cuda').softmax(-1)
            fs={'tree_ladder_hat':lambda:ladder(tokens,p,q,topo,3,overlap_mix=0.),
                'tree_ladder_mix':lambda:ladder(tokens,p,q,topo,3,overlap_mix=.25)}
            stats,_=benchmark(fs,args.rounds,args.repeats,args.replicate+512+b)
            result['cells'].append(dict(model=name,batch=b,positions=n,vocab=v,kind='tree_ladder',timing=stats))
            del fs,p,q;gc.collect();torch.cuda.empty_cache()
    result['status']='complete';args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
