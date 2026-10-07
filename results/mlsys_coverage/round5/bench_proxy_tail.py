"""Captured full-vocabulary root-kernel ablation, no engine-file mutation."""
import argparse,gc,json,time
import numpy as np
import torch
from check_proxy_tail import original,optimized,sampling,root_policy,check,baseline_wrapper,optimized_wrapper


def capture(run):
    for _ in range(3):run()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=run()
    return graph,out


def timed(graph,repeats):
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):graph.replay()
    end.record();end.synchronize()
    return start.elapsed_time(end)/repeats


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--repeats',type=int,default=100);a=p.parse_args()
    torch.manual_seed(311)
    result=dict(gpu=torch.cuda.get_device_name(0),correctness_cases=check('cuda'),rows=[])
    for v,dtype in [(32000,torch.float16),(128256,torch.bfloat16)]:
        for b,n,c in [(1,8,3),(8,8,3),(8,4,1),(8,16,3)]:
            par=[-1 if j<c else (j-c)//c for j in range(n)];sib=[j%c for j in range(n)]
            topo=sampling.pack_topologies([par]*b,[sib]*b,n,'cuda',c)
            e=torch.randn(b,n+1,v,device='cuda',dtype=dtype)
            q=torch.randn(b,n,v,device='cuda',dtype=dtype);tok=torch.randint(v,(b,n),device='cuda')
            temp=torch.full((b,),.7,dtype=torch.float32,device='cuda')
            def run():return root_policy.tree_candidates(e,q,tok,topo,16,n,16,temp,temp,
                source='complement',normalization='full',overlap_mix=.25)
            base,bo=capture(run)
            sampling.ladder=optimized_wrapper
            try:opt,oo=capture(run)
            finally:sampling.ladder=baseline_wrapper
            for x,y in zip(bo,oo):torch.testing.assert_close(x,y,rtol=0,atol=0)
            del x,y
            base_times=[];opt_times=[]
            for repeat in range(6):
                for name,graph in ([('base',base),('opt',opt)] if repeat%2==0 else [('opt',opt),('base',base)]):
                    (base_times if name=='base' else opt_times).append(timed(graph,a.repeats))
            result['rows'].append(dict(b=b,n=n,c=c,v=v,baseline_ms=base_times,optimized_ms=opt_times,
                speedup=float(np.median(base_times)/np.median(opt_times)),bit_identical=True))
            print(result['rows'][-1],flush=True)
            del base,opt,bo,oo,topo,e,q,tok,temp
            gc.collect();torch.cuda.empty_cache()
    from pathlib import Path
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
