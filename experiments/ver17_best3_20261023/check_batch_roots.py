"""Same-input batched root preparation overhead, eager and captured GPU."""
import json
from pathlib import Path
import statistics
import time
import torch
from check_overhead import load_function, BASE, OUT
from ssd.engine.helpers.p2_tree import selected_q_probs_from_logits
from ssd.engine.helpers.batch_tree_roots import root_candidates


def main():
    old=load_function(BASE/'ssd/ssd/engine/helpers/batch_tree_roots.py','root_candidates',
                      {'selected_q_probs_from_logits':selected_q_probs_from_logits})
    rows=[]
    for b,p in ((1,24),(16,24)):
        torch.manual_seed(41)
        logits=torch.randn(b,p,32000,device='cuda',dtype=torch.bfloat16)
        tokens=torch.randint(32000,(b,p),device='cuda')
        vis=torch.eye(p,dtype=torch.uint8)
        for i in range(1,p):vis[i]|=vis[(i-1)//3]
        vis=vis.expand(b,p,p).contiguous().cuda()
        lengths=torch.full((b,),p,dtype=torch.int64,device='cuda')
        temps=torch.full((b,),.7,device='cuda')
        args=(logits,tokens,vis,lengths,temps,4,None,5)
        fns={'before':old,'after':root_candidates}
        graphs={}
        for arm,fn in fns.items():
            for _ in range(10):fn(*args)
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):fn(*args)
            graphs[arm]=graph
        for mode in ('eager','graph'):
            raw={arm:[] for arm in fns}
            for i in range(12):
                for arm in (('before','after') if i%2==0 else ('after','before')):
                    torch.cuda.synchronize();start=time.perf_counter()
                    a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True);a.record()
                    for _ in range(80):
                        if mode=='eager':fns[arm](*args)
                        else:graphs[arm].replay()
                    z.record();z.synchronize()
                    raw[arm].append({'wall_us':(time.perf_counter()-start)*1e6/80,'gpu_us':a.elapsed_time(z)*1000/80})
            row=dict(batch=b,contexts=p,mode=mode,raw=raw,
                     mean={arm:{k:statistics.mean(v[k] for v in vv) for k in ('wall_us','gpu_us')} for arm,vv in raw.items()})
            rows.append(row);print(json.dumps(row['mean']),b,mode,flush=True)
    (OUT/'BATCH_ROOT_OVERHEAD.json').write_text(json.dumps({'rows':rows},indent=2)+'\n')


if __name__=='__main__':main()
