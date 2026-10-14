"""Fixed-tensor, alternating-order latency of batched tree probability work."""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('SSD_HF_CACHE','/tmp')
os.environ.setdefault('SSD_DATASET_DIR','/tmp')
import torch
from ssd.engine.helpers.batch_tree_sampling import BatchedTreeProxy,BatchedTreeAccept


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=dict(status='running',gpu=torch.cuda.get_device_name(),cells=[])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for vocab in (32000,128256):
        for batch in (2,8):
            torch.manual_seed(42);n=8
            logits=torch.randn(batch,n+1,vocab,device='cuda',dtype=torch.float16)
            q=torch.randn(batch,n,vocab,device='cuda',dtype=torch.float16);q[:,1]=q[:,0]
            tokens=torch.arange(1,n+1,device='cuda').expand(batch,n)
            parents=[[-1,-1,0,1,2,3,4,5]]*batch
            siblings=[[0,1,0,0,0,0,0,0]]*batch
            calls={};keepers=[]
            for trim in (False,True):
                os.environ['SSD_TREE_LADDER_TRIM']=str(int(trim))
                proxy=BatchedTreeProxy(batch,n,vocab,torch.float16,'cuda',25,4,13)
                proxy.prepare(parents,siblings,tokens,q,[.7]*batch,[.7]*batch)
                calls[f'proxy_trim{int(trim)}']=lambda p=proxy:p.replay(logits)
                keepers.append(proxy)
                for specialized in (False,True) if trim else (False,):
                    accept=BatchedTreeAccept(proxy,None,3,False,stochastic_only=specialized)
                    name=f'accept_trim{int(trim)}_positive{int(specialized)}'
                    calls[name]=lambda g=accept:g.replay(logits,[.7]*batch,[.7]*batch)
                    keepers.append(accept)
            parity=[]
            for seed in range(10):
                pair=[]
                for name in ('accept_trim1_positive0','accept_trim1_positive1'):
                    torch.cuda.manual_seed(seed)
                    pair.append(tuple(x.clone() for x in calls[name]()))
                parity.append(all(torch.equal(x,y) for x,y in zip(*pair)))
            samples={k:[] for k in calls}
            for f in calls.values():
                for _ in range(3):f()
            torch.cuda.synchronize()
            for repeat in range(40):
                order=list(calls)
                if repeat%2:order.reverse()
                for name in order:
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize();wall=time.perf_counter()
                    start.record();calls[name]();end.record();end.synchronize()
                    samples[name].append(dict(cuda_ms=start.elapsed_time(end),wall_ms=(time.perf_counter()-wall)*1000))
            out['cells'].append(dict(vocab=vocab,batch=batch,specialized_same_seed_equal=all(parity),
                samples=samples,summary={k:{metric:statistics.median(x[metric] for x in v)
                    for metric in ('cuda_ms','wall_ms')} for k,v in samples.items()}))
            a.output.write_text(json.dumps(out,indent=2))
            print(json.dumps(out['cells'][-1]['summary']),flush=True)
            del calls,keepers,proxy,accept,logits,q,pair
            gc.collect();torch.cuda.empty_cache()
    out['status']='complete';a.output.write_text(json.dumps(out,indent=2))


if __name__=='__main__':main()
