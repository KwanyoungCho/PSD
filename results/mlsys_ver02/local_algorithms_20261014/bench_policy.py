"""Same-input graph replay cost of old versus local root/position scores."""
from pathlib import Path
import json,hashlib,os,sys,time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'ssd'))
os.environ.setdefault('SSD_HF_CACHE','/tmp');os.environ.setdefault('SSD_DATASET_DIR','/tmp')
import numpy as np
import torch
from ssd.engine.helpers.batch_tree_sampling import BatchedTreeProxy

@torch.inference_mode()
def main():
    torch.manual_seed(663)
    rows=[]
    for b,n,v in ((1,8,32000),(8,8,32000),(1,8,128256),(8,8,128256)):
        x=torch.randn(b,n+1,v,device='cuda',dtype=torch.bfloat16)
        q=torch.randn(b,n,v,device='cuda',dtype=torch.bfloat16)
        q[:,1]=q[:,0];q[:,3]=q[:,2]
        tokens=torch.arange(1,n+1,device='cuda').expand(b,n)
        parents=[[-1,-1,0,0,1,2,3,4]]*b;siblings=[[0,1,0,1,0,0,0,0]]*b
        variants={}
        for name,policy in [('paper',dict(source='residual',normalization='topm',overlap_mix=0.)),('local',dict(source='complement',normalization='full',overlap_mix=.25))]:
            p=BatchedTreeProxy(b,n,v,torch.bfloat16,'cuda',20,n,16,policy=policy)
            p.prepare(parents,siblings,tokens,q,[.7]*b,[.7]*b)
            p.exit.copy_(x)
            for _ in range(10):p.graph.replay()
            variants[name]=p
        torch.cuda.synchronize()
        samples={k:[] for k in variants};events=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
        for i in range(100):
            for name in (('paper','local') if i%2==0 else ('local','paper')):
                events[0].record()
                for _ in range(5):variants[name].graph.replay()
                events[1].record();events[1].synchronize()
                samples[name].append(events[0].elapsed_time(events[1])/5)
        delta=np.asarray(samples['local'])-samples['paper'];rng=np.random.default_rng(77)
        ci=np.quantile(delta[rng.integers(len(delta),size=(3000,len(delta)))].mean(1),[.025,.975])
        rows.append(dict(batch=b,nodes=n,vocab=v,samples_ms=samples,median_ms={k:float(np.median(x)) for k,x in samples.items()},delta_mean_ms=float(delta.mean()),delta_ci95_ms=ci.tolist()))
        print(rows[-1]['batch'],v,rows[-1]['median_ms'],flush=True)
    out=dict(status='complete',scope='GPU graph root scoring only, synthetic logits, same topology; no transfer/model time',cells=rows,
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'ssd/ssd').rglob('*.py')})
    Path(__file__).with_name('ROOT_COMPONENT.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
