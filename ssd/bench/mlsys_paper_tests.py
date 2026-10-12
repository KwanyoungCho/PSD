"""Coupled stochastic correctness and latency tests against paper functions."""
import argparse,json,os,sys,time,statistics
from pathlib import Path
sys.path.insert(0,'/home/chokwans99/PSD-paper-review/ssd')
os.environ.setdefault('SSD_HF_CACHE','/tmp');os.environ.setdefault('SSD_DATASET_DIR','/tmp')
import torch
from ssd.engine.helpers.p2_tree import tree_verify_walk_tensor
from mlsys_paper_tree import PaperTreeLadder


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    results=[]
    for v in (32000,128256):
      for n in (4,8,16):
       for c in (1,3):
        graph=PaperTreeLadder(n,v,c,'cuda')
        rng=torch.Generator().manual_seed(483+n+c)
        # Breadth-first C-ary tree, original q shared by siblings.
        par=[-1 if j<c else (j-c)//c for j in range(n)] if c>1 else list(range(-1,n-1))
        sib=[j%c for j in range(n)] if c>1 else [0]*n
        refs=[x+1 for x in par]
        q=torch.randn(n,v,device='cuda').softmax(-1)
        toks=[]
        for j in range(n):
            d=q[refs[j]].clone()
            for k in range(j):
                if par[k]==par[j]:d[toks[k]]=0
            toks.append(int(torch.multinomial(d,1)))
        tree=dict(valid=n,u_valid=max(refs)+1,tok=torch.tensor(toks),parent_local=torch.tensor(par),
                  sib_order=torch.tensor(sib),parent_q_ref=torch.tensor(refs))
        # Both close-to-draft and highly concentrated target distributions.
        for pattern in ('diffuse','aligned','sharp'):
         logits=torch.randn(n+1,v,device='cuda')*(4 if pattern=='sharp' else 1)
         if pattern=='aligned':
            for row in range(n+1):
                children=[j for j,p in enumerate(par) if p+1==row]
                if children:logits[row]=q[refs[children[0]]].log()
         max_tv=0.;matched=0
         for seed in range(24):
            coins=torch.rand(n+1,generator=rng).tolist()
            def call(fn):
                ci=iter(coins);distributions=[]
                def sample(p):
                    distributions.append(p.detach().clone())
                    return int(torch.multinomial(p,1))
                torch.manual_seed(seed+121)
                out=fn(tree,logits,q,.7,lambda:next(ci),sample)
                return out,distributions[0]
            reference,rp=call(tree_verify_walk_tensor)
            candidate,cp=call(graph.walk)
            tv=float((rp-cp).abs().sum()/2);max_tv=max(max_tv,tv)
            if reference!=candidate:raise AssertionError((v,n,c,pattern,seed,reference,candidate,tv))
            if tv>1e-5:raise AssertionError(('residual mismatch',tv))
            matched+=1
         timings={}
         for label,fn in [('base',tree_verify_walk_tensor),('candidate',graph.walk)]:
            times=[]
            for rep in range(5):
                torch.cuda.synchronize();t=time.perf_counter()
                for _ in range(10):fn(tree,logits,q,.7,lambda:.55,lambda p:int(torch.multinomial(p,1)))
                torch.cuda.synchronize();times.append((time.perf_counter()-t)*100)
            timings[label]=statistics.median(times)
         results.append(dict(vocab=v,nodes=n,siblings=c,pattern=pattern,matched=matched,max_tv=max_tv,ms=timings))
         a.output.write_text(json.dumps(dict(status='running',cases=results),indent=2))
         print('TREE_PARITY',results[-1],flush=True)
    a.output.write_text(json.dumps(dict(status='complete',cases=results),indent=2))

if __name__=='__main__':main()
