"""Paired captured dense vs flat proxy/accept kernels; no model/TPS claim."""
import argparse,json,os,sys,time,gc,statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('SSD_HF_CACHE','/tmp');os.environ.setdefault('SSD_DATASET_DIR','/tmp')
import torch
import mlsys_flat_tree as flat
from ssd.engine.helpers.batch_tree_sampling import candidates,verify_batch,pack_topologies

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--dtype',choices=['float16','bfloat16'],default='float16')
    ap.add_argument('--vocab',type=int,default=0)
    args=ap.parse_args();report=dict(status='running',gpu=torch.cuda.get_device_name(),dtype=args.dtype,cells=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for vocab in ((args.vocab,) if args.vocab else (32000,128256)):
      for shape,lengths in [('uniform',[8]*8),('half_short',[8]*4+[2]*4),('one_hit',[8]+[0]*7)]:
        b=8;n=8;qc=((sum(lengths)+b+7)//8)*8
        ps=[[-1,-1,0,0,1,2,3,4][:z] for z in lengths]
        ss=[[0,1,0,1,0,0,0,0][:z] for z in lengths]
        dense=pack_topologies(ps,ss,n,'cuda');topo=flat.topology(ps,ss,n,3,'cuda',qc)
        live_c=max(1,max((max(s,default=0)+1 for s in ss),default=1))
        slim=pack_topologies(ps,ss,n,'cuda',live_c)
        torch.manual_seed(132)
        lp=torch.randn(b,n+1,vocab,device='cuda',dtype=getattr(torch,args.dtype))
        lq=torch.randn(b,n,vocab,device='cuda',dtype=getattr(torch,args.dtype))
        tok=torch.arange(1,n+1,device='cuda').expand(b,n).clone()
        tt=torch.full((b,),.7,device='cuda')
        fp=lp[topo['req'],topo['local']].contiguous();fq=flat.pack_nodes(lq,topo).contiguous()
        ft=flat.pack_nodes(tok,topo)
        fns=dict(dense_proxy=lambda:candidates(lp,lq,tok,dense,25,n,13,tt,tt),
                 flat_proxy=lambda:flat.candidates(fp,fq,ft,topo,25,n,13,tt,tt),
                 dense_accept=lambda:verify_batch(lp,lq,tok,dense,tt,tt,stochastic_only=True),
                 flat_accept=lambda:flat.verify(fp,fq,ft,topo,n,tt,tt),
                 live_proxy=lambda:candidates(lp,lq,tok,slim,25,n,13,tt,tt),
                 live_accept=lambda:verify_batch(lp,lq,tok,slim,tt,tt,stochastic_only=True))
        graphs={};buffers={};samples={k:[] for k in fns}
        for name,fn in fns.items():
            for _ in range(3):fn()
            torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):buffers[name]=fn()
            graphs[name]=graph
        parity=[]
        for seed in range(20):
            torch.cuda.manual_seed(seed);graphs['dense_accept'].replay()
            dense_out=tuple(x.clone() for x in buffers['dense_accept'])
            torch.cuda.manual_seed(seed);graphs['flat_accept'].replay()
            parity.append(all(torch.equal(a,z) for a,z in zip(dense_out,buffers['flat_accept'])))
            torch.cuda.manual_seed(seed);graphs['live_accept'].replay()
            if any(not torch.equal(a,z) for a,z in zip(dense_out,buffers['live_accept'])):
                raise AssertionError('Live-sibling acceptance mismatch')
        if not all(parity):raise AssertionError(f'Captured acceptance mismatch: {vocab} {shape} {parity}')
        for repeat in range(60):
            keys=list(graphs)
            if repeat%2:keys.reverse()
            for key in keys:
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize();wall=time.perf_counter()
                start.record();graphs[key].replay();end.record();end.synchronize()
                samples[key].append(dict(cuda_ms=start.elapsed_time(end),wall_ms=(time.perf_counter()-wall)*1000))
        cell=dict(vocab=vocab,shape=shape,lengths=lengths,queries=qc,dense_queries=b*(n+1),samples=samples,same_seed_accept_parity=parity,
                  median={k:{m:statistics.median(x[m] for x in rows) for m in ('cuda_ms','wall_ms')} for k,rows in samples.items()})
        report['cells'].append(cell);args.output.write_text(json.dumps(report,indent=2));print(vocab,shape,cell['median'],flush=True)
        del graphs,buffers,fns,lp,lq,fp,fq;gc.collect();torch.cuda.empty_cache()
    report['status']='complete';args.output.write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
