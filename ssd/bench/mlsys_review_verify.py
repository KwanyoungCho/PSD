"""Fixed-input verification timings; model forward and transport are excluded."""
import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('SSD_HF_CACHE','/tmp')
os.environ.setdefault('SSD_DATASET_DIR','/tmp')
import torch
from ssd.utils.verify import verify
from ssd.utils.verify_fast import verify_stochastic
from ssd.engine.helpers.batch_tree_sampling import pack_topologies, ladder, verify_batch, BatchedTreeAccept
from ssd.engine.helpers.p2_tree import q_probs_from_logits, tree_verify_walk_tensor, tree_verify_walk_greedy


def timed(fn, iterations):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    start=time.perf_counter()
    for _ in range(iterations): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-start)*1000/iterations


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kind',choices=['chain','tree'],required=True)
    ap.add_argument('--vocab',type=int,nargs='+',default=[32000,128256])
    ap.add_argument('--batches',type=int,nargs='+',default=[1,2,4,8])
    ap.add_argument('--nodes',type=int,nargs='+',default=[4,8])
    ap.add_argument('--repeats',type=int,default=5)
    ap.add_argument('--iterations',type=int,default=20)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',args=vars(a)|{'output':str(a.output)},
                gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                scope='fixed synthetic logits; per-call wall ms including final host materialization; excludes forward/transport',
                cases=[])
    def save(): a.output.write_text(json.dumps(report,indent=2))
    save()
    dev='cuda'; dtype=torch.bfloat16
    try:
      for v in a.vocab:
       for b in a.batches:
        for n in a.nodes:
         for t in (.7,0.):
          for c in ([1] if a.kind=='chain' else [1,3]):
            torch.manual_seed(891+b+n+c)
            pl=torch.randn(b,n+1,v,device=dev,dtype=dtype)
            ql=torch.randn(b,n,v,device=dev,dtype=dtype)
            tt=torch.full((b,),t,device=dev,dtype=torch.float32)
            graphs=[]
            if a.kind=='chain':
                tokens=torch.cat([torch.zeros(b,1,device=dev,dtype=torch.long),ql.argmax(-1)],1)
                lengths=torch.tensor([max(1,n-i%3) for i in range(b)],device=dev)
                args=(pl,ql,tokens,tt,tt)
                funcs={'general':lambda:verify(*args,jit_speculate=True,valid_k=lengths)}
                if t:
                    funcs['fast']=lambda:verify_stochastic(*args,valid_k=lengths)
                else:
                    funcs['fast']=lambda:verify(*args,jit_speculate=True,valid_k=lengths,all_greedy=True)
                torch.manual_seed(515)
                ref=funcs['general']()
                torch.manual_seed(515)
                opt=funcs['fast']()
                if str(ref)!=str(opt): raise AssertionError('Chain reference/fast output differs')
                parity=True
            else:
                pars=[];sibs=[];ints=[]
                tokens=torch.zeros(b,n,device=dev,dtype=torch.long)
                for bi in range(b):
                    count=max(1,n-bi%3)
                    par=[j//c-1 for j in range(count)];sib=[j%c for j in range(count)]
                    pars.append(par);sibs.append(sib)
                    for parent in sorted(set(par)):
                        group=[j for j,p in enumerate(par) if p==parent]
                        common=torch.randn(v,device=dev,dtype=dtype)
                        ql[bi,group]=common
                        pl[bi,parent+1]=common+torch.randn_like(common)*.8
                        probs=(common.float()/(t or 1.)).softmax(-1)
                        tokens[bi,group]=torch.multinomial(probs,len(group),replacement=False) if t else probs.topk(len(group)).indices
                    ints.append(dict(valid=count,u_valid=count,tok=tokens[bi,:count].cpu().tolist(),
                                     parent_local=par,sib_order=sib,parent_q_ref=[par.index(p) for p in par]))
                topo=pack_topologies(pars,sibs,n,dev,c_max=c)
                # Verify the actual acceptance probabilities and correction
                # distributions exactly, before timing any implementation.
                pp=(pl.float()/(t or 1.)).softmax(-1)
                qq=(ql.float()/(t or 1.)).softmax(-1)
                l0=ladder(tokens,pp,qq,topo,n,exact=True,trim=False)
                l1=ladder(tokens,pp,qq,topo,n,exact=True,trim=True)
                torch.testing.assert_close(l0[0],l1[0],rtol=0,atol=0)
                torch.testing.assert_close(l0[2],l1[2],rtol=0,atol=0)
                del l0,l1,pp,qq
                proxy=SimpleNamespace(exit=pl,q=ql,tokens=tokens,topology=topo)
                def materialize(out): return torch.cat(out,1).cpu().tolist()
                funcs={}
                for trim in (0,1):
                    os.environ['SSD_TREE_LADDER_TRIM']=str(trim)
                    graph=BatchedTreeAccept(proxy,None,1,t==0)
                    graphs.append(graph)
                    def replay(g=graph): return materialize(g.replay(pl,[t]*b,[t]*b))
                    funcs[f'graph_trim{trim}']=replay
                def eager():
                    os.environ['SSD_TREE_LADDER_TRIM']='0'
                    return materialize(verify_batch(pl,ql,tokens,topo,tt,tt,greedy=t==0))
                funcs['eager']=eager
                def serial():
                    if t:
                        q=q_probs_from_logits(ql.reshape(b*n,v),tt[:,None].expand(b,n).reshape(-1),None,1).view(b,n,v)
                        return [tree_verify_walk_tensor(ints[i],pl[i],q[i],t,
                                lambda:float(torch.rand((),device=dev)),
                                lambda x:int(torch.multinomial(x,1))) for i in range(b)]
                    return [tree_verify_walk_greedy(ints[i],pl[i]) for i in range(b)]
                funcs['serial_legacy_walk']=serial
                parity=True
            rows=[]
            for rep in range(a.repeats):
                order=list(funcs)
                if rep%2: order.reverse()
                for name in order:
                    rows.append(dict(repeat=rep,implementation=name,
                                     ms=timed(funcs[name],a.iterations)))
            stats={name:statistics.median(r['ms'] for r in rows if r['implementation']==name) for name in funcs}
            report['cases'].append(dict(vocab=v,batch=b,nodes=n,siblings=c,temperature=t,
                                        ragged=True,parity=parity,timings=rows,median_ms=stats))
            save();print('CASE',v,b,n,c,t,stats,flush=True)
            del funcs,graphs,pl,ql,tokens,tt
            if a.kind=='tree': del proxy,graph,topo,eager,serial,replay
            gc.collect();torch.cuda.empty_cache()
      report['status']='complete';save()
    except BaseException as e:
      report.update(status='failed',error=f'{type(e).__name__}: {e}');save();raise


if __name__=='__main__': main()
