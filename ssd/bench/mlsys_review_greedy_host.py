"""Benchmark-only alternative for the small-batch greedy tree regression.

Export all row argmax IDs in one D2H, walk independent CPU topologies.
Production dispatch/defaults are unchanged. Full logits/forward excluded.
"""
import argparse,json,os,sys,time,statistics as st
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('SSD_HF_CACHE','/tmp');os.environ.setdefault('SSD_DATASET_DIR','/tmp')
import torch
from ssd.engine.helpers.batch_tree_sampling import pack_topologies,BatchedTreeAccept,verify_batch
from ssd.engine.helpers.p2_tree import tree_verify_walk_greedy

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    result=dict(status='running',scope=__doc__,gpu=torch.cuda.get_device_name(),cases=[])
    def save():
        temporary=a.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result,indent=2))
        temporary.replace(a.output)
    save()
    try:
      for v in (32000,128256):
       for b in (1,2,4,8):
        for n in (4,8):
         for c in (1,3):
            torch.manual_seed(612+b+n+c)
            pl=torch.randn(b,n+1,v,device='cuda',dtype=torch.bfloat16)
            q=torch.randn(b,n,v,device='cuda',dtype=torch.bfloat16)
            pars=[];sibs=[];ints=[];host=[]
            tokens=torch.zeros(b,n,device='cuda',dtype=torch.long)
            for bi in range(b):
                count=max(0,n-bi%5);par=[j//c-1 for j in range(count)];sib=[j%c for j in range(count)]
                ts=[]
                for j in range(count):
                    parent=par[j];candidates=pl[bi,parent+1].topk(c+1).indices
                    # Some parents contain their argmax; others miss it.
                    token=int(candidates[sib[j]+int(bi!=0 and (parent+bi)%3==0)]);ts.append(token)
                pars.append(par);sibs.append(sib);host.append(ts)
                if ts:tokens[bi,:count]=torch.tensor(ts,device='cuda')
                ints.append(dict(valid=count,u_valid=count,tok=ts,parent_local=par,sib_order=sib,parent_q_ref=[par.index(x) for x in par]))
            topo=pack_topologies(pars,sibs,n,'cuda',c_max=c)
            proxy=SimpleNamespace(exit=pl,q=q,tokens=tokens,topology=topo)
            graph=BatchedTreeAccept(proxy,None,1,True)
            def captured():return torch.cat(graph.replay(pl,[0]*b,[0]*b),1).cpu().tolist()
            def batched_host():
                argmax=pl.argmax(-1).cpu().tolist();out=[]
                for bi in range(b):
                    children={(pa,t):j for j,(pa,t) in enumerate(zip(pars[bi],host[bi]))}
                    path=[];ctx=-1
                    for _ in range(len(host[bi])+1):
                        token=argmax[bi][ctx+1];child=children.get((ctx,token))
                        if child is None:break
                        path.append(child);ctx=child
                    out.append(path+[-1]*(n-len(path))+[token,ctx+1])
                return out
            def serial():
                out=[]
                for bi in range(b):
                    path,token=tree_verify_walk_greedy(ints[bi],pl[bi]);out.append(path+[-1]*(n-len(path))+[token,path[-1]+1 if path else 0])
                return out
            ref=torch.cat(verify_batch(pl,q,tokens,topo,[0]*b,[0]*b,greedy=True),1).cpu().tolist()
            assert captured()==ref and batched_host()==ref and serial()==ref
            fns=dict(graph=captured,batched_argmax_host=batched_host,serial_legacy=serial);times={k:[] for k in fns}
            for fn in fns.values():
                for _ in range(3):fn()
            for rep in range(7):
                order=list(fns)
                if rep%2:order.reverse()
                for k in order:
                    torch.cuda.synchronize();start=time.perf_counter()
                    for _ in range(50):fns[k]()
                    torch.cuda.synchronize();times[k].append((time.perf_counter()-start)*1000/50)
            result['cases'].append(dict(vocab=v,batch=b,nodes=n,siblings=c,parity=True,
                median_ms={k:st.median(x) for k,x in times.items()},timings_ms=times))
            save()
      result['status']='complete';save()
    except BaseException as e:result.update(status='failed',error=str(e));save();raise

if __name__=='__main__':main()
