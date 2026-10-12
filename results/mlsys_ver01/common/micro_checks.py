"""Idle-GPU proxy replay and wire-cost A/B; never infer network time from bytes."""
import importlib.util, json, os, statistics, sys, time
from pathlib import Path
R=Path(__file__).resolve().parent;root=R.parents[2]
sys.path.insert(0,str(root/'ssd'));os.environ.setdefault('SSD_HF_CACHE','/tmp');os.environ.setdefault('SSD_DATASET_DIR','/tmp')
import torch
from ssd.engine.helpers import p2_tree as new
spec=importlib.util.spec_from_file_location('old_common_tree','/home/chokwans99/PSD-ver01-common-base/ssd/ssd/engine/helpers/p2_tree.py')
old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
torch.manual_seed(8);dev=torch.device('cuda');dtype=torch.float16
K,V=4,32000;ex=torch.randn(K+1,V,device=dev,dtype=dtype);q=torch.randn(K,V,device=dev,dtype=dtype);tok=torch.tensor([1,2,3,4],device=dev)
results={}
for kind in ['chain','tree']:
    graphs=[]
    for mod in (old,new):
        if kind=='chain':g=mod.ChainProxyCUDAGraph(K,V,10,15,True,dtype,dev)
        else:
            g=mod.TreeProxyCUDAGraph(K,V,15,K,dtype,dev,top_k=10);g.prepare_topology([-1,-1,0,1],[0,1,0,0])
        graphs.append(g)
    for g in graphs:
        for _ in range(30):g.replay(ex,q,tok)
    outs=[tuple(x.clone() for x in g.replay(ex,q,tok)) for g in graphs]
    assert torch.equal(outs[0][0],outs[1][0]) and torch.equal(outs[0][2],outs[1][2])
    oldid,oldp=old.unpack_piv(outs[0][1]);newid,newp=new.unpack_piv(outs[1][1]);assert torch.equal(oldid,newid) and torch.equal(oldp,newp)
    samples=[[],[]]
    for rep in range(15):
        for arm in ([0,1] if rep%2==0 else [1,0]):
            torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            t=time.perf_counter();a.record()
            for _ in range(100):graphs[arm].replay(ex,q,tok)
            b.record();b.synchronize()
            samples[arm].append(dict(host_gpu_us=(time.perf_counter()-t)*1e4,cuda_span_us=a.elapsed_time(b)*10))
    results[kind]=dict(before_us=statistics.median(x['cuda_span_us'] for x in samples[0]),after_us=statistics.median(x['cuda_span_us'] for x in samples[1]),samples=samples,default_t1_unpacked_exact=True)
ids=torch.arange(15,device=dev);p=torch.linspace(.001,1,15,device=dev)
res=[]
for mod in (old,new):
    packed=mod.pack_piv(ids,p)
    res.append(dict(token_payload_bytes=packed.numel()*packed.element_size(),position_plus_token_bytes=2*packed.numel()*packed.element_size(),dtype=str(packed.dtype)))
assert res[0]==res[1]
results['wire']=dict(before=res[0],after=res[1],note='Same int64 payload size and call sites; not a measured NCCL latency claim. New version requires both endpoints to upgrade.')
results['scope']='Same fixed tensors, alternate order, idle GPU; graph replay includes input copies and launch gaps. Component microbenchmark, not model TPS.'
(R/'MICRO.json').write_text(json.dumps(results,indent=2));print({k:(v.get('before_us'),v.get('after_us')) for k,v in results.items() if isinstance(v,dict)})
