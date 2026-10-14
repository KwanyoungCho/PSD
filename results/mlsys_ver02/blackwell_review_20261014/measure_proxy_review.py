"""Fixed-input component timing; excludes model/AL/hit/TPS claims."""
import argparse,hashlib,json,os,statistics,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
os.environ.setdefault('SSD_HF_CACHE','/tmp');os.environ.setdefault('SSD_DATASET_DIR','/tmp')
sys.path.insert(0,str(a.root/'ssd'))
import torch
from ssd.engine.helpers.batch_tree_sampling import BatchedTreeProxy
from ssd.engine.helpers.flat_tree_sampling import FlatProxy
rows=[]
with torch.inference_mode():
 for v,dt in [(32000,torch.float16),(128256,torch.bfloat16)]:
  b,n=8,8;torch.manual_seed(712)
  tokens=torch.arange(1,n+1,device='cuda').expand(b,n)
  logits=torch.randn(b,n+1,v,device='cuda',dtype=dt);q=torch.randn(b,n,v,device='cuda',dtype=dt)
  lengths=[8]*4+[2]*4
  ps=[[-1,-1,0,0,1,2,3,4][:z] for z in lengths];ss=[[0,1,0,1,0,0,0,0][:z] for z in lengths]
  dense=BatchedTreeProxy(b,n,v,dt,'cuda',25,n,13,c_max=2)
  flat=FlatProxy(b,n,v,dt,'cuda',25,n,13,c_max=2)
  dense.prepare(ps,ss,tokens,q,.3,.3);flat.prepare(ps,ss,tokens,q,.3,.3)
  top=flat.current['topo'];packed=logits[top['req'],top['local']]
  for name,proxy,inp in [('dense',dense,logits),('flat',flat,packed)]:
   for _ in range(10):proxy.replay(inp)
   torch.cuda.synchronize();samples=[]
   for _ in range(9):
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    start.record()
    for __ in range(50):proxy.replay(inp)
    end.record();end.synchronize();samples.append(start.elapsed_time(end)/50)
   rows.append(dict(path=name,vocab=v,dtype=str(dt),median_ms=statistics.median(samples),samples_ms=samples))
  del dense,flat
  torch.cuda.empty_cache()
a.output.write_text(json.dumps(dict(rows=rows,device=torch.cuda.get_device_name(),
 root=str(a.root),hashes={str(p.relative_to(a.root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.root/'ssd/ssd').rglob('*.py')}),indent=2))
print(rows)
