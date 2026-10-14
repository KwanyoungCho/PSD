"""Measure complete preparation + proxy + acceptance, including packing cost."""
import os,sys,json,time,statistics,argparse,hashlib
from pathlib import Path
sys.path.insert(0,os.environ.get('PAPER_CHECKOUT','/data/chokwans99/PSD-tree-stream-20261014')+'/ssd')
os.environ.setdefault('SSD_HF_CACHE','/tmp');os.environ.setdefault('SSD_DATASET_DIR','/tmp')
import torch
from ssd.engine.helpers.flat_tree_sampling import FlatProxy,FlatAccept
from ssd.engine.helpers.batch_tree_sampling import BatchedTreeProxy,BatchedTreeAccept

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 report=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),gpu=torch.cuda.get_device_name(),scope='Host preparation, H2D, packing, proxy score and acceptance; no target forward',cells=[])
 for n,c in [(4,2),(4,3),(6,3),(8,3)]:
  b,v=8,128256;dtype=torch.bfloat16;lengths=[n]*4+[2]*4
  parent=[-1,-1,0,0,1,2,3,4][:n];sibling=[0,1,0,1,0,0,0,0][:n]
  ps=[parent[:z] for z in lengths];ss=[sibling[:z] for z in lengths]
  torch.manual_seed(42);lp=torch.randn(b,n+1,v,device='cuda',dtype=dtype);lq=torch.randn(b,n,v,device='cuda',dtype=dtype)
  tokens=torch.arange(1,n+1,device='cuda').expand(b,n).clone();tt=[.7]*b
  dense=BatchedTreeProxy(b,n,v,dtype,'cuda',25,n,13,c_max=c)
  flat=FlatProxy(b,n,v,dtype,'cuda',25,n,13,c_max=c)
  dense.prepare(ps,ss,tokens,lq,tt,tt);flat.prepare(ps,ss,tokens,lq,tt,tt)
  da=BatchedTreeAccept(dense,None,1,False,stochastic_only=True);fa=FlatAccept(flat,None,1,False,stochastic_only=True)
  def run(which):
   proxy,accept=(dense,da) if which=='dense' else (flat,fa)
   proxy.prepare(ps,ss,tokens,lq,tt,tt)
   x=lp if which=='dense' else lp[flat.current['topo']['req'],flat.current['topo']['local']]
   proxy.replay(x)
   # Exit logits and final logits are separate in serving: charge both gathers.
   y=lp if which=='dense' else lp[flat.current['topo']['req'],flat.current['topo']['local']]
   return accept.replay(y,tt,tt)
  for name in ('dense','compact'):
   for _ in range(5):run(name)
  values={name:[] for name in ('dense','compact')}
  for i in range(60):
   for name in (('dense','compact') if i%2==0 else ('compact','dense')):
    torch.cuda.synchronize();start=time.perf_counter();run(name);torch.cuda.synchronize();values[name].append((time.perf_counter()-start)*1000)
  cell=dict(batch=b,width=n,children=c,dtype=str(dtype),queries=((sum(lengths)+b+7)//8)*8,dense_queries=b*(n+1),samples_ms=values,median_ms={k:statistics.median(v) for k,v in values.items()})
  report['cells'].append(cell);print(n,c,cell['median_ms'],flush=True)
 report['status']='complete';a.output.write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
