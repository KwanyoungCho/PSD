"""Original-paper vs fast-chain parity, with independent seeds and ragged rows."""
import json,os,sys,time,statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.sched_setaffinity(0,set(range(24,32)) | set(range(56,64)))
os.environ.setdefault('SSD_HF_CACHE','/tmp');os.environ.setdefault('SSD_DATASET_DIR','/tmp')
import torch
from mlsys_ver01_hooks import load
root=Path(__file__).resolve().parents[1]
new=load('paper_chain_test_candidate',root/'ssd/utils/verify.py')
load('ssd.utils.verify_fast',root/'ssd/utils/verify_fast.py')
old=load('ver01_original_chain_for_gpu',Path(os.getenv('PAPER_BASE_CHECKOUT','/home/chokwans99/PSD-paper-review'))/'ssd/ssd/utils/verify.py')
records=[]
for v in (32000,128256):
 for b in (1,2,4,8):
  for k in (2,4):
   for temp in (1.,.7):
    torch.manual_seed(999)
    p=torch.randn(b,k+1,v,device='cuda',dtype=torch.float16)
    q=torch.randn(b,k,v,device='cuda',dtype=torch.float16)
    toks=torch.randint(v,(b,k+1),device='cuda')
    tt=torch.full((b,),temp,device='cuda');lengths=torch.tensor([max(1,k-i%k) for i in range(b)],device='cuda')
    args=(p,q,toks,tt,tt)
    kwargs=dict(jit_speculate=True,valid_k=lengths)
    funcs=[lambda:old.verify(*args,**kwargs),lambda:new.verify(*args,**kwargs,all_stochastic=True)]
    for seed in range(16):
     torch.manual_seed(seed);expected=funcs[0]()
     torch.manual_seed(seed);actual=funcs[1]()
     if expected!=actual:raise AssertionError((v,b,k,temp,seed,expected,actual))
    times=[[],[]]
    for rep in range(7):
     for arm in (0,1) if rep%2==0 else (1,0):
      torch.cuda.synchronize();t=time.perf_counter()
      for _ in range(15):funcs[arm]()
      torch.cuda.synchronize();times[arm].append((time.perf_counter()-t)*1000/15)
    records.append(dict(vocab=v,batch=b,k=k,temperature=temp,paired_outputs=16,
                        base_ms=statistics.median(times[0]),fast_ms=statistics.median(times[1])))
    print('CHAIN_PARITY',records[-1],flush=True)
Path(sys.argv[1]).write_text(json.dumps(dict(status='complete',cases=records),indent=2))
