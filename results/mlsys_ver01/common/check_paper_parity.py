"""Coupled stochastic law checks against the unmodified paper checkout."""
import importlib.util
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('SSD_HF_CACHE', '/tmp')
os.environ.setdefault('SSD_DATASET_DIR', '/tmp')
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parents[2]/'ssd'))
import torch
from ssd.utils.verify import verify
from ssd.engine.helpers import p2_tree as new_tree
from ssd.layers.sampler import Sampler

def load(name, file):
    spec=importlib.util.spec_from_file_location(name,file)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod
    spec.loader.exec_module(mod);return mod

base=Path(os.environ.get('PAPER_BASE_CHECKOUT','/home/chokwans99/PSD-paper-review'))
old_verify=load('ver01_original_verify',base/'ssd/ssd/utils/verify.py')
old_tree=load('ver01_original_tree',base/'ssd/ssd/engine/helpers/p2_tree.py')
checks={'chain':0,'tree':0,'sampler':0,'q_probs':0}
for dtype in (torch.float16,torch.bfloat16):
 for b in (1,2,4,8):
  for k in (2,4):
   for temp in (.3,.7,1.):
    torch.manual_seed(911)
    p=torch.randn(b,k+1,257,dtype=dtype)
    q=torch.randn(b,k,257,dtype=dtype)
    toks=torch.randint(257,(b,k+1))
    t=torch.full((b,),temp);dt=torch.full((b,),.8)
    lengths=torch.tensor([max(0,k-i%3) for i in range(b)])
    for seed in range(8):
     torch.manual_seed(seed)
     expected=old_verify.verify(p.clone(),q.clone(),toks,t,dt,jit_speculate=True,valid_k=lengths)
     state=torch.get_rng_state().clone()
     torch.manual_seed(seed)
     actual=verify(p.clone(),q.clone(),toks,t,dt,jit_speculate=True,valid_k=lengths,all_stochastic=True)
     assert expected==actual,(dtype,b,k,temp,seed,expected,actual)
     assert torch.equal(state,torch.get_rng_state())
     checks['chain']+=1
 for temp in (.3,.7,1.):
  for sx in (None,.5):
   logits=torch.randn(8,257,dtype=dtype);t=torch.full((8,),temp)
   expected=old_tree.q_probs_from_logits(logits.clone(),t,sx,3)
   actual=new_tree.q_probs_from_logits(logits.clone(),t,sx,3)
   assert torch.equal(expected,actual)
   checks['q_probs']+=1
   torch.manual_seed(17);expected=Sampler(sampler_x=sx)(logits.clone(),t,is_tree=True)
   state=torch.get_rng_state().clone()
   torch.manual_seed(17);actual=Sampler(sampler_x=sx,stochastic_only=True)(logits.clone(),t,is_tree=True)
   assert torch.equal(expected,actual) and torch.equal(state,torch.get_rng_state())
   checks['sampler']+=1
 for n,c in ((4,2),(8,3)):
  parents=[-1 if j<c else (j-c)//c for j in range(n)]
  refs=[x+1 for x in parents]
  tree=dict(valid=n,u_valid=max(refs)+1,tok=list(range(1,n+1)),parent_local=parents,
            sib_order=[j%c for j in range(n)],parent_q_ref=refs)
  for temp in (.3,.7,1.):
   for pattern in ('diffuse','sharp','aligned'):
    logits=torch.randn(n+1,257,dtype=dtype)*(5 if pattern=='sharp' else 1)
    q_logits=torch.randn(max(refs)+1,257,dtype=dtype)
    if pattern=='aligned':
     for j in range(n):logits[parents[j]+1]=q_logits[refs[j]]
    for seed in range(8):
     def run(mod):
      torch.manual_seed(seed)
      q=mod.q_probs_from_logits(q_logits.clone(),torch.full((len(q_logits),),temp),None,3)
      draws=[]
      def sample(p):
       draws.append(p.clone());return int(torch.multinomial(p,1))
      out=mod.tree_verify_walk_tensor(tree,logits.float().clone(),q,temp,lambda:float(torch.rand(1)),sample)
      return out,draws,torch.get_rng_state().clone()
     expected,ep,es=run(old_tree);actual,ap,ast=run(new_tree)
     assert expected==actual and len(ep)==len(ap) and torch.equal(es,ast)
     assert all(torch.equal(a,b) for a,b in zip(ep,ap))
     checks['tree']+=1
(R/'PAPER_PARITY.json').write_text(json.dumps(dict(status='complete',device='cpu',checks=checks,reference=str(base),scope='Exact output, recovery distribution and RNG equality on supplied tensors; not an end-to-end model or GPU timing test.'),indent=2)+'\n')
print('PASS',checks,flush=True)
