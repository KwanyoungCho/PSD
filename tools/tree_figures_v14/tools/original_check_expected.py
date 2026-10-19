from pathlib import Path
import json,math,random,statistics,sys,os
import torch
ROOT=Path('/data/chokwans99/duet-v14-tree-figures-20261019')
sys.path[:0]=[str(ROOT),'/home/chokwans99/PSD-MLSys-ver1.4/ssd']
os.environ.setdefault('SSD_HF_CACHE','/tmp')
os.environ.setdefault('SSD_DATASET_DIR','/tmp')
from figure_math import expected,first_depth
from ssd.engine.helpers.p2_tree import tree_verify_walk_tensor,q_probs_from_logits

tree=dict(tokens=[0,1,2,1,2],parents=[-1,-1,-1,0,0],siblings=[0,1,2,0,1],qrefs=[0,0,0,1,1])
ti=dict(valid=5,tok=tree['tokens'],parent_local=tree['parents'],sib_order=tree['siblings'],parent_q_ref=tree['qrefs'],u_valid=2)
rng=random.Random(37101);cases=[]
for case in range(8):
 def probability():
  x=torch.tensor([rng.random()+.02 for _ in range(3)]);return x/x.sum()
 q=torch.stack([probability(),probability()])
 p=torch.stack([probability() for _ in range(6)])
 if case==0:
  p[:]=q[0];q[:]=q[0]
 logits_p=p.log()*.7;logits_q=q.log()*.7
 pred=expected(tree,logits_p,logits_q,eos=999)
 qp=q_probs_from_logits(logits_q,torch.full((2,),.7))
 if case==0:
  assert abs(pred['expected_al']-(1+first_depth(tree)))<1e-6
 draws=[]
 for _ in range(6000):
  path,_=tree_verify_walk_tensor(ti,logits_p,qp,.7,rng.random,lambda p:0)
  draws.append(1+len(path))
 measured=statistics.mean(draws);se=statistics.stdev(draws)/math.sqrt(len(draws))
 if abs(pred['expected_al']-measured)>6*se+.004:
  raise AssertionError((case,pred,measured,se))
 cases.append(dict(case=case,exact=pred['expected_al'],sampled=measured,standard_error=se,draws=len(draws)))
out=dict(status='pass',native_residual_verifier=True,cases=cases,p_equals_q_first_sibling_identity=True,target_matching=False)
dest=Path(__file__).resolve().parents[1]/'EXPECTATION_CHECKS.json'
dest.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
