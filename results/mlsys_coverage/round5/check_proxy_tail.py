"""Standalone optimization prototype; does not mutate engine sources on disk.

The last draft renormalization is unused in every ladder. The last proxy
residual is unused for complement/proxy root scores. Exact verification and
residual-source ranking still require the final residual.
"""
import argparse,inspect,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'ssd'))
os.environ.setdefault('SSD_HF_CACHE','/tmp');os.environ.setdefault('SSD_DATASET_DIR','/tmp')
import torch
from ssd.engine.helpers import batch_tree_sampling as sampling
from ssd.engine.helpers import root_policy

from ladder_before_trim import ladder as original


def baseline_wrapper(*args,**kwargs):
    kwargs.pop('need_residual',None);kwargs.pop('trim',None)
    return original(*args,**kwargs)


sampling.ladder=baseline_wrapper
source=inspect.getsource(original)
source=source.replace('overlap_mix=0.0):','overlap_mix=0.0,need_residual=True):',1)
a=source.index('        newer=(residual-draft)')
b=source.index('    alpha=torch.stack',a)
block=source[a:b]
split=block.index('        newer=draft.scatter')
source=source[:a]+('        if s < child.shape[-1]-1 or need_residual:\n'+
    ''.join('    '+line+'\n' for line in block[:split].splitlines())+
    '        if s < child.shape[-1]-1:\n'+
    ''.join('    '+line+'\n' for line in block[split:].splitlines()))+source[b:]
source=source.replace('residual=p.clone()', 'residual=p')
source=source.replace('draft=q_ext[bi,child[:,:,0]].clone()', 'draft=q_ext[bi,child[:,:,0]]')
source=source.replace('    residual=torch.where(residual.sum(-1,keepdim=True)>1e-12,residual,p)',
    '    if need_residual:\n        residual=torch.where(residual.sum(-1,keepdim=True)>1e-12,residual,p)')
namespace=dict(sampling.__dict__)
exec(compile(source,'proxy_tail_prototype','exec'),namespace)
optimized=namespace['ladder']


def optimized_wrapper(*args,**kwargs):
    kwargs.pop('need_residual',None);kwargs.pop('trim',None)
    return optimized(*args,**kwargs,need_residual=False)


def check(device):
    torch.manual_seed(302)
    cases=0
    for n,c in [(4,1),(8,3),(16,3)]:
        # Level-order siblings, plus a shorter second row: real padding is exercised.
        parents=[-1 if i<c else (i-c)//c for i in range(n)]
        siblings=[i%c for i in range(n)]
        topo=sampling.pack_topologies([parents,parents[:-2]],[siblings,siblings[:-2]],n,device,c)
        for temperature in (.1,.7,1):
            e=torch.randn(2,n+1,127,device=device); q=torch.randn(2,n,127,device=device)
            tokens=torch.randint(127,(2,n),device=device)
            p=(e/temperature).softmax(-1);d=(q/temperature).softmax(-1)
            for exact in (False,True):
                for mix in (0,.25):
                    ref=original(tokens,p,d,topo,n,exact=exact,overlap_mix=mix)
                    got=optimized(tokens,p,d,topo,n,exact=exact,overlap_mix=mix)
                    for x,y in zip(ref,got):
                        if x is not None:torch.testing.assert_close(x,y,rtol=0,atol=0)
                    cases+=1
            for name in ('proxy','complement'):
                kw=dict(source=name,normalization='full',overlap_mix=.25)
                ref=root_policy.tree_candidates(e,q,tokens,topo,8,n,8,temperature,temperature,**kw)
                sampling.ladder=optimized_wrapper
                try:got=root_policy.tree_candidates(e,q,tokens,topo,8,n,8,temperature,temperature,**kw)
                finally:sampling.ladder=baseline_wrapper
                for x,y in zip(ref,got):torch.testing.assert_close(x,y,rtol=0,atol=0)
                cases+=1
    return cases


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');a=p.parse_args()
    print(json.dumps(dict(device=a.device,bit_identical_cases=check(a.device),engine_modified=False)))

if __name__=='__main__':main()
