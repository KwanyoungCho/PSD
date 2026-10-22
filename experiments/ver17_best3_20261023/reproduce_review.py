"""Small real-CUDA counterexamples; not a model-output bias estimate."""
import json
import torch
from check_overhead import BASE,OUT,load_function
from test_ver17_review import SchedulerOwnership
from ssd.engine.sequence import Sequence, SequenceStatus
from ssd.engine.helpers import p2_tree as pt
from ssd.engine.helpers.tree_rng import tree_rng_seed


def main():
    old=load_function(BASE/'ssd/ssd/engine/scheduler.py','schedule',
                      dict(Sequence=Sequence,SequenceStatus=SequenceStatus,debug_tokens_enabled=lambda:False))
    f=SchedulerOwnership();s,rows=f.make_scheduler([20,2041,30])
    returned,kind=old(s)
    before=dict(returned=[r.seq_id for r in returned],running=[r.seq_id for r in s.running],
                allocated=sorted(f.allocations),missing_unfinished=[r.seq_id for r in rows
                if r not in returned and r not in s.running])
    s,rows=f.make_scheduler([20,2041,30]);returned,kind=s.schedule()
    after=dict(returned=[r.seq_id for r in returned],running=[r.seq_id for r in s.running],
               allocated=sorted(f.allocations),missing_unfinished=[r.seq_id for r in rows
               if r not in returned and r not in s.running])
    n=131072;logits=torch.zeros(n,2,device='cuda');temps=torch.ones(n,device='cuda')
    joint={}
    for mode,seeds in [('old',[42,42]),('new',[tree_rng_seed(42,'p1',6),tree_rng_seed(42,'p1',10)])]:
        samples=[pt.tree_sample_wor(logits,temps,1,generator=torch.Generator(device='cuda').manual_seed(s))[0].flatten() for s in seeds]
        joint[mode]=dict(seeds=seeds,counts=torch.bincount(2*samples[0]+samples[1],minlength=4).tolist(),
                        same_token_fraction=(samples[0]==samples[1]).float().mean().item())
    result=dict(scheduler=dict(before=before,after=after),rng=dict(samples=n,joint=joint,
        limitation='Production sampler on fixed uniform logits, not measured bias of a full LLM.'))
    (OUT/'REPRODUCTION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
