"""Ablate research policies only in benchmark processes; serving has one default.

Run through mlsys_tree_campaign with DUET_LOCAL_REVIEW_ROOT in the plan.
No changes to acceptance, actual q, sampling, KV, scheduling or timing adapters.
"""
import os
from pathlib import Path
import sys
os.environ.setdefault('SSD_HF_CACHE','/tmp')
os.environ.setdefault('SSD_DATASET_DIR','/tmp')
CHECKOUT=Path(os.environ.get('PAPER_CHECKOUT',str(Path(__file__).resolve().parents[2])))
sys.path.insert(0,str(CHECKOUT/'ssd'))
from ssd.engine.helpers import root_policy as root
from ssd.engine.helpers import flat_tree_sampling as flat

POLICIES={
    'local':dict(source='complement',normalization='full',overlap_mix=.25),
    'paper':dict(source='residual',normalization='topm',overlap_mix=0.),
    'normalization':dict(source='residual',normalization='full',overlap_mix=0.),
    'position':dict(source='residual',normalization='full',overlap_mix=.25),
    'proxy':dict(source='proxy',normalization='full',overlap_mix=.25),
}
name=os.environ.get('DUET_LOCAL_REVIEW_ROOT','local')
policy=POLICIES[name]
def wrap(fn):
    def call(*args,**kwargs):
        kwargs.update(policy)
        return fn(*args,**kwargs)
    return call
root.chain_candidates=wrap(root.chain_candidates)
root.tree_candidates=wrap(root.tree_candidates)
flat.candidates=wrap(flat.candidates)
print('[research ablation]',name,policy,'tree='+os.getenv('SSD_TREE_EXPANSION_POLICY','reach_gain_frontier'),flush=True)

from mlsys_tree_followup import main
if __name__=='__main__':main()
