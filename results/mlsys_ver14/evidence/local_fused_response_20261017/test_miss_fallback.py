"""Compare actual old/new hit-serving entry points on cache misses."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import importlib.util,sys,unittest
import torch
import ssd.engine.draft_runner as new

class Miss(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path('/data/chokwans99/duet-local-followup-20261017/code/baseline/ssd/ssd/engine/draft_runner.py')
        spec=importlib.util.spec_from_file_location('fused_old_draft_reference',path)
        cls.old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=cls.old;spec.loader.exec_module(cls.old)

    def fixture(self,empty):
        config=SimpleNamespace(use_eagle=False,duet_enabled=True,verbose=False,
            duet_proxy_on_draft=False,duet_response_token_width=15,
            duet_p1_tree_verify_nodes=15,duet_p2_tree_verify_nodes=15,
            duet_phase1_k=8,duet_phase2_k=5,jit_speculate=True)
        r=SimpleNamespace(config=config,device='cpu',
            hf_config=SimpleNamespace(vocab_size=32,torch_dtype=torch.float32,hidden_size=8),
            tree_cache_keys=torch.empty(0,3,dtype=torch.long) if empty else torch.tensor([[10,1,5]]),
            tree_cache_is_tree=None,tree_cache_valid_k=torch.tensor([5]),_last_n_draft_keys=1)
        def jit(keys,nt,lg,tok,temps,bt,acts):
            tok[:,:5].copy_(torch.arange(5).view(1,5));lg[:,:5].fill_(.7)
        r.jit_speculate=jit
        return r

    def test_nonempty_and_empty_cache_keep_jit_output_and_rng(self):
        for empty in (False,True):
            args=(torch.tensor([[12,2,6]]),1,13,torch.tensor([33]),torch.tensor([.7]),torch.tensor([[0,1,2]]))
            reference=self.fixture(empty);candidate=self.fixture(empty)
            torch.manual_seed(19);a=self.old.DraftRunner.hit_cache_and_respond(reference,*args);state=torch.get_rng_state().clone()
            torch.manual_seed(19);b=new.DraftRunner.hit_cache_and_respond(candidate,*args)
            self.assertTrue(torch.equal(state,torch.get_rng_state()))
            for aa,bb in zip(a,b):
                if torch.is_tensor(aa):self.assertTrue(torch.equal(aa,bb))
                else:self.assertEqual(aa,bb)
            if not empty:self.assertIsNotNone(candidate._fused_miss_match)

if __name__=='__main__':unittest.main()
