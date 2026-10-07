import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch

os.environ.setdefault('SSD_HF_CACHE', '/tmp')
os.environ.setdefault('SSD_DATASET_DIR', '/tmp')
from ssd.engine.helpers.tree_gain_allocation import make_constants, allocate, exhaustive
from ssd.engine.helpers.tree_expansion_policy import TreeExpansionPolicy
from ssd.engine.draft_runner import DraftRunner


class TreeRound4Tests(unittest.TestCase):
    def test_bulk_export_matches_request_phase_and_prefix_semantics(self):
        from ssd.engine.helpers.batch_tree_draft import BatchedDuetDraft
        cfg=SimpleNamespace(duet_p1_tree_policy='on',duet_p2_tree_policy='on',
            duet_p1_tree_verify_nodes=2,duet_p2_tree_verify_nodes=3)
        records=[]
        for phase,N in ((1,4),(2,3)):
            for seq,safe in ((10,True),(11,True),(12,False)):
                ar=SimpleNamespace(NV=N,out_valid=torch.tensor([N,N-1]),
                    view_tok=torch.arange(2*N).reshape(2,N)+10*phase,
                    view_par=torch.full((2,N),-1),view_sib=torch.arange(N).repeat(2,1),
                    view_pcell=torch.zeros(2,N,dtype=torch.long),cell_logits=torch.randn(1,32))
                records.append((ar,seq,torch.tensor([0,1]),torch.tensor([5,6]),phase,safe))
        ordinary=SimpleNamespace(cfg=cfg,entries={})
        bulk=SimpleNamespace(cfg=cfg,entries={})
        for record in records:BatchedDuetDraft._save_entries(ordinary,*record)
        BatchedDuetDraft._save_entries_bulk(bulk,records)
        self.assertEqual(ordinary.entries.keys(),bulk.entries.keys())
        for key in ordinary.entries:
            a,b=ordinary.entries[key],bulk.entries[key]
            self.assertIs(a['logits'],b['logits'])
            self.assertEqual({k:v for k,v in a.items() if k!='logits'},
                             {k:v for k,v in b.items() if k!='logits'})

    def test_gain_optimality_and_request_independence(self):
        curve = [0., .55, .82, .90]
        constants = make_constants(4, curve)
        weights = [[.8, .6, .2], [.99, .3, .1]]
        root = torch.tensor([0, 0, 0, 1, 1, 1])
        log = torch.tensor(np.log(weights).reshape(-1))
        valid = torch.ones(6, dtype=torch.bool)
        remaining = torch.tensor([4, 2])
        result = allocate(root, log, valid, remaining, 0, constants)
        for b in range(2):
            f = result[b*3:b*3+3].tolist()
            best, _ = exhaustive(weights[b], curve, int(remaining[b]))
            self.assertAlmostEqual(sum(w*curve[c] for w,c in zip(weights[b],f)), best)
        changed = log.clone(); changed[3:] += 20
        self.assertTrue(torch.equal(result[:3], allocate(root, changed, valid, remaining, 0, constants)[:3]))

    def test_reach_includes_earlier_sibling_rejections(self):
        policy = TreeExpansionPolicy.__new__(TreeExpansionPolicy)
        policy.alpha = torch.tensor([[.8], [.5], [.25]], dtype=torch.double)
        policy.bins = torch.tensor([], dtype=torch.double)
        raw = torch.tensor([[.6, .3, .1]], dtype=torch.double)
        saved = raw.clone()
        result = policy.priority(torch.tensor([.5]).double().log(), torch.tensor([0,0,0]), raw)
        torch.testing.assert_close(result.exp(), torch.tensor([.4,.05,.0125], dtype=torch.double))
        self.assertTrue(torch.equal(saved, raw))

    def test_legacy_prefix_ignores_realized_q_and_stale_cache(self):
        for phase in (1,2):
            for score in (.001,.999):
                stub = SimpleNamespace(config=SimpleNamespace(
                    duet_p1_tree_max_nodes=3, duet_p2_tree_max_nodes=3,
                    duet_p1_tree_verify_nodes=2, duet_p2_tree_verify_nodes=2,
                    duet_tree_wire_nodes=2), hf_config=SimpleNamespace(vocab_size=8),
                    device=torch.device('cpu'), _p1_rerank_cache={'invalid_old_cache':True})
                stub._tree_compact_view = {k:torch.zeros(1,2,dtype=torch.int64)
                    for k in ('tok','parent_local','sib_order','parent_q_ref','parent_q_cells')}
                stub._tree_compact_view.update(raw_q=torch.zeros(1,2),valid=torch.zeros(1,dtype=torch.int64),
                    u_valid=torch.zeros(1,dtype=torch.int64),selected_old=torch.zeros(2,dtype=torch.int64))
                views=dict(tok=torch.tensor([[0,1,2]]),parent_local=torch.tensor([[-1,-1,0]]),
                    sib_order=torch.tensor([[0,1,0]]),raw_q=torch.tensor([[.6,.25,score]]),
                    valid=torch.tensor([3]),parent_q_ref=torch.tensor([[0,0,1]]),
                    parent_q_cells=torch.tensor([[0,1,-1]]),u_valid=torch.tensor([2]),
                    cell_logits=torch.randn(2,8))
                served,*_=DraftRunner._rerank_tree_hit_view(stub,views,0,phase)
                self.assertEqual(served['tok'][0].tolist(),[0,1])
                self.assertEqual(served['parent_local'][0].tolist(),[-1,-1])
                self.assertEqual(served['parent_q_ref'][0].tolist(),[0,0])
                DraftRunner._precompute_p1_rerank_views(stub,views,1)
                self.assertIsNone(stub._p1_rerank_cache)

    def test_shallow_miss_keeps_parent_q_and_first_sibling_backbone(self):
        runner=DraftRunner.__new__(DraftRunner)
        runner.config=SimpleNamespace(use_eagle=False,speculate_k=4,duet_phase1_k=None,
                                      greedy_only=True,sampler_x=None,async_fan_out=3)
        runner.hf_config=SimpleNamespace(torch_dtype=torch.float32)
        runner.device=torch.device('cpu'); runner.block_size=256
        seen=[]
        def forward(ids, positions, **kwargs):
            seen.append(ids.tolist())
            return torch.tensor([[1.,3.,2.,0.]]).expand(len(ids),-1)
        runner.run_model=forward
        q=torch.empty(2,4,4); tokens=torch.empty(2,4,dtype=torch.long)
        runner.jit_speculate(torch.tensor([[0,0,3],[1,0,0]]),torch.tensor([5,10]),
            q,tokens,torch.zeros(2),torch.zeros(2,8,dtype=torch.int32),
            k_override=2,branch_width=2)
        self.assertEqual(tokens.tolist(),[[1,2,1,2],[1,2,1,2]])
        self.assertEqual(seen,[[3,0],[1,1]])
        torch.testing.assert_close(q[:,0],q[:,1]);torch.testing.assert_close(q[:,2],q[:,3])


if __name__=='__main__': unittest.main()
