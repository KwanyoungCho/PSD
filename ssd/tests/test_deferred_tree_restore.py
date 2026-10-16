"""Accepted-path restore preserves fragmented pages and termination guards."""
from types import SimpleNamespace
import unittest
import torch
from ssd.engine.draft_runner import DraftRunner
from ssd.engine.helpers.p2_tree import pack_tree_ints


class DeferredRestoreTests(unittest.TestCase):
    def make(self):
        view={k:torch.tensor(v) for k,v in dict(valid=[4],u_valid=[3],
            tok=[[1,2,3,4]],parent_local=[[-1,-1,0,2]],
            sib_order=[[0,1,0,0]],parent_q_ref=[[0,0,1,2]]).items()}
        served=pack_tree_ints(view,0,4)
        runner=SimpleNamespace(config=SimpleNamespace(duet_tree_wire_nodes=4),
            hf_config=SimpleNamespace(vocab_size=10),block_size=4,
            kv_cache=torch.zeros(1,2,8,4,1,1),
            _tree_served_numtok=100,_tree_served_seq=999)
        staged=torch.arange(10.).reshape(1,2,5,1,1)
        return runner,(served,staged,3,7)

    def restore(self,runner,state,num_tokens=7,seq=7,terminal=4):
        DraftRunner._restore_previous_tree_kv(runner,state,
            torch.tensor([[seq,terminal,5]]),torch.tensor([num_tokens]),
            torch.tensor([[5,2,7,1,6,0,3,4]]))

    def test_branch_path_crosses_fragmented_pages_using_previous_snapshot(self):
        r,state=self.make();self.restore(r,state)
        actual=r.kv_cache.reshape(1,2,32,1,1)
        expected=torch.zeros_like(actual)
        expected[:,:,23]=state[1][:,:,1]
        expected[:,:,8]=state[1][:,:,3]
        expected[:,:,9]=state[1][:,:,4]
        self.assertTrue(torch.equal(actual,expected))
        self.assertEqual(r._tree_served_numtok,100)
        self.assertEqual(r._tree_served_seq,999)

    def test_output_truncation_restores_only_actual_accepted_prefix(self):
        r,state=self.make();self.restore(r,state,num_tokens=6)
        actual=r.kv_cache.reshape(1,2,32,1,1)
        self.assertTrue(torch.equal(actual[:,:,23],state[1][:,:,1]))
        self.assertTrue(torch.equal(actual[:,:,8],state[1][:,:,3]))
        self.assertTrue(torch.equal(actual[:,:,9],torch.zeros(1,2,1,1)))

    def test_changed_sequence_empty_accept_and_missing_stage_do_not_copy(self):
        for changes in [dict(seq=8),dict(terminal=0),dict(num_tokens=4),dict(missing=True)]:
            with self.subTest(changes=changes):
                r,state=self.make()
                if changes.pop('missing',False):state=(state[0],None,*state[2:])
                self.restore(r,state,**changes)
                self.assertEqual(int(torch.count_nonzero(r.kv_cache)),0)


if __name__=='__main__':unittest.main()
