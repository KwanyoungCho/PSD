from types import SimpleNamespace
import unittest
import torch
from ssd.engine.helpers.fused_tree_response import request_packet_cpu

class Request(unittest.TestCase):
    def test_wire_bytes_match_original_for_ragged_tables_and_float32_temperatures(self):
        seqs=[SimpleNamespace(seq_id=12,recovery_token_id=999,num_tokens=33,
            last_spec_step_accepted_len=4,tree_terminal_node=None,
            draft_block_table=[9,2],draft_temperature=None,temperature=0.7),
            SimpleNamespace(seq_id=13,recovery_token_id=100,num_tokens=140,
            last_spec_step_accepted_len=2,tree_terminal_node=15,
            draft_block_table=[5,3,2],draft_temperature=0.,temperature=1.)]
        keys=torch.tensor([[12,3,999],[13,15,100]],dtype=torch.int64)
        lengths=torch.tensor([33,140]);tables=torch.tensor([[9,2,-1,-1],[5,3,2,-1]],dtype=torch.int64)
        temperatures=torch.tensor([.7,0.],dtype=torch.float32).view(torch.int32).to(torch.int64)
        expected=torch.cat([keys.reshape(-1),lengths,tables.reshape(-1),temperatures])
        self.assertTrue(torch.equal(request_packet_cpu(seqs,4),expected))

    def test_excess_block_table_rejected(self):
        seq=SimpleNamespace(seq_id=1,recovery_token_id=2,num_tokens=3,
            last_spec_step_accepted_len=1,tree_terminal_node=0,
            draft_block_table=[1,2],draft_temperature=None,temperature=1.)
        with self.assertRaises(ValueError):request_packet_cpu([seq],1)

if __name__=='__main__':unittest.main()
