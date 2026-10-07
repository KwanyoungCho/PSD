"""Independent phase capacities must not inherit legacy short/long ordering."""
import os
os.environ.setdefault('SSD_HF_CACHE','/tmp');os.environ.setdefault('SSD_DATASET_DIR','/tmp')
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from ssd.config import Config
from ssd.engine.draft_runner import DraftRunner
from ssd.engine.helpers.batch_tree_common import enabled


def hf(_):
    return SimpleNamespace(max_position_embeddings=2048,hidden_size=128,num_attention_heads=4,
        num_key_value_heads=4,num_hidden_layers=32,model_type='llama',vocab_size=127,
        rope_theta=10000.,torch_dtype=torch.float32,head_dim=32)


def config(batch=1,tree=True):
    return Config(model='/target',draft='/draft',speculate=True,draft_async=True,jit_speculate=True,
        num_gpus=2,max_num_seqs=batch,max_model_len=128,speculate_k=4,
        async_fan_out=2,duet_draft_fan_out=1,duet_enabled=True,duet_exit_layer=12,
        duet_phase1_k=1,duet_phase2_k=3,duet_p1_tree_policy='on' if tree else 'off',
        duet_p2_tree_policy='on' if tree else 'off',duet_p1_tree_max_nodes=2,
        duet_p2_tree_max_nodes=6,duet_p2_budget=3)


class PhaseBudgetTests(unittest.TestCase):
    def setUp(self):
        self.path=patch('ssd.config.os.path.isdir',return_value=True);self.path.start();self.addCleanup(self.path.stop)
        self.hf=patch('ssd.config.AutoConfig.from_pretrained',side_effect=hf);self.hf.start();self.addCleanup(self.hf.stop)

    def test_legacy_chain_and_b1_tree_keep_order_guard(self):
        with patch.dict(os.environ,{'SSD_BATCHED_TREE':'0'}):
            with self.assertRaisesRegex(ValueError,'K2 <= K1'):config(1,True)
            with self.assertRaisesRegex(ValueError,'K2 <= K1'):config(8,False)

    def test_unified_b1_and_b8_accept_longer_phase2_with_envelope(self):
        for b in (1,8):
            with patch.dict(os.environ,{'SSD_BATCHED_TREE':'1' if b==1 else '0'}):
                cfg=config(b)
                self.assertTrue(enabled(cfg))
                self.assertEqual(cfg.duet_response_token_width,6)
                self.assertEqual(cfg.duet_p2_active_root_count,3)
                self.assertGreaterEqual(cfg.duet_proxy_wire_N,3+7*cfg.duet_p1_roots_per_position)
                draft=DraftRunner.create_draft_config(cfg)
                self.assertEqual((draft.duet_phase1_k,draft.duet_phase2_k),(1,3))
                runner=DraftRunner.__new__(DraftRunner)
                runner.config=draft;runner.device=torch.device('cpu');runner.hf_config=hf('draft')
                runner.block_size=256;runner.batched_tree_enabled=True
                runner._init_prealloc_buffers()
                self.assertEqual(runner._arange_kp1.numel(),7)

if __name__=='__main__':unittest.main()
