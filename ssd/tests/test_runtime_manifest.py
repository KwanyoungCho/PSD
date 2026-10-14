import os
os.environ.setdefault('SSD_HF_CACHE','/tmp')
os.environ.setdefault('SSD_DATASET_DIR','/tmp')
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from ssd.config import Config
from ssd.utils.runtime_manifest import runtime_manifest


class RuntimeManifest(unittest.TestCase):
    def test_draft_model_tp_is_distinct_from_target_tp(self):
        cfg=Config.__new__(Config)
        cfg.num_gpus=3;cfg.draft_async=True
        hf=SimpleNamespace(torch_dtype=torch.bfloat16,vocab_size=32000,
            head_dim=128,num_attention_heads=32,num_key_value_heads=4)
        with patch('torch.cuda.get_device_name',return_value='test'), \
             patch('torch.cuda.get_device_capability',return_value=(12,0)), \
             patch('ssd.layers.fi_attn.attention_backend',return_value='flashinfer'):
            draft=runtime_manifest(cfg,hf,2,True,1,torch.device('cuda:2'))
            target=runtime_manifest(cfg,hf,0,False,2,torch.device('cuda:0'))
        self.assertEqual(draft['target_tp'],2)
        self.assertEqual(draft['model_tp'],1)
        self.assertEqual(draft['local_q_heads'],32)
        self.assertEqual(target['target_tp'],2)
        self.assertEqual(target['model_tp'],2)
        self.assertEqual(target['local_q_heads'],16)
        self.assertEqual(target['attention_backend'],'flashinfer')
