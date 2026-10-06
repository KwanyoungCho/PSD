"""Noncontiguous prefix KV pages and bounded batch admission."""
import os
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("SSD_HF_CACHE", "/tmp")
os.environ.setdefault("SSD_DATASET_DIR", "/tmp")
import torch
from ssd.layers.attention import Attention
from ssd.utils.context import set_context, reset_context
from ssd.engine.scheduler import Scheduler
from ssd.engine.sequence import Sequence


class TestAdmission(unittest.TestCase):
    def test_prefill_respects_live_batch_limit(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_num_seqs = 2
        scheduler.max_num_batched_tokens = 10000
        scheduler.running = deque([Sequence([1, 2, 3])])
        scheduler.waiting = deque(Sequence([4, 5, 6]) for _ in range(4))
        scheduler.speculate = False
        scheduler.bms_can_allocate = lambda seq: True
        scheduler.block_manager = SimpleNamespace(allocate=lambda seq: None)
        admitted, prefill = scheduler.schedule()
        self.assertTrue(prefill)
        self.assertEqual(len(admitted), 1)
        self.assertEqual(len(scheduler.running), 2)
        self.assertEqual(len(scheduler.waiting), 3)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestCachedPrefill(unittest.TestCase):
    def test_mixed_cached_uncached_rows_match_dense_attention(self):
        torch.manual_seed(9)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), patch.dict(os.environ, {"SSD_ATTN_BACKEND": "sgl"}):
                h, kvh, dim, page = 4, 2, 64, 256
                q = torch.randn(8, h, dim, device="cuda", dtype=dtype)
                keys = [torch.randn(n, kvh, dim, device="cuda", dtype=dtype)
                        for n in (259, 5)]
                vals = [torch.randn_like(k) for k in keys]
                attn = Attention(h, dim, dim**-.5, kvh).cuda()
                attn.k_cache = torch.zeros(8, page, kvh, dim, device="cuda", dtype=dtype)
                attn.v_cache = torch.zeros_like(attn.k_cache)
                attn.k_cache[5] = keys[0][:page]
                attn.v_cache[5] = vals[0][:page]
                new_k = torch.cat([keys[0][page:], keys[1]])
                new_v = torch.cat([vals[0][page:], vals[1]])
                slots = torch.tensor([2*page+i for i in range(3)] +
                                     [7*page+i for i in range(5)], device="cuda", dtype=torch.int32)
                set_context(True,
                    cu_seqlens_q=torch.tensor([0, 3, 8], device="cuda", dtype=torch.int32),
                    cu_seqlens_k=torch.tensor([0, 259, 264], device="cuda", dtype=torch.int32),
                    max_seqlen_q=5, max_seqlen_k=259, slot_mapping=slots,
                    block_tables=torch.tensor([[5, 2], [7, -1]], device="cuda", dtype=torch.int32))
                try:
                    actual = attn(q.flatten(1), new_k.flatten(1), new_v.flatten(1)).view(8, h, dim)
                finally:
                    reset_context()
                expected = []
                offset = 0
                for row, prefix in enumerate((256, 0)):
                    length = len(keys[row]) - prefix
                    k = keys[row].repeat_interleave(h//kvh, dim=1).float()
                    v = vals[row].repeat_interleave(h//kvh, dim=1).float()
                    for j in range(length):
                        prob = torch.einsum("hd,nhd->hn", q[offset+j].float(), k[:prefix+j+1]) * dim**-.5
                        expected.append(torch.einsum("hn,nhd->hd", prob.softmax(-1), v[:prefix+j+1]))
                    offset += length
                torch.testing.assert_close(actual.float(), torch.stack(expected), atol=.012, rtol=.015)


if __name__ == "__main__":
    unittest.main()
