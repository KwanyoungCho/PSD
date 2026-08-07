"""Target tree-verify CUDA graph: runtime buffer update vs FlashInfer plan.

The production graph is captured once for a fixed query-width bucket.  This
test varies page count, last-page length, page ids and tree mask after capture
and verifies that direct updates of the graph-owned buffers produce the same
attention output as the former per-hit ``wrapper.plan()`` path.
"""
import unittest

import numpy as np
import torch

try:
    import flashinfer
    HAS_FLASHINFER = True
except Exception:
    HAS_FLASHINFER = False

from ssd.engine.helpers.cudagraph_helpers import (
    update_tree_verify_graph_buffers)


@unittest.skipUnless(HAS_FLASHINFER and torch.cuda.is_available(),
                     "CUDA FlashInfer is required")
class TestTreeVerifyPlanless(unittest.TestCase):
    def test_direct_buffer_update_matches_runtime_plan(self):
        dev = torch.device("cuda:0")
        page, rows, max_pages = 16, 7, 5
        h, hkv, dim = 4, 2, 64
        workspace = torch.empty(96 * 2**20, dtype=torch.uint8, device=dev)
        qo_buf = torch.empty(2, dtype=torch.int32, device=dev)
        kv_buf = torch.empty(2, dtype=torch.int32, device=dev)
        ids_buf = torch.empty(max_pages, dtype=torch.int32, device=dev)
        last_buf = torch.empty(1, dtype=torch.int32, device=dev)
        mask_buf = torch.empty((rows * max_pages * page + 7) // 8,
                               dtype=torch.uint8, device=dev)
        mask_indptr_buf = torch.empty(2, dtype=torch.int32, device=dev)
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, "NHD", backend="fa2", use_cuda_graph=True,
            qo_indptr_buf=qo_buf, paged_kv_indptr_buf=kv_buf,
            paged_kv_indices_buf=ids_buf,
            paged_kv_last_page_len_buf=last_buf,
            custom_mask_buf=mask_buf, mask_indptr_buf=mask_indptr_buf)
        wrapper._kv_lens_buffer = torch.empty(
            1, dtype=torch.int32, device=dev)

        qo = torch.tensor([0, rows], dtype=torch.int32)
        wrapper.plan(
            qo.to(dev), torch.tensor([0, 1], dtype=torch.int32, device=dev),
            torch.tensor([0], dtype=torch.int32, device=dev),
            torch.tensor([page], dtype=torch.int32, device=dev),
            h, hkv, dim, page,
            custom_mask=torch.ones(rows * page, dtype=torch.bool,
                                   device=dev),
            q_data_type=torch.float16, kv_data_type=torch.float16)

        torch.manual_seed(19)
        q = torch.randn(rows, h, dim, dtype=torch.float16, device=dev)
        cache = torch.randn(max_pages, 2, page, hkv, dim,
                            dtype=torch.float16, device=dev)
        out = torch.empty_like(q)
        # Warm and capture only once, with the one-page representative plan.
        out.copy_(wrapper.run(q, cache))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out.copy_(wrapper.run(q, cache))

        for n_pages, last_len in ((1, 1), (2, page - 1), (3, page)):
            kv_len = (n_pages - 1) * page + last_len
            page_ids = torch.arange(
                n_pages - 1, -1, -1, dtype=torch.int32, device=dev)
            mask = torch.zeros(rows, kv_len, dtype=torch.bool)
            prefix = max(1, kv_len - rows)
            mask[:, :prefix] = True
            for row in range(rows):
                mask[row, prefix:min(kv_len, prefix + row + 1)] = True
            packed_cpu = torch.from_numpy(np.packbits(
                mask.numpy().reshape(-1), bitorder="little"))
            host = {
                "kv": torch.tensor([0, n_pages], dtype=torch.int32),
                "last": torch.tensor([last_len], dtype=torch.int32),
                "mask": torch.tensor([0, rows * kv_len],
                                     dtype=torch.int32),
                "klen": torch.tensor([kv_len], dtype=torch.int32),
            }

            # New path first: no plan call for this shape.
            update_tree_verify_graph_buffers(
                wrapper, page_ids, packed_cpu, host)
            graph.replay()
            torch.cuda.synchronize()
            direct = out.clone()

            # Old path: public plan() recopies the same runtime contents.
            wrapper.plan(
                qo, host["kv"], page_ids, host["last"],
                h, hkv, dim, page,
                packed_custom_mask=packed_cpu.to(dev),
                q_data_type=torch.float16, kv_data_type=torch.float16)
            graph.replay()
            torch.cuda.synchronize()
            planned = out.clone()
            self.assertTrue(
                torch.equal(direct, planned),
                f"n_pages={n_pages}, last={last_len}, "
                f"max_diff={(direct - planned).abs().max().item()}")


if __name__ == "__main__":
    unittest.main()
