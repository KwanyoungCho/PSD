"""Target tree-verify CUDA graph page-canvas equivalence.

FlashInfer plan output is launch geometry, not mutable tensor content.  A
graph must therefore be captured per page count.  Within that bucket we use a
full-last-page canvas and zero-mask the tail.  This test compares that captured
canvas execution against an exact-length eager attention run.
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
    def test_page_specific_canvas_graph_matches_exact_eager(self):
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

        torch.manual_seed(19)
        q = torch.randn(rows, h, dim, dtype=torch.float16, device=dev)
        cache = torch.randn(max_pages, 2, page, hkv, dim,
                            dtype=torch.float16, device=dev)
        out = torch.empty_like(q)
        eager_ws = torch.empty(
            96 * 2**20, dtype=torch.uint8, device=dev)
        eager = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            eager_ws, "NHD", backend="fa2")
        qo = torch.tensor([0, rows], dtype=torch.int32)

        for n_pages, last_len in ((1, 1), (2, page - 1), (3, page)):
            kv_len = (n_pages - 1) * page + last_len
            canvas_len = n_pages * page
            page_ids = torch.arange(
                n_pages - 1, -1, -1, dtype=torch.int32, device=dev)
            mask = torch.zeros(rows, kv_len, dtype=torch.bool)
            prefix = max(1, kv_len - rows)
            mask[:, :prefix] = True
            for row in range(rows):
                mask[row, prefix:min(kv_len, prefix + row + 1)] = True
            packed_cpu = torch.from_numpy(np.packbits(
                mask.numpy().reshape(-1), bitorder="little"))
            canvas_mask = torch.zeros(rows, canvas_len, dtype=torch.bool)
            canvas_mask[:, :kv_len] = mask
            canvas_packed = torch.from_numpy(np.packbits(
                canvas_mask.numpy().reshape(-1), bitorder="little"))

            # Capture with page-count-specific launch geometry and a full
            # final page.  Dynamic page ids and mask contents are installed
            # exactly as production does immediately before replay.
            wrapper.plan(
                qo.to(dev),
                torch.tensor([0, n_pages], dtype=torch.int32, device=dev),
                torch.zeros(n_pages, dtype=torch.int32, device=dev),
                torch.tensor([page], dtype=torch.int32, device=dev),
                h, hkv, dim, page,
                custom_mask=torch.zeros(
                    rows * canvas_len, dtype=torch.bool, device=dev),
                q_data_type=torch.float16, kv_data_type=torch.float16)
            out.copy_(wrapper.run(q, cache))
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out.copy_(wrapper.run(q, cache))
            host = {
                "kv": torch.tensor([0, n_pages], dtype=torch.int32),
                "last": torch.tensor([page], dtype=torch.int32),
                "mask": torch.tensor([0, rows * canvas_len],
                                     dtype=torch.int32),
                "klen": torch.tensor([canvas_len], dtype=torch.int32),
            }
            update_tree_verify_graph_buffers(
                wrapper, page_ids, canvas_packed, host)
            graph.replay()
            torch.cuda.synchronize()
            canvas_out = out.clone()

            eager.plan(
                qo, host["kv"], page_ids,
                torch.tensor([last_len], dtype=torch.int32),
                h, hkv, dim, page,
                packed_custom_mask=packed_cpu.to(dev),
                q_data_type=torch.float16, kv_data_type=torch.float16)
            exact_out = eager.run(q, cache)
            torch.cuda.synchronize()
            self.assertTrue(
                torch.allclose(canvas_out, exact_out, atol=2e-3, rtol=2e-3),
                f"n_pages={n_pages}, last={last_len}, "
                f"max_diff={(canvas_out - exact_out).abs().max().item()}")


if __name__ == "__main__":
    unittest.main()
