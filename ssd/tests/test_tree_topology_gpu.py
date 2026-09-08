import unittest

import torch

from ssd.engine.helpers.p2_tree import (
    pack_tree_proxy_topology,
    pack_tree_verify_mask_direct,
)
from ssd.engine.helpers.tree_topology_gpu import (
    build_tree_verify_inputs_gpu_,
    pack_tree_proxy_topology_gpu_,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestTreeTopologyGPU(unittest.TestCase):
    """The GPU preparation path must be byte-identical to the CPU fallback."""

    TOPOLOGIES = (
        ([-1], [0]),
        ([-1, -1, -1, 0], [0, 1, 2, 0]),
        ([-1, -1, 0, 2, 1, 4, 3, 6],
         [0, 1, 0, 0, 0, 0, 0, 0]),
    )

    def test_proxy_topology_matches_cpu_reference(self):
        for parents, siblings in self.TOPOLOGIES:
            nv = len(parents)
            expected = pack_tree_proxy_topology(parents, siblings, nv)
            actual = pack_tree_proxy_topology([], [], nv, device="cuda")
            pack_tree_proxy_topology_gpu_(
                torch.tensor(parents, dtype=torch.int64, device="cuda"),
                torch.tensor(siblings, dtype=torch.int64, device="cuda"),
                actual,
            )
            torch.cuda.synchronize()
            for field in expected:
                self.assertTrue(
                    torch.equal(actual[field].cpu(), expected[field]),
                    f"field={field}, parents={parents}",
                )

    def test_verify_rows_and_mask_match_cpu_reference(self):
        for parents, _ in self.TOPOLOGIES:
            valid = len(parents)
            ancestors = [[] for _ in range(valid)]
            depths = [0] * valid
            for node, parent in enumerate(parents):
                if parent >= 0:
                    ancestors[node] = ancestors[parent] + [parent]
                    depths[node] = depths[parent] + 1

            parent_gpu = torch.tensor(
                parents, dtype=torch.int64, device="cuda")
            source_ids = torch.arange(
                valid + 1, dtype=torch.int64, device="cuda") + 101
            source_slots = torch.arange(
                valid + 1, dtype=torch.int32, device="cuda") + 301
            for rows in (valid + 1, valid + 2):
                # Include byte boundaries and long-context regimes used by
                # the 4096-token extended-RoPE experiments.
                for prefix_len in (1, 7, 8, 31, 255, 1023, 2047):
                    kv_len = prefix_len + valid
                    pos0 = prefix_len - 1
                    out_ids = torch.empty(
                        rows, dtype=torch.int64, device="cuda")
                    out_rope = torch.empty_like(out_ids)
                    out_slots = torch.empty(
                        rows, dtype=torch.int32, device="cuda")
                    out_context = torch.empty(
                        1, dtype=torch.int32, device="cuda")
                    out_mask = torch.empty(
                        (rows * kv_len + 7) // 8,
                        dtype=torch.uint8,
                        device="cuda",
                    )

                    mask_bytes = build_tree_verify_inputs_gpu_(
                        parent_gpu,
                        source_ids,
                        source_slots,
                        out_ids,
                        out_rope,
                        out_slots,
                        out_context,
                        out_mask,
                        valid=valid,
                        rows=rows,
                        pos0=pos0,
                        prefix_len=prefix_len,
                        kv_len=kv_len,
                    )
                    torch.cuda.synchronize()

                    expected_ids = torch.zeros(rows, dtype=torch.int64)
                    expected_ids[:valid + 1] = source_ids.cpu()
                    expected_slots = torch.full(
                        (rows,), -1, dtype=torch.int32)
                    expected_slots[:valid + 1] = source_slots.cpu()
                    expected_rope = torch.full(
                        (rows,), pos0, dtype=torch.int64)
                    expected_rope[1:valid + 1] = pos0 + 1 + torch.tensor(
                        depths, dtype=torch.int64)
                    expected_mask = pack_tree_verify_mask_direct(
                        ancestors, valid, rows, prefix_len, kv_len)

                    self.assertEqual(mask_bytes, expected_mask.numel())
                    self.assertTrue(torch.equal(out_ids.cpu(), expected_ids))
                    self.assertTrue(torch.equal(out_slots.cpu(), expected_slots))
                    self.assertTrue(torch.equal(out_rope.cpu(), expected_rope))
                    self.assertEqual(int(out_context.cpu()[0]), kv_len)
                    self.assertTrue(torch.equal(
                        out_mask[:mask_bytes].cpu(), expected_mask))


if __name__ == "__main__":
    unittest.main()
