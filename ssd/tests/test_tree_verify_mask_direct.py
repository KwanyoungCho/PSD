import unittest

import torch

from ssd.engine.helpers.p2_tree import (
    pack_tree_verify_mask, pack_tree_verify_mask_direct)


class TestTreeVerifyMaskDirect(unittest.TestCase):
    def test_matches_dense_reference_across_unaligned_rows(self):
        parents = [-1, -1, 0, 2, 1, 4, 3, 6]
        for valid in range(1, len(parents) + 1):
            ancestors = [[] for _ in range(valid)]
            for node in range(valid):
                parent = parents[node]
                if parent >= 0:
                    ancestors[node] = ancestors[parent] + [parent]
            for rows in (valid + 1, valid + 2):
                for prefix in (1, 7, 8, 9, 255):
                    kv_len = prefix + valid
                    dense = torch.zeros(rows, kv_len, dtype=torch.bool)
                    dense[:, :prefix] = True
                    for node in range(valid):
                        row = node + 1
                        for ancestor in ancestors[node]:
                            dense[row, prefix + ancestor] = True
                        dense[row, prefix + node] = True
                    expected = pack_tree_verify_mask(dense)
                    actual = pack_tree_verify_mask_direct(
                        ancestors, valid, rows, prefix, kv_len)
                    self.assertTrue(torch.equal(actual, expected), (
                        f"valid={valid}, rows={rows}, prefix={prefix}"))


if __name__ == "__main__":
    unittest.main()
