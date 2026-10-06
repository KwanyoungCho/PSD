"""Greedy recovery must use the verified prefix, including ragged batches."""
import os
import unittest

os.environ.setdefault("SSD_HF_CACHE", "/tmp")
os.environ.setdefault("SSD_DATASET_DIR", "/tmp")

import torch
from ssd.utils.verify import verify


class TestGreedyRecovery(unittest.TestCase):
    def _check_ragged(self, device, mixed_temperature=False):
        # Short row accepts its two real proposals and then matches padding.
        # Recovery at column 2 differs from the eventual padded mismatch.
        p = torch.full((3, 5, 8), -100.0, device=device)
        q = torch.full((3, 4, 8), -100.0, device=device)
        pred = torch.tensor([[1, 2, 0, 5, 6], [1, 2, 3, 4, 7],
                             [1, 5, 3, 4, 6]], device=device)
        p.scatter_(2, pred.unsqueeze(-1), 100.0)
        spec = torch.tensor([[7, 1, 2, 0, 3], [7, 1, 2, 3, 4],
                             [7, 1, 2, 3, 4]], device=device)
        q.scatter_(2, spec[:, 1:].unsqueeze(-1), 100.0)
        temps = torch.tensor([0., .7 if mixed_temperature else 0., 0.],
                             device=device)
        suffix, recovery = verify(p, q, spec, temps, temps,
                                  jit_speculate=True,
                                  valid_k=torch.tensor([2, 4, 3], device=device))
        self.assertEqual(suffix, [[7, 1, 2], [7, 1, 2, 3, 4], [7, 1]])
        self.assertEqual(recovery, [0, 7, 5])

    def test_cpu_ragged(self):
        self._check_ragged("cpu")

    def test_cpu_mixed_temperature(self):
        self._check_ragged("cpu", mixed_temperature=True)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_ragged(self):
        self._check_ragged("cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_mixed_temperature(self):
        self._check_ragged("cuda", mixed_temperature=True)


if __name__ == "__main__":
    unittest.main()
