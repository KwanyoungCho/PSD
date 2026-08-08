import os
import unittest
from unittest.mock import patch

from ssd.layers.fi_attn import (
    attention_backend,
    duet_graph_name,
    graph_batch_sizes,
)


class TestFlashInferAttentionBackend(unittest.TestCase):
    def test_explicit_attention_backend_wins(self):
        with patch.dict(os.environ, {"SSD_ATTN_BACKEND": "flashinfer"}):
            self.assertEqual(attention_backend(), "flashinfer")
        with patch.dict(os.environ, {"SSD_ATTN_BACKEND": "sgl"}):
            self.assertEqual(attention_backend(), "sgl")

    def test_auto_backend_uses_build_arch_without_cuda(self):
        with patch("torch.cuda.is_available", return_value=False):
            with patch.dict(
                os.environ,
                {"SSD_ATTN_BACKEND": "auto", "SSD_CUDA_ARCH": "12.0"},
            ):
                self.assertEqual(attention_backend(), "flashinfer")
            with patch.dict(
                os.environ,
                {"SSD_ATTN_BACKEND": "auto", "SSD_CUDA_ARCH": "8.0"},
            ):
                self.assertEqual(attention_backend(), "sgl")

    def test_invalid_attention_backend_fails_loudly(self):
        with patch.dict(os.environ, {"SSD_ATTN_BACKEND": "unknown"}):
            with self.assertRaisesRegex(ValueError, "auto\\|sgl\\|flashinfer"):
                attention_backend()

    def test_graph_buckets_fit_backing_rows(self):
        cases = [
            (1, [1]),
            (2, [1, 2]),
            (3, [1, 2, 3]),
            (8, [1, 2, 4, 8]),
            (17, [1, 2, 4, 8, 16, 17]),
        ]
        for max_bs, expected in cases:
            with self.subTest(max_bs=max_bs):
                self.assertEqual(graph_batch_sizes(max_bs), expected)
                self.assertEqual(max(graph_batch_sizes(max_bs)), max_bs)

    def test_duet_graph_name_is_width_stable(self):
        self.assertEqual(duet_graph_name(8), "duet_verify_kp8")
        with self.assertRaises(ValueError):
            duet_graph_name(0)


if __name__ == "__main__":
    unittest.main()
