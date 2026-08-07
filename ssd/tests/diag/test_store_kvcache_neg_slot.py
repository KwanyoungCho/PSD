"""store_kvcache 범위 밖 slot이 KV를 일절 변경하지 않아야 한다."""
import unittest
import torch

from ssd.layers.attention import store_kvcache


@unittest.skipUnless(torch.cuda.is_available(), "no cuda")
class TestStoreKvNegSlot(unittest.TestCase):
    def test_out_of_range_slots_no_write(self):
        dev = "cuda:0"
        N, H, D = 8, 4, 64
        k = torch.randn(N, H, D, device=dev, dtype=torch.float16)
        v = torch.randn(N, H, D, device=dev, dtype=torch.float16)
        # 실형상: (blocks, block_size, H, D) — slot은 flat 인덱스
        kc = torch.full((2, 64, H, D), 7.0, device=dev,
                        dtype=torch.float16)
        vc = kc.clone()
        kcf = kc.view(-1, H, D)
        vcf = vc.view(-1, H, D)
        slots = torch.tensor([0, 1, -1, -2, -256, 5, 128, 999],
                             dtype=torch.int32, device=dev)
        kc0, vc0 = kcf.clone(), vcf.clone()
        store_kvcache(k, v, kc, vc, slots)
        torch.cuda.synchronize()
        pos = [0, 1, 5]
        for i, s in enumerate(slots.tolist()):
            if 0 <= s < 128:
                self.assertTrue(torch.equal(
                    kcf[s], k[i]), f"slot {s} 기록 누락")
        untouched_rows = [r for r in range(128) if r not in pos]
        self.assertTrue(torch.equal(kcf[untouched_rows], kc0[untouched_rows]),
                        "범위 밖 slot이 KV를 변경함")
        self.assertTrue(torch.equal(
            vcf[untouched_rows], vc0[untouched_rows]))


if __name__ == "__main__":
    unittest.main()
