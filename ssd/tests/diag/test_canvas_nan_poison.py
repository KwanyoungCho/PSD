"""canvas 페이지 유한성 필수 조건 (2026-08-06 실증).

mask=0이어도 canvas 페이지에 Inf/NaN이 있으면 fa2 출력 전체가
NaN이 된다 (산술식 마스킹의 0×inf=NaN). 따라서 canvas 슬롯에
-1(OOB 읽기 — 내용 비보장)을 두는 것은 금지이며, 반드시 유한값이
보장되는 유효 페이지를 넣어야 한다 (draft_runner _pages_fill 규약).

이 테스트는 그 사실 자체를 고정한다:
- 유한 sentinel(777) canvas → 기준과 bit-동일 (중복-페이지 포함
  무해성은 test_canvas_dup_page가 고정)
- Inf/NaN canvas → 출력 NaN 전파 (이 성질이 사라지면 fa2 동작이
  바뀐 것 — 규약 재평가 필요)
"""
import unittest
import torch

try:
    import flashinfer
    HAS_FI = True
except Exception:
    HAS_FI = False


@unittest.skipUnless(HAS_FI and torch.cuda.is_available(), "no fi/cuda")
class TestCanvasNanPoison(unittest.TestCase):
    def _run(self, fill):
        dev = "cuda:0"
        H, HKV, D, PAGE = 32, 4, 64, 256
        W, ctx0, p0 = 10, 533, 3
        g = torch.Generator(device=dev).manual_seed(0)
        cache = (torch.randn(p0 + 1, 2, PAGE, HKV, D, generator=g,
                             device=dev) * 0.1).half()
        cache[p0].fill_(fill)
        q = (torch.randn(W, H, D, generator=g, device=dev) * 0.1).half()
        m = torch.zeros(W, (p0 + 1) * PAGE, dtype=torch.bool,
                        device=dev)
        m[:, :ctx0] = True
        for i in range(W):
            m[i, ctx0 + i] = True
        ws = torch.empty(128 * 2**20, dtype=torch.uint8, device=dev)
        wr = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws, "NHD", backend="fa2")
        wr.plan(torch.tensor([0, W], dtype=torch.int32, device=dev),
                torch.tensor([0, p0 + 1], dtype=torch.int32,
                             device=dev),
                torch.arange(p0 + 1, dtype=torch.int32, device=dev),
                torch.tensor([PAGE], dtype=torch.int32, device=dev),
                H, HKV, D, PAGE, custom_mask=m.reshape(-1),
                q_data_type=torch.float16, kv_data_type=torch.float16)
        return wr.run(q, (cache[:, 0], cache[:, 1])).float()

    def test_finite_canvas_is_inert(self):
        o0 = self._run(0.0)
        o777 = self._run(777.0)
        self.assertEqual(float((o0 - o777).abs().max()), 0.0)

    def test_inf_nan_canvas_poisons_output(self):
        for bad in (float("inf"), float("nan")):
            o = self._run(bad)
            self.assertTrue(bool(torch.isnan(o).any()),
                            f"canvas={bad}: NaN 전파가 사라짐 — "
                            f"fa2 마스킹 방식 변경? 규약 재평가 필요")


if __name__ == "__main__":
    unittest.main()
