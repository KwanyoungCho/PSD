"""단계0 — fa2 preplanned vs auto JIT kernel 직접 대조 (assertion판).

동일 입력·동일 mask에서 attention 출력 차를 측정. **미니 형상과
실모델 형상(TinyLlama-1B draft: H32/HKV4/D64, block 256)을 분리
기록** (고정 지침 — 미니 결과만으로 실엔진 배제 선언 금지).

실행: python -m unittest tests.diag.test_fa2_vs_auto_kernel
"""
import unittest
import torch

try:
    import flashinfer
    HAS_FI = True
except Exception:
    HAS_FI = False


def run_pair(H, HKV, D, PAGE, ctx0, W, trial, dev):
    g = torch.Generator(device=dev).manual_seed(trial)
    kv_len = ctx0 + W
    p = (kv_len + PAGE - 1) // PAGE
    lpl = kv_len - (p - 1) * PAGE
    cache = (torch.randn(p, 2, PAGE, HKV, D, generator=g,
                         device=dev) * 0.1).half()
    q = (torch.randn(W, H, D, generator=g, device=dev) * 0.1).half()
    mask = torch.zeros(W, kv_len, dtype=torch.bool, device=dev)
    mask[:, :ctx0] = True
    for i in range(W):
        mask[i, ctx0 + i] = True
    qo = torch.tensor([0, W], dtype=torch.int32, device=dev)
    po = torch.tensor([0, p], dtype=torch.int32, device=dev)
    kvi = torch.arange(p, dtype=torch.int32, device=dev)
    lastp = torch.tensor([lpl], dtype=torch.int32, device=dev)

    outs = {}
    for backend in ("auto", "fa2"):
        ws = torch.empty(128 * 2**20, dtype=torch.uint8, device=dev)
        wr = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws, "NHD", backend=backend)
        wr.plan(qo, po, kvi, lastp, H, HKV, D, PAGE,
                custom_mask=mask.reshape(-1),
                q_data_type=torch.float16, kv_data_type=torch.float16)
        outs[backend] = wr.run(q, (cache[:, 0], cache[:, 1])) \
            .float().reshape(W, H * D)
    d = outs["fa2"] - outs["auto"]
    return (float(d.abs().max()), float(d.mean()),
            float(torch.nn.functional.cosine_similarity(
                outs["auto"].flatten(), outs["fa2"].flatten(),
                dim=0)))


@unittest.skipUnless(HAS_FI and torch.cuda.is_available(), "no fi/cuda")
class TestFa2VsAutoKernel(unittest.TestCase):
    """확인 판정 어휘: '배제'는 각 형상에서의 직접 측정으로만.
    - 미니 형상: abs_max==0 요구 (2026-08-06 실측 bit-동일)
    - 실모델 형상: 결과를 별도 기준으로 assert (측정 후 고정)"""

    def test_mini_shape_bit_identical(self):
        dev = "cuda:0"
        for trial in range(5):
            mx, mean, cos = run_pair(H=4, HKV=2, D=64, PAGE=64,
                                     ctx0=85, W=10, trial=trial,
                                     dev=dev)
            self.assertEqual(mx, 0.0,
                             f"미니 형상 trial {trial}: abs_max={mx}")

    def test_real_draft_shape(self):
        """실모델 형상 (TinyLlama-1B: H32/HKV4/D64, block 256,
        ctx≈512+): bit-동일이면 kernel을 실형상에서도 배제 가능.
        아니면 abs_max를 기록하고 systematic bias(mean)만 0 요구."""
        dev = "cuda:0"
        results = []
        for trial in range(5):
            mx, mean, cos = run_pair(H=32, HKV=4, D=64, PAGE=256,
                                     ctx0=512 + 21, W=10, trial=trial,
                                     dev=dev)
            results.append((mx, mean, cos))
        print(f"[real-shape] {[(f'{m:.2e}', f'{mn:+.2e}', f'{c:.6f}') for m, mn, c in results]}")
        for i, (mx, mean, cos) in enumerate(results):
            self.assertEqual(
                mx, 0.0,
                f"실형상 trial {i}: fa2≠auto (abs_max={mx:.3e}, "
                f"cos={cos:.6f}) — 실형상에서 kernel 배제 불가, "
                f"단계1 로짓 대조에서 kernel 요인 고려 필요")


if __name__ == "__main__":
    unittest.main()
