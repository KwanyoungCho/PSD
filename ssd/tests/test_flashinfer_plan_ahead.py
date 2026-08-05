"""23번 — FlashInfer round별 plan-상태 분리 검증 (리뷰10-6 기준 ⓐ-ⓓ,ⓕ).

질문: P2 4 round의 plan을 '미리' 각자 wrapper에 준비해 두고 나중에
run만 순서대로 호출해도, 매 round 직전 plan하는 현행 경로와 결과가
정확히 같은가? (같아야 전체-P2 graph에서 plan을 밖으로 뺄 수 있다)

ⓔ(시간)는 엔진 PoC 단계에서 전체 step wall로 측정.
"""
import unittest
import torch

try:
    import flashinfer
    HAS_FI = True
except Exception:
    HAS_FI = False

PAGE = 16          # 검증용 소형 page (엔진 256과 의미 동일)
H, HKV, D = 4, 2, 64
W, F = 10, 4       # P2 폭·round 수


def _mk_cache(num_pages, device):
    return torch.randn(num_pages, 2, PAGE, HKV, D, dtype=torch.float16,
                       device=device)


def _geometry(ctx, f):
    """round f의 kv 길이/페이지 구성 (트리 rollout과 동일: ctx+f·W)."""
    kv_len = ctx + f * W
    n_pages = (kv_len + PAGE - 1) // PAGE
    lpl = kv_len - (n_pages - 1) * PAGE
    return kv_len, n_pages, lpl


def _mk_mask(ctx, f, device):
    """(ctx, f) 결정적 mask — A/B 두 경로에 '동일' 주입 (RNG는 고정
    제너레이터: 경로별 호출 순서와 무관)."""
    kv_len, _n, _l = _geometry(ctx, f)
    g = torch.Generator().manual_seed(ctx * 100 + f)
    mask = torch.zeros(W, kv_len, dtype=torch.bool)
    mask[:, :ctx] = True
    if f:
        for r in range(W):
            mask[r, ctx:ctx + f * W] = torch.rand(
                f * W, generator=g) > 0.5
            mask[r, ctx + (f - 1) * W + (r % W)] = True
    return mask.to(device)


def _plan(wr, ctx, f, mask, device):
    kv_len, n_pages, lpl = _geometry(ctx, f)
    qo_indptr = torch.tensor([0, W], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, n_pages], dtype=torch.int32,
                             device=device)
    kv_indices = torch.arange(n_pages, dtype=torch.int32, device=device)
    lpl_t = torch.tensor([lpl], dtype=torch.int32, device=device)
    wr.plan(qo_indptr, kv_indptr, kv_indices, lpl_t,
            H, HKV, D, PAGE, custom_mask=mask.reshape(-1),
            q_data_type=torch.float16, kv_data_type=torch.float16)
    return kv_len


@unittest.skipUnless(HAS_FI and torch.cuda.is_available(), "no flashinfer")
class TestPlanAheadParity(unittest.TestCase):
    def _run_case(self, ctx, note):
        dev = "cuda:0"
        torch.manual_seed(11)
        max_pages = (ctx + F * W + PAGE - 1) // PAGE + 1
        cache = _mk_cache(max_pages, dev)
        qs = [torch.randn(W, H, D, dtype=torch.float16, device=dev)
              for _ in range(F)]
        float_ws = torch.empty(96 * 1024 * 1024, dtype=torch.uint8,
                               device=dev)
        # ── 방식 A: round별 wrapper 4개, plan 전부 선행 (ⓐ)
        masks = [_mk_mask(ctx, f, dev) for f in range(F)]
        wrs = [flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            float_ws, "NHD") for _ in range(F)]
        for f in range(F):
            _plan(wrs[f], ctx, f, masks[f], dev)
        outs_pre = []
        for f in range(F):
            outs_pre.append(wrs[f].run(qs[f], cache))
        # ── 방식 B(현행): 매 round 직전 plan (같은 wrapper 재사용)
        wr_jit = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            float_ws, "NHD")
        outs_jit = []
        for f in range(F):
            _plan(wr_jit, ctx, f, masks[f], dev)
            outs_jit.append(wr_jit.run(qs[f], cache))
        for f in range(F):
            self.assertTrue(
                torch.allclose(outs_pre[f], outs_jit[f], atol=2e-3,
                               rtol=2e-3),
                f"{note}: round {f} preplanned != JIT-planned "
                f"(max diff {(outs_pre[f]-outs_jit[f]).abs().max()})")

    def test_page_boundary_lpl_1(self):
        # 마지막 page 길이 1 (ⓑ)
        self._run_case(ctx=PAGE * 3 + 1 - 0 * W, note="lpl=1")

    def test_page_boundary_lpl_max(self):
        self._run_case(ctx=PAGE * 3, note="lpl=PAGE")

    def test_page_boundary_lpl_minus1(self):
        self._run_case(ctx=PAGE * 3 - 1, note="lpl=PAGE-1")

    def test_round_crosses_new_page(self):
        # round 진행 중 새 page 진입 (ctx가 page 끝 근처 → f·W가 넘김)
        self._run_case(ctx=PAGE * 2 + PAGE - 3, note="page-cross")

    def test_poisoned_canvas_padding(self):
        """ⓒ: page-끝 canvas — 실KV 밖 슬롯에 쓰레기 KV + mask=0이
        정확-길이 결과와 일치해야 canvas 캡처가 안전하다."""
        dev = "cuda:0"
        torch.manual_seed(7)
        ctx = PAGE * 2 + 5
        f = 2
        kv_len, n_pages, lpl = _geometry(ctx, f)
        pad_to = n_pages * PAGE                 # page 끝까지 canvas
        cache = _mk_cache(n_pages + 1, dev)
        # 실KV 밖 슬롯 오염
        poison = cache.clone()
        flat = poison.view(-1, HKV, D)
        # 마지막 page의 [lpl:] 슬롯 오염 (k·v 모두)
        base = (n_pages - 1) * 2 * PAGE
        for kv in range(2):
            poison[n_pages - 1, kv, lpl:] = 1e4
        q = torch.randn(W, H, D, dtype=torch.float16, device=dev)
        float_ws = torch.empty(96 * 1024 * 1024, dtype=torch.uint8,
                               device=dev)
        qo = torch.tensor([0, W], dtype=torch.int32, device=dev)
        kvp = torch.tensor([0, n_pages], dtype=torch.int32, device=dev)
        kvi = torch.arange(n_pages, dtype=torch.int32, device=dev)
        mask_exact = torch.ones(W, kv_len, dtype=torch.bool, device=dev)
        # 정확 길이 (lpl 실값)
        wr1 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            float_ws, "NHD")
        wr1.plan(qo, kvp, kvi, torch.tensor([lpl], dtype=torch.int32,
                                            device=dev),
                 H, HKV, D, PAGE, custom_mask=mask_exact.reshape(-1),
                 q_data_type=torch.float16, kv_data_type=torch.float16)
        out_exact = wr1.run(q, cache)
        # canvas (lpl=PAGE로 확장 + 오염 + 초과분 mask=0)
        mask_canvas = torch.zeros(W, pad_to, dtype=torch.bool,
                                  device=dev)
        mask_canvas[:, :kv_len] = True
        wr2 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            float_ws, "NHD")
        wr2.plan(qo, kvp, kvi, torch.tensor([PAGE], dtype=torch.int32,
                                            device=dev),
                 H, HKV, D, PAGE, custom_mask=mask_canvas.reshape(-1),
                 q_data_type=torch.float16, kv_data_type=torch.float16)
        out_canvas = wr2.run(q, poison)
        self.assertTrue(
            torch.allclose(out_exact, out_canvas, atol=2e-3, rtol=2e-3),
            f"canvas != exact (max diff "
            f"{(out_exact-out_canvas).abs().max()})")

    def test_memory_report(self):
        """ⓕ: wrapper별 int workspace 크기 보고 (float ws는 공유)."""
        dev = "cuda:0"
        float_ws = torch.empty(96 * 1024 * 1024, dtype=torch.uint8,
                               device=dev)
        wr = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            float_ws, "NHD")
        int_ws = wr._int_workspace_buffer
        print(f"[plan-ahead ⓕ] float ws(공유) {float_ws.numel()/2**20:.0f}"
              f"MB, wrapper당 int ws {int_ws.numel()/2**20:.1f}MB "
              f"→ round×4 추가 {4*int_ws.numel()/2**20:.1f}MB")
        self.assertLess(4 * int_ws.numel(), 512 * 2**20)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAS_FI and torch.cuda.is_available(), "no flashinfer")
class TestBucketCanvasExtended(unittest.TestCase):
    """리뷰11-2: p+1 전체-page canvas와 비연속 page ID 검증
    (기존 테스트는 마지막 page 꼬리만 오염 + kv_indices가 항상
    arange — 'page ID 변경' 미검증이었음)."""

    def test_full_extra_page_poisoned(self):
        # 초기 p page + 예비 1 page 전체를 canvas로; 예비 page 전체
        # 오염 + mask=0 → 정확-길이 결과와 일치 (p+1 고정 canvas 설계)
        dev = "cuda:0"
        torch.manual_seed(3)
        ctx = PAGE * 3 - 4                 # p=3, 예비 page 미진입
        kv_len = ctx
        p_used = (kv_len + PAGE - 1) // PAGE
        p_canvas = p_used + 1              # 예비 1 page
        cache = _mk_cache(p_canvas, dev)
        poison = cache.clone()
        poison[p_used] = 1e4               # 예비 page 전체 오염
        q = torch.randn(W, H, D, dtype=torch.float16, device=dev)
        ws = torch.empty(96 * 2**20, dtype=torch.uint8, device=dev)
        qo = torch.tensor([0, W], dtype=torch.int32, device=dev)
        # 정확
        wr1 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
        wr1.plan(qo,
                 torch.tensor([0, p_used], dtype=torch.int32, device=dev),
                 torch.arange(p_used, dtype=torch.int32, device=dev),
                 torch.tensor([kv_len - (p_used - 1) * PAGE],
                              dtype=torch.int32, device=dev),
                 H, HKV, D, PAGE,
                 custom_mask=torch.ones(W * kv_len, dtype=torch.bool,
                                        device=dev),
                 q_data_type=torch.float16, kv_data_type=torch.float16)
        out_exact = wr1.run(q, cache)
        # canvas p+1 (예비 page 포함, lpl=PAGE, 초과분 전부 mask=0)
        canvas = p_canvas * PAGE
        m = torch.zeros(W, canvas, dtype=torch.bool, device=dev)
        m[:, :kv_len] = True
        wr2 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
        wr2.plan(qo,
                 torch.tensor([0, p_canvas], dtype=torch.int32,
                              device=dev),
                 torch.arange(p_canvas, dtype=torch.int32, device=dev),
                 torch.tensor([PAGE], dtype=torch.int32, device=dev),
                 H, HKV, D, PAGE, custom_mask=m.reshape(-1),
                 q_data_type=torch.float16, kv_data_type=torch.float16)
        out_canvas = wr2.run(q, poison)
        self.assertTrue(
            torch.allclose(out_exact, out_canvas, atol=2e-3, rtol=2e-3),
            f"max diff {(out_exact-out_canvas).abs().max()}")

    def test_noncontiguous_page_ids(self):
        # 실 블록테이블처럼 임의 page ID — arange 참조와 동일해야 함
        dev = "cuda:0"
        torch.manual_seed(5)
        ctx = PAGE * 2 + 7
        p = (ctx + PAGE - 1) // PAGE
        lpl = ctx - (p - 1) * PAGE
        big = _mk_cache(p + 6, dev)        # 넉넉한 물리 페이지
        q = torch.randn(W, H, D, dtype=torch.float16, device=dev)
        ws = torch.empty(96 * 2**20, dtype=torch.uint8, device=dev)
        qo = torch.tensor([0, W], dtype=torch.int32, device=dev)
        mask = (torch.rand(W, ctx, device=dev) > 0.3)
        perm = torch.tensor([5, 1, 7], dtype=torch.int32,
                            device=dev)[:p]  # 비연속 ID
        # 참조: 같은 '논리' KV를 arange 페이지에 배치
        ref_cache = _mk_cache(p, dev)
        for li, pid in enumerate(perm.tolist()):
            ref_cache[li] = big[pid]
        wr_ref = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
        wr_ref.plan(qo, torch.tensor([0, p], dtype=torch.int32,
                                     device=dev),
                    torch.arange(p, dtype=torch.int32, device=dev),
                    torch.tensor([lpl], dtype=torch.int32, device=dev),
                    H, HKV, D, PAGE, custom_mask=mask.reshape(-1),
                    q_data_type=torch.float16,
                    kv_data_type=torch.float16)
        out_ref = wr_ref.run(q, ref_cache)
        # 비연속 ID로 big 캐시 직접 사용
        wr_nc = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
        wr_nc.plan(qo, torch.tensor([0, p], dtype=torch.int32,
                                    device=dev),
                   perm,
                   torch.tensor([lpl], dtype=torch.int32, device=dev),
                   H, HKV, D, PAGE, custom_mask=mask.reshape(-1),
                   q_data_type=torch.float16,
                   kv_data_type=torch.float16)
        out_nc = wr_nc.run(q, big)
        self.assertTrue(
            torch.allclose(out_ref, out_nc, atol=2e-3, rtol=2e-3),
            f"max diff {(out_ref-out_nc).abs().max()}")

    def test_memory_fa2_vs_auto_real_dims(self):
        # 리뷰11-3: 실측 — auto는 wrapper당 ~72MB(vector sparse),
        # fa2 명시 시 ~8MB. 8버킷×4 설계는 fa2 명시 필수.
        dev = "cuda:0"
        ws = torch.empty(96 * 2**20, dtype=torch.uint8, device=dev)
        a0 = torch.cuda.memory_allocated()
        wr_auto = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws, "NHD")
        a1 = torch.cuda.memory_allocated()
        wr_fa2 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws, "NHD", backend="fa2")
        a2 = torch.cuda.memory_allocated()
        auto_mb = (a1 - a0) / 2**20
        fa2_mb = (a2 - a1) / 2**20
        print(f"[mem] wrapper auto {auto_mb:.1f}MB vs fa2 {fa2_mb:.1f}MB"
              f" → 8버킷×4: auto {32*auto_mb/1024:.2f}GB vs fa2 "
              f"{32*fa2_mb/1024:.2f}GB")
        self.assertLess(fa2_mb, 16)
