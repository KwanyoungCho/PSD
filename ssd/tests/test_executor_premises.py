"""23번 — 실행기 전제 3종의 '프로덕션 조건' 완성판 (리뷰12 §1).

1a. RNG 무오염의 참증명: 기본 RNG 상태 저장→기준샘플→복원→graph
    A/B 교차→기본 RNG 샘플이 기준과 '정확히 동일' (e1≠e2 수준 아님)
    + 동일 seed 재구축 시 replay 수열 재현.
1b. 실그래프 버킷: fa2·use_cuda_graph·PAGE=256·실 draft 치수에서
    plan '캡처 전 1회' → replay 사이 page-ID 버퍼 A→B→A 교체 →
    매번 fresh-plan eager 기준과 일치 + runtime plan 호출 0 계수.
1c. glue 폭 통합: 최대-폭 canvas 0-mask가 좁은 glue 정확 결과와
    일치하면 (page_count)만 버킷 키 — 아니면 (page_count, glue) 키.
"""
import unittest
import torch

try:
    import flashinfer
    HAS_FI = True
except Exception:
    HAS_FI = False

PAGE = 256                    # 실 블록 크기
H, HKV, D = 32, 4, 64         # TinyLlama-1.1B draft 치수
W = 10


@unittest.skipUnless(HAS_FI and torch.cuda.is_available(), "no fi/cuda")
class TestRNGStrictNonPollution(unittest.TestCase):
    def test_default_rng_exact_preservation(self):
        dev = "cuda:0"
        tree_gen = torch.Generator(device=dev)
        tree_gen.manual_seed(5)
        buf = torch.zeros(32, device=dev)

        def draw():
            buf.copy_(torch.empty_like(buf).exponential_(
                1, generator=tree_gen))

        draw(); torch.cuda.synchronize()
        ga = torch.cuda.CUDAGraph(); ga.register_generator_state(tree_gen)
        with torch.cuda.graph(ga):
            draw()
        gb = torch.cuda.CUDAGraph(); gb.register_generator_state(tree_gen)
        with torch.cuda.graph(gb):
            draw()
        torch.cuda.synchronize()
        # 기본 RNG 기준 수열 (graph 미실행 세계)
        st = torch.cuda.get_rng_state(dev)
        ref1 = torch.randn(64, device=dev).clone()
        ref2 = torch.randn(64, device=dev).clone()
        # 상태 복원 후 graph들 교차 실행 사이에 기본 RNG 샘플
        torch.cuda.set_rng_state(st, dev)
        ga.replay(); gb.replay(); torch.cuda.synchronize()
        got1 = torch.randn(64, device=dev).clone()
        ga.replay(); torch.cuda.synchronize()
        got2 = torch.randn(64, device=dev).clone()
        self.assertTrue(torch.equal(ref1, got1),
                        "graph 실행이 기본 RNG 수열을 오염")
        self.assertTrue(torch.equal(ref2, got2),
                        "graph 실행이 기본 RNG 수열을 오염(2)")
        del ga, gb

    def test_same_seed_rebuild_reproducible(self):
        dev = "cuda:0"

        def build_and_run():
            gen = torch.Generator(device=dev)
            gen.manual_seed(42)
            out = torch.zeros(16, device=dev)

            def draw():
                out.copy_(torch.empty_like(out).exponential_(
                    1, generator=gen))
            draw(); torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            g.register_generator_state(gen)
            with torch.cuda.graph(g):
                draw()
            seq = []
            for _ in range(4):
                g.replay(); torch.cuda.synchronize()
                seq.append(out.clone())
            del g
            return seq

        s1 = build_and_run()
        s2 = build_and_run()
        for a, b in zip(s1, s2):
            self.assertTrue(torch.equal(a, b),
                            "동일 seed 재구축 replay 수열 불일치")


@unittest.skipUnless(HAS_FI and torch.cuda.is_available(), "no fi/cuda")
class TestRealGraphBucket(unittest.TestCase):
    """plan 1회(캡처 전) → replay 사이 page-ID 버퍼 교체 유효성 —
    실행기의 'runtime plan 0회' 설계가 성립하는지의 결정 실험."""

    def test_page_id_swap_between_replays(self):
        dev = "cuda:0"
        torch.manual_seed(2)
        ctx = PAGE * 2 + 77
        p = (ctx + PAGE - 1) // PAGE
        lpl = ctx - (p - 1) * PAGE
        n_phys = p + 6
        cache = torch.randn(n_phys, 2, PAGE, HKV, D,
                            dtype=torch.float16, device=dev)
        q_buf = torch.randn(W, H, D, dtype=torch.float16, device=dev)
        out_buf = torch.zeros(W, H, D, dtype=torch.float16, device=dev)
        ws = torch.empty(128 * 2**20, dtype=torch.uint8, device=dev)
        canvas = p * PAGE
        mask_logical = (torch.rand(W, ctx, device=dev) > 0.3)
        mask_canvas = torch.zeros(W, canvas, dtype=torch.bool, device=dev)
        mask_canvas[:, :ctx] = mask_logical
        qo = torch.tensor([0, W], dtype=torch.int32, device=dev)
        kvp = torch.tensor([0, p], dtype=torch.int32, device=dev)
        idsA = torch.tensor([0, 1, 2], dtype=torch.int32, device=dev)[:p]
        idsB = torch.tensor([5, 3, 7], dtype=torch.int32, device=dev)[:p]
        kvi_buf = idsA.clone()               # 고정 주소 버퍼
        lpl_buf = torch.tensor([PAGE], dtype=torch.int32, device=dev)
        n_pk = (W * canvas + 7) // 8
        mask_buf = torch.zeros(n_pk, dtype=torch.uint8, device=dev)
        wr = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws, "NHD", backend="fa2", use_cuda_graph=True,
            qo_indptr_buf=qo, paged_kv_indptr_buf=kvp,
            paged_kv_indices_buf=kvi_buf,
            paged_kv_last_page_len_buf=lpl_buf,
            custom_mask_buf=mask_buf,
            mask_indptr_buf=torch.tensor([0, W * canvas],
                                         dtype=torch.int32, device=dev))
        # plan 1회 (캡처 전) — 이후 plan 호출 금지·계수
        wr.plan(qo, kvp, kvi_buf, lpl_buf, H, HKV, D, PAGE,
                custom_mask=mask_canvas.reshape(-1),
                q_data_type=torch.float16, kv_data_type=torch.float16)
        plan_calls = {"n": 0}
        orig_plan = wr.plan
        wr.plan = lambda *a, **k: (plan_calls.__setitem__("n",
                                   plan_calls["n"] + 1),
                                   orig_plan(*a, **k))[1]
        out_buf.copy_(wr.run(q_buf, cache))   # 워밍업
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out_buf.copy_(wr.run(q_buf, cache))
        torch.cuda.synchronize()

        def eager_ref(ids):
            wr2 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                ws, "NHD", backend="fa2")
            wr2.plan(qo, kvp, ids, lpl_buf, H, HKV, D, PAGE,
                     custom_mask=mask_canvas.reshape(-1),
                     q_data_type=torch.float16,
                     kv_data_type=torch.float16)
            return wr2.run(q_buf, cache)

        for ids, tag in [(idsA, "A"), (idsB, "B"), (idsA, "A2")]:
            kvi_buf.copy_(ids)               # replay 사이 ID 교체
            g.replay(); torch.cuda.synchronize()
            ref = eager_ref(ids)
            diff = (out_buf - ref).abs().max()
            self.assertTrue(
                torch.allclose(out_buf, ref, atol=2e-3, rtol=2e-3),
                f"page-ID 교체 {tag} 불일치 (max {diff}) — indices가 "
                f"plan에 bake되면 runtime-plan-0 설계 불성립")
        self.assertEqual(plan_calls["n"], 0, "runtime plan 호출 발생")
        del g


@unittest.skipUnless(HAS_FI and torch.cuda.is_available(), "no fi/cuda")
class TestGlueWidthUnification(unittest.TestCase):
    def test_max_glue_canvas_equals_narrow(self):
        # 좁은 glue(K2+1=5) 정확 mask vs 최대 glue(K1+1=10) canvas에
        # 초과 열 0-mask — 일치하면 버킷 키에서 glue 폭 제거 가능
        dev = "cuda:0"
        torch.manual_seed(9)
        ctx = PAGE + 33
        g_narrow, g_max = 5, 10
        kv_n = ctx + g_narrow
        kv_m = ctx + g_max
        p_m = (kv_m + PAGE - 1) // PAGE
        cache = torch.randn(p_m, 2, PAGE, HKV, D, dtype=torch.float16,
                            device=dev)
        q = torch.randn(W, H, D, dtype=torch.float16, device=dev)
        ws = torch.empty(128 * 2**20, dtype=torch.uint8, device=dev)
        qo = torch.tensor([0, W], dtype=torch.int32, device=dev)
        glue_bits = (torch.rand(W, g_narrow, device=dev) > 0.4)

        def run_with(kv_len, glue_cols):
            p = (kv_len + PAGE - 1) // PAGE
            lpl = kv_len - (p - 1) * PAGE
            m = torch.zeros(W, kv_len, dtype=torch.bool, device=dev)
            m[:, :ctx] = True
            m[:, ctx:ctx + glue_cols.shape[1]] = glue_cols
            wr = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                ws, "NHD", backend="fa2")
            wr.plan(qo, torch.tensor([0, p], dtype=torch.int32,
                                     device=dev),
                    torch.arange(p, dtype=torch.int32, device=dev),
                    torch.tensor([lpl], dtype=torch.int32, device=dev),
                    H, HKV, D, PAGE, custom_mask=m.reshape(-1),
                    q_data_type=torch.float16,
                    kv_data_type=torch.float16)
            return wr.run(q, cache)

        out_n = run_with(kv_n, glue_bits)
        # 최대 폭: glue 5열 실값 + 5열 0 (kv는 g_max까지 확장 — 그
        # 구간 KV는 존재하나 mask=0)
        pad = torch.zeros(W, g_max - g_narrow, dtype=torch.bool,
                          device=dev)
        out_m = run_with(kv_m, torch.cat([glue_bits, pad], 1))
        diff = (out_n - out_m).abs().max()
        self.assertTrue(
            torch.allclose(out_n, out_m, atol=2e-3, rtol=2e-3),
            f"glue 통합 불일치 (max {diff}) — (page,glue) 2-키 필요")


if __name__ == "__main__":
    unittest.main()
