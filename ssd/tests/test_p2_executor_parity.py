"""23번 단계 2/3 — 실행기 모듈 결정적 parity (미니 모델).

게이트: 동일 입력 버퍼 + 동일 고정 noise에서
  eager run_once == captured replay
정수/구조 exact, 실수 allclose. (실모델 parity는 엔진 배선 스모크
단계 — 이 테스트는 실행기 '모듈'의 캡처-등가성을 고정한다.)
"""
import unittest
from unittest import mock
import torch
import torch.nn as nn

try:
    import flashinfer
    HAS_FI = True
except Exception:
    HAS_FI = False

from ssd.utils.context import get_context
from ssd.engine.helpers import p2_tree as PT
from ssd.engine.helpers.p1_tree import P1TreeExecutor
from ssd.engine.helpers.p2_tree_executor import (
    P2TreeExecutor,
    _gather_backbone_logits,
)


class _MiniCfg:
    duet_proxy_total_budget = 10
    duet_tree_root_count = 6
    duet_phase1_k = 9
    duet_phase2_k = 4
    duet_p1_roots_per_position = 2
    duet_p1_tree_forward_scale = 1.0
    duet_p1_tree_max_nodes = 13
    duet_p2_tree_max_nodes = 8
    duet_tree_c_tensor = 3
    duet_tree_nv = 8
    duet_tree_beta = 0.5
    duet_tree_policy = "level"
    duet_tree_proxy_threshold = 0.01
    duet_tree_conf_threshold = 0.03
    sampler_x = None
    async_fan_out = 3

    @property
    def duet_p2_seed_count(self):
        return self.duet_tree_root_count


class _MiniDraft(nn.Module):
    """실모델 대역: KV 기록→attention→hidden (attention.py 순서 미러).
    get_context에서 slot/wrapper를 읽는다 (실 계약과 동일)."""

    def __init__(self, V, H, HKV, D, cache, dev):
        super().__init__()
        g = torch.Generator().manual_seed(1)
        self.emb = ((torch.randn(V, HKV * D, generator=g) * .05)
                    .to(dtype=torch.float16, device=dev))
        self.qp = ((torch.randn(HKV * D, H * D, generator=g) * .05)
                   .to(dtype=torch.float16, device=dev))
        self.head = ((torch.randn(H * D, V, generator=g) * .05)
                     .to(dtype=torch.float16, device=dev))
        self.cache = cache            # [pages, 2, PAGE, HKV, D]
        self.H, self.HKV, self.D = H, HKV, D
        self.PAGE = cache.shape[2]

    def forward(self, ids, rope):
        ctx = get_context()
        x = self.emb.index_select(0, ids.clamp(min=0))      # [W, HKV*D]
        kv = x.view(-1, self.HKV, self.D)
        slots = ctx.slot_mapping.long()
        pg, off = slots // self.PAGE, slots % self.PAGE
        self.cache[pg, 0, off] = kv                          # KV 먼저
        self.cache[pg, 1, off] = kv
        q = (x @ self.qp).view(-1, self.H, self.D)
        wr = ctx.active_wrappers[1]
        o = wr.run(q, self.cache)                            # attention
        return o.reshape(-1, self.H * self.D)

    def logits_fn(self, hidden, last_only):
        return (hidden @ self.head).float()


@unittest.skipUnless(HAS_FI and torch.cuda.is_available(), "no fi/cuda")
class TestExecutorModuleParity(unittest.TestCase):
    def tearDown(self):
        import gc
        gc.collect()
        torch.cuda.synchronize()

    def test_backbone_logit_gather_bounds_and_tail(self):
        """Invalid parent cells must zero-fill without any out-of-bounds read.

        Use a vocabulary width that is not divisible by the Triton block so
        both row and final-column masks are exercised.
        """
        dev = "cuda:0"
        vocab = 1031
        source = torch.arange(7 * vocab, dtype=torch.float32,
                              device=dev).view(7, vocab)
        parent_cells = torch.tensor(
            [[-1, 0, 6], [7, 100, 3]], dtype=torch.int64, device=dev)
        out = torch.empty(2, 3, vocab, dtype=torch.float16, device=dev)
        _gather_backbone_logits(source, parent_cells, out)
        torch.cuda.synchronize()

        expected = torch.zeros_like(out)
        expected[0, 1] = source[0].to(out.dtype)
        expected[0, 2] = source[6].to(out.dtype)
        expected[1, 2] = source[3].to(out.dtype)
        self.assertTrue(torch.equal(expected, out))

    def _mk(self, dev, PAGE=64, policy="level"):
        V, H, HKV, D = 128, 4, 2, 64
        cfg = _MiniCfg()
        cfg.duet_tree_policy = policy
        if policy in ("coverage", "backbone", "dynamic", "eagle", "hybrid",
                      "adaptive"):
            cfg.duet_tree_root_count = cfg.duet_proxy_total_budget
        max_blocks = 8
        cache = torch.zeros(max_blocks, 2, PAGE, HKV, D,
                            dtype=torch.float16, device=dev)
        model = _MiniDraft(V, H, HKV, D, cache, dev)
        ex = P2TreeExecutor(model, model.logits_fn, cfg, dev,
                            PAGE, max_blocks, V, H, HKV, D)
        return ex, cfg, PAGE, V

    def _fill_inputs(self, ex, PAGE, ctx0):
        R, W, F = ex.R, ex.W, ex.F
        g = torch.Generator().manual_seed(7)
        ex.in_root_tok.copy_(torch.randint(0, 100, (R,), generator=g)
                             .to(ex.dev))
        if R == 6:
            piv = torch.tensor([.4, .2, .1, .06, .03, .01])
        else:
            piv = torch.linspace(.4, .01, R)
        ex.in_root_piv.copy_(piv.to(ex.dev))
        ex.in_rope_base.fill_(ctx0 - 1)
        gw = ex.F + 1                       # 이 테스트의 실 glue 폭
        ex.in_glue.zero_()
        ex.in_glue[:, :gw] = 1
        ex.in_glue_w.fill_(gw)
        ex.in_temps.fill_(0.8)
        ex.in_prefix_len.fill_(ctx0 - gw - W)
        p0 = (ctx0 + PAGE - 1) // PAGE
        for f in range(F):
            base = ctx0 + f * W
            pos = base + torch.arange(W)
            ex.in_slot[f].copy_((pos % ((p0 + 1) * PAGE))
                                .to(torch.int32).to(ex.dev))
            ex.in_ctx_len[f].fill_(base + W)
            wr = ex.wrappers[p0][f]
            wr._paged_kv_indices_buf[:p0 + 1].copy_(
                torch.arange(p0 + 1, dtype=torch.int32, device=ex.dev))
        return p0

    def test_eager_equals_replay_with_fixed_noise(self):
        dev = "cuda:0"
        ex, cfg, PAGE, V = self._mk(dev)
        ctx0 = PAGE + 21
        p0 = (ctx0 + PAGE - 1) // PAGE
        ex.prepare_bucket(p0)
        self._fill_inputs(ex, PAGE, ctx0)
        gN = torch.Generator().manual_seed(33)
        noise = [
            (torch.empty(ex.W, V).exponential_(1, generator=gN))
            .to(dev) for _ in range(ex.F)]
        ex.parity_noise = noise
        # ── eager 기준
        ex._local_idx = torch.full((ex.arena.capacity,), -1,
                                   dtype=torch.int64, device=dev)
        ex.model.cache.zero_()
        ex._local_idx.fill_(-1)
        ex.run_once(p0)
        ref = {k: getattr(ex, k).clone()
               for k in ("view_tok", "view_par", "view_sib", "view_rawq",
                         "view_pcell", "out_valid", "out_pq_ref",
                         "out_pq_cells", "out_u_valid",
                         "out_backbone_tok", "out_backbone_logits")}
        ref_logits = ex.cell_logits.clone()
        ref_cache = ex.model.cache.clone()
        # ── 캡처 + replay (동일 입력·동일 noise)
        ex.model.cache.zero_()
        g = ex.capture(p0)
        ex.model.cache.zero_()
        for t in ref.values():
            pass
        ex.replay(p0)
        torch.cuda.synchronize()
        for k, v in ref.items():
            got = getattr(ex, k)
            if got.dtype.is_floating_point:
                self.assertTrue(torch.allclose(v, got, atol=1e-3,
                                               rtol=1e-3), k)
            else:
                self.assertTrue(torch.equal(v, got),
                                f"{k} 불일치")
        self.assertTrue(torch.allclose(ref_logits, ex.cell_logits,
                                       atol=2e-2, rtol=2e-2))
        self.assertTrue(torch.allclose(ref_cache.float(),
                                       ex.model.cache.float(),
                                       atol=2e-2, rtol=2e-2))
        # 구조 불변량
        vt = ex.out_valid.cpu()
        self.assertTrue(int(vt.sum()) > ex.R)
        self.assertTrue(bool((vt <= ex.NV).all()))
        self.assertEqual(tuple(vt.shape), (ex.W,))
        self.assertTrue(torch.equal(vt[ex.R:],
                                    torch.zeros(ex.W - ex.R,
                                                dtype=vt.dtype)))
        del g

    def test_executor_passes_sampler_rescaling_to_wor(self):
        """Captured sampling and target verification must use the same q."""
        dev = "cuda:0"
        ex, cfg, PAGE, _ = self._mk(dev, policy="backbone")
        cfg.sampler_x = 0.7
        cfg.async_fan_out = 3
        ctx0 = PAGE + 21
        p0 = (ctx0 + PAGE - 1) // PAGE
        ex.prepare_bucket(p0)
        self._fill_inputs(ex, PAGE, ctx0)
        ex._local_idx = torch.full(
            (ex.arena.capacity,), -1, dtype=torch.int64, device=dev)
        with mock.patch.object(
                PT, "tree_sample_wor", wraps=PT.tree_sample_wor) as sample:
            ex.run_once(p0)
        self.assertEqual(sample.call_count, ex.F)
        for call in sample.call_args_list:
            self.assertEqual(call.kwargs["sampler_x"], 0.7)
            self.assertEqual(call.kwargs["F"], 3)

    def test_p1_dynamic_executor_eager_equals_replay(self):
        """P1 evaluates every root, then reallocates lanes globally."""
        dev = "cuda:0"
        V, H, HKV, D, PAGE = 128, 4, 2, 64, 64
        cfg = _MiniCfg()
        cfg.duet_tree_policy = "dynamic"
        cfg.duet_tree_conf_threshold = 0.005
        max_blocks = 8
        cache = torch.zeros(max_blocks, 2, PAGE, HKV, D,
                            dtype=torch.float16, device=dev)
        model = _MiniDraft(V, H, HKV, D, cache, dev)
        ex = P1TreeExecutor(
            model, model.logits_fn, cfg, dev, PAGE, max_blocks,
            V, H, HKV, D, context_bucket=10,
            materialize_backbone_logits=False)
        self.assertEqual((ex.F, ex.W, ex.R), (9, 20, 20))
        self.assertEqual(ex.arena.anc_words, 3)

        ctx0 = 2 * PAGE + 21
        p0 = (ctx0 + PAGE - 1) // PAGE
        ex.prepare_bucket(p0)
        self._fill_inputs(ex, PAGE, ctx0)
        # P1 uses its glue-derived start score exactly like P2 uses its proxy
        # prior.  A root below the shared start floor must still receive the
        # round-zero C children, but none of those leaves may be expanded.
        ex.in_root_piv[-1].fill_(0.001)
        gN = torch.Generator().manual_seed(330)
        ex.parity_noise = [
            torch.empty(ex.W, V).exponential_(1, generator=gN).to(dev)
            for _ in range(ex.F)]
        ex._local_idx = torch.full((ex.arena.capacity,), -1,
                                   dtype=torch.int64, device=dev)
        ex.model.cache.zero_()
        # Production dynamic-tree serving consumes cell_logits through the
        # exact parent-q sidecar.  Its old chain-projection tensor must not be
        # gathered just to be discarded on a tree hit.
        ex.out_backbone_logits.fill_(17)
        ex.run_once(p0)
        self.assertTrue(bool((ex.out_backbone_logits == 17).all()))
        ref = {k: getattr(ex, k).clone() for k in (
            "view_tok", "view_par", "view_sib", "out_valid",
            "out_pq_ref", "out_pq_cells")}
        ex.model.cache.zero_()
        ex.capture(p0)
        ex.model.cache.zero_()
        ex.replay(p0)
        torch.cuda.synchronize()
        for name, expected in ref.items():
            self.assertTrue(torch.equal(expected, getattr(ex, name)), name)
        vt_p1 = ex.out_valid.cpu()
        # Round zero covers every root, but later rounds are not obliged to
        # keep one full-depth path per root.  Unequal root priors therefore
        # produce unequal response depths/node counts.
        self.assertTrue(bool((vt_p1 >= 1).all()))
        self.assertGreater(len(set(vt_p1.tolist())), 1)
        self.assertEqual(int(vt_p1[-1]), ex.C)
        self.assertTrue(bool((vt_p1[:-1] > ex.C).any()))

    def test_round_wrappers_share_only_same_page_plan_workspace(self):
        """Identical serial round plans should not allocate F private 8MiB buffers."""
        dev = "cuda:0"
        ex, _, PAGE, _ = self._mk(dev, policy="dynamic")
        ex.prepare_bucket(2)
        same_page = ex.wrappers[2]
        self.assertTrue(all(
            w._int_workspace_buffer.data_ptr()
            == same_page[0]._int_workspace_buffer.data_ptr()
            for w in same_page))
        self.assertTrue(all(
            w._pin_memory_int_workspace_buffer.data_ptr()
            == same_page[0]._pin_memory_int_workspace_buffer.data_ptr()
            for w in same_page))

        ex.prepare_bucket(3)
        self.assertNotEqual(
            ex.wrappers[2][0]._int_workspace_buffer.data_ptr(),
            ex.wrappers[3][0]._int_workspace_buffer.data_ptr())

    def test_z_backbone_executor_keeps_chain_plus_siblings_for_all_roots(self):
        dev = "cuda:0"
        ex, cfg, PAGE, V = self._mk(dev, policy="backbone")
        ctx0 = PAGE + 21
        p0 = (ctx0 + PAGE - 1) // PAGE
        ex.prepare_bucket(p0)
        self._fill_inputs(ex, PAGE, ctx0)
        gN = torch.Generator().manual_seed(44)
        ex.parity_noise = [
            torch.empty(ex.W, V).exponential_(1, generator=gN).to(dev)
            for _ in range(ex.F)]
        ex._local_idx = torch.full((ex.arena.capacity,), -1,
                                   dtype=torch.int64, device=dev)
        ex.run_once(p0)
        torch.cuda.synchronize()

        self.assertEqual(ex.R, ex.W)
        self.assertEqual(ex.out_valid.cpu().tolist(), [ex.NV] * ex.W)
        expected_par = [-1, -1, -1, 0, 0, 0, 3, 6]
        expected_sib = [0, 1, 2, 0, 1, 2, 0, 0]
        for r in range(ex.W):
            self.assertEqual(ex.view_par[r].cpu().tolist(), expected_par)
            self.assertEqual(ex.view_sib[r].cpu().tolist(), expected_sib)
        # FlashInfer wrappers own persistent CUDA workspaces.  Release this
        # deliberately different R=10 fixture before inherited R=6 capture
        # tests construct their wrappers in the same unittest process.
        del ex
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def test_z_dynamic_executor_capture_and_dynamic_root_depths(self):
        dev = "cuda:0"
        ex, cfg, PAGE, V = self._mk(dev, policy="dynamic")
        # This synthetic V=128 model is close to uniform (q≈1/128), whereas
        # the production 32k model's calibrated floor is 0.03.  Use the same
        # threshold code path at the fixture's probability scale; applying a
        # model-specific production value here would intentionally make every
        # synthetic child a leaf and invalidate this topology-diversity test.
        cfg.duet_tree_proxy_threshold = 0.01
        cfg.duet_tree_conf_threshold = 0.005
        ctx0 = PAGE + 21
        p0 = (ctx0 + PAGE - 1) // PAGE
        ex.prepare_bucket(p0)
        self._fill_inputs(ex, PAGE, ctx0)
        # Make root probability differences large enough that the globally
        # selected later parents are not one mandatory tip per root.
        ex.in_root_piv.copy_(torch.tensor(
            [.70, .12, .06, .04, .025, .018, .014, .01, .008, .005],
            device=dev))
        gN = torch.Generator().manual_seed(45)
        ex.parity_noise = [
            torch.empty(ex.W, V).exponential_(1, generator=gN).to(dev)
            for _ in range(ex.F)]
        ex._local_idx = torch.full((ex.arena.capacity,), -1,
                                   dtype=torch.int64, device=dev)
        ex.model.cache.zero_()
        ex._local_idx.fill_(-1)
        ex.run_once(p0)
        ref = {k: getattr(ex, k).clone() for k in (
            "view_tok", "view_par", "view_sib", "view_rawq",
            "view_pcell", "out_valid", "out_pq_ref", "out_pq_cells",
            "out_u_valid")}
        # Every root is evaluated in round zero and therefore has a usable
        # view, but later depth is allocated by the global score.
        valid = ex.out_valid.cpu()
        self.assertTrue(bool((valid >= 1).all()))
        self.assertGreater(len(set(valid.tolist())), 1)

        ex.model.cache.zero_()
        ex.capture(p0)
        ex.model.cache.zero_()
        ex.replay(p0)
        torch.cuda.synchronize()
        for k, v in ref.items():
            got = getattr(ex, k)
            if got.dtype.is_floating_point:
                self.assertTrue(torch.allclose(v, got, atol=1e-3,
                                               rtol=1e-3), k)
            else:
                self.assertTrue(torch.equal(v, got), k)
        del ex
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def test_z_hybrid_executor_depth_floor_and_capture_parity(self):
        dev = "cuda:0"
        ex, cfg, PAGE, V = self._mk(dev, policy="hybrid")
        cfg.duet_tree_conf_threshold = 0.005
        ctx0 = PAGE + 21
        p0 = (ctx0 + PAGE - 1) // PAGE
        ex.prepare_bucket(p0)
        self._fill_inputs(ex, PAGE, ctx0)
        ex.in_root_piv.copy_(torch.tensor(
            [.70, .12, .06, .04, .025, .018, .014, .01, .008, .005],
            device=dev))
        gN = torch.Generator().manual_seed(48)
        ex.parity_noise = [
            torch.empty(ex.W, V).exponential_(1, generator=gN).to(dev)
            for _ in range(ex.F)]
        ex._local_idx = torch.full((ex.arena.capacity,), -1,
                                   dtype=torch.int64, device=dev)
        ex.model.cache.zero_()
        ex.run_once(p0)
        ref = {k: getattr(ex, k).clone() for k in (
            "view_tok", "view_par", "view_sib", "view_rawq",
            "view_pcell", "out_valid", "out_pq_ref", "out_pq_cells",
            "out_u_valid")}
        valid = ex.out_valid.cpu()
        par = ex.view_par.cpu()
        for r in range(ex.R):
            n = int(valid[r])
            depth = [0] * n
            for j in range(n):
                p = int(par[r, j])
                depth[j] = 1 if p < 0 else depth[p] + 1
            self.assertGreaterEqual(max(depth), 2)

        ex.model.cache.zero_()
        ex.capture(p0)
        ex.model.cache.zero_()
        ex.replay(p0)
        torch.cuda.synchronize()
        for k, v in ref.items():
            got = getattr(ex, k)
            if got.dtype.is_floating_point:
                self.assertTrue(torch.allclose(v, got, atol=1e-3,
                                               rtol=1e-3), k)
            else:
                self.assertTrue(torch.equal(v, got), k)
        del ex
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def test_z_adaptive_executor_keeps_depth_and_varies_only_width(self):
        dev = "cuda:0"
        ex, cfg, PAGE, V = self._mk(dev, policy="adaptive")
        ctx0 = PAGE + 21
        p0 = (ctx0 + PAGE - 1) // PAGE
        ex.prepare_bucket(p0)
        self._fill_inputs(ex, PAGE, ctx0)
        ex.in_root_piv.copy_(torch.tensor(
            [.70, .12, .06, .04, .025, .018, .014, .01, .008, .005],
            device=dev))
        gN = torch.Generator().manual_seed(47)
        ex.parity_noise = [
            torch.empty(ex.W, V).exponential_(1, generator=gN).to(dev)
            for _ in range(ex.F)]
        ex._local_idx.fill_(-1)
        ex.run_once(p0)
        torch.cuda.synchronize()
        valid = ex.out_valid.cpu()
        self.assertTrue(bool((valid >= ex.F).all()))
        self.assertGreater(len(set(valid.tolist())), 1)
        par = ex.view_par.cpu()
        sib = ex.view_sib.cpu()
        for r in range(ex.R):
            cur = -1
            depth = 0
            for _ in range(ex.F):
                kids = [j for j in range(int(valid[r]))
                        if int(par[r, j]) == cur and int(sib[r, j]) == 0]
                self.assertTrue(kids, (r, depth, valid[r].item()))
                cur = kids[0]
                depth += 1
            self.assertEqual(depth, ex.F)

        ref = {k: getattr(ex, k).clone() for k in (
            "view_tok", "view_par", "view_sib", "view_rawq",
            "view_pcell", "out_valid", "out_pq_ref", "out_pq_cells",
            "out_u_valid")}
        ex.capture(p0)
        ex.replay(p0)
        torch.cuda.synchronize()
        for k, v in ref.items():
            got = getattr(ex, k)
            if got.dtype.is_floating_point:
                self.assertTrue(torch.allclose(v, got, atol=1e-3,
                                               rtol=1e-3), k)
            else:
                self.assertTrue(torch.equal(v, got), k)
        del ex
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def test_z_eagle_multiple_bucket_graphs_keep_live_local_index(self):
        """Later bucket capture must not orphan earlier graph state.

        Production warmup captures several page-count buckets up front.  A
        former capture()-local allocation left every graph writing a
        different unreachable ``_local_idx`` tensor, while a later attempted
        global shared tensor broke real-model graph isolation.  Buffers are
        now retained per bucket and replay selects the matching one.
        """
        dev = "cuda:0"
        ex, cfg, PAGE, V = self._mk(dev, policy="eagle")
        gN = torch.Generator().manual_seed(46)
        ex.parity_noise = [
            torch.empty(ex.W, V).exponential_(1, generator=gN).to(dev)
            for _ in range(ex.F)]
        buckets = []
        local_ptrs = []
        workspace_ptrs = []
        for ctx0 in (PAGE + 21, 2 * PAGE + 21):
            p0 = (ctx0 + PAGE - 1) // PAGE
            ex.prepare_bucket(p0)
            workspace_ptrs.append(
                ex._float_ws_by_bucket[p0].data_ptr())
            self._fill_inputs(ex, PAGE, ctx0)
            ex.capture(p0)
            local_ptrs.append(ex._local_idx.data_ptr())
            self.assertEqual(
                ex._local_idx_by_bucket[p0].data_ptr(),
                ex._local_idx.data_ptr())
            buckets.append((p0, ctx0))
        self.assertEqual(len(set(local_ptrs)), len(local_ptrs))
        self.assertEqual(len(set(workspace_ptrs)), len(workspace_ptrs))

        # Replay the first graph after the second was captured.  Its selected
        # non-root parents must still map to the local ids emitted in views.
        p0, ctx0 = buckets[0]
        self._fill_inputs(ex, PAGE, ctx0)
        ex.replay(p0)
        torch.cuda.synchronize()
        self.assertEqual(ex._local_idx.data_ptr(),
                         ex._local_idx_by_bucket[p0].data_ptr())
        for f in range(1, ex.F):
            sel = ex.dbg_sel[f]
            valid = ex.dbg_selv[f]
            if bool(valid.any()):
                self.assertTrue(bool(
                    (ex._local_idx.gather(0, sel[valid]) >= 0).all()))
        del ex
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def test_z_prime_capture_inputs_are_finite_and_active(self):
        dev = "cuda:0"
        ex, _cfg, PAGE, _V = self._mk(dev, policy="eagle")
        p0 = 3
        ex.prepare_bucket(p0)
        ex.prime_capture_inputs(p0)
        self.assertTrue(all(bool((x > 0).all()) for x in ex.in_ctx_len))
        self.assertTrue(all(bool((x >= 0).all()) for x in ex.in_slot))
        self.assertTrue(bool(torch.isfinite(ex.in_root_piv).all()))
        self.assertTrue(bool((ex.in_root_piv > 0).all()))
        for wr in ex.wrappers[p0]:
            self.assertTrue(bool((wr._paged_kv_indices_buf >= 0).all()))


@unittest.skipUnless(HAS_FI and torch.cuda.is_available(), "no fi/cuda")
class TestExecutorVsArenaSemantics(unittest.TestCase):
    """판별 parity: 같은 미니모델·같은 noise에서 arena(run_rollout_
    arena, 자체 mask 경로) vs executor — round별 logits 일치 여부로
    mask 버그를, topology/views 일치 여부로 기록 버그를 분리 판별."""

    def tearDown(self):
        import gc
        gc.collect(); torch.cuda.synchronize()

    def test_same_noise_same_tree(self):
        import numpy as _np
        import ssd.engine.helpers.p2_tree as PT2
        dev = "cuda:0"
        PAGE = 64
        V, H, HKV, D = 128, 4, 2, 64
        cfg = _MiniCfg()
        max_blocks = 8
        cache = torch.zeros(max_blocks, 2, PAGE, HKV, D,
                            dtype=torch.float16, device=dev)
        model = _MiniDraft(V, H, HKV, D, cache, dev)
        ex = P2TreeExecutor(model, model.logits_fn, cfg, dev,
                            PAGE, max_blocks, V, H, HKV, D)
        ctx0 = PAGE + 21
        p0 = (ctx0 + PAGE - 1) // PAGE
        ex.prepare_bucket(p0)
        # 입력
        R, W, F, C, NV = ex.R, ex.W, ex.F, ex.C, ex.NV
        g = torch.Generator().manual_seed(7)
        root_toks = torch.randint(0, 100, (R,), generator=g)
        piv = torch.tensor([.4, .2, .1, .06, .03, .01])
        gw = F + 1
        glue_np = _np.ones((R, gw), dtype=_np.uint8)
        rope_base = [ctx0 - 1] * R
        gN = torch.Generator().manual_seed(33)
        noise = [(torch.empty(W, V).exponential_(1, generator=gN))
                 .to(dev) for _ in range(F)]
        # ── executor 실행 (eager run_once — 동일 noise)
        ex.in_root_tok.copy_(root_toks.to(dev))
        ex.in_root_piv.copy_(piv.to(dev))
        ex.in_rope_base.fill_(ctx0 - 1)
        ex.in_glue.zero_(); ex.in_glue[:, :gw] = 1
        ex.in_glue_w.fill_(gw)
        ex.in_temps.fill_(0.8)
        ex.in_prefix_len.fill_(ctx0 - gw - W)
        for f in range(F):
            base = ctx0 + f * W
            pos = base + torch.arange(W)
            ex.in_slot[f].copy_((pos % ((p0 + 1) * PAGE))
                                .to(torch.int32).to(dev))
            ex.in_ctx_len[f].fill_(base + W)
            ex.wrappers[p0][f]._paged_kv_indices_buf[:p0 + 1].copy_(
                torch.arange(p0 + 1, dtype=torch.int32, device=dev))
        ex.parity_noise = noise
        ex._local_idx = torch.full((ex.arena.capacity,), -1,
                                   dtype=torch.int64, device=dev)
        cache.zero_()
        ex._local_idx.fill_(-1)
        ex.run_once(p0)
        exec_logits = ex.cell_logits.clone()
        # ── arena 기준 (같은 모델을 forward_fn으로 — JIT-plan mask)
        cache2 = torch.zeros_like(cache)
        model2 = _MiniDraft(V, H, HKV, D, cache2, dev)
        ws2 = torch.empty(96 * 2**20, dtype=torch.uint8, device=dev)
        noise_iter = {"f": 0}
        orig_wor = PT2.tree_sample_wor

        def wor_with_noise(logits, temps, c, **kw):
            kw.pop("noise", None)
            n = noise[noise_iter["f"]]
            noise_iter["f"] += 1
            return orig_wor(logits, temps, c,
                            assume_pos_temps=True, noise=n)

        def fwd(f, ids, rope, packed, indptr):
            # arena가 만든 packed mask로 JIT-plan attention
            kv_len = ctx0 + f * W + W
            p = (kv_len + PAGE - 1) // PAGE
            lpl = kv_len - (p - 1) * PAGE
            wr = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                ws2, "NHD")
            # packed → bool mask 복원
            n_bits = W * (ctx0 + f * W)
            pk = packed.cpu().numpy() if torch.is_tensor(packed) \
                else packed
            bits = _np.unpackbits(pk, bitorder="little")[:n_bits]
            mlog = torch.from_numpy(
                bits.astype(_np.bool_)).view(W, ctx0 + f * W)
            mfull = torch.zeros(W, kv_len, dtype=torch.bool)
            # arena mask 열: [prefix|glue|spec f·W] = ctx0+f·W 열 —
            # 실 kv에는 이번 라운드 신규 W열이 추가되므로 self 열만
            # 뒤에 이어붙음 (arena 경로의 self_cols 규약)
            mfull[:, :ctx0 + f * W] = mlog
            lane = torch.arange(W)
            mfull[lane, ctx0 + f * W + lane] = True
            wr.plan(torch.tensor([0, W], dtype=torch.int32, device=dev),
                    torch.tensor([0, p], dtype=torch.int32, device=dev),
                    torch.arange(p, dtype=torch.int32, device=dev),
                    torch.tensor([lpl], dtype=torch.int32, device=dev),
                    H, HKV, D, PAGE,
                    custom_mask=mfull.reshape(-1).to(dev),
                    q_data_type=torch.float16,
                    kv_data_type=torch.float16)
            from ssd.utils.context import set_context, reset_context
            base = ctx0 + f * W
            pos = base + torch.arange(W)
            set_context(is_prefill=False,
                        slot_mapping=(pos % ((p0 + 1) * PAGE))
                        .to(torch.int32).to(dev),
                        context_lens=torch.tensor([base + W],
                                                  dtype=torch.int32,
                                                  device=dev),
                        block_tables=torch.arange(
                            max_blocks, dtype=torch.int32,
                            device=dev).unsqueeze(0),
                        active_mq_len=W, active_wrappers={1: wr})
            h = model2(ids.to(dev), rope.to(dev))
            reset_context()
            return model2.logits_fn(h, False)

        try:
            PT2.tree_sample_wor = wor_with_noise
            ar, trace, cell_logits_ref = PT2.run_rollout_arena(
                root_toks.tolist(), piv.clone(), policy="level", W=W,
                F_total=F, c_tensor=C, nv=NV, beta=0.5, depth_cap=F,
                temps=torch.full((W,), 0.8), forward_fn=fwd,
                glue_rows_by_root=glue_np, rope_base_by_root=rope_base,
                K_glue=gw - 1, context_len=ctx0, device=dev)
        finally:
            PT2.tree_sample_wor = orig_wor
        pool = ar.to_pool(R)
        views_ref = PT2.build_root_views(pool, R, NV,
                                         cell_logits=cell_logits_ref)
        # ── ① logits-경로 검증: 두 attention 경로(참조 auto JIT-plan
        # vs 실행기 fa2 preplanned)는 커널이 달라 bit-동일이 아님 —
        # fp16 허용오차 검증 (mask/slot/rope 정합의 증거).
        for f in range(F):
            d = (exec_logits[f * W:(f + 1) * W]
                 - cell_logits_ref[f * W:(f + 1) * W].float()) \
                .abs().max()
            self.assertLess(float(d), 2e-2,
                            f"round {f} logits 불일치 (max {d}) — "
                            f"mask/slot/rope 경로 상이")
        # ── ② 기록기 게이트: 실행기 [R,Nv] 직접 기록 == 자기 arena의
        # build_root_views (같은 트리의 두 서술 — kernel 비결정성과
        # 무관한 exact 비교). 주의: 참조-트리와의 노드-동일성은
        # 근접-동률 priority가 커널 오차로 뒤집혀 원리적 비보장
        # (18/40 민감성 — 실측 확인) → arena-vs-exec 의미 판정은
        # 분포 지표(인터리브 AL)로 (리뷰12 §7 fallback 규정).
        pe = ex.arena.to_pool(R)
        ve = PT2.build_root_views(pe, R, NV,
                                  cell_logits=ex.cell_logits)
        self.assertEqual(tuple(ex.out_valid.shape), (W,))
        self.assertTrue(torch.equal(ve["valid"],
                                    ex.out_valid[:R].cpu()))
        self.assertTrue(torch.equal(
            ex.out_valid[R:].cpu(),
            torch.zeros(W - R, dtype=ex.out_valid.dtype)))
        for r in range(R):
            n = int(ve["valid"][r])
            for key, exbuf in (("tok", ex.view_tok),
                               ("parent_local", ex.view_par),
                               ("sib_order", ex.view_sib)):
                self.assertTrue(
                    torch.equal(ve[key][r, :n], exbuf[r, :n].cpu()),
                    f"기록기: root {r} {key} 불일치")
            self.assertTrue(torch.allclose(
                ve["raw_q"][r, :n], ex.view_rawq[r, :n].cpu(),
                atol=1e-5), f"기록기: root {r} raw_q")

        # ── ③ 최종 소비자 계약 exact: parent-q 번호, wire 블록,
        # cache populate용 backbone까지 CPU arena 기준과 동일해야 한다.
        for key, got in (("parent_q_ref", ex.out_pq_ref),
                         ("parent_q_cells", ex.out_pq_cells),
                         ("u_valid", ex.out_u_valid)):
            self.assertTrue(torch.equal(ve[key], got[:R].cpu()), key)

        exec_view = {
            "valid": ex.out_valid, "u_valid": ex.out_u_valid,
            "tok": ex.view_tok, "parent_local": ex.view_par,
            "sib_order": ex.view_sib, "parent_q_ref": ex.out_pq_ref,
        }
        bt_ref = torch.zeros(R, F, dtype=torch.int64)
        bl_ref = torch.zeros(R, F, V, dtype=ex.dtype)
        for r in range(R):
            wire_ref = PT2.pack_tree_ints(ve, r, NV)
            wire_gpu = PT2.pack_tree_ints(exec_view, r, NV).cpu()
            self.assertTrue(torch.equal(wire_ref, wire_gpu),
                            f"root {r} wire")
            n = int(ve["valid"][r])
            first_child = {}
            for j in range(n):
                par = int(ve["parent_local"][r, j])
                if int(ve["sib_order"][r, j]) == 0 \
                        and par not in first_child:
                    first_child[par] = j
            cur = -1
            for depth in range(F):
                if cur not in first_child:
                    break
                child = first_child[cur]
                bt_ref[r, depth] = ve["tok"][r, child]
                qref = int(ve["parent_q_ref"][r, child])
                pcell = int(ve["parent_q_cells"][r, qref])
                bl_ref[r, depth] = ex.cell_logits[pcell].cpu().to(ex.dtype)
                cur = child
        self.assertTrue(torch.equal(
            bt_ref, ex.out_backbone_tok[:R].cpu()), "backbone token")
        self.assertTrue(torch.equal(
            bl_ref, ex.out_backbone_logits[:R].cpu()), "backbone logits")
        # Cache/layout consumers use W physical rows.  Non-root tail rows are
        # deterministic invalid padding, not a shortened R-row payload.
        self.assertEqual(tuple(ex.view_tok.shape), (W, NV))
        self.assertTrue(bool((ex.out_valid[R:] == 0).all()))
        self.assertTrue(bool((ex.out_backbone_tok[R:] == 0).all()))
        self.assertTrue(bool((ex.out_backbone_logits[R:] == 0).all()))


@unittest.skipUnless(HAS_FI and torch.cuda.is_available(), "no fi/cuda")
class TestExecutorMandatoryParity(TestExecutorModuleParity):
    """리뷰12 deterministic-parity 필수 케이스 (모듈 수준 — 같은
    커널이므로 결정성 보장 영역):
      - page 경계 (round kv_len의 lpl ∈ {1, PAGE-1, PAGE})
      - page/slot 교체 후 replay == 교체 전 (page-ID buffer 내용
        교체 가능성 — 논리 계산 동일이므로 결과 exact)
      - graph↔eager 20회 교차 (fallback 교차 시 graph 상태 무오염)
      - sentinel KV (replay의 KV 기록이 in_slot 슬롯에만 국한)
    round간 KV 영향은 판별 ①이 커버 (round f logits가 f'<f 기록
    KV에 의존하는 arena 참조와 일치)."""

    def _noise(self, ex, V, seed=33):
        gN = torch.Generator().manual_seed(seed)
        return [(torch.empty(ex.W, V).exponential_(1, generator=gN))
                .to(ex.dev) for _ in range(ex.F)]

    def _prep_local(self, ex):
        ex._local_idx = torch.full((ex.arena.capacity,), -1,
                                   dtype=torch.int64, device=ex.dev)

    def _snap(self, ex):
        out = {k: getattr(ex, k).clone()
               for k in ("view_tok", "view_par", "view_sib",
                         "view_rawq", "view_pcell", "out_valid",
                         "out_pq_ref", "out_pq_cells", "out_u_valid",
                         "out_backbone_tok", "out_backbone_logits")}
        out["logits"] = ex.cell_logits.clone()
        return out

    def _assert_snap(self, a, b, msg):
        for k, v in a.items():
            got = b[k]
            if v.dtype.is_floating_point:
                self.assertTrue(
                    torch.allclose(v, got, atol=2e-2, rtol=2e-2),
                    f"{msg}:{k}")
            else:
                self.assertTrue(torch.equal(v, got), f"{msg}:{k}")

    def test_page_boundary_lpl(self):
        # PAGE=64, W=10, F=4: round kv_len = ctx0+f·W+W.
        # ctx0=119 → round0 kv 129 (lpl=1) / ctx0=87 → round2 kv
        # 127 (lpl=PAGE-1) / ctx0=88 → round3 kv 128 (lpl=PAGE).
        dev = "cuda:0"
        for ctx0 in (119, 87, 88):
            ex, cfg, PAGE, V = self._mk(dev)
            p0 = (ctx0 + PAGE - 1) // PAGE
            ex.prepare_bucket(p0)
            self._fill_inputs(ex, PAGE, ctx0)
            ex.parity_noise = self._noise(ex, V)
            self._prep_local(ex)
            ex.model.cache.zero_()
            ex._local_idx.fill_(-1)
            ex.run_once(p0)
            ref = self._snap(ex)
            ex.model.cache.zero_()
            g = ex.capture(p0)
            ex.model.cache.zero_()
            ex.replay(p0)
            torch.cuda.synchronize()
            self._assert_snap(ref, self._snap(ex), f"ctx0={ctx0}")
            del g
            torch.cuda.synchronize()

    def test_page_swap_between_replays(self):
        # 캡처 후 wrapper page-ID 버퍼·in_slot을 물리 재배치
        # (논리 순서 불변) → replay 결과는 교체 전과 동일해야 함.
        dev = "cuda:0"
        ex, cfg, PAGE, V = self._mk(dev)
        ctx0 = PAGE + 21
        p0 = (ctx0 + PAGE - 1) // PAGE          # 2
        ex.prepare_bucket(p0)
        self._fill_inputs(ex, PAGE, ctx0)
        ex.parity_noise = self._noise(ex, V)
        self._prep_local(ex)
        ex.model.cache.zero_()
        g = ex.capture(p0)
        ex.model.cache.zero_()
        ex.replay(p0)
        torch.cuda.synchronize()
        ref = self._snap(ex)
        # 물리 재배치: 논리 page 0,1,2 → 물리 5,3,7 (max_blocks=8)
        kvi = torch.tensor([5, 3, 7], dtype=torch.int32, device=dev)
        for f in range(ex.F):
            ex.wrappers[p0][f]._paged_kv_indices_buf[:p0 + 1] \
                .copy_(kvi)
            base = ctx0 + f * ex.W
            pos = base + torch.arange(ex.W)
            phys = kvi.cpu()[pos // PAGE] * PAGE + pos % PAGE
            ex.in_slot[f].copy_(phys.to(torch.int32).to(dev))
        ex.model.cache.zero_()
        ex.replay(p0)
        torch.cuda.synchronize()
        self._assert_snap(ref, self._snap(ex), "page-swap")
        del g
        torch.cuda.synchronize()

    def test_graph_eager_interleave_20(self):
        # graph replay(A) ↔ eager run_once(B_i, 다른 입력) 20회 교차:
        # 교차 후 A 입력 복원 replay가 항상 최초 A 결과와 동일 —
        # eager(fallback) 개입이 graph 상태를 오염시키지 않음.
        dev = "cuda:0"
        ex, cfg, PAGE, V = self._mk(dev)
        ctx0 = PAGE + 21
        p0 = (ctx0 + PAGE - 1) // PAGE
        ex.prepare_bucket(p0)
        self._fill_inputs(ex, PAGE, ctx0)
        ex.parity_noise = self._noise(ex, V)
        self._prep_local(ex)
        ex.model.cache.zero_()
        g = ex.capture(p0)
        ex.model.cache.zero_()
        ex.replay(p0)
        torch.cuda.synchronize()
        ref = self._snap(ex)
        for i in range(20):
            # eager: 다른 root 토큰 (fallback 대역) — 엔진 게이트와
            # 동일하게 inference_mode 안에서 실행
            with torch.inference_mode():
                gB = torch.Generator().manual_seed(100 + i)
                ex.in_root_tok.copy_(
                    torch.randint(0, 100, (ex.R,), generator=gB)
                    .to(dev))
                ex.model.cache.zero_()
                ex._local_idx.fill_(-1)
                ex.run_once(p0)
            self.assertGreater(int(ex.out_valid.sum()), 0)
            # A 입력 복원 → replay == 최초 A
            self._fill_inputs(ex, PAGE, ctx0)
            ex.model.cache.zero_()
            ex.replay(p0)
            torch.cuda.synchronize()
            self._assert_snap(ref, self._snap(ex), f"iter{i}")
        del g
        torch.cuda.synchronize()

    def test_sentinel_kv_replay_writes_only_slots(self):
        # replay의 KV 기록은 in_slot의 F·W 슬롯에만 — 나머지 cache는
        # sentinel 유지 (KV 오염 부재).
        dev = "cuda:0"
        ex, cfg, PAGE, V = self._mk(dev)
        ctx0 = PAGE + 21
        p0 = (ctx0 + PAGE - 1) // PAGE
        ex.prepare_bucket(p0)
        self._fill_inputs(ex, PAGE, ctx0)
        ex.parity_noise = self._noise(ex, V)
        self._prep_local(ex)
        ex.model.cache.zero_()
        g = ex.capture(p0)
        SEN = 7.0
        ex.model.cache.fill_(SEN)
        ex.replay(p0)
        torch.cuda.synchronize()
        cache = ex.model.cache
        written = torch.zeros(cache.shape[0] * PAGE,
                              dtype=torch.bool, device=dev)
        for f in range(ex.F):
            written[ex.in_slot[f].long()] = True
        wmask = written.view(cache.shape[0], 1, PAGE, 1, 1) \
            .expand_as(cache)
        untouched = cache[~wmask]
        self.assertTrue(bool((untouched == SEN).all()),
                        "in_slot 밖 cache가 변조됨 (KV 오염)")
        self.assertFalse(bool((cache[wmask] == SEN).all()),
                         "in_slot 슬롯에 기록이 없음")
        del g
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
