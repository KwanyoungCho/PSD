"""23번 단계 2/3 — 실행기 모듈 결정적 parity (미니 모델).

게이트: 동일 입력 버퍼 + 동일 고정 noise에서
  eager run_once == captured replay
정수/구조 exact, 실수 allclose. (실모델 parity는 엔진 배선 스모크
단계 — 이 테스트는 실행기 '모듈'의 캡처-등가성을 고정한다.)
"""
import unittest
import torch
import torch.nn as nn

try:
    import flashinfer
    HAS_FI = True
except Exception:
    HAS_FI = False

from ssd.utils.context import get_context
from ssd.engine.helpers.p2_tree_executor import P2TreeExecutor


class _MiniCfg:
    duet_proxy_total_budget = 10
    duet_tree_root_count = 6
    duet_phase2_k = 4
    duet_tree_c_tensor = 3
    duet_tree_nv = 8
    duet_tree_beta = 0.5


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

    def _mk(self, dev, PAGE=64):
        V, H, HKV, D = 128, 4, 2, 64
        cfg = _MiniCfg()
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
        piv = torch.tensor([.4, .2, .1, .06, .03, .01])
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
               for k in ("out_tok", "out_par", "out_sib", "out_rawq",
                         "out_pcell", "out_valid")}
        ref_logits = ex.cell_logits.clone()
        ref_cache = ex.model.cache.clone()
        # ── 캡처 + replay (동일 입력·동일 noise)
        ex.model.cache.zero_()
        g = ex.capture(p0)
        ex.model.cache.zero_()
        for t in ref.values():
            pass
        g.replay()
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
        del g


if __name__ == "__main__":
    unittest.main()
