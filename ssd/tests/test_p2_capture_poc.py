"""23번 — 전체-P2 CUDA graph PoC (리뷰9 순서 4 / 리뷰10 승인 경로).

목표 (kernel 융합 전, 기존 arena 텐서 연산 그대로):
  [arena reset → GPU 예산 → (select → fanout → mask 기록 →
   attention(=raw forward 대역) → 샘플 → 자식 삽입) ×4]
전체를 **한 개의 torch.cuda.graph로 캡처**하고,
  ① 캡처 성공 + 재실행 무오류
  ② 재실행마다 RNG 신선 (동적 트리 내용 변화)
  ③ 토폴로지 불변량 유지 (부모<자식, 예산 준수, R/W 계약)
  ④ host-gap: 캡처 replay가 eager 순차 실행 대비 launch 공백 제거
를 확인한다. 실모델 대신 FlashInfer attention + 선형 head를
"raw forward 대역"으로 사용 — 캡처 메커니즘(wrapper×4 in-graph,
mask 런타임 갱신, KV 라운드별 기록, RNG-in-graph)이 검증 대상.

주의: RNG-in-graph는 replay마다 philox가 전진 — eager와의 비트
패리티는 기대치가 아니며 (분포 동일), 실행기 채택 게이트는 분포
지표(인터리브 AL)로 판정한다 (문서 23).
"""
import unittest
import torch

try:
    import flashinfer
    HAS_FI = True
except Exception:
    HAS_FI = False

import ssd.engine.helpers.p2_tree as PT

PAGE = 16
H, HKV, D = 4, 2, 64
W, F, C, R, NV = 10, 4, 3, 6, 8
V = 256                      # 소형 vocab (head 선형)
CTX = PAGE * 2 + 5           # page 중간에서 시작 (canvas 검증 겸)


class MiniP2Executor:
    """캡처 가능한 미니 P2 실행기 — 전 상태 고정 주소 버퍼."""

    def __init__(self, dev):
        self.dev = dev
        g = torch.Generator().manual_seed(3)
        # "모델" 파라미터 — CPU 생성 후 전송 (기본 CUDA RNG를 in-graph
        # 전용으로 격리: 캡처된 제너레이터를 setup에서 건드리면
        # "Offset increment outside graph capture" — PoC 관찰 사실)
        def rnd(*shape, scale=1.0):
            return (torch.randn(*shape, generator=g) * scale) \
                .to(dtype=torch.float16, device=dev)
        n_pages = (CTX + F * W + PAGE - 1) // PAGE
        self.n_pages = n_pages
        self.cache = rnd(n_pages, 2, PAGE, HKV, D)
        self.q_proj = rnd(D, H * D, scale=0.05)
        self.head = rnd(H * D, V, scale=0.05)
        self.tok_emb = rnd(V, D, scale=0.05)
        # arena (persistent)
        self.ar = PT.TreeArena(R + F * W * C, dev)
        self.piv = torch.tensor([.4, .2, .1, .06, .03, .01],
                                device=dev)
        self.root_toks = torch.arange(10, 10 + R, device=dev)
        self.rope_base = torch.full((R,), CTX - 1, dtype=torch.int64,
                                    device=dev)
        self.glue = torch.ones(R, 1, dtype=torch.uint8, device=dev)
        self.temps = torch.full((W,), 0.8, device=dev)
        # round별 preplanned wrapper (고정 buf + canvas plan)
        self.float_ws = torch.empty(96 * 2**20, dtype=torch.uint8,
                                    device=dev)
        self.wrs = []
        self.mask_bufs = []
        for f in range(F):
            kv_len = CTX + f * W
            npg = (kv_len + W + PAGE - 1) // PAGE   # 이번 라운드 기록 포함
            canvas = npg * PAGE
            mask_buf = torch.zeros(W * canvas, dtype=torch.uint8,
                                   device=dev)
            qo = torch.tensor([0, W], dtype=torch.int32)
            kvp = torch.tensor([0, npg], dtype=torch.int32)
            kvi = torch.arange(npg, dtype=torch.int32)
            lpl = torch.tensor([PAGE], dtype=torch.int32)  # canvas 끝
            wr = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.float_ws, "NHD", use_cuda_graph=True,
                qo_indptr_buf=qo.to(dev),
                paged_kv_indptr_buf=kvp.to(dev),
                paged_kv_indices_buf=kvi.to(dev),
                paged_kv_last_page_len_buf=lpl.to(dev),
                custom_mask_buf=mask_buf.view(torch.uint8),
                mask_indptr_buf=torch.tensor([0, W * canvas],
                                             dtype=torch.int32,
                                             device=dev))
            wr.plan(qo.to(dev), kvp.to(dev), kvi.to(dev),
                    lpl.to(dev), H, HKV, D, PAGE,
                    custom_mask=torch.zeros(W * canvas,
                                            dtype=torch.bool,
                                            device=dev),
                    q_data_type=torch.float16,
                    kv_data_type=torch.float16)
            self.wrs.append(wr)
            self.mask_bufs.append((mask_buf, canvas))
        # 출력 고정 버퍼
        self.out_par = torch.zeros(R + F * W * C + 1, dtype=torch.int64,
                                   device=dev)
        # 캡처 호환 상수 (advanced-index 대입의 스칼라 RHS는 H2D — 금지)
        self.ones_w_u8 = torch.ones(W, dtype=torch.uint8, device=dev)
        self.lane_w = torch.arange(W, device=dev)

    def _round(self, ar, f, tip_idx, tip_depth, remaining):
        sel, sel_valid = PT._arena_select(ar, "level", W, f, F,
                                          tip_idx, remaining)
        reserve = (F - tip_depth).clamp(min=0)
        fan = PT._arena_fanout_backbone(ar, sel, sel_valid, tip_idx,
                                        remaining, reserve, C, R)
        r_of = torch.where(sel_valid,
                           ar.root.gather(0, sel.clamp(min=0)),
                           torch.zeros_like(sel))
        remaining.scatter_add_(0, r_of, -fan)
        ids = torch.where(sel_valid,
                          ar.tok.gather(0, sel.clamp(min=0)),
                          torch.zeros_like(sel))
        # mask: canvas에 [prefix | 조상 셀] 기록 (bit 단위 대신 bool
        # canvas — FlashInfer packed와 등가는 본 PoC 범위 밖, 내용
        # 갱신이 graph 안에서 됨을 검증)
        mask_buf, canvas = self.mask_bufs[f]
        mb = mask_buf.view(W, canvas)
        mb.zero_()
        mb[:, :CTX] = 1
        anc = ar.anc_bits.gather(0, sel.clamp(min=0)) * sel_valid.long()
        if f:                                   # 기존 셀 조상 (f·W 폭)
            shifts = torch.arange(f * W, device=self.dev)
            bits = ((anc.unsqueeze(1) >> shifts) & 1).to(torch.uint8)
            mb[:, CTX:CTX + f * W] = bits
        lane = self.lane_w
        mb[lane, CTX + f * W + lane] = self.ones_w_u8   # 자기 새 슬롯
        # "forward": emb → q → attention → head
        x = self.tok_emb.index_select(0, ids.clamp(min=0))
        q = (x @ self.q_proj).view(W, H, D).to(torch.float16)
        att = self.wrs[f].run(q, self.cache)          # [W, H, D]
        logits = (att.reshape(W, H * D) @ self.head).float()
        toks, raws = PT.tree_sample_wor(logits, self.temps, C,
                                        assume_pos_temps=True)
        # KV 기록 (라운드 f의 W행을 cache의 [CTX+f·W ...) 슬롯에)
        pos = CTX + f * W + torch.arange(W, device=self.dev)
        pg, off = pos // PAGE, pos % PAGE
        kv_new = x.view(W, 1, D).expand(W, HKV, D).to(torch.float16)
        self.cache[pg, 0, off] = kv_new
        self.cache[pg, 1, off] = kv_new
        # 자식 삽입 (run_rollout_arena와 동일 시퀀스)
        lane_cell = f * W + torch.arange(W, device=self.dev)
        ar.cell.scatter_(0, sel.clamp(min=0),
                         torch.where(sel_valid, lane_cell,
                                     ar.cell.gather(0, sel.clamp(min=0))))
        ar.state.scatter_(0, sel.clamp(min=0),
                          torch.where(sel_valid, torch.ones_like(sel),
                                      ar.state.gather(0, sel.clamp(min=0))))
        offs = ar.n + torch.cumsum(fan, 0) - fan
        cgrid = torch.arange(C, device=self.dev)
        slot = offs.unsqueeze(1) + cgrid.unsqueeze(0)
        child_ok = cgrid.unsqueeze(0) < fan.unsqueeze(1)
        scratch = ar.capacity - 1
        sl = torch.where(child_ok, slot,
                         torch.full_like(slot, scratch)).reshape(-1)
        par = sel.unsqueeze(1).expand(W, C).reshape(-1)
        cix = cgrid.unsqueeze(0).expand(W, C).reshape(-1)
        rq = raws.double().reshape(-1)
        ok_q = (rq > 0) & child_ok.reshape(-1)
        ar.tok.scatter_(0, sl, toks.reshape(-1))
        ar.parent_idx.scatter_(0, sl, par)
        ar.depth.scatter_(0, sl, ar.depth.gather(0, par) + 1)
        ar.root.scatter_(0, sl, ar.root.gather(0, par))
        ar.sib.scatter_(0, sl, cix)
        lp = ar.logpri.gather(0, par) + rq.clamp_min(1e-9).log()
        ar.logpri.scatter_(0, sl, torch.where(
            ok_q, lp, torch.full_like(lp, float("-inf"))))
        ar.raw_q.scatter_(0, sl, rq)
        ar.valid.scatter_(0, sl, ok_q)
        ar.state.scatter_(0, sl, torch.where(
            ok_q, torch.zeros_like(sl), torch.ones_like(sl)))
        ar.anc_bits.scatter_(
            0, sl, ar.anc_bits.gather(0, par)
            | (torch.ones_like(par)
               << ar.cell.gather(0, par).clamp(min=0)))
        ar.n = ar.n + fan.sum()
        tip_adv = sel_valid & (sel == tip_idx.gather(0, r_of)) & (fan > 0)
        old_tip = tip_idx.gather(0, r_of)
        delta = torch.where(tip_adv, offs - old_tip,
                            torch.zeros_like(offs))
        tip_idx.scatter_add_(0, r_of, delta)
        tip_depth.scatter_add_(0, r_of, tip_adv.long())

    def run_once(self):
        """전체 P2 (reset 포함) — 캡처 대상."""
        ar = self.ar
        ar.reset()
        budgets = PT.alloc_root_budgets_gpu(self.piv, total=F * W,
                                            beta=0.5, cap=NV)
        remaining = budgets.clone()
        ar.tok[:R] = self.root_toks
        ar.root[:R] = torch.arange(R, device=self.dev)
        ar.logpri[:R] = self.piv.clamp_min(1e-9).float().log().double()
        ar.valid[:R] = True
        ar.n += R
        tip_idx = torch.arange(R, device=self.dev)
        tip_depth = torch.zeros(R, dtype=torch.int64, device=self.dev)
        for f in range(F):
            self._round(ar, f, tip_idx, tip_depth, remaining)


@unittest.skipUnless(HAS_FI and torch.cuda.is_available(), "no fi/cuda")
class TestP2CapturePoC(unittest.TestCase):
    def tearDown(self):
        # 캡처 그래프 파기 + RNG 상태 복구 — 그래프가 살아있는 동안
        # 기본 CUDA 제너레이터는 graph-모드 (eager exponential_이
        # "Offset increment outside graph capture"로 실패; PoC 관찰)
        import gc
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.manual_seed(999)

    def test_capture_replay_and_invariants(self):
        dev = "cuda:0"
        ex = MiniP2Executor(dev)
        # 워밍업 (capture 전 필수 — allocator/커널 준비)
        torch.cuda.manual_seed(100)
        for _ in range(3):
            ex.run_once()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            ex.run_once()
        # ① 재실행 무오류 ×3
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        pool1 = ex.ar.to_pool(R)
        # ③ 불변량: 노드 수·부모<자식·root 범위·예산 상한
        self.assertGreater(pool1.n, R * 2)
        for i in range(pool1.n):
            p = int(pool1.parent_idx[i])
            self.assertLess(p, i)             # 부모 선행
            self.assertLess(int(pool1.root[i]), R)
        gen = [0] * R
        for i in range(pool1.n):
            if int(pool1.parent_idx[i]) >= 0:
                gen[int(pool1.root[i])] += 1
        self.assertTrue(all(x <= NV for x in gen), gen)
        # ② RNG 신선: 재실행하면 다른 트리
        g.replay()
        torch.cuda.synchronize()
        pool2 = ex.ar.to_pool(R)
        same = (pool1.n == pool2.n and
                pool1.tok[:pool1.n].tolist() ==
                pool2.tok[:pool2.n].tolist())
        self.assertFalse(same, "replay가 같은 트리 — RNG 미전진")
        del g

    def test_gap_removed_vs_eager(self):
        """④ host-gap: 캡처 replay 총 시간 vs eager 4-round 순차."""
        dev = "cuda:0"
        ex = MiniP2Executor(dev)
        for _ in range(3):
            ex.run_once()
        torch.cuda.synchronize()
        import time
        t0 = time.perf_counter()
        for _ in range(20):
            ex.run_once()
        torch.cuda.synchronize()
        t_eager = (time.perf_counter() - t0) / 20 * 1e3
        gph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gph):
            ex.run_once()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            gph.replay()
        torch.cuda.synchronize()
        t_replay = (time.perf_counter() - t0) / 20 * 1e3
        print(f"[PoC ④] eager 4-round {t_eager:.2f}ms → captured "
              f"replay {t_replay:.2f}ms (×{t_eager/max(t_replay,1e-9):.1f})")
        _ratio_ok = t_replay < t_eager * 0.6
        del gph
        self.assertTrue(_ratio_ok,
                        f"replay {t_replay:.2f} !< eager {t_eager:.2f}·0.6")


if __name__ == "__main__":
    unittest.main()
