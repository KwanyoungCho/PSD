"""23번 단계 2 — 전체-P2 CUDA graph 실행기 (실모델, 단일 버킷 v1).

캡처 범위: [arena reset → GPU 예산 → (select → fanout → 입력/rope
gather → packed mask 기록 → raw draft forward(KV 기록→attention) →
compute_logits → WOR 샘플(전용 generator) → 자식 삽입 + [R,Nv] 직접
기록) ×F].

계약 (검증된 전제 위에서):
- plan은 capture 시 버킷당 1회 (runtime plan 0회) — page-ID/slot/
  mask/입력은 고정 주소 버퍼, 내용은 replay 전 host copy로 갱신.
- wrapper: backend="fa2" (auto는 +64MB/wrapper), use_cuda_graph=True.
- RNG: 전용 generator + register_generator_state (기본 RNG 무오염
  참증명 통과 패턴).
- set_context는 라운드당 capture 시 1회 (replay 중 파이썬 0).
- 미지원 조건(비-Llama draft/EAGLE/B>1/temp0/페이지 초과)은 호출측
  arena fallback.

v1 범위: [R,Nv] tok/par/sib/raw_q/parent_cell/valid 를 graph가 직접
기록. parent-q uniq(U-slot) 매핑만 임시 debug 경로(호출측 소형 CPU
변환)로 허용 — 리뷰12 §2 단서.
"""
import torch

import flashinfer

from ssd.utils.context import set_context, reset_context
from ssd.engine.helpers import p2_tree as PT


class P2TreeExecutor:
    def __init__(self, model, compute_logits_fn, config, device,
                 block_size, max_blocks, vocab_size):
        self.model = model
        self.compute_logits = compute_logits_fn
        self.cfg = config
        self.dev = device
        self.bs = block_size
        self.max_blocks = max_blocks
        self.V = vocab_size
        self.W = int(config.duet_proxy_total_budget)
        self.R = int(config.duet_tree_root_count or self.W)
        self.F = int(config.duet_phase2_k)
        self.C = int(config.duet_tree_c_tensor)
        self.NV = int(config.duet_tree_nv)
        W, R, F, C, NV = self.W, self.R, self.F, self.C, self.NV
        d = device
        # ── 입력 고정 버퍼 (replay 전 host가 내용 갱신)
        self.in_root_tok = torch.zeros(R, dtype=torch.int64, device=d)
        self.in_root_piv = torch.zeros(R, dtype=torch.float32, device=d)
        self.in_rope_base = torch.zeros(R, dtype=torch.int64, device=d)
        self.in_glue = torch.zeros(R, F + 1, dtype=torch.uint8, device=d)
        self.in_temps = torch.zeros(W, dtype=torch.float32, device=d)
        self.in_slot = [torch.zeros(W, dtype=torch.int32, device=d)
                        for _ in range(F)]
        self.in_ctx_len = [torch.zeros(1, dtype=torch.int32, device=d)
                           for _ in range(F)]
        self.in_block_tables = torch.zeros(1, max_blocks,
                                           dtype=torch.int32, device=d)
        # prefix 경계는 버킷 내 요청마다 달라짐 — 캡처에 박히면 안
        # 되는 '내용' (버퍼 구동; python int 슬라이싱 금지)
        self.in_prefix_len = torch.zeros(1, dtype=torch.int64, device=d)
        # ── arena / RNG / 출력
        self.arena = PT.TreeArena(R + F * W * C, d)
        self.gen = torch.Generator(device=d)
        self.gen.manual_seed(torch.initial_seed() % (2**31))
        self.out_tok = torch.zeros(R, NV, dtype=torch.int64, device=d)
        self.out_par = torch.full((R, NV), -1, dtype=torch.int64,
                                  device=d)
        self.out_sib = torch.zeros(R, NV, dtype=torch.int64, device=d)
        self.out_rawq = torch.zeros(R, NV, dtype=torch.float32, device=d)
        self.out_pcell = torch.full((R, NV), -1, dtype=torch.int64,
                                    device=d)
        self.out_valid = torch.zeros(R, dtype=torch.int64, device=d)
        self.cell_logits = torch.zeros(F * W, vocab_size,
                                       dtype=torch.float32, device=d)
        # 캡처 호환 상수
        self.ones_w = torch.ones(W, dtype=torch.uint8, device=d)
        self.lane_w = torch.arange(W, device=d)
        self.arange_R = torch.arange(R, device=d)
        self.graphs = {}                       # n_pages → CUDAGraph
        self.wrappers = {}                     # n_pages → [wrapper]*F
        self._float_ws = torch.empty(128 * 2**20, dtype=torch.uint8,
                                     device=d)

    # ---------- 버킷 준비 ----------
    def _mk_round_wrapper(self, n_pages_r, canvas_cols):
        d = self.dev
        W = self.W
        qo = torch.tensor([0, W], dtype=torch.int32, device=d)
        kvp = torch.tensor([0, n_pages_r], dtype=torch.int32, device=d)
        kvi = torch.zeros(n_pages_r, dtype=torch.int32, device=d)
        lpl = torch.tensor([self.bs], dtype=torch.int32, device=d)
        n_pk = (W * canvas_cols + 7) // 8
        mask_buf = torch.zeros(n_pk, dtype=torch.uint8, device=d)
        wr = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self._float_ws, "NHD", backend="fa2", use_cuda_graph=True,
            qo_indptr_buf=qo, paged_kv_indptr_buf=kvp,
            paged_kv_indices_buf=kvi, paged_kv_last_page_len_buf=lpl,
            custom_mask_buf=mask_buf,
            mask_indptr_buf=torch.tensor([0, W * canvas_cols],
                                         dtype=torch.int32, device=d))
        wr.plan(qo, kvp, kvi, lpl,
                self.model.config.num_attention_heads
                if hasattr(self.model, "config") else 32,
                getattr(getattr(self.model, "config", None),
                        "num_key_value_heads", 4),
                getattr(getattr(self.model, "config", None),
                        "head_dim", 64),
                self.bs,
                custom_mask=torch.zeros(W * canvas_cols,
                                        dtype=torch.bool, device=d),
                q_data_type=torch.float16, kv_data_type=torch.float16)
        wr._canvas_cols = canvas_cols
        return wr

    def prepare_bucket(self, n_pages0):
        """버킷 = 시작 page 수. round r의 canvas = (p0+1) page 고정
        (전제 검증: p+1 전체-page 0-mask 안전)."""
        F, W = self.F, self.W
        wrappers = []
        for f in range(F):
            canvas_pages = n_pages0 + 1
            wrappers.append(self._mk_round_wrapper(
                canvas_pages, canvas_pages * self.bs))
        self.wrappers[n_pages0] = wrappers
        return wrappers

    # ---------- 본체 (캡처 대상) ----------
    def _pack_row_mask(self, wr, f):
        """행별 [prefix 1s | glue | 조상셀 | self] canvas — 열 배치가
        prefix 길이(요청별 상이)에 의존하므로 전부 버퍼-구동 텐서
        연산 (python int 슬라이싱은 캡처에 박힘 — 금지)."""
        W, F = self.W, self.F
        canvas = wr._canvas_cols
        ar = self.arena
        sel, sel_valid = self._sel[f]
        plen = self.in_prefix_len                     # [1] int64 버퍼
        col = torch.arange(canvas, device=self.dev)   # [canvas]
        gW = self.in_glue.shape[1]
        r_of = torch.where(sel_valid,
                           ar.root.gather(0, sel.clamp(min=0)),
                           torch.zeros_like(sel))
        # prefix: col < plen
        m = (col.unsqueeze(0) < plen).expand(W, canvas) \
            .to(torch.uint8).clone()
        # glue: col ∈ [plen, plen+gW) → glue[r_of, col-plen]
        g_off = col.unsqueeze(0) - plen               # [1, canvas]
        in_glue_rng = (g_off >= 0) & (g_off < gW)
        g_idx = g_off.clamp(min=0, max=gW - 1)
        g_bits = self.in_glue.index_select(0, r_of) \
            .gather(1, g_idx.expand(W, canvas)) \
            * sel_valid.unsqueeze(1).to(torch.uint8)
        m = torch.where(in_glue_rng.expand(W, canvas), g_bits, m)
        # 조상 셀: col ∈ [plen+gW, plen+gW+f·W) → anc bit (col-spec0)
        spec_off = g_off - gW
        anc = ar.anc_bits.gather(0, sel.clamp(min=0)) * sel_valid.long()
        in_spec = (spec_off >= 0) & (spec_off < f * W) if f else \
            torch.zeros(1, canvas, dtype=torch.bool, device=self.dev)
        if f:
            a_bits = ((anc.unsqueeze(1)
                       >> spec_off.clamp(min=0, max=max(f * W - 1, 0)))
                      & 1).to(torch.uint8)
            m = torch.where(in_spec.expand(W, canvas), a_bits, m)
        # self 셀: col == plen+gW+f·W+lane
        self_col = plen + gW + f * W + self.lane_w.unsqueeze(1)  # [W,1]
        is_self = col.unsqueeze(0) == self_col        # [W, canvas]
        m = torch.where(is_self, self.ones_w.unsqueeze(1)
                        .expand(W, canvas), m)
        flat = m.reshape(-1)
        pad = (-flat.numel()) % 8
        if pad:
            flat = torch.cat([flat, torch.zeros(
                pad, dtype=torch.uint8, device=self.dev)])
        wbits = (1 << torch.arange(8, device=self.dev)).to(torch.uint8)
        packed = (flat.view(-1, 8) * wbits).sum(
            1, dtype=torch.int64).to(torch.uint8)
        wr._custom_mask_buf[:packed.numel()].copy_(packed)

    def run_once(self, n_pages0):
        """캡처 시 1회 트레이스 (replay 의미는 버퍼 내용에 의해)."""
        W, R, F, C, NV = self.W, self.R, self.F, self.C, self.NV
        ar = self.arena
        ar.reset()
        budgets = PT.alloc_root_budgets_gpu(
            self.in_root_piv, total=F * W, beta=self.cfg.duet_tree_beta,
            cap=NV)
        remaining = budgets.clone()
        ar.tok[:R] = self.in_root_tok
        ar.root[:R] = self.arange_R
        ar.logpri[:R] = self.in_root_piv.clamp_min(1e-9) \
            .log().double()
        ar.valid[:R] = True
        ar.n += R
        self.out_valid.zero_()
        self.out_par.fill_(-1)
        self.out_pcell.fill_(-1)
        tip_idx = self.arange_R.clone()
        tip_depth = torch.zeros(R, dtype=torch.int64, device=self.dev)
        self._sel = {}
        wrappers = self.wrappers[n_pages0]
        for f in range(F):
            sel, sel_valid = PT._arena_select(ar, "level", W, f, F,
                                              tip_idx, remaining)
            self._sel[f] = (sel, sel_valid)
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
            rope = torch.where(
                sel_valid,
                self.in_rope_base.gather(0, r_of)
                + ar.depth.gather(0, sel.clamp(min=0)),
                self.in_rope_base[0].expand(W))
            self._pack_row_mask(wrappers[f], f)
            # ── raw draft forward (capture-시 context 1회 bake)
            set_context(
                is_prefill=False,
                slot_mapping=self.in_slot[f],
                context_lens=self.in_ctx_len[f],
                block_tables=self.in_block_tables,
                active_mq_len=W,
                active_wrappers={1: wrappers[f]},
            )
            hidden = self.model(ids, rope)
            logits = self.compute_logits(hidden, False)[:W].float()
            reset_context()
            self.cell_logits[f * W:(f + 1) * W] = logits
            toks, raws = PT.tree_sample_wor(
                logits, self.in_temps, C, assume_pos_temps=True,
                generator=self.gen)
            # ── 삽입 + [R,NV] 직접 기록
            lane_cell = f * W + self.lane_w
            ar.cell.scatter_(0, sel.clamp(min=0),
                             torch.where(sel_valid, lane_cell,
                                         ar.cell.gather(
                                             0, sel.clamp(min=0))))
            ar.state.scatter_(0, sel.clamp(min=0),
                              torch.where(sel_valid,
                                          torch.ones_like(sel),
                                          ar.state.gather(
                                              0, sel.clamp(min=0))))
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
            child_root = ar.root.gather(0, par)
            ar.root.scatter_(0, sl, child_root)
            ar.sib.scatter_(0, sl, cix)
            lp = ar.logpri.gather(0, par) + rq.clamp_min(1e-9).log()
            ar.logpri.scatter_(0, sl, torch.where(
                ok_q, lp, torch.full_like(lp, float("-inf"))))
            ar.raw_q.scatter_(0, sl, rq)
            ar.valid.scatter_(0, sl, ok_q)
            ar.state.scatter_(0, sl, torch.where(
                ok_q, torch.zeros_like(sl), torch.ones_like(sl)))
            parent_cell_of_child = ar.cell.gather(0, par).clamp(min=0)
            ar.anc_bits.scatter_(
                0, sl, ar.anc_bits.gather(0, par)
                | (torch.ones_like(par) << parent_cell_of_child))
            # [R,NV] 직접 기록: 이 라운드 자식들의 root-local index =
            # out_valid[root] + (동일 root 내 순번). lane-major 순서
            # 보존: one_hot cumsum (고정 shape).
            oh = torch.nn.functional.one_hot(child_root, R).long() \
                * ok_q.unsqueeze(1).long()
            rank_in_round = (oh.cumsum(0) - oh)
            local = (self.out_valid.gather(
                0, child_root) + (rank_in_round * oh).sum(1))
            local_c = local.clamp(max=NV - 1)
            dst = child_root * NV + local_c
            wmask = ok_q & (local < NV)
            dst_safe = torch.where(wmask, dst,
                                   torch.zeros_like(dst))
            def _w(outbuf, val):
                flatb = outbuf.view(-1)
                cur = flatb.gather(0, dst_safe)
                flatb.scatter_(0, dst_safe,
                               torch.where(wmask, val, cur))
            _w(self.out_tok, toks.reshape(-1))
            _w(self.out_sib, cix)
            _w(self.out_pcell, ar.cell.gather(0, par))
            _w(self.out_rawq, rq.float())
            # parent_local: 부모의 (root,local) — 부모가 root면 -1.
            # 부모 local은 arena에 없으므로 slot별 local을 arena에
            # 병기 저장 (sib 필드와 별개 local_idx 텐서).
            par_local = self._local_idx.gather(0, par)
            _w(self.out_par, par_local)
            self._local_idx.scatter_(0, sl, torch.where(
                wmask, local, torch.full_like(local, -1)))
            self.out_valid.scatter_add_(0, self.arange_R,
                                        oh.sum(0))
            ar.n = ar.n + fan.sum()
            tip_adv = sel_valid & (sel == tip_idx.gather(0, r_of)) \
                & (fan > 0)
            old_tip = tip_idx.gather(0, r_of)
            delta = torch.where(tip_adv, offs - old_tip,
                                torch.zeros_like(offs))
            tip_idx.scatter_add_(0, r_of, delta)
            tip_depth.scatter_add_(0, r_of, tip_adv.long())

    def capture(self, n_pages0):
        if n_pages0 not in self.wrappers:
            self.prepare_bucket(n_pages0)
        self._local_idx = torch.full((self.arena.capacity,), -1,
                                     dtype=torch.int64, device=self.dev)
        # 워밍업 ×2 (allocator/커널 준비 — eager)
        for _ in range(2):
            self._local_idx.fill_(-1)
            self.run_once(n_pages0)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        g.register_generator_state(self.gen)
        with torch.cuda.graph(g):
            self._local_idx.fill_(-1)
            self.run_once(n_pages0)
        self.graphs[n_pages0] = g
        return g
