import os
import time
import torch
import torch.distributed as dist
import dataclasses

from ssd.engine.model_runner import ModelRunner
from ssd.config import Config
from ssd.utils.context import set_context, reset_context
from ssd.utils.async_helpers.async_spec_helpers import get_forked_recovery_tokens_from_logits, make_glue_decode_input_ids
from ssd.utils.async_helpers.nccl_pack import recv_int64
from ssd.engine.helpers.cudagraph_helpers import flush_draft_profile

PROFILE_DRAFT = os.environ.get("SSD_PROFILE_DRAFT", "0") == "1"

# Split-only K1/K2 mode (per docs/mesa/04-split-k1k2-design.md):
#   - Draft pass = K1 forwards → draft-sourced rows of depth K1
#   - Proxy pass = K2 forwards → proxy-sourced rows of depth K2
#   - NO continuation. valid_k space = {K1, K2}.
# Hybrid path is untouched when this flag is off.
SPLIT_K1K2_MODE = os.environ.get("SSD_FORCE_SPLIT_K1K2", "0") == "1"
# Trace gate for split-K1/K2 contract validation. Active only when
# SSD_TRACE_SPLIT_K1K2=1. Logs every step's valid_k / shapes / per-position
# proxy fan_out so we can prove what's real data vs zero-pad.
TRACE_SPLIT_K1K2 = os.environ.get("SSD_TRACE_SPLIT_K1K2", "0") == "1"

ttl = 0
ttl_hit = 0

class DraftRunner(ModelRunner):
    
    @classmethod
    def create_draft_config(cls, cfg: Config) -> Config:
        """Create a draft config from the main config without instantiating DraftRunner."""
        draft_cfg = dataclasses.replace(
            cfg,
            model=cfg.draft,
            gpu_memory_utilization = (0.75 if not cfg.draft_async else 0.8), # REMAINING SPACE if not draft_async
            tokenizer_path=cfg.model if cfg.use_eagle else None,
            d_model_target=cfg.hf_config.hidden_size if cfg.use_eagle and cfg.hf_config else None,
            enforce_eager=cfg.enforce_eager,
        )
        return draft_cfg

    def __init__(self, cfg: Config, rank: int = 0, init_q = None):
        self.draft_cfg = self.create_draft_config(cfg)
        self.is_draft = True # this is is_draft, use self.config.draft for the draft model path 
        self.prev_num_tokens = None
        super().__init__(self.draft_cfg, rank=rank, event=None, is_draft=True, num_tp_gpus=1, init_q=init_q)
        
        if self.config.use_eagle:
            assert self.config.jit_speculate, \
                "EAGLE requires jit_speculate=True (cache misses need draft activations)"

        if self.is_draft and self.draft_async:
            self._reset_tree_cache_tensors()
            self._init_prealloc_buffers()
            self._draft_step_times = []
            # MESA: capture draft/proxy layout CudaGraphs.
            # Phase 3 (K1 split): also capture phase1_layout_long (K=K1 forwards,
            # same MQ_LEN as draft so capture shape is identical).
            if self.config.mesa_enabled and not self.enforce_eager:
                from ssd.engine.helpers.cudagraph_helpers import (
                    capture_fi_tree_decode_cudagraph,
                    capture_phase2_hybrid_cudagraph,
                )
                # Legacy draft/proxy layouts: skipped in split-only mode (split
                # path uses split_k1_{long,short} + split_k2 only).
                if SPLIT_K1K2_MODE:
                    _layouts_to_capture = []
                else:
                    _layouts_to_capture = [self.draft_layout, self.proxy_layout]
                # Hybrid phase1 long/short — only present in hybrid mode.
                if self.phase1_layout_long is not None:
                    _layouts_to_capture.append(self.phase1_layout_long)
                if self.phase1_layout_short is not None:
                    _layouts_to_capture.append(self.phase1_layout_short)
                # Split-only K1/K2 mode: capture phase1 long/short + phase2 K2.
                if SPLIT_K1K2_MODE:
                    if self.split_k1_long_layout is not None:
                        _layouts_to_capture.append(self.split_k1_long_layout)
                    if self.split_k1_short_layout is not None:
                        _layouts_to_capture.append(self.split_k1_short_layout)
                    if self.split_k2_layout is not None:
                        _layouts_to_capture.append(self.split_k2_layout)
                for _layout in _layouts_to_capture:
                    _gv, _pool, _graphs, _bs = capture_fi_tree_decode_cudagraph(self, layout=_layout)
                    self.graph_vars[_layout.graph_key] = _gv
                    self.graph_pools[_layout.graph_key] = _pool
                    self.graphs[_layout.graph_key] = _graphs
                    self.graph_bs_list[_layout.graph_key] = _bs
                print(f'[MESA] Captured FI tree decode CudaGraphs ({len(_layouts_to_capture)} layouts)', flush=True)

                # Phase 9B-1: capture phase2_hybrid CG for both buckets.
                # Skipped in split-only K1/K2 mode (split path doesn't use them).
                if self.config.mesa_phase1_k is not None and not SPLIT_K1K2_MODE:
                    _hybrid_cg_buckets = [
                        ("long", self.config.mesa_phase1_k + self.config.mesa_phase2_k),
                        ("short", self.config.mesa_phase2_k),
                    ]
                    for _bk, _Kstep in _hybrid_cg_buckets:
                        _key = f"phase2_hybrid_{_bk}_cg"
                        if _key not in self.prefill_wrappers_by_layout:
                            continue
                        _gv2, _pool2, _g2, _total2 = capture_phase2_hybrid_cudagraph(
                            self, bucket=_bk, K_step=_Kstep,
                        )
                        self.graph_vars[_key] = _gv2
                        self.graph_pools[_key] = _pool2
                        self.graphs[_key] = _g2
                    print(f'[MESA hybrid] Captured phase2_hybrid CGs '
                          f'({len(_hybrid_cg_buckets)} buckets)', flush=True)
            print(f'DraftRunner set up, starting draft_loop', flush=True)
            self.draft_loop()

    def draft_async_prefill(self):
        assert self.draft_async and self.is_draft

        # 1) Receive metadata then individual tensors
        # First recv metadata to learn sizes
        metadata = torch.zeros(5, dtype=torch.int64, device=self.device)
        dist.recv(metadata, src=0, group=self.async_pg)
        total_new_tokens, batch_size, max_blocks, use_eagle, eagle_act_dim = metadata.tolist()
        if use_eagle:
            assert eagle_act_dim == 3 * self.config.d_model_target, (
                f"EAGLE activation dimension {eagle_act_dim} does not match expected dimension 3 * {self.config.d_model_target}"
            )

        # 2) receive fused int64 payload (input_ids + num_tokens + draft_block_table)
        fused_total = total_new_tokens + batch_size + batch_size * max_blocks
        fused = recv_int64(self.async_pg, src=0, total_length=fused_total, device=self.device)
        off = 0
        input_ids = fused[off:off + total_new_tokens]; off += total_new_tokens
        num_tokens = fused[off:off + batch_size]; off += batch_size
        draft_block_table = fused[off:off + batch_size * max_blocks].view(batch_size, max_blocks).to(torch.int32); off += batch_size * max_blocks
        assert off == fused_total

        eagle_acts = None
        if use_eagle:
            eagle_acts = torch.zeros(
                total_new_tokens, eagle_act_dim, dtype=self.hf_config.torch_dtype, device=self.device,
            )
            dist.recv(eagle_acts, src=0, group=self.async_pg)

        prefill_ctxt = self.prepare_prefill_ctxt(num_tokens, draft_block_table)

        # 5) set up context exactly like prepare_prefill() does:
        set_context(
            is_prefill=True,
            cu_seqlens_q=prefill_ctxt["cu_seqlens_q"],
            cu_seqlens_k=prefill_ctxt["cu_seqlens_k"],
            max_seqlen_q=prefill_ctxt["max_seqlen_q"],
            max_seqlen_k=prefill_ctxt["max_seqlen_k"],
            slot_mapping=prefill_ctxt["slot_map"],
            context_lens=None,
        ) # , block_tables=block_tables, commenting this out essentially removes prefix caching

        # 6) run the draft model in prefill mode
        positions = prefill_ctxt["positions"]
        if self.config.use_eagle:
            self.run_model(input_ids, positions, is_prefill=True, last_only=True, hidden_states=eagle_acts)
        else:
            self.run_model(input_ids, positions, is_prefill=True, last_only=True, hidden_states=eagle_acts)

        # 7) clean up
        reset_context()

    def _reset_tree_cache_tensors(self):
        """Reset tensor-backed tree cache to empty."""
        # initialize as empty keys on correct device; tokens/logits set to None until first populate
        self.tree_cache_keys = torch.zeros(
            (0, 3), dtype=torch.int64, device=self.device)
        self.tree_cache_tokens = None
        self.tree_cache_logits = None
        self.tree_cache_activations = None
        # MESA: keys[:_last_n_draft_keys] = phase 1 (draft-sourced),
        # keys[_last_n_draft_keys:] = phase 2 (proxy-sourced). 0 = non-MESA / not yet populated.
        self._last_n_draft_keys = 0
        # Per-row valid_k. Phase 1 (Phase 2 hybrid plan) Phase 4 will populate this with
        # K_long for draft-sourced rows and K_short for proxy-sourced rows. For now,
        # empty / None (current code treats all rows as K).
        self.tree_cache_valid_k = None

    def _init_prealloc_buffers(self):
        # PERFORMANCE: pre-allocate constant tensors used every draft step to avoid repeated CUDA mallocs
        from ssd.engine.helpers.tree_layout import create_tree_layout
        K = self.config.speculate_k
        d = self.device

        # Layout-independent tensors
        self._arange_kp1 = torch.arange(K + 1, device=d, dtype=torch.int64)
        self._arange_2kp1 = torch.arange(2 * K + 1, device=d, dtype=torch.int64)

        # full_layout: 기존 SSD용 (non-MESA + MESA 비활성)
        self.full_layout = create_tree_layout(
            name="full",
            fan_out_list=self.config.fan_out_list,
            fan_out_list_miss=self.config.fan_out_list_miss,
            K=K, device=d)

        # Backward compat: 기존 전역 변수를 full_layout으로 위임
        self._step_pos_offsets = self.full_layout.step_pos_offsets
        self._step_rope_offsets = self.full_layout.step_rope_offsets
        self._fan_idx_hit = self.full_layout.fan_idx_hit
        self._fan_idx_miss = self.full_layout.fan_idx_miss
        self._arange_mq = self.full_layout.arange_mq

        # MESA: draft_layout + proxy_layout (legacy two-pass).
        # Phase 2 hybrid (`docs/mesa/01-design.md Part 5` Phase 3b):
        # additionally creates `phase1_layout_long` / `phase1_layout_short` if
        # `mesa_phase1_k` / `mesa_phase2_k` are set in config. These have
        # forward_depth = K1 (config) and position_count = K_long+1 / K_short+1.
        if self.config.mesa_enabled:
            draft_fo = self.config.mesa_draft_fan_out
            proxy_fo = self.config.mesa_proxy_fan_out
            self.draft_layout = create_tree_layout(
                name="draft",
                fan_out_list=[draft_fo] * (K + 1),
                fan_out_list_miss=[draft_fo] * (K + 1),
                K=K, device=d)
            self.proxy_layout = create_tree_layout(
                name="proxy",
                fan_out_list=[proxy_fo] * (K + 1),
                fan_out_list_miss=[proxy_fo] * (K + 1),
                K=K, device=d)
            # Hybrid Phase 1 layouts (only created when K1/K2 are configured).
            # forward_depth = K1, position_count varies per bucket.
            self.phase1_layout_long = None
            self.phase1_layout_short = None
            # v1 (proxy K2 단축) layout — Phase 2 proxy uses this in the hybrid
            # config path. forward_depth = K2 (= K_short). position_count tracks
            # the incoming valid_k bucket; long-hit uses K_long+1.
            self.proxy_layout_short_long = None   # long-hit bucket: pos=K_long+1, K=K2
            self.proxy_layout_short_short = None  # short-hit bucket: pos=K_short+1, K=K2
            # Split-only K1/K2 mode layouts (per docs/mesa/04-split-k1k2-design.md):
            #   - split_k1_long  : K=K1, position_count=K1+1 (draft pass, valid_k=K1 hit / miss)
            #   - split_k1_short : K=K1, position_count=K2+1 (draft pass, valid_k=K2 hit; K2<=K1 only)
            #   - split_k2       : K=K2, position_count=K2+1 (proxy pass)
            # No continuation. Supported scope: K2 <= K1 (asserted at runtime entry).
            self.split_k1_long_layout = None
            self.split_k1_short_layout = None
            self.split_k2_layout = None
            if self.config.mesa_phase1_k is not None:
                K1 = self.config.mesa_phase1_k
                K2 = self.config.mesa_phase2_k
                K_long = K1 + K2  # = speculate_k
                K_short = K2
                # Hybrid phase1/proxy_short layouts — skipped in split-only
                # mode (split path uses split_k1_long/short + split_k2 only).
                if not SPLIT_K1K2_MODE:
                    self.phase1_layout_long = create_tree_layout(
                        name="phase1_long",
                        fan_out_list=[draft_fo] * (K_long + 1),
                        fan_out_list_miss=[draft_fo] * (K_long + 1),
                        K=K1, device=d, position_count=K_long + 1,
                    )
                    self.phase1_layout_short = create_tree_layout(
                        name="phase1_short",
                        fan_out_list=[draft_fo] * (K_short + 1),
                        fan_out_list_miss=[draft_fo] * (K_short + 1),
                        K=K1, device=d, position_count=K_short + 1,
                    )
                    # v1 proxy short layouts. proxy_fo per position; forward depth K2.
                    self.proxy_layout_short_long = create_tree_layout(
                        name="proxy_short_long",
                        fan_out_list=[proxy_fo] * (K_long + 1),
                        fan_out_list_miss=[proxy_fo] * (K_long + 1),
                        K=K2, device=d, position_count=K_long + 1,
                    )
                    self.proxy_layout_short_short = create_tree_layout(
                        name="proxy_short_short",
                        fan_out_list=[proxy_fo] * (K_short + 1),
                        fan_out_list_miss=[proxy_fo] * (K_short + 1),
                        K=K2, device=d, position_count=K_short + 1,
                    )
                    print(f'[MESA hybrid] phase1 layouts: '
                          f'long MQ_LEN={self.phase1_layout_long.MQ_LEN} (K1={K1}, pos={K_long + 1}), '
                          f'short MQ_LEN={self.phase1_layout_short.MQ_LEN} (K1={K1}, pos={K_short + 1})',
                          flush=True)
                    print(f'[MESA hybrid] proxy_short layouts: '
                          f'long MQ_LEN={self.proxy_layout_short_long.MQ_LEN} (K2={K2}, pos={K_long + 1}), '
                          f'short MQ_LEN={self.proxy_layout_short_short.MQ_LEN} (K2={K2}, pos={K_short + 1})',
                          flush=True)
                # Split-only K1/K2 mode: phase1 long/short bucket dispatch
                # (per docs/mesa/04-split-k1k2-design.md). Supports K2 <= K1 only.
                if SPLIT_K1K2_MODE:
                    # Hard check (assert is stripped under python -O).
                    if K2 > K1:
                        raise ValueError(
                            f"split-K1K2 mode requires K2 <= K1, got K1={K1}, K2={K2}. "
                            f"K2>K1 is not supported (proxy_horizon tail source undefined). "
                            f"See docs/mesa/04-split-k1k2-design.md."
                        )
                    # Phase 1 fan_out_list (non-uniform supported).
                    # Phase 2 always uniform proxy_fo per position (Phase 2 selection
                    # logic / target proxy compute is uniform; non-uniform Phase 2
                    # would need policy-based dynamic fan_out — separate design).
                    _p1_fol_user = self.config.mesa_split_phase1_fan_out_list
                    if _p1_fol_user is not None:
                        if len(_p1_fol_user) != K1 + 1:
                            raise ValueError(
                                f"mesa_split_phase1_fan_out_list len={len(_p1_fol_user)} "
                                f"must equal K1+1={K1 + 1}, got {_p1_fol_user}"
                            )
                        _p1_fol_long = list(_p1_fol_user)
                    else:
                        _p1_fol_long = [draft_fo] * (K1 + 1)
                    if self.config.mesa_split_phase2_fan_out_list is not None:
                        raise NotImplementedError(
                            "split-K1/K2 Phase 2 non-uniform fan_out is not supported "
                            "(Phase 2 selection is uniform; would need policy-based "
                            "dynamic fan_out — separate design)."
                        )

                    self.split_k1_long_layout = create_tree_layout(
                        name="split_k1_long",
                        fan_out_list=_p1_fol_long,
                        fan_out_list_miss=_p1_fol_long,
                        K=K1, device=d, position_count=K1 + 1,
                    )
                    if K2 < K1:
                        # Short bucket: first K2+1 positions of phase1 list.
                        _p1_fol_short = _p1_fol_long[:K2 + 1]
                        self.split_k1_short_layout = create_tree_layout(
                            name="split_k1_short",
                            fan_out_list=_p1_fol_short,
                            fan_out_list_miss=_p1_fol_short,
                            K=K1, device=d, position_count=K2 + 1,
                        )
                    # Phase 2 layout — capture-time placeholder for unified Policy B.
                    # docs/mesa/05-policy-b-fix.md Section 3.7 (옵션 A-1):
                    #   K = K2 (forward depth, fixed)
                    #   MQ_LEN = pfo*(K_max+1) (worst-case sizing)
                    #   position_count / fan_out_list updated per-step in caller.
                    K_rank_max = K1 if K1 >= K2 else K2     # = K1 (K2 ≤ K1 invariant)
                    self.split_k2_layout = create_tree_layout(
                        name="split_k2",
                        fan_out_list=[proxy_fo] * (K_rank_max + 1),
                        fan_out_list_miss=[proxy_fo] * (K_rank_max + 1),
                        K=K2, device=d, position_count=K_rank_max + 1,
                    )
                    _short_str = (
                        f'split_k1_short MQ={self.split_k1_short_layout.MQ_LEN} '
                        f'(K=K1={K1}, pos={K2+1}), '
                        if self.split_k1_short_layout is not None else
                        'split_k1_short SKIPPED (K1==K2, long bucket reused), '
                    )
                    print(f'[MESA split-K1K2] layouts: '
                          f'split_k1_long MQ={self.split_k1_long_layout.MQ_LEN} (K=K1={K1}, pos={K1+1}), '
                          f'{_short_str}'
                          f'split_k2 MQ={self.split_k2_layout.MQ_LEN} (K=K2={K2}, pos={K2+1})',
                          flush=True)
                # Step 2: HybridPhase2Plan single-instance allocation (max-size).
                # Skipped in split-only K1/K2 mode (split path does not use
                # HybridPhase2Plan / 5-region scratch / phase2_hybrid CG).
                if SPLIT_K1K2_MODE:
                    self.hybrid_phase2_plan = None
                    print('[MESA split-K1K2] hybrid_phase2_plan / 5-region scratch: SKIPPED', flush=True)
                else:
                    from ssd.engine.helpers.hybrid_phase2_plan import HybridPhase2Plan
                    _max_pages = self.config.max_blocks
                    _max_total = (K_long + 1) * (draft_fo + proxy_fo)
                    _max_mask_size = _max_total * self.config.max_model_len
                    _MQ_p1_max = (K_long + 1) * draft_fo
                    _MQ_proxy_max = (K_long + 1) * proxy_fo
                    _max_L = (
                        self.config.max_model_len
                        + K_long
                        + K_long * _MQ_p1_max
                        + K2 * _MQ_proxy_max
                    )
                    self.hybrid_phase2_plan = HybridPhase2Plan.create(
                        K1=K1, K2=K2,
                        mesa_draft_fan_out=draft_fo,
                        mesa_proxy_fan_out=proxy_fo,
                        max_pages_per_row=_max_pages,
                        max_packed_mask_size=_max_mask_size,
                        max_L=_max_L,
                        device=d,
                    )
                    print(f'[MESA hybrid] HybridPhase2Plan allocated: K_long={K_long}, '
                          f'K_short={K2}, max_total_rows={_max_total}, '
                          f'max_pages={_max_pages}', flush=True)

                    # 5-region scratch slot layout metadata (hybrid path only).
                    MQ_p1 = self.phase1_layout_long.MQ_LEN
                    MQ_proxy_long_max = proxy_fo * (K_long + 1)
                    self.hybrid_glue_slot_count = K_long + 1
                    self.hybrid_phase1_slot_base = self.hybrid_glue_slot_count
                    self.hybrid_phase1_slot_count = K1 * MQ_p1
                    self.hybrid_a_tail_slot_base = (
                        self.hybrid_phase1_slot_base + self.hybrid_phase1_slot_count
                    )
                    self.hybrid_a_tail_slot_count = K2 * MQ_p1
                    self.hybrid_b_proxy_slot_base = (
                        self.hybrid_a_tail_slot_base + self.hybrid_a_tail_slot_count
                    )
                    self.hybrid_b_proxy_slot_count = K2 * MQ_proxy_long_max
                    _hybrid_required_slots = (
                        self.hybrid_glue_slot_count
                        + self.hybrid_phase1_slot_count
                        + self.hybrid_a_tail_slot_count
                        + self.hybrid_b_proxy_slot_count
                    )
                    from ssd.utils.async_helpers.async_spec_helpers import (
                        compute_megaspec_lookahead as _cmsl,
                    )
                    _reserved_slots = _cmsl(self.full_layout.MQ_LEN, K_long)
                    assert _hybrid_required_slots <= _reserved_slots, (
                        f"Hybrid 5-region scratch needs {_hybrid_required_slots} slots "
                        f"but scheduler reserves only {_reserved_slots}."
                    )
                    print(f'[MESA hybrid] 5-region scratch layout: '
                          f'glue={self.hybrid_glue_slot_count} '
                          f'phase1_kv={self.hybrid_phase1_slot_count} '
                          f'a_tail={self.hybrid_a_tail_slot_count} '
                          f'b_proxy={self.hybrid_b_proxy_slot_count} '
                          f'(required={_hybrid_required_slots}/reserved={_reserved_slots})',
                          flush=True)
            print(f'[MESA] TreeLayouts: full MQ_LEN={self.full_layout.MQ_LEN}, '
                  f'draft MQ_LEN={self.draft_layout.MQ_LEN}, '
                  f'proxy MQ_LEN={self.proxy_layout.MQ_LEN}', flush=True)

        # #D Pre-allocate spec buffers for _decode_tree (per-step 8 MB alloc/zero-fill 제거).
        # Rev1 invariant: sum(fan_out_list) 항상 고정 (proxy_fan_out × (K+1)).
        # 따라서 runtime proxy layout의 MQ_LEN ≡ self.proxy_layout.MQ_LEN (static).
        # B=1이므로 N = max_mq. EAGLE은 MESA와 함께 안 쓰임 (config assert).
        #
        # MESA는 Phase 1, 2 결과를 동시에 보관해야 merge 가능 → **슬롯 2개** 준비.
        # Baseline은 슬롯 1개만 사용.
        mq_list = [self.full_layout.MQ_LEN]
        if self.config.mesa_enabled:
            mq_list.extend([self.draft_layout.MQ_LEN, self.proxy_layout.MQ_LEN])
        # Include split-K1/K2 layouts — non-uniform Phase 1 list can have
        # sum(_p1_list) > uniform draft_layout.MQ_LEN, undersizing spec
        # buffers. Same for split_k1_short / split_k2.
        for _layout_attr in ("split_k1_long_layout", "split_k1_short_layout",
                              "split_k2_layout"):
            _layout = getattr(self, _layout_attr, None)
            if _layout is not None:
                mq_list.append(_layout.MQ_LEN)
        max_mq = max(mq_list)
        max_N = self.config.max_num_seqs * max_mq
        V = self.hf_config.vocab_size
        H = self.hf_config.hidden_size
        dt = self.hf_config.torch_dtype

        n_slots = 2 if self.config.mesa_enabled else 1
        self._spec_tokens_bufs = [
            torch.empty((max_N, K), dtype=torch.int64, device=d) for _ in range(n_slots)
        ]
        self._spec_logits_bufs = [
            torch.empty((max_N, K, V), dtype=dt, device=d) for _ in range(n_slots)
        ]
        self._spec_activations_bufs = (
            [torch.empty((max_N, K, H), dtype=dt, device=d) for _ in range(n_slots)]
            if self.config.use_eagle else [None] * n_slots
        )
        self._spec_buf_counter = 0  # round-robin index (Phase 1 = 0, Phase 2 = 1)

    def jit_speculate(self, 
                      request_keys: torch.Tensor, 
                      num_tokens: torch.Tensor, 
                      out_logits: torch.Tensor, 
                      out_tokens: torch.Tensor, 
                      temperatures: torch.Tensor, 
                      draft_block_tables: torch.Tensor,
                      target_recovery_activations: torch.Tensor = None):
        
        input_ids = request_keys[:, -1]
        pos_offset = -1 if self.config.use_eagle else 0
        positions = num_tokens - 1 + pos_offset # want to write rec token at post N-1 since [0, ..., N-2] filled by prefill 
        context_lens = num_tokens + pos_offset # N+1
        # Calculate slot mapping vectorized
        block_idx = positions // self.block_size
        pos_in_block = positions % self.block_size
        batch_indices = torch.arange(input_ids.shape[0], device=self.device)
        slot_map = draft_block_tables[batch_indices, block_idx] * self.block_size + pos_in_block

        hidden_states = None
        spec_activations = None
        
        if self.config.use_eagle:
            assert target_recovery_activations is not None
            hidden_states = self.model.fc(target_recovery_activations.to(self.model.fc.weight.dtype))
            spec_activations = torch.empty(
                input_ids.shape[0], self.config.speculate_k,
                self.hf_config.hidden_size,
                dtype=self.hf_config.torch_dtype, device=self.device)

        # Split-only K1/K2 mode: JIT only K_max forwards (= K1) per miss step.
        # The remaining K_long-K_max positions of out_tokens stay at random
        # init — speculator slices to valid_k+1=K_max+1 anyway, dropping them.
        # This saves K2 wasted forwards per miss step (K_max=K1, K_long=K1+K2).
        _jit_K = self.config.speculate_k
        if SPLIT_K1K2_MODE and self.config.mesa_phase1_k is not None:
            _K1c = self.config.mesa_phase1_k
            _K2c = self.config.mesa_phase2_k
            _jit_K = _K1c if _K1c >= _K2c else _K2c  # = K_max
        for i in range(_jit_K): # we're going to glue after this anyways, and by sending the spec request target has verified we have K more slots left in our last page
            set_context(
                is_prefill=False,
                slot_mapping=slot_map,
                context_lens=context_lens.to(torch.int32),
                block_tables=draft_block_tables,
                is_jit=True,
            )
            
            if self.config.use_eagle:
                logits, prenorm = self.run_model(input_ids, positions, is_prefill=False, last_only=True, hidden_states=hidden_states)
                spec_activations[:, i] = prenorm
                hidden_states = prenorm
            else:
                logits = self.run_model(input_ids, positions, is_prefill=False, last_only=True)
            
            out_logits[:, i, :] = logits
            reset_context()
            next_tokens = self.sampler(logits, temperatures, is_tree=True)
            out_tokens[:, i] = next_tokens
            
            # Update for next iteration
            input_ids = next_tokens
            positions = positions + 1
            context_lens = context_lens + 1
            # Update slot mapping for next position
            block_idx = positions // self.block_size
            pos_in_block = positions % self.block_size
            slot_map = draft_block_tables[batch_indices, block_idx] * self.block_size + pos_in_block

        return spec_activations

    def hit_cache_and_respond(self, request_keys, B, K, num_tokens, temperatures, draft_block_tables, target_recovery_activations=None):
        """Hits the cache (tensor-backed) and returns tensors to respond to the spec request."""
        global ttl, ttl_hit
        # Draft model now returns full target vocab size logits (after d2t expansion)
        V = self.hf_config.vocab_size

        # Init miss slots with valid random logits so token IDs are in-vocab (fixes B>1 crash)
        out_logits = torch.empty((B, K, V), dtype=self.hf_config.torch_dtype, device=self.device).uniform_()
        out_tokens = out_logits.argmax(dim=-1)
        cache_hits = torch.zeros(B, dtype=torch.int64, device=self.device)
        # Per-row valid_k: defaults to K (= K_long for MESA / speculate_k for non-MESA).
        # Phase 4 will override per-row to K_short for proxy-sourced hits.
        # Phase 5 will set miss/JIT path to K_short as well.
        # Split-only K1/K2 mode (per docs/mesa/04-split-k1k2-design.md):
        # default valid_k = K_max = max(K1, K2). Cache-hit overwrites with the
        # row's own valid_k (∈ {K1, K2}). NEVER use K_long here in this mode.
        if (
            SPLIT_K1K2_MODE
            and self.config.mesa_phase1_k is not None
        ):
            K1 = self.config.mesa_phase1_k
            K2 = self.config.mesa_phase2_k
            K_max = K1 if K1 >= K2 else K2
            valid_k = torch.full((B,), K_max, dtype=torch.int64, device=self.device)
        else:
            valid_k = torch.full((B,), K, dtype=torch.int64, device=self.device)

        assert request_keys.shape == (B, 3), f"ERROR in hit_cache_and_respond: request_keys should be (B, 3), got {request_keys.shape}"
        
        hidden_size = self.hf_config.hidden_size
        out_activations = torch.zeros(
            B, K, hidden_size,
            dtype=self.hf_config.torch_dtype, device=self.device
        ) if self.config.use_eagle else None
        
        # Statistics
        ttl += int(B)
        
        if self.config.verbose:
            print(f"[hit_cache_and_respond] Request keys: {request_keys}", flush=True)
            for i in range(B):
                rec_token = request_keys[i, 2].item()
                rec_text = self.tokenizer.decode([rec_token])
                print(f"  Req {i}: token={rec_token} ('{rec_text}')", flush=True)
        
        # MESA-only: per-seq phase classification (0=miss, 1=phase 1 draft, 2=phase 2 proxy).
        # All zeros for non-MESA — verifier silently accumulates 0s and reports 0 phase rates.
        phase_source = torch.zeros(B, dtype=torch.int64, device=self.device)
        if self.tree_cache_keys.numel() > 0:
            # Vectorized membership against tensor cache
            eq = (request_keys.unsqueeze(1) == self.tree_cache_keys.unsqueeze(0))  # [B,T,3]
            match = torch.all(eq, dim=2)  # [B,T]
            cache_hits = match.any(dim=1)  # [B]
            ttl_hit += int(cache_hits.sum().item())

            if self.config.mesa_enabled and self._last_n_draft_keys > 0:
                _hit_idx = match.float().argmax(dim=1).to(torch.int64)
                _is_phase1 = cache_hits & (_hit_idx < self._last_n_draft_keys)
                phase_source = torch.where(_is_phase1, torch.full_like(phase_source, 1),
                                            torch.where(cache_hits.bool(),
                                                        torch.full_like(phase_source, 2),
                                                        phase_source))
                # Pull per-row valid_k from cache. v1 hybrid: heterogeneous
                # K_long (draft rows) / K_short (proxy rows). Speculator
                # slices tokens / logits to vk; verifier dispatches the
                # matching verify_long / verify_short bucket. Always honored
                # (no longer gated) — gating it caused a correctness bug
                # where speculator extended seq by padded K_long-K_short
                # zero tokens for proxy hits.
                if self.tree_cache_valid_k is not None:
                    _hit_valid_k = self.tree_cache_valid_k[_hit_idx]
                    valid_k = torch.where(cache_hits.bool(), _hit_valid_k, valid_k)
            
            if self.config.verbose:
                print(f"[hit_cache_and_respond] Cache hits: {cache_hits.sum().item()}/{B}", flush=True)
                print(f"[hit_cache_and_respond] Cache: {self.tree_cache_keys.shape[0]} entries", flush=True)
                
                # Build set of hit cache indices for marking
                hit_indices = set()
                if cache_hits.any():
                    idx = match.float().argmax(dim=1).to(torch.int64)
                    for i in range(B):
                        if cache_hits[i]:
                            hit_indices.add(idx[i].item())
                
                # Print cache entries with hit markers
                for i, key in enumerate(self.tree_cache_keys):
                    seq_id, k_idx, rec_token = key.tolist()
                    rec_text = self.tokenizer.decode([rec_token])
                    hit_marker = "[HIT]" if i in hit_indices else ""
                    print(f"    [{i}]: key=({seq_id}, {k_idx}, {rec_token}) -> value=('{rec_text}') {hit_marker}", flush=True)
            
            # Fill hits
            if (cache_hits.any() and not self.config.jit_speculate) or (cache_hits.all() and self.config.jit_speculate):
                # print(f'[hit_cache_and_respond] got all cache hits, using cached logits and tokens', flush=True)
                # [B], arbitrary if no match but masked out
                idx = match.float().argmax(dim=1).to(torch.int64)
                sel = cache_hits
                # Cache row width may be < out_tokens K (split-only K1/K2 mode
                # pads to K_max < K_long). Copy first cache_row_width cols, leave
                # the rest zero — valid_k tells consumers the meaningful prefix.
                _cache_w = self.tree_cache_tokens.shape[1]
                # tokens [T,K]
                out_tokens[sel, :_cache_w] = self.tree_cache_tokens[idx[sel]]
                # logits [T,K+1,V]
                out_logits[sel, :_cache_w] = self.tree_cache_logits[idx[sel]]
                if self.config.use_eagle:
                    out_activations[sel, :_cache_w] = self.tree_cache_activations[idx[sel]]
            elif self.config.jit_speculate: 
                # print(f'[hit_cache_and_respond] found a cache miss, running jit speculate', flush=True)
                if self.config.verbose:
                    print(f"[hit_cache_and_respond] Running JIT speculate for cache misses", flush=True)
                jit_acts = self.jit_speculate(
                    request_keys, 
                    num_tokens, 
                    out_logits, 
                    out_tokens,
                    temperatures,
                    draft_block_tables,
                    target_recovery_activations
                    ) # write into out_logits, out_tokens
                if self.config.use_eagle:
                    out_activations = jit_acts
        elif self.config.jit_speculate:
            # Cache is empty (first iteration), must JIT all
            if self.config.verbose:
                print(f"[hit_cache_and_respond] Cache empty, running JIT speculate for all", flush=True)
            jit_acts = self.jit_speculate(
                request_keys, 
                num_tokens, 
                out_logits, 
                out_tokens,
                temperatures,
                draft_block_tables,
                target_recovery_activations
                )
            if self.config.use_eagle:
                out_activations = jit_acts
            
        rec_toks = request_keys[:, 2]

        # Step 9B-0: glue input width follows valid_k (long-hit=K_long+1,
        # short-hit=K_short+1). B=1 invariant lets us pass a scalar.
        # Sync valid_k to Python ONLY when hybrid mode actually needs the
        # per-step bucket dispatch. Split / pure-MESA: width is always
        # K_long; pass None so make_glue_decode_input_ids defaults to it
        # without forcing a CUDA stream flush. Returned as the 8th element
        # so downstream callers (_glue_decode, _decode_speculate_async)
        # avoid re-syncing the same value.
        if self.config.mesa_phase1_k is not None and valid_k.numel() > 0:
            _vk_scalar = int(valid_k[0].item())
        else:
            _vk_scalar = None
        # ===== TRACE point 1: hit_cache_and_respond boundary =====
        if TRACE_SPLIT_K1K2 and SPLIT_K1K2_MODE and self.config.mesa_phase1_k is not None:
            K1 = self.config.mesa_phase1_k
            K2 = self.config.mesa_phase2_k
            K_max = K1 if K1 >= K2 else K2
            _vk_list = valid_k.tolist() if valid_k is not None else None
            _hits_list = cache_hits.tolist()
            _cache_w = self.tree_cache_tokens.shape[1] if self.tree_cache_tokens is not None and self.tree_cache_tokens.numel() > 0 else 0
            _cache_n = self.tree_cache_keys.shape[0] if self.tree_cache_keys is not None and self.tree_cache_keys.numel() > 0 else 0
            print(f"[TRACE-split-k1k2 #1 hit_cache] K1={K1} K2={K2} K_max={K_max} "
                  f"cache_hits={_hits_list} valid_k={_vk_list} _vk_scalar={_vk_scalar} "
                  f"out_tokens.shape={list(out_tokens.shape)} "
                  f"cache_n_rows={_cache_n} cache_row_width={_cache_w}",
                  flush=True)
        # ===== END TRACE =====
        glue_input_ids = make_glue_decode_input_ids(out_tokens, rec_toks, valid_k=_vk_scalar)
        return out_tokens, out_logits, glue_input_ids, cache_hits, out_activations, phase_source, valid_k, _vk_scalar

    def _service_spec_request(self):
        """Receives a speculation request, serves it from cache, and sends results back in a single response."""
        meta = self.recv_tensor((3,), torch.int64)
        B, K, F = meta.tolist()

        # Receive all request payload in one fused int64 burst (includes temperatures encoded as int64)
        max_blocks = self.config.max_blocks
        fused_total = (3 * B) + B + (B * max_blocks) + B  # +B for temps_as_int64
        fused_req = recv_int64(self.async_pg, src=0,
                               total_length=fused_total, device=self.device)
        off = 0
        cache_keys = fused_req[off:off + (3 * B)].view(B, 3)
        off += 3 * B
        seq_ids = cache_keys[:, 0]
        num_tokens = fused_req[off:off + B].to(torch.int64)
        off += B
        draft_block_tables = fused_req[off:off + B *
                                       max_blocks].view(B, max_blocks).to(torch.int32)
        off += B * max_blocks
        temps_as_int64 = fused_req[off:off + B]
        off += B
        assert off == fused_total
        temperatures = temps_as_int64.to(torch.int32).view(torch.float32)

        target_recovery_activations = torch.zeros(
            B, 3 * self.config.d_model_target, dtype=self.hf_config.torch_dtype, device=self.device
        ) if self.config.use_eagle else None

        extend_counts = None
        extend_eagle_acts = None
        extend_token_ids = None

        if self.config.use_eagle:
            dist.recv(target_recovery_activations, src=0, group=self.async_pg)

            # Receive extend data for fused glue decode
            act_dim = 3 * self.config.d_model_target
            extend_counts = torch.zeros(B, dtype=torch.int64, device=self.device)
            extend_eagle_acts = torch.zeros(B, K, act_dim, dtype=self.hf_config.torch_dtype, device=self.device)
            extend_token_ids = torch.zeros(B, K, dtype=torch.int64, device=self.device)
            dist.recv(extend_counts, src=0, group=self.async_pg)
            dist.recv(extend_eagle_acts, src=0, group=self.async_pg)
            dist.recv(extend_token_ids, src=0, group=self.async_pg)

            if self.config.verbose:
                recovery_tokens_target = cache_keys[:, 2].clone()
                print(f"\n{'='*80}", flush=True)
                print(f"[CACHE REQUEST] Batch size: {B}, Spec depth: {K}", flush=True)
                for i in range(B):
                    seq_id = cache_keys[i, 0].item()
                    keep_idx = cache_keys[i, 1].item()
                    rec_token_target = recovery_tokens_target[i].item()
                    rec_token_text = self.tokenizer.decode([rec_token_target])
                    n_ext = extend_counts[i].item()
                    print(f"  Seq {seq_id}: keep_idx={keep_idx}, recovery_token={rec_token_target} ('{rec_token_text}'), n_ext={n_ext}", flush=True)
                print(f"{'='*80}\n", flush=True)

        from ssd.engine.helpers.cudagraph_helpers import mesa_record as _mr_h, mesa_close as _mc_h
        _mev_hc = _mr_h("hit_cache_respond")
        out_tokens, out_logits, glue_decode_input_ids, cache_hits, out_activations, phase_source, valid_k, valid_k_scalar = self.hit_cache_and_respond(
            cache_keys, B, K, num_tokens, temperatures, draft_block_tables, target_recovery_activations)
        _mc_h("hit_cache_respond", _mev_hc)

        if self.config.verbose:
            print(f"[CACHE RESPONSE]", flush=True)
            for i in range(B):
                hit_status = "HIT" if cache_hits[i].item() == 1 else "MISS"
                print(f"  Seq {cache_keys[i, 0].item()}: {hit_status}", flush=True)
                if cache_hits[i].item() == 1 or self.config.jit_speculate:
                    tokens_list = out_tokens[i, :K].tolist()
                    tokens_text = [self.tokenizer.decode([t]) for t in tokens_list]
                    print(f"    Tokens: {tokens_list}", flush=True)
                    print(f"    Detokenized: {tokens_text}", flush=True)
            print(f"", flush=True)

        # Wire layout matches speculator_async._fused_response: [cache_hits, phase_source, valid_k, out_tokens].
        fused_response = torch.cat([cache_hits.reshape(-1).to(torch.int64),
                                    phase_source.reshape(-1),
                                    valid_k.reshape(-1).to(torch.int64),
                                    out_tokens.reshape(-1).to(torch.int64)])
        from ssd.engine.helpers.cudagraph_helpers import mesa_record as _mr_s, mesa_close as _mc_s
        _mev_ds = _mr_s("draft_send_response")
        dist.send(fused_response, dst=0, group=self.async_pg)
        dist.send(out_logits[:, :K, :].contiguous(), dst=0, group=self.async_pg)
        _mc_s("draft_send_response", _mev_ds)

        partial_tree_decode_args = {
            "num_tokens": num_tokens,
            "seq_ids": seq_ids,
            "temperatures": temperatures,
            "dbt": draft_block_tables,
            "cache_hits": cache_hits,
            "returned_tokens": out_tokens,
            "target_recovery_activations": target_recovery_activations,
            "previous_activations": out_activations,
            "extend_counts": extend_counts,
            "extend_eagle_acts": extend_eagle_acts,
            "extend_token_ids": extend_token_ids,
            # Step 9A: per-row valid_k from previous step's matched cache row
            # (or K_long fallback for misses). Drives glue / phase1 / proxy /
            # hybrid layout dispatch in _build_tree_batch_mesa.
            "valid_k": valid_k,
            # Pre-sync'd Python int (hybrid mode) or None (split / pure-MESA).
            # Lets _glue_decode and the main hot path skip redundant .item()
            # syncs — the .item() that produced this scalar happened once
            # inside hit_cache_and_respond above.
            "valid_k_scalar": valid_k_scalar,
        }

        return glue_decode_input_ids, partial_tree_decode_args

    def prepare_prefill_ctxt(
        self,
        num_tokens: torch.Tensor,  # [B]
        draft_block_table: torch.Tensor,  # [B, max_blocks]
    ) -> dict:
        """
        Prepare context for prefill forward pass.
        """
        B = num_tokens.shape[0]
        total = num_tokens.sum().item()
        cu_seqlens_q = torch.zeros(B + 1, dtype=torch.int32, device=self.device)
        cu_seqlens_q[1:] = torch.cumsum(num_tokens, dim=0)
        batch_indices = torch.arange(B, device=self.device, dtype=torch.int64).repeat_interleave(num_tokens)
        positions = torch.arange(total, device=self.device, dtype=torch.int64) - cu_seqlens_q[:-1].to(torch.int64).repeat_interleave(num_tokens)
        max_seqlen_q = num_tokens.max().item()

        # Calculate block indices and offsets for ALL positions
        block_indices = (positions // self.block_size).to(torch.int64)
        offsets = (positions % self.block_size).to(torch.int32)

        # Get block IDs for each position from dbt
        block_ids = draft_block_table[batch_indices, block_indices]

        # Calculate slot_map for each position
        slot_map = (block_ids * self.block_size + offsets).to(torch.int32)

        return {
            "positions": positions,
            "slot_map": slot_map,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_q.clone(),
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_q,
        }

    
    def prepare_glue_decode_ctxt(self, num_tokens, input_ids, dbt, B, valid_k=None):
        """Prepare glue decode context.

        Step 9B-0: ``valid_k`` (default = config.speculate_k = K_long) drives
        glue width. positions / slot_map / context_lens / cu_seqlens_q /
        max_seqlen_q all follow valid_k+1.
        """
        K = valid_k if valid_k is not None else self.config.speculate_k
        pos_offset = -1 if self.config.use_eagle else 0
        positions_start = (num_tokens - 1 + pos_offset).unsqueeze(-1)
        # arange(K+1) — sliced from cached _arange_kp1 (which is K_long+1 wide)
        if valid_k is not None and valid_k != self.config.speculate_k:
            arange_kp1 = self._arange_kp1[:K + 1]
        else:
            arange_kp1 = self._arange_kp1
        positions_grid = positions_start + arange_kp1

        # Calculate block indices and offsets for ALL positions
        block_indices = (positions_grid // self.block_size).to(torch.int64)
        offsets = (positions_grid % self.block_size).to(torch.int32)

        # Get block IDs for each position from dbt
        B_expanded = torch.arange(B, device=self.device).unsqueeze(-1).expand(-1, K + 1)
        blk_ids = dbt[B_expanded, block_indices]

        # Calculate slot_map for each position
        slot_map_grid = blk_ids * self.block_size + offsets

        # Flattened tensors for varlen decode
        positions_flat = positions_grid.reshape(-1).to(torch.int64)
        slot_map_flat = slot_map_grid.reshape(-1).to(torch.int32)

        context_lens = (num_tokens + pos_offset + K).to(torch.int32)
        seqlen_q = torch.full((B,), K + 1, dtype=torch.int32, device=self.device)
        cu_seqlens_q = torch.zeros(B + 1, dtype=torch.int32, device=self.device)
        cu_seqlens_q[1:] = torch.cumsum(seqlen_q, dim=0)

        return {
            "input_ids": input_ids,
            "positions": positions_flat,
            "slot_map": slot_map_flat,
            "cu_seqlens_q": cu_seqlens_q,
            "max_seqlen_q": K + 1,
            "context_lens": context_lens,
            "block_tables": dbt,
        }

    def prepare_glue_decode_ctxt_eagle(self, num_tokens, fused_ids, fused_hs, extend_counts, seqlens_q, cu_seqlens_q, dbt, B):
        """Prepare context for EAGLE glue decode with FA varlen causal.

        Tokens packed contiguously: [ext_0..ext_{n0-1}, rec_0, spec_0..spec_{K-1}, ext_1..., ...]
        No padding within sequences. cu_seqlens_q has variable per-seq lengths.
        """
        K = self.config.speculate_k
        total_real = int(cu_seqlens_q[-1].item())

        # Per-token batch index and local offset within each seq
        batch_idx = torch.repeat_interleave(torch.arange(B, device=self.device), seqlens_q)  # [total_real]
        local_off = torch.arange(total_real, device=self.device) - cu_seqlens_q[:-1].long().repeat_interleave(seqlens_q)

        # Positions: extend starts at num_tokens-2-n_ext, then rec, then spec
        # base_pos[b] = num_tokens[b] - 2 - extend_counts[b] (position of first extend token)
        base_pos = (num_tokens - 2 - extend_counts).long()  # [B]
        positions = (base_pos[batch_idx] + local_off).to(torch.int64)

        # Context lens: last token (spec K-1) at pos num_tokens-2+K, cache has 0..num_tokens-2+K
        context_lens = (num_tokens - 1 + K).to(torch.int32)

        # Slot mapping
        block_idx = (positions // self.block_size).clamp(0, dbt.shape[1] - 1).to(torch.int64)
        block_off = (positions % self.block_size).to(torch.int32)
        blk_ids = dbt[batch_idx, block_idx]
        slot_map = (blk_ids * self.block_size + block_off).to(torch.int32)

        return {
            "input_ids": fused_ids,
            "positions": positions,
            "slot_map": slot_map,
            "hidden_states": fused_hs,
            "cu_seqlens_q": cu_seqlens_q,
            "max_seqlen_q": 2 * K + 1,
            "context_lens": context_lens,
            "block_tables": dbt,
        }

    def _construct_tree_decode_args(self, partial_tree_decode_args, rec_flat, dbt):
        # tree decode needs (input_ids, positions) that are [N], wrapper plan handles batch size of attn computation 
        # rec_flat is [N]
        
        B = dbt.shape[0]
        K = self.config.speculate_k
        F = self.config.async_fan_out
        N = rec_flat.shape[0]
        cache_hits = partial_tree_decode_args["cache_hits"]

        _layout = self.full_layout  # _construct_tree_decode_args is only used in non-MESA path
        if __debug__:
            assert N == B*_layout.MQ_LEN, f"ERROR in _construct_tree_decode_args: N should be B*MQ_LEN={B*_layout.MQ_LEN}, got {N}"

        b_flat = torch.arange(B, device=self.device, dtype=torch.int64)[:, None].expand(B, _layout.MQ_LEN).flatten()
        fkp1_flat = _layout.arange_mq.repeat(B)
        j_idx_flat = torch.cat([_layout.fan_idx_hit if hit else _layout.fan_idx_miss for hit in cache_hits])
        metadata = torch.tensor([B, K, F, N], dtype=torch.int64, device=self.device)

        seq_ids = partial_tree_decode_args["seq_ids"]
        seq_ids_expanded = seq_ids[b_flat]
        pos_offset = -1 if self.config.use_eagle else 0
        positions = (partial_tree_decode_args["num_tokens"][b_flat] - 1 + pos_offset) + (K + 1) + fkp1_flat
        rope_positions = (partial_tree_decode_args["num_tokens"][b_flat] - 1 + pos_offset) + j_idx_flat + 1
        temperatures = partial_tree_decode_args["temperatures"][b_flat]

        tree_decode_args = {
            "metadata": metadata,
            "input_ids": rec_flat,  # [N]
            "positions": positions,  # [N]
            "rope_positions": rope_positions, # [N], these are to be passed into model fwd 
            # the dbt is now [B, M] in the seq fan out codebase
            "block_tables": dbt,
            "temps": temperatures,  # [N]
            "rec_flat": rec_flat,  # [N]
            "seq_ids_expanded": seq_ids_expanded,  # [N]
            "cache_hits": cache_hits,  # [B] # we also want returned_tokens which is [B, K]
        }

        return tree_decode_args

    def _glue_decode(self, partial_tree_decode_args, glue_decode_input_ids):
        """Glue decode only (no tree args construction). Non-EAGLE scope.

        Step 9B-0: dispatches on valid_k (long-hit vs short-hit). Glue width
        = valid_k+1. Long-hit uses glue_long context / CG, short-hit uses
        glue_short.

        Returns: (glue_logits [B, valid_k+1, V], gd_for_fork [B, valid_k+1],
                  cache_hits, cache_hits_list, dbt, B)
        """
        from ssd.engine.helpers.cudagraph_helpers import mesa_record, mesa_close
        _mev_glue = mesa_record("glue")
        dbt = partial_tree_decode_args["dbt"]
        cache_hits = partial_tree_decode_args["cache_hits"]
        cache_hits_list = cache_hits.tolist()

        if self.config.use_eagle:
            # EAGLE glue decode — full path (unchanged, passes through to _build_tree_batch)
            raise NotImplementedError("_glue_decode does not support EAGLE; use _build_tree_batch directly")

        # Step 9B-0: glue width follows valid_k. Read the pre-sync'd Python
        # scalar that hit_cache_and_respond stashed in partial_tree_decode_args
        # — None means split / pure-MESA path, default to K_long. No new
        # GPU→CPU sync here.
        _vk_glue = partial_tree_decode_args.get("valid_k_scalar")
        if _vk_glue is None:
            # Split-only K1/K2 mode: default = K_max, NOT K_long.
            if SPLIT_K1K2_MODE and self.config.mesa_phase1_k is not None:
                K1 = self.config.mesa_phase1_k
                K2 = self.config.mesa_phase2_k
                _vk_glue = K1 if K1 >= K2 else K2
            else:
                _vk_glue = self.config.speculate_k
        K = _vk_glue
        B = glue_decode_input_ids.shape[0] // (K + 1)
        assert B == partial_tree_decode_args["num_tokens"].shape[0]
        glue_decode_ctxt = self.prepare_glue_decode_ctxt(
            num_tokens=partial_tree_decode_args["num_tokens"],
            input_ids=glue_decode_input_ids,
            dbt=dbt, B=B, valid_k=_vk_glue,
        )

        set_context(
            is_prefill=False,
            cu_seqlens_q=glue_decode_ctxt["cu_seqlens_q"],
            max_seqlen_q=glue_decode_ctxt["max_seqlen_q"],
            slot_mapping=glue_decode_ctxt["slot_map"],
            context_lens=glue_decode_ctxt["context_lens"],
            block_tables=glue_decode_ctxt["block_tables"],
            # Step 9B-0: glue bucket dispatch — model_runner reads
            # context.glue_valid_k to pick glue_long vs glue_short CG.
            glue_valid_k=_vk_glue,
        )

        glue_decode_logits_flat = self.run_model(
            glue_decode_ctxt["input_ids"], glue_decode_ctxt["positions"],
            is_prefill=False, last_only=False)

        reset_context()

        glue_decode_logits = glue_decode_logits_flat.view(B, K + 1, -1)
        gd_for_fork = glue_decode_input_ids.reshape(B, K + 1)

        mesa_close("glue", _mev_glue)
        return glue_decode_logits, gd_for_fork, cache_hits, cache_hits_list, dbt, B

    def _build_tree_batch(self, partial_tree_decode_args, glue_decode_input_ids):
        if self.config.verbose:
            print(f'about to build tree batch')
        K = self.config.speculate_k
        dbt = partial_tree_decode_args["dbt"]
        cache_hits = partial_tree_decode_args["cache_hits"]
        cache_hits_list = cache_hits.tolist()
        pos_offset = -1 if self.config.use_eagle else 0

        if self.config.use_eagle:
            B = partial_tree_decode_args["num_tokens"].shape[0]
            extend_counts = partial_tree_decode_args.get("extend_counts")
            if extend_counts is None:
                extend_counts = torch.zeros(B, dtype=torch.int64, device=self.device)
            extend_eagle_acts_batch = partial_tree_decode_args.get("extend_eagle_acts")
            extend_token_ids_batch = partial_tree_decode_args.get("extend_token_ids")
            target_acts = partial_tree_decode_args["target_recovery_activations"]
            prev_acts = partial_tree_decode_args["previous_activations"]
            hidden_size = self.hf_config.hidden_size
            fc_dtype = self.model.fc.weight.dtype

            gd_view = glue_decode_input_ids.view(B, K + 1)
            rec_tok_ids = gd_view[:, 0]
            spec_tok_ids = gd_view[:, 1:]

            # Variable per-seq lengths: n_ext[b] + K + 1
            seqlens_q = (extend_counts + K + 1).to(torch.int32)
            cu_seqlens_q = torch.zeros(B + 1, dtype=torch.int32, device=self.device)
            cu_seqlens_q[1:] = torch.cumsum(seqlens_q, 0)
            total_real = int(cu_seqlens_q[-1].item())

            # Build packed fused_ids and fused_hs (no padding, no for loops)
            fused_ids = torch.zeros(total_real, dtype=torch.int64, device=self.device)
            fused_hs = torch.zeros(total_real, hidden_size, dtype=self.hf_config.torch_dtype, device=self.device)

            # Per-token batch index and local offset
            batch_idx = torch.repeat_interleave(torch.arange(B, device=self.device), seqlens_q)
            local_off = torch.arange(total_real, device=self.device) - cu_seqlens_q[:-1].long().repeat_interleave(seqlens_q)
            n_ext = extend_counts.long()  # [B]
            n_ext_per_tok = n_ext[batch_idx]  # [total_real]

            # Classify each token: extend (local < n_ext), rec (local == n_ext), spec (local > n_ext)
            is_extend = local_off < n_ext_per_tok
            is_rec = local_off == n_ext_per_tok
            is_spec = local_off > n_ext_per_tok

            # Extend + rec tokens: batch fc into single call
            is_target_conditioned = is_extend | is_rec
            tc_b = batch_idx[is_target_conditioned]
            tc_local = local_off[is_target_conditioned]
            tc_n_ext = n_ext_per_tok[is_target_conditioned]

            # Gather target acts: extend uses extend_eagle_acts_batch[b,j], rec uses target_acts[b]
            tc_is_ext = tc_local < tc_n_ext
            tc_acts = torch.empty(tc_b.size(0), target_acts.size(1), dtype=fc_dtype, device=self.device)
            if tc_is_ext.any() and extend_eagle_acts_batch is not None:
                ext_b = tc_b[tc_is_ext]
                ext_j = tc_local[tc_is_ext]
                tc_acts[tc_is_ext] = extend_eagle_acts_batch[ext_b, ext_j].to(fc_dtype)
                fused_ids[is_extend] = extend_token_ids_batch[ext_b, ext_j]
            tc_acts[~tc_is_ext] = target_acts[tc_b[~tc_is_ext]].to(fc_dtype)
            fused_ids[is_rec] = rec_tok_ids[batch_idx[is_rec]]

            # Single batched fc call
            fused_hs[is_target_conditioned] = self.model.fc(tc_acts)

            # Spec tokens: ids from spec_tok_ids, hs from prev_acts (self-conditioned, no fc)
            spec_j = local_off[is_spec] - n_ext_per_tok[is_spec] - 1  # 0..K-1
            fused_ids[is_spec] = spec_tok_ids[batch_idx[is_spec], spec_j]
            fused_hs[is_spec] = prev_acts[batch_idx[is_spec], spec_j]

            glue_decode_ctxt = self.prepare_glue_decode_ctxt_eagle(
                num_tokens=partial_tree_decode_args["num_tokens"],
                fused_ids=fused_ids, fused_hs=fused_hs,
                extend_counts=extend_counts, seqlens_q=seqlens_q,
                cu_seqlens_q=cu_seqlens_q, dbt=dbt, B=B,
            )
        else:
            # Non-EAGLE: K+1 per seq, uses verify CG path
            B = glue_decode_input_ids.shape[0] // (K + 1)
            assert B == partial_tree_decode_args["num_tokens"].shape[0]
            glue_decode_ctxt = self.prepare_glue_decode_ctxt(
                num_tokens=partial_tree_decode_args["num_tokens"],
                input_ids=glue_decode_input_ids,
                dbt=dbt, B=B,
            )

        # Pre-compute tree decode args (overlap CPU with GPU)
        # Uses full_layout for non-MESA path. MESA path uses _build_tree_batch_mesa() instead.
        _layout = self.full_layout
        _pre_b_flat = torch.arange(B, device=self.device, dtype=torch.int64)[:, None].expand(B, _layout.MQ_LEN).flatten()
        _pre_fkp1_flat = _layout.arange_mq.repeat(B)
        _pre_j_idx_flat = torch.cat([_layout.fan_idx_hit if int(h) else _layout.fan_idx_miss for h in cache_hits_list])
        N_pre = _pre_b_flat.shape[0]
        _pre_metadata_ints = (B, K, self.config.async_fan_out, N_pre)
        _pre_seq_ids_expanded = partial_tree_decode_args["seq_ids"][_pre_b_flat]
        _pre_positions = (partial_tree_decode_args["num_tokens"][_pre_b_flat] - 1 + pos_offset) + (K + 1) + _pre_fkp1_flat
        _pre_rope_positions = (partial_tree_decode_args["num_tokens"][_pre_b_flat] - 1 + pos_offset) + _pre_j_idx_flat + 1
        _pre_temperatures = partial_tree_decode_args["temperatures"][_pre_b_flat]

        # --- Run glue decode forward ---
        from ssd.engine.helpers.cudagraph_helpers import mesa_record as _mr_g, mesa_close as _mc_g
        _mev_gb = _mr_g("glue")
        set_context(
            is_prefill=False,
            cu_seqlens_q=glue_decode_ctxt["cu_seqlens_q"],
            max_seqlen_q=glue_decode_ctxt["max_seqlen_q"],
            slot_mapping=glue_decode_ctxt["slot_map"],
            context_lens=glue_decode_ctxt["context_lens"],
            block_tables=glue_decode_ctxt["block_tables"],
        )

        glue_prenorm = None
        if self.config.use_eagle:
            fused_hs_flat = glue_decode_ctxt["hidden_states"]
            glue_decode_logits_flat, glue_prenorm = self.run_model(
                glue_decode_ctxt["input_ids"], glue_decode_ctxt["positions"],
                is_prefill=False, last_only=False, hidden_states=fused_hs_flat)
        else:
            glue_decode_logits_flat = self.run_model(
                glue_decode_ctxt["input_ids"], glue_decode_ctxt["positions"],
                is_prefill=False, last_only=False)

        reset_context()
        _mc_g("glue", _mev_gb)

        # --- Extract K+1 logits/prenorms at rec+spec positions ---
        if self.config.use_eagle:
            # Packed layout: rec at cu_seqlens_q[b] + n_ext[b], spec follows
            cu_q = glue_decode_ctxt["cu_seqlens_q"]
            rec_offsets = cu_q[:-1].long() + extend_counts.long()  # [B]
            extract_idx = rec_offsets.unsqueeze(1) + self._arange_kp1.unsqueeze(0)  # [B, K+1]
            flat_idx = extract_idx.flatten()
            glue_decode_logits = glue_decode_logits_flat[flat_idx].view(B, K + 1, -1)
            if glue_prenorm is not None:
                glue_prenorm_kp1 = glue_prenorm[flat_idx].view(B, K + 1, -1)
        else:
            glue_decode_logits = glue_decode_logits_flat.view(B, K + 1, -1)
            if glue_prenorm is not None:
                glue_prenorm_kp1 = glue_prenorm.view(B, K + 1, -1)

        # --- Build tree hidden states from K+1 prenorms ---
        tree_hidden_states = None
        if glue_prenorm is not None:
            # Vectorized: for each (b, depth), repeat prenorm by fan_out[depth]
            # fan_out_t[depth] for hits, fan_out_t_miss[depth] for misses
            fan_hit = self.config.fan_out_t  # [K+1]
            fan_miss = self.config.fan_out_t_miss  # [K+1]
            # Per-batch fan_out: [B, K+1]
            per_batch_fan = torch.where(
                cache_hits.bool().unsqueeze(1).expand(B, K + 1),
                fan_hit.unsqueeze(0).expand(B, K + 1),
                fan_miss.unsqueeze(0).expand(B, K + 1),
            )  # [B, K+1]
            reps_flat = per_batch_fan.reshape(-1)  # [B*(K+1)]
            prenorms_flat = glue_prenorm_kp1.reshape(B * (K + 1), -1)  # [B*(K+1), d]
            tree_hidden_states = torch.repeat_interleave(prenorms_flat, reps_flat, dim=0)

        # --- Fork tokens from K+1 logits ---
        # Need [B, K+1] input_ids for forking (rec + spec tokens)
        if self.config.use_eagle:
            gd_for_fork = gd_view  # [B, K+1] already computed above
        else:
            gd_for_fork = glue_decode_input_ids.reshape(B, K + 1)

        forked_rec_tokens = get_forked_recovery_tokens_from_logits(
            self.config,
            glue_decode_logits,
            cache_hits,
            gd_for_fork,
            tokenizer=self.tokenizer,
        ).view(-1)

        tree_decode_args = {
            "metadata_ints": _pre_metadata_ints,
            "input_ids": forked_rec_tokens,
            "positions": _pre_positions,
            "rope_positions": _pre_rope_positions,
            "block_tables": dbt,
            "temps": _pre_temperatures,
            "rec_flat": forked_rec_tokens,
            "seq_ids_expanded": _pre_seq_ids_expanded,
            "cache_hits": cache_hits,
            "cache_hits_list": cache_hits_list,
        }
        tree_decode_args["hidden_states"] = tree_hidden_states
        # MESA: store glue_decode_logits and gd_for_fork for proxy token swap
        if self.config.mesa_enabled:
            tree_decode_args["_mesa_glue_logits"] = glue_decode_logits  # [B, K+1, V]
            tree_decode_args["_mesa_gd_for_fork"] = gd_for_fork          # [B, K+1]
        return tree_decode_args

    @torch.inference_mode()
    def _compute_step_positions_and_slot_maps(self, initial_positions, initial_rope_positions, dbt, B, K, F, N, MQ_LEN, layout=None):
        # PERFORMANCE: pre-allocated _step_pos_offsets/_step_rope_offsets avoid per-step torch.arange calls
        _layout = layout or self.full_layout
        step_positions = initial_positions[None, :] + _layout.step_pos_offsets
        step_rope_positions = initial_rope_positions[None, :] + _layout.step_rope_offsets
        step_context_lens = step_positions.view(K, B, _layout.MQ_LEN)[:, :, -1] + 1

        # Precompute slot_maps for all steps: [K, N]
        b_flat = torch.arange(B, device=self.device, dtype=torch.int64)[
            :, None].expand(B, _layout.MQ_LEN).flatten()
        batch_indices = torch.arange(N, device=self.device)
        dbt_expanded = dbt[b_flat]  # [N, M] - constant across steps

        step_offsets = (step_positions % self.block_size).to(torch.int32)  # [K, N]
        step_last_blks = (step_positions // self.block_size).to(torch.int64)  # [K, N]
        step_blk_ids = dbt_expanded[batch_indices[None, :], step_last_blks]  # [K, N]
        step_slot_maps = step_blk_ids * self.block_size + step_offsets  # [K, N]

        return step_positions, step_rope_positions, step_context_lens, step_slot_maps

    def _decode_tree_step(self, depth, current_input_ids, step_rope_positions, step_slot_maps, step_context_lens, dbt, payload, spec_tokens, spec_logits, spec_activations):
        """Execute a single tree decode step."""
        if self.config.mesa_enabled:
            _layout = payload.get("_active_layout")
            _active_mq = _layout.MQ_LEN if _layout and _layout.name != "full" else None
            _active_wrappers = None
            if _active_mq is not None:
                _active_wrappers = self.prefill_wrappers_by_layout.get(_layout.name)
            set_context(
                is_prefill=False,
                slot_mapping=step_slot_maps[depth],
                context_lens=step_context_lens[depth].to(torch.int32),
                block_tables=dbt,
                active_mq_len=_active_mq,
                active_wrappers=_active_wrappers,
                active_layout=_layout,  # runtime layout for dynamic fan_out
            )
        else:
            set_context(
                is_prefill=False,
                slot_mapping=step_slot_maps[depth],
                context_lens=step_context_lens[depth].to(torch.int32),
                block_tables=dbt,
            )

        hidden_states = payload.get("hidden_states")
        if self.config.use_eagle:
            logits, prenorm = self.run_model(current_input_ids, step_rope_positions[depth], is_prefill=False, last_only=False, tree_decode_step=depth, cache_hits=payload["cache_hits"], hidden_states=hidden_states)
            assert spec_activations is not None
            spec_activations[:, depth] = prenorm
            payload["hidden_states"] = prenorm
        else:
            logits = self.run_model(current_input_ids, step_rope_positions[depth], is_prefill=False, last_only=False, tree_decode_step=depth, cache_hits=payload["cache_hits"])
        
        reset_context()
        
        V = self.hf_config.vocab_size  # Draft returns full target vocab size after d2t expansion
        logits_flat = logits.view(-1, V)  # [N, V]
        spec_logits[:, depth, :] = logits_flat
        # Inline greedy: payload["_all_greedy"] checked once in _decode_tree
        next_tokens = logits_flat.argmax(dim=-1) if payload["_all_greedy"] else self.sampler(logits_flat, payload["temps"], is_tree=True)
        spec_tokens[:, depth] = next_tokens
        
        return next_tokens

    def _decode_tree(self, payload, layout=None):
        """Decodes the speculation tree. layout=None → full_layout (backward compat)."""
        _layout = layout or self.full_layout
        payload["_active_layout"] = _layout  # for _decode_tree_step context

        # setup
        B, K, F, N = payload["metadata_ints"]

        V = self.hf_config.vocab_size
        # #D: slice pre-allocated buffers (no per-step alloc / zero-fill).
        # MESA는 Phase 1/2 결과를 merge까지 보관해야 하므로 슬롯 2개 round-robin.
        # _decode_tree_step fully overwrites spec_tokens[:, depth] and spec_logits[:, depth, :]
        # per iter over all K depths → garbage init OK.
        n_slots = len(self._spec_tokens_bufs)
        slot_id = self._spec_buf_counter % n_slots
        self._spec_buf_counter += 1
        assert N <= self._spec_tokens_bufs[slot_id].shape[0], \
            f"spec buf too small: N={N} > {self._spec_tokens_bufs[slot_id].shape[0]}"
        spec_tokens = self._spec_tokens_bufs[slot_id][:N, :K]
        spec_logits = self._spec_logits_bufs[slot_id][:N, :K, :V]
        spec_activations = (
            self._spec_activations_bufs[slot_id][:N, :K, :]
            if self.config.use_eagle else None
        )

        initial_positions = payload["positions"]
        initial_rope_positions = payload["rope_positions"]
        current_input_ids = payload["input_ids"]
        dbt = payload["block_tables"]

        _, step_rope_positions, step_context_lens, step_slot_maps = self._compute_step_positions_and_slot_maps(
            initial_positions, initial_rope_positions, dbt, B, K, F, N, _layout.MQ_LEN, layout=_layout
        )

        _prof = os.environ.get("SSD_PROFILE", "0") == "1"
        payload["_all_greedy"] = bool((payload["temps"] == 0).all())
        _step_times = []
        for depth in range(K):
            if _prof or PROFILE_DRAFT:
                torch.cuda.synchronize()
                _st = time.perf_counter()
            current_input_ids = self._decode_tree_step(
                depth, current_input_ids, step_rope_positions, step_slot_maps,
                step_context_lens, dbt, payload, spec_tokens, spec_logits, spec_activations
            )
            if _prof or PROFILE_DRAFT:
                torch.cuda.synchronize()
                _et = time.perf_counter()
                _step_times.append((_et - _st) * 1000)
                if _prof:
                    print(f"[PROFILE draft] tree_step[{depth}]={_step_times[-1]:.2f}ms", flush=True)
        if PROFILE_DRAFT and _step_times:
            avg = sum(_step_times) / len(_step_times)
            print(f"[PROFILE draft] tree_decode: K={K} steps={' '.join(f'{t:.2f}' for t in _step_times)} avg={avg:.2f}ms total={sum(_step_times):.2f}ms", flush=True)

        return spec_tokens, spec_logits, spec_activations

    def _populate_tree_cache(self, payload, tokens, logits, cache_hits, activations=None, layout=None):
        """Populates the tensor-backed tree_cache with the results of the decoding.
        layout=None → full_layout (backward compat).
        """
        _layout = layout or self.full_layout
        seq_ids_expanded = payload["seq_ids_expanded"].to(torch.int64)
        rec_flat = payload["rec_flat"].to(torch.int64)

        k_flat = torch.cat([_layout.fan_idx_hit if hit else _layout.fan_idx_miss for hit in payload["cache_hits_list"]])

        assert k_flat.shape[0] == payload["block_tables"].shape[0] * _layout.MQ_LEN, f"ERROR in _populate_tree_cache: k_flat should be {payload['block_tables'].shape[0] * _layout.MQ_LEN}, got {k_flat.shape[0]}"
        
        keys = torch.stack([seq_ids_expanded, k_flat, rec_flat], dim=1).contiguous()  # [N,3]

        assert self.tree_cache_keys.numel() == 0
        self.tree_cache_keys = keys
        self.tree_cache_tokens = tokens
        self.tree_cache_logits = logits
        self.tree_cache_activations = activations
        # Phase 1 plumbing: every row in the non-MESA / legacy populate path is
        # treated as full-K (= K_long for MESA, speculate_k otherwise). Phase 4
        # of hybrid will distinguish K_long vs K_short per row.
        self.tree_cache_valid_k = torch.full(
            (keys.shape[0],), self.config.speculate_k,
            dtype=torch.int64, device=self.device,
        )
        
        # Print cache population details
        if self.config.verbose:
            N = keys.shape[0]
            print(f"\n{'='*80}", flush=True)
            print(f"[CACHE POPULATED] {N} entries", flush=True)
            
            # Show sample entries per sequence
            for seq_id in keys[:, 0].unique()[:1]:  # Just show first sequence
                seq_mask = keys[:, 0] == seq_id
                seq_entries = keys[seq_mask]
                seq_tokens = tokens[seq_mask]
                
                print(f"  Seq {seq_id.item()}: {seq_mask.sum().item()} entries", flush=True)
                
                # Show first 2 unique recovery tokens
                for rec_token in seq_entries[:, 2].unique()[:2]:
                    rec_mask = seq_entries[:, 2] == rec_token
                    if rec_mask.any():
                        idx = rec_mask.nonzero(as_tuple=True)[0][0]
                        k_idx = seq_entries[idx, 1].item()
                        
                        rec_text = self.tokenizer.decode([rec_token.item()])
                        spec_tokens = seq_tokens[idx].tolist()
                        spec_text = [self.tokenizer.decode([t]) for t in spec_tokens]
                        print(f"    k={k_idx}, rec={rec_token.item()} ('{rec_text}') -> {spec_text}", flush=True)
            print(f"{'='*80}\n", flush=True)

    # ============================================================
    # MESA-SSD: 2-pass tree decode methods
    # ============================================================

    def _irecv_mesa_proxy(self, B, K):
        """Post non-blocking recv for proxy. Returns (work, buffer).

        Wire schema branches by policy (docs/mesa/05-policy-b-fix.md):
          - Policy "b" (default, unified K+1): chosen_pos[wire_N] + chosen_tok[wire_N]
          - Policy "a" (legacy, dead): fan_out_list[K+1] + topk_ids[B*K*top_k] + topk_probs[B*K*top_k]
        """
        import torch.distributed as dist
        if self.config.mesa_policy == "b":
            wire_N = self.config.mesa_proxy_wire_N
            total_len = 2 * wire_N
        else:
            top_k = self.config.mesa_proxy_top_k
            total_len = (K + 1) + B * K * top_k + B * K * top_k
        buf = torch.empty(total_len, dtype=torch.int64, device=self.device)
        work = dist.irecv(buf, src=0, group=self.async_pg)
        return work, buf

    def _unpack_mesa_proxy(self, buf, B, K):
        """Unpack proxy data from irecv buffer.

        Returns dict whose keys depend on policy:
          - "b": {"chosen_pos": [wire_N], "chosen_tok": [wire_N]}
          - "a": {"fan_out_list", "topk_ids", "topk_probs"} (legacy)
        """
        if self.config.mesa_policy == "b":
            wire_N = self.config.mesa_proxy_wire_N
            chosen_pos = buf[:wire_N]
            chosen_tok = buf[wire_N:2 * wire_N]
            return {"chosen_pos": chosen_pos, "chosen_tok": chosen_tok}
        # Legacy path (dead — Policy A retained for emergency rollback)
        top_k = self.config.mesa_proxy_top_k
        off = 0
        fan_out_list = buf[off:off + (K + 1)].tolist()  # [K+1] ints
        off += K + 1
        topk_ids = buf[off:off + B * K * top_k].view(B, K, top_k)
        off += B * K * top_k
        topk_probs = buf[off:].to(torch.int32).view(torch.float32).view(B, K, top_k)
        return {"fan_out_list": fan_out_list, "topk_ids": topk_ids, "topk_probs": topk_probs}

    def _select_draft_sourced_tokens(self, logits, returned_tokens, draft_fan_out):
        """Select fork tokens from draft logits (uniform top-k per position)."""
        logits = logits.clone()
        logits[:, :-1, :] = logits[:, :-1, :].scatter(
            dim=2, index=returned_tokens[:, 1:].unsqueeze(2), value=float('-inf'))
        _, topk_idx = torch.topk(logits, draft_fan_out, dim=-1)  # [B, K+1, draft_fan_out]
        return topk_idx

    def _select_draft_sourced_tokens_perpos(self, logits, returned_tokens, fan_out_list):
        """Per-position fan_out selector for Phase 1 (non-uniform fan_out_list).

        Args:
            logits: [B, P, V] glue logits at P=len(fan_out_list) positions.
            returned_tokens: [B, P] target's accepted tokens at each glue pos.
            fan_out_list: list[int] of length P, sum = MQ_LEN.

        Returns:
            flat:    [B, MQ_LEN] int64 — Phase 1 decode input. slot p occupies
                     [offset_p, offset_p + fan_out_list[p]) with top-fan_out_list[p]
                     draft tokens (returned_tokens excluded).
            padded:  [B, P, max_fo] int64 — Policy B dedup input. position p
                     has fan_out_list[p] real entries followed by 0-padded slots.
            mask:    [P, max_fo] bool — True where padded[*, p, j] is real
                     (j < fan_out_list[p]). Required to avoid false-match on
                     0-padding when chosen_tok happens to be 0.
        """
        B, P, V = logits.shape
        assert P == len(fan_out_list), \
            f"glue width {P} != fan_out_list len {len(fan_out_list)}"
        logits = logits.clone()
        # Mask returned token at non-leaf positions (already-accepted tokens
        # excluded from fork candidates).
        if P > 1:
            logits[:, :-1, :] = logits[:, :-1, :].scatter(
                dim=2, index=returned_tokens[:, 1:].unsqueeze(2), value=float('-inf'))
        MQ_LEN = sum(fan_out_list)
        max_fo = max(fan_out_list) if fan_out_list else 0
        device = logits.device
        flat = torch.empty(B, MQ_LEN, dtype=torch.int64, device=device)
        # zero-padded; mask[p, j] tells which slots are real
        padded = torch.zeros(B, P, max_fo, dtype=torch.int64, device=device)
        mask = torch.zeros(P, max_fo, dtype=torch.bool, device=device)
        offset = 0
        for p, fo in enumerate(fan_out_list):
            if fo == 0:
                continue
            _, topk = torch.topk(logits[:, p, :], fo, dim=-1)  # [B, fo]
            flat[:, offset:offset + fo] = topk
            padded[:, p, :fo] = topk
            mask[p, :fo] = True
            offset += fo
        return flat, padded, mask

    def _update_phase2_layout_inplace(self, fan_out_tensor, K_rank):
        """In-place update of self.split_k2_layout for unified Policy B.

        Avoids per-step ``create_tree_layout(...)`` which allocates 6+ new
        GPU tensors. Static fields (MQ_LEN, K, arange_mq, step_pos_offsets,
        step_rope_offsets, graph_key, name) stay; only fan_out-derived
        tensors and position_count update.

        See docs/mesa/05-policy-b-fix.md Section 3.7 + Fix ④.

        Args:
            fan_out_tensor: [K_rank+1] int64 GPU tensor; sum == total_budget.
            K_rank: int. position_count = K_rank+1.

        Returns:
            self.split_k2_layout (mutated).
        """
        layout = self.split_k2_layout
        K_plus_1 = K_rank + 1
        device = fan_out_tensor.device
        # Update mutable fields:
        layout.position_count = K_plus_1
        layout.fan_out_t = fan_out_tensor
        layout.fan_out_t_miss = fan_out_tensor                  # same in unified path
        # fan_idx_hit = arange(K_plus_1).repeat_interleave(fan_out_tensor)
        layout.fan_idx_hit = torch.arange(
            K_plus_1, device=device, dtype=torch.int64
        ).repeat_interleave(fan_out_tensor)
        layout.fan_idx_miss = layout.fan_idx_hit
        # fan_out_list (Python list) — REQUIRED to match position_count by
        # cudagraph_helpers.run_fi_tree_decode_cudagraph (line 329:
        # np.repeat(_tril, _fol)). For K1==K2 case, placeholder length
        # accidentally matched K_rank+1; for K1!=K2 it didn't → ValueError.
        # 1 GPU sync / step (≤9 elements). Cheap, but necessary for CG mask.
        layout.fan_out_list = fan_out_tensor.tolist()
        layout.fan_out_list_miss = layout.fan_out_list
        return layout

    def _select_proxy_sourced_tokens_unified(self, mesa_proxy, draft_forked,
                                              K_rank, total_budget,
                                              draft_forked_mask=None):
        """K+1 unified Policy B selector.

        See docs/mesa/05-policy-b-fix.md Step 3.

        Args:
            mesa_proxy: {"chosen_pos": [wire_N], "chosen_tok": [wire_N]}
                        score-sorted desc. chosen_pos ∈ [0, K_rank] by target-side
                        construction (Section 3.3 invariant).
            draft_forked: [B, P, max_fo] Phase 1 candidates per ranking position.
                          - uniform Phase 1: max_fo = dfo, all slots real.
                          - non-uniform: max_fo = max(fan_out_list); padded with
                            zeros where fan_out_list[p] < max_fo.
                          P ≥ K_rank + 1.
            K_rank: int  ranking horizon = current step's valid_k.
                    Phase 2 forward depth (K2) is independent.
            total_budget: int  Phase 2 tree size (= sum(fan_out_list) returned).
            draft_forked_mask: optional [P, max_fo] bool. None for uniform Phase 1
                               (all-real). Required for non-uniform Phase 1 to
                               avoid false in_draft match on zero-padded slots
                               when chosen_tok happens to be 0.

        Returns:
            result_tokens: [B, total_budget] int64 — pos-grouped (stable sort by
                           pos), score order preserved within each pos.
            fan_out_list:  list[int] length K_rank+1, sum == total_budget.
        """
        chosen_pos = mesa_proxy["chosen_pos"]    # [wire_N]
        chosen_tok = mesa_proxy["chosen_tok"]    # [wire_N]
        N = chosen_pos.shape[0]
        B = draft_forked.shape[0]
        assert B == 1, "MESA invariant: B=1"
        # Invariant: chosen_pos ∈ [0, K_rank] by construction.
        if __debug__:
            assert (chosen_pos <= K_rank).all().item(), \
                f"chosen_pos out of range [0, {K_rank}]: max={chosen_pos.max().item()}"

        # in_draft dedup. Uniform: all slots real. Non-uniform: mask filters
        # zero-padded slots so chosen_tok=0 doesn't false-match on padding.
        df_per_cand = draft_forked[0, chosen_pos, :]                # [N, max_fo]
        eq = (df_per_cand == chosen_tok.unsqueeze(-1))                # [N, max_fo]
        if draft_forked_mask is not None:
            msk = draft_forked_mask[chosen_pos, :]                    # [N, max_fo]
            eq = eq & msk
        in_draft = eq.any(-1)                                         # [N]
        valid = ~in_draft                                            # [N]

        # Pick first total_budget valid (score-sorted preserved).
        rank = valid.to(torch.int64).cumsum(0)                       # [N]
        take = valid & (rank <= total_budget)                        # [N]

        # === Fix ③ (correctness guard) ===
        # Buffer sizing (Section 3.3 / 3.5) is supposed to ensure
        # take.sum() == total_budget. If it doesn't, fan_out_list.sum()
        # would silently differ from proxy_forked.shape[1] (= total_budget)
        # and downstream Phase 2 decode reads zero-padded slots as real
        # tokens. Hard-fail with a clear message instead of silent corruption.
        # NOTE: this is a 1 GPU sync per step; acceptable for safety.
        _take_sum = int(take.sum().item())
        assert _take_sum == total_budget, (
            f"Policy B underfill: take.sum()={_take_sum} != total_budget={total_budget}. "
            f"This means buffer sizing was insufficient. Check config.mesa_proxy_wire_N "
            f"and config.mesa_proxy_top_k auto-raise; see docs/mesa/05-policy-b-fix.md "
            f"Section 3.5."
        )

        taken_pos = chosen_pos[take]                                  # [total_budget]
        taken_tok = chosen_tok[take]

        # fan_out_list 동적 재구성 (length = K_rank+1)
        K_plus_1 = K_rank + 1
        fan_out_tensor = torch.zeros(
            K_plus_1, dtype=torch.int64, device=chosen_pos.device)
        fan_out_tensor.scatter_add_(
            0, taken_pos, torch.ones_like(taken_pos))

        # result tensor [B, total_budget] — group by pos (stable sort preserves score)
        result = torch.zeros(
            B, total_budget, dtype=torch.int64, device=chosen_pos.device)
        order = taken_pos.argsort(stable=True)
        result[0, :taken_tok.shape[0]] = taken_tok[order]

        # Return fan_out_tensor (GPU) — caller does .tolist() once (cheap
        # for K_plus_1 ~9 elements) and updates layout in-place.
        # See Fix ④ (avoid per-step create_tree_layout).
        return result, fan_out_tensor

    def _select_proxy_sourced_tokens_policy_a(self, glue_logits, gd_for_fork,
                                                mesa_proxy, draft_forked, fan_out_list,
                                                valid_k=None):
        """Policy A: h_i-based dynamic fan_out. Fully vectorized.
        Rev1 post-#4: pos<K uses proxy-only (no draft fallback). pos==K uses draft top-k only.
        Step 9A: ``valid_k`` (per-step incoming bucket; defaults to speculate_k=K_long).
        Iterates positions [0, valid_k); pos==valid_k uses all-accept draft top-k.
        For short-hit (valid_k=K_short=K2), trailing K_long-K_short slots in
        proxy_topk_ids / draft_forked are unused.
        """
        B = glue_logits.shape[0]
        assert B == 1, "Policy A vectorized path assumes B=1"
        K = valid_k if valid_k is not None else self.config.speculate_k
        # fan_out_list must be sliced to length valid_k+1 by caller
        assert len(fan_out_list) == K + 1, (
            f"fan_out_list length {len(fan_out_list)} != K+1={K+1}"
        )
        MQ_LEN = sum(fan_out_list)
        device = glue_logits.device
        # Step 9A: slice proxy_topk_ids / proxy_topk_probs to first K positions.
        # Target wire ships [B, K_long, P]; for short-hit only first K=K_short
        # positions are meaningful.
        proxy_topk_ids = mesa_proxy["topk_ids"][:, :K, :]      # [B, K, P]  P = proxy_top_k
        P = proxy_topk_ids.shape[-1]
        dfo = draft_forked.shape[-1]

        # ----- pos < K: proxy-only dedup (no fallback) -----
        # dedup within proxy itself (prefix duplicates → invalid)
        # proxy_exp: [B, K, P, 1], proxy_prev: [B, K, 1, P]
        proxy_exp = proxy_topk_ids.unsqueeze(-1)                # [B,K,P,1]
        proxy_prev = proxy_topk_ids.unsqueeze(-2)               # [B,K,1,P]
        # Mask[..., i, j]: proxy[i] == proxy[j]. Only count j<i as a duplicate.
        eq_prev = (proxy_exp == proxy_prev)                     # [B,K,P,P]
        lower_triu = torch.tril(torch.ones(P, P, device=device, dtype=torch.bool), diagonal=-1)
        in_prev = (eq_prev & lower_triu.view(1, 1, P, P)).any(dim=-1)   # [B,K,P]

        # draft overlap: proxy vs draft_forked[:, :K, :]
        draft_for_pos = draft_forked[:, :K, :]                  # [B,K,dfo]
        in_draft = (proxy_topk_ids.unsqueeze(-1) == draft_for_pos.unsqueeze(-2)).any(dim=-1)  # [B,K,P]

        valid = (~in_prev) & (~in_draft)                        # [B,K,P]
        # Rank within valid: cumsum → 1-indexed rank; rank ≤ fo[pos] → take
        rank = valid.to(torch.int64).cumsum(dim=-1)             # [B,K,P], 0 for invalid
        fan_out_tensor = torch.tensor(fan_out_list[:K], dtype=torch.int64, device=device)  # [K]
        take_mask = valid & (rank <= fan_out_tensor.view(1, K, 1))                          # [B,K,P]

        # ----- pos == K (all-accept): draft logits top-k, excluding draft_forked[K] -----
        fo_K = fan_out_list[K]
        if fo_K > 0:
            logits_K = glue_logits[:, K, :].clone()             # [B,V]
            # Mask out draft_forked tokens (they're already in draft tree at pos K).
            logits_K.scatter_(1, draft_forked[:, K, :], float('-inf'))
            _, all_accept_topk = torch.topk(logits_K, fo_K, dim=-1)  # [B, fo_K]
        else:
            all_accept_topk = torch.empty(B, 0, dtype=torch.int64, device=device)

        # ----- Assemble result tensor [B, MQ_LEN] in fan_idx order -----
        result = torch.zeros(B, MQ_LEN, dtype=torch.int64, device=device)

        # Per-position offsets into result (prefix sum of fan_out_list)
        offsets = [0]
        for fo in fan_out_list:
            offsets.append(offsets[-1] + fo)
        # offsets[pos]: starting index of pos in result

        # Fill pos < K: scatter taken proxy tokens compacted per position
        # For B=1 we unroll per-pos; vectorized gather would be more complex and B=1 here anyway.
        for pos in range(K):
            fo = fan_out_list[pos]
            if fo == 0:
                continue
            # take_mask[0, pos] is [P] bool; proxy_topk_ids[0, pos] is [P] int64
            sel = torch.masked_select(proxy_topk_ids[0, pos], take_mask[0, pos])  # up to fo tokens
            # safety: clip to fo (should be exactly fo post-config guarantee)
            sel = sel[:fo]
            if __debug__ and sel.numel() < fo:
                assert False, f"MESA underfill: pos={pos} fo={fo} got={sel.numel()} " \
                              f"(proxy_top_k={P}, dfo={dfo}; raise mesa_proxy_top_k)"
            result[0, offsets[pos]:offsets[pos]+sel.numel()] = sel

        # Fill pos == K
        if fo_K > 0:
            result[:, offsets[K]:offsets[K]+fo_K] = all_accept_topk

        return result

    def _build_tree_decode_args_for_layout(self, partial_tree_decode_args, forked_tokens,
                                             layout, cache_hits_list, pos_offset=0):
        """Build tree_decode_args using the given layout. Used for both draft and proxy passes.

        Note (Phase 2 of MESA hybrid plan): glue position offset uses
        ``layout.position_count``, not ``K + 1``. For non-MESA / current MESA
        layouts these are equal (legacy invariant); for the hybrid Phase 1
        layout they diverge (forward depth K1 vs position count valid_k+1).
        """
        B = partial_tree_decode_args["num_tokens"].shape[0]
        K = layout.K
        MQ_LEN = layout.MQ_LEN
        glue_offset = layout.position_count

        _b_flat = torch.arange(B, device=self.device, dtype=torch.int64)[:, None].expand(B, MQ_LEN).flatten()
        _fkp1_flat = layout.arange_mq.repeat(B)
        _j_idx_flat = torch.cat([layout.fan_idx_hit if int(h) else layout.fan_idx_miss for h in cache_hits_list])
        N = _b_flat.shape[0]

        _pos_offset = -1 if self.config.use_eagle else 0
        _positions = (partial_tree_decode_args["num_tokens"][_b_flat] - 1 + _pos_offset) + glue_offset + _fkp1_flat
        _rope_positions = (partial_tree_decode_args["num_tokens"][_b_flat] - 1 + _pos_offset) + _j_idx_flat + 1
        _temperatures = partial_tree_decode_args["temperatures"][_b_flat]

        return {
            "metadata_ints": (B, K, layout.fan_out_list[0], N),
            "input_ids": forked_tokens.view(-1),
            "positions": _positions,
            "rope_positions": _rope_positions,
            "block_tables": partial_tree_decode_args["dbt"],
            "temps": _temperatures,
            "rec_flat": forked_tokens.view(-1),
            "seq_ids_expanded": partial_tree_decode_args["seq_ids"][_b_flat],
            "cache_hits": partial_tree_decode_args["cache_hits"],
            "cache_hits_list": cache_hits_list,
            "hidden_states": None,  # non-EAGLE
        }

    def _build_tree_batch_mesa(self, partial_tree_decode_args, glue_decode_input_ids):
        """MESA 2-pass tree decode:
        Pass 1: draft-sourced tokens → decode with draft_layout (immediate, no proxy wait)
        Pass 2: proxy-sourced tokens → decode with proxy_layout (after proxy arrives)
        Both passes reuse same KV scratch positions (safe: results extracted to tensors).

        Split-only K1/K2 mode (SSD_FORCE_SPLIT_K1K2=1): branch to a separate
        path that runs Phase 1 with K1 forwards and Phase 2 with K2 forwards
        independently. NO continuation. See docs/mesa/04-split-k1k2-design.md.
        """
        if SPLIT_K1K2_MODE and self.split_k1_long_layout is not None:
            return self._build_tree_batch_split_k1k2(
                partial_tree_decode_args, glue_decode_input_ids,
            )
        B = partial_tree_decode_args["num_tokens"].shape[0]
        K = self.config.speculate_k

        # Post irecv FIRST — so target send doesn't block
        proxy_recv_work, proxy_buf = self._irecv_mesa_proxy(B, K)

        # Glue decode only (no full tree args waste)
        glue_logits, gd_for_fork, cache_hits, cache_hits_list, dbt, B_glue = \
            self._glue_decode(partial_tree_decode_args, glue_decode_input_ids)

        from ssd.engine.helpers.cudagraph_helpers import mesa_record as _mr, mesa_close as _mc

        # === Pass 1: draft-sourced tokens (즉시 시작) ===
        # Step 9A: layout dispatch by incoming valid_k. B=1 invariant so
        # we read the scalar valid_k. Long-hit (= K_long, draft cache row
        # matched) → phase1_layout_long. Short-hit (= K_short = K2, proxy
        # cache row matched) → phase1_layout_short. Miss → K_long fallback.
        _mev_p1b = _mr("phase1_build")
        # Read pre-sync'd Python scalar (set by hit_cache_and_respond). None
        # means split / pure-MESA path → no bucket dispatch needed; use K_long.
        _step_valid_k = partial_tree_decode_args.get("valid_k_scalar")
        if _step_valid_k is None:
            _step_valid_k = self.config.speculate_k
        _is_short_hit = (
            self.config.mesa_phase1_k is not None
            and _step_valid_k == self.config.mesa_phase2_k  # = K_short
        )
        # Step 9A optional dispatch trace.
        import os as _os_trace
        if _os_trace.environ.get("SSD_TRACE_BUCKET", "0") == "1":
            print(f"[bucket] valid_k={_step_valid_k} "
                  f"{'short' if _is_short_hit else 'long'}", flush=True)
        if self.config.mesa_phase1_k is not None and self.phase1_layout_long is not None:
            _phase1_layout = (
                self.phase1_layout_short if _is_short_hit
                else self.phase1_layout_long
            )
        elif self.phase1_layout_long is not None:
            _phase1_layout = self.phase1_layout_long
        else:
            _phase1_layout = self.draft_layout
        draft_forked = self._select_draft_sourced_tokens(
            glue_logits, gd_for_fork, self.config.mesa_draft_fan_out)
        # Step 9A: slice draft_forked to the layout's position_count for
        # short-hit. glue_logits/draft_forked are computed at K_long+1
        # positions always (glue decode width is fixed); short bucket only
        # uses the first K_short+1 positions.
        _phase1_pos_count = _phase1_layout.position_count
        draft_forked = draft_forked[:, :_phase1_pos_count, :].contiguous()
        draft_tree_args = self._build_tree_decode_args_for_layout(
            partial_tree_decode_args, draft_forked, _phase1_layout, cache_hits_list)
        _mc("phase1_build", _mev_p1b)
        draft_tokens, draft_logits, draft_acts = self._decode_tree(
            draft_tree_args, layout=_phase1_layout)

        # === Step 8: hybrid-default mode resolution ===
        # Default = hybrid (when configured). Split fallback only via explicit
        # env gate (SSD_FORCE_SPLIT_PHASE2=1) for debug/regression. Parity
        # harness (SSD_HYBRID_PARITY=1) runs BOTH paths to compare.
        import os as _os_step8
        _hybrid_available = (
            self.config.mesa_phase1_k is not None
            and getattr(self, "hybrid_phase2_plan", None) is not None
        )
        _force_split = _os_step8.environ.get("SSD_FORCE_SPLIT_PHASE2", "0") == "1"
        _hybrid_default = _hybrid_available and not _force_split
        _parity_check = (
            _hybrid_available
            and _os_step8.environ.get("SSD_HYBRID_PARITY", "0") == "1"
        )
        # Run split continuation if: (a) hybrid not available (legacy path),
        # (b) force-split requested, or (c) parity check (need split outputs
        # to diff against hybrid). Hybrid-default skips it.
        _run_split_cont_decode = (not _hybrid_default) or _parity_check

        if _run_split_cont_decode and (
            self.config.mesa_phase1_k is not None
            and self.phase1_layout_long is not None
        ):
            import dataclasses as _dc
            K1_cfg = self.config.mesa_phase1_k
            K2_cfg = self.config.mesa_phase2_k
            MQ_p1 = _phase1_layout.MQ_LEN
            cont_layout = _dc.replace(
                _phase1_layout,
                K=K2_cfg,
                step_pos_offsets=torch.arange(K1_cfg, K1_cfg + K2_cfg, device=self.device, dtype=torch.int64)[:, None] * MQ_p1,
                step_rope_offsets=torch.arange(K1_cfg, K1_cfg + K2_cfg, device=self.device, dtype=torch.int64)[:, None],
            )
            cont_input_ids = draft_tokens[:, K1_cfg - 1].contiguous()
            cont_tree_args = dict(draft_tree_args)
            cont_tree_args["input_ids"] = cont_input_ids
            cont_tokens, cont_logits, cont_acts = self._decode_tree(cont_tree_args, layout=cont_layout)
            draft_tokens = torch.cat([draft_tokens, cont_tokens], dim=1)
            draft_logits = torch.cat([draft_logits, cont_logits], dim=1)
            if draft_acts is not None and cont_acts is not None:
                draft_acts = torch.cat([draft_acts, cont_acts], dim=1)

        # === Pass 중간: proxy 수신 ===
        _mev_pw = _mr("proxy_wait")
        proxy_recv_work.wait()
        _mc("proxy_wait", _mev_pw)

        # === Pass 2: Policy A — dynamic fan_out from h_i ===
        _mev_p2b = _mr("phase2_build")
        mesa_proxy = self._unpack_mesa_proxy(proxy_buf, B, K)
        fan_out_list = mesa_proxy["fan_out_list"]  # [K+1] from target

        # Step 9A: HybridPhase2Plan begin_step uses ACTUAL incoming valid_k.
        # B=1 invariant. valid_k drives proxy layout position_count and the
        # plan's internal long/short scalars. fan_out_list already covers
        # (valid_k+1) positions on the target side (per fused_response wire
        # contract); slice to the meaningful prefix.
        if self.config.mesa_phase1_k is not None and getattr(self, "hybrid_phase2_plan", None) is not None:
            _vk_step = _step_valid_k
            _fol_meaningful = fan_out_list[:_vk_step + 1]
            # Phase D debug gates: skip cont or proxy in plan + mask build
            import os as _os_local
            _skip_cont = _os_local.environ.get("SSD_HYBRID_SKIP_CONT", "0") == "1"
            _skip_proxy = _os_local.environ.get("SSD_HYBRID_SKIP_PROXY", "0") == "1"
            self.hybrid_phase2_plan.begin_step(
                valid_k=_vk_step, fan_out_list=_fol_meaningful,
                skip_cont=_skip_cont, skip_proxy=_skip_proxy,
            )

        from ssd.engine.helpers.tree_layout import create_tree_layout
        # Step 9A: proxy layout dispatch by valid_k. position_count =
        # valid_k+1 (matches phase1's glue width for this step). K=K2
        # (proxy forward depth) for hybrid; K=K_long for legacy non-hybrid.
        # fan_out_list comes from target wire (length K_long+1); slice to
        # the meaningful prefix that matches position_count.
        if self.config.mesa_phase1_k is not None:
            proxy_forward_depth = self.config.mesa_phase2_k
            proxy_position_count = _step_valid_k + 1
        else:
            proxy_forward_depth = K
            proxy_position_count = K + 1
        proxy_fan_out_list = fan_out_list[:proxy_position_count]
        step_proxy_layout = create_tree_layout(
            name="proxy",
            fan_out_list=proxy_fan_out_list,
            fan_out_list_miss=proxy_fan_out_list,
            K=proxy_forward_depth, device=self.device,
            position_count=proxy_position_count)

        # Token selection with dynamic per-position fan_out. Step 9A: pass
        # the trimmed fan_out_list (length valid_k+1) and valid_k explicitly
        # so short-hit iterates only over the meaningful prefix.
        proxy_forked = self._select_proxy_sourced_tokens_policy_a(
            glue_logits, gd_for_fork, mesa_proxy, draft_forked,
            proxy_fan_out_list, valid_k=_step_valid_k)
        proxy_tree_args = self._build_tree_decode_args_for_layout(
            partial_tree_decode_args, proxy_forked, step_proxy_layout, cache_hits_list)
        _mc("phase2_build", _mev_p2b)
        # Split proxy decode runs in: split-default (no hybrid) OR parity check.
        # In hybrid-default, hybrid forward produces proxy outputs directly
        # (writes B_proxy region; never overwrites Phase 1 KV → no re-run needed).
        _run_split_proxy_decode = (not _hybrid_default) or _parity_check
        if _run_split_proxy_decode:
            proxy_tokens, proxy_logits, proxy_acts = self._decode_tree(
                proxy_tree_args, layout=step_proxy_layout)
        else:
            # Native hybrid-default: proxy outputs come from hybrid forward below
            proxy_tokens = None
            proxy_logits = None
            proxy_acts = None

        # Step 8: hybrid plan build. Required for hybrid forward (default
        # path) AND for parity oracle. Skipped only when in pure split-fallback
        # mode without parity (legacy).
        import os as _os
        _use_hybrid = _hybrid_default  # Step 8: default path = hybrid
        if _hybrid_default or _parity_check:
            _mev_phb = _mr("phase2_hybrid_build")
            self._build_phase2_hybrid_plan(
                draft_tree_args=draft_tree_args,
                proxy_tree_args=proxy_tree_args,
                step_proxy_layout=step_proxy_layout,
                fan_out_list=fan_out_list,
            )
            _mc("phase2_hybrid_build", _mev_phb)
        # Phase A: split eager mirror gate. When SSD_MIRROR_PROXY=1 (independent
        # of full hybrid parity), run mirror on split's proxy outputs to verify
        # bool-eager planning vs CG packed planning produce same proxy results.
        # Step 8: requires split path to have run (proxy_tokens not None).
        _mirror_check = (
            self.config.mesa_phase1_k is not None
            and _os.environ.get("SSD_MIRROR_PROXY", "0") == "1"
            and proxy_tokens is not None
        )
        if _mirror_check:
            mirror_tokens, mirror_logits, mirror_meta = self._split_eager_mirror_proxy(
                proxy_tree_args=proxy_tree_args,
                step_proxy_layout=step_proxy_layout,
            )
            self._mirror_parity_diff(
                split_proxy_tokens=proxy_tokens,
                split_proxy_logits=proxy_logits,
                mirror_tokens=mirror_tokens,
                mirror_logits=mirror_logits,
                mirror_meta=mirror_meta,
            )

        # Phase C-1: fresh eager-only proxy wrapper experiment. Both bool and
        # packed paths run on a wrapper that has NEVER been used by CG capture
        # (proxy_eager_debug, use_cuda_graph=False). Tests if wrapper-state
        # pollution from CG-shared "proxy" wrapper was contaminating Phase B-2.
        # Step 8: requires split path to have run.
        _fresh_check = (
            self.config.mesa_phase1_k is not None
            and _os.environ.get("SSD_FRESH_EAGER_PROXY", "0") == "1"
            and proxy_tokens is not None
        )
        if _fresh_check:
            # Bool path on fresh wrapper
            fresh_bool_tokens, fresh_bool_logits, _ = self._split_eager_mirror_proxy(
                proxy_tree_args=proxy_tree_args,
                step_proxy_layout=step_proxy_layout,
                wrapper_key="proxy_eager_debug",
            )
            # Packed path on fresh wrapper
            fresh_pkg_tokens, fresh_pkg_logits, _ = self._split_low_level_mirror_proxy(
                proxy_tree_args=proxy_tree_args,
                step_proxy_layout=step_proxy_layout,
                wrapper_key="proxy_eager_debug",
            )

            def _diff_summary(a_tok, a_logits, b_tok, b_logits, label_a, label_b):
                eq = (a_tok == b_tok)
                n_total = eq.numel()
                n_mis = int((~eq).sum().item())
                d = (a_logits - b_logits).abs()
                return (f"{label_a} vs {label_b}: "
                        f"tokens {n_mis}/{n_total} mismatch "
                        f"({100*n_mis/max(n_total,1):.1f}%); "
                        f"logits max_abs={float(d.max().item()):.6f} "
                        f"mean_abs={float(d.mean().item()):.6f}")

            print(f"[FRESH C-1] " + _diff_summary(
                proxy_tokens, proxy_logits, fresh_bool_tokens, fresh_bool_logits,
                "split_CG", "fresh_bool"), flush=True)
            print(f"[FRESH C-1] " + _diff_summary(
                proxy_tokens, proxy_logits, fresh_pkg_tokens, fresh_pkg_logits,
                "split_CG", "fresh_packed"), flush=True)
            print(f"[FRESH C-1] " + _diff_summary(
                fresh_bool_tokens, fresh_bool_logits, fresh_pkg_tokens, fresh_pkg_logits,
                "fresh_bool", "fresh_packed"), flush=True)

        # Phase B-2: low-level packed mirror gate. SSD_LOWLEVEL_MIRROR_PROXY=1
        # Step 8: requires split path to have run.
        _lowlevel_mirror = (
            self.config.mesa_phase1_k is not None
            and _os.environ.get("SSD_LOWLEVEL_MIRROR_PROXY", "0") == "1"
            and proxy_tokens is not None
        )
        if _lowlevel_mirror:
            ll_tokens, ll_logits, ll_meta = self._split_low_level_mirror_proxy(
                proxy_tree_args=proxy_tree_args,
                step_proxy_layout=step_proxy_layout,
            )
            tok_eq = (proxy_tokens == ll_tokens)
            n_total = tok_eq.numel()
            n_mismatch = int((~tok_eq).sum().item())
            diff = (proxy_logits - ll_logits).abs()
            print(f"[LL MIRROR] split CG vs split low-level mirror: "
                  f"tokens {n_mismatch}/{n_total} mismatch "
                  f"({100*n_mismatch/max(n_total,1):.1f}%); "
                  f"logits max_abs_diff={float(diff.max().item()):.6f} "
                  f"mean_abs_diff={float(diff.mean().item()):.6f}", flush=True)
            if n_mismatch == 0:
                print("[LL MIRROR] PASSES — low-level packed path is faithful. "
                      "Hybrid can be ported to this path for parity.", flush=True)
            else:
                first_idx = (~tok_eq).nonzero(as_tuple=False)[0]
                row, depth = int(first_idx[0].item()), int(first_idx[1].item())
                print(f"[LL MIRROR] first mismatch row={row} depth={depth} "
                      f"split={int(proxy_tokens[row,depth])} "
                      f"ll={int(ll_tokens[row,depth])}", flush=True)

        if _use_hybrid:
            # Step 7 parity harness: capture split's continuation tail +
            # proxy outputs BEFORE overwriting, so we can diff them against
            # hybrid outputs.
            if _parity_check:
                K1_cfg_diff = self.config.mesa_phase1_k
                _split_cont_tokens = draft_tokens[:, K1_cfg_diff:].clone()  # [cont, K2]
                _split_cont_logits = draft_logits[:, K1_cfg_diff:].clone()
                _split_proxy_tokens = proxy_tokens.clone()
                _split_proxy_logits = proxy_logits.clone()

            _cont_oracle = (
                _parity_check
                and _os.environ.get("SSD_HYBRID_CONT_ORACLE", "0") == "1"
            )
            _proxy_oracle = (
                _parity_check
                and _os.environ.get("SSD_HYBRID_PROXY_ORACLE", "0") == "1"
            )

            # Phase 1 KV state: in native hybrid-default mode, split proxy
            # decode was SKIPPED → Phase 1 KV is still clean from Pass 1's
            # decode. No re-run needed. Under parity_check, split proxy ran
            # and overwrote P1 KV; re-run Phase 1 to restore for hybrid.
            if _parity_check:
                self._decode_tree(draft_tree_args, layout=_phase1_layout)

            # Phase D-4: correct_split_cont oracle. Runs AFTER Phase 1 re-run
            # so it sees clean Phase 1 KV (same as hybrid). Oracle uses
            # split's physical layout (cont slot positions) but plan-correct
            # mask, providing a third reference point for 3-way diff.
            # Cont oracle writes A_tail only — does NOT touch Phase 1 KV.
            _oracle_cont_tokens = None
            _oracle_cont_logits = None
            if _cont_oracle:
                _oracle_cont_tokens, _oracle_cont_logits = (
                    self._decode_correct_split_cont(
                        draft_tree_args=draft_tree_args,
                        draft_tokens_phase1=draft_tokens[:, :self.config.mesa_phase1_k],
                    )
                )

            # Phase D-5: fresh_proxy_oracle. Eager bool-mask path on a
            # use_cuda_graph=False wrapper (proxy_eager_debug) that NEVER
            # touched CG capture — fresh state, no buffer pollution.
            # Proxy oracle writes overlap Phase 1 KV slots (split's natural
            # layout), so a Phase 1 re-run is required AFTER the oracle to
            # restore P1 KV before hybrid runs.
            _oracle_proxy_tokens = None
            _oracle_proxy_logits = None
            _oracle_proxy_meta = None
            # Phase D-6: self-consistency. SSD_HYBRID_SELF_CONSISTENCY=1
            # runs each path twice with same args (P1 restored between runs)
            # and diffs run-1 vs run-2. Determinism check: oracle should be
            # 0/72 with itself; hybrid should be 0/72 with itself.
            _self_consistency = (
                _parity_check
                and _os.environ.get("SSD_HYBRID_SELF_CONSISTENCY", "0") == "1"
            )
            _oracle_proxy_tokens_2 = None
            _oracle_proxy_logits_2 = None
            if _proxy_oracle:
                _oracle_proxy_tokens, _oracle_proxy_logits, _oracle_proxy_meta = (
                    self._split_eager_mirror_proxy(
                        proxy_tree_args=proxy_tree_args,
                        step_proxy_layout=step_proxy_layout,
                        wrapper_key="proxy_eager_debug",
                    )
                )
                # Restore Phase 1 KV after proxy oracle's overwrite.
                self._decode_tree(draft_tree_args, layout=_phase1_layout)

                if _self_consistency:
                    # Run oracle a SECOND time with same args, same restored
                    # initial state. Should produce IDENTICAL output (deterministic).
                    _oracle_proxy_tokens_2, _oracle_proxy_logits_2, _ = (
                        self._split_eager_mirror_proxy(
                            proxy_tree_args=proxy_tree_args,
                            step_proxy_layout=step_proxy_layout,
                            wrapper_key="proxy_eager_debug",
                        )
                    )
                    # Restore P1 again before hybrid runs.
                    self._decode_tree(draft_tree_args, layout=_phase1_layout)

            # Step 9B-2: hybrid runtime dispatch:
            #   default       → CG hybrid (run_phase2_hybrid_cudagraph)
            #   force_eager   → eager hybrid (_decode_phase2_hybrid)
            #   force_split   → reached via _force_split branch (no hybrid)
            _force_eager_hybrid = (
                _os.environ.get("SSD_FORCE_EAGER_HYBRID_PHASE2", "0") == "1"
            )
            _bucket_h = "long" if self.hybrid_phase2_plan.valid_k == self.hybrid_phase2_plan.K_long else "short"
            _hybrid_cg_key = f"phase2_hybrid_{_bucket_h}_cg"
            _hybrid_cg_available = (
                not _force_eager_hybrid
                and not self.enforce_eager
                and _hybrid_cg_key in self.graphs
            )
            if _hybrid_cg_available:
                # Per-depth `phase2_hybrid_prep_{bucket}` and
                # `phase2_hybrid_replay_{bucket}` fire INSIDE the K2 loop
                # in run_phase2_hybrid_cudagraph (matches split's per-depth
                # phase{1,2}_prep / phase{1,2}_replay semantics so the K2
                # forwards show as K2 separate events, not one fused event).
                from ssd.engine.helpers.cudagraph_helpers import run_phase2_hybrid_cudagraph
                cont_tokens_h, cont_logits_h, proxy_tokens_h, proxy_logits_h = (
                    run_phase2_hybrid_cudagraph(
                        self,
                        plan=self.hybrid_phase2_plan,
                        draft_tree_args=draft_tree_args,
                        proxy_tree_args=proxy_tree_args,
                        step_proxy_layout=step_proxy_layout,
                        draft_tokens_phase1=draft_tokens[:, :self.config.mesa_phase1_k],
                        bucket=_bucket_h,
                    )
                )
            else:
                # Eager fallback — per-depth label fires inside the eager
                # K2 loop (see _decode_phase2_hybrid).
                cont_tokens_h, cont_logits_h, proxy_tokens_h, proxy_logits_h = (
                    self._decode_phase2_hybrid(
                        draft_tree_args=draft_tree_args,
                        proxy_tree_args=proxy_tree_args,
                        step_proxy_layout=step_proxy_layout,
                        draft_tokens_phase1=draft_tokens[:, :self.config.mesa_phase1_k],
                    )
                )

            # Phase D-6 hybrid self-consistency: run hybrid AGAIN with same
            # inputs. Hybrid writes A_tail + B_proxy (disjoint from P1 KV),
            # so no P1 restore is needed between runs. Output should be
            # bit-identical if hybrid is deterministic.
            _proxy_tokens_h_2 = None
            _proxy_logits_h_2 = None
            _cont_tokens_h_2 = None
            _cont_logits_h_2 = None
            if _self_consistency:
                _cont_tokens_h_2, _cont_logits_h_2, _proxy_tokens_h_2, _proxy_logits_h_2 = (
                    self._decode_phase2_hybrid(
                        draft_tree_args=draft_tree_args,
                        proxy_tree_args=proxy_tree_args,
                        step_proxy_layout=step_proxy_layout,
                        draft_tokens_phase1=draft_tokens[:, :self.config.mesa_phase1_k],
                    )
                )

            if _parity_check:
                # Skip-mode short-circuits (Phase D debug).
                _skip_cont_diff = _os.environ.get("SSD_HYBRID_SKIP_CONT", "0") == "1"
                _skip_proxy_diff = _os.environ.get("SSD_HYBRID_SKIP_PROXY", "0") == "1"
                if _skip_cont_diff and not _skip_proxy_diff:
                    print(f"[HYBRID PARITY skip_cont] proxy-only mode", flush=True)
                    eq = (_split_proxy_tokens == proxy_tokens_h)
                    n_total = eq.numel()
                    n_mis = int((~eq).sum().item())
                    d = (_split_proxy_logits - proxy_logits_h).abs()
                    print(f"[HYBRID PARITY skip_cont] split_proxy vs hybrid_proxy: "
                          f"tokens {n_mis}/{n_total} mismatch "
                          f"({100*n_mis/max(n_total,1):.1f}%); "
                          f"logits max_abs={float(d.max().item()):.6f} "
                          f"mean_abs={float(d.mean().item()):.6f}", flush=True)
                elif _skip_proxy_diff and not _skip_cont_diff:
                    print(f"[HYBRID PARITY skip_proxy] cont-only mode", flush=True)
                    eq = (_split_cont_tokens == cont_tokens_h)
                    n_total = eq.numel()
                    n_mis = int((~eq).sum().item())
                    d = (_split_cont_logits - cont_logits_h).abs()
                    print(f"[HYBRID PARITY skip_proxy] split_cont vs hybrid_cont: "
                          f"tokens {n_mis}/{n_total} mismatch "
                          f"({100*n_mis/max(n_total,1):.1f}%); "
                          f"logits max_abs={float(d.max().item()):.6f} "
                          f"mean_abs={float(d.mean().item()):.6f}", flush=True)
                else:
                    # Phase D-5: oracle-aware parity output. Two primary lines
                    # (hybrid vs class-specific oracle); split-vs-* are diags.
                    self._oracle_parity_report(
                        split_cont_tokens=_split_cont_tokens,
                        split_cont_logits=_split_cont_logits,
                        split_proxy_tokens=_split_proxy_tokens,
                        split_proxy_logits=_split_proxy_logits,
                        hybrid_cont_tokens=cont_tokens_h,
                        hybrid_cont_logits=cont_logits_h,
                        hybrid_proxy_tokens=proxy_tokens_h,
                        hybrid_proxy_logits=proxy_logits_h,
                        oracle_cont_tokens=_oracle_cont_tokens,
                        oracle_cont_logits=_oracle_cont_logits,
                        oracle_proxy_tokens=_oracle_proxy_tokens,
                        oracle_proxy_logits=_oracle_proxy_logits,
                        oracle_proxy_meta=_oracle_proxy_meta,
                        step_proxy_layout=step_proxy_layout,
                        oracle_proxy_tokens_2=_oracle_proxy_tokens_2,
                        oracle_proxy_logits_2=_oracle_proxy_logits_2,
                        hybrid_proxy_tokens_2=_proxy_tokens_h_2,
                        hybrid_proxy_logits_2=_proxy_logits_h_2,
                        hybrid_cont_tokens_2=_cont_tokens_h_2,
                        hybrid_cont_logits_2=_cont_logits_h_2,
                    )

            # In Phase D skip modes, keep split outputs for the skipped class
            # so cache emit doesn't break.
            _skip_cont_replace = _os.environ.get("SSD_HYBRID_SKIP_CONT", "0") == "1"
            _skip_proxy_replace = _os.environ.get("SSD_HYBRID_SKIP_PROXY", "0") == "1"
            if not _skip_cont_replace:
                draft_tokens = torch.cat(
                    [draft_tokens[:, :self.config.mesa_phase1_k], cont_tokens_h], dim=1
                )
                draft_logits = torch.cat(
                    [draft_logits[:, :self.config.mesa_phase1_k], cont_logits_h], dim=1
                )
            if not _skip_proxy_replace:
                proxy_tokens = proxy_tokens_h
                proxy_logits = proxy_logits_h

        # === Merge + populate cache (use runtime layouts for correct keys) ===
        # Step 9A: pass _phase1_layout so cache keys reflect short bucket
        # for short-hit steps. Without this the legacy long-bucket draft_layout
        # would size-mismatch with draft_args (which was built using
        # phase1_layout_short for short-hit).
        self._merge_and_populate_cache(
            draft_tree_args, draft_tokens, draft_logits,
            proxy_tree_args, proxy_tokens, proxy_logits,
            cache_hits_list, draft_acts, proxy_acts,
            proxy_layout=step_proxy_layout,
            draft_layout=_phase1_layout)

    def _build_tree_batch_split_k1k2(self, partial_tree_decode_args, glue_decode_input_ids):
        """Split-only K1/K2 mode (per docs/mesa/04-split-k1k2-design.md).

        Two independent passes, NO continuation:
          - Phase 1 (Draft pass): K1 forwards with split_k1_layout (pos=K1+1).
            Output: draft-sourced rows of depth K1.
          - Phase 2 (Proxy pass): K2 forwards with split_k2_layout (pos=K2+1).
            Output: proxy-sourced rows of depth K2.

        Cache contract: draft rows valid_k=K1, proxy rows valid_k=K2.
        valid_k space = {K1, K2}.
        """
        from ssd.engine.helpers.cudagraph_helpers import mesa_record as _mr, mesa_close as _mc

        B = partial_tree_decode_args["num_tokens"].shape[0]
        K = self.config.speculate_k
        K1 = self.config.mesa_phase1_k
        K2 = self.config.mesa_phase2_k
        # Split-only K1/K2 mode: K2 <= K1 contract (asserted at init time).
        # No need to re-assert here, but document the invariant:
        #   K_max = K1, K_min = K2, valid_k space = {K1, K2} ⊆ [1, K1].

        # Post irecv FIRST so target send doesn't block.
        proxy_recv_work, proxy_buf = self._irecv_mesa_proxy(B, K)

        # Glue decode (single forward returning logits at the accept-frontier).
        # Glue width follows valid_k+1 (matched row depth + 1):
        #   - valid_k=K1 (or first-step K_max=K1) → glue width = K1+1
        #   - valid_k=K2                          → glue width = K2+1
        glue_logits, gd_for_fork, cache_hits, cache_hits_list, dbt, B_glue = \
            self._glue_decode(partial_tree_decode_args, glue_decode_input_ids)

        # === Phase 1 layout dispatch by step's valid_k (long/short bucket) ===
        # valid_k=K1 → split_k1_long_layout (pos=K1+1) — glue width matches.
        # valid_k=K2 → split_k1_short_layout (pos=K2+1) — glue width matches.
        # Read pre-sync'd Python scalar from partial_tree_decode_args.
        _step_valid_k = partial_tree_decode_args.get("valid_k_scalar")
        if _step_valid_k is None:
            _step_valid_k = K1  # K_max default for first-step / no-hit path
        assert _step_valid_k in (K1, K2), (
            f"split-K1K2 phase1 dispatch: unexpected valid_k={_step_valid_k}; "
            f"expected K1={K1} or K2={K2}"
        )
        _is_short_hit = (_step_valid_k == K2 and K2 != K1)
        _layout_k1 = (
            self.split_k1_short_layout if _is_short_hit
            else self.split_k1_long_layout
        )

        # === Phase 1: Draft pass, K1 forwards ===
        _mev_p1b = _mr("phase1_build")
        # Per-position selector when Phase 1 fan_out_list is non-uniform.
        # Glue is sized to position_count (= K1+1 long / K2+1 short); slice
        # both glue_logits and gd_for_fork before passing.
        _pc = _layout_k1.position_count
        _is_uniform = all(
            f == _layout_k1.fan_out_list[0] for f in _layout_k1.fan_out_list
        )
        if _is_uniform:
            # Uniform Phase 1: 3D [B, P, dfo] direct. No padded/mask needed
            # (all slots real, mask=None signals all-real to selector).
            draft_forked_full = self._select_draft_sourced_tokens(
                glue_logits, gd_for_fork, _layout_k1.fan_out_list[0])
            draft_forked_k1 = draft_forked_full[:, :_pc, :].contiguous()
            draft_forked_p1_padded = draft_forked_k1   # same as 3D form
            draft_forked_p1_mask = None                # all-real
        else:
            # Non-uniform Phase 1: perpos selector returns 3-tuple.
            # flat is for Phase 1 decode, padded/mask are for Policy B dedup.
            (draft_forked_k1, draft_forked_p1_padded,
             draft_forked_p1_mask) = self._select_draft_sourced_tokens_perpos(
                glue_logits[:, :_pc, :].contiguous(),
                gd_for_fork[:, :_pc].contiguous(),
                _layout_k1.fan_out_list,
            )
        draft_tree_args = self._build_tree_decode_args_for_layout(
            partial_tree_decode_args, draft_forked_k1, _layout_k1, cache_hits_list)
        _mc("phase1_build", _mev_p1b)
        draft_tokens, draft_logits, draft_acts = self._decode_tree(
            draft_tree_args, layout=_layout_k1)

        # === Wait for proxy from target ===
        _mev_pw = _mr("proxy_wait")
        proxy_recv_work.wait()
        _mc("proxy_wait", _mev_pw)
        mesa_proxy = self._unpack_mesa_proxy(proxy_buf, B, K)
        # ===== TRACE point 4 (early): proxy unpack =====
        if TRACE_SPLIT_K1K2:
            K_max = K1 if K1 >= K2 else K2
            _vk_step = partial_tree_decode_args.get("valid_k_scalar")
            if self.config.mesa_policy == "b":
                print(f"[TRACE-split-k1k2 #4a proxy_unpack policy=b] K1={K1} K2={K2} K_max={K_max} "
                      f"step_valid_k_scalar={_vk_step} "
                      f"chosen_pos.shape={list(mesa_proxy['chosen_pos'].shape)} "
                      f"chosen_tok.shape={list(mesa_proxy['chosen_tok'].shape)}",
                      flush=True)
            else:
                _topk_ids_shape = list(mesa_proxy["topk_ids"].shape)
                _topk_probs_shape = list(mesa_proxy["topk_probs"].shape)
                _per_pos_sum = []
                _src_pos = min(K2, mesa_proxy["topk_ids"].shape[1])
                for _p in range(_src_pos):
                    _per_pos_sum.append(round(float(mesa_proxy["topk_probs"][0, _p].sum().item()), 4))
                print(f"[TRACE-split-k1k2 #4a proxy_unpack policy=a] K1={K1} K2={K2} K_max={K_max} "
                      f"step_valid_k_scalar={_vk_step} "
                      f"topk_ids.shape={_topk_ids_shape} topk_probs.shape={_topk_probs_shape} "
                      f"fan_out_list={mesa_proxy['fan_out_list']} "
                      f"topk_probs_sum_per_pos[0..min(K2,wire)-1]={_per_pos_sum}",
                      flush=True)
        # ===== END TRACE =====

        # === Phase 2: Proxy pass, K2 forwards (independent) ===
        _mev_p2b = _mr("phase2_build")
        pfo = self.config.mesa_proxy_fan_out

        if self.config.mesa_policy == "b":
            # === Policy B unified path (default) ===
            # docs/mesa/05-policy-b-fix.md Step 4.
            # Per-step:
            #   K = K2 (forward depth, captured in self.split_k2_layout, fixed)
            #   position_count = K_rank+1 (= valid_k+1, dynamic, in-place update)
            #   fan_out_tensor = selector output (dedup-aware, sum=total_budget)
            #
            # Fix ④: avoid per-step create_tree_layout(...) by updating
            # self.split_k2_layout in-place. Selector returns fan_out as
            # GPU tensor (no .tolist() sync); _update_phase2_layout_inplace
            # rebuilds only the mutable parts.
            K_rank = _step_valid_k
            total_budget = self.config.mesa_proxy_total_budget   # = pfo*(K_max+1)
            # Pass padded [B, P, max_fo] + mask [P, max_fo] for Policy B dedup.
            # Uniform: padded == 3D draft_forked_k1, mask=None (all-real).
            # Non-uniform: padded zero-filled beyond fan_out_list[p], mask
            # filters to avoid false-match on chosen_tok=0.
            proxy_forked, proxy_fan_out_tensor = self._select_proxy_sourced_tokens_unified(
                mesa_proxy, draft_forked_p1_padded,
                K_rank=K_rank, total_budget=total_budget,
                draft_forked_mask=draft_forked_p1_mask,
            )
            _layout_k2 = self._update_phase2_layout_inplace(proxy_fan_out_tensor, K_rank)
            if TRACE_SPLIT_K1K2:
                _fol_dbg = proxy_fan_out_tensor.tolist()      # debug-only sync
                print(f"[TRACE-split-k1k2 #4b proxy_seed policy=b] K2={K2} K_rank={K_rank} "
                      f"total_budget={total_budget} fan_out_list={_fol_dbg}",
                      flush=True)
        else:
            # === Legacy Policy A path (dead — config 검증으로 차단됨) ===
            # TODO(policy-a): see docs/mesa/05-policy-b-fix.md Section 3.1
            _layout_k2 = self.split_k2_layout  # K=K2, position_count=K2+1, fan_out=pfo
            proxy_topk_ids = mesa_proxy["topk_ids"]
            proxy_seed_3d = torch.zeros(
                B, _layout_k2.position_count, pfo,
                dtype=torch.int64, device=self.device,
            )
            _src_pos = min(K2, proxy_topk_ids.shape[1])
            proxy_seed_3d[:, :_src_pos, :] = proxy_topk_ids[:, :_src_pos, :pfo]
            if _src_pos <= _layout_k2.position_count - 1:
                _, leaf_topk = torch.topk(
                    glue_logits[:, _src_pos, :] if _src_pos < glue_logits.shape[1]
                    else glue_logits[:, -1, :],
                    pfo, dim=-1,
                )
                proxy_seed_3d[:, _src_pos, :] = leaf_topk
            proxy_forked = proxy_seed_3d.view(B, -1)
            if TRACE_SPLIT_K1K2:
                _seeds = []
                for _p in range(_layout_k2.position_count):
                    _seeds.append(proxy_seed_3d[0, _p, :].tolist())
                print(f"[TRACE-split-k1k2 #4b proxy_seed_consumed policy=a] K2={K2} pfo={pfo} "
                      f"_src_pos={_src_pos} layout.position_count={_layout_k2.position_count} "
                      f"proxy_seed_3d_per_pos={_seeds}",
                      flush=True)

        proxy_tree_args = self._build_tree_decode_args_for_layout(
            partial_tree_decode_args, proxy_forked, _layout_k2, cache_hits_list)
        _mc("phase2_build", _mev_p2b)
        proxy_tokens, proxy_logits, proxy_acts = self._decode_tree(
            proxy_tree_args, layout=_layout_k2)

        # === Cache write: draft rows valid_k=K1, proxy rows valid_k=K2 ===
        # _merge_and_populate_cache infers per-row valid_k from
        # draft_tokens.shape[1] (=K1) and proxy_tokens.shape[1] (=K2).
        self._merge_and_populate_cache(
            draft_tree_args, draft_tokens, draft_logits,
            proxy_tree_args, proxy_tokens, proxy_logits,
            cache_hits_list, draft_acts, proxy_acts,
            proxy_layout=_layout_k2,
            draft_layout=_layout_k1)

    def _build_phase2_hybrid_plan(self, *, draft_tree_args, proxy_tree_args,
                                    step_proxy_layout, fan_out_list):
        """Fill HybridPhase2Plan per-row × per-depth tensors in-place.

        Called when ``mesa_phase1_k`` is configured. ``_decode_phase2_hybrid``
        consumes these tensors (default hot path). Split-fallback path reads
        nothing from the plan.

        Per reviewer guidance:
        - per_row_context_lens_by_depth = physical KV length in the shared
          5-region layout (uniform across all rows at fixed depth). Per-row
          semantic visibility differences are handled by the custom mask
          (Step 4).
        - per_row_block_tables keeps per-row contract structurally. With
          current (block_size=256) the 5-region scratch fits in 1-2 pages so
          the fill is identical across rows; that's a current-config collapse,
          not a semantic simplification.
        - depth-major slot placement: cont row i at depth d writes scratch
          position cont_initial[i] + d * MQ_p1; proxy row j at depth d writes
          proxy_initial[j] + d * MQ_proxy. step_pos_offsets is NOT shared
          (different stride per region), so slot_maps_by_depth is computed
          row-wise directly.
        - proxy initial positions / rope positions come from the current
          proxy path's glue-based semantics (matches step_proxy_layout's
          fan_idx). Cont initial rope position = num_tokens + j_idx_p1[i] +
          K1 (extending past Phase 1's K1 ancestors).
        """
        from ssd.engine.helpers.cudagraph_helpers import (
            mesa_record as _mr_b, mesa_close as _mc_b,
        )
        plan = self.hybrid_phase2_plan
        K1 = self.config.mesa_phase1_k
        K2 = self.config.mesa_phase2_k
        K_long = K1 + K2
        block_size = self.block_size
        device = self.device

        cont_count = plan.cont_row_count
        proxy_count = plan.proxy_row_count
        total = plan.total_row_count

        _mev_b_setup = _mr_b("phase2_hybrid_build_setup")
        # Step 9A: phase1 layout dispatch by plan.valid_k. Long-hit uses
        # phase1_layout_long (MQ_LEN=(K_long+1)*dfo); short-hit uses
        # phase1_layout_short (MQ_LEN=(K_short+1)*dfo).
        K_step = plan.valid_k  # this step's incoming bucket: K_long or K_short
        if K_step == plan.K_long:
            _phase1_layout_step = self.phase1_layout_long
        else:
            assert K_step == plan.K_short, f"unexpected valid_k={K_step}"
            _phase1_layout_step = self.phase1_layout_short
        MQ_p1 = _phase1_layout_step.MQ_LEN  # = (K_step+1) × dfo
        # When skip_proxy debug gate is on, proxy_count=0 → use placeholder
        # MQ_proxy=1 to avoid div-by-zero. Mask logic gates on in_bp_region
        # which requires kv_pos >= b_proxy_base AND < b_proxy_base + K2*MQ_proxy;
        # with MQ_proxy=1, K2*MQ_proxy=K2 = small range, but in_bp_region
        # filter still works for proxy-row visibility (which is empty anyway).
        MQ_proxy = max(proxy_count, 1)  # dynamic = sum(fan_out_list) per Policy A/B

        # 1) per_row_region_id (0=cont, 1=proxy)
        plan.per_row_region_id[:cont_count] = 0
        plan.per_row_region_id[cont_count:total] = 1

        # 2) per_row_valid_source_kind (cache-emit valid_k per row)
        plan.per_row_valid_source_kind[:cont_count] = K_long
        plan.per_row_valid_source_kind[cont_count:total] = K2  # = K_short

        # 3) per_row_block_tables — per-row contract preserved. Current config
        # collapses to identical fill (5-region fits in seq's first scratch
        # pages). Per-row hybrid path will use this with mask + context_lens
        # to express semantic differences.
        seq_dbt = draft_tree_args["block_tables"]  # [B, num_blocks_seq]
        seq_bt = seq_dbt[0]  # B=1
        n_pages_seq = seq_bt.shape[0]
        n_pages_filled = min(n_pages_seq, plan.max_pages_per_row)
        # Broadcast seq's block table to every plan row
        plan.per_row_block_tables[:total, :n_pages_filled] = (
            seq_bt[:n_pages_filled].view(1, -1).expand(total, n_pages_filled).to(torch.int32)
        )
        if n_pages_filled < plan.max_pages_per_row:
            plan.per_row_block_tables[:total, n_pages_filled:] = -1

        # 4) Reconstruct num_tokens (B=1 invariant) from draft positions.
        # draft_tree_args["positions"][0] = num_tokens - 1 + (K_step+1) + 0
        # ⇒ num_tokens = draft_tree_args["positions"][0] - K_step
        # Phase 9C: cache as `plan.step_num_tokens` (single .item() per step;
        # avoids K2 .item() calls in run_phase2_hybrid_cudagraph's depth loop).
        p1_pos0 = int(draft_tree_args["positions"][0].item())
        num_tokens = p1_pos0 - K_step
        plan.step_num_tokens = num_tokens

        # 5) cont_initial_positions / cont_initial_rope_positions.
        # PHYSICAL DECODE COORDINATE in seq's expanding token space:
        #   A_tail base = num_tokens + K_step + K1*MQ_p1
        # where K_step = valid_k (this step's bucket), MQ_p1 = (K_step+1)*dfo.
        # depth d adds d*MQ_p1. Phase 9D: re-use plan-level arange.
        cont_idx = plan.cont_idx_arange[:cont_count]
        plan.cont_initial_positions[:cont_count] = (
            num_tokens + K_step + K1 * MQ_p1 + cont_idx
        )
        # LOGICAL rope: num_tokens + j_idx_p1[i] + K1 (extends past Phase 1's
        # K1 ancestors of branch i forked at glue position j_idx_p1[i]).
        cache_hits = draft_tree_args["cache_hits"]
        plan.step_cache_hit = bool(int(cache_hits[0]))
        j_idx_p1_full = (
            _phase1_layout_step.fan_idx_hit if int(cache_hits[0])
            else _phase1_layout_step.fan_idx_miss
        )
        plan.cont_initial_rope_positions[:cont_count] = (
            num_tokens + j_idx_p1_full[:cont_count] + K1
        )

        # 6) proxy_initial_positions / proxy_initial_rope_positions.
        # B_proxy base = num_tokens + K_step + K_long*MQ_p1 (placed AFTER
        # Phase 1's K1*MQ_p1 + cont's K2*MQ_p1 = K_long*MQ_p1 worth of slots
        # in the A_tail region of the 5-region scratch). depth d adds
        # d*MQ_proxy. Phase 9D: re-use plan-level arange.
        proxy_idx = plan.proxy_idx_arange[:proxy_count]
        plan.proxy_initial_positions[:proxy_count] = (
            num_tokens + K_step + K_long * MQ_p1 + proxy_idx
        )
        # proxy_initial_rope_positions[j] is the LOGICAL branch position for
        # RoPE: num_tokens + j_idx_proxy[j] (j_idx_proxy from step_proxy_layout's
        # dynamic fan_idx — Policy A/B preserved).
        proxy_j_idx = (
            step_proxy_layout.fan_idx_hit if int(cache_hits[0])
            else step_proxy_layout.fan_idx_miss
        )
        plan.proxy_initial_rope_positions[:proxy_count] = (
            num_tokens + proxy_j_idx[:proxy_count]
        )

        _mc_b("phase2_hybrid_build_setup", _mev_b_setup)

        # 7) per_row_slot_maps_by_depth — row-wise direct compute.
        # Cont row i at depth d: scratch_pos = cont_initial[i] + d * MQ_p1
        # Proxy row j at depth d: scratch_pos = proxy_initial[j] + d * MQ_proxy
        _mev_b_slots = _mr_b("phase2_hybrid_build_slots")
        for d in range(K2):
            cont_scratch_pos = plan.cont_initial_positions[:cont_count] + d * MQ_p1
            cont_block_id = (cont_scratch_pos // block_size).to(torch.int64)
            cont_offset = (cont_scratch_pos % block_size).to(torch.int32)
            plan.per_row_slot_maps_by_depth[:cont_count, d] = (
                seq_bt[cont_block_id].to(torch.int32) * block_size + cont_offset
            )

            proxy_scratch_pos = plan.proxy_initial_positions[:proxy_count] + d * MQ_proxy
            proxy_block_id = (proxy_scratch_pos // block_size).to(torch.int64)
            proxy_offset = (proxy_scratch_pos % block_size).to(torch.int32)
            plan.per_row_slot_maps_by_depth[cont_count:total, d] = (
                seq_bt[proxy_block_id].to(torch.int32) * block_size + proxy_offset
            )

        _mc_b("phase2_hybrid_build_slots", _mev_b_slots)

        # 8) per_row_context_lens_by_depth + Phase 9C per-depth caches.
        # physical_end = num_tokens + K_step + K_long*MQ_p1 + (d+1)*MQ_proxy.
        # Pre-compute Python ints for L / n_pages to eliminate K2×.item() syncs
        # in the hot path runtime.
        _mev_b_kv = _mr_b("phase2_hybrid_build_kv")
        per_depth_L = []
        per_depth_n_pages = []
        for d in range(K2):
            ctx_len_d = num_tokens + K_step + K_long * MQ_p1 + (d + 1) * MQ_proxy
            plan.per_row_context_lens_by_depth[:total, d] = ctx_len_d
            per_depth_L.append(ctx_len_d)
            n_pages_d_step = (ctx_len_d + block_size - 1) // block_size
            n_pages_d_step = min(n_pages_d_step, n_pages_filled)
            per_depth_n_pages.append(n_pages_d_step)
        plan.per_depth_L = per_depth_L
        plan.per_depth_n_pages = per_depth_n_pages

        # 9) per_row_kv_indptr / kv_indices — shared page list across rows
        # (current 1-page collapse). Per-row contract preserved.
        # Phase 9C: also pre-build CPU kv_indptr and GPU kv_lens / kv_lpl per
        # depth so runtime can low_level_packed_plan() without any per-depth
        # tensor allocations or CPU↔GPU transfers in the hot path.
        # Phase 9D: re-use plan.total_arange_p1 instead of fresh torch.arange.
        seq_bt_int32 = seq_bt[:n_pages_filled].to(torch.int32)
        for d in range(K2):
            n_pages_d_step = per_depth_n_pages[d]
            ctx_len_d_int = per_depth_L[d]
            n_idx = total * n_pages_d_step
            # GPU indptr — write directly into plan with mul out=.
            torch.mul(
                plan.total_arange_p1[:total + 1], n_pages_d_step,
                out=plan.per_row_kv_indptr_by_depth[:total + 1, d],
            )
            plan.per_row_kv_indices_by_depth[d, :n_idx] = (
                seq_bt_int32[:n_pages_d_step].repeat(total)
            )
            # CPU pinned indptr — multiply pre-built CPU arange by stride.
            indptr_cpu = plan.per_depth_kv_indptr_cpu[d]
            torch.mul(
                plan.total_arange_p1_cpu[:total + 1], n_pages_d_step,
                out=indptr_cpu[:total + 1],
            )
            indptr_cpu[total + 1 :] = total * n_pages_d_step
            # kv_lens = (num_pages - 1) * block_size + last_page_len
            # last_page_len = ((L - 1) % block_size) + 1
            lpl = ((ctx_len_d_int - 1) % block_size) + 1
            kv_len_val = (n_pages_d_step - 1) * block_size + lpl
            plan.per_depth_kv_lpl_gpu[d, :].fill_(lpl)
            plan.per_depth_kv_lens_gpu[d, :].fill_(kv_len_val)

        # step_pos_offsets / step_rope_offsets remain as set by create() —
        # depth d → arange(K2)[d]. These are not used directly for slot
        # mapping (different stride for cont vs proxy); per_row_slot_maps_by_depth
        # is the source of truth for write slots in Step 6.
        _mc_b("phase2_hybrid_build_kv", _mev_b_kv)

        # Step 4: build per-depth packed mask consuming the metadata above.
        # K_step = this step's valid_k = phase1_kv_base offset.
        _mev_b_mask = _mr_b("phase2_hybrid_build_mask")
        self._build_hybrid_packed_mask_inplace(
            num_tokens=num_tokens,
            K1=K1, K2=K2, K_step=K_step, K_long=K_long,
            MQ_p1=MQ_p1, MQ_proxy=MQ_proxy,
            cont_count=cont_count, proxy_count=proxy_count, total=total,
            j_idx_p1=j_idx_p1_full[:cont_count],
            j_idx_proxy=proxy_j_idx[:proxy_count],
        )
        _mc_b("phase2_hybrid_build_mask", _mev_b_mask)

    def _build_hybrid_packed_mask_inplace(self, *, num_tokens, K1, K2, K_step,
                                            K_long, MQ_p1, MQ_proxy, cont_count,
                                            proxy_count, total,
                                            j_idx_p1, j_idx_proxy):
        """Fill HybridPhase2Plan.per_depth_packed_masks +
        per_depth_mask_indptr in-place, with per-row j_idx-aware visibility.

        Build-only. Reference docstring; runtime hybrid forward uses an
        inline bool mask (`_compute_hybrid_bool_mask_for_depth`) since
        FlashInfer's high-level plan() takes BOOL not bit-packed.

        Per-row visibility (per reviewer):
        - continuation row i at depth d:
            * persistent (kv_pos < num_tokens) visible
            * glue prefix [num_tokens, num_tokens + j_idx_p1[i] + 1) visible
              (lower-triangular: glue position p visible iff p ≤ j_idx_p1[i])
            * own Phase 1 KV K1 ancestors visible (depth-major slots
              phase1_kv_base + d_p1*MQ_p1 + i for d_p1 in [0,K1))
            * own A_tail prefix visible (slots a_tail_base + (K1+d_at)*MQ_p1
              + i for d_at in [0, d+1) — including current depth's write,
              causal self)
            * everything else hidden (other rows' Phase 1 KV / A_tail / all
              B_proxy / other glue tail / other rows' glue tail)
        - proxy row j at depth d:
            * persistent visible
            * glue prefix [num_tokens, num_tokens + j_idx_proxy[j] + 1) visible
            * own B_proxy prefix visible (slots b_proxy_base + d_b*MQ_proxy
              + j for d_b in [0, d+1))
            * everything else hidden (all Phase 1 KV, all A_tail, other
              proxy rows' B_proxy, other rows' glue tail)

        FlashInfer packed_mask convention: bits LSB-first within each byte;
        per-query mask length is padded to multiple of 8 bits, then packed.
        per_depth_mask_indptr[d, i] = byte offset of row i's mask within
        per_depth_packed_masks[d].
        """
        plan = self.hybrid_phase2_plan
        device = self.device

        # K_step = this step's bucket (K_long or K_short). phase1_kv_base
        # offset comes from glue end = num_tokens + K_step (per Step 9A).
        glue_base = num_tokens - 1
        phase1_kv_base = num_tokens + K_step
        a_tail_base = phase1_kv_base + K1 * MQ_p1
        b_proxy_base = a_tail_base + K2 * MQ_p1

        # Phase 9D: re-use plan-level pre-allocated workspaces instead of
        # allocating fresh tensors per step / per depth. cont_idx_arange /
        # proxy_idx_arange were allocated max-size at create() time.
        cont_idx_t = plan.cont_idx_arange[:cont_count]
        proxy_idx_t = plan.proxy_idx_arange[:proxy_count]
        bit_weights = plan.bit_weights_u8     # [8] constant
        full_mask_buf = plan.full_mask_buf    # [max_total, max_L_padded] bool

        # Per-row j_idx as column vectors for broadcasting (no per-depth alloc).
        j_idx_p1_col = j_idx_p1.view(cont_count, 1)
        j_idx_proxy_col = j_idx_proxy.view(proxy_count, 1)
        cont_idx_col = cont_idx_t.view(cont_count, 1)
        proxy_idx_col = proxy_idx_t.view(proxy_count, 1)

        # Phase 9C: cache bytes_per_row per depth so runtime can slice plan
        # buffer directly without re-running the bool-mask + pack work.
        per_depth_bytes_per_row: list[int] = []

        for d in range(K2):
            # Physical KV upper bound at depth d (after current depth's write).
            L = num_tokens + K_step + K_long * MQ_p1 + (d + 1) * MQ_proxy
            L_padded = ((L + 7) // 8) * 8
            bytes_per_row = L_padded // 8

            # Phase 9D: slice the pre-built kv_pos arange instead of allocating
            # a fresh torch.arange(L) per depth. View [1, L] for broadcasting.
            kv_pos_2d = plan.kv_pos_arange[:L].view(1, L)

            # Phase 9D: write directly into pre-allocated full_mask_buf rows.
            # Layout: rows [0:cont_count) = cont rows; rows [cont_count:total)
            # = proxy rows. Pad columns [L:L_padded) zeroed. Skip torch.cat.
            mask_out = full_mask_buf[:total, :L_padded]
            # Zero out the active region (workspace is reused step-to-step).
            mask_out.zero_()

            cont_out = full_mask_buf[:cont_count, :L]      # [cont_count, L] view
            proxy_out = full_mask_buf[cont_count:total, :L]  # [proxy_count, L] view

            # persistent visible: kv_pos < num_tokens (broadcast 1×L → row×L)
            persistent_vis = kv_pos_2d < num_tokens  # [1, L]
            # Use copy_ + broadcast so we write into the buf in place.
            cont_out.copy_(persistent_vis.expand(cont_count, L))
            proxy_out.copy_(persistent_vis.expand(proxy_count, L))

            # ---- continuation rows ----
            # glue prefix lower-triangular: kv_pos in [glue_base, glue_base + j_idx + 1)
            cont_glue_vis = (
                (kv_pos_2d >= glue_base) &
                (kv_pos_2d <= (glue_base + j_idx_p1_col))
            )  # [cont_count, L]
            cont_out.bitwise_or_(cont_glue_vis)

            # own Phase 1 KV: kv_pos in Phase 1 region AND
            # (kv_pos - phase1_kv_base) % MQ_p1 == cont_idx
            p1_off = kv_pos_2d - phase1_kv_base
            in_p1_region = (kv_pos_2d >= phase1_kv_base) & (kv_pos_2d < a_tail_base)
            own_p1_branch = (p1_off % MQ_p1) == cont_idx_col
            cont_out.bitwise_or_(in_p1_region & own_p1_branch)

            # own A_tail prefix [0..d] (current depth INCLUSIVE for causal)
            at_off = kv_pos_2d - a_tail_base
            in_at_region = (kv_pos_2d >= a_tail_base) & (kv_pos_2d < b_proxy_base)
            own_at_branch = (at_off % MQ_p1) == cont_idx_col
            at_depth = at_off // MQ_p1
            at_depth_ok = at_depth <= d
            cont_out.bitwise_or_(in_at_region & own_at_branch & at_depth_ok)

            # ---- proxy rows ----
            proxy_glue_vis = (
                (kv_pos_2d >= glue_base) &
                (kv_pos_2d <= (glue_base + j_idx_proxy_col))
            )
            proxy_out.bitwise_or_(proxy_glue_vis)

            # own B_proxy prefix [0..d]
            bp_off = kv_pos_2d - b_proxy_base
            in_bp_region = (kv_pos_2d >= b_proxy_base) & (kv_pos_2d < (b_proxy_base + K2 * MQ_proxy))
            own_bp_branch = (bp_off % MQ_proxy) == proxy_idx_col
            bp_depth = bp_off // MQ_proxy
            bp_depth_ok = bp_depth <= d
            proxy_out.bitwise_or_(in_bp_region & own_bp_branch & bp_depth_ok)

            # ---- pack ----
            # mask_out is [total, L_padded] bool with L_padded a multiple of 8.
            # Pack 8 consecutive bools per byte, LSB-first.
            packed = mask_out.to(torch.uint8).view(total, bytes_per_row, 8)
            packed_bytes = (packed * bit_weights.view(1, 1, 8)).sum(dim=2).to(torch.uint8)

            # Write to plan tensor (depth-major). Phase 9C: zero padding so
            # runtime can pass the slice directly.
            flat = packed_bytes.reshape(-1)
            n_bytes = total * bytes_per_row
            assert n_bytes <= plan.per_depth_packed_masks.shape[1], (
                f"packed mask depth {d} needs {n_bytes} bytes but plan has "
                f"{plan.per_depth_packed_masks.shape[1]} per-depth slots"
            )
            plan.per_depth_packed_masks[d, :n_bytes] = flat
            max_total_bytes = plan.per_depth_packed_masks.shape[1]
            if n_bytes < max_total_bytes:
                plan.per_depth_packed_masks[d, n_bytes:].zero_()

            # Phase 9D: indptr from pre-built total_arange_p1 (no fresh arange).
            torch.mul(
                plan.total_arange_p1[:total + 1], bytes_per_row,
                out=plan.per_depth_mask_indptr[d, :total + 1],
            )
            plan.per_depth_mask_indptr[d, total + 1 :] = total * bytes_per_row

            per_depth_bytes_per_row.append(bytes_per_row)

        # Phase 9C: expose per-depth bytes_per_row to the runtime caller.
        plan.per_depth_bytes_per_row = per_depth_bytes_per_row

    @torch.inference_mode()
    def _split_eager_mirror_proxy(self, *, proxy_tree_args, step_proxy_layout,
                                    wrapper_key="proxy"):
        """Phase A mirror: rerun split's proxy decode in eager bool-mask mode.

        Same proxy semantics as split CG path (same proxy_tree_args, same
        step_proxy_layout, same wrapper instance, same get_custom_mask helper)
        but uses HIGH-LEVEL `wrapper.plan(custom_mask=bool, ...)` API and
        direct `self.model(...)` call instead of low-level
        `wrapper._custom_mask_buf write + _cached_module.plan(...)` + captured
        CG replay.

        Run AFTER split's proxy completes — KV state at base region has
        split's proxy KV. Mirror reads/writes the same scratch slots; final
        KV content is identical (deterministic).

        Returns: (spec_tokens [N, K2], spec_logits [N, K2, V], mirror_meta dict)
        """
        # NOTE: get_custom_mask_cached uses (K, fan_out_list) as cache key.
        # Split CG's run_fi_tree_decode_cudagraph builds masks with K =
        # config.speculate_k (= K_long) and fan_out_list = layout.fan_out_list
        # (dynamic Policy A). Mirror must match.
        from ssd.engine.helpers.mask_helpers import get_custom_mask_cached

        K2 = step_proxy_layout.K
        MQ_proxy = step_proxy_layout.MQ_LEN
        cache_hits = proxy_tree_args["cache_hits"]
        B, K_layout, F_layout, N = proxy_tree_args["metadata_ints"]
        assert K_layout == K2, f"layout K {K_layout} != K2 {K2}"

        wrapper_dict = self.prefill_wrappers_by_layout[wrapper_key]
        wrapper_bs = next(x for x in wrapper_dict if x >= B)
        wrapper = wrapper_dict[wrapper_bs]

        initial_positions = proxy_tree_args["positions"]
        initial_rope_positions = proxy_tree_args["rope_positions"]
        dbt = proxy_tree_args["block_tables"]

        _, step_rope_positions, step_context_lens, step_slot_maps = (
            self._compute_step_positions_and_slot_maps(
                initial_positions, initial_rope_positions, dbt, B, K2,
                F_layout, N, MQ_proxy, layout=step_proxy_layout
            )
        )

        V = self.hf_config.vocab_size
        spec_tokens = torch.empty(N, K2, dtype=torch.int64, device=self.device)
        spec_logits = torch.empty(N, K2, V, dtype=self.hf_config.torch_dtype, device=self.device)

        # Capture per-depth metadata for parity dump
        mirror_meta = {
            "step_rope_positions": step_rope_positions,
            "step_context_lens": step_context_lens,
            "step_slot_maps": step_slot_maps,
            "kv_indptr_per_d": [],
            "kv_indices_per_d": [],
            "kv_lpl_per_d": [],
            "bool_mask_per_d": [],
        }

        current_input_ids = proxy_tree_args["input_ids"]

        # Step 9B-3: K_full = layout-derived (= position_count - 1). For long
        # bucket = K_long; for short bucket = K_short. Was hardcoded to
        # speculate_k pre-9B-3 which broke short bucket (fan_out_list
        # length mismatch in mask helper).
        K_full = step_proxy_layout.position_count - 1
        for d in range(K2):
            # Bool mask: use K_full + layout.fan_out_list (= split CG's path)
            bool_mask = get_custom_mask_cached(
                self.config, step_context_lens[d], d, K_full, F_layout, B,
                device=self.device,
                fan_out_list=step_proxy_layout.fan_out_list,
                fan_out_list_miss=step_proxy_layout.fan_out_list_miss,
                cache_hits=cache_hits,
            )

            # FlashInfer plan inputs (mirrors eager_tree_decode_plan logic)
            block_tables = dbt
            context_lens = step_context_lens[d]
            counts = (context_lens + self.block_size - 1) // self.block_size
            kv_indptr = torch.cat([
                torch.tensor([0], device=block_tables.device),
                counts.cumsum(0),
            ]).to(torch.int32)
            mask_pages = torch.arange(block_tables.size(1), device=block_tables.device)[None, :] < counts[:, None]
            kv_indices = block_tables[mask_pages]
            kv_last_page_len = (context_lens % self.block_size)
            kv_last_page_len[kv_last_page_len == 0] = self.block_size
            kv_last_page_len = kv_last_page_len.to(torch.int32)
            cu_seqlens_q = torch.arange(B + 1, device=self.device, dtype=torch.int32) * MQ_proxy

            wrapper.plan(
                cu_seqlens_q, kv_indptr, kv_indices, kv_last_page_len,
                self.hf_config.num_attention_heads,
                self.hf_config.num_key_value_heads,
                self.hf_config.head_dim, self.block_size,
                custom_mask=bool_mask,
                q_data_type=self.hf_config.torch_dtype,
                kv_data_type=self.hf_config.torch_dtype,
            )

            mirror_meta["kv_indptr_per_d"].append(kv_indptr.clone())
            mirror_meta["kv_indices_per_d"].append(kv_indices.clone())
            mirror_meta["kv_lpl_per_d"].append(kv_last_page_len.clone())
            mirror_meta["bool_mask_per_d"].append(bool_mask.clone())

            # CRITICAL: cu_seqlens_q MUST be None to route through attention.py's
            # tree_decode branch (which uses prefill_wrapper.run with our custom
            # mask). If cu_seqlens_q is set, attention.py routes to
            # flash_attn_with_kvcache (verify_or_glue branch) which IGNORES the
            # custom_mask and does only causal attention — wrong for tree decode.
            set_context(
                is_prefill=False,
                slot_mapping=step_slot_maps[d],
                context_lens=step_context_lens[d].to(torch.int32),
                block_tables=dbt,
                cu_seqlens_q=None,
                max_seqlen_q=0,
                active_mq_len=MQ_proxy,
                active_wrappers=wrapper_dict,
                active_layout=step_proxy_layout,
            )

            outputs = self.model(current_input_ids, step_rope_positions[d])
            logits = self.model.compute_logits(outputs, last_only=False)
            reset_context()

            logits_flat = logits.view(N, V)
            spec_logits[:, d, :] = logits_flat
            next_tokens = logits_flat.argmax(dim=-1)
            spec_tokens[:, d] = next_tokens
            current_input_ids = next_tokens

        return spec_tokens, spec_logits, mirror_meta

    def _split_low_level_mirror_proxy(self, *, proxy_tree_args, step_proxy_layout,
                                        wrapper_key="proxy"):
        # Explicit no_grad + inference_mode wrapper (decorator path was failing
        # with autograd backward inside model forward via Inductor).
        with torch.inference_mode(), torch.no_grad():
            return self._split_low_level_mirror_proxy_impl(
                proxy_tree_args=proxy_tree_args,
                step_proxy_layout=step_proxy_layout,
                wrapper_key=wrapper_key,
            )

    def _split_low_level_mirror_proxy_impl(self, *, proxy_tree_args, step_proxy_layout,
                                              wrapper_key="proxy"):
        """Phase B-2 low-level mirror: rerun split's proxy decode using the
        EXACT same low-level packed planning path as split CG, but in eager
        mode (no CG capture/replay).

        Should produce IDENTICAL outputs to split CG if low-level setup is
        faithful regardless of CG capture mode. Confirms Phase A finding
        that planning path was the bug.

        Returns: (spec_tokens [N, K2], spec_logits [N, K2, V], meta dict)
        """
        from ssd.engine.helpers.cudagraph_helpers import (
            low_level_packed_plan, build_packed_mask_for_proxy_step,
        )

        K2 = step_proxy_layout.K
        K_full = self.config.speculate_k  # K_long
        MQ_proxy = step_proxy_layout.MQ_LEN
        cache_hits = proxy_tree_args["cache_hits"]
        B, K_layout, F_layout, N = proxy_tree_args["metadata_ints"]
        assert K_layout == K2

        wrapper_dict = self.prefill_wrappers_by_layout[wrapper_key]
        wrapper_bs = next(x for x in wrapper_dict if x >= B)
        wrapper = wrapper_dict[wrapper_bs]

        initial_positions = proxy_tree_args["positions"]
        initial_rope_positions = proxy_tree_args["rope_positions"]
        dbt = proxy_tree_args["block_tables"]

        _, step_rope_positions, step_context_lens, step_slot_maps = (
            self._compute_step_positions_and_slot_maps(
                initial_positions, initial_rope_positions, dbt, B, K2,
                F_layout, N, MQ_proxy, layout=step_proxy_layout
            )
        )

        V = self.hf_config.vocab_size
        spec_tokens = torch.empty(N, K2, dtype=torch.int64, device=self.device)
        spec_logits = torch.empty(N, K2, V, dtype=self.hf_config.torch_dtype, device=self.device)

        meta = {
            "step_rope_positions": step_rope_positions,
            "step_context_lens": step_context_lens,
            "step_slot_maps": step_slot_maps,
        }

        current_input_ids = proxy_tree_args["input_ids"]

        # cache_hits_list / context_lens_list for packed mask builder
        cache_hits_list = cache_hits.tolist()
        # Initial context_lens = step_context_lens[0] - 0*MQ_proxy (= the
        # base seq context length pre-spec)
        # Actually run_fi_tree_decode_cudagraph uses context.context_lens
        # (= initial seq length, not step-shifted). step s adds s*MQ_LEN.
        # Match by using context_lens[0] as base.
        initial_ctx_lens = step_context_lens[0] - 0 * MQ_proxy
        # Wait — split CG's step 0 cache uses `context_lens.tolist()` from
        # context, which is the seq's initial spec scratch context length
        # (= num_tokens + K_long for our config). This INCLUDES glue but
        # NOT spec. step s adds s*MQ_LEN to get cols_b at depth s.
        # step_context_lens[d] from _compute_step_positions_and_slot_maps =
        # (initial_positions[..., -1] + d*MQ_LEN) + 1 = base + d*MQ_LEN.
        # So initial = step_context_lens[0] - 0*MQ_LEN matches.
        # Hmm but build_packed_mask_for_proxy_step uses
        # cols_b = context_lens[b] + s*MQ_LEN, where context_lens is the
        # *initial* (passed in once). Let's compute initial_context as
        # step_context_lens[0] (which is base + 0*MQ_LEN = base).
        base_ctx_lens = step_context_lens[0].tolist()  # B-sized
        # But build_packed_mask wants context_lens_list = the BASE seq len
        # (without step shift). Looking at run_fi_tree_decode line 280:
        # `step_cls = [int(cl) + s * MQ_LEN for cl in context_lens_list]`
        # so context_lens_list is raw context.context_lens.tolist() pre-step.
        # That equals num_tokens + K_long for our hybrid config.
        # step_context_lens[0] = base + 0*MQ_LEN = base. So base_ctx_lens
        # is the right input for context_lens_list.

        for d in range(K2):
            # Build packed mask for this step (numpy path identical to split CG)
            packed_mask, packed_indptr = build_packed_mask_for_proxy_step(
                MQ_LEN=MQ_proxy, K=K_full, B=B,
                context_lens_list=base_ctx_lens,
                cache_hits_list=cache_hits_list,
                fan_out_list=step_proxy_layout.fan_out_list,
                fan_out_list_miss=step_proxy_layout.fan_out_list_miss,
                step=d, device=self.device,
            )

            # Plan inputs (same logic as run_fi_tree_decode step 0 cache)
            block_tables = dbt
            context_lens = step_context_lens[d]
            counts = (context_lens + self.block_size - 1) // self.block_size
            kv_indptr_cpu = torch.zeros(B + 1, dtype=torch.int32)
            kv_indptr_cpu[1:] = counts.cpu().to(torch.int32).cumsum(0)
            mask_pages = torch.arange(block_tables.size(1), device=block_tables.device)[None, :] < counts[:, None]
            kv_indices = block_tables[mask_pages].to(torch.int32)
            kv_last_page_len = (context_lens % self.block_size)
            kv_last_page_len[kv_last_page_len == 0] = self.block_size
            kv_last_page_len = kv_last_page_len.to(torch.int32)
            qo_indptr_cpu = torch.arange(B + 1, dtype=torch.int32) * MQ_proxy
            kv_lens_gpu = ((kv_indptr_cpu[1:].to(self.device) - kv_indptr_cpu[:-1].to(self.device) - 1)
                           * self.block_size + kv_last_page_len).to(torch.int32)

            low_level_packed_plan(
                wrapper, model_runner=self,
                qo_indptr_cpu=qo_indptr_cpu, kv_indptr_cpu=kv_indptr_cpu,
                kv_indices=kv_indices, kv_last_page_len_gpu=kv_last_page_len,
                kv_lens_gpu=kv_lens_gpu,
                packed_mask=packed_mask, packed_indptr=packed_indptr, B=B,
            )

            # cu_seqlens_q=None → attention.py routes to tree_decode branch
            # which uses prefill_wrapper.run() with our planned custom mask.
            # If we set cu_seqlens_q, it routes to flash_attn_with_kvcache
            # which IGNORES the custom mask.
            set_context(
                is_prefill=False,
                slot_mapping=step_slot_maps[d],
                context_lens=context_lens.to(torch.int32),
                block_tables=dbt,
                cu_seqlens_q=None,
                max_seqlen_q=0,
                active_mq_len=MQ_proxy,
                active_wrappers=wrapper_dict,
                active_layout=step_proxy_layout,
            )

            outputs = self.model(current_input_ids, step_rope_positions[d])
            logits = self.model.compute_logits(outputs, last_only=False)
            reset_context()

            logits_flat = logits.view(N, V)
            spec_logits[:, d, :] = logits_flat
            next_tokens = logits_flat.argmax(dim=-1)
            spec_tokens[:, d] = next_tokens
            current_input_ids = next_tokens

        return spec_tokens, spec_logits, meta

    def _mirror_parity_diff(self, *, split_proxy_tokens, split_proxy_logits,
                              mirror_tokens, mirror_logits, mirror_meta):
        """Diff split CG vs split eager mirror (proxy-only).

        First mismatch row/depth dump:
        - tokens
        - logits max_abs_diff
        - rope (mirror_meta provides; split CG would compute identically)
        - slot_mapping (same)
        - context_len (same)
        - kv_indptr / kv_indices / kv_last_page_len (mirror_meta vs split CG)
        - bool mask visible positions (mirror's bool mask)
        """
        tok_eq = (split_proxy_tokens == mirror_tokens)
        n_total = tok_eq.numel()
        n_mismatch = int((~tok_eq).sum().item())
        diff = (split_proxy_logits - mirror_logits).abs()
        max_diff = float(diff.max().item())
        mean_diff = float(diff.mean().item())

        print(f"[MIRROR PARITY] split CG vs split eager mirror: tokens "
              f"{n_mismatch}/{n_total} mismatch ({100*n_mismatch/max(n_total,1):.1f}%); "
              f"logits max_abs_diff={max_diff:.6f} mean_abs_diff={mean_diff:.6f}",
              flush=True)

        if n_mismatch > 0:
            first_idx = (~tok_eq).nonzero(as_tuple=False)[0]
            row = int(first_idx[0].item())
            depth = int(first_idx[1].item())
            split_tok = int(split_proxy_tokens[row, depth].item())
            mirror_tok = int(mirror_tokens[row, depth].item())

            rope = int(mirror_meta["step_rope_positions"][depth, row].item())
            slot = int(mirror_meta["step_slot_maps"][depth, row].item())
            ctx_len = int(mirror_meta["step_context_lens"][depth, row].item())
            kv_indptr = mirror_meta["kv_indptr_per_d"][depth].tolist()
            kv_indices = mirror_meta["kv_indices_per_d"][depth].tolist()
            kv_lpl = mirror_meta["kv_lpl_per_d"][depth].tolist()
            bool_mask = mirror_meta["bool_mask_per_d"][depth]
            # bool_mask is 1D flat over all queries × kv positions
            # Per-row visibility for this query is encoded by FlashInfer's
            # interpretation; we just dump count of True bits in mask
            n_mask_true = int(bool_mask.sum().item())

            print(f"[MIRROR PARITY] first mismatch row={row} depth={depth}", flush=True)
            print(f"  tokens: split_CG={split_tok} mirror_eager={mirror_tok}", flush=True)
            print(f"  rope={rope} slot={slot} ctx_len={ctx_len}", flush=True)
            print(f"  kv_indptr={kv_indptr} kv_indices_len={len(kv_indices)} kv_lpl={kv_lpl}", flush=True)
            print(f"  bool_mask total True={n_mask_true} dtype={bool_mask.dtype} numel={bool_mask.numel()}",
                  flush=True)
        else:
            print(f"[MIRROR PARITY] PASSES — split CG path == split eager mirror path. "
                  f"bool eager planning is faithful. "
                  f"Hybrid mismatch is in metadata/mask, not planning path.", flush=True)

    def _oracle_parity_report(self, *,
                                split_cont_tokens, split_cont_logits,
                                split_proxy_tokens, split_proxy_logits,
                                hybrid_cont_tokens, hybrid_cont_logits,
                                hybrid_proxy_tokens, hybrid_proxy_logits,
                                oracle_cont_tokens, oracle_cont_logits,
                                oracle_proxy_tokens, oracle_proxy_logits,
                                oracle_proxy_meta, step_proxy_layout,
                                oracle_proxy_tokens_2=None, oracle_proxy_logits_2=None,
                                hybrid_proxy_tokens_2=None, hybrid_proxy_logits_2=None,
                                hybrid_cont_tokens_2=None, hybrid_cont_logits_2=None):
        """Phase D-5: oracle-aware parity output.

        Two primary lines (gated by SSD_HYBRID_*_ORACLE):
          [CONT ORACLE]  hybrid_cont  vs correct_split_cont
          [PROXY ORACLE] hybrid_proxy vs fresh_proxy_oracle

        Diagnostics (split direct comparisons):
          [diag] split_CG_cont   vs correct_split_cont
          [diag] split_CG_proxy  vs fresh_proxy_oracle
          [diag] hybrid_proxy    vs split_CG_proxy
          [diag] hybrid_cont     vs split_CG_cont

        First-mismatch detail dump for [PROXY ORACLE] when nonzero.
        """
        def _fmt(a_tok, a_log, b_tok, b_log):
            eq = (a_tok == b_tok)
            n = eq.numel()
            nm = int((~eq).sum().item())
            d = (a_log - b_log).abs()
            return (nm, n,
                    f"tokens {nm}/{n} ({100*nm/max(n,1):.1f}%); "
                    f"logits max_abs={float(d.max().item()):.6f} "
                    f"mean_abs={float(d.mean().item()):.6f}")

        # Step 9B-3: tag bucket (long/short) so post-process can aggregate.
        plan = self.hybrid_phase2_plan
        bucket_tag = "long" if plan.valid_k == plan.K_long else "short"

        # Primary: cont oracle
        cont_oracle_mismatch = 0
        if oracle_cont_tokens is not None:
            nm_c, _, line = _fmt(
                hybrid_cont_tokens, hybrid_cont_logits,
                oracle_cont_tokens, oracle_cont_logits,
            )
            cont_oracle_mismatch = nm_c
            print(f"[CONT ORACLE {bucket_tag}]  hybrid_cont  vs correct_split_cont: {line}",
                  flush=True)

        # Primary: proxy oracle
        proxy_oracle_mismatch = 0
        if oracle_proxy_tokens is not None:
            nm, _, line = _fmt(
                hybrid_proxy_tokens, hybrid_proxy_logits,
                oracle_proxy_tokens, oracle_proxy_logits,
            )
            proxy_oracle_mismatch = nm
            print(f"[PROXY ORACLE {bucket_tag}] hybrid_proxy vs fresh_proxy_oracle: {line}",
                  flush=True)

        # Diagnostics
        if oracle_cont_tokens is not None:
            _, _, line = _fmt(
                split_cont_tokens, split_cont_logits,
                oracle_cont_tokens, oracle_cont_logits,
            )
            print(f"[diag] split_CG_cont   vs correct_split_cont: {line}",
                  flush=True)
        if oracle_proxy_tokens is not None:
            _, _, line = _fmt(
                split_proxy_tokens, split_proxy_logits,
                oracle_proxy_tokens, oracle_proxy_logits,
            )
            print(f"[diag] split_CG_proxy  vs fresh_proxy_oracle: {line}",
                  flush=True)
            _, _, line = _fmt(
                hybrid_proxy_tokens, hybrid_proxy_logits,
                split_proxy_tokens, split_proxy_logits,
            )
            print(f"[diag] hybrid_proxy    vs split_CG_proxy:    {line}",
                  flush=True)
        _, _, line = _fmt(
            hybrid_cont_tokens, hybrid_cont_logits,
            split_cont_tokens, split_cont_logits,
        )
        print(f"[diag] hybrid_cont     vs split_CG_cont:     {line}",
              flush=True)

        # Phase D-6: self-consistency lines
        if oracle_proxy_tokens_2 is not None and oracle_proxy_tokens is not None:
            _, _, line = _fmt(
                oracle_proxy_tokens, oracle_proxy_logits,
                oracle_proxy_tokens_2, oracle_proxy_logits_2,
            )
            print(f"[SELF] fresh_proxy_oracle vs fresh_proxy_oracle: {line}",
                  flush=True)
        if hybrid_proxy_tokens_2 is not None:
            _, _, line = _fmt(
                hybrid_proxy_tokens, hybrid_proxy_logits,
                hybrid_proxy_tokens_2, hybrid_proxy_logits_2,
            )
            print(f"[SELF] hybrid_proxy        vs hybrid_proxy:        {line}",
                  flush=True)
        if hybrid_cont_tokens_2 is not None:
            _, _, line = _fmt(
                hybrid_cont_tokens, hybrid_cont_logits,
                hybrid_cont_tokens_2, hybrid_cont_logits_2,
            )
            print(f"[SELF] hybrid_cont         vs hybrid_cont:         {line}",
                  flush=True)

        # Proxy oracle first-mismatch detail
        if proxy_oracle_mismatch > 0 and oracle_proxy_tokens is not None:
            self._dump_proxy_oracle_first_mismatch(
                hybrid_proxy_tokens=hybrid_proxy_tokens,
                hybrid_proxy_logits=hybrid_proxy_logits,
                oracle_proxy_tokens=oracle_proxy_tokens,
                oracle_proxy_logits=oracle_proxy_logits,
                oracle_proxy_meta=oracle_proxy_meta,
                step_proxy_layout=step_proxy_layout,
            )
        # Phase D-7: cont oracle first-mismatch top-1/top-2 margin (same
        # drift-consistency check as proxy).
        if cont_oracle_mismatch > 0 and oracle_cont_tokens is not None:
            self._dump_cont_oracle_first_mismatch(
                hybrid_cont_tokens=hybrid_cont_tokens,
                hybrid_cont_logits=hybrid_cont_logits,
                oracle_cont_tokens=oracle_cont_tokens,
                oracle_cont_logits=oracle_cont_logits,
            )

    def _dump_cont_oracle_first_mismatch(self, *,
                                            hybrid_cont_tokens, hybrid_cont_logits,
                                            oracle_cont_tokens, oracle_cont_logits):
        """Phase D-7: top-1/top-2 margin dump for first cont mismatch.
        Same drift-consistency check as proxy."""
        eq = (hybrid_cont_tokens == oracle_cont_tokens)
        nz = (~eq).nonzero(as_tuple=False)
        if nz.numel() == 0:
            return
        row = int(nz[0, 0].item())
        depth = int(nz[0, 1].item())
        h_tok = int(hybrid_cont_tokens[row, depth].item())
        o_tok = int(oracle_cont_tokens[row, depth].item())
        log_d = float(
            (hybrid_cont_logits[row, depth] - oracle_cont_logits[row, depth])
            .abs().max().item()
        )
        h_top2 = torch.topk(hybrid_cont_logits[row, depth], 2)
        o_top2 = torch.topk(oracle_cont_logits[row, depth], 2)
        h_t1, h_t2 = int(h_top2.indices[0].item()), int(h_top2.indices[1].item())
        h_l1, h_l2 = float(h_top2.values[0].item()), float(h_top2.values[1].item())
        o_t1, o_t2 = int(o_top2.indices[0].item()), int(o_top2.indices[1].item())
        o_l1, o_l2 = float(o_top2.values[0].item()), float(o_top2.values[1].item())
        h_margin = h_l1 - h_l2
        o_margin = o_l1 - o_l2
        # Phase D-7: drift-consistency check. Each vocab position's logit
        # can drift by up to log_d (max abs vocab diff). The MARGIN between
        # any two positions can shift by up to 2*log_d (one side up, the
        # other side down). So if o_margin <= 2*log_d, drift can flip.
        # Tied oracle (o_margin=0) is always flippable by any drift > 0.
        h_at_o_t1 = float(hybrid_cont_logits[row, depth, o_t1].item())
        h_at_o_t2 = float(hybrid_cont_logits[row, depth, o_t2].item())
        margin_bound = 2.0 * log_d
        drift_explains = o_margin <= margin_bound + 1e-6
        print(f"[CONT ORACLE first mismatch] row={row} depth={depth} "
              f"hybrid_tok={h_tok} oracle_tok={o_tok} "
              f"logit_max_diff={log_d:.6f}", flush=True)
        print(f"  top1/top2 margin:", flush=True)
        print(f"    oracle: top1={o_t1} (logit={o_l1:.4f}) top2={o_t2} "
              f"(logit={o_l2:.4f}) margin={o_margin:.4f}", flush=True)
        print(f"    hybrid: top1={h_t1} (logit={h_l1:.4f}) top2={h_t2} "
              f"(logit={h_l2:.4f}) margin={h_margin:.4f}", flush=True)
        print(f"    cross: hybrid_at_o_top1={h_at_o_t1:.4f} "
              f"hybrid_at_o_top2={h_at_o_t2:.4f}", flush=True)
        print(f"    structural drift consistent: o_margin({o_margin:.4f}) "
              f"<= 2*log_d({margin_bound:.4f}) → {drift_explains}",
              flush=True)

    def _dump_proxy_oracle_first_mismatch(self, *,
                                            hybrid_proxy_tokens, hybrid_proxy_logits,
                                            oracle_proxy_tokens, oracle_proxy_logits,
                                            oracle_proxy_meta, step_proxy_layout):
        """Phase D-5: side-by-side metadata dump for first
        hybrid_proxy != fresh_proxy_oracle mismatch."""
        plan = self.hybrid_phase2_plan
        eq = (hybrid_proxy_tokens == oracle_proxy_tokens)
        nz = (~eq).nonzero(as_tuple=False)
        if nz.numel() == 0:
            return
        row = int(nz[0, 0].item())
        depth = int(nz[0, 1].item())
        h_tok = int(hybrid_proxy_tokens[row, depth].item())
        o_tok = int(oracle_proxy_tokens[row, depth].item())
        log_d = float(
            (hybrid_proxy_logits[row, depth] - oracle_proxy_logits[row, depth])
            .abs().max().item()
        )
        print(f"[PROXY ORACLE first mismatch] row={row} depth={depth} "
              f"hybrid_tok={h_tok} oracle_tok={o_tok} "
              f"logit_max_diff={log_d:.6f}", flush=True)

        # Phase D-6: top-1 / top-2 margin dump. If oracle's margin <= the
        # observed hybrid-vs-oracle logit drift, the argmax flip is consistent
        # with structural FP drift on a near-tie row.
        h_top2 = torch.topk(hybrid_proxy_logits[row, depth], 2)
        o_top2 = torch.topk(oracle_proxy_logits[row, depth], 2)
        h_t1, h_t2 = int(h_top2.indices[0].item()), int(h_top2.indices[1].item())
        h_l1, h_l2 = float(h_top2.values[0].item()), float(h_top2.values[1].item())
        o_t1, o_t2 = int(o_top2.indices[0].item()), int(o_top2.indices[1].item())
        o_l1, o_l2 = float(o_top2.values[0].item()), float(o_top2.values[1].item())
        h_margin = h_l1 - h_l2
        o_margin = o_l1 - o_l2
        # Logit at oracle's top1 position in hybrid (and vice versa) — shows
        # how close hybrid was to oracle's argmax choice.
        h_at_o_t1 = float(hybrid_proxy_logits[row, depth, o_t1].item())
        o_at_h_t1 = float(oracle_proxy_logits[row, depth, h_t1].item())
        print(f"  top1/top2 margin:", flush=True)
        print(f"    oracle: top1={o_t1} (logit={o_l1:.4f}) top2={o_t2} "
              f"(logit={o_l2:.4f}) margin={o_margin:.4f}", flush=True)
        print(f"    hybrid: top1={h_t1} (logit={h_l1:.4f}) top2={h_t2} "
              f"(logit={h_l2:.4f}) margin={h_margin:.4f}", flush=True)
        print(f"    cross: hybrid_logit_at_oracle_top1({o_t1})={h_at_o_t1:.4f} "
              f"oracle_logit_at_hybrid_top1({h_t1})={o_at_h_t1:.4f}", flush=True)
        # Phase D-7: drift-consistency check using 2*log_d bound (margin
        # between any two positions can shift by up to 2*log_d under bounded
        # per-position drift). Captures both direct flips and tied tie-break
        # flips.
        h_at_o_t2 = float(hybrid_proxy_logits[row, depth, o_t2].item())
        margin_bound = 2.0 * log_d
        drift_explains = o_margin <= margin_bound + 1e-6
        print(f"    cross: hybrid_at_o_top1={h_at_o_t1:.4f} "
              f"hybrid_at_o_top2={h_at_o_t2:.4f}", flush=True)
        print(f"    structural drift consistent: o_margin({o_margin:.4f}) "
              f"<= 2*log_d({margin_bound:.4f}) → {drift_explains}",
              flush=True)

        # Hybrid metadata
        cont_count = plan.cont_row_count
        global_row_h = cont_count + row
        h_rope = int(plan.proxy_initial_rope_positions[row].item()) + depth
        h_slot = int(plan.per_row_slot_maps_by_depth[global_row_h, depth].item())
        h_ctx = int(plan.per_row_context_lens_by_depth[global_row_h, depth].item())
        h_kv_indptr_full = plan.per_row_kv_indptr_by_depth[
            :plan.total_row_count + 1, depth,
        ]
        h_kv_indices_full = plan.per_row_kv_indices_by_depth[depth]
        h_lo = int(h_kv_indptr_full[global_row_h].item())
        h_hi = int(h_kv_indptr_full[global_row_h + 1].item())
        h_pages = h_kv_indices_full[h_lo:h_hi].tolist()
        h_lpl_block = (h_ctx - 1) % self.block_size + 1

        # Oracle metadata. Oracle uses per-seq KV plan (B=1), so ctx_len /
        # kv_indptr / kv_indices / kv_last_page_len are seq-level (not per-row).
        # Row-level fields (rope, slot) are per row within MQ_proxy.
        o_rope = int(oracle_proxy_meta["step_rope_positions"][depth, row].item())
        o_slot = int(oracle_proxy_meta["step_slot_maps"][depth, row].item())
        o_ctx = int(oracle_proxy_meta["step_context_lens"][depth, 0].item())
        o_indptr = oracle_proxy_meta["kv_indptr_per_d"][depth]
        o_indices = oracle_proxy_meta["kv_indices_per_d"][depth]
        o_lpl = oracle_proxy_meta["kv_lpl_per_d"][depth]
        # B=1: row 0's pages = indices[0:indptr[1]] (shared across all MQ_proxy
        # rows of the seq). 'pages' here is per seq, not per row.
        o_pages = o_indices[int(o_indptr[0].item()):int(o_indptr[1].item())].tolist()

        print(f"  hybrid: rope={h_rope} slot={h_slot} ctx={h_ctx} "
              f"kv_pages={h_pages} kv_lpl(block_calc)={h_lpl_block}",
              flush=True)
        print(f"  oracle: rope={o_rope} slot={o_slot} ctx={o_ctx} "
              f"kv_pages={o_pages} kv_lpl={o_lpl.tolist()}",
              flush=True)

        # Hybrid bool mask visible set for this proxy row
        K1 = self.config.mesa_phase1_k
        K2 = self.config.mesa_phase2_k
        K_long = K1 + K2
        MQ_p1 = self.phase1_layout_long.MQ_LEN
        # num_tokens via cont_initial_rope[0] - K1 (j_idx_p1[0]=0 for hit case)
        if cont_count > 0:
            num_tokens = int(plan.cont_initial_rope_positions[0].item()) - K1
        else:
            num_tokens = h_rope - depth  # fallback (j_idx_proxy[0]=0)
        j_idx_p1_full = (
            self.phase1_layout_long.fan_idx_hit if plan.step_cache_hit
            else self.phase1_layout_long.fan_idx_miss
        )
        j_idx_proxy_vec = (
            plan.proxy_initial_rope_positions[:plan.proxy_row_count] - num_tokens
        )
        # Step 9A: K_step = plan.valid_k (this step's bucket).
        K_step = plan.valid_k
        bool_mask_h = self._compute_hybrid_bool_mask_for_depth(
            d=depth, K1=K1, K2=K2, K_step=K_step,
            MQ_p1=MQ_p1, L=h_ctx,
            cont_count=cont_count, proxy_count=plan.proxy_row_count,
            num_tokens=num_tokens,
            j_idx_p1=j_idx_p1_full[:cont_count],
            j_idx_proxy=j_idx_proxy_vec,
        )
        n_total = cont_count + plan.proxy_row_count
        h_visible = bool_mask_h.view(n_total, h_ctx)[global_row_h]
        h_vis_count = int(h_visible.sum().item())
        h_vis_pos = h_visible.nonzero(as_tuple=False).flatten().tolist()

        # Oracle bool mask visible set (from mirror_meta — get_custom_mask_cached)
        o_bool = oracle_proxy_meta["bool_mask_per_d"][depth]
        n_proxy = plan.proxy_row_count
        o_L = o_bool.numel() // n_proxy
        o_visible = o_bool.view(n_proxy, o_L)[row]
        o_vis_count = int(o_visible.sum().item())
        o_vis_pos = o_visible.nonzero(as_tuple=False).flatten().tolist()

        print(f"  hybrid mask: visible={h_vis_count} L={h_ctx} "
              f"first10={h_vis_pos[:10]} last5={h_vis_pos[-5:]}",
              flush=True)
        print(f"  oracle mask: visible={o_vis_count} L={o_L} "
              f"first10={o_vis_pos[:10]} last5={o_vis_pos[-5:]}",
              flush=True)
        # Note: hybrid and oracle visible positions are at DIFFERENT physical
        # slots (hybrid uses B_proxy region, oracle uses split's overlap with
        # P1 region). The semantic equivalence is: both should attend to
        # persistent + glue prefix(j_idx_proxy[r]) + own proxy ancestors at
        # depths [0, d+1). Matched K/V values via deterministic forward.

    def _hybrid_parity_diff(self, *, split_cont_tokens, split_cont_logits,
                              split_proxy_tokens, split_proxy_logits,
                              hybrid_cont_tokens, hybrid_cont_logits,
                              hybrid_proxy_tokens, hybrid_proxy_logits):
        """Step 7 parity harness: diff hybrid vs split outputs.

        Logs:
        - cont/proxy token mismatch counts
        - cont/proxy logits max/mean abs diff
        - first mismatch row/depth + j_idx + slot_map + context_len dump

        Triggered by SSD_HYBRID_PARITY=1 on the Step-8 hybrid-default path.
        temp=0 only — hybrid uses argmax, not sampler.
        """
        plan = self.hybrid_phase2_plan

        # Token mismatch counts
        cont_tok_eq = (split_cont_tokens == hybrid_cont_tokens)  # [cont, K2]
        proxy_tok_eq = (split_proxy_tokens == hybrid_proxy_tokens)
        n_cont = cont_tok_eq.numel()
        n_proxy = proxy_tok_eq.numel()
        cont_mismatch = int((~cont_tok_eq).sum().item())
        proxy_mismatch = int((~proxy_tok_eq).sum().item())

        # Logits diff
        cont_diff = (split_cont_logits - hybrid_cont_logits).abs()
        proxy_diff = (split_proxy_logits - hybrid_proxy_logits).abs()
        cont_max_diff = float(cont_diff.max().item())
        cont_mean_diff = float(cont_diff.mean().item())
        proxy_max_diff = float(proxy_diff.max().item())
        proxy_mean_diff = float(proxy_diff.mean().item())

        print(f"[HYBRID PARITY] tokens: cont {cont_mismatch}/{n_cont} mismatch "
              f"({100*cont_mismatch/max(n_cont,1):.1f}%); proxy {proxy_mismatch}/{n_proxy} "
              f"mismatch ({100*proxy_mismatch/max(n_proxy,1):.1f}%)", flush=True)
        print(f"[HYBRID PARITY] logits: cont max_abs_diff={cont_max_diff:.6f} "
              f"mean_abs_diff={cont_mean_diff:.6f}; proxy max_abs_diff={proxy_max_diff:.6f} "
              f"mean_abs_diff={proxy_mean_diff:.6f}", flush=True)

        # First mismatch dump. j_idx_p1 selection MUST follow actual seq cache
        # hit, NOT per_row_region_id (which is 0/1 for cont/proxy region tag).
        if cont_mismatch > 0:
            first_idx = (~cont_tok_eq).nonzero(as_tuple=False)[0]  # [2] = (row, depth)
            row = int(first_idx[0].item())
            depth = int(first_idx[1].item())
            split_tok = int(split_cont_tokens[row, depth].item())
            hybrid_tok = int(hybrid_cont_tokens[row, depth].item())
            _hit = bool(plan.step_cache_hit)
            j_idx_p1 = (
                self.phase1_layout_long.fan_idx_hit if _hit
                else self.phase1_layout_long.fan_idx_miss
            )
            j_idx_val = int(j_idx_p1[row].item())
            slot_map = int(plan.per_row_slot_maps_by_depth[row, depth].item())
            ctx_len = int(plan.per_row_context_lens_by_depth[row, depth].item())
            print(f"[HYBRID PARITY] first cont mismatch row={row} depth={depth} "
                  f"split_tok={split_tok} hybrid_tok={hybrid_tok} j_idx_p1={j_idx_val} "
                  f"slot_map={slot_map} ctx_len={ctx_len}", flush=True)

            # Phase D-3 mask diff dump: show what hybrid sees vs what split's
            # CG mask formula sees for this row at this depth. Probes hypothesis
            # that split's continuation mask formula (run_fi_tree_decode_cudagraph
            # at cont step 0 with K=K_long) exposes cross-branch Phase 1 KV
            # because prefix_len_b = num_tokens + K1*MQ_p1 - 1 instead of num_tokens-1.
            try:
                K1 = self.config.mesa_phase1_k
                K2 = self.config.mesa_phase2_k
                K_long = K1 + K2
                MQ_p1 = self.phase1_layout_long.MQ_LEN
                # Recover num_tokens from hybrid metadata
                cont_rope0 = int(plan.cont_initial_rope_positions[0].item())
                num_tokens = cont_rope0 - K1  # cont_rope[0] = nt + j_idx[0] + K1, j_idx[0]=0
                # Hybrid bool mask for this row at this depth — use Step 3's
                # ctx_len_d as L (Phase D-3a consistency fix).
                cont_count = plan.cont_row_count
                proxy_count = plan.proxy_row_count
                L_d = int(plan.per_row_context_lens_by_depth[0, depth].item())
                bool_mask = self._compute_hybrid_bool_mask_for_depth(
                    d=depth, K1=K1, K2=K2, K_step=plan.valid_k,
                    MQ_p1=MQ_p1, L=L_d,
                    cont_count=cont_count, proxy_count=proxy_count,
                    num_tokens=num_tokens,
                    j_idx_p1=j_idx_p1[:cont_count],
                    j_idx_proxy=plan.proxy_initial_rope_positions[:proxy_count] - num_tokens,
                )
                total_rows = cont_count + proxy_count
                L = bool_mask.numel() // total_rows
                hybrid_row = bool_mask.view(total_rows, L)[row]
                hyb_visible = hybrid_row.nonzero(as_tuple=False).flatten().tolist()
                # Split's mask formula at cont step `depth` (K=K_long internal):
                # prefix_len_b = num_tokens + K1*MQ_p1 - 1
                # glue_lower_tri at [prefix, prefix+K_long+1) using j_idx
                # diag blks for blk in [0, depth+1) at diag_start + blk*MQ_p1 + row
                split_prefix = num_tokens + K1 * MQ_p1 - 1
                split_diag_start = split_prefix + K_long + 1
                # Build split's predicted visible KV for this row
                split_visible = list(range(split_prefix))  # all prefix visible
                # glue lower-tri: cols [prefix, prefix+j_idx_val] visible
                for c in range(split_prefix, split_prefix + j_idx_val + 1):
                    split_visible.append(c)
                # diagonals: own row at depths 0..depth (current step inclusive)
                for blk in range(depth + 1):
                    split_visible.append(split_diag_start + blk * MQ_p1 + row)
                # Set diff
                hyb_set = set(hyb_visible)
                split_set = set(split_visible)
                only_split = sorted(split_set - hyb_set)
                only_hybrid = sorted(hyb_set - split_set)
                print(f"[HYBRID PARITY MASK] row={row} depth={depth} "
                      f"num_tokens={num_tokens} hyb_count={len(hyb_visible)} "
                      f"split_count_predicted={len(split_visible)} "
                      f"only_split_count={len(only_split)} "
                      f"only_hybrid_count={len(only_hybrid)}", flush=True)
                # Decode only_split into physical regions for clarity
                phase1_kv_base = num_tokens + K_long
                a_tail_base = phase1_kv_base + K1 * MQ_p1
                osp_persistent = sum(1 for c in only_split if c < num_tokens)
                osp_glue = sum(1 for c in only_split if num_tokens - 1 <= c < num_tokens + K_long)
                osp_p1 = sum(1 for c in only_split if phase1_kv_base <= c < a_tail_base)
                osp_other = len(only_split) - osp_persistent - osp_glue - osp_p1
                print(f"  only_split breakdown: persistent={osp_persistent} "
                      f"glue={osp_glue} phase1_kv={osp_p1} other={osp_other}", flush=True)
                print(f"  only_split sample (first 10): {only_split[:10]} "
                      f"... (last 5): {only_split[-5:] if len(only_split) > 10 else []}",
                      flush=True)
                print(f"  only_hybrid sample (first 10): {only_hybrid[:10]}", flush=True)
            except Exception as e:
                print(f"[HYBRID PARITY MASK] dump failed: {e}", flush=True)

        if proxy_mismatch > 0:
            # Dump ALL proxy mismatches (small count usually)
            all_idx = (~proxy_tok_eq).nonzero(as_tuple=False)
            print(f"[HYBRID PARITY] all proxy mismatches ({proxy_mismatch} total):", flush=True)
            for i in range(min(proxy_mismatch, 10)):
                r = int(all_idx[i, 0].item())
                d = int(all_idx[i, 1].item())
                st = int(split_proxy_tokens[r, d].item())
                ht = int(hybrid_proxy_tokens[r, d].item())
                ld = float((split_proxy_logits[r, d] - hybrid_proxy_logits[r, d]).abs().max().item())
                print(f"  proxy row={r} depth={d} split={st} hybrid={ht} logit_max_diff={ld:.4f}",
                      flush=True)

            # For mismatched rows: dump j_idx_proxy and rope (need access to last-step proxy_layout)
            # store from build_phase2_hybrid_plan via plan.proxy_initial_rope_positions
            mismatch_rows = sorted(set(int(all_idx[i, 0].item()) for i in range(proxy_mismatch)))
            print(f"[HYBRID PARITY] proxy mismatch rows: {mismatch_rows}", flush=True)
            for r in mismatch_rows[:5]:
                rope = int(plan.proxy_initial_rope_positions[r].item())
                # j_idx = rope - num_tokens (since proxy_initial_rope = num_tokens + j_idx_proxy[r])
                # We don't have num_tokens here directly; derive from any cont row's metadata
                if plan.cont_row_count > 0:
                    cont_rope0 = int(plan.cont_initial_rope_positions[0].item())
                    K1 = self.config.mesa_phase1_k
                    # cont_rope[0] = num_tokens + j_idx_p1[0] + K1, j_idx_p1[0] usually = 0
                    num_tokens_est = cont_rope0 - K1
                    j_idx_proxy_r = rope - num_tokens_est
                    print(f"  row={r} rope={rope} j_idx_proxy~={j_idx_proxy_r}", flush=True)

            first_idx = all_idx[0]
            row = int(first_idx[0].item())
            depth = int(first_idx[1].item())
            cont_count = plan.cont_row_count
            global_row = cont_count + row
            split_tok = int(split_proxy_tokens[row, depth].item())
            hybrid_tok = int(hybrid_proxy_tokens[row, depth].item())

            # Hybrid metadata (from HybridPhase2Plan)
            hybrid_slot = int(plan.per_row_slot_maps_by_depth[global_row, depth].item())
            hybrid_ctx_len = int(plan.per_row_context_lens_by_depth[global_row, depth].item())
            hybrid_rope = int(plan.proxy_initial_rope_positions[row].item()) + depth

            # Reconstruct split-side equivalent metadata from current MESA path:
            # - split's proxy scratch position = num_tokens + K_long + d*MQ_proxy + j
            #   (overlaps Phase 1 KV; tree mask handles ancestor visibility)
            # - split's proxy ctx_len at depth d ≈ num_tokens + K_long + (d+1)*MQ_proxy
            # - split's proxy rope = num_tokens + j_idx_proxy[j] + d (same as hybrid)
            K_long_dbg = self.config.speculate_k
            MQ_proxy_dbg = plan.proxy_row_count
            num_tokens_dbg = hybrid_ctx_len - K_long_dbg - K_long_dbg * (
                self.phase1_layout_long.MQ_LEN
            ) - (depth + 1) * MQ_proxy_dbg
            split_scratch_pos = num_tokens_dbg + K_long_dbg + depth * MQ_proxy_dbg + row
            split_ctx_len_eq = num_tokens_dbg + K_long_dbg + (depth + 1) * MQ_proxy_dbg

            # Bool mask for current depth (matches what wrapper used).
            # Phase D-3a: pass L = ctx_len_d directly. Use actual cache_hits
            # (from plan-stored value) to pick fan_idx_hit vs miss.
            L_d = int(plan.per_row_context_lens_by_depth[0, depth].item())
            _hit = bool(plan.step_cache_hit)  # default hit
            j_idx_p1_dbg = (
                self.phase1_layout_long.fan_idx_hit if _hit
                else self.phase1_layout_long.fan_idx_miss
            )[:cont_count]
            bool_mask_d = self._compute_hybrid_bool_mask_for_depth(
                d=depth, K1=self.config.mesa_phase1_k,
                K2=self.config.mesa_phase2_k, K_step=plan.valid_k,
                MQ_p1=self.phase1_layout_long.MQ_LEN,
                L=L_d,
                cont_count=cont_count, proxy_count=plan.proxy_row_count,
                num_tokens=num_tokens_dbg,
                j_idx_p1=j_idx_p1_dbg,
                j_idx_proxy=plan.proxy_initial_rope_positions[:plan.proxy_row_count] - num_tokens_dbg,
            )
            n_total = cont_count + plan.proxy_row_count
            row_mask = bool_mask_d.view(n_total, L_d)[global_row]
            n_visible = int(row_mask.sum().item())
            visible_pos = row_mask.nonzero(as_tuple=False).flatten().tolist()
            visible_summary = (
                visible_pos[:5] + ["..."] + visible_pos[-5:]
                if len(visible_pos) > 12 else visible_pos
            )

            print(f"[HYBRID PARITY] first proxy mismatch row={row} depth={depth}", flush=True)
            print(f"  tokens: split={split_tok} hybrid={hybrid_tok}", flush=True)
            print(f"  rope: hybrid={hybrid_rope} (split=same; both = num_tokens + j_idx_proxy[{row}] + {depth})",
                  flush=True)
            print(f"  slot: split={split_scratch_pos} hybrid={hybrid_slot} (different scratch regions)",
                  flush=True)
            print(f"  ctx_len: split_eq={split_ctx_len_eq} hybrid={hybrid_ctx_len}", flush=True)
            print(f"  visible_kv: count={n_visible} positions={visible_summary}", flush=True)

    def _compute_hybrid_bool_mask_for_depth(self, *, d, K1, K2, K_step,
                                              MQ_p1, L,
                                              cont_count, proxy_count,
                                              num_tokens, j_idx_p1, j_idx_proxy):
        """Compute per-row bool visibility mask for hybrid attention at depth d.

        K_step = this step's bucket (K_long for long-hit, K_short for
        short-hit). Drives glue width and phase1_kv_base offset. K1, K2 are
        Phase 1 / Phase 2 forward depths (constant per config).

        L is the per-row KV length (= runtime ctx_len_d). Caller must pass
        the SAME value used by Step 3's per_row_context_lens_by_depth, so
        wrapper.plan(custom_mask=...) and wrapper.run() see consistent shapes.

        When proxy_count == 0, proxy-region modulo/division is skipped
        entirely; the trailing (d+1) cols beyond a_tail (placeholder cols
        from Step 3's MQ_proxy=max(proxy_count,1)=1) remain 0 and no row
        attends to them.
        """
        device = self.device
        total = cont_count + proxy_count

        # Glue: K_step+1 tokens at [num_tokens-1, num_tokens+K_step). Phase 1
        # first slot at num_tokens + K_step (= positions[0] from
        # _build_tree_decode_args_for_layout with glue_offset=K_step+1).
        glue_base = num_tokens - 1
        phase1_kv_base = num_tokens + K_step
        a_tail_base = phase1_kv_base + K1 * MQ_p1
        b_proxy_base = a_tail_base + K2 * MQ_p1

        kv_pos = torch.arange(L, device=device, dtype=torch.int64)
        kv_pos_2d = kv_pos.view(1, L)

        cont_idx_t = torch.arange(cont_count, device=device, dtype=torch.int64)

        persistent_vis = kv_pos_2d < num_tokens

        # cont rows — independent of proxy_count
        if cont_count > 0:
            cont_glue_vis = (
                (kv_pos_2d >= glue_base) &
                (kv_pos_2d <= (glue_base + j_idx_p1.view(cont_count, 1)))
            )
            p1_off = kv_pos_2d - phase1_kv_base
            in_p1 = (kv_pos_2d >= phase1_kv_base) & (kv_pos_2d < a_tail_base)
            cont_p1_vis = in_p1 & ((p1_off % MQ_p1) == cont_idx_t.view(cont_count, 1))

            at_off = kv_pos_2d - a_tail_base
            in_at = (kv_pos_2d >= a_tail_base) & (kv_pos_2d < b_proxy_base)
            cont_at_vis = (
                in_at
                & ((at_off % MQ_p1) == cont_idx_t.view(cont_count, 1))
                & ((at_off // MQ_p1) <= d)
            )

            cont_mask = (
                persistent_vis.expand(cont_count, L)
                | cont_glue_vis | cont_p1_vis | cont_at_vis
            )
        else:
            cont_mask = torch.zeros((0, L), dtype=torch.bool, device=device)

        # proxy rows — only when proxy_count > 0. When 0, skip modulo/division
        # over a B_proxy region whose width is meaningless (Step 3's placeholder
        # MQ_proxy=1 doesn't reflect real proxy fan_out).
        if proxy_count > 0:
            proxy_idx_t = torch.arange(proxy_count, device=device, dtype=torch.int64)
            MQ_proxy = proxy_count  # Step 3 uses sum(fan_out_list) = proxy_count when proxy_count>0
            proxy_glue_vis = (
                (kv_pos_2d >= glue_base) &
                (kv_pos_2d <= (glue_base + j_idx_proxy.view(proxy_count, 1)))
            )
            bp_off = kv_pos_2d - b_proxy_base
            in_bp = (kv_pos_2d >= b_proxy_base) & (kv_pos_2d < (b_proxy_base + K2 * MQ_proxy))
            proxy_b_vis = (
                in_bp
                & ((bp_off % MQ_proxy) == proxy_idx_t.view(proxy_count, 1))
                & ((bp_off // MQ_proxy) <= d)
            )

            proxy_mask = persistent_vis.expand(proxy_count, L) | proxy_glue_vis | proxy_b_vis
        else:
            proxy_mask = torch.zeros((0, L), dtype=torch.bool, device=device)

        # Concat + flatten (FlashInfer custom_mask is 1D bool/uint8)
        full_mask = torch.cat([cont_mask, proxy_mask], dim=0)  # [total, L]
        return full_mask.reshape(-1).to(torch.bool)  # [total * L]

    @torch.inference_mode()
    def _decode_correct_split_cont(self, *, draft_tree_args, draft_tokens_phase1):
        """Phase D-4 / Phase 9B-3: continuation oracle, valid_k-aware.

        Uses split's physical layout (cont_layout slot positions = same
        as hybrid's A_tail region) but applies plan-correct continuation
        bool mask. Generalized for both buckets via plan.valid_k.

        Returns: (cont_tokens [cont_count, K2], cont_logits [cont_count, K2, V])
        """
        plan = self.hybrid_phase2_plan
        K1 = self.config.mesa_phase1_k
        K2 = self.config.mesa_phase2_k
        # Step 9B-3: K_step = plan.valid_k (long or short bucket)
        K_step = plan.valid_k
        if K_step == plan.K_long:
            _phase1_layout_step = self.phase1_layout_long
        else:
            _phase1_layout_step = self.phase1_layout_short
        MQ_p1 = _phase1_layout_step.MQ_LEN
        cont_count = MQ_p1  # phase1.MQ_LEN = (K_step+1) * dfo
        device = self.device
        V = self.hf_config.vocab_size
        dt = self.hf_config.torch_dtype

        # Recover num_tokens and j_idx_p1 from draft_tree_args.
        p1_pos0 = int(draft_tree_args["positions"][0].item())
        num_tokens = p1_pos0 - K_step
        cache_hits = draft_tree_args["cache_hits"]
        j_idx_p1 = (
            _phase1_layout_step.fan_idx_hit if int(cache_hits[0])
            else _phase1_layout_step.fan_idx_miss
        )[:cont_count]

        # Slot positions: cont_layout writes at
        #   step_positions[d, r] = num_tokens + K_step + (K1+d)*MQ_p1 + r
        # = hybrid's a_tail_base + d*MQ_p1 + r.
        cont_idx = torch.arange(cont_count, device=device, dtype=torch.int64)
        phase1_kv_base = num_tokens + K_step
        a_tail_base = phase1_kv_base + K1 * MQ_p1
        cont_initial_pos = a_tail_base + cont_idx  # depth 0 slot

        # Logical rope: num_tokens + j_idx_p1[r] + K1.
        cont_initial_rope = num_tokens + j_idx_p1 + K1

        # Wrapper: dedicated eager-only correct_split_cont
        wrapper_dict = self.prefill_wrappers_by_layout["correct_split_cont"]
        wrapper_total = next(iter(wrapper_dict.keys()))
        assert cont_count <= wrapper_total, (
            f"correct_split_cont total={cont_count} > wrapper max={wrapper_total}"
        )
        wrapper = wrapper_dict[wrapper_total]

        # Block table from seq (B=1)
        seq_dbt = draft_tree_args["block_tables"]
        seq_bt = seq_dbt[0]
        n_pages_seq = seq_bt.shape[0]

        # Output buffers
        cont_tokens = torch.empty((cont_count, K2), dtype=torch.int64, device=device)
        cont_logits = torch.empty((cont_count, K2, V), dtype=dt, device=device)

        cur_cont_ids = draft_tokens_phase1[:cont_count, K1 - 1].contiguous()
        cu_seqlens_q_full = torch.arange(cont_count + 1, device=device, dtype=torch.int32)

        for d in range(K2):
            # Per-depth slot map: scratch_pos = a_tail_base + d*MQ_p1 + r
            scratch_pos = cont_initial_pos + d * MQ_p1
            block_id = (scratch_pos // self.block_size).to(torch.int64)
            offset = (scratch_pos % self.block_size).to(torch.int32)
            slot_map_d = (
                seq_bt[block_id].to(torch.int32) * self.block_size + offset
            ).contiguous()

            # ctx_len at depth d: split's natural extent =
            #   num_tokens + K_step + (K1+d+1)*MQ_p1
            ctx_len_val = num_tokens + K_step + (K1 + d + 1) * MQ_p1
            ctx_len_d = torch.full(
                (cont_count,), ctx_len_val, dtype=torch.int32, device=device,
            )

            # kv_indptr / kv_indices: per-row, all rows reference seq's pages
            pages_needed = (ctx_len_val + self.block_size - 1) // self.block_size
            pages_needed = int(min(pages_needed, n_pages_seq))
            kv_indptr_d = torch.arange(
                cont_count + 1, device=device, dtype=torch.int32,
            ) * pages_needed
            kv_indices_d = seq_bt[:pages_needed].to(torch.int32).repeat(cont_count)

            kv_last_page_len = ((ctx_len_d - 1) % self.block_size + 1).to(torch.int32)

            # Per-row block_tables (collapse-identical fill)
            block_tables_d = seq_bt[:n_pages_seq].view(1, -1).expand(
                cont_count, n_pages_seq,
            ).contiguous().to(torch.int32)

            # Plan-correct continuation bool mask (cont-only).
            bool_mask_d = self._compute_hybrid_bool_mask_for_depth(
                d=d, K1=K1, K2=K2, K_step=K_step,
                MQ_p1=MQ_p1, L=ctx_len_val,
                cont_count=cont_count, proxy_count=0,
                num_tokens=num_tokens,
                j_idx_p1=j_idx_p1,
                j_idx_proxy=torch.empty(0, dtype=torch.int64, device=device),
            )

            wrapper.plan(
                cu_seqlens_q_full,
                kv_indptr_d,
                kv_indices_d,
                kv_last_page_len,
                self.hf_config.num_attention_heads,
                self.hf_config.num_key_value_heads,
                self.hf_config.head_dim,
                self.block_size,
                custom_mask=bool_mask_d,
                q_data_type=dt,
                kv_data_type=dt,
            )

            cur_cont_rope = cont_initial_rope + d
            set_context(
                is_prefill=False,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=0,
                max_seqlen_k=0,
                slot_mapping=slot_map_d,
                context_lens=ctx_len_d,
                block_tables=block_tables_d,
                active_mq_len=1,
                active_wrappers=wrapper_dict,
                active_layout=None,
            )

            outputs = self.model(cur_cont_ids, cur_cont_rope.contiguous())
            logits_flat = self.model.compute_logits(outputs, last_only=False)
            reset_context()

            logits_flat = logits_flat.view(-1, V)
            next_tokens = logits_flat.argmax(dim=-1)

            cont_logits[:, d, :] = logits_flat
            cont_tokens[:, d] = next_tokens
            cur_cont_ids = next_tokens

        return cont_tokens, cont_logits

    @torch.inference_mode()
    def _decode_phase2_hybrid(self, *, draft_tree_args, proxy_tree_args,
                                step_proxy_layout, draft_tokens_phase1):
        """Eager hybrid Phase 2 forward (default hot path post-Step-8).

        Combines continuation rows (extending Phase 1's K1 ancestors) and
        proxy rows in a single flat batch. Runs K2 depth forwards with
        per-row attention plumbing from HybridPhase2Plan. Writes A_tail +
        B_proxy regions; never overwrites Phase 1 KV.

        Current runtime is eager-only. ``plan.valid_k`` already drives the
        long/short metadata dispatch inside the loop; Step 9B adds hybrid
        CG-captured variants for the per-bucket runtime path.

        Returns: (cont_tokens [cont_count, K2], cont_logits [cont_count, K2, V],
                  proxy_tokens [proxy_count, K2], proxy_logits [proxy_count, K2, V])
                 same shape as split reference's continuation+proxy outputs
                 so cache emit stays unchanged.

        Args:
            draft_tokens_phase1: Phase 1 forward outputs [cont_count, K1].
                Last column (depth K1-1) is the input to continuation depth 0.
        """
        plan = self.hybrid_phase2_plan
        K1 = self.config.mesa_phase1_k
        K2 = self.config.mesa_phase2_k
        K_long = K1 + K2

        # Step 9A: eager hybrid handles BOTH long and short buckets via
        # plan.valid_k dispatch. Step 9B will add CG-captured variants per
        # bucket (phase2_hybrid_long / phase2_hybrid_short).
        assert plan.valid_k in (plan.K_long, plan.K_short), (
            f"hybrid eager path: unexpected valid_k={plan.valid_k} "
            f"(expected K_long={plan.K_long} or K_short={plan.K_short})"
        )

        cont_count = plan.cont_row_count
        proxy_count = plan.proxy_row_count
        total = plan.total_row_count
        device = self.device
        V = self.hf_config.vocab_size
        dt = self.hf_config.torch_dtype

        # Per-row wrapper — eager hybrid wrapper family, dispatched by
        # plan.valid_k (long vs short bucket). Step 9B added separate
        # phase2_hybrid_short_eager wrapper alongside the existing long.
        if plan.valid_k == plan.K_long:
            wrapper_dict = self.prefill_wrappers_by_layout["phase2_hybrid_long"]
        else:
            wrapper_dict = self.prefill_wrappers_by_layout["phase2_hybrid_short"]
        wrapper_total = next(iter(wrapper_dict.keys()))
        assert total <= wrapper_total, (
            f"hybrid runtime total={total} > wrapper allocated max={wrapper_total}"
        )
        wrapper = wrapper_dict[wrapper_total]

        # Output buffers (eager — no buf reuse for first land; refine later)
        cont_tokens = torch.empty((cont_count, K2), dtype=torch.int64, device=device)
        cont_logits = torch.empty((cont_count, K2, V), dtype=dt, device=device)
        proxy_tokens = torch.empty((proxy_count, K2), dtype=torch.int64, device=device)
        proxy_logits = torch.empty((proxy_count, K2, V), dtype=dt, device=device)

        # Initial input ids per Step 6 design:
        # - continuation depth 0 input = Phase 1's K1-th token (= last Phase 1 output)
        # - proxy depth 0 input = proxy_forked tokens (= proxy_tree_args.input_ids)
        cur_cont_ids = draft_tokens_phase1[:cont_count, K1 - 1].contiguous()
        cur_proxy_ids = proxy_tree_args["input_ids"][:proxy_count].contiguous()

        cu_seqlens_q_full = torch.arange(total + 1, device=device, dtype=torch.int32)

        # Bucket for per-depth profiling labels (matches CG path naming).
        _eager_bucket = "long" if plan.valid_k == plan.K_long else "short"
        from ssd.engine.helpers.cudagraph_helpers import (
            mesa_record as _mr_eg, mesa_close as _mc_eg,
        )

        for d in range(K2):
            # Per-depth prep — KV plan, custom mask, set_context.
            _mev_eg_prep = _mr_eg(f"phase2_hybrid_eager_prep_{_eager_bucket}")
            # Flat batch: [cont rows | proxy rows]
            flat_input_ids = torch.cat([cur_cont_ids, cur_proxy_ids], dim=0)  # [total]

            # LOGICAL rope positions (per Step 6 invariant 4: model.run gets
            # logical rope, NOT physical hybrid coords)
            cont_rope = plan.cont_initial_rope_positions[:cont_count] + d
            proxy_rope = plan.proxy_initial_rope_positions[:proxy_count] + d
            flat_rope = torch.cat([cont_rope, proxy_rope], dim=0).contiguous()  # [total]

            # Per-row attention metadata (PHYSICAL — from HybridPhase2Plan)
            slot_map_d = plan.per_row_slot_maps_by_depth[:total, d].contiguous()
            ctx_len_d = plan.per_row_context_lens_by_depth[:total, d].contiguous().to(torch.int32)
            kv_indptr_d = plan.per_row_kv_indptr_by_depth[:total + 1, d].contiguous()
            n_idx = int(kv_indptr_d[total].item())
            kv_indices_d = plan.per_row_kv_indices_by_depth[d, :n_idx].contiguous()

            # kv_last_page_len: per-row, last page valid token count
            # = ((ctx_len - 1) % block_size) + 1
            kv_last_page_len = ((ctx_len_d - 1) % self.block_size + 1).to(torch.int32)

            # Per-row block_tables — collapse-identical fill from Step 3
            # (current 1-page case). Plan field shape preserved per-row.
            block_tables_d = plan.per_row_block_tables[:total]
            L_d = int(ctx_len_d[0].item())  # uniform per row at fixed depth

            # Custom mask — FlashInfer's custom_mask= arg expects BOOL tensor
            # of shape [total_qo_len * total_kv_len] (1 element per (q, kv)
            # pair), not bit-packed. Pass L = ctx_len_d directly so
            # wrapper.plan(custom_mask) and wrapper.run() see consistent
            # buffer shapes (Phase D-3a).
            # Step 9A: K_step = plan.valid_k (long or short bucket).
            # Phase 1 layout dispatch follows the same bucket. MQ_p1 is
            # smaller for short bucket → mask formula adapts via K_step
            # offset and MQ_p1 stride.
            K_step_h = plan.valid_k
            _phase1_layout_h = (
                self.phase1_layout_long if K_step_h == plan.K_long
                else self.phase1_layout_short
            )
            MQ_p1_h = _phase1_layout_h.MQ_LEN
            bool_mask_d = self._compute_hybrid_bool_mask_for_depth(
                d=d,
                K1=K1, K2=K2, K_step=K_step_h,
                MQ_p1=MQ_p1_h,
                L=L_d,
                cont_count=cont_count,
                proxy_count=proxy_count,
                num_tokens=int(draft_tree_args["positions"][0].item()) - K_step_h,
                j_idx_p1=(
                    _phase1_layout_h.fan_idx_hit if int(draft_tree_args["cache_hits"][0])
                    else _phase1_layout_h.fan_idx_miss
                )[:cont_count],
                j_idx_proxy=(
                    step_proxy_layout.fan_idx_hit if int(draft_tree_args["cache_hits"][0])
                    else step_proxy_layout.fan_idx_miss
                )[:proxy_count],
            )

            # Plan the wrapper for depth d
            wrapper.plan(
                cu_seqlens_q_full[:total + 1],
                kv_indptr_d,
                kv_indices_d,
                kv_last_page_len,
                self.hf_config.num_attention_heads,
                self.hf_config.num_key_value_heads,
                self.hf_config.head_dim,
                self.block_size,
                custom_mask=bool_mask_d,
                q_data_type=dt,
                kv_data_type=dt,
            )

            # CRITICAL: cu_seqlens_q=None routes through attention.py's
            # tree_decode branch which calls prefill_wrapper.run() with our
            # planned custom mask. Setting cu_seqlens_q (the bug we just
            # found in proxy mirror) routes to flash_attn_with_kvcache which
            # IGNORES custom_mask and does only causal attention — wrong.
            # active_mq_len=1 because hybrid has 1 query per row.
            set_context(
                is_prefill=False,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=0,
                max_seqlen_k=0,
                slot_mapping=slot_map_d,
                context_lens=ctx_len_d,
                block_tables=block_tables_d,
                active_mq_len=1,
                active_wrappers=wrapper_dict,
                active_layout=None,
            )

            _mc_eg(f"phase2_hybrid_eager_prep_{_eager_bucket}", _mev_eg_prep)

            # Eager forward — bypass run_model's CG dispatch by calling
            # self.model directly. tree_decode_step is NOT used here since
            # we're not going through the captured tree CG.
            _mev_eg_fwd = _mr_eg(f"phase2_hybrid_eager_replay_{_eager_bucket}")
            outputs = self.model(flat_input_ids, flat_rope)
            logits_flat = self.model.compute_logits(outputs, last_only=False)
            _mc_eg(f"phase2_hybrid_eager_replay_{_eager_bucket}", _mev_eg_fwd)
            reset_context()

            logits_flat = logits_flat.view(-1, V)  # [total, V]

            # Sample greedy (current MESA test config) — temps unused
            next_tokens = logits_flat.argmax(dim=-1)  # [total]

            # Split cont/proxy
            cont_logits[:, d, :] = logits_flat[:cont_count]
            cont_tokens[:, d] = next_tokens[:cont_count]
            proxy_logits[:, d, :] = logits_flat[cont_count:total]
            proxy_tokens[:, d] = next_tokens[cont_count:total]

            cur_cont_ids = next_tokens[:cont_count]
            cur_proxy_ids = next_tokens[cont_count:total]

        return cont_tokens, cont_logits, proxy_tokens, proxy_logits

    def _merge_and_populate_cache(self, draft_args, draft_tokens, draft_logits,
                                    proxy_args, proxy_tokens, proxy_logits,
                                    cache_hits_list, draft_acts=None, proxy_acts=None,
                                    proxy_layout=None, draft_layout=None):
        """Merge draft + proxy tree decode results into single cache.
        proxy_layout: runtime layout for dynamic fan_out (Policy A). Falls back to self.proxy_layout.

        v1 hybrid path: when ``mesa_phase1_k`` is set, proxy_tokens / proxy_logits
        come back with depth K2 (< K_long). Cache row width is uniformly K_long;
        the K_long - K2 tail of proxy rows is zero-padded and ignored downstream
        (verify reads only the first ``valid_k`` positions).
        """
        from ssd.engine.helpers.cudagraph_helpers import mesa_record as _mr, mesa_close as _mc
        _mev_mc = _mr("merge_cache")
        _proxy_layout = proxy_layout or self.proxy_layout
        # Step 9A: draft layout dispatch by step's valid_k (long/short bucket).
        # When not passed (legacy path), fall back to the static draft_layout.
        _draft_layout = draft_layout or self.draft_layout
        # Cache row width: split-only K1/K2 mode pads to K_max=max(K1,K2)
        # so wire/cache layout is independent of K1+K2. Hybrid stays at K_long.
        if SPLIT_K1K2_MODE and self.config.mesa_phase1_k is not None:
            K1_cfg = self.config.mesa_phase1_k
            K2_cfg = self.config.mesa_phase2_k
            K_long = K1_cfg if K1_cfg >= K2_cfg else K2_cfg  # = K_max in this mode
        else:
            K_long = self.config.speculate_k
        # Phase 3: draft row valid_k = draft_tokens.shape[1] (K1 without
        # continuation, K_long with continuation). proxy row valid_k =
        # proxy_tokens.shape[1] (K_short = K2).
        draft_row_vk = int(draft_tokens.shape[1])
        proxy_row_vk = int(proxy_tokens.shape[1])

        # Build keys with layout-specific fan_idx
        draft_k = torch.cat([_draft_layout.fan_idx_hit if int(h) else _draft_layout.fan_idx_miss
                              for h in cache_hits_list])
        draft_keys = torch.stack([
            draft_args["seq_ids_expanded"].to(torch.int64),
            draft_k, draft_args["rec_flat"].to(torch.int64)], dim=1)

        proxy_k = torch.cat([_proxy_layout.fan_idx_hit if int(h) else _proxy_layout.fan_idx_miss
                              for h in cache_hits_list])
        proxy_keys = torch.stack([
            proxy_args["seq_ids_expanded"].to(torch.int64),
            proxy_k, proxy_args["rec_flat"].to(torch.int64)], dim=1)

        # Pad draft_tokens / proxy_tokens from per-row K (K1 / K_short) to
        # K_long width so cache row width is uniform. The matched row's
        # valid_k tells consumers (speculator, verify) the meaningful prefix.
        def _pad_to_klong(toks, logs, acts, src_k):
            if toks.shape[1] == K_long:
                return toks, logs, acts
            pad_w = K_long - src_k
            n = toks.shape[0]
            tok_pad = torch.zeros((n, pad_w), dtype=toks.dtype, device=toks.device)
            toks = torch.cat([toks, tok_pad], dim=1)
            log_pad = torch.zeros((n, pad_w, logs.shape[2]), dtype=logs.dtype, device=logs.device)
            logs = torch.cat([logs, log_pad], dim=1)
            if acts is not None:
                act_pad = torch.zeros((n, pad_w, acts.shape[2]), dtype=acts.dtype, device=acts.device)
                acts = torch.cat([acts, act_pad], dim=1)
            return toks, logs, acts

        draft_tokens, draft_logits, draft_acts = _pad_to_klong(draft_tokens, draft_logits, draft_acts, draft_row_vk)
        proxy_tokens, proxy_logits, proxy_acts = _pad_to_klong(proxy_tokens, proxy_logits, proxy_acts, proxy_row_vk)

        self.tree_cache_keys = torch.cat([draft_keys, proxy_keys], dim=0)
        # Boundary for phase 1 (draft-sourced) vs phase 2 (proxy-sourced) classification
        # in the next hit_cache_and_respond lookup.
        self._last_n_draft_keys = draft_keys.shape[0]
        self.tree_cache_tokens = torch.cat([draft_tokens, proxy_tokens], dim=0)
        self.tree_cache_logits = torch.cat([draft_logits, proxy_logits], dim=0)
        if draft_acts is not None and proxy_acts is not None:
            self.tree_cache_activations = torch.cat([draft_acts, proxy_acts], dim=0)
        else:
            self.tree_cache_activations = None
        # Per-row valid_k. Legacy MESA: all rows = K_long. Hybrid: draft rows
        # = K1 (Phase 1 only) or K_long (with continuation), proxy rows = K_short.
        n_total = self.tree_cache_keys.shape[0]
        n_draft = draft_keys.shape[0]
        self.tree_cache_valid_k = torch.empty(n_total, dtype=torch.int64, device=self.device)
        self.tree_cache_valid_k[:n_draft] = draft_row_vk
        self.tree_cache_valid_k[n_draft:] = proxy_row_vk
        _mc("merge_cache", _mev_mc)

    # new one, with true asynchrony
    def draft_loop(self):
        """
        Runs the asynchronous draft model loop. 
        Handles three commands:
          1 = prefill, 0 = spec request, 2 = exit.
        """
        assert self.draft_async, "draft_loop only runs in async-draft mode"

        while True:
            # 1) Wait for the next command (may be PREFILL, SPEC_REQUEST, or EXIT)
            from ssd.engine.helpers.cudagraph_helpers import mesa_record as _mr_c, mesa_close as _mc_c
            _mev_rc = _mr_c("draft_recv_cmd")
            cmd = self.recv_cmd()
            _mc_c("draft_recv_cmd", _mev_rc)

            # PREFILL: run the draft prefill and then loop back
            if cmd == 1:
                self.draft_async_prefill()
                continue

            # SPECULATE request: serve out-of-cache or random speculations
            elif cmd == 0:
                _ds0 = time.perf_counter()
                _prof = os.environ.get("SSD_PROFILE", "0") == "1"
                if _prof or PROFILE_DRAFT:
                    torch.cuda.synchronize()
                    _d0 = time.perf_counter()

                glue_decode_input_ids, partial_tree_decode_args = self._service_spec_request()

                if _prof or PROFILE_DRAFT:
                    torch.cuda.synchronize()
                    _d1 = time.perf_counter()

                self._reset_tree_cache_tensors()

                if self.config.mesa_enabled:
                    # MESA: 2-pass tree decode (draft-sourced → proxy-sourced)
                    self._build_tree_batch_mesa(partial_tree_decode_args, glue_decode_input_ids)
                    # Set profiling vars to avoid NameError in shared print below
                    if _prof or PROFILE_DRAFT:
                        torch.cuda.synchronize()
                        _d2 = _d3 = time.perf_counter()
                else:
                    # Standard SSD: single-pass tree decode
                    tree_decode_args = self._build_tree_batch(partial_tree_decode_args, glue_decode_input_ids)

                    if _prof or PROFILE_DRAFT:
                        torch.cuda.synchronize()
                        _d2 = time.perf_counter()

                    tokens, logits, activations = self._decode_tree(tree_decode_args)

                    if _prof or PROFILE_DRAFT:
                        torch.cuda.synchronize()
                        _d3 = time.perf_counter()

                    self._populate_tree_cache(tree_decode_args, tokens, logits, tree_decode_args["cache_hits"], activations)
                self._draft_step_times.append(time.perf_counter() - _ds0)

                if _prof or PROFILE_DRAFT:
                    torch.cuda.synchronize()
                    _d4 = time.perf_counter()
                    print(f"[PROFILE draft] service={(_d1-_d0)*1000:.2f}ms build_tree={(_d2-_d1)*1000:.2f}ms decode_tree={(_d3-_d2)*1000:.2f}ms populate={(_d4-_d3)*1000:.2f}ms total={(_d4-_d0)*1000:.2f}ms", flush=True)

                if PROFILE_DRAFT:
                    flush_draft_profile()

                continue

            # EXIT: clean up and break out of the loop
            elif cmd == 2:
                if self._draft_step_times:
                    avg_ms = sum(self._draft_step_times) * 1000 / len(self._draft_step_times)
                    print(f"[metrics] Avg draft step time (ms): {avg_ms:.2f}", flush=True)
                try:
                    from ssd.engine.helpers.cudagraph_helpers import mesa_dump
                    mesa_dump("draft")
                except Exception:
                    pass
                self.exit()
                break

            else:
                raise RuntimeError(f"draft_loop: unknown command {cmd}")
