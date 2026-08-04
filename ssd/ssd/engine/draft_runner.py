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

# Split-only K1/K2 mode (per docs/duet/04-split-k1k2-design.md):
#   - Draft pass = K1 forwards → draft-sourced rows of depth K1
#   - Proxy pass = K2 forwards → proxy-sourced rows of depth K2
#   - NO continuation. valid_k space = {K1, K2}.
# Hybrid path is untouched when this flag is off.
from ssd.engine.helpers.e0_trace import E0_TRACE as _E0_TRACE
from ssd.engine.helpers import e0_trace as _e0
SPLIT_K1K2_MODE = os.environ.get("SSD_FORCE_SPLIT_K1K2", "0") == "1"
# Trace gate for split-K1/K2 contract validation. Active only when
# SSD_TRACE_SPLIT_K1K2=1. Logs every step's valid_k / shapes / per-position
# proxy fan_out so we can prove what's real data vs zero-pad.
TRACE_SPLIT_K1K2 = os.environ.get("SSD_TRACE_SPLIT_K1K2", "0") == "1"
# JIT-short gate (split mode only): serve cache misses with a K2-deep JIT
# chain (valid_k=K2, short-bucket path — same shape as a phase2 hit)
# instead of K_max=K1. Saves K1-K2 unbatched draft forwards on the miss
# critical path; costs L_miss truncation beyond depth K2.
DUET_JIT_SHORT = os.environ.get("SSD_DUET_JIT_SHORT", "0") == "1"
# Subset-JIT A/B gate: on a MIXED batch, run the JIT only over the miss
# rows (gather -> compact JIT -> scatter) instead of all B rows. The
# response CONTENT is identical either way (hit rows are overwritten from
# the cache after the JIT regardless); this changes only the JIT compute
# width. Default OFF — the JIT forward is a 1-token/seq decode (width-
# latency-bound), so the expected gain is small; this gate exists to
# measure it (b_gt1/bscale32 jit_subset experiment). Not applied on the
# EAGLE path (activation gather/scatter not wired).
DUET_JIT_SUBSET = os.environ.get("SSD_DUET_JIT_SUBSET", "0") == "1"

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
            # DUET: capture split-K1/K2 layout CudaGraphs
            # (split_k1_{long,short} + split_k2).
            if self.config.duet_enabled and not self.enforce_eager:
                from ssd.engine.helpers.cudagraph_helpers import (
                    capture_fi_tree_decode_cudagraph,
                )
                _layouts_to_capture = []
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
                print(f'[DUET] Captured FI tree decode CudaGraphs ({len(_layouts_to_capture)} layouts)', flush=True)
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
        # DUET: keys[:_last_n_draft_keys] = phase 1 (draft-sourced),
        # keys[_last_n_draft_keys:] = phase 2 (proxy-sourced). 0 = non-DUET / not yet populated.
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

        # full_layout: 기존 SSD용 (non-DUET + DUET 비활성)
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

        # DUET split-only K1/K2 mode layouts (per docs/duet/04-split-k1k2-design.md):
        #   - split_k1_long  : K=K1, position_count=K1+1 (draft pass, valid_k=K1 hit / miss)
        #   - split_k1_short : K=K1, position_count=K2+1 (draft pass, valid_k=K2 hit; K2<=K1 only)
        #   - split_k2       : K=K2, position_count=K2+1 (proxy pass)
        # No continuation. Supported scope: K2 <= K1.
        # Hybrid / legacy two-pass layouts were removed (2026-07); config
        # enforces SSD_FORCE_SPLIT_K1K2=1 + duet_phase1_k/duet_phase2_k set.
        if self.config.duet_enabled:
            draft_fo = self.config.duet_draft_fan_out
            proxy_fo = self.config.duet_proxy_fan_out
            self.split_k1_long_layout = None
            self.split_k1_short_layout = None
            self.split_k2_layout = None
            K1 = self.config.duet_phase1_k
            K2 = self.config.duet_phase2_k
            # Hard check (assert is stripped under python -O).
            if K2 > K1:
                raise ValueError(
                    f"split-K1K2 mode requires K2 <= K1, got K1={K1}, K2={K2}. "
                    f"K2>K1 is not supported (proxy_horizon tail source undefined). "
                    f"See docs/duet/04-split-k1k2-design.md."
                )
            # Phase 1 fan_out_list (non-uniform supported).
            # Phase 2 always uniform proxy_fo per position (Phase 2 selection
            # logic / target proxy compute is uniform; non-uniform Phase 2
            # would need policy-based dynamic fan_out — separate design).
            _p1_fol_user = self.config.duet_split_phase1_fan_out_list
            if _p1_fol_user is not None:
                if len(_p1_fol_user) != K1 + 1:
                    raise ValueError(
                        f"duet_split_phase1_fan_out_list len={len(_p1_fol_user)} "
                        f"must equal K1+1={K1 + 1}, got {_p1_fol_user}"
                    )
                _p1_fol_long = list(_p1_fol_user)
            else:
                _p1_fol_long = [draft_fo] * (K1 + 1)
            if self.config.duet_split_phase2_fan_out_list is not None:
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
            # docs/duet/05-policy-b-fix.md Section 3.7 (옵션 A-1):
            #   K = K2 (forward depth, fixed)
            #   MQ_LEN = pfo*(K_max+1) (worst-case sizing)
            #   position_count / fan_out_list updated per-step in caller.
            K_rank_max = K1 if K1 >= K2 else K2     # = K1 (K2 ≤ K1 invariant)
            # Tier-3: capture-time placeholder list sums to the single-source
            # total budget (== [pfo]*(K_max+1) exactly when duet_p2_budget
            # unset; per-step values are rewritten by the caller anyway).
            _p2_total = self.config.duet_proxy_total_budget
            _p2_base, _p2_rem = divmod(int(_p2_total), K_rank_max + 1)
            _p2_list = [_p2_base + (1 if i < _p2_rem else 0)
                        for i in range(K_rank_max + 1)]
            self.split_k2_layout = create_tree_layout(
                name="split_k2",
                fan_out_list=_p2_list,
                fan_out_list_miss=list(_p2_list),
                K=K2, device=d, position_count=K_rank_max + 1,
            )
            _short_str = (
                f'split_k1_short MQ={self.split_k1_short_layout.MQ_LEN} '
                f'(K=K1={K1}, pos={K2+1}), '
                if self.split_k1_short_layout is not None else
                'split_k1_short SKIPPED (K1==K2, long bucket reused), '
            )
            print(f'[DUET split-K1K2] layouts: '
                  f'split_k1_long MQ={self.split_k1_long_layout.MQ_LEN} (K=K1={K1}, pos={K1+1}), '
                  f'{_short_str}'
                  f'split_k2 MQ={self.split_k2_layout.MQ_LEN} (K=K2={K2}, pos={K2+1})',
                  flush=True)

        # #D Pre-allocate spec buffers for _decode_tree (per-step 8 MB alloc/zero-fill 제거).
        # Rev1 invariant: sum(fan_out_list) 항상 고정 (proxy_fan_out × (K+1)).
        # B=1이므로 N = max_mq. EAGLE은 DUET와 함께 안 쓰임 (config assert).
        #
        # DUET는 Phase 1, 2 결과를 동시에 보관해야 merge 가능 → **슬롯 2개** 준비.
        # Baseline은 슬롯 1개만 사용.
        mq_list = [self.full_layout.MQ_LEN]
        # Include split-K1/K2 layouts (DUET) — the runtime layouts consumed by
        # _decode_tree; non-uniform Phase 1 lists change their MQ_LEN.
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

        n_slots = 2 if self.config.duet_enabled else 1
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
        if SPLIT_K1K2_MODE and self.config.duet_phase1_k is not None:
            _K1c = self.config.duet_phase1_k
            _K2c = self.config.duet_phase2_k
            # SSD_DUET_JIT_SHORT=1: K2-deep JIT (miss rows become valid_k=K2
            # short-bucket rows; hit_cache_and_respond default must match).
            _jit_K = _K2c if DUET_JIT_SHORT else (_K1c if _K1c >= _K2c else _K2c)
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
        # Draft model now returns full target vocab size logits (after d2t expansion)
        V = self.hf_config.vocab_size
        # T3.4-b3: 트리 wire 필드는 step-국소 — 이전 step 잔재 제거
        self._tree_wire_ints = None
        self._tree_wire_parent_q = None
        self._tree_hit_root = None

        # Init miss slots with valid random logits so token IDs are in-vocab (fixes B>1 crash)
        out_logits = torch.empty((B, K, V), dtype=self.hf_config.torch_dtype, device=self.device).uniform_()
        out_tokens = out_logits.argmax(dim=-1)
        cache_hits = torch.zeros(B, dtype=torch.int64, device=self.device)
        # Per-row valid_k: defaults to K (= K_long for DUET / speculate_k for non-DUET).
        # Phase 4 will override per-row to K_short for proxy-sourced hits.
        # Phase 5 will set miss/JIT path to K_short as well.
        # Split-only K1/K2 mode (per docs/duet/04-split-k1k2-design.md):
        # default valid_k = K_max = max(K1, K2). Cache-hit overwrites with the
        # row's own valid_k (∈ {K1, K2}). NEVER use K_long here in this mode.
        if (
            SPLIT_K1K2_MODE
            and self.config.duet_phase1_k is not None
        ):
            K1 = self.config.duet_phase1_k
            K2 = self.config.duet_phase2_k
            K_max = K1 if K1 >= K2 else K2
            # JIT-short: miss/JIT rows default to valid_k=K2 to match the
            # K2-deep JIT chain (jit_speculate). Hit rows are overwritten
            # with their cached row's valid_k below either way.
            _miss_vk = K2 if DUET_JIT_SHORT else K_max
            valid_k = torch.full((B,), _miss_vk, dtype=torch.int64, device=self.device)
        else:
            valid_k = torch.full((B,), K, dtype=torch.int64, device=self.device)

        assert request_keys.shape == (B, 3), f"ERROR in hit_cache_and_respond: request_keys should be (B, 3), got {request_keys.shape}"
        
        hidden_size = self.hf_config.hidden_size
        out_activations = torch.zeros(
            B, K, hidden_size,
            dtype=self.hf_config.torch_dtype, device=self.device
        ) if self.config.use_eagle else None
        
        # Statistics
        
        if self.config.verbose:
            print(f"[hit_cache_and_respond] Request keys: {request_keys}", flush=True)
            for i in range(B):
                rec_token = request_keys[i, 2].item()
                rec_text = self.tokenizer.decode([rec_token])
                print(f"  Req {i}: token={rec_token} ('{rec_text}')", flush=True)
        
        # DUET-only: per-seq phase classification (0=miss, 1=phase 1 draft, 2=phase 2 proxy).
        # All zeros for non-DUET — verifier silently accumulates 0s and reports 0 phase rates.
        phase_source = torch.zeros(B, dtype=torch.int64, device=self.device)
        if self.tree_cache_keys.numel() > 0:
            # Vectorized membership against tensor cache
            eq = (request_keys.unsqueeze(1) == self.tree_cache_keys.unsqueeze(0))  # [B,T,3]
            match = torch.all(eq, dim=2)  # [B,T]
            cache_hits = match.any(dim=1)  # [B]

            if self.config.duet_enabled and self._last_n_draft_keys > 0:
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
            
            # Fill logic (M2, docs/duet/13 §2 — mixed hit/miss fix):
            #   - no JIT: fill hit rows from cache (miss rows stay random-init).
            #   - JIT + any miss: run the batched JIT for ALL rows (latency-
            #     bound, extra rows ~free), THEN overwrite hit rows from the
            #     cache below. Pre-M2 this case skipped the cache fill
            #     entirely — one miss clobbered every hit row in the batch
            #     with JIT output while their valid_k/phase_source still said
            #     "hit" (the JIT-all-clobbers-hits bug; impossible at B=1).
            #   - JIT + all hit: cache fill only (JIT skipped), unchanged.
            # The _any_hit / cache_hits.all() __bool__ syncs replace the
            # identical cache_hits.any()/.all() syncs the pre-M2 branch
            # condition already paid — no new hot-path GPU sync (.all() is
            # only evaluated on the JIT path, exactly as before).
            _any_hit = bool(cache_hits.any())
            if self.config.jit_speculate and not (_any_hit and bool(cache_hits.all())):
                # print(f'[hit_cache_and_respond] found a cache miss, running jit speculate', flush=True)
                if self.config.verbose:
                    print(f"[hit_cache_and_respond] Running JIT speculate for cache misses", flush=True)
                if DUET_JIT_SUBSET and _any_hit and not self.config.use_eagle:
                    # Mixed batch, subset gate on: JIT only the miss rows.
                    # nonzero() syncs on the data-dependent shape — this
                    # replaces the .all() sync of the branch condition class,
                    # not an extra hot-path sync category. Hit rows are never
                    # touched by the JIT here; the cache overwrite below is
                    # then a pure fill (nothing of theirs to restore).
                    miss_idx = (~cache_hits).nonzero(as_tuple=False).squeeze(1)
                    sub_logits = out_logits[miss_idx]
                    sub_tokens = out_tokens[miss_idx]
                    self.jit_speculate(
                        request_keys[miss_idx],
                        num_tokens[miss_idx],
                        sub_logits,
                        sub_tokens,
                        temperatures[miss_idx],
                        draft_block_tables[miss_idx],
                        None,
                        )  # write into sub_logits, sub_tokens (compact [n])
                    out_logits[miss_idx] = sub_logits
                    out_tokens[miss_idx] = sub_tokens
                else:
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
            if _any_hit:
                # Overwrite hit rows from the cache — runs AFTER the JIT so
                # mixed batches keep cached tokens/logits on hit rows. Their
                # valid_k was already pulled from the matched cache row above;
                # miss rows keep the JIT default (_miss_vk) and phase_source 0.
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
                # === T3.4-b3: P2-tree hit 서빙 (B=1) — 뷰가 체인 행을 대체.
                # out_tokens = 뷰 노드(생성 순서), valid_k = 유효 노드 수 →
                # 기존 extend/prepare 기계가 트리 행을 그대로 나른다 (창 =
                # [rec]+뷰, scratch 셀 선형 — 리뷰4 row 계약). topology와
                # parent_q는 wire 트리 블록(b2 스플라이스)으로 동승.
                _tviews = getattr(self, "_tree_views", None)
                if (_tviews is not None and B == 1
                        and bool(cache_hits[0])
                        and int(phase_source[0]) == 2):
                    from ssd.engine.helpers.p2_tree import pack_tree_ints
                    _root = int(idx[0]) - self._last_n_draft_keys
                    _nv = self.config.duet_tree_nv
                    _n_valid = int(_tviews["valid"][_root])
                    if _n_valid > 0:
                        out_tokens[0, :_nv] = \
                            _tviews["tok"][_root].to(self.device)
                        valid_k[0] = _n_valid
                        _packed_ints = pack_tree_ints(_tviews, _root, _nv)
                        if int(_packed_ints[0]) != _n_valid:
                            raise RuntimeError(
                                f"tree serve invariant: pack valid="
                                f"{int(_packed_ints[0])} != _n_valid="
                                f"{_n_valid} root={_root}")
                        self._tree_wire_ints = _packed_ints.to(self.device)
                        self._tree_wire_parent_q = \
                            _tviews["parent_q_logits"][_root].to(
                                device=self.device,
                                dtype=out_logits.dtype).unsqueeze(0)
                        self._tree_hit_root = _root
                        # b6-2 (D14): 다음 요청에서 수락 경로 재실체화에
                        # 쓸 서빙 스냅샷 (double buffer — 다음 build가
                        # _tree_views를 덮어써도 유지)
                        self._tree_served_ints = _packed_ints.clone()
                        self._tree_served_numtok = int(num_tokens[0])
                        # 이슈 #35: staging 소비 가드용 seq 정체 기록
                        self._tree_served_seq = int(request_keys[0, 0])
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
        # short-hit=K_short+1). M2 (docs/duet/13 §1): the batch dispatch
        # scalar is vk_max = max(valid_k) — a sync SWAP of the old seq-0
        # [0].item() capture, still exactly one GPU→CPU sync.
        # Rows with vk_i < vk_max feed filler to the vk_max-wide glue
        # beyond their own vk_i (cache-padding zeros for short hit rows,
        # random-init in-vocab tokens for JIT-short miss rows). That is
        # ACCEPTABLE:
        # their glue logits past vk_i can never surface as accepted
        # tokens — phase-1 fork selection slices positions per layout
        # (uniform width), and the M1 verify clamp
        # (accept_until = min(accept_until, vk_i)) blocks acceptance past
        # vk_i, so forks seeded past a short row's vk_i are unreachable
        # cache rows (v1 padding cost, not a correctness issue).
        # Per-seq valid_k [B] rides the response wire unchanged; this
        # scalar is ONLY the batch-level dispatch width (glue bucket /
        # phase-1 layout / TP-broadcast scalar).
        # Sync valid_k to Python ONLY when split-K1/K2 mode actually needs
        # the per-step bucket dispatch. Non-DUET: width is always
        # K_long; pass None so make_glue_decode_input_ids defaults to it
        # without forcing a CUDA stream flush. Returned as the 8th element
        # so downstream callers (_glue_decode, _decode_speculate_async)
        # avoid re-syncing the same value.
        if self.config.duet_phase1_k is not None and valid_k.numel() > 0:
            _vk_scalar = int(valid_k.max().item())
        else:
            _vk_scalar = None
        _profile_cache_status = None
        if os.environ.get("SSD_PROFILE_DUET", "0") == "1" and cache_hits is not None:
            if B == 1:
                _hit = int(cache_hits[0].item())
                if _hit == 0:
                    _profile_cache_status = "miss"
                else:
                    _src = int(phase_source[0].item()) if phase_source is not None else 0
                    if _src == 1:
                        _profile_cache_status = "hit_k1"
                    elif _src == 2:
                        _profile_cache_status = "hit_k2"
                    else:
                        _profile_cache_status = "hit"
            else:
                _hits = cache_hits.tolist()
                _profile_cache_status = (
                    "miss" if all(h == 0 for h in _hits)
                    else "hit" if all(h == 1 for h in _hits)
                    else "mixed"
                )
        # ===== TRACE point 1: hit_cache_and_respond boundary =====
        if TRACE_SPLIT_K1K2 and SPLIT_K1K2_MODE and self.config.duet_phase1_k is not None:
            K1 = self.config.duet_phase1_k
            K2 = self.config.duet_phase2_k
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
        return (
            out_tokens, out_logits, glue_input_ids, cache_hits, out_activations,
            phase_source, valid_k, _vk_scalar, _profile_cache_status,
        )

    def _service_spec_request(self):
        """Receives a speculation request, serves it from cache, and sends results back in a single response."""
        # Phase B: meta wire is [B, K, F, step_id]. step_id is the per-target
        # spec request id; we set it on the profiler context so all draft
        # spans for this request inherit it. See docs/duet/06 §4.4.
        from ssd.engine.helpers.cudagraph_helpers import (
            duet_record as _mr_dr, duet_close as _mc_dr,
            duet_set_context as _mctx_dr, duet_clear_status as _mclr_dr,
        )
        _mev_dr = _mr_dr("draft_recv_request")
        meta = self.recv_tensor((4,), torch.int64)
        B, K, F, step_id = meta.tolist()
        # Announce new step_id + proc=draft. Clear any leftover status from
        # the previous request so spans before hit_cache_and_respond don't
        # inherit it; status will be set explicitly below once known.
        _mctx_dr(step_id=step_id, proc="draft")
        _mclr_dr()

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
        if _E0_TRACE:  # E0: 직전 step의 실제 outcome (P0, docs/duet/17 §2)
            self._e0_step_id = step_id
            _e0.record_draft_request(step_id, cache_keys, temps_as_int64)
        assert off == fused_total
        temperatures = temps_as_int64.to(torch.int32).view(torch.float32)

        # === T3.4-b6-2 (D14): 직전 step이 트리 서빙이었으면 수락 경로
        # KV를 뷰-순서 셀 → canonical 셀로 복사. rope는 트리 rope
        # (pos0+1+depth)가 이미 canonical 위치와 같아 셀 이동만으로
        # 재실체화 완성. k_idx = 종단 노드 id (b6-1), a_eff는
        # num_tokens 델타 (EOS/max-token 절단을 자동 반영 — 경로
        # prefix == 수락 prefix). one-shot 소거.
        _served = getattr(self, "_tree_served_ints", None)
        if _served is not None:
            self._tree_served_ints = None
            _staged = getattr(self, "_tree_staged_kv", None)
            self._tree_staged_kv = None
            _a_eff = int(num_tokens[0]) - self._tree_served_numtok - 1
            _T = int(cache_keys[0, 1])
            # 이슈 #35: 서빙 시점 seq와 다른 seq의 요청이면 (종료 후 새
            # seq 진입 등) staged KV는 폐기 — a_eff/k_idx가 새 seq
            # 좌표로 우연히 유효해 보여도 KV 오염이므로 정체 대조 필수.
            if int(seq_ids[0]) != getattr(self, "_tree_served_seq", -1):
                _staged = None
            if _a_eff > 0 and _T > 0 and _staged is not None:
                _nv_s = self.config.duet_tree_nv
                _par = _served[3 + _nv_s:3 + 2 * _nv_s]
                _path = []
                _j = _T - 1
                while _j >= 0:
                    _path.append(_j)
                    _j = int(_par[_j])
                _path.reverse()
                _path = _path[:_a_eff]
                # 이슈 #21: src = staging(글루 시점 gather — 물리블록
                # 해제와 무관), dst = 현재 dbt의 canonical 슬롯.
                # staging 행 j+1 = 뷰 노드 j ([0]=rec).
                _p0 = self._tree_served_numtok - 1
                _bs = self.block_size
                _max_pos = _p0 + 1 + len(_path) + 1
                _dbt0 = draft_block_tables[0, :(_max_pos // _bs) + 1].cpu()
                _src_rows, _dst = [], []
                for _k, _jj in enumerate(_path):
                    _dp = _p0 + 1 + _k
                    _src_rows.append(1 + _jj)
                    _dst.append(int(_dbt0[_dp // _bs]) * _bs + _dp % _bs)
                if _src_rows:
                    kc = self.kv_cache
                    _flat = kc.view(kc.shape[0], kc.shape[1], -1,
                                    kc.shape[4], kc.shape[5])
                    _sr = torch.tensor(_src_rows, dtype=torch.int64,
                                       device=kc.device)
                    _dt = torch.tensor(_dst, dtype=torch.int64,
                                       device=kc.device)
                    _flat[:, :, _dt] = _staged[:, :, _sr]

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

        # Close draft_recv_request after the last payload recv (fused + EAGLE
        # if applicable). docs/duet/06 §4.5.
        _mc_dr("draft_recv_request", _mev_dr)

        from ssd.engine.helpers.cudagraph_helpers import duet_record as _mr_h, duet_close as _mc_h
        _mev_hc = _mr_h("hit_cache_respond")
        (
            out_tokens, out_logits, glue_decode_input_ids, cache_hits,
            out_activations, phase_source, valid_k, valid_k_scalar,
            profile_cache_status,
        ) = self.hit_cache_and_respond(
            cache_keys, B, K, num_tokens, temperatures, draft_block_tables,
            target_recovery_activations,
        )
        # Set close-time status so hit_cache_respond row carries it AND any
        # subsequent draft spans for this request inherit it.
        _mctx_dr(status=profile_cache_status)
        _hc_label = f"hit_cache_respond_{profile_cache_status}" if profile_cache_status else "hit_cache_respond"
        _mc_h(_hc_label, _mev_hc)

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

        if _E0_TRACE:  # E0: 이번에 서빙한 응답 (hit 유형/깊이/토큰 — P0)
            _e0.record_draft_response(
                getattr(self, "_e0_step_id", -1), phase_source, valid_k,
                out_tokens[:, :K])
        # Wire layout matches speculator_async._fused_response: [cache_hits, phase_source, valid_k, out_tokens].
        _fused_parts = [cache_hits.reshape(-1).to(torch.int64),
                        phase_source.reshape(-1),
                        valid_k.reshape(-1).to(torch.int64),
                        out_tokens.reshape(-1).to(torch.int64)]
        if self.config.duet_tree_policy != "off":
            # T3.4-b2: 트리 블록 동승 (max-padded — hit 없으면 zero 블록,
            # 크기·호출 순서 불변). 뷰 내용 채움은 respond 경로(b3).
            _tb = getattr(self, "_tree_wire_ints", None)
            _nv = self.config.duet_tree_nv
            from ssd.engine.helpers.p2_tree import tree_wire_ints_len
            if _tb is None:
                _tb = torch.zeros(B * tree_wire_ints_len(_nv),
                                  dtype=torch.int64, device=self.device)
            _fused_parts.append(_tb.reshape(-1))
        fused_response = torch.cat(_fused_parts)
        from ssd.engine.helpers.cudagraph_helpers import duet_record as _mr_s, duet_close as _mc_s
        _mev_ds = _mr_s("draft_send_response")
        dist.send(fused_response, dst=0, group=self.async_pg)
        dist.send(out_logits[:, :K, :].contiguous(), dst=0, group=self.async_pg)
        if self.config.duet_tree_policy != "off":
            _pq = getattr(self, "_tree_wire_parent_q", None)
            if _pq is None:
                _pq = torch.zeros(B, self.config.duet_tree_nv,
                                  self.hf_config.vocab_size,
                                  dtype=out_logits.dtype, device=self.device)
            dist.send(_pq.contiguous(), dst=0, group=self.async_pg)
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
            # (or K_long fallback for misses). Drives glue / phase1 / proxy
            # layout dispatch in _build_tree_batch_split_k1k2.
            "valid_k": valid_k,
            # Pre-sync'd Python int (DUET split mode) or None (non-DUET).
            # Lets _glue_decode and the main hot path skip redundant .item()
            # syncs — the .item() that produced this scalar happened once
            # inside hit_cache_and_respond above.
            "valid_k_scalar": valid_k_scalar,
        }
        if self.config.duet_proxy_on_draft:
            # Raw-proxy mode needs this step's logits_q to compute Policy B
            # locally (_policy_b_from_raw_proxy).
            partial_tree_decode_args["out_logits"] = out_logits

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

        _layout = self.full_layout  # _construct_tree_decode_args is only used in non-DUET path
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
        from ssd.engine.helpers.cudagraph_helpers import duet_record, duet_close
        _mev_glue = duet_record("glue")
        dbt = partial_tree_decode_args["dbt"]
        cache_hits = partial_tree_decode_args["cache_hits"]
        cache_hits_list = cache_hits.tolist()

        if self.config.use_eagle:
            # EAGLE glue decode — full path (unchanged, passes through to _build_tree_batch)
            raise NotImplementedError("_glue_decode does not support EAGLE; use _build_tree_batch directly")

        # Step 9B-0: glue width follows valid_k. Read the pre-sync'd Python
        # scalar that hit_cache_and_respond stashed in partial_tree_decode_args
        # — None means split / pure-DUET path, default to K_long. No new
        # GPU→CPU sync here.
        _vk_glue = partial_tree_decode_args.get("valid_k_scalar")
        if _vk_glue is None:
            # Split-only K1/K2 mode: default = K_max, NOT K_long.
            if SPLIT_K1K2_MODE and self.config.duet_phase1_k is not None:
                K1 = self.config.duet_phase1_k
                K2 = self.config.duet_phase2_k
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

        duet_close("glue", _mev_glue)
        return glue_decode_logits, gd_for_fork, cache_hits, cache_hits_list, dbt, B

    def _tree_glue_decode(self, partial_tree_decode_args,
                          glue_decode_input_ids):
        """TREE_GLUE (T3.4-b3-3, v6 §7.5 ⓐ / 결정④): 트리-hit step의 글루.

        입력 = [rec] + 뷰 노드(생성 순서, 폭 n_valid+1). 체인 글루와 달리
        노드 j는 prefix + rec + 조상(j)만 봐야 하므로 causal varlen 대신
        **split_k2 tree-decode CG(step 0) + packed mask override**로 한
        번의 W-폭 MQ forward를 돌린다. KV는 체인과 동일한 canonical
        scratch 셀(pos0+1+j, 뷰 순서 — target verify row 계약과 일치)에
        쓰이고, rope만 pos0+1+depth(j)로 덮어쓴다. B=1 전용.

        Returns: (glue_logits [1, n_valid+1, V], gd_for_fork [1, n_valid+1],
                  cache_hits, cache_hits_list, dbt, B)
        """
        import numpy as _np
        from ssd.engine.helpers import cudagraph_helpers as _CH
        from ssd.engine.helpers.cudagraph_helpers import duet_record, duet_close
        _mev_glue = duet_record("glue")
        dbt = partial_tree_decode_args["dbt"]
        cache_hits = partial_tree_decode_args["cache_hits"]
        cache_hits_list = cache_hits.tolist()
        num_tokens = partial_tree_decode_args["num_tokens"]
        V = self.hf_config.vocab_size
        _layout = self.split_k2_layout
        W = _layout.MQ_LEN

        _root = self._tree_hit_root
        _views = self._tree_views
        par = _views["parent_local"][_root]
        n_valid = int(_views["valid"][_root])
        n_rows = n_valid + 1
        if glue_decode_input_ids.numel() != n_rows:
            raise RuntimeError(
                f"tree glue width mismatch: ids="
                f"{glue_decode_input_ids.numel()} vs n_valid+1={n_rows}")
        if n_rows > W:
            raise RuntimeError(f"tree glue rows {n_rows} > W {W}")

        # 노드별 depth/조상 (parent_local은 항상 자기보다 앞 — 뷰 invariant)
        depth = [0] * n_valid
        anc = [[] for _ in range(n_valid)]
        for j in range(n_valid):
            p = int(par[j])
            if p >= 0:
                depth[j] = depth[p] + 1
                anc[j] = anc[p] + [p]

        ctxt = self.prepare_glue_decode_ctxt(
            num_tokens=num_tokens, input_ids=glue_decode_input_ids,
            dbt=dbt, B=1, valid_k=n_valid)
        pos0 = int(ctxt["positions"][0])
        cols = int(ctxt["context_lens"][0])

        # W-폭 패딩: pad 행은 slot -1 (KV 미기록), rope pos0, prefix-only mask
        input_w = torch.zeros(W, dtype=torch.int64, device=self.device)
        input_w[:n_rows] = glue_decode_input_ids.reshape(-1)
        rope_w = torch.full((W,), pos0, dtype=torch.int64, device=self.device)
        rope_w[1:n_rows] = pos0 + 1 + torch.tensor(
            depth, dtype=torch.int64, device=self.device)
        slot_w = torch.full((W,), -1, dtype=torch.int32, device=self.device)
        slot_w[:n_rows] = ctxt["slot_map"]

        # mask: [W, cols] = [prefix 1s | rec+노드 셀 (n_rows)]
        prefix_len = cols - n_rows
        m = _np.zeros((W, cols), dtype=_np.uint8)
        m[:, :prefix_len] = 1                      # pad 행 포함 (NaN 방지)
        m[0, prefix_len] = 1                       # rec: prefix+self
        for j in range(n_valid):
            r = 1 + j
            m[r, prefix_len] = 1                   # rec
            for a in anc[j]:
                m[r, prefix_len + 1 + a] = 1       # 조상 노드
            m[r, prefix_len + 1 + j] = 1           # self
        packed = _np.packbits(m.ravel(), bitorder="little")
        indptr = _np.array([0, len(packed)], dtype=_np.int32)

        _CH.cache["_tree_mask_override"] = {0: (
            torch.from_numpy(packed).to(self.device),
            torch.from_numpy(indptr).to(self.device))}
        set_context(
            is_prefill=False,
            slot_mapping=slot_w,
            context_lens=ctxt["context_lens"].to(torch.int32),
            block_tables=dbt,
            active_mq_len=W,
            active_wrappers=self.prefill_wrappers_by_layout.get(_layout.name),
            active_layout=_layout,
            active_cache_hits_list=cache_hits_list,
        )
        try:
            logits = self.run_model(
                input_w, rope_w, is_prefill=False, last_only=False,
                tree_decode_step=0, cache_hits=cache_hits)
        finally:
            _CH.cache.pop("_tree_mask_override", None)
            reset_context()

        glue_logits = logits.view(-1, V)[:n_rows].view(1, n_rows, V)
        gd_for_fork = glue_decode_input_ids.reshape(1, n_rows)
        # 이슈 #21: 뷰 노드 KV를 staging으로 gather — scheduler가 excess
        # draft block을 해제해도 다음 요청의 수락경로 재실체화가 물리
        # 블록 생존에 의존하지 않게 한다 (리뷰 2D). [rec+노드] 전체
        # (n_rows 셀), scatter는 b6-2에서 경로만.
        kc = self.kv_cache
        _flat = kc.view(kc.shape[0], kc.shape[1], -1, kc.shape[4],
                        kc.shape[5])
        _slots_dev = ctxt["slot_map"].to(torch.int64)
        self._tree_staged_kv = _flat[:, :, _slots_dev].clone()
        duet_close("glue", _mev_glue)
        return glue_logits, gd_for_fork, cache_hits, cache_hits_list, dbt, 1

    def _invalidate_zero_valid_tree_keys(self):
        """이슈 #14: valid==0 root는 서빙 실체(뷰)가 없다 — zero backbone
        행이 체인 응답으로 나가면 q가 실제 제안분포가 아니게 되어 수락
        보존이 깨진다. 키를 -1로 무효화해 hit 자체를 봉쇄 (miss → JIT)."""
        _zv = (self._tree_views["valid"] == 0).nonzero().flatten()
        if _zv.numel():
            self.tree_cache_keys[
                self._last_n_draft_keys + _zv.to(self.device)] = -1

    def _tree_backbone_project(self, pool, R, K2, cell_logits):
        """rollout pool → 체인-호환 populate 입력 [R, K2] (backbone=맏이 사슬).

        키/valid_k/serving 형식을 체인과 동일하게 유지하기 위한 투영 —
        실제 응답 내용은 뷰(self._tree_views)가 담당한다.
        """
        V = self.hf_config.vocab_size
        _bt = torch.zeros(R, K2, dtype=torch.int64)
        # 이슈 #13: fp32면 fp16 draft_logits와 cat에서 dtype 크래시
        _dev = cell_logits.device if cell_logits is not None else "cpu"
        _bl = torch.zeros(R, K2, V, dtype=self.hf_config.torch_dtype,
                          device=_dev)
        _first_child = {}
        for i in range(pool.n):
            _pi = int(pool.parent_idx[i])
            if _pi >= 0 and int(pool.sib_order[i]) == 0 \
                    and _pi not in _first_child:
                _first_child[_pi] = i
        for r in range(R):
            cur, d = r, 0
            while cur in _first_child and d < K2:
                nx = _first_child[cur]
                _bt[r, d] = pool.tok[nx]
                _pc = int(pool.parent_cell[nx])
                if cell_logits is not None and _pc >= 0:
                    _bl[r, d] = cell_logits[_pc]
                cur, d = nx, d + 1
        return _bt.to(self.device), _bl.to(self.device)

    def _select_tree_fork_tokens(self, glue_logits, fan_counts, child_toks):
        """트리 step P1 fork 선택 (T3.4-b3-4, 결정④ ⓑ). B=1 전용.

        체인 선택기의 '다음 토큰 제외'를 일반화: 컨텍스트 c의 제외 집합 =
        뷰 안에 이미 있는 c의 자식 토큰들. 컨텍스트별 top-fan_counts[c].

        Returns: (flat [1, sum(fan_counts)], padded [1, n_rows, max_fo],
                  mask [n_rows, max_fo] bool)
        """
        lg = glue_logits[0].clone()                     # [n_rows, V]
        n_rows = lg.shape[0]
        for c, toks in enumerate(child_toks):
            if toks:
                lg[c, torch.tensor(toks, dtype=torch.int64,
                                   device=lg.device)] = float("-inf")
        max_fo = max(fan_counts)
        top = lg.topk(max_fo, dim=-1).indices           # [n_rows, max_fo]
        flat = torch.cat([top[c, :fan_counts[c]] for c in range(n_rows)])
        padded = torch.zeros(1, n_rows, max_fo, dtype=torch.int64,
                             device=lg.device)
        mask = torch.zeros(n_rows, max_fo, dtype=torch.bool,
                           device=lg.device)
        for c in range(n_rows):
            padded[0, c, :fan_counts[c]] = top[c, :fan_counts[c]]
            mask[c, :fan_counts[c]] = True
        return flat.view(1, -1), padded, mask

    def _tree_step_p1p2(self, partial_tree_decode_args, glue_logits,
                        cache_hits, cache_hits_list, dbt,
                        proxy_recv_work, proxy_buf):
        """트리-hit step의 P1/P2 (T3.4-b3-4/-5, 결정④ ⓑⓒ). B=1 전용.

        P1: fork 컨텍스트 = [rec(root-종단), 뷰 노드 0..n_valid-1] —
        fan_idx가 그대로 종단 노드 id 네임스페이스가 된다. CG 불변을
        위해 MQ 폭은 split_k1_long 그대로 두고 컨텍스트별 fanout을
        균등-우선 재배분 (F7 예산 재검토 전 v1). fork 행 mask = 조상
        비트맵 (전 step override), rope = 컨텍스트 depth 기반.
        """
        import numpy as _np
        from ssd.engine.helpers import cudagraph_helpers as _CH
        from ssd.engine.helpers.cudagraph_helpers import (
            duet_record as _mr, duet_close as _mc)

        K1 = self.config.duet_phase1_k
        _layout_k1 = self.split_k1_long_layout
        W1 = _layout_k1.MQ_LEN
        n_rows = glue_logits.shape[1]
        n_valid = n_rows - 1
        _views = self._tree_views
        _root = self._tree_hit_root
        par = _views["parent_local"][_root]
        vtok = _views["tok"][_root]

        # 컨텍스트 topology: depth / 조상 / 뷰-내 자식 토큰(fork 제외 집합)
        depths = [0] * n_valid
        anc = [[] for _ in range(n_valid)]
        child_toks = [[] for _ in range(n_rows)]
        depth_ctx = [0] * n_rows              # ctx0(rec)=0, ctx 1+j = 1+depth_j
        for j in range(n_valid):
            p = int(par[j])
            if p >= 0:
                depths[j] = depths[p] + 1
                anc[j] = anc[p] + [p]
            child_toks[p + 1].append(int(vtok[j]))
            depth_ctx[1 + j] = 1 + depths[j]

        # === P1: 노드-fork, K1 forwards ===
        _mev_p1b = _mr("phase1_build")
        # 이슈 #31 (리뷰2-5): 균등 배분(W1//n_rows)은 rec(최빈 종단
        # 컨텍스트)에도 소수 lane만 줘 P1 궤적 품질을 깎는다 (verdict
        # 회계: P1 축 −0.054 tok/step — AL 동률의 주범). 종단질량
        # prior ∝ a^depth·(1−a)^{자식수} (a = per-depth 수락률 적합
        # ~0.52 — docs/duet/18 λ·E1 α)로 가중, 바닥 1 lane (전 ctx
        # 생존), largest-remainder 반올림 (합 = W1, CG 폭 불변).
        if W1 >= n_rows:
            # 리뷰3-8의 presib 보정(reach에 (1−A)^sib_order 포함)은
            # 고정-A '모델'로는 정확하나 **동일-시드 A/B에서 P1AL
            # 4.13→3.88·P2AL 2.13→1.94 회귀** (2026-08-04, 8×256) —
            # 실제 둘째-형제 조건부 수락은 λ-할인(18번: a₂≈1−(1−α)^0.52)
            # 으로 고정-A 예측보다 높아, 미보정형이 우연히 보상한다.
            # 경험 우선으로 #31 형태 유지; 진짜 교정은 depth/sib별
            # calibrated prior (T6 부채 — 리뷰3도 동일 제안).
            _A = 0.52
            w_ctx = [(_A ** depth_ctx[c])
                     * ((1.0 - _A) ** len(child_toks[c]))
                     for c in range(n_rows)]
            _ws = sum(w_ctx)
            _extra = W1 - n_rows
            _quota = [_extra * w / _ws for w in w_ctx]
            fan_counts = [1 + int(q) for q in _quota]
            _rem = W1 - sum(fan_counts)
            _ord = sorted(range(n_rows),
                          key=lambda c: _quota[c] - int(_quota[c]),
                          reverse=True)
            for c in _ord[:_rem]:
                fan_counts[c] += 1
        else:
            fan_counts = [W1 // n_rows + (1 if i < W1 % n_rows else 0)
                          for i in range(n_rows)]
        draft_forked_k1, draft_forked_p1_padded, draft_forked_p1_mask = \
            self._select_tree_fork_tokens(glue_logits, fan_counts, child_toks)
        draft_tree_args = self._build_tree_decode_args_for_layout(
            partial_tree_decode_args, draft_forked_k1, _layout_k1,
            cache_hits_list)
        _ctx_of_row = torch.repeat_interleave(
            torch.arange(n_rows, dtype=torch.int64),
            torch.tensor(fan_counts, dtype=torch.int64))
        pos0 = int(partial_tree_decode_args["num_tokens"][0]) - 1
        draft_tree_args["rope_positions"] = torch.tensor(
            [pos0 + 1 + depth_ctx[int(c)] for c in _ctx_of_row],
            dtype=torch.int64, device=self.device)

        # 전 step packed mask: [prefix+rec | 조상 노드 셀 | 자기 행 체인 셀]
        base = int(draft_tree_args["positions"][0])   # = pos0 + glue_offset
        ov = {}
        for f in range(K1):
            cols = base + (f + 1) * W1
            m = _np.zeros((W1, cols), dtype=_np.uint8)
            m[:, :pos0 + 1] = 1                       # prefix + rec 셀
            for r in range(W1):
                c = int(_ctx_of_row[r])
                if c > 0:
                    j = c - 1
                    for a in anc[j]:
                        m[r, pos0 + 1 + a] = 1
                    m[r, pos0 + 1 + j] = 1
                for s in range(f + 1):
                    m[r, base + s * W1 + r] = 1
            packed = _np.packbits(m.ravel(), bitorder="little")
            indptr = _np.array([0, len(packed)], dtype=_np.int32)
            ov[f] = (torch.from_numpy(packed).to(self.device),
                     torch.from_numpy(indptr).to(self.device))
        _mc("phase1_build", _mev_p1b)
        _CH.cache["_tree_mask_override"] = ov
        try:
            draft_tokens, draft_logits, draft_acts = self._decode_tree(
                draft_tree_args, layout=_layout_k1)
        finally:
            _CH.cache.pop("_tree_mask_override", None)

        # === proxy 대기 ===
        _mev_pw = _mr("proxy_wait")
        proxy_recv_work.wait()
        _mc("proxy_wait", _mev_pw)
        B = partial_tree_decode_args["num_tokens"].shape[0]
        K = self.config.speculate_k
        duet_proxy = self._unpack_duet_proxy(proxy_buf, B, K)

        # === P2: node-seed rollout (결정④ ⓒ v1) ===
        # 선택기의 위치축이 노드 id 축이 된다 (chosen_pos = 노드 id —
        # 형식 불변, v6 §7.5ⓒ). K_rank = n_valid → position_count = n_rows.
        _mev_p2b = _mr("phase2_build")
        from ssd.engine.helpers.p2_tree import build_root_views
        K2 = self.config.duet_phase2_k
        K_rank = n_valid
        total_budget = self.config.duet_proxy_total_budget
        _sel_out = self._select_proxy_sourced_tokens_unified(
            duet_proxy, draft_forked_p1_padded,
            K_rank=K_rank, total_budget=total_budget,
            draft_forked_mask=draft_forked_p1_mask,
        )
        if len(_sel_out) == 3:
            proxy_forked, proxy_fan_out_tensor, proxy_piv = _sel_out
        else:
            proxy_forked, proxy_fan_out_tensor = _sel_out
            proxy_piv = None
        _layout_k2 = self._update_phase2_layout_inplace(
            proxy_fan_out_tensor, K_rank)
        proxy_tree_args = self._build_tree_decode_args_for_layout(
            partial_tree_decode_args, proxy_forked, _layout_k2,
            cache_hits_list)

        # seed(노드-fork)별 rope/글루 가시성: 컨텍스트 depth·조상 비트맵
        _seed_ctx = torch.repeat_interleave(
            torch.arange(n_rows, dtype=torch.int64),
            proxy_fan_out_tensor[0].cpu())
        proxy_tree_args["rope_positions"] = torch.tensor(
            [pos0 + 1 + depth_ctx[int(c)] for c in _seed_ctx],
            dtype=torch.int64, device=self.device)
        _gro = _np.zeros((len(_seed_ctx), n_rows), dtype=_np.uint8)
        for r, c in enumerate(_seed_ctx.tolist()):
            _gro[r, 0] = 1                        # rec
            if c > 0:
                for a in anc[c - 1]:
                    _gro[r, 1 + a] = 1
                _gro[r, 1 + (c - 1)] = 1
        proxy_tree_args["glue_rows_override"] = _gro
        proxy_tree_args["K_glue_override"] = n_valid
        proxy_tree_args["proxy_piv"] = proxy_piv

        _temps_p2 = proxy_tree_args["temps"]
        _mB, _mK, _mF, _mN = proxy_tree_args["metadata_ints"]
        _, _srp, _scl, _ssm = self._compute_step_positions_and_slot_maps(
            proxy_tree_args["positions"],
            proxy_tree_args["rope_positions"],
            proxy_tree_args["block_tables"],
            _mB, _mK, _mF, _mN, _layout_k2.MQ_LEN, layout=_layout_k2)
        _mc("phase2_build", _mev_p2b)
        pool, _elog, _cell_logits = self._p2tree_rollout(
            duet_proxy, proxy_forked, proxy_fan_out_tensor,
            proxy_tree_args, _layout_k2, _ssm, _scl, _temps_p2)
        _R = _layout_k2.MQ_LEN
        _nv = self.config.duet_tree_nv
        self._tree_views = build_root_views(
            pool, _R, _nv, cell_logits=_cell_logits)
        proxy_tokens, proxy_logits = self._tree_backbone_project(
            pool, _R, K2, _cell_logits)

        self._merge_and_populate_cache(
            draft_tree_args, draft_tokens, draft_logits,
            proxy_tree_args, proxy_tokens, proxy_logits,
            cache_hits_list, draft_acts, None,
            proxy_layout=_layout_k2,
            draft_layout=_layout_k1,
            draft_fan_idx_override=_ctx_of_row.to(self.device))
        self._invalidate_zero_valid_tree_keys()

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
        # Uses full_layout for non-DUET path. DUET path uses _build_tree_batch_duet() instead.
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
        from ssd.engine.helpers.cudagraph_helpers import duet_record as _mr_g, duet_close as _mc_g
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
        # DUET: store glue_decode_logits and gd_for_fork for proxy token swap
        if self.config.duet_enabled:
            tree_decode_args["_duet_glue_logits"] = glue_decode_logits  # [B, K+1, V]
            tree_decode_args["_duet_gd_for_fork"] = gd_for_fork          # [B, K+1]
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
        if self.config.duet_enabled:
            _layout = payload.get("_active_layout")
            _active_mq = _layout.MQ_LEN if _layout and _layout.name != "full" else None
            _active_wrappers = None
            if _active_mq is not None:
                _active_wrappers = self.prefill_wrappers_by_layout.get(_layout.name)
            # Batch 1b: thread cache_hits_list through context so
            # run_fi_tree_decode_cudagraph avoids re-tolist()-ing at step 0
            # (value already known from _glue_decode).
            set_context(
                is_prefill=False,
                slot_mapping=step_slot_maps[depth],
                context_lens=step_context_lens[depth].to(torch.int32),
                block_tables=dbt,
                active_mq_len=_active_mq,
                active_wrappers=_active_wrappers,
                active_layout=_layout,  # runtime layout for dynamic fan_out
                active_cache_hits_list=payload.get("cache_hits_list"),
            )
        else:
            set_context(
                is_prefill=False,
                slot_mapping=step_slot_maps[depth],
                context_lens=step_context_lens[depth].to(torch.int32),
                block_tables=dbt,
                active_cache_hits_list=payload.get("cache_hits_list"),
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
        # DUET는 Phase 1/2 결과를 merge까지 보관해야 하므로 슬롯 2개 round-robin.
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
        # Phase 1 plumbing: every row in the non-DUET / legacy populate path is
        # treated as full-K (= K_long for DUET, speculate_k otherwise). Phase 4
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
    # DUET-SSD: 2-pass tree decode methods
    # ============================================================

    def _irecv_duet_proxy(self, B, K):
        """Post non-blocking recv for proxy. Returns (work, buffer).

        Wire schema (Policy B, unified K+1; docs/duet/05-policy-b-fix.md,
        batched M1 docs/duet/13 §3):
        chosen_pos[B*wire_N] + chosen_tok[B*wire_N] (at B=1 identical to the
        old 2*wire_N single-seq wire).
        Raw-proxy mode (SSD_DUET_PROXY_ON_DRAFT=1, docs/duet/09 WS3):
        fixed-length fused payload, see config.duet_raw_proxy_wire_len
        (B=1 only — docs/duet/13 §6).
        """
        import torch.distributed as dist
        if self.config.duet_proxy_on_draft:
            total_len = self.config.duet_raw_proxy_wire_len
        else:
            wire_N = self.config.duet_proxy_wire_N
            total_len = 2 * B * wire_N
        buf = torch.empty(total_len, dtype=torch.int64, device=self.device)
        work = dist.irecv(buf, src=0, group=self.async_pg)
        return work, buf

    def _unpack_duet_proxy(self, buf, B, K):
        """Unpack proxy data from irecv buffer.

        Returns {"chosen_pos": [B, wire_N], "chosen_tok": [B, wire_N]}
        (Policy B, views into buf — no copy), or the raw top-M dict in
        raw-proxy mode (feed to _policy_b_from_raw_proxy to obtain
        chosen_pos/chosen_tok).
        """
        if self.config.duet_proxy_on_draft:
            M = self.config.duet_proxy_topm
            K_max = max(self.config.duet_phase1_k, self.config.duet_phase2_k)
            Kp1w = K_max + 1
            L1 = Kp1w * M
            return {
                "topk_ids": buf[:L1].view(Kp1w, M),
                "topk_logits": buf[L1:2 * L1].view(torch.float64).float().view(Kp1w, M),
                "lse": buf[2 * L1:2 * L1 + Kp1w].view(torch.float64).float(),
                "y_logit": buf[2 * L1 + Kp1w:].view(torch.float64).float(),
            }
        wire_N = self.config.duet_proxy_wire_N
        chosen_pos = buf[:B * wire_N].view(B, wire_N)
        chosen_tok = buf[B * wire_N:2 * B * wire_N].view(B, wire_N)
        if self.config.duet_tree_policy != "off":
            # T2.0 (v6 D2): dedup **이전** 단일 지점에서 unpack —
            # selector/캐시/rollout에는 순수 토큰만 흘린다.
            from ssd.engine.helpers.p2_tree import unpack_piv
            chosen_tok, chosen_piv = unpack_piv(chosen_tok)
            return {"chosen_pos": chosen_pos, "chosen_tok": chosen_tok,
                    "chosen_piv": chosen_piv}
        return {"chosen_pos": chosen_pos, "chosen_tok": chosen_tok}

    def _p2tree_rollout(self, duet_proxy, proxy_forked, proxy_fan_out_tensor,
                        tree_args, layout, step_slot_maps,
                        step_context_lens, temps):
        """P2-tree rollout 엔진 어댑터 (T1.4b-b — docs/duet/20).

        B=1 전용 (v6: B>1 게이트 OFF). run_rollout 코어에 실엔진
        forward_fn을 주입: 셀 그리드 slot/context는 기존 체인 것을
        재사용하고 input/rope/mask만 동적으로 교체 (mask는
        cudagraph_helpers cache["_tree_mask_override"] 훅으로).
        반환: (pool, eval_log) — 응답 조립은 T2.
        """
        import numpy as _np
        from ssd.engine.helpers import cudagraph_helpers as _CH
        from ssd.engine.helpers import p2_tree as _PT
        from ssd.utils.context import set_context, reset_context

        cfg = self.config
        W = layout.MQ_LEN
        K2 = cfg.duet_phase2_k
        V = self.hf_config.vocab_size
        fo = proxy_fan_out_tensor[0].tolist()
        toks = proxy_forked[0]
        seeds, i = [], 0
        for p, c in enumerate(fo):
            for _ in range(c):
                seeds.append((p, int(toks[i])))
                i += 1
        # T2.1: selector가 관통시킨 seed별 P_iv (pos-그룹 순서 = seeds 순서)
        seed_piv = tree_args.get("proxy_piv")
        # T6 1a (docs/duet/22): SSD_TREE_ARENA=1이면 GPU 상주 rollout —
        # piv를 CPU로 내리지 않는다 (pre 구간 sync 제거).
        _use_arena = os.environ.get("SSD_TREE_ARENA", "0") == "1"
        if _use_arena:
            root_piv = (seed_piv[0].float() if seed_piv is not None
                        else torch.full((len(seeds),), 1e-6,
                                        device=self.device))
        else:
            root_piv = (seed_piv[0].cpu().float()
                        if seed_piv is not None
                        else torch.full((len(seeds),), 1e-6))
        # 이슈 #24: R-W 분리 — 상위 R root 외에는 piv를 0으로 눌러
        # 예산이 가지 않게 한다 (구조·CG·키 폭은 불변; 무예산 root는
        # 뷰 0 → populate 후 #14 키 무효화 경로로 명시적 miss).
        _rc = getattr(cfg, "duet_tree_root_count", None)
        if _rc is not None and _rc < len(seeds):
            _keep = torch.argsort(root_piv, descending=True,
                                  stable=True)[:_rc]
            _mask_r = torch.zeros(len(seeds), dtype=torch.bool,
                                  device=root_piv.device)
            _mask_r[_keep] = True
            root_piv = torch.where(_mask_r, root_piv,
                                   torch.zeros_like(root_piv))
        # 글루 가시성/rope base: seed 행의 것 — 체인 step은 위치-prefix,
        # 트리 step은 조상 비트맵 override (T3.4-b3-5)
        _gro = tree_args.get("glue_rows_override")
        if _gro is not None:
            glue_rows = _gro
            K_glue_used = int(tree_args["K_glue_override"])
        else:
            # 이슈 #7 수정: 글루 폭 = 위치축 길이 (vk+1) — K2+1로 자르면
            # vk=K1 step에서 미래 토큰 누출/조기 절단 (stub은 vk=K2라 통과)
            _n_pos = len(fo)
            glue_rows = _np.zeros((len(seeds), _n_pos), dtype=_np.uint8)
            for r, (p, _t) in enumerate(seeds):
                glue_rows[r, :p + 1] = 1
            K_glue_used = _n_pos - 1
        rope0 = tree_args["rope_positions"]
        if _use_arena and torch.is_tensor(rope0):
            rope_base = rope0[:len(seeds)].to(torch.int64)  # sync 0회
        else:
            rope_base = [int(rope0[r]) for r in range(len(seeds))]
        ctx_len = int(step_context_lens[0][0]) - 0  # chain 빌더 입력과 동일

        dbt = tree_args["block_tables"]
        cache_hits_list = tree_args.get("cache_hits_list") or [1]
        _CH.cache["_tree_mask_override"] = {}

        def forward_fn(f, input_ids_w, rope_w, packed, indptr):
            if torch.is_tensor(packed):            # arena: device 상주
                _CH.cache["_tree_mask_override"][f] = (
                    packed.to(self.device), indptr.to(self.device))
            else:
                _CH.cache["_tree_mask_override"][f] = (
                    torch.from_numpy(packed).to(self.device),
                    torch.from_numpy(indptr.astype(_np.int32))
                    .to(self.device))
            set_context(
                is_prefill=False,
                slot_mapping=step_slot_maps[f],
                context_lens=step_context_lens[f].to(torch.int32),
                block_tables=dbt,
                active_mq_len=W,
                active_wrappers=self.prefill_wrappers_by_layout.get(
                    layout.name),
                active_layout=layout,
                active_cache_hits_list=cache_hits_list,
            )
            logits = self.run_model(
                input_ids_w.to(self.device), rope_w.to(self.device),
                is_prefill=False, last_only=False, tree_decode_step=f,
                cache_hits=tree_args.get("cache_hits"))
            reset_context()
            return logits.view(-1, V)[:W].float()   # GPU 상주 (CPU 왕복 제거)

        try:
            if _use_arena:
                _ar, eval_log, cell_logits = _PT.run_rollout_arena(
                    toks[:len(seeds)], root_piv,
                    policy=cfg.duet_tree_policy, W=W, F_total=K2,
                    c_tensor=cfg.duet_tree_c_tensor,
                    nv=cfg.duet_tree_nv, beta=cfg.duet_tree_beta,
                    depth_cap=K2, temps=temps[:1].expand(W).float(),
                    forward_fn=forward_fn, glue_rows_by_root=glue_rows,
                    rope_base_by_root=rope_base, K_glue=K_glue_used,
                    fanout_policy=cfg.duet_tree_fanout_policy,
                    context_len=ctx_len, sampler_x=cfg.sampler_x,
                    F_x=cfg.async_fan_out, device=self.device)
                # 1a 경계: 단일 sync로 기존 view/wire 경로에 접속
                # (1b에서 view/wire GPU화로 제거 예정 — docs/duet/22)
                pool = _ar.to_pool(len(seeds))
                if os.environ.get("SSD_TREE_ALLOC_CHECK", "0") == "1":
                    st = pool.alloc_stats
                    if st["generated"] != st["allocated"]:
                        print(f"[tree-alloc #27/arena] {st}", flush=True)
            else:
                pool, eval_log, cell_logits = _PT.run_rollout(
                    [t for _p, t in seeds], root_piv,
                    policy=cfg.duet_tree_policy, W=W, F_total=K2,
                    c_tensor=cfg.duet_tree_c_tensor,
                    nv=cfg.duet_tree_nv,
                    beta=cfg.duet_tree_beta, depth_cap=K2,
                    temps=temps[:1].expand(W).float(),
                    forward_fn=forward_fn, glue_rows_by_root=glue_rows,
                    rope_base_by_root=rope_base, K_glue=K_glue_used,
                    fanout_policy=cfg.duet_tree_fanout_policy,
                    context_len=ctx_len,
                    sampler_x=cfg.sampler_x, F_x=cfg.async_fan_out)
        finally:
            _CH.cache.pop("_tree_mask_override", None)
        return pool, eval_log, cell_logits

    def _policy_b_from_raw_proxy(self, raw, out_logits, out_tokens, K_step):
        """Compute Policy B {chosen_pos, chosen_tok} draft-side from the raw
        top-M exit proxy (SSD_DUET_PROXY_ON_DRAFT=1). Mirrors the verifier's
        Policy B math (docs/duet/05) exactly, except the residual candidate
        set is the target's top-M exit tokens per position instead of the
        full vocab — residual <= p_E pointwise, so top-M(p_E) contains the
        true residual top-k up to the p_E mass below rank M.

        Args:
            raw: dict from _unpack_duet_proxy (raw-proxy mode).
            out_logits: [B, K_alloc, V] this step's response logits (= the
                        logits_q the target verifies against).
            out_tokens: [B, K_alloc] this step's response tokens.
            K_step: the step's valid_k (verify width - 1).
        """
        from ssd.utils.async_helpers.duet_policy import policy_b_from_candidates
        return policy_b_from_candidates(
            raw, out_logits[0], out_tokens[0, :K_step], K_step,
            self.config.duet_proxy_top_k, self.config.duet_proxy_wire_N)

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

        See docs/duet/05-policy-b-fix.md Section 3.7 + Fix ④; batched over B
        per docs/duet/13 §4 (M3).

        Args:
            fan_out_tensor: [B, K_rank+1] int64 GPU tensor; each row sums to
                total_budget (= layout MQ_LEN).
            K_rank: int. position_count = K_rank+1 (uniform across the batch).

        Returns:
            self.split_k2_layout (mutated).
        """
        layout = self.split_k2_layout
        K_plus_1 = K_rank + 1
        device = fan_out_tensor.device
        B = fan_out_tensor.shape[0]
        # Update mutable fields:
        layout.position_count = K_plus_1
        layout.fan_out_t = fan_out_tensor                       # [B, K_plus_1]
        layout.fan_out_t_miss = fan_out_tensor                  # same in unified path
        # fan_idx = per-seq repeat_interleave, concatenated → [B*MQ_p2].
        # arange(K_plus_1).repeat(B) tiles positions per seq; interleaving by
        # the row-major flattened counts yields seq 0's fan_idx, then seq 1's,
        # ... At B=1 this is bit-identical to the pre-M3
        # arange(K_plus_1).repeat_interleave(fan_out_tensor).
        layout.fan_idx_hit = torch.arange(
            K_plus_1, device=device, dtype=torch.int64
        ).repeat(B).repeat_interleave(fan_out_tensor.reshape(-1))
        layout.fan_idx_miss = layout.fan_idx_hit
        # M3: fan_idx above already spans ALL seqs — consumers must not re-cat
        # it per seq (see _build_tree_decode_args_for_layout / merge).
        layout.fan_idx_per_seq = True
        # fan_out_list (Python list) — REQUIRED to match position_count by
        # cudagraph_helpers.run_fi_tree_decode_cudagraph (line 329:
        # np.repeat(_tril, _fol)). M3: list of per-seq lists [B][K_plus_1]
        # via ONE .tolist() (the same single allowed sync as pre-M3, now
        # B×(≤9) elements).
        layout.fan_out_list = fan_out_tensor.tolist()
        layout.fan_out_list_miss = layout.fan_out_list
        return layout

    @staticmethod
    def _select_proxy_sourced_tokens_unified(duet_proxy, draft_forked,
                                              K_rank, total_budget,
                                              draft_forked_mask=None):
        """K+1 unified Policy B selector, batched over B (docs/duet/13 §4, M3).
        Static (does not use self) so unit tests can call it directly via
        DraftRunner._select_proxy_sourced_tokens_unified without instance setup.

        See docs/duet/05-policy-b-fix.md Step 3.

        Args:
            duet_proxy: {"chosen_pos": [B, wire_N], "chosen_tok": [B, wire_N]}
                        score-sorted desc PER SEQ. chosen_pos ∈ [0, K_rank] by
                        target-side construction (Section 3.3 invariant; short
                        seqs with vk_i < K_rank never land beyond vk_i because
                        their padded positions get h=0 — docs/duet/13 §3).
            draft_forked: [B, P, max_fo] Phase 1 candidates per ranking position.
                          - uniform Phase 1: max_fo = dfo, all slots real.
                          - non-uniform: max_fo = max(fan_out_list); padded with
                            zeros where fan_out_list[p] < max_fo.
                          P ≥ K_rank + 1.
            K_rank: int  ranking horizon = current step's vk_max.
                    Phase 2 forward depth (K2) is independent.
            total_budget: int  Phase 2 tree size (= sum(fan_out_list) returned)
                          — EXACTLY total_budget per seq by construction.
            draft_forked_mask: optional [P, max_fo] bool, shared across seqs
                               (Phase 1 fan_out_list is uniform per seq). None
                               for uniform Phase 1 (all-real). Required for
                               non-uniform Phase 1 to avoid false in_draft match
                               on zero-padded slots when chosen_tok happens to be 0.

        Returns:
            result_tokens:  [B, total_budget] int64 — per seq pos-grouped
                            (stable sort by pos), score order preserved within
                            each pos.
            fan_out_tensor: [B, K_rank+1] int64, each row sums to total_budget.
        """
        chosen_pos = duet_proxy["chosen_pos"]    # [B, wire_N]
        chosen_tok = duet_proxy["chosen_tok"]    # [B, wire_N]
        B, N = chosen_pos.shape
        assert draft_forked.shape[0] == B, \
            f"draft_forked B={draft_forked.shape[0]} != proxy B={B}"
        # Invariant: chosen_pos ∈ [0, K_rank] by construction.
        if __debug__:
            assert (chosen_pos <= K_rank).all().item(), \
                f"chosen_pos out of range [0, {K_rank}]: max={chosen_pos.max().item()}"

        # in_draft dedup — per-seq membership against that seq's Phase 1
        # candidates. Uniform: all slots real. Non-uniform: mask filters
        # zero-padded slots so chosen_tok=0 doesn't false-match on padding.
        _b_idx = torch.arange(B, device=chosen_pos.device)[:, None]   # [B, 1]
        df_per_cand = draft_forked[_b_idx, chosen_pos]                # [B, N, max_fo]
        eq = (df_per_cand == chosen_tok.unsqueeze(-1))                # [B, N, max_fo]
        if draft_forked_mask is not None:
            msk = draft_forked_mask[chosen_pos, :]                    # [B, N, max_fo]
            eq = eq & msk
        in_draft = eq.any(-1)                                         # [B, N]
        valid = ~in_draft                                             # [B, N]

        # Pick first total_budget valid per seq (score-sorted preserved).
        rank = valid.to(torch.int64).cumsum(1)                        # [B, N]
        take = valid & (rank <= total_budget)                         # [B, N]

        # === Fix ③ (correctness guard) ===
        # Buffer sizing (Section 3.3 / 3.5) is supposed to ensure
        # take.sum() == total_budget PER SEQ. If it doesn't, fan_out rows
        # would silently differ from proxy_forked.shape[1] (= total_budget)
        # and downstream Phase 2 decode reads zero-padded slots as real
        # tokens. This assert was found to be a per-step GPU→CPU sync on the
        # critical path (immediately after proxy_wait). Since the invariant
        # is guaranteed by config-time buffer sizing (see 05-policy-b-fix.md
        # §3.3), gate under __debug__ so it's stripped under `python -O`.
        if __debug__:
            _take_sums = take.sum(dim=1)
            assert bool((_take_sums == total_budget).all().item()), (
                f"Policy B underfill: take.sum(per seq)={_take_sums.tolist()} != "
                f"total_budget={total_budget}. This means buffer sizing was "
                f"insufficient. Check config.duet_proxy_wire_N and "
                f"config.duet_proxy_top_k auto-raise; see docs/duet/05-policy-b-fix.md "
                f"Section 3.5."
            )

        # Boolean indexing on [B, N] yields row-major order; each seq
        # contributes EXACTLY total_budget elements (invariant above), so the
        # view recovers per-seq groups. Same boolean-index op as pre-M3 — no
        # new sync.
        taken_pos = chosen_pos[take].view(B, total_budget)            # [B, total_budget]
        taken_tok = chosen_tok[take].view(B, total_budget)
        # T2.1 (P2-tree): 잔존 seed의 P_iv를 토큰과 동일 인덱싱으로 관통
        # (기존 역매칭 브릿지 제거 — 결정 ③ 라이브 값 사용).
        _piv = duet_proxy.get("chosen_piv") if isinstance(duet_proxy, dict) \
            else None
        taken_piv = _piv[take].view(B, total_budget) if _piv is not None \
            else None

        # fan_out 동적 재구성 — per seq (length = K_rank+1 per row)
        K_plus_1 = K_rank + 1
        fan_out_tensor = torch.zeros(
            B, K_plus_1, dtype=torch.int64, device=chosen_pos.device)
        fan_out_tensor.scatter_add_(
            1, taken_pos, torch.ones_like(taken_pos))

        # result tensor [B, total_budget] — group by pos per seq (stable sort
        # preserves score order within each pos). Identical to the pre-M3
        # taken_tok[order] at B=1 (take.sum()==total_budget guaranteed).
        order = taken_pos.argsort(dim=1, stable=True)                 # [B, total_budget]
        result = taken_tok.gather(1, order)

        # Return fan_out_tensor (GPU) — caller does .tolist() once (cheap
        # for B×(K_plus_1 ~9) elements) and updates layout in-place.
        # See Fix ④ (avoid per-step create_tree_layout).
        if taken_piv is not None:
            return result, fan_out_tensor, taken_piv.gather(1, order)
        return result, fan_out_tensor

    def _build_tree_decode_args_for_layout(self, partial_tree_decode_args, forked_tokens,
                                             layout, cache_hits_list, pos_offset=0):
        """Build tree_decode_args using the given layout. Used for both draft and proxy passes.

        Note: glue position offset uses ``layout.position_count``, not
        ``K + 1``. For non-DUET layouts these are equal; for split-K1/K2
        layouts they diverge (e.g. split_k1_short: forward depth K1 vs
        position count K2+1).
        """
        B = partial_tree_decode_args["num_tokens"].shape[0]
        K = layout.K
        MQ_LEN = layout.MQ_LEN
        glue_offset = layout.position_count

        _b_flat = torch.arange(B, device=self.device, dtype=torch.int64)[:, None].expand(B, MQ_LEN).flatten()
        _fkp1_flat = layout.arange_mq.repeat(B)
        if layout.fan_idx_per_seq:
            # M3 (docs/duet/13 §4): split_k2 fan_idx is already the per-seq
            # concatenated [B*MQ_LEN] (hit == miss in the unified path) — do
            # NOT re-cat per seq. At B=1 identical to cat of the single row.
            _j_idx_flat = layout.fan_idx_hit
        else:
            _j_idx_flat = torch.cat([layout.fan_idx_hit if int(h) else layout.fan_idx_miss for h in cache_hits_list])
        N = _b_flat.shape[0]
        if __debug__:
            assert _j_idx_flat.shape[0] == N, \
                f"fan_idx length {_j_idx_flat.shape[0]} != N={N} (layout {layout.name})"

        _pos_offset = -1 if self.config.use_eagle else 0
        _positions = (partial_tree_decode_args["num_tokens"][_b_flat] - 1 + _pos_offset) + glue_offset + _fkp1_flat
        _rope_positions = (partial_tree_decode_args["num_tokens"][_b_flat] - 1 + _pos_offset) + _j_idx_flat + 1
        _temperatures = partial_tree_decode_args["temperatures"][_b_flat]

        # metadata F: first-position fan_out as an int. M3: split_k2's
        # fan_out_list is nested per-seq ([B][P]) — take seq 0's first entry
        # (same value as the pre-M3 flat list at B=1).
        _f0 = layout.fan_out_list[0]
        if isinstance(_f0, list):
            _f0 = _f0[0]
        return {
            "metadata_ints": (B, K, _f0, N),
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

    def _build_tree_batch_duet(self, partial_tree_decode_args, glue_decode_input_ids):
        """DUET tree decode dispatch — split-only K1/K2 mode.

        See docs/duet/04-split-k1k2-design.md. The hybrid / legacy two-pass
        implementations were removed (2026-07); see git history.
        """
        if SPLIT_K1K2_MODE and self.split_k1_long_layout is not None:
            return self._build_tree_batch_split_k1k2(
                partial_tree_decode_args, glue_decode_input_ids,
            )
        raise RuntimeError(
            "DUET hybrid / legacy two-pass paths were removed (2026-07). "
            "Run with SSD_FORCE_SPLIT_K1K2=1 and duet_phase1_k/duet_phase2_k set. "
            "The removed implementation is preserved in git history "
            "(feat/mesa-proxy-async-overlap @ 19c8f73 and earlier).")

    def _build_tree_batch_split_k1k2(self, partial_tree_decode_args, glue_decode_input_ids):
        """Split-only K1/K2 mode (per docs/duet/04-split-k1k2-design.md).

        Two independent passes, NO continuation:
          - Phase 1 (Draft pass): K1 forwards with split_k1_layout (pos=K1+1).
            Output: draft-sourced rows of depth K1.
          - Phase 2 (Proxy pass): K2 forwards with split_k2_layout (pos=K2+1).
            Output: proxy-sourced rows of depth K2.

        Cache contract: draft rows valid_k=K1, proxy rows valid_k=K2.
        valid_k space = {K1, K2}.
        """
        from ssd.engine.helpers.cudagraph_helpers import duet_record as _mr, duet_close as _mc

        B = partial_tree_decode_args["num_tokens"].shape[0]
        K = self.config.speculate_k
        K1 = self.config.duet_phase1_k
        K2 = self.config.duet_phase2_k
        # Split-only K1/K2 mode: K2 <= K1 contract (asserted at init time).
        # No need to re-assert here, but document the invariant:
        #   K_max = K1, K_min = K2, valid_k space = {K1, K2} ⊆ [1, K1].

        # Post irecv FIRST so target send doesn't block.
        proxy_recv_work, proxy_buf = self._irecv_duet_proxy(B, K)

        # Glue decode (single forward returning logits at the accept-frontier).
        # Glue width follows valid_k+1 (matched row depth + 1):
        #   - valid_k=K1 (or first-step K_max=K1) → glue width = K1+1
        #   - valid_k=K2                          → glue width = K2+1
        _tree_step = getattr(self, "_tree_hit_root", None) is not None
        if _tree_step:
            # T3.4-b3-3 (결정④ ⓐ): 트리-hit step은 TREE_GLUE — 뷰를 트리
            # topology로 실체화한 뒤 노드-fork P1/P2 (ⓑⓒ)로 이어간다.
            glue_logits, gd_for_fork, cache_hits, cache_hits_list, dbt, \
                B_glue = self._tree_glue_decode(
                    partial_tree_decode_args, glue_decode_input_ids)
            return self._tree_step_p1p2(
                partial_tree_decode_args, glue_logits, cache_hits,
                cache_hits_list, dbt, proxy_recv_work, proxy_buf)
        glue_logits, gd_for_fork, cache_hits, cache_hits_list, dbt, B_glue = \
            self._glue_decode(partial_tree_decode_args, glue_decode_input_ids)

        # === Phase 1 layout dispatch by step's valid_k (long/short bucket) ===
        # valid_k=K1 → split_k1_long_layout (pos=K1+1) — glue width matches.
        # valid_k=K2 → split_k1_short_layout (pos=K2+1) — glue width matches.
        # Read pre-sync'd Python scalar from partial_tree_decode_args.
        # M2 (docs/duet/13 §1): the scalar is vk_max over the batch — a
        # mixed K1/K2 batch dispatches the long bucket; only an all-short
        # batch takes the short bucket (cheap max, v1 padding cost).
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
        # Batch 1d: proxy_wait is I/O (NCCL recv) that fires BEFORE the
        # phase2_build span opens — the earlier parent="phase2_build" tag
        # caused the aligned plotter to render proxy_wait to the LEFT of
        # the translucent phase2_build parent bar (misleading). Render as
        # a root span instead.
        _mev_pw = _mr("proxy_wait")
        proxy_recv_work.wait()
        _mc("proxy_wait", _mev_pw)
        duet_proxy = self._unpack_duet_proxy(proxy_buf, B, K)
        if self.config.duet_proxy_on_draft:
            # Raw-proxy mode: Policy B selection runs HERE (draft side,
            # post-proxy window) instead of on the target verify path.
            _mev_plb = _mr("proxy_local_b")
            duet_proxy = self._policy_b_from_raw_proxy(
                duet_proxy,
                partial_tree_decode_args["out_logits"],
                partial_tree_decode_args["returned_tokens"],
                _step_valid_k,
            )
            _mc("proxy_local_b", _mev_plb)
        # ===== TRACE point 4 (early): proxy unpack =====
        if TRACE_SPLIT_K1K2:
            K_max = K1 if K1 >= K2 else K2
            _vk_step = partial_tree_decode_args.get("valid_k_scalar")
            print(f"[TRACE-split-k1k2 #4a proxy_unpack policy=b] K1={K1} K2={K2} K_max={K_max} "
                  f"step_valid_k_scalar={_vk_step} "
                  f"chosen_pos.shape={list(duet_proxy['chosen_pos'].shape)} "
                  f"chosen_tok.shape={list(duet_proxy['chosen_tok'].shape)}",
                  flush=True)
        # ===== END TRACE =====

        # === Phase 2: Proxy pass, K2 forwards (independent) ===
        _mev_p2b = _mr("phase2_build")

        # === Policy B unified path ===
        # docs/duet/05-policy-b-fix.md Step 4.
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
        total_budget = self.config.duet_proxy_total_budget   # = pfo*(K_max+1)
        # Pass padded [B, P, max_fo] + mask [P, max_fo] for Policy B dedup.
        # Uniform: padded == 3D draft_forked_k1, mask=None (all-real).
        # Non-uniform: padded zero-filled beyond fan_out_list[p], mask
        # filters to avoid false-match on chosen_tok=0.
        # M3 (docs/duet/13 §4): the selector is batched — feed the [B, wire_N]
        # wire tensors directly (M1's seq-0 collapse removed). Raw-proxy mode
        # (B==1-guarded off-champion gate, §6) yields 1-D single-seq tensors
        # via _policy_b_from_raw_proxy — lift to [1, wire_N].
        if duet_proxy["chosen_pos"].dim() == 1:
            duet_proxy = {"chosen_pos": duet_proxy["chosen_pos"].unsqueeze(0),
                          "chosen_tok": duet_proxy["chosen_tok"].unsqueeze(0)}
        _sel_out = self._select_proxy_sourced_tokens_unified(
            duet_proxy, draft_forked_p1_padded,
            K_rank=K_rank, total_budget=total_budget,
            draft_forked_mask=draft_forked_p1_mask,
        )
        if len(_sel_out) == 3:                 # tree ON: piv 관통 (T2.1)
            proxy_forked, proxy_fan_out_tensor, proxy_piv = _sel_out
        else:
            proxy_forked, proxy_fan_out_tensor = _sel_out
            proxy_piv = None
        if _E0_TRACE:  # E0: wire(=rank 순) + dedup 후 잔존 P2 seed (P0)
            _e0.record_draft_selector(
                getattr(self, "_e0_step_id", -1),
                duet_proxy["chosen_pos"], duet_proxy["chosen_tok"],
                proxy_forked, proxy_fan_out_tensor)
        _layout_k2 = self._update_phase2_layout_inplace(proxy_fan_out_tensor, K_rank)
        if TRACE_SPLIT_K1K2:
            _fol_dbg = proxy_fan_out_tensor.tolist()      # debug-only sync
            print(f"[TRACE-split-k1k2 #4b proxy_seed policy=b] K2={K2} K_rank={K_rank} "
                  f"total_budget={total_budget} fan_out_list={_fol_dbg}",
                  flush=True)

        proxy_tree_args = self._build_tree_decode_args_for_layout(
            partial_tree_decode_args, proxy_forked, _layout_k2, cache_hits_list)
        _mc("phase2_build", _mev_p2b)
        _temps_p2 = proxy_tree_args["temps"]
        if (self.config.duet_tree_policy != "off" and B == 1
                and bool((_temps_p2 > 0).all())):
            # === P2-tree rollout (T3.4-b3) — 체인 P2 decode 대체 ===
            from ssd.engine.helpers.p2_tree import build_root_views
            _tree_args = dict(proxy_tree_args)
            _tree_args["proxy_piv"] = proxy_piv
            _mB, _mK, _mF, _mN = proxy_tree_args["metadata_ints"]
            _, _srp, _scl, _ssm = self._compute_step_positions_and_slot_maps(
                proxy_tree_args["positions"],
                proxy_tree_args["rope_positions"],
                proxy_tree_args["block_tables"],
                _mB, _mK, _mF, _mN, _layout_k2.MQ_LEN, layout=_layout_k2)
            pool, _elog, _cell_logits = self._p2tree_rollout(
                duet_proxy, proxy_forked, proxy_fan_out_tensor,
                _tree_args, _layout_k2, _ssm, _scl, _temps_p2)
            _R = _layout_k2.MQ_LEN
            _nv = self.config.duet_tree_nv
            self._tree_views = build_root_views(
                pool, _R, _nv, cell_logits=_cell_logits)
            # 체인-호환 populate 입력: backbone(맏이 사슬) [R, K2] 투영
            proxy_tokens, proxy_logits = self._tree_backbone_project(
                pool, _R, K2, _cell_logits)
            proxy_acts = None
        else:
            self._tree_views = None
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
        if getattr(self, "_tree_views", None) is not None:
            self._invalidate_zero_valid_tree_keys()


    def _merge_and_populate_cache(self, draft_args, draft_tokens, draft_logits,
                                    proxy_args, proxy_tokens, proxy_logits,
                                    cache_hits_list, draft_acts=None, proxy_acts=None,
                                    proxy_layout=None, draft_layout=None,
                                    draft_fan_idx_override=None):
        """Merge draft + proxy tree decode results into single cache.
        proxy_layout / draft_layout: the step's runtime layouts (split-K1/K2
        caller always passes both).

        Proxy rows come back with depth K2 (<= K1). Cache row width is
        uniformly K_max; the K_max - K2 tail of proxy rows is zero-padded and
        ignored downstream (verify reads only the first ``valid_k`` positions).
        """
        from ssd.engine.helpers.cudagraph_helpers import duet_record as _mr, duet_close as _mc
        _mev_mc = _mr("merge_cache")
        _proxy_layout = proxy_layout
        _draft_layout = draft_layout
        # Cache row width: split-only K1/K2 mode pads to K_max=max(K1,K2)
        # so wire/cache layout is independent of K1+K2. Hybrid stays at K_long.
        if SPLIT_K1K2_MODE and self.config.duet_phase1_k is not None:
            K1_cfg = self.config.duet_phase1_k
            K2_cfg = self.config.duet_phase2_k
            K_long = K1_cfg if K1_cfg >= K2_cfg else K2_cfg  # = K_max in this mode
        else:
            K_long = self.config.speculate_k
        # Phase 3: draft row valid_k = draft_tokens.shape[1] (K1 without
        # continuation, K_long with continuation). proxy row valid_k =
        # proxy_tokens.shape[1] (K_short = K2).
        draft_row_vk = int(draft_tokens.shape[1])
        proxy_row_vk = int(proxy_tokens.shape[1])

        # Build keys with layout-specific fan_idx.
        # T3.4-b3-4: 트리 step은 fan_idx = 종단 노드 id (0=root-종단,
        # 1+j=뷰 노드 j) — 호출자가 컨텍스트 매핑을 직접 넘긴다.
        if draft_fan_idx_override is not None:
            draft_k = draft_fan_idx_override.to(torch.int64)
        else:
            draft_k = torch.cat([_draft_layout.fan_idx_hit if int(h) else _draft_layout.fan_idx_miss
                                  for h in cache_hits_list])
        draft_keys = torch.stack([
            draft_args["seq_ids_expanded"].to(torch.int64),
            draft_k, draft_args["rec_flat"].to(torch.int64)], dim=1)

        if _proxy_layout.fan_idx_per_seq:
            # M3 (docs/duet/13 §4): split_k2 fan_idx already spans all seqs
            # ([B*MQ_p2], hit == miss) — no per-seq re-cat.
            proxy_k = _proxy_layout.fan_idx_hit
        else:
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
        # Per-row valid_k. Legacy DUET: all rows = K_long. Hybrid: draft rows
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
            from ssd.engine.helpers.cudagraph_helpers import duet_record as _mr_c, duet_close as _mc_c
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

                if self.config.duet_enabled:
                    # DUET: 2-pass tree decode (draft-sourced → proxy-sourced)
                    self._build_tree_batch_duet(partial_tree_decode_args, glue_decode_input_ids)
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
                    from ssd.engine.helpers.cudagraph_helpers import duet_dump
                    duet_dump("draft")
                except Exception as e:
                    if os.environ.get("SSD_PROFILE_DUET", "0") == "1":
                        print(f"[duet_profile draft warning] failed to dump profile: {type(e).__name__}: {e}", flush=True)
                self.exit()
                break

            else:
                raise RuntimeError(f"draft_loop: unknown command {cmd}")
