import os
from dataclasses import dataclass
from transformers import AutoConfig
import torch
from ssd.paths import DEFAULT_TARGET, DEFAULT_DRAFT


def _ensure_head_dim(hf_config: AutoConfig) -> None:
    """Inject ``head_dim`` on configs that omit it (e.g. Qwen2 / Qwama).

    Several downstream paths (model_runner.allocate_kv_cache, FlashInfer
    plan in cudagraph_helpers, capture_fi_tree_decode_cudagraph) read
    ``hf_config.head_dim`` as a plain attribute. Llama and Qwen3 configs
    expose it; Qwen2 does not (the value is derived from hidden_size /
    num_attention_heads). Inject once at config-load time so consumers
    do not need a getattr fallback.
    """
    if not hasattr(hf_config, 'head_dim') or getattr(hf_config, 'head_dim', None) is None:
        hf_config.head_dim = hf_config.hidden_size // hf_config.num_attention_heads


@dataclass
class Config:
    model: str = DEFAULT_TARGET
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 1 
    max_model_len: int = 4096 
    gpu_memory_utilization: float = 0.7
    num_gpus: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # spec config args
    draft_hf_config: AutoConfig | None = None
    speculate: bool = False 
    draft: str = DEFAULT_DRAFT
    speculate_k: int = 1
    draft_async: bool = False
    
    # async spec only
    async_fan_out: int = 3
    fan_out_list: list[int] | None = None
    fan_out_list_miss: list[int] | None = None
    sampler_x: float | None = None 
    jit_speculate: bool = False 

    # eagle3
    use_eagle: bool = False 
    eagle_layers: list[int] | None = None   
    d_model_target: int | None = None
    tokenizer_path: str | None = None

    # DUET-SSD
    duet_enabled: bool = False
    duet_exit_layer: int | None = None      # None=auto: 2*L//3
    duet_proxy_top_k: int = 3              # proxy correction token count
    duet_draft_fan_out: int | None = None   # draft-sourced branches per position (None=auto: fan_out//2)
    duet_policy: str = "b"                  # Phase-2 budget policy: "b" = unified K+1 P_iv (default).
                                            # "a" retained as dead branch; see docs/duet/05-policy-b-fix.md.
    # Phase 2 hybrid (per docs/duet/01-design.md Part 5).
    # K1 = Phase 1 forward depth, K2 = Phase 2 forward depth.
    # Constraint: K1 + K2 == speculate_k. K2 also = K_short (proxy-sourced row depth).
    # None means "use legacy two-pass DUET path" (no hybrid).
    duet_phase1_k: int | None = None
    duet_phase2_k: int | None = None
    # Split-only K1/K2 mode: per-position fan_out list for Phase 1 / Phase 2.
    # Length must be K1+1 / K2+1. None → uniform [draft_fo]*(K1+1) / [proxy_fo]*(K2+1).
    # Allows non-uniform speculation tree (e.g., wider at root, narrower at leaves).
    duet_split_phase1_fan_out_list: list[int] | None = None
    duet_split_phase2_fan_out_list: list[int] | None = None

    # AWQ W4A16 quantization (target + draft, role-aware).
    # Public path is `awq_marlin`; legacy torchao backends remain as an
    # internal fallback only. See `INT8-WEIGHT-ONLY-PLAN-v2.md`.
    target_quant_enabled: bool = False
    # Backends — both torchao WO paths are documented as **bf16 activation**
    # workflows (see torchao inference docs). Combining either with a fp16
    # checkpoint is not supported by the selected backend:
    #   - int4_wo_tile: API-level dtype assert fails ("Expected zeros fp16, got bf16")
    #   - int8_wo     : no assert, but produces inf in MLP output (numerically unreliable)
    # → fp16 checkpoint + these backends requires either a different backend
    #   (e.g. GemliteUIntXWeightOnlyConfig is fp16-native) or opt-in bf16 upcast.
    # "awq_marlin" (default, supported public path) | "int4_wo_tile" | "int8_wo"
    # int4_wo_tile / int8_wo are LEGACY torchao paths kept as internal
    # fallback only. Setting them is allowed but no longer driven by any CLI;
    # the runner logs `[quant][LEGACY]` if they ever activate. See
    # `INT8-WEIGHT-ONLY-PLAN-v2.md` and `INT8-v2-IMPL-ISSUE.md`.
    target_quant_backend: str = "awq_marlin"
    # AWQ-specific options. Only used when target_quant_backend == "awq_marlin".
    target_quant_awq_artifact: str | None = None    # SSD-native artifact prefix
    target_quant_external_awq_path: str | None = None   # external AutoAWQ hf dir
    target_quant_group_size: int = 128

    # Draft-side AWQ (role-aware AWQ — independent from target).
    # Llama-family non-EAGLE only, tp_size=1 draft.
    draft_quant_enabled: bool = False
    draft_quant_backend: str = "awq_marlin"              # only awq_marlin supported for draft
    draft_quant_awq_artifact: str | None = None          # SSD-native artifact prefix
    draft_quant_external_awq_path: str | None = None     # external AutoAWQ hf dir
    draft_quant_group_size: int = 128
    # NOTE: draft lm_head / embeddings quantization is NOT implemented.
    # The fields used to exist as opt-ins but were removed because they
    # had no effect — the runner kept lm_head / embeddings dense regardless.
    # Default False: ParallelLMHead is a per-step hot path (gather + cat per call)
    # and DUET also calls lm_head at exit layer for proxy logits. Quantizing it
    # hurts throughput and accept rate (DUET accept -4~8%p observed). Turn on
    # explicitly only when memory is critical or for benchmarking.
    target_quant_lm_head: bool = False
    # [LEGACY torchao fields, kept only because the deprecated runner branch
    # still references them. Do not introduce new uses; will be deleted in
    # the next cleanup PR.]
    target_quant_mode: str = "load_time"
    target_quant_force_bf16_runtime: bool = False
    target_quant_artifact_prefix: str | None = None

    # Debugging
    verbose: bool = False
    debug_mode: bool = False
    max_steps: int | None = None

    @property
    def max_blocks(self):
        return (self.max_model_len + self.kvcache_block_size - 1) // self.kvcache_block_size

    @property
    def duet_proxy_wire_N(self) -> int:
        """Total (chosen_pos, chosen_tok) entries on Policy B wire = total_budget + buffer.

        total_budget = pfo × (K_max+1)        (Phase 2 tree size; layout MQ_LEN)
        buffer       = max(p1_sum_full, p1_sum_short) + 2  (max dedup loss)

        Where p1_sum_full = sum(phase1_fan_out_list) and
              p1_sum_short = sum(phase1_fan_out_list[:K2+1]).

        For uniform fallback (no list): p1_sum_full = dfo*(K1+1),
        p1_sum_short = dfo*(K2+1) — equivalent to old `(K_max+1)*dfo + 2`
        when K1 >= K2 and K_max = K1.

        K_max = max(K1, K2) in split mode (= K1 by K2 ≤ K1), else speculate_k.
        See docs/duet/05-policy-b-fix.md Section 3.5.
        """
        import os as _os_cfg
        _split_mode = (
            self.duet_phase1_k is not None
            and _os_cfg.environ.get("SSD_FORCE_SPLIT_K1K2", "0") == "1"
        )
        if _split_mode:
            K_max = max(self.duet_phase1_k, self.duet_phase2_k)
        else:
            K_max = self.speculate_k
        K_plus_1 = K_max + 1
        total_budget = self.duet_proxy_fan_out * K_plus_1
        # List-aware Phase 1 dedup loss bound
        _p1_list = self.duet_split_phase1_fan_out_list
        if _split_mode and _p1_list is not None:
            _K2p1 = self.duet_phase2_k + 1
            p1_sum_full = sum(_p1_list)
            p1_sum_short = sum(_p1_list[:_K2p1])
        else:
            _K1p1 = (self.duet_phase1_k + 1) if self.duet_phase1_k is not None else self.speculate_k + 1
            _K2p1 = (self.duet_phase2_k + 1) if self.duet_phase2_k is not None else self.speculate_k + 1
            p1_sum_full = self.duet_draft_fan_out * _K1p1
            p1_sum_short = self.duet_draft_fan_out * _K2p1
        buffer = max(p1_sum_full, p1_sum_short) + 2
        return total_budget + buffer

    @property
    def duet_proxy_total_budget(self) -> int:
        """Phase 2 tree size = sum(fan_out_list) at runtime = layout MQ_LEN.

        See docs/duet/05-policy-b-fix.md Section 3.5 / 3.7.
        """
        import os as _os_cfg
        _split_mode = (
            self.duet_phase1_k is not None
            and _os_cfg.environ.get("SSD_FORCE_SPLIT_K1K2", "0") == "1"
        )
        if _split_mode:
            K_max = max(self.duet_phase1_k, self.duet_phase2_k)
        else:
            K_max = self.speculate_k
        return self.duet_proxy_fan_out * (K_max + 1)

    def __post_init__(self):
        model = self.model 
        assert os.path.isdir(model)

        assert 1 <= self.num_gpus <= 8 # this codebase only works on one node 
        self.hf_config = AutoConfig.from_pretrained(model)
        _ensure_head_dim(self.hf_config)
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings)
        if self.speculate:
            draft = self.draft
            self.draft_hf_config = AutoConfig.from_pretrained(draft)
            _ensure_head_dim(self.draft_hf_config)
            self.max_model_len = min(
                self.max_model_len, self.draft_hf_config.max_position_embeddings)
            if self.draft_async:
                if self.fan_out_list is None: 
                    self.fan_out_list = [self.async_fan_out] * (self.speculate_k + 1)
                    self.MQ_LEN = sum(self.fan_out_list)
                if self.fan_out_list_miss is None:
                    self.fan_out_list_miss = self.fan_out_list 
                assert sum(self.fan_out_list_miss) == sum(self.fan_out_list), "ERROR in Config: fan_out_list_miss must be the same as fan_out_list"
                
        if self.use_eagle:
            if self.eagle_layers is None:
                L = self.hf_config.num_hidden_layers
                # self.eagle_layers = [3, L//2, L-3]
                self.eagle_layers = [2, L//2, L-3] # [2, 16, 29] outputs, ie. [3, L//2+1, L-2] inputs
                print(f'[Config] just set eagle_layers={self.eagle_layers}', flush=True)
            # Eagle draft must use target's rope_theta (draft config may default to wrong value)
            if self.speculate and self.draft_hf_config is not None:
                target_rope_theta = getattr(self.hf_config, 'rope_theta', 500000.0)
                draft_rope_theta = getattr(self.draft_hf_config, 'rope_theta', 10000.0)
                if target_rope_theta != draft_rope_theta:
                    print(f'[Config] Overriding eagle draft rope_theta: {draft_rope_theta} -> {target_rope_theta}', flush=True)
                    self.draft_hf_config.rope_theta = target_rope_theta
                # Also override max_position_embeddings for correct RoPE cache size
                # NOTE: Do NOT change max_model_len here - it was already correctly capped.
                # Only change draft_hf_config.max_position_embeddings for RoPE.
                target_max_pos = getattr(self.hf_config, 'max_position_embeddings', 8192)
                draft_max_pos = getattr(self.draft_hf_config, 'max_position_embeddings', 2048)
                if target_max_pos != draft_max_pos:
                    print(f'[Config] Overriding eagle draft max_position_embeddings: {draft_max_pos} -> {target_max_pos}', flush=True)
                    self.draft_hf_config.max_position_embeddings = target_max_pos
        
        if self.duet_enabled:
            assert self.draft_async, "DUET-SSD requires draft_async=True"
            assert self.speculate, "DUET-SSD requires speculate=True"
            assert self.hf_config.model_type == "llama", "DUET-SSD only supports Llama models"
            assert not self.use_eagle, "DUET-SSD + EAGLE: not yet implemented (eagle_acts split collection needed)"
            assert not self.enforce_eager, "DUET-SSD requires CudaGraph mode (enforce_eager must be False)"
            assert self.jit_speculate, "DUET-SSD requires jit_speculate=True (miss rows need valid logits_q)"
            # #3 B=1 only: Policy A uses accept_probs[0] as single h_i for whole batch.
            assert self.max_num_seqs == 1, \
                "DUET-SSD Rev1 only supports B=1 (max_num_seqs=1); " \
                "Policy A uses accept_probs[0] as a single h_i distribution for the whole batch"
            if self.duet_exit_layer is None:
                L = self.hf_config.num_hidden_layers
                self.duet_exit_layer = (2 * L) // 3
            assert 0 < self.duet_exit_layer < self.hf_config.num_hidden_layers, \
                f"duet_exit_layer must be in (0, {self.hf_config.num_hidden_layers}), got {self.duet_exit_layer}"
            if self.duet_draft_fan_out is None:
                self.duet_draft_fan_out = max(1, self.async_fan_out // 2)
            assert 0 < self.duet_draft_fan_out < self.async_fan_out, \
                f"duet_draft_fan_out must be in (0, {self.async_fan_out}), got {self.duet_draft_fan_out}"
            self.duet_proxy_fan_out = self.async_fan_out - self.duet_draft_fan_out

            # Phase 1/2 K validation moved up (auto-raise needs validated K1/K2).
            if (self.duet_phase1_k is None) != (self.duet_phase2_k is None):
                raise ValueError(
                    "duet_phase1_k and duet_phase2_k must both be set or both None; "
                    f"got K1={self.duet_phase1_k}, K2={self.duet_phase2_k}"
                )
            if self.duet_phase1_k is not None:
                assert self.duet_phase1_k > 0, f"duet_phase1_k must be > 0, got {self.duet_phase1_k}"
                assert self.duet_phase2_k > 0, f"duet_phase2_k must be > 0, got {self.duet_phase2_k}"
                assert self.duet_phase1_k + self.duet_phase2_k == self.speculate_k, (
                    f"duet_phase1_k + duet_phase2_k must equal speculate_k; "
                    f"got K1={self.duet_phase1_k} + K2={self.duet_phase2_k} != "
                    f"speculate_k={self.speculate_k}"
                )

            import os as _os_cfg
            _split_mode = (
                self.duet_phase1_k is not None
                and _os_cfg.environ.get("SSD_FORCE_SPLIT_K1K2", "0") == "1"
            )
            # Phase 1 fan_out_list validation (split mode only). User-provided
            # list overrides uniform [draft_fo]*(K1+1). Used by buffer/top_k
            # sizing below — list-aware in ALL cases when list is provided
            # (even uniform list with values != duet_draft_fan_out).
            _p1_list = self.duet_split_phase1_fan_out_list
            if _split_mode and _p1_list is not None:
                _K1p1 = self.duet_phase1_k + 1
                _K2p1 = self.duet_phase2_k + 1
                if len(_p1_list) != _K1p1:
                    raise ValueError(
                        f"duet_split_phase1_fan_out_list len={len(_p1_list)} "
                        f"must equal K1+1={_K1p1}; got {_p1_list}"
                    )
                if any(f < 0 for f in _p1_list):
                    raise ValueError(
                        f"duet_split_phase1_fan_out_list must have all entries >= 0; "
                        f"got {_p1_list}"
                    )
                if sum(_p1_list) <= 0:
                    raise ValueError(
                        f"duet_split_phase1_fan_out_list sum must be > 0 "
                        f"(layout would have MQ_LEN=0); got {_p1_list}"
                    )
                # short-hit (K2+1) prefix sum > 0 — split_k1_short layout
                # would have MQ_LEN=0 otherwise, breaking K2-hit dispatch.
                if sum(_p1_list[:_K2p1]) <= 0:
                    raise ValueError(
                        f"duet_split_phase1_fan_out_list[:K2+1] sum must be > 0 "
                        f"(short-hit layout MQ_LEN=0). Got prefix={_p1_list[:_K2p1]} "
                        f"from {_p1_list}"
                    )
            # Phase 2 non-uniform fan_out is unsupported (independent of Phase 1
            # list — reject early to avoid delayed DraftRunner-init failure when
            # only Phase 2 list is set).
            if _split_mode and self.duet_split_phase2_fan_out_list is not None:
                raise NotImplementedError(
                    "split-K1/K2 Phase 2 non-uniform fan_out is not supported "
                    "(Phase 2 selection is uniform; would need policy-based "
                    "dynamic fan_out — separate design)."
                )
            # Auto-raise proxy_top_k. Two constraints (per docs/duet/05-policy-b-fix.md):
            #   per-pos:  top_k ≥ total_budget + max(p1_fanout) + 2
            #   total:    top_k ≥ ceil(wire_N / (K_min+1))
            # buffer: max possible dedup loss across positions.
            #   uniform: (K_max+1) * dfo
            #   non-uniform: max(sum(p1_list), sum(p1_list[:K2+1]))
            #     (long-hit + miss use full sum; short-hit uses prefix sum)
            # split mode K_max = max(K1, K2) = K1 (K2 ≤ K1 invariant).
            if _split_mode:
                K_max = max(self.duet_phase1_k, self.duet_phase2_k)
                K_min = min(self.duet_phase1_k, self.duet_phase2_k)
            else:
                K_max = self.speculate_k
                K_min = self.speculate_k
            K_plus_1 = K_max + 1
            pfo = self.duet_proxy_fan_out
            dfo = self.duet_draft_fan_out
            total_budget = pfo * K_plus_1

            # List-aware Phase 1 fan-out stats — used in ALL cases when user
            # provides list, otherwise fall back to uniform [dfo]*(K1+1).
            if _split_mode and _p1_list is not None:
                _p1_eff = list(_p1_list)
                _K2p1 = self.duet_phase2_k + 1
                p1_sum_full = sum(_p1_eff)
                p1_sum_short = sum(_p1_eff[:_K2p1])
                p1_max = max(_p1_eff) if _p1_eff else dfo
            else:
                # uniform fallback: list = [dfo] * (K1+1) when phase1_k set,
                # else degenerate ([dfo] * (speculate_k+1) for legacy compat).
                _K1p1 = (self.duet_phase1_k + 1) if self.duet_phase1_k is not None else self.speculate_k + 1
                _K2p1 = (self.duet_phase2_k + 1) if self.duet_phase2_k is not None else self.speculate_k + 1
                p1_sum_full = dfo * _K1p1
                p1_sum_short = dfo * _K2p1
                p1_max = dfo

            buffer = max(p1_sum_full, p1_sum_short) + 2
            wire_N = total_budget + buffer
            per_pos_min = total_budget + p1_max + 2
            total_min = -(-wire_N // (K_min + 1))                # ceil(wire_N / (K_min+1))
            required_top_k = max(per_pos_min, total_min)
            if self.duet_proxy_top_k < required_top_k:
                print(f'[Config] duet_proxy_top_k raised {self.duet_proxy_top_k} → {required_top_k} '
                      f'(K_max={K_max} K_min={K_min} p1_sum_full={p1_sum_full} '
                      f'p1_sum_short={p1_sum_short} p1_max={p1_max} '
                      f'per_pos={per_pos_min} total={total_min} wire_N={wire_N})',
                      flush=True)
                self.duet_proxy_top_k = required_top_k
            assert self.duet_proxy_top_k >= 1, "duet_proxy_top_k must be >= 1"
            assert self.duet_policy in ("a", "b"), \
                f"duet_policy must be 'a' or 'b', got {self.duet_policy!r}"
            # Policy "b" (unified K+1) is only implemented in split-K1/K2 mode
            # (docs/duet/05-policy-b-fix.md). For legacy / hybrid paths the
            # _build_tree_batch_duet() codepath still reads duet_proxy["fan_out_list"]
            # which Policy B's wire schema doesn't provide. Force "a" with warning.
            if self.duet_policy == "b" and not _split_mode:
                print(f"[Config] duet_policy='b' is only implemented in split-K1/K2 "
                      f"mode (SSD_FORCE_SPLIT_K1K2=1); forcing 'a' for "
                      f"legacy/hybrid path.", flush=True)
                self.duet_policy = "a"
            print(f'[Config] DUET-SSD enabled: exit_layer={self.duet_exit_layer}, '
                  f'proxy_top_k={self.duet_proxy_top_k}, '
                  f'draft_fan_out={self.duet_draft_fan_out}, proxy_fan_out={self.duet_proxy_fan_out}, '
                  f'K1={self.duet_phase1_k}, K2={self.duet_phase2_k}',
                  flush=True)

        assert self.max_num_batched_tokens >= self.max_model_len
