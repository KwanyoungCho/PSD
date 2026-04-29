import os
from dataclasses import dataclass
from transformers import AutoConfig
import torch
from ssd.paths import DEFAULT_TARGET, DEFAULT_DRAFT

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

    # MESA-SSD
    mesa_enabled: bool = False
    mesa_exit_layer: int | None = None      # None=auto: 2*L//3
    mesa_proxy_top_k: int = 3              # proxy correction token count
    mesa_draft_fan_out: int | None = None   # draft-sourced branches per position (None=auto: fan_out//2)
    mesa_policy: str = "b"                  # Phase-2 budget policy: "b" = unified K+1 P_iv (default).
                                            # "a" retained as dead branch; see docs/mesa/05-policy-b-fix.md.
    # Phase 2 hybrid (per docs/mesa/01-design.md Part 5).
    # K1 = Phase 1 forward depth, K2 = Phase 2 forward depth.
    # Constraint: K1 + K2 == speculate_k. K2 also = K_short (proxy-sourced row depth).
    # None means "use legacy two-pass MESA path" (no hybrid).
    mesa_phase1_k: int | None = None
    mesa_phase2_k: int | None = None
    # Split-only K1/K2 mode: per-position fan_out list for Phase 1 / Phase 2.
    # Length must be K1+1 / K2+1. None → uniform [draft_fo]*(K1+1) / [proxy_fo]*(K2+1).
    # Allows non-uniform speculation tree (e.g., wider at root, narrower at leaves).
    mesa_split_phase1_fan_out_list: list[int] | None = None
    mesa_split_phase2_fan_out_list: list[int] | None = None

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
    # and MESA also calls lm_head at exit layer for proxy logits. Quantizing it
    # hurts throughput and accept rate (MESA accept -4~8%p observed). Turn on
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
    def mesa_proxy_wire_N(self) -> int:
        """Total (chosen_pos, chosen_tok) entries on Policy B wire = total_budget + buffer.

        total_budget = pfo × (K_max+1)        (Phase 2 tree size; layout MQ_LEN)
        buffer       = (K_max+1) × dfo + 2    (dedup loss safety margin)

        K_max = max(K1, K2) in split mode (= K1 by K2 ≤ K1), else speculate_k.
        See docs/mesa/05-policy-b-fix.md Section 3.5.
        """
        import os as _os_cfg
        _split_mode = (
            self.mesa_phase1_k is not None
            and _os_cfg.environ.get("SSD_FORCE_SPLIT_K1K2", "0") == "1"
        )
        if _split_mode:
            K_max = max(self.mesa_phase1_k, self.mesa_phase2_k)
        else:
            K_max = self.speculate_k
        K_plus_1 = K_max + 1
        total_budget = self.mesa_proxy_fan_out * K_plus_1
        buffer = K_plus_1 * self.mesa_draft_fan_out + 2
        return total_budget + buffer

    @property
    def mesa_proxy_total_budget(self) -> int:
        """Phase 2 tree size = sum(fan_out_list) at runtime = layout MQ_LEN.

        See docs/mesa/05-policy-b-fix.md Section 3.5 / 3.7.
        """
        import os as _os_cfg
        _split_mode = (
            self.mesa_phase1_k is not None
            and _os_cfg.environ.get("SSD_FORCE_SPLIT_K1K2", "0") == "1"
        )
        if _split_mode:
            K_max = max(self.mesa_phase1_k, self.mesa_phase2_k)
        else:
            K_max = self.speculate_k
        return self.mesa_proxy_fan_out * (K_max + 1)

    def __post_init__(self):
        model = self.model 
        assert os.path.isdir(model)

        assert 1 <= self.num_gpus <= 8 # this codebase only works on one node 
        self.hf_config = AutoConfig.from_pretrained(model)
        self.max_model_len = min(
            self.max_model_len, self.hf_config.max_position_embeddings) 
        if self.speculate: 
            draft = self.draft
            self.draft_hf_config = AutoConfig.from_pretrained(draft)
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
        
        if self.mesa_enabled:
            assert self.draft_async, "MESA-SSD requires draft_async=True"
            assert self.speculate, "MESA-SSD requires speculate=True"
            assert self.hf_config.model_type == "llama", "MESA-SSD only supports Llama models"
            assert not self.use_eagle, "MESA-SSD + EAGLE: not yet implemented (eagle_acts split collection needed)"
            assert not self.enforce_eager, "MESA-SSD requires CudaGraph mode (enforce_eager must be False)"
            assert self.jit_speculate, "MESA-SSD requires jit_speculate=True (miss rows need valid logits_q)"
            # #3 B=1 only: Policy A uses accept_probs[0] as single h_i for whole batch.
            assert self.max_num_seqs == 1, \
                "MESA-SSD Rev1 only supports B=1 (max_num_seqs=1); " \
                "Policy A uses accept_probs[0] as a single h_i distribution for the whole batch"
            if self.mesa_exit_layer is None:
                L = self.hf_config.num_hidden_layers
                self.mesa_exit_layer = (2 * L) // 3
            assert 0 < self.mesa_exit_layer < self.hf_config.num_hidden_layers, \
                f"mesa_exit_layer must be in (0, {self.hf_config.num_hidden_layers}), got {self.mesa_exit_layer}"
            if self.mesa_draft_fan_out is None:
                self.mesa_draft_fan_out = max(1, self.async_fan_out // 2)
            assert 0 < self.mesa_draft_fan_out < self.async_fan_out, \
                f"mesa_draft_fan_out must be in (0, {self.async_fan_out}), got {self.mesa_draft_fan_out}"
            self.mesa_proxy_fan_out = self.async_fan_out - self.mesa_draft_fan_out
            # Auto-raise proxy_top_k. Two constraints (per docs/mesa/05-policy-b-fix.md):
            #   per-pos:  top_k ≥ max_fo + dfo + 2     (한 position에 budget 몰빵 시)
            #   total:    top_k ≥ ceil(wire_N / (K_min+1))   (short-hit candidate count >= wire_N)
            # split mode K_max = max(K1, K2) = K1 (K2 ≤ K1 invariant).
            # wire_N = total_budget + buffer = pfo*(K_max+1) + (K_max+1)*dfo + 2.
            import os as _os_cfg
            _split_mode = (
                self.mesa_phase1_k is not None
                and _os_cfg.environ.get("SSD_FORCE_SPLIT_K1K2", "0") == "1"
            )
            if _split_mode:
                K_max = max(self.mesa_phase1_k, self.mesa_phase2_k)
                K_min = min(self.mesa_phase1_k, self.mesa_phase2_k)
            else:
                K_max = self.speculate_k
                K_min = self.speculate_k
            K_plus_1 = K_max + 1
            pfo = self.mesa_proxy_fan_out
            dfo = self.mesa_draft_fan_out
            total_budget = pfo * K_plus_1
            buffer = K_plus_1 * dfo + 2
            wire_N = total_budget + buffer
            per_pos_min = total_budget + dfo + 2                 # = pfo*(K_max+1) + dfo + 2
            total_min = -(-wire_N // (K_min + 1))                # ceil(wire_N / (K_min+1))
            required_top_k = max(per_pos_min, total_min)
            if self.mesa_proxy_top_k < required_top_k:
                print(f'[Config] mesa_proxy_top_k raised {self.mesa_proxy_top_k} → {required_top_k} '
                      f'(K_max={K_max} K_min={K_min} per_pos={per_pos_min} total={total_min} '
                      f'wire_N={wire_N})', flush=True)
                self.mesa_proxy_top_k = required_top_k
            assert self.mesa_proxy_top_k >= 1, "mesa_proxy_top_k must be >= 1"
            assert self.mesa_policy in ("a", "b"), \
                f"mesa_policy must be 'a' or 'b', got {self.mesa_policy!r}"
            # Policy "b" (unified K+1) is only implemented in split-K1/K2 mode
            # (docs/mesa/05-policy-b-fix.md). For legacy / hybrid paths the
            # _build_tree_batch_mesa() codepath still reads mesa_proxy["fan_out_list"]
            # which Policy B's wire schema doesn't provide. Force "a" with warning.
            if self.mesa_policy == "b" and not _split_mode:
                print(f"[Config] mesa_policy='b' is only implemented in split-K1/K2 "
                      f"mode (SSD_FORCE_SPLIT_K1K2=1); forcing 'a' for "
                      f"legacy/hybrid path.", flush=True)
                self.mesa_policy = "a"
            # Policy "b" + non-uniform Phase 1 fan-out list is not yet implemented:
            # _select_draft_sourced_tokens_perpos returns flat [B, MQ_LEN] but the
            # unified selector indexes draft_forked as 3D [B, P, dfo].
            if self.mesa_policy == "b" and self.mesa_split_phase1_fan_out_list is not None:
                # Check uniformity. If non-uniform, raise.
                _fol = self.mesa_split_phase1_fan_out_list
                if not all(f == _fol[0] for f in _fol):
                    raise NotImplementedError(
                        "Policy 'b' (unified) + non-uniform mesa_split_phase1_fan_out_list "
                        "is not implemented. Selector assumes 3D draft_forked [B, P, dfo] "
                        "but per-position selector returns flat [B, MQ_LEN]. "
                        "Use uniform list (or omit) with policy='b', or use policy='a' "
                        "for non-uniform Phase 1."
                    )
            # Phase 2 hybrid (K1, K2) validation. Both None → legacy two-pass path.
            # Both set → hybrid path with K1 + K2 == speculate_k invariant.
            if (self.mesa_phase1_k is None) != (self.mesa_phase2_k is None):
                raise ValueError(
                    "mesa_phase1_k and mesa_phase2_k must both be set or both None; "
                    f"got K1={self.mesa_phase1_k}, K2={self.mesa_phase2_k}"
                )
            if self.mesa_phase1_k is not None:
                assert self.mesa_phase1_k > 0, f"mesa_phase1_k must be > 0, got {self.mesa_phase1_k}"
                assert self.mesa_phase2_k > 0, f"mesa_phase2_k must be > 0, got {self.mesa_phase2_k}"
                assert self.mesa_phase1_k + self.mesa_phase2_k == self.speculate_k, (
                    f"mesa_phase1_k + mesa_phase2_k must equal speculate_k; "
                    f"got K1={self.mesa_phase1_k} + K2={self.mesa_phase2_k} != "
                    f"speculate_k={self.speculate_k}"
                )
            print(f'[Config] MESA-SSD enabled: exit_layer={self.mesa_exit_layer}, '
                  f'proxy_top_k={self.mesa_proxy_top_k}, '
                  f'draft_fan_out={self.mesa_draft_fan_out}, proxy_fan_out={self.mesa_proxy_fan_out}, '
                  f'K1={self.mesa_phase1_k}, K2={self.mesa_phase2_k}',
                  flush=True)

        assert self.max_num_batched_tokens >= self.max_model_len
