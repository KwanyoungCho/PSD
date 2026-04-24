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

    # Weight-only quantization (target only)
    target_quant_enabled: bool = False
    # Backends — both torchao WO paths are documented as **bf16 activation**
    # workflows (see torchao inference docs). Combining either with a fp16
    # checkpoint is not supported by the selected backend:
    #   - int4_wo_tile: API-level dtype assert fails ("Expected zeros fp16, got bf16")
    #   - int8_wo     : no assert, but produces inf in MLP output (numerically unreliable)
    # → fp16 checkpoint + these backends requires either a different backend
    #   (e.g. GemliteUIntXWeightOnlyConfig is fp16-native) or opt-in bf16 upcast.
    # "int4_wo_tile" | "int8_wo" | "awq_marlin"
    #   - int4_wo_tile / int8_wo : legacy torchao weight-only paths
    #   - awq_marlin            : new AWQ-style W4A16 via sgl-kernel Marlin
    #                             (plan v2 primary direction — see INT8-WEIGHT-ONLY-PLAN-v2.md)
    target_quant_backend: str = "int4_wo_tile"
    # AWQ-specific options. Only used when target_quant_backend == "awq_marlin".
    target_quant_awq_artifact: str | None = None    # SSD-native artifact prefix
    target_quant_external_awq_path: str | None = None   # external AutoAWQ hf dir
    target_quant_group_size: int = 128
    # Default False: ParallelLMHead is a per-step hot path (gather + cat per call)
    # and MESA also calls lm_head at exit layer for proxy logits. Quantizing it
    # hurts throughput and accept rate (MESA accept -4~8%p observed). Turn on
    # explicitly only when memory is critical or for benchmarking.
    target_quant_lm_head: bool = False
    target_quant_mode: str = "load_time"    # "load_time" | "persistent"
    # Opt-in workaround for fp16 checkpoints: override runtime dtype to bf16
    # so the torchao WO backend (which expects bf16 activation) can be used.
    # Default False → fp16 checkpoint + quant raises ValueError, surfacing the
    # unsupported combination rather than silently switching runtime dtype.
    # Set True only if you accept that "fp16 checkpoint" becomes effectively a
    # bf16 runtime (weights/activations/KV cache/graph buffers all bf16).
    target_quant_force_bf16_runtime: bool = False
    # Path prefix for persistent artifacts. Per-rank files at <prefix>.rank{r}.pt
    # - mode=load_time: if set and artifact exists → load; else quantize+save (dump) then use
    # - mode=persistent: must exist, load-only
    target_quant_artifact_prefix: str | None = None

    # Debugging
    verbose: bool = False
    debug_mode: bool = False
    max_steps: int | None = None

    @property
    def max_blocks(self): 
        return (self.max_model_len + self.kvcache_block_size - 1) // self.kvcache_block_size

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
            # #4 Auto-raise proxy_top_k to eliminate draft fallback.
            # Worst case: fan_out_list skewed → max position fo ≤ pfo*(K+1). Need proxy_top_k ≥ max_fo + dfo + margin.
            K_plus_1 = self.speculate_k + 1
            max_possible_fo = self.mesa_proxy_fan_out * K_plus_1
            required_top_k = max_possible_fo + self.mesa_draft_fan_out + 2
            if self.mesa_proxy_top_k < required_top_k:
                print(f'[Config] mesa_proxy_top_k raised {self.mesa_proxy_top_k} → {required_top_k} '
                      f'(to eliminate draft fallback; max_fo={max_possible_fo} + dfo={self.mesa_draft_fan_out} + margin=2)',
                      flush=True)
                self.mesa_proxy_top_k = required_top_k
            assert self.mesa_proxy_top_k >= 1, "mesa_proxy_top_k must be >= 1"
            print(f'[Config] MESA-SSD enabled: exit_layer={self.mesa_exit_layer}, '
                  f'proxy_top_k={self.mesa_proxy_top_k}, '
                  f'draft_fan_out={self.mesa_draft_fan_out}, proxy_fan_out={self.mesa_proxy_fan_out}',
                  flush=True)

        assert self.max_num_batched_tokens >= self.max_model_len
