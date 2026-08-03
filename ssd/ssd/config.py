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
    duet_policy: str = "b"                  # Phase-2 budget policy: "b" = unified K+1 P_iv (only option;
                                            # Policy "a" was removed 2026-07 — see git history).
    # JIT-short (docs/duet/12): on miss, JIT builds a K2-deep chain instead
    # of K_max — champion standard. Promoted from the SSD_DUET_JIT_SHORT env
    # (docs/duet/16 Tier-2, default ON); an explicitly-set env still wins
    # for old-script compat and the resolved value is re-exported for the
    # draft process import-time read.
    duet_jit_short: bool = True
    # Direct Phase-2 seed budget (docs/duet/16 Tier-3). None → derived
    # pfo*(K_max+1) (100% 재현). When set it is THE single source: layout
    # MQ_LEN / scheduler reservation / wire sizing read
    # duet_proxy_total_budget, and the verifier's per-step K-position
    # allocation scales proportionally (duet_p2_budget_at).
    duet_p2_budget: int | None = None
    # Split-K1/K2 mode (per docs/duet/04-split-k1k2-design.md).
    # K1 = Phase 1 forward depth, K2 = Phase 2 forward depth.
    # Constraint: K1 + K2 == speculate_k, K2 <= K1.
    # REQUIRED when duet_enabled (the hybrid / legacy two-pass paths that
    # allowed None were removed 2026-07; __post_init__ hard-errors on None).
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
        total_budget = self.duet_proxy_total_budget   # Tier-3 single source
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
    def duet_proxy_on_draft(self) -> bool:
        """SSD_DUET_PROXY_ON_DRAFT=1 — target sends raw top-M exit proxy
        (ids + logits + lse + draft-token logit) and the DRAFT computes
        Policy B locally in its proxy_wait window. Removes the residual
        softmax/topk + pack from the target verify critical path
        (docs/duet/09 WS3)."""
        import os as _os_cfg
        return _os_cfg.environ.get("SSD_DUET_PROXY_ON_DRAFT", "0") == "1"

    @property
    def duet_proxy_topm(self) -> int:
        """Top-M exit candidates per position on the raw-proxy wire.
        Must satisfy M >= duet_proxy_top_k (residual candidate coverage;
        top_k is auto-raised to ~total_budget + p1_max + 2, e.g. 12-14)."""
        import os as _os_cfg
        return int(_os_cfg.environ.get("SSD_DUET_PROXY_TOPM", "24"))

    @property
    def duet_exit_topm_gather(self) -> bool:
        """SSD_DUET_EXIT_TOPM_GATHER=1 — each TP rank reduces its exit
        lm_head vocab shard to top-M candidates (+ logsumexp partial +
        draft-token logit) before the gather, and rank 0 runs Policy B on
        the merged candidate set. Replaces the full-[flat, V] exit logits
        gather (~640KB) with ~16KB and drops the full-vocab proxy
        softmax/topk from the rank-0 verify path (docs/duet/09 WS3).
        Wire to the draft is unchanged ({chosen_pos, chosen_tok})."""
        import os as _os_cfg
        return _os_cfg.environ.get("SSD_DUET_EXIT_TOPM_GATHER", "0") == "1"

    @property
    def duet_exit_replica(self) -> bool:
        """SSD_DUET_EXIT_REPLICA=1 — target rank 0 keeps a full-vocab
        lm_head replica (~V×D fp16, 512MB at 70B) so the mid-verify exit
        proxy needs NO TP collective at all: ranks 1+ go straight from
        graph_pre to graph_post (the exit rendezvous point disappears),
        and rank 0 runs norm + replica lm_head + Policy B + send on a
        side stream overlapped with graph_post. Motivated by the
        exit-topm-gather null result: the exit cost is rendezvous-
        dominated, not volume-dominated (docs/duet/09 WS3c)."""
        import os as _os_cfg
        return _os_cfg.environ.get("SSD_DUET_EXIT_REPLICA", "0") == "1"

    @property
    def duet_raw_proxy_wire_len(self) -> int:
        """Fused int64 payload length for the raw-proxy wire (worst-case
        K_max sizing; short steps pad the tail):
        [ids (K_max+1)*M | logits-f64 (K_max+1)*M | lse-f64 (K_max+1) |
         y_logit-f64 (K_max)]."""
        K_max = max(self.duet_phase1_k, self.duet_phase2_k)
        Kp1 = K_max + 1
        M = self.duet_proxy_topm
        return 2 * Kp1 * M + Kp1 + K_max

    @property
    def duet_proxy_total_budget(self) -> int:
        """Phase 2 tree size = sum(fan_out_list) at runtime = layout MQ_LEN.

        See docs/duet/05-policy-b-fix.md Section 3.5 / 3.7.
        """
        if self.duet_p2_budget is not None:
            return self.duet_p2_budget
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

    def duet_p2_budget_at(self, K: int) -> int:
        """Per-step proxy budget over K+1 positions (verifier-side h/fan-out
        allocation; K = the step's vk_max, varies per step — NOT the same
        quantity as duet_proxy_total_budget). Default path reproduces the
        historical pfo*(K+1) exactly; a direct duet_p2_budget scales
        proportionally by position count (== exact on full-K steps)."""
        if self.duet_p2_budget is None:
            return self.duet_proxy_fan_out * (K + 1)
        K_max = max(self.duet_phase1_k, self.duet_phase2_k)
        return max(1, round(self.duet_p2_budget * (K + 1) / (K_max + 1)))

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

            # docs/duet/08 §4: SSD_PROXY_STREAM requires SSD_ASYNC_PROXY_SEND
            # (Policy B compute on proxy_stream is meaningless without
            # non-blocking send — the blocking send would re-serialize).
            # Fail-fast at config time; do NOT silently fallback.
            if os.environ.get("SSD_PROXY_STREAM", "0") == "1" \
               and os.environ.get("SSD_ASYNC_PROXY_SEND", "0") != "1":
                raise ValueError(
                    "SSD_PROXY_STREAM=1 requires SSD_ASYNC_PROXY_SEND=1; "
                    "Policy B compute on proxy_stream is meaningless without "
                    "non-blocking send (the blocking send would re-serialize)."
                )
            # B>1 (docs/duet/13): the historical B=1 constraint was the
            # single-seq Policy B pipeline (proxy wire / selector / phase-2
            # layout), batched in stages M1-M3 — not the long-removed
            # Policy A this comment used to blame. Cap raised 8 -> 32 for
            # the bscale32 campaign (CG bucket axis already derives
            # {1,2,4,8,16,32} from max_bs; no new CG families).
            assert self.max_num_seqs <= 32, \
                "DUET-SSD supports max_num_seqs <= 32 (docs/duet/13 B>1); " \
                f"got {self.max_num_seqs}"
            # B==1-only gates (docs/duet/13 §6, out of scope v1): fail fast
            # at config time instead of hitting the runtime B==1 asserts
            # mid-run.
            if self.max_num_seqs > 1 and (
                    self.duet_exit_topm_gather or self.duet_exit_replica
                    or self.duet_proxy_on_draft):
                raise ValueError(
                    "SSD_DUET_EXIT_TOPM_GATHER / SSD_DUET_EXIT_REPLICA / "
                    "SSD_DUET_PROXY_ON_DRAFT are B==1-only gates "
                    "(docs/duet/13 §6, out of scope v1); unset them or run "
                    f"with max_num_seqs=1 (got max_num_seqs={self.max_num_seqs})."
                )
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
                # Tier-3: early fail (moved up from DraftRunner init; the
                # runner check stays as defense). -O 생존형 raise.
                if self.duet_phase2_k > self.duet_phase1_k:
                    raise ValueError(
                        f"DUET split requires K2 <= K1 (short/long CG bucket "
                        f"invariant); got K1={self.duet_phase1_k}, "
                        f"K2={self.duet_phase2_k}")
                if self.duet_p2_budget is not None and self.duet_p2_budget < 1:
                    raise ValueError(
                        f"duet_p2_budget must be >= 1; got {self.duet_p2_budget}")

            import os as _os_cfg
            # Tier-2 (docs/duet/16): split-K1/K2 is the ONLY DUET path, so
            # `--duet` implies it — the historical SSD_FORCE_SPLIT_K1K2=1
            # export is auto-set here. Spawned child processes inherit
            # environ, and the remaining runtime readers keep working
            # unchanged during the transition (read-site consolidation is
            # deferred; see docs/duet/17).
            if _os_cfg.environ.get("SSD_FORCE_SPLIT_K1K2", "0") != "1":
                _os_cfg.environ["SSD_FORCE_SPLIT_K1K2"] = "1"
            _split_mode = self.duet_phase1_k is not None
            if not _split_mode:
                raise ValueError(
                    "DUET-SSD requires duet_phase1_k / duet_phase2_k "
                    "(split-K1/K2 is the only path; the hybrid / legacy "
                    "two-pass implementations were removed 2026-07 — "
                    "preserved in git history at 19c8f73 and earlier)."
                )
            # Tier-2: SSD_DUET_JIT_SHORT promoted to config.duet_jit_short.
            _env_js = _os_cfg.environ.get("SSD_DUET_JIT_SHORT")
            if _env_js is not None and (_env_js == "1") != self.duet_jit_short:
                print(f"[Config][deprecated] SSD_DUET_JIT_SHORT={_env_js} env "
                      f"overrides duet_jit_short={self.duet_jit_short} "
                      f"(prefer --duet_no_jit_short)", flush=True)
                self.duet_jit_short = (_env_js == "1")
            _os_cfg.environ["SSD_DUET_JIT_SHORT"] = \
                "1" if self.duet_jit_short else "0"
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
            total_budget = self.duet_proxy_total_budget  # Tier-3 single source

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
            # Raw-proxy wire (SSD_DUET_PROXY_ON_DRAFT=1): the draft-side
            # residual topk(top_k) runs over M wire candidates — M must
            # cover the (auto-raised) top_k.
            if self.duet_proxy_on_draft and self.duet_proxy_topm < self.duet_proxy_top_k:
                raise ValueError(
                    f"SSD_DUET_PROXY_TOPM={self.duet_proxy_topm} < auto-raised "
                    f"duet_proxy_top_k={self.duet_proxy_top_k}; raise TOPM."
                )
            if self.duet_exit_topm_gather and self.duet_proxy_topm < self.duet_proxy_top_k:
                raise ValueError(
                    f"SSD_DUET_PROXY_TOPM={self.duet_proxy_topm} < auto-raised "
                    f"duet_proxy_top_k={self.duet_proxy_top_k}; raise TOPM."
                )
            if self.duet_exit_topm_gather and self.duet_proxy_on_draft:
                raise ValueError(
                    "SSD_DUET_EXIT_TOPM_GATHER=1 and SSD_DUET_PROXY_ON_DRAFT=1 "
                    "are mutually exclusive: the top-M gather keeps Policy B "
                    "on rank 0 with the standard chosen wire, while "
                    "proxy-on-draft changes the wire to raw candidates."
                )
            if self.duet_exit_replica and (
                    self.duet_exit_topm_gather or self.duet_proxy_on_draft):
                raise ValueError(
                    "SSD_DUET_EXIT_REPLICA=1 is mutually exclusive with the "
                    "other exit-proxy gates (it removes the exit collective "
                    "entirely and keeps the legacy full-vocab Policy B)."
                )
            # Policy A was removed (2026-07) along with the hybrid / legacy
            # two-pass paths; only the unified K+1 Policy B remains
            # (docs/duet/05-policy-b-fix.md).
            if self.duet_policy != "b":
                raise ValueError(
                    f"duet_policy={self.duet_policy!r} is no longer supported; "
                    "only 'b' (unified K+1) remains. Policy A was removed "
                    "(2026-07); see git history."
                )
            print(f'[Config] DUET-SSD enabled: exit_layer={self.duet_exit_layer}, '
                  f'proxy_top_k={self.duet_proxy_top_k}, '
                  f'draft_fan_out={self.duet_draft_fan_out}, proxy_fan_out={self.duet_proxy_fan_out}, '
                  f'K1={self.duet_phase1_k}, K2={self.duet_phase2_k}',
                  flush=True)

        assert self.max_num_batched_tokens >= self.max_model_len
