import os
from dataclasses import dataclass
from transformers import AutoConfig
import torch
from ssd.paths import DEFAULT_TARGET, DEFAULT_DRAFT
from ssd.engine.helpers.tree_expansion_config import resolve_tree_expansion, validate_tree_score_mode
from ssd.engine.helpers.duet_parameters import canonical_config, positive_int


def _normalize_tree_switches(p1_policy: str, p2_policy: str,
                             legacy_policy: str | None) \
        -> tuple[str, str, str]:
    """Normalize public phase switches without changing them on re-entry.

    ``Config`` is reconstructed with ``dataclasses.replace`` for the draft
    runner, so its already-normalized internal selector appears here again as
    ``legacy_policy``.  Internal ``dynamic``/``backbone`` values must not turn
    P2 on during that second pass.  Only the old explicit ``eagle`` spelling
    and ``off`` retain their historical P2-switch override.
    """
    if p1_policy not in ("off", "on"):
        raise ValueError(
            f"duet_p1_tree_policy must be off|on; got {p1_policy!r}")
    if p2_policy not in ("off", "on"):
        raise ValueError(
            f"duet_p2_tree_policy must be off|on; got {p2_policy!r}")
    reproduction = {
        "dynamic", "backbone", "hybrid", "adaptive", "coverage",
        "confidence", "level", "frontier",
    }
    if legacy_policy == "eagle":
        return p1_policy, "on", "eagle"
    if legacy_policy == "off":
        p2_policy = "off"
        selector = "dynamic" if p1_policy == "on" else "off"
        return p1_policy, p2_policy, selector
    if legacy_policy in reproduction:
        return p1_policy, p2_policy, legacy_policy
    if legacy_policy is not None:
        raise ValueError(
            "deprecated duet_tree_policy accepts only off|eagle or a "
            "legacy reproduction policy; new runs must use "
            "duet_p2_tree_policy=off|on")
    selector = "dynamic" if (
        p1_policy == "on" or p2_policy == "on") else "off"
    return p1_policy, p2_policy, selector


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


@canonical_config
@dataclass
class Config:
    model: str = DEFAULT_TARGET
    dtype: str | None = None  # None=checkpoint; explicit float16 or bfloat16
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 1 
    max_model_len: int = 4096 
    gpu_memory_utilization: float = 0.7
    num_gpus: int = 1
    enforce_eager: bool = False
    greedy_only: bool = False  # Compile homogeneous T=0 draft paths without RNG.
    stochastic_only: bool = False  # CPU request contract; no unused greedy sampler work.
    hf_config: AutoConfig | None = None
    # Internal runner role: early exit always belongs to the target model.
    runner_role: str = "target"
    target_hf_config: AutoConfig | None = None
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
    # Permit RoPE cache extrapolation when the target context window is
    # larger than the draft model's advertised window.  This is opt-in:
    # positions outside the draft's training window can reduce proposal
    # quality even though rotary positions themselves remain computable.
    extend_draft_rope: bool = False
    
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
    # Cache-policy ablation: expose logical K1=0 while retaining the K1=K2
    # graph bucket required by the split engine.  The draft runner skips P1
    # candidate generation, deduplication, and cache population entirely;
    # every cache root is sourced from the early-exit proxy (P2).
    duet_only_proxy: bool = False
    # Cache-root ranking only. Target verification and proposal q are unchanged.
    duet_root_source: str = "complement"  # complement=e(1-q), residual, proxy
    duet_root_normalization: str = "topm"  # retained top-M (default) | full vocabulary
    duet_root_overlap_mix: float = 0.25  # weight of h_bar, not of h_hat
    duet_exit_layer: int | None = None      # None=auto: 2*L//3
    duet_draft_fan_out: int | None = None   # draft-sourced branches per position (None=auto: fan_out//2)
    duet_policy: str = "b"                  # Phase-2 budget policy: "b" = unified K+1 P_iv (only option;
                                            # Policy "a" was removed 2026-07 — see git history).
    # JIT-short (docs/duet/12): on miss, JIT builds a K2-deep chain instead
    # of K_max — champion standard. Promoted from the SSD_DUET_JIT_SHORT env
    # (docs/duet/16 Tier-2, default ON); an explicitly-set env still wins
    # for old-script compat and the resolved value is re-exported for the
    # draft process import-time read.
    duet_jit_short: bool = True
    # Per-forward input widths. DS is a continuation-row cap; its first
    # forward always evaluates all context roots. None uses all DS roots.
    duet_p1_forward_width: int | None = None
    duet_p2_forward_width: int | None = None
    # DUET dynamic trees.  Public policy names deliberately describe whether
    # *our* method is enabled; they do not expose the name of an external
    # baseline that inspired part of the selector. Both phases default to
    # dynamic trees; a chain arm explicitly sets both phase switches off.
    duet_p1_tree_policy: str = "on"           # off | on
    duet_p2_tree_policy: str = "on"           # off | on
    # Deprecated compatibility input for old experiment scripts.  New code
    # and documentation use only the phase-specific on/off controls above.
    # Public ``on`` normalizes to DUET's internal ``dynamic`` selector; the
    # old ``eagle`` spelling remains only for exact experiment reproduction.
    duet_tree_policy: str | None = None
    # Same q-path score in both modes. Last: only newly generated depth;
    # frontier: every unexpanded leaf, across depths. No fitted acceptance.
    duet_tree_expansion: str = "last"
    duet_p1_tree_expansion: str | None = None
    duet_p2_tree_expansion: str | None = None
    duet_tree_c_tensor: int = 3              # 노드당 일괄 샘플 폭 C_tensor
    # One node cap per phase: generation G and verification M are identical.
    duet_p1_tree_verify_nodes: int = 18
    duet_p2_tree_verify_nodes: int = 8
    # DS roots are per position; PS roots are total after DS deduplication.
    duet_p1_roots_per_position: int = 2
    duet_p2_root_count: int | None = None
    # P1 is a future cache-key forest rather than one current-request tree.
    # ``dynamic`` uses the last/frontier choice above; ``backbone`` reserves
    # one continuation lane per root and therefore guarantees one full-depth
    # chain for every root before retaining ordered siblings; ``hybrid`` uses
    # the chain-sized compute budget, preserves two levels where capacity
    # permits, then switches to global selection.
    duet_p1_tree_allocation_policy: str = "dynamic"
    duet_tree_beta: float = 0.5              # 예산 배분 지수 (E1 근거 0.5)
    # Legacy fixed-backbone reproduction knob.  The production dynamic
    # selector decides fanout with its global allocation rule instead.
    duet_tree_fanout_policy: str = "backbone"   # backbone | ctensor
    # Split-K1/K2 mode (per docs/duet/04-split-k1k2-design.md).
    # K1 = Phase 1 forward depth, K2 = Phase 2 forward depth.
    # K1 + K2 == speculate_k. Legacy split-chain graphs require K2 <= K1;
    # the unified tree service has independent phase/verification capacities.
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
    def duet_p1_tree_max_nodes(self):
        return self.duet_p1_tree_verify_nodes

    @property
    def duet_p2_tree_max_nodes(self):
        return self.duet_p2_tree_verify_nodes

    @property
    def duet_tree_nv(self):
        return self.duet_p2_tree_verify_nodes

    @property
    def duet_p2_budget(self):
        return self.duet_p2_forward_width

    @property
    def duet_tree_root_count(self):
        return self.duet_p2_root_count

    @property
    def duet_proxy_top_k(self):
        """Local top-L candidates; target sends only global wire_N entries."""
        return self.duet_proxy_wire_N if self.duet_enabled else 3

    @property
    def duet_tree_proxy_threshold(self):
        return 0.0

    @property
    def duet_tree_conf_threshold(self):
        return 0.0

    @property
    def duet_p1_tree_start_threshold(self):
        return 0.0

    @property
    def duet_p1_tree_conf_threshold(self):
        return 0.0

    @property
    def max_blocks(self):
        return (self.max_model_len + self.kvcache_block_size - 1) // self.kvcache_block_size

    @property
    def duet_tree_enabled(self) -> bool:
        """Whether either phase may serve a dynamic tree response."""
        return (self.duet_enabled and
                (self.duet_p1_tree_policy == "on"
                 or self.duet_p2_tree_policy == "on"))

    @property
    def duet_tree_wire_nodes(self) -> int:
        """Fixed topology-metadata capacity of one draft response.

        This is also the largest dynamic-tree token view and parent-q
        sidecar.  The ordinary chain logits keep their ``speculate_k`` width;
        tree-specific tensors grow only when at least one tree phase is on.
        """
        p1 = (int(self.duet_p1_tree_verify_nodes)
              if self.duet_p1_tree_policy == "on" else 0)
        p2 = (int(self.duet_p2_tree_verify_nodes)
              if self.duet_p2_tree_policy == "on" else 0)
        return max(1, p1, p2)

    @property
    def duet_response_token_width(self) -> int:
        """Token slots carried by one async draft response.

        ``speculate_k`` is the logical chain depth (K1+K2 in DUET) and still
        sizes ordinary draft logits.  A dynamic tree may expose more nodes
        than that depth without performing more sequential rounds, so its
        token envelope must be sized independently.  Keeping this property
        separate avoids inflating chain logits/NCCL traffic by ``V`` while
        allowing, for example, a K1=9 tree to return 18 candidate nodes.
        """
        tree = self.duet_tree_wire_nodes if self.duet_tree_enabled else 0
        return max(int(self.speculate_k), int(tree))

    @property
    def duet_p2_active_root_count(self) -> int:
        """Number of roots that receive the production P2 node budget."""
        if getattr(self, "duet_tree_policy", "off") in (
                "dynamic", "eagle", "hybrid", "adaptive"):
            # Global dynamic expansion keeps root/cache coverage and
            # forward width as separate concepts.  By default every one of
            # the W chain roots is retained; an explicit smaller R remains a
            # supported experiment knob.  R>W is rejected in __post_init__
            # because round zero must evaluate every root in one W-wide
            # forward.
            return (int(self.duet_p2_root_count)
                    if self.duet_p2_root_count is not None
                    else self.duet_proxy_total_budget)
        if getattr(self, "duet_tree_policy", "off") == "coverage":
            # Legacy coverage always keeps every root retained by the chain
            # selector.  The remaining reproduction modes below still honour
            # an explicitly configured root count.
            return self.duet_proxy_total_budget
        if getattr(self, "duet_tree_policy", "off") == "confidence":
            width = self.duet_proxy_total_budget
            k2 = int(self.duet_phase2_k or self.speculate_k)
            return min(width, max(1, (width * k2) // (k2 + 2)))
        if getattr(self, "duet_tree_policy", "off") != "off" \
                and self.duet_p2_root_count is not None:
            return int(self.duet_p2_root_count)
        return self.duet_proxy_total_budget

    @property
    def duet_p2_seed_count(self) -> int:
        """P2 selector width (23번 단계1 — R/W 분리).

        트리 ON + root_count 설정 시 R (선택기가 dedup 후 R개만 취함;
        나머지 W-R 행은 실행 padding — root/키 미생성). 그 외엔 종전
        W(total_budget) — 체인 경로 불변.

        ``SSD_TREE_ROOT_SHADOW`` is a diagnostic-only exception: retain all
        W ranked roots but still allocate production budget to exactly
        ``duet_p2_active_root_count`` roots.  This lets one request stream
        measure whether a real miss matched discarded ranks without changing
        the active R=6 topology.
        """
        # Canonical confidence mode reserves one complete K2-deep backbone
        # plus two rescue nodes for every active root.  With the production
        # W=10, K2=4 layout this deterministically selects floor(40/6)=6
        # roots.  Keeping all ten roots consumed the entire 40-node budget in
        # backbones and measured no useful branching gain; making R a manual
        # knob merely rediscovered the same point by sweep.  The selector
        # still ranks the roots by P_iv, and unused forward lanes expand the
        # selected roots, so this does not reduce the 4xW model workload.
        # Legacy level/frontier retain root_count for exact reproduction.
        if (getattr(self, "duet_tree_policy", "off") == "confidence"
                and os.environ.get("SSD_TREE_ROOT_SHADOW", "") not in
                ("", "0")):
            return self.duet_proxy_total_budget
        return self.duet_p2_active_root_count

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
        # Tier-3: field-based split detection — the property must not depend
        # on call-time environ (post-init the env is guaranteed set for DUET,
        # but a Config object's derived values should be self-contained).
        _split_mode = self.duet_phase1_k is not None
        if _split_mode:
            K_max = max(self.duet_phase1_k, self.duet_phase2_k)
        else:
            K_max = self.speculate_k
        # 23번 단계1: 송신량 = seed_count + dedup 여유 (종전 W + 여유
        # = 28 → 트리 R6에선 24). 체인(off)에선 종전과 동일.
        # The shadow diagnostic deliberately keeps the production wire size;
        # it asks the existing dedup buffer for four additional retained
        # roots, so target compute/traffic and the top-six candidates stay
        # unchanged.
        total_budget = self.duet_p2_active_root_count
        # The only-proxy ablation has no P1 candidates to deduplicate.  Keep
        # two spare wire entries for the existing selector's safety margin.
        if self.duet_only_proxy:
            return total_budget + 2
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
        if self.duet_p1_tree_policy == "on":
            _ctx_max = max(
                int(self.duet_phase1_k or self.speculate_k) + 1,
                int(self.duet_phase2_k or self.speculate_k) + 1,
                int(self.duet_p1_tree_verify_nodes) + 1,
                (int(self.duet_p2_tree_verify_nodes) + 1
                 if self.duet_p2_tree_policy == "on" else 0))
            p1_sum_tree = _ctx_max * int(
                self.duet_p1_roots_per_position)
            p1_sum_full = max(p1_sum_full, p1_sum_tree)
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
        # Fixed-input gains did not establish a full-serving gain for B>1.
        # Preserve the paper default; the existing explicit override remains
        # available for measured deployments and diagnostic comparisons.
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
        if self.duet_p2_forward_width is not None:
            return self.duet_p2_forward_width
        # Tier-3: field-based split detection — the property must not depend
        # on call-time environ (post-init the env is guaranteed set for DUET,
        # but a Config object's derived values should be self-contained).
        _split_mode = self.duet_phase1_k is not None
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
        if self.duet_p2_forward_width is None:
            return self.duet_proxy_fan_out * (K + 1)
        K_max = max(self.duet_phase1_k, self.duet_phase2_k)
        return max(1, round(self.duet_p2_forward_width * (K + 1) / (K_max + 1)))

    def __post_init__(self):
        if self.dtype not in (None, 'float16', 'bfloat16'):
            raise ValueError('dtype must be None|float16|bfloat16')
        from ssd.engine.helpers.root_policy_config import validate_root_policy
        validate_root_policy(self)
        # Normalize the public phase-specific on/off controls before any
        # derived DUET property or legacy implementation branch reads them.
        # Old scripts remain reproducible, but ``eagle`` is never a public
        # name in the new interface.
        _legacy_tree_policy = self.duet_tree_policy
        (self.duet_p1_tree_policy, self.duet_p2_tree_policy,
         self.duet_tree_policy) = _normalize_tree_switches(
            self.duet_p1_tree_policy, self.duet_p2_tree_policy,
            _legacy_tree_policy)
        if self.duet_enabled and self.duet_tree_enabled:
            validate_tree_score_mode()
        for _phase in ("p1", "p2"):
            _expansion = resolve_tree_expansion(self, _phase)
            _allocation = (self.duet_p1_tree_allocation_policy if _phase == "p1"
                           else self.duet_tree_policy)
            if (getattr(self, f"duet_{_phase}_tree_policy") == "on"
                    and _expansion == "frontier"
                    and _allocation not in ("dynamic", "eagle")):
                raise ValueError(
                    f"{_phase} frontier requires dynamic allocation, not {_allocation!r}")
        for name in ('duet_p1_tree_verify_nodes', 'duet_p2_tree_verify_nodes',
                     'duet_p1_roots_per_position'):
            positive_int(name, getattr(self, name))
        for name in ('duet_p1_forward_width', 'duet_p2_forward_width',
                     'duet_p2_root_count'):
            positive_int(name, getattr(self, name), optional=True)

        model = self.model 
        if not (os.path.isdir(model)):
            raise ValueError('Invalid config.py contract: os.path.isdir(model)')

        if not (1 <= self.num_gpus <= 8):
            raise ValueError('Invalid config.py contract: 1 <= self.num_gpus <= 8')
        if self.max_num_seqs < 1 or self.kvcache_block_size < 1:
            raise ValueError("max_num_seqs and kvcache_block_size must be positive")
        if self.speculate and self.speculate_k < 1:
            raise ValueError("speculate_k must be positive")
        if self.draft_async and self.async_fan_out < 1:
            raise ValueError("async_fan_out must be positive")
        requested_max_model_len = int(self.max_model_len)
        if requested_max_model_len < 1:
            raise ValueError(
                f"max_model_len must be positive; got {self.max_model_len}")

        self.hf_config = AutoConfig.from_pretrained(model)
        if self.dtype is not None:
            self.hf_config.torch_dtype = getattr(torch, self.dtype)
        _ensure_head_dim(self.hf_config)
        if self.runner_role not in ("target", "draft"):
            raise ValueError("runner_role must be target|draft")
        if self.runner_role == "target":
            self.target_hf_config = self.hf_config
        elif self.target_hf_config is None:
            raise ValueError("Draft config requires the original target_hf_config")
        target_hf = self.target_hf_config
        target_max_position_embeddings = int(
            self.hf_config.max_position_embeddings)
        self.max_model_len = min(
            requested_max_model_len, target_max_position_embeddings)
        if self.speculate:
            draft = self.draft
            self.draft_hf_config = AutoConfig.from_pretrained(draft)
            if self.dtype is not None:
                self.draft_hf_config.torch_dtype = getattr(torch, self.dtype)
            _ensure_head_dim(self.draft_hf_config)
            draft_max_position_embeddings = int(
                self.draft_hf_config.max_position_embeddings)
            if self.extend_draft_rope:
                # RoPE is generated analytically from inv_freq.  Extending
                # this cache avoids an artificial context clamp, while the
                # target model and the user-requested limit remain hard caps.
                if draft_max_position_embeddings < self.max_model_len:
                    print(
                        "[Config] Extending draft RoPE cache: "
                        f"{draft_max_position_embeddings} -> "
                        f"{self.max_model_len} positions "
                        "(draft quality beyond its trained window is not "
                        "guaranteed)",
                        flush=True,
                    )
                    self.draft_hf_config.max_position_embeddings = \
                        self.max_model_len
            else:
                self.max_model_len = min(
                    self.max_model_len, draft_max_position_embeddings)
            if self.draft_async:
                if self.fan_out_list is None: 
                    self.fan_out_list = [self.async_fan_out] * (self.speculate_k + 1)
                    self.MQ_LEN = sum(self.fan_out_list)
                if (len(self.fan_out_list) != self.speculate_k + 1
                        or any(x < 0 for x in self.fan_out_list)
                        or sum(self.fan_out_list) <= 0):
                    raise ValueError("fan_out_list requires K+1 nonnegative entries and a positive total")
                self.MQ_LEN = sum(self.fan_out_list)
                if self.fan_out_list_miss is None:
                    self.fan_out_list_miss = self.fan_out_list
                if (len(self.fan_out_list_miss) != self.speculate_k + 1
                        or any(x < 0 for x in self.fan_out_list_miss)
                        or sum(self.fan_out_list_miss) <= 0):
                    raise ValueError("fan_out_list_miss requires K+1 nonnegative entries and a positive total")
                if not (sum(self.fan_out_list_miss) == sum(self.fan_out_list)):
                    raise ValueError("ERROR in Config: fan_out_list_miss must be the same as fan_out_list")
                
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
            if not (self.draft_async):
                raise ValueError("DUET-SSD requires draft_async=True")
            if not (self.speculate):
                raise ValueError("DUET-SSD requires speculate=True")
            if not (target_hf.model_type == "llama"):
                raise ValueError("DUET-SSD only supports Llama models")
            if not (not self.use_eagle):
                raise ValueError("DUET-SSD + EAGLE: not yet implemented (eagle_acts split collection needed)")
            if not (not self.enforce_eager):
                raise ValueError("DUET-SSD requires CudaGraph mode (enforce_eager must be False)")
            if not (self.jit_speculate):
                raise ValueError("DUET-SSD requires jit_speculate=True (miss rows need valid logits_q)")
            if self.duet_only_proxy and self.duet_tree_enabled:
                raise ValueError(
                    "duet_only_proxy is a chain-only ablation; set both "
                    "duet_p1_tree_policy=off and duet_p2_tree_policy=off")

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
            if not (self.max_num_seqs <= 32):
                raise ValueError("DUET-SSD supports max_num_seqs <= 32 (docs/duet/13 B>1); " \
                f"got {self.max_num_seqs}")
            # B==1-only gates (docs/duet/13 §6, out of scope v1): fail fast
            # at config time instead of hitting the runtime B==1 asserts
            # mid-run.
            if self.max_num_seqs > 1 and (
                    self.duet_exit_topm_gather
                    or (self.duet_exit_replica and not self.duet_tree_enabled)
                    or self.duet_proxy_on_draft):
                raise ValueError(
                    "Top-M/proxy-on-draft and non-tree exit-replica modes "
                    "are B==1-only gates "
                    "(docs/duet/13 §6, out of scope v1); unset them or run "
                    f"with max_num_seqs=1 (got max_num_seqs={self.max_num_seqs})."
                )
            if self.duet_exit_layer is None:
                L = target_hf.num_hidden_layers
                self.duet_exit_layer = (2 * L) // 3
            if not (0 <= self.duet_exit_layer < target_hf.num_hidden_layers):
                raise ValueError(f"duet_exit_layer must be in [0, {target_hf.num_hidden_layers}), got {self.duet_exit_layer}")
            if self.duet_draft_fan_out is None:
                self.duet_draft_fan_out = max(1, self.async_fan_out // 2)
            explicit_split = (self.duet_phase1_k is not None
                              and self.duet_phase2_k is not None
                              and self.duet_p2_forward_width is not None
                              and self.duet_p2_forward_width > 0)
            limit = self.async_fan_out if explicit_split else self.async_fan_out - 1
            if not 1 <= self.duet_draft_fan_out <= limit:
                raise ValueError(
                    f"duet_draft_fan_out must be in [1, {limit}]; got "
                    f"{self.duet_draft_fan_out}. Equality with async_fan_out "
                    "requires explicit positive split K1/K2 and P2 budget.")
            self.duet_proxy_fan_out = self.async_fan_out - self.duet_draft_fan_out

            # Phase 1/2 K validation moved up (auto-raise needs validated K1/K2).
            if (self.duet_phase1_k is None) != (self.duet_phase2_k is None):
                raise ValueError(
                    "duet_phase1_k and duet_phase2_k must both be set or both None; "
                    f"got K1={self.duet_phase1_k}, K2={self.duet_phase2_k}"
                )
            if self.duet_phase1_k is not None:
                # Runtime shape contracts must survive ``python -O``.  The
                # former asserts disappeared in production and let a stale
                # ``--k`` size response/logit buffers differently from
                # K1/K2.  Changing K1=9 to K1=8 while leaving ``--k 13`` was
                # one concrete way to enter that corrupt configuration.
                if self.duet_phase1_k <= 0:
                    raise ValueError(
                        "duet_phase1_k must be > 0; got "
                        f"{self.duet_phase1_k}")
                if self.duet_phase2_k <= 0:
                    raise ValueError(
                        "duet_phase2_k must be > 0; got "
                        f"{self.duet_phase2_k}")
                if (self.duet_phase1_k + self.duet_phase2_k
                        != self.speculate_k):
                    raise ValueError(
                        "duet_phase1_k + duet_phase2_k must equal "
                        "speculate_k; got "
                        f"K1={self.duet_phase1_k} + "
                        f"K2={self.duet_phase2_k} != "
                        f"speculate_k={self.speculate_k}. Omit --k so "
                        "bench.py derives it, or set --k to K1+K2.")
                # Tier-3: early fail (moved up from DraftRunner init; the
                # runner check stays as defense). -O 생존형 raise.
                from ssd.engine.helpers.batch_tree_common import enabled as batch_tree_enabled
                if (self.duet_phase2_k > self.duet_phase1_k
                        and not batch_tree_enabled(self)):
                    raise ValueError(
                        f"DUET split requires K2 <= K1 (short/long CG bucket "
                        f"invariant); got K1={self.duet_phase1_k}, "
                        f"K2={self.duet_phase2_k}")
                if self.duet_p2_forward_width is not None and self.duet_p2_forward_width < 1:
                    raise ValueError(
                        f"duet_p2_forward_width must be >= 1; got {self.duet_p2_forward_width}")
                # Dynamic-tree knobs — -O 생존형 raise.
                if self.duet_tree_policy not in (
                        "off", "dynamic", "hybrid", "adaptive", "eagle",
                        "coverage", "backbone", "confidence", "level",
                        "frontier"):
                    raise ValueError(
                        f"duet_tree_policy must be "
                        f"off|dynamic|adaptive|eagle|coverage|backbone|"
                        f"confidence|level|frontier; "
                        f"got {self.duet_tree_policy!r}")
                if self.duet_tree_enabled:
                    # Packed token IDs use uint32; wire payload remains int64.
                    if self.hf_config.vocab_size > 2**32:
                        raise ValueError(
                            f"DUET tree pack requires vocab_size <= 2**32 "
                            f"(token bits 0-31); got "
                            f"{self.hf_config.vocab_size}")
                    if self.duet_tree_fanout_policy not in (
                            "backbone", "ctensor"):
                        raise ValueError(
                            f"duet_tree_fanout_policy must be "
                            f"backbone|ctensor; got "
                            f"{self.duet_tree_fanout_policy!r}")
                    if self.duet_p1_tree_allocation_policy not in (
                            "dynamic", "backbone", "hybrid"):
                        raise ValueError(
                            "duet_p1_tree_allocation_policy must be "
                            "dynamic|backbone|hybrid; got "
                            f"{self.duet_p1_tree_allocation_policy!r}")
                    if not (1 <= self.duet_tree_c_tensor <= 8):
                        raise ValueError(
                            f"duet_tree_c_tensor must be in [1,8]; "
                            f"got {self.duet_tree_c_tensor}")
                    # Tree response width is independent of sequential draft
                    # depth.  The fixed int64 token envelope and topology
                    # sidecar grow to the active phase maximum, while the
                    # ordinary [K,V] logits wire remains speculate_k wide.
                    for _phase, _enabled, _nodes in (
                            ("P1", self.duet_p1_tree_policy == "on",
                             self.duet_p1_tree_max_nodes),
                            ("P2", self.duet_p2_tree_policy == "on",
                             self.duet_p2_tree_max_nodes)):
                        if _enabled and not (1 <= _nodes < self.max_model_len):
                            raise ValueError(
                                f"{_phase} tree max nodes must be in "
                                f"[1,max_model_len={self.max_model_len}); "
                                f"got {_nodes}")
                    if self.duet_p2_tree_policy == "on" \
                            and self.duet_p2_root_count is not None and \
                            self.duet_p2_root_count < 1:
                        raise ValueError(
                            f"duet_p2_root_count must be >= 1; got "
                            f"{self.duet_p2_root_count}")
                    if (self.duet_p2_tree_policy == "on"
                            and self.duet_tree_policy == "coverage"
                            and self.duet_p2_root_count is not None):
                        raise ValueError(
                            "duet_p2_root_count cannot be set with "
                            "duet_tree_policy=coverage: coverage keeps all "
                            "chain roots by definition")
                    if (self.duet_p2_tree_policy == "on"
                            and self.duet_tree_policy in (
                                "coverage", "backbone", "adaptive")
                            and self.duet_tree_nv < self.duet_phase2_k):
                        raise ValueError(
                            f"{self.duet_tree_policy} requires "
                            f"duet_tree_nv >= K2 to keep "
                            f"the complete chain backbone; got Nv="
                            f"{self.duet_tree_nv}, K2={self.duet_phase2_k}")
                    if (self.duet_p2_tree_policy == "on"
                            and self.duet_tree_policy in (
                                "coverage", "backbone", "adaptive")
                            and self.duet_tree_nv
                            > self.duet_phase2_k * self.duet_tree_c_tensor):
                        raise ValueError(
                            f"{self.duet_tree_policy} Nv="
                            f"{self.duet_tree_nv} is not "
                            f"generatable by K2={self.duet_phase2_k} rounds "
                            f"and C={self.duet_tree_c_tensor}")
                    # 이슈 #27: tip 의무 lane이 W(=P2 예산)를 초과하면
                    # rollout이 구조적으로 불가 — config에서 조기 차단.
                    if (self.duet_p2_tree_policy == "on"
                            and self.duet_p2_root_count is not None
                            and self.duet_p2_root_count
                            > self.duet_proxy_total_budget):
                        # 리뷰9-2: p2_budget 명시 여부와 무관하게 유효
                        # W(파생값 포함)와 비교 (이슈 #27)
                        raise ValueError(
                            f"duet_p2_root_count "
                            f"({self.duet_p2_root_count}) must be <= "
                            f"W ({self.duet_proxy_total_budget}) — "
                            f"tip 의무 lane > W (이슈 #27)")
                    # Tree-hit materialization now has its own common-width
                    # layout (max active phase nodes + recovery).  The old
                    # split_k2-width restriction ``P2 nodes + 1 <= W`` no
                    # longer applies: response capacity is the only bound.
                    # 이슈 #15: 트리 v1 미지원 proxy 게이트 — raw-proxy는
                    # _tree_step_p1p2에 변환 분기가 없고(KeyError),
                    # topm_gather dict-wire는 pack_piv가 없어 트리 셀렉터
                    # 계약(P_iv 관통)을 깨뜨린다. 지원 전까지 명시 차단.
                    if self.duet_proxy_on_draft:
                        raise ValueError(
                            "DUET dynamic trees are not supported with "
                            "SSD_DUET_PROXY_ON_DRAFT=1 (v1 — raw-proxy 변환 "
                            "분기 미구현)")
                    if self.duet_exit_topm_gather:
                        raise ValueError(
                            "DUET dynamic trees are not supported with "
                            "SSD_DUET_EXIT_TOPM_GATHER=1 (v1 — dict-wire에 "
                            "P_iv pack 미구현)")

            import os as _os_cfg
            # Tier-2 (docs/duet/16): split-K1/K2 is the ONLY DUET path, so
            # `--duet` implies it — the historical SSD_FORCE_SPLIT_K1K2=1
            # export is auto-set here. Spawned child processes inherit
            # environ, and the remaining runtime readers keep working
            # unchanged during the transition (read-site consolidation is
            # deferred; see docs/duet/internal/17).
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
            # Each position contributes top-L with L equal to the global
            # wire count. Global top-N is then selected on target. The wire
            # includes the derived DS dedup margin; L is not a tuning knob.
            if self.duet_proxy_on_draft or self.duet_exit_topm_gather:
                raise ValueError('Local root policy requires full-vocabulary exit logits; raw/top-M proxy gates are unsupported')
            print(f'[DUET policy] root={self.duet_root_source} '
                  f'normalization={self.duet_root_normalization} '
                  f'position={1-self.duet_root_overlap_mix:g} observed + '
                  f'{self.duet_root_overlap_mix:g} overlap; '
                  f'tree_requested={os.getenv("SSD_TREE_EXPANSION_POLICY", "q_path")} '
                  f'expansion_p1={resolve_tree_expansion(self, "p1")} '
                  f'expansion_p2={resolve_tree_expansion(self, "p2")} '
                  '(resolved per executor)',flush=True)
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
            print(f'[Config] DUET v1.1: exit={self.duet_exit_layer}, '
                  f'K_DS/PS={self.duet_phase1_k}/{self.duet_phase2_k}, '
                  f'DS_roots_per_position={self.duet_p1_roots_per_position}, '
                  f'PS_roots={self.duet_p2_active_root_count}, '
                  f'DS_continuation_input_cap={self.duet_p1_forward_width}, '
                  f'PS_forward_inputs={self.duet_proxy_total_budget}, '
                  f'DS/PS_G=M={self.duet_p1_tree_verify_nodes}/{self.duet_p2_tree_verify_nodes}, '
                  f'C={self.duet_tree_c_tensor}, '
                  f'auto_top_L={self.duet_proxy_top_k}, wire_N={self.duet_proxy_wire_N}, '
                  f'normalization={self.duet_root_normalization}, '
                  f'expansion={self.duet_tree_expansion}, '
                  f'DS/PS_tree={self.duet_p1_tree_policy}/{self.duet_p2_tree_policy}',
                  flush=True)


        if not (self.max_num_batched_tokens >= self.max_model_len):
            raise ValueError('Invalid config.py contract: self.max_num_batched_tokens >= self.max_model_len')
