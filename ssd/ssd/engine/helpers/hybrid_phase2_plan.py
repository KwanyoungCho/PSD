"""HybridPhase2Plan — per-step plan object for the MESA Phase 2 hybrid loop.

Per ``docs/mesa/01-design.md Part 5``:

- One ``HybridPhase2Plan`` instance per ``DraftRunner``.
- All tensor fields are allocated **once** at engine init, sized for the long
  bucket worst case. ``phase2_build`` then fills these in-place per step from
  the incoming ``valid_k`` and the dynamically-received proxy ``fan_out_list``.
- The depth loop (`_decode_phase2_hybrid`) reads from these tensors only —
  no Python-side tensor construction in the hot path.
- Two CUDAGraphs (long / short) are captured at engine init; runtime selects
  one via ``graph_key`` per step.

Row layout (B=1):

    flat batch = [ continuation rows (cont_row_count) | proxy rows (proxy_row_count) ]
                 \_______ phase_split_offset ________/
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class HybridPhase2Plan:
    """Per-step Phase 2 hybrid plan.

    Tensor fields are allocated once at construction time at max-size (long
    bucket). ``fill(valid_k, ...)`` overwrites them in-place per step. The
    depth loop reads from these without any further allocation.
    """

    # ─────────────────────────────────────────────────────────────────────
    # Static config (set once at __init__)
    # ─────────────────────────────────────────────────────────────────────
    K_long: int                    # = mesa_phase1_k + mesa_phase2_k = speculate_k
    K_short: int                   # = mesa_phase2_k
    K1: int                        # Phase 1 forward depth
    K2: int                        # Phase 2 forward depth (= K_short)
    mesa_draft_fan_out: int        # fan_out per Phase 1 fork position
    mesa_proxy_fan_out: int        # fan_out per proxy fork position
    max_pages_per_row: int         # block-table width (max KV pages a row can address)
    max_packed_mask_size: int      # per-depth packed mask buffer size

    # ─────────────────────────────────────────────────────────────────────
    # Per-step scalars (filled by ``fill``)
    # ─────────────────────────────────────────────────────────────────────
    valid_k: int = 0
    cont_row_count: int = 0        # = (valid_k + 1) × mesa_draft_fan_out
    proxy_row_count: int = 0       # = (valid_k + 1) × mesa_proxy_fan_out
    total_row_count: int = 0       # = cont_row_count + proxy_row_count
    phase_split_offset: int = 0    # = cont_row_count
    name: str = ""                 # "phase2_hybrid_long" or "_short"
    graph_key: str = ""            # CUDAGraph dispatch key
    step_cache_hit: bool = True    # actual seq cache_hit at this step (B=1);
                                    # used by parity dumps to pick the right
                                    # fan_idx_hit vs fan_idx_miss vector.
                                    # NOT a region tag — distinct from
                                    # per_row_region_id which is 0=cont/1=proxy.
    # Hot-path scalar caches (filled by _build_phase2_hybrid_plan to avoid
    # GPU→CPU syncs in run_phase2_hybrid_cudagraph's depth loop).
    step_num_tokens: int = 0
    # Per-depth scalar caches (Python list, K2 entries).
    # Populated by _build_phase2_hybrid_plan; consumed by runtime.
    per_depth_L: list = field(default_factory=list)              # ctx_len per depth
    per_depth_n_pages: list = field(default_factory=list)         # ceil(L/block_size)
    per_depth_bytes_per_row: list = field(default_factory=list)   # L_padded // 8
    per_depth_n_pad_bytes: list = field(default_factory=list)     # padding bytes per depth (zero-filled in pre-built mask buffer)

    # ─────────────────────────────────────────────────────────────────────
    # Allocated-once buffers (max-sized; long bucket worst case)
    #
    # All shapes use ``MAX_TOTAL`` = (K_long + 1) × (mesa_draft_fan_out +
    # mesa_proxy_fan_out). The per-step ``fill`` writes the
    # ``[:total_row_count]`` slice; the rest is undefined and unused.
    # ─────────────────────────────────────────────────────────────────────

    # Row classification
    per_row_region_id: torch.Tensor = field(default=None)            # int8 [MAX_TOTAL]
    per_row_valid_source_kind: torch.Tensor = field(default=None)    # int64 [MAX_TOTAL]

    # Per-row × per-depth attention plumbing
    per_row_context_lens_by_depth: torch.Tensor = field(default=None)   # int32 [MAX_TOTAL, K2]
    per_row_slot_maps_by_depth: torch.Tensor = field(default=None)      # int32 [MAX_TOTAL, K2]
    per_row_kv_indptr_by_depth: torch.Tensor = field(default=None)      # int32 [MAX_TOTAL + 1, K2]
    per_row_kv_indices_by_depth: torch.Tensor = field(default=None)     # int32 [K2, max_total_indices]
    per_row_block_tables: torch.Tensor = field(default=None)            # int32 [MAX_TOTAL, max_pages_per_row]

    # Per-depth packed masks (built from the above by build_hybrid_packed_mask)
    per_depth_packed_masks: torch.Tensor = field(default=None)          # uint8 [K2, max_packed_mask_size]
    per_depth_mask_indptr: torch.Tensor = field(default=None)           # int32 [K2, MAX_TOTAL + 1]

    # Phase 9C: pre-built runtime-ready tensors per depth.
    # Populated in _build_phase2_hybrid_plan; consumed by run_phase2_hybrid_*
    # to avoid per-depth allocation and CPU↔GPU syncs.
    per_depth_kv_indptr_cpu: torch.Tensor = field(default=None)         # int32 CPU [K2, MAX_TOTAL + 1]
    per_depth_kv_lens_gpu: torch.Tensor = field(default=None)           # int32 GPU [K2, MAX_TOTAL]
    per_depth_kv_lpl_gpu: torch.Tensor = field(default=None)            # int32 GPU [K2, MAX_TOTAL]
    qo_indptr_cpu: torch.Tensor = field(default=None)                   # int32 CPU [MAX_TOTAL + 1] (invariant)

    # Position math (re-used arange, sized by K2)
    step_pos_offsets: torch.Tensor = field(default=None)                # int64 [K2]
    step_rope_offsets: torch.Tensor = field(default=None)               # int64 [K2]

    # Per-row initial positions (rope + absolute) — filled per step
    cont_initial_positions: torch.Tensor = field(default=None)          # int64 [max_cont]
    cont_initial_rope_positions: torch.Tensor = field(default=None)
    proxy_initial_positions: torch.Tensor = field(default=None)
    proxy_initial_rope_positions: torch.Tensor = field(default=None)

    # Pre-allocated workspaces for the per-step build (replaces per-step
    # torch.arange / torch.zeros allocations in the hot path). All sized for
    # the long-bucket worst case at engine init.
    bit_weights_u8: torch.Tensor = field(default=None)                  # uint8 [8] — constant 1<<arange(8)
    cont_idx_arange: torch.Tensor = field(default=None)                 # int64 [max_cont]
    proxy_idx_arange: torch.Tensor = field(default=None)                # int64 [max_proxy]
    kv_pos_arange: torch.Tensor = field(default=None)                   # int64 [max_L]
    total_arange_p1: torch.Tensor = field(default=None)                 # int32 GPU [max_total + 1]
    total_arange_p1_cpu: torch.Tensor = field(default=None)             # int32 CPU [max_total + 1]
    full_mask_buf: torch.Tensor = field(default=None)                   # bool  [max_total, max_L_padded]

    # ─────────────────────────────────────────────────────────────────────
    # Construction helper
    # ─────────────────────────────────────────────────────────────────────
    @classmethod
    def create(
        cls,
        *,
        K1: int,
        K2: int,
        mesa_draft_fan_out: int,
        mesa_proxy_fan_out: int,
        max_pages_per_row: int,
        max_packed_mask_size: int,
        max_L: int,
        device: torch.device,
    ) -> "HybridPhase2Plan":
        """Allocate all tensor fields max-sized for the long bucket."""
        K_long = K1 + K2
        K_short = K2

        # Worst-case row counts (long-hit step)
        max_cont = (K_long + 1) * mesa_draft_fan_out
        max_proxy = (K_long + 1) * mesa_proxy_fan_out
        max_total = max_cont + max_proxy

        # KV indices are depth-major; max total indices ≤ MAX_TOTAL × max_pages_per_row
        max_total_indices = max_total * max_pages_per_row

        plan = cls(
            K_long=K_long,
            K_short=K_short,
            K1=K1,
            K2=K2,
            mesa_draft_fan_out=mesa_draft_fan_out,
            mesa_proxy_fan_out=mesa_proxy_fan_out,
            max_pages_per_row=max_pages_per_row,
            max_packed_mask_size=max_packed_mask_size,
        )

        plan.per_row_region_id = torch.zeros(max_total, dtype=torch.int8, device=device)
        plan.per_row_valid_source_kind = torch.full(
            (max_total,), K_long, dtype=torch.int64, device=device,
        )

        plan.per_row_context_lens_by_depth = torch.zeros(
            (max_total, K2), dtype=torch.int32, device=device,
        )
        plan.per_row_slot_maps_by_depth = torch.full(
            (max_total, K2), -1, dtype=torch.int32, device=device,
        )
        plan.per_row_kv_indptr_by_depth = torch.zeros(
            (max_total + 1, K2), dtype=torch.int32, device=device,
        )
        plan.per_row_kv_indices_by_depth = torch.zeros(
            (K2, max_total_indices), dtype=torch.int32, device=device,
        )
        plan.per_row_block_tables = torch.full(
            (max_total, max_pages_per_row), -1, dtype=torch.int32, device=device,
        )
        plan.per_depth_packed_masks = torch.zeros(
            (K2, max_packed_mask_size), dtype=torch.uint8, device=device,
        )
        plan.per_depth_mask_indptr = torch.zeros(
            (K2, max_total + 1), dtype=torch.int32, device=device,
        )

        plan.step_pos_offsets = torch.arange(K2, dtype=torch.int64, device=device)
        plan.step_rope_offsets = torch.arange(K2, dtype=torch.int64, device=device)

        plan.cont_initial_positions = torch.zeros(max_cont, dtype=torch.int64, device=device)
        plan.cont_initial_rope_positions = torch.zeros(max_cont, dtype=torch.int64, device=device)
        plan.proxy_initial_positions = torch.zeros(max_proxy, dtype=torch.int64, device=device)
        plan.proxy_initial_rope_positions = torch.zeros(max_proxy, dtype=torch.int64, device=device)

        # Phase 9C: pre-allocate runtime-ready buffers.
        plan.per_depth_kv_indptr_cpu = torch.zeros(
            (K2, max_total + 1), dtype=torch.int32, device='cpu', pin_memory=True,
        )
        plan.per_depth_kv_lens_gpu = torch.zeros(
            (K2, max_total), dtype=torch.int32, device=device,
        )
        plan.per_depth_kv_lpl_gpu = torch.zeros(
            (K2, max_total), dtype=torch.int32, device=device,
        )
        # qo_indptr is invariant: arange(max_total + 1).
        plan.qo_indptr_cpu = torch.arange(
            max_total + 1, dtype=torch.int32, device='cpu',
        )

        # Phase 9D: build-time workspaces (replace per-step allocations).
        # bit_weights = [1, 2, 4, 8, 16, 32, 64, 128] — completely constant.
        plan.bit_weights_u8 = (
            1 << torch.arange(8, dtype=torch.uint8, device=device)
        )
        # Pre-built per-row index aranges; the build slices [:cont_count] /
        # [:proxy_count] each step.
        plan.cont_idx_arange = torch.arange(
            max_cont, dtype=torch.int64, device=device,
        )
        plan.proxy_idx_arange = torch.arange(
            max_proxy, dtype=torch.int64, device=device,
        )
        # max-L kv_pos arange — sized to fit the worst-case physical KV bound
        # (num_tokens + K_step + K_long*MQ_p1 + K2*MQ_proxy). Caller passes
        # ``max_L``; we slice [:L] each step instead of allocating a fresh
        # arange.
        plan.kv_pos_arange = torch.arange(
            max_L, dtype=torch.int64, device=device,
        )
        # Pre-allocated full mask workspace ([max_total, max_L_padded] bool).
        # Lets the build write rows in-place without torch.cat / pad allocs.
        max_L_padded = ((max_L + 7) // 8) * 8
        plan.full_mask_buf = torch.zeros(
            (max_total, max_L_padded), dtype=torch.bool, device=device,
        )
        # Pre-built (max_total + 1) int32 arange for kv_indptr / mask_indptr
        # construction (multiplied by per-depth strides each step). Both GPU
        # and CPU variants — caller picks based on target tensor's device.
        plan.total_arange_p1 = torch.arange(
            max_total + 1, dtype=torch.int32, device=device,
        )
        plan.total_arange_p1_cpu = torch.arange(
            max_total + 1, dtype=torch.int32, device='cpu',
        )

        return plan

    # ─────────────────────────────────────────────────────────────────────
    # Per-step fill — set scalars from incoming valid_k. Tensor contents
    # are written by ``_build_phase2_hybrid_plan`` in draft_runner.
    # ─────────────────────────────────────────────────────────────────────
    def begin_step(self, valid_k: int, fan_out_list: list[int] | None = None,
                     skip_cont: bool = False, skip_proxy: bool = False) -> None:
        """Compute per-step row counts and dispatch keys.

        Args:
            valid_k: incoming hit's row depth (K_long or K_short).
            fan_out_list: per-position proxy fan-outs (length = valid_k + 1)
                from this step's Policy A/B allocation. If None, falls back
                to a uniform `mesa_proxy_fan_out` per position (legacy default
                — preserved for non-Policy paths).

        Invariants (per reviewer guidance):
          - cont_row_count = (valid_k + 1) × mesa_draft_fan_out  (Phase 1
            seeds — always uniform fan-out per position)
          - proxy_row_count = sum(fan_out_list)  (dynamic Policy A/B
            semantics; equals proxy_fan_out_total = pfo × (valid_k + 1) in
            steady state but kept dynamic so per-position skew flows
            through)
        """
        self.valid_k = valid_k
        self.cont_row_count = 0 if skip_cont else (valid_k + 1) * self.mesa_draft_fan_out
        if skip_proxy:
            self.proxy_row_count = 0
        elif fan_out_list is None:
            self.proxy_row_count = (valid_k + 1) * self.mesa_proxy_fan_out
        else:
            assert len(fan_out_list) == valid_k + 1, (
                f"fan_out_list length {len(fan_out_list)} != valid_k+1 "
                f"= {valid_k + 1}"
            )
            self.proxy_row_count = int(sum(fan_out_list))
        self.total_row_count = self.cont_row_count + self.proxy_row_count
        self.phase_split_offset = self.cont_row_count

        if valid_k == self.K_long:
            self.name = "phase2_hybrid_long"
            self.graph_key = "fi_phase2_hybrid_long"
        else:
            assert valid_k == self.K_short, (
                f"HybridPhase2Plan only supports valid_k ∈ {{K_long={self.K_long}, "
                f"K_short={self.K_short}}}; got {valid_k}"
            )
            self.name = "phase2_hybrid_short"
            self.graph_key = "fi_phase2_hybrid_short"

    # ─────────────────────────────────────────────────────────────────────
    # Convenience views (do not allocate)
    # ─────────────────────────────────────────────────────────────────────
    @property
    def cont_slice(self) -> slice:
        """Index slice for continuation rows in the flat hybrid batch."""
        return slice(0, self.cont_row_count)

    @property
    def proxy_slice(self) -> slice:
        """Index slice for proxy-sourced rows in the flat hybrid batch."""
        return slice(self.cont_row_count, self.total_row_count)
