import os
import torch
import numpy as np
from typing import List
from ssd.utils.context import set_context, get_context, reset_context
from ssd.engine.helpers.mask_helpers import get_custom_mask
from time import perf_counter


## RUN CUDAGRAPHS
@torch.inference_mode()
def run_verify_cudagraph(model_runner, input_ids, positions, last_only, graph_vars,
                          bucket="verify"):
    """Replay multi-query (K+1) decode CG.

    Step 9B-0: ``bucket`` selects the CG family. Default = "verify"
    (K_long+1 wide). Draft glue short-bucket uses bucket="verify_short"
    (K_short+1 wide). k_plus_1 is derived from graph_bs_list / input shape.
    """
    context = get_context()
    if bucket == "verify_short":
        k_plus_1 = model_runner.config.mesa_phase2_k + 1
    elif bucket == "verify_k1":
        k_plus_1 = model_runner.config.mesa_phase1_k + 1
    elif bucket == "verify_k2":
        k_plus_1 = model_runner.config.mesa_phase2_k + 1
    else:
        k_plus_1 = model_runner.config.speculate_k + 1
    orig_bs = input_ids.size(0) // k_plus_1  # orig_bs = N here

    wrapper_bs = next(
        x for x in model_runner.graph_bs_list[bucket] if x >= orig_bs)
    graph = model_runner.graphs[bucket][wrapper_bs]

    for k, v in graph_vars.items():
        if k != "outputs":
            v.zero_()

    # Pad to graph bucket size if needed (fixes B>=6 crash from non-monotonic cu_seqlens_q)
    if wrapper_bs > orig_bs:
        pad_bs = wrapper_bs - orig_bs
        pad_flat = pad_bs * k_plus_1
        dev = input_ids.device

        input_ids = torch.cat([input_ids, torch.zeros(pad_flat, dtype=input_ids.dtype, device=dev)])
        positions = torch.cat([positions, torch.zeros(pad_flat, dtype=positions.dtype, device=dev)])
        slot_mapping = torch.cat([
            context.slot_mapping,
            torch.full((pad_flat,), -1, dtype=context.slot_mapping.dtype, device=dev)])
        # Repeat last real row for ghost sequences (valid page table / context len)
        bt = context.block_tables
        cl = context.context_lens
        block_tables = torch.cat([bt, bt[orig_bs-1:orig_bs].expand(pad_bs, -1).contiguous()])
        context_lens = torch.cat([cl, cl[orig_bs-1:orig_bs].expand(pad_bs).contiguous()])
        bs = wrapper_bs
    else:
        slot_mapping = context.slot_mapping
        block_tables = context.block_tables
        context_lens = context.context_lens
        bs = orig_bs

    graph_vars["input_ids"][:bs * k_plus_1] = input_ids
    graph_vars["positions"][:bs * k_plus_1] = positions
    graph_vars["slot_mapping"][:bs * k_plus_1] = slot_mapping
    graph_vars["context_lens"][:bs] = context_lens
    # Construct cu_seqlens_q for FULL padded batch (monotonically increasing)
    seqlen_q = torch.full(
        (bs,), k_plus_1, dtype=torch.int32, device=graph_vars["cu_seqlens_q"].device)
    cu = graph_vars["cu_seqlens_q"][:bs + 1]
    cu.zero_()
    cu[1:].copy_(torch.cumsum(seqlen_q, 0))

    if block_tables is not None:
        graph_vars["block_tables"][:bs, :block_tables.size(1)] = block_tables

    _pt = os.environ.get("SSD_PROFILE_TARGET", "0") == "1"
    if _pt:
        torch.cuda.synchronize()
        _t0 = perf_counter()

    _vr_label = "draft_glue_replay" if model_runner.is_draft else "verify_replay"
    _ev_vr = mesa_record(_vr_label)
    graph.replay()
    mesa_close(_vr_label, _ev_vr)

    if _pt:
        torch.cuda.synchronize()
        _t1 = perf_counter()

    # Extract outputs for the ORIGINAL batch size only
    outputs = graph_vars["outputs"][:orig_bs * k_plus_1]
    logits = model_runner.model.compute_logits(outputs, last_only)

    if _pt:
        torch.cuda.synchronize()
        _t2 = perf_counter()
        has_eagle = "eagle_acts" in graph_vars
        print(f"[PROFILE verify_cg] replay={(_t1-_t0)*1000:.2f}ms logits={(_t2-_t1)*1000:.2f}ms eagle={has_eagle} bs={orig_bs} rank={model_runner.rank}", flush=True)

    # For eagle target, also return eagle_acts
    if "eagle_acts" in graph_vars:
        eagle_acts = graph_vars["eagle_acts"][:orig_bs * k_plus_1]
        return logits, eagle_acts
    return logits


@torch.inference_mode()
def run_decode_cudagraph(model_runner, input_ids, positions, last_only, graph_vars, hidden_states=None):
    context = get_context()

    flat_batch_size = input_ids.size(0)

    graph = model_runner.graphs["decode"][next(
        x for x in model_runner.graph_bs_list["decode"] if x >= flat_batch_size)]

    for k, v in graph_vars.items():
            if k != "outputs":
                v.zero_()

    graph_vars["input_ids"][:flat_batch_size] = input_ids
    graph_vars["positions"][:flat_batch_size] = positions
    graph_vars["slot_mapping"][:flat_batch_size] = context.slot_mapping
    graph_vars["context_lens"][:flat_batch_size] = context.context_lens

    if hidden_states is not None and "hidden_states" in graph_vars:
        graph_vars["hidden_states"][:flat_batch_size] = hidden_states

    if context.block_tables is not None:
        graph_vars["block_tables"][:flat_batch_size,
                                :context.block_tables.size(1)] = context.block_tables

    graph.replay()

    outputs = graph_vars["outputs"][:flat_batch_size]
    logits = model_runner.model.compute_logits(outputs, last_only)
    # EAGLE draft: outputs is prenorm, return both
    if "hidden_states" in graph_vars:
        return logits, outputs
    return logits


cache = {}

_plan_event = None  # Lazy-init CUDA event for plan() sync
PROFILE = os.environ.get("SSD_PROFILE", "0") == "1"
PROFILE_DRAFT = os.environ.get("SSD_PROFILE_DRAFT", "0") == "1"
_draft_events = []  # [(step, label, start_event, end_event), ...]

def flush_draft_profile():
    """Sync once, read all CUDA events, print per-step breakdown, clear list."""
    if not _draft_events:
        return
    torch.cuda.synchronize()
    by_step = {}
    for step, label, ev0, ev1 in _draft_events:
        by_step.setdefault(step, []).append((label, ev0.elapsed_time(ev1)))
    parts = []
    total = 0.0
    for step in sorted(by_step):
        step_total = sum(t for _, t in by_step[step])
        detail = " ".join(f"{l}={t:.2f}" for l, t in by_step[step])
        parts.append(f"s{step}={step_total:.2f}({detail})")
        total += step_total
    print(f"[PROFILE draft_detail] K={len(by_step)} total={total:.2f}ms avg_step={total/len(by_step):.2f}ms | {' '.join(parts)}", flush=True)
    _draft_events.clear()

@torch.inference_mode()
def run_fi_tree_decode_cudagraph(model_runner, input_ids, positions, last_only, graph_vars, step, cache_hits, hidden_states=None, layout=None):
    # bs != len(input_ids, positions) now in multi-query seting, also need step-dependent mask
    _prep_label = ("phase1_prep" if (layout is not None and (
                       layout.name == "draft"
                       or (layout.name is not None and layout.name.startswith("phase1_"))
                       or (layout.name is not None and layout.name.startswith("split_k1"))))
                   else "phase2_prep" if (layout is not None and (
                       layout.name == "proxy" or layout.name == "split_k2"))
                   else "tree_prep")
    _mev_prep = mesa_record(_prep_label)
    context = get_context()
    assert context.cu_seqlens_q is None, "ERROR in run_fi_tree_decode_cudagraph: cu_seqlens_q should be set to None so we don't take FA path"

    K, F = model_runner.config.speculate_k, model_runner.config.async_fan_out
    # Layout-aware MQ_LEN and fan_out_list resolution.
    # Step 9A: K_for_mask = position_count - 1 (= glue width). For
    # phase1_long this is K_long; for phase1_short this is K_short. The
    # outer K (config.speculate_k) is used for the mask cache loop count,
    # but the per-step glue/diag dimensions follow the layout.
    if layout is not None:
        MQ_LEN = layout.MQ_LEN
        _graph_key = layout.graph_key
        _fan_out_list = layout.fan_out_list
        _fan_out_list_miss = layout.fan_out_list_miss
        K_for_mask = layout.position_count - 1
        # Step-0 precompute loop count = forward_depth (layout.K), not speculate_k.
        # Split-K1/K2 mode: Phase 1 forwards K1 times, Phase 2 forwards K2 times.
        # Prior code used speculate_k = K1+K2 → wasted (K_long-K1) or (K_long-K2)
        # iterations of kv-meta + mask-build + GPU transfers per step-0.
        K_loop = layout.K
    else:
        MQ_LEN = sum(model_runner.config.fan_out_list)
        _graph_key = "fi_tree_decode"
        _fan_out_list = model_runner.config.fan_out_list
        _fan_out_list_miss = model_runner.config.fan_out_list_miss
        K_for_mask = K
        K_loop = K
    orig_flat = input_ids.size(0)
    assert orig_flat % MQ_LEN == 0, f"ERROR in run_fi_tree_decode_cudagraph: flat_batch_size should be divisible by MQ_LEN, got {orig_flat} and {MQ_LEN}"
    orig_B = orig_flat // MQ_LEN

    # Pick CUDA graph and wrapper bucket (layout-aware key)
    wrapper_bs = next(
        x for x in model_runner.graph_bs_list[_graph_key] if x >= orig_B)
    graph = model_runner.graphs[_graph_key][wrapper_bs]
    # Layout-aware wrapper: use layout-specific wrappers if available
    _wrappers = model_runner.prefill_wrappers
    if layout is not None and hasattr(model_runner, 'prefill_wrappers_by_layout'):
        _wrappers = model_runner.prefill_wrappers_by_layout.get(layout.name, _wrappers)
    wrapper = _wrappers[wrapper_bs]

    # Prepare padded inputs/context if needed
    if wrapper_bs > orig_B:
        # print(f'PADDING--')
        pad_B = wrapper_bs - orig_B
        pad_flat = pad_B * MQ_LEN

        # Pad queries (ids/rope positions)
        pad_ids = torch.zeros(
            pad_flat, dtype=input_ids.dtype, device=input_ids.device)
        pad_pos = torch.zeros(
            pad_flat, dtype=positions.dtype, device=positions.device)
        input_ids = torch.cat([input_ids, pad_ids], dim=0)
        positions = torch.cat([positions, pad_pos], dim=0)

        # Pad slot_mapping with -1 to skip KV writes for padded queries
        slot_map = torch.cat(
            [context.slot_mapping,
             torch.full((pad_flat,), -1, dtype=context.slot_mapping.dtype, device=context.slot_mapping.device)]
        )

        # Pad block_tables/context_lens by repeating the last real row
        bt = context.block_tables
        cl = context.context_lens
        pad_bt = bt[orig_B - 1:orig_B].expand(pad_B, -1).contiguous()
        pad_cl = cl[orig_B - 1:orig_B].expand(pad_B).contiguous()
        bt = torch.cat([bt, pad_bt], dim=0)
        cl = torch.cat([cl, pad_cl], dim=0)

        # Set padded context for this replay
        set_context(is_prefill=False, slot_mapping=slot_map,
                    context_lens=cl, block_tables=bt)

        block_tables = bt
        context_lens = cl
        flat_batch_size = input_ids.size(0)  # == wrapper_bs * MQ_LEN
        B = wrapper_bs
    else:
        block_tables = context.block_tables
        context_lens = context.context_lens
        flat_batch_size = orig_flat
        B = orig_B

    if PROFILE:
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()

    # in the case where we pad, we'll need cache_hits.shape[0] to match the padded batch size
    if cache_hits.shape[0] < B:
        cache_hits = torch.cat([cache_hits, torch.zeros(B - cache_hits.shape[0], device=cache_hits.device)])

    # PERFORMANCE: Step 0 -- precompute KV page metadata on CPU for all K steps.
    # CPU tensors let plan() skip its internal .to("cpu") GPU->CPU syncs.
    # For B<=8, CPU slicing also avoids GPU boolean indexing.
    if step == 0:
        # Layout change: clear cache if MQ_LEN changed (2-pass MESA reuses global cache)
        if cache.get("_mq_len") != MQ_LEN:
            cache.clear()
            cache["_mq_len"] = MQ_LEN
        cache["cu_seqlens_q_cpu"] = torch.arange(B + 1, dtype=torch.int32) * MQ_LEN
        context_lens_list = context_lens.tolist()   # GPU sync
        cache["block_tables"] = block_tables
        block_size = model_runner.block_size
        cache["precomputed_kv"] = []
        cache["plan_cpu_args"] = []

        if B <= 8:
            # PERFORMANCE: CPU-only kv_indices via slicing (no GPU boolean indexing)
            for s in range(K_loop):
                step_cls = [int(cl) + s * MQ_LEN for cl in context_lens_list]
                step_counts = [(cl + block_size - 1) // block_size for cl in step_cls]
                if B == 1:
                    kv_indices_s = block_tables[0, :step_counts[0]]
                else:
                    kv_indices_s = torch.cat([block_tables[b, :step_counts[b]] for b in range(B)])
                cache["precomputed_kv"].append(kv_indices_s)
                kv_indptr_cpu = torch.zeros(B + 1, dtype=torch.int32)
                kv_indptr_cpu[1:] = torch.tensor(step_counts, dtype=torch.int32).cumsum(0)
                kv_lpl_cpu = torch.tensor(
                    [cl % block_size if cl % block_size != 0 else block_size for cl in step_cls],
                    dtype=torch.int32)
                cache["plan_cpu_args"].append((kv_indptr_cpu, kv_lpl_cpu))
        else:
            # Large batch: GPU boolean indexing for kv_indices, CPU tensors for plan args
            bt_upcast = torch.arange(block_tables.size(1), device=block_tables.device)[None, :]
            step_offsets = torch.arange(K_loop + 2, device=context_lens.device) * MQ_LEN
            all_step_cls = context_lens.unsqueeze(1) + step_offsets.unsqueeze(0)
            all_counts = (all_step_cls + block_size - 1) // block_size
            all_masks = bt_upcast.unsqueeze(1) < all_counts.unsqueeze(2)
            for s in range(K_loop):
                cache["precomputed_kv"].append(block_tables[all_masks[:, s, :]])
                step_cls = [int(cl) + s * MQ_LEN for cl in context_lens_list]
                step_counts = [(cl + block_size - 1) // block_size for cl in step_cls]
                kv_indptr_cpu = torch.zeros(B + 1, dtype=torch.int32)
                kv_indptr_cpu[1:] = torch.tensor(step_counts, dtype=torch.int32).cumsum(0)
                kv_lpl_cpu = torch.tensor(
                    [cl % block_size if cl % block_size != 0 else block_size for cl in step_cls],
                    dtype=torch.int32)
                cache["plan_cpu_args"].append((kv_indptr_cpu, kv_lpl_cpu))

        # CPU mask precompute: build all K packed masks using numpy at step 0.
        # Eliminates per-step get_custom_mask (GPU) + segment_packbits + GPU->CPU syncs.
        cache_hits_list = cache_hits[:B].tolist()

        # Layout change detection: if fan_out changed, recompute glue masks
        _needs_recompute = "glue_hit_np" not in cache
        if not _needs_recompute and cache.get("_cached_fol") != _fan_out_list:
            _needs_recompute = True
        if _needs_recompute:
            cache["_cached_fol"] = _fan_out_list
            _fol = _fan_out_list
            _fol_miss = _fan_out_list_miss
            # Step 9A: glue tril dim follows the layout's position_count
            # (= K_for_mask+1). For phase1_short bucket K_for_mask=K_short<K_long.
            _tril = np.tril(np.ones((K_for_mask + 1, K_for_mask + 1), dtype=np.uint8))
            cache["glue_hit_np"] = np.repeat(_tril, _fol, axis=0)
            cache["glue_miss_np"] = np.repeat(_tril, _fol_miss, axis=0)

        _glue_hit = cache["glue_hit_np"]
        _glue_miss = cache["glue_miss_np"]
        _rows_np = np.arange(MQ_LEN)

        cache["cpu_packed_masks"] = []
        cache["cpu_packed_indptrs"] = []

        # ─────────────────────────────────────────────────────────────────
        # KNOWN BUG (separate track from hybrid landing).
        #
        # This generic mask formula assumes the layout
        #     [persistent | glue (K_long+1) | diag blocks of MQ_LEN]
        # which holds only for single-pass tree decode where K = layout.K
        # and there is no prior spec scratch already written.
        #
        # Under MESA-SSD's K1-split (mesa_phase1_k < speculate_k), the
        # **continuation pass** runs separately AFTER Phase 1 has already
        # written K1*MQ_LEN slots of Phase 1 KV. The formula treats those
        # K1*MQ_LEN slots as fully-visible "persistent prefix" and shifts
        # the (K_long+1)-wide glue lower-tri region forward by K1*MQ_LEN —
        # which physically lands inside the LAST K_long+1 slots of Phase 1's
        # last depth (not the actual glue at all). Effect: cont rows attend
        # to ALL Phase 1 KV across branches and ALL glue without j_idx
        # filtering. See `_compute_hybrid_bool_mask_for_depth` for the
        # plan-correct continuation mask and `_decode_correct_split_cont`
        # for an oracle that confirms split CG cont diverges by ~28-31%
        # tokens vs plan-correct semantics.
        #
        # NOT FIXED HERE because split is retained as a fallback/reference
        # path while hybrid-default lands. This is tracked separately from
        # the hybrid path.
        # ─────────────────────────────────────────────────────────────────
        for s in range(K_loop):
            # Step 9A: ttl_added_s uses K_for_mask (= layout glue width)
            # not the outer K (config.speculate_k). For phase1_short the
            # glue width is K_short+1; for phase1_long / non-MESA it's
            # K_long+1.
            ttl_added_s = (s + 1) * MQ_LEN + (K_for_mask + 1)
            packed_segs = []
            seg_packed_sizes = []

            for b in range(B):
                cols_b = int(context_lens_list[b]) + s * MQ_LEN
                prefix_len_b = cols_b - ttl_added_s

                mask_b = np.zeros((MQ_LEN, cols_b), dtype=np.uint8)
                mask_b[:, :prefix_len_b] = 1
                glue = _glue_hit if int(cache_hits_list[b]) == 1 else _glue_miss
                mask_b[:, prefix_len_b:prefix_len_b + K_for_mask + 1] = glue
                diag_start = prefix_len_b + K_for_mask + 1
                for blk in range(s + 1):
                    mask_b[_rows_np, diag_start + blk * MQ_LEN + _rows_np] = 1

                packed = np.packbits(mask_b.ravel(), bitorder='little')
                packed_segs.append(packed)
                seg_packed_sizes.append(len(packed))

            full_packed = np.concatenate(packed_segs) if B > 1 else packed_segs[0]
            indptr = np.zeros(B + 1, dtype=np.int32)
            indptr[1:] = np.cumsum(seg_packed_sizes)

            cache["cpu_packed_masks"].append(
                torch.from_numpy(full_packed.copy()).to(model_runner.device, non_blocking=True))
            cache["cpu_packed_indptrs"].append(
                torch.from_numpy(indptr.copy()).to(model_runner.device, non_blocking=True))

        # Pre-transfer KV metadata to GPU (eliminates per-step pageable H2D transfers)
        cache["qo_indptr_gpu"] = cache["cu_seqlens_q_cpu"].to(model_runner.device, non_blocking=True)
        cache["kv_indptr_gpu"] = []
        cache["kv_lpl_gpu"] = []
        cache["kv_lens_gpu"] = []
        cache["kv_lens_cpu"] = []   # for plan() (avoid GPU dependency)
        for s in range(K_loop):
            ki, kl = cache["plan_cpu_args"][s]
            cache["kv_indptr_gpu"].append(ki.to(model_runner.device, non_blocking=True))
            cache["kv_lpl_gpu"].append(kl.to(model_runner.device, non_blocking=True))
            kv_lens_cpu = ((ki[1:] - ki[:-1] - 1) * model_runner.block_size + kl).to(torch.int32)
            cache["kv_lens_cpu"].append(kv_lens_cpu)
            cache["kv_lens_gpu"].append(kv_lens_cpu.to(model_runner.device, non_blocking=True))

    if PROFILE:
        end_time.record()
        torch.cuda.synchronize()
        precompute_time = start_time.elapsed_time(end_time)
        start_time.record()

    # Use precomputed CPU-packed masks (built at step 0)
    if PROFILE_DRAFT:
        _ev_mask0 = torch.cuda.Event(enable_timing=True); _ev_mask0.record()

    kv_indices = cache["precomputed_kv"][step]
    kv_indptr_cpu, kv_lpl_cpu = cache["plan_cpu_args"][step]
    qo_indptr_cpu = cache["cu_seqlens_q_cpu"]

    packed_mask = cache["cpu_packed_masks"][step]
    packed_indptr = cache["cpu_packed_indptrs"][step]

    # Phase C-2: optional perturbation of packed_mask buffer at runtime to
    # test if captured CG kernel actually reads it. Gated by env var:
    #   SSD_CG_MASK_PERTURB=ones      → all 0xFF (= every bit visible)
    #   SSD_CG_MASK_PERTURB=zeros     → all 0x00 (= no bit visible)
    # If output is INSENSITIVE to perturbation, captured kernel doesn't read
    # this buffer (or reads from a different place); strong evidence that the
    # captured CG has attention metadata baked in at capture time.
    _perturb = os.environ.get("SSD_CG_MASK_PERTURB", "")
    if _perturb == "ones":
        packed_mask = torch.full_like(packed_mask, 0xFF)
    elif _perturb == "zeros":
        packed_mask = torch.zeros_like(packed_mask)

    wrapper._custom_mask_buf[:len(packed_mask)].copy_(packed_mask, non_blocking=True)
    wrapper._mask_indptr_buf.copy_(packed_indptr, non_blocking=True)

    # GPU-to-GPU copies from pre-transferred tensors (no pageable H2D)
    wrapper._qo_indptr_buf.copy_(cache["qo_indptr_gpu"], non_blocking=True)
    wrapper._paged_kv_indptr_buf.copy_(cache["kv_indptr_gpu"][step], non_blocking=True)
    wrapper._paged_kv_last_page_len_buf.copy_(cache["kv_lpl_gpu"][step], non_blocking=True)
    wrapper._paged_kv_indices_buf[:len(kv_indices)].copy_(kv_indices, non_blocking=True)

    total_num_rows = int(qo_indptr_cpu[-1].item())
    wrapper._kv_lens_buffer[:len(kv_indptr_cpu) - 1].copy_(cache["kv_lens_gpu"][step], non_blocking=True)

    # Event-based sync: only wait for this stream's copies, not all CUDA streams.
    global _plan_event
    if _plan_event is None:
        _plan_event = torch.cuda.Event()
    _plan_event.record()
    _plan_event.synchronize()

    if PROFILE_DRAFT:
        _ev_plan0 = torch.cuda.Event(enable_timing=True); _ev_plan0.record()

    plan_args = [
        wrapper._float_workspace_buffer, wrapper._int_workspace_buffer,
        wrapper._pin_memory_int_workspace_buffer,
        # kv_lens passed as CPU tensor (FlashInfer's standard wrapper.plan()
        # contract is kv_lens_arr_host on CPU).
        qo_indptr_cpu, kv_indptr_cpu, cache["kv_lens_cpu"][step],
        wrapper._max_total_num_rows or total_num_rows,
        B, model_runner.hf_config.num_attention_heads,
        model_runner.hf_config.num_key_value_heads,
        model_runner.block_size, wrapper.is_cuda_graph_enabled,
        model_runner.hf_config.head_dim, model_runner.hf_config.head_dim,
        False, -1,
    ]
    if wrapper._backend == "fa2":
        plan_args.extend([-1, False])
    wrapper._plan_info = wrapper._cached_module.plan(*plan_args)

    if PROFILE_DRAFT:
        _ev_plan1 = torch.cuda.Event(enable_timing=True); _ev_plan1.record()

    if PROFILE:
        end_time.record()
        torch.cuda.synchronize()
        plan_time = start_time.elapsed_time(end_time)
        start_time.record()

    # Copy inputs/context into graph buffers for padded size
    graph_vars["input_ids"][:flat_batch_size] = input_ids
    graph_vars["positions"][:flat_batch_size] = positions
    graph_vars["slot_mapping"][:flat_batch_size] = get_context().slot_mapping
    graph_vars["context_lens"][:B] = context_lens
    if hidden_states is not None and "hidden_states" in graph_vars:
        if hidden_states.shape[0] < flat_batch_size:
            # Pad hidden_states to match padded batch
            pad_n = flat_batch_size - hidden_states.shape[0]
            hidden_states = torch.cat([hidden_states, torch.zeros(pad_n, hidden_states.shape[1], dtype=hidden_states.dtype, device=hidden_states.device)])
        graph_vars["hidden_states"][:flat_batch_size] = hidden_states
    if step == 0:
        graph_vars["block_tables"][:B, :block_tables.size(1)] = block_tables

    if PROFILE:
        end_time.record()
        torch.cuda.synchronize()
        buffer_prep_time = start_time.elapsed_time(end_time)
        start_time.record()

    if PROFILE_DRAFT:
        _ev_replay0 = torch.cuda.Event(enable_timing=True); _ev_replay0.record()

    _layout_name = layout.name if layout is not None else None
    if _layout_name == "draft" or (_layout_name is not None and (
        _layout_name.startswith("phase1_") or _layout_name.startswith("split_k1")
    )):
        _mesa_label = "phase1_replay"
    elif _layout_name == "proxy" or _layout_name == "split_k2":
        _mesa_label = "phase2_replay"
    else:
        _mesa_label = "tree_replay"
    mesa_close(_prep_label, _mev_prep)
    _mev = mesa_record(_mesa_label)
    graph.replay()
    mesa_close(_mesa_label, _mev)

    if PROFILE_DRAFT:
        _ev_replay1 = torch.cuda.Event(enable_timing=True); _ev_replay1.record()
        _draft_events.append((step, "mask+buf", _ev_mask0, _ev_plan0))
        _draft_events.append((step, "plan", _ev_plan0, _ev_plan1))
        _draft_events.append((step, "replay", _ev_replay0, _ev_replay1))

    if PROFILE:
        end_time.record()
        torch.cuda.synchronize()
        replay_time = start_time.elapsed_time(end_time)

    # Extract logits from graph_vars instead of computing them separately
    logits_all = graph_vars["logits"][:flat_batch_size]

    if PROFILE:
        print(f"[run_fi_tree_decode_cudagraph] step {step}: precompute={precompute_time:.3f}ms, plan={plan_time:.3f}ms, buffer={buffer_prep_time:.3f}ms, replay={replay_time:.3f}ms", flush=True)

    logits_out = logits_all[:orig_flat]
    # EAGLE draft: also return prenorm (outputs) for self-conditioning
    if "hidden_states" in graph_vars:
        prenorm = graph_vars["outputs"][:orig_flat]
        return logits_out, prenorm
    return logits_out


## CAPTURE CUDAGRAPHS
@torch.inference_mode()
def capture_cudagraph(model_runner):
    config = model_runner.config
    hf_config = config.hf_config
    max_seqs = min(model_runner.config.max_num_seqs, 512)
    if model_runner.config.speculate and model_runner.config.draft_async and model_runner.is_draft:
        N = max_seqs * (model_runner.config.speculate_k + 1) * \
            model_runner.config.async_fan_out
        max_bs = N * (model_runner.config.speculate_k + 1)
    else:
        max_bs = max_seqs + 1
    max_num_blocks = (config.max_model_len +
                      model_runner.block_size - 1) // model_runner.block_size
    input_ids = torch.zeros(max_bs, dtype=torch.int64)
    positions = torch.zeros(max_bs, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs, hf_config.hidden_size)

    if model_runner.config.speculate and model_runner.config.draft_async and model_runner.is_draft:
        # Power-of-two buckets: max_bs = max_seqs*(K+1)^2*F can be huge (e.g. 9600 for k=9,f=3,B=32),
        # linear step-16 would create ~600 graphs. Power-of-two gives ~15.
        N = max_seqs * (model_runner.config.speculate_k + 1) * model_runner.config.async_fan_out
        graph_bs_list = []
        bs = 1
        while bs < max_bs:
            graph_bs_list.append(bs)
            bs *= 2
        if max_bs not in graph_bs_list:
            graph_bs_list.append(max_bs)
        # Ensure N (tree decode batch size) is a bucket for exact-fit replay
        if N not in graph_bs_list:
            graph_bs_list.append(N)
            graph_bs_list.sort()
    else:
        graph_bs_list = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        if max_bs % 16 != 0:
            graph_bs_list.append(max_bs)

    graphs = {}
    graph_pool = None

    is_jit = (model_runner.config.speculate and model_runner.config.draft_async and model_runner.is_draft)

    # Eagle models need special handling during CUDA graph capture
    is_eagle_draft = config.use_eagle and model_runner.is_draft
    is_eagle_target = config.use_eagle and not model_runner.is_draft
    hidden_states = None
    if is_eagle_draft:
        # Use hidden_size (d_model_draft) so CG captures the pass-through branch in Eagle3DraftForCausalLM.forward()
        # All callers project target acts via fc() BEFORE passing to CG
        hidden_states = torch.zeros(max_bs, hf_config.hidden_size,
                                    dtype=hf_config.torch_dtype, device=input_ids.device)

    for bs in reversed(graph_bs_list):
        graph = torch.cuda.CUDAGraph()
        set_context(
            False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs], is_jit=is_jit)
        if is_eagle_draft:
            outputs[:bs] = model_runner.model(
                input_ids[:bs], positions[:bs], hidden_states[:bs])    # warmup
        elif is_eagle_target:
            out, _ = model_runner.model(
                input_ids[:bs], positions[:bs])    # warmup
            outputs[:bs] = out
        else:
            outputs[:bs] = model_runner.model(
                input_ids[:bs], positions[:bs])    # warmup
        with torch.cuda.graph(graph, graph_pool):
            if is_eagle_draft:
                outputs[:bs] = model_runner.model(
                    input_ids[:bs], positions[:bs], hidden_states[:bs])    # capture
            elif is_eagle_target:
                out, _ = model_runner.model(
                    input_ids[:bs], positions[:bs])    # capture
                outputs[:bs] = out
            else:
                outputs[:bs] = model_runner.model(
                    input_ids[:bs], positions[:bs])    # capture
        if graph_pool is None:
            graph_pool = graph.pool()
        graphs[bs] = graph
        torch.cuda.synchronize()
        reset_context()

    graph_vars = dict(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        outputs=outputs,
    )
    if hidden_states is not None:
        graph_vars["hidden_states"] = hidden_states

    return graph_vars, graph_pool, graphs, graph_bs_list


@torch.inference_mode()
def capture_verify_cudagraph(model_runner, k_plus_1=None):
    """Capture multi-query (K+1) decode CG.

    Step 9B-0: ``k_plus_1`` (default = config.speculate_k+1 = K_long+1)
    drives query width. For draft glue short-bucket dispatch, call with
    K_short+1 to get a smaller CG family.
    """
    config = model_runner.config
    # assert not model_runner.is_draft, "ERROR in capture_verify_cudagraph: verify path only supported for target model"
    hf_config = config.hf_config
    max_bs = min(model_runner.config.max_num_seqs, 512)
    if k_plus_1 is None:
        k_plus_1 = model_runner.config.speculate_k + 1

    is_eagle_target = config.use_eagle and not model_runner.is_draft

    # For verify, we need to handle k+1 tokens per sequence, and use cu_seqlens_q and max_seqlen_q
    input_ids = torch.zeros(max_bs * k_plus_1, dtype=torch.int64)
    positions = torch.zeros(max_bs * k_plus_1, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs * k_plus_1, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(
        max_bs, model_runner.max_num_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs * k_plus_1, hf_config.hidden_size)
    cu_seqlens_q = torch.zeros(max_bs + 1, dtype=torch.int32)

    # Eagle target: also capture eagle_acts from model forward
    eagle_acts = None
    if is_eagle_target:
        # eagle_acts has shape [num_tokens, 3 * hidden_size] for 3 layers
        eagle_acts = torch.zeros(max_bs * k_plus_1, 3 * hf_config.hidden_size,
                                  dtype=hf_config.torch_dtype)

    base = [1, 2, 4, 8]
    dynamic = list(range(16, max_bs+1, 16))
    all_b = base + dynamic
    if max_bs not in all_b:
        all_b.append(max_bs)
    all_b.sort()
    all_N = [b for b in all_b if b <= max_bs]

    graphs = {}
    graph_pool = None

    for bs in reversed(all_N):
        graph = torch.cuda.CUDAGraph()
        # For verify, each sequence is length K+1, so seqlen_q is [K+1]*bs
        seqlen_q = torch.full((bs,), k_plus_1, dtype=torch.int32)
        cu = cu_seqlens_q[:bs + 1]
        cu.zero_()
        cu[1:].copy_(torch.cumsum(seqlen_q, 0))
        context_lens[:bs] = seqlen_q

        set_context(
            is_prefill=False,
            slot_mapping=slot_mapping[:bs * k_plus_1],
            context_lens=context_lens[:bs],
            block_tables=block_tables[:bs],
            cu_seqlens_q=cu,
            max_seqlen_q=k_plus_1,
        )

        # warmup
        model_out = model_runner.model(
            input_ids[:bs * k_plus_1], positions[:bs * k_plus_1])
        if isinstance(model_out, tuple):
            outputs[:bs * k_plus_1] = model_out[0]
            if eagle_acts is not None:
                eagle_acts[:bs * k_plus_1] = model_out[1]
        else:
            outputs[:bs * k_plus_1] = model_out
        with torch.cuda.graph(graph, graph_pool):
            # capture
            model_out = model_runner.model(
                input_ids[:bs * k_plus_1], positions[:bs * k_plus_1])
            if isinstance(model_out, tuple):
                outputs[:bs * k_plus_1] = model_out[0]
                if eagle_acts is not None:
                    eagle_acts[:bs * k_plus_1] = model_out[1]
            else:
                outputs[:bs * k_plus_1] = model_out

        if graph_pool is None:
            graph_pool = graph.pool()
        graphs[bs] = graph
        torch.cuda.synchronize()
        reset_context()

    graph_vars = dict(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        cu_seqlens_q=cu_seqlens_q,
        outputs=outputs,
    )
    if eagle_acts is not None:
        graph_vars["eagle_acts"] = eagle_acts

    return graph_vars, graph_pool, graphs, all_N


@torch.inference_mode()
def run_glue_decode_cudagraph(model_runner, input_ids, positions, last_only, graph_vars, hidden_states=None):
    """Run EAGLE glue decode with FA causal + varlen cu_seqlens_q. No padding within sequences."""
    context = get_context()
    K = model_runner.config.speculate_k
    two_kp1 = 2 * K + 1
    orig_flat = input_ids.size(0)
    orig_B = context.context_lens.size(0)
    dev = input_ids.device

    wrapper_bs = next(
        x for x in model_runner.graph_bs_list["glue_decode"] if x >= orig_B)
    graph = model_runner.graphs["glue_decode"][wrapper_bs]
    max_flat = wrapper_bs * two_kp1

    # Zero all non-output graph vars
    for k, v in graph_vars.items():
        if k != "outputs":
            v.zero_()

    # Copy real data into graph buffers (orig_flat <= max_flat always)
    graph_vars["input_ids"][:orig_flat] = input_ids
    graph_vars["positions"][:orig_flat] = positions
    graph_vars["slot_mapping"][:orig_flat] = context.slot_mapping
    # Pad remaining flat slots with -1 slot_mapping (no KV write)
    if orig_flat < max_flat:
        graph_vars["slot_mapping"][orig_flat:max_flat] = -1

    graph_vars["context_lens"][:orig_B] = context.context_lens
    graph_vars["block_tables"][:orig_B, :context.block_tables.size(1)] = context.block_tables

    # cu_seqlens_q: real seqs, then ghost seqs (repeat last cumsum = 0-length queries)
    cu = context.cu_seqlens_q  # [orig_B + 1]
    graph_vars["cu_seqlens_q"][:orig_B + 1] = cu
    if wrapper_bs > orig_B:
        # Ghost seqs get 0-length queries
        graph_vars["cu_seqlens_q"][orig_B + 1:wrapper_bs + 1] = cu[-1]
        # Ghost seqs need valid block_tables/context_lens (copy last real seq)
        pad_B = wrapper_bs - orig_B
        graph_vars["context_lens"][orig_B:wrapper_bs] = context.context_lens[orig_B - 1]
        graph_vars["block_tables"][orig_B:wrapper_bs] = context.block_tables[orig_B - 1]

    if hidden_states is not None and "eagle_hidden_states" in graph_vars:
        graph_vars["eagle_hidden_states"][:orig_flat] = hidden_states

    graph.replay()

    outputs = graph_vars["outputs"][:orig_flat]
    logits = model_runner.model.compute_logits(outputs, last_only)
    if "eagle_hidden_states" in graph_vars:
        return logits, outputs
    return logits


@torch.inference_mode()
def capture_glue_decode_cudagraph(model_runner):
    """Capture CG for EAGLE glue decode: FA causal + varlen cu_seqlens_q, max flat = B*(2K+1)."""
    config = model_runner.config
    hf_config = config.hf_config
    max_bs = min(config.max_num_seqs, 512)
    K = config.speculate_k
    two_kp1 = 2 * K + 1
    max_flat = max_bs * two_kp1
    max_num_blocks = (config.max_model_len + model_runner.block_size - 1) // model_runner.block_size

    input_ids = torch.zeros(max_flat, dtype=torch.int64, device=model_runner.device)
    positions = torch.zeros(max_flat, dtype=torch.int64, device=model_runner.device)
    slot_mapping = torch.zeros(max_flat, dtype=torch.int32, device=model_runner.device)
    context_lens = torch.full((max_bs,), config.max_model_len, dtype=torch.int32, device=model_runner.device)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=model_runner.device)
    outputs = torch.empty(max_flat, hf_config.hidden_size, device=model_runner.device)
    cu_seqlens_q = torch.zeros(max_bs + 1, dtype=torch.int32, device=model_runner.device)

    eagle_hs = None
    if config.use_eagle and model_runner.is_draft:
        eagle_hs = torch.zeros(max_flat, hf_config.hidden_size, dtype=hf_config.torch_dtype, device=model_runner.device)

    graph_bs_list = [1]
    for bs in [2, 4, 8] + list(range(16, max_bs + 1, 16)):
        if bs <= max_bs:
            graph_bs_list.append(bs)
    if max_bs not in graph_bs_list:
        graph_bs_list.append(max_bs)
    graph_bs_list.sort()

    graphs = {}
    graph_pool = None

    print(f'[capture_glue_decode_cudagraph] Capturing for bs={graph_bs_list}', flush=True)

    for bs in reversed(graph_bs_list):
        graph = torch.cuda.CUDAGraph()
        flat = bs * two_kp1

        # Uniform cu_seqlens_q for capture (each seq gets 2K+1 queries)
        seqlen_q = torch.full((bs,), two_kp1, dtype=torch.int32, device=model_runner.device)
        cu = cu_seqlens_q[:bs + 1]
        cu.zero_()
        cu[1:].copy_(torch.cumsum(seqlen_q, 0))

        set_context(
            is_prefill=False,
            cu_seqlens_q=cu,
            max_seqlen_q=two_kp1,
            slot_mapping=slot_mapping[:flat],
            context_lens=context_lens[:bs],
            block_tables=block_tables[:bs],
        )

        if eagle_hs is not None:
            outputs[:flat] = model_runner.model(input_ids[:flat], positions[:flat], eagle_hs[:flat])
        else:
            outputs[:flat] = model_runner.model(input_ids[:flat], positions[:flat])

        with torch.cuda.graph(graph, graph_pool):
            if eagle_hs is not None:
                outputs[:flat] = model_runner.model(input_ids[:flat], positions[:flat], eagle_hs[:flat])
            else:
                outputs[:flat] = model_runner.model(input_ids[:flat], positions[:flat])

        if graph_pool is None:
            graph_pool = graph.pool()
        graphs[bs] = graph
        torch.cuda.synchronize()
        reset_context()

    graph_vars = dict(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        cu_seqlens_q=cu_seqlens_q,
        outputs=outputs,
    )
    if eagle_hs is not None:
        graph_vars["eagle_hidden_states"] = eagle_hs

    return graph_vars, graph_pool, graphs, graph_bs_list


@torch.inference_mode()
def low_level_packed_plan(wrapper, *, model_runner, qo_indptr_cpu, kv_indptr_cpu,
                            kv_indices, kv_last_page_len_gpu, kv_lens_gpu,
                            packed_mask, packed_indptr, B):
    """Phase B-1 helper: write FlashInfer wrapper internal buffers + call
    low-level `_cached_module.plan(...)`.

    Mirrors the exact buffer-write order used by run_fi_tree_decode_cudagraph
    (line 369-406) so that split CG path and any caller using this helper
    produce identical wrapper plan state.

    Args:
        wrapper: FlashInfer BatchPrefillWithPagedKVCacheWrapper instance with
            internal buffers (_custom_mask_buf, _mask_indptr_buf, etc.).
        model_runner: source of head dims / block_size / hf_config.
        qo_indptr_cpu: int32 CPU tensor [B+1].
        kv_indptr_cpu: int32 CPU tensor [B+1].
        kv_indices: int32 GPU tensor [n_indices_total].
        kv_last_page_len_gpu: int32 GPU tensor [B].
        kv_lens_gpu: int32 GPU tensor [B] = (num_pages-1)*block_size + last_page_len.
        packed_mask: uint8 GPU tensor (numpy.packbits little-endian).
        packed_indptr: int32 GPU tensor [B+1] (per-batch packed_mask byte offsets).
        B: batch size in wrapper terms.
    """
    # 1. Mask buffers
    wrapper._custom_mask_buf[:len(packed_mask)].copy_(packed_mask, non_blocking=True)
    wrapper._mask_indptr_buf[:len(packed_indptr)].copy_(packed_indptr, non_blocking=True)

    # 2. KV / qo metadata buffers (qo_indptr/kv_indptr GPU sources)
    qo_indptr_gpu = qo_indptr_cpu.to(wrapper._qo_indptr_buf.device, non_blocking=True)
    kv_indptr_gpu = kv_indptr_cpu.to(wrapper._paged_kv_indptr_buf.device, non_blocking=True)
    wrapper._qo_indptr_buf[:len(qo_indptr_gpu)].copy_(qo_indptr_gpu, non_blocking=True)
    wrapper._paged_kv_indptr_buf[:len(kv_indptr_gpu)].copy_(kv_indptr_gpu, non_blocking=True)
    wrapper._paged_kv_last_page_len_buf[:len(kv_last_page_len_gpu)].copy_(kv_last_page_len_gpu, non_blocking=True)
    wrapper._paged_kv_indices_buf[:len(kv_indices)].copy_(kv_indices, non_blocking=True)

    total_num_rows = int(qo_indptr_cpu[-1].item())
    wrapper._kv_lens_buffer[:len(kv_lens_gpu)].copy_(kv_lens_gpu, non_blocking=True)

    # Sync event
    ev = torch.cuda.Event()
    ev.record()
    ev.synchronize()

    # Low-level plan call
    plan_args = [
        wrapper._float_workspace_buffer, wrapper._int_workspace_buffer,
        wrapper._pin_memory_int_workspace_buffer,
        qo_indptr_cpu, kv_indptr_cpu, kv_lens_gpu,
        wrapper._max_total_num_rows or total_num_rows,
        B, model_runner.hf_config.num_attention_heads,
        model_runner.hf_config.num_key_value_heads,
        model_runner.block_size, wrapper.is_cuda_graph_enabled,
        model_runner.hf_config.head_dim, model_runner.hf_config.head_dim,
        False, -1,
    ]
    if wrapper._backend == "fa2":
        plan_args.extend([-1, False])
    wrapper._plan_info = wrapper._cached_module.plan(*plan_args)


def build_packed_mask_for_proxy_step(*, MQ_LEN, K, B, context_lens_list,
                                       cache_hits_list, fan_out_list,
                                       fan_out_list_miss, step, device):
    """Build packed mask + indptr for a single step using the SAME numpy
    construction as run_fi_tree_decode_cudagraph step 0 (lines 313-340).

    Used by Phase B-2 proxy-first low-level mirror for parity.
    """
    import numpy as np
    _tril = np.tril(np.ones((K + 1, K + 1), dtype=np.uint8))
    _glue_hit = np.repeat(_tril, fan_out_list, axis=0)
    _glue_miss = np.repeat(_tril, fan_out_list_miss, axis=0)
    _rows_np = np.arange(MQ_LEN)

    s = step
    ttl_added_s = (s + 1) * MQ_LEN + (K + 1)
    packed_segs = []
    seg_packed_sizes = []
    for b in range(B):
        cols_b = int(context_lens_list[b]) + s * MQ_LEN
        prefix_len_b = cols_b - ttl_added_s
        mask_b = np.zeros((MQ_LEN, cols_b), dtype=np.uint8)
        mask_b[:, :prefix_len_b] = 1
        glue = _glue_hit if int(cache_hits_list[b]) == 1 else _glue_miss
        mask_b[:, prefix_len_b:prefix_len_b + K + 1] = glue
        diag_start = prefix_len_b + K + 1
        for blk in range(s + 1):
            mask_b[_rows_np, diag_start + blk * MQ_LEN + _rows_np] = 1
        packed = np.packbits(mask_b.ravel(), bitorder='little')
        packed_segs.append(packed)
        seg_packed_sizes.append(len(packed))
    full_packed = np.concatenate(packed_segs) if B > 1 else packed_segs[0]
    indptr = np.zeros(B + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(seg_packed_sizes)
    return (
        torch.from_numpy(full_packed.copy()).to(device, non_blocking=True),
        torch.from_numpy(indptr.copy()).to(device, non_blocking=True),
    )


@torch.inference_mode()
def capture_fi_tree_decode_cudagraph(model_runner, layout=None):
    config = model_runner.config
    hf_config = config.hf_config
    max_bs = min(model_runner.config.max_num_seqs, 512)
    K, F = model_runner.config.speculate_k, model_runner.config.async_fan_out
    if layout is not None:
        MQ_LEN = layout.MQ_LEN
        _graph_key = layout.graph_key
    else:
        MQ_LEN = sum(model_runner.config.fan_out_list)
        _graph_key = "fi_tree_decode"
    max_flat_batch_size = max_bs * MQ_LEN

    max_num_blocks = (config.max_model_len +
                      model_runner.block_size - 1) // model_runner.block_size
    input_ids = torch.zeros(max_flat_batch_size, dtype=torch.int64, device=model_runner.device)
    positions = torch.zeros(max_flat_batch_size, dtype=torch.int64, device=model_runner.device)
    slot_mapping = torch.zeros(max_flat_batch_size, dtype=torch.int32, device=model_runner.device)
    context_lens = torch.full((max_bs,), config.max_model_len, dtype=torch.int32, device=model_runner.device) # make sure these are consistent with our dummy example
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=model_runner.device)
    outputs = torch.empty(max_flat_batch_size, hf_config.hidden_size, dtype=hf_config.torch_dtype, device=model_runner.device)
    logits = torch.empty(max_flat_batch_size, hf_config.vocab_size, dtype=hf_config.torch_dtype, device=model_runner.device)

    # Create graph_bs_list to match what will be used in cudagraph_helpers.py
    graph_bs_list = [1]
    for bs in [2, 4, 8] + list(range(16, max_bs + 1, 16)):
        if bs <= max_bs:
            graph_bs_list.append(bs)
    if max_bs not in graph_bs_list:
        graph_bs_list.append(max_bs)
    graph_bs_list.sort()

    graphs = {}
    graph_pool = None

    # Eagle draft needs hidden_states for forward (d_model_draft, NOT 3*d_model_target)
    # All callers project target acts via fc() BEFORE passing to CG
    # MUST be outside the for-loop so all graphs share the same tensor
    fi_hidden_states = None
    if config.use_eagle and model_runner.is_draft:
        fi_hidden_states = torch.zeros(max_flat_batch_size, hf_config.hidden_size,
                                       dtype=hf_config.torch_dtype, device=model_runner.device)

    print(f'About to capture FI cudagraphs for bs={graph_bs_list} key={_graph_key} MQ_LEN={MQ_LEN}', flush=True)

    for bs in reversed(graph_bs_list):
        print(f'  [FI capture] bs={bs} key={_graph_key} starting...', flush=True)
        graph = torch.cuda.CUDAGraph()

        # Build a self-consistent fake plan for capture:
        # - q_len = MQ_LEN for each request
        # - k_len = max_model_len for each request (use maximum context length)

        cu_seqlens_q = torch.arange(
            bs + 1, dtype=torch.int32, device=model_runner.device) * MQ_LEN
        # Use max_num_blocks pages per request for maximum context length
        kv_indptr = torch.arange(
            bs + 1, dtype=torch.int32, device=model_runner.device) * max_num_blocks
        kv_indices = torch.zeros(int(
            kv_indptr[-1].item()), dtype=torch.int32, device=model_runner.device)  # page ids (dummy)
        # Last page length for max model len context
        last_page_len = config.max_model_len % model_runner.block_size
        if last_page_len == 0:
            last_page_len = model_runner.block_size
        kv_last_page_len = torch.full(
            (bs,), last_page_len, dtype=torch.int32, device=model_runner.device)
        custom_mask = torch.ones(bs * MQ_LEN * config.max_model_len,
                                 dtype=torch.bool, device=model_runner.device)

        # Set the fi_tensors buffers with our fake data
        # Layout-aware: use layout-specific wrapper if available
        if layout is not None and hasattr(model_runner, 'prefill_wrappers_by_layout'):
            _capture_wrapper = model_runner.prefill_wrappers_by_layout[layout.name][bs]
        else:
            _capture_wrapper = model_runner.prefill_wrappers[bs]
        _capture_wrapper.plan(
            cu_seqlens_q,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            hf_config.num_attention_heads,
            hf_config.num_key_value_heads,
            hf_config.head_dim,
            model_runner.block_size,
            custom_mask=custom_mask,
            q_data_type=hf_config.torch_dtype,
            kv_data_type=hf_config.torch_dtype,
        )

        # Set minimal context needed for run
        # Layout-aware: pass active_mq_len and active_wrappers so attention uses correct wrapper
        _active_wrappers = None
        _active_mq_len = None
        if layout is not None and hasattr(model_runner, 'prefill_wrappers_by_layout'):
            _active_mq_len = layout.MQ_LEN
            _active_wrappers = model_runner.prefill_wrappers_by_layout[layout.name]
        set_context(
            is_prefill=False,
            slot_mapping=slot_mapping[:bs * MQ_LEN],
            context_lens=context_lens[:bs],
            block_tables=block_tables[:bs],
            active_mq_len=_active_mq_len,
            active_wrappers=_active_wrappers,
        )

        # Warmup run
        print(f'  [FI capture] bs={bs} key={_graph_key} warmup...', flush=True)
        if fi_hidden_states is not None:
            outputs[:bs * MQ_LEN] = model_runner.model(
                input_ids[:bs * MQ_LEN], positions[:bs * MQ_LEN], fi_hidden_states[:bs * MQ_LEN])
        else:
            outputs[:bs * MQ_LEN] = model_runner.model(
                input_ids[:bs * MQ_LEN], positions[:bs * MQ_LEN])
        logits[:bs * MQ_LEN] = model_runner.model.compute_logits(outputs[:bs * MQ_LEN], False)
        print(f'  [FI capture] bs={bs} key={_graph_key} warmup done, capturing...', flush=True)

        # Capture both model run and logits computation
        with torch.cuda.graph(graph, graph_pool):
            if fi_hidden_states is not None:
                outputs[:bs * MQ_LEN] = model_runner.model(
                    input_ids[:bs * MQ_LEN], positions[:bs * MQ_LEN], fi_hidden_states[:bs * MQ_LEN])
            else:
                outputs[:bs * MQ_LEN] = model_runner.model(input_ids[:bs * MQ_LEN], positions[:bs * MQ_LEN])
            logits[:bs * MQ_LEN] = model_runner.model.compute_logits(outputs[:bs * MQ_LEN], False)

        if graph_pool is None:
            graph_pool = graph.pool()
        graphs[bs] = graph

        torch.cuda.synchronize()
        reset_context()

    graph_vars = dict(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        block_tables=block_tables,
        context_lens=context_lens,
        outputs=outputs,
        logits=logits,
    )
    if fi_hidden_states is not None:
        graph_vars["hidden_states"] = fi_hidden_states

    return graph_vars, graph_pool, graphs, graph_bs_list


# ============================================================
# MESA-SSD: Split Verify CudaGraph (pre + post)
# ============================================================

@torch.inference_mode()
def capture_mesa_verify_cudagraph(model_runner, lookahead=None, graph_pool=None):
    """MESA split verify CudaGraph.
    graph_pre: layers [0, exit_layer] → exit_hidden, exit_residual
    graph_post: layers [exit_layer+1, L-1] + norm → outputs

    Args:
        lookahead: number of speculative tokens to verify per seq (cu_seqlens
            length is lookahead+1 because of recovery slot). Default is
            ``config.speculate_k`` (= K_long). For v1 hybrid path, called twice
            with lookahead=K_long and lookahead=K_short.
        graph_pool: optional CUDA graph pool to share across captures.
    """
    config = model_runner.config
    hf_config = config.hf_config
    max_bs = min(config.max_num_seqs, 512)
    if lookahead is None:
        lookahead = config.speculate_k
    k_plus_1 = lookahead + 1
    exit_layer = config.mesa_exit_layer
    H = hf_config.hidden_size

    input_ids = torch.zeros(max_bs * k_plus_1, dtype=torch.int64)
    positions = torch.zeros(max_bs * k_plus_1, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs * k_plus_1, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, model_runner.max_num_blocks, dtype=torch.int32)
    cu_seqlens_q = torch.zeros(max_bs + 1, dtype=torch.int32)
    exit_hidden = torch.zeros(max_bs * k_plus_1, H, dtype=hf_config.torch_dtype)
    exit_residual = torch.zeros(max_bs * k_plus_1, H, dtype=hf_config.torch_dtype)
    outputs = torch.zeros(max_bs * k_plus_1, H, dtype=hf_config.torch_dtype)

    base = [1, 2, 4, 8]
    dynamic = list(range(16, max_bs + 1, 16))
    all_b = sorted(set(base + dynamic + [max_bs]))
    all_N = [b for b in all_b if b <= max_bs]

    graphs_pre = {}
    graphs_post = {}
    # graph_pool: passed in (for short bucket sharing pool with long); else None at first capture

    for bs in reversed(all_N):
        flat = bs * k_plus_1
        seqlen_q = torch.full((bs,), k_plus_1, dtype=torch.int32)
        cu = cu_seqlens_q[:bs + 1]
        cu.zero_()
        cu[1:].copy_(torch.cumsum(seqlen_q, 0))
        context_lens[:bs] = seqlen_q

        set_context(
            is_prefill=False,
            slot_mapping=slot_mapping[:flat],
            context_lens=context_lens[:bs],
            block_tables=block_tables[:bs],
            cu_seqlens_q=cu,
            max_seqlen_q=k_plus_1,
        )

        # --- graph_pre: layers [0, exit_layer] ---
        hs, res = model_runner.model(
            input_ids[:flat], positions[:flat], end_layer=exit_layer + 1)
        exit_hidden[:flat].copy_(hs)
        exit_residual[:flat].copy_(res)

        graph_pre = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_pre, graph_pool):
            hs, res = model_runner.model(
                input_ids[:flat], positions[:flat], end_layer=exit_layer + 1)
            exit_hidden[:flat].copy_(hs)
            exit_residual[:flat].copy_(res)
        if graph_pool is None:
            graph_pool = graph_pre.pool()
        graphs_pre[bs] = graph_pre

        # --- graph_post: layers [exit_layer+1, L-1] + norm ---
        out = model_runner.model(
            input_ids[:flat], positions[:flat],
            start_layer=exit_layer + 1,
            init_hidden_states=exit_hidden[:flat],
            init_residual=exit_residual[:flat])
        outputs[:flat] = out if not isinstance(out, tuple) else out[0]

        graph_post = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph_post, graph_pool):
            out = model_runner.model(
                input_ids[:flat], positions[:flat],
                start_layer=exit_layer + 1,
                init_hidden_states=exit_hidden[:flat],
                init_residual=exit_residual[:flat])
            outputs[:flat] = out if not isinstance(out, tuple) else out[0]
        graphs_post[bs] = graph_post

        torch.cuda.synchronize()
        reset_context()

    graph_vars = dict(
        input_ids=input_ids, positions=positions,
        slot_mapping=slot_mapping, context_lens=context_lens,
        block_tables=block_tables, cu_seqlens_q=cu_seqlens_q,
        exit_hidden=exit_hidden, exit_residual=exit_residual,
        outputs=outputs,
        lookahead=lookahead,  # so run_mesa_verify_cudagraph picks the right k_plus_1
    )
    return graph_vars, graph_pool, graphs_pre, graphs_post, all_N


@torch.inference_mode()
def run_mesa_verify_cudagraph(model_runner, input_ids, positions, last_only,
                               graph_vars, mesa_proxy_fn=None, bucket="mesa_verify"):
    """Split CudaGraph verify: pre → proxy → post → logits.

    Args:
        graph_vars: dict from capture; ``graph_vars["lookahead"]`` determines k_plus_1.
        bucket: name prefix for ``model_runner.graphs`` / ``graph_bs_list`` keys.
            Default ``"mesa_verify"`` (legacy single-bucket). v1 hybrid uses
            ``"mesa_verify_long"`` and ``"mesa_verify_short"``.
    """
    context = get_context()
    config = model_runner.config
    # lookahead key was added in v1; fallback to speculate_k for legacy graph_vars.
    lookahead = graph_vars.get("lookahead", config.speculate_k)
    k_plus_1 = lookahead + 1
    orig_bs = input_ids.size(0) // k_plus_1

    _ev_setup = mesa_record("verify_setup")
    wrapper_bs = next(
        x for x in model_runner.graph_bs_list[bucket] if x >= orig_bs)
    graph_pre = model_runner.graphs[f"{bucket}_pre"][wrapper_bs]
    graph_post = model_runner.graphs[f"{bucket}_post"][wrapper_bs]

    for k, v in graph_vars.items():
        if k not in ("outputs", "exit_hidden", "exit_residual", "lookahead"):
            v.zero_()

    # Padding (same pattern as run_verify_cudagraph)
    if wrapper_bs > orig_bs:
        pad_bs = wrapper_bs - orig_bs
        pad_flat = pad_bs * k_plus_1
        dev = input_ids.device
        input_ids = torch.cat([input_ids, torch.zeros(pad_flat, dtype=input_ids.dtype, device=dev)])
        positions = torch.cat([positions, torch.zeros(pad_flat, dtype=positions.dtype, device=dev)])
        slot_mapping = torch.cat([
            context.slot_mapping,
            torch.full((pad_flat,), -1, dtype=context.slot_mapping.dtype, device=dev)])
        bt = context.block_tables
        cl = context.context_lens
        block_tables = torch.cat([bt, bt[orig_bs-1:orig_bs].expand(pad_bs, -1).contiguous()])
        context_lens = torch.cat([cl, cl[orig_bs-1:orig_bs].expand(pad_bs).contiguous()])
        bs = wrapper_bs
    else:
        slot_mapping = context.slot_mapping
        block_tables = context.block_tables
        context_lens = context.context_lens
        bs = orig_bs

    graph_vars["input_ids"][:bs * k_plus_1] = input_ids
    graph_vars["positions"][:bs * k_plus_1] = positions
    graph_vars["slot_mapping"][:bs * k_plus_1] = slot_mapping
    graph_vars["context_lens"][:bs] = context_lens
    seqlen_q = torch.full(
        (bs,), k_plus_1, dtype=torch.int32, device=graph_vars["cu_seqlens_q"].device)
    cu = graph_vars["cu_seqlens_q"][:bs + 1]
    cu.zero_()
    cu[1:].copy_(torch.cumsum(seqlen_q, 0))
    if block_tables is not None:
        graph_vars["block_tables"][:bs, :block_tables.size(1)] = block_tables

    mesa_close("verify_setup", _ev_setup)

    # ====== graph_pre.replay() ======
    _ev = mesa_record("graph_pre")
    graph_pre.replay()
    mesa_close("graph_pre", _ev)

    # ====== Mid-forward: exit logits (norm + lm_head on exit_hidden) ======
    _ev_el = mesa_record("exit_logits")
    flat = orig_bs * k_plus_1
    exit_h = graph_vars["exit_hidden"][:flat] + graph_vars["exit_residual"][:flat]
    normed = model_runner.model.model.norm(exit_h, None)
    # ALL TP ranks call compute_logits → gather participation.
    # rank 0: exit_logits = [flat, V]; rank 1+: exit_logits = None
    exit_logits = model_runner.model.compute_logits(normed, last_only=False)
    mesa_close("exit_logits", _ev_el)

    # ====== proxy compute + isend (rank 0 only does real work) ======
    _ev = mesa_record("proxy_compute_send")
    # mesa_proxy_fn: set on rank 0's ModelRunner only. rank 1+ skips.
    if mesa_proxy_fn is not None:
        mesa_proxy_fn(exit_logits, orig_bs)
    mesa_close("proxy_compute_send", _ev)

    # ====== graph_post.replay() ======
    _ev = mesa_record("graph_post")
    graph_post.replay()
    mesa_close("graph_post", _ev)

    # ====== Final logits ======
    _ev_fl = mesa_record("final_logits")
    outputs = graph_vars["outputs"][:flat]
    logits = model_runner.model.compute_logits(outputs, last_only)
    mesa_close("final_logits", _ev_fl)
    return logits


# ────────────────────────────────────────────────────────────────────────────
# Phase 9B-1: hybrid Phase 2 CG capture + run.
#
# Captures a single-depth model forward (cont + proxy rows in flat batch)
# per bucket. Runtime replays K2 times after writing per-depth metadata
# to baked buffers via low_level_packed_plan.
# ────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def capture_phase2_hybrid_cudagraph(model_runner, *, bucket, K_step):
    """Capture phase2_hybrid CG for one bucket (long or short).

    Each bucket has a fixed total_rows = (K_step+1) * (dfo + pfo). Capture
    is one depth's forward; runtime replays K2 times. Separate graph_pool
    per bucket to avoid aliasing (lesson from verify_short capture).

    Returns: (graph_vars, graph_pool, graph, total_rows)
    """
    config = model_runner.config
    hf_config = config.hf_config
    K1 = config.mesa_phase1_k
    K2 = config.mesa_phase2_k
    K_long = K1 + K2
    dfo = config.mesa_draft_fan_out
    pfo = config.mesa_proxy_fan_out
    total = (K_step + 1) * (dfo + pfo)
    max_num_blocks = (config.max_model_len + model_runner.block_size - 1) // model_runner.block_size

    bucket_key = f"phase2_hybrid_{bucket}_cg"
    wrappers = model_runner.prefill_wrappers_by_layout[bucket_key]
    assert total in wrappers, f"phase2_hybrid CG wrapper for total={total} missing"
    wrapper = wrappers[total]

    # Pre-allocated runtime-write buffers (graph_vars).
    input_ids = torch.zeros(total, dtype=torch.int64, device=model_runner.device)
    rope_positions = torch.zeros(total, dtype=torch.int64, device=model_runner.device)
    slot_mapping = torch.zeros(total, dtype=torch.int32, device=model_runner.device)
    context_lens = torch.zeros(total, dtype=torch.int32, device=model_runner.device)
    block_tables = torch.zeros(total, max_num_blocks, dtype=torch.int32, device=model_runner.device)
    outputs = torch.empty(total, hf_config.hidden_size,
                           dtype=hf_config.torch_dtype, device=model_runner.device)

    # Build a self-consistent fake plan for capture (real plan written
    # before each replay via low_level_packed_plan).
    cu_seqlens_q = torch.arange(total + 1, dtype=torch.int32, device=model_runner.device)
    pages_per_row = max_num_blocks
    kv_indptr = torch.arange(total + 1, dtype=torch.int32, device=model_runner.device) * pages_per_row
    kv_indices = torch.zeros(total * pages_per_row, dtype=torch.int32, device=model_runner.device)
    last_page_len = config.max_model_len % model_runner.block_size
    if last_page_len == 0:
        last_page_len = model_runner.block_size
    kv_last_page_len = torch.full((total,), last_page_len, dtype=torch.int32, device=model_runner.device)
    # Fake mask: all visible (will be overwritten per-depth via packed mask)
    fake_mask = torch.ones(total * config.max_model_len, dtype=torch.bool, device=model_runner.device)

    wrapper.plan(
        cu_seqlens_q, kv_indptr, kv_indices, kv_last_page_len,
        hf_config.num_attention_heads, hf_config.num_key_value_heads,
        hf_config.head_dim, model_runner.block_size,
        custom_mask=fake_mask,
        q_data_type=hf_config.torch_dtype,
        kv_data_type=hf_config.torch_dtype,
    )

    set_context(
        is_prefill=False,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=0,
        max_seqlen_k=0,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        active_mq_len=1,
        active_wrappers=wrappers,
        active_layout=None,
    )

    # Warmup
    out = model_runner.model(input_ids, rope_positions)
    outputs.copy_(out)

    # Capture
    graph_pool = None
    graph = torch.cuda.CUDAGraph()
    print(f'[MESA hybrid] capturing phase2_hybrid CG bucket={bucket} '
          f'total_rows={total}', flush=True)
    with torch.cuda.graph(graph, graph_pool):
        out = model_runner.model(input_ids, rope_positions)
        outputs.copy_(out)
    graph_pool = graph.pool()
    torch.cuda.synchronize()
    reset_context()

    graph_vars = dict(
        input_ids=input_ids,
        rope_positions=rope_positions,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        outputs=outputs,
        bucket=bucket,
        K_step=K_step,
        total=total,
    )
    return graph_vars, graph_pool, graph, total


@torch.inference_mode()
def run_phase2_hybrid_cudagraph(model_runner, *, plan, draft_tree_args,
                                  proxy_tree_args, step_proxy_layout,
                                  draft_tokens_phase1, bucket):
    """Run captured phase2_hybrid CG for the given bucket. Replays K2
    depth times with per-depth runtime metadata.

    Phase 9C optimized hot path:
      - Pre-built packed mask & mask_indptr on plan (no bool-mask compute,
        no packing in the loop).
      - Pre-built kv_indptr_cpu, kv_lens_gpu, kv_lpl_gpu on plan.
      - Pre-cached per_depth_L / per_depth_n_pages / per_depth_bytes_per_row
        as Python ints (no `.item()` syncs in the loop).
      - Invariant qo_indptr_cpu cached on plan (no per-depth GPU→CPU copy).

    Mirrors `_decode_phase2_hybrid` (eager path) semantics:
      - cu_seqlens_q=None
      - active_mq_len=1
      - per-row B_proxy region (no overwrite of Phase 1 KV)
      - low-level packed plan path

    Returns: (cont_tokens, cont_logits, proxy_tokens, proxy_logits)
    """
    K1 = model_runner.config.mesa_phase1_k
    K2 = model_runner.config.mesa_phase2_k

    cont_count = plan.cont_row_count
    proxy_count = plan.proxy_row_count
    total = plan.total_row_count
    device = model_runner.device
    V = model_runner.hf_config.vocab_size
    dt = model_runner.hf_config.torch_dtype

    bucket_key = f"phase2_hybrid_{bucket}_cg"
    graph_vars = model_runner.graph_vars[bucket_key]
    graph = model_runner.graphs[bucket_key]
    wrappers = model_runner.prefill_wrappers_by_layout[bucket_key]
    wrapper_total = next(iter(wrappers.keys()))
    wrapper = wrappers[wrapper_total]
    assert total <= wrapper_total, (
        f"hybrid CG total={total} > wrapper allocated={wrapper_total}"
    )

    # Output collectors. proxy_count varies per step (Policy A) so we keep
    # per-call alloc; cont/proxy logits are large enough that preallocation
    # benefits would require additional plan-level slot tracking — leave
    # for a follow-up if profiling shows it's hot.
    cont_tokens = torch.empty((cont_count, K2), dtype=torch.int64, device=device)
    cont_logits = torch.empty((cont_count, K2, V), dtype=dt, device=device)
    proxy_tokens = torch.empty((proxy_count, K2), dtype=torch.int64, device=device)
    proxy_logits = torch.empty((proxy_count, K2, V), dtype=dt, device=device)

    # Initial input ids
    cur_cont_ids = draft_tokens_phase1[:cont_count, K1 - 1].contiguous()
    cur_proxy_ids = proxy_tree_args["input_ids"][:proxy_count].contiguous()

    # qo_indptr — invariant per bucket. Plan caches arange(max_total+1) and
    # we pass [:wrapper_total+1]. Padding entries (>= total) saturate at
    # total via padding-row indptr saturation (handled below in plan).
    # We need qo_indptr to monotonically equal [0..total] then total..total
    # (no rows for padding). Build once outside the loop.
    qo_indptr_cpu = plan.qo_indptr_cpu[: wrapper_total + 1].clone()
    if total < wrapper_total:
        qo_indptr_cpu[total:].fill_(total)

    for d in range(K2):
        # Per-depth Python scalars (no .item() syncs).
        L_d = plan.per_depth_L[d]
        bytes_per_row = plan.per_depth_bytes_per_row[d]

        # Pre-built tensors from plan.
        slot_map_d = plan.per_row_slot_maps_by_depth[:total, d].contiguous()
        ctx_len_d = plan.per_row_context_lens_by_depth[:total, d].contiguous().to(torch.int32)

        # Pre-built kv plan inputs (no per-depth alloc).
        # Plan stores [K2, max_total+1] CPU indptr already padded to wrapper.
        kv_indptr_cpu_full = plan.per_depth_kv_indptr_cpu[d, : wrapper_total + 1]
        # kv_indices: per_row_kv_indices_by_depth[d] holds total*n_pages_d
        # entries. For wrapper padding, pass slice; padding rows have empty
        # mask so they won't read from invalid pages.
        n_pages_d = plan.per_depth_n_pages[d]
        kv_indices_d = plan.per_row_kv_indices_by_depth[d, : total * n_pages_d]
        kv_lens_gpu_d = plan.per_depth_kv_lens_gpu[d, : wrapper_total]
        kv_lpl_gpu_d = plan.per_depth_kv_lpl_gpu[d, : wrapper_total]

        # Pre-built packed mask + indptr from plan (was duplicated at runtime
        # pre-9C — biggest hot-path win).
        n_total_bytes = wrapper_total * bytes_per_row
        packed_mask = plan.per_depth_packed_masks[d, :n_total_bytes]
        packed_indptr = plan.per_depth_mask_indptr[d, : wrapper_total + 1]

        # Low-level plan: writes to wrapper baked buffers.
        low_level_packed_plan(
            wrapper, model_runner=model_runner,
            qo_indptr_cpu=qo_indptr_cpu,
            kv_indptr_cpu=kv_indptr_cpu_full,
            kv_indices=kv_indices_d,
            kv_last_page_len_gpu=kv_lpl_gpu_d,
            kv_lens_gpu=kv_lens_gpu_d,
            packed_mask=packed_mask,
            packed_indptr=packed_indptr,
            B=wrapper_total,
        )

        # Flat input_ids and rope per depth (small allocs; can also be
        # pre-allocated per bucket but cont_count + proxy_count varies).
        flat_input_ids = torch.cat([cur_cont_ids, cur_proxy_ids], dim=0)
        cont_rope = plan.cont_initial_rope_positions[:cont_count] + d
        proxy_rope = plan.proxy_initial_rope_positions[:proxy_count] + d
        flat_rope = torch.cat([cont_rope, proxy_rope], dim=0).contiguous()
        block_tables_d = plan.per_row_block_tables[:total]

        # Per-depth prep label (matches split's phase{1,2}_prep semantics —
        # KV plan + buffer copies just before graph.replay).
        _mev_php = mesa_record(f"phase2_hybrid_prep_{bucket}")
        # Write per-depth runtime metadata to graph buffers.
        graph_vars["input_ids"][:total].copy_(flat_input_ids, non_blocking=True)
        graph_vars["rope_positions"][:total].copy_(flat_rope, non_blocking=True)
        graph_vars["slot_mapping"][:total].copy_(slot_map_d, non_blocking=True)
        graph_vars["context_lens"][:total].copy_(ctx_len_d, non_blocking=True)
        graph_vars["block_tables"][:total, :block_tables_d.shape[1]].copy_(
            block_tables_d, non_blocking=True,
        )
        mesa_close(f"phase2_hybrid_prep_{bucket}", _mev_php)

        # Per-depth replay label (matches split's phase{1,2}_replay — fires
        # K2 times per spec step, once per depth).
        _mev_phr = mesa_record(f"phase2_hybrid_replay_{bucket}")
        graph.replay()
        mesa_close(f"phase2_hybrid_replay_{bucket}", _mev_phr)

        outputs = graph_vars["outputs"][:total]
        logits_flat = model_runner.model.compute_logits(outputs, last_only=False).view(-1, V)
        next_tokens = logits_flat.argmax(dim=-1)

        cont_logits[:, d, :] = logits_flat[:cont_count]
        cont_tokens[:, d] = next_tokens[:cont_count]
        proxy_logits[:, d, :] = logits_flat[cont_count:total]
        proxy_tokens[:, d] = next_tokens[cont_count:total]

        cur_cont_ids = next_tokens[:cont_count]
        cur_proxy_ids = next_tokens[cont_count:total]

    return cont_tokens, cont_logits, proxy_tokens, proxy_logits


# ---------- MESA per-phase profiling (zero-sync, additions only) ----------
# Doc: ssd/docs/mesa/06-timeline-cleanup-plan.md §4.2-4.5, §5 Phase B.
#
# Design summary:
#   - One CUDA/CPU anchor per process, captured lazily on first mesa_record()
#     when SSD_PROFILE_MESA=1. The only allowed syncs are anchor init and dump.
#   - mesa_record(label, parent=None) starts a span: records a CUDA event +
#     CPU dispatch ns, snapshots current context (step_id, proc) for fallback.
#   - mesa_close(label, start_handle) ends a span and CRITICALLY reads the
#     CURRENT context (especially `status`) at close time so target_spec_wait
#     can be labeled with the hit-class learned only after speculate() returns.
#   - mesa_dump(tag) syncs once and computes wall-clock + cuda_ms derived
#     fields per row, writes a single JSON with an _anchor metadata block.
import time as _time_mod_for_mesa

PROFILE_MESA = os.environ.get("SSD_PROFILE_MESA", "0") == "1"

# Per-span open record:
#   (idx, label, parent_label, start_ev, end_ev,
#    cpu_dispatch_start_ns, cpu_dispatch_end_ns,
#    open_step_id, open_proc, close_step_id, close_status, close_proc)
_mesa_events = []
_mesa_idx = 0      # monotonic call index (per process)

# CUDA/CPU anchor (set once per process, lazily on first record)
_mesa_anchor_event = None
_mesa_anchor_cpu_ns = None
_mesa_anchor_device = None

# Profiler context (process-local, single-threaded for the profiled path).
# Used so individual call sites do not need to thread step_id/status/proc
# through every helper signature. mesa_close() reads this AT close time.
_mesa_context = {"step_id": None, "status": None, "proc": None}


def _ensure_mesa_anchor():
    """Lazily initialize the per-process CUDA/CPU anchor.

    Only synchronizes when PROFILE_MESA=1. Idempotent; no-op when off or
    already initialized. This is the only sync in the profile path outside
    mesa_dump().
    """
    global _mesa_anchor_event, _mesa_anchor_cpu_ns, _mesa_anchor_device
    if not PROFILE_MESA:
        return
    if _mesa_anchor_event is not None:
        return
    torch.cuda.synchronize()
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    torch.cuda.synchronize()
    _mesa_anchor_event = ev
    _mesa_anchor_cpu_ns = _time_mod_for_mesa.perf_counter_ns()
    _mesa_anchor_device = torch.cuda.current_device()


def mesa_set_context(step_id=None, status=None, proc=None):
    """Update the process-local profiler context.

    Each argument is applied only if not None, so callers can update a
    subset (e.g. only `status` after speculate() returns). No-op when
    PROFILE_MESA=0 to keep cold path zero-cost.
    """
    if not PROFILE_MESA:
        return
    if step_id is not None:
        _mesa_context["step_id"] = step_id
    if status is not None:
        _mesa_context["status"] = status
    if proc is not None:
        _mesa_context["proc"] = proc


def mesa_clear_status():
    """Explicitly clear close-time status. Used between draft requests so a
    new request does not inherit the previous request's hit/miss label."""
    if not PROFILE_MESA:
        return
    _mesa_context["status"] = None


def mesa_record(label, parent=None):
    """Open a profiling span. Returns an opaque handle for mesa_close().

    Captures: start CUDA event, CPU dispatch ns, and a snapshot of
    (step_id, proc) at open time. `status` is intentionally NOT snapshotted
    here — it is read fresh in mesa_close() so spans like target_spec_wait
    can pick up status learned after the body runs.
    """
    if not PROFILE_MESA:
        return None
    _ensure_mesa_anchor()
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    cpu_ns = _time_mod_for_mesa.perf_counter_ns()
    # Snapshot open-time context (fallback for step_id/proc).
    open_step_id = _mesa_context["step_id"]
    open_proc = _mesa_context["proc"]
    return (ev, cpu_ns, label, parent, open_step_id, open_proc)


def mesa_close(label, start_handle):
    """Close a profiling span opened by mesa_record().

    Reads the CURRENT context (step_id, status, proc) at close time. This
    is essential for target_spec_wait_* whose status is set after
    speculator.speculate() returns. Open-time step_id/proc are used as a
    fallback when the close-time context is unset.
    """
    global _mesa_idx
    if start_handle is None:
        return
    start_ev, cpu_start_ns, _open_label, parent, open_step_id, open_proc = start_handle
    end_ev = torch.cuda.Event(enable_timing=True)
    end_ev.record()
    cpu_end_ns = _time_mod_for_mesa.perf_counter_ns()
    # Close-time context wins; fall back to open-time snapshot if unset.
    close_step_id = _mesa_context["step_id"]
    if close_step_id is None:
        close_step_id = open_step_id
    close_proc = _mesa_context["proc"]
    if close_proc is None:
        close_proc = open_proc
    close_status = _mesa_context["status"]
    _mesa_events.append((
        _mesa_idx, label, parent,
        start_ev, end_ev,
        cpu_start_ns, cpu_end_ns,
        close_step_id, close_status, close_proc,
    ))
    _mesa_idx += 1


def mesa_reset():
    """Reset profiling state — mesa_dump/mesa_flush call this after writing."""
    global _mesa_idx, _mesa_anchor_event, _mesa_anchor_cpu_ns, _mesa_anchor_device
    _mesa_events.clear()
    _mesa_idx = 0
    _mesa_anchor_event = None
    _mesa_anchor_cpu_ns = None
    _mesa_anchor_device = None
    _mesa_context["step_id"] = None
    _mesa_context["status"] = None
    _mesa_context["proc"] = None


def mesa_flush(tag, run_id=None):
    """Public API: write JSON and reset. Call between runs or at end of generate().
    run_id: appended to filename to avoid overwrites; defaults to HMS timestamp."""
    return mesa_dump(tag, run_id=run_id)


def mesa_dump(tag, run_id=None):
    if not _mesa_events:
        mesa_reset()
        return
    import json
    # Single sync — convert CUDA event times to anchored wall-clock at dump
    # time so the per-step path stays sync-free.
    torch.cuda.synchronize()
    anchor_ev = _mesa_anchor_event
    anchor_cpu_ns = _mesa_anchor_cpu_ns
    rows = []
    for (
        idx, label, parent,
        start_ev, end_ev,
        cpu_start_ns, cpu_end_ns,
        step_id, status, proc,
    ) in _mesa_events:
        gpu_start_ms = anchor_ev.elapsed_time(start_ev)
        gpu_end_ms = anchor_ev.elapsed_time(end_ev)
        cuda_ms = start_ev.elapsed_time(end_ev)
        wall_start_ns = anchor_cpu_ns + int(gpu_start_ms * 1e6)
        wall_end_ns = anchor_cpu_ns + int(gpu_end_ms * 1e6)
        rows.append({
            "idx": idx,
            "proc": proc,
            "step_id": step_id,
            "status": status,
            "label": label,
            "parent_label": parent,
            "gpu_start_ms_since_anchor": gpu_start_ms,
            "gpu_end_ms_since_anchor": gpu_end_ms,
            "wall_start_ns": wall_start_ns,
            "wall_end_ns": wall_end_ns,
            "cuda_ms": cuda_ms,
            "cpu_dispatch_start_ns": cpu_start_ns,
            "cpu_dispatch_end_ns": cpu_end_ns,
            # Backward-compat aliases for legacy consumers
            # (summarize_ssd_run.py reads `ms`, `start_ms`, `end_ms`).
            "ms": cuda_ms,
            "start_ms": gpu_start_ms,
            "end_ms": gpu_end_ms,
        })
    outdir = os.environ.get("SSD_PROFILE_DIR", "/tmp")
    os.makedirs(outdir, exist_ok=True)
    if run_id is None:
        run_id = _time_mod_for_mesa.strftime("%H%M%S")
    path = f"{outdir}/mesa_profile_{tag}_{run_id}.json"
    # Wire format: top-level JSON is a list of row dicts. To keep the legacy
    # `summarize_ssd_run.py` parser working (it does `json.load` and filters
    # by `label`), the anchor metadata is emitted as a sentinel row with
    # `label="_anchor"`. New consumers (Phase C plotter) look this row up by
    # label and extract anchor_cpu_ns / anchor_device from it.
    anchor_row = {
        "idx": -1,
        "proc": None,
        "step_id": None,
        "status": None,
        "label": "_anchor",
        "parent_label": None,
        "anchor_cpu_ns": anchor_cpu_ns,
        "anchor_device": _mesa_anchor_device,
        "anchor_note": "GPU event times converted to host monotonic clock",
    }
    payload = [anchor_row] + rows
    with open(path, "w") as f:
        json.dump(payload, f)
    print(f"[mesa_profile] {len(rows)} events -> {path}", flush=True)
    mesa_reset()
