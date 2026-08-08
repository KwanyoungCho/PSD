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

    ``bucket`` selects the CG family. Default = "verify" (K+1 wide).
    Split-K1/K2 draft glue uses bucket="verify_k1" / "verify_k2".
    k_plus_1 is derived from graph_bs_list / input shape.
    """
    context = get_context()
    if bucket == "verify_k1":
        k_plus_1 = model_runner.config.duet_phase1_k + 1
    elif bucket == "verify_k2":
        k_plus_1 = model_runner.config.duet_phase2_k + 1
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
    _ev_vr = duet_record(_vr_label)
    graph.replay()
    duet_close(_vr_label, _ev_vr)

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
def cg_input_range_check(input_ids, positions, slot_mapping,
                         context_lens, vocab_size, rope_len,
                         n_kv_slots, step, active_mask=None):
    """리뷰 2단계: graph.replay() 직전 입력 범위 검사 (debug 전용,
    SSD_CG_INPUT_CHECK=1). 위반 시 CUDA graph를 실행하지 않고 CPU
    RuntimeError로 종료 — GPU context 보존 + 정확한 최초 위반 기록.
    규약: active lane은 반드시 유효한 slot을 가져야 한다. padding
    lane만 active_mask=False일 때 slot=-1을 사용할 수 있다. position/
    token 범위 밖과 ctx<=0도 오류다."""
    bad = []
    im, ix = int(input_ids.min()), int(input_ids.max())
    if im < 0 or ix >= vocab_size:
        lane = int(((input_ids < 0) | (input_ids >= vocab_size))
                   .nonzero()[0])
        bad.append(f"token[min={im},max={ix},V={vocab_size},"
                   f"lane={lane}]")
    pm, px = int(positions.min()), int(positions.max())
    if pm < 0 or px >= rope_len:
        lane = int(((positions < 0) | (positions >= rope_len))
                   .nonzero()[0])
        bad.append(f"pos[min={pm},max={px},rope_len={rope_len},"
                   f"lane={lane}]")
    if slot_mapping is not None:
        sm, sx = int(slot_mapping.min()), int(slot_mapping.max())
        if active_mask is not None:
            act = active_mask.to(slot_mapping.device,
                                 dtype=torch.bool)
            bad_sl = ((slot_mapping < 0) & act) \
                | (slot_mapping < -1) | (slot_mapping >= n_kv_slots)
        else:
            # mask 미제공 기본: -1은 정상 padding (run_fi의 기존
            # pad 규약) — < -1 과 상한 초과만 위반
            bad_sl = (slot_mapping < -1) \
                | (slot_mapping >= n_kv_slots)
        if bool(bad_sl.any()):
            lane = int(bad_sl.nonzero()[0])
            bad.append(f"slot[min={sm},max={sx},N={n_kv_slots},"
                       f"lane={lane}]")
    if context_lens is not None and int(context_lens.min()) <= 0:
        bad.append(f"ctx[min={int(context_lens.min())}]")
    if bad:
        raise RuntimeError(
            f"[cg-input-check] step={step} 범위 위반: "
            + " ".join(bad))


def _rope_len_of(model_runner):
    rl = getattr(model_runner, "_cg_rope_len", None)
    if rl is None:
        rl = 0
        for m in model_runner.model.modules():
            c = getattr(m, "cos_sin_cache", None)
            if c is not None:
                rl = int(c.shape[0])
                break
        model_runner._cg_rope_len = rl
    return rl


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
    _mev_prep = duet_record(_prep_label)
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
        # Layout change: clear cache if MQ_LEN changed (2-pass DUET reuses global cache)
        if cache.get("_mq_len") != MQ_LEN:
            # 이슈 #10: 레이아웃 전환 clear가 트리 mask override 훅을
            # 삭제하면 안 된다 (P1↔P2 폭이 다른 모든 트리 step에서
            # f=0 무음 오마스크 + f=1 KeyError) — pop/복원으로 보존.
            _tree_ov_keep = cache.pop("_tree_mask_override", None)
            cache.clear()
            cache["_mq_len"] = MQ_LEN
            if _tree_ov_keep is not None:
                cache["_tree_mask_override"] = _tree_ov_keep
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
        # Batch 1b: reuse cache_hits_list threaded through context (set in
        # _decode_tree_step from payload) to avoid a duplicate .tolist() sync
        # per step. Fall back to the tolist path when it's not pre-computed
        # (non-DUET / non-tree paths).
        _ctx_chl = getattr(context, "active_cache_hits_list", None)
        if _ctx_chl is not None and len(_ctx_chl) >= B:
            cache_hits_list = _ctx_chl[:B]
        else:
            cache_hits_list = cache_hits[:B].tolist()

        # 이슈 #11: 트리 override 활성 진입은 실행되는 모든 step의 mask를
        # override가 공급한다 (TREE_GLUE=step0 한정, P1=전 step 사전등록,
        # rollout=매 forward 직전 등록). 체인 mask 재빌드는 글루 폭 전제
        # (K_for_mask+1)가 트리 글루(n_valid+1)와 달라 음수 prefix 등으로
        # 깨질 수 있고 결과물도 읽히지 않으므로 통째로 생략 — 체인 호출은
        # 자기 step-0 진입에서 재빌드한다.
        if cache.get("_tree_mask_override") is not None:
            cache.pop("cpu_packed_masks", None)
            cache.pop("cpu_packed_indptrs", None)
        else:
            # M3 (docs/duet/13 §4): split_k2's fan_out_list is a list of per-seq
            # lists ([B][position_count]); build one glue block per seq. Flat
            # lists (all other layouts + non-DUET) keep the shared-glue path.
            _per_seq_fol = bool(_fan_out_list) and isinstance(_fan_out_list[0], list)
            # Layout change detection: if fan_out changed, recompute glue masks.
            # The _cached_fol key is the FULL per-seq structure when nested, so any
            # per-seq distribution change (the norm at B>1) forces a rebuild; at
            # B=1 a repeated distribution still hits the cache exactly as pre-M3.
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
                if _per_seq_fol:
                    cache["glue_hit_np"] = [np.repeat(_tril, f, axis=0) for f in _fol]
                    cache["glue_miss_np"] = (
                        cache["glue_hit_np"] if _fol_miss is _fol
                        else [np.repeat(_tril, f, axis=0) for f in _fol_miss])
                else:
                    cache["glue_hit_np"] = np.repeat(_tril, _fol, axis=0)
                    cache["glue_miss_np"] = np.repeat(_tril, _fol_miss, axis=0)

            _glue_hit = cache["glue_hit_np"]
            _glue_miss = cache["glue_miss_np"]
            _rows_np = np.arange(MQ_LEN)

            cache["cpu_packed_masks"] = []
            cache["cpu_packed_indptrs"] = []

            # ─────────────────────────────────────────────────────────────────
            # NOTE: this generic mask formula assumes the layout
            #     [persistent | glue (K+1) | diag blocks of MQ_LEN]
            # i.e. single-pass tree decode where K = layout.K and no prior spec
            # scratch was written. That holds for every live caller (non-DUET
            # full layout + split-K1/K2 passes, which each start from a clean
            # scratch region). The removed hybrid "continuation pass" violated
            # it — see git history (2026-07 removal) for the KNOWN BUG writeup.
            # ─────────────────────────────────────────────────────────────────
            for s in range(K_loop):
                # Step 9A: ttl_added_s uses K_for_mask (= layout glue width)
                # not the outer K (config.speculate_k). For phase1_short the
                # glue width is K_short+1; for phase1_long / non-DUET it's
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
                    if _per_seq_fol:
                        # per-seq glue block; padded rows (b >= real B when the
                        # CG bucket pads) reuse the last real seq's block — their
                        # outputs are discarded (slot_map -1).
                        glue = glue[b] if b < len(glue) else glue[-1]
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

    # P2-tree (T1.4b, docs/duet/internal/20): per-forward 동적 mask 주입 —
    # rollout 어댑터가 채우는 override. 체인 경로는 키 부재로 무영향.
    _tree_ov = cache.get("_tree_mask_override")
    if _tree_ov is not None and step in _tree_ov:
        packed_mask, packed_indptr = _tree_ov[step]
    else:
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
    if os.environ.get("SSD_CG_INPUT_CHECK", "0") == "1":
        cg_input_range_check(
            input_ids, positions, get_context().slot_mapping,
            context_lens,
            int(model_runner.hf_config.vocab_size),
            _rope_len_of(model_runner),
            int(model_runner.kv_cache.shape[2]
                * model_runner.kv_cache.shape[3]),
            step)
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
        _duet_label = "phase1_replay"
    elif _layout_name == "proxy" or _layout_name == "split_k2":
        _duet_label = "phase2_replay"
    else:
        _duet_label = "tree_replay"
    duet_close(_prep_label, _mev_prep)
    _mev = duet_record(_duet_label)
    graph.replay()
    duet_close(_duet_label, _mev)

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
# DUET-SSD: Split Verify CudaGraph (pre + post)
# ============================================================

def update_tree_verify_graph_buffers(wrapper, kv_indices, packed_mask,
                                     plan_host):
    """Update only the tensors read by a captured tree-attention graph.

    ``BatchPrefillWithPagedKVCacheWrapper.plan()`` cannot itself be captured.
    More importantly, calling it immediately before ``CUDAGraph.replay()``
    cannot change the graph's already-recorded launch geometry; in the old
    target path it merely recopied these small runtime tensors and then ran
    the host planner anyway.  The graph is captured for a fixed query bucket,
    while page count, last-page length, page ids and mask remain ordinary
    buffer *contents*.  Updating those contents directly is the same contract
    used by :class:`P2TreeExecutor`.

    ``plan_host`` owns persistent CPU int32 tensors ``kv``, ``last``,
    ``mask`` and ``klen``.  Keeping them persistent avoids both CUDA scalar
    assignments and the hidden device-to-host reads inside ``plan()``.
    """
    n_blocks = int(plan_host["kv"][1])
    if n_blocks <= 0:
        raise ValueError("tree verify requires at least one KV page")
    if n_blocks > wrapper._paged_kv_indices_buf.numel():
        raise ValueError(
            f"tree verify page count {n_blocks} exceeds wrapper capacity "
            f"{wrapper._paged_kv_indices_buf.numel()}")
    if packed_mask.numel() > wrapper._custom_mask_buf.numel():
        raise ValueError(
            f"tree verify packed mask {packed_mask.numel()} exceeds "
            f"wrapper capacity {wrapper._custom_mask_buf.numel()}")

    # All copies are enqueued on the current stream before graph replay.
    # CPU sources are tiny and persistent; the page-id source is already on
    # the target GPU.  No allocation, plan call, or GPU->CPU synchronization
    # is performed here.
    wrapper._paged_kv_indptr_buf.copy_(plan_host["kv"], non_blocking=True)
    wrapper._paged_kv_last_page_len_buf.copy_(
        plan_host["last"], non_blocking=True)
    wrapper._paged_kv_indices_buf[:n_blocks].copy_(
        kv_indices[:n_blocks], non_blocking=True)
    wrapper._custom_mask_buf[:packed_mask.numel()].copy_(
        packed_mask, non_blocking=True)
    wrapper._mask_indptr_buf.copy_(plan_host["mask"], non_blocking=True)
    wrapper._kv_lens_buffer.copy_(plan_host["klen"], non_blocking=True)


@torch.inference_mode()
def capture_tree_verify_cudagraph(model_runner, graph_pool=None):
    """P2-tree verify CG (T3.2 bucket capture — docs/duet/internal/20).

    bucket = (N_v, page_count), B=1.  FlashInfer's host ``plan()`` chooses
    launch geometry from the KV page shape.  Rewriting its buffers after a
    graph was captured does *not* rewrite that launch geometry, so a graph
    captured with one page cannot safely serve a three-page request.  Capture
    every reachable page count once.  Within a page bucket the last page is a
    full-page canvas and the packed mask hides columns beyond the real
    ``kv_len``; this keeps shape/launch fixed while token values and topology
    remain dynamic.
    """
    config = model_runner.config
    hf_config = config.hf_config
    exit_layer = config.duet_exit_layer
    H = hf_config.hidden_size
    bs_blk = model_runner.block_size
    _tp = max(1, model_runner.num_tp_gpus)
    dev = model_runner.device
    out = {}
    max_pages = (config.max_model_len + bs_blk - 1) // bs_blk
    for nv_b in sorted(model_runner.tree_verify_wrappers):
        wrapper = model_runner.tree_verify_wrappers[nv_b]
        r = nv_b + 1
        bufs = {
            "input_ids": torch.zeros(r, dtype=torch.int64),
            "rope": torch.zeros(r, dtype=torch.int64),
            "slot_mapping": torch.zeros(r, dtype=torch.int32),
            "context_lens": torch.full((1,), r, dtype=torch.int32),
            "exit_hidden": torch.zeros(r, H, dtype=hf_config.torch_dtype),
            "exit_residual": torch.zeros(r, H, dtype=hf_config.torch_dtype),
            "outputs": torch.zeros(r, H, dtype=hf_config.torch_dtype),
        }
        for n_pages in range(1, max_pages + 1):
            canvas_cols = n_pages * bs_blk
            bufs["context_lens"].fill_(canvas_cols)
            # Capture with the exact page-count launch geometry.  Page ids
            # are contents and are replaced before replay; all-zero ids are
            # finite and valid for warmup/capture.
            wrapper.plan(
                torch.tensor([0, r], dtype=torch.int32, device=dev),
                torch.tensor([0, n_pages], dtype=torch.int32, device=dev),
                torch.zeros(n_pages, dtype=torch.int32, device=dev),
                torch.tensor([bs_blk], dtype=torch.int32, device=dev),
                max(1, hf_config.num_attention_heads // _tp),
                max(1, hf_config.num_key_value_heads // _tp),
                hf_config.head_dim, bs_blk,
                custom_mask=torch.ones(
                    r * canvas_cols, dtype=torch.bool, device=dev),
                q_data_type=hf_config.torch_dtype,
                kv_data_type=hf_config.torch_dtype,
            )
            set_context(
                is_prefill=False,
                slot_mapping=bufs["slot_mapping"],
                context_lens=bufs["context_lens"],
                tree_verify_wrapper=wrapper,
            )
            # --- pre: layers [0, exit] ---
            hs, res = model_runner.model(
                bufs["input_ids"], bufs["rope"], end_layer=exit_layer + 1)
            bufs["exit_hidden"].copy_(hs)
            bufs["exit_residual"].copy_(res)
            g_pre = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_pre, graph_pool):
                hs, res = model_runner.model(
                    bufs["input_ids"], bufs["rope"],
                    end_layer=exit_layer + 1)
                bufs["exit_hidden"].copy_(hs)
                bufs["exit_residual"].copy_(res)
            if graph_pool is None:
                graph_pool = g_pre.pool()
            # --- post: layers [exit+1, L) + norm ---
            o = model_runner.model(
                bufs["input_ids"], bufs["rope"],
                start_layer=exit_layer + 1,
                init_hidden_states=bufs["exit_hidden"],
                init_residual=bufs["exit_residual"])
            bufs["outputs"].copy_(o if not isinstance(o, tuple) else o[0])
            g_post = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_post, graph_pool):
                o = model_runner.model(
                    bufs["input_ids"], bufs["rope"],
                    start_layer=exit_layer + 1,
                    init_hidden_states=bufs["exit_hidden"],
                    init_residual=bufs["exit_residual"])
                bufs["outputs"].copy_(
                    o if not isinstance(o, tuple) else o[0])
            reset_context()
            out[(nv_b, n_pages)] = {
                "bufs": bufs, "pre": g_pre, "post": g_post,
                "wrapper": wrapper, "canvas_cols": canvas_cols,
                "n_pages": n_pages}
    return out, graph_pool


@torch.inference_mode()
def capture_duet_verify_cudagraph(model_runner, lookahead=None, graph_pool=None):
    """DUET split verify CudaGraph.
    graph_pre: layers [0, exit_layer] → exit_hidden, exit_residual
    graph_post: layers [exit_layer+1, L-1] + norm → outputs

    Args:
        lookahead: number of speculative tokens to verify per seq (cu_seqlens
            length is lookahead+1 because of recovery slot). Default is
            ``config.speculate_k``. Split-K1/K2 mode captures per bucket with
            lookahead=K1 and lookahead=K2.
        graph_pool: optional CUDA graph pool to share across captures.
    """
    config = model_runner.config
    hf_config = config.hf_config
    max_bs = min(config.max_num_seqs, 512)
    if lookahead is None:
        lookahead = config.speculate_k
    k_plus_1 = lookahead + 1
    exit_layer = config.duet_exit_layer
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
        lookahead=lookahead,  # so run_duet_verify_cudagraph picks the right k_plus_1
    )
    return graph_vars, graph_pool, graphs_pre, graphs_post, all_N


@torch.inference_mode()
def _duet_exit_topm_gather(model_runner, normed, input_ids_flat, M):
    """Rank-local top-M exit reduction (SSD_DUET_EXIT_TOPM_GATHER=1).

    Each TP rank computes its lm_head vocab-shard logits (same GEMM as the
    full path), reduces to top-M values/ids + a logsumexp partial + the
    shard's draft-token logit, and gathers ONE small fp32 payload
    [flat, 2M+2] to rank 0 (vs the full [flat, V] logits). Rank 0 merges
    the 4 partials into an EXACT global top-M candidate dict with exact
    probabilities (p = exp(logit - lse_global)) in the raw-proxy schema
    consumed by policy_b_from_candidates.

    y (draft token at each position) is derived from the verify input ids:
    position i's draft token = input_ids[i+1] (B=1; the all-accept slot K
    has no draft token). Vocab ids fit fp32 exactly (V < 2^24).

    Returns: raw dict on rank 0, None on other ranks.
    """
    import torch.distributed as dist
    lm_head = model_runner.model.lm_head
    flat = normed.size(0)
    K_ = flat - 1
    shard = torch.nn.functional.linear(normed, lm_head.weight)  # [flat, V/tp]
    shard_f = shard.float()
    lse_local = torch.logsumexp(shard_f, dim=-1)                # [flat]
    vals, idx = shard_f.topk(M, dim=-1)                         # [flat, M]
    tp_size = getattr(lm_head, "tp_size", 1)
    vstart = getattr(lm_head, "vocab_start_idx", 0)
    vend = getattr(lm_head, "vocab_end_idx", shard.size(1))
    ids = (idx + vstart).to(torch.float32)

    y_tok = input_ids_flat[1:flat]                              # [K_]
    in_range = (y_tok >= vstart) & (y_tok < vend)
    y_idx = (y_tok - vstart).clamp(0, shard.size(1) - 1)
    y_val = shard_f[:K_].gather(1, y_idx.unsqueeze(1)).squeeze(1)
    y_val = torch.where(in_range, y_val,
                        torch.full_like(y_val, float("-inf")))

    payload = torch.empty(flat, 2 * M + 2, dtype=torch.float32,
                          device=normed.device)
    payload[:, :M] = vals
    payload[:, M:2 * M] = ids
    payload[:, 2 * M] = lse_local
    payload[:K_, 2 * M + 1] = y_val
    payload[K_:, 2 * M + 1] = float("-inf")

    if tp_size > 1:
        parts = ([torch.empty_like(payload) for _ in range(tp_size)]
                 if lm_head.tp_rank == 0 else None)
        dist.gather(payload, parts, 0, group=lm_head.tp_group)
        if lm_head.tp_rank != 0:
            return None
    else:
        parts = [payload]

    vals4 = torch.cat([p[:, :M] for p in parts], dim=1)         # [flat, tp*M]
    ids4 = torch.cat([p[:, M:2 * M] for p in parts], dim=1)
    lse_g = torch.logsumexp(
        torch.stack([p[:, 2 * M] for p in parts], dim=0), dim=0)  # [flat]
    y_g = torch.stack(
        [p[:, 2 * M + 1] for p in parts], dim=0).max(dim=0).values  # [flat]
    top_lg, sel = vals4.topk(M, dim=-1)                         # exact global top-M
    top_ids = ids4.gather(1, sel).to(torch.int64)
    return {
        "topk_ids": top_ids,
        "topk_logits": top_lg,
        "lse": lse_g,
        "y_logit": y_g[:K_],
    }


def run_duet_verify_cudagraph(model_runner, input_ids, positions, last_only,
                               graph_vars, duet_proxy_fn=None, bucket="duet_verify"):
    """Split CudaGraph verify: pre → proxy → post → logits.

    Args:
        graph_vars: dict from capture; ``graph_vars["lookahead"]`` determines k_plus_1.
        bucket: name prefix for ``model_runner.graphs`` / ``graph_bs_list`` keys.
            Split-K1/K2 mode uses ``"duet_verify_k1"`` / ``"duet_verify_k2"``.
    """
    context = get_context()
    config = model_runner.config
    # lookahead key was added in v1; fallback to speculate_k for legacy graph_vars.
    lookahead = graph_vars.get("lookahead", config.speculate_k)
    k_plus_1 = lookahead + 1
    orig_bs = input_ids.size(0) // k_plus_1

    _ev_setup = duet_record("verify_setup")
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

    duet_close("verify_setup", _ev_setup)

    # ====== graph_pre.replay() ======
    if getattr(config, "duet_exit_replica", False):
        # Exit-replica: graph_pre rewrites exit_hidden/exit_residual —
        # make sure the PREVIOUS step's side-stream read finished (it
        # completes ~35 ms before this point in practice; the wait is a
        # no-op guard against pathological stalls).
        _prev_done = getattr(model_runner, "_duet_exit_done_ev", None)
        if _prev_done is not None:
            torch.cuda.current_stream().wait_event(_prev_done)
    _ev = duet_record("graph_pre")
    graph_pre.replay()
    duet_close("graph_pre", _ev)

    if getattr(config, "duet_exit_replica", False):
        # B>1 not supported for this gate (docs/duet/13 §6).
        assert orig_bs == 1, "duet_exit_replica: B>1 not supported (docs/duet/13 §6)"
        # ====== Exit-replica overlap (docs/duet/09 WS3c) ======
        # NO TP collective: ranks 1+ fall straight through to graph_post
        # (the exit rendezvous point disappears). Rank 0 — the only rank
        # with duet_proxy_fn set — runs norm + full-vocab lm_head
        # (local replica) + Policy B + send on a side stream: the CPU
        # dispatch below is hidden behind graph_pre's GPU execution and
        # the side work itself overlaps graph_post. The exit_logits /
        # Record dispatch on the default stream and the actual norm + head +
        # proxy calculation on the side stream.  The old ``exit_logits``
        # event put both endpoints on the default stream, so a long-looking
        # bar could be stream contention rather than exit computation.
        _ev_el = duet_record("exit_proxy_launch")
        flat = orig_bs * k_plus_1
        _replica = getattr(model_runner, "_duet_lm_head_replica", None)
        if duet_proxy_fn is not None and _replica is not None:
            _es = getattr(model_runner, "_duet_exit_stream", None)
            if _es is None:
                _es = torch.cuda.Stream(device=_replica.device)
                model_runner._duet_exit_stream = _es
            _ev_ready = torch.cuda.Event()
            _ev_ready.record()  # default stream: fires when graph_pre is done
            with torch.cuda.stream(_es):
                _es.wait_event(_ev_ready)
                _ev_side = duet_record("exit_proxy_side")
                exit_h = (graph_vars["exit_hidden"][:flat]
                          + graph_vars["exit_residual"][:flat])
                normed = model_runner.model.model.norm(exit_h, None)
                _el_full = torch.nn.functional.linear(normed, _replica)
                duet_proxy_fn(_el_full, orig_bs)
                duet_close("exit_proxy_side", _ev_side)
            _done = torch.cuda.Event()
            _done.record(_es)
            model_runner._duet_exit_done_ev = _done
        duet_close("exit_proxy_launch", _ev_el)
        exit_logits = None
    else:
        # ====== Mid-forward: exit logits (norm + lm_head on exit_hidden) ======
        _ev_el = duet_record("exit_logits")
        flat = orig_bs * k_plus_1
        exit_h = graph_vars["exit_hidden"][:flat] + graph_vars["exit_residual"][:flat]
        normed = model_runner.model.model.norm(exit_h, None)
        if getattr(config, "duet_exit_topm_gather", False):
            # B>1 not supported for this gate (docs/duet/13 §6) — the y_tok
            # slice inside _duet_exit_topm_gather crosses seq boundaries.
            assert orig_bs == 1, "duet_exit_topm_gather: B>1 not supported (docs/duet/13 §6)"
            # Rank-local top-M reduction (docs/duet/09 WS3): every rank shrinks
            # its vocab shard to top-M candidates + lse partial + draft-token
            # logit BEFORE the gather. rank 0 gets a raw-candidate dict (same
            # schema as the raw-proxy wire); rank 1+ get None.
            exit_logits = _duet_exit_topm_gather(
                model_runner, normed, input_ids[:flat], config.duet_proxy_topm)
        else:
            # ALL TP ranks call compute_logits → gather participation.
            # rank 0: exit_logits = [flat, V]; rank 1+: exit_logits = None
            exit_logits = model_runner.model.compute_logits(normed, last_only=False)
        duet_close("exit_logits", _ev_el)

        # ====== proxy compute + isend (rank 0 only does real work) ======
        # Batch 3b (docs/duet/08 §1.2): if SSD_PROXY_STREAM=1, dispatch the
        # Policy B compute + isend onto a dedicated proxy_stream so the default
        # stream returns to graph_post.replay() immediately. Cross-stream
        # tensors (exit_logits, and inside duet_proxy_fn: draft_tokens,
        # logits_q, cache_hits — captured from Verifier closure) need
        # record_stream so the caching allocator does not free them while
        # proxy_stream is still reading. NEVER make default stream wait on
        # proxy_stream (that would defeat the purpose).
        _ev = duet_record("exit_proxy_launch")
        if duet_proxy_fn is not None:
            _proxy_stream = getattr(model_runner, "_duet_proxy_stream", None)
            if _proxy_stream is None and os.environ.get("SSD_PROXY_STREAM", "0") == "1":
                _proxy_stream = torch.cuda.Stream(device=model_runner.device)
                model_runner._duet_proxy_stream = _proxy_stream
            if _proxy_stream is not None:
                # Cross-stream lifetime guard for the ONE tensor visible at
                # this boundary. draft_tokens / logits_q / cache_hits are
                # captured in the closure of duet_proxy_fn — Verifier
                # record_stream's them defensively inside _compute_and_send_proxy.
                if exit_logits is not None and torch.is_tensor(exit_logits):
                    exit_logits.record_stream(_proxy_stream)
                # Event on default stream signals "exit_logits ready"; proxy
                # stream waits, does its work, and default stream immediately
                # returns to graph_post below.
                _ev_data_ready = torch.cuda.Event()
                _ev_data_ready.record()
                with torch.cuda.stream(_proxy_stream):
                    _proxy_stream.wait_event(_ev_data_ready)
                    _ev_side = duet_record("exit_proxy_side")
                    duet_proxy_fn(exit_logits, orig_bs)
                    duet_close("exit_proxy_side", _ev_side)
                # NO default stream wait_stream(proxy_stream) here — that would
                # re-serialize (docs/duet/08 §1.3 wrong pattern).
            else:
                _ev_side = duet_record("exit_proxy_side")
                duet_proxy_fn(exit_logits, orig_bs)
                duet_close("exit_proxy_side", _ev_side)
        duet_close("exit_proxy_launch", _ev)

    # ====== graph_post.replay() ======
    _ev = duet_record("graph_post")
    graph_post.replay()
    duet_close("graph_post", _ev)

    # ====== Final logits ======
    _ev_fl = duet_record("final_logits")
    outputs = graph_vars["outputs"][:flat]
    logits = model_runner.model.compute_logits(outputs, last_only)
    duet_close("final_logits", _ev_fl)
    return logits


# ---------- DUET per-phase profiling (zero-sync, additions only) ----------
# Doc: ssd/docs/duet/06-timeline-cleanup-plan.md §4.2-4.5, §5 Phase B.
#
# Design summary:
#   - One CUDA/CPU anchor per process, captured lazily on first duet_record()
#     when SSD_PROFILE_DUET=1. The only allowed syncs are anchor init and dump.
#   - duet_record(label, parent=None) starts a span: records a CUDA event +
#     CPU dispatch ns, snapshots current context (step_id, proc) for fallback.
#   - duet_close(label, start_handle) ends a span and CRITICALLY reads the
#     CURRENT context (especially `status`) at close time so target_spec_wait
#     can be labeled with the hit-class learned only after speculate() returns.
#   - duet_dump(tag) syncs once and computes wall-clock + cuda_ms derived
#     fields per row, writes a single JSON with an _anchor metadata block.
import time as _time_mod_for_duet

PROFILE_DUET = os.environ.get("SSD_PROFILE_DUET", "0") == "1"

# Per-span open record:
#   (idx, label, parent_label, start_ev, end_ev,
#    cpu_dispatch_start_ns, cpu_dispatch_end_ns,
#    open_step_id, open_proc, close_step_id, close_status, close_proc)
_duet_events = []
_duet_idx = 0      # monotonic call index (per process)
_duet_cap_warned = False

# CUDA/CPU anchor (set once per process, lazily on first record)
_duet_anchor_event = None
_duet_anchor_cpu_ns = None
_duet_anchor_device = None

# Profiler context (process-local, single-threaded for the profiled path).
# Used so individual call sites do not need to thread step_id/status/proc
# through every helper signature. duet_close() reads this AT close time.
_duet_context = {"step_id": None, "status": None, "proc": None}


def _ensure_duet_anchor():
    """Lazily initialize the per-process CUDA/CPU anchor.

    Only synchronizes when PROFILE_DUET=1. Idempotent; no-op when off or
    already initialized. This is the only sync in the profile path outside
    duet_dump().
    """
    global _duet_anchor_event, _duet_anchor_cpu_ns, _duet_anchor_device
    if not PROFILE_DUET:
        return
    if _duet_anchor_event is not None:
        return
    # Batch 1e: capture anchor_cpu_ns BETWEEN the two syncs (matches doc
    # 06 §4.2 canonical pattern). Previously the CPU stamp was taken after
    # the second synchronize + event assignment + attribute lookup, adding
    # a systematic 10-100 µs offset that biased cross-process alignment.
    torch.cuda.synchronize()
    _duet_anchor_cpu_ns = _time_mod_for_duet.perf_counter_ns()
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    torch.cuda.synchronize()
    _duet_anchor_event = ev
    _duet_anchor_device = torch.cuda.current_device()


def duet_set_context(step_id=None, status=None, proc=None):
    """Update the process-local profiler context.

    Each argument is applied only if not None, so callers can update a
    subset (e.g. only `status` after speculate() returns). No-op when
    PROFILE_DUET=0 to keep cold path zero-cost.
    """
    if not PROFILE_DUET:
        return
    if step_id is not None:
        _duet_context["step_id"] = step_id
    if status is not None:
        _duet_context["status"] = status
    if proc is not None:
        _duet_context["proc"] = proc


def duet_clear_status():
    """Explicitly clear close-time status. Used between draft requests so a
    new request does not inherit the previous request's hit/miss label."""
    if not PROFILE_DUET:
        return
    _duet_context["status"] = None


def duet_record(label, parent=None):
    """Open a profiling span. Returns an opaque handle for duet_close().

    Captures: start CUDA event, CPU dispatch ns, and a snapshot of
    (step_id, proc) at open time. `status` is intentionally NOT snapshotted
    here — it is read fresh in duet_close() so spans like target_spec_wait
    can pick up status learned after the body runs.
    """
    if not PROFILE_DUET:
        return None
    # Long profile runs used to retain every CUDA Event until process exit.
    # Around 23k live events both chain and tree traces showed an unrelated
    # 0.8-second stall inside whichever span happened to allocate the next
    # event.  Timeline profiles need representative steps, not an unbounded
    # event archive, so allow the run script to stop recording after a safe
    # window without perturbing the serving path further.
    global _duet_cap_warned
    _max_events = int(os.environ.get("SSD_PROFILE_DUET_MAX_EVENTS", "0"))
    if _max_events > 0 and len(_duet_events) >= _max_events:
        if not _duet_cap_warned:
            _duet_cap_warned = True
            print(f"[duet_profile] event cap {_max_events} reached; "
                  "later spans are intentionally not recorded", flush=True)
        return None
    _ensure_duet_anchor()
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    cpu_ns = _time_mod_for_duet.perf_counter_ns()
    # Snapshot open-time context (fallback for step_id/proc).
    open_step_id = _duet_context["step_id"]
    open_proc = _duet_context["proc"]
    return (ev, cpu_ns, label, parent, open_step_id, open_proc)


def duet_close(label, start_handle):
    """Close a profiling span opened by duet_record().

    Reads the CURRENT context (step_id, status, proc) at close time. This
    is essential for target_spec_wait_* whose status is set after
    speculator.speculate() returns. Open-time step_id/proc are used as a
    fallback when the close-time context is unset.
    """
    global _duet_idx
    if start_handle is None:
        return
    start_ev, cpu_start_ns, _open_label, parent, open_step_id, open_proc = start_handle
    end_ev = torch.cuda.Event(enable_timing=True)
    end_ev.record()
    cpu_end_ns = _time_mod_for_duet.perf_counter_ns()
    # Close-time context wins; fall back to open-time snapshot if unset.
    close_step_id = _duet_context["step_id"]
    if close_step_id is None:
        close_step_id = open_step_id
    close_proc = _duet_context["proc"]
    if close_proc is None:
        close_proc = open_proc
    close_status = _duet_context["status"]
    _duet_events.append((
        _duet_idx, label, parent,
        start_ev, end_ev,
        cpu_start_ns, cpu_end_ns,
        close_step_id, close_status, close_proc,
    ))
    _duet_idx += 1


def duet_reset():
    """Reset profiling state — duet_dump/duet_flush call this after writing."""
    global _duet_idx, _duet_anchor_event, _duet_anchor_cpu_ns, \
        _duet_anchor_device, _duet_cap_warned
    _duet_events.clear()
    _duet_idx = 0
    _duet_anchor_event = None
    _duet_anchor_cpu_ns = None
    _duet_anchor_device = None
    _duet_cap_warned = False
    _duet_context["step_id"] = None
    _duet_context["status"] = None
    _duet_context["proc"] = None


def duet_flush(tag, run_id=None):
    """Public API: write JSON and reset. Call between runs or at end of generate().
    run_id: appended to filename to avoid overwrites; defaults to HMS timestamp."""
    return duet_dump(tag, run_id=run_id)


def duet_dump(tag, run_id=None):
    if not _duet_events:
        duet_reset()
        return
    import json
    # Single sync — convert CUDA event times to anchored wall-clock at dump
    # time so the per-step path stays sync-free.
    torch.cuda.synchronize()
    anchor_ev = _duet_anchor_event
    anchor_cpu_ns = _duet_anchor_cpu_ns
    rows = []
    for (
        idx, label, parent,
        start_ev, end_ev,
        cpu_start_ns, cpu_end_ns,
        step_id, status, proc,
    ) in _duet_events:
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
        run_id = _time_mod_for_duet.strftime("%H%M%S")
    path = f"{outdir}/duet_profile_{tag}_{run_id}.json"
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
        "anchor_device": _duet_anchor_device,
        "anchor_note": "GPU event times converted to host monotonic clock",
    }
    payload = [anchor_row] + rows
    with open(path, "w") as f:
        json.dump(payload, f)
    print(f"[duet_profile] {len(rows)} events -> {path}", flush=True)
    duet_reset()
