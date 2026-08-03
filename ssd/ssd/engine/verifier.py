import os
import torch
from time import perf_counter
from transformers import AutoTokenizer

from ssd.engine.sequence import Sequence
from ssd.engine.model_runner import ModelRunner
from ssd.utils.verify import verify
from ssd.engine.helpers.speculate_types import SpeculateResult, VerifyResult, VerifierBase


class Verifier(VerifierBase):
    def __init__(
        self,
        lookahead: int,
        device: torch.device,
        target_model_runner: ModelRunner,
        sampler_x: float | None = None,
        async_fan_out: int | None = None,
        jit_speculate: bool = False,
        tokenizer: AutoTokenizer = None,
        metrics: dict = None,
    ):
        super().__init__(lookahead, device)
        self.target_model_runner = target_model_runner
        self.sampler_x = sampler_x
        self.async_fan_out = async_fan_out
        self.jit_speculate = jit_speculate
        self.tokenizer = tokenizer
        self.metrics = metrics
        # Batch 3 (proxy async send, docs/duet/08 §3): lazy-inited ring buffer
        # for non-blocking proxy send. Only allocated on rank 0 + DUET + gate.
        self._proxy_send_ring = None
        # Counts every ring.send() call — denominator for slot_wait_rate.
        self._proxy_send_call_count = 0

    def drain_proxy_send_ring(self):
        """Public shutdown hook — llm_engine calls this before joining the
        draft process (see docs/duet/08 §3.3).

        Rationale: outstanding dist.isend interacting with a closed process
        group can hang or segfault. Explicit drain from a known shutdown
        hook is required; Python __del__ is unreliable at interpreter exit.
        """
        if self._proxy_send_ring is not None:
            self._proxy_send_ring.drain()
            if os.environ.get("SSD_PROFILE_DUET", "0") == "1":
                outdir = os.environ.get("SSD_PROFILE_DIR", "/tmp")
                try:
                    self._proxy_send_ring.dump_stats(
                        outdir,
                        decode_steps_seen=self._proxy_send_call_count,
                    )
                except Exception as e:
                    print(f"[verifier] proxy_send_ring dump_stats failed: {e}", flush=True)

    def __del__(self):
        """Backup only. Do NOT rely on this as primary cleanup."""
        try:
            self.drain_proxy_send_ring()
        except Exception:
            pass

    def prefill(self, seqs: list[Sequence], eagle: bool = False) -> VerifyResult:
        result = self.target_model_runner.call("run", seqs, True)
        if eagle:
            token_ids, eagle_acts = result
        else:
            token_ids = result

        offset = 0
        for seq, token_id in zip(seqs, token_ids):
            seq.recovery_token_id = token_id
            if eagle:
                seq_len = seq.num_prompt_tokens
                # this doesn't move acts onto cpu does it? 
                seq.last_target_hidden_state = eagle_acts[offset + seq_len - 1].clone()
                offset += seq_len

        return VerifyResult(
            [], # no accepted tokens for prefill, just recovery tokens (first sampled token).
            [seq.recovery_token_id for seq in seqs],
            eagle_acts if eagle else None,
        )

    def verify(self, seqs: list[Sequence], speculate_result: SpeculateResult, eagle: bool = False) -> VerifyResult:
        """Verify speculative tokens using the target model."""
        _prof = os.environ.get("SSD_PROFILE", "0") == "1"
        batch_size = len(seqs)
        config = self.target_model_runner.config

        # Phase B: announce step_id + proc=target_rank0 + status for ALL spans
        # inside verify (graph_pre/post, exit_logits, proxy_compute_send,
        # verify_sample_accept, ...). step_id comes from the matching spec
        # request; status is what speculate() learned. See docs/duet/06 §4.4.
        from ssd.engine.helpers.cudagraph_helpers import duet_set_context as _mctx_v
        _mctx_v(
            step_id=getattr(speculate_result, "step_id", None),
            proc="target_rank0",
            status=getattr(speculate_result, "profile_cache_status", None),
        )

        # DUET: set proxy function on target model runner (rank 0 only)
        if config.duet_enabled:
            async_pg = self.target_model_runner.async_pg
            draft_rank = self.target_model_runner.draft_rank
            # M1 (docs/duet/13 §1/§5): per-step lookahead dispatches on
            # vk_max = max(valid_k) over the batch. Short seqs are padded to
            # vk_max; their REAL width rides in valid_k and is enforced by the
            # per-seq accept clamp in verify() below. The .max().item() here
            # REPLACES the former torch.unique() GPU→CPU sync (not an extra
            # one). For non-hybrid path defaults to self.lookahead = K_long.
            if speculate_result.valid_k is not None and config.duet_phase1_k is not None:
                _step_lookahead = int(speculate_result.valid_k.max().item())
            else:
                _step_lookahead = self.lookahead
            # ===== TRACE point 2: speculator → verifier boundary =====
            _trace_split = (
                os.environ.get("SSD_TRACE_SPLIT_K1K2", "0") == "1"
                and os.environ.get("SSD_FORCE_SPLIT_K1K2", "0") == "1"
                and config.duet_phase1_k is not None
            )
            if _trace_split:
                _K1 = config.duet_phase1_k
                _K2 = config.duet_phase2_k
                _K_max = _K1 if _K1 >= _K2 else _K2
                _vk_list = speculate_result.valid_k.tolist() if speculate_result.valid_k is not None else None
                print(f"[TRACE-split-k1k2 #2 spec→verify] K1={_K1} K2={_K2} K_max={_K_max} "
                      f"valid_k={_vk_list} _step_lookahead={_step_lookahead} "
                      f"speculations.shape={list(speculate_result.speculations.shape)} "
                      f"logits_q.shape={list(speculate_result.logits_q.shape)}",
                      flush=True)
            # ===== END TRACE =====
            # v1 hybrid: _step_lookahead comes from speculate_result.valid_k
            # (uniform per batch at B=1). Passed through call("run", ..., step_lookahead)
            # below so all TP ranks see the same value via SHM.
            self.target_model_runner._duet_step_lookahead = _step_lookahead

            # Slice draft_tokens / logits_q to step_lookahead since target ran
            # only K_short+1 positions on a short-hit step.
            draft_tokens = speculate_result.speculations[:, 1:_step_lookahead + 1]  # [B, vk]
            logits_q = speculate_result.logits_q[:, :_step_lookahead, :]            # [B, vk, V]
            cache_hits = speculate_result.cache_hits                                  # [B] or None

            def _proxy_fn(exit_logits, orig_bs, _vk=_step_lookahead):
                self._compute_and_send_proxy(
                    exit_logits, draft_tokens, logits_q, orig_bs,
                    _vk, async_pg, draft_rank, cache_hits=cache_hits,
                    valid_k=speculate_result.valid_k)

            self.target_model_runner._duet_proxy_fn = _proxy_fn

        if _prof:
            torch.cuda.synchronize()
            _vt0 = perf_counter()

        _pt = os.environ.get("SSD_PROFILE_TARGET", "0") == "1"
        _tv0 = perf_counter()
        # v1 hybrid: pass step_lookahead so it broadcasts to all TP ranks via
        # SHM, keeping verify CG bucket dispatch in sync.
        _step_lh_arg = getattr(self.target_model_runner, "_duet_step_lookahead", None) \
            if config.duet_enabled and config.duet_phase1_k is not None else None
        result = self.target_model_runner.call(
            "run", seqs, False, False, True, None, _step_lh_arg)

        # DUET: clear proxy function
        if config.duet_enabled:
            self.target_model_runner._duet_proxy_fn = None

        if _prof:
            torch.cuda.synchronize()
            _vt1 = perf_counter()

        if _pt:
            torch.cuda.synchronize()
            _vt_call = perf_counter()
            print(f"[PROFILE verifier] target_call={(_vt_call-_tv0)*1000:.2f}ms eagle={eagle} bs={batch_size}", flush=True)

        if eagle:
            logits_p_flat, eagle_acts_flat = result
        else:
            logits_p_flat = result

        # v1 hybrid: per-step variable lookahead. _step_lookahead read above
        # from speculate_result.valid_k. For non-hybrid path it equals
        # self.lookahead (= K_long).
        _step_lookahead = getattr(self.target_model_runner, "_duet_step_lookahead", self.lookahead)
        for s in seqs:
            s.num_cached_tokens += _step_lookahead + 1

        logits_p = logits_p_flat.view(
            batch_size, _step_lookahead + 1, -1)  # [b, vk+1, v]

        # Build per-seq temps for target verify and draft q respectively.
        temps_target = [seq.temperature for seq in seqs]
        temps_draft = [
            seq.draft_temperature if seq.draft_temperature is not None else seq.temperature
            for seq in seqs
        ]
        temperatures_target = torch.tensor(temps_target, dtype=torch.float32, device=self.device)
        temperatures_draft = torch.tensor(temps_draft, dtype=torch.float32, device=self.device)

        from ssd.engine.helpers.cudagraph_helpers import duet_record as _mr, duet_close as _mc
        _mev_vs = _mr("verify_sample_accept")
        new_suffixes, recovery_tokens = verify(
            logits_p=logits_p,
            logits_q=speculate_result.logits_q,
            speculations=speculate_result.speculations,
            temperatures_target=temperatures_target,
            temperatures_draft=temperatures_draft,
            cache_hits=speculate_result.cache_hits,
            sampler_x=self.sampler_x,
            async_fan_out=self.async_fan_out,
            jit_speculate=self.jit_speculate,
            valid_k=speculate_result.valid_k,
        )
        _mc("verify_sample_accept", _mev_vs)

        self.metrics["target_verify_times"].append(perf_counter() - _tv0)

        if _prof:
            torch.cuda.synchronize()
            _vt2 = perf_counter()
            print(f"[PROFILE verify] target_fwd={(_vt1-_vt0)*1000:.2f}ms verify_compute={(_vt2-_vt1)*1000:.2f}ms", flush=True)


        # # Debug: print recovery tokens detokenized
        if __debug__ and recovery_tokens is not None and len(recovery_tokens) > 0:
            recovery_texts = []
            for token in recovery_tokens:
                try:
                    text = self.tokenizer.decode([token], skip_special_tokens=False)
                    recovery_texts.append(text)
                except Exception:
                    recovery_texts.append(f"<token_id:{token}>")
            print(f"[verify] recovery tokens: {recovery_texts}", flush=True)

        self.metrics["accepted_suffix_lens_with_recovery"].extend(
            [len(s) for s in new_suffixes])

        # For async mode, also track accepted suffix lengths only for cache hits
        if speculate_result.cache_hits is not None:
            _ch_cpu = speculate_result.cache_hits.cpu()
            self.metrics["cache_hits"].append(_ch_cpu.float().mean().item())
            # DUET-only: split cache hits by which phase populated them.
            # Non-DUET paths send all-zero phase_source and the rates stay 0.
            if speculate_result.phase_source is not None:
                _ps_cpu = speculate_result.phase_source.cpu()
                self.metrics["phase1_hits"].append((_ps_cpu == 1).float().mean().item())
                self.metrics["phase2_hits"].append((_ps_cpu == 2).float().mean().item())
            for i, suffix_len in enumerate([len(s) for s in new_suffixes]):
                if _ch_cpu[i] == 1:
                    self.metrics["accepted_suffix_lens_on_hit"].append(suffix_len)
                    if speculate_result.phase_source is not None:
                        _accepted_len = max(0, suffix_len - 1)
                        _src = int(_ps_cpu[i].item())
                        if _src == 1:
                            self.metrics["accepted_lens_phase1_hit"].append(_accepted_len)
                        elif _src == 2:
                            self.metrics["accepted_lens_phase2_hit"].append(_accepted_len)
                else:
                    self.metrics["accepted_suffix_lens_on_miss"].append(suffix_len)

        # Print mean length of new suffixes for monitoring
        if __debug__ and new_suffixes:
            mean_suffix_len = sum([len(suffix) for suffix in new_suffixes]) / len(new_suffixes)
            print(f"[verify] mean new suffix length: {mean_suffix_len:.2f}", flush=True)

        eagle_acts = None
        if eagle:
            eagle_acts = eagle_acts_flat.view(batch_size, self.lookahead + 1, -1)
        
        return VerifyResult(
            new_suffixes=new_suffixes,
            recovery_tokens=recovery_tokens,
            eagle_acts=eagle_acts,
        )

    def _compute_and_send_proxy(self, exit_logits, draft_tokens, logits_q,
                                 B, K, async_pg, draft_rank, cache_hits=None,
                                 valid_k=None):
        """Compute DUET proxy from early-exit logits and send to draft.

        Args:
            exit_logits: [B*(K+1), V] — norm+lm_head+TP gather done. None on non-rank-0.
            draft_tokens: [B, K] — draft's speculated tokens
            logits_q: [B, K, V] — draft model logits
            cache_hits: [B] or None
            valid_k: [B] or None — per-seq real suffix width (M6,
                docs/duet/13). K is the batch's vk_max; a short seq's
                columns >= vk_i are padding and get α̂ = 0 so its h mass
                past vk_i is 0 (the design §3 claim, implemented in M6).
        """
        import torch.distributed as dist
        from ssd.utils.async_helpers.nccl_pack import send_int64
        config = self.target_model_runner.config
        top_k = config.duet_proxy_top_k
        _detail_profile = os.environ.get("SSD_PROFILE_DUET_DETAIL", "0") == "1"
        if _detail_profile:
            from ssd.engine.helpers.cudagraph_helpers import duet_record as _mr_d, duet_close as _mc_d
        else:
            _mr_d = _mc_d = None

        # === Exit top-M gather (SSD_DUET_EXIT_TOPM_GATHER=1, docs/duet/09) ===
        # exit_logits arrives as the merged raw-candidate dict from
        # _duet_exit_topm_gather; run Policy B on the candidate set (exact
        # same math/wire as the full-vocab path — see duet_policy).
        if isinstance(exit_logits, dict):
            # B>1 not supported for this gate (docs/duet/13 §6).
            assert B == 1, "duet_exit_topm_gather: B>1 not supported (docs/duet/13 §6)"
            from ssd.utils.async_helpers.duet_policy import policy_b_from_candidates
            chosen = policy_b_from_candidates(
                exit_logits, logits_q[0], draft_tokens[0, :K], K,
                config.duet_proxy_top_k, config.duet_proxy_wire_N)
            send_int64(async_pg, draft_rank,
                       chosen["chosen_pos"], chosen["chosen_tok"])
            return

        if exit_logits.dim() == 2:
            exit_logits = exit_logits.view(B, K + 1, -1)  # [B, K+1, V]

        # Batch 3b: when this callback runs on a dedicated proxy_stream (see
        # cudagraph_helpers.run_duet_verify_cudagraph), the caching allocator
        # must be told the closure-captured default-stream tensors are still
        # alive. `exit_logits.record_stream` is done at the dispatch boundary;
        # here we cover draft_tokens / logits_q / cache_hits which are read
        # below (docs/duet/08 §2 cross-stream lifetime checklist).
        if (os.environ.get("SSD_PROXY_STREAM", "0") == "1"
                or config.duet_exit_replica):
            _cur = torch.cuda.current_stream()
            if draft_tokens is not None:
                draft_tokens.record_stream(_cur)
            if logits_q is not None:
                logits_q.record_stream(_cur)
            if cache_hits is not None:
                cache_hits.record_stream(_cur)
            if valid_k is not None:
                valid_k.record_stream(_cur)

        # === Raw-proxy wire (SSD_DUET_PROXY_ON_DRAFT=1, docs/duet/09 WS3) ===
        # Send top-M exit candidates (ids + logits) + logsumexp + the exit
        # logit at each draft token; the draft reconstructs exact p_E for
        # those candidates and computes Policy B locally in its proxy_wait
        # window. Removes the 2× fp32 softmax + residual + double topk +
        # pack from the target verify critical path. Fixed worst-case
        # (K_max) payload; short steps zero-pad the tail rows.
        if config.duet_proxy_on_draft:
            # B>1 not supported for this gate (docs/duet/13 §6).
            assert B == 1, "duet_proxy_on_draft: B>1 not supported (docs/duet/13 §6)"
            M = config.duet_proxy_topm
            K_max = max(config.duet_phase1_k, config.duet_phase2_k)
            Kp1w = K_max + 1
            el = exit_logits[0].float()                        # [K+1, V]
            lse = torch.logsumexp(el, dim=-1)                  # [K+1]
            top_lg, top_ids = el.topk(M, dim=-1)               # [K+1, M]
            y_lg = el[:K].gather(
                1, draft_tokens[0, :K].unsqueeze(1)).squeeze(1)  # [K]
            dev = el.device
            ids_pad = torch.zeros(Kp1w, M, dtype=torch.int64, device=dev)
            lg_pad = torch.zeros(Kp1w, M, dtype=torch.float64, device=dev)
            lse_pad = torch.zeros(Kp1w, dtype=torch.float64, device=dev)
            y_pad = torch.zeros(K_max, dtype=torch.float64, device=dev)
            ids_pad[:K + 1] = top_ids
            lg_pad[:K + 1] = top_lg.to(torch.float64)
            lse_pad[:K + 1] = lse.to(torch.float64)
            y_pad[:K] = y_lg.to(torch.float64)
            payload = torch.cat([
                ids_pad.view(-1),
                lg_pad.view(-1).view(torch.int64),
                lse_pad.view(torch.int64),
                y_pad.view(torch.int64),
            ])
            send_int64(async_pg, draft_rank, payload)
            return

        _ev_compute = _mr_d("proxy_compute") if _detail_profile else None
        # p_E (early-exit proxy), p_D (draft)
        p_E = torch.softmax(exit_logits[:, :K, :].float(), dim=-1)  # [B, K, V]
        p_D = torch.softmax(logits_q.float(), dim=-1)                # [B, K, V]

        # Accept probability proxy: â_i = min(1, p_E(y_i) / p_D(y_i))
        gather_idx = draft_tokens.unsqueeze(-1)  # [B, K, 1]
        p_E_y = p_E.gather(2, gather_idx).squeeze(-1)  # [B, K]
        p_D_y = p_D.gather(2, gather_idx).squeeze(-1)  # [B, K]
        accept_probs = (p_E_y / (p_D_y + 1e-10)).clamp(max=1.0)  # [B, K]

        # Residual proxy: [p_E - p_D]_+
        residual = (p_E - p_D).clamp(min=0)  # [B, K, V]
        residual.scatter_(2, gather_idx, 0.0)  # exclude draft token y_i
        topk_probs, topk_ids = residual.topk(top_k, dim=-1)  # [B, K, top_k]
        topk_sum = topk_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        topk_probs = topk_probs / topk_sum

        # M6 (docs/duet/13): zero α̂ at a short seq's padded columns
        # (j >= vk_i). Then cumprod stalls at the real prefix, h[b, vk_i]
        # carries the FULL all-real-accept mass ( = cumprod[vk_i-1]·(1-0) )
        # and h[b, j > vk_i] = 0 → P_iv mass 0 → chosen never lands beyond
        # vk_i (the M1 design §3 claim; previously unimplemented, so short
        # seqs leaked budget to unreachable positions). No-op when valid_k
        # is uniform == K (B=1 / all-long batches).
        if valid_k is not None:
            _pad_cols = (torch.arange(K, device=accept_probs.device)[None, :]
                         >= valid_k[:, None])                       # [B, K]
            accept_probs = accept_probs.masked_fill(_pad_cols, 0.0)

        # Cache miss handling: if jit_speculate=False, miss rows have random logits_q
        if cache_hits is not None and not config.jit_speculate:
            miss_mask = ~cache_hits.to(torch.bool)
            if miss_mask.any():
                accept_probs[miss_mask] = 0.0
                miss_p_E = p_E[miss_mask].clone()
                miss_p_E.scatter_(2, gather_idx[miss_mask], 0.0)
                miss_topk_probs, miss_topk_ids = miss_p_E.topk(top_k, dim=-1)
                topk_ids[miss_mask] = miss_topk_ids
                topk_probs[miss_mask] = miss_topk_probs / miss_topk_probs.sum(-1, keepdim=True).clamp(min=1e-10)

        # Compute h_i (first reject position distribution) and fan_out_list on target side
        # This removes ~0.5ms from draft critical path
        # M1 (docs/duet/13 §3): batched over B — h [B, K+1]. At B=1 this is
        # numerically identical to the former accept_probs[0] single-seq path.
        # Tier-3: per-step budget helper (K = step vk_max, varies per step —
        # a DIFFERENT quantity from the constant duet_proxy_total_budget;
        # == pfo*(K+1) exactly when duet_p2_budget is unset).
        proxy_fan_out_total = config.duet_p2_budget_at(K)
        cumprod = torch.cumprod(accept_probs, dim=1)     # [B, K]
        h = torch.zeros(B, K + 1, device=accept_probs.device)
        h[:, 0] = 1 - accept_probs[:, 0]
        if K > 1:
            h[:, 1:K] = cumprod[:, :-1] * (1 - accept_probs[:, 1:])
        h[:, K] = cumprod[:, -1]  # all-accept
        if _detail_profile:
            _mc_d("proxy_compute", _ev_compute)

        # === Policy B unified K+1 path (default since 2026-04-29) ===
        # docs/duet/05-policy-b-fix.md Step 2.
        # Replaces legacy Policy A/B branches below. GPU-only, no .cpu().tolist().
        # Wire schema (Policy B): {chosen_pos[N], chosen_tok[N]} score-sorted.
        if config.duet_policy == "b":
            _ev_pack = _mr_d("proxy_pack") if _detail_profile else None
            # pos == K (all-accept): target's full distribution at exit_logits[:, K, :].
            # Note: existing p_E only covers [:, :K, :] (line 246) — read separately.
            pE_K = torch.softmax(exit_logits[:, K, :].float(), dim=-1)        # [B, V]
            pE_K_topk_probs, pE_K_topk_ids = pE_K.topk(top_k, dim=-1)         # [B, top_k]
            pE_K_topk_probs = pE_K_topk_probs / pE_K_topk_probs.sum(-1, keepdim=True).clamp(min=1e-10)

            # concat positions [0..K-1] (residual) + position K (all-accept) → [B, K+1, top_k]
            correction_topk_probs = torch.cat(
                [topk_probs, pE_K_topk_probs.unsqueeze(1)], dim=1)            # [B, K+1, top_k]
            correction_topk_ids = torch.cat(
                [topk_ids, pE_K_topk_ids.unsqueeze(1)], dim=1)                # [B, K+1, top_k]

            # Global top-N. wire_N is config-fixed (K_max+1 worst-case).
            # top_k auto-raise (config.py) ensures (K+1)*top_k ≥ wire_N for current step's K.
            # M1 (docs/duet/13 §3): per-seq flatten+topk over the B axis. At
            # B=1 the [1, ...] row-wise topk is identical to the old flat topk.
            wire_N = config.duet_proxy_wire_N
            P_iv = h.unsqueeze(-1) * correction_topk_probs                    # [B, K+1, top_k]
            _, top_idx = P_iv.flatten(1).topk(wire_N, dim=-1)                 # [B, wire_N]
            chosen_pos = top_idx // top_k                                     # [B, wire_N], ∈ [0, K]
            chosen_tok = correction_topk_ids.flatten(1).gather(1, top_idx)    # [B, wire_N]
            if _detail_profile:
                _mc_d("proxy_pack", _ev_pack)

            # ===== TRACE point 3 (Policy B) =====
            if (
                os.environ.get("SSD_TRACE_SPLIT_K1K2", "0") == "1"
                and os.environ.get("SSD_FORCE_SPLIT_K1K2", "0") == "1"
                and config.duet_phase1_k is not None
            ):
                _K1 = config.duet_phase1_k
                _K2 = config.duet_phase2_k
                print(f"[TRACE-split-k1k2 #3 proxy_compute policy=b] K1={_K1} K2={_K2} "
                      f"K(=valid_k)={K} wire_N={wire_N} top_k={top_k} "
                      f"chosen_pos.shape={list(chosen_pos.shape)}",
                      flush=True)
            # NCCL send: chosen_pos [B*wire_N int64] + chosen_tok [B*wire_N int64]
            # (M1, docs/duet/13 §3 — flattened batched wire; at B=1 the length
            # is 2*wire_N, byte-identical to the old single-seq wire).
            # Batch 3a (docs/duet/08 §3): env-gated non-blocking send via ring
            # buffer. SSD_ASYNC_PROXY_SEND=1 → dist.isend + persistent buf;
            # else keep the legacy blocking send. Ring is lazy-inited on first
            # use so non-rank-0 processes (which never fire this callback)
            # don't allocate.
            _ev_send = _mr_d("proxy_send") if _detail_profile else None
            if os.environ.get("SSD_ASYNC_PROXY_SEND", "0") == "1":
                if self._proxy_send_ring is None:
                    from ssd.utils.async_helpers.nccl_pack import AsyncSendRing
                    # Buf = 2 × B_max × wire_N int64 (chosen_pos + chosen_tok
                    # packed; B_max = config.max_num_seqs, actual send length
                    # is 2·B·wire_N for the step's B).
                    # 2 slots is enough at hit-dominated rates (docs/duet/08 §3.2);
                    # if slot_wait_rate > 1% in Phase 4, bump to 4.
                    self._proxy_send_ring = AsyncSendRing(
                        n_slots=2,
                        buf_size=2 * config.max_num_seqs * config.duet_proxy_wire_N,
                        device=chosen_pos.device,
                        pg=async_pg,
                    )
                self._proxy_send_ring.send(
                    draft_rank, chosen_pos.reshape(-1), chosen_tok.reshape(-1))
                self._proxy_send_call_count += 1
            else:
                send_int64(async_pg, draft_rank,
                           chosen_pos.reshape(-1), chosen_tok.reshape(-1))
            if _detail_profile:
                _mc_d("proxy_send", _ev_send)
            return

        # === Legacy Policy A / non-unified Policy B: removed (2026-07) ===
        raise RuntimeError(
            "duet_policy='a' / non-unified Policy B proxy path was removed "
            "(2026-07). Only duet_policy='b' (unified K+1) is supported; the "
            "removed implementation is preserved in git history "
            "(feat/mesa-proxy-async-overlap @ 19c8f73 and earlier).")
