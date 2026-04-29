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

        # MESA: set proxy function on target model runner (rank 0 only)
        if config.mesa_enabled:
            async_pg = self.target_model_runner.async_pg
            draft_rank = self.target_model_runner.draft_rank
            # v1 hybrid: per-step lookahead (= valid_k of the matched cache row).
            # speculate_result.valid_k is uniform across the B=1 batch (Config
            # invariant). For non-hybrid path defaults to self.lookahead = K_long.
            if speculate_result.valid_k is not None and config.mesa_phase1_k is not None:
                _vk_unique = torch.unique(speculate_result.valid_k)
                assert _vk_unique.numel() == 1, \
                    f"v1 expects uniform valid_k across batch (B=1), got {speculate_result.valid_k}"
                _step_lookahead = int(_vk_unique.item())
            else:
                _step_lookahead = self.lookahead
            # ===== TRACE point 2: speculator → verifier boundary =====
            _trace_split = (
                os.environ.get("SSD_TRACE_SPLIT_K1K2", "0") == "1"
                and os.environ.get("SSD_FORCE_SPLIT_K1K2", "0") == "1"
                and config.mesa_phase1_k is not None
            )
            if _trace_split:
                _K1 = config.mesa_phase1_k
                _K2 = config.mesa_phase2_k
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
            self.target_model_runner._mesa_step_lookahead = _step_lookahead

            # Slice draft_tokens / logits_q to step_lookahead since target ran
            # only K_short+1 positions on a short-hit step.
            draft_tokens = speculate_result.speculations[:, 1:_step_lookahead + 1]  # [B, vk]
            logits_q = speculate_result.logits_q[:, :_step_lookahead, :]            # [B, vk, V]
            cache_hits = speculate_result.cache_hits                                  # [B] or None

            def _proxy_fn(exit_logits, orig_bs, _vk=_step_lookahead):
                self._compute_and_send_proxy(
                    exit_logits, draft_tokens, logits_q, orig_bs,
                    _vk, async_pg, draft_rank, cache_hits=cache_hits)

            self.target_model_runner._mesa_proxy_fn = _proxy_fn

        if _prof:
            torch.cuda.synchronize()
            _vt0 = perf_counter()

        _pt = os.environ.get("SSD_PROFILE_TARGET", "0") == "1"
        _tv0 = perf_counter()
        # v1 hybrid: pass step_lookahead so it broadcasts to all TP ranks via
        # SHM, keeping verify CG bucket dispatch in sync.
        _step_lh_arg = getattr(self.target_model_runner, "_mesa_step_lookahead", None) \
            if config.mesa_enabled and config.mesa_phase1_k is not None else None
        result = self.target_model_runner.call(
            "run", seqs, False, False, True, None, _step_lh_arg)

        # MESA: clear proxy function
        if config.mesa_enabled:
            self.target_model_runner._mesa_proxy_fn = None

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
        _step_lookahead = getattr(self.target_model_runner, "_mesa_step_lookahead", self.lookahead)
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

        from ssd.engine.helpers.cudagraph_helpers import mesa_record as _mr, mesa_close as _mc
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
            # MESA-only: split cache hits by which phase populated them.
            # Non-MESA paths send all-zero phase_source and the rates stay 0.
            if speculate_result.phase_source is not None:
                _ps_cpu = speculate_result.phase_source.cpu()
                self.metrics["phase1_hits"].append((_ps_cpu == 1).float().mean().item())
                self.metrics["phase2_hits"].append((_ps_cpu == 2).float().mean().item())
            for i, suffix_len in enumerate([len(s) for s in new_suffixes]):
                if _ch_cpu[i] == 1:
                    self.metrics["accepted_suffix_lens_on_hit"].append(suffix_len)
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
                                 B, K, async_pg, draft_rank, cache_hits=None):
        """Compute MESA proxy from early-exit logits and send to draft.

        Args:
            exit_logits: [B*(K+1), V] — norm+lm_head+TP gather done. None on non-rank-0.
            draft_tokens: [B, K] — draft's speculated tokens
            logits_q: [B, K, V] — draft model logits
            cache_hits: [B] or None
        """
        import torch.distributed as dist
        from ssd.utils.async_helpers.nccl_pack import send_int64
        config = self.target_model_runner.config
        top_k = config.mesa_proxy_top_k

        if exit_logits.dim() == 2:
            exit_logits = exit_logits.view(B, K + 1, -1)  # [B, K+1, V]

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
        proxy_fan_out_total = config.mesa_proxy_fan_out * (K + 1)  # total proxy budget
        cumprod = torch.cumprod(accept_probs[0], dim=0)  # [K] (B=1 scope)
        h = torch.zeros(K + 1, device=accept_probs.device)
        h[0] = 1 - accept_probs[0, 0]
        if K > 1:
            h[1:K] = cumprod[:-1] * (1 - accept_probs[0, 1:])
        h[K] = cumprod[-1]  # all-accept

        # === Policy B unified K+1 path (default since 2026-04-29) ===
        # docs/mesa/05-policy-b-fix.md Step 2.
        # Replaces legacy Policy A/B branches below. GPU-only, no .cpu().tolist().
        # Wire schema (Policy B): {chosen_pos[N], chosen_tok[N]} score-sorted.
        if config.mesa_policy == "b":
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
            wire_N = config.mesa_proxy_wire_N
            P_iv = h.view(K + 1, 1) * correction_topk_probs[0]                # [K+1, top_k]
            _, top_idx = P_iv.flatten().topk(wire_N)                          # [wire_N]
            chosen_pos = top_idx // top_k                                     # [wire_N], ∈ [0, K]
            chosen_tok = correction_topk_ids[0].view(-1).gather(0, top_idx)   # [wire_N]

            # ===== TRACE point 3 (Policy B) =====
            if (
                os.environ.get("SSD_TRACE_SPLIT_K1K2", "0") == "1"
                and os.environ.get("SSD_FORCE_SPLIT_K1K2", "0") == "1"
                and config.mesa_phase1_k is not None
            ):
                _K1 = config.mesa_phase1_k
                _K2 = config.mesa_phase2_k
                print(f"[TRACE-split-k1k2 #3 proxy_compute policy=b] K1={_K1} K2={_K2} "
                      f"K(=valid_k)={K} wire_N={wire_N} top_k={top_k} "
                      f"chosen_pos.shape={list(chosen_pos.shape)}",
                      flush=True)
            # NCCL send: chosen_pos [wire_N int64] + chosen_tok [wire_N int64]
            send_int64(async_pg, draft_rank, chosen_pos, chosen_tok)
            return

        # === Legacy Policy A / Policy B (dead branches) ===
        # TODO(policy-a): Policy A retained as dead branch. Default is "b" since
        # 2026-04-29; see docs/mesa/05-policy-b-fix.md Section 3.1.

        # Budget allocation
        if h[:K].sum() < 1e-6:
            # All accept expected → uniform fan_out (both policies)
            fan_out_list = [proxy_fan_out_total // (K + 1)] * (K + 1)
            remainder = proxy_fan_out_total - sum(fan_out_list)
            for i in range(remainder):
                fan_out_list[i] += 1
        elif config.mesa_policy == "b":
            # Policy B: joint (i, v) ranking by P̂(i, v) = ĥ_i × r̂_i(v)
            # All-accept (pos=K) has no correction tokens → allocate proportionally to h[K],
            # remaining budget over (pos<K, v) by P̂ topk. Reorder topk_ids/topk_probs so
            # Policy-B-chosen tokens come first per position — draft's existing "first
            # valid fo[pos]" selection then naturally consumes them.
            fo_K = int(round(h[K].item() / h.sum().item() * proxy_fan_out_total))
            fo_K = max(0, min(fo_K, proxy_fan_out_total))
            remaining = proxy_fan_out_total - fo_K
            # K * top_k >> remaining (K=6, top_k≥17, remaining ≤ pfo*K) so topk has enough.
            topN = min(remaining, K * top_k)
            P_iv = h[:K].view(K, 1) * topk_probs[0]                 # [K, top_k]
            _, top_idx = P_iv.reshape(-1).topk(topN)                # [topN]
            positions = (top_idx // top_k).cpu().tolist()
            ranks = (top_idx % top_k).cpu().tolist()

            fan_out_list = [0] * (K + 1)
            fan_out_list[K] = fo_K
            new_topk_ids = topk_ids.clone()
            new_topk_probs = topk_probs.clone()
            for pos in range(K):
                chosen = [r for p, r in zip(positions, ranks) if p == pos]
                fan_out_list[pos] = len(chosen)
                rest = [r for r in range(top_k) if r not in chosen]
                new_order = torch.tensor(chosen + rest, device=topk_ids.device, dtype=torch.long)
                new_topk_ids[0, pos] = topk_ids[0, pos, new_order]
                new_topk_probs[0, pos] = topk_probs[0, pos, new_order]
            topk_ids, topk_probs = new_topk_ids, new_topk_probs
            # Padding underflow guard: if topN < remaining (impossible in normal config but
            # keep invariant), top up the largest fan_out_list[i<K] entries.
            shortage = remaining - topN
            for _ in range(shortage):
                pos = int(torch.tensor(fan_out_list[:K]).argmax().item())
                fan_out_list[pos] += 1
        else:
            # Policy A: h_i proportional per-position
            raw = (h / h.sum() * proxy_fan_out_total).floor().int()
            remainder = proxy_fan_out_total - raw.sum().item()
            _, sorted_idx = h.sort(descending=True)
            for i in range(int(remainder)):
                raw[sorted_idx[i]] += 1
            fan_out_list = raw.tolist()

        fan_out_tensor = torch.tensor(fan_out_list, dtype=torch.int64, device=accept_probs.device)

        # v1 hybrid: wire payload is fixed max-size (K_long-based) per the plan's
        # Wire Contracts. When the per-step compute K (= step_lookahead) is
        # smaller than K_long, pad the wire tensors with zeros up to K_long.
        K_wire = config.speculate_k
        if K < K_wire:
            pad_pos = K_wire - K
            fan_out_pad = torch.zeros(pad_pos, dtype=fan_out_tensor.dtype, device=fan_out_tensor.device)
            fan_out_tensor = torch.cat([fan_out_tensor, fan_out_pad])  # [K_long+1]
            tok_pad = torch.zeros(B, pad_pos, top_k, dtype=topk_ids.dtype, device=topk_ids.device)
            topk_ids = torch.cat([topk_ids, tok_pad], dim=1)            # [B, K_long, top_k]
            prb_pad = torch.zeros(B, pad_pos, top_k, dtype=topk_probs.dtype, device=topk_probs.device)
            topk_probs = torch.cat([topk_probs, prb_pad], dim=1)         # [B, K_long, top_k]

        # ===== TRACE point 3: target _compute_and_send_proxy =====
        if (
            os.environ.get("SSD_TRACE_SPLIT_K1K2", "0") == "1"
            and os.environ.get("SSD_FORCE_SPLIT_K1K2", "0") == "1"
            and config.mesa_phase1_k is not None
        ):
            _K1 = config.mesa_phase1_k
            _K2 = config.mesa_phase2_k
            _K_max = _K1 if _K1 >= _K2 else _K2
            assert K in (_K1, _K2), (
                f"split-k1k2 proxy compute: K={K} not in {{K1={_K1}, K2={_K2}}}"
            )
            # per-position topk_probs sum (real positions [0..K-1], padded [K..K_long-1])
            _per_pos_sum = []
            for _p in range(min(_K2, topk_probs.shape[1])):
                _per_pos_sum.append(round(float(topk_probs[0, _p].sum().item()), 4))
            print(f"[TRACE-split-k1k2 #3 proxy_compute] K1={_K1} K2={_K2} K_max={_K_max} "
                  f"K(=valid_k passed in)={K} K_long={config.speculate_k} "
                  f"fan_out_list={fan_out_list} "
                  f"topk_ids.shape={list(topk_ids.shape)} "
                  f"topk_probs.shape={list(topk_probs.shape)} "
                  f"topk_probs_sum_per_pos[0..K2-1]={_per_pos_sum}",
                  flush=True)
        # ===== END TRACE =====
        # NCCL send: fan_out_list [K_long+1] + topk_ids [B,K_long,top_k] + topk_probs [B,K_long,top_k]
        send_int64(async_pg, draft_rank,
                   fan_out_tensor,
                   topk_ids.reshape(-1),
                   topk_probs.view(-1).to(torch.float32).view(torch.int32).to(torch.int64))
