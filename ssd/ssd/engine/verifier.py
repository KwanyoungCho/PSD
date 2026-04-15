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
            draft_tokens = speculate_result.speculations[:, 1:]  # [B, K]
            logits_q = speculate_result.logits_q                 # [B, K, V]
            cache_hits = speculate_result.cache_hits              # [B] or None

            def _proxy_fn(exit_logits, orig_bs):
                self._compute_and_send_proxy(
                    exit_logits, draft_tokens, logits_q, orig_bs,
                    self.lookahead, async_pg, draft_rank, cache_hits=cache_hits)

            self.target_model_runner._mesa_proxy_fn = _proxy_fn

        if _prof:
            torch.cuda.synchronize()
            _vt0 = perf_counter()

        _pt = os.environ.get("SSD_PROFILE_TARGET", "0") == "1"
        _tv0 = perf_counter()
        result = self.target_model_runner.call("run", seqs, False, False, True)

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

        for s in seqs:
            s.num_cached_tokens += self.lookahead + 1

        logits_p = logits_p_flat.view(
            batch_size, self.lookahead + 1, -1)  # [b, k+1, v]

        # Build per-seq temps for target verify and draft q respectively.
        temps_target = [seq.temperature for seq in seqs]
        temps_draft = [
            seq.draft_temperature if seq.draft_temperature is not None else seq.temperature
            for seq in seqs
        ]
        temperatures_target = torch.tensor(temps_target, dtype=torch.float32, device=self.device)
        temperatures_draft = torch.tensor(temps_draft, dtype=torch.float32, device=self.device)

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

        # NCCL send (packed, blocking — 280 bytes ~3μs)
        send_int64(async_pg, draft_rank,
                   accept_probs.view(-1).to(torch.float32).view(torch.int32).to(torch.int64),
                   topk_ids.reshape(-1),
                   topk_probs.view(-1).to(torch.float32).view(torch.int32).to(torch.int64))
