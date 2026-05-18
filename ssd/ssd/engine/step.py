from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
import torch
from time import perf_counter
from transformers import AutoTokenizer

from ssd.engine.model_runner import ModelRunner
from ssd.engine.sequence import Sequence
from ssd.engine.scheduler import Scheduler
from ssd.engine.helpers.speculate_types import SpeculatorBase, VerifierBase, VerifyResult
from ssd.utils.misc import decode_tokens


class InferenceStep(ABC):

    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler

    @abstractmethod
    def decode(self, seqs: list[Sequence]) -> int:
        pass

    @abstractmethod
    def prefill(self, seqs: list[Sequence]) -> int:
        pass


class AutoRegressiveStep(InferenceStep):

    def __init__(self, scheduler: Scheduler, model_runner: ModelRunner, tokenizer: AutoTokenizer):
        super().__init__(scheduler)
        self.model_runner = model_runner
        self.tokenizer = tokenizer

    def step(self, seqs: list[Sequence], is_prefill: bool) -> int:
        if __debug__:
            print(f'[auto_regressive_step] is_prefill={is_prefill}', flush=True)

        token_ids = self.model_runner.call("run", seqs, is_prefill)

        if __debug__:
            decoded_tokens = decode_tokens(token_ids, self.tokenizer)
            print(f"[auto_regressive_step] generated tokens: {decoded_tokens}", flush=True)

        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        return len(seqs) if not is_prefill else sum(len(seq) for seq in seqs)

    def prefill(self, seqs: list[Sequence]) -> int:
        return self.step(seqs, is_prefill=True)

    def decode(self, seqs: list[Sequence]) -> int:
        return self.step(seqs, is_prefill=False)


class SpecDecodeStep(InferenceStep):

    def __init__(
        self,
        scheduler: Scheduler,
        speculator: SpeculatorBase,
        verifier: VerifierBase,
        eagle: bool,
        tokenizer: AutoTokenizer,
        async_spec: bool,
    ):
        super().__init__(scheduler)
        self.speculator = speculator
        self.verifier = verifier
        self.eagle = eagle
        self.tokenizer = tokenizer
        self.async_spec = async_spec

    def prefill(self, seqs: list[Sequence]) -> int:
        # When doing async speculation and not Eagle, we can do draft and target prefills in parallel.
        if not self.eagle and self.async_spec:
            empty_verify_result = VerifyResult([], [], None)
            self.speculator.prefill(seqs, empty_verify_result)
            verify_result = self.verifier.prefill(seqs, eagle=False)
        else:
            verify_result = self.verifier.prefill(seqs, eagle=self.eagle)
            self.speculator.prefill(seqs, verify_result)

        for seq in seqs:
            assert seq.recovery_token_id is not None
            seq.num_cached_tokens = seq.num_prompt_tokens
            seq.num_draft_cached_tokens = seq.num_prompt_tokens

        return sum(len(seq) for seq in seqs)

    def decode(self, seqs: list[Sequence]) -> int:
        # Scheduler may preempt the only running sequence on a step when KV slot
        # pre-reservation (target_lookahead + draft_lookahead = K+1 + K*MQ_LEN)
        # cannot be satisfied — common at high async_fan_out with B=1. The
        # preempted seq re-enters on the next schedule() call; this step has no
        # work. Without this guard the empty batch cascades through speculate()
        # → speculator_async.valid_k[0] (IndexError) and through the verify CG
        # padding path → torch.cat on empty bt[-1:0] (RuntimeError).
        if not seqs:
            return 0

        _prof = os.environ.get("SSD_PROFILE", "0") == "1"
        if _prof:
            torch.cuda.synchronize()
            _t0 = perf_counter()

        # Save lightweight state instead of expensive clone_spec deep copy.
        # speculate() modifies: token_ids (append+extend), num_tokens, last_token, num_draft_cached_tokens
        # verify() modifies: num_cached_tokens (line 77 of verifier.py)
        # postprocess_speculate() needs the ORIGINAL state to apply new suffixes.
        saved = [(len(seq.token_ids), seq.num_tokens, seq.last_token, seq.num_draft_cached_tokens, seq.num_cached_tokens) for seq in seqs]

        eagle_sentinel = True if self.eagle else None
        in_verify_result = VerifyResult(
            new_suffixes=[],
            recovery_tokens=[],
            eagle_acts=eagle_sentinel,
        )
        #### STEP 1: SPECULATE ####
        # Phase B (docs/mesa/06 §4.4 final paragraph): target_spec_wait wraps
        # the blocking speculate() call. proc + step_id are set inside
        # _speculation_request once the request id is incremented; we set
        # proc here pre-emptively so events on this process are tagged even
        # before the request id is known. status is learned only after
        # speculate() returns and is late-bound via mesa_set_context so
        # mesa_close() picks it up at close time. Label retains the
        # legacy `_{status}` suffix (Option (i)) for back-compat with
        # summarize_ssd_run.py; the row also carries status as a field.
        from ssd.engine.helpers.cudagraph_helpers import (
            mesa_record as _mr, mesa_close as _mc, mesa_set_context as _mctx,
        )
        _mctx(proc="target_rank0")
        _mev_sw = _mr("target_spec_wait")
        speculate_result = self.speculator.speculate(seqs, in_verify_result)
        _status = getattr(speculate_result, "profile_cache_status", None)
        _step_id = getattr(speculate_result, "step_id", None)
        _mctx(step_id=_step_id, status=_status)
        _wait_label = f"target_spec_wait_{_status}" if _status else "target_spec_wait"
        _mc(_wait_label, _mev_sw)

        if _prof:
            torch.cuda.synchronize()
            _t1 = perf_counter()

        if __debug__:
            speculations = speculate_result.speculations
            print(f"[SpecDecodeStep] speculations: {speculations}", flush=True)
            speculations_list = speculations.tolist()

            for i, speculation in enumerate(speculations_list):
                decoded_tokens = decode_tokens(speculation, self.tokenizer)
                print(f"[SpecDecodeStep] speculation {i}: {decoded_tokens}", flush=True)

        #### STEP 2: VERIFY ####
        out_verify_result = self.verifier.verify(seqs, speculate_result, eagle=self.eagle)

        if _prof:
            torch.cuda.synchronize()
            _t2 = perf_counter()

        if __debug__:
            recovery_tokens = out_verify_result.recovery_tokens
            new_suffixes = out_verify_result.new_suffixes
            for i, new_suffix in enumerate(new_suffixes):
                decoded_tokens = decode_tokens(new_suffix + [recovery_tokens[i]], self.tokenizer)
                print(f"[SpecDecodeStep] verification {i}: {decoded_tokens}", flush=True)

        # Restore original seq state before postprocess (undo speculate + verify modifications)
        for seq, (orig_len, orig_nt, orig_lt, orig_ndc, orig_nct) in zip(seqs, saved):
            del seq.token_ids[orig_len:]
            seq.num_tokens = orig_nt
            seq.last_token = orig_lt
            seq.num_draft_cached_tokens = orig_ndc
            seq.num_cached_tokens = orig_nct

        #### STEP 3: POSTPROCESS ####
        _mev_pp = _mr("target_postprocess")
        self.scheduler.postprocess_speculate(
            seqs,
            out_verify_result.new_suffixes,
            out_verify_result.recovery_tokens,
            eagle_acts=out_verify_result.eagle_acts if self.eagle else None,
        )
        _mc("target_postprocess", _mev_pp)

        if _prof:
            torch.cuda.synchronize()
            _t3 = perf_counter()
            cache_hits = speculate_result.cache_hits
            hits_str = f"hits={cache_hits.sum().item()}/{len(cache_hits)}" if cache_hits is not None else ""
            toks = sum(len(s) for s in out_verify_result.new_suffixes)
            print(f"[PROFILE target] handshake={(_t1-_t0)*1000:.2f}ms verify={(_t2-_t1)*1000:.2f}ms postprocess={(_t3-_t2)*1000:.2f}ms total={(_t3-_t0)*1000:.2f}ms {hits_str} toks={toks}", flush=True)

        return sum(len(s) for s in out_verify_result.new_suffixes)
