import os
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from ssd.engine.helpers.speculate_types import SpeculateResult, VerifyResult, SpeculatorBase
from ssd.engine.helpers.runner_helpers import prepare_prefill_payload
from ssd.engine.sequence import Sequence
from ssd.utils.misc import decode_tokens
from ssd.utils.async_helpers.nccl_pack import send_int64


class SpeculatorAsync(SpeculatorBase):

    def __init__(
        self,
        lookahead: int,
        device: torch.device,
        async_fan_out: int,
        max_blocks: int,
        vocab_size: int,
        draft_dtype: torch.dtype,
        kvcache_block_size: int,
        max_model_len: int,
        async_pg: dist.ProcessGroup,
        draft_runner_rank: int,
        tokenizer: AutoTokenizer,
        verbose: bool,
    ):
        super().__init__(lookahead, device)
        self.async_fan_out = async_fan_out
        self.max_blocks = max_blocks
        self.vocab_size = vocab_size
        self.draft_dtype = draft_dtype
        self.kvcache_block_size = kvcache_block_size
        self.max_model_len = max_model_len
        self.async_pg = async_pg
        self.draft_runner_rank = draft_runner_rank
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.K = lookahead

        # Aligned-timeline (Phase B, docs/mesa/06 §4.4): monotonically
        # increasing per-process spec request id. Mirrored on draft via the
        # spec metadata tensor. Always populated; sub-µs cost.
        self._request_step_id = 0

        # Pre-allocate handshake send/recv buffers (reused every step)
        self._alloc_handshake_bufs(1)

        # Pre-allocate speculate() output buffers (avoid torch.tensor(device=cuda) sync)
        self._recovery_buf = torch.empty(1, dtype=torch.int64, device=device)
        self._speculations_buf = torch.empty(1, lookahead + 1, dtype=torch.int64, device=device)

    def _alloc_handshake_bufs(self, B):
        self._hs_B = B
        d = self.device
        self._cmd = torch.zeros(1, dtype=torch.int64, device=d)
        # Meta layout: [B, K, F, step_id]. step_id is overwritten in-place
        # each request (see _speculation_request). Initialized to 0.
        self._meta = torch.tensor(
            [B, self.K, self.async_fan_out, 0], dtype=torch.int64, device=d)
        self._cache_keys = torch.empty(B, 3, dtype=torch.int64, device=d)
        self._num_tokens_buf = torch.empty(B, dtype=torch.int64, device=d)
        self._temps_buf = torch.empty(B, dtype=torch.float32, device=d)
        self._block_tables_buf = torch.full((B, self.max_blocks), -1, dtype=torch.int32, device=d)
        # Wire layout: [cache_hits (B), phase_source (B), valid_k (B), out_tokens (B*K)].
        # phase_source: 0=miss, 1=phase 1 (draft) hit, 2=phase 2 (proxy) hit. All zeros for non-MESA.
        # valid_k: per-row suffix length (K_long for draft-sourced/miss, K_short for proxy-sourced
        #   in MESA hybrid; uniform K for non-MESA / pre-Phase-4 MESA).
        self._fused_response = torch.empty(3 * B + B * self.K, dtype=torch.int64, device=d)
        self._logits_q = torch.empty(B, self.K, self.vocab_size, dtype=self.draft_dtype, device=d)
        self._extend_counts = torch.zeros(B, dtype=torch.int64, device=d)

    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        eagle_acts = verify_result.eagle_acts
        input_id_list = [seq.token_ids for seq in seqs]

        # EAGLE token-conditioning shift: token at position j gets conditioning
        # from target act at position j-1. Skip first token per seq and drop
        # last eagle_act per seq so they align correctly.
        if eagle_acts is not None:
            sliced = []
            offset = 0
            for ids in input_id_list:
                seq_len = len(ids)
                sliced.append(eagle_acts[offset:offset + seq_len - 1])
                offset += seq_len
            eagle_acts = torch.cat(sliced, dim=0)
            input_id_list = [ids[1:] for ids in input_id_list]

        max_blocks = (self.max_model_len + self.kvcache_block_size - 1) // self.kvcache_block_size
        cmd, metadata, input_ids, num_tokens, draft_block_table, eagle_acts = prepare_prefill_payload(
            input_id_list, eagle_acts, self.device, max_blocks,
            [seq.draft_block_table for seq in seqs],
        )
        dist.send(cmd, dst=self.draft_runner_rank, group=self.async_pg)
        dist.send(metadata, dst=self.draft_runner_rank, group=self.async_pg)
        send_int64(self.async_pg, self.draft_runner_rank,
                   input_ids, num_tokens, draft_block_table.to(torch.int64))
        if eagle_acts is not None:
            dist.send(eagle_acts, dst=self.draft_runner_rank, group=self.async_pg)
        return SpeculateResult([], [])

    def speculate(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        for seq in seqs:
            assert seq.recovery_token_id is not None
            seq.append_token(seq.recovery_token_id)

        if self.verbose:
            sep = '=' * 80
            print(f"\n{sep}", flush=True)
            print(f"[TARGET SEQUENCE TRUNK] Batch size: {len(seqs)}", flush=True)
            for i, seq in enumerate(seqs):
                trunk = seq.token_ids[-20:] if len(seq.token_ids) > 20 else seq.token_ids
                print(f"  Seq {seq.seq_id} (len={len(seq.token_ids)}):", flush=True)
                print(f"    Trunk: ...{decode_tokens(trunk, self.tokenizer)}", flush=True)
                print(f"    Recovery: {seq.recovery_token_id} ({decode_tokens([seq.recovery_token_id], self.tokenizer)})", flush=True)
            print(f"{sep}\n", flush=True)

        eagle = verify_result.eagle_acts is not None
        speculations_tokens, logits_q, cache_hits, phase_source, valid_k = self._speculation_request(seqs, eagle)

        # Build speculations using pre-allocated buffers (avoids torch.tensor(device=cuda) sync)
        B = len(seqs)
        if B != self._recovery_buf.shape[0]:
            self._recovery_buf = torch.empty(B, dtype=torch.int64, device=self.device)
            self._speculations_buf = torch.empty(B, self.K + 1, dtype=torch.int64, device=self.device)
        _rec_cpu = torch.tensor([seq.recovery_token_id for seq in seqs], dtype=torch.int64)
        self._recovery_buf.copy_(_rec_cpu, non_blocking=True)
        self._speculations_buf[:, 0] = self._recovery_buf
        self._speculations_buf[:, 1:] = speculations_tokens
        speculations = self._speculations_buf
        profile_cache_status = None
        _profile_mesa = os.environ.get("SSD_PROFILE_MESA", "0") == "1"
        _hit_statuses: list[int] = []
        _phase_statuses: list[int] = []

        # v1 hybrid: when valid_k differs from K, only extend by valid_k tokens
        # (proxy-sourced rows store K_short; rest of the K_long-shaped buffer is
        # padding from _merge_and_populate_cache and must not enter seq tokens).
        for i, seq in enumerate(seqs):
            vk_i = int(valid_k[i].item()) if valid_k is not None else self.K
            seq.token_ids.extend(speculations_tokens[i, :vk_i].tolist())
            seq.num_tokens = len(seq.token_ids)
            seq.last_token = seq.token_ids[-1]
            seq.num_draft_cached_tokens += vk_i + 1
            if _profile_mesa and cache_hits is not None:
                _hit_statuses.append(int(cache_hits[i].item()))
                _phase_statuses.append(int(phase_source[i].item()) if phase_source is not None else 0)

        # Slice speculations to per-step valid_k width before returning. For B=1
        # (Config invariant) valid_k is uniform; slice with the scalar.
        # B==0 (empty batch from scheduler preemption) is normally filtered at
        # SpecDecodeStep.decode top-level guard; keep a defensive numel() check
        # here for any direct callers that bypass that path.
        if valid_k is not None and valid_k.numel() > 0:
            vk = int(valid_k[0].item())
            if vk != self.K:
                speculations = speculations[:, :vk + 1].contiguous()
                logits_q = logits_q[:, :vk, :].contiguous()

        if _hit_statuses:
            if all(h == 0 for h in _hit_statuses):
                profile_cache_status = "miss"
            elif all(h == 1 for h in _hit_statuses):
                if len(_phase_statuses) == 1:
                    if _phase_statuses[0] == 1:
                        profile_cache_status = "hit_k1"
                    elif _phase_statuses[0] == 2:
                        profile_cache_status = "hit_k2"
                    else:
                        profile_cache_status = "hit"
                elif all(p == _phase_statuses[0] for p in _phase_statuses):
                    if _phase_statuses[0] == 1:
                        profile_cache_status = "hit_k1"
                    elif _phase_statuses[0] == 2:
                        profile_cache_status = "hit_k2"
                    else:
                        profile_cache_status = "hit"
                else:
                    profile_cache_status = "mixed"
            else:
                profile_cache_status = "mixed"

        return SpeculateResult(
            speculations, logits_q, cache_hits, phase_source, valid_k,
            profile_cache_status=profile_cache_status,
            step_id=self._request_step_id,
        )

    def _speculation_request(self, seqs: list[Sequence], eagle: bool):
        B = len(seqs)
        if B != self._hs_B:
            self._alloc_handshake_bufs(B)

        # Phase B: increment step_id once per spec request and mirror it into
        # the meta tensor. Draft picks it up from meta[3] (see
        # draft_runner._service_spec_request). See docs/mesa/06 §4.4.
        self._request_step_id += 1
        self._meta[3] = self._request_step_id

        # Set target-side profiler context for ALL spans inside this request.
        # status is filled in later (in step.py after speculate() returns).
        from ssd.engine.helpers.cudagraph_helpers import (
            mesa_record as _mr, mesa_close as _mc, mesa_set_context as _mctx,
        )
        _mctx(step_id=self._request_step_id, proc="target_rank0")

        # Fill send buffers in-place (avoids torch.tensor from Python lists)
        for i, seq in enumerate(seqs):
            self._cache_keys[i, 0] = seq.seq_id
            self._cache_keys[i, 1] = seq.last_spec_step_accepted_len - 1
            self._cache_keys[i, 2] = seq.recovery_token_id
            self._num_tokens_buf[i] = seq.num_tokens
            self._temps_buf[i] = seq.draft_temperature if seq.draft_temperature is not None else seq.temperature
            bt = seq.draft_block_table
            bt_len = len(bt)
            if bt_len > 0:
                self._block_tables_buf[i, :bt_len] = torch.tensor(bt, dtype=torch.int32, device=self.device)
            self._block_tables_buf[i, bt_len:] = -1

        # Handshake marker: target_send_request — wraps cmd + meta + fused
        # payload send (and EAGLE payloads if applicable). Parent label
        # `target_spec_wait` (assigned by mesa_dump via parent arg).
        _mev_send = _mr("target_send_request", parent="target_spec_wait")
        # Send cmd + meta + fused payload (temps fused into int64 burst)
        dist.send(self._cmd, dst=self.draft_runner_rank, group=self.async_pg)
        dist.send(self._meta, dst=self.draft_runner_rank, group=self.async_pg)
        temps_as_int64 = self._temps_buf.view(torch.int32).to(torch.int64)
        send_int64(
            self.async_pg, self.draft_runner_rank,
            self._cache_keys, self._num_tokens_buf,
            self._block_tables_buf.to(torch.int64), temps_as_int64,
        )

        if eagle:
            recovery_activations = torch.stack(
                [seq.last_target_hidden_state for seq in seqs], dim=0,
            ).to(self.device)
            dist.send(recovery_activations.to(self.draft_dtype),
                      dst=self.draft_runner_rank, group=self.async_pg)

            # Send extend data for glue decode with fused extend
            K = self.K
            act_dim = recovery_activations.shape[-1]
            for i, seq in enumerate(seqs):
                self._extend_counts[i] = seq.extend_count
            extend_eagle_acts = torch.zeros(B, K, act_dim, dtype=self.draft_dtype, device=self.device)
            extend_token_ids = torch.zeros(B, K, dtype=torch.int64, device=self.device)
            for i, seq in enumerate(seqs):
                n = seq.extend_count
                if n > 0 and seq.extend_eagle_acts is not None:
                    extend_eagle_acts[i, :n] = seq.extend_eagle_acts[:n].to(self.draft_dtype)
                    extend_token_ids[i, :n] = seq.extend_token_ids[:n]
            dist.send(self._extend_counts, dst=self.draft_runner_rank, group=self.async_pg)
            dist.send(extend_eagle_acts, dst=self.draft_runner_rank, group=self.async_pg)
            dist.send(extend_token_ids, dst=self.draft_runner_rank, group=self.async_pg)
        _mc("target_send_request", _mev_send)

        # Handshake marker: target_recv_response_wait — wraps the blocking
        # recvs for the draft response (fused tokens + logits_q). This is
        # where pipeline wait time accumulates on the target.
        _mev_recv = _mr("target_recv_response_wait", parent="target_spec_wait")
        # Recv into pre-allocated buffers
        dist.recv(self._fused_response, src=self.draft_runner_rank, group=self.async_pg)
        cache_hits = self._fused_response[:B]
        phase_source = self._fused_response[B:2 * B]
        valid_k = self._fused_response[2 * B:3 * B]
        speculations = self._fused_response[3 * B:].view(B, self.K)
        dist.recv(self._logits_q, src=self.draft_runner_rank, group=self.async_pg)
        _mc("target_recv_response_wait", _mev_recv)

        # Point marker: target_response_received (zero-duration timestamp
        # immediately after logits recv completes).
        _mev_rr = _mr("target_response_received", parent="target_spec_wait")
        _mc("target_response_received", _mev_rr)

        return speculations, self._logits_q, cache_hits, phase_source, valid_k
