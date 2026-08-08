import os
from ssd.engine.helpers.e0_trace import E0_TRACE as _E0_TRACE
from ssd.engine.helpers import e0_trace as _e0
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
        # Target tree Policy-B uses fixed-shape CUDA graphs by default.  The
        # dynamic topology is copied into persistent buffers before verify;
        # exit-time work is then one replay instead of dozens of Python GPU
        # launches.  Keep an explicit off switch for clean fallback A/B.
        self._chain_proxy_graphs = getattr(
            self.target_model_runner, "_chain_proxy_graphs_prebuilt", {})
        self._tree_proxy_graphs = {}
        self._active_tree_proxy_graph = None
        _cfg = self.target_model_runner.config
        _prebuilt = getattr(
            self.target_model_runner, "_tree_proxy_graphs_prebuilt", None)
        if _prebuilt is not None:
            self._tree_proxy_graphs = _prebuilt
        elif (os.environ.get("SSD_TREE_PROXY_GRAPH", "1") != "0"
                and getattr(_cfg, "duet_tree_enabled", False)
                and self.device.type == "cuda"
                and getattr(self.target_model_runner, "rank", 0) == 0):
            from ssd.engine.helpers.p2_tree import TreeProxyCUDAGraph
            _verify_buckets = sorted(
                (self.target_model_runner.tree_verify_wrappers or {}).keys())
            # Proxy top-k must see exactly (valid+1)*V elements.  Padding
            # to the 4/6/8 target-model bucket preserves values but can
            # change the ordering of bit-identical ties inside topk.
            # Capture one tiny metadata envelope per exact valid width.
            _proxy_widths = range(1, max(_verify_buckets) + 1) \
                if _verify_buckets else ()
            _tree_depth_steps = max(
                int(_cfg.duet_phase2_k),
                (int(_cfg.duet_phase1_k)
                 if _cfg.duet_p1_tree_policy == "on" else 0))
            for _nv in _proxy_widths:
                self._tree_proxy_graphs[int(_nv)] = TreeProxyCUDAGraph(
                    nv=int(_nv),
                    vocab_size=_cfg.hf_config.vocab_size,
                    wire_n=_cfg.duet_proxy_wire_N,
                    depth_steps=_tree_depth_steps,
                    top_k=_cfg.duet_proxy_top_k,
                    dtype=_cfg.hf_config.torch_dtype,
                    device=self.device)
            if self._tree_proxy_graphs:
                print("[DUET tree] captured target proxy graphs lazily "
                      f"(buckets={sorted(self._tree_proxy_graphs)})",
                      flush=True)
        self._warmup_duet_acceptance_kernels()

    def _warmup_duet_acceptance_kernels(self):
        """Move the common verify softmax/sampling cold start out of step 1.

        Model and tree CUDA graphs are captured during ``ModelRunner`` init,
        but the temperature>0 acceptance path is ordinary PyTorch code and
        was first exercised by the first real decode step.  Its one-time
        softmax/multinomial setup showed up as a 79--86 ms
        ``verify_sample_accept`` outlier in both chain and tree profiles.

        Warm both DUET widths and both cache-hit branches with synthetic
        tensors.  Preserve the default CUDA RNG state so enabling DUET does
        not change the production sampling sequence.  This routine runs only
        in the rank-0 CUDA verifier; CPU/unit-test constructions are no-ops.
        """
        cfg = self.target_model_runner.config
        if (self.device.type != "cuda"
                or getattr(self.target_model_runner, "rank", 0) != 0
                or not getattr(cfg, "duet_enabled", False)
                or cfg.duet_phase1_k is None):
            return

        from ssd.utils.verify import verify as _verify

        dev_index = (self.device.index if self.device.index is not None
                     else torch.cuda.current_device())
        widths = sorted({int(cfg.duet_phase1_k),
                         int(cfg.duet_phase2_k)})
        vocab = int(cfg.hf_config.vocab_size)
        dtype = cfg.hf_config.torch_dtype
        # fork_rng restores CPU and the selected CUDA generator even if a
        # future warmup operation raises.  Synchronize before leaving so all
        # RNG-consuming kernels have completed before restoration.
        with torch.random.fork_rng(devices=[dev_index], enabled=True):
            for k in widths:
                logits_p = torch.zeros(
                    1, k + 1, vocab, dtype=dtype, device=self.device)
                logits_q = torch.zeros(
                    1, k, vocab, dtype=dtype, device=self.device)
                speculations = torch.zeros(
                    1, k + 1, dtype=torch.int64, device=self.device)
                temperatures = torch.full(
                    (1,), 0.7, dtype=torch.float32, device=self.device)
                valid_k = torch.full(
                    (1,), k, dtype=torch.int64, device=self.device)
                for cache_hit in (False, True):
                    _verify(
                        logits_p=logits_p,
                        logits_q=logits_q,
                        speculations=speculations,
                        temperatures_target=temperatures,
                        temperatures_draft=temperatures,
                        cache_hits=torch.full(
                            (1,), cache_hit, dtype=torch.bool,
                            device=self.device),
                        sampler_x=self.sampler_x,
                        async_fan_out=self.async_fan_out,
                        # Exercise both the miss and ratio branches even if
                        # the live short-JIT option treats every row as ratio.
                        jit_speculate=False,
                        valid_k=valid_k,
                    )
            torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        print(f"[DUET] warmed acceptance kernels (K={widths})", flush=True)

    def _send_proxy_wire(self, config, async_pg, draft_rank, *parts):
        """Send one fixed-schema proxy payload on the shared async path.

        Tree verification used to bypass the chain path's persistent
        ``AsyncSendRing`` and call blocking ``dist.send`` directly.  That
        serialized target graph-post behind the proxy message exactly on the
        steps where tree verification was already wider.  Both chain and tree
        now use the same transport contract and shutdown accounting.
        """
        from ssd.utils.async_helpers.nccl_pack import send_int64
        if os.environ.get("SSD_ASYNC_PROXY_SEND", "0") != "1":
            send_int64(async_pg, draft_rank, *parts)
            return
        if self._proxy_send_ring is None:
            from ssd.utils.async_helpers.nccl_pack import AsyncSendRing
            self._proxy_send_ring = AsyncSendRing(
                n_slots=2,
                buf_size=(2 * config.max_num_seqs
                          * config.duet_proxy_wire_N),
                device=parts[0].device,
                pg=async_pg,
            )
        self._proxy_send_ring.send(
            draft_rank, *(part.reshape(-1) for part in parts))
        self._proxy_send_call_count += 1

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

        # T3.4-b5: 트리 step 감지 플래그 — 비-DUET 경로에서도 정의돼야
        # 아래 verify 분기가 안전하다.
        _tree_meta_arg = None
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

            # T3.4-b4/b5: 트리 응답 감지 (valid>0) — proxy q와 run() 메타
            # 양쪽에서 쓰므로 여기서 한 번 계산.
            _tree_meta_arg = None
            self._active_tree_proxy_graph = None
            if (getattr(config, "duet_tree_enabled", False)
                    and getattr(speculate_result, "tree_ints", None)
                    is not None):
                _ti_row = speculate_result.tree_ints[0]
                if int(_ti_row[0]) > 0:
                    _tree_meta_arg = _ti_row.tolist()
                    # Validate on rank 0 before indexing parent_q_ref or
                    # sending the metadata through SHM to target TP ranks.
                    # This readback already existed as .tolist().
                    from ssd.engine.helpers.p2_tree import (
                        parse_tree_ints, validate_tree_ints)
                    _nv_t = int(config.duet_tree_wire_nodes)
                    _ti_checked = parse_tree_ints(
                        torch.tensor(_tree_meta_arg, dtype=torch.int64),
                        _nv_t)
                    validate_tree_ints(
                        _ti_checked, _nv_t, config.hf_config.vocab_size)
                    _phase_t = int(speculate_result.phase_source[0]) \
                        if speculate_result.phase_source is not None else 0
                    _phase_cap = (int(config.duet_p1_tree_max_nodes)
                                  if _phase_t == 1 else
                                  int(config.duet_p2_tree_max_nodes))
                    if int(_tree_meta_arg[0]) > _phase_cap:
                        raise RuntimeError(
                            f"tree valid={int(_tree_meta_arg[0])} exceeds "
                            f"phase {_phase_t} cap={_phase_cap}")
                    if _tree_meta_arg[0] != _step_lookahead:
                        raise RuntimeError(
                            f"tree wire invariant: meta valid="
                            f"{_tree_meta_arg[0]} != step vk="
                            f"{_step_lookahead} "
                            f"(valid_k={speculate_result.valid_k.tolist()}, "
                            f"ints={_tree_meta_arg})")

                    # Prepare only tiny fixed-shape topology buffers here,
                    # while graph_pre has not started.  The exit callback
                    # then needs no Python topology construction or H2D copy.
                    self._active_tree_proxy_graph = None
                    if self._tree_proxy_graphs:
                        _valid_t = int(_tree_meta_arg[0])
                        _pg = self._tree_proxy_graphs[_valid_t]
                        _par_t = _tree_meta_arg[
                            3 + _nv_t:3 + 2 * _nv_t][:_valid_t]
                        _sib_t = _tree_meta_arg[
                            3 + 2 * _nv_t:3 + 3 * _nv_t][:_valid_t]
                        _pg.prepare_topology(_par_t, _sib_t)
                        self._active_tree_proxy_graph = _pg

            # Slice draft_tokens / logits_q to step_lookahead since target ran
            # only K_short+1 positions on a short-hit step.
            draft_tokens = speculate_result.speculations[:, 1:_step_lookahead + 1]  # [B, vk]
            if _tree_meta_arg is not None:
                # 트리 step: 노드 j의 제안분포 q = 부모 셀 logits —
                # parent_q_ref로 gather (backbone 캐시 행 logits_q는
                # 트리 노드의 q가 아니다). α̂ = min(1, p^E/q_parent).
                _nv_t = int(config.duet_tree_wire_nodes)
                _pq_ref = speculate_result.tree_ints[
                    0, 3 + 3 * _nv_t:3 + 4 * _nv_t][:_step_lookahead]
                logits_q = speculate_result.parent_q_logits[
                    0, _pq_ref.to(speculate_result.parent_q_logits.device)
                ].unsqueeze(0)                                          # [1, vk, V]
            else:
                logits_q = speculate_result.logits_q[:, :_step_lookahead, :]        # [B, vk, V]
            cache_hits = speculate_result.cache_hits                                  # [B] or None

            _tree_meta_for_proxy = _tree_meta_arg

            # 리뷰3-9: 트리 Policy-B는 실제 verify 보행과 같은 분포로 —
            # p^E는 target temp, q는 draft temp+sampler_x (B=1 스칼라).
            _tp0 = seqs[0].temperature
            _td0 = (seqs[0].draft_temperature
                    if seqs[0].draft_temperature is not None
                    else seqs[0].temperature)
            if _tree_meta_arg is not None and _tp0 <= 0:
                # 리뷰3-10: target temp=0 + 트리 미정의 — draft측 WOR
                # 게이트와 동일하게 명시 차단 (조용한 오분포 금지).
                raise NotImplementedError(
                    "DUET dynamic tree requires target temperature > 0 "
                    "(temp-0 tree verify is gated; use chain policy)")

            def _proxy_fn(exit_logits, orig_bs, _vk=_step_lookahead,
                          _tm=_tree_meta_for_proxy, _tp=_tp0, _td=_td0):
                if _tm is not None:
                    # 이슈 #25 (리뷰 2C): 트리 step은 체인 수식 금지 —
                    # p^E row = parent+1 페어링 + terminal-mass DP.
                    self._compute_and_send_proxy_tree(
                        exit_logits, draft_tokens, logits_q, orig_bs,
                        _vk, _tm, async_pg, draft_rank,
                        temp_p=_tp, temp_d=_td)
                else:
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
        # T3.4-b4: 트리 응답이면 tree_ints를 SHM으로 전 rank에 전달 —
        # target이 eager 트리 verify 분기를 탄다. (위 duet 블록에서 계산)
        result = self.target_model_runner.call(
            "run", seqs, False, False, True, None, _step_lh_arg,
            _tree_meta_arg)

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

        # ===== E0: 최종층 분포 기록 (P0 확장 — 기본 OFF, 전용 런 전용) =====
        if _E0_TRACE:
            _e0.record_target_final(logits_p, temperatures_target)

        from ssd.engine.helpers.cudagraph_helpers import duet_record as _mr, duet_close as _mc
        _mev_vs = _mr("verify_sample_accept")
        if _tree_meta_arg is not None:
            # T3.4-b5: 트리 보행 (잔차 사다리 — tree_verify_walk_tensor,
            # 참조 보행과 동일-코인 동등성으로 고정된 프로덕션 경로).
            new_suffixes, recovery_tokens, _term_node, _walk_path = \
                self._tree_verify_walk(
                    speculate_result, logits_p,
                    temperatures_target, temperatures_draft)
            # b6-1: 종단 노드 id (0=root-종단, 1+j=노드 j) — 다음 요청
            # 키 k_idx로 나간다 (speculator). 절단 시에도 경로-prefix가
            # 수락-prefix와 일치하므로 id는 그대로 유효 (키는 miss 가능
            # — 무해).
            seqs[0].tree_terminal_node = _term_node
            # b6-3: 수락 경로 KV commit — scratch(뷰-순서 셀) → canonical
            # (경로-순서 셀). 전 rank가 SHM 명령 순서로 실행 — B=1에서는
            # 다음 run()보다 항상 먼저 완료 (per-rank 순차 loop) → ack
            # 없이 race 없음 (B>1 tree는 게이트 OFF; ack는 그때 신설).
            _pos0_t = seqs[0].num_tokens - (_step_lookahead + 1)
            _bt = seqs[0].block_table
            _bs = self.target_model_runner.block_size
            _src, _dst = [], []
            for k, j in enumerate(_walk_path):
                if j == k:
                    continue
                sp = _pos0_t + 1 + j
                dp = _pos0_t + 1 + k
                _src.append(_bt[sp // _bs] * _bs + sp % _bs)
                _dst.append(_bt[dp // _bs] * _bs + dp % _bs)
            if _src:
                self.target_model_runner.call("commit_tree_kv", _src, _dst)
        else:
            for _s in seqs:
                _s.tree_terminal_node = None
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

    def _tree_verify_walk(self, speculate_result, logits_p,
                          temperatures_target, temperatures_draft):
        """P2-tree 잔차-사다리 보행 (T3.4-b5). B=1 전용.

        q_parent_probs는 draft 샘플측과 **동일 함수**
        (q_probs_from_logits — temp/sampler_x 동일 처리)로 빌드 — 수락
        보존의 전제. 반환 관례는 체인 verify()와 동일: suffix =
        [rec] + 수락 경로 토큰, recovery 별도 (postprocess가 내용-무관
        append — 트리 경로가 그대로 실린다).
        """
        from ssd.engine.helpers.p2_tree import (
            parse_tree_ints, tree_verify_walk_tensor, q_probs_from_logits)
        cfg = self.target_model_runner.config
        nv = int(cfg.duet_tree_wire_nodes)
        ti = parse_tree_ints(speculate_result.tree_ints[0].cpu(), nv)
        pq = speculate_result.parent_q_logits[0].float()         # [nv, V] GPU
        td = float(temperatures_draft[0])
        tt = float(temperatures_target[0])
        q_probs = q_probs_from_logits(
            pq, torch.full((pq.shape[0],), max(td, 1e-8),
                           device=pq.device),
            self.sampler_x, self.async_fan_out)
        p_rows = logits_p[0].float()             # [valid+1, V] GPU 상주 —
        # 사다리 벡터 연산([V])을 GPU에서 수행 (CPU 복사 2.3MB/step 제거;
        # walk_tensor는 device-agnostic, 스칼라 sync는 형제당 소수)

        def _coin():
            return float(torch.rand(1).item())

        def _mult(probs):
            return int(torch.multinomial(probs, 1).item())

        path, terminal = tree_verify_walk_tensor(
            ti, p_rows, q_probs, tt, _coin, _mult)
        _calib = os.environ.get("SSD_TREE_CALIB_TRACE", "")
        if _calib:
            # Diagnostic-only post-hoc calibration.  Keep three labels
            # separate: an unattempted later sibling is not a rejection.
            # The offline analyzer derives ``expansion_useful`` from the
            # accepted path: this node must have an accepted child below it.
            # That is the relevant label because a threshold keeps this node
            # as a verifiable leaf and only suppresses its later expansion.
            import json as _json
            _v = int(ti["valid"])
            _par = [int(ti["parent_local"][j]) for j in range(_v)]
            _sib = [int(ti["sib_order"][j]) for j in range(_v)]
            _tok_cpu = [int(ti["tok"][j]) for j in range(_v)]
            _tok = torch.tensor(
                _tok_cpu, dtype=torch.int64, device=q_probs.device)
            _qref = ti["parent_q_ref"][:_v].to(
                q_probs.device, torch.int64)
            _q_node = q_probs[_qref, _tok].detach().float().cpu().tolist()
            _depth, _path_conf = [0] * _v, [0.0] * _v
            for _j in range(_v):
                _p = _par[_j]
                _depth[_j] = 1 if _p < 0 else _depth[_p] + 1
                _path_conf[_j] = float(_q_node[_j]) * (
                    1.0 if _p < 0 else _path_conf[_p])

            _accepted_by_parent = {}
            _reached_ctx = {-1}
            _ctx = -1
            for _j in path:
                _jj = int(_j)
                if _par[_jj] != _ctx:
                    raise RuntimeError(
                        "tree calibration path/topology mismatch: "
                        f"node={_jj} parent={_par[_jj]} expected={_ctx}")
                _accepted_by_parent[_ctx] = _jj
                _reached_ctx.add(_jj)
                _ctx = _jj

            _nodes = []
            _path_set = {int(x) for x in path}
            for _j in range(_v):
                _p = _par[_j]
                _parent_reached = _p in _reached_ctx
                _winner = _accepted_by_parent.get(_p)
                _attempted = bool(
                    _parent_reached and
                    (_winner is None or _sib[_j] <= _sib[_winner]))
                _accepted = bool(_winner == _j)
                _nodes.append({
                    "node": _j,
                    "parent": _p,
                    "sibling": _sib[_j],
                    "depth": _depth[_j],
                    "token": _tok_cpu[_j],
                    "q": float(_q_node[_j]),
                    "path_conf": float(_path_conf[_j]),
                    "parent_reached": bool(_parent_reached),
                    "attempted": _attempted,
                    "accepted": _accepted,
                    "rejected": bool(_attempted and not _accepted),
                    "on_path": bool(_j in _path_set),
                })
            self._tree_calib_seq = getattr(self, "_tree_calib_seq", 0) + 1
            _parent = os.path.dirname(_calib)
            if _parent:
                os.makedirs(_parent, exist_ok=True)
            with open(_calib + ".jsonl", "a") as _f:
                _f.write(_json.dumps({
                    "seq": self._tree_calib_seq,
                    "phase": (int(speculate_result.phase_source[0])
                              if speculate_result.phase_source is not None
                              else 0),
                    "policy": cfg.duet_tree_policy,
                    "valid": _v,
                    "temperature_target": tt,
                    "temperature_draft": td,
                    "path": [int(x) for x in path],
                    "terminal": int(terminal),
                    "nodes": _nodes,
                }) + "\n")
        if os.environ.get("SSD_TREE_DIAG", "0") == "1":
            # 판별 진단: 수락이 서브트리 "상한"에 막히는가(예산 문제) vs
            # 깊은 트리에서도 일찍 기각되는가(보행/q 문제)
            _v = int(ti["valid"])
            _par_d = ti["parent_local"]
            _dep = [0] * _v
            for _j in range(_v):
                _pp = int(_par_d[_j])
                _dep[_j] = (_dep[_pp] + 1) if _pp >= 0 else 1
            _dmax = max(_dep) if _v else 0
            print(f"[tree-diag] valid={_v} dmax={_dmax} "
                  f"path={len(path)} ceil={int(len(path) >= _dmax)} "
                  f"term={(1 + path[-1]) if path else 0}", flush=True)
        rec0 = int(speculate_result.speculations[0, 0])
        suffix = [rec0] + [int(ti["tok"][j]) for j in path]
        term_node = (1 + path[-1]) if path else 0
        _topo = os.environ.get("SSD_TREE_TOPO_TRACE", "")
        if _topo:
            # 23번 단계2: 서빙된 root의 topology + 수락 경로 (계측 전용)
            import json as _json
            _v = int(ti["valid"])
            with open(_topo + ".walk.jsonl", "a") as _f:
                _f.write(_json.dumps({
                    "phase": (int(speculate_result.phase_source[0])
                              if speculate_result.phase_source is not None
                              else 0),
                    "par": [int(ti["parent_local"][j]) for j in range(_v)],
                    "sib": [int(ti["sib_order"][j]) for j in range(_v)],
                    "path": [int(x) for x in path],
                    "accepted_len": len(path),
                    "term": term_node,
                }) + "\n")
        return [suffix], [terminal], term_node, path

    def _compute_and_send_proxy_tree(self, exit_logits, draft_tokens,
                                     logits_q, B, vk, tree_meta,
                                     async_pg, draft_rank,
                                     temp_p=1.0, temp_d=1.0):
        """트리 step Policy-B (이슈 #25, v6 §7.5ⓒ — 리뷰 2C 수용).

        체인 수식과의 차이:
        - α̂_j의 p^E는 **부모 컨텍스트 row** (parent_local+1; root직결=
          row 0)에서 gather — 종전 row j 페어링은 형제/사촌에서 오배열.
        - ĥ는 cumprod가 아니라 topology-따라 전파하는 terminal-mass DP:
          reach(j) = reach(parent)·∏앞형제(1−α̂)·α̂_j,
          terminal(ctx) = reach(ctx)·∏자식(1−α̂) (독립 근사 — 체인과
          동일 수준; 사다리 정밀 반영은 후속).
        - residual(ctx): 내부 ctx는 (p^E_ctx − q_ctx)+ 에 자식 토큰
          제외 (q_ctx = 그 ctx 자식의 parent_q row); 잎 ctx는 p^E_ctx
          그대로 (잎 보너스 = plain p 샘플과 정합).
        wire 형식/pack은 체인과 동일 (chosen_pos = ctx id: 0=rec,
        1+j=노드 j — draft fork 네임스페이스와 정합).

        이슈 #28+#34 (리뷰2): ① 형제 α̂를 원본 p/q 독립곱으로 계산하던
        근사를 verify 보행과 동일한 **정확 사다리**(R/D 순차 갱신)로
        교체 — 2형제 반례에서 all-reject 질량 0 vs 실제 0.28125.
        ② residual도 사다리 최종 R (보행의 recovery 원천과 동일 분포).
        ③ 전면 GPU 텐서화 — 종전 파이썬 구현(노드별 float()/int() 동기
        + CPU [valid+1, V] 행 조립)이 exit_logits 스팬을 0.3→17.6ms로
        부풀림 (tree-hit step 실측). 이제 sync/CPU 왕복 0회 — 토폴로지
        인덱스는 tree_meta(CPU 리스트)에서 빌드.
        """
        import torch.distributed as dist
        from ssd.engine.helpers.p2_tree import pack_piv, tree_policy_b_ladder
        config = self.target_model_runner.config
        _detail_profile = os.environ.get(
            "SSD_PROFILE_DUET_DETAIL", "0") == "1"
        if _detail_profile:
            from ssd.engine.helpers.cudagraph_helpers import (
                duet_record as _mr_tree, duet_close as _mc_tree)
        else:
            _mr_tree = _mc_tree = None
        if exit_logits is None:
            return
        if (os.environ.get("SSD_PROXY_STREAM", "0") == "1"
                or config.duet_exit_replica):
            # The callback may outlive the default stream dispatch.  Keep the
            # closure-captured proposal tensors alive until the tree policy
            # and async send queued on the side stream have consumed them.
            _cur = torch.cuda.current_stream()
            exit_logits.record_stream(_cur)
            if draft_tokens is not None:
                draft_tokens.record_stream(_cur)
            if logits_q is not None:
                logits_q.record_stream(_cur)
        if exit_logits.dim() == 2:
            exit_logits = exit_logits.view(B, vk + 1, -1)
        V = exit_logits.shape[-1]
        nv = int(config.duet_tree_wire_nodes)
        valid = int(tree_meta[0])
        par = [int(x) for x in tree_meta[3 + nv:3 + 2 * nv][:valid]]
        sib = [int(x) for x in tree_meta[3 + 2 * nv:3 + 3 * nv][:valid]]

        _proxy_graph = self._active_tree_proxy_graph
        if _proxy_graph is not None:
            # These two coarse spans are useful in every timeline profile;
            # the finer eager fallback labels remain behind DETAIL.
            from ssd.engine.helpers.cudagraph_helpers import (
                duet_record as _mr_tree_coarse,
                duet_close as _mc_tree_coarse)
            _ev_graph = _mr_tree_coarse(
                "tree_proxy_graph_replay", parent="exit_proxy_side")
            chosen_pos, chosen_tok, _top_v = _proxy_graph.replay(
                exit_logits[0], logits_q[0], draft_tokens[0])
            _mc_tree_coarse("tree_proxy_graph_replay", _ev_graph)
            _ev_send = _mr_tree_coarse(
                "proxy_send_enqueue", parent="exit_proxy_side")
            Verifier._send_proxy_wire(
                self, config, async_pg, draft_rank,
                chosen_pos, chosen_tok)
            _mc_tree_coarse("proxy_send_enqueue", _ev_send)
            return

        # 리뷰3-9(분포 미러)는 **동일-시드 A/B로 원복** (2026-08-04):
        # temp/sampler_x 반영판은 hit +0.011에 P2AL 2.13→1.94 —
        # 날카로워진 p^E가 wire 후보를 얕은 ctx로 몰아 깊이를 깎았다
        # (tok/step 4.55→4.36 순손실). plain softmax가 체인 proxy와도
        # 일관된 경험적 동작점 — 재도전은 P_iv 랭킹·β·prior 공동
        # recalibration으로만 (T6 부채; 20번 판정표 참조).
        _ev_softmax = (_mr_tree("tree_proxy_softmax")
                       if _detail_profile else None)
        p_E = torch.softmax(exit_logits[0].float(), dim=-1)      # [vk+1, V]
        q_rows = torch.softmax(logits_q[0].float(), dim=-1)      # [vk, V]
        tokens = draft_tokens[0, :valid].to(p_E.device)
        if _detail_profile:
            _mc_tree("tree_proxy_softmax", _ev_softmax)

        _ev_ladder = (_mr_tree("tree_proxy_accept_residual")
                      if _detail_profile else None)
        _alpha, term, resid = tree_policy_b_ladder(
            par, sib, tokens, p_E, q_rows)
        if _detail_profile:
            _mc_tree("tree_proxy_accept_residual", _ev_ladder)

        # Preserve the chain Policy-B score scale: remove already-drafted
        # children, keep top-k per context, renormalize that retained set,
        # and only then rank contexts on the common wire.  Ranking the full
        # vocabulary here changes roots even for a C=1 chain-shaped tree.
        _ev_rank = (_mr_tree("tree_proxy_rank_candidates")
                    if _detail_profile else None)
        correction = resid.clone()
        if valid:
            par_t = torch.tensor(par, dtype=torch.int64,
                                 device=p_E.device)
            correction[par_t + 1, tokens] = 0.0
        ctx_k = min(int(config.duet_proxy_top_k), V)
        correction_prob, correction_id = correction.topk(ctx_k, dim=-1)
        correction_prob = correction_prob / correction_prob.sum(
            -1, keepdim=True).clamp(min=1e-10)
        piv = correction_prob * term.unsqueeze(1)
        wire_N = config.duet_proxy_wire_N
        flat = piv.flatten()
        k = min(wire_N, flat.numel())
        top_v, top_i = flat.topk(k)
        chosen_pos = (top_i // ctx_k).to(torch.int64)
        chosen_tok = correction_id.flatten().gather(0, top_i).to(torch.int64)
        if k < wire_N:                       # pad (드묾)
            pad = wire_N - k
            chosen_pos = torch.cat([chosen_pos, chosen_pos.new_zeros(pad)])
            chosen_tok = torch.cat([chosen_tok, chosen_tok.new_zeros(pad)])
            top_v = torch.cat([top_v, top_v.new_zeros(pad)])
        if getattr(config, "duet_tree_enabled", False):
            chosen_tok = pack_piv(chosen_tok, top_v)
        if _detail_profile:
            _mc_tree("tree_proxy_rank_candidates", _ev_rank)
        dev = self.device
        _ev_send = (_mr_tree("tree_proxy_send")
                    if _detail_profile else None)
        Verifier._send_proxy_wire(
            self,
            config, async_pg, draft_rank,
            chosen_pos.to(dev), chosen_tok.to(dev))
        if _detail_profile:
            _mc_tree("tree_proxy_send", _ev_send)

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

        # Fixed-shape B=1 chain fast path.  This covers K1, ordinary K2 and
        # cache miss (when JIT logits_q is valid).  Tree-hit steps are routed
        # to _compute_and_send_proxy_tree before reaching this function.
        # Keeping the whole Policy-B calculation in one CUDA graph removes
        # the many small launches that appeared as a 4--6 ms proxy bar.
        _chain_graph = getattr(self, "_chain_proxy_graphs", {}).get(int(K))
        if (_chain_graph is not None
                and B == 1
                and self.jit_speculate
                and config.duet_policy == "b"
                and not _E0_TRACE):
            from ssd.engine.helpers.cudagraph_helpers import (
                duet_record as _mr_chain, duet_close as _mc_chain)
            _ev_graph = _mr_chain(
                "chain_proxy_graph_replay", parent="exit_proxy_side")
            chosen_pos, chosen_tok, _ = _chain_graph.replay(
                exit_logits[0], logits_q[0], draft_tokens[0, :K])
            _mc_chain("chain_proxy_graph_replay", _ev_graph)
            _ev_send = _mr_chain(
                "proxy_send_enqueue", parent="exit_proxy_side")
            Verifier._send_proxy_wire(
                self, config, async_pg, draft_rank,
                chosen_pos, chosen_tok)
            _mc_chain("proxy_send_enqueue", _ev_send)
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
            _tree_enabled = getattr(
                config, "duet_tree_enabled",
                getattr(config, "duet_tree_policy", "off") != "off")
            if _tree_enabled:
                # T2.0 (v6 D2): 라이브 P_iv를 wire에 동승 (양자화 pack).
                from ssd.engine.helpers.p2_tree import pack_piv
                _piv_chosen = P_iv.flatten(1).gather(1, top_idx)
                chosen_tok = pack_piv(chosen_tok, _piv_chosen)
            if _detail_profile:
                _mc_d("proxy_pack", _ev_pack)

            # ===== E0 calibration trace (P0, docs/duet/internal/17 §2; default OFF —
            # dedicated runs only, never TPS measurement) =====
            if _E0_TRACE:
                _e0.record_target_wire(
                    config, exit_logits, logits_q, draft_tokens, B, K,
                    valid_k, cache_hits, P_iv, top_idx, chosen_pos,
                    chosen_tok, h)

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
            Verifier._send_proxy_wire(
                self,
                config, async_pg, draft_rank,
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
