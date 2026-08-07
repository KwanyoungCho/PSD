"""E0 calibration trace gate (P0 — docs/duet/internal/15 §9 E0, docs/duet/internal/17 §2).

Standalone, wholesale-removable module. The engine carries only small
guarded blocks (`if E0_TRACE: e0_trace.record_*`); deleting this file +
those blocks removes the feature entirely.

User conditions (docs/duet/internal/15 §10 P0, 2026-08-04):
- Gate `SSD_DUET_E0_TRACE=1`, default OFF. NEVER enabled in TPS
  measurement runs (same principle as PROFILE=0); dedicated runs only.
- OFF cost at call sites = one module-level bool check.
- ON: tensors are copied to pinned host buffers with non_blocking=True on
  the caller's stream; a background thread waits on a CUDA event and
  serializes to JSONL — no sync on the engine critical path. The TPS of
  an ON run is never reported.
- `SSD_DUET_E0_SUBSAMPLE=N` records every N-th step (default 1 = all).
  `SSD_DUET_E0_DIR` sets the output dir (default ./e0_trace).
- Drop accounting: queue overflow increments a counter written in the
  final summary line — E0 hygiene requires drops == 0 (docs/duet/internal/15 §9).

Schema (JSONL, one record per line, joined offline on (n_step, seq)):
- target "wire": full wire candidate set BEFORE dedup with raw P_iv values
  + sufficient statistics for offline temp-matched recomputation (per-pos
  y-logits, exact lse@temp1, top-M exit/draft logits — the wire itself
  carries no P_iv, so this is the only source; docs/duet/internal/15 E0 ④).
- draft "request": incoming outcome (seq, accepted_len-1, recovery_tok).
- draft "response": phase_source / valid_k / response tokens.
- draft "selector": parsed wire (order = rank) + retained P2 seeds after
  dedup + per-position fan-out (dedup survival is reconstructible).
"""
import atexit
import json
import os
import queue
import threading

import torch

E0_TRACE = os.environ.get("SSD_DUET_E0_TRACE", "0") == "1"
_SUBSAMPLE = int(os.environ.get("SSD_DUET_E0_SUBSAMPLE", "1"))
_OUT_DIR = os.environ.get("SSD_DUET_E0_DIR", "e0_trace")
_TOPM = int(os.environ.get("SSD_DUET_E0_TOPM", "32"))


class _Writer:
    """Background JSONL writer: the engine thread enqueues (event, record)
    pairs; this thread waits on the event (so device→pinned-host copies
    have landed) and serializes. Queue overflow drops the record and
    counts it (never blocks the engine)."""

    def __init__(self, path):
        self.drops = 0
        self.written = 0
        self._q = queue.Queue(maxsize=8192)
        # Line-buffered: the engine may hard-exit (no atexit) — an unflushed
        # block buffer silently loses the tail (관측: draft 꼬리 ~700 step
        # 유실, docs/duet/internal/17 이슈 #5). Writes happen on this background
        # thread, so per-line flush costs nothing on the engine path.
        self._f = open(path, "a", buffering=1)
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            evt, rec = item
            if evt is not None:
                evt.synchronize()
            for k, v in list(rec.items()):
                if torch.is_tensor(v):
                    rec[k] = v.tolist()
            self._f.write(json.dumps(rec) + "\n")
            self.written += 1
            # Heartbeat: the engine hard-exits (atexit unreliable), so the
            # drop counter is persisted periodically — E0 hygiene (drop=0)
            # is verifiable from the last heartbeat even without a clean
            # close (docs/duet/internal/17 이슈 #5 후속).
            if self.written % 1000 == 0:
                self._f.write(json.dumps(
                    {"kind": "heartbeat", "written": self.written,
                     "drops": self.drops}) + "\n")
        self._f.write(json.dumps(
            {"kind": "summary", "written": self.written,
             "drops": self.drops}) + "\n")
        self._f.flush()
        self._f.close()

    def put(self, evt, rec):
        try:
            self._q.put_nowait((evt, rec))
        except queue.Full:
            self.drops += 1

    def close(self):
        self._q.put(None)
        self._t.join(timeout=30)


class E0Trace:
    def __init__(self, role):
        os.makedirs(_OUT_DIR, exist_ok=True)
        self._w = _Writer(os.path.join(
            _OUT_DIR, f"e0_{role}_{os.getpid()}.jsonl"))
        self._n = 0

    def _record(self, kind, gpu_tensors=None, **fields):
        self._n += 1
        if _SUBSAMPLE > 1 and (self._n % _SUBSAMPLE):
            return
        rec = {"kind": kind, "n": self._n, **fields}
        evt = None
        any_cuda = gpu_tensors and any(t.is_cuda for t in gpu_tensors.values())
        if gpu_tensors:
            for name, t in gpu_tensors.items():
                if t.is_cuda:
                    host = torch.empty(
                        t.shape, dtype=t.dtype, pin_memory=True)
                    host.copy_(t, non_blocking=True)
                    rec[name] = host
                else:
                    rec[name] = t.clone()
            if any_cuda:
                evt = torch.cuda.Event()
                evt.record()
        self._w.put(evt, rec)

    def close(self):
        self._w.close()


_INSTANCES = {}


def get(role):
    inst = _INSTANCES.get(role)
    if inst is None:
        inst = _INSTANCES[role] = E0Trace(role)
    return inst


def close_all():
    for inst in _INSTANCES.values():
        inst.close()
    _INSTANCES.clear()


atexit.register(close_all)


# ---------------------------------------------------------------------------
# Record assemblers — all heavy logic lives here, not at the call sites.
# ---------------------------------------------------------------------------

def record_target_wire(config, exit_logits, logits_q, draft_tokens,
                       B, K, valid_k, cache_hits,
                       P_iv, top_idx, chosen_pos, chosen_tok, h):
    """Target rank-0, once per spec step, right after the wire is packed.

    Sufficient statistics for offline raw AND temp-matched P_iv
    (docs/duet/internal/15 E0 ④): exact per-position lse@temp1 + y-logits +
    top-M exit/draft logits (temp-matched is top-M-approximate, which
    the design explicitly allows), + exit logit of every wire candidate.
    """
    tr = get("target")
    V = logits_q.shape[-1]
    exit_view = exit_logits.view(B, K + 1, V).float()
    lq = logits_q.float()
    piv_chosen = P_iv.flatten(1).gather(1, top_idx)                # [B, N]
    y_logit_E = exit_view[:, :K, :].gather(
        2, draft_tokens.unsqueeze(-1)).squeeze(-1)                 # [B, K]
    y_logit_D = lq.gather(2, draft_tokens.unsqueeze(-1)).squeeze(-1)
    lse1_E = torch.logsumexp(exit_view, dim=-1)                    # [B, K+1]
    lse1_D = torch.logsumexp(lq, dim=-1)                           # [B, K]
    topm = min(_TOPM, V)
    e_top = exit_view.topk(topm, dim=-1)
    d_top = lq.topk(topm, dim=-1)
    # Tree Policy-B packs P_iv into bits 15..31 of chosen_tok before the
    # communication call.  E0 is invoked after that packing, so using the
    # wire integer as a vocabulary index triggers a device-side gather OOB.
    # Store and gather with the clean token bits; P_iv itself is already
    # recorded losslessly enough in the separate ``piv`` field.
    tree_enabled = bool(
        config is not None
        and (getattr(config, "duet_tree_enabled", False)
             or getattr(config, "duet_tree_policy", "off") != "off"))
    wire_tok = (chosen_tok & ((1 << 15) - 1)
                if tree_enabled else chosen_tok)
    cand_logit_E = exit_view.gather(
        1, chosen_pos.unsqueeze(-1).expand(-1, -1, V)).gather(
        2, wire_tok.unsqueeze(-1)).squeeze(-1)                     # [B, N]
    tr._record(
        "wire",
        gpu_tensors=dict(
            chosen_pos=chosen_pos, chosen_tok=wire_tok,
            piv=piv_chosen, h=h,
            y_logit_E=y_logit_E, y_logit_D=y_logit_D,
            lse1_E=lse1_E, lse1_D=lse1_D,
            exit_top_ids=e_top.indices, exit_top_logits=e_top.values,
            draft_top_ids=d_top.indices, draft_top_logits=d_top.values,
            cand_logit_E=cand_logit_E,
            valid_k=valid_k if valid_k is not None else torch.full(
                (B,), K, dtype=torch.int64, device=chosen_pos.device),
            cache_hits=cache_hits if cache_hits is not None else torch.zeros(
                (B,), dtype=torch.int64, device=chosen_pos.device),
        ),
        K=int(K), B=int(B))


def record_target_final(logits_p, temperatures):
    """Target rank-0, verify forward 직후: **최종층** 분포의 top-M + 정확
    lse. 형제-수락(λ) 계산은 exit 근사가 아니라 이 분포로 해야 한다는
    사용자 지적(2026-08-04) 반영 — k번째 "final" 레코드는 같은 step의
    k번째 "wire" 레코드와 짝 (둘 다 spec step당 1회; K로 교차 확인)."""
    tr = get("target")
    lp = logits_p.float()
    topm = min(_TOPM, lp.shape[-1])
    t = lp.topk(topm, dim=-1)
    lse = torch.logsumexp(lp, dim=-1)
    tr._record(
        "final",
        gpu_tensors=dict(final_top_ids=t.indices,
                         final_top_logits=t.values,
                         lse1_P=lse, temps=temperatures),
        K=int(lp.shape[1] - 1), B=int(lp.shape[0]))


def record_draft_request(step_id, cache_keys, temps):
    """Draft, once per spec request: the previous step's ACTUAL outcome —
    cache key (seq_id, accepted_len-1, recovery_token) + temperature."""
    get("draft")._record(
        "request",
        gpu_tensors=dict(cache_keys=cache_keys, temps=temps),
        step_id=int(step_id))


def record_draft_response(step_id, phase_source, valid_k, out_tokens):
    """Draft, once per response: what was served (hit type per seq,
    per-seq chain depth, the response tokens themselves)."""
    get("draft")._record(
        "response",
        gpu_tensors=dict(phase_source=phase_source, valid_k=valid_k,
                         out_tokens=out_tokens),
        step_id=int(step_id))


def record_draft_selector(step_id, chosen_pos, chosen_tok,
                          proxy_forked, proxy_fan_out, proxy_piv=None):
    """Draft, once per tree build: the parsed wire (order = wire rank —
    score-descending) and the retained P2 seeds after P1 dedup, plus the
    per-position fan-out. Dedup survival / original rank of each retained
    seed is reconstructible offline (wire order + selector semantics)."""
    tensors = dict(chosen_pos=chosen_pos, chosen_tok=chosen_tok,
                   proxy_forked=proxy_forked,
                   proxy_fan_out=proxy_fan_out)
    # New calibration traces are self-contained: the retained root score is
    # recorded in exactly the same grouped order as ``proxy_forked``.  Older
    # traces did not have this field and need a target-wire join offline.
    if proxy_piv is not None:
        tensors["proxy_piv"] = proxy_piv
    get("draft")._record(
        "selector",
        gpu_tensors=tensors,
        step_id=int(step_id))
