"""Opt-in diagnostic for captured draft forwards, with no per-step CPU readback.

This is a benchmark adapter, never imported by serving. GPU timestamps and
root node counts are copied only at request boundaries. Timing marker kernels
have a cost; compare with an uninstrumented run before interpreting latency.
"""
import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _stamp(clock, counter, slot: tl.constexpr, stride: tl.constexpr,
           limit: tl.constexpr, begin: tl.constexpr):
    if begin:
        row = tl.load(counter)
        tl.store(counter, row + 1)
    else:
        row = tl.load(counter) - 1
    now = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", constraints="=l",
                                   args=[], dtype=tl.uint64, is_pure=False, pack=1)
    tl.store(clock + row * stride + slot, now, mask=(row >= 0) & (row < limit))


@triton.jit
def _snapshot(counter, counts, live, prefixes, valid, scores, prefix,
              roots: tl.constexpr, limit: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.load(counter) - 1
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = (i < roots) & (row >= 0) & (row < limit)
    n = tl.load(valid + i, mask=i < roots, other=0)
    active = tl.load(scores + i, mask=i < roots, other=0) > 0
    tl.store(counts + row * roots + i, n, mask=ok)
    tl.store(live + row * roots + i, active, mask=ok)
    if tl.program_id(0) == 0:
        p = tl.load(prefix)
        tl.store(prefixes + row, p, mask=(row >= 0) & (row < limit))


_executors = []
_epoch = 0
_installed = False


def instrument(ex, limit=2048):
    """Attach history to an executor before capture, preserving its model/RNG."""
    from ssd.utils.context import set_context, reset_context
    if hasattr(ex, '_phase_probe'):
        return
    history = dict(counter=torch.zeros((), device=ex.dev, dtype=torch.int64),
                   clock=torch.zeros((limit, 3 * ex.F), device=ex.dev, dtype=torch.uint64),
                   counts=torch.zeros((limit, ex.R), device=ex.dev, dtype=torch.int64),
                   live=torch.zeros((limit, ex.R), device=ex.dev, dtype=torch.uint8),
                   prefix=torch.zeros(limit, device=ex.dev, dtype=torch.int64),
                   steps=[], limit=limit)
    ex._phase_probe = history

    def run_once(n_pages0, finalize=True):
        rounds = ex.iter_rounds(n_pages0, finalize=finalize)
        request = next(rounds)
        f = 0
        while True:
            ids, rope, context = request
            set_context(**context)
            try:
                _stamp[(1,)](history['clock'], history['counter'], 3*f,
                             3*ex.F, limit, f == 0)
                hidden = ex.model(ids, rope)
                _stamp[(1,)](history['clock'], history['counter'], 3*f+1,
                             3*ex.F, limit, False)
                logits = ex.compute_logits(hidden, False).reshape(-1, ex.V).float()
                _stamp[(1,)](history['clock'], history['counter'], 3*f+2,
                             3*ex.F, limit, False)
            finally:
                reset_context()
            f += 1
            try:
                request = rounds.send(logits)
            except StopIteration:
                if f != ex.F:
                    raise AssertionError('Forward count changed')
                break

    ex.run_once = run_once
    capture = ex.capture
    def wrapped_capture(*args, **kwargs):
        if history['steps']:
            raise RuntimeError('Unexpected graph capture during measured request; prewarm first')
        result = capture(*args, **kwargs)
        history['counter'].zero_()
        return result
    ex.capture = wrapped_capture
    replay = ex.replay
    def wrapped_replay(n_pages0):
        from ssd.engine.helpers import cudagraph_helpers as profiling
        if len(history['steps']) >= limit:
            raise RuntimeError('Phase probe history exhausted; flush at request boundaries')
        result = replay(n_pages0)
        # P1 exposes every real root, including any zero-score candidate;
        # its larger context bucket also contains truly padded root rows.
        active = getattr(ex, 'in_root_valid', ex.in_root_piv)
        _snapshot[(triton.cdiv(ex.R, 128),)](
            history['counter'], history['counts'], history['live'], history['prefix'],
            ex.out_valid, active, ex.in_prefix_len, ex.R, limit, 128)
        history['steps'].append(dict(step_id=profiling._duet_context['step_id'],
                                     pages=n_pages0))
        return result
    ex.replay = wrapped_replay
    _executors.append(ex)


def drain(ex):
    """Call at a safe request boundary; never during overlapped phase work."""
    h = ex._phase_probe
    n = int(h['counter'].cpu())
    if n != len(h['steps']):
        raise AssertionError(f'Probe replay count mismatch: GPU {n}, CPU {len(h["steps"])}')
    clocks = h['clock'][:n].cpu().tolist()
    counts = h['counts'][:n].cpu().tolist()
    live = h['live'][:n].cpu().tolist()
    prefixes = h['prefix'][:n].cpu().tolist()
    records = []
    for i, meta in enumerate(h['steps']):
        timings = []
        for f in range(ex.F):
            start, middle, end = clocks[i][3*f:3*f+3]
            if not 0 < start < middle < end:
                raise AssertionError('Invalid device forward timestamps')
            timings.append(dict(round=f+1, width=ex.round_widths[f],
                                model_ms=(middle-start)/1e6,
                                head_ms=(end-middle)/1e6,
                                forward_ms=(end-start)/1e6))
        active = [n for n, flag in zip(counts[i], live[i]) if flag]
        if any(n > ex.NV for n in active):
            raise AssertionError('Generated nodes exceed configured cap')
        records.append(dict(**meta, phase=ex.phase, roots=ex.R,
                            active_roots=len(active), nodes=active,
                            physical_nodes=counts[i], live=live[i],
                            prefix_tokens=prefixes[i], C=ex.C, M=ex.NV,
                            round_widths=list(ex.round_widths), forward=timings))
    h['steps'].clear()
    h['counter'].zero_()
    return records


def flush(folder):
    global _epoch
    records = [r for ex in _executors if ex._phase_probe['steps'] for r in drain(ex)]
    if records:
        path = Path(folder) / f'phase_probe_request_{_epoch:04d}.json'
        path.write_text(json.dumps(dict(schema='duet-phase-probe-v1', epoch=_epoch,
            note='GPU globaltimer envelopes; marker kernels included. All-root counts captured on GPU. No per-step CPU readback.',
            records=records), indent=2)+'\n')
        _epoch += 1


def install():
    global _installed
    if _installed:
        return
    from ssd.engine.helpers.p2_tree_executor import P2TreeExecutor
    from ssd.engine.helpers import cudagraph_helpers as profiling
    from ssd.engine.draft_runner import DraftRunner
    original_init = P2TreeExecutor.__init__
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        instrument(self)
    P2TreeExecutor.__init__ = init
    original_prefill = DraftRunner.draft_async_prefill
    def prefill(self, *args, **kwargs):
        if _executors:
            flush(os.environ['SSD_PROFILE_DIR'])
        return original_prefill(self, *args, **kwargs)
    DraftRunner.draft_async_prefill = prefill
    original_exit = DraftRunner.exit
    def exit(self, *args, **kwargs):
        if _executors:
            flush(os.environ['SSD_PROFILE_DIR'])
        return original_exit(self, *args, **kwargs)
    DraftRunner.exit = exit
    _installed = True

