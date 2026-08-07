from dataclasses import dataclass

import torch
import torch.distributed as dist


def concat_int64(*tensors: torch.Tensor) -> torch.Tensor:
    """Concatenate tensors into a single flat int64 payload."""
    parts = []
    for t in tensors:
        if t is None:
            continue
        if t.dtype != torch.int64:
            t = t.to(torch.int64)
        parts.append(t.reshape(-1))
    if not parts:
        return torch.empty(0, dtype=torch.int64)
    return torch.cat(parts, dim=0)


def send_int64(pg, dst: int, *tensors: torch.Tensor):
    """Send many int64-compatible tensors as one fused payload in a fixed order."""
    payload = concat_int64(*tensors)
    if payload.numel() == 0:
        return
    dist.send(payload, dst=dst, group=pg)


def recv_int64(pg, src: int, total_length: int, device: torch.device) -> torch.Tensor:
    """Receive a fused int64 payload of known total length."""
    t = torch.empty((total_length,), dtype=torch.int64, device=device)
    if total_length > 0:
        dist.recv(t, src=src, group=pg)
    return t


@dataclass
class AsyncSendSlot:
    """One slot of an AsyncSendRing.

    The persistent ``buf`` holds the most recently issued payload; ``work`` is
    the outstanding :class:`dist.Work` handle returned by ``dist.isend`` (or
    ``None`` if the slot is idle). ``wait_count`` / ``wait_ms_total`` track
    how often / how long we CPU-blocked when the slot was reused while its
    previous send was still in flight (see ``AsyncSendRing.send``).
    """

    buf: torch.Tensor              # persistent GPU int64 [buf_size]
    work: object | None = None     # outstanding dist.Work
    wait_count: int = 0            # slot-reuse wait fires
    wait_ms_total: float = 0.0     # cumulative CPU wait at slot reuse


class AsyncSendRing:
    """N-slot round-robin for non-blocking proxy send.

    Each slot owns a persistent int64 GPU buffer plus an outstanding
    :class:`dist.Work` handle. ``send`` packs the payload into the next
    slot's buffer (no allocation on the hot path) and issues
    ``dist.isend``; the handle is stored on the slot. Before reusing a
    slot we CPU-wait on the previous send (rare path; ``n_slots`` is
    tuned so this almost never fires). ``drain`` is the explicit cleanup
    hook called from engine shutdown.

    See ``ssd/docs/duet/08-proxy-overlap-experiment.md`` §3 for the full
    design rationale.
    """

    def __init__(self, n_slots: int, buf_size: int, device, pg):
        self.slots = [
            AsyncSendSlot(
                buf=torch.empty(buf_size, dtype=torch.int64, device=device)
            )
            for _ in range(n_slots)
        ]
        self.next_idx = 0
        self.pg = pg

    def send(self, dst: int, *tensors: torch.Tensor):
        slot = self.slots[self.next_idx]

        # Slot-reuse safety: rare path (n_slots tuned so this almost never fires).
        # ``work.wait()`` is a CPU-side blocking call, so we measure it with
        # ``time.perf_counter`` — CUDA events would not capture the CPU wait
        # and would add hot-path event overhead (see doc 08 §3.1 reviewer #5).
        if slot.work is not None:
            import time
            t0 = time.perf_counter()
            slot.work.wait()
            slot.wait_ms_total += (time.perf_counter() - t0) * 1000.0
            slot.wait_count += 1
            slot.work = None

        # Copy payload into the persistent slot buffer (no alloc on hot path).
        # copy_ runs on the current CUDA stream (proxy_stream when called from
        # within the proxy callback). We deliberately do NOT use
        # ``concat_int64`` here — that allocates a fresh tensor every call,
        # which defeats the point of the ring buffer.
        offset = 0
        for t in tensors:
            n = t.numel()
            if offset + n > slot.buf.numel():
                raise RuntimeError(
                    f"AsyncSendRing payload overflow: need={offset + n}, "
                    f"capacity={slot.buf.numel()}")
            slot.buf[offset:offset + n].copy_(t.view(-1), non_blocking=True)
            offset += n

        # Async send. dist.isend returns immediately with a Work handle.
        # PyTorch's NCCL backend stream-orders the isend kernel after the
        # preceding copy_ on the current stream (see doc 08 §3.1 reviewer #3).
        slot.work = dist.isend(slot.buf[:offset], dst=dst, group=self.pg)

        self.next_idx = (self.next_idx + 1) % len(self.slots)

    def drain(self):
        """Explicit cleanup. Called from engine shutdown path."""
        for slot in self.slots:
            if slot.work is not None:
                slot.work.wait()
                slot.work = None

    def dump_stats(self, outdir: str, decode_steps_seen: int):
        """Persist slot wait counters so Phase 4 / summarize can compute
        slot_wait_rate. Called by Verifier.drain_proxy_send_ring() under
        SSD_PROFILE_DUET=1 only — gate kept by caller (see doc 08 §3.1)."""
        import json
        out = {
            "slots": [
                {
                    "idx": i,
                    "wait_count": s.wait_count,
                    "wait_ms_total": s.wait_ms_total,
                }
                for i, s in enumerate(self.slots)
            ],
            "decode_steps_seen": decode_steps_seen,
        }
        with open(f"{outdir}/proxy_send_ring_stats.json", "w") as f:
            json.dump(out, f)
