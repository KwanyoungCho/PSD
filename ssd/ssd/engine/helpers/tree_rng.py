"""Stable per-phase/context/lane RNG domains; no per-step work."""
import hashlib
import struct


def tree_rng_seed(seed: int, phase: str, contexts: int, lane: int = 0) -> int:
    if phase not in ('p1', 'p2') or contexts < 1 or lane < 0:
        raise ValueError('invalid tree RNG domain')
    payload = struct.pack('<QQQQ', int(seed) & ((1 << 64) - 1),
                          int(phase[-1]), int(contexts), int(lane))
    return int.from_bytes(hashlib.blake2b(
        payload, digest_size=8, person=b'DUET-tree-rng-v1').digest(), 'little')
