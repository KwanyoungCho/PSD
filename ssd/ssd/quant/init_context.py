"""Thread-local flag that switches TP linear modules to quant-mode init.

In quant mode, `__init__` must not allocate a dense GPU `weight` tensor —
it allocates a `meta` placeholder instead. The adapter/loader then replaces
the placeholder with an `AwqQuantState` before forward is called.

This is the minimal implementation of plan §6.3.1 option (2).
"""
from __future__ import annotations

import threading
from contextlib import contextmanager


_state = threading.local()


def is_quant_init_active() -> bool:
    return bool(getattr(_state, "active", False))


@contextmanager
def quant_init_context(active: bool = True):
    """Activate quant-aware construction inside the with-block.

    While active, TP linear modules skip dense GPU weight allocation and
    allocate meta-device placeholders only.
    """
    prev = getattr(_state, "active", False)
    _state.active = active
    try:
        yield
    finally:
        _state.active = prev
