"""Opt-in token diagnostics, independent of Python assertion optimization."""
import os

def debug_tokens_enabled():
    return os.environ.get("SSD_DEBUG_TOKENS", "0") == "1"
