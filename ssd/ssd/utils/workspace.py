"""Validate scratch budgets before capture; never resize captured buffers."""
import os
from contextlib import contextmanager


def workspace_bytes(variable, default_mib):
    value = os.environ.get(variable, str(default_mib))
    try:
        mib = int(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{variable} must be a positive integer MiB; got {value!r}") from exc
    if mib <= 0:
        raise ValueError(f"{variable} must be a positive integer MiB; got {value!r}")
    return mib * 2**20


@contextmanager
def planning_workspace(variable, workspace, **shape):
    """Attach allocation and shape to the original planner failure.

    The planner's message carries required bytes when available. This is
    deliberately fail-fast: changing storage after capture invalidates graphs.
    """
    try:
        yield
    except RuntimeError as exc:
        available = workspace.numel() * workspace.element_size()
        raise RuntimeError(
            f"FlashInfer plan failed: {variable}={available / 2**20:g} MiB "
            f"({available} bytes), shape={shape}. No captured buffer was "
            f"resized. If scratch is insufficient, increase {variable} before "
            f"initialization and rerun prewarm. Original error: {exc}"
        ) from exc
