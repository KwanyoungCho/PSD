"""Report a queue's observed state, not the name of its last launched job.

Reusable by server-specific report adapters. Caller checks PID identity (not
just PID reuse) and supplies a monotonic/UTC-consistent heartbeat age.
"""
import math

LEVELS = ('diagnostic','correctness','model_functional','performance')


def queue_state(*, planned, validated, active_job=None, process_alive=False,
                heartbeat_age=None, stale_after=120, failed=False,
                stop_requested=False, waiting=False, queue_exhausted=False):
    if planned < 0 or not 0 <= validated <= planned:
        raise ValueError('Invalid planned/validated population')
    if stop_requested:
        return 'stopped'
    if failed:
        return 'needs_attention'
    if active_job is not None:
        if (process_alive and heartbeat_age is not None
                and math.isfinite(heartbeat_age) and 0 <= heartbeat_age <= stale_after):
            return 'running'
        return 'needs_attention'
    if planned and validated == planned:
        return 'complete'
    if waiting:
        return 'waiting'
    if queue_exhausted:
        return 'drained'
    return 'needs_attention'


def validation_record(level, success, *, reason=None, evidence=None):
    if level not in LEVELS:
        raise ValueError(f'Unknown validation level: {level}')
    return dict(level=level, success=bool(success), reason=reason, evidence=evidence)
