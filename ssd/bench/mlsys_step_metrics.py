"""Performance population: complete nominal batches, excluding cap/clip steps.

This does not alter generation or discard raw counters. Every retained
event, emitted token and second belongs to exactly the same decode steps.
"""
from collections import Counter


def performance_summary(metrics, expected_batch):
    if expected_batch < 1:
        raise ValueError("expected_batch must be positive")
    events = metrics.get("phase_events", [])
    steps = metrics.get("decode_steps", [])
    reasons = Counter()
    kept = []
    kept_events = []
    for step in steps:
        start, end = step["event_start"], step["event_end"]
        row_events = events[start:end]
        batch = step.get("batch_size", len(row_events) if row_events else None)
        if batch is None:
            raise ValueError("Step lacks actual batch size; cannot infer AR batch from events")
        if batch > expected_batch:
            raise ValueError("Actual step batch exceeds the nominal batch")
        boundary = bool(step.get("output_cap_reached") or step.get("clipped"))
        short = batch != expected_batch
        if boundary: reasons["cap_or_clip"] += 1
        if short: reasons["partial_batch"] += 1
        if boundary and short: reasons["both"] += 1
        if boundary or short:
            continue
        kept.append(step)
        kept_events.extend(row_events)
    seconds = sum(s["seconds"] for s in kept)
    tokens = sum(s["emitted_tokens"] for s in kept)
    def mean(name):
        return sum(e[name] for e in kept_events) / len(kept_events) if kept_events else None
    return dict(
        policy="full_batch_without_cap_or_clip_v1", nominal_batch=expected_batch,
        tps=tokens / seconds if seconds > 0 else None,
        al=mean("accepted_len"), cache_hit=mean("cache_hit"),
        emitted_tokens=tokens, seconds=seconds, included_steps=len(kept),
        included_events=len(kept_events), total_steps=len(steps),
        excluded_steps=len(steps)-len(kept), exclusion_reasons=dict(reasons),
        included_step_indices=[i for i,s in enumerate(steps)
                               if not(s.get("output_cap_reached") or s.get("clipped"))
                               and s.get("batch_size", s["event_end"]-s["event_start"])==expected_batch])
