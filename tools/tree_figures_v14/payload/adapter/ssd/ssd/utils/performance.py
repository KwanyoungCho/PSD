"""Whole-step performance selection without CUDA profiling.

This does not alter generation or discard raw counters. Every retained
event, emitted token and second belongs to exactly the same decode steps.
"""
from collections import Counter
from collections import defaultdict
import math
import os
from statistics import median


LONG_STEP_MEDIAN_FACTOR = 2.0
LONG_STEP_MIN_SAMPLES = 5


def profiling_enabled():
    return any(os.getenv(name, '0') == '1' for name in
               ('SSD_PROFILE_DUET', 'SSD_PROFILE', 'SSD_PROFILE_DRAFT'))


def duration_step_filter(rows, *, enabled=True):
    """One upper-tail pass, grouped by request scope, batch and cache mix.

    Each row has seconds, batch_size and events (CPU records). No model
    operation, GPU synchronization or mutation of input records occurs here.
    Scope is optional: a native generate call already provides one scope;
    benchmark campaigns assign a scope to each fixed request group.
    """
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        seconds = row['seconds']
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('Decode step time must be finite and positive')
        mix = Counter()
        for event in row.get('events', []):
            hit = event.get('cache_hit')
            key = ('miss' if hit == 0 else
                   f"hit_p{event.get('source', 0)}" if hit == 1 else
                   'speculative_without_cache_source')
            mix[key] += 1
        key = (row.get('duration_scope'), row['batch_size'], tuple(sorted(mix.items())))
        groups[key].append(index)
    excluded = []; references = []
    for (scope, batch, mix), indices in groups.items():
        center = median(rows[i]['seconds'] for i in indices)
        threshold = LONG_STEP_MEDIAN_FACTOR * center
        sufficient = len(indices) >= LONG_STEP_MIN_SAMPLES
        references.append(dict(scope=scope, batch_size=batch, cache_mix=dict(mix),
            samples=len(indices), median_seconds=center, threshold_seconds=threshold,
            applied=bool(enabled and sufficient)))
        if enabled and sufficient:
            excluded.extend(dict(index=i, seconds=rows[i]['seconds'],
                threshold_seconds=threshold, scope=scope, batch_size=batch,
                cache_mix=dict(mix)) for i in indices if rows[i]['seconds'] > threshold)
    removed = {r['index'] for r in excluded}
    return dict(policy='upper_step_duration_by_scope_batch_cache_mix_v1',
        enabled=bool(enabled), median_factor=LONG_STEP_MEDIAN_FACTOR,
        minimum_group_samples=LONG_STEP_MIN_SAMPLES, comparison='strictly_greater',
        grouping='generate/request-group scope, actual batch, P1/P2/other-hit/miss counts',
        excluded_steps=len(excluded), excluded_seconds=sum(r['seconds'] for r in excluded),
        excluded=sorted(excluded, key=lambda r:r['index']), groups=references,
        included_indices=[i for i in range(len(rows)) if i not in removed])


def performance_summary(metrics, expected_batch):
    if expected_batch < 1:
        raise ValueError("expected_batch must be positive")
    events = metrics.get("phase_events", [])
    steps = metrics.get("decode_steps", [])
    reasons = Counter()
    eligible = []
    for index, step in enumerate(steps):
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
        eligible.append((index, step, row_events, batch))
    selection = duration_step_filter([dict(seconds=s['seconds'], batch_size=b,
        events=es, duration_scope=s.get('duration_scope')) for _,s,es,b in eligible],
        enabled=not profiling_enabled())
    # Expose original indices, not indices into the cap/partial-filtered list.
    for row in selection['excluded']:
        row['index'] = eligible[row['index']][0]
    included = [eligible[i] for i in selection.pop('included_indices')]
    indices = [i for i,_,_,_ in included]
    selection['excluded_emitted_tokens'] = sum(
        steps[r['index']]['emitted_tokens'] for r in selection['excluded'])
    if selection['excluded_steps']:
        reasons['long_step'] = selection['excluded_steps']
    kept = [s for _,s,_,_ in included]
    kept_events = [e for _,_,es,_ in included for e in es]
    seconds = sum(s["seconds"] for s in kept)
    tokens = sum(s["emitted_tokens"] for s in kept)
    def mean(name):
        values = [e[name] for e in kept_events if e.get(name) is not None]
        return sum(values) / len(values) if values else None
    def source_stats(source):
        es = [e for e in kept_events if e.get('cache_hit') == 1 and e.get('source') == source]
        return dict(events=len(es), rate=len(es)/len(kept_events) if kept_events else None,
            accepted_spec_len=sum(e.get('accepted_spec_len',max(0,e['accepted_len']-1)) for e in es)/len(es) if es else None)
    hit_lengths = [e['accepted_len'] for e in kept_events if e.get('cache_hit') == 1]
    miss_lengths = [e['accepted_len'] for e in kept_events if e.get('cache_hit') == 0]
    return dict(
        policy="full_batch_without_cap_clip_or_long_step_v2", nominal_batch=expected_batch,
        tps=tokens / seconds if seconds > 0 else None,
        al=mean("accepted_len"), cache_hit=mean("cache_hit"),
        emitted_tokens=tokens, seconds=seconds, included_steps=len(kept),
        included_events=len(kept_events), total_steps=len(steps),
        excluded_steps=len(steps)-len(kept), exclusion_reasons=dict(reasons),
        included_step_indices=indices,
        step_ms=1000*seconds/len(kept) if kept else None,
        duration_filter=selection,
        before_duration_filter=dict(steps=len(eligible),
            seconds=sum(s['seconds'] for _,s,_,_ in eligible),
            emitted_tokens=sum(s['emitted_tokens'] for _,s,_,_ in eligible)),
        phase1=source_stats(1), phase2=source_stats(2),
        hit_lengths=hit_lengths, miss_lengths=miss_lengths)
