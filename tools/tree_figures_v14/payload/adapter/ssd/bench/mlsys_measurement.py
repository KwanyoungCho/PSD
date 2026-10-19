"""Performance population from complete decode-step durations.

Never subtract preparation seconds while retaining the same tokens. Exclude
the entire affected step from time, tokens, AL and hit metrics together.
Raw summaries are retained. Legacy capture helpers remain available for
diagnostics, but ordinary performance measurement does not require them.
"""
from collections import Counter
import hashlib
import importlib.util
from pathlib import Path
import json
import os
from mlsys_request_reporting import safe_tps

# A/B runs can import the engine from an older PAPER_CHECKOUT. Load the
# aggregation policy beside this adapter so both arms use the current rule.
_POLICY_FILE = Path(__file__).resolve().parents[1]/'ssd'/'utils'/'performance.py'
_POLICY_SPEC = importlib.util.spec_from_file_location('_mlsys_measurement_policy', _POLICY_FILE)
_POLICY = importlib.util.module_from_spec(_POLICY_SPEC)
_POLICY_SPEC.loader.exec_module(_POLICY)
_POLICY_SHA256 = hashlib.sha256(_POLICY_FILE.read_bytes()).hexdigest()


def configure_capture_audit():
    # Performance uses existing step clocks. Explicit diagnostic audit env
    # settings are still honored by the optional capture adapter itself.
    return


def capture_intervals(folder):
    rows={}
    for path in sorted(Path(folder).glob('*.jsonl')):
        for line in path.read_text().splitlines():
            try: row=json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f'Incomplete capture record in {path}; aggregate after workers finish')
            key=(row['pid'],row['label'],row['start_ns'])
            rows.setdefault(key,{}).update(row)
    return list(rows.values())


def overlap_ns(start,end,captures):
    intervals=[]
    for c in captures:
        lo=max(start,c['start_ns']);hi=min(end,c.get('end_ns',end))
        if hi>lo:intervals.append((lo,hi))
    total=0;right=start
    for lo,hi in sorted(intervals):
        total+=max(0,hi-max(lo,right));right=max(right,hi)
    return total


def base_eligible(s,batch):
    return (s['batch']==batch and not s['cap'] and not s['clipped']
            and s.get('performance_included', True))


def performance_eligible(s,batch):
    if 'performance_included' in s:
        return base_eligible(s,batch)
    # Historical reports used capture-interval selection; do not silently
    # reinterpret their recorded population when reading an old result.
    return base_eligible(s,batch) and not s.get('graph_preparation',False)


def annotate(steps,captures):
    for s in steps:
        if 'audit_start_ns' not in s or 'audit_end_ns' not in s:
            raise ValueError('Capture-free metrics require step timestamps, not a latency cutoff')
        ns=overlap_ns(s['audit_start_ns'],s['audit_end_ns'],captures)
        s['graph_preparation']=ns>0
        s['graph_preparation_overlap_seconds']=ns/1e9


def batch_metrics(steps,batch):
    hist=Counter();jit_hist=Counter();ar_hist=Counter();hits=0;count=0
    for s in steps:
        es=s['events']
        if len(es)!=batch:
            raise ValueError('Batch-hit metrics require one verification event per request')
        if any('valid_k' not in e for e in es):
            raise ValueError('Cannot infer actual JIT fallback without recorded proposal lengths')
        misses=[e for e in es if not e['cache_hit']]
        m=len(misses);hist[m]+=1;hits+=batch-m;count+=1
        # These DUET arms use positive-length JIT for served miss proposals
        # and zero-length target AR for skipped miss proposals. This measures
        # actual served fallback, not the number of draft forward calls.
        jit_hist[sum(e['valid_k']>0 for e in misses)]+=1
        ar_hist[sum(e['valid_k']==0 for e in misses)]+=1
    if not count:return dict(steps=0,request_hit=None,miss_histogram={})
    return dict(steps=count,request_hit=hits/(batch*count),
        all_hit=hist[0]/count,any_miss=1-hist[0]/count,
        mixed=sum(n for m,n in hist.items() if 0<m<batch)/count,
        all_miss=hist[batch]/count,
        actual_jit_step=1-jit_hist[0]/count,
        actual_jit_request=sum(m*n for m,n in jit_hist.items())/(batch*count),
        miss_ar_step=1-ar_hist[0]/count,
        miss_ar_request=sum(m*n for m,n in ar_hist.items())/(batch*count),
        hypothetical_jit_step_if_all_partial_misses_use_ar=hist[batch]/count,
        miss_histogram={str(m):hist[m] for m in range(batch+1)})


def summarize(steps,batch,outputs):
    original=[s for s in steps if base_eligible(s,batch)]
    kept=[s for s in original if not s['graph_preparation']]
    es=[e for s in kept for e in s['events']]
    secs=sum(s['seconds'] for s in kept);tokens=sum(s['emitted'] for s in kept)
    mean=lambda xs:sum(xs)/len(xs) if xs else None
    summary=dict(policy='full_batch_without_cap_clip_or_observed_preparation_v2',
        steps=len(kept),all_steps=len(steps),tokens=tokens,seconds=secs,
        tps=safe_tps(tokens,secs,len(kept)),al=mean([e['accepted_len'] for e in es]),
        hit=mean([e['cache_hit'] for e in es]),
        verify_ms=mean([v*1000 for s in kept for v in s.get('verify',[])]),
        step_ms=mean([s['seconds']*1000 for s in kept]),
        actual_output_tokens=sum(map(len,outputs)),
        emitted_decode_tokens=sum(s['emitted'] for s in steps))
    def prep(population):
        total=sum(s['seconds'] for s in population)
        marked=[s for s in population if s['graph_preparation']]
        overlap=sum(s['graph_preparation_overlap_seconds'] for s in population)
        return dict(steps=len(population),seconds=total,affected_steps=len(marked),
            affected_step_seconds=sum(s['seconds'] for s in marked),
            interval_union_seconds=overlap,
            interval_union_fraction=overlap/total if total else None)
    batch_report=batch_metrics(kept,batch) if es else None
    after_startup=[s for s in kept if s['events'] and s['events'][0]['step_id']>1]
    return dict(summary=summary,batch_metrics=batch_report,
        batch_metrics_excluding_group_startup=batch_metrics(after_startup,batch) if es else None,
        preparation_full_batch=prep(original),preparation_all_decode=prep(steps))


def finalize_measurement(report):
    """Called at end of a benchmark; raw JSON rows remain available."""
    if _POLICY.profiling_enabled():
        report['measurement_status']='instrumented CUDA profile; not TPS evidence'
        return
    steps=report['steps'];batch=report['args']['batch']
    scopes=[];copies=[]
    for scope,g in enumerate(report['groups']):
        for s in g['steps']:
            scopes.append(scope);copies.append(s)
    if len(copies)!=len(steps):
        raise ValueError('Group/flat step population mismatch')
    events=[];native=[]
    for i,(s,copy) in enumerate(zip(steps,copies)):
        if any(s[k]!=copy[k] for k in ('seconds','batch','emitted')):
            raise ValueError('Group/flat step order mismatch')
        start=len(events);events.extend(s['events'])
        native.append(dict(seconds=s['seconds'],emitted_tokens=s['emitted'],batch_size=s['batch'],
            event_start=start,event_end=len(events),output_cap_reached=s['cap'],clipped=s['clipped'],
            duration_scope=scopes[i]))
    selected=_POLICY.performance_summary(dict(decode_steps=native,phase_events=events),batch)
    indices=set(selected['included_step_indices'])
    long_indices={r['index'] for r in selected['duration_filter']['excluded']}
    for i,(s,copy) in enumerate(zip(steps,copies)):
        for row in (s,copy):
            row['performance_included']=i in indices
            row['long_step_outlier']=i in long_indices
    kept=[steps[i] for i in selected['included_step_indices']]
    es=[e for s in kept for e in s['events']]
    mean=lambda xs:sum(xs)/len(xs) if xs else None
    if 'summary_unfiltered' not in report:
        report['summary_unfiltered']=report['summary']
        # Compatibility alias: this is the original full-batch/cap-only row.
        report['summary_with_preparation']=report['summary']
    report['summary']=dict(policy=selected['policy'],steps=len(kept),all_steps=len(steps),
        tokens=selected['emitted_tokens'],seconds=selected['seconds'],tps=selected['tps'],
        excluded_long_steps=selected['duration_filter']['excluded_steps'],
        al=selected['al'],hit=selected['cache_hit'],step_ms=selected['step_ms'],
        verify_ms=mean([v*1000 for s in kept for v in s.get('verify',[])]),
        actual_output_tokens=sum(map(len,report['outputs'])),emitted_decode_tokens=sum(s['emitted'] for s in steps))
    after_startup=[s for s in kept if s['events'] and s['events'][0]['step_id']>1]
    report['measurement']=dict(policy='complete_step_duration_filter_v1',
        policy_source_sha256=_POLICY_SHA256,
        included_step_indices=selected['included_step_indices'],
        duration_filter=selected['duration_filter'],exclusion_reasons=selected['exclusion_reasons'],
        included_steps=len(kept),excluded_steps=selected['excluded_steps'],
        before_duration_filter=selected['before_duration_filter'],
        batch_metrics=batch_metrics(kept,batch) if es else None,
        batch_metrics_excluding_group_startup=batch_metrics(after_startup,batch) if es else None,
        raw_request_population='Request records retain complete output/time; not duration-filtered aggregate performance')
    report['measurement_status']='whole-step duration filtering; no CUDA profiling or capture inspection required'
    report['measurement_note']='Same fixed rule for all methods. Raw steps, tokens and times retained; excluded steps leave both performance numerator and denominator. Long elapsed time alone does not identify its cause.'
