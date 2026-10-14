"""Benchmark-only steady-state population and per-batch fallback reporting.

Never subtract preparation seconds while retaining the same tokens. Exclude
the entire affected step from time, tokens, AL and hit metrics together.
Raw summaries are retained. CPU capture intervals are evidence of exposure,
not a counterfactual measure of time saved by removing preparation.
"""
from collections import Counter
from pathlib import Path
import json
import os
import sys
from mlsys_request_reporting import safe_tps


def configure_capture_audit():
    if os.getenv('SSD_PROFILE_DUET', '0') == '1':
        return
    if '--output' not in sys.argv:
        return
    output=Path(sys.argv[sys.argv.index('--output')+1]).resolve()
    os.environ.setdefault('DUET_REVIEW_CAPTURE_AUDIT', str(output.with_suffix('.captures')))
    os.environ.setdefault('SSD_PROFILE_DUET_DETAIL', '1')


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
    return s['batch']==batch and not s['cap'] and not s['clipped']


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
    folder=os.getenv('DUET_REVIEW_CAPTURE_AUDIT')
    if os.getenv('SSD_PROFILE_DUET','0')=='1':
        report['measurement_status']='instrumented CUDA profile; not TPS evidence'
        return
    if not folder:
        raise ValueError('Missing capture audit; cannot label a run preparation-free')
    captures=capture_intervals(folder)
    annotate(report['steps'],captures)
    # Loaded JSON duplicates group and flat step objects; handle both forms.
    for g in report['groups']:annotate(g['steps'],captures)
    out=summarize(report['steps'],report['args']['batch'],report['outputs'])
    report['summary_with_preparation']=report['summary']
    report['summary']=out.pop('summary')
    report['measurement']=out
    report['measurement_status']='excludes whole steps overlapping observed capture/setup intervals'
    report['measurement_note']='No duration outlier filter; raw time preserved. Successful full prewarm still needs a no-new-capture check.'
    report['capture_intervals']=captures
