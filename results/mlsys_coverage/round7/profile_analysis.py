"""Measured-prompt target spans; do not add overlapping CUDA/host scopes."""
import collections
import json
import re
import statistics
from pathlib import Path

R = Path(__file__).resolve().parent


def stats(values):
    return dict(n=len(values), mean=statistics.mean(values),
                median=statistics.median(values)) if values else None


result = {}
for arm in ('base', 'unified'):
    raw = R / 'unified' / f'unified_b1_{arm}_diagnostic_r0.json'
    data = json.loads(raw.read_text())
    if data['status'] != 'complete':
        raise RuntimeError('Diagnostic is still running')
    log = raw.with_suffix('.log').read_text()
    paths = re.findall(r'events -> (\S*duet_profile_target_rank0_\S+\.json)', log)
    if not paths:
        raise ValueError('Target trace missing')
    trace_path = R / 'profiles' / arm / Path(paths[-1]).name
    events = json.loads(trace_path.read_text())
    grouped = []
    for event in events:
        if event['label'] == 'target_send_request' and event.get('step_id') == 1:
            grouped.append(collections.defaultdict(lambda: collections.defaultdict(list)))
        if grouped and event.get('step_id') is not None:
            grouped[-1][event['step_id']][event['label']].append(event)
    groups = data['warm_groups'] + data['groups']
    if len(grouped) != len(groups):
        raise AssertionError(('Prompt/trace boundary mismatch', arm, len(grouped), len(groups)))
    spans = collections.defaultdict(list)
    host = collections.defaultdict(list)
    by_source = collections.defaultdict(list)
    counts = collections.Counter()
    for group, labels in zip(groups[len(data['warm_groups']):], grouped[len(data['warm_groups']):]):
        captures = {sid for sid, labels_at_step in labels.items()
                    if any('capture' in name for name in labels_at_step)}
        excluded = captures | {sid + 1 for sid in captures}
        counts['capture_steps'] += len(captures)
        for step in group['steps']:
            sid = step['events'][0]['step_id']
            if step['batch'] != data['args']['batch'] or step['cap'] or step['clipped'] or sid in excluded:
                counts['excluded_measured_steps'] += 1
                continue
            counts['included_measured_steps'] += 1
            source = step['events'][0]['source']
            by_source[source].extend(v * 1000 for v in step['verify'])
            for name, values in labels[sid].items():
                spans[name].append(sum(v['cuda_ms'] for v in values))
                host[name].append(sum((v['cpu_dispatch_end_ns'] - v['cpu_dispatch_start_ns']) / 1e6
                                      for v in values))
    result[arm] = dict(counts=dict(counts), trace=str(trace_path.relative_to(R)),
                       verification_by_source_ms={k: stats(v) for k, v in by_source.items()},
                       cuda_span_ms={k: stats(v) for k, v in spans.items()},
                       host_dispatch_ms={k: stats(v) for k, v in host.items()})
out = dict(scope='16 measured prompts after 4 warm prompts. Nominal B, no cap/clip; exclude target capture steps and their next step. Means conditional on each label occurring. CUDA spans include host launch gaps; labels can overlap and are not summed into verification wall time.',
           arms=result)
(R / 'PROFILE_BREAKDOWN.json').write_text(json.dumps(out, indent=2))
print('Detailed profile analysis complete')
