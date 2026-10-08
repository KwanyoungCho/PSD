"""CPU-only, auditable data preparation for the complete Round5 figure atlas."""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_screen import grouped
from result_metrics import collect, read

HERE = Path(__file__).resolve().parent
OUT = HERE / 'paper_view'
STATUS_ORDER = ['P1 hit', 'P2 hit', 'Hit', 'Mixed hits', 'Mixed hit/miss', 'Miss']
# Only disjoint/coarse stages: never add a replay child to its total parent.
TARGET = {
    'target_spec_wait': 'Wait / sync', 'batch_tree_prepare': 'Prepare',
    'batch_target_pre': 'Target pre', 'batch_target_proxy': 'Proxy / launch',
    'batch_target_post': 'Target post', 'batch_target_final_logits': 'Final logits',
    'batch_tree_accept': 'Accept', 'batch_tree_commit': 'Commit',
    'verify_replay': 'Target verify', 'verify_sample_accept': 'Accept',
    'verify_accept_prep': 'Accept prepare', 'target_postprocess': 'Postprocess',
}
DRAFT = {
    'batch_miss_draft': 'Miss draft', 'batch_glue': 'Context',
    'batch_p1_total': 'P1', 'batch_proxy_wait': 'Proxy wait', 'batch_p2_total': 'P2',
    'draft_recv_request': 'Request', 'hit_cache_respond_hit': 'Cache response',
    'hit_cache_respond_miss': 'Miss response', 'hit_cache_respond_mixed': 'Mixed response',
    'draft_send_response': 'Send response', 'glue': 'Context',
    'ssd_build_tree': 'SSD build', 'ssd_decode_tree': 'SSD decode',
    'ssd_populate_cache': 'Cache insert',
}


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def mean(xs):
    return float(np.mean(xs)) if len(xs) else None


def status(es, mode):
    hits = [bool(e['cache_hit']) for e in es]
    if not any(hits):
        return 'Miss'
    if not all(hits):
        return 'Mixed hit/miss'
    if mode == 'ssd':
        return 'Hit'
    sources = {e['source'] for e in es}
    return 'P1 hit' if sources == {1} else 'P2 hit' if sources == {2} else 'Mixed hits'


def parameters(raw):
    a, e, k = raw['args'], raw['env'], raw['engine_kwargs']
    duet = a['mode'] == 'duet-tree'
    p = dict(model='llama2' if 'llama2' in a['target'].lower() else 'llama3',
             method='DUET' if duet else 'SSD', target=Path(a['target']).name,
             draft=Path(a['draft']).name, target_dtype=raw.get('target_dtype'),
             draft_dtype=raw.get('draft_dtype'), target_tp=a['target_tp'],
             input_cap=a['input_cap'], output_cap=a['max_new_tokens'],
             max_model_len=a['max_model_len'], gpus=e['CUDA_VISIBLE_DEVICES'],
             K1=a['k1'] if duet else None, K2=a['k2'] if duet else None,
             SSD_K=k['speculate_k'] if not duet else None,
             SSD_F=k['async_fan_out'] if not duet else None,
             exit=a['exit_layer'] if duet else None, C=a['tree_width'] if duet else None,
             N1=k.get('duet_p1_tree_max_nodes') if duet else None,
             N2=k.get('duet_p2_tree_max_nodes') if duet else None,
             M1=k.get('duet_p1_tree_verify_nodes') if duet else None,
             M2=k.get('duet_p2_tree_verify_nodes') if duet else None,
             P1_roots=a['p1_roots'] if duet else None,
             draft_fanout=a['draft_fan_out'] if duet else None,
             proxy_fanout=a['proxy_fan_out'] if duet else None,
             W=(a['p2_budget'] if a['p2_budget'] is not None else
                a['proxy_fan_out'] * (max(a['k1'], a['k2']) + 1)) if duet else None,
             R=a['tree_root_count'] if duet else None,
             proxy_floor=a['tree_proxy_threshold'] if duet else None,
             confidence_floor=a['tree_conf_threshold'] if duet else None,
             beta_recorded_inactive=a['tree_beta'] if duet else None,
             root=a['root_source'] if duet else None,
             normalization=a['root_normalization'] if duet else None,
             overlap_mix=a['root_overlap_mix'] if duet else None,
             tree_policy=e.get('SSD_TREE_EXPANSION_POLICY') if duet else None,
             miss_K=raw.get('effective_features', {}).get('miss_depth') if duet else k['speculate_k'],
             trim=int(e.get('SSD_TREE_LADDER_TRIM', '0')) if duet else None,
             stream=int(e.get('SSD_BATCH_TREE_PROXY_STREAM', '0')) if duet else None,
             fused=int(e.get('SSD_TREE_FUSED_MATH', '0')) if duet else None,
             bulk=int(e.get('SSD_BATCH_TREE_BULK_EXPORT', '0')) if duet else None,
             parallel_insert=int(e.get('SSD_TREE_PARALLEL_INSERT', '0')) if duet else None,
             fast_verify=e.get('SSD_FAST_VERIFY'), backend=e.get('SSD_ATTN_BACKEND'),
             ignore_eos=a['ignore_eos'], greedy_only=a['greedy_only'],
             git_commit=raw['git_commit'])
    return p


def config_key(p, batch, temperature):
    ignored = {'input_cap', 'output_cap', 'max_model_len', 'gpus', 'git_commit'}
    # A companion is a same-setting diagnostic, not a same-workload replay.
    return json.dumps([batch, temperature, {k: v for k, v in p.items() if k not in ignored}], sort_keys=True)


def pass_role(raw, i):
    n = len(raw['cells'])
    if n == 6:
        return 'warmup' if i < 3 else 'measured-stability'
    if n == 2 and len({(c['batch'], c['temperature']) for c in raw['cells']}) == 1:
        return 'warmup' if i == 0 else 'measured'
    if raw['env'].get('SSD_PROFILE_DUET') == '1':
        return 'profile-diagnostic'
    return 'single-pass / smoke' if n == 1 else 'smoke-matrix'


def wall_breakdown(cell, mode):
    events = cell['metrics']['phase_events']
    buckets = collections.defaultdict(list)
    included = excluded = 0
    for s in cell['metrics']['decode_steps']:
        if s['output_cap_reached'] or s['clipped']:
            excluded += 1
            continue
        es = events[s['event_start']:s['event_end']]
        assert es
        buckets[status(es, mode)].append((s['seconds'] * 1000, s['emitted_tokens'], len(es)))
        included += 1
    total = sum(sum(v[0] for v in vs) for vs in buckets.values())
    rows = []
    for st in STATUS_ORDER:
        vs = buckets.get(st, [])
        if not vs:
            continue
        duration = sum(v[0] for v in vs)
        rows.append(dict(status=st, steps=len(vs), mean_ms=duration / len(vs),
                         p50_ms=float(np.median([v[0] for v in vs])),
                         p90_ms=float(np.quantile([v[0] for v in vs], .9)),
                         time_share=duration / total, time_ms=duration,
                         tokens=sum(v[1] for v in vs),
                         mean_active_batch=mean([v[2] for v in vs])))
    assert sum(r['steps'] for r in rows) == included
    return dict(groups=rows, included_steps=included, excluded_steps=excluded,
                total_ms=total, tps=sum(r['tokens'] for r in rows) / (total / 1000))


def interval_rows(group, labels, origin, lane):
    result = []
    for label, display in labels.items():
        for e in group.get(label, []):
            if 'wall_start_ns' not in e or 'wall_end_ns' not in e:
                continue
            start = e['wall_start_ns'] / 1e6 - origin
            end = e['wall_end_ns'] / 1e6 - origin
            assert end >= start
            result.append(dict(label=display, start=start, end=end, lane=lane, original_label=label))
    return result


def union_length(spans):
    cursor = None
    length = 0.
    for start, end in sorted(spans):
        length += max(0., end - max(start, cursor if cursor is not None else start))
        cursor = max(cursor if cursor is not None else end, end)
    return length


def profile_detail(screen, raw):
    tg = grouped(HERE / screen['profile_paths'][0], 'target_send_request')
    dg = grouped(HERE / screen['profile_paths'][1],
                 'batch_miss_draft' if raw['args']['mode'] == 'duet-tree' else 'draft_recv_request')
    cell = raw['cells'][-1]
    events = cell['metrics']['phase_events']
    capture = {sid for sid, labels in list(tg.items()) + list(dg.items())
               if any('capture' in label for label in labels)}
    blocked = capture | {sid + 1 for sid in capture}
    groups = collections.defaultdict(list)
    counts = collections.Counter()
    overlap_max = 0.
    for s in cell['metrics']['decode_steps']:
        es = events[s['event_start']:s['event_end']]
        sid = es[0]['step_id']
        if sid < 20:
            counts['initial'] += 1
            continue
        if sid in blocked:
            counts['capture_or_next'] += 1
            continue
        if len(es) != cell['batch']:
            counts['partial_batch'] += 1
            continue
        if s['output_cap_reached'] or s['clipped']:
            counts['output_boundary'] += 1
            continue
        t, d = tg.get(sid, {}), dg.get(sid, {})
        required_t = ['target_spec_wait', 'target_postprocess']
        required_d = ['batch_glue', 'batch_p1_total', 'batch_p2_total', 'batch_proxy_receive'] if raw['args']['mode'] == 'duet-tree' else ['glue', 'ssd_build_tree', 'ssd_decode_tree', 'ssd_populate_cache']
        if any(k not in t for k in required_t) or any(k not in d for k in required_d):
            counts['incomplete_trace'] += 1
            continue
        origin = t['target_spec_wait'][0]['wall_start_ns'] / 1e6
        end = t['target_postprocess'][-1]['wall_end_ns'] / 1e6 - origin
        if end <= 0:
            counts['invalid_window'] += 1
            continue
        target = interval_rows(t, TARGET, origin, 'Target')
        proxy = interval_rows(t, {'batch_proxy_side': 'Proxy compute'}, origin, 'Proxy stream')
        draft = interval_rows(d, DRAFT, origin, 'Draft current')
        # Earlier P2/cache-build work can explain this step's target wait.
        previous = interval_rows(dg.get(sid - 1, {}),
                                 {'batch_p2_total': 'Previous P2', 'ssd_decode_tree': 'Previous SSD decode',
                                  'ssd_populate_cache': 'Previous cache insert'}, origin, 'Draft previous')
        previous = [v for v in previous if v['end'] > 0 and v['start'] < end]
        span = [(max(0., x['start']), min(end, x['end'])) for x in target]
        union = union_length(span)
        overlap_max = max(overlap_max, sum(b-a for a,b in span)-union)
        unlabelled = max(0., end-union)
        # Explicit gaps make the target's total auditable without calling them CPU compute.
        cursor = 0.
        gaps = []
        for a, b in sorted(span):
            if a > cursor:
                gaps.append(dict(label='Unlabelled gap', start=cursor, end=a, lane='Target', original_label='derived_gap'))
            cursor = max(cursor, b)
        if cursor < end:
            gaps.append(dict(label='Unlabelled gap', start=cursor, end=end, lane='Target', original_label='derived_gap'))
        target += gaps
        source_count = collections.Counter('hit' if raw['args']['mode'] == 'ssd' and e['cache_hit'] else
                                           str(e['source']) for e in es)
        markers = {'Target ready': end}
        if 'batch_target_final_logits' in t:
            markers['Final logits'] = t['batch_target_final_logits'][-1]['wall_end_ns']/1e6-origin
        if 'batch_proxy_receive' in d:
            markers['Proxy received'] = d['batch_proxy_receive'][-1]['wall_end_ns']/1e6-origin
        x = dict(step_id=sid, status=status(es, raw['args']['mode']), target_ms=end,
                 decode_ms=1000*s['seconds'], source_counts=dict(source_count),
                 physical_queries=es[0].get('physical_batch_queries'),
                 max_valid_nodes=max(e['valid_k'] for e in es), unlabelled_target_ms=unlabelled,
                 intervals=target+proxy+draft, previous=previous, markers=markers)
        groups[x['status']].append(x)
    result = []
    for st in STATUS_ORDER:
        samples = groups.get(st, [])
        if not samples:
            continue
        center = float(np.median([s['target_ms'] for s in samples]))
        representative = min(samples, key=lambda s: abs(s['target_ms']-center))
        # One population for every component; optional absent work contributes zero duration.
        # Mean aligned offsets are a schematic, not an observed single step.
        labels = sorted({(v['lane'], v['label']) for s in samples for v in s['intervals']})
        common = []
        components = []
        for lane, label in labels:
            starts, ends, durations = [], [], []
            for s in samples:
                vs = [v for v in s['intervals'] if v['lane'] == lane and v['label'] == label]
                if vs:
                    starts.append(min(v['start'] for v in vs))
                    ends.append(max(v['end'] for v in vs))
                durations.append(sum(v['end']-v['start'] for v in vs))
            components.append(dict(lane=lane, label=label, mean_ms=mean(durations),
                                   present_steps=len(starts), population=len(samples)))
            # Gaps are not contiguous; do not turn their min/max into a false stage.
            # Conditional optional spans are shown in representative panels, not mean schematic.
            if len(starts) == len(samples) and label != 'Unlabelled gap':
                common.append(dict(lane=lane, label=label, start=mean(starts), end=mean(ends)))
        result.append(dict(status=st, n=len(samples), mean_target_ms=mean([s['target_ms'] for s in samples]),
                           mean_decode_ms=mean([s['decode_ms'] for s in samples]),
                           mean_intervals=common, components=components,
                           mean_markers={k:mean([s['markers'][k] for s in samples if k in s['markers']])
                                         for k in {k for s in samples for k in s['markers']}},
                           representative=representative,
                           step_ids=[s['step_id'] for s in samples],
                           query_shapes=dict(collections.Counter(str(s['physical_queries']) for s in samples))))
    return dict(groups=result, excluded=dict(counts), eligible_steps=sum(g['n'] for g in result),
                trace_paths=screen['profile_paths'], labelled_capture_count=len(capture),
                max_target_stage_overlap_ms=overlap_max,
                detail_labels_present=any('batch_tree_accept' in t for t in tg.values()))


def build_data():
    screen = read(HERE / 'SCREEN_RESULTS.json')
    inventory = read(HERE / 'INVENTORY.json')
    complete = {(r['directory'], r['name']) for r in inventory['runs'] if r['status'] == 'complete'}
    assert len(complete) == len(screen) == 487
    rows, passes = [], []
    for i, sc in enumerate(screen):
        raw = read(HERE / sc['path'])
        assert (Path(sc['path']).parent.name, sc['name']) in complete
        ps = parameters(raw)
        summaries = []
        for j, c in enumerate(raw['cells']):
            m = collect(raw, j)
            m.pop('per_question')
            m.pop('question_indexes')
            m.pop('query_shape_counts')
            m.update(name=sc['name'], path=sc['path'], pass_index=j, pass_role=pass_role(raw,j),
                     model=ps['model'], method=ps['method'], instrumented=bool(sc.get('profile_paths')),
                     timing_excluded=sc['timing_excluded'], input_cap=ps['input_cap'], output_cap=ps['output_cap'])
            summaries.append(m)
            passes.append(m)
        last = summaries[-1]
        wall = wall_breakdown(raw['cells'][-1], raw['args']['mode'])
        assert np.isclose(wall['tps'], last['boundary_excluded_step_tps'], rtol=1e-12)
        detail = profile_detail(sc, raw) if sc.get('profile_paths') else None
        row = dict(name=sc['name'], path=sc['path'], parameters=ps,
                   summaries=summaries, last=last, wall=wall, detail=detail,
                   full_parameters=dict(args=raw['args'], engine_kwargs=raw['engine_kwargs'],
                                        env=raw['env'], effective_features=raw['effective_features'],
                                        resolved_env=raw.get('resolved_env'),
                                        runtime_source_sha256=raw.get('runtime_source_sha256')),
                   config_key=config_key(ps, last['batch'], last['temperature']))
        rows.append(row)
        if (i+1)%40 == 0:
            print(f'Data {i+1}/{len(screen)}', flush=True)
    assert len(passes) == 691
    assert sum(bool(r['detail']) for r in rows) == 337
    for r in rows:
        matches = [p for p in rows if p['detail'] and p['detail']['eligible_steps'] and
                   not p['last']['timing_excluded'] and p['config_key'] == r['config_key']]
        matches.sort(key=lambda p:(p['parameters']['input_cap'] != r['parameters']['input_cap'],
                                   p['parameters']['output_cap'] != r['parameters']['output_cap'],
                                   not p['detail']['detail_labels_present'], -p['detail']['eligible_steps'], p['name']))
        r['companion_profiles'] = [p['name'] for p in matches]
    payload = dict(rows=rows, passes=passes, failures=[r for r in inventory['runs'] if r['status'] != 'complete'],
                   source_sha256={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in
                                  ['SCREEN_RESULTS.json','INVENTORY.json','POSTOPT_RESULTS.json','FINAL_RESULTS.json','FOLLOWUP_RESULTS.json']})
    dump(OUT/'DATA.json', payload)
    print('Data complete:',len(rows),'jobs,',len(passes),'passes',flush=True)
    return payload


if __name__ == '__main__':
    build_data()
