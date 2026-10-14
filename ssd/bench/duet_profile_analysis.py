"""Strict request/step alignment for chain AND dynamic-tree DUET profiles.

Sparse labels must never determine request epochs. GPU timestamps are used for
durations; CPU dispatch timestamps assign spans to the complete request stream.
No long-latency sample is silently censored. Only the documented warmup, first
two steps per request, and terminal step are excluded from reported statistics.
"""
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

VERSION = 'request-anchor-v2'


class ProfileAlignmentError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise ProfileAlignmentError(message)


def index_events(events, anchor_label, wanted):
    anchors = sorted((e for e in events if e.get('label') == anchor_label),
                     key=lambda e: e['cpu_dispatch_start_ns'])
    require(bool(anchors), f'Missing complete anchor stream: {anchor_label}')
    starts = [e['cpu_dispatch_start_ns'] for e in anchors]
    require(len(set(starts)) == len(starts), 'Duplicate anchor start')
    keys, epochs = [], []
    for e in anchors:
        sid = int(e['step_id'])
        if not epochs or sid <= epochs[-1][-1]:
            require(sid == 1, f'Request starts with unexpected step {sid}')
            epochs.append([])
        require(sid == len(epochs[-1]) + 1, 'Missing/noncontiguous request step')
        epochs[-1].append(sid)
        keys.append((len(epochs) - 1, sid))
    out = defaultdict(dict)
    for e in events:
        label = e.get('label')
        if label not in wanted:
            continue
        i = bisect_right(starts, e['cpu_dispatch_start_ns']) - 1
        require(i >= 0, f'{label} precedes first request')
        key = keys[i]
        require(int(e['step_id']) == key[1], f'{label}: step/anchor mismatch {key}')
        require(label not in out[key], f'Duplicate {label} at {key}')
        require(e['wall_end_ns'] >= e['wall_start_ns'], f'Negative duration {label}')
        out[key][label] = e
    return out, epochs


def read_profile(path, pattern):
    files = list(path.glob(pattern))
    require(len(files) == 1, f'Expected one {pattern}, found {len(files)}')
    return json.loads(files[0].read_text())


def parse_events(draft, target, records, cfg):
    target_labels = {'target_send_request', 'graph_pre', 'graph_post',
                     'final_logits', 'proxy_send_enqueue'}
    draft_labels = {'phase1_build', 'p1_root_build', 'p1_graph_replay',
                    'proxy_wait', 'merge_cache'}
    ti, te = index_events(target, 'target_send_request', target_labels)
    di, de = index_events(draft, 'draft_recv_request', draft_labels)
    require(te == de, 'Target/draft request boundaries or step counts disagree')
    active = [(i, r) for i, r in enumerate(records) if r['metrics']['phase_events']]
    expected = [[int(e['step_id']) for e in r['metrics']['phase_events']]
                for _, r in active]
    require(te[1:] == expected and len(te) == len(active) + 1,
            'Raw outcomes disagree with profile requests (one warmup required)')

    # Work with relative ns before converting to ms; preserve sub-ms precision.
    origin_ns = min(e['wall_start_ns'] for e in target if 'wall_start_ns' in e)
    def begin(e): return (e['wall_start_ns'] - origin_ns) / 1e6
    def end(e): return (e['wall_end_ns'] - origin_ns) / 1e6

    lags = []
    for key in ti:
        pw, ps = di[key].get('proxy_wait'), ti[key].get('proxy_send_enqueue')
        if pw and ps and end(pw) - begin(pw) > .2:
            lag = end(pw) - end(ps)
            if 0 <= lag < 5:
                lags.append(lag)
    transport = float(np.median(lags)) if lags else None
    rows, estimated_errors, p1_labels = [], [], Counter()
    eligible = 0
    for epoch, (record_i, record) in enumerate(active, start=1):
        events = record['metrics']['phase_events']
        prefix = record.get('prompt_tokens', record.get('input_tokens'))
        for idx, event in enumerate(events):
            sid = int(event['step_id'])
            cur, dr = ti[(epoch, sid)], di[(epoch, sid)]
            for label in target_labels:
                require(label in cur, f'Missing {label} at {(epoch, sid)}')
            for label in ('proxy_wait', 'merge_cache'):
                require(label in dr, f'Missing {label} at {(epoch, sid)}')
            # A tree-hit bypasses phase1_build. Root selection/preparation,
            # replay and output preparation all precede proxy_wait.
            p1_label = 'phase1_build' if 'phase1_build' in dr else 'p1_root_build'
            require(p1_label in dr, f'Missing complete P1 start at {(epoch, sid)}')
            if p1_label == 'p1_root_build':
                require('p1_graph_replay' in dr, 'Tree P1 root without replay')
            p1_start = begin(dr[p1_label])
            p1_end = begin(dr['proxy_wait'])
            require(p1_start <= end(dr[p1_label]) <= p1_end + .01,
                    f'P1 not contained before proxy wait at {(epoch, sid)}')
            if 'p1_graph_replay' in dr:
                require(p1_start <= begin(dr['p1_graph_replay']) and
                        end(dr['p1_graph_replay']) <= p1_end + .01,
                        'Tree replay outside P1 span')
            require(end(dr['proxy_wait']) <= begin(dr['merge_cache']) + .01,
                    'Merge precedes completed proxy receive')
            context = prefix
            prefix += event['accepted_len']
            if idx < 2 or idx == len(events) - 1:
                continue
            eligible += 1
            nxt = ti[(epoch, sid + 1)]
            o = begin(cur['graph_pre'])
            F = begin(nxt['target_send_request']) - o
            cache = end(dr['merge_cache']) - o
            wait = end(dr['proxy_wait']) - o
            S = begin(nxt['graph_pre']) - max(begin(nxt['target_send_request']),
                                             end(dr['merge_cache']))
            actual = begin(nxt['graph_pre']) - o
            D1, D2 = p1_end - p1_start, cache - wait
            require(min(F, D1, D2, actual) >= 0 and S >= -.5,
                    f'Impossible cross-request/clock ordering at {(epoch, sid)}')
            # Explicit estimate, not a direct measurement of uncensored arrival.
            P = end(cur['proxy_send_enqueue']) - o
            estimated_P = P + transport if transport is not None else None
            if estimated_P is not None:
                estimated_errors.append(max(F, max(p1_end-o, estimated_P)+D2)+S-actual)
            p1_labels[p1_label] += 1
            c, nc = event['source'], events[idx+1]['source']
            rows.append(dict(config=cfg, prompt=record_i, uid=record.get('uid', str(record_i)),
                epoch=epoch, step_id=sid, state=c, next_state=nc,
                n=cfg['k1'] if c == 1 else cfg['k2'],
                next_n=cfg['k1'] if nc == 1 else cfg['k2'], context=context,
                F=F, x=p1_start-o, D1=D1, D2=D2, P=estimated_P, S=S, cycle=actual,
                p1_span_label=p1_label, proxy_wait_ms=wait-(p1_end-o),
                gap1=estimated_P-(p1_end-o) if estimated_P is not None else None,
                gap1_send=P-(p1_end-o), gap2=F-cache,
                gap2_verify=end(cur['final_logits'])-o-cache))
    require(len(rows) == eligible and eligible > 0, 'Incomplete/empty eligible step coverage')
    return rows, dict(parser_version=VERSION, requests=len(active),
        profiled_steps=sum(map(len, te)), eligible_steps=eligible, rows=len(rows),
        coverage=1., dropped={}, p1_span_labels=dict(p1_labels),
        transport_ms=transport, transport_samples=len(lags),
        transport_p05_p95_ms=np.quantile(lags,[.05,.95]).tolist() if lags else None,
        estimated_arrival_reconstruction_mae_ms=float(np.mean(np.abs(estimated_errors)))
        if estimated_errors else None,
        notes='Warmup/first2/final excluded; no latency-outlier censoring. P1 deadline '
              'uses estimated arrival; gap2 is next-request readiness, gap2_verify is '
              'target forward completion including final LM head. GPU anchors have '
              'cross-process clock uncertainty; counts within 1ms need caution.')


def profile_rows(run):
    path = Path(run['path'])
    return parse_events(read_profile(path, 'duet_profile_draft_*.json'),
                        read_profile(path, 'duet_profile_target_rank0_*.json'),
                        run['records'], run['config'])
