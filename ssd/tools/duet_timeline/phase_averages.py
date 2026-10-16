"""Source-conditioned phase averages from complete measured steps.

Export compact timestamp records so another server can recompute the means
without the original profile paths. No inferred transport delay or clipping of
negative margins. Graph/K is a per-round average, not isolated model latency.
"""
import argparse
import gzip
import json
from pathlib import Path

from plot_model_breakdowns import load, one, read_arm, require, sha, STATES, union_ms
from plot_consecutive_steps import read_profiles, transition
from plot_model_wait_timelines import command_index

FIELDS = ('step_ms', 'ds_total_ms', 'ds_margin_ms', 'ds_round_ms',
          'ps_total_ms', 'ps_margin_ms', 'ps_round_ms')
FORWARD_FIELDS = ('ds_forward_once_ms','ds_compute_window_ms','ps_forward_once_ms','ps_compute_window_ms')
FORWARD_COLUMNS = ('step_ms','ds_forward_once_ms','ds_compute_window_ms','ds_margin_ms',
                   'ps_forward_once_ms','ps_compute_window_ms','ps_margin_ms')
FORWARD_HEADERS = ['Current outcome','Steps','Step (ms)','DS forward (ms)',
                   'DS compute* (ms)','DS spare* (ms)','PS forward (ms)','PS compute* (ms)','PS spare (ms)']
HEADERS = ['Current outcome', 'Steps', 'Step (ms)', 'DS total (ms)',
           'DS spare* (ms)', 'DS graph/K1 (ms)', 'PS total (ms)',
           'PS spare (ms)', 'PS graph/K2 (ms)']
PERFORMANCE_HEADERS = ['Current outcome', 'Steps', 'Rate (%)',
                       'AL (tokens/step)', 'Profile TPS (tokens/s)']
DEFINITIONS = {
    'grouping': 'Current step cache source; DS/PS prepare future outcomes.',
    'weighting': 'Arithmetic mean over eligible steps, including all repetitions; overall is step-weighted.',
    'step_ms': 'Next target_send_request.start minus current target_send_request.start, same request.',
    'ds_total_ms': 'P1 root/phase preparation start to proxy_wait.start; excludes glue and proxy wait.',
    'ds_margin_ms': 'Target proxy_send_enqueue.end minus proxy_wait.start (DS complete). Send-side reference, not measured arrival.',
    'ds_round_ms': 'P1 graph replay elapsed / K1; includes sampling/selection inside graph.',
    'ps_total_ms': 'proxy_wait.end to merge_cache.end.',
    'ps_margin_ms': 'Next target_send_request.start minus next draft_recv_cmd.start; includes post-cache cleanup.',
    'ps_round_ms': 'P2 graph replay elapsed / K2; includes sampling/selection inside graph.',
    'share_pct': 'Number of steps in the source divided by all eligible steps; Overall = 100%.',
    'hit_rate_pct': 'Hit fraction within the row. P1/P2 = 100%, miss = 0%; Overall = (P1+P2)/all.',
    'al': 'Mean accepted_len including correction/recovery or bonus token, from recorded verification outcomes.',
    'tps': 'Sum of actually emitted tokens / sum of target-request-start-to-next-start seconds on the same eligible profiled steps. Ratio of totals, not mean of per-step TPS; not full-dataset or unprofiled TPS.',
    'limitations': 'Cross-device clocks have alignment uncertainty. Signed spare time is not guaranteed usable compute budget. Samples without optional forward probes contain graph/K only, not per-forward timings.',
    'optional_forward_probe': 'When present: actual model+LMhead GPU envelopes, including timing markers. Compute window = sum of recorded forwards + signed spare. Current non-forward work held fixed; not a guarantee for a larger K.',
    'native_chain_forward': 'Chain samples use the sum of K separately recorded phase1/phase2 replay events: model+LMhead and captured output-buffer stores, excluding between-round planning/sampling. Their legacy ds/ps_round_ms is first-to-last replay envelope / K, not the isolated forward sum/K.',
}


def sample(groups, commands, row, k1, k2, run_id):
    epoch, sid = row['epoch'], row['step_id']
    d, t = groups['draft'][(epoch, sid)], groups['target'][(epoch, sid)]
    tr = transition(groups, commands, epoch, sid)
    p1 = 'phase1_build' if any(e['label'] == 'phase1_build' for e in d) else 'p1_root_build'
    spans={};native_forward={}
    for prefix,label,single,count in [('ds','p1_graph_replay','phase1_replay',k1),
                                      ('ps','p2_graph_replay','phase2_replay',k2)]:
        if any(e['label']==label for e in d):
            e=one(d,label);spans[prefix]=(e['wall_start_ns'],e['wall_end_ns'])
        else:
            events=sorted((e for e in d if e['label']==single),key=lambda e:e['wall_start_ns'])
            require(len(events)==count,f'Expected {count} native {single} forwards, found {len(events)}')
            spans[prefix]=(events[0]['wall_start_ns'],events[-1]['wall_end_ns'])
            native_forward[prefix]=union_ms(events)
    result = dict(run=run_id, epoch=epoch, step_id=sid, uid=row['uid'],
                state=row['state'], next_state=row['next_state'], k1=k1, k2=k2,
                ds_start_ns=one(d, p1)['wall_start_ns'],
                ds_end_ns=one(d, 'proxy_wait')['wall_start_ns'],
                proxy_send_ns=one(t, 'proxy_send_enqueue')['wall_end_ns'],
                ps_start_ns=one(d, 'proxy_wait')['wall_end_ns'],
                ps_end_ns=one(d, 'merge_cache')['wall_end_ns'],
                draft_ready_ns=tr['draft_receive_ready_ns'],
                step_start_ns=one(t, 'target_send_request')['wall_start_ns'],
                next_step_start_ns=tr['deadline_ns'],
                proxy_start_ns=one(t, 'exit_proxy_side')['wall_start_ns'],
                proxy_end_ns=one(t, 'exit_proxy_side')['wall_end_ns'],
                target_post_start_ns=one(t, 'graph_post')['wall_start_ns'],
                target_post_end_ns=one(t, 'graph_post')['wall_end_ns'],
                ds_graph_start_ns=spans['ds'][0],ds_graph_end_ns=spans['ds'][1],
                ps_graph_start_ns=spans['ps'][0],ps_graph_end_ns=spans['ps'][1])
    if native_forward:
        require(len(native_forward)==2,'Mixed chain/tree phases need separate timing definitions')
        for prefix,k in [('ds',k1),('ps',k2)]:
            result[prefix+'_forward_once_ms']=native_forward[prefix]/k
            margin=((result['proxy_send_ns']-result['ds_end_ns']) if prefix=='ds' else
                    (result['next_step_start_ns']-result['draft_ready_ns']))/1e6
            result[prefix+'_compute_window_ms']=native_forward[prefix]+margin
        result['forward_measurement']='Native chain replay: model + LM head + graph output stores; excludes between-forward preparation/sampling.'
    for field in ('accepted_len', 'emitted_len', 'context_tokens', 'verify_width'):
        result[field] = row.get(field)
    for field,label in [('target_pre_ms','graph_pre'), ('target_post_ms','graph_post'),
                        ('target_head_ms','final_logits'), ('target_wait_ms','target_spec_wait'),
                        ('target_setup_ms','verify_setup'), ('target_accept_ms','verify_sample_accept')]:
        events = [e for e in t if e['label'] == label]
        result[field] = union_ms(events) if events else None
    return result


def metrics(s):
    require(s['k1'] > 0 and s['k2'] > 0, 'Missing forward count')
    delta = lambda a, b: (s[a] - s[b]) / 1e6
    result = dict(step_ms=delta('next_step_start_ns', 'step_start_ns'),
                  ds_total_ms=delta('ds_end_ns', 'ds_start_ns'),
                  ds_margin_ms=delta('proxy_send_ns', 'ds_end_ns'),
                  ds_round_ms=delta('ds_graph_end_ns', 'ds_graph_start_ns') / s['k1'],
                  ps_total_ms=delta('ps_end_ns', 'ps_start_ns'),
                  ps_margin_ms=delta('next_step_start_ns', 'draft_ready_ns'),
                  ps_round_ms=delta('ps_graph_end_ns', 'ps_graph_start_ns') / s['k2'])
    require(all(result[k] >= 0 for k in FIELDS if 'margin' not in k), 'Negative duration')
    require(s['ds_start_ns'] <= s['ds_graph_start_ns'] <= s['ds_graph_end_ns'] <= s['ds_end_ns'], 'DS graph outside phase')
    require(s['ps_start_ns'] <= s['ps_graph_start_ns'] <= s['ps_graph_end_ns'] <= s['ps_end_ns'] <= s['draft_ready_ns'], 'PS graph outside phase')
    return result


def summarize(samples):
    require(bool(samples), 'Empty population')
    keys = [(s['run'], s['epoch'], s['step_id']) for s in samples]
    require(len(set(keys)) == len(keys), 'Duplicate step')
    require(all(s['state'] in STATES for s in samples), 'Unknown cache source')
    for s in samples:
        for field in ('accepted_len', 'emitted_len'):
            value=s.get(field)
            require(value is None or (type(value) is int and value >= 0), 'Invalid token count')
        require(s.get('accepted_len') is None or s.get('emitted_len') is None or
                s['emitted_len'] <= s['accepted_len'], 'Emitted tokens exceed accepted length')
    extras=FORWARD_FIELDS if any(k in s for s in samples for k in FORWARD_FIELDS) else ()
    require(all(all(k in s for k in extras) for s in samples), 'Incomplete forward probe population')
    values = [dict(state=s['state'], accepted_len=s.get('accepted_len'),
                   emitted_len=s.get('emitted_len'), **metrics(s), **{k:s[k] for k in extras}) for s in samples]
    rows = []
    for state in (1, 2, 0, 'all'):
        subset = [v for v in values if state == 'all' or v['state'] == state]
        n = len(subset)
        al_n=sum(v['accepted_len'] is not None for v in subset)
        emitted_n=sum(v['emitted_len'] is not None for v in subset)
        elapsed_ms=sum(v['step_ms'] for v in subset)
        rows.append(dict(state=state, label='Overall' if state == 'all' else STATES[state], n=n,
                         share_pct=100*n/len(values),
                         hit_rate_pct=100*sum(v['state'] in (1,2) for v in subset)/n if n else None,
                         al=sum(v['accepted_len'] for v in subset)/n if n and al_n==n else None,
                         tps=sum(v['emitted_len'] for v in subset)*1000/elapsed_ms
                             if n and emitted_n==n and elapsed_ms>0 else None,
                         al_samples=al_n,emitted_samples=emitted_n,
                         **{k: sum(v[k] for v in subset) / n if n else None for k in (*FIELDS,*extras)}))
    return rows


def table_cells(rows):
    fields=FORWARD_COLUMNS if rows and 'ds_forward_once_ms' in rows[0] else FIELDS
    return [[r['label'], str(r['n'])] +
            ['N/A' if r[k] is None else format(r[k], '+.3f' if 'margin' in k else '.3f')
             for k in fields] for r in rows]


def table_headers(rows):
    return FORWARD_HEADERS if rows and 'ds_forward_once_ms' in rows[0] else HEADERS


def performance_cells(rows):
    return [[r['label'], str(r['n'])] +
            ['N/A' if value is None else format(value, '.3f')
             for value in (r['hit_rate_pct'] if r['state']=='all' else r['share_pct'],
                           r['al'], r['tps'])] for r in rows]


def workload_summary(samples):
    rows=[]
    for state in (1,2,0,'all'):
        group=[s for s in samples if state=='all' or s['state']==state]
        n=len(group)
        def mean(key):
            vals=[s.get(key) for s in group]
            return sum(vals)/n if n and all(v is not None for v in vals) else None
        width=mean('verify_width')
        counts={}
        for s in group:
            v=s.get('verify_width')
            if v is not None:counts[str(v)]=counts.get(str(v),0)+1
        rows.append(dict(state=state,n=n,mean_context_tokens=mean('context_tokens'),
                         mean_logical_query_rows=width+1 if width is not None else None,
                         verify_width_counts=counts,
                         **{k:mean(k) for k in ('target_pre_ms','target_post_ms','target_head_ms',
                                               'target_wait_ms','target_setup_ms','target_accept_ms')}))
    return dict(rows=rows,note='Logical query rows = verify_width + 1, before graph-bucket padding. Context is the committed prefix before this step, including earlier generated tokens. These compare different measured conditions, not a controlled draft-model ablation.')


def overlap_summary(samples):
    spans = [max(0, min(s['proxy_end_ns'], s['target_post_end_ns']) -
                 max(s['proxy_start_ns'], s['target_post_start_ns'])) / 1e6 for s in samples]
    return dict(steps=len(spans), overlapping_steps=sum(v > 0 for v in spans),
                mean_overlap_ms=sum(spans)/len(spans),
                note='exit_proxy_side versus graph_post elapsed envelopes; send enqueue is not transfer completion.')


def read_samples(path):
    with gzip.open(path, 'rt') as f:
        data = json.load(f)
    require(data['schema'] == 'duet-phase-samples-v1', 'Unknown averages schema')
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--historical-source', type=Path, required=True)
    p.add_argument('--accepted-runs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    source = args.historical_source
    selected = load(source/'selected.json')
    selections = list(selected.items()) + [('code34b_throughput', selected['code34b']['throughput_alternative'])]
    ids = dict(l2_7b='llama2_7b_llama68m', l2_13b='llama2_13b_amd135m',
               l3_8b='llama3_8b_drafter340m', code34b='codellama34b_amd135m_code',
               code34b_throughput='codellama34b_amd135m_code')
    sets = []
    for key, selection in selections:
        arm = read_arm(source, key, selection)
        raw = [e for f in sorted((arm['folder']/'profile').glob('*draft*.json')) for e in load(f)]
        index = command_index(raw)
        c = arm['result']['config']
        records=[r for r in arm['result']['records'] if r['metrics']['phase_events']]
        for row in arm['rows']:
            ev=records[row['epoch']-1]['metrics']['phase_events'][row['step_id']-1]
            require(ev['step_id']==row['step_id'],'Outcome alignment mismatch')
            require(not ev['clipped'] and not ev['output_cap_reached'],'Historical eligible step reaches cap')
            row.update(emitted_len=ev['emitted_len'],context_tokens=row['context'],verify_width=ev['verify_width'])
        samples = [sample(arm['grouped'], index, r, c['k1'], c['k2'], key) for r in arm['rows']]
        for r, s in zip(arm['rows'], samples):
            m = metrics(s)
            for old, new in [('D1', 'ds_total_ms'), ('D2', 'ps_total_ms'), ('gap1_send', 'ds_margin_ms')]:
                require(abs(r[old] - m[new]) < 1e-6, 'Frozen analysis mismatch')
        sets.append(dict(key=key, model=ids[key], setting='K1/K2 16/6 (throughput alternative)' if key.endswith('throughput') else 'Selected setting',
                         k1=c['k1'], k2=c['k2'], samples=samples, source_hashes=arm['input_hashes'],
                         eligibility=arm['audit'], measurement='Historical v1.0 / RTX3090 / 4 profiled turns'))
        print(key, len(samples), flush=True)
    samples = []; hashes = {}; skipped = {}; runs = []
    for rep in (1, 2, 3):
        folder = args.accepted_runs/f'after_graph{rep}'
        run = load(folder/'run.json'); require(run['status'] == 'complete', 'Incomplete accepted run')
        groups, commands, profile_hashes = read_profiles(folder)
        hashes.update(profile_hashes); hashes[str(folder/'run.json')] = sha(folder/'run.json')
        index = command_index(commands)
        request_groups = run['warm_groups'] + run['groups']
        expected = {(ep, i+1) for ep, g in enumerate(request_groups) for i in range(len(g['steps']))}
        require(expected == set(groups['target']) == set(groups['draft']), 'Request alignment mismatch')
        count = 0
        for ep, g in enumerate(request_groups):
            require(g['n'] == 1, 'Only B1 traces supported')
            context=g['request_records'][0]['input_tokens']
            for i, step in enumerate(g['steps']):
                prior_context=context; context+=step['emitted']
                reason = ('warm' if g['phase'] == 'warm' else 'first_two' if i < 2 else
                          'terminal' if i == len(g['steps'])-1 else 'cap_clip' if step['cap'] or step['clipped'] else None)
                if reason:
                    skipped[reason] = skipped.get(reason, 0) + 1
                    continue
                r = dict(epoch=ep, step_id=i+1, uid=g['request_records'][0]['paper_uid'],
                         state=step['events'][0]['source'], next_state=g['steps'][i+1]['events'][0]['source'],
                         accepted_len=step['events'][0]['accepted_len'],emitted_len=step['emitted'],
                         context_tokens=prior_context,verify_width=step['events'][0]['verify_width'])
                samples.append(sample(groups, index, r, run['args']['k1'], run['args']['k2'], folder.name))
                count += 1
        runs.append(dict(run=folder.name, eligible_steps=count))
        print(folder.name, count, flush=True)
    sets.append(dict(key='accepted', model='llama2_7b_amd135m', setting='PS preparation validation (fixed parameters)',
                     k1=6, k2=3, samples=samples, source_hashes=hashes,
                     eligibility=dict(skipped=skipped, runs=runs),
                     measurement='Accepted optimization / RTX4090 / same 24 turns repeated 3 times; fixed parameters'))
    for item in sets:
        summarize(item['samples'])
        for path, digest in item['source_hashes'].items():
            require(sha(Path(path)) == digest, 'Source changed during extraction')
    data = dict(schema='duet-phase-samples-v1', definitions=DEFINITIONS, sets=sets,
                extractor_sha256=sha(Path(__file__)), new_inference_run=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip.compress((json.dumps(data, ensure_ascii=False)+'\n').encode(), mtime=0))


if __name__ == '__main__':
    main()
