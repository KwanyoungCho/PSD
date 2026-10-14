"""Independent aggregation of extension A/B results and CUDA components.

Bootstrap units are fixed groups of eight prompts, never correlated tokens.
Profiles are separate instrumented runs and are not used as TPS evidence.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def load(path):
    return json.loads(Path(path).read_text())


def eligible(step, batch):
    return step['batch'] == batch and not step['cap'] and not step['clipped'] and not step.get('graph_preparation', False)


def vector(group, batch):
    steps = [s for s in group['steps'] if eligible(s, batch)]
    events = [e for s in steps for e in s['events']]
    return np.array([sum(s['emitted'] for s in steps), sum(s['seconds'] for s in steps),
                     sum(e['accepted_len'] for e in events), len(events),
                     sum(e['cache_hit'] for e in events)], dtype=float)


def metrics(data):
    assert not data.get('diagnostic_only'), 'Diagnostic readbacks invalidate TPS'
    batch = data['args']['batch']
    v = sum((vector(g, batch) for g in data['groups']), np.zeros(5))
    out = dict(data['summary'])
    assert abs(v[0] / v[1] - out['tps']) < 1e-6
    out['raw_decode_tps'] = out['actual_output_tokens'] / sum(s['seconds'] for s in data['steps'])
    out['retained_step_fraction'] = out['steps'] / out['all_steps']
    out['events'] = int(v[3])
    return out


def compare(a, b):
    assert a['prompt_ids'] == b['prompt_ids']
    assert a['source_hashes'] == b['source_hashes']
    batch = a['args']['batch']
    va = np.stack([vector(g, batch) for g in a['groups']])
    vb = np.stack([vector(g, batch) for g in b['groups']])
    assert len(va) == len(vb)
    idx = np.random.default_rng(7191).integers(0, len(va), (10000, len(va)))
    sa, sb = va[idx].sum(1), vb[idx].sum(1)
    tps = (sb[:, 0] / sb[:, 1]) / (sa[:, 0] / sa[:, 1]) - 1
    al = sb[:, 2] / sb[:, 3] - sa[:, 2] / sa[:, 3]
    hit = sb[:, 4] / sb[:, 3] - sa[:, 4] / sa[:, 3]
    out = dict(tps_percent=100*(b['summary']['tps']/a['summary']['tps']-1),
               tps_percent_ci=(np.quantile(tps, [.025, .975])*100).tolist(),
               al_delta=b['summary']['al']-a['summary']['al'],
               al_delta_ci=np.quantile(al, [.025, .975]).tolist(),
               hit_pp=100*(b['summary']['hit']-a['summary']['hit']),
               hit_pp_ci=(np.quantile(hit, [.025, .975])*100).tolist(),
               outputs_identical=a['outputs'] == b['outputs'],
               events_identical=[s['events'] for s in a['steps']] == [s['events'] for s in b['steps']],
               groups=len(va))
    return out


def campaign(directory):
    directory = Path(directory)
    manifest = {r['name']: r for r in load(directory/'campaign.json')}
    runs = {}
    for name in ('base', 'optimized', 'packed', 'mixed', 'stream'):
        path = directory/(name+'.json')
        if path.exists():
            data = load(path)
            if data['status'] == 'complete':
                runs[name] = data
    out = dict(directory=str(directory), cells={}, comparisons={})
    for name, data in runs.items():
        meta = manifest.get(name, {})
        out['cells'][name] = dict(metrics=metrics(data), external_pids=meta.get('external_pids', []),
                                 args=data['args'], kwargs=data['kwargs'], env=data['env'],
                                 sha256=hashlib.sha256((directory/(name+'.json')).read_bytes()).hexdigest())
    for left, right in [('base','optimized'),('optimized','packed'),('packed','mixed'),('optimized','stream')]:
        if left in runs and right in runs:
            out['comparisons'][left+'->'+right] = compare(runs[left], runs[right])
    return out


def sensitivity(directory):
    directory=Path(directory)
    out={}
    for k in (1,2,3):
        a_path,b_path=(directory/f'k2_{k}_wait.json'),(directory/f'k2_{k}_ar.json')
        if not a_path.exists() or not b_path.exists():continue
        a,b=load(a_path),load(b_path)
        if a['status']!='complete' or b['status']!='complete':continue
        out[str(k)]=dict(wait=metrics(a),ar=metrics(b),comparison=compare(a,b),kwargs=a['kwargs'])
    return out


def profile(directory, name):
    directory = Path(directory)
    data = load(directory/(name+'.json'))
    if data['status'] != 'complete':
        return {'status': data['status']}
    warm = len(data['warm_groups'])
    allowed = set()
    for i, group in enumerate(data['groups']):
        for s in group['steps']:
            if eligible(s, data['args']['batch']):
                allowed.add((warm+i, s['events'][0]['step_id']))
    rows = []
    counts = {}
    for path in (directory/name).glob('duet_profile_*.json'):
        proc = 'draft' if 'draft_' in path.name else 'target'
        first = 'batch_kv_restore' if proc == 'draft' else 'target_send_request'
        group = -1
        for row in sorted(load(path)[1:], key=lambda x:x['idx']):
            if row['label'] == first and row['step_id'] == 1:
                group += 1
            key = group, row['step_id']
            if key in allowed:
                rows.append((proc, key, row))
        counts[proc] = group+1
    assert all(n == warm+len(data['groups']) for n in counts.values()), counts
    captured = {key for _, key, row in rows if 'capture' in row['label']}
    samples = defaultdict(list)
    step_components = defaultdict(dict)
    for proc, key, row in rows:
        if key not in captured:
            samples[proc+'/'+row['label']].append(row['cuda_ms'])
            step_components[f'{key[0]-warm}:{key[1]}'][proc+'/'+row['label']]=row['cuda_ms']
    return dict(status='complete', groups=counts, allowed_steps=len(allowed),
                excluded_capture_steps=len(captured),
                step_components=dict(step_components),
                components={k:dict(mean_ms=float(np.mean(v)), median_ms=float(np.median(v)),
                                   count=len(v)) for k,v in samples.items()})


def profile_compare(directory, left, right):
    directory=Path(directory)
    a,b=load(directory/(left+'.json')),load(directory/(right+'.json'))
    assert a['outputs']==b['outputs'], 'Component comparison needs matching outputs'
    assert [s['events'] for s in a['steps']]==[s['events'] for s in b['steps']]
    pa,pb=profile(directory,left),profile(directory,right)
    common=pa['step_components'].keys() & pb['step_components'].keys()
    result={}
    for label in pa['components'].keys() & pb['components'].keys():
        keys=[k for k in common if label in pa['step_components'][k] and label in pb['step_components'][k]]
        va=np.array([pa['step_components'][k][label] for k in keys])
        vb=np.array([pb['step_components'][k][label] for k in keys])
        result[label]=dict(steps=len(keys),before_ms=float(va.mean()),after_ms=float(vb.mean()),
                           delta_ms=float((vb-va).mean()),percent=float((vb.mean()/va.mean()-1)*100))
    return result


def mixed_expected(directory):
    directory=Path(directory)
    out={}
    for model in ('llama2','llama3'):
        expected=load(directory/'expected'/(model+'_expected.json'))
        ordinary=load(directory/'profiles_v2'/(model+'_optimized.json'))
        strip=lambda s:[{k:v for k,v in e.items() if not k.startswith('conditional_')} for e in s['events']]
        assert expected['outputs']==ordinary['outputs']
        assert list(map(strip,expected['steps']))==list(map(strip,ordinary['steps']))
        pr=profile(directory/'profiles_v2',model+'_optimized')
        components=pr['step_components']
        zero=[]
        for g,group in enumerate(ordinary['groups']):
            for step in group['steps']:
                if not step['events']:continue
                key=f"{g}:{step['events'][0]['step_id']}"
                if key in components and all(e['cache_hit'] for e in step['events']):
                    zero.append(components[key]['draft/batch_miss_draft'])
        ar_metadata=float(np.median(zero)) if zero else 0.
        cells=defaultdict(list)
        for g,(eg,og) in enumerate(zip(expected['groups'],ordinary['groups'])):
            for es,os in zip(eg['steps'],og['steps']):
                if not eligible(os,ordinary['args']['batch']):continue
                key=f"{g}:{es['events'][0]['step_id']}"
                if key not in components:continue
                events=es['events']; misses=sum(not e['cache_hit'] for e in events)
                if not 0<misses<len(events):continue
                aw=sum(e['conditional_expected_len'] for e in events)
                aa=sum(e['conditional_expected_len_if_miss_ar'] for e in events)
                # Padded target/acceptance canvas is held fixed. Approximation:
                # only exposed JIT cost is removed; future state is not modeled.
                saved=max(0,components[key]['draft/batch_miss_draft']-ar_metadata)
                required=os['seconds']*1000*(aw-aa)/aw
                cells[misses].append(dict(expected_wait=aw,expected_ar=aa,loss=aw-aa,
                    estimated_saved_ms=saved,break_even_saved_ms=required,
                    predicted_ar_win=saved>required))
        out[model]=dict(scope='fixed proposals; ordinary profiled step time; modeled one-step criterion, not measured AR TPS',
            matching_outputs=True,matching_events=True,ar_metadata_ms=ar_metadata,zero_miss_steps=len(zero),
            miss_counts={str(m):dict(steps=len(v),
                expected_token_loss=float(np.mean([x['loss'] for x in v])),
                estimated_saved_ms=float(np.median([x['estimated_saved_ms'] for x in v])),
                break_even_saved_ms=float(np.median([x['break_even_saved_ms'] for x in v])),
                predicted_win_fraction=float(np.mean([x['predicted_ar_win'] for x in v]))) for m,v in cells.items()})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--campaign', action='append', default=[])
    p.add_argument('--profiles', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = {'campaigns': {Path(d).name:campaign(d) for d in args.campaign}}
    if args.profiles:
        out['profiles'] = {}
        for path in args.profiles.glob('llama*.json'):
            if path.with_suffix('').is_dir():
                out['profiles'][path.stem] = profile(args.profiles, path.stem)
    args.output.write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
