#!/usr/bin/env python3
"""Export fresh B=1 DS/PS profiles to a portable timeline bundle and render it.

Input paths are relative to the supplied spec. Re-rendering the exported bundle
does not require the original server, GPUs, model weights or inference runtime.
"""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from plot_model_breakdowns import load, require, sha, STATES
from plot_consecutive_steps import make_window, read_profiles
from plot_model_wait_timelines import command_index
from phase_averages import sample, summarize, metrics, DEFINITIONS
from build_model_timelines import main as build


def normalize_outcomes(data):
    """Keep warmup requests for exact anchor alignment; never infer cache source."""
    require(data.get('status') == 'complete', 'Outcomes must be complete')
    if data.get('schema') == 'duet-timeline-outcomes-v1':
        result = data
    else:
        require(all(k in data for k in ('args', 'warm_groups', 'groups')), 'Unsupported outcomes format; see README')
        require(data['args']['batch'] == 1, 'B>1 needs per-request source accounting; this exporter supports B=1')
        requests = []
        for epoch, group in enumerate(data['warm_groups'] + data['groups']):
            require(group['n'] == 1 and len(group['request_records']) == 1, 'Expected one request per group')
            steps = []
            for i, step in enumerate(group['steps'], 1):
                require(len(step['events']) == 1 and step['events'][0]['batch_size'] == 1, 'Expected B=1 event')
                ev = step['events'][0]
                require(ev['step_id'] == i, 'Noncontiguous outcome steps')
                steps.append(dict(step_id=i, source=ev['source'], cap=step['cap'], clipped=step['clipped']))
            record = group['request_records'][0]
            requests.append(dict(epoch=epoch, uid=record['paper_uid'], warmup=group['phase'] == 'warm', steps=steps))
        result = dict(schema='duet-timeline-outcomes-v1', status='complete', batch_size=1,
                      k1=data['args']['k1'], k2=data['args']['k2'], requests=requests)
    require(result['batch_size'] == 1, 'Only B=1 supported')
    require(all(type(result[k]) is int and result[k] > 0 for k in ('k1', 'k2')), 'Invalid forward counts')
    require(bool(result['requests']), 'Empty request list')
    for epoch, request in enumerate(result['requests']):
        require(request['epoch'] == epoch, 'Request epochs must match chronological profile anchors, starting at zero')
        require(type(request['warmup']) is bool and bool(request['uid']), 'Missing request identity/warmup flag')
        require(bool(request['steps']), 'Requests without decode anchors must not be assigned a profile epoch')
        for sid, step in enumerate(request['steps'], 1):
            require(step['step_id'] == sid and step['source'] in STATES, 'Invalid step/source')
            require(type(step['cap']) is bool and type(step['clipped']) is bool, 'Explicit cap/clipped flags required')
    return result


def export(spec_path, output):
    spec_path = spec_path.resolve(); output = output.resolve()
    spec = load(spec_path)
    require(spec['schema'] == 'duet-timeline-inputs-v1' and spec['datasets'], 'Invalid input spec')
    require(not output.exists() or not any(output.iterdir()), 'Use a new/empty output directory; use build_model_timelines.py to redraw an exported bundle')
    data_dir = output/'data'; data_dir.mkdir(parents=True, exist_ok=True)
    models = {}; sets = []; seen = set()
    for dataset_i, entry in enumerate(spec['datasets']):
        ident, name, setting = entry['model_id'], entry['model_name'], entry['setting']
        require(ident not in ('.', '..') and all(c.isascii() and (c.isalnum() or c in '_-') for c in ident) and bool(ident), 'Use a plain ASCII model_id')
        require((ident, setting) not in seen, 'Combine repeats of one setting in its runs list')
        seen.add((ident, setting))
        require(bool(entry['measurement']) and bool(entry['runs']), 'Measurement description and runs required')
        if ident not in models:
            models[ident] = dict(id=ident, name=name, note=entry['measurement'], panels=[])
        else:
            require(models[ident]['name'] == name, 'Model name mismatch')
            if entry['measurement'] not in models[ident]['note']:
                models[ident]['note'] += ' / ' + entry['measurement']
        samples = []; hashes = {}; skipped = Counter(); candidates = {s: [] for s in (1, 2, 0)}
        forward_counts = None; run_details = []
        for run_i, run_spec in enumerate(entry['runs']):
            outcomes_path = (spec_path.parent/run_spec['outcomes']).resolve()
            profile_dir = (spec_path.parent/run_spec['profile_dir']).resolve()
            original = load(outcomes_path); outcomes = normalize_outcomes(original)
            counts = (outcomes['k1'], outcomes['k2'])
            require(forward_counts in (None, counts), 'Different K1/K2 must use different settings')
            forward_counts = counts
            groups, commands, input_hashes = read_profiles(profile_dir)
            input_hashes[str(outcomes_path)] = sha(outcomes_path); hashes.update(input_hashes)
            expected = {(r['epoch'], s['step_id']) for r in outcomes['requests'] for s in r['steps']}
            require(expected == set(groups['target']) == set(groups['draft']), 'Outcomes do not cover exactly the recorded profile anchors')
            index = command_index(commands); run_id = f'dataset{dataset_i}_run{run_i}'
            run_details.append(dict(id=run_id, outcomes=str(outcomes_path), profile_dir=str(profile_dir),
                                    commit=original.get('paper_commit', original.get('commit')),
                                    args=original.get('args'), env=original.get('env')))
            for request in outcomes['requests']:
                steps = request['steps']; ep = request['epoch']
                for i, step in enumerate(steps):
                    reason = ('warmup' if request['warmup'] else 'first_two' if i < 2 else
                              'terminal' if i == len(steps)-1 else 'cap_clip' if step['cap'] or step['clipped'] else None)
                    if reason:
                        skipped[reason] += 1
                        continue
                    sid = step['step_id']; state = step['source']
                    row = dict(epoch=ep, step_id=sid, uid=request['uid'], state=state, next_state=steps[i+1]['source'])
                    record = sample(groups, index, row, *counts, run_id)
                    samples.append(record)
                    if (ep, sid+2) in groups['target'] and (ep, sid+2) in groups['draft']:
                        window = make_window(groups, commands, ep, sid,
                            title=f'{name} | {setting} | {STATES[state]} -> {STATES[row["next_state"]]} | UID {row["uid"]} | steps {sid}-{sid+1}',
                            metadata=dict(description=f'{entry["measurement"]} | {setting} | B=1, K1/K2 {counts[0]}/{counts[1]}',
                                          state=state, next_state=row['next_state'], uid=row['uid'],
                                          source_hashes=input_hashes, run=run_id,
                                          selection='median current target-send-to-next-send duration among eligible same-source steps with two following anchors'),
                            k1=counts[0], k2=counts[1])
                        candidates[state].append((metrics(record)['step_ms'], run_i, ep, sid, window))
            for path, digest in input_hashes.items():
                require(sha(Path(path)) == digest, 'Source changed during extraction; wait for profiling to finish')
        summarize(samples)
        selection_counts = {}
        for state, options in candidates.items():
            selection_counts[str(state)] = len(options)
            if not options:
                continue
            options.sort(key=lambda x: x[:4])
            window = options[(len(options)-1)//2][-1]
            dest = data_dir/f'dataset{dataset_i}_source{state}.json'
            dest.write_text(json.dumps(window, indent=2, ensure_ascii=False)+'\n')
            models[ident]['panels'].append(dict(window=str(dest.relative_to(output)), sha256=sha(dest), source=state, setting=setting))
        require(any(candidates.values()), 'Need at least one eligible step with two following anchors to draw consecutive steps')
        sets.append(dict(key=f'dataset{dataset_i}', model=ident, setting=setting, measurement=entry['measurement'],
                         k1=forward_counts[0], k2=forward_counts[1], samples=samples, source_hashes=hashes,
                         eligibility=dict(skipped=dict(skipped), eligible_steps=len(samples),
                                          representative_candidates=selection_counts), runs=run_details))
        print(ident, setting, len(samples), 'eligible steps', flush=True)
    portable = dict(schema='duet-phase-samples-v1', definitions=DEFINITIONS, sets=sets,
                    extractor_sha256=sha(Path(__file__)), new_inference_run=False,
                    input_spec_sha256=sha(spec_path))
    dest = data_dir/'PHASE_SAMPLES.json.gz'
    dest.write_bytes(gzip.compress((json.dumps(portable, ensure_ascii=False)+'\n').encode(), mtime=0))
    manifest = dict(schema='duet-model-timelines-v1', models=list(models.values()),
                    phase_samples=dict(path=str(dest.relative_to(output)), sha256=sha(dest)))
    path = output/'SOURCES.json'; path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    manifest = export(args.spec, args.output)
    build(['--manifest', str(manifest), '--output', str(args.output.resolve())])


if __name__ == '__main__':
    main()
