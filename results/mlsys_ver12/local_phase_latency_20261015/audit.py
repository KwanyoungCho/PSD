"""Check telemetry ownership, paired probe overhead and per-shape scaling."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
RAW=Path('/data/chokwans99/duet-local-phase-20261015')


def paired(left,right):
    lookup={(r['epoch'],r['step_id']):r for r in left['rows']}
    pairs=[(lookup[(r['epoch'],r['step_id'])],r) for r in right['rows'] if (r['epoch'],r['step_id']) in lookup]
    if not all(a['state']==b['state'] and a['accepted_len']==b['accepted_len'] for a,b in pairs):
        raise AssertionError('Matched run outcomes differ')
    delta=defaultdict(list)
    for a,b in pairs:delta[b['uid']].append(b['step_ms']-a['step_ms'])
    means=np.array([np.mean(v) for v in delta.values()])
    draws=np.random.default_rng(77).choice(means,(10000,len(means)),replace=True).mean(1)
    return dict(before=left['name'],after=right['name'],paired_steps=len(pairs),unique_uids=len(delta),
        before_mean_step_ms=float(np.mean([a['step_ms'] for a,b in pairs])),
        after_mean_step_ms=float(np.mean([b['step_ms'] for a,b in pairs])),
        mean_step_difference_ms=float(np.mean([b['step_ms']-a['step_ms'] for a,b in pairs])),
        equal_uid_mean_difference_ms=float(means.mean()),uid_bootstrap95_ms=np.quantile(draws,[.025,.975]).tolist(),
        outputs_equal=left['outputs']==right['outputs'],
        note='Independent process timings include run-to-run variation; interval is UID-clustered, not a per-kernel cost estimate.')


def main():
    with gzip.open(HERE/'SAMPLES.json.gz','rt') as f:runs=json.load(f)
    out=dict(telemetry={},probe_comparison={},repeat_comparison={},preparation={},by_shape={})
    by={r['name']:r for r in runs}
    for run in runs:
        folder=RAW/run['name'];cmd=json.loads((folder/'COMMAND.json').read_text())
        devices=set(map(int,cmd['env']['CUDA_VISIBLE_DEVICES'].split(',')))
        pids={str(int(p.stem.split('_')[-1])) for p in folder.glob('preparation_*.jsonl')}
        times=[];foreign=[];clocks=defaultdict(list);temps=defaultdict(list)
        for line in (folder/'telemetry.jsonl').read_text().splitlines():
            t=json.loads(line);times.append(t['wall_ns'])
            uuids=set()
            for g in t['gpu']:
                parts=[s.strip() for s in g.split(',')]
                if int(parts[0]) in devices:
                    uuids.add(parts[1]);clocks[parts[0]].append(float(parts[5]));temps[parts[0]].append(float(parts[6]))
            for j in t['jobs']:
                if not j:continue
                parts=[s.strip() for s in j.split(',')]
                if parts[0] in uuids and parts[1] not in pids:foreign.append(dict(time=t['wall_ns'],job=j))
        out['telemetry'][run['name']]=dict(devices=sorted(devices),samples=len(times),foreign=foreign,
            maximum_sampling_gap_s=max(np.diff(times))/1e9 if len(times)>1 else None,
            clocks_mean_mhz={k:float(np.mean(v)) for k,v in clocks.items()},
            temperature_max_c={k:max(v) for k,v in temps.items()})
        preparation=[json.loads(line) for f in folder.glob('preparation_*.jsonl') for line in f.read_text().splitlines()]
        out['preparation'][run['name']]=dict(events=len(preparation),
            excluded_measured_steps=run['eligibility']['skipped'].get('preparation',0),
            largest=[dict(r,duration_ms=(r['end_ns']-r['start_ns'])/1e6)
                     for r in sorted(preparation,key=lambda r:r['end_ns']-r['start_ns'],reverse=True)[:10]])
        groups=defaultdict(list)
        for row in run['rows']:
            for phase in ('ds','ps'):
                probe=row.get(phase+'_probe')
                if probe is None:continue
                for forward in probe['forward']:
                    key=(phase,forward['round'],forward['width'],probe['pages'])
                    groups[key].append(forward['forward_ms'])
        out['by_shape'][run['name']]=[dict(phase=k[0],round=k[1],width=k[2],prefix_pages=k[3],n=len(v),
            mean_ms=float(np.mean(v)),p50_ms=float(np.median(v)),p95_ms=float(np.quantile(v,.95))) for k,v in sorted(groups.items())]
    for model in ('7b','13b'):
        if model+'_base' not in by or model+'_control' not in by:continue
        out['probe_comparison'][model]=paired(by[model+'_control'],by[model+'_base'])
        if model+'_base_repeat' in by:
            out['repeat_comparison'][model]=paired(by[model+'_base'],by[model+'_base_repeat'])
    (HERE/'AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':main()
