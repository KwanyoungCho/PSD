"""Align GPU telemetry by the monotonic clock recorded during sampling."""
import collections
import json
from pathlib import Path

RAW = Path('/data/chokwans99/duet-ps-device-20261015')
OUT = Path(__file__).resolve().parent
samples = [json.loads(l) for l in (RAW/'telemetry.jsonl').read_text().splitlines()]
result = {}
for path in sorted((RAW/'runs').glob('*/run.json')):
    if 'fallback' in path.parent.name:
        continue
    run = json.loads(path.read_text())
    lo = min(s['audit_start_ns'] for s in run['steps'])
    hi = max(s['audit_end_ns'] for s in run['steps'])
    selected = [s for s in samples if lo <= s['monotonic_ns'] <= hi]
    gpus = collections.defaultdict(list)
    processes = set()
    for sample in selected:
        uuids = set()
        for line in sample['gpus'].splitlines():
            values = [x.strip() for x in line.split(',')]
            if values[0] in ('4','5'):
                gpus[values[0]].append(values)
                uuids.add(values[1])
        processes.update(line for line in sample['processes'].splitlines()
                         if line.split(',')[0].strip() in uuids)
    result[path.parent.name] = dict(samples=len(selected),
        gpu={index:dict(sm_clock_mean=sum(float(x[3]) for x in rows)/len(rows),
                        max_temp=max(float(x[4]) for x in rows))
             for index,rows in gpus.items()},
        observed_processes=sorted(processes))
(OUT/'TELEMETRY_SUMMARY.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
