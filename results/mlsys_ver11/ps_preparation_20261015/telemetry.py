import collections,json,time
from pathlib import Path
raw=Path('/data/chokwans99/duet-ps-prep-20261015');out=Path(__file__).resolve().parent
offset=time.time_ns()-time.monotonic_ns();samples=[json.loads(l) for l in (raw/'telemetry.jsonl').read_text().splitlines()]
result={'clock_alignment':'Approximate wall-clock telemetry to monotonic offset sampled after runs. Native plots use unchanged event timestamps.','epoch_minus_monotonic_ns':offset,'runs':{}}
for p in sorted(raw.glob('runs/*/run.json')):
 if '_graph' not in p.parent.name and 'unprofiled' not in p.parent.name:continue
 r=json.loads(p.read_text());steps=r['steps'];lo=min(s['audit_start_ns'] for s in steps);hi=max(s['audit_end_ns'] for s in steps)
 ss=[s for s in samples if lo <= s['time']*1e9-offset <= hi];g=collections.defaultdict(list);processes=set()
 for s in ss:
  for line in s['gpus'].splitlines():
   a=[x.strip() for x in line.split(',')]
   if a[0] in ('4','5'):g[a[0]].append(a)
  processes.update(s['processes'].splitlines())
 result['runs'][p.parent.name]={'samples':len(ss),'gpu':{k:{'sm_clock_mean':sum(float(x[3]) for x in v)/len(v),'max_temp':max(float(x[4]) for x in v)} for k,v in g.items()},'observed_process_rows':sorted(processes)}
(out/'TELEMETRY_SUMMARY.json').write_text(json.dumps(result,indent=2))
