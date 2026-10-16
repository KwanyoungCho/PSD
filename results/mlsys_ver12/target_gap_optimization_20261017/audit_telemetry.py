"""Audit assigned GPU residency/clocks during the measured-request interval."""
import argparse,json
from collections import defaultdict
from pathlib import Path
from statistics import mean
p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
offset=json.loads((args.raw/'CLOCK_OFFSET.json').read_text())['offset_ns'];rows=[]
for folder in sorted((args.raw/'runs').iterdir()):
 if not (folder/'status.json').exists():continue
 status=json.loads((folder/'status.json').read_text())
 if status['status']!='complete':continue
 run=json.loads((folder/'run.json').read_text());prov=json.loads((folder/'PROVENANCE.json').read_text())
 steps=run['steps'];start=min(s['audit_start_ns'] for s in steps)+offset;end=max(s['audit_end_ns'] for s in steps)+offset
 gpu=defaultdict(list);seen=defaultdict(set);maxjobs=defaultdict(int)
 uuid_index={line.split(',')[1].strip():int(line.split(',')[0]) for line in prov['initial_gpu_state']['gpu']}
 for line in (folder/'telemetry.jsonl').read_text().splitlines():
  sample=json.loads(line)
  if not start<=sample['time_ns']<=end:continue
  jobs=defaultdict(set)
  for row in sample['processes']:
   values=[x.strip() for x in row.split(',')]
   if values and values[0] in uuid_index:jobs[uuid_index[values[0]]].add(int(values[1]))
  for row in sample['gpu']:
   v=[x.strip() for x in row.split(',')];idx=int(v[0])
   if idx not in prov['gpus']:continue
   gpu[idx].append(dict(util=int(v[4].split()[0]),clock=int(v[5].split()[0]),temp=int(v[6]),power=float(v[7].split()[0])))
   seen[idx].update(jobs[idx]);maxjobs[idx]=max(maxjobs[idx],len(jobs[idx]))
 rows.append(dict(tag=folder.name,profile=status['profile'],source_unchanged=status['source_unchanged'],
  assigned_gpus=prov['gpus'],all_assigned_gpus_initially_free=not prov['initial_gpu_state']['processes'] or not any(uuid_index.get(x.split(',')[0].strip()) in prov['gpus'] for x in prov['initial_gpu_state']['processes']),
  sample_scope='Measured-request wall envelope (includes gaps/prefill between those requests); wall offset sampled once, not a GPU-kernel-only clock measurement',
  devices={str(i):dict(samples=len(x),max_simultaneous_processes=maxjobs[i],pids=sorted(seen[i]),clock_mean_mhz=mean(v['clock'] for v in x),temperature_max_c=max(v['temp'] for v in x),utilization_mean_pct=mean(v['util'] for v in x),power_mean_w=mean(v['power'] for v in x)) for i,x in gpu.items()}))
args.output.write_text(json.dumps(dict(runs=rows,clock_offset_ns=offset,note='One serving PID per assigned GPU expected. This checks unexpected concurrent residency, not isolation of all host or hardware noise.'),indent=2)+'\n')
print('Audited',len(rows),'completed runs')
