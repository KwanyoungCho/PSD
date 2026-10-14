import json,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/data/chokwans99/duet-tree-20261014')
out={}
for path in ROOT.rglob('*.telemetry.jsonl'):
 result=path.with_name(path.name.replace('.telemetry.jsonl','.json'))
 if not result.exists():continue
 try:j=json.loads(result.read_text())
 except Exception:continue
 if not isinstance(j,dict) or j.get('status')!='complete':continue
 steps=j.get('steps',[])
 if not steps:continue
 remote='remote' in path.relative_to(ROOT).parts
 offset=json.loads((HERE/('CLOCK_REMOTE.json' if remote else 'CLOCK_LOCAL.json')).read_text())['epoch_minus_monotonic_ns']
 start=(min(s['audit_start_ns'] for s in steps)+offset)/1e9
 end=(max(s['audit_end_ns'] for s in steps)+offset)/1e9
 stats={};external=set()
 for line in path.read_text().splitlines():
  r=json.loads(line);external.update(r.get('external_pids',[]))
  if not start<=r['time']<=end:continue
  for line in r.get('gpu','').splitlines():
   try:idx,mem,util,clock,power,temp=map(float,line.split(','))
   except Exception:continue
   if util<20:continue
   d=stats.setdefault(int(idx),{'sm_clock':[],'temp':[],'memory_mib':[]})
   d['sm_clock'].append(clock);d['temp'].append(temp);d['memory_mib'].append(mem)
 out[str(result.relative_to(ROOT))]={'approximate_decode_window':True,'external_pids':sorted(external),'gpus':{k:{'samples':len(d['sm_clock']),'mean_sm_clock':statistics.mean(d['sm_clock']),'max_temp':max(d['temp']),'max_memory_mib':max(d['memory_mib'])} for k,d in stats.items()}}
(HERE/'TELEMETRY.json').write_text(json.dumps(out,indent=2))
for name in ['candidate_full_l2/base.json','candidate_full_l2/candidate.json','candidate_full_l3/base.json','candidate_full_l3/candidate.json']:
 print(name,out.get(name))
