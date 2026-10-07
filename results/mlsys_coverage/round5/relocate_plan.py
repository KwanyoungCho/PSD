"""Write a portable replay plan without mutating the original experiment plan.

Pass --target/--draft for a single-model plan. Historical opt-in defaults are
made explicit so a future default change does not silently change the arm.
"""
from pathlib import Path
import argparse,json
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('plan',type=Path);p.add_argument('--output',type=Path,required=True)
p.add_argument('--target');p.add_argument('--draft');p.add_argument('--prompts')
p.add_argument('--job',help='Select one existing job by its exact name')
p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3])
a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
old='/home/chokwans99/PSD-mlsys-coverage'
def relocate(s):return s.replace(old,str(a.root))
jobs=json.loads(a.plan.read_text())
if a.job:
 jobs=[j for j in jobs if j['name']==a.job]
 if len(jobs)!=1:raise ValueError('Expected one matching job: '+a.job)
for j in jobs:
 j['args']=[relocate(s) for s in j['args']]
 for key,value in [('--target',a.target),('--draft',a.draft),('--prompts',a.prompts)]:
  if value is not None:
   # Plans intentionally append final values after screening defaults.
   for i,token in enumerate(j['args'][:-1]):
    if token==key:j['args'][i+1]=value
 j['env']={k:relocate(v) for k,v in j.get('env',{}).items()}
 for key in ['SSD_TREE_FUSED_MATH','SSD_TREE_PARALLEL_INSERT','SSD_BATCH_TREE_BULK_EXPORT','SSD_TREE_LADDER_TRIM','SSD_BATCH_TREE_PROXY_STREAM']:
  j['env'].setdefault(key,'0')
 j['env'].setdefault('SSD_TREE_EXPANSION_POLICY','q_path')
 j['env'].setdefault('SSD_DUET_JIT_SHORT','1')
 if j['env']['SSD_TREE_EXPANSION_POLICY']!='q_path':
  for key in ['SSD_TREE_REACH_CALIBRATION','SSD_TREE_GAIN_CALIBRATION']:
   if key in j['env'] and not Path(j['env'][key]).exists():raise FileNotFoundError(j['env'][key])
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(jobs,indent=2)+'\n')
print('Wrote',a.output,'jobs',len(jobs))
