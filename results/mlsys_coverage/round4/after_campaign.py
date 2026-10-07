"""Start a dependent campaign after a known campaign succeeds and GPUs idle."""
import argparse,json,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--after',type=Path,required=True);p.add_argument('--count',type=int,required=True)
p.add_argument('--plan',required=True);p.add_argument('--directory',required=True);p.add_argument('--gpus',required=True);p.add_argument('--port',required=True);a=p.parse_args()
while True:
 if a.after.exists():
  try:rows=json.loads(a.after.read_text())
  except json.JSONDecodeError:time.sleep(2);continue
  if any(r['status']!='complete' for r in rows):raise RuntimeError('Prerequisite campaign failed')
  if len(rows)>=a.count:break
 time.sleep(5)
ids=subprocess.check_output(['nvidia-smi','-i',a.gpus,'--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
for attempt in range(12):
 active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
 if not set(ids).intersection(active):break
 time.sleep(5)
else:raise RuntimeError('GPUs still occupied after prerequisite; no process killed')
cmd=[sys.executable,'ssd/bench/mlsys_campaign.py','--plan',a.plan,'--directory',a.directory,'--gpus',a.gpus,'--port',a.port]
print('START_DEPENDENT',cmd,flush=True)
subprocess.run(cmd,check=True)
