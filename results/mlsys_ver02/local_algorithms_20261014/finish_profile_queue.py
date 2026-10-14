"""Run the diagnostic profile after the matched L2 jobs release GPUs 4/5."""
import json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
folder=Path('/data/chokwans99/ver02-local-algorithms-full-l2-final-20261014')
def main():
 start=time.time()
 while time.time()-start<3600:
  p=folder/'campaign.json'
  try:jobs=json.loads(p.read_text()) if p.exists() else []
  except json.JSONDecodeError:
   time.sleep(5);continue
  if any(j['status']!='complete' or j['returncode'] for j in jobs):
   raise RuntimeError('Inspect failed L2 attempt before profiling')
  if len(jobs)==7:break
  time.sleep(5)
 else:raise TimeoutError('L2 did not complete')
 uuids=set(subprocess.check_output(['nvidia-smi','-i','4,5','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines())
 quiet=0;start=time.time()
 while time.time()-start<3600:
  rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()
  quiet=0 if any(r.split(',')[0] in uuids for r in rows) else quiet+1
  if quiet>=3:break
  time.sleep(5)
 else:raise TimeoutError('GPU 4/5 busy')
 subprocess.run([sys.executable,'-B',str(ROOT/'ssd/bench/mlsys_tree_campaign.py'),
  '--plan',str(HERE/'PROFILE_PLAN.json'),'--directory','/data/chokwans99/ver02-local-algorithms-profiles-20261014',
  '--gpus','4,5','--port','29400','--timeout','2400'],cwd=ROOT,check=True)
 print('Profile queue complete',flush=True)

if __name__=='__main__':main()
