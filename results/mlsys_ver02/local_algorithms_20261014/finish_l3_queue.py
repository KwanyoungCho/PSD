"""Wait for our active L3 campaign, then repair contaminated arms and check greedy."""
import json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
raw=Path('/data/chokwans99/ver02-local-algorithms-full-l3-20261014')
def campaign(plan,directory,port):
 uuids=set(subprocess.check_output(['nvidia-smi','-i','6,7','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines())
 quiet=0;deadline=time.time()+3600
 while time.time()<deadline:
  rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()
  quiet=0 if any(r.split(',')[0] in uuids for r in rows) else quiet+1
  if quiet>=25:break
  time.sleep(5)
 else:raise TimeoutError('No stable GPU 6/7 availability')
 print('GPU 6/7 quiet for 120s; starting',plan.name,flush=True)
 cmd=[sys.executable,'-B',str(ROOT/'ssd/bench/mlsys_tree_campaign.py'),
      '--plan',str(plan),'--directory',directory,'--gpus','6,7','--port',str(port),'--timeout','2400']
 subprocess.run(cmd,cwd=ROOT,check=True)

def main():
 deadline=time.time()+3600
 while time.time()<deadline:
  p=raw/'campaign.json';jobs=json.loads(p.read_text()) if p.exists() else []
  bad=[j for j in jobs if j['status']!='complete' or j['returncode']]
  if any(not j['external_pids'] for j in bad):
   raise RuntimeError('Unexplained failure; inspect before scheduling more GPU work')
  if len(jobs)==7:break
  time.sleep(5)
 else:raise TimeoutError('Main campaign did not finish')
 names={j['name'] for j in jobs if j['external_pids'] or j['status']!='complete'}
 plan=json.loads((HERE/'l3_FULL_PLAN.json').read_text())
 retry=HERE/'L3_RECHECK_PLAN.json';retry.write_text(json.dumps([j for j in plan if j['name'] in names],indent=2)+'\n')
 time.sleep(5)
 if names:campaign(retry,'/data/chokwans99/ver02-local-algorithms-l3-recheck3-20261014',29200)
 campaign(HERE/'FINAL_GREEDY_PLAN.json','/data/chokwans99/ver02-local-algorithms-final-greedy-20261014',29300)
 (HERE/'FOLLOWUP_QUEUE_COMPLETE.json').write_text(json.dumps(dict(time=time.time(),rechecked=sorted(names),status='complete'),indent=2)+'\n')
 print('Follow-up queue complete',flush=True)

if __name__=='__main__':main()
