import json,subprocess,sys,time
from pathlib import Path
r=Path('/data2/chokwans99/target-gap-20261017');done=[]
assert json.loads((r/'queue_restore_7b.json').read_text())['status']=='complete'
for variant,rep in [('before','a'),('inputs','a'),('inputs','b'),('before','b')]:
 tag=f'{variant}_7b_chain_perf_{rep}';p=r/'runs'/tag
 if not p.exists():
  time.sleep(3);p.mkdir()
  with (p/'controller.log').open('w') as f:subprocess.run([sys.executable,str(r/'remote_run_chain.py'),'--variant',variant,'--model','7b','--tag',tag,'--passes','2'],stdout=f,stderr=subprocess.STDOUT,check=True)
 assert json.loads((p/'status.json').read_text())['status']=='complete'
 done.append(tag);(r/'queue_chain.json').write_text(json.dumps(dict(status='running',completed=done),indent=2)+'\n');print('DONE',tag,flush=True)
(r/'queue_chain.json').write_text(json.dumps(dict(status='complete',completed=done),indent=2)+'\n')
