"""Warm profile attribution and fresh ABBA validation after the ablation queue."""
import argparse,json,subprocess,sys,time
from pathlib import Path
R=Path('/data2/chokwans99/target-gap-20261017')
p=argparse.ArgumentParser();p.add_argument('--model',required=True);m=p.parse_args().model
while True:
 f=R/f'queue_{m}.json'
 if f.exists() and json.loads(f.read_text())['status']=='complete':break
 time.sleep(3)
plan=[('before','profile_warm',1),('restore','profile_warm',1),
      ('before','final_a',2),('restore','final_a',2),('restore','final_b',2),('before','final_b',2)]
done=[]
for variant,stage,passes in plan:
 tag=f'{variant}_{m}_{stage}';folder=R/'runs'/tag
 if not folder.exists():
  time.sleep(3);folder.mkdir()
  command=[sys.executable,str(R/'remote_run.py'),'--variant',variant,'--model',m,'--tag',tag,'--passes',str(passes)]
  if stage.startswith('profile'):command.append('--profile')
  with (folder/'controller.log').open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
 status=json.loads((folder/'status.json').read_text())
 if status['status']!='complete':raise RuntimeError(status)
 if variant=='restore':
  ref='before_'+m+'_'+stage if stage!='final_b' else 'before_'+m+'_final_a'
  a=json.loads((R/'runs'/ref/'run.json').read_text());b=json.loads((folder/'run.json').read_text())
  def events(j):return [(e.get('source'),e.get('cache_hit'),e.get('valid_k'),e.get('accepted_len')) for s in j['steps'] for e in s['events']]
  parity=dict(output_equal=a['outputs']==b['outputs'],events_equal=events(a)==events(b),outputs=len(b['outputs']),events=len(events(b)))
  (folder/'PARITY.json').write_text(json.dumps(parity,indent=2)+'\n')
  if not all(parity[k] for k in ('output_equal','events_equal')):raise RuntimeError(parity)
 done.append(tag);(R/f'queue_restore_{m}.json').write_text(json.dumps(dict(status='running',completed=done,schedule=plan),indent=2)+'\n')
 print('DONE',tag,flush=True)
(R/f'queue_restore_{m}.json').write_text(json.dumps(dict(status='complete',completed=done,schedule=plan),indent=2)+'\n')
