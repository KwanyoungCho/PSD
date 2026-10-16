"""One-pass combined-code integration check; not a repeated A/B claim."""
import argparse,json,subprocess,sys,time
from pathlib import Path
r=Path('/data2/chokwans99/target-gap-20261017');p=argparse.ArgumentParser();p.add_argument('--model',required=True);m=p.parse_args().model
key='queue_chain.json' if m=='7b' else 'queue_restore_13b.json'
while not (r/key).exists() or json.loads((r/key).read_text())['status']!='complete':time.sleep(3)
time.sleep(3);tag=f'combined_{m}_integration';folder=r/'runs'/tag
if not folder.exists():
 folder.mkdir()
 with (folder/'controller.log').open('w') as f:subprocess.run([sys.executable,str(r/'remote_run.py'),'--variant','combined','--model',m,'--tag',tag,'--passes','1'],stdout=f,stderr=subprocess.STDOUT,check=True)
assert json.loads((folder/'status.json').read_text())['status']=='complete'
a=json.loads((r/'runs'/f'before_{m}_profile_warm'/'run.json').read_text());b=json.loads((folder/'run.json').read_text())
def events(j):return [(e.get('source'),e.get('cache_hit'),e.get('valid_k'),e.get('accepted_len')) for s in j['steps'] for e in s['events']]
result=dict(output_equal=a['outputs']==b['outputs'],events_equal=events(a)==events(b),outputs=len(b['outputs']),events=len(events(b)),measurement='Single profile-OFF process, integration/parity only; no repeated combined-code performance claim',summary=b['summary'])
(folder/'PARITY.json').write_text(json.dumps(result,indent=2)+'\n')
assert result['output_equal'] and result['events_equal'],result
(r/f'queue_combined_{m}.json').write_text(json.dumps(dict(status='complete',completed=[tag]),indent=2)+'\n');print(result,flush=True)
