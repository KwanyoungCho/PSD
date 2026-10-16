"""Run each model's matched campaign serially on its dedicated GPU set."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/data2/chokwans99/target-gap-20261017')
BASELINE='profile_v3'


def wait_status(tag):
    path=ROOT/'runs'/tag/'status.json'
    while not path.exists():time.sleep(2)
    row=json.loads(path.read_text())
    if row['status']!='complete':raise RuntimeError(f'{tag} failed: {row}')


def same_profile(model,tag):
    def read(name):return json.loads((ROOT/'runs'/name/'run.json').read_text())
    a=read(f'before_{model}_{BASELINE}');b=read(tag)
    def events(j):
        return [(e.get('source'),e.get('cache_hit'),e.get('valid_k'),e.get('accepted_len'))
                for s in j['steps'] for e in s['events']]
    result=dict(tag=tag,output_equal=a['outputs']==b['outputs'],events_equal=events(a)==events(b),
                output_requests=len(a['outputs']),events=len(events(a)))
    (ROOT/'runs'/tag/'PARITY.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['output_equal'] or not result['events_equal']:
        raise RuntimeError(f'Model parity failed: {result}')


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['7b','13b'],required=True)
    args=p.parse_args();model=args.model
    wait_status(f'before_{model}_{BASELINE}')
    schedule=[('snapshot','profile',1),('inputs','profile',1),
              ('before','perf_a',2),('snapshot','perf_a',2),('inputs','perf_a',2),
              ('inputs','perf_b',2),('snapshot','perf_b',2),('before','perf_b',2)]
    done=[]
    for variant,stage,passes in schedule:
        tag=f'{variant}_{model}_{stage}'
        path=ROOT/'runs'/tag
        if not path.exists():
            # Let the previous CUDA context disappear from nvidia-smi first.
            time.sleep(3)
            path.mkdir()
            command=[sys.executable,str(ROOT/'remote_run.py'),'--variant',variant,
                     '--model',model,'--tag',tag,'--passes',str(passes)]
            if stage=='profile':command.append('--profile')
            with (path/'controller.log').open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        wait_status(tag)
        if stage=='profile':same_profile(model,tag)
        done.append(tag)
        (ROOT/f'queue_{model}.json').write_text(json.dumps(dict(status='running',completed=done,schedule=schedule),indent=2)+'\n')
        print('DONE',tag,flush=True)
    (ROOT/f'queue_{model}.json').write_text(json.dumps(dict(status='complete',completed=done,schedule=schedule),indent=2)+'\n')


if __name__=='__main__':main()
