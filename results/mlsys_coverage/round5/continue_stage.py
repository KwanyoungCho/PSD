"""Dependency-aware Round5 coordinator; abort failed jobs, never kill others."""
import argparse,json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--batch',type=int,required=True)
p.add_argument('--after',required=True);p.add_argument('--stage',choices=['combo','warm'],required=True)
p.add_argument('--gpus',required=True);p.add_argument('--port',required=True);a=p.parse_args()
prefix=f'{a.model}_b{a.batch}';manifest=HERE/f'{prefix}_{a.after}/campaign.json'
plan=HERE/f'{prefix}_{a.after}_plan.json'
while True:
    if manifest.exists() and plan.exists():
        try:rows=json.loads(manifest.read_text());expected=len(json.loads(plan.read_text()))
        except json.JSONDecodeError:time.sleep(2);continue
        if any(r['status']!='complete' for r in rows):raise RuntimeError('Prerequisite failed')
        if len(rows)==expected:break
    time.sleep(5)
subprocess.run([sys.executable,str(HERE/f'make_{a.stage}_plan.py'),'--model',a.model,'--batch',str(a.batch)],check=True)
subprocess.run([sys.executable,str(HERE.parent/'round4/after_campaign.py'),
    '--after',str(manifest),'--count',str(expected),'--plan',str(HERE/f'{prefix}_{a.stage}_plan.json'),
    '--directory',str(HERE/f'{prefix}_{a.stage}'),'--gpus',a.gpus,'--port',a.port],check=True)
