"""Wait for the four measured anchors, then execute their recorded neighborhood."""
import argparse,json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--batch',type=int,required=True)
p.add_argument('--gpus',required=True);p.add_argument('--port',required=True);a=p.parse_args()
prefix=f'{a.model}_b{a.batch}';manifest=HERE/f'{prefix}_duet_anchor/campaign.json'
while True:
    if manifest.exists():
        rows=json.loads(manifest.read_text())
        if any(r['status']!='complete' for r in rows):raise RuntimeError('Anchor failed')
        if len(rows)==4:break
    time.sleep(5)
subprocess.run([sys.executable,str(HERE/'make_refine.py'),'--model',a.model,'--batch',str(a.batch)],check=True)
subprocess.run([sys.executable,str(HERE.parent/'round4/after_campaign.py'),
    '--after',str(manifest),'--count','4','--plan',str(HERE/f'{prefix}_refine_plan.json'),
    '--directory',str(HERE/f'{prefix}_refine'),'--gpus',a.gpus,'--port',a.port],check=True)
