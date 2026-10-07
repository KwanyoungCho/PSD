"""Depend on phase AND independently tuned SSD diagnostics before tree study."""
import argparse,json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--batch',type=int,required=True)
p.add_argument('--gpus',required=True);p.add_argument('--port',required=True);a=p.parse_args()
prefix=f'{a.model}_b{a.batch}'
manifest=HERE/f'{prefix}_ssd_refine/campaign.json'
count=len(json.loads((HERE/f'{prefix}_ssd_refine_plan.json').read_text()))
while True:
    if manifest.exists():
        try:rows=json.loads(manifest.read_text())
        except json.JSONDecodeError:time.sleep(2);continue
        if any(r['status']!='complete' for r in rows):raise RuntimeError('SSD refinement failed')
        if len(rows)==count:break
    time.sleep(5)
subprocess.run([sys.executable,str(HERE/'make_tree_plan.py'),'--model',a.model,'--batch',str(a.batch)],check=True)
subprocess.run([sys.executable,str(HERE.parent/'round4/after_campaign.py'),
    '--after',str(manifest),'--count',str(count),'--plan',str(HERE/f'{prefix}_tree_plan.json'),
    '--directory',str(HERE/f'{prefix}_tree'),'--gpus',a.gpus,'--port',a.port],check=True)
