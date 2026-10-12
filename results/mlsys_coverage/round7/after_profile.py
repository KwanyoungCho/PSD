"""Run frozen full-corpus plan only after every predecessor job completes."""
import argparse,json,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True)
p.add_argument('--plan',type=Path,required=True);p.add_argument('--gpus',required=True)
p.add_argument('--port',required=True);p.add_argument('--directory',required=True)
a=p.parse_args();required=len(json.loads((a.previous/'plan.json').read_text()))
while True:
 f=a.previous/'campaign.json'
 rows=json.loads(f.read_text()) if f.exists() else []
 if any(r['status']!='complete' or r['exit_code'] for r in rows):raise RuntimeError('Predecessor failure')
 if len(rows)==required:break
 time.sleep(5)
time.sleep(5)
subprocess.run([sys.executable,str(Path(__file__).with_name('run_pair.py')),'--plan',str(a.plan),
                '--gpus',a.gpus,'--port',a.port,'--directory',a.directory],check=True)
