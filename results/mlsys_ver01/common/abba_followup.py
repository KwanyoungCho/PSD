import json,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent
while True:
    cp=R/'tree_r0/campaign.json';rows=json.loads(cp.read_text()) if cp.exists() else []
    if any(x['exit_code'] or x['status']!='complete' for x in rows):raise SystemExit('Primary tree failure')
    if len(rows)==3:break
    time.sleep(5)
subprocess.run([sys.executable,str(R.parent/'run_pair.py'),'--plan',str(R/'abba_plan.json'),'--gpus','2,3','--port','34000','--directory',str(R/'abba')],cwd=R.parents[2],check=True)
