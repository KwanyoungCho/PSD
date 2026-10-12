import json,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent
while True:
    cp=R/'chain/campaign.json';rows=json.loads(cp.read_text()) if cp.exists() else []
    if any(x['exit_code'] or x['status']!='complete' for x in rows):raise SystemExit('Chain failure')
    if len(rows)>=2:break
    time.sleep(5)
subprocess.run([sys.executable,str(R.parent/'run_pair.py'),'--plan',str(R/'chain_paper_plan.json'),'--gpus','0,1','--port','33800','--directory',str(R/'chain')],cwd=R.parents[2],check=True)
