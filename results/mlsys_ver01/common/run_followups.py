"""Reuse GPU6/7 only after their functional campaign releases the pair."""
import json,os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent;root=R.parents[2]
while True:
    cp=R/'features/campaign.json'
    rows=json.loads(cp.read_text()) if cp.exists() else []
    if any(x['status']!='complete' or x['exit_code'] for x in rows):raise SystemExit('Feature failure: inspect before continuing')
    if len(rows)==len(json.loads((R/'features_plan.json').read_text())):break
    time.sleep(5)
for name,port in [('prefix',33500),('bulk',33600),('profiles',33700)]:
    print('FOLLOWUP',name,flush=True)
    subprocess.run([sys.executable,str(R.parent/'run_pair.py'),'--plan',str(R/(name+'_plan.json')),'--gpus','6,7','--port',str(port),'--directory',str(R/name)],cwd=root,check=True)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='7',OMP_NUM_THREADS='4')
for script in ['check_suites.py','micro_checks.py']:
    print('FOLLOWUP',script,flush=True)
    with (R/(script+'.final.log')).open('w') as log:
        subprocess.run([sys.executable,str(R/script)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
print('FOLLOWUPS COMPLETE',flush=True)
