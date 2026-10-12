import argparse,json,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--gpus',required=True);p.add_argument('--port',required=True);a=p.parse_args()
here=Path(__file__).resolve().parent
manifest=here/f'{a.model}_b8_full/campaign.json'
while True:
    records=json.loads(manifest.read_text()) if manifest.exists() else []
    if any(r['status']!='complete' for r in records):raise RuntimeError('B8 campaign failed')
    if len(records)==6:break
    time.sleep(10)
if a.model=='llama2':
    while True:
        p=here/'greedy_host_micro.json'
        if p.exists() and json.loads(p.read_text())['status']!='running':break
        time.sleep(10)
uuids=subprocess.check_output(['nvidia-smi','-i',a.gpus,'--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
while True:
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if not any(u in active for u in uuids):break
    time.sleep(10)
with (here/f'{a.model}_chain_profile_campaign.log').open('w') as log:
    r=subprocess.run([sys.executable,str(here.parents[2]/'ssd/bench/mlsys_campaign.py'),'--plan',str(here/f'{a.model}_chain_profile_plan.json'),'--directory',str(here/f'{a.model}_chain_profile'),'--gpus',a.gpus,'--port',a.port,'--timeout','1200'],stdout=log,stderr=subprocess.STDOUT)
raise SystemExit(r.returncode)
