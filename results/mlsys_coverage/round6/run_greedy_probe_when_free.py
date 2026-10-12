import json,os,subprocess,sys,time
from pathlib import Path
here=Path(__file__).resolve().parent
manifest=here/'llama2_b8_full/campaign.json'
while True:
    records=json.loads(manifest.read_text()) if manifest.exists() else []
    if any(r['status']!='complete' for r in records):raise RuntimeError('B8 campaign failed; review before reuse')
    if len(records)==6:break
    time.sleep(10)
uuid=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
while True:
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if uuid not in active:break
    time.sleep(10)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',SSD_CUDA_ARCH='8.9',OMP_NUM_THREADS='4')
with (here/'greedy_host_micro.log').open('w') as log:
    run=subprocess.run([sys.executable,str(here.parents[2]/'ssd/bench/mlsys_review_greedy_host.py'),'--output',str(here/'greedy_host_micro.json')],env=env,stdout=log,stderr=subprocess.STDOUT)
raise SystemExit(run.returncode)
