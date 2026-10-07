"""Freeze and run final data independently for each (model,B) selection cell."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--batch',type=int,required=True)
p.add_argument('--replicate',type=int,choices=[0,1],required=True);p.add_argument('--gpus',required=True);p.add_argument('--port',required=True)
a=p.parse_args();prefix=f'{a.model}_b{a.batch}'
if a.replicate==0:
    manifest=HERE/f'{prefix}_warm/campaign.json';plan=HERE/f'{prefix}_warm_plan.json'
else:
    other=8 if a.batch==1 else 1
    manifest=HERE/f'final_{a.model}_b{other}_r0/campaign.json'
    plan=HERE/f'{a.model}_b{other}_final_r0_plan.json'
future=HERE/f'{prefix}_final_r{a.replicate}_plan.json'
while True:
    if manifest.exists() and plan.exists():
        try:rows=json.loads(manifest.read_text());expected=len(json.loads(plan.read_text()))
        except json.JSONDecodeError:time.sleep(2);continue
        if any(r['status']!='complete' for r in rows):raise RuntimeError('Prerequisite failed')
        if len(rows)==expected and (a.replicate==0 or future.exists()):break
    time.sleep(5)
if a.replicate==0:
    subprocess.run([sys.executable,str(HERE/'freeze_final.py'),'--model',a.model,'--batch',str(a.batch)],check=True)
# The kernel prototype runs on an idle owned pair, ahead of final generation.
# It does not modify engine files; its result never selects model parameters.
if a.model=='llama3' and a.batch==8 and a.replicate==0:
    ids=set(subprocess.check_output(['nvidia-smi','-i',a.gpus,'--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines())
    for _ in range(12):
        active=set(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines())
        if not ids.intersection(active):break
        time.sleep(5)
    else:raise RuntimeError('Pair occupied before isolated kernel benchmark')
    with (HERE/'proxy_tail_gpu.log').open('w') as log:
        subprocess.run([sys.executable,str(HERE/'bench_proxy_tail.py'),'--output',str(HERE/'PROXY_TAIL_GPU.json')],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus,OMP_NUM_THREADS='4',SSD_CUDA_ARCH='8.9'),
            stdout=log,stderr=subprocess.STDOUT,check=True)
subprocess.run([sys.executable,str(HERE.parent/'round4/after_campaign.py'),
    '--after',str(manifest),'--count',str(expected),'--plan',str(future),
    '--directory',str(HERE/f'final_{prefix}_r{a.replicate}'),'--gpus',a.gpus,'--port',a.port],check=True)
