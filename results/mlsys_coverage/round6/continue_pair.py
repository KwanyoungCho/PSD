"""Wait for this round's profile campaign, then prefix diagnosis and full B8."""
import argparse,json,subprocess,sys,time,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model',choices=['llama2','llama3'],required=True);p.add_argument('--gpus',required=True);p.add_argument('--port',type=int,required=True);a=p.parse_args()
here=Path(__file__).resolve().parent;repo=here.parents[2]
folder=here/('llama2_profile' if a.model=='llama2' else 'llama3_profile_retry1')
while True:
    manifest=json.loads((folder/'campaign.json').read_text()) if (folder/'campaign.json').exists() else []
    if any(x['status']!='complete' for x in manifest):raise RuntimeError('Profile campaign failed')
    if len(manifest)==13:break
    time.sleep(10)
target='/home/chokwans99/models_mlsys/layerskip-llama2-7b' if a.model=='llama2' else '/data/chokwans99/models/layerskip-llama3-8B'
env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpus.split(',')[0],SSD_CUDA_ARCH='8.9',SSD_DIST_PORT=str(a.port),SSD_ATTN_BACKEND='sgl',OMP_NUM_THREADS='4')
with (here/f'{a.model}_prefix.log').open('w') as out:
    result=subprocess.run([sys.executable,str(repo/'ssd/bench/mlsys_review_prefix.py'),'--model',target,'--prompts',str(here.parent/'questions.json'),'--output',str(here/f'{a.model}_prefix.json')],env=env,cwd=repo,stdout=out,stderr=subprocess.STDOUT)
print('PREFIX',a.model,result.returncode,flush=True)
# A diagnostic failure is retained and does not invalidate unrelated full tests.
with (here/f'{a.model}_b8_full_campaign.log').open('w') as out:
    result=subprocess.run([sys.executable,str(repo/'ssd/bench/mlsys_campaign.py'),'--plan',str(here/f'{a.model}_b8_full_plan.json'),'--directory',str(here/f'{a.model}_b8_full'),'--gpus',a.gpus,'--port',str(a.port+20),'--timeout','2400'],cwd=repo,stdout=out,stderr=subprocess.STDOUT)
print('FULL_B8',a.model,result.returncode,flush=True)
raise SystemExit(result.returncode)
