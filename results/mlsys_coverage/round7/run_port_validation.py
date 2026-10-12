"""After Llama3 corpus controls free GPUs0/1, validate narrow physical patches."""
import json,os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent;root=R.parents[2]
previous=R/'llama3_chain_b1_full'
while True:
    p=previous/'campaign.json'
    rows=json.loads(p.read_text()) if p.exists() else []
    if any(x['status']!='complete' for x in rows):raise RuntimeError('Prior campaign failure')
    if len(rows)==len(json.loads((previous/'plan.json').read_text())):break
    time.sleep(5)
time.sleep(5)
children=[]
for gpu,name,model in [('0','llama2','/home/chokwans99/models_mlsys/layerskip-llama2-7b'),
                       ('1','llama3','/data/chokwans99/models/layerskip-llama3-8B')]:
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,SSD_CUDA_ARCH='8.9',SSD_ATTN_BACKEND='auto',
             SSD_SHM_NAME='paper7_prefix_'+name,OMP_NUM_THREADS='4')
    log=(R/(name+'_paper_prefix.log')).open('w')
    cmd=[sys.executable,'-O',str(root/'ssd/bench/mlsys_paper_prefix.py'),'--model',model,
         '--prompts',str(R.parent/'questions.json'),'--output',str(R/(name+'_paper_prefix.json'))]
    children.append((subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT),log))
for proc,log in children:
    code=proc.wait();log.close()
    if code:raise RuntimeError('Prefix model verification failed')
jobs=[]
for mode in ('chain','tree'):
 for arm in ('base','ported'):
    jobs.append(dict(name=f'integration_{mode}_{arm}_smoke_r0',arm='base',draft_seed=131,
        env={} if arm=='base' else dict(PAPER_CHECKOUT='/home/chokwans99/PSD-paper-port-review',SSD_PAPER_TREE_ACCEPT_GRAPH='1'),
        args=['--mode',mode,'--batch','1','--prompts',str(R.parent/'round5/tuning48.json'),
              '--limit','4','--warm-prompts','2','--output-tokens','32','--seed','9501']))
plan=R/'integration_plan.json';plan.write_text(json.dumps(jobs,indent=2))
subprocess.run([sys.executable,str(R/'run_pair.py'),'--plan',str(plan),'--gpus','0,1','--port','28300',
                '--directory',str(R/'integration')],check=True)
checks=[]
for mode in ('chain','tree'):
    a=json.loads((R/'integration'/f'integration_{mode}_base_smoke_r0.json').read_text())
    b=json.loads((R/'integration'/f'integration_{mode}_ported_smoke_r0.json').read_text())
    check=dict(mode=mode,outputs_equal=a['outputs']==b['outputs'],
               warm_outputs_equal=[g['outputs'] for g in a['warm_groups']]==[g['outputs'] for g in b['warm_groups']])
    checks.append(check)
    if not all(check[k] for k in ('outputs_equal','warm_outputs_equal')):raise AssertionError(check)
(R/'PORT_INTEGRATION.json').write_text(json.dumps(checks,indent=2))
