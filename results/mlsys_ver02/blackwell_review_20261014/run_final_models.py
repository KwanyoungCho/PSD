"""Same frozen release source as the full regression suite, two GPU pairs."""
import concurrent.futures,json,os,signal,subprocess,time,threading
from pathlib import Path
ROOT=Path('/home/chokwans99/duet_blackwell_review_release_20261014')
OUT=ROOT/'checks/models';OUT.mkdir(exist_ok=True)
ENV=dict(os.environ,SSD_HF_CACHE='/tmp',SSD_DATASET_DIR='/tmp',OMP_NUM_THREADS='2',
    PYTHONPATH='/home/chokwans99/duet_blackwell_review_final_20261014/deps311:'+str(ROOT/'ssd'),
    SSD_TREE_EXPANSION_POLICY='q_path',SSD_DEBUG_TOKENS='0',SSD_TREE_EXEC_WARMUP='all')
ROWS=[];LOCK=threading.Lock()

def save():
    with LOCK:
        p=OUT/'status.tmp';p.write_text(json.dumps(ROWS,indent=2));p.replace(OUT/'status.json')

def family(b,gpus):
    if b==1:
        deadline=time.time()+1200
        while not (ROOT/'checks/full_final/CHECKS_PROVENANCE.json').exists():
            if time.time()>deadline:raise RuntimeError('Final suite completion missing')
            time.sleep(5)
        rows=json.loads((ROOT/'checks/full_final/CHECKS.json').read_text())
        if any(x['returncode'] for x in rows):raise RuntimeError('Final regression failed')
    for mode,temp in [('ar',0),('tree',0),('tree',.7)]:
        name=f'{mode}_b{b}_t{temp}'
        cmd=['/home/chokwans99/anaconda3/envs/ssd/bin/python','ssd/bench/mlsys_blackwell_smoke.py',
            '--target','/home/chokwans99/equal_budget_models/layerskip-llama2-7b',
            '--draft','/home/chokwans99/equal_budget_models/AMD-Llama-135m',
            '--mode',mode,'--batch',str(b),'--temperature',str(temp),'--output',str(OUT/(name+'.json'))]
        row=dict(name=name,command=cmd,gpus=gpus if mode=='tree' else gpus.split(',')[0],status='running',started=time.time())
        ROWS.append(row);save()
        with (OUT/(name+'.log')).open('w') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=dict(ENV,CUDA_VISIBLE_DEVICES=row['gpus'],SSD_DIST_PORT=str(29850+b)),
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            row['pid']=proc.pid;save()
            try:code=proc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                code=124
        row.update(status='complete' if code==0 else 'failed',returncode=code,ended=time.time());save()
        if code:raise RuntimeError(f'{name} failed: {code}')

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    fs=[pool.submit(family,1,'2,3'),pool.submit(family,2,'4,5')]
    for f in fs:f.result()
print('FINAL_FUNCTIONAL_COMPLETE',flush=True)
