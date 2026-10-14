"""Isolated verification queue: never touch ongoing AL experiment snapshots."""
import concurrent.futures,hashlib,json,os,signal,subprocess,time,threading
from pathlib import Path
ROOT=Path('/home/chokwans99/duet_blackwell_review_tie_20261014')
PY='/home/chokwans99/anaconda3/envs/ssd/bin/python'
BASE='/home/chokwans99/equal_budget_models'
OUT=ROOT/'checks';OUT.mkdir(exist_ok=True)
ENV=dict(os.environ,SSD_HF_CACHE='/tmp',SSD_DATASET_DIR='/tmp',OMP_NUM_THREADS='2',
    PYTHONPATH='/home/chokwans99/duet_blackwell_review_final_20261014/deps311:'+str(ROOT/'ssd'))
ROWS=[]
LOCK=threading.Lock()

def save():
    with LOCK:
        tmp=OUT/'validation.tmp';tmp.write_text(json.dumps(ROWS,indent=2));tmp.replace(OUT/'validation.json')

def run(name,cmd,gpus,extra=None,timeout=1800):
    row=dict(name=name,command=cmd,gpus=gpus,status='running',started=time.time())
    ROWS.append(row);save()
    with (OUT/(name+'.log')).open('w') as log:
        proc=subprocess.Popen(cmd,cwd=ROOT,env=dict(ENV,CUDA_VISIBLE_DEVICES=gpus,**(extra or {})),
                              stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        row['pid']=proc.pid;save()
        try:rc=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            rc=124
    row.update(status='complete' if rc==0 else 'failed',returncode=rc,ended=time.time());save()
    return rc

def smoke(mode,batch,temp,gpu):
    name=f'{mode}_b{batch}_t{temp}'
    cmd=[PY,'ssd/bench/mlsys_blackwell_smoke.py','--target',BASE+'/layerskip-llama2-7b',
         '--draft',BASE+'/AMD-Llama-135m','--mode',mode,'--batch',str(batch),
         '--temperature',str(temp),'--output',str(OUT/(name+'.json'))]
    return run(name,cmd,gpu,dict(SSD_DIST_PORT=str(29740+batch),
        SSD_TREE_EXPANSION_POLICY='q_path',SSD_DEBUG_TOKENS='0',
        SSD_TREE_EXEC_WARMUP='all'))

def final_suites():
    # Previous attempt is preserved. Let its isolated test worker finish first.
    while True:
        try:
            if Path('/proc/2926754/stat').read_text().split()[2]=='Z':break
        except FileNotFoundError:break
        time.sleep(10)
    return run('full_final',[PY,'results/mlsys_ver01/extensions/check_suites.py',
        '--output-dir',str(OUT/'full_final')],'4')

with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    suites=pool.submit(final_suites)
    for i,arm in enumerate(('before','after','after','before')):
        source=('/home/chokwans99/duet_blackwell_review_final_20261014'
                if arm=='before' else str(ROOT))
        run(f'proxy_{i}_{arm}',[PY,str(ROOT/'checks/measure_proxy_review.py'),
            '--root',source,'--output',str(OUT/f'proxy_{i}_{arm}.json')],'5')
    for b in (1,2):
        if smoke('ar',b,0,'5'):raise SystemExit(1)
    if suites.result():raise SystemExit(1)
# Same GPUs now perform short, unadapted two-device functional runs.
for b in (1,2):
    for t in (0,.7):
        if smoke('tree',b,t,'4,5'):
            # Preserve original failure, do not launch an endless retry loop.
            raise SystemExit(1)
print('VALIDATION_QUEUE_DRAINED',flush=True)
