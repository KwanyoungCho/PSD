"""Run focused checks only after the timing queue releases local GPUs6/7."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=Path('/data/chokwans99/duet-local-followup-20261017')
while not (ROOT/'runs/overlap_pair1_profile/status.json').exists():time.sleep(2)
folder=HERE/'checks';folder.mkdir(exist_ok=False)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='6,7',PYTHONDONTWRITEBYTECODE='1',
    PYTHONPATH=str(ROOT/'code/overlap/ssd')+':'+str(ROOT/'code/overlap/ssd/tests'),
    SSD_CUDA_ARCH='8.9',SSD_HF_CACHE='/tmp',SSD_DATASET_DIR='/tmp',SSD_PROFILE_DUET='0',
    OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',DEFERRED_TEST_OUTPUT=str(folder/'DEFERRED_RECEIVE.json'),
    TRITON_CACHE_DIR=str(ROOT/'cache/triton'))
os.sched_setaffinity(0,{20,21,22,23,52,53,54,55})
python='/home/chokwans99/PSD/ssd/.venv/bin/python'
commands=[('dependency',[python,'-B',str(HERE/'test_deferred_receive.py')])]
tests=['test_response_snapshot','test_response_topology','test_tree_verify_planless',
       'test_tree_topology_gpu','test_tree_verify_mask_direct','test_tree_accept_port']
commands.append(('regressions',[python,'-B','-m','unittest','-v',*tests]))
results=[]
for name,command in commands:
    with (folder/(name+'.log')).open('w') as log:
        proc=subprocess.Popen(command,cwd=ROOT/'code/overlap',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=proc.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            code=-1
    results.append(dict(name=name,command=command,exit_code=code))
    if code!=0:break
(HERE/'CHECKS.json').write_text(json.dumps(dict(status='pass' if all(r['exit_code']==0 for r in results) and len(results)==2 else 'failed',results=results),indent=2)+'\n')
