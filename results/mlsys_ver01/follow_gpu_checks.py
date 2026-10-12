"""Use the completed tree-repeat campaign's GPU for correctness, never during TPS."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

R = Path(__file__).resolve().parent
ROOT = R.parents[1]
deadline = time.time() + 7200
while True:
    f = R / 'tree_r1/campaign.json'
    rows = json.loads(f.read_text()) if f.exists() else []
    if any(v['status'] != 'complete' or v['exit_code'] for v in rows):
        raise RuntimeError('Tree-repeat prerequisite failed')
    if len(rows) == len(json.loads((R / 'tree_r1_plan.json').read_text())):
        break
    if time.time() > deadline:
        raise TimeoutError('Tree-repeat campaign did not complete')
    time.sleep(5)
time.sleep(3)
active = subprocess.check_output(['nvidia-smi', '-i', '6', '--query-compute-apps=pid',
                                   '--format=csv,noheader'], text=True).strip()
if active:
    raise RuntimeError('GPU6 still occupied: ' + active)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='6', SSD_HF_CACHE='/tmp', SSD_DATASET_DIR='/tmp',
           PYTHONPATH=str(ROOT/'ssd'), OMP_NUM_THREADS='4')
os.sched_setaffinity(0, set(range(24, 32)) | set(range(56, 64)))
checks = [
    ('p2_gpu_tests', [sys.executable, '-m', 'unittest', 'discover', '-s', 'ssd/tests', '-p', 'test_p2_tree_alloc.py']),
    ('p1_gpu_tests', [sys.executable, '-m', 'unittest', 'discover', '-s', 'ssd/tests', '-p', 'test_p1_dynamic_tree.py']),
    ('chain_gpu_parity', [sys.executable, 'ssd/bench/mlsys_ver01_chain_check.py', str(R/'CHAIN_GPU_PARITY.json')]),
]
result = []
for name, command in checks:
    print('START', name, flush=True)
    with (R/(name+'.log')).open('w') as out:
        process = subprocess.run(command, env=env, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
    result.append(dict(name=name, command=command, exit_code=process.returncode))
    (R/'GPU_CHECKS.json').write_text(json.dumps(result, indent=2))
    if process.returncode:
        raise RuntimeError('GPU check failed: '+name)
    print('PASS', name, flush=True)
