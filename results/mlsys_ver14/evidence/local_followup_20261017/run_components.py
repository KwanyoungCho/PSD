import os,time,subprocess,json
from pathlib import Path
p=Path(__file__).resolve().parent
r=Path('/data/chokwans99/duet-local-followup-20261017')
while not (r/'runs/mixed_prep/status.json').exists():time.sleep(2)
os.sched_setaffinity(0,{16,17,18,19,48,49,50,51})
env=dict(os.environ,CUDA_VISIBLE_DEVICES='4',PYTHONPATH=str(r/'code/prep/ssd'),SSD_HF_CACHE='/tmp',SSD_DATASET_DIR='/tmp',SSD_CUDA_ARCH='8.9',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TRITON_CACHE_DIR=str(r/'cache/triton'))
with (p/'COMPONENTS.log').open('w') as f:
 result=subprocess.run(['/home/chokwans99/PSD/ssd/.venv/bin/python','-B',str(p/'component_bench.py')],env=env,stdout=f,stderr=subprocess.STDOUT)
(p/'COMPONENT_STATUS.json').write_text(json.dumps(dict(exit_code=result.returncode))+'\n')
