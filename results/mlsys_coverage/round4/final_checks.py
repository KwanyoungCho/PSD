"""Wait for this task's B1 campaign, then run both correctness configurations."""
import json,os,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
campaign=HERE/'llama2_b1_short/campaign.json'
while True:
 if campaign.exists():
  rows=json.loads(campaign.read_text())
  if any(r['status']!='complete' for r in rows):raise RuntimeError('Prerequisite failed')
  if len(rows)==3:break
 time.sleep(5)
uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
for _ in range(12):
 if uuid not in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines():break
 time.sleep(5)
else:raise RuntimeError('GPU0 is busy; no process killed')
results=[]
for name,flag in [('default','0'),('fused_bulk_parallel','1')]:
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',MLSYS_PYTHON='/home/chokwans99/PSD/ssd/.venv/bin/python',
          SSD_TREE_FUSED_MATH=flag,SSD_TREE_PARALLEL_INSERT=flag,SSD_BATCH_TREE_BULK_EXPORT=flag,
          SSD_TREE_EXPANSION_POLICY='q_path')
 start=time.time()
 with (HERE/f'regressions_final_{name}.log').open('w') as f:
  r=subprocess.run(['bash','results/mlsys_coverage/run_regressions.sh'],env=env,stdout=f,stderr=subprocess.STDOUT)
 results.append(dict(name=name,returncode=r.returncode,seconds=time.time()-start))
 (HERE/'final_checks.json').write_text(json.dumps(results,indent=2))
 print(results[-1],flush=True)
 if r.returncode:raise RuntimeError('Final regression failed; inspect preserved log')
