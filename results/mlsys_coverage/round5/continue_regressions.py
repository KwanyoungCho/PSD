"""Run final CUDA correctness suite after this lane releases its GPUs."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--model',default='llama2');p.add_argument('--lane',type=int,default=1)
p.add_argument('--gpu',required=True);p.add_argument('--stream-only',action='store_true');a=p.parse_args()
directory=HERE/f'{a.model}_{"stream" if a.stream_only else "opt"}_l{a.lane}_full'
while True:
    manifest=directory/'campaign.json';plan=directory/'plan.json'
    if manifest.exists() and plan.exists():
        try:rows=json.loads(manifest.read_text());expected=len(json.loads(plan.read_text()))
        except json.JSONDecodeError:time.sleep(2);continue
        if any(r['status']!='complete' for r in rows):raise RuntimeError('Optimization campaign failed')
        if len(rows)==expected:break
    time.sleep(5)
for _ in range(12):
    active=subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if not active:break
    time.sleep(5)
else:raise RuntimeError('Regression GPU is still occupied')
records=[]
for trim in (('1',) if a.stream_only else ('0','1')):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,MLSYS_PYTHON=sys.executable,
        SSD_TREE_LADDER_TRIM=trim,SSD_TREE_FUSED_MATH='1',SSD_BATCH_TREE_BULK_EXPORT='1',
        SSD_TREE_PARALLEL_INSERT='0',SSD_BATCH_TREE_PROXY_STREAM=str(int(a.stream_only)),OMP_NUM_THREADS='4')
    logpath=HERE/(f'regressions_final_stream.log' if a.stream_only else f'regressions_final_trim{trim}.log')
    with logpath.open('w') as log:
        subprocess.run(['bash',str(HERE.parent/'run_regressions.sh')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    records.append(dict(trim=trim,gpu=a.gpu,log=logpath.name,status='passed'))
(HERE/('FINAL_STREAM_REGRESSIONS.json' if a.stream_only else 'FINAL_REGRESSIONS.json')).write_text(json.dumps(records,indent=2)+'\n')
