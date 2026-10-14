"""Re-run the contaminated arm after the authorized local GPU4/5 queue exits."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
R=Path(__file__).resolve().parents[3];D=Path(__file__).resolve().parent
name='local_45_retry_external';out=Path('/data/chokwans99/equal-tree-paper-full-20261014')/name
previous=3342688
uuids=set(subprocess.check_output(['nvidia-smi','-i','4,5','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines())
while True:
    old=Path('/proc')/str(previous)/'cmdline'
    previous_alive=old.exists() and b'mlsys_tree_campaign.py' in old.read_bytes()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()
    busy=[s for s in apps if s.split(',')[0] in uuids]
    if not previous_alive and not busy:break
    time.sleep(30)
manifest=json.loads((D/'SOURCE_SHA256.json').read_text())
changed=[k for k,v in manifest.items() if hashlib.sha256((R/k).read_bytes()).hexdigest()!=v]
if changed:raise RuntimeError('Frozen runtime changed: '+repr(changed))
print('START clean rerun on GPU4/5',time.time(),flush=True)
os.environ['SSD_CUDA_ARCH']='8.9';os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.chdir(R)
os.execv(sys.executable,[sys.executable,'-B','ssd/bench/mlsys_tree_campaign.py','--plan',str(D/(name+'.json')),'--directory',str(out),'--gpus','4,5','--port','29010','--timeout','14400'])
