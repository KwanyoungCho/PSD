"""Acquire the same GPU locks as the paper campaigns; preserve every attempt."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OLD = Path('/home/eslab/chokwans99/PSD-tree-quality-v14')
BASE = Path('/home/eslab/chokwans99')
PYTHON = BASE/'baseline/.venv-ssd/bin/python'
TARGET = '/home/eslab/models/hub/models--facebook--layerskip-llama2-70B/snapshots/0a1815fcbd11543b0de227aa8dcbced952149411'
DRAFT = '/home/eslab/models/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6'


def hashes():
    paths = sorted((ROOT/'ssd/ssd').rglob('*.py'))
    paths += sorted(Path(__file__).parent.glob('*.py'))
    paths += sorted((OLD/'ssd/ssd').rglob('*.py'))
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    locks=[]
    for i in (0, 1):
        for path in (BASE/f'PSD/results/duet_gpu{i}.lock', BASE/f'DUET-reference-70b-20261020/gpu_locks/gpu{i}.lock'):
            h=path.open('a')
            fcntl.flock(h, fcntl.LOCK_EX|fcntl.LOCK_NB)
            locks.append(h)
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'], text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'], text=True)
    selected={r.split(',')[1].strip() for r in gpu.splitlines() if int(r.split(',')[0]) in (0,1)}
    assert not any(r.split(',')[0].strip() in selected for r in apps.splitlines()), 'GPU0/1 occupied'
    assert subprocess.check_output(['git','-C',str(OLD),'rev-parse','HEAD'],text=True).strip() == 'dedc85ccb299df2d76fcf34d50f05f2d8747a32a'
    assert not subprocess.check_output(['git','-C',str(OLD),'status','--porcelain'],text=True).strip()
    before=hashes()
    (args.out/'SOURCE_MANIFEST.json').write_text(json.dumps(before, indent=2)+'\n')
    (args.out/'CHANGE.patch').write_text(subprocess.check_output(['git','-C',str(ROOT),'diff'],text=True))
    env={k:v for k,v in os.environ.items() if not k.startswith(('SSD_', 'DUET_', 'PAPER_', 'CUDA_', 'TORCH_')) and k not in ('PYTHONPATH','LD_PRELOAD','LD_LIBRARY_PATH')}
    env.update(CUDA_VISIBLE_DEVICES='0,1',SSD_CUDA_ARCH='12.0',TORCH_CUDA_ARCH_LIST='12.0',
        SSD_HF_CACHE='/home/eslab/models/hub',SSD_DATASET_DIR=str(BASE/'baseline/data'),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',SSD_DIST_PORT='31257',
        SSD_SHM_NAME='duet_padding_check_20261022',SSD_PROFILE='0',SSD_PROFILE_DRAFT='0',
        SSD_PROFILE_DUET='0',SSD_PROFILE_DUET_DETAIL='0',SSD_PROFILE_TARGET='0',
        SSD_DUET_EXIT_REPLICA='0',SSD_PROXY_STREAM='0',SSD_BATCH_TREE_PROXY_STREAM='0',
        SSD_TREE_TOPOLOGY_GPU='1',SSD_TREE_VERIFY_BUCKETS='2,4',SSD_TREE_VERIFY_WORKSPACE_MB='256',
        OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',
        PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTHONPATH=str(ROOT/'ssd'))
    cmd=[str(PYTHON),'-B','-m','torch.distributed.run','--nproc_per_node=2','--master_port=31256',
        str(Path(__file__).with_name('validate.py')),'--out',str(args.out),'--baseline',str(OLD),
        '--target',TARGET,'--draft',DRAFT,'--inputs',str(BASE/'DUET-reference-70b-20261020/inputs/CANONICAL_DUET.json')]
    started=time.time()
    (args.out/'PROCESS.json').write_text(json.dumps(dict(state='starting',command=cmd,started=started),indent=2)+'\n')
    with (args.out/'process.log').open('x') as log:
        child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        (args.out/'PROCESS.json').write_text(json.dumps(dict(pid=child.pid,command=cmd,started=started),indent=2)+'\n')
        code=child.wait()
    unchanged=hashes()==before
    (args.out/'PROCESS.json').write_text(json.dumps(dict(pid=child.pid,command=cmd,exit_code=code,source_unchanged=unchanged,finished=time.time()),indent=2)+'\n')
    assert code==0 and unchanged, 'Validation failed: preserve this attempt'


if __name__ == '__main__':
    main()
