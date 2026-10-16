"""One isolated nova run, with source hashes and external-GPU-use telemetry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT=Path('/data2/chokwans99/target-gap-20261017')
PYTHON='/home/chokwans99/anaconda3/envs/ssd/bin/python'


def gpu_state():
    def query(args):
        return subprocess.check_output(['nvidia-smi',*args],text=True).strip().splitlines()
    return dict(time_ns=time.time_ns(),
        gpu=query(['--query-gpu=index,uuid,name,memory.used,utilization.gpu,clocks.sm,temperature.gpu,power.draw','--format=csv,noheader']),
        processes=query(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader']))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--variant',required=True)
    p.add_argument('--model',choices=['7b','13b'],required=True)
    p.add_argument('--tag',required=True)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--passes',type=int,default=2)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--launch',action='store_true')
    args=p.parse_args()
    if socket.gethostname().split('.')[0]!='nova':raise RuntimeError('nova only')
    folder=ROOT/'runs'/args.tag
    if args.launch:
        folder.mkdir(parents=True,exist_ok=False)
        argv=[sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]]
        argv.remove('--launch')
        with (folder/'controller.log').open('w') as log:
            proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (folder/'controller.json').write_text(json.dumps(dict(pid=proc.pid,argv=argv))+'\n')
        print(json.dumps(dict(tag=args.tag,pid=proc.pid)),flush=True)
        return
    folder.mkdir(parents=True,exist_ok=True)
    checkout=ROOT/args.variant
    gpus=(2,3) if args.model=='7b' else (4,5,6)
    state=gpu_state()
    uuids={r.split(',')[1].strip() for r in state['gpu'] if int(r.split(',')[0]) in gpus}
    if any(r.split(',')[0].strip() in uuids for r in state['processes'] if r):
        raise RuntimeError('Assigned GPU busy; no process killed')
    inputs=checkout/'results/mlsys_ver12/target_gap_optimization_20261017/INPUTS.json'
    sources={str(f.relative_to(checkout)):hashlib.sha256(f.read_bytes()).hexdigest()
        for base in ('ssd/ssd','ssd/bench') for f in (checkout/base).rglob('*.py')}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),PAPER_CHECKOUT=str(checkout),
        PAPER_CPUSET='0-7,32-35' if args.model=='7b' else '8-15,40-43',PAPER_ARM=args.tag,PAPER_SEED='9101',
        PYTHONPATH=str(checkout/'ssd'),SSD_CUDA_ARCH='8.6',SSD_HF_CACHE='/tmp',SSD_DATASET_DIR='/tmp',
        SSD_TREE_EXEC='0',SSD_TREE_EXEC_WARMUP='all',SSD_P1_TREE_EXEC_WARMUP='all',SSD_ASYNC_PROXY_SEND='1',
        SSD_PROFILE_DUET='1' if args.profile else '0',SSD_PROFILE='0',SSD_PROFILE_DRAFT='0',
        SSD_PROFILE_DUET_DETAIL='1' if args.profile else '0',SSD_PROFILE_DUET_MAX_EVENTS='18000',
        SSD_PROFILE_DIR=str(folder),SSD_FORCE_SPLIT_K1K2='1',SSD_DUET_EXIT_REPLICA='0',
        SSD_SHM_NAME='duet_gap_'+args.tag,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        SSD_DIST_PORT='29817' if args.model=='7b' else '29818',
        TRITON_CACHE_DIR=str(ROOT/'cache/triton'),TORCHINDUCTOR_CACHE_DIR=str(ROOT/'cache/inductor'),
        TMPDIR='/data2/chokwans99/duet-build-cache/tmp')
    cmd=[PYTHON,'-O','-B',str(ROOT/'benchmark_chain.py'),
        '--target',f'/data2/chokwans99/models/layerskip-llama2-{args.model[:-1]}B',
        '--draft','/data2/chokwans99/models/llama-68m','--target-tp','1' if args.model=='7b' else '2',
        '--mode','chain','--batch','1','--temperature','0.7','--prompts',str(inputs),
        '--output-tokens','192','--input-cap','512','--warm-prompts','3','--passes',str(args.passes),
        '--exit-layer','21' if args.model=='7b' else '26','--k1','10','--k2','10',
        '--p1-verify','10','--p1-nodes','10','--p2-verify','10','--p2-nodes','10',
        '--roots','4','--siblings','1','--ds-width','40','--ps-width','40','--ps-roots','40',
        '--expansion','last','--limit',str(args.limit),'--output',str(folder/'run.json')]
    # Preserve the original spelling of the on-disk model directories.
    cmd[cmd.index('--target')+1]='/data2/chokwans99/models/layerskip-llama2-7B' if args.model=='7b' else '/data2/chokwans99/models/layerskip-llama2-13B'
    provenance=dict(argv=cmd,external_wrapper_sha256=hashlib.sha256((ROOT/'benchmark_chain.py').read_bytes()).hexdigest(),env={k:v for k,v in env.items() if k.startswith(('PAPER_','SSD_','TRITON_','TORCHINDUCTOR_','OMP_','MKL_')) or k in ('CUDA_VISIBLE_DEVICES','PYTHONPATH')},
        source_sha256=sources,gpus=gpus,initial_gpu_state=state)
    (folder/'PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
    with (folder/'process.log').open('w') as log,(folder/'telemetry.jsonl').open('w') as tel:
        proc=subprocess.Popen(cmd,cwd=checkout,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (folder/'process.json').write_text(json.dumps(dict(pid=proc.pid,pgid=proc.pid))+'\n')
        while proc.poll() is None:
            tel.write(json.dumps(gpu_state())+'\n');tel.flush();time.sleep(2)
    unchanged=all(hashlib.sha256((checkout/f).read_bytes()).hexdigest()==h for f,h in sources.items())
    status=dict(status='complete' if proc.returncode==0 and unchanged else 'failed',exit_code=proc.returncode,
                source_unchanged=unchanged,model=args.model,variant=args.variant,profile=args.profile)
    (folder/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status),flush=True)
    if proc.returncode or not unchanged:raise SystemExit(1)


if __name__=='__main__':main()
