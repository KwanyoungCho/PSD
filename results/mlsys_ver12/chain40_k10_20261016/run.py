"""Run native chains locally; retain command, source hashes and GPU telemetry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RAW=Path('/data/chokwans99/duet-chain40-k10-20261016')
sys.path.insert(0,str(HERE.parent/'local_phase_latency_20261015'))
from run_campaign import gpu_state,write


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='+',default=['7b','13b'])
    parser.add_argument('--pilot',action='store_true');a=parser.parse_args()
    if socket.gethostname().split('.')[0]!='orion':raise RuntimeError('Local server only')
    RAW.mkdir(parents=True,exist_ok=True)
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
             for directory in ('ssd/ssd','ssd/bench') for p in (ROOT/directory).rglob('*.py')}
    statuses=[]
    for model in a.models:
        tp,devices,exit_layer=(1,[4,5],21) if model=='7b' else (2,[4,5,6],26)
        name=('pilot_' if a.pilot else '')+model;folder=RAW/name
        if (folder/'run.json').exists():raise FileExistsError(folder)
        state=gpu_state();uuids={r.split(',')[1].strip() for r in state['gpu'] if int(r.split(',')[0]) in devices}
        if any(r.split(',')[0].strip() in uuids for r in state['jobs'] if r):
            raise RuntimeError(f'GPUs {devices} occupied; no processes terminated')
        folder.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,devices)),
            PAPER_CHECKOUT=str(ROOT),PAPER_CPUSET='16-23,48-55',PAPER_ARM=name,PAPER_SEED='9101',
            SSD_CUDA_ARCH='8.9',SSD_HF_CACHE='/tmp',SSD_DATASET_DIR='/tmp',SSD_TREE_EXEC='0',
            SSD_ASYNC_PROXY_SEND='1',SSD_PROFILE_DUET='1',SSD_PROFILE_DUET_DETAIL='1',
            SSD_PROFILE_DUET_MAX_EVENTS='60000',SSD_PROFILE_DIR=str(folder),
            DUET_REVIEW_CAPTURE_AUDIT=str(folder/'captures'),SSD_FORCE_SPLIT_K1K2='1',
            SSD_DUET_EXIT_REPLICA='0',SSD_DIST_PORT='29751')
        cmd=[sys.executable,'-B',str(ROOT/'ssd/bench/mlsys_chain_phase_latency.py'),
             '--target',f'/home/chokwans99/models_mlsys/layerskip-llama2-{model}',
             '--draft','/home/chokwans99/models_mlsys/llama-68m','--mode','chain','--target-tp',str(tp),
             '--batch','1','--temperature','0.7','--prompts',str(HERE/'INPUTS.json'),
             '--input-cap','512','--output-tokens','96' if a.pilot else '192',
             '--warm-prompts','1' if a.pilot else '3','--passes','1' if a.pilot else '2',
             '--exit-layer',str(exit_layer),'--k1','10','--k2','10','--roots','4',
             '--p1-nodes','10','--p1-verify','10','--p2-nodes','10','--p2-verify','10',
             '--ds-width','40','--ps-width','40','--ps-roots','40','--siblings','1',
             '--output',str(folder/'run.json')]
        if a.pilot:cmd+=['--limit','2']
        write(folder/'COMMAND.json',dict(argv=cmd,env={k:v for k,v in env.items() if k.startswith(('PAPER_','SSD_','DUET_')) or k=='CUDA_VISIBLE_DEVICES'}))
        write(folder/'SOURCE_MANIFEST.json',sources)
        print('START',name,flush=True)
        with (folder/'process.log').open('w') as log,(folder/'telemetry.jsonl').open('w') as tel:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(folder/'PROCESS.json',dict(pid=proc.pid,pgid=proc.pid))
            while proc.poll() is None:
                tel.write(json.dumps(gpu_state())+'\n');tel.flush();time.sleep(2)
        status=dict(name=name,returncode=proc.returncode,status='complete' if proc.returncode==0 else 'failed')
        statuses.append(status);write(RAW/('PILOT_STATUS.json' if a.pilot else 'STATUS.json'),dict(status='running',runs=statuses))
        if proc.returncode:raise RuntimeError(f'{name} failed, preserve {folder}')
        r=json.loads((folder/'run.json').read_text());assert r['status']=='complete'
        assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==s for p,s in sources.items())
        print('COMPLETE',name,r['summary'],flush=True)
    write(RAW/('PILOT_STATUS.json' if a.pilot else 'STATUS.json'),dict(status='complete',runs=statuses))


if __name__=='__main__':main()
