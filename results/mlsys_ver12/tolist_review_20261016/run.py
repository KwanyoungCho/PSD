"""Same local GPU triple, alternating before/after/after/before."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
RAW=Path('/data/chokwans99/duet-tolist-review-20261016')
sys.path.insert(0,str(HERE.parent/'local_phase_latency_20261015'))
from run_campaign import gpu_state,write


def main():
    if socket.gethostname().split('.')[0]!='orion':raise RuntimeError('Local server only')
    schedule=[('before1','before'),('after1','after'),('after2','after'),('before2','before')]
    main_sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'ssd/ssd').rglob('*.py')}
    write(HERE/'MAIN_RUNTIME_BEFORE.json',main_sources)
    completed=[]
    for name,arm in schedule:
        folder=RAW/'runs'/name;checkout=RAW/arm
        if (folder/'run.json').exists():raise FileExistsError(folder)
        st=gpu_state();uuids={r.split(',')[1].strip() for r in st['gpu'] if int(r.split(',')[0]) in (4,5,6)}
        if any(r and r.split(',')[0].strip() in uuids for r in st['jobs']):raise RuntimeError('Assigned GPUs busy; no jobs terminated')
        folder.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='4,5,6',PAPER_CHECKOUT=str(checkout),
            PAPER_CPUSET='16-23,48-55',PAPER_ARM=name,PAPER_SEED='9101',
            SSD_CUDA_ARCH='8.9',SSD_HF_CACHE='/tmp',SSD_DATASET_DIR='/tmp',
            SSD_TREE_EXEC='1',SSD_TREE_EXEC_WARMUP='all',SSD_P1_TREE_EXEC_WARMUP='all',
            SSD_ASYNC_PROXY_SEND='1',SSD_PROFILE_DUET='1',SSD_PROFILE_DUET_DETAIL='1',
            SSD_PROFILE_DUET_MAX_EVENTS='18000',SSD_PROFILE_DIR=str(folder),
            DUET_REVIEW_CAPTURE_AUDIT=str(folder/'captures'),SSD_FORCE_SPLIT_K1K2='1',
            SSD_DUET_EXIT_REPLICA='0',SSD_DIST_PORT='29761')
        cmd=[sys.executable,'-B',str(HERE/'benchmark.py'),'--target','/home/chokwans99/models_mlsys/layerskip-llama2-13b',
            '--draft','/home/chokwans99/models_mlsys/llama-68m','--target-tp','2','--mode','tree','--batch','1',
            '--temperature','0.7','--prompts',str(HERE.parent/'local_phase_latency_20261015/INPUTS.json'),
            '--output-tokens','192','--input-cap','512','--warm-prompts','3','--passes','2',
            '--exit-layer','26','--k1','3','--k2','3','--p1-verify','40','--p1-nodes','40',
            '--p2-verify','40','--p2-nodes','40','--roots','3','--siblings','3',
            '--ds-width','9','--ps-width','9','--ps-roots','9','--expansion','last','--output',str(folder/'run.json')]
        sources={str(p.relative_to(checkout)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for base in ('ssd/ssd','ssd/bench') for p in (checkout/base).rglob('*.py')}
        write(folder/'SOURCE_MANIFEST.json',sources)
        write(folder/'COMMAND.json',dict(argv=cmd,env={k:v for k,v in env.items() if k.startswith(('PAPER_','SSD_','DUET_')) or k=='CUDA_VISIBLE_DEVICES'}))
        print('START',name,flush=True)
        with (folder/'process.log').open('w') as log,(folder/'telemetry.jsonl').open('w') as tel:
            p=subprocess.Popen(cmd,cwd=checkout,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write(folder/'PROCESS.json',dict(pid=p.pid,pgid=p.pid))
            while p.poll() is None:
                tel.write(json.dumps(gpu_state())+'\n');tel.flush();time.sleep(2)
        if p.returncode:raise RuntimeError(f'{name} failed: {p.returncode}; keep logs')
        assert all(hashlib.sha256((checkout/f).read_bytes()).hexdigest()==s for f,s in sources.items())
        r=json.loads((folder/'run.json').read_text());assert r['status']=='complete'
        completed.append(name);write(RAW/'STATUS.json',dict(status='running',completed=completed,schedule=schedule))
        print('COMPLETE',name,r['summary'],flush=True)
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==s for f,s in main_sources.items())
    write(RAW/'STATUS.json',dict(status='complete',completed=completed,schedule=schedule))


if __name__=='__main__':main()
