"""Local GPU worker; independent workers use disjoint GPUs and CPU cores."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RAW = Path('/data/chokwans99/duet-local-phase-20261015')


def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2)+'\n')
    tmp.replace(path)


def gpu_state():
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu,clocks.sm,temperature.gpu',
                                   '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    jobs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                    '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    return dict(wall_ns=time.time_ns(), gpu=gpu, jobs=jobs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=['7b', '13b'])
    p.add_argument('--arms', nargs='+')
    p.add_argument('--devices')
    p.add_argument('--cpuset',default='16-23,48-55')
    p.add_argument('--worker',default='main')
    p.add_argument('--port',default='29741')
    args = p.parse_args()
    if socket.gethostname().split('.')[0] != 'orion':
        raise RuntimeError('Local orion only')
    RAW.mkdir(parents=True, exist_ok=True)
    plan = json.loads((HERE/'PLAN.json').read_text())
    sources = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
               for base in ('ssd/ssd', 'ssd/bench') for f in (ROOT/base).rglob('*.py')}
    write(RAW/f'SOURCE_MANIFEST_{args.worker}.json', sources)
    statuses = []
    for model in args.models:
        tp, devices, exit_layer = (1, [4,5],21) if model=='7b' else (2,[4,5,6],26)
        if args.devices:devices=list(map(int,args.devices.split(',')))
        if len(devices)!=tp+1:raise ValueError('Expected target TP GPUs plus one draft GPU')
        target=Path('/home/chokwans99/models_mlsys')/f'layerskip-llama2-{model}'
        idx=json.loads((target/'model.safetensors.index.json').read_text())
        if not all((target/f).exists() for f in set(idx['weight_map'].values())):
            raise RuntimeError(f'Model download incomplete: {target}')
        # Control remains a native timeline run, but has no inner-forward markers.
        arms=[dict(name='control',ds_width=9,ps_width=9,probe=False)]+[
            dict(a,probe=True) for a in plan['arms']]
        arms.append(dict(name='psbase',ds_width=9,ps_width=9,probe=True))
        arms.append(dict(name='base_repeat',ds_width=9,ps_width=9,probe=True))
        if args.arms:arms=[next(a for a in arms if a['name']==name) for name in args.arms]
        else:arms=[a for a in arms if a['name'] not in ('psbase','base_repeat')]
        for arm in arms:
            name=f'{model}_{arm["name"]}'
            folder=RAW/name
            existing=folder/'run.json'
            if existing.exists():
                if json.loads(existing.read_text()).get('status')=='complete':
                    statuses.append(dict(name=name,status='complete',reused=True));continue
                raise RuntimeError(f'Existing incomplete run {existing}; preserve and review it')
            state=gpu_state()
            uuids={line.split(',')[1].strip() for line in state['gpu'] if int(line.split(',')[0]) in devices}
            if any(line.split(',')[0].strip() in uuids for line in state['jobs'] if line):
                raise RuntimeError(f'Required GPUs {devices} occupied; do not terminate another job')
            folder.mkdir(parents=True)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,devices)),
                PAPER_CHECKOUT=str(ROOT),PAPER_CPUSET=args.cpuset,PAPER_ARM=name,PAPER_SEED='9101',
                SSD_CUDA_ARCH='8.9',SSD_HF_CACHE='/tmp',SSD_DATASET_DIR='/tmp',
                SSD_TREE_EXEC='1',SSD_TREE_EXEC_WARMUP='all',SSD_P1_TREE_EXEC_WARMUP='all',
                SSD_ASYNC_PROXY_SEND='1',SSD_PROFILE_DUET='1',SSD_PROFILE_DUET_DETAIL='1',
                SSD_PROFILE_DUET_MAX_EVENTS='18000',SSD_PROFILE_DIR=str(folder),
                DUET_REVIEW_CAPTURE_AUDIT=str(folder/'captures'),SSD_FORCE_SPLIT_K1K2='1',
                SSD_DUET_EXIT_REPLICA='0',DUET_PHASE_PROBE=str(int(arm['probe'])),SSD_DIST_PORT=args.port)
            cmd=[sys.executable,'-B',str(ROOT/'ssd/bench/mlsys_local_phase_latency.py'),
                '--target',str(target),'--draft','/home/chokwans99/models_mlsys/llama-68m',
                '--target-tp',str(tp),'--mode','tree','--batch','1','--temperature','0.7',
                '--prompts',str(HERE/'INPUTS.json'),'--output-tokens','192','--input-cap','512',
                '--warm-prompts','3','--passes','2','--exit-layer',str(exit_layer),
                '--k1','3','--k2','3','--p1-verify','40','--p1-nodes','40',
                '--p2-verify','40','--p2-nodes','40','--roots','3','--siblings','3',
                '--ds-width',str(arm['ds_width']),'--ps-width',str(arm['ps_width']),
                '--ps-roots','9','--expansion','last','--output',str(folder/'run.json')]
            write(folder/'COMMAND.json',dict(argv=cmd,env={k:v for k,v in env.items() if k.startswith(('PAPER_','SSD_','DUET_')) or k in ('CUDA_VISIBLE_DEVICES','MASTER_PORT')}))
            print('START',name,flush=True)
            with (folder/'process.log').open('w') as log, (folder/'telemetry.jsonl').open('w') as tel:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                while proc.poll() is None:
                    tel.write(json.dumps(gpu_state())+'\n');tel.flush()
                    time.sleep(2)
            row=dict(name=name,status='complete' if proc.returncode==0 else 'failed',returncode=proc.returncode)
            statuses.append(row);write(RAW/f'STATUS_{args.worker}.json',dict(runs=statuses,status='running'))
            if proc.returncode:
                raise RuntimeError(f'{name} failed; inspect {folder}/process.log')
            run=json.loads((folder/'run.json').read_text())
            if run['status']!='complete':raise RuntimeError(f'{name} incomplete')
            if not all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==digest for f,digest in sources.items()):
                raise RuntimeError('Runtime/benchmark changed during campaign')
            print('COMPLETE',name,run['summary'],flush=True)
    write(RAW/f'STATUS_{args.worker}.json',dict(runs=statuses,status='complete',models=args.models))


if __name__=='__main__':main()
