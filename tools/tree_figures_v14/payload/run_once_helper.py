"""Frozen local v1.5 run; source changes invalidate the saved result."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(os.environ['V14_REVIEW_RAW'])
PYTHON = sys.executable
PAIRS = [(6, 7), (4, 5), (1, 2)]
CPUSETS = ['20-23,52-55','16-19,48-51','0-3,32-35']


def process_groups(snapshot):
    groups = {}
    for line in snapshot['processes']:
        if not line: continue
        pid = int(line.split(',')[1])
        try: groups[str(pid)] = os.getpgid(pid)
        except ProcessLookupError: groups[str(pid)] = None
    return groups


def state():
    def q(args):
        return subprocess.check_output(['nvidia-smi', *args], text=True).strip().splitlines()
    return dict(time_ns=time.time_ns(), gpu=q([
        '--query-gpu=index,uuid,memory.used,utilization.gpu,clocks.sm,temperature.gpu,power.draw',
        '--format=csv,noheader']), processes=q([
        '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader']))


def run(job, slot):
    tag = job['tag']; folder = ROOT/'runs'/tag
    if (folder/'status.json').exists():
        return json.loads((folder/'status.json').read_text())
    gpus = job.get('gpus', PAIRS[slot])
    initial = state()
    uuids = {r.split(',')[1].strip() for r in initial['gpu'] if int(r.split(',')[0]) in gpus}
    if any(r.split(',')[0].strip() in uuids for r in initial['processes'] if r):
        result = dict(status='busy', tag=tag, gpus=gpus)
        return result
    folder.mkdir(parents=True, exist_ok=False)
    checkout = Path(job['checkout'])
    sources = {str(f.relative_to(checkout)): hashlib.sha256(f.read_bytes()).hexdigest()
               for base in ('ssd/ssd', 'ssd/bench') for f in (checkout/base).rglob('*.py')}
    config = dict(exit_layer=22, k1=8, k2=5, p1_verify=15, p2_verify=15,
                  roots=4, siblings=3, ds_width=32, ps_width=15, ps_roots=15,
                  passes=1, output_tokens=192, input_cap=512, warm_prompts=3,
                  prompts='INPUTS12.json', limit=0, seed=9101, temperature=0.7)
    config.update(job.get('config', {}))
    profile = bool(job.get('profile', False))
    cache_root=Path(os.environ.get('VER15_CACHE_ROOT',str(ROOT)))
    for name in ('cache/triton','cache/inductor','tmp'):
        (cache_root/name).mkdir(parents=True,exist_ok=True)
    base_env = {k:v for k,v in os.environ.items() if not k.startswith(('SSD_', 'PAPER_', 'DUET_', 'P1_SWEEP_'))}
    env = dict(base_env, FIGURE_STRIDE=str(job.get('figure_stride',1)), REVIEW_SHAPES=str(int(job.get('shapes',False))), SSD_TREE_SHAPE_METRICS=str(int(job.get('shapes',False))), P1_SWEEP_DRAFT_KV_FRACTION=str(job.get('draft_kv_fraction',.35)), CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)),
        PAPER_CHECKOUT=str(checkout), PAPER_CPUSET=CPUSETS[slot], PAPER_ARM=tag,
        PAPER_SEED=str(config['seed']), SSD_SEED=str(config['seed']), PAPER_EXPERIMENT_PROTOCOL=str(ROOT/'PROTOCOL.json'), PAPER_COLD_PREFIX='1', PYTHONPATH=str(checkout/'ssd'),
        SSD_CUDA_ARCH=os.environ.get('SSD_CUDA_ARCH','8.9'), SSD_HF_CACHE='/tmp', SSD_DATASET_DIR='/tmp',
        SSD_TREE_EXEC='1', SSD_TREE_EXEC_WARMUP='all', SSD_P1_TREE_EXEC_WARMUP='all',
        SSD_ASYNC_PROXY_SEND=str(job.get('async_send', 1)), SSD_PROXY_STREAM=str(job.get('proxy_stream', 0)), SSD_DUET_EXIT_TOPM_GATHER='0', SSD_PROFILE_DUET=str(int(profile)), SSD_PROFILE='0',
        SSD_PROFILE_DRAFT='0', SSD_PROFILE_DUET_DETAIL=str(int(profile)),
        SSD_PROFILE_DUET_MAX_EVENTS='24000', SSD_PROFILE_DIR=str(folder),
        SSD_FORCE_SPLIT_K1K2='1', SSD_DUET_EXIT_REPLICA=str(job.get('replica', 0)),
        SSD_SHM_NAME='duet_v14_residual_review_' +tag, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        SSD_DIST_PORT=str(35440+slot),
        TRITON_CACHE_DIR=str(cache_root/'cache/triton'),
        TORCHINDUCTOR_CACHE_DIR=str(cache_root/'cache/inductor'),
        TMPDIR=str(cache_root/'tmp'))
    wrapper = ROOT/('benchmark_probe.py' if job.get('probe') else 'benchmark_wall.py')
    cmd = [PYTHON, *([] if job.get('smoke') else ['-O']), '-B', str(wrapper),
        '--target', os.environ.get('VER15_TARGET','/home/chokwans99/models_mlsys/layerskip-llama2-7b'),
        '--draft', os.environ.get('VER15_DRAFT','/home/chokwans99/models_mlsys/llama-68m'),
        '--target-tp', str(job.get('target_tp', 1)), '--mode', job.get('mode','tree'), '--batch', '1',
        '--prompts', str(ROOT/config.pop('prompts')), '--output', str(folder/'run.json'),
        '--p1-nodes', str(config['p1_verify']), '--p2-nodes', str(config['p2_verify']),
        '--expansion', 'last']
    for name, value in config.items():
        cmd += ['--'+name.replace('_', '-'), str(value)]
    provenance = dict(tag=tag, slot=slot, gpus=gpus, job=job, argv=cmd,
        source_sha256=sources, wrapper_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest(),
        initial_gpu_state=initial,
        env={k:v for k,v in env.items() if k.startswith(('PAPER_', 'SSD_', 'OMP_', 'MKL_', 'TRITON_', 'TORCHINDUCTOR_', 'P1_SWEEP_'))
             or k in ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH','FIGURE_STRIDE')})
    (folder/'PROVENANCE.json').write_text(json.dumps(provenance, indent=2)+'\n')
    helper_hashes = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.rglob('*.py') if 'runs' not in f.parts}
    provenance['experiment_sha256'] = helper_hashes
    (folder/'PROVENANCE.json').write_text(json.dumps(provenance, indent=2)+'\n')
    started = time.time()
    with (folder/'process.log').open('w') as log, (folder/'telemetry.jsonl').open('w') as tel:
        proc = subprocess.Popen(cmd, cwd=checkout, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        (folder/'process.json').write_text(json.dumps(dict(pid=proc.pid, pgid=proc.pid))+'\n')
        while proc.poll() is None:
            sample=state(); sample['process_groups']=process_groups(sample)
            tel.write(json.dumps(sample)+'\n'); tel.flush(); time.sleep(2)
    unchanged = all(hashlib.sha256((checkout/f).read_bytes()).hexdigest()==h for f,h in sources.items())
    experiment_unchanged=all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in helper_hashes.items())
    unchanged = unchanged and experiment_unchanged
    result = dict(status='complete' if proc.returncode==0 and unchanged else 'failed',
        tag=tag, exit_code=proc.returncode, source_unchanged=unchanged, gpus=gpus,
        elapsed_seconds=time.time()-started, profile=profile)
    (folder/'status.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('plan', type=Path); p.add_argument('--launch', action='store_true')
    args=p.parse_args()
    if socket.gethostname().split('.')[0]!='orion': raise RuntimeError('local orion only')
    if args.launch:
        logpath=args.plan.with_suffix('.controller.log')
        with logpath.open('x') as log:
            proc=subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), str(args.plan.resolve())],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(pid=proc.pid, log=str(logpath))), flush=True); return
    plan=json.loads(args.plan.read_text())
    import queue
    pending=queue.Queue()
    for job in plan['jobs']:pending.put(job)
    def worker(slot):
        results=[]
        while True:
            if pending.empty():break
            snapshot=state()
            assigned={r.split(',')[1].strip() for r in snapshot['gpu'] if int(r.split(',')[0]) in PAIRS[slot]}
            busy=any(r.split(',')[0].strip() in assigned for r in snapshot['processes'] if r)
            if busy:
                time.sleep(10)
                continue
            try:job=pending.get_nowait()
            except queue.Empty:break
            result=run(job, slot); results.append(result)
            pending.task_done()
            if result['status']!='complete':break
        return results
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(worker, range(2)))
    args.plan.with_suffix('.status.json').write_text(json.dumps(dict(results=results), indent=2)+'\n')


if __name__=='__main__': main()
