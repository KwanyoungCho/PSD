"""Finite, matched full-Spec-Bench comparison; native runtime is immutable."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[1]
OUT = ROOT / 'results/mlsys_ver17/best3_20261023'
PYTHON = '/home/chokwans99/PSD/ssd/.venv/bin/python'
PROTOCOL = json.loads((OUT / 'PROTOCOL.json').read_text())
INPUTS = json.loads((CODE / 'INPUTS_FULL.json').read_text())
CPU = {2: '0-7,32-39', 0: '16-23,48-55', 10: '24-31,56-63'}
LOCK = threading.RLock()
STOP = threading.Event()
STATE = dict(status='running', active={}, done={}, errors=[], started=time.time(),
             pid=os.getpid(), planned_runs=6, protocol=str(OUT / 'PROTOCOL.json'))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def progress():
    with LOCK:
        dump(OUT / 'STATUS.json', dict(STATE, updated=time.time()))


def runtime_hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / 'ssd/ssd').rglob('*.py'))}


def measurement_hashes():
    paths = list((CODE / 'adapter').rglob('*.py'))
    paths += [CODE / n for n in ('benchmark_wall.py', 'capacity_adapter.py', 'observe_runtime.py')]
    return {str(p.relative_to(CODE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def telemetry():
    def query(fields):
        return subprocess.check_output(['nvidia-smi', *fields], text=True, timeout=15).strip().splitlines()
    gpu = query(['--query-gpu=index,uuid,memory.used,utilization.gpu,clocks.sm,temperature.gpu',
                 '--format=csv,noheader,nounits'])
    processes = []
    for line in query(['--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits']):
        if not line:
            continue
        uuid, pid, memory = [s.strip() for s in line.split(',')]
        try:
            pgid = os.getpgid(int(pid))
        except ProcessLookupError:
            continue
        processes.append(dict(uuid=uuid, pid=int(pid), pgid=pgid, memory=memory))
    return dict(time_ns=time.time_ns(), gpus=gpu, processes=processes)


def validate(folder, arm, seed, before, helpers):
    sys.path.insert(0, str(CODE / 'adapter/ssd/bench'))
    from mlsys_request_reporting import classify
    from question_mean import summarize
    assert before == runtime_hashes(Path(PROTOCOL['arms'][arm]['root'])), 'Runtime changed during measurement'
    assert helpers == measurement_hashes(), 'Measurement helper changed during measurement'
    run = json.loads((folder / 'run.json').read_text())
    assert run['status'] == 'complete'
    assert run['paper_commit'] == PROTOCOL['arms'][arm]['commit']
    groups = run['warm_groups'] + run['groups']
    assert len(run['warm_groups']) == 3 and len(run['groups']) == 557
    assert len(groups) == 560 and len(run['prompt_ids']) == 560
    assert run['args']['seed'] == seed and run['args']['input_cap'] == 0
    assert run['args']['output_tokens'] == 1024 and run['args']['temperature'] == .7
    kw = run['kwargs']
    assert kw['dtype'] == 'bfloat16' and kw['max_num_seqs'] == 1
    assert kw['duet_p1_tree_policy'] == kw['duet_p2_tree_policy'] == 'on'
    assert kw['duet_tree_expansion'] == 'last'
    for k in ('k1', 'k2'):
        assert run['args'][k] == PROTOCOL['arms'][arm][k]
    counts = {}
    guards = []
    for item, prompt, group in zip(INPUTS, run['prompt_ids'], groups):
        assert prompt == item['input_ids'], item['uid']
        assert group['n'] == len(group['request_records']) == 1
        rec = group['request_records'][0]
        assert rec['paper_uid'] == item['uid'] and str(rec['question_id']) == str(item['question_id'])
        assert rec['input_tokens'] == len(prompt)
        assert rec['target_temperature'] == rec['draft_temperature'] == .7
        assert rec['ignore_eos'] is False
        assert rec['effective_output_cap'] == min(1024, 2048-len(prompt))
        assert rec['completion_tokens'] <= rec['effective_output_cap']
        assert rec['output_ids'] == group['outputs'][0]
        kind = classify(rec)
        counts[kind] = counts.get(kind, 0) + 1
        if kind == 'context_guard_no_decode':
            guards.append(item['uid'])
        assert rec['completion_tokens'] == sum(s['emitted'] for s in group['steps'])
        for step in group['steps']:
            assert step['batch'] == 1 and len(step['events']) == 1
            e = step['events'][0]
            assert e['source'] in (0, 1, 2)
            assert bool(e['cache_hit']) == (e['source'] != 0)
    for role in ('target', 'draft'):
        actual = json.loads((folder / f'ACTUAL_{role}.json').read_text())
        assert all(actual[k] == 'torch.bfloat16' for k in ('parameter', 'kv_cache', 'hf_dtype'))
        assert not actual['enforce_eager']
        if role == 'target':
            assert actual['fused_tree_graphs'] > 0 and actual['fused_chain_graphs']
    assert not (folder / 'UNEXPECTED_PREEMPTION.json').exists()
    summary = summarize(run['groups'])
    raw = summarize(run['groups'], filtered=False)
    dump(folder / 'QUESTION_MEAN.json', summary)
    dump(folder / 'QUESTION_MEAN_RAW.json', raw)
    evidence = dict(status='passed', turns=560, warmup_turns=3, measured_turns=557,
                    classifications=counts, context_guard_no_decode_uids=guards,
                    both_tree_phases_enabled=True, expansion='last',
                    source_unchanged=True, input_order_and_ids_identical=True,
                    dtype='bfloat16', graph_path=True,
                    question_mean=summarize(run['groups'])['sources'],
                    long_excluded=sum(bool(s['long_step_outlier']) for s in run['steps']),
                    raw_steps=len(run['steps']),
                    run_sha256=hashlib.sha256((folder/'run.json').read_bytes()).hexdigest())
    dump(folder / 'CHECKS.json', evidence)
    return evidence


def stop_process(proc):
    if proc.poll() is not None:
        return proc.returncode
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        return proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        return proc.wait()


def job(gpus, seed, arm):
    tag = f'{arm}_s{seed}'
    runtime = Path(PROTOCOL['arms'][arm]['root'])
    folder = OUT / 'runs' / tag
    if (folder / 'STATUS.json').exists():
        old = json.loads((folder / 'STATUS.json').read_text())
        if old['status'] == 'complete':
            with LOCK:
                STATE['done'][tag] = old
            progress()
            return
        raise RuntimeError(f'Preserved incomplete attempt requires review: {folder}')
    snap = telemetry()
    uuids = {line.split(',')[1].strip() for line in snap['gpus'] if int(line.split(',')[0]) in gpus}
    while any(p['uuid'] in uuids for p in snap['processes']):
        with LOCK:
            STATE['active'][tag] = dict(status='waiting_for_free_pair', gpus=gpus)
        progress()
        if STOP.wait(15) or (OUT/'STOP').exists():
            raise RuntimeError('Stopped before launch')
        snap = telemetry()
    folder.mkdir(parents=True, exist_ok=False)
    before, helpers = runtime_hashes(runtime), measurement_hashes()
    env = {k:v for k,v in os.environ.items() if not k.startswith(('SSD_', 'PAPER_', 'DUET_', 'NCCL_', 'CUDA_')) and k != 'PYTHONPATH'}
    env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)), PYTHONPATH=str(runtime/'ssd'),
        PAPER_CHECKOUT=str(runtime), PAPER_CPUSET=CPU[seed], PAPER_ARM=tag,
        PAPER_SEED=str(seed), SSD_SEED=str(seed), DUET_MEASURE_DTYPE='bfloat16',
        PAPER_COLD_PREFIX='1', PAPER_EXPERIMENT_PROTOCOL=str(OUT/'PROTOCOL.json'),
        SSD_CUDA_ARCH='8.9', TORCH_CUDA_ARCH_LIST='8.9', SSD_HF_CACHE='/tmp', SSD_DATASET_DIR='/tmp',
        SSD_TREE_EXEC='1', SSD_TREE_EXEC_WARMUP='all', SSD_P1_TREE_EXEC_WARMUP='all',
        SSD_ASYNC_PROXY_SEND='1', SSD_PROXY_STREAM='0', SSD_BATCH_TREE_PROXY_STREAM='0',
        SSD_DUET_EXIT_REPLICA='0', SSD_DUET_EXIT_TOPM_GATHER='0', SSD_VERIFY_GRAPH_LOGITS='1',
        NCCL_GRAPH_MIXING_SUPPORT='0', NCCL_P2P_DISABLE='1', SSD_TREE_VERIFY_BUCKETS='',
        SSD_PROFILE_DUET='0', SSD_PROFILE_DUET_DETAIL='0', SSD_PROFILE='0', SSD_PROFILE_DRAFT='0',
        SSD_PROFILE_DIR=str(folder), SSD_FORCE_SPLIT_K1K2='1', SSD_SHM_NAME='v17best3_'+tag,
        SSD_DIST_PORT=str(47000+seed%1000+gpus[0]*100), OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', P1_SWEEP_DRAFT_KV_FRACTION='.35',
        PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        TRITON_CACHE_DIR='/data/chokwans99/ver16_diag_20261022/triton',
        TORCHINDUCTOR_CACHE_DIR='/data/chokwans99/ver16_diag_20261022/inductor')
    k = PROTOCOL['arms'][arm]
    cmd = [PYTHON,'-O','-B',str(CODE/'benchmark_wall.py'),
        '--target','/home/chokwans99/models_mlsys/layerskip-llama2-7b',
        '--draft','/home/chokwans99/models_mlsys/llama-68m', '--target-tp','1','--mode','tree','--batch','1',
        '--prompts',str(CODE/'INPUTS_FULL.json'),'--output',str(folder/'run.json'),
        '--p1-verify','23','--p1-nodes','23','--p2-verify','15','--p2-nodes','15',
        '--k1',str(k['k1']),'--k2',str(k['k2']),'--roots','4','--siblings','3',
        '--ds-width','56','--ps-width','28','--ps-roots','24','--expansion','last',
        '--exit-layer','21','--passes','1','--output-tokens','1024','--input-cap','0',
        '--warm-prompts','3','--seed',str(seed),'--temperature','.7']
    dump(folder/'PROVENANCE.json',dict(command=cmd, source_hashes=before, measurement_hashes=helpers,
        env={k:v for k,v in env.items() if k.startswith(('SSD_','PAPER_','DUET_','CUDA_','NCCL_'))},
        gpus=gpus, seed=seed, arm=arm, input_sha256=PROTOCOL['input_sha256']))
    start=time.time()
    foreign=[]
    aborted=None
    with (folder/'process.log').open('w') as log, (folder/'telemetry.jsonl').open('w') as tel:
        proc=subprocess.Popen(cmd, cwd=runtime, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        with LOCK:
            STATE['active'][tag]=dict(status='running',pid=proc.pid,gpus=gpus,started=start,folder=str(folder))
        progress()
        print('START',tag,gpus,proc.pid,flush=True)
        while proc.poll() is None:
            snap=telemetry()
            tel.write(json.dumps(snap)+'\n');tel.flush()
            foreign=[p for p in snap['processes'] if p['uuid'] in uuids and p['pgid']!=proc.pid]
            if foreign or STOP.is_set() or (OUT/'STOP').exists():
                aborted='foreign GPU process' if foreign else 'user/controller stop'
                stop_process(proc)
                break
            STOP.wait(5)
        code=proc.wait()
    error=None
    try:
        assert code==0, ('process exit',code)
        assert aborted is None, aborted
        evidence=validate(folder,arm,seed,before,helpers)
    except Exception as exc:
        error=repr(exc)
        evidence=None
    row=dict(status='complete' if error is None else 'failed', tag=tag, arm=arm, seed=seed,
        gpus=gpus, elapsed=time.time()-start, exit_code=code, error=error, foreign=foreign,
        folder=str(folder), checks=str(folder/'CHECKS.json') if evidence else None)
    dump(folder/'STATUS.json',row)
    with LOCK:
        STATE['active'].pop(tag,None)
        STATE['done'][tag]=row
    progress()
    print('DONE',json.dumps(row),flush=True)
    if error:
        raise RuntimeError(row)


def lane(spec):
    handles=[]
    for gpu in sorted(spec['gpus']):
        handle=(OUT/f'gpu{gpu}.lock').open('a')
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        handles.append(handle)
    for arm in spec['order']:
        if STOP.is_set() or (OUT/'STOP').exists():
            break
        job(spec['gpus'],spec['seed'],arm)


def main():
    assert socket.gethostname().split('.')[0]=='orion'
    handle=(OUT/'controller.lock').open('a')
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,lambda *_:STOP.set())
    signal.signal(signal.SIGINT,lambda *_:STOP.set())
    progress()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(lane,spec) for spec in PROTOCOL['lanes']]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                with LOCK:
                    STATE['errors'].append(repr(exc))
                progress()
    STATE['status']='failed' if STATE['errors'] else ('stopped' if STOP.is_set() or (OUT/'STOP').exists() else 'measurements_complete')
    progress()
    if STATE['status']=='measurements_complete':
        command=[sys.executable,'-B',str(CODE/'report.py')]
        if (CODE/'report.py').exists():
            subprocess.run(command,check=True,cwd=ROOT)
            STATE['status']='complete'
            progress()


if __name__=='__main__':
    main()
