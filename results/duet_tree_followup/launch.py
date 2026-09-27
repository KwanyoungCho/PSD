"""Wait for idle GPUs, correctness-gate, then run all frozen online jobs.

No occupied GPU is reclaimed. STOP pauses between jobs or while waiting; failed
attempts are preserved and stop the queue. File changes after queue submission
also stop execution, so a stale background process cannot run edited policies.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from policy_hook import POLICIES

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PYTHON=ROOT/'ssd/.venv/bin/python'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def status(state,**kwargs):
    out=dict(state=state,pid=os.getpid(),updated=time.time(),**kwargs)
    tmp=HERE/'queue_status.tmp';tmp.write_text(json.dumps(out,indent=2));tmp.replace(HERE/'queue_status.json')


def check_frozen():
    manifest=json.loads((HERE/'execution_manifest.json').read_text())
    for path,digest in manifest['files'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Queued source changed: '+path)
    for path,info in manifest.get('external_weight_metadata',{}).items():
        stat=Path(path).stat()
        if stat.st_size!=info['size'] or stat.st_mtime_ns!=info['mtime_ns']:
            raise ValueError('Model artifact changed after submission: '+path)


def wait_devices(chosen):
    while True:
        if (HERE/'STOP').exists():raise RuntimeError('STOP requested; no new jobs started')
        check_frozen()
        data=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
        devices=[s.split(',') for s in data.splitlines()]
        usage={int(a):int(c) for a,_,c in devices}
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
        occupied={s.strip() for s in processes.splitlines() if s.strip().startswith('GPU-')}
        free=[int(a) for a,b,c in devices if int(c)<1000 and b.strip() not in occupied]
        if chosen is None and len(free)>=5:
            preferred=[3,4,5,6,7]
            chosen=preferred if all(g in free for g in preferred) else free[:5]
            (HERE/'assigned_gpus.json').write_text(json.dumps(chosen))
        if chosen is not None and all(g in free for g in chosen):return chosen
        status('waiting_for_gpus',required=5,assigned=chosen,free=free,memory_mib=usage,
               availability_rule='No compute process and memory <1000 MiB on every selected GPU')
        time.sleep(30)


def execute(name,job,gpus,*,smoke=False,trace=False):
    dest=HERE/'runs'/name
    if dest.exists():
        if (dest/'completion.json').exists() and (dest/'validated.json').exists():
            cp=json.loads((dest/'completion.json').read_text());vp=json.loads((dest/'validated.json').read_text())
            if cp['exit_code']==0 and sha(dest/'records.jsonl')==vp['records_sha256']:
                return
        raise ValueError('Prior incomplete attempt preserved at '+str(dest))
    base=json.loads((HERE.parent/'duet_tree_al_full/runs/s1_q_path/command.json').read_text())
    env=os.environ.copy();env.update(base['env'])
    for k in list(env):
        if k.startswith(('SSD_DUET_PROBE','SSD_TREE_TOPO_TRACE','SSD_TREE_CALIB_TRACE','SSD_TREE_NODE_AUDIT','DUET_TREE_AUDIT','SSD_E0_','SSD_CONF_')):env.pop(k)
    dest.mkdir(parents=True);env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),
        DUET_TREE_POLICY=job['policy'],DUET_TREE_SCORE_MODE=POLICIES[job['policy']]['score'],DUET_FULL_OUT=str(dest),
        DUET_FULL_SMOKE=str(int(smoke)),SSD_DIST_PORT=str(25700+(job['seed']%100)+list(POLICIES).index(job['policy'])*100),
        DUET_FOLLOWUP_TRACE=str(int(trace)),DUET_FOLLOWUP_PARITY=str(int(smoke)),
        SSD_TREE_EXEC_WARMUP='all',SSD_PROFILE='0',SSD_PROFILE_DUET='0')
    if trace:env['SSD_TREE_NODE_AUDIT']=str(dest/'node_audit')
    cmd=base['command'][:];cmd[2]=str(HERE/'run_full.py');cmd[cmd.index('--seed')+1]=str(job['seed'])
    (dest/'command.json').write_text(json.dumps(dict(command=cmd,
        env={k:v for k,v in env.items() if k.startswith(('DUET_','SSD_','CUDA_VISIBLE'))},
        execution_manifest_sha256=sha(HERE/'execution_manifest.json')),indent=2))
    start=time.time()
    with (dest/'run.log').open('x') as stream:
        proc=subprocess.Popen(cmd,cwd=ROOT/'ssd',env=env,stdout=stream,stderr=subprocess.STDOUT)
        (dest/'process.json').write_text(json.dumps(dict(pid=proc.pid,started=start)))
        while proc.poll() is None:
            status('running',job=name,worker_pid=proc.pid,gpus=gpus,wall_s=time.time()-start)
            time.sleep(15)
    code=proc.returncode
    (dest/'completion.json').write_text(json.dumps(dict(exit_code=code,wall_s=time.time()-start),indent=2))
    if code:raise RuntimeError('Experiment failed; preserved log: '+str(dest/'run.log'))
    if smoke:
        parity=[json.loads(line) for p in dest.glob('parity.*.jsonl') for line in p.open()]
        if {r['phase'] for r in parity}!={'p1','p2'}:raise ValueError('Both-phase real-model graph/eager checks not observed')
        if not all(r['passed'] for r in parity):raise ValueError('Parity gate failed')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wait',action='store_true',required=True);args=ap.parse_args()
    lock=(HERE/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((HERE/'plan.json').read_text())
    try:
        check_frozen()
        for name in ('math_checks.json','frontier_checks.json','cuda_checks.json','analysis_checks.json'):
            if not json.loads((HERE/name).read_text())['passed']:raise ValueError('Failed prerequisite '+name)
        for policy in plan['policies']:
            if not json.loads((HERE/f'executor_{policy}.json').read_text())['passed']:raise ValueError('Failed executor fixture '+policy)
        gpus=json.loads((HERE/'assigned_gpus.json').read_text()) if (HERE/'assigned_gpus.json').exists() else None
        for policy in plan['policies']:
            gpus=wait_devices(gpus)
            execute('smoke_'+policy,dict(seed=1,policy=policy),gpus,smoke=True,trace=True)
        for job in plan['jobs']:
            gpus=wait_devices(gpus);execute(f's{job["seed"]}_{job["policy"]}',job,gpus)
        for policy in ('reach','reach_gain_frontier'):
            gpus=wait_devices(gpus);execute('trace_'+policy,dict(seed=1,policy=policy),gpus,trace=True)
        for script in ('validate_online.py','analyze_online.py','analyze_trace.py','make_report.py'):
            status('analyzing',script=script)
            subprocess.run([str(PYTHON),str(HERE/script)],cwd=ROOT,check=True)
        status('complete',online_jobs=len(plan['jobs']),trace_jobs=2,smoke_jobs=len(plan['policies']))
    except Exception as exc:
        status('stopped',error=str(exc));raise


if __name__=='__main__':main()
