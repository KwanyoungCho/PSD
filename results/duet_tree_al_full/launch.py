"""Run the frozen full-data jobs. Completed jobs are checksum-validated, never overwritten."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--only',type=int)
    args=ap.parse_args()
    plan=json.loads((HERE/'plan.json').read_text())
    if hashlib.sha256(Path(plan['calibration_path']).read_bytes()).hexdigest()!=plan['calibration_sha256']:raise ValueError('Calibration changed')
    base=json.loads((ROOT/'results/duet_tree_analysis/probe/command.json').read_text())
    jobs=[dict(seed=1,policy='phase_sibling_q_bin')] if args.smoke else plan['jobs']
    for i,job in enumerate(jobs):
        if args.only is not None and i!=args.only:continue
        name=('smoke_' if args.smoke else '')+f"s{job['seed']}_{job['policy']}"
        dest=HERE/'runs'/name
        if dest.exists():
            cp=dest/'completion.json';vp=dest/'validated.json'
            if cp.exists() and json.loads(cp.read_text())['exit_code']==0 and vp.exists():
                v=json.loads(vp.read_text())
                if hashlib.sha256((dest/'records.jsonl').read_bytes()).hexdigest()!=v['records_sha256']:raise ValueError('Completed data changed')
                if not args.smoke and (v['questions']!=480 or v['turns']!=560):raise ValueError('Incomplete completed job')
                print('skip validated',name,flush=True);continue
            raise ValueError(f'Incomplete prior attempt preserved at {dest}; investigate before retry')
        # Do not collide with unrelated jobs on the same devices.
        used=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        usage={int(a):int(b) for a,b in (s.split(',') for s in used.splitlines())}
        if any(usage[g]>1000 for g in plan['runtime']['gpus']):raise RuntimeError('Assigned GPUs are busy')
        dest.mkdir(parents=True)
        env=os.environ.copy();env.update(base['env'])
        for k in list(env):
            if k.startswith(('SSD_DUET_PROBE','SSD_TREE_TOPO_TRACE','SSD_TREE_CALIB_TRACE','SSD_TREE_NODE_AUDIT','DUET_TREE_AUDIT','SSD_E0_','SSD_CONF_')):env.pop(k)
        env.update(SSD_DATASET_DIR=str(HERE/'dataset'),DUET_TREE_SCORE_MODE=job['policy'],
            DUET_TREE_SCORE_CALIBRATION=plan['calibration_path'],DUET_FULL_OUT=str(dest),
            DUET_FULL_DRAFT_KV_FRACTION=str(plan['runtime']['draft_kv_memory_fraction']),
            SSD_TREE_EXEC_WARMUP='all',SSD_DIST_PORT=str(25100+(99 if args.smoke else i)),
            DUET_FULL_SMOKE='1' if args.smoke else '0',SSD_PROFILE='0',SSD_PROFILE_DUET='0')
        cmd=base['command'][:];cmd[2]=str(HERE/'run_full.py')
        for flag,value in [('--numseqs',1),('--seed',job['seed']),('--output_len',plan['max_new_tokens']),('--max_model_len',plan['max_model_len'])]:
            cmd[cmd.index(flag)+1]=str(value)
        if plan['extend_draft_rope']:cmd.append('--extend_draft_rope')
        sources=list(HERE.glob('*.py'))+[ROOT/'results/duet_tree_analysis/score_hook.py']
        (dest/'command.json').write_text(json.dumps(dict(command=cmd,env={k:v for k,v in env.items() if k.startswith(('SSD_','DUET_','CUDA_VISIBLE'))},
            source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2))
        start=time.time()
        with (dest/'run.log').open('w') as f:
            proc=subprocess.Popen(cmd,cwd=ROOT/'ssd',env=env,stdout=f,stderr=subprocess.STDOUT)
            (dest/'process.json').write_text(json.dumps(dict(pid=proc.pid,started=start)))
            code=proc.wait()
        (dest/'completion.json').write_text(json.dumps(dict(exit_code=code,wall_s=time.time()-start),indent=2))
        if code:raise RuntimeError('Failed '+name+'; inspect preserved log')
        print('completed',name,flush=True)
        if not args.smoke:
            subprocess.run([str(ROOT/'ssd/.venv/bin/python'),str(HERE/'analyze.py'),'--partial'],check=True)


if __name__=='__main__':main()
