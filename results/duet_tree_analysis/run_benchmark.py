"""Two-seed untraced confirmation; fixed model, score and existing holdout bank."""
import json
import os
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def main():
    plan=json.loads((HERE/'rollout_plan.json').read_text())
    jobs=[(921,'q_path'),(921,'phase_sibling_q_bin'),(922,'phase_sibling_q_bin'),(922,'q_path')]
    plan=dict(plan,jobs=jobs,purpose='Check instrumentation concern with untraced two-seed AL/TPS.',
        scope='Same finite validation bank; confirmation of fixed policy, not new unseen prompts.',
        primary_timing='sum emitted output tokens / sum per-request generation wall time; includes prefill, excludes initialization and excluded warmup')
    path=HERE/'benchmark_plan.json'
    if path.exists():raise ValueError('Benchmark plan exists')
    path.write_text(json.dumps(plan,indent=2))
    base=json.loads((HERE/'probe/command.json').read_text())
    for index,(seed,policy) in enumerate(jobs):
        dest=HERE/f'bench_{seed}_{policy}';dest.mkdir()
        env=os.environ.copy();env.update(base['env'])
        for k in list(env):
            if k.startswith(('SSD_DUET_PROBE','SSD_TREE_TOPO_TRACE','SSD_TREE_CALIB_TRACE','SSD_TREE_NODE_AUDIT','DUET_TREE_AUDIT')):env.pop(k)
        env.update(SSD_DATASET_DIR=str(HERE/'rollout_datasets'),DUET_TREE_SCORE_MODE=policy,
            DUET_TREE_SCORE_CALIBRATION=str(HERE/'calibration_frozen.json'),DUET_TREE_BENCH_OUT=str(dest/'result.json'),
            SSD_DIST_PORT=str(24951+index),SSD_PROFILE='0',SSD_PROFILE_DUET='0')
        cmd=base['command'][:];cmd[2]=str(HERE/'bench_score.py')
        cmd[cmd.index('--numseqs')+1]='16';cmd[cmd.index('--seed')+1]=str(seed)
        (dest/'command.json').write_text(json.dumps(dict(command=cmd,env={k:v for k,v in env.items() if k.startswith(('SSD_','DUET_','CUDA_VISIBLE'))}),indent=2))
        start=time.time()
        with (dest/'run.log').open('w') as f:
            proc=subprocess.Popen(cmd,cwd=ROOT/'ssd',env=env,stdout=f,stderr=subprocess.STDOUT)
            (dest/'process.json').write_text(json.dumps(dict(pid=proc.pid,started=start)))
            code=proc.wait()
        (dest/'completion.json').write_text(json.dumps(dict(exit_code=code,wall_s=time.time()-start),indent=2))
        if code:raise RuntimeError(f'{dest.name} failed')
        print(dest.name,'complete',flush=True)


if __name__=='__main__':main()
