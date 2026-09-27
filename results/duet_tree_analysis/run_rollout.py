"""Prespecified two-policy G=M actual-tree construction diagnostic.

Run after analyze_probe.py; no performance claim because observers synchronize.
"""
import json
import os
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]


def main():
    policies=['q_path','phase_sibling_q_bin']
    plan=json.loads((HERE/'probe_plan.json').read_text())
    mapping=plan['mapping']; records=(HERE/'datasets/alpaca/alpaca_data_10000.jsonl').read_text().splitlines()
    validation=[(m,row) for m,row in zip(mapping,records) if m['split']=='validation']
    plan=dict(plan,mapping=[m for m,_ in validation],purpose='Actual safe expansion-score ablation; diagnostic synchronization; not TPS.',
              policies=policies,primary_estimand='Per-prompt accepted descendant tokens per served tree hit.',
              fixed_before_validation_analysis=True,calibration='calibration_frozen.json')
    plan_path=HERE/'rollout_plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text())!=plan:raise ValueError('Different existing rollout plan')
    else:plan_path.write_text(json.dumps(plan,indent=2))
    data=HERE/'rollout_datasets/alpaca/alpaca_data_10000.jsonl';data.parent.mkdir(parents=True,exist_ok=True)
    data.write_text('\n'.join(row for _,row in validation)+'\n')
    # --prepare records policy choice before the held-out diagnostics are opened.
    import sys
    if '--prepare' in sys.argv:return
    if not (HERE/'calibration_frozen.json').exists():raise ValueError('Frozen calibration missing')
    base=json.loads((HERE/'probe/command.json').read_text())
    for index,policy in enumerate(policies):
        dest=HERE/('rollout_'+policy)
        if dest.exists():raise ValueError('Refuse to overwrite rollout artifacts')
        dest.mkdir()
        env=os.environ.copy();env.update(base['env'])
        env.update(SSD_DATASET_DIR=str(data.parents[1]),DUET_TREE_AUDIT_OUT=str(dest),
            DUET_TREE_AUDIT_PLAN=str(plan_path),DUET_TREE_SCORE_MODE=policy,
            DUET_TREE_SCORE_CALIBRATION=str(HERE/'calibration_frozen.json'),SSD_DIST_PORT=str(24931+index))
        cmd=base['command'][:];cmd[cmd.index('--numseqs')+1]=str(len(validation))
        (dest/'command.json').write_text(json.dumps(dict(command=cmd,env={k:v for k,v in env.items() if k.startswith(('SSD_','DUET_','CUDA_VISIBLE'))}),indent=2))
        start=time.time()
        with (dest/'run.log').open('w') as f:
            proc=subprocess.Popen(cmd,cwd=ROOT/'ssd',env=env,stdout=f,stderr=subprocess.STDOUT)
            (dest/'process.json').write_text(json.dumps(dict(pid=proc.pid,started=start)))
            code=proc.wait()
        (dest/'completion.json').write_text(json.dumps(dict(exit_code=code,wall_s=time.time()-start),indent=2))
        if code:raise RuntimeError(f'Rollout {policy} failed')
        print(policy,'complete',flush=True)


if __name__=='__main__':main()
