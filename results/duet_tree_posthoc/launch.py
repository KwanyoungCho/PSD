"""Separate, immutable diagnostic runs; do not alter original AL artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FULL = HERE.parent / 'duet_tree_al_full'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    for i, policy in enumerate(['q_path', 'phase_sibling_q_bin']):
        if args.smoke and i: continue
        name = ('smoke_' if args.smoke else '') + policy
        dest = HERE / 'runs' / name
        if dest.exists(): raise ValueError(f'Refuse overwrite: {dest}')
        usage = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
        if any(int(line.split(',')[1]) > 1000 for line in usage.splitlines() if int(line.split(',')[0]) in range(3, 8)):
            raise RuntimeError('Assigned GPUs busy')
        dest.mkdir(parents=True)
        base = json.loads((FULL / 'runs' / f's1_{policy}' / 'command.json').read_text())
        env = os.environ.copy(); env.update(base['env'])
        env.update(DUET_FULL_OUT=str(dest), SSD_DIST_PORT=str(25320+i+10*args.smoke),
            DUET_FULL_SMOKE=str(int(args.smoke)))
        cmd = base['command'][:]; cmd[2] = str(HERE / 'observe.py')
        sources = list(HERE.glob('*.py')) + [FULL / 'run_full.py', FULL / 'runtime.py',
            HERE.parent / 'duet_tree_analysis/core.py', HERE.parent / 'duet_tree_analysis/score_hook.py']
        (dest / 'command.json').write_text(json.dumps(dict(command=cmd,
            env={k:v for k,v in env.items() if k.startswith(('SSD_', 'DUET_', 'CUDA_VISIBLE'))},
            sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}), indent=2))
        start = time.time()
        with (dest / 'run.log').open('x') as log:
            proc = subprocess.Popen(cmd, cwd=ROOT / 'ssd', env=env, stdout=log, stderr=subprocess.STDOUT)
            (dest / 'process.json').write_text(json.dumps(dict(pid=proc.pid, started=start)))
            code = proc.wait()
        (dest / 'completion.json').write_text(json.dumps(dict(exit_code=code, wall_s=time.time()-start)))
        if code: raise RuntimeError(f'Failed: {dest / "run.log"}')
        print('completed', name, flush=True)


if __name__ == '__main__':
    main()
