"""Native merged-runtime checks; short functional runs, not AL/TPS evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]


def hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / 'ssd/ssd').rglob('*.py'))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', required=True)
    p.add_argument('--draft', required=True)
    p.add_argument('--gpus', default='4,5')
    p.add_argument('--port', type=int, default=29894)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    before = hashes()
    report = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  status='running', source_hashes=before, runs=[])

    def save():
        tmp = out / 'status.tmp'
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(out / 'status.json')

    env = dict(os.environ, PYTHONPATH=str(ROOT / 'ssd'), SSD_HF_CACHE='/tmp',
               SSD_DATASET_DIR='/tmp', SSD_DEBUG_TOKENS='0', SSD_TREE_EXEC_WARMUP='all',
               OMP_NUM_THREADS='2', SSD_DIST_PORT=str(args.port))
    # Exercise the merged default, rather than override it to q_path.
    env.pop('SSD_TREE_EXPANSION_POLICY', None)
    for b in (1, 2):
        for mode, temp in (('ar', 0), ('tree', 0), ('tree', .7)):
            name = f'{mode}_b{b}_t{temp}'
            cmd = [sys.executable, str(ROOT / 'ssd/bench/mlsys_blackwell_smoke.py'),
                   '--target', args.target, '--draft', args.draft, '--mode', mode,
                   '--batch', str(b), '--temperature', str(temp), '--output', str(out / (name + '.json'))]
            gpus = args.gpus if mode == 'tree' else args.gpus.split(',')[0]
            row = dict(name=name, command=cmd, gpus=gpus, started=time.time())
            report['runs'].append(row)
            save()
            with (out / (name + '.log')).open('w') as log:
                proc = subprocess.Popen(cmd, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=gpus),
                                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                row['pid'] = proc.pid
                save()
                try:
                    code = proc.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    code = 124
            row.update(returncode=code, ended=time.time())
            save()
            if code:
                report['status'] = 'failed'
                save()
                raise RuntimeError(f'{name} failed ({code}); see retained log')
    parity = []
    for b in (1, 2):
        ar = json.loads((out / f'ar_b{b}_t0.json').read_text())
        tree = json.loads((out / f'tree_b{b}_t0.json').read_text())
        a = [row for group in ar['groups'] for row in group['outputs']]
        z = [row for group in tree['groups'] for row in group['outputs']]
        parity.append(dict(batch=b, prompts=len(a), exact=sum(x == y for x, y in zip(a, z)),
                           same_count=len(a) == len(z)))
    stable = before == hashes()
    report.update(greedy_parity=parity, source_unchanged=stable,
                  status='complete' if stable and all(x['same_count'] and x['prompts'] == x['exact'] for x in parity) else 'failed')
    save()
    if report['status'] != 'complete':
        raise RuntimeError('Source stability or greedy parity failed')
    print(json.dumps(dict(status=report['status'], greedy_parity=parity)))


if __name__ == '__main__':
    main()
