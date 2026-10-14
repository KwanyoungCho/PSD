"""Small Last/Frontier adapter checks on the merged source, not a full AL run."""
import argparse
import gzip
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from collections import defaultdict
from run_models import ROOT, hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', required=True)
    p.add_argument('--draft', required=True)
    p.add_argument('--gpus', default='4,5')
    p.add_argument('--port', type=int, default=29896)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--wait-model-status', type=Path)
    a = p.parse_args()
    if a.wait_model_status:
        deadline = time.monotonic() + 3600
        while True:
            state = json.loads(a.wait_model_status.read_text())['status']
            if state == 'complete':
                break
            if state != 'running' or time.monotonic() > deadline:
                raise RuntimeError('Native model validation failed or timed out')
            time.sleep(5)
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = hashes()
    report = dict(status='running', source_hashes=source, runs=[],
                  scope='4 paper input turns, 32 output cap, one warmup; merged adapter functional only')
    def save():
        (out / 'status.json').write_text(json.dumps(report, indent=2))
    data = ROOT / 'results/mlsys_ver02/equal_budget_20261014'
    jobs = json.loads((data / 'paper_local_45_m32_s16101.json').read_text())
    for selection in ('last', 'frontier'):
        job = next(j for j in jobs if j['name'] == selection + '_c3')
        args = list(job['args'])
        for key, value in {'--target': a.target, '--draft': a.draft,
                           '--prompts': str(data / 'SPEC_BENCH_FULL.json'),
                           '--limit': '4', '--warm-prompts': '1', '--output-tokens': '32'}.items():
            args[args.index(key)+1] = value
        profile = out / (selection + '_profile')
        profile.mkdir()
        env = dict(os.environ, **job['env'])
        env.pop('PAPER_CPUSET', None)
        env.update(PAPER_CHECKOUT=str(ROOT), PYTHONPATH=str(ROOT / 'ssd'),
                   SSD_HF_CACHE='/tmp', SSD_DATASET_DIR='/tmp', SSD_SEED=str(job['draft_seed']),
                   OMP_NUM_THREADS='2', SSD_DEBUG_TOKENS='0', SSD_PROFILE_DIR=str(profile),
                   CUDA_VISIBLE_DEVICES=a.gpus, SSD_DIST_PORT=str(a.port))
        output = out / (selection + '.json')
        cmd = [sys.executable, str(ROOT / 'ssd/bench/mlsys_equal_budget.py'),
               *args, '--output', str(output)]
        row = dict(selection=selection, command=cmd, started=time.time(), env=job['env'])
        report['runs'].append(row)
        save()
        with (out / (selection + '.log')).open('w') as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
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
            raise RuntimeError(f'{selection}: return code {code}')
        result = json.loads(output.read_text())
        if result['status'] != 'complete' or len(result['outputs']) != 4 or result['source_hashes'] != source:
            raise RuntimeError('Incomplete run or source mismatch')
        if result['equal_budget']['calibration'] or result['equal_budget']['selection'] != selection:
            raise RuntimeError('Wrong selection policy')
        calls = defaultdict(lambda: [0, 0, 0])
        with gzip.open(profile / 'equal_forest.jsonl.gz', 'rt') as f:
            for line in f:
                x = json.loads(line)
                if x['physical_rows'] != 64 or x['roots'] != 8 or x['C'] != 3:
                    raise RuntimeError('Wrong forest geometry')
                v = calls[(x['seq'], x['step'], x['phase'])]
                v[0] += x['processed']; v[1] += x['generated']; v[2] += 1
        if not calls or any(v != [64, 192, 8] for v in calls.values()):
            raise RuntimeError('Unequal active forward budget')
        row.update(turns=4, exported_phase_forests=len(calls), active_rows_per_phase=64)
        save()
    report.update(status='complete' if source == hashes() else 'failed', source_unchanged=source == hashes())
    save()
    if report['status'] != 'complete':
        raise RuntimeError('Runtime changed during validation')
    print('MERGED_ADAPTER_CHECK_COMPLETE')


if __name__ == '__main__':
    main()
