"""Portable ver1.3 source identity and focused regression checks.

CPU is the default. --cuda runs the same ten suites as the recorded
52-test combined-runtime validation, plus the optional forward probe.
Each suite runs in a fresh process so test-only import flags cannot leak.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
CPU_SUITES = (
    'test_response_snapshot', 'test_deferred_tree_restore',
    'test_b_gt1_m2', 'test_b_gt1_jit_subset',
    'test_step_performance_metrics', 'test_long_step_filter',
    'test_mlsys_measurement', 'test_consecutive_timeline',
    'test_chain_latency_reporting', 'test_model_breakdowns',
    'test_plot_duet_aligned_timeline', 'test_ver10_profile_analysis',
)
CUDA_SUITES = (
    'test_deferred_tree_restore', 'test_response_snapshot',
    'test_single_verify_inputs', 'test_common_port', 'test_ver01_review',
    'test_tree_accept_port', 'test_b_gt1_m6_verify_window',
    'test_step_performance_metrics', 'test_long_step_filter',
    'test_mlsys_measurement', 'test_phase_probe',
)


def hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for directory in ('ssd/ssd', 'ssd/bench', 'ssd/tests', 'ssd/tools/duet_timeline')
            for p in sorted((ROOT / directory).rglob('*.py'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--worker', choices=sorted(set(CPU_SUITES + CUDA_SUITES)))
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.worker:
        sys.path[:0] = [str(ROOT / 'ssd'), str(ROOT / 'ssd/tests')]
        suite = unittest.defaultTestLoader.loadTestsFromName(args.worker)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        row = dict(tests=result.testsRun, errors=len(result.errors),
                   failures=len(result.failures), skipped=len(result.skipped))
        (out / (args.worker + '.json')).write_text(json.dumps(row, indent=2) + '\n')
        return int(not (result.testsRun and result.wasSuccessful() and not result.skipped))

    before = hashes()
    manifest = json.loads((Path(__file__).parent / 'RUNTIME_MANIFEST.json').read_text())
    actual = {k: v for k, v in before.items() if k.startswith('ssd/ssd/')}
    if actual != manifest['runtime_sha256']:
        raise SystemExit('Runtime differs from the tested ver1.3 source; create new validation provenance.')
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'ssd') + os.pathsep + str(ROOT / 'ssd/tests'),
               PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    # Required import paths. Real CUDA runs must supply their own architecture.
    env.setdefault('SSD_HF_CACHE', '/tmp')
    env.setdefault('SSD_DATASET_DIR', '/tmp')
    env.setdefault('SSD_CUDA_ARCH', '8.9')
    rows = []
    for name in CUDA_SUITES if args.cuda else CPU_SUITES:
        suite_env = dict(env)
        if name in ('test_b_gt1_m2', 'test_b_gt1_jit_subset'):
            suite_env.update(SSD_FORCE_SPLIT_K1K2='1', SSD_DUET_JIT_SHORT='1')
        with (out / (name + '.log')).open('w') as log:
            run = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()),
                                  '--worker', name, '--output', str(out)],
                                 cwd=ROOT, env=suite_env, stdout=log, stderr=subprocess.STDOUT)
        row_path = out / (name + '.json')
        row = json.loads(row_path.read_text()) if row_path.exists() else dict(worker_did_not_report=True)
        row.update(name=name, returncode=run.returncode)
        rows.append(row)
        print(row, flush=True)
    unchanged = hashes() == before
    passed = unchanged and all(r['returncode'] == 0 for r in rows)
    report = dict(status='complete' if passed else 'failed', cuda=args.cuda,
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  tests=sum(r.get('tests', 0) for r in rows), suites=rows,
                  source_unchanged=unchanged, runtime_matches_tested_combined=True,
                  source_sha256=before, python=sys.version)
    (out / 'CHECKS.json').write_text(json.dumps(report, indent=2) + '\n')
    return int(not passed)


if __name__ == '__main__':
    raise SystemExit(main())
