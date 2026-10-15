"""Run every review suite in a fresh interpreter and retain source evidence.

Zero discovered tests and unexpected skips fail this CUDA release check.
An explicitly unsupported SGL/SM120 case is recorded, never counted as passed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
SUITES = (
    'test_p2_preparation', 'test_duet_parameters_v11', 'test_args_cleanup_equiv',
    'test_common_port', 'test_output_accounting', 'test_step_performance_metrics',
    'test_cached_prefill', 'test_ver01_review', 'test_tree_accept_port',
    'test_p1_dynamic_tree', 'test_p2_tree_alloc', 'test_p2_executor_parity',
    'test_tree_verify_planless', 'test_b_gt1_m6_verify_window',
    'test_model_pair_contract', 'test_batched_tree_executor', 'test_batch_tree_serving',
    'test_tree_fused_math', 'test_tree_extension_review', 'test_mlsys_measurement',
    'test_flat_tree_review', 'test_flat_tree_runtime', 'test_tree_live_siblings',
    'test_tree_warmup_bounds', 'test_tree_proxy_defaults',
    'test_runtime_manifest', 'test_fi_attn_backend', 'test_blackwell_review', 'test_blackwell_attention_shapes', 'test_job_status_review',
    'test_request_reporting', 'test_release_review', 'test_tree_payload_prepare', 'test_local_research_policy',
    'test_equal_budget_tree', 'test_equal_dataset',
    'test_tree_expansion_modes',
    'test_ver10_root_settings', 'test_ver10_profile_analysis',
    'test_ver10_checkpoint_rope',
)


def source_hashes():
    return {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
            for folder in ('ssd/ssd', 'ssd/tests', 'ssd/bench')
            for f in sorted((ROOT / folder).rglob('*.py'))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--suite', choices=SUITES, action='append')
    p.add_argument('--worker', choices=SUITES, help=argparse.SUPPRESS)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / 'ssd'))
    os.environ.setdefault('SSD_HF_CACHE', '/tmp')
    os.environ.setdefault('SSD_DATASET_DIR', '/tmp')
    if args.worker:
        suite = unittest.defaultTestLoader.discover(str(ROOT / 'ssd/tests'),
                                                    pattern=args.worker + '.py')
        with (args.output_dir / (args.worker + '.suite.log')).open('w') as log:
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        allowed_skips = [(test.id(), reason) for test,reason in result.skipped
            if test.id().endswith('TestCachedPrefill.test_sgl')
            and reason.startswith('SGL Hopper kernels unsupported on SM120')]
        unexpected_skips = len(result.skipped)-len(allowed_skips)
        row = dict(name=args.worker, tests=result.testsRun, failures=len(result.failures),
                   errors=len(result.errors), skipped=len(result.skipped),
                   unsupported=allowed_skips, unexpected_skips=unexpected_skips)
        (args.output_dir / (args.worker + '.result.json')).write_text(json.dumps(row, indent=2))
        raise SystemExit(0 if result.testsRun and result.wasSuccessful() and not unexpected_skips else 1)

    before = source_hashes()
    git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    git_branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    source_status = subprocess.check_output(['git', 'status', '--porcelain', '--',
                                            'ssd/ssd', 'ssd/tests', 'ssd/bench'], cwd=ROOT, text=True)
    started = time.time()
    rows = []
    for name in args.suite or SUITES:
        cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', name,
               '--output-dir', str(args.output_dir.resolve())]
        with (args.output_dir / (name + '.process.log')).open('w') as log:
            run = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        result = args.output_dir / (name + '.result.json')
        row = json.loads(result.read_text()) if result.exists() else dict(name=name, tests=0)
        row['returncode'] = run.returncode
        rows.append(row)
        print(row, flush=True)
        (args.output_dir / 'CHECKS.json').write_text(json.dumps(rows, indent=2))
    stable = before == source_hashes()
    provenance = dict(started=started, ended=time.time(), source_unchanged=stable,
                      git_commit=git_commit, git_branch=git_branch, source_status=source_status,
                      source_hashes=before, command=sys.argv,
                      cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'), python=sys.executable,
                      runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name',
                                                   '--format=csv'], text=True))
    (args.output_dir / 'CHECKS_PROVENANCE.json').write_text(json.dumps(provenance, indent=2))
    raise SystemExit(0 if stable and all(r['returncode'] == 0 for r in rows) else 1)


if __name__ == '__main__':
    main()
