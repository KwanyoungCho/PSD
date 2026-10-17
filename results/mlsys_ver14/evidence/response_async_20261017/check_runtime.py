"""Focused checks in fresh processes; run before timed comparisons."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path('/data2/chokwans99/response-async-20261017')
BASE=['test_deferred_tree_restore','test_response_snapshot','test_single_verify_inputs',
      'test_common_port','test_ver01_review','test_tree_accept_port',
      'test_b_gt1_m6_verify_window','test_step_performance_metrics',
      'test_long_step_filter','test_mlsys_measurement','test_b_gt1_m2','test_b_gt1_jit_subset']


def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['early','prepack'],required=True)
    p.add_argument('--worker');args=p.parse_args();checkout=ROOT/args.variant
    out=ROOT/('checks_'+args.variant);out.mkdir(exist_ok=True)
    sys.path[:0]=[str(checkout/'ssd'),str(checkout/'ssd/tests')]
    if args.worker:
        if args.worker == 'fixtures':
            os.environ['SSD_FORCE_SPLIT_K1K2']='1'
            os.environ['SSD_DUET_JIT_SHORT']='1'
            names=BASE[-2:]
        else:
            names=BASE[:-2]+(['test_response_send'] if args.variant=='early' else ['test_response_topology'])
        suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in names)
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        (out/(args.worker+'.json')).write_text(json.dumps(dict(tests=result.testsRun,
            errors=len(result.errors),failures=len(result.failures),skipped=len(result.skipped)))+'\n')
        return int(not result.wasSuccessful() or bool(result.skipped))
    def source():
        return {str(p.relative_to(checkout)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (checkout/'ssd/ssd').rglob('*.py')}
    before=source();rows=[]
    for name in ('common','fixtures'):
        env=dict(os.environ,PYTHONPATH=str(checkout/'ssd')+':'+str(checkout/'ssd/tests'),
                 PYTHONDONTWRITEBYTECODE='1',SSD_HF_CACHE='/tmp',SSD_DATASET_DIR='/tmp',
                 SSD_CUDA_ARCH='8.6',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (out/(name+'.log')).open('w') as f:
            proc=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),
                '--variant',args.variant,'--worker',name],cwd=checkout,env=env,stdout=f,stderr=subprocess.STDOUT)
        path=out/(name+'.json');row=json.loads(path.read_text()) if path.exists() else {}
        row.update(name=name,returncode=proc.returncode);rows.append(row);print(row,flush=True)
    stable=source()==before
    report=dict(status='complete' if stable and all(r['returncode']==0 for r in rows) else 'failed',
        source_unchanged=stable,source_sha256=before,suites=rows,tests=sum(r.get('tests',0) for r in rows))
    (out/'CHECKS.json').write_text(json.dumps(report,indent=2)+'\n')
    return int(report['status']!='complete')


if __name__=='__main__':raise SystemExit(main())
