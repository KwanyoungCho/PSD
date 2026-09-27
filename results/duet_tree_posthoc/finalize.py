"""Finish independent audits/analysis once both full-corpus collections succeed."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wait',action='store_true');args=ap.parse_args()
    while True:
        paths=[HERE/'runs'/p/'completion.json' for p in ['q_path','phase_sibling_q_bin']]
        for p in paths:
            if p.exists() and json.loads(p.read_text())['exit_code']:
                raise RuntimeError(f'Collection failed: {p}')
        if all(p.exists() for p in paths):break
        if not args.wait:raise RuntimeError('Collection still running')
        time.sleep(15)
    jobs=[];env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for script in ['validate.py','analyze.py','examples.py','fanout.py']:
        log=(HERE/(script[:-3]+'.log')).open('w')
        proc=subprocess.Popen([sys.executable,str(HERE/script)],env=env,stdout=log,stderr=subprocess.STDOUT)
        jobs.append((script,proc,log))
    failures=[]
    for script,proc,log in jobs:
        code=proc.wait();log.close()
        if code:failures.append((script,code))
        print(script,'exit',code,flush=True)
    if failures:raise RuntimeError(f'Analysis failure: {failures}')
    subprocess.run([sys.executable,str(HERE/'sibling_bound.py')],check=True,env=env)
    subprocess.run([sys.executable,str(HERE/'make_report.py')],check=True,env=env)
    (HERE/'analysis_completed.json').write_text(json.dumps(dict(complete=True,
        reviewed=False,completed_unix=time.time(),scope='Data collection, audits and automatic report; interpretation review pending.'),indent=2))


if __name__=='__main__':main()
