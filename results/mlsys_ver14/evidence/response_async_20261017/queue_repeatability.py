"""Finish the three remaining runs as diagnostics after a parity failure.

Do not silently replace the failed run or turn unequal workloads into a
latency claim. Retain the original failure and compare every subsequent run.
"""
import json
from pathlib import Path
import subprocess
import sys
from queue_runs import signature

ROOT=Path('/data2/chokwans99/response-async-20261017')


def main():
    plan=[('hostlist','hostlist_13b_host_b',2,False,'before_13b_final_b'),
          ('before','before_13b_host_b',2,False,'before_13b_final_b'),
          ('hostlist','hostlist_13b_profile',1,True,'before_13b_profile')]
    results=[]
    for variant,tag,passes,profile,reference in plan:
        folder=ROOT/'runs'/tag
        if not folder.exists():
            argv=[sys.executable,str(ROOT/'remote_run.py'),'--variant',variant,'--model','13b',
                  '--tag',tag,'--passes',str(passes)]
            if profile:argv.append('--profile')
            subprocess.run(argv,check=True)
        assert json.loads((folder/'status.json').read_text())['status']=='complete'
        a=json.loads((ROOT/'runs'/reference/'run.json').read_text())
        b=json.loads((folder/'run.json').read_text())
        parity=dict(outputs_equal=a['outputs']==b['outputs'],events_equal=signature(a)==signature(b),
                    outputs=len(b['outputs']),events=len(signature(b)),purpose='repeatability diagnostic after host_a mismatch; not replacement evidence')
        (folder/'PARITY.json').write_text(json.dumps(parity,indent=2)+'\n')
        results.append(dict(tag=tag,**parity))
        (ROOT/'REPEATABILITY.json').write_text(json.dumps(dict(status='running',runs=results,
            original_failed_run='hostlist_13b_host_a',discarded_or_overwritten=False),indent=2)+'\n')
        print(results[-1],flush=True)
    (ROOT/'REPEATABILITY.json').write_text(json.dumps(dict(status='complete',runs=results,
        original_failed_run='hostlist_13b_host_a',discarded_or_overwritten=False),indent=2)+'\n')


if __name__=='__main__':main()
