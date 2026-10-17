"""Per-GPU-pair dependency queues; independent pairs never wait at a barrier."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import time
from run_mixed import ROOT, run


def main():
    p=argparse.ArgumentParser();p.add_argument('plan',type=Path);p.add_argument('--launch',action='store_true')
    a=p.parse_args()
    if a.launch:
        log=a.plan.with_suffix('.controller.log')
        with log.open('x') as f:
            proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),str(a.plan.resolve())],
                stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=proc.pid,log=str(log))),flush=True);return
    plan=json.loads(a.plan.read_text())
    def worker(slot):
        q=plan['queues'][slot]
        for dep in q.get('after',[]):
            while not (ROOT/dep).exists():time.sleep(2)
        results=[]
        for job in q['jobs']:
            r=run(job,slot);results.append(r)
            if r['status']!='complete':break
        return results
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(worker,range(2)))
    a.plan.with_suffix('.status.json').write_text(json.dumps(dict(results=results),indent=2)+'\n')


if __name__=='__main__':main()
