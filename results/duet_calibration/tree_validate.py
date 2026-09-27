"""Hold out a tree verify shape; separate generation from verification size."""
import json
import time
from campaign import HERE, config, run_one
from calibrator import load_run, metrics
import numpy as np


def phase_verify(run,phase):
    values=[]
    for rec in run['records']:
        ev=rec['metrics']['phase_events'];times=rec['metrics']['target_verify_times']
        if len(ev)!=len(times):raise ValueError('Unaligned verification metrics')
        values.extend(t*1000 for e,t in zip(ev[2:],times[2:]) if e['source']==phase)
    return dict(n=len(values),mean_ms=float(np.mean(values)) if values else None,
                median_ms=float(np.median(values)) if values else None)


def main():
    while not (HERE/'candidate_summary.json').exists():time.sleep(10)
    anchors=[config(mode='p2tree',v2=v) for v in [4,8]]
    runs=[]
    for cfg in anchors:
        path=HERE/'runs/extensions'/f'{cfg["tag"]}_s913'
        if not (path/'complete.json').exists():
            (HERE/'tree_validation.json').write_text(json.dumps(dict(status='blocked_by_anchor_failure',path=str(path)),indent=2));return
        runs.append(load_run(path))
    pred=np.mean([phase_verify(r,2)['mean_ms'] for r in runs])
    freeze=HERE/'tree_frozen.json'
    if not freeze.exists():freeze.write_text(json.dumps(dict(predicted_p2_verify6_ms=float(pred),anchors=[phase_verify(r,2) for r in runs]),indent=2))
    tests=[config(mode='p2tree',v2=6),config(mode='p2tree',n2=12,v2=8)]
    for cfg in tests:run_one(cfg,'extensions',profile=True)
    tested=[load_run(HERE/'runs/extensions'/f'{cfg["tag"]}_s913') for cfg in tests]
    out=dict(status='complete',predicted_p2_verify6_ms=float(pred),measured_p2_verify6=phase_verify(tested[0],2),
             all=[dict(config=r['config'],metrics=metrics(r),p2_verify=phase_verify(r,2)) for r in runs+tested],
             note='Interpolation of conditional target verification latency only; all runs are profiled. No tree-quality or final TPS optimality guarantee.')
    out['prediction_error_pct']=100*abs(pred-out['measured_p2_verify6']['mean_ms'])/out['measured_p2_verify6']['mean_ms']
    (HERE/'tree_validation.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))


if __name__=='__main__':main()
