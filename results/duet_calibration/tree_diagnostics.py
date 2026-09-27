"""Preserve the failed tree interpolation and explain its large latency tail."""
import json
from campaign import HERE,config
from calibrator import load_run,metrics
from tree_validate import phase_verify


def large_events(run):
    return [dict(prompt=i,step=j,source=e['source'],valid_k=e['valid_k'],milliseconds=t*1000)
            for i,rec in enumerate(run['records'])
            for j,(e,t) in enumerate(zip(rec['metrics']['phase_events'],rec['metrics']['target_verify_times']))
            if j>=2 and t>.1]


def main():
    cfg=config(mode='p2tree')
    old=load_run(HERE/'runs/extensions'/f'{cfg["tag"]}_s913')
    new=load_run(HERE/'runs/tree_repeat'/f'{cfg["tag"]}_s913')
    data=json.loads((HERE/'tree_validation.json').read_text())
    prediction=sum(r['p2_verify']['median_ms'] for r in data['all'][:2])/2
    measured=data['measured_p2_verify6']['median_ms']
    out=dict(original=dict(metrics=metrics(old),p2_verify=phase_verify(old,2),over100ms=large_events(old)),
        repeat=dict(metrics=metrics(new),p2_verify=phase_verify(new,2),over100ms=large_events(new)),
        posthoc_median_interpolation_ms=prediction,measured_nv6_median_ms=measured,
        posthoc_median_error_pct=100*abs(prediction-measured)/measured,
        scope='Post-hoc diagnostic after the original mean interpolation failed; same prompts/seed, new process, existing kernel caches. Preserve original failure. Exact cause of stalls was not isolated. Median error is descriptive, not a separately held-out model validation.')
    (HERE/'tree_diagnostics.json').write_text(json.dumps(out,indent=2))


if __name__=='__main__':main()
