"""Rao-Blackwell reward diagnostic on actual verified chain snapshots.

This does not refit or change the recommendation. It tests a future calibration
statistic with already-computed p/q, without a learned proxy or extra model.
"""
import json
import numpy as np
from campaign import HERE
from calibrator import load_run


def moments(alpha):
    reach=np.cumprod(np.asarray(alpha,dtype=np.float64))
    mean=reach.sum()
    second=np.dot(2*np.arange(1,len(reach)+1)-1,reach)
    return 1+mean,max(0.,second-mean*mean)


def check_moments():
    rng=np.random.default_rng(37)
    for K in range(1,9):
        a=rng.uniform(0,1,K);reach=np.r_[1,np.cumprod(a)]
        h=np.r_[reach[:-1]*(1-a),reach[-1]]
        values=np.arange(K+1)+1
        mean,var=moments(a)
        assert np.isclose(mean,h@values)
        assert np.isclose(var,h@(values-mean)**2)


def main():
    check_moments();plan=json.loads((HERE/'plan.json').read_text());out=[]
    for stage,seed in [('calibration',4913),('validation',5913)]:
        path=HERE/'runs'/f'probe_{stage}'/f'chain_e56_k4_2_w15_s{seed}'
        run=load_run(path);meta=json.loads((path/'distributions.json').read_text());rows=[]
        for index,record in enumerate(run['records']):
            if meta['seq_steps'].get(str(index+2))!=len(record['metrics']['phase_events']):
                raise ValueError('Probe/verification request-step counts differ')
        for chunk in meta['chunks']:
            with np.load(path/chunk['file']) as z:
                for step in np.unique(z['step']):
                    ix=z['step']==step;sid=int(z['seq_id'][ix][0])
                    if sid==1:continue
                    local=int(z['local_step'][ix][0]);K=int(ix.sum())-1
                    event=run['records'][sid-2]['metrics']['phase_events'][local]
                    if int(event['valid_k'])!=K:raise ValueError('Snapshot/phase length mismatch')
                    p,q,y=z['p_T'][ix],z['p_D'][ix],z['y'][ix]
                    a=np.minimum(p[np.arange(K),y[:K]]/(q[np.arange(K),y[:K]]+1e-10),1.)
                    mean,var=moments(a)
                    rows.append(dict(prompt=sid-2,local_step=local,source=event['source'],K=K,
                                     actual=event['accepted_len'],expected=mean,conditional_variance=var))
        n=len(run['records']);residual=[]
        for i in range(n):
            r=[x for x in rows if x['prompt']==i]
            if not r:raise ValueError('Missing reward observations')
            residual.append(np.mean([x['actual']-x['expected'] for x in r]))
        ds=[m['dataset'] for m in plan['datasets'][stage]['mapping']]
        groups=[np.flatnonzero(np.array(ds)==name) for name in dict.fromkeys(ds)]
        rng=np.random.default_rng(913)
        bootstrap=np.array([np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]) for _ in range(4000)])
        actual=np.array([r['actual'] for r in rows]);expected=np.array([r['expected'] for r in rows])
        out.append(dict(stage=stage,samples=len(rows),actual_mean=float(actual.mean()),expected_mean=float(expected.mean()),
            empirical_actual_variance=float(actual.var()),empirical_expected_variance=float(expected.var()),
            conditional_variance_mean=float(np.mean([r['conditional_variance'] for r in rows])),
            observed_residual_mse=float(np.mean((actual-expected)**2)),
            prompt_mean_actual_minus_expected=float(np.mean(residual)),
            paired_prompt_residual_ci95=np.quantile(np.array(residual)[bootstrap].mean(1),[.025,.975]).tolist(),
            rows=rows))
    payload=dict(moment_identity_check='passed',results=out,
        scope='Diagnostic only. E[U|current prefix, draft suffix, p, q] = 1 + sum_j prod_i<=j alpha_i. Uses the verifier epsilon 1e-10. No improvement in held-out recommendation claimed.',
        limitation='Removes acceptance-coin noise conditionally; prompt, draft trajectory, phase-hit and unseen-configuration uncertainty remain. Finite-sample variances need not satisfy the population inequality exactly.')
    (HERE/'reward_expectation.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k!='rows'} for r in out],indent=2))


if __name__=='__main__':main()
