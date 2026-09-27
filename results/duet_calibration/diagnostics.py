"""Supplementary diagnostics; never modifies frozen primary predictions."""
import json
import numpy as np
from campaign import HERE,config
from calibrator import Calibrator,load_run,metrics,profile_rows,features


def main():
    frozen=json.loads((HERE/'frozen.json').read_text());model=Calibrator(frozen['model'])
    plan=json.loads((HERE/'plan.json').read_text());contexts=[m['input_tokens'] for m in plan['datasets']['calibration']['mapping']]
    low=[]
    for layer in [40,56]:
        cfg=config(exit=layer,k1=2,k2=2);path=HERE/'runs/extensions'/f'{cfg["tag"]}_s913'
        if not (path/'complete.json').exists():continue
        run=load_run(path);rows,diag=profile_rows(run);pred=model.predict(cfg,contexts)
        actual=float(np.mean([x for r in run['records'] for x in r['metrics']['target_step_times'][3:-1]])*1000)
        components={}
        for kind in ['F','P','x','D1','D2','S']:
            observed=np.mean([r[kind] for r in rows])
            predicted=np.mean([np.dot(frozen['model']['components'][kind]['coef'],
                features(kind,r['config'],r['next_state'] if kind=='S' else r['state'],r['context'])) for r in rows])
            components[kind]=dict(observed_ms=float(observed),predicted_ms=float(predicted),
                                  residual_ms=float(observed-predicted))
        low.append(dict(config=cfg,metrics=metrics(run),profile_diagnostics=diag,prediction=pred,
            actual_cycle_ms=actual,cycle_error_pct=100*abs(pred['cycle_ms']-actual)/actual,
            components_at_observed_states=components,
            mean_gap1_ms=float(np.mean([r['gap1'] for r in rows])),
            fraction_p1_complete_before_proxy=float(np.mean([r['gap1']>0 for r in rows]))))
    reward=[]
    for stage,seed in [('calibration',913),('validation',1913)]:
        for layer in [40,56,72]:
            run=load_run(HERE/'runs'/stage/f'{config(exit=layer)["tag"]}_s{seed}')
            per_dataset={}
            for ds in dict.fromkeys(m['dataset'] for m in plan['datasets'][stage]['mapping']):
                ev=[e for r,m in zip(run['records'],plan['datasets'][stage]['mapping']) if m['dataset']==ds for e in r['metrics']['phase_events']]
                per_dataset[ds]=dict(steps=len(ev),u=float(np.mean([e['accepted_len'] for e in ev])))
            reward.append(dict(stage=stage,exit=layer,metrics=metrics(run),dataset_reward=per_dataset))
    (HERE/'diagnostics.json').write_text(json.dumps(dict(low_forward=low,anchor_reward_transfer=reward,
        note='Supplementary profiled low-K extrapolation and descriptive reward transfer. No retroactive refit.'),indent=2))


if __name__=='__main__':main()
