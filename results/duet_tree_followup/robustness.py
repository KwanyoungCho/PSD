"""Separate numerical integration, calibration instability and rare losses."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from diagnose_c3 import best,macro,interval

HERE=Path(__file__).resolve().parent


def main():
    calibration=json.loads((HERE/'gain_calibration.json').read_text())
    contexts={}
    for line in (HERE/'c3_contexts.jsonl').open():
        r=json.loads(line);contexts[r['policy'],r['serial'],r['parent']]=r
    rows=[]
    for line in (HERE/'c3_allocations.jsonl').open():
        r=json.loads(line)
        if r['kind']!='observed_round':continue
        cc=[contexts[r['policy'],r['serial'],p] for p in r['parents']]
        rho=np.array([x['rho'] for x in cc]);weights=np.array([x['frozen'] for x in cc]);truth=np.array([x['g'] for x in cc])
        before=r['fans']['frozen_breadth'];after=r['fans']['frozen_gain']
        low=high=0.
        for j,(a,b) in enumerate(zip(after,before)):
            value=truth[j,a]-truth[j,b]
            if a==3 and b!=3:
                lo,hi=cc[j]['bound'];low+=rho[j]*(lo-truth[j,b]);high+=rho[j]*(hi-truth[j,b])
            elif b==3 and a!=3:
                lo,hi=cc[j]['bound'];low+=rho[j]*(truth[j,a]-hi);high+=rho[j]*(truth[j,a]-lo)
            else:low+=rho[j]*value;high+=rho[j]*value
        phase=str(r['phase']);curves=calibration['curves'][phase]
        fans=[]
        for limit in calibration['numerical_integration'][phase]['deterministic_bound']:
            c=curves[:];c[3]=limit;fans.append(best(weights,[c]*len(cc),r['budget']))
        row=dict(r,positive=max(0,r['delta']),negative=min(0,r['delta']),
            integration_delta_lower=low,integration_delta_upper=high,
            calibration_numeric_decision_stable=float(all(np.array_equal(x,after) for x in fans)))
        for i,curve in enumerate(calibration['calibration_stability'][phase]['leave_one_prompt_out']):
            fan=best(weights,[curve]*len(cc),r['budget'])
            row['loo_'+str(i)]=float(sum(rho[j]*truth[j,c] for j,c in enumerate(fan)))-r['frozen_breadth']
        rows.append(row)
    result=dict(scope='Exploratory local C=3 allocation; no online AL claim',summary={})
    for phase in ('all',1,2):
        rr=[r for r in rows if phase=='all' or r['phase']==phase]
        result['summary'][str(phase)]=dict(cases=len(rr),questions=len({r['question_id'] for r in rr}),
            means={k:macro(rr,k) for k in ['delta','positive','negative','integration_delta_lower','integration_delta_upper','calibration_numeric_decision_stable']},
            leave_one_prompt_out_mean_deltas=[macro(rr,'loo_'+str(i)) for i in range(8)],
            all_group_ci=interval(rr,'delta'),
            worst_losses=sorted(rr,key=lambda r:r['delta'])[:5],
            largest_gains=sorted(rr,key=lambda r:-r['delta'])[:5])
    result['within_collection_policy']={p:{k:interval([r for r in rows if r['policy']==p],k,'frozen_breadth')
        for k in ('q_gain','frozen_gain','oracle')} for p in sorted({r['policy'] for r in rows})}
    result['within_task']={g:interval([r for r in rows if r['group']==g],'delta') for g in sorted({r['group'] for r in rows})}
    result['direct_q_gain_vs_frozen_gain']=interval(rows,'q_gain','frozen_gain')
    result['q_gain_vs_q_breadth']=interval(rows,'q_gain','q_breadth')
    (HERE/'robustness.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk not in ('worst_losses','largest_gains')} for k,v in result['summary'].items()},indent=2))


if __name__=='__main__':main()
