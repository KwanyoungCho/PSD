"""C=3 local prospective allocation on the frozen full-corpus raw sample.

This integrates unsampled children only at already observed expanded parents.
It cannot supply target distributions on missing frontier branches or online AL.
"""
from collections import defaultdict
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np
from allocation import partitions
from gain import third_gain

HERE=Path(__file__).resolve().parent
POST=HERE.parent/'duet_tree_posthoc'
sys.path.insert(0,str(POST))
from features import groups
from calibrate import frozen_predict, from_edges
from analyze import macro, interval


def best(weights, curves, budget):
    # Small observed groups; parent-specific oracle curves need unrestricted
    # enumeration rather than rearrangement of a common estimated curve.
    patterns=np.array([x for x in itertools.product(range(4),repeat=len(weights)) if sum(x)<=budget])
    values=(np.asarray(weights)[None,:]*np.take_along_axis(
        np.asarray(curves)[None,:,:],patterns[:,:,None],axis=2)[:,:,0]).sum(1)
    return patterns[int(np.argmax(values))]


def main():
    calibration=json.loads((HERE/'gain_calibration.json').read_text())
    frozen=json.loads((HERE.parent/'duet_tree_analysis/calibration_frozen.json').read_text())
    contexts=[];rows=[];start=time.time();counter=0
    out=HERE/'c3_contexts.jsonl'
    if out.exists():raise FileExistsError('Preserve completed or partial diagnostic data')
    with out.open('x',buffering=1) as sink:
        for policy in ('q_path','phase_sibling_q_bin'):
            folder=POST/'runs'/policy
            for line in (folder/'trees.jsonl').open():
                t=json.loads(line)
                if t['is_final_event'] or 'raw_file' not in t:continue
                fr,_=from_edges(t,frozen_predict(t,frozen));local={};bydepth=defaultdict(list)
                with np.load(folder/t['raw_file']) as z:
                    for parent,kids in groups(t['par'],t['sib']).items():
                        if parent<0:continue
                        g=third_gain(z['p'][parent+1],z['q'][kids[0]],seed=9122000+counter,head=32,samples=256)
                        row={k:t[k] for k in ('question_id','group','phase','policy','serial','uid')}
                        row.update(parent=parent,depth=t['depth'][parent],fanout=len(kids),
                            rho=t['reach'][parent],frozen=float(fr[parent]),q_path=t['q_path'][parent],**g)
                        contexts.append(row);sink.write(json.dumps(row)+'\n');counter+=1
                        local[parent]=row;bydepth[row['depth']].append(parent)
                for depth,parents in bydepth.items():
                    if len(parents)<2:continue
                    cases=[('observed_round',parents,sum(local[p]['fanout'] for p in parents))]
                    cases += [('pair_budget3',[u,v],3) for u,v in itertools.combinations(parents,2)]
                    for kind,pp,budget in cases:
                        truth=np.array([local[p]['g'] for p in pp]);rho=np.array([local[p]['rho'] for p in pp])
                        curve=calibration['curves'][str(t['phase'])];curves=[curve]*len(pp)
                        q=np.array([local[p]['q_path'] for p in pp]);reach=np.array([local[p]['frozen'] for p in pp])
                        def breadth(weights):
                            fan=np.zeros(len(pp),dtype=int);left=budget;order=np.argsort(-weights,kind='stable')
                            for _ in range(3):
                                for j in order:
                                    if left>0:fan[j]+=1;left-=1
                            return fan
                        def value(fan):return float(sum(rho[j]*truth[j,c] for j,c in enumerate(fan)))
                        fans=dict(q_breadth=breadth(q),frozen_breadth=breadth(reach),
                            q_gain=best(q,curves,budget),frozen_gain=best(reach,curves,budget),
                            true_reach_gain=best(rho,curves,budget),
                            frozen_true_gain=best(reach,truth,budget),oracle=best(rho,truth,budget))
                        if kind=='observed_round':fans['observed']=np.array([local[p]['fanout'] for p in pp])
                        result={k:t[k] for k in ('question_id','group','phase','policy','serial','uid')}
                        result.update(kind=kind,depth=depth,parents=pp,budget=budget,
                            **{k:value(v) for k,v in fans.items()},fans={k:v.tolist() for k,v in fans.items()})
                        base=fans['frozen_breadth'];new=fans['frozen_gain'];delta=value(new)-value(base)
                        result.update(changed=float(not np.array_equal(base,new)),win=float(delta>1e-10),
                            loss=float(delta< -1e-10),delta=delta,
                            g3_error_se=float(np.sqrt(sum((rho[j]*local[p]['se'])**2 for j,p in enumerate(pp) if (base[j]==3)!=(new[j]==3)))))
                        # Sensitivity to the tiny calibration set, no retuning.
                        leave=calibration['calibration_stability'][str(t['phase'])]['leave_one_prompt_out']
                        vals=[value(best(reach,[cc]*len(pp),budget))-value(base) for cc in leave]
                        result.update(calibration_loo_min=min(vals),calibration_loo_max=max(vals),
                            calibration_loo_decision_change=float(any(not np.array_equal(best(reach,[cc]*len(pp),budget),new) for cc in leave)))
                        rows.append(result)
                if counter%100<5:
                    print('contexts',counter,'elapsed',round(time.time()-start,1),flush=True)
    (HERE/'c3_allocations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result=dict(scope='Local prospective C=3 gain on known expanded contexts; not a same-prefix super-tree or online trial.',
        contexts=len(contexts),questions=len({r['question_id'] for r in contexts}),allocation_cases=len(rows),
        integration_max_se=max(r['se'] for r in contexts),wall_s=time.time()-start,summary={})
    for kind in ('pair_budget3','observed_round'):
        result['summary'][kind]={}
        for phase in ('all',1,2):
            rr=[r for r in rows if r['kind']==kind and (phase=='all' or r['phase']==phase)]
            fields=['q_breadth','frozen_breadth','q_gain','frozen_gain','true_reach_gain','frozen_true_gain','oracle',
                    'changed','win','loss','calibration_loo_decision_change','calibration_loo_min','calibration_loo_max']
            result['summary'][kind][str(phase)]=dict(cases=len(rr),questions=len({r['question_id'] for r in rr}),
                means={k:macro(rr,k) for k in fields},
                ci={k:interval(rr,k,'frozen_breadth') for k in ['q_breadth','q_gain','frozen_gain','true_reach_gain','frozen_true_gain','oracle']},
                delta_integration_max_se=max(r['g3_error_se'] for r in rr))
    (HERE/'c3_analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
