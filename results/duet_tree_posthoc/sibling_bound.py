"""Certify the frozen lookup cannot prefer a later sibling to the first one."""
import itertools
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    model=json.loads((HERE.parent/'duet_tree_analysis/calibration_frozen.json').read_text())
    result=dict(passed=True,scope='Same parent, frozen phase/sibling/q-bin table. Not all nodes across different roots.',phases={})
    for phase in ['1','2']:
        table=[]
        for sibling in range(3):
            key=phase+':'+str(sibling)
            fallback=model['means'].get(key,model['means'].get(phase,model['means']['all']))
            table.append([model['means'].get(key+':'+str(b),fallback) for b in range(len(model['bins'])-1)])
        lo=np.min(table,axis=1);hi=np.max(table,axis=1);minimum_gap=1.
        for a,b,c in itertools.product(*table):
            beta=np.array([a,(1-a)*b,(1-a)*(1-b)*c])
            minimum_gap=min(minimum_gap,float(beta[0]-beta[1:].max()))
        assert lo[0]>.5 and minimum_gap>0
        result['phases'][phase]=dict(first_alpha_lower=float(lo[0]),
            total_later_beta_upper=float(1-lo[0]),
            second_beta_upper=float((1-lo[0])*hi[1]),
            third_beta_upper=float((1-lo[0])*(1-lo[1])*hi[2]),
            combinations_checked=len(table[0])**3,minimum_first_vs_later_gap=minimum_gap)
    (HERE/'sibling_bound.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
