"""Hand-computed weighting, conditional-transition and observable-feature checks."""
import json
from pathlib import Path
import numpy as np
from analyze import macro, decorate
from calibrate import keys, fit, predict


def main():
    rows=[dict(group='a',question_id=1,x=10.),dict(group='a',question_id=1,x=0.),
        dict(group='a',question_id=2,x=1.),dict(group='b',question_id=3,x=9.)]
    assert macro(rows,'x')==6. # ((mean(10,0)+1)/2 + 9)/2
    for r in rows:r['delta']=r['x']-2
    assert macro(rows,'delta')==4.
    t=dict(n=3,par=[-1,-1,1],sib=[0,1,0],alpha=[1.,.5,1.],reach=[1.,0.,0.])
    decorate(t)
    assert np.array_equal(t['_beta'],[1.,0.,1.])
    # Features from unseen next p/q cannot accidentally enter this lookup.
    t.update(phase=1,raw=[.9,.1,.6],q_cond=[.9,1.,.6],depth=[1,1,2],q_entropy=[.3,.3,.5],
        attempt=[1.,0.,0.],overlap=[.1,.2,.3],p_token=[.01,.99,.6])
    model=fit([t],'draft_rich');before=predict(t,model)
    t['overlap']=[.99,.99,.99];t['p_token']=[1.,0.,0.];t['alpha']=[0.,0.,0.]
    assert np.array_equal(before,predict(t,model))
    out=dict(passed=True,checks=['hand-computed task/question/tree weighting',
        'local transition at zero-reach parents', 'no oracle target fields in lookup features'])
    (Path(__file__).resolve().parent/'statistics_checks.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out))


if __name__=='__main__':main()
