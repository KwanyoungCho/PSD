"""Independent float64 simplex checks for stated mathematical claims.

Numerical checks complement the algebra; they are not a proof or model test.
"""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    rng=np.random.default_rng(20260912)
    tv=lambda x,y: .5*np.abs(x-y).sum(-1)
    norm=lambda a:a/a.sum(-1,keepdims=True)
    pick=lambda a,k:np.argsort(-a,axis=-1,kind='stable')[:,:k]
    mass=lambda a,ids:np.take_along_axis(a,ids,-1).sum(-1)
    counts=dict(contexts=0,bound=0,direct_TV_certificate=0,raw_score_certificate=0,
                topk_margin_certificate=0,regret_certificate=0)
    for vocabulary in (8,32):
        n=30000;k=3
        concentrations=np.exp(rng.uniform(-3,1,(n,1)))
        def sample():
            a=rng.gamma(concentrations,size=(n,vocabulary))
            return norm(a)
        p,q,u=sample(),sample(),sample()
        error_scale=np.exp(rng.uniform(-12,0,(n,1)))
        e=(1-error_scale)*p+error_scale*u
        a,b=np.maximum(p-q,0),np.maximum(e-q,0)
        z,zh=a.sum(-1),b.sum(-1)
        assert min(z.min(),zh.min())>0
        r,rh=norm(a),norm(b)
        delta,eta=tv(p,e),np.max(abs(p-e),axis=-1)
        bound=np.minimum(1,delta/np.maximum(z,zh))
        assert np.max(tv(r,rh)-bound)<1e-12
        counts['bound']+=n
        sr,se,star=pick(b,k),pick(e,k),pick(a,k)
        cr,ce,cs=mass(r,sr),mass(r,se),mass(r,star)
        assert np.min(cs-cr)>-1e-12
        assert np.min(cs-mass(r,pick(p,k)))>-1e-12
        assert np.max((cs-cr)-2*tv(r,rh))<1e-12
        gamma=mass(b,sr)-mass(b,se)
        threshold=np.minimum(2*delta,2*k*eta)
        certificate=gamma>threshold+1e-12
        assert (cr[certificate]>ce[certificate]).all()
        counts['raw_score_certificate']+=int(certificate.sum())
        topvalues=np.sort(a,axis=-1)[:,-(k+1):]
        margin=topvalues[:,1]-topvalues[:,0]
        stable=margin>2*eta+1e-12
        np.testing.assert_array_equal(np.sort(sr[stable],axis=1),np.sort(star[stable],axis=1))
        counts['topk_margin_certificate']+=int(stable.sum())
        direct=bound < tv(r,p)-delta-1e-12
        assert (tv(r,rh)[direct]<tv(r,e)[direct]).all()
        counts['direct_TV_certificate']+=int(direct.sum())
        enough=cs-ce>2*bound+1e-12
        assert (cr[enough]>ce[enough]).all()
        counts['regret_certificate']+=int(enough.sum())
        # Exact support/ranking regret, including actual zero-score fillers.
        reachable=b>0
        np.put_along_axis(reachable,sr,True,axis=1)
        constrained=mass(r,pick(np.where(reachable,r,-np.inf),k))
        np.testing.assert_allclose(cr-ce,(cs-ce)-(cs-constrained)-(constrained-cr),atol=1e-12)
        assert min((cs-constrained).min(),(constrained-cr).min())>-1e-12
        counts['contexts']+=n
    e=np.array([[.4,.35,.25]])
    q=np.array([[.39,.10,.51]])
    p=np.array([[.49,.10,.41],[.39,.20,.41]])
    r=norm(np.maximum(p-q,0))
    ir=pick(np.maximum(e-q,0),1);ie=pick(e,1)
    gains=(r[:,ir[0,0]]-r[:,ie[0,0]]).tolist()
    assert gains==[-1.,1.]
    # Z can be maximal without any residual ranking advantage.
    p=np.array([[.6,.4,0.]])
    q=np.array([[0.,0.,1.]])
    a=np.maximum(p-q,0)
    np.testing.assert_allclose(a.sum(-1),1)
    np.testing.assert_allclose(norm(a),p)
    result=dict(passed=True,seed=20260912,precision='float64',counts=counts,
                same_proxy_draft_opposite_target_gains=gains,
                large_Z_no_gain=True,scope='unmasked, fixed cost top-k; excludes zero normalization cases')
    (HERE/'theory_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
