"""Independent numerical, allocation and losslessness checks (CPU)."""
import itertools
import json
from pathlib import Path
import sys
import numpy as np
import torch
from allocation import allocate, make_constants, exhaustive
from gain import third_gain, brute_gains

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'duet_tree_analysis'))
from core import production_functions
from audit_math import bias_audit


def first_output(p, q, count):
    """Integrate token proposal draws and actual residual verifier transitions."""
    def walk(r, d, left):
        if left == 0 or d.sum() == 0:
            return r
        d = d/d.sum(); out = np.zeros(len(p))
        for t in np.flatnonzero(d > 0):
            a = min(1., r[t]/d[t]); out[t] += d[t]*a
            if a < 1:
                rr = np.maximum(r-d, 0); rr /= rr.sum()
                dd = d.copy(); dd[t] = 0
                out += d[t]*(1-a)*walk(rr, dd, left-1)
        return out
    return walk(p, q, count)


def main():
    torch.set_num_threads(1)
    rng = np.random.default_rng(9022026)
    error_gain = error_allocate = error_lossless = 0.
    gain_cases = alloc_cases = distribution_cases = 0
    for v in range(2, 10):
        for i in range(60):
            p = rng.dirichlet(np.full(v, .5)); q = rng.dirichlet(np.full(v, .5))
            a = third_gain(p, q, exact=True)['g']; b = brute_gains(p, q)
            error_gain = max(error_gain, float(np.max(np.abs(np.array(a)-b))))
            gain_cases += 1
    sparse = [([0, 1], [.9, .1]), ([1,0,0], [0,1,0]),
              ([.3,.7,0],[.3,.7,0]), ([.2,.3,.5], [1-1e-12,5e-13,5e-13])]
    for p, q in sparse:
        error_gain = max(error_gain, float(np.max(np.abs(np.array(third_gain(p,q,exact=True)['g'])-brute_gains(p,q)))))
    for case in range(600):
        r = int(rng.integers(1, 5)); b = int(rng.integers(1, 9))
        # A monotone curve need NOT have decreasing marginal gains.
        curve = np.r_[0., np.sort(rng.random(3))]
        if case % 3 == 0: curve = np.array([0., .1, .95, 1.])
        remaining = rng.integers(0, b+1, r); future = int(rng.integers(0, 4))
        counts = [int(rng.integers(0, min(4, x)+1)) for x in remaining]
        root = np.concatenate([np.full(n, i) for i,n in enumerate(counts)]).astype(int)
        weights = rng.random(len(root))
        if case % 9 == 0: weights[:] = .5
        order = rng.permutation(len(root)); root=root[order]; weights=weights[order]
        root = np.r_[root, 0, 0]; weights = np.r_[weights, .99, .99]
        valid = np.r_[np.ones(len(root)-2,dtype=bool), False, False]
        result = allocate(torch.tensor(root), torch.tensor(weights).log(), torch.tensor(valid),
            torch.tensor(remaining), future, make_constants(b, curve)).numpy()
        if np.any(result[~valid]) or np.any(result < 0) or np.any(result > 3):
            raise ValueError('Invalid fanout')
        for rr in range(r):
            mask = (root == rr)&valid; now=max(1, int(remaining[rr])-future) if remaining[rr] else 0
            got=float(np.dot(weights[mask],curve[result[mask]]))
            best,_ = exhaustive(weights[mask],curve,now)
            error_allocate=max(error_allocate,abs(got-best))
            if result[mask].sum()>now: raise ValueError('Node reserve exceeded')
        alloc_cases += 1
    # Fanout chosen from existing parent information. Conditional target p/q
    # may differ arbitrarily across parents; next-token sampling stays exact.
    for case in range(100):
        v=3; p0=rng.dirichlet(np.ones(v)); q0=rng.dirichlet(np.ones(v))
        pnext=rng.dirichlet(np.ones(v), v); qnext=rng.dirichlet(np.ones(v), v)
        weights=rng.random(v);curve=np.r_[0.,np.sort(rng.random(3))]
        fan=allocate(torch.zeros(v,dtype=torch.long),torch.tensor(weights).log(),
            torch.ones(v,dtype=torch.bool),torch.tensor([4]),0,make_constants(4,curve)).numpy()
        # Both root and each token-conditioned next context integrate to p.
        rootout=first_output(p0,q0,int(rng.integers(0,4)))
        got=np.array([rootout[t]*first_output(pnext[t],qnext[t],int(fan[t])) for t in range(v)])
        truth=p0[:,None]*pnext
        error_lossless=max(error_lossless,float(np.max(abs(got-truth))))
        distribution_cases += 1
    if max(error_gain,error_allocate,error_lossless)>1e-9:
        raise ValueError((error_gain,error_allocate,error_lossless))
    prod=production_functions()
    negative=bias_audit(prod['rerank_tree_indices'],prod['tree_verify_walk_tensor'])
    if negative['tv']<=0:raise ValueError('Negative control failed')
    result=dict(passed=True, gain_cases=gain_cases+len(sparse), gain_max_error=error_gain,
        allocation_cases=alloc_cases, allocation_max_error=error_allocate,
        two_token_distribution_cases=distribution_cases, lossless_max_error=error_lossless,
        post_sampling_negative_control_tv=negative['tv'],
        scope='CPU mathematical checks; does not claim CUDA/full engine validation')
    (HERE/'math_checks.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))


if __name__=='__main__':main()
