"""Reference math ported without semantic changes; source hashes in calibration_math_sources.json."""
import numpy as np
from collections import defaultdict

def groups(par, sib):
    kids = defaultdict(list)
    for j, p in enumerate(par):
        if not -1 <= int(p) < j:
            raise ValueError('Invalid parent order')
        kids[int(p)].append(j)
    for p, js in kids.items():
        js.sort(key=lambda j: int(sib[j]))
        if [int(sib[j]) for j in js] != list(range(len(js))):
            raise ValueError('Sibling prefix required')
    return kids

def normalize(x, fallback=None):
    x = np.asarray(x, dtype=np.float64)
    total = x.sum()
    return x / total if total > 1e-14 else (
        np.zeros_like(x) if fallback is None else np.asarray(fallback).copy())

def exact_ladder(par, sib, tokens, p_rows, q_rows):
    """q_rows[j] is the ORIGINAL parent q, before sibling exclusions.

    Returns conditional alpha, joint node reach, joint attempt probability,
    terminal mass, terminal residual, and expected accepted token count.
    All quantities condition on the already realized, unpruned candidate tree.
    """
    par = list(map(int, par)); sib = list(map(int, sib))
    tokens = np.asarray(tokens, dtype=np.int64)
    p_rows = np.asarray(p_rows, dtype=np.float64)
    q_rows = np.asarray(q_rows, dtype=np.float64)
    n = len(par)
    kids = groups(par, sib)
    alpha = np.zeros(n); reach = np.zeros(n); attempt = np.zeros(n)
    residual = p_rows.copy(); terminal = np.zeros(n + 1)
    for ctx in [-1] + list(range(n)):
        incoming = 1.0 if ctx == -1 else reach[ctx]
        R = p_rows[ctx + 1].copy()
        remaining = incoming
        js = kids.get(ctx, [])
        D = q_rows[js[0]].copy() if js else None
        for j in js:
            t = tokens[j]
            if D[t] <= 0:
                raise ValueError('Zero proposal probability')
            a = min(1.0, R[t] / D[t])
            alpha[j] = a; attempt[j] = remaining
            reach[j] = remaining * a
            remaining *= 1.0 - a
            R = normalize(np.maximum(R - D, 0))
            D[t] = 0; D = normalize(D)
        terminal[ctx + 1] = remaining
        residual[ctx + 1] = normalize(R, p_rows[ctx + 1])
    assert abs(terminal.sum() - 1) < 1e-8
    depth = np.zeros(n, dtype=int)
    for j, p in enumerate(par): depth[j] = 1 if p < 0 else depth[p] + 1
    assert abs(reach.sum() - np.dot(terminal[1:], depth)) < 1e-8
    return dict(alpha=alpha, reach=reach, attempt=attempt,
                terminal=terminal, residual=residual, al=float(reach.sum()))

def fit(trees, bins, shrink):
    stats = defaultdict(lambda: [0., 0.])
    for t in trees:
        for j, (a, w) in enumerate(zip(t['calc']['alpha'], t['calc']['attempt'])):
            phase = str(t['phase']); sibling = str(t['sib'][j]); b = str(np.searchsorted(bins, t['raw'][j], side='right') - 1)
            for key in ['all', phase, phase+':'+sibling, phase+':'+sibling+':'+b]:
                stats[key][0] += w*a; stats[key][1] += w
    means = {}
    for key in sorted(stats, key=lambda k: (k != 'all', k.count(':'), k)):
        num, den = stats[key]
        parent = ':'.join(key.split(':')[:-1]) if ':' in key else 'all'
        prior = means.get(parent, .5)
        means[key] = num/den if key == 'all' and den > 0 else (num + shrink*prior)/(den+shrink)
    return dict(bins=bins, shrinkage=shrink, means=means, sufficient_statistics=dict(stats),
                label='Expected conditional acceptance; weighted by exact attempt probability.',
                training_prompts=sorted({t['prompt'] for t in trees}),
                validation_used_for_fitting=False)

def gains(p,q):
    p=np.asarray(p,dtype=float);q=np.asarray(q,dtype=float)
    p=p/p.sum();q=q/q.sum()
    g1=float(np.minimum(p,q).sum());z=1-g1
    if z<1e-14:return g1,g1
    residual=np.maximum(p-q,0);residual/=residual.sum()
    support=q>0
    ratios=residual[support]/q[support]
    order=np.argsort(ratios);ratios=ratios[order]
    qq=q[support][order];rr=residual[support][order]
    rprefix=np.r_[0.,np.cumsum(rr)]
    # Suffix sums avoid losing tiny q tails via 1-cumsum cancellation.
    qsuffix=np.r_[np.cumsum(qq[::-1])[::-1],0.]
    ids=np.flatnonzero(q>p);weights=q[ids]-p[ids]
    denom=1-q[ids];normal=denom>1e-12
    overlap=np.zeros(len(ids))
    scale=1/denom[normal]
    pos=np.searchsorted(ratios,scale,side='right')
    overlap[normal]=rprefix[pos]+scale*qsuffix[pos]
    for ix in np.flatnonzero(~normal):
        d=q.copy();d[ids[ix]]=0;mass=d.sum()
        if mass>0:overlap[ix]=np.minimum(residual,d/mass).sum()
    g2=g1+float(np.dot(weights,np.clip(overlap,0,1)))
    if not g1-1e-10<=g2<=1+1e-10:raise ValueError('Invalid gain')
    return g1,min(1.,g2)

def distribution(x):
    x = np.asarray(x, dtype=np.float64)
    if np.any(x < 0) or not np.isfinite(x).all() or x.sum() <= 0:
        raise ValueError('Invalid probability vector')
    return x / x.sum()

def third_gain(p, q, *, seed=0, head=32, samples=256, exact=False):
    p, q = distribution(p), distribution(q)
    g1, g2 = gains(p, q)
    z = np.maximum(q-p, 0)
    ids = np.flatnonzero(z > 0)
    if len(ids) == 0 or g2 >= 1-1e-14:
        return dict(g=[0., g1, g2, g2], se=0., tail_mass=0., bound=[g2, g2],
                    evaluations=0, samples=0)
    residual = np.maximum(p-q, 0); residual /= residual.sum()
    # Common R/q order is unchanged by subtracting c*q and renormalizing R.
    support = np.flatnonzero(q > 0)
    ratios = residual[support] / q[support]
    order = np.argsort(ratios)
    support = support[order]; ratios = ratios[order]
    qs = q[support]; rs = residual[support]
    qsuffix = np.r_[np.cumsum(qs[::-1])[::-1], 0.]
    cache = {}

    def increment(t):
        t = int(t)
        if t in cache:
            return cache[t]
        denom = 1-q[t]
        if denom < 1e-10:
            d = q.copy(); d[t] = 0
            if d.sum() == 0:
                value = 0.
            else:
                a, b = gains(residual, d/d.sum()); value = b-a
            cache[t] = value
            return value
        c = 1/denom
        rr = np.maximum(residual-c*q, 0)
        zz = rr.sum()
        if zz < 1e-14:
            cache[t] = 0.
            return 0.
        prefix = np.r_[0., np.cumsum(rr[support] / zz)]
        weights = np.maximum(c*q-residual, 0); weights[t] = 0
        uu = np.flatnonzero(weights > 0)
        den3 = 1-q[t]-q[uu]
        normal = den3 > 1e-10
        overlap = np.zeros(len(uu))
        c3 = 1/den3[normal]
        pos = np.searchsorted(ratios, c+zz*c3, side='right')
        overlap[normal] = prefix[pos]+c3*qsuffix[pos]
        for j in np.flatnonzero(~normal):
            d = q.copy(); d[t] = 0; d[uu[j]] = 0
            if d.sum() > 0:
                overlap[j] = np.minimum(rr/zz, d/d.sum()).sum()
        value = float(np.dot(weights[uu], np.clip(overlap, 0, 1)))
        cache[t] = value
        return value

    ids = ids[np.argsort(-z[ids], kind='stable')]
    top = ids if exact else ids[:head]
    tail = np.empty(0, dtype=int) if exact else ids[head:]
    base = g2 + sum(float(z[t])*increment(t) for t in top)
    mass = float(z[tail].sum())
    se = 0.; value = base; n = 0
    if mass > 0:
        if samples < 2:
            raise ValueError('At least two integration draws required')
        rng = np.random.default_rng(seed)
        draws = rng.choice(tail, size=samples, p=z[tail]/mass)
        y = np.array([increment(t) for t in draws])
        value += mass*float(y.mean())
        se = mass*float(y.std(ddof=1))/np.sqrt(samples)
        n = samples
    # Sampling estimates can exceed 1 slightly; expose raw estimate and clamp
    # only when creating deployment constants, with the adjustment recorded.
    return dict(g=[0., g1, g2, value], se=se, tail_mass=mass,
                bound=[base, min(1., base+mass)], evaluations=len(cache), samples=n)
