"""Prospective C=3 WOR acceptance, with explicit integration uncertainty.

g3 = g2 + sum_t [q_t-p_t]+ * (g2(R,D^t)-g1(R,D^t)).
The head is summed exactly; the residual mass is integrated with fixed-seed IID
draws. This is Rao-Blackwell integration, not token-dependent online selection.
"""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'duet_tree_posthoc'))
from fanout import gains


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


def brute_gains(p, q, c=3):
    """Independent enumeration of every ordered proposal/rejection path."""
    p, q = distribution(p), distribution(q)
    def accepted(r, d, left):
        if not left or d.sum() == 0:
            return 0.
        d = d/d.sum(); total = 0.
        for t in np.flatnonzero(d > 0):
            a = min(1., r[t]/d[t])
            rest = 0.
            if a < 1:
                rr = np.maximum(r-d, 0); rr /= rr.sum()
                dd = d.copy(); dd[t] = 0
                rest = accepted(rr, dd, left-1)
            total += d[t]*(a+(1-a)*rest)
        return total
    return [0.]+[accepted(p, q, n) for n in range(1, c+1)]
