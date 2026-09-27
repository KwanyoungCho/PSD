"""Small hierarchical scalar tables, grouped cross-fitting; no neural head."""
from collections import defaultdict
import numpy as np
from features import groups

BINS = [0, .01, .03, .1, .3, .6, .9, 1.00001]
MODES = ['refit_q', 'depth_q', 'cond_depth_q', 'draft_rich', 'direct_beta']


def keys(t, j, mode):
    value = t['q_cond'][j] if mode == 'cond_depth_q' else t['raw'][j]
    b = int(np.searchsorted(BINS, value, side='right') - 1)
    p = str(t['phase']); ps = p + ':' + str(t['sib'][j])
    result = ['all', p, ps, ps + ':q' + str(b)]
    if mode != 'refit_q': result.append(result[-1] + ':d' + str(t['depth'][j]))
    if mode in ('draft_rich', 'direct_beta'):
        h = int(np.searchsorted([.5, 1.5, 3., 6.], t['q_entropy'][j], side='right'))
        result.append(result[-1] + ':h' + str(h))
    return result


def fit(trees, mode, shrink=5.):
    stats = defaultdict(lambda: [0., 0.]); parents = {}
    for t in trees:
        for j, a in enumerate(t['alpha']):
            w = t['attempt'][j]
            if mode == 'direct_beta':
                parent = t['par'][j]
                w = 1. if parent < 0 else t['reach'][parent]
                a = t['reach'][j] / w if w > 1e-14 else 0.
            previous = None
            for key in keys(t, j, mode):
                stats[key][0] += w * a; stats[key][1] += w
                parents[key] = previous; previous = key
    means = {}
    for key in sorted(stats, key=lambda k: (k != 'all', k.count(':'), k)):
        num, den = stats[key]
        prior = means.get(parents[key], .5)
        means[key] = num / den if key == 'all' and den else (num + shrink * prior) / (den + shrink)
    return dict(mode=mode, shrink=shrink, means=means, stats=dict(stats))


def predict(t, model):
    means = model['means']; mode = model['mode']
    values = []
    for j in range(t['n']):
        value = .5
        for key in keys(t, j, mode): value = means.get(key, value)
        values.append(value)
    return np.asarray(values)


def frozen_predict(t, model):
    values = []
    for j in range(t['n']):
        p = str(t['phase']); key = p + ':' + str(t['sib'][j])
        a = model['means'].get(key, model['means'].get(p, model['means']['all']))
        b = int(np.searchsorted(model['bins'], t['raw'][j], side='right') - 1)
        values.append(model['means'].get(key + ':' + str(b), a))
    return np.asarray(values)


def from_edges(t, values, beta=False):
    reach = np.zeros(t['n']); local = np.zeros(t['n'])
    for parent, children in sorted(groups(t['par'], t['sib']).items()):
        incoming = 1. if parent < 0 else reach[parent]
        if beta:
            vv = values[children].copy(); vv /= max(1., vv.sum())
        else:
            a = values[children]; vv = a * np.r_[1., np.cumprod(1-a[:-1])]
        local[children] = vv; reach[children] = incoming * vv
    return reach, local
