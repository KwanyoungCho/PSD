"""Exact, compact diagnostics from a realized served tree. No sampling changes."""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'duet_tree_analysis'))
from core import exact_ladder, groups, path_product


def entropy(x):
    return float(-np.sum(x * np.log(np.maximum(x, 1e-300))))


def extract(par, sib, tok, p, q, path):
    par = np.asarray(par, dtype=int); sib = np.asarray(sib, dtype=int)
    tok = np.asarray(tok, dtype=int)
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    p /= p.sum(axis=1, keepdims=True); q /= q.sum(axis=1, keepdims=True)
    calc = exact_ladder(par, sib, tok, p, q)
    n = len(par); depth = np.zeros(n, dtype=int)
    for j, parent in enumerate(par):
        depth[j] = 1 if parent < 0 else depth[parent] + 1
    raw = q[np.arange(n), tok]
    fields = {k: np.zeros(n) for k in ['q_cond', 'q_entropy', 'q_max',
        'previous_q_mass', 'overlap', 'p_token', 'p_entropy', 'top1_agree', 'p_rank']}
    for parent, children in groups(par, sib).items():
        qq = q[children[0]]; pp = p[parent + 1]; previous = 0.
        for j in children:
            fields['q_cond'][j] = raw[j] / max(1 - previous, 1e-14)
            fields['previous_q_mass'][j] = previous
            fields['q_entropy'][j] = entropy(qq)
            fields['q_max'][j] = qq.max()
            fields['overlap'][j] = np.minimum(pp, qq).sum()
            fields['p_token'][j] = pp[tok[j]]
            fields['p_entropy'][j] = entropy(pp)
            fields['top1_agree'][j] = pp.argmax() == qq.argmax()
            fields['p_rank'][j] = 1 + np.sum(pp > pp[tok[j]])
            previous += raw[j]
    result = dict(par=par, sib=sib, tok=tok, depth=depth, raw=raw,
        q_path=path_product(par, raw), alpha=calc['alpha'], reach=calc['reach'],
        attempt=calc['attempt'], terminal=calc['terminal'], **fields)
    result = {k: v.tolist() for k, v in result.items()}
    result.update(n=n, true_al=calc['al'], observed_al=len(path),
        coin_variance=max(0., float(np.dot(calc['terminal'][1:], depth**2) - calc['al']**2)))
    return result
