"""Independent checks for exact expectation, variance and raw feature extraction."""
import json
from pathlib import Path
import numpy as np
from features import extract, exact_ladder, groups
from calibrate import from_edges


def main():
    rng = np.random.default_rng(25322)
    maximum = 0.; cases = 0
    # Enumerate terminal-path probabilities independently by recursive walking.
    for _ in range(300):
        par = [-1, -1, 0, 0, 1, 2]; sib = [0, 1, 0, 1, 0, 0]
        tokens = [0, 1, 2, 3, 4, 0]
        p = rng.dirichlet(np.ones(5), size=7)
        parentq = rng.dirichlet(np.ones(5), size=7)
        q = parentq[np.asarray(par)+1]
        data = extract(par, sib, tokens, p.copy(), q.copy(), [])
        probs = []; kids = groups(par, sib)
        def walk(ctx, mass, depth):
            residual = p[ctx+1].copy(); draft = parentq[ctx+1].copy()
            for j in kids.get(ctx, []):
                a = min(1., residual[tokens[j]] / draft[tokens[j]])
                walk(j, mass*a, depth+1); mass *= 1-a
                residual = np.maximum(residual-draft, 0.)
                residual /= max(residual.sum(), 1e-300)
                draft[tokens[j]] = 0; draft /= draft.sum()
            probs.append((depth, mass))
        walk(-1, 1., 0)
        mean = sum(d*w for d,w in probs)
        var = sum((d-mean)**2*w for d,w in probs)
        rho, _ = from_edges(data, np.asarray(data['alpha']))
        maximum = max(maximum, abs(mean-data['true_al']), abs(var-data['coin_variance']),
            float(np.max(abs(rho-data['reach']))))
        if maximum > 1e-10: raise ValueError('Expectation/variance mismatch')
        cases += 1
    # Same online q and sampled token, different p gives opposing acceptance.
    q = np.asarray([[.9, .1]])
    a = extract([-1], [0], [0], np.asarray([[.9,.1],[.5,.5]]), q.copy(), [])
    b = extract([-1], [0], [0], np.asarray([[0.,1.],[.5,.5]]), q.copy(), [])
    assert a['alpha'] == [1.] and b['alpha'] == [0.]
    result = dict(passed=True, random_exact_enumerations=cases,
        maximum_error=maximum, draft_feature_nonidentifiability_example=True)
    path = Path(__file__).resolve().parent / 'math_audit.json'
    path.write_text(json.dumps(result, indent=2)); print(json.dumps(result))


if __name__ == '__main__': main()
