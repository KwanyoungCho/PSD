"""Compare both candidate distributions to the SAME true residual.

Read-only reanalysis of the original 8-run, layer-56 snapshots. No generation
or policy tuning. Run from repository root with ssd/.venv/bin/python.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    out = Path(__file__).resolve().parent
    root = out.parents[2]
    raw = root / 'ssd/experiments/proxy_source_ablation/probe_replay_20260910/out'
    parts = []
    for metric in sorted((out / 'metrics').glob('*.npz')):
        meta = json.loads(metric.with_suffix('.json').read_text())
        with np.load(metric) as m:
            h = m['row::h_true'][:, :4].reshape(-1).astype(np.float64)
            existing = m['row::residual_tvd'][:, :4].reshape(-1)
            local_r = m['row::local_residual'][:, :4].reshape(-1)
            local_e = m['row::local_proxy'][:, :4].reshape(-1)
            expected_step = np.repeat(m['step'], 4)
        rows = []
        for chunk in meta['probe']['chunks']:
            with np.load(raw / chunk['file']) as data:
                keep = ~data['is_bonus']
                p = data['p_T'][keep].astype(np.float64)
                q = data['p_D'][keep].astype(np.float64)
                e = data['p_E'][keep, 0].astype(np.float64)
                y = data['y'][keep]
                a, b = np.maximum(p-q, 0), np.maximum(e-q, 0)
                z, zh = a.sum(-1), b.sum(-1)
                assert (z > 0).all() and (zh > 0).all()
                r, rh = a/z[:, None], b/zh[:, None]
                # Also record the full-vocabulary e-only distribution after
                # the implementation's exclusion of the observed draft token.
                e_excluded = e.copy()
                e_excluded[np.arange(len(y)), y] = 0
                e_excluded /= e_excluded.sum(-1, keepdims=True)
                b_excluded = b.copy()
                b_excluded[np.arange(len(y)), y] = 0
                b_mass = b_excluded.sum(-1)
                valid_excluded = b_mass > 0
                rh_excluded = b_excluded/np.maximum(b_mass[:, None], 1e-300)
                tv = lambda x, y: .5*np.abs(x-y).sum(-1)
                rows.append(dict(z=z, z_proxy=zh, tv_pe=tv(p, e),
                                 tv_r_rh=tv(r, rh), tv_r_e=tv(r, e),
                                 tv_r_e_excluded=tv(r, e_excluded),
                                 tv_r_rh_excluded=tv(r, rh_excluded),
                                 valid_excluded=valid_excluded,
                                 step=data['step'][keep],
                                 seq_id=data['seq_id'][keep]))
        d = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
        np.testing.assert_array_equal(d['step'], expected_step)
        np.testing.assert_allclose(d['tv_r_rh'], existing, atol=2e-5, rtol=0)
        d.update(h=h, local_residual=local_r, local_proxy=local_e,
                 dataset=np.full(len(h), meta['record']['dataset']),
                 seed=np.full(len(h), meta['record']['seed']))
        parts.append(d)
        print(f'checked {metric.stem}: {len(h)} rejection positions', flush=True)
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    names = ['z', 'z_proxy', 'tv_pe', 'tv_r_rh', 'tv_r_e', 'tv_r_e_excluded',
             'local_residual', 'local_proxy']

    def summarize(mask):
        w = d['h'][mask]
        return dict(rows=int(mask.sum()), row_fraction=float(mask.mean()),
                    reject_event_fraction=float(w.sum()/d['h'].sum()),
                    equal_row_means={k: float(d[k][mask].mean()) for k in names},
                    reject_event_means={k: float(np.average(d[k][mask], weights=w)) for k in names},
                    residual_tv_better_row_fraction=float((d['tv_r_rh'][mask] < d['tv_r_e'][mask]).mean()),
                    residual_tv_better_event_fraction=float(np.average(d['tv_r_rh'][mask] < d['tv_r_e'][mask], weights=w)))

    masks = {'all': np.ones(len(d['h']), dtype=bool), 'heldout': d['seq_id'] > 16}
    for threshold in (.03, .05, .1, .2):
        masks[f'z_lt_{threshold:g}'] = d['z'] < threshold
    for lo, hi in ((0, .05), (.05, .1), (.1, .2), (.2, .4), (.4, .6), (.6, 1.01)):
        masks[f'z_bin_{lo:g}_{hi:g}'] = (d['z'] >= lo) & (d['z'] < hi)
    masks['small_proxy_error_large_residual_error'] = (d['tv_pe'] < .05) & (d['tv_r_rh'] > .5)
    result = dict(
        scope='8 runs; layer 56; T=1; chain B=1; 11980 non-bonus positions; descriptive post-hoc diagnostics',
        weighting='Equal rows and true first-rejection h weights are reported separately; neither is prompt-balanced coverage.',
        tv_definition='Full-vocabulary, unmasked TV(R,Rhat) and TV(R,e); excluded e is separately labeled. No top-M truncation for TV.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        z_quantiles=dict(zip(['min', 'p10', 'p25', 'median', 'p75', 'p90', 'max'],
                            np.quantile(d['z'], [0, .1, .25, .5, .75, .9, 1]).tolist())),
        groups={k: summarize(m) for k, m in masks.items() if m.any()},
    )
    valid = d['valid_excluded']
    result['matched_excluded_distributions'] = dict(
        explanation='Both e and (e-q)+ exclude observed y, then normalize over full vocabulary; zero-sum residual rows excluded from BOTH policies.',
        valid_rows=int(valid.sum()), invalid_rows=int((~valid).sum()),
        valid_event_mass_fraction=float(d['h'][valid].sum()/d['h'].sum()),
        tv_r_rh_event_mean=float(np.average(d['tv_r_rh_excluded'][valid], weights=d['h'][valid])),
        tv_r_e_event_mean=float(np.average(d['tv_r_e_excluded'][valid], weights=d['h'][valid])),
    )
    (out / 'followup_stats.json').write_text(json.dumps(result, indent=2))
    np.savez_compressed(out / 'followup_stats.npz', **d)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
